"""Training loop for the 9-class CodeBERT classifier (Milestone 7.4).

What one run does
-----------------
Per epoch: train on shuffled batches, validate, record metrics, keep the best epoch
(by validation macro-F1), save a resumable checkpoint, and stop early after
``patience`` epochs without improvement. The TEST split is never touched here; test
evaluation happens once, in M8.

Loss and gradient accumulation (the part that is easy to get subtly wrong)
--------------------------------------------------------------------------
An optimizer step uses a GROUP of micro-batches (``grad_accum_steps`` of them, because
a full batch of 16 does not fit in 4 GB). Each sample's loss is its class weight times
its cross-entropy. The group loss is the SUM of those over every sample in the group,
divided by the ACTUAL number of samples in the group. Consequences:

* Every sample counts with exactly its class weight, whichever micro-batch it landed in.
  (PyTorch's default weighted "mean" divides by the sum of weights in each micro-batch,
  which with micro-batches of 2 would flatten the intended weighting: a LINKER file would
  count about 1.7 times a SEMANTIC file instead of about 6.4 times.)
* The final group of an epoch is usually smaller than the rest. For 28,458 training files
  and micro-batches of 2 that is 14,229 micro-batches = 1,778 full groups of 8 plus one
  group of 5 (10 samples). It still takes an optimizer step, and it is divided by 10, not
  by 16 and not by 8.
* ``tests/test_trainer.py`` proves that accumulating micro-batches gives the same gradients
  as one big batch, including a partial-size group.

Class weights are scaled to a sample-weighted mean of 1 (see class_weights.py), so the
loss scale stays comparable to an unweighted run. Validation loss uses the same weighted
sum divided by the number of validation samples; the weights come from TRAIN counts, so
it is "train-weighted" and comparable with the training loss.

Resume
------
A checkpoint is written after every epoch, so an interruption loses at most one epoch.
``resume=True`` restores model, optimizer, schedule and early-stopping state and continues
with the next epoch. It refuses to resume if the configuration differs from the saved one,
so the schedule can never change silently. Each epoch's shuffle is seeded with
``seed + epoch`` instead of relying on global random state, so a resumed run sees the same
data order as an uninterrupted one. Dropout masks after a resume differ from an
uninterrupted run; GPU kernels are not bit-exact either, so runs are reproducible from the
seed up to GPU nondeterminism.
"""

import itertools
import json
import math
import time
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from sdp.data.dataset import collate_batch
from sdp.data.taxonomy import TAXONOMY_VERSION
from sdp.evaluation.metrics import LEAF_NAMES, accuracy, macro_f1, per_class_report
from sdp.model.classifier import CodeBERTClassifier
from sdp.training.checkpoint import load_last, save_best, save_last
from sdp.training.config import TrainConfig


class NonFiniteLossError(RuntimeError):
    """Raised when the loss becomes NaN or infinite, so the run stops instead of wasting hours."""


# --------------------------------------------------------------------------- #
# Small pure pieces (each is unit-tested)
# --------------------------------------------------------------------------- #


def steps_per_epoch(n_micro_batches: int, accum: int) -> int:
    """Optimizer steps in one epoch, counting the final partial group."""
    if n_micro_batches < 1 or accum < 1:
        raise ValueError(
            f"need n_micro_batches >= 1 and accum >= 1, got {n_micro_batches}, {accum}"
        )
    return math.ceil(n_micro_batches / accum)


def warmup_decay_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Learning-rate multiplier: linear 0 -> 1 over warmup, then linear 1 -> 0 to the end."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


class EarlyStopper:
    """Tracks the best score and how many epochs in a row failed to beat it."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("-inf")
        self.best_epoch = 0
        self.bad_epochs = 0

    def update(self, score: float, epoch: int) -> bool:
        """Record an epoch's score; return True if it is a new best."""
        if score > self.best:
            self.best, self.best_epoch, self.bad_epochs = score, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, float | int]:
        return {"best": self.best, "best_epoch": self.best_epoch, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.best = float(state["best"])
        self.best_epoch = int(state["best_epoch"])
        self.bad_epochs = int(state["bad_epochs"])


def make_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Two groups: weight decay on weight matrices, none on biases and LayerNorm (convention)."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exempt = name.endswith("bias") or "LayerNorm" in name
        (no_decay if exempt else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def autocast_dtype(precision: str) -> torch.dtype | None:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]


def make_scaler(cfg: TrainConfig, device: torch.device) -> torch.amp.GradScaler:
    """Loss scaling is only needed (and only enabled) for fp16 on a GPU."""
    return torch.amp.GradScaler(
        device.type, enabled=cfg.precision == "fp16" and device.type == "cuda"
    )


def make_train_loader(
    dataset: Dataset, micro_batch_size: int, pad_token_id: int, seed: int, epoch: int
) -> DataLoader:
    """Shuffled micro-batches whose order depends only on (seed, epoch)."""
    generator = torch.Generator().manual_seed(seed + epoch)
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=partial(collate_batch, pad_token_id=pad_token_id),
    )


def weighted_sum_loss(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Sum over samples of (class weight x cross-entropy), computed in float32."""
    return torch.nn.functional.cross_entropy(
        logits.float(), labels, weight=weights, reduction="sum"
    )


def make_logger(log_path: Path | None = None) -> Callable[[str], None]:
    """A logger that prints with a timestamp and appends to ``log_path`` if given."""

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
        print(line, flush=True)
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    return log


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #


def _to_device(
    batch: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["input_ids"].to(device, non_blocking=True),
        batch["attention_mask"].to(device, non_blocking=True),
        batch["labels"].to(device, non_blocking=True),
    )


def _batches_in_groups(loader: DataLoader, accum: int) -> Iterator[list[dict]]:
    """Yield lists of up to ``accum`` micro-batches; the last list may be shorter."""
    iterator = iter(loader)
    while True:
        group = list(itertools.islice(iterator, accum))
        if not group:
            return
        yield group


def accumulate_group(
    model: torch.nn.Module,
    group: list[dict],
    weights: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler,
) -> tuple[torch.Tensor, int]:
    """Forward and backward over one group; gradients accumulate, no optimizer step.

    Returns (summed weighted loss over the group, number of samples in the group). The
    gradient is that of (summed weighted loss / number of samples), the group's ACTUAL size.
    """
    n_samples = sum(int(batch["labels"].shape[0]) for batch in group)
    total = torch.zeros((), device=device)
    for batch in group:
        ids, mask, labels = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(ids, mask)
        loss_sum = weighted_sum_loss(logits, labels, weights)
        scaler.scale(loss_sum / n_samples).backward()
        total += loss_sum.detach()
    return total, n_samples


def _optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    max_grad_norm: float,
) -> None:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


def train_one_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    loader: DataLoader,
    weights: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype | None,
    accum: int,
    max_grad_norm: float,
    epoch: int,
    steps_total: int,
    log: Callable[[str], None],
    log_every: int,
) -> dict[str, float]:
    """One pass over the training data; returns loss, throughput and the final learning rate."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    samples_total = 0
    window_loss = 0.0
    window_samples = 0
    started = window_start = time.perf_counter()
    for step, group in enumerate(_batches_in_groups(loader, accum), start=1):
        group_loss, n_samples = accumulate_group(model, group, weights, device, dtype, scaler)
        value = float(group_loss)
        if not math.isfinite(value):
            log(f"epoch {epoch} step {step}: loss is {value}; aborting the run")
            raise NonFiniteLossError(f"non-finite loss {value} at epoch {epoch}, step {step}")
        _optimizer_step(model, optimizer, scheduler, scaler, max_grad_norm)
        loss_total += value
        samples_total += n_samples
        window_loss += value
        window_samples += n_samples
        if step % log_every == 0 or step == steps_total:
            now = time.perf_counter()
            eta_minutes = (now - started) / step * (steps_total - step) / 60.0
            log(
                f"epoch {epoch} step {step}/{steps_total} | "
                f"loss {window_loss / window_samples:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                f"{window_samples / (now - window_start):.1f} samples/s | "
                f"epoch eta {eta_minutes:.1f} min"
            )
            window_loss, window_samples, window_start = 0.0, 0, now
    seconds = time.perf_counter() - started
    return {
        "loss": loss_total / samples_total,
        "seconds": seconds,
        "samples_per_sec": samples_total / seconds,
        "lr_end": scheduler.get_last_lr()[0],
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    weights: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict[str, object]:
    """No-gradient pass: weighted loss plus accuracy, macro-F1 and a per-class report."""
    model.eval()
    loss_sum = 0.0
    n_samples = 0
    predictions: list[torch.Tensor] = []
    truths: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            ids, mask, labels = _to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(ids, mask).float()
            loss_sum += weighted_sum_loss(logits, labels, weights).item()
            n_samples += int(labels.shape[0])
            predictions.append(logits.argmax(dim=-1).cpu())
            truths.append(labels.cpu())
    y_pred = torch.cat(predictions).numpy()
    y_true = torch.cat(truths).numpy()
    return {
        "loss": loss_sum / n_samples,
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "report": per_class_report(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def _summary(history: list[dict], stopper: EarlyStopper, cfg: TrainConfig) -> dict[str, object]:
    return {
        "epochs_run": len(history),
        "best_epoch": stopper.best_epoch,
        "best_val_macro_f1": stopper.best,
        "stopped_early": stopper.should_stop,
        "finished": stopper.should_stop or len(history) >= cfg.max_epochs,
    }


def _config_differences(saved: dict, current: dict) -> list[str]:
    keys = sorted(set(saved) | set(current))
    return [key for key in keys if saved.get(key) != current.get(key)]


def run_training(
    *,
    cfg: TrainConfig,
    model: CodeBERTClassifier,
    train_dataset: Dataset,
    val_dataset: Dataset,
    class_counts: Sequence[int],
    weights: torch.Tensor,
    pad_token_id: int,
    device: torch.device,
    run_dir: Path | None = None,
    resume: bool = False,
    log: Callable[[str], None] = print,
    stop_after_epoch: int | None = None,
    log_every: int = 100,
) -> dict[str, object]:
    """Train, validate every epoch, checkpoint, and return a summary dict.

    ``stop_after_epoch`` simulates an interruption (used by tests and the smoke check).
    """
    run_dir = run_dir or cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"
    dtype = autocast_dtype(cfg.precision)

    n_train = len(train_dataset)
    if n_train < 1:
        raise ValueError("the training dataset is empty")
    steps_per_ep = steps_per_epoch(math.ceil(n_train / cfg.micro_batch_size), cfg.grad_accum_steps)
    total_steps = steps_per_ep * cfg.max_epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    model.to(device)
    optimizer = torch.optim.AdamW(
        make_param_groups(model, cfg.weight_decay),
        lr=cfg.learning_rate,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_decay_factor(step, warmup_steps, total_steps)
    )
    scaler = make_scaler(cfg, device)
    stopper = EarlyStopper(cfg.patience)
    history: list[dict] = []
    start_epoch = 1

    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"nothing to resume: {last_path} does not exist")
        state = load_last(last_path)
        different = _config_differences(state["config"], cfg.to_dict())
        if different:
            raise ValueError(f"config differs from the checkpoint in: {different}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        stopper.load_state_dict(state["stopper"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        log(
            f"resumed from {last_path.name}: {state['epoch']} epoch(s) done, continuing at "
            f"epoch {start_epoch}"
        )
        if stopper.should_stop or start_epoch > cfg.max_epochs:
            log("this run is already finished; nothing to do")
            return _summary(history, stopper, cfg)
    else:
        if last_path.exists():
            raise FileExistsError(
                f"{last_path} already exists; use resume=True or choose a new run_name"
            )

    meta = {
        "model_name": cfg.model_name,
        "num_labels": len(LEAF_NAMES),
        "label_names": list(LEAF_NAMES),
        "taxonomy_version": TAXONOMY_VERSION,
        "dropout": cfg.dropout,
        "precision": cfg.precision,
        "config": cfg.to_dict(),
        "class_counts": [int(count) for count in class_counts],
        "class_weights": [float(weight) for weight in weights.detach().cpu().tolist()],
    }
    if not resume:
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "config": meta["config"],
                    "class_counts": meta["class_counts"],
                    "class_weights": meta["class_weights"],
                    "taxonomy_version": TAXONOMY_VERSION,
                    "steps_per_epoch": steps_per_ep,
                    "total_steps": total_steps,
                    "warmup_steps": warmup_steps,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log(
            f"run {cfg.run_name}: {n_train} train / {len(val_dataset)} val samples, "
            f"{steps_per_ep} optimizer steps per epoch, {total_steps} total, "
            f"{warmup_steps} warmup, effective batch {cfg.effective_batch_size}"
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=partial(collate_batch, pad_token_id=pad_token_id),
    )

    for epoch in range(start_epoch, cfg.max_epochs + 1):
        train_loader = make_train_loader(
            train_dataset, cfg.micro_batch_size, pad_token_id, cfg.seed, epoch
        )
        train_stats = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader=train_loader,
            weights=weights,
            device=device,
            dtype=dtype,
            accum=cfg.grad_accum_steps,
            max_grad_norm=cfg.max_grad_norm,
            epoch=epoch,
            steps_total=steps_per_ep,
            log=log,
            log_every=log_every,
        )
        val = evaluate(model, val_loader, weights, device, dtype)
        improved = stopper.update(float(val["macro_f1"]), epoch)
        report = val["report"]
        if improved:
            save_best(
                run_dir / "best.pt",
                {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                {**meta, "best_epoch": epoch, "val_macro_f1": float(val["macro_f1"])},
            )
            report.round(5).to_csv(run_dir / "val_report_best.csv", index=False)

        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": float(val["loss"]),
            "val_accuracy": float(val["accuracy"]),
            "val_macro_f1": float(val["macro_f1"]),
        }
        for name, f1 in zip(LEAF_NAMES, report["f1"].tolist(), strict=True):
            row[f"f1_{name}"] = f1
        row["epoch_seconds"] = train_stats["seconds"]
        row["train_samples_per_sec"] = train_stats["samples_per_sec"]
        row["lr_end"] = train_stats["lr_end"]
        row["is_best"] = improved
        history.append(row)
        pd.DataFrame(history).round(5).to_csv(run_dir / "metrics.csv", index=False)

        save_last(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            stopper=stopper.state_dict(),
            config=cfg.to_dict(),
            history=history,
        )
        log(
            f"epoch {epoch}/{cfg.max_epochs} | train loss {train_stats['loss']:.4f} | "
            f"val loss {float(val['loss']):.4f} | val macro-F1 {float(val['macro_f1']):.4f} | "
            f"val acc {float(val['accuracy']):.4f} | {train_stats['seconds'] / 60:.1f} min | "
            f"{train_stats['samples_per_sec']:.1f} samples/s" + (" | NEW BEST" if improved else "")
        )
        if stopper.should_stop:
            log(f"early stop: no improvement for {cfg.patience} epoch(s)")
            break
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            log(f"stopping after epoch {epoch} (requested)")
            break

    summary = _summary(history, stopper, cfg)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
