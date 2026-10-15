"""Local GPU feasibility benchmark for CodeBERT fine-tuning (Milestone 7).

Run BEFORE the training loop is written. It answers one question on this laptop's
RTX 3050 (4 GB): what fits, how fast does it run, and how long would an epoch take?
The answers fix micro_batch_size, grad_accum_steps, gradient_checkpointing and the
epoch budget in configs/train.yaml.

Three modes (python scripts/benchmark_gpu.py <mode>):

sweep    Worst-case memory. Real forward, backward and AdamW steps on batches where
         every sequence is the full 512 tokens, for a grid of micro-batch sizes with
         and without gradient checkpointing, plus no-gradient evaluation at several
         batch sizes. A configuration "fits" if it did not run out of memory and at
         least HEADROOM_MB of GPU memory was still free afterwards (measured on the
         whole device, so the CUDA context and other programs are counted).
         Writes reports/m7_gpu_sweep.csv and reports/m7_gpu_env.json.
sustain  Real throughput. Takes the fastest configurations that fit, trains each on
         real shuffled train batches for a short time, then trains the best one for
         longer to expose laptop thermal throttling. Prints the projected epoch time
         and which row of the decision gate it falls in. Writes
         reports/m7_gpu_sustain_windows.csv and reports/m7_gpu_sustain_summary.csv.
sanity   Numerical check. Trains on one fixed batch (one example per class) with a
         constant learning rate and reports whether the loss stays finite and goes
         down. The loss is NOT required to reach zero. Writes reports/m7_gpu_sanity.csv.

Nothing here is a unit test: it measures hardware. Model weights are loaded with
local_files_only=True, so no network is used.
"""

import argparse
import gc
import json
import math
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader, Subset

from sdp.config import PROJECT_ROOT
from sdp.data.dataset import SourceCodeDataset, build_dataloaders, collate_batch
from sdp.data.taxonomy import NUM_LEAVES
from sdp.model.classifier import CodeBERTClassifier
from sdp.training.class_weights import compute_class_weights, count_classes
from sdp.training.config import PRECISIONS, TrainConfig

REPORTS_DIR = PROJECT_ROOT / "reports"
SWEEP_CSV = "m7_gpu_sweep.csv"
ENV_JSON = "m7_gpu_env.json"
SUSTAIN_WINDOWS_CSV = "m7_gpu_sustain_windows.csv"
SUSTAIN_SUMMARY_CSV = "m7_gpu_sustain_summary.csv"
SANITY_CSV = "m7_gpu_sanity.csv"

SEQ_LEN = 512
SYNTHETIC_VOCAB = 50000
HEADROOM_MB = 400.0
SWEEP_OPT_STEPS = 3
EVAL_TIMED_BATCHES = 5
PROBE_EVAL_BATCHES = 60

#: (gradient_checkpointing, micro_batch_size) pairs tried by the sweep, smallest first.
TRAIN_GRID: tuple[tuple[bool, int], ...] = (
    (False, 1),
    (False, 2),
    (False, 4),
    (True, 4),
    (True, 8),
    (True, 16),
)
EVAL_BATCH_SIZES: tuple[int, ...] = (8, 16, 32)

SWEEP_COLUMNS: tuple[str, ...] = (
    "mode",
    "checkpointing",
    "micro_batch_size",
    "accum",
    "status",
    "fits",
    "samples_per_sec",
    "peak_allocated_mb",
    "peak_reserved_mb",
    "free_after_mb",
    "attn",
)

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class Candidate:
    """One training configuration under test."""

    checkpointing: bool
    micro_batch_size: int

    @property
    def label(self) -> str:
        kind = "ckpt" if self.checkpointing else "plain"
        return f"mb{self.micro_batch_size}_{kind}"


@dataclass
class TrainState:
    """Everything one training configuration needs to run steps."""

    model: CodeBERTClassifier
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    loss_fn: torch.nn.Module
    dtype: torch.dtype | None
    max_grad_norm: float


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _mb(num_bytes: float) -> float:
    return num_bytes / (1024 * 1024)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this benchmark needs the GPU build of PyTorch.")
    return torch.device("cuda")


def _check_precision(cfg: TrainConfig) -> None:
    if cfg.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 is not supported on this GPU; rerun with --precision fp16.")


def _autocast_dtype(precision: str) -> torch.dtype | None:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]


def _cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _accum_steps(cfg: TrainConfig, candidate: Candidate) -> int:
    """Micro-batches per optimizer step so the effective batch stays at the config's size."""
    return max(1, math.ceil(cfg.effective_batch_size / candidate.micro_batch_size))


def _class_weights(cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    labels = pd.read_parquet(cfg.cache_path / "train.parquet", columns=["label"])["label"]
    weights = compute_class_weights(count_classes(labels.to_numpy()), cfg.class_weight_power)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _pad_token_id(cfg: TrainConfig) -> int:
    model_config = transformers.AutoConfig.from_pretrained(cfg.model_name, local_files_only=True)
    return int(model_config.pad_token_id)


def _to_device(batch: dict, device: torch.device) -> Batch:
    return (
        batch["input_ids"].to(device, non_blocking=True),
        batch["attention_mask"].to(device, non_blocking=True),
        batch["labels"].to(device, non_blocking=True),
    )


def _synthetic_batch(batch_size: int, device: torch.device) -> Batch:
    """Worst case for memory: every sequence is the full 512 tokens."""
    ids = torch.randint(3, SYNTHETIC_VOCAB, (batch_size, SEQ_LEN), device=device)
    mask = torch.ones_like(ids)
    labels = torch.randint(0, NUM_LEAVES, (batch_size,), device=device)
    return ids, mask, labels


def _environment(device: torch.device, cfg: TrainConfig) -> dict[str, object]:
    torch.zeros(1, device=device)  # create the CUDA context before measuring free memory
    free, total = torch.cuda.mem_get_info()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "total_vram_mb": round(_mb(total), 1),
        "free_vram_mb_at_start": round(_mb(free), 1),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "precision": cfg.precision,
        "effective_batch_size": cfg.effective_batch_size,
        "headroom_target_mb": HEADROOM_MB,
    }


def _to_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def _gpu_stats() -> dict[str, float]:
    """Temperature, SM clock, power and memory use from nvidia-smi (empty if unavailable)."""
    query = "temperature.gpu,clocks.sm,power.draw,memory.used"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        parts = result.stdout.strip().splitlines()[0].split(",")
    except (OSError, subprocess.SubprocessError, IndexError):
        return {}
    if len(parts) != 4:
        return {}
    names = ("temp_c", "sm_clock_mhz", "power_w", "mem_used_mb")
    return {name: _to_float(part.strip()) for name, part in zip(names, parts, strict=True)}


def _series(windows: list[dict[str, object]], name: str) -> list[float]:
    values = [float(w[name]) for w in windows if name in w]
    return [v for v in values if not math.isnan(v)]


# --------------------------------------------------------------------------- #
# Training building blocks (a deliberately minimal version of the real loop)
# --------------------------------------------------------------------------- #


def _make_state(
    cfg: TrainConfig, candidate: Candidate, device: torch.device, weights: torch.Tensor
) -> TrainState:
    model = CodeBERTClassifier.from_pretrained(
        cfg.model_name,
        dropout=cfg.dropout,
        gradient_checkpointing=candidate.checkpointing,
        local_files_only=True,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, fused=True
    )
    return TrainState(
        model=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16"),
        loss_fn=torch.nn.CrossEntropyLoss(weight=weights),
        dtype=_autocast_dtype(cfg.precision),
        max_grad_norm=cfg.max_grad_norm,
    )


def _micro_step(
    state: TrainState,
    ids: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    accum: int,
) -> torch.Tensor:
    """One forward and backward pass; returns the (unscaled, detached) loss."""
    with torch.autocast("cuda", dtype=state.dtype, enabled=state.dtype is not None):
        logits = state.model(ids, mask)
    loss = state.loss_fn(logits.float(), labels)
    state.scaler.scale(loss / accum).backward()
    return loss.detach()


def _optimizer_step(state: TrainState) -> None:
    state.scaler.unscale_(state.optimizer)
    torch.nn.utils.clip_grad_norm_(state.model.parameters(), state.max_grad_norm)
    state.scaler.step(state.optimizer)
    state.scaler.update()
    state.optimizer.zero_grad(set_to_none=True)


# --------------------------------------------------------------------------- #
# Mode 1: sweep (worst-case memory)
# --------------------------------------------------------------------------- #


def _measure_training(
    cfg: TrainConfig, candidate: Candidate, device: torch.device, weights: torch.Tensor
) -> dict[str, object]:
    state = _make_state(cfg, candidate, device, weights)
    accum = _accum_steps(cfg, candidate)
    ids, mask, labels = _synthetic_batch(candidate.micro_batch_size, device)

    def one_optimizer_step() -> None:
        for _ in range(accum):
            _micro_step(state, ids, mask, labels, accum)
        _optimizer_step(state)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    one_optimizer_step()  # warm-up; also allocates the optimizer state
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(SWEEP_OPT_STEPS):
        one_optimizer_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    free_after, _ = torch.cuda.mem_get_info()
    samples = SWEEP_OPT_STEPS * accum * candidate.micro_batch_size
    return {
        "samples_per_sec": samples / elapsed,
        "peak_allocated_mb": _mb(torch.cuda.max_memory_allocated()),
        "peak_reserved_mb": _mb(torch.cuda.max_memory_reserved()),
        "free_after_mb": _mb(free_after),
        "attn": str(state.model.encoder.config._attn_implementation),
    }


def _measure_eval(cfg: TrainConfig, batch_size: int, device: torch.device) -> dict[str, object]:
    model = CodeBERTClassifier.from_pretrained(
        cfg.model_name, dropout=cfg.dropout, local_files_only=True
    ).to(device)
    model.eval()
    dtype = _autocast_dtype(cfg.precision)
    ids, mask, _ = _synthetic_batch(batch_size, device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        model(ids, mask)  # warm-up
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(EVAL_TIMED_BATCHES):
            model(ids, mask)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    free_after, _ = torch.cuda.mem_get_info()
    return {
        "samples_per_sec": EVAL_TIMED_BATCHES * batch_size / elapsed,
        "peak_allocated_mb": _mb(torch.cuda.max_memory_allocated()),
        "peak_reserved_mb": _mb(torch.cuda.max_memory_reserved()),
        "free_after_mb": _mb(free_after),
        "attn": str(model.encoder.config._attn_implementation),
    }


def _guarded(measure: Callable[[], dict[str, object]], **identity: object) -> dict[str, object]:
    """Run one measurement; an out-of-memory error becomes a row, not a crash."""
    row: dict[str, object] = dict(identity)
    try:
        row.update(measure())
        row["status"] = "ok"
    except torch.cuda.OutOfMemoryError:
        row["status"] = "oom"
    _cleanup()
    row["fits"] = row["status"] == "ok" and float(row.get("free_after_mb", 0.0)) >= HEADROOM_MB
    return row


def _format_row(row: dict[str, object]) -> str:
    if row["status"] != "ok":
        return f"  -> {str(row['status']).upper()} (out of memory)"
    return (
        f"  -> {float(row['samples_per_sec']):6.2f} samples/s | "
        f"peak reserved {float(row['peak_reserved_mb']):6.0f} MB | "
        f"free after {float(row['free_after_mb']):6.0f} MB | fits={row['fits']}"
    )


def run_sweep(cfg: TrainConfig, out_dir: Path) -> int:
    device = _require_cuda()
    _check_precision(cfg)
    torch.manual_seed(cfg.seed)
    weights = _class_weights(cfg, device)
    env = _environment(device, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ENV_JSON).write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(json.dumps(env, indent=2))

    rows: list[dict[str, object]] = []
    for checkpointing, micro in TRAIN_GRID:
        candidate = Candidate(checkpointing, micro)
        print(f"\ntraining {candidate.label} ...", flush=True)
        row = _guarded(
            partial(_measure_training, cfg, candidate, device, weights),
            mode="train",
            checkpointing=checkpointing,
            micro_batch_size=micro,
            accum=_accum_steps(cfg, candidate),
        )
        print(_format_row(row), flush=True)
        rows.append(row)

    for batch_size in EVAL_BATCH_SIZES:
        print(f"\nevaluation (no gradients) at batch {batch_size} ...", flush=True)
        row = _guarded(
            partial(_measure_eval, cfg, batch_size, device),
            mode="eval",
            checkpointing=False,
            micro_batch_size=batch_size,
            accum=0,
        )
        print(_format_row(row), flush=True)
        rows.append(row)

    table = pd.DataFrame(rows).reindex(columns=list(SWEEP_COLUMNS))
    table.round(2).to_csv(out_dir / SWEEP_CSV, index=False)
    print("\nSweep results:")
    print(table.round(1).to_string(index=False))
    fitting = table[(table["mode"] == "train") & table["fits"].astype(bool)]
    print(f"\n{len(fitting)} training configuration(s) fit with >= {HEADROOM_MB:.0f} MB headroom.")
    print(f"Wrote {out_dir / SWEEP_CSV}")
    return 0


# --------------------------------------------------------------------------- #
# Mode 2: sustain (real data, thermals, epoch projection)
# --------------------------------------------------------------------------- #


def _sustain_trial(
    cfg: TrainConfig,
    candidate: Candidate,
    device: torch.device,
    weights: torch.Tensor,
    pad_id: int,
    minutes: float,
    window_seconds: float,
    label: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Train on real shuffled batches for ``minutes``; return per-window rows and a summary."""
    loaders = build_dataloaders(cfg.cache_path, pad_id, batch_size=candidate.micro_batch_size)
    loader = loaders["train"]
    state = _make_state(cfg, candidate, device, weights)
    accum = _accum_steps(cfg, candidate)
    windows: list[dict[str, object]] = []
    status = "ok"
    micro_count = 0
    window_samples = 0
    last_loss = torch.zeros((), device=device)
    started = window_start = time.perf_counter()
    deadline = started + minutes * 60.0
    finished = False
    try:
        while not finished:
            for batch in loader:
                ids, mask, labels = _to_device(batch, device)
                last_loss = _micro_step(state, ids, mask, labels, accum)
                micro_count += 1
                window_samples += int(ids.shape[0])
                if micro_count % accum == 0:
                    _optimizer_step(state)
                if time.perf_counter() - window_start >= window_seconds:
                    torch.cuda.synchronize()
                    now = time.perf_counter()
                    windows.append(
                        {
                            "trial": label,
                            "window": len(windows) + 1,
                            "elapsed_s": now - started,
                            "samples_per_sec": window_samples / (now - window_start),
                            "loss": float(last_loss),
                            **_gpu_stats(),
                        }
                    )
                    window_start, window_samples = now, 0
                    if now >= deadline:
                        finished = True
                        break
    except torch.cuda.OutOfMemoryError:
        status = "oom"

    steady = windows[1:] or windows  # the first window includes start-up effects
    rates = [float(w["samples_per_sec"]) for w in steady]
    temps = _series(windows, "temp_c")
    clocks = _series(windows, "sm_clock_mhz")
    powers = _series(windows, "power_w")
    mems = _series(windows, "mem_used_mb")
    summary: dict[str, object] = {
        "trial": label,
        "micro_batch_size": candidate.micro_batch_size,
        "checkpointing": candidate.checkpointing,
        "accum": accum,
        "minutes_requested": minutes,
        "status": status,
        "windows": len(windows),
        "steady_samples_per_sec": statistics.median(rates) if rates else 0.0,
        "loss_finite": all(math.isfinite(float(w["loss"])) for w in windows),
        "temp_max_c": max(temps) if temps else math.nan,
        "sm_clock_first_mhz": clocks[0] if clocks else math.nan,
        "sm_clock_last_mhz": clocks[-1] if clocks else math.nan,
        "power_median_w": statistics.median(powers) if powers else math.nan,
        "mem_used_max_mb": max(mems) if mems else math.nan,
    }
    return windows, summary


def _format_summary(summary: dict[str, object]) -> str:
    return (
        f"  -> {summary['status']} | steady "
        f"{float(summary['steady_samples_per_sec']):.2f} samples/s | "
        f"max temp {summary['temp_max_c']} C | "
        f"SM clock {summary['sm_clock_first_mhz']} -> {summary['sm_clock_last_mhz']} MHz | "
        f"loss finite: {summary['loss_finite']}"
    )


def _eval_probe(cfg: TrainConfig, device: torch.device, pad_id: int) -> tuple[float, int]:
    """Samples per second of a no-gradient pass over random validation batches."""
    dataset = SourceCodeDataset(cfg.cache_path / "val.parquet")
    generator = torch.Generator().manual_seed(cfg.seed)
    wanted = PROBE_EVAL_BATCHES * cfg.eval_batch_size
    indices = torch.randperm(len(dataset), generator=generator)[:wanted].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=cfg.eval_batch_size,
        collate_fn=partial(collate_batch, pad_token_id=pad_id),
    )
    model = CodeBERTClassifier.from_pretrained(
        cfg.model_name, dropout=cfg.dropout, local_files_only=True
    ).to(device)
    model.eval()
    dtype = _autocast_dtype(cfg.precision)
    seen = 0
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        for index, batch in enumerate(loader):
            ids, mask, _ = _to_device(batch, device)
            model(ids, mask)
            if index == 0:  # the first batch is a warm-up and is not timed
                torch.cuda.synchronize()
                start = time.perf_counter()
            else:
                seen += int(ids.shape[0])
    torch.cuda.synchronize()
    return seen / (time.perf_counter() - start), len(dataset)


def _gate(total_minutes: float) -> str:
    if total_minutes <= 120:
        return "<= 2 h/epoch: up to 5 epochs, early stopping; extra seeds realistic"
    if total_minutes <= 240:
        return "2-4 h/epoch: max 3 epochs, single seed, no ablation"
    return "> 4 h/epoch: use the fallbacks (freeze embeddings or lower layers), then re-measure"


def run_sustain(cfg: TrainConfig, args: argparse.Namespace) -> int:
    device = _require_cuda()
    _check_precision(cfg)
    torch.manual_seed(cfg.seed)
    sweep_path = args.out_dir / SWEEP_CSV
    if not sweep_path.exists():
        raise SystemExit(f"{sweep_path} not found; run the sweep mode first.")
    sweep = pd.read_csv(sweep_path)
    fitting = sweep[(sweep["mode"] == "train") & sweep["fits"].astype(bool)]
    fitting = fitting.sort_values("samples_per_sec", ascending=False).head(args.top)
    if fitting.empty:
        raise SystemExit(
            "No training configuration fit with the headroom target. "
            "Use the fallbacks (freeze embeddings or lower layers) before continuing."
        )
    candidates = [
        Candidate(bool(row.checkpointing), int(row.micro_batch_size))
        for row in fitting.itertuples()
    ]
    weights = _class_weights(cfg, device)
    pad_id = _pad_token_id(cfg)

    all_windows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for candidate in candidates:
        print(f"\nshort trial {candidate.label} ({args.short_minutes:g} min) ...", flush=True)
        windows, summary = _sustain_trial(
            cfg,
            candidate,
            device,
            weights,
            pad_id,
            args.short_minutes,
            args.window_seconds,
            f"{candidate.label}_short",
        )
        _cleanup()
        print(_format_summary(summary), flush=True)
        all_windows += windows
        summaries.append(summary)

    finished = [
        (summary, candidate)
        for summary, candidate in zip(summaries, candidates, strict=True)
        if summary["status"] == "ok"
    ]
    if not finished:
        raise SystemExit("Every short trial ran out of memory; rerun the sweep after closing apps.")
    _, best = max(finished, key=lambda pair: float(pair[0]["steady_samples_per_sec"]))

    print(f"\nlong trial {best.label} ({args.long_minutes:g} min) ...", flush=True)
    windows, long_summary = _sustain_trial(
        cfg,
        best,
        device,
        weights,
        pad_id,
        args.long_minutes,
        args.window_seconds,
        f"{best.label}_long",
    )
    _cleanup()
    print(_format_summary(long_summary), flush=True)
    all_windows += windows
    train_sps = float(long_summary["steady_samples_per_sec"])
    if long_summary["status"] != "ok" or train_sps <= 0:
        raise SystemExit(f"The long trial failed (status {long_summary['status']}).")

    print("\nmeasuring validation speed ...", flush=True)
    eval_sps, n_val = _eval_probe(cfg, device, pad_id)
    _cleanup()
    n_train = len(pd.read_parquet(cfg.cache_path / "train.parquet", columns=["label"]))
    train_min = n_train / train_sps / 60.0
    val_min = n_val / eval_sps / 60.0
    total_min = train_min + val_min
    gate = _gate(total_min)
    long_summary.update(
        eval_samples_per_sec=eval_sps,
        projected_train_min=train_min,
        projected_val_min=val_min,
        projected_epoch_min=total_min,
        gate=gate,
    )
    summaries.append(long_summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_windows).round(3).to_csv(args.out_dir / SUSTAIN_WINDOWS_CSV, index=False)
    pd.DataFrame(summaries).round(3).to_csv(args.out_dir / SUSTAIN_SUMMARY_CSV, index=False)

    flag = "--checkpointing" if best.checkpointing else "--no-checkpointing"
    print("\n" + "=" * 70)
    print(f"Best configuration: {best.label} (precision {cfg.precision})")
    print(f"Steady training speed: {train_sps:.2f} samples/s, validation {eval_sps:.2f} samples/s")
    print(
        f"Projected epoch: {train_min:.0f} min training + {val_min:.0f} min validation "
        f"= {total_min:.0f} min ({total_min / 60:.1f} h)"
    )
    print(f"Decision gate: {gate}")
    print("Suggested configs/train.yaml values:")
    print(f"  micro_batch_size: {best.micro_batch_size}")
    print(f"  grad_accum_steps: {_accum_steps(cfg, best)}")
    print(f"  gradient_checkpointing: {str(best.checkpointing).lower()}")
    print(f"  precision: {cfg.precision}")
    print(
        f"Next: python scripts/benchmark_gpu.py sanity --micro-batch-size "
        f"{best.micro_batch_size} {flag}"
    )
    print(f"Wrote {args.out_dir / SUSTAIN_WINDOWS_CSV} and {args.out_dir / SUSTAIN_SUMMARY_CSV}")
    return 0


# --------------------------------------------------------------------------- #
# Mode 3: sanity (does the loss go down?)
# --------------------------------------------------------------------------- #


def run_sanity(cfg: TrainConfig, args: argparse.Namespace) -> int:
    device = _require_cuda()
    _check_precision(cfg)
    torch.manual_seed(cfg.seed)
    checkpointing = args.checkpointing
    if checkpointing is None:
        checkpointing = cfg.gradient_checkpointing
    candidate = Candidate(checkpointing, args.micro_batch_size or cfg.micro_batch_size)
    cfg = replace(cfg, learning_rate=args.lr)
    weights = _class_weights(cfg, device)
    pad_id = _pad_token_id(cfg)

    train_path = cfg.cache_path / "train.parquet"
    labels = pd.read_parquet(train_path, columns=["label"])["label"]
    one_per_class = labels.groupby(labels).head(1).index.tolist()
    dataset = SourceCodeDataset(train_path)
    batch = collate_batch([dataset[i] for i in one_per_class], pad_token_id=pad_id)
    ids, mask, targets = _to_device(batch, device)
    size = candidate.micro_batch_size
    chunks = list(zip(ids.split(size), mask.split(size), targets.split(size), strict=True))
    accum = len(chunks)
    print(
        f"sanity: {len(one_per_class)} examples (one per class), {accum} micro-batch(es) of "
        f"<= {size}, checkpointing={candidate.checkpointing}, precision={cfg.precision}, "
        f"lr={cfg.learning_rate:g}, {args.steps} steps"
    )

    state = _make_state(cfg, candidate, device, weights)
    losses: list[float] = []
    for step in range(1, args.steps + 1):
        total = torch.zeros((), device=device)
        for chunk_ids, chunk_mask, chunk_labels in chunks:
            total += _micro_step(state, chunk_ids, chunk_mask, chunk_labels, accum)
        _optimizer_step(state)
        losses.append(float(total) / accum)
        if step == 1 or step % 10 == 0:
            print(f"step {step:3d}  loss {losses[-1]:.4f}", flush=True)

    first, last = losses[:10], losses[-10:]
    finite = all(math.isfinite(value) for value in losses)
    improved = statistics.mean(last) < statistics.mean(first)
    verdict = "PASS" if finite and improved else "FAIL"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"step": range(1, len(losses) + 1), "loss": losses}).to_csv(
        args.out_dir / SANITY_CSV, index=False
    )
    print(
        f"\nfirst-10 mean loss {statistics.mean(first):.4f} -> last-10 mean loss "
        f"{statistics.mean(last):.4f} (min {min(losses):.4f}); all finite: {finite}"
    )
    print(f"Sanity verdict: {verdict} (finite and decreasing; zero is not required)")
    print(f"Wrote {args.out_dir / SANITY_CSV}")
    return 0 if verdict == "PASS" else 1


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("sweep", "sustain", "sanity"))
    parser.add_argument("--config", type=Path, default=None, help="train.yaml (default: repo)")
    parser.add_argument("--precision", choices=PRECISIONS, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--top", type=int, default=3, help="sustain: configs to test briefly")
    parser.add_argument("--short-minutes", type=float, default=3.0)
    parser.add_argument("--long-minutes", type=float, default=10.0)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=60, help="sanity: optimizer steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="sanity: constant learning rate")
    parser.add_argument("--micro-batch-size", type=int, default=None, help="sanity only")
    parser.add_argument(
        "--checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="sanity only",
    )
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    cfg = TrainConfig.from_yaml(args.config)
    if args.precision is not None:
        cfg = replace(cfg, precision=args.precision)
    if args.eval_batch_size is not None:
        cfg = replace(cfg, eval_batch_size=args.eval_batch_size)

    if args.mode == "sweep":
        return run_sweep(cfg, args.out_dir)
    if args.mode == "sustain":
        return run_sustain(cfg, args)
    return run_sanity(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
