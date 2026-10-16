"""Tests for sdp.training.trainer: small pure pieces, gradient accumulation, and full runs.

Full runs use a tiny random encoder on CPU with dropout off, so they are fast and
deterministic: no network, no GPU.
"""

import random

import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset
from transformers import RobertaConfig, RobertaModel

from sdp.data.dataset import collate_batch
from sdp.data.taxonomy import NUM_LEAVES
from sdp.evaluation.metrics import LEAF_NAMES
from sdp.model.classifier import CodeBERTClassifier
from sdp.training.checkpoint import load_best
from sdp.training.class_weights import compute_class_weights, count_classes
from sdp.training.config import TrainConfig
from sdp.training.trainer import (
    EarlyStopper,
    NonFiniteLossError,
    accumulate_group,
    make_param_groups,
    make_scaler,
    make_train_loader,
    run_training,
    steps_per_epoch,
    warmup_decay_factor,
    weighted_sum_loss,
)

_PAD_ID = 1
_CPU = torch.device("cpu")
# Train-split class counts from reports/m6_class_split_counts.csv, in taxonomy order.
_M6_TRAIN_COUNTS = [5968, 4232, 9404, 232, 1208, 390, 741, 311, 5972]


def _tiny_encoder() -> RobertaModel:
    config = RobertaConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=70,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=_PAD_ID,
        type_vocab_size=1,
    )
    return RobertaModel(config, add_pooling_layer=False)


def _model(seed: int = 0) -> CodeBERTClassifier:
    torch.manual_seed(seed)
    return CodeBERTClassifier(_tiny_encoder(), dropout=0.0)


class _TinyDataset(Dataset):
    """Class c is the token pattern [CLS, (10 + c) x k, SEP]: easy to learn."""

    def __init__(self, n: int, seed: int) -> None:
        rng = random.Random(seed)
        self.items = []
        for i in range(n):
            label = i % NUM_LEAVES
            ids = [0] + [10 + label] * rng.randint(3, 6) + [2]
            self.items.append(
                {
                    "submission_id": f"s{seed}_{i}",
                    "input_ids": torch.tensor(ids),
                    "label": torch.tensor(label),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def _cfg(**overrides) -> TrainConfig:
    raw = {
        "run_name": "t",
        "seed": 42,
        "output_root": "runs",
        "cache_dir": "unused",
        "model_name": "tiny",
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_epochs": 4,
        "patience": 10,
        "max_grad_norm": 1.0,
        "micro_batch_size": 4,
        "grad_accum_steps": 2,
        "eval_batch_size": 8,
        "precision": "fp32",
        "gradient_checkpointing": False,
        "class_weight_power": 0.5,
    }
    raw.update(overrides)
    return TrainConfig.from_dict(raw)


def _run(tmp_path, cfg, *, resume=False, stop_after_epoch=None, model=None, name="run"):
    counts = count_classes([i % NUM_LEAVES for i in range(36)])
    weights = torch.tensor(
        compute_class_weights(counts, cfg.class_weight_power), dtype=torch.float32
    )
    return run_training(
        cfg=cfg,
        model=model if model is not None else _model(),
        train_dataset=_TinyDataset(36, 1),
        val_dataset=_TinyDataset(18, 2),
        class_counts=counts,
        weights=weights,
        pad_token_id=_PAD_ID,
        device=_CPU,
        run_dir=tmp_path / name,
        resume=resume,
        log=lambda _: None,
        stop_after_epoch=stop_after_epoch,
    )


# --------------------------------------------------------------------------- #
# Small pure pieces
# --------------------------------------------------------------------------- #


def test_warmup_decay_factor_shape():
    assert warmup_decay_factor(0, 10, 100) == 0.0
    assert warmup_decay_factor(5, 10, 100) == pytest.approx(0.5)
    assert warmup_decay_factor(10, 10, 100) == pytest.approx(1.0)
    assert warmup_decay_factor(55, 10, 100) == pytest.approx(0.5)
    assert warmup_decay_factor(100, 10, 100) == 0.0
    assert warmup_decay_factor(150, 10, 100) == 0.0
    assert warmup_decay_factor(0, 0, 100) == pytest.approx(1.0)


def test_steps_per_epoch_counts_the_partial_group():
    assert steps_per_epoch(14229, 8) == 1779  # 1,778 full groups of 8 plus one group of 5
    assert steps_per_epoch(16, 8) == 2
    assert steps_per_epoch(8, 8) == 1
    assert steps_per_epoch(1, 8) == 1


def test_epoch_shuffle_depends_only_on_seed_and_epoch():
    dataset = _TinyDataset(36, 1)

    def order(epoch: int) -> list[str]:
        loader = make_train_loader(dataset, 4, _PAD_ID, 42, epoch)
        return [sid for batch in loader for sid in batch["submission_ids"]]

    assert order(1) == order(1)
    assert order(1) != order(2)
    assert sorted(order(1)) == sorted(order(2))


def test_early_stopper_tracks_best_and_patience():
    stopper = EarlyStopper(patience=2)
    assert stopper.update(0.1, 1) is True
    assert stopper.update(0.2, 2) is True
    assert stopper.update(0.15, 3) is False
    assert not stopper.should_stop
    assert stopper.update(0.15, 4) is False
    assert stopper.should_stop
    assert (stopper.best, stopper.best_epoch) == (0.2, 2)

    restored = EarlyStopper(patience=2)
    restored.load_state_dict(stopper.state_dict())
    assert restored.state_dict() == stopper.state_dict()
    assert restored.should_stop
    assert restored.update(0.3, 5) is True
    assert not restored.should_stop


def test_param_groups_exempt_bias_and_layernorm_from_decay():
    model = _model()
    decay_group, no_decay_group = make_param_groups(model, 0.01)
    assert decay_group["weight_decay"] == 0.01
    assert no_decay_group["weight_decay"] == 0.0
    exempt_ids = {id(p) for p in no_decay_group["params"]}
    for name, parameter in model.named_parameters():
        expected_exempt = name.endswith("bias") or "LayerNorm" in name
        assert (id(parameter) in exempt_ids) == expected_exempt, name
    assert len(decay_group["params"]) + len(no_decay_group["params"]) == len(
        list(model.parameters())
    )


# --------------------------------------------------------------------------- #
# Gradient accumulation: the normalization the whole design hinges on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_micro", [5, 8])
def test_accumulated_gradients_match_one_big_batch(n_micro):
    """5 micro-batches (a partial group) and 8 (a full group) must equal one big batch."""
    model = _model(0)
    weights = torch.tensor(compute_class_weights(_M6_TRAIN_COUNTS, 0.5), dtype=torch.float32)
    n = 2 * n_micro
    items = [_TinyDataset(n, 3)[i] for i in range(n)]
    scaler = make_scaler(_cfg(), _CPU)

    big = collate_batch(items, pad_token_id=_PAD_ID)
    model.zero_grad(set_to_none=True)
    reference_loss = weighted_sum_loss(
        model(big["input_ids"], big["attention_mask"]), big["labels"], weights
    )
    (reference_loss / n).backward()
    reference = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    assert reference

    model.zero_grad(set_to_none=True)
    group = [collate_batch(items[i : i + 2], pad_token_id=_PAD_ID) for i in range(0, n, 2)]
    total, count = accumulate_group(model, group, weights, _CPU, None, scaler)

    assert count == n
    assert float(total) == pytest.approx(float(reference_loss), rel=1e-5)
    accumulated = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert set(accumulated) == set(reference)
    for name, gradient in accumulated.items():
        assert torch.allclose(gradient, reference[name], atol=1e-5, rtol=1e-4), name


# --------------------------------------------------------------------------- #
# Full runs
# --------------------------------------------------------------------------- #


def test_training_reduces_loss_and_writes_all_outputs(tmp_path):
    summary = _run(tmp_path, _cfg(max_epochs=8))
    run_dir = tmp_path / "run"
    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert len(metrics) == 8
    assert metrics["train_loss"].iloc[-1] < metrics["train_loss"].iloc[0]
    for name in LEAF_NAMES:
        assert f"f1_{name}" in metrics.columns
    for filename in ("best.pt", "last.pt", "config.json", "summary.json", "val_report_best.csv"):
        assert (run_dir / filename).exists(), filename
    assert len(pd.read_csv(run_dir / "val_report_best.csv")) == NUM_LEAVES
    assert summary["finished"] is True
    assert summary["epochs_run"] == 8


def test_best_checkpoint_matches_best_validation_epoch(tmp_path):
    _run(tmp_path, _cfg(max_epochs=4))
    metrics = pd.read_csv(tmp_path / "run" / "metrics.csv")
    best_row = metrics.loc[metrics["val_macro_f1"].idxmax()]
    meta = load_best(tmp_path / "run" / "best.pt")["meta"]
    assert meta["best_epoch"] == int(best_row["epoch"])
    assert meta["val_macro_f1"] == pytest.approx(float(best_row["val_macro_f1"]), abs=1e-4)
    assert meta["label_names"] == list(LEAF_NAMES)
    assert meta["num_labels"] == NUM_LEAVES


def test_interrupted_run_resumes_to_the_same_result(tmp_path):
    cfg = _cfg(max_epochs=4)
    _run(tmp_path, cfg, name="full")

    interrupted = _run(tmp_path, cfg, stop_after_epoch=2, name="split")
    assert interrupted["finished"] is False
    assert interrupted["epochs_run"] == 2
    resumed = _run(tmp_path, cfg, resume=True, model=_model(99), name="split")
    assert resumed["finished"] is True
    assert resumed["epochs_run"] == 4

    columns = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"]
    full = pd.read_csv(tmp_path / "full" / "metrics.csv")[columns]
    split = pd.read_csv(tmp_path / "split" / "metrics.csv")[columns]
    pd.testing.assert_frame_equal(full, split, atol=1e-6)


def test_resume_refuses_a_changed_config(tmp_path):
    _run(tmp_path, _cfg(max_epochs=4), stop_after_epoch=1)
    with pytest.raises(ValueError, match="config"):
        _run(tmp_path, _cfg(max_epochs=4, learning_rate=5e-4), resume=True)


def test_non_finite_loss_aborts_training(tmp_path):
    model = _model()
    with torch.no_grad():
        model.classifier.weight.fill_(float("nan"))
    with pytest.raises(NonFiniteLossError):
        _run(tmp_path, _cfg(), model=model)


def test_resuming_a_finished_run_does_nothing(tmp_path):
    cfg = _cfg(max_epochs=3)
    _run(tmp_path, cfg)
    again = _run(tmp_path, cfg, resume=True)
    assert again["finished"] is True
    assert again["epochs_run"] == 3
    assert len(pd.read_csv(tmp_path / "run" / "metrics.csv")) == 3


def test_fresh_run_refuses_to_overwrite_an_existing_checkpoint(tmp_path):
    cfg = _cfg(max_epochs=2)
    _run(tmp_path, cfg)
    with pytest.raises(FileExistsError):
        _run(tmp_path, cfg)
