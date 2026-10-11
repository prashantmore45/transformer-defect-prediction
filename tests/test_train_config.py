"""Tests for sdp.training.config. The real configs/train.yaml is also checked."""

from dataclasses import FrozenInstanceError

import pytest
import yaml

from sdp.config import PROJECT_ROOT
from sdp.training.config import TrainConfig


def _valid() -> dict[str, object]:
    return {
        "run_name": "test-run",
        "seed": 42,
        "output_root": "runs",
        "cache_dir": "data/processed/tokenized_9class",
        "model_name": "microsoft/codebert-base",
        "dropout": 0.1,
        "learning_rate": 2.0e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_epochs": 3,
        "patience": 2,
        "max_grad_norm": 1.0,
        "micro_batch_size": 2,
        "grad_accum_steps": 8,
        "eval_batch_size": 8,
        "precision": "bf16",
        "gradient_checkpointing": False,
        "class_weight_power": 0.5,
    }


def test_committed_train_yaml_is_valid():
    cfg = TrainConfig.from_yaml()
    assert cfg.effective_batch_size >= 1


def test_to_dict_round_trip():
    cfg = TrainConfig.from_yaml()
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg


def test_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig.from_dict({**_valid(), "learnin_rate": 1.0})


def test_rejects_missing_key():
    raw = _valid()
    del raw["seed"]
    with pytest.raises(ValueError, match="missing"):
        TrainConfig.from_dict(raw)


def test_rejects_string_float_with_yaml_hint():
    with pytest.raises(TypeError, match="decimal point"):
        TrainConfig.from_dict({**_valid(), "learning_rate": "2e-5"})


def test_rejects_bool_for_int_field():
    with pytest.raises(TypeError, match="micro_batch_size"):
        TrainConfig.from_dict({**_valid(), "micro_batch_size": True})


def test_accepts_int_for_float_field_and_converts():
    cfg = TrainConfig.from_dict({**_valid(), "dropout": 0, "warmup_ratio": 0})
    assert cfg.dropout == 0.0
    assert isinstance(cfg.dropout, float)
    assert isinstance(cfg.warmup_ratio, float)


@pytest.mark.parametrize("learning_rate", [1e-9, 1e-2])
def test_learning_rate_out_of_range(learning_rate):
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig.from_dict({**_valid(), "learning_rate": learning_rate})


def test_rejects_unknown_precision():
    with pytest.raises(ValueError, match="precision"):
        TrainConfig.from_dict({**_valid(), "precision": "fp8"})


@pytest.mark.parametrize("run_name", ["has space", "../escape", "..", ""])
def test_rejects_unsafe_run_name(run_name):
    with pytest.raises(ValueError):
        TrainConfig.from_dict({**_valid(), "run_name": run_name})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("micro_batch_size", 0),
        ("grad_accum_steps", 0),
        ("eval_batch_size", 0),
        ("max_epochs", 0),
        ("patience", 0),
        ("seed", -1),
    ],
)
def test_rejects_bad_integer_settings(field, value):
    with pytest.raises(ValueError, match=field):
        TrainConfig.from_dict({**_valid(), field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dropout", 1.0),
        ("dropout", -0.1),
        ("warmup_ratio", 0.6),
        ("class_weight_power", 1.5),
        ("class_weight_power", -0.1),
    ],
)
def test_rejects_out_of_range_fractions(field, value):
    with pytest.raises(ValueError, match=field):
        TrainConfig.from_dict({**_valid(), field: value})


def test_effective_batch_size_is_micro_batch_times_accumulation():
    cfg = TrainConfig.from_dict({**_valid(), "micro_batch_size": 2, "grad_accum_steps": 8})
    assert cfg.effective_batch_size == 16


def test_paths_resolve_under_project_root():
    cfg = TrainConfig.from_dict(_valid())
    assert cfg.run_dir == PROJECT_ROOT / "runs" / "test-run"
    assert cfg.cache_path == PROJECT_ROOT / "data" / "processed" / "tokenized_9class"


def test_absolute_cache_dir_is_kept(tmp_path):
    cfg = TrainConfig.from_dict({**_valid(), "cache_dir": str(tmp_path)})
    assert cfg.cache_path == tmp_path


def test_config_is_frozen():
    cfg = TrainConfig.from_dict(_valid())
    with pytest.raises(FrozenInstanceError):
        cfg.seed = 1


def test_from_yaml_reads_a_file(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(_valid()), encoding="utf-8")
    assert TrainConfig.from_yaml(path) == TrainConfig.from_dict(_valid())


def test_from_yaml_rejects_non_mapping(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        TrainConfig.from_yaml(path)
