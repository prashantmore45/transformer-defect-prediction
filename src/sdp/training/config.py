"""Training configuration for the 9-class CodeBERT fine-tuning run (Milestone 7.2).

``configs/train.yaml`` is the single place every training value lives. This
module turns that YAML into a frozen, validated ``TrainConfig``:

* There are NO defaults in the class. Every key must appear in the YAML, so a
  value can never silently come from somewhere the reader cannot see.
* Unknown or missing keys are errors, so a typo (``learnin_rate``) fails
  immediately instead of quietly training with something unintended.
* Types are checked strictly (``True`` is not an integer; ``"2e-5"`` is not a
  float). PyYAML reads ``2e-5`` as TEXT because YAML 1.1 floats need a decimal
  point, so the error message tells you to write ``2.0e-5``.
* Ranges are checked at construction. The learning-rate floor of 1e-7 exists
  because the base paper's reported 1e-9 was judged an error (the model would
  barely move); that value must never be reused by accident.

This module deliberately does not import torch: loading and validating a
configuration must work (and be tested) without a GPU.
"""

import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Self

from sdp.config import PROJECT_ROOT, load_config

DEFAULT_TRAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"

#: Allowed values for ``TrainConfig.precision``.
PRECISIONS: tuple[str, ...] = ("bf16", "fp16", "fp32")

# Must start with a letter or digit so a run name can never be "." or "..".
_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _coerce(name: str, value: Any, expected: type) -> Any:
    """Return ``value`` if it has the expected type, else raise TypeError.

    ``bool`` is a subclass of ``int`` in Python, so it is rejected explicitly
    for int and float fields. An int is accepted for a float field and converted.
    """
    if expected is bool:
        if isinstance(value, bool):
            return value
    elif expected is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif expected is float:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    elif expected is str:
        if isinstance(value, str):
            return value

    hint = ""
    if expected is float and isinstance(value, str):
        hint = " (YAML reads 2e-5 as text; write it with a decimal point, e.g. 2.0e-5)"
    raise TypeError(
        f"{name} must be {expected.__name__}, got {type(value).__name__} {value!r}{hint}"
    )


@dataclass(frozen=True)
class TrainConfig:
    """Every value a training run needs. See configs/train.yaml for what each means."""

    run_name: str
    seed: int
    output_root: str
    cache_dir: str
    model_name: str
    dropout: float
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_epochs: int
    patience: int
    max_grad_norm: float
    micro_batch_size: int
    grad_accum_steps: int
    eval_batch_size: int
    precision: str
    gradient_checkpointing: bool
    class_weight_power: float

    def __post_init__(self) -> None:
        for name in ("run_name", "output_root", "cache_dir", "model_name"):
            _check(bool(getattr(self, name).strip()), f"{name} must not be empty")
        _check(
            _RUN_NAME_PATTERN.fullmatch(self.run_name) is not None,
            f"run_name {self.run_name!r} must start with a letter or digit and contain "
            "only letters, digits, '.', '_' and '-'",
        )
        _check(self.seed >= 0, f"seed must be >= 0, got {self.seed}")
        _check(0.0 <= self.dropout < 1.0, f"dropout must be in [0, 1), got {self.dropout}")
        _check(
            1e-7 <= self.learning_rate <= 1e-3,
            f"learning_rate must be in [1e-7, 1e-3], got {self.learning_rate} "
            "(the base paper's reported 1e-9 is below this range and was judged an error)",
        )
        _check(self.weight_decay >= 0.0, f"weight_decay must be >= 0, got {self.weight_decay}")
        _check(
            0.0 <= self.warmup_ratio <= 0.5,
            f"warmup_ratio must be in [0, 0.5], got {self.warmup_ratio}",
        )
        _check(self.max_epochs >= 1, f"max_epochs must be >= 1, got {self.max_epochs}")
        _check(self.patience >= 1, f"patience must be >= 1, got {self.patience}")
        _check(self.max_grad_norm > 0.0, f"max_grad_norm must be > 0, got {self.max_grad_norm}")
        _check(
            self.micro_batch_size >= 1,
            f"micro_batch_size must be >= 1, got {self.micro_batch_size}",
        )
        _check(
            self.grad_accum_steps >= 1,
            f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}",
        )
        _check(
            self.eval_batch_size >= 1,
            f"eval_batch_size must be >= 1, got {self.eval_batch_size}",
        )
        _check(
            self.precision in PRECISIONS,
            f"precision must be one of {PRECISIONS}, got {self.precision!r}",
        )
        _check(
            0.0 <= self.class_weight_power <= 1.0,
            f"class_weight_power must be in [0, 1], got {self.class_weight_power}",
        )

    @property
    def effective_batch_size(self) -> int:
        """Sequences contributing to each optimizer step."""
        return self.micro_batch_size * self.grad_accum_steps

    @property
    def cache_path(self) -> Path:
        """Token-cache directory (an absolute ``cache_dir`` is kept as is)."""
        return PROJECT_ROOT / self.cache_dir

    @property
    def run_dir(self) -> Path:
        """Where this run's checkpoints and logs go (git-ignored)."""
        return PROJECT_ROOT / self.output_root / self.run_name

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, e.g. to save the resolved config next to a run's results."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        """Build and validate a config from a mapping with exactly the right keys."""
        if not isinstance(raw, dict):
            raise TypeError(f"train config must be a mapping, got {type(raw).__name__}")
        names = {f.name for f in fields(cls)}
        unknown = sorted(str(key) for key in set(raw) - names)
        if unknown:
            raise ValueError(f"unknown train config keys: {unknown}")
        missing = sorted(names - set(raw))
        if missing:
            raise ValueError(f"missing train config keys: {missing}")
        values = {f.name: _coerce(f.name, raw[f.name], f.type) for f in fields(cls)}
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> Self:
        """Load ``configs/train.yaml`` (or ``path``) and validate it."""
        return cls.from_dict(load_config(path or DEFAULT_TRAIN_CONFIG_PATH))
