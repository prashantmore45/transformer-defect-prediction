"""Checkpoint files for training runs (Milestone 7.4).

Two kinds of file, both written atomically (to a temporary file, then renamed), so a
power cut or crash mid-save can never leave a half-written checkpoint behind:

* ``last.pt``: everything needed to resume after an interruption (model, optimizer,
  learning-rate schedule, gradient scaler, epoch counter, early-stopping state, the
  config, and the metrics history). Written after every epoch, overwriting the last one.
* ``best.pt``: the model weights of the best epoch plus a ``meta`` dictionary that makes
  the file self-describing (label names in class-ID order, taxonomy version, model name,
  config, best epoch and its validation macro-F1), so evaluation (M8) and serving (M9)
  can load it without knowing anything about the training code.

Files are loaded with ``weights_only=True``, which refuses to run arbitrary pickled code.
That works because everything saved here is tensors and plain Python values.
"""

import os
from pathlib import Path
from typing import Any

import torch


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_last(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    stopper: dict[str, Any],
    config: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    """Write the full resumable state after ``epoch`` epochs have completed."""
    _atomic_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "stopper": stopper,
            "config": config,
            "history": history,
        },
        path,
    )


def load_last(path: Path) -> dict[str, Any]:
    """Read a ``last.pt`` file (tensors are loaded onto the CPU)."""
    return torch.load(path, map_location="cpu", weights_only=True)


def save_best(path: Path, model_state: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    """Write the best epoch's weights plus the metadata that describes them."""
    _atomic_save({"model_state": model_state, "meta": meta}, path)


def load_best(path: Path) -> dict[str, Any]:
    """Read a ``best.pt`` file: a dict with ``model_state`` and ``meta``."""
    return torch.load(path, map_location="cpu", weights_only=True)
