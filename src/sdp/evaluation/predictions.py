"""Prediction dumps for evaluation (Milestone 8.1).

What this module is for
-----------------------
A "prediction dump" stores, for every file in one split, what the trained model
said: the 9 raw scores (logits) and the predicted class, next to the true class.
Every later M8 number (headline metrics, bootstrap intervals, confusion matrix,
slice analysis) is computed from this file, so the model runs over the test split
exactly once. Re-running inference for each analysis would make it easy to start
tuning against test results by accident.

What a dump row contains
------------------------
``submission_id``  join key back to the manifest (problem, verdict, input source, ...)
``y_true``         true class ID (checked against the manifest, never trusted blindly)
``y_pred``         argmax of the logits
``cached_length``  number of real tokens the model saw (at most 512)
``logit_<CLASS>``  the 9 raw scores, in class-ID order, float32

Logits are kept, not just the argmax, so probabilities, confidence and top-2 accuracy
can be derived later without touching the GPU again.

Safety checks (each one raises instead of producing a plausible wrong number)
-----------------------------------------------------------------------------
* file hashes (checkpoint, manifest) are compared with the recorded values;
* the checkpoint's label names and taxonomy version must match ``taxonomy.py``;
* the dump must cover exactly the manifest rows of the split, once each, with the
  same true labels;
* the validation split must reproduce the validation macro-F1 stored in the checkpoint
  (proves this inference path equals the training-time evaluation) before the test
  split is touched;
* a dump file is never overwritten.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sdp.data.taxonomy import TAXONOMY_VERSION
from sdp.evaluation.metrics import LEAF_NAMES, accuracy, macro_f1
from sdp.model.classifier import CodeBERTClassifier
from sdp.training.checkpoint import load_best

#: Logit column names in class-ID order.
LOGIT_COLUMNS: tuple[str, ...] = tuple(f"logit_{name}" for name in LEAF_NAMES)

#: Every column of a dump, in order.
DUMP_COLUMNS: tuple[str, ...] = (
    "submission_id",
    "y_true",
    "y_pred",
    "cached_length",
    *LOGIT_COLUMNS,
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks so large checkpoints do not fill memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    """Return the file's SHA-256, or raise if it differs from ``expected`` (case-insensitive)."""
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"{label} hash mismatch for {path}: expected {expected.lower()}, got {actual}"
        )
    return actual


def check_label_contract(meta: dict[str, Any]) -> None:
    """The checkpoint must have been trained against exactly this taxonomy."""
    if list(meta["label_names"]) != list(LEAF_NAMES):
        raise ValueError(
            f"checkpoint label order {meta['label_names']} differs from taxonomy {list(LEAF_NAMES)}"
        )
    if int(meta["num_labels"]) != len(LEAF_NAMES):
        raise ValueError(f"checkpoint has {meta['num_labels']} labels, expected {len(LEAF_NAMES)}")
    if meta["taxonomy_version"] != TAXONOMY_VERSION:
        raise ValueError(
            f"checkpoint taxonomy version {meta['taxonomy_version']!r} "
            f"differs from {TAXONOMY_VERSION!r}"
        )


def load_model(best_path: Path, device: torch.device) -> tuple[CodeBERTClassifier, dict[str, Any]]:
    """Rebuild the classifier from ``best.pt`` (offline, strict weight loading), in eval mode."""
    bundle = load_best(best_path)
    meta = bundle["meta"]
    check_label_contract(meta)
    model = CodeBERTClassifier.from_pretrained(
        meta["model_name"],
        num_labels=int(meta["num_labels"]),
        dropout=float(meta["dropout"]),
        local_files_only=True,
    )
    model.load_state_dict(bundle["model_state"])  # strict: any missing/extra key raises
    model.to(device).eval()
    return model, meta


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
) -> pd.DataFrame:
    """Run the model over ``loader`` once and return one dump row per file, in loader order."""
    model.eval()
    submission_ids: list[str] = []
    labels: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    logit_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(input_ids, attention_mask).float()
            logit_chunks.append(logits.cpu().numpy())
            labels.append(batch["labels"].numpy())
            lengths.append(batch["attention_mask"].sum(dim=1).numpy())
            submission_ids.extend(batch["submission_ids"])

    logits_all = np.concatenate(logit_chunks).astype(np.float32)
    frame = pd.DataFrame(
        {
            "submission_id": submission_ids,
            "y_true": np.concatenate(labels).astype(np.int64),
            "y_pred": logits_all.argmax(axis=1).astype(np.int64),
            "cached_length": np.concatenate(lengths).astype(np.int64),
        }
    )
    for index, column in enumerate(LOGIT_COLUMNS):
        frame[column] = logits_all[:, index]
    return frame[list(DUMP_COLUMNS)]


def validate_against_manifest(frame: pd.DataFrame, manifest: pd.DataFrame, split: str) -> None:
    """The dump must be exactly the manifest's rows for ``split``, once each, same labels."""
    rows = manifest[manifest["split"] == split]
    if len(rows) == 0:
        raise ValueError(f"manifest has no rows for split {split!r}")
    if len(frame) != len(rows):
        raise ValueError(f"dump has {len(frame)} rows, manifest split {split!r} has {len(rows)}")
    if frame["submission_id"].duplicated().any():
        raise ValueError("dump contains duplicate submission_id values")
    if set(frame["submission_id"]) != set(rows["submission_id"]):
        raise ValueError(f"dump submission_ids differ from manifest split {split!r}")
    expected = rows.set_index("submission_id")["leaf_id"].loc[frame["submission_id"]].to_numpy()
    if not np.array_equal(expected, frame["y_true"].to_numpy()):
        raise ValueError("dump y_true differs from the manifest leaf_id for some rows")


def check_reproduction(
    frame: pd.DataFrame, expected_macro_f1: float, tolerance: float = 0.002
) -> dict[str, float | bool]:
    """Compare the dump's macro-F1 with the value recorded at training time."""
    y_true = frame["y_true"].to_numpy()
    y_pred = frame["y_pred"].to_numpy()
    got = macro_f1(y_true, y_pred)
    return {
        "macro_f1": got,
        "accuracy": accuracy(y_true, y_pred),
        "expected_macro_f1": float(expected_macro_f1),
        "abs_difference": abs(got - float(expected_macro_f1)),
        "tolerance": tolerance,
        "passed": abs(got - float(expected_macro_f1)) <= tolerance,
    }


def write_dump(frame: pd.DataFrame, path: Path) -> None:
    """Write a dump atomically; refuse to overwrite an existing file (one-time rule)."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; dumps are written once and never replaced")
    if list(frame.columns) != list(DUMP_COLUMNS):
        raise ValueError("dump columns differ from DUMP_COLUMNS")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def read_dump(path: Path) -> pd.DataFrame:
    """Read a dump and check its columns."""
    frame = pd.read_parquet(path)
    if list(frame.columns) != list(DUMP_COLUMNS):
        raise ValueError(f"{path} has unexpected columns: {list(frame.columns)}")
    return frame
