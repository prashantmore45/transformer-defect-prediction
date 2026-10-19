"""Descriptive tables for the M8.4 slice analysis (no bootstrap).

* ``confidence_by_predicted_class``: for each class the model predicts, how many predictions,
  how confident (mean top probability), how close a call (median margin between the top two
  probabilities, and the share of predictions with a margin below ``low_margin``), and how
  often right (precision). Tests the "LOGICAL is the default class" hypothesis.
* ``predicted_parent_shares``: for each slice, the share of its files predicted under each
  parent class (ERROR_FREE, COMPILE_ERROR, RUNTIME_ERROR, LOGICAL): where the misses go.
* ``calibration_table``: per slice, mean confidence against accuracy; a model is
  well calibrated when the two agree. ``overconfidence`` is confidence minus accuracy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from sdp.data.taxonomy import COARSE_ORDER, NUM_COARSE
from sdp.evaluation.hierarchy import parent_ids
from sdp.evaluation.metrics import LEAF_NAMES


def _slice_mask(mask: object, n_files: int, name: str) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (n_files,):
        raise ValueError(f"slice {name!r}: mask must have one entry per file (length {n_files})")
    if not selected.any():
        raise ValueError(f"slice {name!r} is empty")
    return selected


def confidence_by_predicted_class(
    y_true: object,
    y_pred: object,
    max_prob: object,
    margin: object,
    label_names: Sequence[str] = LEAF_NAMES,
    low_margin: float = 0.10,
) -> pd.DataFrame:
    """One row per predicted class that occurs, plus a final '(all predictions)' row."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    top = np.asarray(max_prob, dtype=np.float64)
    gap = np.asarray(margin, dtype=np.float64)
    if true.ndim != 1 or true.size == 0 or not (true.shape == pred.shape == top.shape == gap.shape):
        raise ValueError("inputs must be non-empty 1-D arrays of the same length")

    def summarize(label: str, selected: np.ndarray) -> dict[str, object]:
        return {
            "predicted_class": label,
            "n_predicted": int(selected.sum()),
            "share_of_predictions": float(selected.mean()),
            "mean_max_prob": float(top[selected].mean()),
            "median_margin": float(np.median(gap[selected])),
            "share_low_margin": float((gap[selected] < low_margin).mean()),
            "precision": float((true[selected] == pred[selected]).mean()),
        }

    rows = []
    for index, label in enumerate(label_names):
        selected = pred == index
        if selected.any():
            rows.append(summarize(label, selected))
    rows.append(summarize("(all predictions)", np.ones(pred.shape, dtype=bool)))
    return pd.DataFrame(rows)


def predicted_parent_shares(y_pred: object, slices: Mapping[str, object]) -> pd.DataFrame:
    """Per slice, the share of files whose predicted class falls under each parent class."""
    pred = np.asarray(y_pred)
    parents = parent_ids()
    rows = []
    for name, mask in slices.items():
        selected = _slice_mask(mask, pred.shape[0], name)
        counts = np.bincount(parents[pred[selected]], minlength=NUM_COARSE)
        row: dict[str, object] = {"slice": name, "n_files": int(selected.sum())}
        for index, parent in enumerate(COARSE_ORDER):
            row[f"pred_{parent.value}"] = float(counts[index] / selected.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(
    max_prob: object, correct: object, slices: Mapping[str, object]
) -> pd.DataFrame:
    """Per slice: mean confidence, accuracy, and their difference (overconfidence)."""
    confidence = np.asarray(max_prob, dtype=np.float64)
    right = np.asarray(correct, dtype=bool)
    if confidence.ndim != 1 or confidence.shape != right.shape:
        raise ValueError("max_prob and correct must be 1-D arrays of the same length")
    rows = []
    for name, mask in slices.items():
        selected = _slice_mask(mask, confidence.shape[0], name)
        mean_confidence = float(confidence[selected].mean())
        accuracy = float(right[selected].mean())
        rows.append(
            {
                "slice": name,
                "n_files": int(selected.sum()),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "overconfidence": mean_confidence - accuracy,
            }
        )
    return pd.DataFrame(rows)
