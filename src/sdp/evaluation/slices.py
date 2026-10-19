"""Slice analysis: performance on subsets of the test files (Milestone 8.4).

Concepts
--------
A SLICE is a subset of the test files (for example "files of at most 128 tokens", or
"files whose author also appears in train"). For each slice we report accuracy, per-class
recall and a macro-recall, each with a problem-level bootstrap interval, and optionally the
DIFFERENCE between two slices with its own interval.

Shared problem universe
-----------------------
Every slice is bootstrapped over the SAME list of test problems with the SAME draws. In a
draw, a slice's confusion matrix is built from its own files inside the resampled problems.
This matters for differences: the same problem can contribute files to both slices, and a
paired interval accounts for that, exactly as in ``sdp.evaluation.paired``. With all files
in one slice, the draws (and so the intervals) equal ``summarize_bootstrap``'s.

Metrics
-------
Inside a slice a class may be missing or tiny, and macro-F1 would score absent classes as 0.
So slices report accuracy and per-class recall (with the file count), and a macro-recall:
the mean recall over the classes with at least ``min_class_support`` files in the slice
(decided from the point estimate, so the class set is fixed across draws). Where a quantity
is undefined in a draw (no files, or a class with no files) it is NaN and the percentile
interval ignores that draw. A difference's macro-recall uses the classes that qualify in
BOTH slices.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sdp.data.taxonomy import NUM_LEAVES
from sdp.evaluation.bootstrap import (
    DEFAULT_LEVEL,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    draw_group_counts,
    resample_confusions,
)
from sdp.evaluation.metrics import LEAF_NAMES, confusion_matrix

DEFAULT_MIN_CLASS_SUPPORT = 20

OVERALL_COLUMNS = [
    "slice",
    "n_files",
    "n_problems",
    "accuracy",
    "accuracy_lo",
    "accuracy_hi",
    "macro_recall",
    "macro_recall_lo",
    "macro_recall_hi",
    "n_classes_scored",
]
PER_CLASS_COLUMNS = ["slice", "class", "support", "recall", "recall_lo", "recall_hi"]
DIFFERENCE_COLUMNS = ["first", "second", "metric", "value", "ci_low", "ci_high"]


@dataclass(frozen=True)
class SliceResults:
    """Result of ``bootstrap_slices``."""

    overall: pd.DataFrame
    per_class: pd.DataFrame
    differences: pd.DataFrame


def interval_masks(
    values: object, edges: Sequence[float], labels: Sequence[str] | None = None
) -> dict[str, np.ndarray]:
    """Boolean masks for right-closed bins (edges[i], edges[i+1]]; values outside get none."""
    array = np.asarray(values, dtype=np.float64)
    bounds = np.asarray(edges, dtype=np.float64)
    if bounds.ndim != 1 or bounds.size < 2 or np.any(np.diff(bounds) <= 0):
        raise ValueError("edges must be strictly increasing with at least two entries")
    names = (
        list(labels)
        if labels is not None
        else [f"({bounds[i]:g}, {bounds[i + 1]:g}]" for i in range(bounds.size - 1)]
    )
    if len(names) != bounds.size - 1:
        raise ValueError(f"need {bounds.size - 1} labels for {bounds.size} edges")
    return {name: (array > bounds[i]) & (array <= bounds[i + 1]) for i, name in enumerate(names)}


def softmax_confidence(logits: object) -> tuple[np.ndarray, np.ndarray]:
    """Top probability and the margin (top minus second probability) for each row of logits."""
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("logits must be 2-D with at least two classes")
    scores = scores - scores.max(axis=1, keepdims=True)  # stabilises the exponentials
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    top_two = np.sort(probabilities, axis=1)[:, -2:]
    return top_two[:, 1], top_two[:, 1] - top_two[:, 0]


def common_classes(
    y_true: object,
    slices: Mapping[str, object],
    min_class_support: int = DEFAULT_MIN_CLASS_SUPPORT,
    num_classes: int = NUM_LEAVES,
) -> list[int]:
    """Class IDs with at least ``min_class_support`` files in EVERY slice."""
    true = np.asarray(y_true)
    counts = np.stack(
        [
            np.bincount(true[np.asarray(mask, dtype=bool)], minlength=num_classes)
            for mask in slices.values()
        ]
    )
    return np.flatnonzero((counts >= min_class_support).all(axis=0)).tolist()


def _rates(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Accuracy, per-class recall and support from confusion matrices (..., K, K);
    NaN if undefined."""
    matrix = np.asarray(cm, dtype=np.float64)
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    support = matrix.sum(axis=-1)
    total = support.sum(axis=-1)
    accuracy = np.where(total > 0, diagonal.sum(axis=-1) / np.maximum(total, 1.0), np.nan)
    recall = np.where(support > 0, diagonal / np.maximum(support, 1.0), np.nan)
    return {"accuracy": accuracy, "recall": recall, "support": support}


def _macro(recall: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Mean recall over ``classes`` (NaN-aware); NaN if no class qualifies."""
    if classes.size == 0:
        return np.full(recall.shape[:-1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(recall[..., classes], axis=-1)


def _interval(draws: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval along axis 0, ignoring NaN draws."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    tail = (1.0 - level) / 2.0 * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        low, high = np.nanpercentile(draws, [tail, 100.0 - tail], axis=0)
    return low, high


def _difference_row(
    first: str, second: str, metric: str, value: float, draws: np.ndarray, level: float
) -> dict[str, object]:
    low, high = _interval(draws, level)
    return {
        "first": first,
        "second": second,
        "metric": metric,
        "value": float(value),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def bootstrap_slices(
    y_true: object,
    y_pred: object,
    groups: object,
    slices: Mapping[str, object],
    *,
    pairs: Sequence[tuple[str, str]] = (),
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    min_class_support: int = DEFAULT_MIN_CLASS_SUPPORT,
    macro_classes: Sequence[int] | None = None,
    label_names: Sequence[str] = LEAF_NAMES,
) -> SliceResults:
    """Per-slice accuracy / recall with problem-bootstrap intervals, plus slice differences.

    ``slices`` maps a name to a boolean mask with one entry per file (slices may overlap and
    need not cover every file). ``pairs`` lists (first, second) slice names; each gives the
    difference ``first - second`` for accuracy, macro-recall and the per-class recall of
    classes present in both.

    If ``macro_classes`` is given, macro-recall uses exactly those classes in every slice
    and in every difference (a class with no files in a slice is skipped there, not scored
    as 0), which makes macro-recall comparable across slices. Otherwise each slice uses the
    classes with at least ``min_class_support`` files, and a difference uses the classes
    that qualify in both.
    """
    if not slices:
        raise ValueError("no slices given")
    num_classes = len(label_names)
    fixed_classes: np.ndarray | None = None
    if macro_classes is not None:
        fixed_classes = np.unique(np.asarray(list(macro_classes), dtype=np.int64))
        if fixed_classes.size == 0:
            raise ValueError("macro_classes must not be empty")
        if fixed_classes.min() < 0 or fixed_classes.max() >= num_classes:
            raise ValueError(f"macro_classes must be class IDs in [0, {num_classes})")
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    confusion_matrix(true, pred, num_classes)  # validates: 1-D, integer, in range, same length
    group_values = np.asarray(groups)
    if group_values.shape != true.shape:
        raise ValueError("groups must have one entry per file")
    if pd.isna(group_values).any():
        raise ValueError("groups contains missing values")
    universe, group_index = np.unique(group_values, return_inverse=True)
    n_groups = universe.shape[0]

    matrices: dict[str, np.ndarray] = {}
    for name, mask in slices.items():
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != true.shape:
            raise ValueError(
                f"slice {name!r}: mask must have one entry per file (length {true.shape[0]})"
            )
        if not selected.any():
            raise ValueError(f"slice {name!r} is empty")
        cms = np.zeros((n_groups, num_classes, num_classes), dtype=np.int64)
        np.add.at(cms, (group_index[selected], true[selected], pred[selected]), 1)
        matrices[name] = cms
    for first, second in pairs:
        for name in (first, second):
            if name not in matrices:
                raise KeyError(f"pair refers to missing slice {name!r}")

    counts = draw_group_counts(n_groups, n_boot, seed)  # the SAME draws for every slice
    point = {name: _rates(cms.sum(axis=0)) for name, cms in matrices.items()}
    draws = {name: _rates(resample_confusions(counts, cms)) for name, cms in matrices.items()}

    overall_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for name, cms in matrices.items():
        rates, boot = point[name], draws[name]
        if fixed_classes is not None:
            scored = fixed_classes
        else:
            scored = np.flatnonzero(rates["support"] >= min_class_support)
        accuracy_lo, accuracy_hi = _interval(boot["accuracy"], level)
        macro_lo, macro_hi = _interval(_macro(boot["recall"], scored), level)
        overall_rows.append(
            {
                "slice": name,
                "n_files": int(rates["support"].sum()),
                "n_problems": int((cms.sum(axis=(1, 2)) > 0).sum()),
                "accuracy": float(rates["accuracy"]),
                "accuracy_lo": float(accuracy_lo),
                "accuracy_hi": float(accuracy_hi),
                "macro_recall": float(_macro(rates["recall"], scored)),
                "macro_recall_lo": float(macro_lo),
                "macro_recall_hi": float(macro_hi),
                "n_classes_scored": int(scored.size),
            }
        )
        recall_lo, recall_hi = _interval(boot["recall"], level)
        for position, label in enumerate(label_names):
            if rates["support"][position] > 0:
                class_rows.append(
                    {
                        "slice": name,
                        "class": label,
                        "support": int(rates["support"][position]),
                        "recall": float(rates["recall"][position]),
                        "recall_lo": float(recall_lo[position]),
                        "recall_hi": float(recall_hi[position]),
                    }
                )

    difference_rows: list[dict[str, object]] = []
    for first, second in pairs:
        a, b = point[first], point[second]
        da, db = draws[first], draws[second]
        shared = np.flatnonzero(
            (a["support"] >= min_class_support) & (b["support"] >= min_class_support)
        )
        difference_rows.append(
            _difference_row(
                first,
                second,
                "accuracy",
                a["accuracy"] - b["accuracy"],
                da["accuracy"] - db["accuracy"],
                level,
            )
        )
        difference_rows.append(
            _difference_row(
                first,
                second,
                "macro_recall",
                _macro(a["recall"], shared) - _macro(b["recall"], shared),
                _macro(da["recall"], shared) - _macro(db["recall"], shared),
                level,
            )
        )
        for position, label in enumerate(label_names):
            if a["support"][position] > 0 and b["support"][position] > 0:
                difference_rows.append(
                    _difference_row(
                        first,
                        second,
                        f"recall:{label}",
                        a["recall"][position] - b["recall"][position],
                        da["recall"][:, position] - db["recall"][:, position],
                        level,
                    )
                )

    return SliceResults(
        overall=pd.DataFrame(overall_rows, columns=OVERALL_COLUMNS),
        per_class=pd.DataFrame(class_rows, columns=PER_CLASS_COLUMNS),
        differences=pd.DataFrame(difference_rows, columns=DIFFERENCE_COLUMNS),
    )
