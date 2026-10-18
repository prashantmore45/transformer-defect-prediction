"""Problem-level bootstrap confidence intervals (Milestone 8.2).

Concepts
--------
* A single test score (say macro-F1 = 0.35) comes from one particular test set. A
  different draw of problems would give a slightly different score. The bootstrap
  estimates that spread: repeatedly rebuild a "pretend test set" by sampling WITH
  replacement from the real one, recompute the score each time, and read off the
  middle 95% of the results (the 2.5th to 97.5th percentile). This is the percentile
  bootstrap, the simplest standard variant.
* We resample whole PROBLEMS, not single files. Files from one problem share a
  statement and have similar solutions and bugs, so they are not independent.
  Resampling single files would pretend there is more independent evidence than there
  is, and the intervals would come out too narrow. (Statisticians call this a cluster
  bootstrap.)
* What the interval does NOT cover: training variability. There is one training run
  (seed 42), so the interval says how much the score could move with a different draw
  of test problems, not with a different training run.

How it is computed (fast and exact)
-----------------------------------
1. Build one confusion matrix per problem (shape: problems x classes x classes).
2. A bootstrap draw is a vector of counts: how many times each problem was picked.
   Summing the picked problems' matrices is then one matrix product (counts @ matrices).
3. Every metric is computed straight from the summed confusion matrix with numpy. The
   formulas equal ``sdp.evaluation.metrics`` (a test checks this), including the rule
   that every class is always scored: a class with no predictions or no support gets
   F1 = 0 and still counts in the macro average.

The draws (step 2) are a separate step on purpose: M8.3 reuses the SAME draws for the
model and for each baseline, so the model-minus-baseline difference gets a proper paired
interval instead of two overlapping separate ones.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sdp.data.taxonomy import NUM_LEAVES
from sdp.evaluation.metrics import LEAF_NAMES, confusion_matrix

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 42
DEFAULT_LEVEL = 0.95


def group_confusions(
    y_true: object,
    y_pred: object,
    groups: object,
    num_classes: int = NUM_LEAVES,
) -> np.ndarray:
    """One confusion matrix per group; shape (n_groups, num_classes, num_classes).

    Rows are true classes, columns are predicted classes (as in ``metrics.py``).
    Groups are returned in sorted order of their values.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    # Validates y_true / y_pred (1-D, integer, in range, equal length) and gives us
    # the whole-split matrix for a consistency check below.
    total = confusion_matrix(true, pred, num_classes)

    group_values = np.asarray(groups)
    if group_values.ndim != 1 or group_values.shape != true.shape:
        raise ValueError(
            f"groups must be 1-D with one entry per file ({true.shape[0]}), "
            f"got shape {group_values.shape}"
        )
    if pd.isna(group_values).any():
        raise ValueError("groups contains missing values")

    _, inverse = np.unique(group_values, return_inverse=True)
    n_groups = int(inverse.max()) + 1
    matrices = np.zeros((n_groups, num_classes, num_classes), dtype=np.int64)
    np.add.at(matrices, (inverse, true, pred), 1)

    if not np.array_equal(matrices.sum(axis=0), total):  # cannot happen; guards future edits
        raise AssertionError("per-group confusion matrices do not add up to the total")
    return matrices


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise division that gives 0 where the denominator is 0."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(np.shape(numerator), dtype=np.float64),
        where=denominator > 0,
    )


def metrics_from_confusion(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Accuracy, macro-F1 and per-class precision/recall/F1 from confusion matrices.

    ``cm`` has shape (..., K, K): one matrix, or a stack of them (leading axes are kept).
    Per-class results have shape (..., K); accuracy and macro_f1 have shape (...).
    F1 is computed as 2*TP / (support + predicted count), which equals the usual
    2PR/(P+R) and is 0 whenever the class has neither support nor predictions.
    """
    matrix = np.asarray(cm, dtype=np.float64)
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"expected square confusion matrices, got shape {matrix.shape}")

    true_positive = np.diagonal(matrix, axis1=-2, axis2=-1)
    support = matrix.sum(axis=-1)  # true count per class (row sums)
    predicted = matrix.sum(axis=-2)  # predicted count per class (column sums)
    total = matrix.sum(axis=(-2, -1))

    f1 = _safe_divide(2.0 * true_positive, support + predicted)
    return {
        "accuracy": _safe_divide(true_positive.sum(axis=-1), total),
        "macro_f1": f1.mean(axis=-1),
        "precision": _safe_divide(true_positive, predicted),
        "recall": _safe_divide(true_positive, support),
        "f1": f1,
    }


def draw_group_counts(
    n_groups: int, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED
) -> np.ndarray:
    """Bootstrap draws as counts: shape (n_boot, n_groups), each row sums to n_groups.

    Entry [b, g] is how many times group g was picked in draw b.
    """
    if n_groups < 1 or n_boot < 1:
        raise ValueError(f"need n_groups >= 1 and n_boot >= 1, got {n_groups}, {n_boot}")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_groups, size=(n_boot, n_groups))
    counts = np.zeros((n_boot, n_groups), dtype=np.float64)
    for draw in range(n_boot):
        counts[draw] = np.bincount(picks[draw], minlength=n_groups)
    return counts


def resample_confusions(counts: np.ndarray, group_cms: np.ndarray) -> np.ndarray:
    """Confusion matrix of each bootstrap draw; shape (n_boot, K, K)."""
    n_groups, num_classes, _ = group_cms.shape
    if counts.ndim != 2 or counts.shape[1] != n_groups:
        raise ValueError(f"counts must have shape (n_boot, {n_groups}), got {counts.shape}")
    flat = group_cms.reshape(n_groups, num_classes * num_classes).astype(np.float64)
    return (counts @ flat).reshape(counts.shape[0], num_classes, num_classes)


def interval(draws: np.ndarray, level: float = DEFAULT_LEVEL) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval along axis 0 (e.g. level 0.95 -> 2.5th and 97.5th percentiles)."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(draws, [tail, 100.0 - tail], axis=0)
    return low, high


def summarize_bootstrap(
    y_true: object,
    y_pred: object,
    groups: object,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    label_names: Sequence[str] = LEAF_NAMES,
    counts: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point estimates plus percentile intervals for the headline and per-class metrics.

    Returns ``(overall, per_class)``:

    * ``overall``: one row each for accuracy and macro_f1 (value, ci_low, ci_high).
    * ``per_class``: one row per class with support (files), problems (distinct groups
      containing the class), and precision / recall / f1, each with ``_lo`` and ``_hi``.

    Pass ``counts`` (from ``draw_group_counts``) to reuse the same draws for several
    prediction vectors; otherwise draws are generated from ``n_boot`` and ``seed``.
    """
    num_classes = len(label_names)
    group_cms = group_confusions(y_true, y_pred, groups, num_classes)
    if counts is None:
        counts = draw_group_counts(group_cms.shape[0], n_boot, seed)

    point_cm = group_cms.sum(axis=0)
    point = metrics_from_confusion(point_cm)
    boot = metrics_from_confusion(resample_confusions(counts, group_cms))

    overall_rows = []
    for name in ("accuracy", "macro_f1"):
        low, high = interval(boot[name], level)
        overall_rows.append(
            {
                "metric": name,
                "value": float(point[name]),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )

    per_class = pd.DataFrame(
        {
            "class": list(label_names),
            "support": point_cm.sum(axis=1).astype(int),
            "problems": (group_cms.sum(axis=2) > 0).sum(axis=0).astype(int),
        }
    )
    for name in ("precision", "recall", "f1"):
        low, high = interval(boot[name], level)
        per_class[name] = point[name]
        per_class[f"{name}_lo"] = low
        per_class[f"{name}_hi"] = high
    return pd.DataFrame(overall_rows), per_class
