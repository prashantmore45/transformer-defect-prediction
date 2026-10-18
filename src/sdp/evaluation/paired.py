"""Paired comparison of systems scored on the same test files (Milestone 8.3).

Concepts
--------
Two systems scored on the SAME files make correlated mistakes: a hard file is hard for
both. Comparing two separate confidence intervals ignores that and is far too cautious. A
paired bootstrap instead recomputes the DIFFERENCE (reference minus other) on every
bootstrap draw, using the same resampled problems for both systems, and reads the
interval off the differences. If most of the spread cancels, the paired interval is
much narrower than either system's own.

Details
-------
* Draws come from ``sdp.evaluation.bootstrap.draw_group_counts`` with the same seed and
  number of problems as M8.2, so the reference system's own intervals here equal the
  ones in ``reports/m8/test_overall.csv``.
* ``frac_draws_not_ahead`` is the share of draws in which the reference did NOT beat the
  other system (difference <= 0). It is an informal bootstrap proportion, not a formal
  p-value.
* ``align_systems`` matches dumps by submission_id, never by row order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sdp.evaluation.bootstrap import (
    DEFAULT_LEVEL,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    draw_group_counts,
    group_confusions,
    interval,
    metrics_from_confusion,
    resample_confusions,
)
from sdp.evaluation.metrics import LEAF_NAMES

HEADLINE_METRICS: tuple[str, ...] = ("accuracy", "macro_f1")


@dataclass(frozen=True)
class PairedComparison:
    """Result of ``compare_systems``."""

    systems: pd.DataFrame  # system, metric, value, ci_low, ci_high
    differences: pd.DataFrame  # versus, metric, value, ci_low, ci_high, frac_draws_not_ahead
    per_class_differences: pd.DataFrame  # versus, class, f1_difference, ci_low, ci_high, ...
    n_files: int
    n_problems: int


def align_systems(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Line up several prediction dumps on submission_id.

    Each frame needs ``submission_id``, ``y_true`` and ``y_pred``. Returns the submission
    IDs (in the first frame's order), the true labels, and each system's predictions in
    that order. Duplicate IDs, differing ID sets or differing true labels raise.
    """
    names = list(frames)
    if not names:
        raise ValueError("no systems to align")
    first = frames[names[0]]
    if first["submission_id"].duplicated().any():
        raise ValueError(f"{names[0]}: duplicate submission_id values")
    ids = first["submission_id"].to_numpy()
    y_true = first["y_true"].to_numpy()

    predictions: dict[str, np.ndarray] = {}
    for name in names:
        frame = frames[name]
        if frame["submission_id"].duplicated().any():
            raise ValueError(f"{name}: duplicate submission_id values")
        if set(frame["submission_id"]) != set(ids):
            raise ValueError(f"{name}: submission_ids differ from {names[0]}")
        ordered = frame.set_index("submission_id").loc[ids]
        if not np.array_equal(ordered["y_true"].to_numpy(), y_true):
            raise ValueError(f"{name}: y_true differs from {names[0]}")
        predictions[name] = ordered["y_pred"].to_numpy()
    return ids, y_true, predictions


def compare_systems(
    y_true: object,
    predictions: Mapping[str, object],
    groups: object,
    *,
    reference: str,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    label_names: tuple[str, ...] = LEAF_NAMES,
) -> PairedComparison:
    """Per-system intervals plus paired 'reference minus other' differences.

    ``predictions`` maps system name to predicted class IDs (all aligned to ``y_true`` and
    ``groups``, one problem ID per file). Differences are computed for every system other
    than ``reference``.
    """
    if reference not in predictions:
        raise KeyError(f"reference system {reference!r} is not in predictions")
    if len(predictions) < 2:
        raise ValueError("need the reference and at least one other system")

    num_classes = len(label_names)
    matrices = {
        name: group_confusions(y_true, predicted, groups, num_classes)
        for name, predicted in predictions.items()
    }
    n_groups = matrices[reference].shape[0]
    counts = draw_group_counts(n_groups, n_boot, seed)  # the SAME draws for every system
    point = {name: metrics_from_confusion(m.sum(axis=0)) for name, m in matrices.items()}
    boot = {
        name: metrics_from_confusion(resample_confusions(counts, m)) for name, m in matrices.items()
    }

    system_rows: list[dict[str, object]] = []
    for name in predictions:
        for metric in HEADLINE_METRICS:
            low, high = interval(boot[name][metric], level)
            system_rows.append(
                {
                    "system": name,
                    "metric": metric,
                    "value": float(point[name][metric]),
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )

    difference_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    for name in predictions:
        if name == reference:
            continue
        for metric in HEADLINE_METRICS:
            draws = boot[reference][metric] - boot[name][metric]
            low, high = interval(draws, level)
            difference_rows.append(
                {
                    "versus": name,
                    "metric": metric,
                    "value": float(point[reference][metric] - point[name][metric]),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "frac_draws_not_ahead": float((draws <= 0).mean()),
                }
            )
        class_draws = boot[reference]["f1"] - boot[name]["f1"]  # (n_boot, num_classes)
        class_low, class_high = interval(class_draws, level)
        class_values = point[reference]["f1"] - point[name]["f1"]
        class_not_ahead = (class_draws <= 0).mean(axis=0)
        for index, label in enumerate(label_names):
            per_class_rows.append(
                {
                    "versus": name,
                    "class": label,
                    "f1_difference": float(class_values[index]),
                    "ci_low": float(class_low[index]),
                    "ci_high": float(class_high[index]),
                    "frac_draws_not_ahead": float(class_not_ahead[index]),
                }
            )

    return PairedComparison(
        systems=pd.DataFrame(system_rows),
        differences=pd.DataFrame(difference_rows),
        per_class_differences=pd.DataFrame(per_class_rows),
        n_files=int(np.asarray(y_true).shape[0]),
        n_problems=int(n_groups),
    )
