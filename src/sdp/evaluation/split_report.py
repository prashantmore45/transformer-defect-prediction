"""Everything the evaluation report needs for ONE split, from one set of bootstrap draws.

Milestone 8.2
-------------
``evaluate_split`` takes a prediction dump's labels plus a problem ID for every file and
returns:

* ``overall``: accuracy and macro-F1 with percentile intervals;
* ``per_class``: support, number of problems, precision / recall / F1 with intervals;
* ``parent``: parent accuracy, parent macro-F1, within-parent error share and its chance
  reference, with intervals (see ``sdp.evaluation.hierarchy``);
* ``confusion`` (9x9) and ``parent_confusion`` (4x4) as integer count matrices.

All intervals come from the SAME bootstrap draws: each draw picks a set of problems with
replacement, and every statistic is recomputed on that pretend test set. That keeps the
tables consistent with each other, and the draws can be reused for baselines (M8.3) so a
model-minus-baseline difference gets a paired interval.
"""

from __future__ import annotations

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
    resample_confusions,
    summarize_bootstrap,
)
from sdp.evaluation.hierarchy import aggregate_to_parents, parent_stats

#: Parent-level statistics, in report order.
PARENT_METRICS: tuple[str, ...] = (
    "parent_accuracy",
    "parent_macro_f1",
    "within_parent_error_share",
    "within_parent_error_share_if_uniform",
)


@dataclass(frozen=True)
class SplitEvaluation:
    """Result of ``evaluate_split``."""

    overall: pd.DataFrame
    per_class: pd.DataFrame
    parent: pd.DataFrame
    confusion: np.ndarray
    parent_confusion: np.ndarray
    n_files: int
    n_problems: int


def problem_ids_for(frame: pd.DataFrame, manifest: pd.DataFrame) -> np.ndarray:
    """Problem ID (as text) for each dump row, in the dump's row order.

    ``manifest`` needs ``submission_id`` and ``problem_id`` columns. Unknown submission IDs
    and duplicate manifest IDs raise instead of silently misaligning rows.
    """
    lookup = manifest.set_index("submission_id")["problem_id"]
    if not lookup.index.is_unique:
        raise ValueError("manifest contains duplicate submission_id values")
    missing = set(frame["submission_id"]) - set(lookup.index)
    if missing:
        raise KeyError(f"{len(missing)} dump submission_ids are not in the manifest")
    return lookup.loc[frame["submission_id"]].astype(str).to_numpy()


def evaluate_split(
    frame: pd.DataFrame,
    groups: object,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> SplitEvaluation:
    """Point estimates and problem-level bootstrap intervals for one split.

    ``frame`` needs ``y_true`` and ``y_pred`` columns; ``groups`` holds one problem ID per
    row of ``frame``.
    """
    missing = {"y_true", "y_pred"} - set(frame.columns)
    if missing:
        raise KeyError(f"frame is missing columns: {sorted(missing)}")
    y_true = frame["y_true"].to_numpy()
    y_pred = frame["y_pred"].to_numpy()

    group_cms = group_confusions(y_true, y_pred, groups)
    counts = draw_group_counts(group_cms.shape[0], n_boot, seed)
    overall, per_class = summarize_bootstrap(y_true, y_pred, groups, level=level, counts=counts)

    point_cm = group_cms.sum(axis=0)
    point = parent_stats(point_cm)
    draws = parent_stats(resample_confusions(counts, group_cms))
    rows = []
    for name in PARENT_METRICS:
        low, high = interval(draws[name], level)
        rows.append(
            {
                "metric": name,
                "value": float(point[name]),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )

    return SplitEvaluation(
        overall=overall,
        per_class=per_class,
        parent=pd.DataFrame(rows),
        confusion=point_cm,
        parent_confusion=aggregate_to_parents(point_cm),
        n_files=len(frame),
        n_problems=int(group_cms.shape[0]),
    )
