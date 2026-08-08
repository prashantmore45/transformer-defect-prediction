"""Leakage-safe dataset splitting.

The problem this module exists to solve
--------------------------------------
Competitive-programming corpora contain resubmission chains: a user submits,
receives a verdict, edits a few characters, and resubmits. Row 0 and row 1 of
the labelled manifest are exactly this — the same user, the same problem, 130
seconds apart, identical code size, verdicts `Compile Error` and `Accepted`.

Under a random submission-level split those two near-identical files land on
opposite sides of the train/test boundary roughly half the time. A model that
memorises the problem then scores well without learning anything about defects,
and the reported accuracy is fiction. 65.6% of candidate submissions belong to
such chains.

Splitting at the `problem_id` level eliminates this by construction: every
submission to a problem goes to the same split, so no chain can straddle the
boundary.

Why allocation is not simply random
-----------------------------------
Problem sizes are severely right-skewed (median 357 submissions, maximum 39,830;
the ten largest problems hold 3.6% of the corpus). Assigning problems uniformly
at random therefore gives no control over the proportion of *submissions* in
each split — the train fraction varies by roughly 1.5 percentage points across
seeds.

`allocate_problems` uses longest-processing-time-first greedy allocation:
problems are shuffled with the seed, sorted largest-first, then each is given to
whichever split is furthest below its submission target. The large problems are
placed early and the thousands of small ones fill the residual gaps, driving the
deviation well below 0.1 percentage points while remaining seed-dependent.

The comparison arm
------------------
`submission_level_split` performs the deliberately-unsafe random split. It is
not a fallback: both splits are used, and the macro-F1 gap between them
quantifies how much a naive split inflates results. That gap is a reported
finding, not a bug.

Freezing
--------
Once a model has been trained, the seed and ratios must not change. Re-splitting
invalidates every metric, figure, and checkpoint already produced. Task 11
records both in `split_config.json` alongside content hashes.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd


class Split(StrEnum):
    """The three dataset partitions."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


SPLIT_ORDER: Final[tuple[Split, ...]] = (Split.TRAIN, Split.VAL, Split.TEST)

DEFAULT_RATIOS: Final[Mapping[Split, float]] = MappingProxyType(
    {Split.TRAIN: 0.6, Split.VAL: 0.2, Split.TEST: 0.2}
)

DEFAULT_SEED: Final[int] = 42


def _validate_ratios(ratios: Mapping[Split, float]) -> None:
    if set(ratios) != set(SPLIT_ORDER):
        raise ValueError(f"ratios must cover exactly {[s.value for s in SPLIT_ORDER]}")
    if any(r <= 0 for r in ratios.values()):
        raise ValueError(f"all ratios must be positive; got {dict(ratios)}")
    total = sum(ratios.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"ratios must sum to 1.0; got {total}")


# --------------------------------------------------------------------------- #
# Core allocation — pure, no pandas, directly testable
# --------------------------------------------------------------------------- #


def allocate_problems(
    sizes: Mapping[str, int],
    *,
    ratios: Mapping[Split, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Split]:
    """Assign each problem to one split, balancing submission counts.

    Largest-first greedy allocation: each problem goes to the split with the
    largest remaining deficit against its target. Problems are shuffled with
    `seed` before sorting, so equal-sized problems land differently per seed
    while the balance guarantee is unaffected.

    Args:
        sizes: problem_id -> number of submissions. Must be non-empty, and every
            size must be positive.
        ratios: target share of *submissions* (not problems) per split.
        seed: controls the pre-sort shuffle. Freeze this once training begins.

    Returns:
        problem_id -> Split, covering every key in `sizes` exactly once.

    Raises:
        ValueError: on empty input, non-positive sizes, invalid ratios, or if
            any split would receive no problems at all.
    """
    _validate_ratios(ratios)
    if not sizes:
        raise ValueError("sizes is empty; nothing to allocate")
    if any(n <= 0 for n in sizes.values()):
        bad = sorted(p for p, n in sizes.items() if n <= 0)[:5]
        raise ValueError(f"sizes must be positive; offending problems: {bad}")
    if len(sizes) < len(SPLIT_ORDER):
        raise ValueError(
            f"need at least {len(SPLIT_ORDER)} problems to fill every split; " f"got {len(sizes)}"
        )

    total = sum(sizes.values())
    targets = {split: ratios[split] * total for split in SPLIT_ORDER}
    allocated = {split: 0 for split in SPLIT_ORDER}

    # Sort keys first so that dict insertion order cannot affect the result,
    # then shuffle with the seed, then sort largest-first. Python's sort is
    # stable, so the shuffled order survives among equal sizes.
    rng = np.random.default_rng(seed)
    problems = sorted(sizes)
    order = [problems[i] for i in rng.permutation(len(problems))]
    order.sort(key=lambda p: sizes[p], reverse=True)

    assignment: dict[str, Split] = {}
    for problem in order:
        # Ties break by SPLIT_ORDER because max() keeps the first maximum.
        split = max(SPLIT_ORDER, key=lambda s: targets[s] - allocated[s])
        assignment[problem] = split
        allocated[split] += sizes[problem]

    empty = [s.value for s in SPLIT_ORDER if not any(v is s for v in assignment.values())]
    if empty:
        raise ValueError(f"splits received no problems: {empty}")

    return assignment


# --------------------------------------------------------------------------- #
# pandas wrappers
# --------------------------------------------------------------------------- #


def _as_split_series(values: Sequence[str], index: pd.Index) -> pd.Series:
    """Build an ordered categorical Series so groupby and plots stay in order."""
    return pd.Series(
        pd.Categorical(values, categories=[s.value for s in SPLIT_ORDER], ordered=True),
        index=index,
        name="split",
    )


def problem_level_split(
    problem_ids: pd.Series,
    *,
    ratios: Mapping[Split, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> pd.Series:
    """Leakage-safe split. Every submission to a problem shares one split.

    Args:
        problem_ids: one entry per submission; may be categorical dtype.

    Returns:
        Ordered categorical Series of split names, aligned to `problem_ids`.
    """
    counts = problem_ids.value_counts()
    counts = counts[counts > 0]  # categorical dtype reports unused categories
    sizes = {str(p): int(n) for p, n in counts.items()}

    assignment = allocate_problems(sizes, ratios=ratios, seed=seed)
    mapped = problem_ids.astype(str).map({p: s.value for p, s in assignment.items()})
    return _as_split_series(mapped.to_numpy(), problem_ids.index)


def submission_level_split(
    index: pd.Index,
    *,
    ratios: Mapping[Split, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> pd.Series:
    """Deliberately-unsafe random split, used only as a comparison arm.

    Ignores `problem_id`, so resubmission chains straddle the boundary. The
    macro-F1 gap against `problem_level_split` measures the leakage effect and
    is reported as a finding.
    """
    _validate_ratios(ratios)
    n = len(index)
    if n < len(SPLIT_ORDER):
        raise ValueError(f"need at least {len(SPLIT_ORDER)} rows; got {n}")

    n_train = int(round(n * ratios[Split.TRAIN]))
    n_val = int(round(n * ratios[Split.VAL]))
    labels = np.array(
        [Split.TRAIN.value] * n_train
        + [Split.VAL.value] * n_val
        + [Split.TEST.value] * (n - n_train - n_val),
        dtype=object,
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    return _as_split_series(labels, index)


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


def problem_overlap(problem_ids: pd.Series, splits: pd.Series) -> dict[str, int]:
    """Count problems appearing in more than one split. All values must be 0."""
    by_split = {
        split.value: set(problem_ids[splits == split.value].astype(str)) for split in SPLIT_ORDER
    }
    pairs = [
        (Split.TRAIN, Split.VAL),
        (Split.TRAIN, Split.TEST),
        (Split.VAL, Split.TEST),
    ]
    return {f"{a.value}&{b.value}": len(by_split[a.value] & by_split[b.value]) for a, b in pairs}


def assert_problem_disjoint(problem_ids: pd.Series, splits: pd.Series) -> None:
    """Raise unless no problem appears in two splits.

    This is the guarantee the whole module exists to provide, so it is checked
    rather than assumed.
    """
    overlap = problem_overlap(problem_ids, splits)
    offending = {k: v for k, v in overlap.items() if v}
    if offending:
        raise AssertionError(f"problem_id leaks across splits: {offending}")


def split_summary(splits: pd.Series, labels: pd.Series | None = None) -> pd.DataFrame:
    """Rows and share per split, optionally broken down by class.

    Used to verify that greedy allocation hit its targets and that no class is
    starved in any split.
    """
    counts = splits.value_counts().sort_index()
    out = pd.DataFrame({"rows": counts, "pct": (counts / len(splits) * 100).round(2)})
    if labels is not None:
        wide = pd.crosstab(splits, labels)
        out = out.join(wide)
    return out
