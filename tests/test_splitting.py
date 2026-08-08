"""Tests for leakage-safe dataset splitting.

The headline assertion is `test_problem_level_split_is_disjoint`: no problem_id
may appear in two splits. Every reported metric in the project depends on it, so
it is verified on every commit rather than checked once in a notebook.

`test_submission_level_split_does_leak` asserts the opposite for the comparison
arm. That is deliberate. If both split strategies happened to produce zero
overlap, the disjointness test would pass for the wrong reason and prove
nothing. Confirming that the unsafe split really is unsafe is what makes the
safe one meaningful — and it encodes the project's leakage claim as an
executable check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp.data.splitting import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    SPLIT_ORDER,
    Split,
    allocate_problems,
    assert_problem_disjoint,
    problem_level_split,
    problem_overlap,
    split_summary,
    submission_level_split,
)


def skewed_sizes(n_problems: int = 400, seed: int = 0) -> dict[str, int]:
    """Problem sizes with a heavy right tail, mimicking the real corpus."""
    rng = np.random.default_rng(seed)
    raw = rng.pareto(1.2, n_problems) * 300 + 1
    return {f"p{i:05d}": int(v) for i, v in enumerate(raw)}


def chained_frame(n_problems: int = 200, per_problem: int = 12) -> pd.DataFrame:
    """Submissions with resubmission chains: many rows share a problem_id."""
    rows = [
        {"problem_id": f"p{p:05d}", "submission_id": f"s{p:05d}_{k:02d}"}
        for p in range(n_problems)
        for k in range(per_problem)
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The guarantee
# --------------------------------------------------------------------------- #


def test_problem_level_split_is_disjoint() -> None:
    """No problem_id may appear in two splits. This is the whole point."""
    df = chained_frame()
    splits = problem_level_split(df["problem_id"])

    assert_problem_disjoint(df["problem_id"], splits)
    assert all(v == 0 for v in problem_overlap(df["problem_id"], splits).values())


def test_submission_level_split_does_leak() -> None:
    """The comparison arm must genuinely leak, or the contrast is vacuous."""
    df = chained_frame()
    splits = submission_level_split(df.index)

    overlap = problem_overlap(df["problem_id"], splits)
    assert sum(overlap.values()) > 0, "random split unexpectedly produced no leakage"


def test_every_submission_in_a_chain_shares_one_split() -> None:
    """Stated from the chain's side rather than the problem's."""
    df = chained_frame()
    df["split"] = problem_level_split(df["problem_id"])

    per_problem = df.groupby("problem_id", observed=True)["split"].nunique()
    assert per_problem.max() == 1


# --------------------------------------------------------------------------- #
# allocate_problems — determinism and completeness
# --------------------------------------------------------------------------- #


def test_allocation_covers_every_problem_exactly_once() -> None:
    sizes = skewed_sizes()
    assignment = allocate_problems(sizes)

    assert set(assignment) == set(sizes)
    assert all(isinstance(s, Split) for s in assignment.values())


def test_allocation_is_deterministic_for_a_fixed_seed() -> None:
    sizes = skewed_sizes()
    assert allocate_problems(sizes, seed=7) == allocate_problems(sizes, seed=7)


def test_allocation_varies_with_seed() -> None:
    """Otherwise the seed is decoration and multi-seed runs are meaningless."""
    sizes = skewed_sizes()
    assert allocate_problems(sizes, seed=1) != allocate_problems(sizes, seed=2)


def test_allocation_ignores_dict_insertion_order() -> None:
    """Same data must give the same split however the dict was built."""
    sizes = skewed_sizes()
    shuffled_keys = list(sizes)
    np.random.default_rng(99).shuffle(shuffled_keys)
    reordered = {k: sizes[k] for k in shuffled_keys}

    assert allocate_problems(sizes) == allocate_problems(reordered)


def test_allocation_fills_every_split() -> None:
    assignment = allocate_problems(skewed_sizes())
    assert set(assignment.values()) == set(SPLIT_ORDER)


# --------------------------------------------------------------------------- #
# Balance
# --------------------------------------------------------------------------- #


def _achieved_shares(sizes, assignment) -> dict[Split, float]:
    total = sum(sizes.values())
    return {
        split: sum(n for p, n in sizes.items() if assignment[p] is split) / total
        for split in SPLIT_ORDER
    }


@pytest.mark.parametrize("seed", [0, 1, 42, 1234])
def test_submission_shares_hit_their_targets(seed: int) -> None:
    """Greedy allocation must beat naive random assignment (~1.5pp deviation)."""
    sizes = skewed_sizes(seed=seed)
    shares = _achieved_shares(sizes, allocate_problems(sizes, seed=seed))

    for split in SPLIT_ORDER:
        assert abs(shares[split] - DEFAULT_RATIOS[split]) < 0.005, split


def test_custom_ratios_are_respected() -> None:
    sizes = skewed_sizes()
    ratios = {Split.TRAIN: 0.8, Split.VAL: 0.1, Split.TEST: 0.1}
    shares = _achieved_shares(sizes, allocate_problems(sizes, ratios=ratios))

    for split in SPLIT_ORDER:
        assert abs(shares[split] - ratios[split]) < 0.01, split


# --------------------------------------------------------------------------- #
# Input validation — bad input must fail loudly, not silently
# --------------------------------------------------------------------------- #


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        allocate_problems({})


def test_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError) as exc:
        allocate_problems({"p1": 10, "p2": 0, "p3": 5, "p4": 7})
    assert "p2" in str(exc.value)


def test_rejects_too_few_problems_to_fill_the_splits() -> None:
    with pytest.raises(ValueError):
        allocate_problems({"p1": 10, "p2": 5})


@pytest.mark.parametrize(
    "ratios",
    [
        {Split.TRAIN: 0.6, Split.VAL: 0.2, Split.TEST: 0.3},  # sums to 1.1
        {Split.TRAIN: 0.6, Split.VAL: 0.2},  # missing a split
        {Split.TRAIN: 1.2, Split.VAL: -0.1, Split.TEST: -0.1},  # negative
    ],
)
def test_rejects_invalid_ratios(ratios) -> None:
    with pytest.raises(ValueError):
        allocate_problems(skewed_sizes(n_problems=20), ratios=ratios)


# --------------------------------------------------------------------------- #
# pandas wrappers
# --------------------------------------------------------------------------- #


def test_split_series_is_aligned_and_complete() -> None:
    df = chained_frame()
    splits = problem_level_split(df["problem_id"])

    assert len(splits) == len(df)
    assert splits.index.equals(df.index)
    assert splits.isna().sum() == 0


def test_split_series_is_an_ordered_categorical() -> None:
    """Preserves train/val/test order in groupby and plots, not alphabetical."""
    splits = problem_level_split(chained_frame()["problem_id"])

    assert isinstance(splits.dtype, pd.CategoricalDtype)
    assert splits.dtype.ordered
    assert list(splits.cat.categories) == [s.value for s in SPLIT_ORDER]


def test_handles_categorical_with_unused_categories() -> None:
    """Regression: value_counts on a categorical reports zero-row categories.

    The real manifest has 4,036 problem categories but only 4,032 with rows.
    Those empty categories must not reach allocate_problems as size-0 entries.
    """
    df = chained_frame(n_problems=30)
    df["problem_id"] = pd.Categorical(
        df["problem_id"],
        categories=[f"p{p:05d}" for p in range(40)],  # 10 categories with no rows
    )

    splits = problem_level_split(df["problem_id"])
    assert splits.isna().sum() == 0
    assert_problem_disjoint(df["problem_id"], splits)


def test_submission_level_split_proportions() -> None:
    index = pd.RangeIndex(10_000)
    splits = submission_level_split(index)

    counts = splits.value_counts()
    assert counts[Split.TRAIN.value] == 6000
    assert counts[Split.VAL.value] == 2000
    assert counts[Split.TEST.value] == 2000


def test_submission_level_split_is_deterministic() -> None:
    index = pd.RangeIndex(5_000)
    a = submission_level_split(index, seed=DEFAULT_SEED)
    b = submission_level_split(index, seed=DEFAULT_SEED)
    assert a.equals(b)


def test_split_summary_reports_rows_and_classes() -> None:
    df = chained_frame()
    df["split"] = problem_level_split(df["problem_id"])
    df["label"] = ["A", "B"] * (len(df) // 2)

    summary = split_summary(df["split"], df["label"])
    assert list(summary.index) == [s.value for s in SPLIT_ORDER]
    assert summary["rows"].sum() == len(df)
    assert {"A", "B"} <= set(summary.columns)


# --------------------------------------------------------------------------- #
# The audit helper itself
# --------------------------------------------------------------------------- #


def test_assert_problem_disjoint_detects_a_planted_leak() -> None:
    """The audit must actually catch leakage, not always pass."""
    df = chained_frame(n_problems=20, per_problem=4)
    splits = problem_level_split(df["problem_id"])

    corrupted = splits.copy()
    corrupted.iloc[0] = (
        Split.TEST.value if splits.iloc[0] != Split.TEST.value else Split.TRAIN.value
    )

    with pytest.raises(AssertionError) as exc:
        assert_problem_disjoint(df["problem_id"], corrupted)
    assert "leaks across splits" in str(exc.value)
