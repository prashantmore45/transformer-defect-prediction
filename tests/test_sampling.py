"""Tests for working-corpus sampling.

Three tests carry most of the weight:

`test_round_robin_beats_naive_sampling_for_diversity` verifies the module's
central claim — that round-robin allocation represents more problems than
uniform random sampling on a concentrated corpus. Everything else here checks
bookkeeping; this one checks the idea.

`test_changing_one_quota_does_not_disturb_other_classes` verifies the per-cell
derived seeds. Without them, raising one quota silently reshuffles every other
class's selection — no error, no visible symptom.

`test_raising_a_quota_extends_rather_than_reshuffles` pins down monotonicity,
which is what makes an incremental second extraction pass possible instead of a
full redo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp.data.sampling import (
    ARCHIVE_ROOT,
    DEFAULT_CLASS_TOTALS,
    archive_paths,
    draw_sample,
    resolve_quotas,
)
from sdp.data.splitting import DEFAULT_RATIOS, SPLIT_ORDER, Split
from sdp.data.taxonomy import COARSE_ORDER, CoarseClass


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["split"] = pd.Categorical(
        df["split"], categories=[s.value for s in SPLIT_ORDER], ordered=True
    )
    df["coarse_label"] = pd.Categorical(
        df["coarse_label"], categories=[c.value for c in COARSE_ORDER], ordered=True
    )
    df["problem_id"] = df["problem_id"].astype("category")
    return df


def uniform_manifest(n_problems: int = 60, per_problem: int = 40) -> pd.DataFrame:
    """Every problem the same size, classes cycled evenly.

    Gives 200 rows in each (split, class) cell for the default 60/40 shape, so
    round-robin runs to completion without exhausting any problem.
    """
    rows = []
    for p in range(n_problems):
        split = SPLIT_ORDER[p % len(SPLIT_ORDER)].value
        for k in range(per_problem):
            rows.append(
                {
                    "submission_id": f"s{p:04d}_{k:04d}",
                    "problem_id": f"p{p:05d}",
                    "split": split,
                    "coarse_label": COARSE_ORDER[k % len(COARSE_ORDER)].value,
                }
            )
    return _frame(rows)


def concentrated_manifest() -> pd.DataFrame:
    """One dominant problem plus a long tail, in every (split, class) cell.

    Mirrors the real corpus, where the top 100 problems hold 19.6% of rows.
    """
    rows = []
    for split in SPLIT_ORDER:
        for cls in COARSE_ORDER:
            tag = f"{split.value[:2]}{cls.value[:2]}"
            for k in range(600):  # dominant problem
                rows.append(
                    {
                        "submission_id": f"s_{tag}_big_{k:04d}",
                        "problem_id": f"p_{tag}_big",
                        "split": split.value,
                        "coarse_label": cls.value,
                    }
                )
            for t in range(50):  # long tail: 50 problems x 8 rows
                for k in range(8):
                    rows.append(
                        {
                            "submission_id": f"s_{tag}_{t:02d}_{k}",
                            "problem_id": f"p_{tag}_{t:02d}",
                            "split": split.value,
                            "coarse_label": cls.value,
                        }
                    )
    return _frame(rows)


SMALL_TOTALS = {cls: 150 for cls in COARSE_ORDER}  # -> 90/30/30 per split


# --------------------------------------------------------------------------- #
# resolve_quotas
# --------------------------------------------------------------------------- #


def test_default_quotas_total_the_working_corpus() -> None:
    quotas = resolve_quotas()
    assert len(quotas) == len(SPLIT_ORDER) * len(COARSE_ORDER)
    assert sum(quotas.values()) == 75_000


def test_each_class_total_is_exact_after_splitting() -> None:
    """Rounding residue must not leak; per-class totals are the contract."""
    quotas = resolve_quotas()
    for cls in COARSE_ORDER:
        got = sum(quotas[(s, cls)] for s in SPLIT_ORDER)
        assert got == DEFAULT_CLASS_TOTALS[cls], cls


def test_quotas_follow_the_split_ratios() -> None:
    quotas = resolve_quotas()
    for cls in COARSE_ORDER:
        total = DEFAULT_CLASS_TOTALS[cls]
        for split in SPLIT_ORDER:
            expected = total * DEFAULT_RATIOS[split]
            assert abs(quotas[(split, cls)] - expected) <= 1


def test_resolve_quotas_rejects_incomplete_class_totals() -> None:
    with pytest.raises(ValueError):
        resolve_quotas({CoarseClass.ERROR_FREE: 100})


def test_resolve_quotas_rejects_non_positive_totals() -> None:
    with pytest.raises(ValueError):
        resolve_quotas({cls: 0 for cls in COARSE_ORDER})


# --------------------------------------------------------------------------- #
# The central claim: diversity
# --------------------------------------------------------------------------- #


def test_round_robin_beats_naive_sampling_for_diversity() -> None:
    """The module exists for this. Everything else is bookkeeping."""
    df = concentrated_manifest()
    quotas = resolve_quotas(SMALL_TOTALS)

    sample, _ = draw_sample(df, quotas=quotas, seed=3)
    naive = df.sample(n=len(sample), random_state=3)

    assert sample["problem_id"].nunique() > naive["problem_id"].nunique()


def test_every_problem_contributes_before_any_contributes_twice() -> None:
    """With uniform problem sizes, per-problem counts differ by at most one.

    `astype(str)` before value_counts is required: on a categorical Series,
    value_counts reports every declared category, so the 40 problems belonging
    to other splits would appear with count 0 and drag min() to zero.
    """
    df = uniform_manifest()
    sample, _ = draw_sample(df, quotas=resolve_quotas(SMALL_TOTALS), seed=1)

    cell = sample[
        (sample["split"].astype(str) == Split.TRAIN.value)
        & (sample["coarse_label"].astype(str) == CoarseClass.ERROR_FREE.value)
    ]
    counts = cell["problem_id"].astype(str).value_counts()

    train_problems = set(
        df.loc[df["split"].astype(str) == Split.TRAIN.value, "problem_id"].astype(str)
    )
    assert set(counts.index) == train_problems, "some problems contributed nothing"
    assert counts.max() - counts.min() <= 1


def test_report_records_problem_coverage() -> None:
    df = concentrated_manifest()
    _, report = draw_sample(df, quotas=resolve_quotas(SMALL_TOTALS), seed=1)

    assert (report["problems"] > 1).all()
    assert set(report.columns) >= {
        "split",
        "coarse_label",
        "requested",
        "available",
        "taken",
        "shortfall",
        "problems",
        "max_per_problem",
    }


# --------------------------------------------------------------------------- #
# Determinism and seed isolation
# --------------------------------------------------------------------------- #


def test_sampling_is_deterministic_for_a_fixed_seed() -> None:
    df = uniform_manifest()
    quotas = resolve_quotas(SMALL_TOTALS)

    a, _ = draw_sample(df, quotas=quotas, seed=11)
    b, _ = draw_sample(df, quotas=quotas, seed=11)
    assert a["submission_id"].tolist() == b["submission_id"].tolist()


def test_sampling_varies_with_seed() -> None:
    df = uniform_manifest()
    quotas = resolve_quotas(SMALL_TOTALS)

    a, _ = draw_sample(df, quotas=quotas, seed=11)
    b, _ = draw_sample(df, quotas=quotas, seed=12)
    assert set(a["submission_id"]) != set(b["submission_id"])


def _ids_for(sample: pd.DataFrame, cls: CoarseClass) -> set[str]:
    return set(sample.loc[sample["coarse_label"].astype(str) == cls.value, "submission_id"])


def test_changing_one_quota_does_not_disturb_other_classes() -> None:
    """Verifies per-cell derived seeds. Fails loudly if they are removed."""
    df = uniform_manifest()
    base = resolve_quotas(SMALL_TOTALS)
    bumped = dict(base)
    bumped[(Split.TRAIN, CoarseClass.COMPILE_ERROR)] += 20

    a, _ = draw_sample(df, quotas=base, seed=5)
    b, _ = draw_sample(df, quotas=bumped, seed=5)

    for cls in (CoarseClass.ERROR_FREE, CoarseClass.RUNTIME_ERROR, CoarseClass.LOGICAL):
        assert _ids_for(a, cls) == _ids_for(b, cls), cls


def test_raising_a_quota_extends_rather_than_reshuffles() -> None:
    """Monotonicity: a later, larger draw is a superset of the earlier one.

    This is what makes an incremental second extraction pass viable — already
    extracted files stay in the corpus and only the additions are fetched.
    """
    df = uniform_manifest()
    base = resolve_quotas(SMALL_TOTALS)
    bumped = dict(base)
    bumped[(Split.TRAIN, CoarseClass.COMPILE_ERROR)] += 20

    a, _ = draw_sample(df, quotas=base, seed=5)
    b, _ = draw_sample(df, quotas=bumped, seed=5)

    assert set(a["submission_id"]) < set(b["submission_id"])
    assert len(b) == len(a) + 20


# --------------------------------------------------------------------------- #
# Correctness of the drawn sample
# --------------------------------------------------------------------------- #


def test_sample_respects_quotas_when_supply_is_ample() -> None:
    df = uniform_manifest()
    quotas = resolve_quotas(SMALL_TOTALS)
    sample, report = draw_sample(df, quotas=quotas, seed=1)

    assert len(sample) == sum(quotas.values())
    assert (report["shortfall"] == 0).all()
    assert (report["taken"] == report["requested"]).all()


def test_sample_rows_land_in_the_right_cell() -> None:
    """Guards against a filter bug silently mixing splits or classes."""
    df = uniform_manifest()
    sample, _ = draw_sample(df, quotas=resolve_quotas(SMALL_TOTALS), seed=1)

    joined = sample.merge(
        df[["submission_id", "split", "coarse_label"]],
        on="submission_id",
        suffixes=("", "_src"),
    )
    assert (joined["split"].astype(str) == joined["split_src"].astype(str)).all()
    assert (joined["coarse_label"].astype(str) == joined["coarse_label_src"].astype(str)).all()


def test_sample_has_no_duplicate_submissions() -> None:
    df = uniform_manifest()
    sample, _ = draw_sample(df, quotas=resolve_quotas(SMALL_TOTALS), seed=1)
    assert not sample["submission_id"].duplicated().any()


def test_sample_is_a_subset_of_the_input() -> None:
    df = uniform_manifest()
    sample, _ = draw_sample(df, quotas=resolve_quotas(SMALL_TOTALS), seed=1)
    assert set(sample.index) <= set(df.index)
    assert list(sample.columns) == list(df.columns)


def test_shortfall_is_reported_not_silently_truncated() -> None:
    """A quietly short class would leave you trusting a count that is wrong."""
    df = uniform_manifest(n_problems=6, per_problem=8)  # 4 rows per cell
    quotas = resolve_quotas({cls: 300 for cls in COARSE_ORDER})

    sample, report = draw_sample(df, quotas=quotas, seed=1)

    assert (report["shortfall"] > 0).any()
    assert (report["taken"] <= report["available"]).all()
    assert len(sample) == int(report["taken"].sum())


def test_missing_column_raises() -> None:
    df = uniform_manifest().drop(columns=["coarse_label"])
    with pytest.raises(KeyError):
        draw_sample(df, quotas=resolve_quotas(SMALL_TOTALS))


# --------------------------------------------------------------------------- #
# Archive paths
# --------------------------------------------------------------------------- #


def test_archive_path_format() -> None:
    df = pd.DataFrame({"problem_id": ["p00000", "p04032"], "submission_id": ["s123", "s456"]})
    paths = archive_paths(df)

    assert paths.iloc[0] == f"{ARCHIVE_ROOT}/p00000/C++/s123.cpp"
    assert paths.iloc[1] == f"{ARCHIVE_ROOT}/p04032/C++/s456.cpp"


def test_archive_paths_are_unique_for_unique_submissions() -> None:
    """A collision would silently overwrite an extracted file."""
    df = uniform_manifest()
    sample, _ = draw_sample(df, quotas=resolve_quotas(SMALL_TOTALS), seed=1)
    paths = archive_paths(sample)

    assert paths.nunique() == len(sample)


def test_archive_paths_handle_categorical_columns() -> None:
    """problem_id is categorical dtype in the real manifest."""
    df = uniform_manifest()
    assert isinstance(df["problem_id"].dtype, pd.CategoricalDtype)

    paths = archive_paths(df.head(5))
    assert all(p.startswith(f"{ARCHIVE_ROOT}/p") for p in paths)
    assert all(p.endswith(".cpp") for p in paths)
