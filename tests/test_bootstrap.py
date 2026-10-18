"""Tests for sdp.evaluation.bootstrap (Milestone 8.2)."""

import numpy as np
import pandas as pd
import pytest

from sdp.data.taxonomy import NUM_LEAVES
from sdp.evaluation.bootstrap import (
    draw_group_counts,
    group_confusions,
    interval,
    metrics_from_confusion,
    resample_confusions,
    summarize_bootstrap,
)
from sdp.evaluation.metrics import (
    LEAF_NAMES,
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_report,
)

K = NUM_LEAVES


def random_problem(
    n: int = 600, n_groups: int = 60, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random labels where class 8 is never true and class 5 is never predicted."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 8, size=n)
    y_pred = rng.integers(0, 9, size=n)
    y_pred[y_pred == 5] = 4
    groups = rng.integers(0, n_groups, size=n)
    return y_true, y_pred, groups


# --------------------------------------------------------------------------- #
# group_confusions
# --------------------------------------------------------------------------- #


def test_group_confusions_add_up_to_total() -> None:
    y_true, y_pred, groups = random_problem()
    matrices = group_confusions(y_true, y_pred, groups)
    assert matrices.shape == (len(np.unique(groups)), K, K)
    assert matrices.sum() == len(y_true)
    assert np.array_equal(matrices.sum(axis=0), confusion_matrix(y_true, y_pred))


def test_group_confusions_accepts_string_groups() -> None:
    y_true = np.array([0, 1, 2, 2])
    groups = np.array(["p2", "p1", "p1", "p3"])
    matrices = group_confusions(y_true, y_true, groups)
    assert matrices.shape[0] == 3
    assert matrices[0].sum() == 2  # "p1" comes first when sorted


def test_group_confusions_rejects_wrong_length_groups() -> None:
    y_true, y_pred, groups = random_problem()
    with pytest.raises(ValueError, match="groups"):
        group_confusions(y_true, y_pred, groups[:-1])


def test_group_confusions_rejects_missing_groups() -> None:
    y_true = np.array([0, 1, 2])
    groups = np.array([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="missing"):
        group_confusions(y_true, y_true, groups)


def test_group_confusions_rejects_out_of_range_predictions() -> None:
    with pytest.raises(ValueError):
        group_confusions(np.array([0, 1]), np.array([0, K]), np.array([1, 2]))


# --------------------------------------------------------------------------- #
# metrics_from_confusion
# --------------------------------------------------------------------------- #


def test_metrics_from_confusion_equal_metrics_module() -> None:
    y_true, y_pred, _ = random_problem()
    got = metrics_from_confusion(confusion_matrix(y_true, y_pred))
    reference = per_class_report(y_true, y_pred)
    assert np.allclose(got["precision"], reference["precision"])
    assert np.allclose(got["recall"], reference["recall"])
    assert np.allclose(got["f1"], reference["f1"])
    assert got["macro_f1"] == pytest.approx(macro_f1(y_true, y_pred))
    assert got["accuracy"] == pytest.approx(accuracy(y_true, y_pred))
    # The never-predicted class (5) and the never-true class (8) are scored 0, not dropped.
    assert got["f1"][5] == 0.0
    assert got["f1"][8] == 0.0


def test_metrics_from_confusion_handles_a_stack() -> None:
    first = confusion_matrix(*random_problem(seed=1)[:2])
    second = confusion_matrix(*random_problem(seed=2)[:2])
    stacked = metrics_from_confusion(np.stack([first, second]))
    single = metrics_from_confusion(second)
    assert stacked["f1"].shape == (2, K)
    assert np.allclose(stacked["f1"][1], single["f1"])
    assert stacked["macro_f1"][1] == pytest.approx(single["macro_f1"])


def test_metrics_from_confusion_all_zero_matrix_gives_zeros() -> None:
    result = metrics_from_confusion(np.zeros((K, K)))
    assert result["accuracy"] == 0.0
    assert result["macro_f1"] == 0.0


def test_metrics_from_confusion_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        metrics_from_confusion(np.zeros((3, 4)))


# --------------------------------------------------------------------------- #
# draws and resampling
# --------------------------------------------------------------------------- #


def test_draw_group_counts_shape_and_row_sums() -> None:
    counts = draw_group_counts(30, n_boot=50, seed=0)
    assert counts.shape == (50, 30)
    assert np.all(counts.sum(axis=1) == 30)
    assert np.all(counts >= 0)
    assert np.all(counts == np.round(counts))


def test_draw_group_counts_is_reproducible_and_seed_sensitive() -> None:
    first = draw_group_counts(30, n_boot=20, seed=7)
    again = draw_group_counts(30, n_boot=20, seed=7)
    other = draw_group_counts(30, n_boot=20, seed=8)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_draw_group_counts_rejects_bad_sizes() -> None:
    with pytest.raises(ValueError):
        draw_group_counts(0)
    with pytest.raises(ValueError):
        draw_group_counts(5, n_boot=0)


def test_resampling_each_group_once_gives_the_total_matrix() -> None:
    y_true, y_pred, groups = random_problem()
    matrices = group_confusions(y_true, y_pred, groups)
    counts = np.ones((1, matrices.shape[0]))
    resampled = resample_confusions(counts, matrices)
    assert np.array_equal(resampled[0], confusion_matrix(y_true, y_pred))


def test_resampling_one_group_twice_doubles_it() -> None:
    y_true, y_pred, groups = random_problem()
    matrices = group_confusions(y_true, y_pred, groups)
    counts = np.zeros((1, matrices.shape[0]))
    counts[0, 0] = 2
    assert np.array_equal(resample_confusions(counts, matrices)[0], 2 * matrices[0])


def test_resample_rejects_mismatched_counts() -> None:
    matrices = group_confusions(*random_problem())
    with pytest.raises(ValueError, match="counts"):
        resample_confusions(np.ones((3, matrices.shape[0] + 1)), matrices)


def test_interval_matches_numpy_percentiles_and_validates_level() -> None:
    values = np.arange(101, dtype=float)
    low, high = interval(values, 0.90)
    assert low == pytest.approx(5.0)
    assert high == pytest.approx(95.0)
    with pytest.raises(ValueError, match="level"):
        interval(values, 1.0)


# --------------------------------------------------------------------------- #
# summarize_bootstrap
# --------------------------------------------------------------------------- #


def test_summarize_shapes_columns_and_point_values() -> None:
    y_true, y_pred, groups = random_problem()
    overall, per_class = summarize_bootstrap(y_true, y_pred, groups, n_boot=200, seed=0)

    assert overall["metric"].tolist() == ["accuracy", "macro_f1"]
    assert list(overall.columns) == ["metric", "value", "ci_low", "ci_high"]
    assert (overall["ci_low"] <= overall["ci_high"]).all()
    values = overall.set_index("metric")["value"]
    assert values["accuracy"] == pytest.approx(accuracy(y_true, y_pred))
    assert values["macro_f1"] == pytest.approx(macro_f1(y_true, y_pred))

    assert per_class["class"].tolist() == list(LEAF_NAMES)
    assert list(per_class.columns[:3]) == ["class", "support", "problems"]
    for name in ("precision", "recall", "f1"):
        assert {name, f"{name}_lo", f"{name}_hi"} <= set(per_class.columns)
        assert (per_class[f"{name}_lo"] <= per_class[f"{name}_hi"]).all()
    reference = per_class_report(y_true, y_pred)
    assert np.allclose(per_class["f1"], reference["f1"])
    assert per_class["support"].tolist() == reference["support"].tolist()


def test_summarize_counts_problems_per_class() -> None:
    y_true = np.array([3, 3, 3, 0, 0, 0])
    groups = np.array(["a", "a", "b", "c", "c", "d"])
    _, per_class = summarize_bootstrap(y_true, y_true, groups, n_boot=50, seed=0)
    table = per_class.set_index("class")
    assert table.loc["LINKER", "support"] == 3
    assert table.loc["LINKER", "problems"] == 2
    assert table.loc["ERROR_FREE", "support"] == 3
    assert table.loc["ERROR_FREE", "problems"] == 2
    assert table.loc["SYNTAX", "problems"] == 0


def test_summarize_with_shared_counts_equals_seeded_run() -> None:
    y_true, y_pred, groups = random_problem()
    n_groups = len(np.unique(groups))
    shared = draw_group_counts(n_groups, n_boot=100, seed=3)
    by_counts = summarize_bootstrap(y_true, y_pred, groups, counts=shared)
    by_seed = summarize_bootstrap(y_true, y_pred, groups, n_boot=100, seed=3)
    pd.testing.assert_frame_equal(by_counts[0], by_seed[0])
    pd.testing.assert_frame_equal(by_counts[1], by_seed[1])


def test_perfect_predictions_give_a_degenerate_accuracy_interval() -> None:
    y_true, _, groups = random_problem()
    overall, _ = summarize_bootstrap(y_true, y_true, groups, n_boot=100, seed=0)
    row = overall.set_index("metric").loc["accuracy"]
    assert row["value"] == 1.0
    assert row["ci_low"] == 1.0
    assert row["ci_high"] == 1.0


def test_problem_level_interval_is_wide_when_files_in_a_problem_agree() -> None:
    # 100 problems x 10 files; within a problem every file is right or every file is wrong.
    # Only 100 independent pieces of evidence exist, so the 95% interval for accuracy should
    # be about 0.2 wide. A file-level bootstrap would wrongly report about 0.06.
    rng = np.random.default_rng(1)
    n_groups, per_group = 100, 10
    correct = rng.random(n_groups) < 0.5
    groups = np.repeat(np.arange(n_groups), per_group)
    y_true = np.zeros(n_groups * per_group, dtype=int)
    y_pred = np.where(np.repeat(correct, per_group), 0, 1)
    overall, _ = summarize_bootstrap(y_true, y_pred, groups, n_boot=1000, seed=0)
    row = overall.set_index("metric").loc["accuracy"]
    assert row["ci_high"] - row["ci_low"] > 0.10
