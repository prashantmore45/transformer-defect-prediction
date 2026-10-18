"""Tests for sdp.evaluation.paired (Milestone 8.3)."""

import numpy as np
import pandas as pd
import pytest

from sdp.data.taxonomy import NUM_LEAVES
from sdp.evaluation.bootstrap import summarize_bootstrap
from sdp.evaluation.metrics import LEAF_NAMES, accuracy, macro_f1
from sdp.evaluation.paired import align_systems, compare_systems

K = NUM_LEAVES


def make_truth(n: int = 600, n_groups: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, K, size=n)
    groups = np.array([f"p{g:02d}" for g in rng.integers(0, n_groups, size=n)])
    return y_true, groups


def corrupt(labels: np.ndarray, rate: float, seed: int) -> np.ndarray:
    """Replace a fraction ``rate`` of the labels with random classes."""
    rng = np.random.default_rng(seed)
    result = labels.copy()
    mask = rng.random(labels.shape[0]) < rate
    result[mask] = rng.integers(0, K, size=int(mask.sum()))
    return result


def frame(ids: list[str], y_true: list[int], y_pred: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"submission_id": ids, "y_true": y_true, "y_pred": y_pred})


# --------------------------------------------------------------------------- #
# align_systems
# --------------------------------------------------------------------------- #


def test_align_systems_reorders_rows_to_the_first_systems_order() -> None:
    first = frame(["a", "b", "c"], [0, 1, 2], [0, 0, 0])
    second = frame(["c", "a", "b"], [2, 0, 1], [5, 6, 7])  # a -> 6, b -> 7, c -> 5
    ids, y_true, predictions = align_systems({"first": first, "second": second})
    assert ids.tolist() == ["a", "b", "c"]
    assert y_true.tolist() == [0, 1, 2]
    assert predictions["first"].tolist() == [0, 0, 0]
    assert predictions["second"].tolist() == [6, 7, 5]


def test_align_systems_rejects_different_id_sets() -> None:
    first = frame(["a", "b", "c"], [0, 1, 2], [0, 0, 0])
    second = frame(["a", "b", "z"], [0, 1, 2], [0, 0, 0])
    with pytest.raises(ValueError, match="differ"):
        align_systems({"first": first, "second": second})


def test_align_systems_rejects_duplicate_ids() -> None:
    first = frame(["a", "b", "c"], [0, 1, 2], [0, 0, 0])
    second = frame(["a", "a", "c"], [0, 0, 2], [0, 0, 0])
    with pytest.raises(ValueError, match="duplicate"):
        align_systems({"first": first, "second": second})


def test_align_systems_rejects_different_true_labels() -> None:
    first = frame(["a", "b", "c"], [0, 1, 2], [0, 0, 0])
    second = frame(["a", "b", "c"], [0, 1, 3], [0, 0, 0])
    with pytest.raises(ValueError, match="y_true"):
        align_systems({"first": first, "second": second})


def test_align_systems_rejects_no_systems() -> None:
    with pytest.raises(ValueError, match="no systems"):
        align_systems({})


# --------------------------------------------------------------------------- #
# compare_systems
# --------------------------------------------------------------------------- #


def test_reference_intervals_match_summarize_bootstrap_with_the_same_seed() -> None:
    y_true, groups = make_truth()
    a, b = corrupt(y_true, 0.4, 1), corrupt(y_true, 0.7, 2)
    result = compare_systems(y_true, {"a": a, "b": b}, groups, reference="a", n_boot=200, seed=5)
    overall, _ = summarize_bootstrap(y_true, a, groups, n_boot=200, seed=5)
    mine = result.systems[result.systems["system"] == "a"].set_index("metric")
    theirs = overall.set_index("metric")
    for metric in ("accuracy", "macro_f1"):
        for column in ("value", "ci_low", "ci_high"):
            assert mine.loc[metric, column] == pytest.approx(theirs.loc[metric, column])


def test_difference_values_equal_the_metric_differences() -> None:
    y_true, groups = make_truth()
    a, b = corrupt(y_true, 0.4, 1), corrupt(y_true, 0.7, 2)
    result = compare_systems(y_true, {"a": a, "b": b}, groups, reference="a", n_boot=100, seed=0)
    table = result.differences.set_index("metric")
    assert table.loc["accuracy", "value"] == pytest.approx(
        accuracy(y_true, a) - accuracy(y_true, b)
    )
    assert table.loc["macro_f1", "value"] == pytest.approx(
        macro_f1(y_true, a) - macro_f1(y_true, b)
    )
    assert (table["versus"] == "b").all()
    assert (table["ci_low"] <= table["ci_high"]).all()


def test_identical_systems_have_a_zero_difference_and_a_degenerate_interval() -> None:
    y_true, groups = make_truth()
    a = corrupt(y_true, 0.4, 1)
    result = compare_systems(y_true, {"a": a, "b": a.copy()}, groups, reference="a", n_boot=100)
    for _, row in result.differences.iterrows():
        assert row["value"] == 0.0
        assert row["ci_low"] == 0.0
        assert row["ci_high"] == 0.0
        assert row["frac_draws_not_ahead"] == 1.0  # the reference is never AHEAD


def test_a_clearly_better_reference_has_an_interval_above_zero() -> None:
    y_true, groups = make_truth()
    constant = np.zeros_like(y_true)
    result = compare_systems(
        y_true, {"perfect": y_true.copy(), "constant": constant}, groups, reference="perfect"
    )
    for _, row in result.differences.iterrows():
        assert row["value"] > 0
        assert row["ci_low"] > 0
        assert row["frac_draws_not_ahead"] == 0.0


def test_paired_interval_is_narrower_than_either_systems_own_interval() -> None:
    # B is A with 5% of its predictions scrambled: their mistakes are strongly correlated,
    # so the spread of the DIFFERENCE is much smaller than the spread of either score.
    y_true, groups = make_truth()
    a = corrupt(y_true, 0.4, 1)
    b = corrupt(a, 0.05, 2)
    result = compare_systems(y_true, {"a": a, "b": b}, groups, reference="a", n_boot=500, seed=0)
    own = result.systems[result.systems["metric"] == "accuracy"].set_index("system")
    width_a = own.loc["a", "ci_high"] - own.loc["a", "ci_low"]
    width_b = own.loc["b", "ci_high"] - own.loc["b", "ci_low"]
    diff = result.differences[result.differences["metric"] == "accuracy"].iloc[0]
    assert diff["ci_high"] - diff["ci_low"] < min(width_a, width_b)


def test_per_class_differences_have_one_row_per_class_and_other_system() -> None:
    y_true, groups = make_truth()
    predictions = {
        "ref": corrupt(y_true, 0.3, 1),
        "b": corrupt(y_true, 0.6, 2),
        "c": corrupt(y_true, 0.8, 3),
    }
    result = compare_systems(y_true, predictions, groups, reference="ref", n_boot=100, seed=0)
    table = result.per_class_differences
    assert len(table) == 2 * K
    assert table["versus"].tolist() == ["b"] * K + ["c"] * K
    assert table[table["versus"] == "b"]["class"].tolist() == list(LEAF_NAMES)
    assert (table["ci_low"] <= table["ci_high"]).all()
    assert result.n_files == len(y_true)
    assert result.n_problems == len(np.unique(groups))


def test_compare_systems_validates_its_arguments() -> None:
    y_true, groups = make_truth()
    a = corrupt(y_true, 0.4, 1)
    with pytest.raises(KeyError, match="reference"):
        compare_systems(y_true, {"a": a, "b": a}, groups, reference="missing")
    with pytest.raises(ValueError, match="at least one other"):
        compare_systems(y_true, {"a": a}, groups, reference="a")


def test_compare_systems_is_deterministic() -> None:
    y_true, groups = make_truth()
    predictions = {"a": corrupt(y_true, 0.4, 1), "b": corrupt(y_true, 0.7, 2)}
    first = compare_systems(y_true, predictions, groups, reference="a", n_boot=100, seed=3)
    again = compare_systems(y_true, predictions, groups, reference="a", n_boot=100, seed=3)
    pd.testing.assert_frame_equal(first.systems, again.systems)
    pd.testing.assert_frame_equal(first.differences, again.differences)
    pd.testing.assert_frame_equal(first.per_class_differences, again.per_class_differences)
