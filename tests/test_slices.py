"""Tests for sdp.evaluation.slices (Milestone 8.4)."""

import numpy as np
import pandas as pd
import pytest

from sdp.data.taxonomy import NUM_LEAVES
from sdp.evaluation.bootstrap import summarize_bootstrap
from sdp.evaluation.metrics import accuracy, per_class_report
from sdp.evaluation.slices import (
    bootstrap_slices,
    common_classes,
    interval_masks,
    softmax_confidence,
)

K = NUM_LEAVES


def make_data(
    n: int = 900, n_groups: int = 60, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, K, size=n)
    y_pred = np.where(rng.random(n) < 0.4, y_true, rng.integers(0, K, size=n))
    groups = np.array([f"p{g:02d}" for g in rng.integers(0, n_groups, size=n)])
    return y_true, y_pred, groups


# --------------------------------------------------------------------------- #
# interval_masks
# --------------------------------------------------------------------------- #


def test_interval_masks_are_right_closed_and_skip_values_outside() -> None:
    values = [0, 128, 129, 256, 300, 511, 512, 600]
    masks = interval_masks(values, [0, 128, 256, 511, 512], labels=["a", "b", "c", "d"])
    assert list(masks) == ["a", "b", "c", "d"]
    assert np.flatnonzero(masks["a"]).tolist() == [1]
    assert np.flatnonzero(masks["b"]).tolist() == [2, 3]
    assert np.flatnonzero(masks["c"]).tolist() == [4, 5]
    assert np.flatnonzero(masks["d"]).tolist() == [6]  # 0 and 600 fall in no bin


def test_interval_masks_default_labels_and_validation() -> None:
    assert list(interval_masks([1, 2], [0, 5, 10])) == ["(0, 5]", "(5, 10]"]
    with pytest.raises(ValueError, match="increasing"):
        interval_masks([1], [0, 5, 5])
    with pytest.raises(ValueError, match="labels"):
        interval_masks([1], [0, 5, 10], labels=["only one"])


# --------------------------------------------------------------------------- #
# softmax_confidence
# --------------------------------------------------------------------------- #


def test_uniform_logits_give_one_over_k_and_zero_margin() -> None:
    max_prob, margin = softmax_confidence(np.zeros((2, K)))
    assert np.allclose(max_prob, 1 / K)
    assert np.allclose(margin, 0.0)


def test_a_dominant_logit_gives_a_confident_prediction() -> None:
    logits = np.zeros((1, K))
    logits[0, 3] = 10.0
    max_prob, margin = softmax_confidence(logits)
    assert max_prob[0] > 0.999
    assert margin[0] > 0.999


def test_softmax_confidence_is_stable_for_huge_logits() -> None:
    logits = np.zeros((1, K))
    logits[0, 0], logits[0, 1] = 1000.0, 999.0
    max_prob, margin = softmax_confidence(logits)
    assert np.isfinite(max_prob).all()
    assert max_prob[0] == pytest.approx(1 / (1 + np.exp(-1.0)), abs=1e-6)
    assert margin[0] == pytest.approx(max_prob[0] - (1 - max_prob[0]), abs=1e-6)


def test_softmax_confidence_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="2-D"):
        softmax_confidence(np.zeros(5))


# --------------------------------------------------------------------------- #
# bootstrap_slices
# --------------------------------------------------------------------------- #


def test_point_values_match_the_metrics_module_on_each_slice() -> None:
    y_true, y_pred, groups = make_data()
    mask_a = np.arange(len(y_true)) % 3 == 0
    result = bootstrap_slices(
        y_true, y_pred, groups, {"a": mask_a, "b": ~mask_a}, n_boot=100, seed=0
    )
    overall = result.overall.set_index("slice")
    assert overall.loc["a", "n_files"] == int(mask_a.sum())
    assert overall.loc["a", "accuracy"] == pytest.approx(accuracy(y_true[mask_a], y_pred[mask_a]))
    report = per_class_report(y_true[mask_a], y_pred[mask_a]).set_index("class")
    table = result.per_class[result.per_class["slice"] == "a"].set_index("class")
    assert len(table) == int((report["support"] > 0).sum())  # only classes that appear
    for label in table.index:
        assert table.loc[label, "recall"] == pytest.approx(report.loc[label, "recall"])
        assert table.loc[label, "support"] == report.loc[label, "support"]


def test_a_slice_of_all_files_reproduces_the_whole_split_interval() -> None:
    y_true, y_pred, groups = make_data()
    everything = np.ones(len(y_true), dtype=bool)
    result = bootstrap_slices(y_true, y_pred, groups, {"all": everything}, n_boot=200, seed=4)
    overall, _ = summarize_bootstrap(y_true, y_pred, groups, n_boot=200, seed=4)
    expected = overall.set_index("metric").loc["accuracy"]
    row = result.overall.iloc[0]
    assert row["accuracy"] == pytest.approx(expected["value"])
    assert row["accuracy_lo"] == pytest.approx(expected["ci_low"])
    assert row["accuracy_hi"] == pytest.approx(expected["ci_high"])


def test_shared_draws_make_parallel_slices_differ_by_exactly_zero() -> None:
    # Every problem contributes 5 files to each slice and is right or wrong as a whole, so in
    # every bootstrap draw the two slices have identical accuracy: the paired difference is
    # exactly 0, although each slice's own interval is wide.
    rng = np.random.default_rng(0)
    n_groups, per_group = 80, 10
    wrong = rng.random(n_groups) < 0.5
    groups = np.repeat([f"p{g:02d}" for g in range(n_groups)], per_group)
    y_true = np.zeros(n_groups * per_group, dtype=int)
    y_pred = np.where(np.repeat(wrong, per_group), 1, 0)
    first_half = np.tile(np.arange(per_group), n_groups) < 5
    result = bootstrap_slices(
        y_true,
        y_pred,
        groups,
        {"first": first_half, "second": ~first_half},
        pairs=[("first", "second")],
        n_boot=300,
        seed=0,
    )
    own = result.overall.set_index("slice")
    assert own.loc["first", "accuracy_hi"] - own.loc["first", "accuracy_lo"] > 0.05
    row = result.differences[result.differences["metric"] == "accuracy"].iloc[0]
    assert (row["value"], row["ci_low"], row["ci_high"]) == (0.0, 0.0, 0.0)


def test_macro_recall_scores_only_classes_with_enough_support() -> None:
    y_true = np.array([0] * 30 + [1] * 5)
    y_pred = np.array([0] * 15 + [1] * 15 + [1] * 5)  # class 0 recall 0.5, class 1 recall 1.0
    groups = np.array([f"p{i % 10}" for i in range(35)])
    result = bootstrap_slices(
        y_true,
        y_pred,
        groups,
        {"all": np.ones(35, dtype=bool)},
        n_boot=50,
        seed=0,
        min_class_support=20,
    )
    row = result.overall.iloc[0]
    assert row["n_classes_scored"] == 1
    assert row["macro_recall"] == pytest.approx(0.5)


def test_difference_rows_equal_the_point_differences() -> None:
    y_true, y_pred, groups = make_data()
    mask_a = np.arange(len(y_true)) % 3 == 0
    mask_b = np.arange(len(y_true)) % 3 == 1
    result = bootstrap_slices(
        y_true,
        y_pred,
        groups,
        {"a": mask_a, "b": mask_b},
        pairs=[("a", "b")],
        n_boot=100,
        seed=0,
    )
    table = result.differences.set_index("metric")
    assert table.loc["accuracy", "value"] == pytest.approx(
        accuracy(y_true[mask_a], y_pred[mask_a]) - accuracy(y_true[mask_b], y_pred[mask_b])
    )
    assert {"accuracy", "macro_recall"} <= set(table.index)
    assert any(name.startswith("recall:") for name in table.index)
    assert (table["ci_low"] <= table["ci_high"]).all()
    assert (result.differences["first"] == "a").all()


def test_overlapping_slices_are_allowed() -> None:
    y_true, y_pred, groups = make_data()
    everything = np.ones(len(y_true), dtype=bool)
    half = np.arange(len(y_true)) % 2 == 0
    result = bootstrap_slices(
        y_true, y_pred, groups, {"all": everything, "half": half}, n_boot=50, seed=0
    )
    assert result.overall["n_files"].tolist() == [len(y_true), int(half.sum())]


def test_bad_slices_and_pairs_are_rejected() -> None:
    y_true, y_pred, groups = make_data()
    everything = np.ones(len(y_true), dtype=bool)
    with pytest.raises(ValueError, match="no slices"):
        bootstrap_slices(y_true, y_pred, groups, {})
    with pytest.raises(ValueError, match="empty"):
        bootstrap_slices(y_true, y_pred, groups, {"e": np.zeros(len(y_true), dtype=bool)})
    with pytest.raises(ValueError, match="length"):
        bootstrap_slices(y_true, y_pred, groups, {"short": np.ones(5, dtype=bool)})
    with pytest.raises(KeyError, match="missing"):
        bootstrap_slices(y_true, y_pred, groups, {"a": everything}, pairs=[("a", "nope")])


def test_bootstrap_slices_is_deterministic() -> None:
    y_true, y_pred, groups = make_data()
    mask = np.arange(len(y_true)) % 2 == 0
    kwargs = {"pairs": [("x", "y")], "n_boot": 60, "seed": 3}
    first = bootstrap_slices(y_true, y_pred, groups, {"x": mask, "y": ~mask}, **kwargs)
    again = bootstrap_slices(y_true, y_pred, groups, {"x": mask, "y": ~mask}, **kwargs)
    pd.testing.assert_frame_equal(first.overall, again.overall)
    pd.testing.assert_frame_equal(first.per_class, again.per_class)
    pd.testing.assert_frame_equal(first.differences, again.differences)


def test_fixed_macro_classes_are_used_in_every_slice() -> None:
    y_true = np.array([0] * 30 + [1] * 30)
    y_pred = np.array([0] * 15 + [1] * 15 + [1] * 30)  # class 0 recall 0.5, class 1 recall 1.0
    groups = np.array([f"p{i % 12}" for i in range(60)])
    result = bootstrap_slices(
        y_true,
        y_pred,
        groups,
        {"zeros": np.arange(60) < 30, "all": np.ones(60, dtype=bool)},
        n_boot=50,
        seed=0,
        macro_classes=[0, 1],
    )
    table = result.overall.set_index("slice")
    assert table.loc["all", "n_classes_scored"] == 2
    assert table.loc["all", "macro_recall"] == pytest.approx(0.75)
    # class 1 has no files in the first slice: it is skipped there, not scored as zero
    assert table.loc["zeros", "macro_recall"] == pytest.approx(0.5)


def test_macro_classes_are_validated() -> None:
    y_true, y_pred, groups = make_data()
    everything = np.ones(len(y_true), dtype=bool)
    for bad in ([], [K], [-1]):
        with pytest.raises(ValueError, match="macro_classes"):
            bootstrap_slices(y_true, y_pred, groups, {"all": everything}, macro_classes=bad)


def test_common_classes_keeps_classes_with_enough_files_in_every_slice() -> None:
    first = [0] * 30 + [1] * 25 + [2] * 30  # 85 files
    second = [0] * 30 + [1] * 10 + [2] * 30  # 70 files
    y_true = np.array(first + second)
    masks = {"first": np.arange(155) < 85, "second": np.arange(155) >= 85}
    assert common_classes(y_true, masks, min_class_support=20) == [0, 2]
    assert common_classes(y_true, masks, min_class_support=5) == [0, 1, 2]
