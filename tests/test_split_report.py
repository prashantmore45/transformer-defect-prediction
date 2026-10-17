"""Tests for sdp.evaluation.split_report (Milestone 8.2)."""

import numpy as np
import pandas as pd
import pytest

from sdp.data.taxonomy import NUM_COARSE, NUM_LEAVES
from sdp.evaluation.bootstrap import summarize_bootstrap
from sdp.evaluation.hierarchy import parent_stats
from sdp.evaluation.metrics import accuracy, confusion_matrix, macro_f1
from sdp.evaluation.split_report import PARENT_METRICS, evaluate_split, problem_ids_for

K = NUM_LEAVES


def make_data(
    n: int = 300, n_problems: int = 40, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """A dump-like frame, a manifest in a DIFFERENT row order, and the true problem IDs."""
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n)]
    groups = np.array([f"p{g:03d}" for g in rng.integers(0, n_problems, n)])
    frame = pd.DataFrame(
        {
            "submission_id": ids,
            "y_true": rng.integers(0, K, n),
            "y_pred": rng.integers(0, K, n),
        }
    )
    order = rng.permutation(n)
    manifest = pd.DataFrame(
        {
            "submission_id": [ids[i] for i in order],
            "problem_id": pd.Categorical([groups[i] for i in order]),
        }
    )
    return frame, manifest, groups


# --------------------------------------------------------------------------- #
# problem_ids_for
# --------------------------------------------------------------------------- #


def test_problem_ids_follow_the_dump_row_order() -> None:
    frame, manifest, groups = make_data()
    assert np.array_equal(problem_ids_for(frame, manifest), groups)


def test_problem_ids_reject_unknown_submission_ids() -> None:
    frame, manifest, _ = make_data()
    frame.loc[0, "submission_id"] = "not-in-manifest"
    with pytest.raises(KeyError, match="not in the manifest"):
        problem_ids_for(frame, manifest)


def test_problem_ids_reject_duplicate_manifest_ids() -> None:
    frame, manifest, _ = make_data()
    manifest.loc[1, "submission_id"] = manifest.loc[0, "submission_id"]
    with pytest.raises(ValueError, match="duplicate"):
        problem_ids_for(frame, manifest)


# --------------------------------------------------------------------------- #
# evaluate_split
# --------------------------------------------------------------------------- #


def test_evaluate_split_point_values_match_the_metrics_module() -> None:
    frame, _, groups = make_data()
    result = evaluate_split(frame, groups, n_boot=200, seed=0)
    y_true, y_pred = frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
    values = result.overall.set_index("metric")["value"]
    assert values["accuracy"] == pytest.approx(accuracy(y_true, y_pred))
    assert values["macro_f1"] == pytest.approx(macro_f1(y_true, y_pred))
    assert np.array_equal(result.confusion, confusion_matrix(y_true, y_pred))


def test_evaluate_split_reports_sizes_and_matrix_shapes() -> None:
    frame, _, groups = make_data()
    result = evaluate_split(frame, groups, n_boot=50, seed=0)
    assert result.n_files == len(frame)
    assert result.n_problems == len(np.unique(groups))
    assert result.confusion.shape == (K, K)
    assert result.parent_confusion.shape == (NUM_COARSE, NUM_COARSE)
    assert int(result.parent_confusion.sum()) == len(frame)
    assert len(result.per_class) == K


def test_parent_table_lists_all_metrics_in_order_with_ordered_intervals() -> None:
    frame, _, groups = make_data()
    result = evaluate_split(frame, groups, n_boot=200, seed=0)
    assert result.parent["metric"].tolist() == list(PARENT_METRICS)
    assert (result.parent["ci_low"] <= result.parent["ci_high"]).all()


def test_parent_table_point_values_match_parent_stats() -> None:
    frame, _, groups = make_data()
    result = evaluate_split(frame, groups, n_boot=50, seed=0)
    expected = parent_stats(result.confusion)
    table = result.parent.set_index("metric")["value"]
    for name in PARENT_METRICS:
        assert table[name] == pytest.approx(float(expected[name]))


def test_overall_and_per_class_equal_summarize_bootstrap_with_the_same_seed() -> None:
    frame, _, groups = make_data()
    result = evaluate_split(frame, groups, n_boot=100, seed=5)
    overall, per_class = summarize_bootstrap(
        frame["y_true"].to_numpy(), frame["y_pred"].to_numpy(), groups, n_boot=100, seed=5
    )
    pd.testing.assert_frame_equal(result.overall, overall)
    pd.testing.assert_frame_equal(result.per_class, per_class)


def test_evaluate_split_is_deterministic_and_seed_sensitive() -> None:
    frame, _, groups = make_data()
    first = evaluate_split(frame, groups, n_boot=100, seed=1)
    again = evaluate_split(frame, groups, n_boot=100, seed=1)
    other = evaluate_split(frame, groups, n_boot=100, seed=2)
    pd.testing.assert_frame_equal(first.parent, again.parent)
    assert not first.parent[["ci_low", "ci_high"]].equals(other.parent[["ci_low", "ci_high"]])


def test_perfect_predictions_give_degenerate_parent_intervals() -> None:
    frame, _, groups = make_data()
    frame["y_pred"] = frame["y_true"]
    result = evaluate_split(frame, groups, n_boot=100, seed=0)
    table = result.parent.set_index("metric")
    assert table.loc["parent_accuracy", ["value", "ci_low", "ci_high"]].tolist() == [1.0] * 3
    assert table.loc["within_parent_error_share", "value"] == 0.0


def test_evaluate_split_requires_label_columns() -> None:
    frame, _, groups = make_data()
    with pytest.raises(KeyError, match="y_pred"):
        evaluate_split(frame.drop(columns=["y_pred"]), groups)
