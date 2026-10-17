"""Tests for sdp.evaluation.hierarchy (Milestone 8.2)."""

import numpy as np
import pytest

from sdp.data.taxonomy import NUM_COARSE, NUM_LEAVES
from sdp.evaluation.hierarchy import (
    aggregate_to_parents,
    membership_matrix,
    parent_blocks,
    parent_ids,
    parent_stats,
)
from sdp.evaluation.metrics import confusion_matrix, macro_f1

K = NUM_LEAVES
# Leaf IDs: 0 ERROR_FREE, 1 SYNTAX, 2 SEMANTIC, 3 LINKER, 4 SIGSEGV, 5 SIGFPE,
# 6 SIGABRT, 7 NZEC, 8 LOGICAL.


def cm_of(y_true: list[int], y_pred: list[int]) -> np.ndarray:
    return confusion_matrix(np.array(y_true), np.array(y_pred))


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_parent_ids_follow_the_taxonomy() -> None:
    assert parent_ids().tolist() == [0, 1, 1, 1, 2, 2, 2, 2, 3]


def test_membership_matrix_has_one_parent_per_leaf() -> None:
    matrix = membership_matrix()
    assert matrix.shape == (K, NUM_COARSE)
    assert matrix.sum(axis=1).tolist() == [1] * K
    assert matrix.sum(axis=0).tolist() == [1, 3, 4, 1]


def test_parent_blocks_are_contiguous_ranges() -> None:
    assert parent_blocks() == [
        ("ERROR_FREE", 0, 1),
        ("COMPILE_ERROR", 1, 4),
        ("RUNTIME_ERROR", 4, 8),
        ("LOGICAL", 8, 9),
    ]


# --------------------------------------------------------------------------- #
# aggregate_to_parents
# --------------------------------------------------------------------------- #


def test_aggregate_equals_relabelled_confusion_matrix() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, K, size=500)
    y_pred = rng.integers(0, K, size=500)
    mapping = parent_ids()
    expected = confusion_matrix(mapping[y_true], mapping[y_pred], num_classes=NUM_COARSE)
    assert np.array_equal(aggregate_to_parents(confusion_matrix(y_true, y_pred)), expected)


def test_aggregate_handles_a_stack_and_keeps_totals() -> None:
    rng = np.random.default_rng(1)
    stack = np.stack(
        [
            confusion_matrix(rng.integers(0, K, size=100), rng.integers(0, K, size=100))
            for _ in range(3)
        ]
    )
    result = aggregate_to_parents(stack)
    assert result.shape == (3, NUM_COARSE, NUM_COARSE)
    assert result.sum(axis=(-2, -1)).tolist() == [100, 100, 100]


def test_aggregate_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        aggregate_to_parents(np.zeros((3, 3)))


# --------------------------------------------------------------------------- #
# parent_stats
# --------------------------------------------------------------------------- #


def test_perfect_predictions() -> None:
    stats = parent_stats(cm_of(list(range(K)), list(range(K))))
    assert float(stats["parent_accuracy"]) == 1.0
    assert float(stats["parent_macro_f1"]) == 1.0
    # No errors at all: the share is defined as 0, not undefined.
    assert float(stats["within_parent_error_share"]) == 0.0
    assert float(stats["within_parent_error_share_if_uniform"]) == 0.0


def test_all_errors_within_parent() -> None:
    y_true = [1, 2, 3, 4, 5, 6, 7]
    y_pred = [2, 1, 1, 5, 4, 7, 6]  # every miss lands on a sibling
    stats = parent_stats(cm_of(y_true, y_pred))
    assert float(stats["parent_accuracy"]) == 1.0
    assert float(stats["within_parent_error_share"]) == 1.0
    # One error per true class; siblings are 2,2,2,3,3,3,3 out of 8 possible wrong classes.
    assert float(stats["within_parent_error_share_if_uniform"]) == pytest.approx(18 / 56)


def test_all_errors_across_parents() -> None:
    stats = parent_stats(cm_of([1, 1, 4, 8], [4, 8, 1, 0]))
    assert float(stats["parent_accuracy"]) == 0.0
    assert float(stats["within_parent_error_share"]) == 0.0


def test_mixed_case_by_hand() -> None:
    # Ten SYNTAX files: 6 right, 2 called SEMANTIC (sibling), 2 called LOGICAL (other parent).
    y_true = [1] * 10
    y_pred = [1] * 6 + [2] * 2 + [8] * 2
    stats = parent_stats(cm_of(y_true, y_pred))
    assert float(stats["parent_accuracy"]) == pytest.approx(0.8)  # 6 + 2 right parent
    assert float(stats["within_parent_error_share"]) == pytest.approx(0.5)  # 2 of 4 errors
    # SYNTAX has 2 siblings; 4 errors x 2/8 = 1 expected within, out of 4 errors.
    assert float(stats["within_parent_error_share_if_uniform"]) == pytest.approx(0.25)


def test_classes_without_siblings_have_no_within_parent_errors() -> None:
    stats = parent_stats(cm_of([0, 0, 0, 0], [0, 1, 4, 8]))
    assert float(stats["parent_accuracy"]) == pytest.approx(0.25)
    assert float(stats["within_parent_error_share"]) == 0.0
    assert float(stats["within_parent_error_share_if_uniform"]) == 0.0


def test_parent_macro_f1_equals_metrics_module_on_relabelled_data() -> None:
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, K, size=400)
    y_pred = rng.integers(0, K, size=400)
    mapping = parent_ids()
    expected = macro_f1(mapping[y_true], mapping[y_pred], num_classes=NUM_COARSE)
    got = parent_stats(confusion_matrix(y_true, y_pred))["parent_macro_f1"]
    assert float(got) == pytest.approx(expected)


def test_parent_stats_handles_a_stack() -> None:
    rng = np.random.default_rng(3)
    stack = np.stack(
        [
            confusion_matrix(rng.integers(0, K, size=200), rng.integers(0, K, size=200))
            for _ in range(4)
        ]
    )
    stats = parent_stats(stack)
    for name, values in stats.items():
        assert values.shape == (4,), name
    single = parent_stats(stack[2])
    assert float(stats["parent_accuracy"][2]) == pytest.approx(float(single["parent_accuracy"]))


def test_parent_stats_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        parent_stats(np.zeros((4, 4)))
