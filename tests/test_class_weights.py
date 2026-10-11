"""Tests for sdp.training.class_weights, including the real M6 training counts."""

import numpy as np
import pytest

from sdp.data.taxonomy import LEAF_ORDER, LEAF_TO_ID, DefectClass
from sdp.training.class_weights import compute_class_weights, count_classes

# Train-split counts from reports/m6_class_split_counts.csv, keyed by class name so
# the order is taken from the taxonomy and not from an assumed list position.
_TRAIN_COUNTS = {
    DefectClass.ERROR_FREE: 5968,
    DefectClass.SYNTAX: 4232,
    DefectClass.SEMANTIC: 9404,
    DefectClass.LINKER: 232,
    DefectClass.SIGSEGV: 1208,
    DefectClass.SIGFPE: 390,
    DefectClass.SIGABRT: 741,
    DefectClass.NZEC: 311,
    DefectClass.LOGICAL: 5972,
}
M6_TRAIN_COUNTS = np.array([_TRAIN_COUNTS[c] for c in LEAF_ORDER])


def test_count_classes_includes_absent_classes():
    assert count_classes([0, 0, 2], num_classes=4).tolist() == [2, 0, 1, 0]


def test_count_classes_rejects_out_of_range():
    with pytest.raises(ValueError):
        count_classes([0, 5], num_classes=4)


def test_count_classes_rejects_empty():
    with pytest.raises(ValueError):
        count_classes([], num_classes=4)


def test_count_classes_rejects_non_integer():
    with pytest.raises(TypeError):
        count_classes(np.array([0.0, 1.0]), num_classes=4)


def test_power_zero_gives_all_ones():
    weights = compute_class_weights([100, 400], 0.0)
    assert weights.tolist() == pytest.approx([1.0, 1.0])


def test_power_one_is_balanced_inverse_frequency():
    weights = compute_class_weights([100, 400], 1.0)
    assert weights.tolist() == pytest.approx([2.5, 0.625])


def test_power_half_is_the_square_root_of_the_ratio():
    counts = np.array([100, 400])
    weights = compute_class_weights(counts, 0.5)
    assert weights[0] / weights[1] == pytest.approx(2.0)
    assert (counts * weights).sum() / counts.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("power", [0.0, 0.25, 0.5, 1.0])
def test_sample_weighted_mean_is_one(power):
    weights = compute_class_weights(M6_TRAIN_COUNTS, power)
    assert (M6_TRAIN_COUNTS * weights).sum() / M6_TRAIN_COUNTS.sum() == pytest.approx(1.0)


def test_reference_values_for_the_m6_training_counts():
    linker = LEAF_TO_ID[DefectClass.LINKER]
    semantic = LEAF_TO_ID[DefectClass.SEMANTIC]
    half = compute_class_weights(M6_TRAIN_COUNTS, 0.5)
    full = compute_class_weights(M6_TRAIN_COUNTS, 1.0)
    assert half[linker] == pytest.approx(4.33, abs=0.01)
    assert half[semantic] == pytest.approx(0.68, abs=0.01)
    assert full[linker] == pytest.approx(13.63, abs=0.01)
    assert full[semantic] == pytest.approx(0.336, abs=0.01)
    assert half.max() / half.min() == pytest.approx((9404 / 232) ** 0.5)
    assert full.max() / full.min() == pytest.approx(9404 / 232)


def test_rejects_a_class_with_zero_examples():
    with pytest.raises(ValueError, match="at least one"):
        compute_class_weights([5, 0, 3], 0.5)


@pytest.mark.parametrize("power", [-0.1, 1.1])
def test_rejects_power_outside_zero_to_one(power):
    with pytest.raises(ValueError, match="power"):
        compute_class_weights([5, 5], power)


def test_rejects_non_integer_counts():
    with pytest.raises(TypeError):
        compute_class_weights(np.array([1.0, 2.0]), 0.5)


def test_rejects_empty_counts():
    with pytest.raises(ValueError):
        compute_class_weights([], 0.5)
