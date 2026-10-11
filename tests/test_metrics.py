"""Tests for sdp.evaluation.metrics. Inputs are tiny so every value can be checked by hand."""

import numpy as np
import pytest

from sdp.data.taxonomy import NUM_LEAVES
from sdp.evaluation.metrics import (
    LEAF_NAMES,
    confusion_matrix,
    macro_f1,
    per_class_report,
    summarize,
)


def test_leaf_names_follow_taxonomy_order():
    assert len(LEAF_NAMES) == NUM_LEAVES
    assert LEAF_NAMES[0] == "ERROR_FREE"
    assert LEAF_NAMES[-1] == "LOGICAL"


def test_confusion_matrix_rows_true_columns_predicted():
    cm = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 2], num_classes=3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_macro_f1_perfect_predictions():
    assert macro_f1([0, 1, 2, 2], [0, 1, 2, 2], num_classes=3) == pytest.approx(1.0)


def test_macro_f1_counts_a_never_predicted_class_as_zero():
    # class 0: precision 1/2, recall 1 -> F1 2/3 ; class 1 never predicted -> F1 0
    assert macro_f1([0, 1], [0, 0], num_classes=2) == pytest.approx(1 / 3)


def test_macro_f1_keeps_a_class_with_no_examples_in_the_denominator():
    # class 2 has no true or predicted examples -> F1 0 -> macro = (1 + 1 + 0) / 3
    assert macro_f1([0, 0, 1, 1], [0, 0, 1, 1], num_classes=3) == pytest.approx(2 / 3)


def test_per_class_report_columns_and_support():
    report = per_class_report([0, 0, 1, 2], [0, 1, 1, 2], label_names=("a", "b", "c"))
    assert list(report.columns) == ["class", "support", "precision", "recall", "f1"]
    assert report["class"].tolist() == ["a", "b", "c"]
    assert report["support"].tolist() == [2, 1, 1]
    assert report.loc[0, "recall"] == pytest.approx(0.5)
    assert report.loc[1, "precision"] == pytest.approx(0.5)


def test_summarize_returns_accuracy_and_macro_f1():
    result = summarize([0, 1, 1, 2], [0, 1, 0, 2], num_classes=3)
    assert set(result) == {"accuracy", "macro_f1"}
    assert result["accuracy"] == pytest.approx(0.75)


def test_rejects_out_of_range_labels():
    with pytest.raises(ValueError):
        macro_f1([0, 1], [0, 3], num_classes=3)


def test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        macro_f1([0, 1, 2], [0, 1], num_classes=3)


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        macro_f1([], [], num_classes=3)


def test_rejects_non_integer_labels():
    with pytest.raises(TypeError):
        macro_f1(np.array([0.0, 1.0]), np.array([0.0, 1.0]), num_classes=2)
