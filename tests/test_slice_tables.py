"""Tests for sdp.evaluation.slice_tables (Milestone 8.4). Hand-computed small cases."""

import numpy as np
import pytest

from sdp.evaluation.slice_tables import (
    calibration_table,
    confidence_by_predicted_class,
    predicted_parent_shares,
)

Y_TRUE = [1, 2, 2, 2, 0]
Y_PRED = [1, 1, 2, 2, 2]
MAX_PROB = [0.9, 0.5, 0.6, 0.7, 0.8]
MARGIN = [0.8, 0.05, 0.3, 0.5, 0.15]


def test_confidence_by_predicted_class_matches_a_hand_computation() -> None:
    table = confidence_by_predicted_class(Y_TRUE, Y_PRED, MAX_PROB, MARGIN)
    table = table.set_index("predicted_class")
    # Class 0 is never predicted, so it has no row; the last row covers every prediction.
    assert table.index.tolist() == ["SYNTAX", "SEMANTIC", "(all predictions)"]

    syntax = table.loc["SYNTAX"]  # predicted for files 0 and 1
    assert syntax["n_predicted"] == 2
    assert syntax["share_of_predictions"] == pytest.approx(0.4)
    assert syntax["mean_max_prob"] == pytest.approx(0.7)
    assert syntax["median_margin"] == pytest.approx(0.425)
    assert syntax["share_low_margin"] == pytest.approx(0.5)
    assert syntax["precision"] == pytest.approx(0.5)

    semantic = table.loc["SEMANTIC"]  # predicted for files 2, 3, 4
    assert semantic["n_predicted"] == 3
    assert semantic["mean_max_prob"] == pytest.approx(0.7)
    assert semantic["median_margin"] == pytest.approx(0.3)
    assert semantic["share_low_margin"] == pytest.approx(0.0)
    assert semantic["precision"] == pytest.approx(2 / 3)

    everything = table.loc["(all predictions)"]
    assert everything["n_predicted"] == 5
    assert everything["share_of_predictions"] == pytest.approx(1.0)
    assert everything["mean_max_prob"] == pytest.approx(0.7)
    assert everything["median_margin"] == pytest.approx(0.3)
    assert everything["share_low_margin"] == pytest.approx(0.2)
    assert everything["precision"] == pytest.approx(0.6)


def test_low_margin_threshold_is_a_parameter() -> None:
    table = confidence_by_predicted_class(Y_TRUE, Y_PRED, MAX_PROB, MARGIN, low_margin=0.2)
    semantic = table.set_index("predicted_class").loc["SEMANTIC"]
    assert semantic["share_low_margin"] == pytest.approx(1 / 3)  # only margin 0.15 is below 0.2


def test_confidence_by_predicted_class_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        confidence_by_predicted_class([0, 1], [0], [0.5, 0.5], [0.1, 0.1])


def test_predicted_parent_shares_by_hand() -> None:
    # leaves: 0 ERROR_FREE, 1 and 2 under COMPILE_ERROR, 4 under RUNTIME_ERROR, 8 LOGICAL
    y_pred = np.array([0, 1, 2, 4, 8, 8])
    slices = {"first": np.arange(6) < 4, "last": np.arange(6) >= 4}
    table = predicted_parent_shares(y_pred, slices).set_index("slice")
    assert list(table.columns) == [
        "n_files",
        "pred_ERROR_FREE",
        "pred_COMPILE_ERROR",
        "pred_RUNTIME_ERROR",
        "pred_LOGICAL",
    ]
    assert table.loc["first", "n_files"] == 4
    assert table.loc["first", "pred_ERROR_FREE"] == pytest.approx(0.25)
    assert table.loc["first", "pred_COMPILE_ERROR"] == pytest.approx(0.5)
    assert table.loc["first", "pred_RUNTIME_ERROR"] == pytest.approx(0.25)
    assert table.loc["first", "pred_LOGICAL"] == pytest.approx(0.0)
    assert table.loc["last", "pred_LOGICAL"] == pytest.approx(1.0)


def test_predicted_parent_shares_rejects_bad_slices() -> None:
    y_pred = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="empty"):
        predicted_parent_shares(y_pred, {"none": np.zeros(3, dtype=bool)})
    with pytest.raises(ValueError, match="length"):
        predicted_parent_shares(y_pred, {"short": np.ones(2, dtype=bool)})


def test_calibration_table_by_hand() -> None:
    confidence = [0.9, 0.9, 0.5, 0.5]
    correct = [True, True, False, True]
    slices = {"high": np.arange(4) < 2, "low": np.arange(4) >= 2}
    table = calibration_table(confidence, correct, slices).set_index("slice")
    assert table.loc["high", "mean_confidence"] == pytest.approx(0.9)
    assert table.loc["high", "accuracy"] == pytest.approx(1.0)
    assert table.loc["high", "overconfidence"] == pytest.approx(-0.1)
    assert table.loc["low", "accuracy"] == pytest.approx(0.5)
    assert table.loc["low", "overconfidence"] == pytest.approx(0.0)


def test_calibration_table_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        calibration_table([0.5, 0.5], [True], {"a": np.ones(2, dtype=bool)})
    with pytest.raises(ValueError, match="empty"):
        calibration_table([0.5, 0.5], [True, False], {"a": np.zeros(2, dtype=bool)})
