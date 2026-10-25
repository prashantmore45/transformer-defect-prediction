"""Tests for sdp.evaluation.plots (Milestone 8.2).

Figures are checked by inspecting their objects (texts, lines, tick labels, bars), not by
comparing pixels, so the tests are fast and stable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdp.data.taxonomy import NUM_LEAVES
from sdp.evaluation.hierarchy import parent_blocks
from sdp.evaluation.metrics import LEAF_NAMES
from sdp.evaluation.plots import (
    build_confusion_figure,
    build_forest_figure,
    build_grouped_bars_figure,
    build_interval_bars_figure,
    build_per_class_f1_figure,
    row_normalize,
    save_figure,
)

K = NUM_LEAVES


def sample_cm() -> np.ndarray:
    cm = np.zeros((K, K), dtype=np.int64)
    cm[0, 0] = 5
    cm[1, 2] = 3
    cm[3, 3] = 1
    cm[8, 8] = 7
    return cm


def main_axes(figure):
    return figure.axes[0]  # axes[1] is the colour bar


def per_class_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class": list(LEAF_NAMES),
            "support": [10 * (i + 1) for i in range(K)],
            "f1": np.linspace(0.1, 0.9, K),
            "f1_lo": np.linspace(0.0, 0.8, K),
            "f1_hi": np.linspace(0.2, 1.0, K),
        }
    )


# --------------------------------------------------------------------------- #
# row_normalize
# --------------------------------------------------------------------------- #


def test_row_normalize_rows_sum_to_one_and_empty_rows_are_zero() -> None:
    cm = np.array([[2, 2], [0, 0]])
    result = row_normalize(cm)
    assert result.tolist() == [[0.5, 0.5], [0.0, 0.0]]


def test_row_normalize_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        row_normalize(np.zeros((2, 3)))


# --------------------------------------------------------------------------- #
# build_confusion_figure
# --------------------------------------------------------------------------- #


def test_counts_mode_writes_each_nonzero_cell_once_and_skips_zeros() -> None:
    figure = build_confusion_figure(sample_cm(), LEAF_NAMES, title="t")
    texts = sorted(text.get_text() for text in main_axes(figure).texts)
    assert texts == ["1", "3", "5", "7"]


def test_percent_mode_rounds_and_marks_tiny_shares() -> None:
    cm = np.zeros((K, K), dtype=np.int64)
    cm[0, 0], cm[0, 1], cm[0, 2] = 900, 99, 1  # 90.0%, 9.9%, 0.1% of 1000
    figure = build_confusion_figure(cm, LEAF_NAMES, title="t", cell_text="percent")
    texts = sorted(text.get_text() for text in main_axes(figure).texts)
    assert texts == ["10", "90", "<1"]


def test_row_labels_show_support_and_columns_show_names() -> None:
    figure = build_confusion_figure(sample_cm(), LEAF_NAMES, title="t")
    axes = main_axes(figure)
    rows = [label.get_text() for label in axes.get_yticklabels()]
    cols = [label.get_text() for label in axes.get_xticklabels()]
    assert rows[0] == "ERROR_FREE (n=5)"
    assert rows[8] == "LOGICAL (n=7)"
    assert rows[4] == "SIGSEGV (n=0)"
    assert cols == list(LEAF_NAMES)


def test_parent_blocks_draw_three_separators_in_each_direction() -> None:
    figure = build_confusion_figure(sample_cm(), LEAF_NAMES, title="t", blocks=parent_blocks())
    assert len(main_axes(figure).lines) == 6  # 3 horizontal + 3 vertical


def test_no_blocks_means_no_separators() -> None:
    figure = build_confusion_figure(sample_cm(), LEAF_NAMES, title="t")
    assert len(main_axes(figure).lines) == 0


def test_four_by_four_parent_matrix_is_supported() -> None:
    cm = np.diag([4, 3, 2, 1]).astype(np.int64)
    names = ["ERROR_FREE", "COMPILE_ERROR", "RUNTIME_ERROR", "LOGICAL"]
    figure = build_confusion_figure(cm, names, title="parents")
    assert len(main_axes(figure).texts) == 4


def test_confusion_figure_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="square"):
        build_confusion_figure(np.zeros((3, 4), dtype=np.int64), ["a", "b", "c"], title="t")
    with pytest.raises(ValueError, match="labels"):
        build_confusion_figure(sample_cm(), LEAF_NAMES[:-1], title="t")
    with pytest.raises(ValueError, match="cell_text"):
        build_confusion_figure(sample_cm(), LEAF_NAMES, title="t", cell_text="bogus")
    with pytest.raises(TypeError, match="integer"):
        build_confusion_figure(sample_cm().astype(float), LEAF_NAMES, title="t")


# --------------------------------------------------------------------------- #
# build_per_class_f1_figure
# --------------------------------------------------------------------------- #


def test_per_class_figure_has_one_bar_per_class_with_support_labels() -> None:
    figure = build_per_class_f1_figure(per_class_table(), title="t")
    axes = main_axes(figure)
    assert len(axes.patches) == K
    labels = [label.get_text() for label in axes.get_yticklabels()]
    assert labels[0] == "ERROR_FREE (n=10)"
    assert labels[-1] == f"LOGICAL (n={10 * K})"


def test_per_class_figure_survives_an_interval_that_misses_the_estimate() -> None:
    table = per_class_table()
    table["f1_lo"] = table["f1"] + 0.05  # inconsistent on purpose
    table["f1_hi"] = table["f1"] - 0.05
    figure = build_per_class_f1_figure(table, title="t")  # whiskers clip to zero, no error
    assert len(main_axes(figure).patches) == K


def test_per_class_figure_rejects_missing_columns() -> None:
    with pytest.raises(KeyError, match="f1_hi"):
        build_per_class_f1_figure(per_class_table().drop(columns=["f1_hi"]), title="t")


# --------------------------------------------------------------------------- #
# save_figure
# --------------------------------------------------------------------------- #


def test_save_figure_writes_a_png_and_creates_folders(tmp_path: Path) -> None:
    figure = build_confusion_figure(sample_cm(), LEAF_NAMES, title="t")
    target = tmp_path / "nested" / "cm.png"
    returned = save_figure(figure, target, dpi=50)
    assert returned == target
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_figure_may_overwrite(tmp_path: Path) -> None:
    figure = build_confusion_figure(sample_cm(), LEAF_NAMES, title="t")
    target = tmp_path / "cm.png"
    save_figure(figure, target, dpi=50)
    save_figure(figure, target, dpi=50)  # regenerable output: must not raise
    assert target.exists()


# --------------------------------------------------------------------------- #
# interval bars, forest and grouped bars (M8.6)
# --------------------------------------------------------------------------- #


def test_interval_bars_have_one_bar_and_one_value_label_per_row() -> None:
    figure = build_interval_bars_figure(
        ["a", "b", "c"], [0.2, 0.5, 0.8], [0.1, 0.4, 0.7], [0.3, 0.6, 0.9], title="t", xlabel="x"
    )
    axes = main_axes(figure)
    assert len(axes.patches) == 3
    assert sorted(text.get_text() for text in axes.texts) == ["0.200", "0.500", "0.800"]
    assert [label.get_text() for label in axes.get_yticklabels()] == ["a", "b", "c"]


def test_interval_bars_reference_adds_exactly_one_line() -> None:
    args = (["a", "b"], [0.2, 0.5], [0.1, 0.4], [0.3, 0.6])
    plain = build_interval_bars_figure(*args, title="t", xlabel="x")
    marked = build_interval_bars_figure(
        *args, title="t", xlabel="x", reference=0.111, reference_label="chance"
    )
    assert len(main_axes(marked).lines) == len(main_axes(plain).lines) + 1


def test_interval_bars_survive_an_interval_that_misses_the_estimate() -> None:
    figure = build_interval_bars_figure(["a"], [0.5], [0.6], [0.4], title="t", xlabel="x")
    assert len(main_axes(figure).patches) == 1


def test_interval_bars_reject_mismatched_or_empty_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        build_interval_bars_figure(["a", "b"], [0.5], [0.4], [0.6], title="t", xlabel="x")
    with pytest.raises(ValueError, match="non-empty"):
        build_interval_bars_figure([], [], [], [], title="t", xlabel="x")


def test_forest_figure_hollows_markers_whose_interval_includes_zero() -> None:
    figure = build_forest_figure(
        ["up", "spans", "down"],
        [0.2, 0.0, -0.2],
        [0.1, -0.1, -0.3],
        [0.3, 0.1, -0.1],
        title="t",
        xlabel="x",
    )
    axes = main_axes(figure)
    assert len(axes.lines) == 2 * 3 + 1  # an interval and a marker per row, plus the zero line
    markers = [line for line in axes.lines if line.get_marker() == "o"]
    assert [marker.get_markerfacecolor() == "white" for marker in markers] == [False, True, False]
    assert [label.get_text() for label in axes.get_yticklabels()] == ["up", "spans", "down"]


def test_forest_figure_rejects_mismatched_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        build_forest_figure(["a"], [0.1, 0.2], [0.0], [0.3], title="t", xlabel="x")


def test_grouped_bars_have_one_bar_per_category_and_series() -> None:
    series = {
        "first": ([0.1, 0.2], [0.0, 0.1], [0.2, 0.3]),
        "second": ([0.3, 0.4], [0.2, 0.3], [0.4, 0.5]),
        "third": ([0.5, 0.6], [0.4, 0.5], [0.6, 0.7]),
    }
    figure = build_grouped_bars_figure(["x", "y"], series, title="t", ylabel="v")
    axes = main_axes(figure)
    assert len(axes.patches) == 6
    legend = axes.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["first", "second", "third"]
    assert [label.get_text() for label in axes.get_xticklabels()] == ["x", "y"]


def test_grouped_bars_apply_the_requested_y_limits() -> None:
    series = {"only": ([0.1], [0.0], [0.2])}
    figure = build_grouped_bars_figure(["x"], series, title="t", ylabel="v", ylim=(0.0, 0.8))
    assert main_axes(figure).get_ylim() == (0.0, 0.8)


def test_grouped_bars_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_grouped_bars_figure([], {"s": ([], [], [])}, title="t", ylabel="v")
    with pytest.raises(ValueError, match="at least one"):
        build_grouped_bars_figure(["x"], {}, title="t", ylabel="v")
    with pytest.raises(ValueError, match="same length"):
        build_grouped_bars_figure(["x", "y"], {"s": ([0.1], [0.0], [0.2])}, title="t", ylabel="v")
