"""Figures for the evaluation report (Milestone 8.2).

Design notes
------------
* Figures are built with ``matplotlib.figure.Figure`` directly, not ``pyplot``. There is no
  backend selection, no global figure registry and nothing to close, so the same code runs
  headless in tests and in the evaluation script.
* The confusion matrix is COLOURED by the share of each true class (each row sums to 1), so
  a thin class such as LINKER stays visible next to SEMANTIC. The cell TEXT is either the raw
  count or the rounded percentage. A nonzero percentage that rounds to 0 is shown as "<1",
  and empty cells are left blank.
* Rows are labelled with their support (``LINKER (n=54)``) so a recall figure is never read
  without seeing how many files it rests on.
* Parent separators come from ``sdp.evaluation.hierarchy.parent_blocks``, not from
  hard-coded positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

CELL_TEXT_MODES = ("counts", "percent")
PER_CLASS_COLUMNS = {"class", "support", "f1", "f1_lo", "f1_hi"}


def row_normalize(cm: object) -> np.ndarray:
    """Divide each row by its sum; a row with no files becomes all zeros."""
    matrix = np.asarray(cm, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)


def build_confusion_figure(
    cm: object,
    labels: Sequence[str],
    *,
    title: str,
    cell_text: str = "counts",
    blocks: Sequence[tuple[str, int, int]] | None = None,
) -> Figure:
    """Confusion-matrix heatmap: rows = true class, columns = predicted class.

    ``blocks`` is a list of (name, start, stop) leaf ranges; a thick line is drawn at the
    start of every block except the first.
    """
    matrix = np.asarray(cm)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")
    if not np.issubdtype(matrix.dtype, np.integer):
        raise TypeError(f"confusion matrix must hold integer counts, got dtype {matrix.dtype}")
    size = matrix.shape[0]
    if len(labels) != size:
        raise ValueError(f"got {len(labels)} labels for a {size}x{size} matrix")
    if cell_text not in CELL_TEXT_MODES:
        raise ValueError(f"cell_text must be one of {CELL_TEXT_MODES}, got {cell_text!r}")

    proportion = row_normalize(matrix)
    support = matrix.sum(axis=1)

    side = max(6.0, 0.9 * size + 2.5)
    figure = Figure(figsize=(side, side * 0.9))
    axes = figure.subplots()
    image = axes.imshow(proportion, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label="share of true class")

    axes.set_xticks(range(size))
    axes.set_xticklabels(list(labels), rotation=45, ha="right")
    axes.set_yticks(range(size))
    axes.set_yticklabels(
        [f"{label} (n={int(count)})" for label, count in zip(labels, support, strict=True)]
    )
    axes.set_xlabel("predicted class")
    axes.set_ylabel("true class")
    axes.set_title(title)

    for row in range(size):
        for col in range(size):
            count = int(matrix[row, col])
            if count == 0:
                continue
            if cell_text == "counts":
                text = str(count)
            else:
                percent = 100.0 * proportion[row, col]
                text = "<1" if percent < 0.5 else f"{percent:.0f}"
            axes.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if proportion[row, col] > 0.5 else "black",
            )

    if blocks:
        for _, start, _ in blocks:
            if start > 0:
                axes.axhline(start - 0.5, color="black", linewidth=1.5)
                axes.axvline(start - 0.5, color="black", linewidth=1.5)

    figure.tight_layout()
    return figure


def build_per_class_f1_figure(
    per_class: pd.DataFrame,
    *,
    title: str,
    interval_label: str = "95% problem-bootstrap interval",
) -> Figure:
    """Horizontal bars of per-class F1 with interval whiskers, first class on top.

    Expects the ``per_class`` table from ``summarize_bootstrap``. Whiskers are clipped to
    zero length if a percentile interval does not contain the point estimate, which can
    legitimately happen for thin classes.
    """
    missing = PER_CLASS_COLUMNS - set(per_class.columns)
    if missing:
        raise KeyError(f"per-class table is missing columns: {sorted(missing)}")

    names = per_class["class"].tolist()
    f1 = per_class["f1"].to_numpy(dtype=float)
    lower = np.clip(f1 - per_class["f1_lo"].to_numpy(dtype=float), 0.0, None)
    upper = np.clip(per_class["f1_hi"].to_numpy(dtype=float) - f1, 0.0, None)

    figure = Figure(figsize=(7.5, 0.55 * len(names) + 1.8))
    axes = figure.subplots()
    positions = np.arange(len(names))
    axes.barh(positions, f1, xerr=[lower, upper], color="#4C78A8", ecolor="black", capsize=3)
    axes.set_yticks(positions)
    axes.set_yticklabels(
        [
            f"{name} (n={int(count)})"
            for name, count in zip(names, per_class["support"], strict=True)
        ]
    )
    axes.invert_yaxis()
    axes.set_xlim(0.0, 1.08)
    axes.set_xticks(np.linspace(0.0, 1.0, 6))
    axes.set_xlabel(f"F1 (bar = point estimate, whisker = {interval_label})")
    axes.set_title(title)
    for position, value, extra in zip(positions, f1, upper, strict=True):
        axes.text(value + extra + 0.01, position, f"{value:.2f}", va="center", fontsize=9)
    figure.tight_layout()
    return figure


def save_figure(figure: Figure, path: Path, dpi: int = 150) -> Path:
    """Write ``figure`` as an image (creating folders); overwriting is allowed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
