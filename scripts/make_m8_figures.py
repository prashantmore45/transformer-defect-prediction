"""Build the M8 figure set from the committed result tables (Milestone 8.6).

    python scripts/make_m8_figures.py

No model is run and no test file is read: every figure is drawn from CSV tables already
committed under reports/m8/. The SHA-256 of every input table is recorded in
reports/figures/m8/figures_provenance.json, so each figure traces to the exact tables that
produced it. Figures are regenerable, so existing PNGs are overwritten.
"""

from __future__ import annotations

import json
import sys
import time

import matplotlib
import pandas as pd

from sdp.config import PROJECT_ROOT
from sdp.evaluation.metrics import LEAF_NAMES
from sdp.evaluation.plots import (
    build_forest_figure,
    build_grouped_bars_figure,
    build_interval_bars_figure,
    save_figure,
)
from sdp.evaluation.predictions import sha256_file

M8 = PROJECT_ROOT / "reports" / "m8"
OUT_DIR = PROJECT_ROOT / "reports" / "figures" / "m8"
SYSTEMS = ("codebert", "tfidf_logreg", "length_only", "stratified_random", "majority")
SYSTEM_LABELS = {
    "codebert": "CodeBERT",
    "tfidf_logreg": "TF-IDF + logistic regression",
    "length_only": "Length only",
    "stratified_random": "Stratified random",
    "majority": "Majority class",
}
LENGTH_BINS = ["<=128", "129-256", "257-512", "513-1024", ">1024"]
TIERS = {
    "exact_twin": "exact content twin",
    "same_problem_and_user": "same problem and same author",
    "same_problem_other_user": "same problem, other author",
    "problem_not_in_train": "problem not in train",
}
VISIBILITY = {
    "visible_untruncated": "error visible, file fits the window",
    "visible_truncated": "error visible, file truncated",
    "not_fully_visible": "error beyond the window",
}
COMPILE_CLASSES = ["SYNTAX", "SEMANTIC"]
CHANCE_MACRO_F1 = 1 / 9
METRIC_LABELS = {"macro_f1": "macro-F1", "accuracy": "accuracy"}
SHORT_LABELS = {"codebert": "CodeBERT", "tfidf_logreg": "TF-IDF"}

used_inputs: dict[str, str] = {}


def read_table(path, columns: list[str]) -> pd.DataFrame:
    """Read a committed CSV, check its columns, and record its SHA-256."""
    if not path.exists():
        raise SystemExit(f"missing input table: {path}")
    table = pd.read_csv(path)
    missing = set(columns) - set(table.columns)
    if missing:
        raise SystemExit(f"{path.name} is missing columns {sorted(missing)}")
    used_inputs[path.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(path)
    return table


def metric_row(table: pd.DataFrame, system: str, metric: str) -> pd.Series:
    return table[(table["system"] == system) & (table["metric"] == metric)].iloc[0]


def systems_figure(systems: pd.DataFrame, metric: str, title: str, chance: float | None):
    rows = systems[systems["metric"] == metric].set_index("system").loc[list(SYSTEMS)]
    return build_interval_bars_figure(
        [SYSTEM_LABELS[name] for name in SYSTEMS],
        rows["value"].to_numpy(),
        rows["ci_low"].to_numpy(),
        rows["ci_high"].to_numpy(),
        title=title,
        xlabel=(
            f"{METRIC_LABELS[metric]} (bar = estimate, whisker = 95% problem-bootstrap interval)"
        ),
        xlim=(0.0, 0.5),
        reference=chance,
        reference_label="chance level (1/9)" if chance is not None else None,
    )


def per_class_figure(differences: pd.DataFrame, per_class: pd.DataFrame):
    rows = differences[differences["versus"] == "tfidf_logreg"].set_index("class")
    rows = rows.loc[list(LEAF_NAMES)]
    support = per_class.set_index("class")["support"]
    return build_forest_figure(
        [f"{name} (n={int(support[name])})" for name in LEAF_NAMES],
        rows["f1_difference"].to_numpy(),
        rows["ci_low"].to_numpy(),
        rows["ci_high"].to_numpy(),
        title="CodeBERT minus TF-IDF: per-class F1 difference (test)",
        xlabel="F1 difference (filled marker = 95% interval excludes 0)",
    )


def length_figure(length_overall: pd.DataFrame):
    series = {}
    for system in ("codebert", "tfidf_logreg"):
        rows = length_overall[length_overall["system"] == system].set_index("slice")
        rows = rows.loc[LENGTH_BINS]
        series[SYSTEM_LABELS[system]] = (
            rows["accuracy"].to_numpy(),
            rows["accuracy_lo"].to_numpy(),
            rows["accuracy_hi"].to_numpy(),
        )
    return build_grouped_bars_figure(
        LENGTH_BINS,
        series,
        title="Test accuracy by file length (tokens)",
        ylabel="accuracy",
        ylim=(0.0, 0.7),
    )


def visibility_figure(per_class: pd.DataFrame, system: str):
    series = {}
    for slice_name, label in VISIBILITY.items():
        rows = per_class[(per_class["system"] == system) & (per_class["slice"] == slice_name)]
        rows = rows.set_index("class").loc[COMPILE_CLASSES]
        series[label] = (
            rows["recall"].to_numpy(),
            rows["recall_lo"].to_numpy(),
            rows["recall_hi"].to_numpy(),
        )
    return build_grouped_bars_figure(
        COMPILE_CLASSES,
        series,
        title=(
            f"Recall for compile-error files by where the first error is ({SHORT_LABELS[system]})"
        ),
        ylabel="recall",
        ylim=(0.0, 0.8),
    )


def leakage_headline_figure(headline: pd.DataFrame, paired: pd.DataFrame, files: dict):
    entries = [
        (
            f"R on its own random test ({files['random_test']} files)",
            metric_row(headline, "random_split_model", "macro_f1"),
        ),
        (
            f"P on the problem-level test ({files['problem_test']} files)",
            metric_row(headline, "problem_split_model", "macro_f1"),
        ),
        (
            f"R on the {files['shared']} shared files",
            metric_row(paired, "random_split_model", "macro_f1"),
        ),
        (
            f"P on the same {files['shared']} shared files",
            metric_row(paired, "problem_split_model", "macro_f1"),
        ),
    ]
    return build_interval_bars_figure(
        [label for label, _ in entries],
        [row["value"] for _, row in entries],
        [row["ci_low"] for _, row in entries],
        [row["ci_high"] for _, row in entries],
        title="Test macro-F1: random-split model R vs problem-split model P",
        xlabel="macro-F1 (bar = estimate, whisker = 95% problem-bootstrap interval)",
        xlim=(0.0, 0.6),
    )


def leakage_tiers_figure(tiers: pd.DataFrame, headline: pd.DataFrame):
    rows = tiers.set_index("slice").loc[list(TIERS)]
    p_accuracy = float(metric_row(headline, "problem_split_model", "accuracy")["value"])
    return build_interval_bars_figure(
        [f"{TIERS[name]} (n={int(rows.loc[name, 'n_files'])})" for name in TIERS],
        rows["accuracy"].to_numpy(),
        rows["accuracy_lo"].to_numpy(),
        rows["accuracy_hi"].to_numpy(),
        title="Random-split model R: test accuracy by leakage tier",
        xlabel="accuracy (bar = estimate, whisker = 95% problem-bootstrap interval)",
        reference=p_accuracy,
        reference_label=f"P overall accuracy ({p_accuracy:.3f})",
    )


def main() -> int:
    interval_columns = ["system", "metric", "value", "ci_low", "ci_high"]
    systems = read_table(M8 / "compare_test_systems.csv", interval_columns)
    differences = read_table(
        M8 / "compare_test_per_class_differences.csv",
        ["versus", "class", "f1_difference", "ci_low", "ci_high"],
    )
    per_class = read_table(M8 / "test_per_class.csv", ["class", "support"])
    length_overall = read_table(
        M8 / "slices" / "length_overall.csv",
        ["system", "slice", "accuracy", "accuracy_lo", "accuracy_hi"],
    )
    visibility = read_table(
        M8 / "slices" / "visibility_per_class.csv",
        ["system", "slice", "class", "recall", "recall_lo", "recall_hi"],
    )
    headline = read_table(M8 / "random_split" / "compare_headline_systems.csv", interval_columns)
    paired = read_table(M8 / "random_split" / "compare_paired_systems.csv", interval_columns)
    tiers = read_table(
        M8 / "random_split" / "compare_tiers_overall.csv",
        ["slice", "n_files", "accuracy", "accuracy_lo", "accuracy_hi"],
    )
    provenance_path = M8 / "random_split" / "compare_provenance.json"
    files = json.loads(provenance_path.read_text(encoding="utf-8"))["files"]
    used_inputs[provenance_path.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(provenance_path)

    figures = {
        "m8_systems_macro_f1.png": systems_figure(
            systems, "macro_f1", "Test macro-F1 by system", CHANCE_MACRO_F1
        ),
        "m8_systems_accuracy.png": systems_figure(
            systems, "accuracy", "Test accuracy by system", None
        ),
        "m8_codebert_vs_tfidf_per_class.png": per_class_figure(differences, per_class),
        "m8_length_accuracy.png": length_figure(length_overall),
        "m8_visibility_recall_codebert.png": visibility_figure(visibility, "codebert"),
        "m8_visibility_recall_tfidf_logreg.png": visibility_figure(visibility, "tfidf_logreg"),
        "m8_leakage_headline_vs_paired.png": leakage_headline_figure(headline, paired, files),
        "m8_leakage_tiers.png": leakage_tiers_figure(tiers, headline),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        print(save_figure(figure, OUT_DIR / name))

    provenance = {
        "milestone": "M8.6",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "matplotlib": matplotlib.__version__,
        "inputs_sha256": used_inputs,
        "outputs": sorted(figures),
    }
    (OUT_DIR / "figures_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
