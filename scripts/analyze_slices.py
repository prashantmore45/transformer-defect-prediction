"""M8.4 slice analysis on the TEST split (descriptive; no model or config changes).

Run after M8.1-M8.3 and scripts/visibility_per_file.py:

    python scripts/analyze_slices.py

Five analyses, each run for CodeBERT and (where it has predictions) the TF-IDF baseline as
a control, all with problem-level bootstrap intervals over the SAME problems and draws:

1. length bins (full token length), 2. confidence quintiles (CodeBERT only),
3. author overlap (test files whose user_id appears in train), 4. runtime-class input source
(all / verified sample input only / empty stdin), 5. first compile error visible or not
(SYNTAX and SEMANTIC files, with a within-truncated-files control).

Gates before anything is computed: every input's SHA-256 is verified; cached_length must
equal min(token_length, 512) for every test file; and a one-slice 'all files' bootstrap must
reproduce CodeBERT's accuracy interval in reports/m8/test_overall.csv. Failure exits with
code 3 and writes nothing.

Outputs go to reports/m8/slices/ (CSV tables) plus reports/m8/slices/provenance.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.config import PROJECT_ROOT
from sdp.data.dataset import MANIFEST_9CLASS_PATH
from sdp.data.taxonomy import LEAF_TO_ID, DefectClass
from sdp.evaluation.baselines import read_baseline_dump
from sdp.evaluation.bootstrap import DEFAULT_LEVEL, DEFAULT_N_BOOT, DEFAULT_SEED
from sdp.evaluation.metrics import LEAF_NAMES
from sdp.evaluation.paired import align_systems
from sdp.evaluation.predictions import (
    LOGIT_COLUMNS,
    read_dump,
    require_hash,
    validate_against_manifest,
)
from sdp.evaluation.slice_tables import (
    calibration_table,
    confidence_by_predicted_class,
    predicted_parent_shares,
)
from sdp.evaluation.slices import (
    bootstrap_slices,
    common_classes,
    interval_masks,
    softmax_confidence,
)

CODEBERT = "codebert"
TFIDF = "tfidf_logreg"
SYSTEMS = (CODEBERT, TFIDF)
MIN_SUPPORT = 20
TOKEN_CAP = 512
CONSISTENCY_TOLERANCE = 1e-5

SPLIT_CONFIG_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "split_config.json"
MANIFEST_KEY = "sample_manifest_9class.parquet"
TOKENS_CSV = PROJECT_ROOT / "reports" / "tokenization_full.csv"

RUNTIME_IDS = [
    LEAF_TO_ID[leaf]
    for leaf in (DefectClass.SIGSEGV, DefectClass.SIGFPE, DefectClass.SIGABRT, DefectClass.NZEC)
]
COMPILE_IDS = [LEAF_TO_ID[DefectClass.SYNTAX], LEAF_TO_ID[DefectClass.SEMANTIC]]


@dataclass(frozen=True)
class SplitData:
    """Everything the analyses need, aligned to one file order (the test split)."""

    y_true: np.ndarray
    predictions: dict[str, np.ndarray]
    logits: np.ndarray
    groups: np.ndarray
    token_length: np.ndarray
    user_seen: np.ndarray
    input_source: np.ndarray
    visibility: np.ndarray
    hashes: dict[str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "m8",
        help="folder with the M8 dumps and provenance files; outputs go to <dir>/slices",
    )
    return parser


def load_data(reports_dir: Path) -> SplitData:
    m8_1 = json.loads((reports_dir / "m8_1_provenance.json").read_text("utf-8"))["dump_sha256"]
    m8_3 = json.loads((reports_dir / "m8_3_provenance.json").read_text("utf-8"))["dump_sha256"]
    expected_manifest = json.loads(SPLIT_CONFIG_PATH.read_text("utf-8"))["artifact_hashes"][
        MANIFEST_KEY
    ]
    codebert_name = "test_predictions.parquet"
    tfidf_name = "baseline_tfidf_logreg_test.parquet"
    hashes = {
        "manifest": require_hash(MANIFEST_9CLASS_PATH, expected_manifest, "9-class manifest"),
        codebert_name: require_hash(
            reports_dir / codebert_name, m8_1[codebert_name], codebert_name
        ),
        tfidf_name: require_hash(reports_dir / tfidf_name, m8_3[tfidf_name], tfidf_name),
    }
    manifest = pd.read_parquet(
        MANIFEST_9CLASS_PATH,
        columns=["submission_id", "problem_id", "user_id", "split", "leaf_id", "input_source"],
    )
    codebert = read_dump(reports_dir / codebert_name)
    tfidf = read_baseline_dump(reports_dir / tfidf_name)
    for frame in (codebert, tfidf):
        validate_against_manifest(frame, manifest, "test")
    ids, y_true, predictions = align_systems(
        {CODEBERT: codebert[["submission_id", "y_true", "y_pred"]], TFIDF: tfidf}
    )

    by_id = codebert.set_index("submission_id").loc[ids]
    test_rows = manifest[manifest["split"] == "test"].set_index("submission_id").loc[ids]
    train_users = set(manifest.loc[manifest["split"] == "train", "user_id"].astype(str))

    tokens = pd.read_csv(TOKENS_CSV, usecols=["submission_id", "token_length"])
    tokens = tokens.set_index("submission_id")
    if not tokens.index.is_unique:
        raise SystemExit("tokenization_full.csv has duplicate submission_id values")
    token_length = tokens["token_length"].loc[ids].to_numpy()
    if not np.array_equal(np.minimum(token_length, TOKEN_CAP), by_id["cached_length"].to_numpy()):
        raise SystemExit("cached_length != min(token_length, 512) for some test files")

    visibility = pd.read_csv(reports_dir / "visibility_per_file.csv")
    visibility = visibility[visibility["split"] == "test"].set_index("submission_id")["outcome"]
    return SplitData(
        y_true=y_true,
        predictions=predictions,
        logits=by_id[list(LOGIT_COLUMNS)].to_numpy(),
        groups=test_rows["problem_id"].astype(str).to_numpy(),
        token_length=token_length,
        user_seen=test_rows["user_id"].astype(str).isin(train_users).to_numpy(),
        input_source=test_rows["input_source"].astype(object).fillna("").to_numpy(dtype=str),
        visibility=visibility.reindex(ids).fillna("").to_numpy(dtype=str),
        hashes=hashes,
    )


def check_against_m8_2(data: SplitData, reports_dir: Path) -> list[str]:
    """Problems found when comparing a one-slice bootstrap with M8.2's accuracy interval."""
    everything = np.ones(data.y_true.shape[0], dtype=bool)
    result = bootstrap_slices(
        data.y_true,
        data.predictions[CODEBERT],
        data.groups,
        {"all": everything},
        n_boot=DEFAULT_N_BOOT,
        seed=DEFAULT_SEED,
        level=DEFAULT_LEVEL,
    )
    stored = pd.read_csv(reports_dir / "test_overall.csv").set_index("metric").loc["accuracy"]
    row = result.overall.iloc[0]
    problems = []
    for mine, column in (
        (row["accuracy"], "value"),
        (row["accuracy_lo"], "ci_low"),
        (row["accuracy_hi"], "ci_high"),
    ):
        difference = abs(float(mine) - float(stored[column]))
        if difference > CONSISTENCY_TOLERANCE:
            problems.append(f"accuracy {column} differs from test_overall.csv by {difference}")
    return problems


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def add_support_flag(differences: pd.DataFrame, per_class: pd.DataFrame) -> pd.DataFrame:
    """Add min_files (files in the smaller slice, per-class rows only) and low_support."""
    support = {
        (system, name, label): count
        for system, name, label, count in zip(
            per_class["system"],
            per_class["slice"],
            per_class["class"],
            per_class["support"],
            strict=True,
        )
    }
    min_files: list[float] = []
    for system, first, second, metric in zip(
        differences["system"],
        differences["first"],
        differences["second"],
        differences["metric"],
        strict=True,
    ):
        if metric.startswith("recall:"):
            label = metric.split(":", 1)[1]
            min_files.append(min(support[(system, first, label)], support[(system, second, label)]))
        else:
            min_files.append(float("nan"))
    flagged = differences.copy()
    flagged["min_files"] = min_files
    flagged["low_support"] = flagged["min_files"] < MIN_SUPPORT  # NaN compares False
    return flagged


def run_systems(
    data: SplitData,
    slices: dict[str, np.ndarray],
    *,
    pairs: list[tuple[str, str]] | None = None,
    macro_classes: list[int] | None = None,
    systems: tuple[str, ...] = SYSTEMS,
) -> dict[str, pd.DataFrame]:
    """Bootstrap the slices for each system and stack the results with a 'system' column."""
    collected: dict[str, list[pd.DataFrame]] = {"overall": [], "per_class": [], "differences": []}
    for system in systems:
        result = bootstrap_slices(
            data.y_true,
            data.predictions[system],
            data.groups,
            slices,
            pairs=pairs or (),
            n_boot=DEFAULT_N_BOOT,
            seed=DEFAULT_SEED,
            level=DEFAULT_LEVEL,
            min_class_support=MIN_SUPPORT,
            macro_classes=macro_classes,
        )
        for key, table in (
            ("overall", result.overall),
            ("per_class", result.per_class),
            ("differences", result.differences),
        ):
            if not table.empty:
                labelled = table.copy()
                labelled.insert(0, "system", system)
                collected[key].append(labelled)

    tables = {
        key: pd.concat(frames, ignore_index=True) for key, frames in collected.items() if frames
    }
    if "differences" in tables:
        tables["differences"] = add_support_flag(tables["differences"], tables["per_class"])
    return tables


def choose_macro(data: SplitData, slices: dict[str, np.ndarray], label: str) -> list[int] | None:
    classes = common_classes(data.y_true, slices, MIN_SUPPORT)
    names = [LEAF_NAMES[index] for index in classes]
    print(f"{label}: macro-recall over classes with >= {MIN_SUPPORT} files in every slice: {names}")
    return classes or None


def interval_text(value: float, low: float, high: float) -> str:
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def excludes_zero(low: float, high: float) -> bool:
    return bool(low > 0 or high < 0)


def show_overall(title: str, overall: pd.DataFrame) -> None:
    print(f"\n{title}")
    rows = list(overall.itertuples())
    view = pd.DataFrame(
        {
            "system": overall["system"],
            "slice": overall["slice"],
            "files": overall["n_files"],
            "problems": overall["n_problems"],
            "accuracy": [interval_text(r.accuracy, r.accuracy_lo, r.accuracy_hi) for r in rows],
            "macro_recall": [
                interval_text(r.macro_recall, r.macro_recall_lo, r.macro_recall_hi) for r in rows
            ],
        }
    )
    print(view.to_string(index=False))


def show_differences(title: str, differences: pd.DataFrame) -> None:
    print(f"\n{title}")
    rows = list(differences.itertuples())
    view = pd.DataFrame(
        {
            "system": differences["system"],
            "first - second": differences["first"] + " - " + differences["second"],
            "metric": differences["metric"],
            "difference [95% interval]": [
                interval_text(r.value, r.ci_low, r.ci_high) for r in rows
            ],
            "excludes 0": ["yes" if excludes_zero(r.ci_low, r.ci_high) else "" for r in rows],
            "min files": ["" if pd.isna(n) else str(int(n)) for n in differences["min_files"]],
            "note": ["LOW n" if flag else "" for flag in differences["low_support"]],
        }
    )
    print(view.to_string(index=False))


def show_recall_pivot(title: str, per_class: pd.DataFrame, system: str, order: list[str]) -> None:
    rows = per_class[per_class["system"] == system]
    recall = rows.pivot(index="class", columns="slice", values="recall").reindex(LEAF_NAMES)
    support = rows.pivot(index="class", columns="slice", values="support").reindex(LEAF_NAMES)
    print(f"\n{title}: recall (first table) and files (second table), {system}")
    print(recall[order].round(2).to_string())
    print(support[order].fillna(0).astype(int).to_string())


def slice_order(tables: dict[str, pd.DataFrame]) -> list[str]:
    return tables["overall"]["slice"].unique().tolist()


# --------------------------------------------------------------------------- #
# The five analyses; each returns (tables, macro class IDs or None, notes)
# --------------------------------------------------------------------------- #


def length_analysis(data: SplitData) -> tuple[dict[str, pd.DataFrame], list[int] | None, dict]:
    slices = interval_masks(
        data.token_length,
        [0, 128, 256, 512, 1024, np.inf],
        ["<=128", "129-256", "257-512", "513-1024", ">1024"],
    )
    macro = choose_macro(data, slices, "length bins")
    tables = run_systems(data, slices, macro_classes=macro)
    show_overall("LENGTH BINS (full token length)", tables["overall"])
    for system in SYSTEMS:
        show_recall_pivot("Length bins", tables["per_class"], system, slice_order(tables))
    return tables, macro, {}


def confidence_analysis(data: SplitData) -> tuple[dict[str, pd.DataFrame], list[int] | None, dict]:
    max_prob, margin = softmax_confidence(data.logits)
    cuts = np.quantile(max_prob, [0.2, 0.4, 0.6, 0.8]).tolist()
    slices = interval_masks(
        max_prob, [-np.inf, *cuts, np.inf], ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]
    )
    macro = choose_macro(data, slices, "confidence quintiles")
    tables = run_systems(data, slices, macro_classes=macro, systems=(CODEBERT,))
    correct = data.predictions[CODEBERT] == data.y_true
    tables["calibration"] = calibration_table(max_prob, correct, slices)
    tables["by_predicted_class"] = confidence_by_predicted_class(
        data.y_true, data.predictions[CODEBERT], max_prob, margin
    )
    show_overall("CONFIDENCE QUINTILES (CodeBERT; Q1 = least confident fifth)", tables["overall"])
    print("\nCalibration: mean confidence vs accuracy per quintile")
    print(tables["calibration"].round(3).to_string(index=False))
    print("\nConfidence by predicted class (is LOGICAL a default class?)")
    print(tables["by_predicted_class"].round(3).to_string(index=False))
    return tables, macro, {"quintile_cuts": cuts}


def user_overlap_analysis(
    data: SplitData,
) -> tuple[dict[str, pd.DataFrame], list[int] | None, dict]:
    slices = {"seen_user": data.user_seen, "unseen_user": ~data.user_seen}
    macro = choose_macro(data, slices, "user overlap")
    tables = run_systems(data, slices, pairs=[("seen_user", "unseen_user")], macro_classes=macro)
    show_overall("USER OVERLAP (author also in train vs not)", tables["overall"])
    show_differences("USER OVERLAP: seen minus unseen", tables["differences"])
    return tables, macro, {}


def runtime_input_analysis(
    data: SplitData,
) -> tuple[dict[str, pd.DataFrame], list[int] | None, dict]:
    runtime = np.isin(data.y_true, RUNTIME_IDS)
    slices = {
        "runtime_all": runtime,
        "runtime_verified_only": runtime & (data.input_source == "sample_verified"),
        "runtime_empty_stdin": runtime & (data.input_source == "empty_stdin"),
    }
    pairs = [
        ("runtime_verified_only", "runtime_all"),
        ("runtime_empty_stdin", "runtime_verified_only"),
    ]
    tables = run_systems(data, slices, pairs=pairs, macro_classes=RUNTIME_IDS)
    show_overall("RUNTIME CLASSES BY INPUT SOURCE", tables["overall"])
    show_recall_pivot("Runtime input source", tables["per_class"], CODEBERT, slice_order(tables))
    print(
        "\nNote: the empty-stdin slice has very few files per class; read its file counts "
        "before reading any difference involving it."
    )
    show_differences("RUNTIME INPUT SOURCE: differences", tables["differences"])
    return tables, RUNTIME_IDS, {}


def visibility_analysis(
    data: SplitData,
) -> tuple[dict[str, pd.DataFrame], list[int] | None, dict]:
    compile_files = np.isin(data.y_true, COMPILE_IDS)
    truncated = data.token_length > TOKEN_CAP
    fully = data.visibility == "fully_visible"
    not_fully = np.isin(data.visibility, ["partly_visible", "not_visible"])
    slices = {
        "visible_untruncated": compile_files & fully & ~truncated,
        "visible_truncated": compile_files & fully & truncated,
        "not_fully_visible": compile_files & not_fully,
    }
    untruncated_but_hidden = int((compile_files & not_fully & ~truncated).sum())
    pairs = [
        ("visible_truncated", "not_fully_visible"),
        ("visible_untruncated", "not_fully_visible"),
        ("visible_untruncated", "visible_truncated"),
    ]
    tables = run_systems(data, slices, pairs=pairs, macro_classes=COMPILE_IDS)
    shares = []
    for system in SYSTEMS:
        table = predicted_parent_shares(data.predictions[system], slices)
        table.insert(0, "system", system)
        shares.append(table)
    tables["predicted_parent"] = pd.concat(shares, ignore_index=True)

    measured = int((compile_files & (fully | not_fully)).sum())
    print(
        f"\nVISIBILITY: {measured} measured SYNTAX/SEMANTIC test files "
        f"({int(compile_files.sum()) - measured} excluded: short file or no usable error line); "
        f"{untruncated_but_hidden} hidden-error files were NOT truncated (expected 0)"
    )
    show_overall("VISIBILITY SLICES (SYNTAX and SEMANTIC files)", tables["overall"])
    show_recall_pivot("Visibility", tables["per_class"], CODEBERT, slice_order(tables))
    show_differences("VISIBILITY: first minus second", tables["differences"])
    print("\nWhere do these files get predicted? (share under each parent class)")
    print(tables["predicted_parent"].round(3).to_string(index=False))
    return tables, COMPILE_IDS, {"hidden_error_but_untruncated": untruncated_but_hidden}


ANALYSES = {
    "length": length_analysis,
    "confidence": confidence_analysis,
    "user_overlap": user_overlap_analysis,
    "runtime_input": runtime_input_analysis,
    "visibility": visibility_analysis,
}


def main(argv: list[str] | None = None) -> int:
    reports_dir: Path = build_parser().parse_args(argv).reports_dir
    data = load_data(reports_dir)
    problems = check_against_m8_2(data, reports_dir)
    if problems:
        for problem in problems:
            print(f"STOP: {problem}")
        print("The slice machinery disagrees with M8.2. Nothing was written.")
        return 3
    print("consistency check passed: one-slice accuracy interval equals test_overall.csv")

    out_dir = reports_dir / "slices"
    out_dir.mkdir(parents=True, exist_ok=True)
    macro_used: dict[str, object] = {}
    sizes: dict[str, dict[str, int]] = {}
    notes: dict[str, dict] = {}
    written = 0
    for name, analysis in ANALYSES.items():
        tables, macro, extra = analysis(data)
        for key, table in tables.items():
            table.round(6).to_csv(out_dir / f"{name}_{key}.csv", index=False)
            written += 1
        macro_used[name] = [LEAF_NAMES[i] for i in macro] if macro else "per-slice class sets"
        overall = tables["overall"]
        sizes[name] = overall[overall["system"] == CODEBERT].set_index("slice")["n_files"].to_dict()
        notes[name] = extra

    provenance = {
        "milestone": "M8.4",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bootstrap": {
            "unit": "problem",
            "n_boot": DEFAULT_N_BOOT,
            "seed": DEFAULT_SEED,
            "level": DEFAULT_LEVEL,
            "method": "percentile; shared draws across all slices and systems",
        },
        "min_class_support": MIN_SUPPORT,
        "input_sha256": data.hashes,
        "macro_recall_classes": macro_used,
        "slice_sizes_files": sizes,
        "notes": notes,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"\nwrote {written} tables and provenance.json to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
