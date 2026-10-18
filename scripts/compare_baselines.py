"""Compare CodeBERT with the baselines on the same files (Milestone 8.3).

Run after ``scripts/predict_split.py`` (M8.1), ``scripts/evaluate_test.py`` (M8.2) and
``scripts/predict_baselines.py`` (M8.3):

    python scripts/compare_baselines.py

What it does
------------
1. Verifies every input: each dump against the SHA-256 recorded by its own provenance file,
   the manifest against ``split_config.json``, and every dump against the manifest.
2. Aligns CodeBERT and the four baselines on submission_id and looks up each file's problem.
3. Runs the paired problem-level bootstrap (``sdp.evaluation.paired``) with CodeBERT as the
   reference, for TEST (the result) and VALIDATION (context). The bootstrap draws are the
   same as in M8.2.
4. Cross-checks CodeBERT's own test intervals against ``reports/m8/test_overall.csv``. If
   they differ, one of the two pipelines is wrong: nothing is written and the script
   exits with code 3.
5. Writes CSV tables to ``reports/m8/`` and prints the headline numbers with intervals.
   This is the first place a baseline test number is read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from sdp.config import PROJECT_ROOT
from sdp.data.dataset import MANIFEST_9CLASS_PATH
from sdp.evaluation.baselines import read_baseline_dump
from sdp.evaluation.bootstrap import DEFAULT_LEVEL, DEFAULT_N_BOOT, DEFAULT_SEED
from sdp.evaluation.metrics import per_class_report
from sdp.evaluation.paired import PairedComparison, align_systems, compare_systems
from sdp.evaluation.predictions import read_dump, require_hash, validate_against_manifest
from sdp.evaluation.split_report import problem_ids_for

SPLIT_CONFIG_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "split_config.json"
MANIFEST_KEY = "sample_manifest_9class.parquet"
REFERENCE = "codebert"
BASELINES = ("tfidf_logreg", "length_only", "stratified_random", "majority")
STRONGEST_BASELINE = "tfidf_logreg"
CONSISTENCY_TOLERANCE = 1e-5  # test_overall.csv is rounded to 6 decimals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "m8",
        help="folder that holds the dumps and provenance files; outputs are written here too",
    )
    return parser


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}", flush=True)


def load_split_frames(
    reports_dir: Path,
    split: str,
    codebert_hashes: dict[str, str],
    baseline_hashes: dict[str, str],
    verified: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Read CodeBERT's and the baselines' dumps for one split, hash-checking each file."""
    frames: dict[str, pd.DataFrame] = {}
    name = f"{split}_predictions.parquet"
    verified[name] = require_hash(reports_dir / name, codebert_hashes[name], name)
    frames[REFERENCE] = read_dump(reports_dir / name)[["submission_id", "y_true", "y_pred"]]
    for baseline in BASELINES:
        name = f"baseline_{baseline}_{split}.parquet"
        verified[name] = require_hash(reports_dir / name, baseline_hashes[name], name)
        frames[baseline] = read_baseline_dump(reports_dir / name)
    return frames


def check_against_m8_2(comparison: PairedComparison, reports_dir: Path) -> list[str]:
    """Differences between CodeBERT's intervals here and the M8.2 table (empty = consistent)."""
    stored = pd.read_csv(reports_dir / "test_overall.csv").set_index("metric")
    mine = comparison.systems[comparison.systems["system"] == REFERENCE].set_index("metric")
    problems: list[str] = []
    for metric in ("accuracy", "macro_f1"):
        for column in ("value", "ci_low", "ci_high"):
            difference = abs(float(mine.loc[metric, column]) - float(stored.loc[metric, column]))
            if difference > CONSISTENCY_TOLERANCE:
                problems.append(f"{metric} {column}: differs from test_overall.csv by {difference}")
    return problems


def interval_text(value: float, low: float, high: float) -> str:
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def print_systems(title: str, systems: pd.DataFrame) -> None:
    print(f"\n{title}")
    for name in systems["system"].unique():
        rows = systems[systems["system"] == name].set_index("metric")
        accuracy, macro = rows.loc["accuracy"], rows.loc["macro_f1"]
        print(
            f"  {name:<18} accuracy "
            f"{interval_text(accuracy['value'], accuracy['ci_low'], accuracy['ci_high'])}   "
            f"macro-F1 {interval_text(macro['value'], macro['ci_low'], macro['ci_high'])}"
        )


def print_differences(title: str, differences: pd.DataFrame) -> None:
    print(f"\n{title}")
    for name in differences["versus"].unique():
        rows = differences[differences["versus"] == name].set_index("metric")
        accuracy, macro = rows.loc["accuracy"], rows.loc["macro_f1"]
        print(
            f"  vs {name:<18} accuracy "
            f"{interval_text(accuracy['value'], accuracy['ci_low'], accuracy['ci_high'])}   "
            f"macro-F1 {interval_text(macro['value'], macro['ci_low'], macro['ci_high'])}   "
            f"draws not ahead (macro-F1) {macro['frac_draws_not_ahead']:.3f}"
        )


def per_class_table(
    y_true: object, predictions: dict[str, object], comparison: PairedComparison
) -> pd.DataFrame:
    """Per-class F1 of CodeBERT and the strongest baseline, with the paired difference."""
    reference_f1 = per_class_report(y_true, predictions[REFERENCE]).set_index("class")
    other_f1 = per_class_report(y_true, predictions[STRONGEST_BASELINE]).set_index("class")
    rows = comparison.per_class_differences
    rows = rows[rows["versus"] == STRONGEST_BASELINE].set_index("class")
    table = pd.DataFrame(
        {
            "support": reference_f1["support"],
            f"{REFERENCE}_f1": reference_f1["f1"].round(3),
            f"{STRONGEST_BASELINE}_f1": other_f1["f1"].round(3),
        }
    )
    table["difference [95% interval]"] = [
        interval_text(row.f1_difference, row.ci_low, row.ci_high)
        for row in rows.loc[table.index].itertuples()
    ]
    return table


def write_table(table: pd.DataFrame, path: Path) -> None:
    table.round(6).to_csv(path, index=False)
    print(path)


def main(argv: list[str] | None = None) -> int:
    reports_dir: Path = build_parser().parse_args(argv).reports_dir
    codebert_hashes = json.loads((reports_dir / "m8_1_provenance.json").read_text("utf-8"))[
        "dump_sha256"
    ]
    baseline_hashes = json.loads((reports_dir / "m8_3_provenance.json").read_text("utf-8"))[
        "dump_sha256"
    ]
    expected_manifest = json.loads(SPLIT_CONFIG_PATH.read_text(encoding="utf-8"))[
        "artifact_hashes"
    ][MANIFEST_KEY]
    manifest_sha = require_hash(MANIFEST_9CLASS_PATH, expected_manifest, "9-class manifest")
    manifest = pd.read_parquet(
        MANIFEST_9CLASS_PATH, columns=["submission_id", "problem_id", "split", "leaf_id"]
    )

    verified: dict[str, str] = {}
    results: dict[str, PairedComparison] = {}
    test_inputs: tuple[object, dict[str, object]] | None = None
    for split in ("val", "test"):
        frames = load_split_frames(reports_dir, split, codebert_hashes, baseline_hashes, verified)
        for frame in frames.values():
            validate_against_manifest(frame, manifest, split)
        ids, y_true, predictions = align_systems(frames)
        groups = problem_ids_for(pd.DataFrame({"submission_id": ids}), manifest)
        results[split] = compare_systems(
            y_true,
            predictions,
            groups,
            reference=REFERENCE,
            n_boot=DEFAULT_N_BOOT,
            seed=DEFAULT_SEED,
            level=DEFAULT_LEVEL,
        )
        if split == "test":
            test_inputs = (y_true, predictions)

    problems = check_against_m8_2(results["test"], reports_dir)
    if problems:
        for problem in problems:
            log(f"STOP: {problem}")
        log("CodeBERT's intervals disagree with M8.2. Nothing was written.")
        return 3
    log("consistency check passed: CodeBERT's test intervals equal test_overall.csv")

    test = results["test"]
    write_table(test.systems, reports_dir / "compare_test_systems.csv")
    write_table(test.differences, reports_dir / "compare_test_differences.csv")
    write_table(test.per_class_differences, reports_dir / "compare_test_per_class_differences.csv")
    write_table(results["val"].systems, reports_dir / "compare_val_systems.csv")

    provenance = {
        "milestone": "M8.3",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference": REFERENCE,
        "bootstrap": {
            "unit": "problem",
            "n_boot": DEFAULT_N_BOOT,
            "seed": DEFAULT_SEED,
            "level": DEFAULT_LEVEL,
            "method": "percentile, paired (same draws for every system)",
        },
        "manifest_sha256": manifest_sha,
        "verified_input_sha256": verified,
        "rows": {split: result.n_files for split, result in results.items()},
        "problems": {split: result.n_problems for split, result in results.items()},
    }
    provenance_path = reports_dir / "compare_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(provenance_path)

    print_systems("TEST: each system, 95% problem-bootstrap intervals", test.systems)
    print_differences("TEST: CodeBERT minus each baseline (paired intervals)", test.differences)
    y_true, predictions = test_inputs
    print(f"\nTEST per-class F1: CodeBERT vs {STRONGEST_BASELINE}")
    print(per_class_table(y_true, predictions, test).to_string())
    print_systems("VALIDATION (context): each system", results["val"].systems)
    return 0


if __name__ == "__main__":
    sys.exit(main())
