"""Compute the M8.2 evaluation tables and figures from the saved prediction dumps.

Run after ``scripts/predict_split.py``:

    python scripts/evaluate_test.py

What it does
------------
1. Checks both dumps against the SHA-256 hashes recorded by M8.1 and against the 9-class
   manifest (hash-checked too). The model is never run here: only saved predictions are read.
2. Computes accuracy, macro-F1, per-class metrics and parent-level statistics for the TEST
   split with problem-level bootstrap intervals (2,000 draws, seed 42, 95% percentile).
3. Does the same for VALIDATION as a context row: validation was used to choose the best
   epoch, so it is slightly optimistic by construction, and the gap to test shows how much.
4. Writes CSV tables and PNG figures to ``reports/m8/`` plus a provenance file.

Only file paths are printed on purpose: the first time anyone reads a test number it should
be in the saved table, next to its interval.
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
from sdp.data.taxonomy import COARSE_ORDER
from sdp.evaluation.bootstrap import DEFAULT_LEVEL, DEFAULT_N_BOOT, DEFAULT_SEED
from sdp.evaluation.hierarchy import parent_blocks
from sdp.evaluation.metrics import LEAF_NAMES
from sdp.evaluation.plots import (
    build_confusion_figure,
    build_per_class_f1_figure,
    save_figure,
)
from sdp.evaluation.predictions import read_dump, require_hash, validate_against_manifest
from sdp.evaluation.split_report import SplitEvaluation, evaluate_split, problem_ids_for

SPLIT_CONFIG_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "split_config.json"
MANIFEST_KEY = "sample_manifest_9class.parquet"
PARENT_NAMES = [parent.value for parent in COARSE_ORDER]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "m8",
        help="folder that holds the M8.1 dumps; outputs are written here too",
    )
    return parser


def write_table(table: pd.DataFrame, path: Path) -> None:
    table.round(6).to_csv(path, index=False)
    print(path)


def write_matrix(matrix, names: list[str], path: Path) -> None:
    pd.DataFrame(matrix, index=names, columns=names).to_csv(path, index_label="true_class")
    print(path)


def write_figure(figure, path: Path) -> None:
    save_figure(figure, path)
    print(path)


def write_test_outputs(result: SplitEvaluation, out_dir: Path) -> None:
    write_table(result.overall, out_dir / "test_overall.csv")
    write_table(result.per_class, out_dir / "test_per_class.csv")
    write_table(result.parent, out_dir / "test_parent_stats.csv")
    write_matrix(result.confusion, list(LEAF_NAMES), out_dir / "test_confusion_9x9.csv")
    write_matrix(result.parent_confusion, PARENT_NAMES, out_dir / "test_confusion_parents.csv")

    size = f"{result.n_files:,} files, {result.n_problems:,} problems"
    blocks = parent_blocks()
    write_figure(
        build_confusion_figure(
            result.confusion,
            LEAF_NAMES,
            title=f"Test confusion matrix, counts ({size})",
            cell_text="counts",
            blocks=blocks,
        ),
        out_dir / "test_confusion_counts.png",
    )
    write_figure(
        build_confusion_figure(
            result.confusion,
            LEAF_NAMES,
            title=f"Test confusion matrix, % of true class ({size})",
            cell_text="percent",
            blocks=blocks,
        ),
        out_dir / "test_confusion_percent.png",
    )
    write_figure(
        build_confusion_figure(
            result.parent_confusion,
            PARENT_NAMES,
            title=f"Test confusion matrix by parent class, counts ({size})",
            cell_text="counts",
        ),
        out_dir / "test_confusion_parents.png",
    )
    write_figure(
        build_per_class_f1_figure(result.per_class, title=f"Test per-class F1 ({size})"),
        out_dir / "test_per_class_f1.png",
    )


def main(argv: list[str] | None = None) -> int:
    out_dir: Path = build_parser().parse_args(argv).reports_dir
    m8_1 = json.loads((out_dir / "m8_1_provenance.json").read_text(encoding="utf-8"))
    expected_manifest = json.loads(SPLIT_CONFIG_PATH.read_text(encoding="utf-8"))[
        "artifact_hashes"
    ][MANIFEST_KEY]

    manifest_sha = require_hash(MANIFEST_9CLASS_PATH, expected_manifest, "9-class manifest")
    manifest = pd.read_parquet(
        MANIFEST_9CLASS_PATH, columns=["submission_id", "problem_id", "split", "leaf_id"]
    )

    results: dict[str, SplitEvaluation] = {}
    dump_hashes: dict[str, str] = {}
    for split in ("test", "val"):
        name = f"{split}_predictions.parquet"
        dump_hashes[name] = require_hash(out_dir / name, m8_1["dump_sha256"][name], name)
        frame = read_dump(out_dir / name)
        validate_against_manifest(frame, manifest, split)
        results[split] = evaluate_split(
            frame,
            problem_ids_for(frame, manifest),
            n_boot=DEFAULT_N_BOOT,
            seed=DEFAULT_SEED,
            level=DEFAULT_LEVEL,
        )

    write_test_outputs(results["test"], out_dir)
    write_table(results["val"].overall, out_dir / "val_overall.csv")

    provenance = {
        "milestone": "M8.2",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bootstrap": {
            "unit": "problem",
            "n_boot": DEFAULT_N_BOOT,
            "seed": DEFAULT_SEED,
            "level": DEFAULT_LEVEL,
            "method": "percentile",
        },
        "manifest_sha256": manifest_sha,
        "dump_sha256": dump_hashes,
        "m8_1_best_pt_sha256": m8_1["best_pt_sha256"],
        "rows": {split: result.n_files for split, result in results.items()},
        "problems": {split: result.n_problems for split, result in results.items()},
    }
    provenance_path = out_dir / "m8_2_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(provenance_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
