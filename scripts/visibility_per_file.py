"""Per-file visibility of the first compile error (Milestone 8.4).

Reuses the exact measurement from scripts/measure_truncation_by_class.py (part b): is the
compiler's first error line inside the 510 content tokens CodeBERT sees? That script writes
only percentages; this one keeps the per-file outcome so M8.4 can slice test results by it.
The functions are imported from that script's file (not copied), so the two cannot drift.

Gate: the per-file results, re-aggregated, must reproduce reports/m6_error_visibility.csv
(every count, all four strata). Failure writes nothing.

Run locally (needs the corpus, the M4 results and the HF tokenizer cache):

    python scripts/visibility_per_file.py

Writes reports/m8/visibility_per_file.csv: one row per SYNTAX/SEMANTIC file, all splits.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

from sdp.config import PROJECT_ROOT
from sdp.data.dataset import MANIFEST_9CLASS_PATH
from sdp.data.tokenization import load_tokenizer

OUT_PATH = PROJECT_ROOT / "reports" / "m8" / "visibility_per_file.csv"
M6_SUMMARY = PROJECT_ROOT / "reports" / "m6_error_visibility.csv"
COUNT_COLUMNS = [
    "n",
    "short_file",
    "no_error_line",
    "measured",
    "fully_visible",
    "partly_visible",
    "not_visible",
    "echo_no_snippet",
    "echo_mismatch",
]


def load_measurement_module() -> ModuleType:
    """Import scripts/measure_truncation_by_class.py as a module (its main() is not run)."""
    path = Path(__file__).with_name("measure_truncation_by_class.py")
    spec = importlib.util.spec_from_file_location("measure_truncation_by_class", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate(module: ModuleType, results: pd.DataFrame) -> list[str]:
    """Differences between the re-aggregated per-file results and the M6 report."""
    syntax = results[results["leaf"] == "SYNTAX"]
    strata = [
        ("SYNTAX (all)", syntax),
        ("SYNTAX (missing-include only)", syntax[syntax["missing_include"]]),
        ("SYNTAX (excl. missing-include)", syntax[~syntax["missing_include"]]),
        ("SEMANTIC", results[results["leaf"] == "SEMANTIC"]),
    ]
    mine = pd.DataFrame([module._summarize(name, frame) for name, frame in strata])
    reference = pd.read_csv(M6_SUMMARY)
    problems: list[str] = []
    if mine["stratum"].tolist() != reference["stratum"].tolist():
        return [f"strata differ: {mine['stratum'].tolist()} vs {reference['stratum'].tolist()}"]
    for column in COUNT_COLUMNS:
        for name, got, want in zip(mine["stratum"], mine[column], reference[column], strict=True):
            if int(got) != int(want):
                problems.append(f"{name} {column}: got {int(got)}, M6 report has {int(want)}")
    return problems


def main() -> int:
    if OUT_PATH.exists():
        print(f"STOP: {OUT_PATH} already exists; delete it deliberately to regenerate.")
        return 1
    module = load_measurement_module()

    manifest = pd.read_parquet(
        MANIFEST_9CLASS_PATH, columns=["submission_id", "leaf_label", "rel_path", "split"]
    )
    m4 = pd.read_csv(
        module.M4_CSV, usecols=["submission_id", "stderr"], dtype=str, keep_default_na=False
    )
    wanted = manifest[manifest["leaf_label"].isin(["SYNTAX", "SEMANTIC"])]
    selected = wanted.merge(m4, on="submission_id", validate="1:1")
    if len(selected) != len(wanted):
        raise SystemExit("some SYNTAX/SEMANTIC files have no M4 stderr row")

    tokenizer = load_tokenizer()
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("a fast tokenizer is required for exact character offsets")

    print(f"Measuring {len(selected):,} SYNTAX/SEMANTIC files (all splits) ...")
    records = []
    for count, row in enumerate(selected.itertuples(), start=1):
        record = module._measure(tokenizer, row)
        record["submission_id"] = row.submission_id
        record["split"] = str(row.split)
        records.append(record)
        if count % 5000 == 0:
            print(f"  {count:,}/{len(selected):,}")
    results = pd.DataFrame(records)

    problems = gate(module, results)
    if problems:
        for problem in problems:
            print(f"GATE FAILED: {problem}")
        print("Nothing was written.")
        return 2
    print("gate passed: per-file results reproduce reports/m6_error_visibility.csv exactly")

    columns = ["submission_id", "split", "leaf", "missing_include", "outcome", "echo"]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results[columns].to_csv(OUT_PATH, index=False)
    print(OUT_PATH)
    print(results.groupby(["split", "outcome"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
