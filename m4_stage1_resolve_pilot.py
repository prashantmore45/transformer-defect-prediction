"""
M4 Stage 1 — Resolve the 500-file LINKER pilot sample to real source paths.

Purpose
-------
The original GCC 6.3.0 pilot (docs/LABELING.md §5) reported a 1.24% LINKER
rate on a fixed 500-file sample (seed 42). To remeasure that rate with a
current compiler (GCC 16.2.0) on the *exact same* 500 files, we first need
to know where each file actually lives on disk and what compiler standard
to use. This script does ONLY that resolution + verification step.

It does NOT invoke g++. It does NOT modify linker_pilot.csv or the manifest.

Inputs
------
- reports/linker_pilot.csv          (500 rows: submission_id, problem_id,
                                      outcome, classification, stderr_tail)
- data/processed/splits/sample_manifest_hashed.parquet
                                     (submission_id, problem_id, user_id,
                                      status, date, code_size,
                                      original_language, coarse_label,
                                      split, random_split, archive_path,
                                      rel_path, sha256, bytes)

Output
------
- reports/m4_stage1_resolved.csv
  One row per pilot submission, with:
    submission_id, problem_id, original_outcome, original_classification,
    original_stderr_tail, rel_path, resolved_path, file_exists, std_flag

Exit behaviour
--------------
Prints a verification summary and exits non-zero if anything doesn't
resolve cleanly, so this never silently proceeds to Stage 2 with a
partial or broken join.
"""

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — the one thing you need to set for your machine.
# This must be the root that `rel_path` values in the manifest are relative
# to (i.e. EXTRACTED_ROOT / rel_path == the actual .cpp file on disk).
# ---------------------------------------------------------------------------
EXTRACTED_ROOT = Path(r"D:\Dev\Github\transformer-defect-prediction\data\processed\sources")

# Fixed compiler standard used by the original pilot (docs/LABELING.md §5).
# Kept fixed here too, deliberately: the only variable this remeasurement
# changes is the compiler binary, not the language standard.
STD_FLAG = "gnu++17"

PILOT_CSV = Path("reports/linker_pilot.csv")
MANIFEST_PARQUET = Path("data/processed/splits/sample_manifest_hashed.parquet")
OUTPUT_CSV = Path("reports/m4_stage1_resolved.csv")
EXCLUDED_LOG_CSV = Path("reports/m4_stage1_excluded.csv")

# Known, already-investigated exclusions: pilot submission_ids that do not
# join to the final frozen manifest. Each reason was confirmed by hand
# against M2 artifacts before being hardcoded here — see M4 notes.
# IMPORTANT: this is a fixed allow-list, not a general "skip unmatched rows"
# mechanism. Any unmatched row NOT in this dict is still a hard failure,
# because that would mean a new, uninvestigated join problem.
KNOWN_MANIFEST_EXCLUSIONS = {
    "s363457044": (
        "Absent from final M2 manifest; present in the pre-dedup sample "
        "but no row-level drop reason survives in dedup_report.csv or "
        "03_labeling_and_splitting.ipynb. Untraceable exclusion — "
        "recorded as a limitation, not assumed to be dedup."
    ),
    "s198272745": (
        "Confirmed cross-split content duplicate: appears in the "
        "duplicate-group listing in 03_labeling_and_splitting.ipynb. "
        "Dropped under M2's plurality-split retention rule for "
        "colliding normalized SHA-256 hashes."
    ),
}


def main() -> int:
    if not PILOT_CSV.exists():
        print(f"ERROR: pilot CSV not found at {PILOT_CSV}")
        return 1
    if not MANIFEST_PARQUET.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PARQUET}")
        return 1

    pilot = pd.read_csv(PILOT_CSV)
    manifest = pd.read_parquet(MANIFEST_PARQUET)

    print(f"Loaded pilot CSV: {len(pilot)} rows")
    print(f"Loaded manifest: {len(manifest)} rows")

    if len(pilot) != 500:
        print(f"WARNING: expected 500 pilot rows, found {len(pilot)}")

    if pilot["submission_id"].duplicated().any():
        dupes = pilot["submission_id"][pilot["submission_id"].duplicated()].tolist()
        print(f"ERROR: duplicate submission_id values in pilot CSV: {dupes}")
        return 1

    # Join on submission_id only — problem_id is redundant with it and
    # joining on both would just be a second, unnecessary check.
    manifest_cols = ["submission_id", "problem_id", "original_language", "rel_path"]
    merged = pilot.merge(
        manifest[manifest_cols],
        on="submission_id",
        how="left",
        suffixes=("", "_manifest"),
    )

    missing_join = merged[merged["rel_path"].isna()]

    if len(missing_join) > 0:
        missing_ids = missing_join["submission_id"].tolist()
        unexpected = [sid for sid in missing_ids if sid not in KNOWN_MANIFEST_EXCLUSIONS]

        if unexpected:
            print(
                f"ERROR: {len(unexpected)} pilot submission_ids had no manifest "
                f"match and are NOT in the known-exclusions allow-list: {unexpected}"
            )
            print(
                "This is a new, uninvestigated join failure — do not add it to "
                "KNOWN_MANIFEST_EXCLUSIONS without checking dedup_report.csv / "
                "03_labeling_and_splitting.ipynb first, the way the first two were."
            )
            return 1

        # All missing rows are known, already-investigated exclusions.
        excluded_log = missing_join[["submission_id", "problem_id"]].copy()
        excluded_log["reason"] = excluded_log["submission_id"].map(KNOWN_MANIFEST_EXCLUSIONS)
        EXCLUDED_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        excluded_log.to_csv(EXCLUDED_LOG_CSV, index=False)

        print(f"{len(missing_join)} known exclusion(s) found and logged to {EXCLUDED_LOG_CSV}:")
        for sid in missing_ids:
            print(f"  - {sid}: {KNOWN_MANIFEST_EXCLUSIONS[sid]}")

        merged = merged[merged["rel_path"].notna()].copy()

    print(
        f"Working sample after known exclusions: {len(merged)} rows " f"(nominal pilot size: 500)"
    )

    # Resolve full paths and check existence.
    merged["resolved_path"] = merged["rel_path"].apply(lambda rp: str(EXTRACTED_ROOT / rp))
    merged["file_exists"] = merged["resolved_path"].apply(lambda p: Path(p).exists())
    merged["std_flag"] = STD_FLAG

    n_missing_files = (~merged["file_exists"]).sum()
    if n_missing_files > 0:
        print(f"ERROR: {n_missing_files} resolved paths do not exist on disk.")
        print("First few missing paths:")
        print(merged.loc[~merged["file_exists"], "resolved_path"].head(10).to_string(index=False))
        print(
            "\nCheck EXTRACTED_ROOT at the top of this script — "
            "it must be the root that `rel_path` is relative to."
        )
        return 1

    print(f"All {len(merged)} resolved source files exist on disk.")

    out = merged.rename(
        columns={
            "outcome": "original_outcome",
            "classification": "original_classification",
            "stderr_tail": "original_stderr_tail",
        }
    )[
        [
            "submission_id",
            "problem_id",
            "original_outcome",
            "original_classification",
            "original_stderr_tail",
            "rel_path",
            "resolved_path",
            "file_exists",
            "std_flag",
        ]
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(out)} resolved rows to {OUTPUT_CSV}")
    print("Stage 1 complete. No compilation was performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
