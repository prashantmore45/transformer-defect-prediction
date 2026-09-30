"""
M4 -- Merge the phantom-failure retry results into the main results file.

Purpose
-------
Replaces the 3,179 rows in m4_full_reserve_results.csv that were phantom
wsl.exe failures (empty stderr, no real compile ever happened) with their
correct results from m4_retry_results.csv. Produces one clean, authoritative
dataset. The original main results file is left untouched; this writes a
new, merged file.

Input
-----
- reports/m4_full_reserve_results.csv (29,876 rows, 3,179 phantom)
- reports/m4_retry_results.csv (3,179 corrected rows)

Output
------
- reports/m4_full_reserve_results_merged.csv
"""

import sys
from pathlib import Path

import pandas as pd

MAIN_CSV = Path("reports/m4_full_reserve_results.csv")
RETRY_CSV = Path("reports/m4_retry_results.csv")
MERGED_CSV = Path("reports/m4_full_reserve_results_merged.csv")


def main() -> int:
    main_df = pd.read_csv(MAIN_CSV)
    retry_df = pd.read_csv(RETRY_CSV)
    print(f"Main: {len(main_df)} rows. Retry: {len(retry_df)} rows.")

    retry_ids = set(retry_df["submission_id"].astype(str))
    dupes_in_retry = retry_df["submission_id"].duplicated().sum()
    if dupes_in_retry > 0:
        print(f"WARNING: {dupes_in_retry} duplicate submission_id(s) in retry file.")

    kept = main_df[~main_df["submission_id"].astype(str).isin(retry_ids)]
    print(f"Keeping {len(kept)} untouched rows from main (not retried).")

    merged = pd.concat([kept, retry_df], ignore_index=True)
    print(f"Merged total: {len(merged)} rows (expected {len(main_df)}).")

    if len(merged) != len(main_df):
        print("ERROR: row count mismatch after merge -- investigate before proceeding.")
        return 1

    if merged["submission_id"].duplicated().any():
        print("ERROR: duplicate submission_ids in merged output -- investigate before proceeding.")
        return 1

    merged.to_csv(MERGED_CSV, index=False)
    print(f"\nWrote merged file to {MERGED_CSV}")

    print("\nMerged classification breakdown:")
    print(merged["classification"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
