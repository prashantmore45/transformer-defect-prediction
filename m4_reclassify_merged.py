"""
M4 -- Reclassify the merged, clean full-reserve results with the updated
tier2.py (2nd pattern-table pass, built from real full-reserve frequency
evidence). No recompilation -- reuses the already-captured returncode/stderr.

Input
-----
- reports/m4_full_reserve_results_merged.csv

Output
------
- reports/m4_full_reserve_results_final.csv (adds a 'classification' column
  recomputed with the updated tier2.classify(); original 'classification'
  column is kept as 'classification_v1' for comparison)
"""

import sys
from pathlib import Path

import pandas as pd

from sdp.data.labeling import tier2

INPUT_CSV = Path("reports/m4_full_reserve_results_merged.csv")
OUTPUT_CSV = Path("reports/m4_full_reserve_results_final.csv")


def main() -> int:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        return 1

    df = pd.read_csv(INPUT_CSV)
    df["stderr"] = df["stderr"].fillna("")
    print(f"Loaded {len(df)} rows")

    df["classification_v1"] = df["classification"]
    df["classification"] = df.apply(
        lambda row: str(
            tier2.classify(row["returncode"], row["stderr"], timed_out=bool(row["timed_out"]))
        ),
        axis=1,
    )

    print("\nOld (v1) classification breakdown:")
    print(df["classification_v1"].value_counts().to_string())

    print("\nNew (v2, final) classification breakdown:")
    counts = df["classification"].value_counts()
    print(counts.to_string())
    print(f"\nAs % of {len(df)}:")
    print((100 * counts / len(df)).round(2).to_string())

    changed = df[df["classification"] != df["classification_v1"]]
    print(f"\n{len(changed)} rows changed classification.")
    print("\nTransition breakdown (old -> new):")
    print(
        changed.groupby(["classification_v1", "classification"]).size().sort_values(ascending=False)
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
