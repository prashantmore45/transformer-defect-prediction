"""
M4 -- Full-reserve summary: per-class counts, discard breakdown, and the
UNRECOGNIZED/AMBIGUOUS batch-review lists.

Purpose
-------
Turns the raw per-file results from run_tier2_full_reserve.py into the
actual M4 deliverables: final Tier-2 leaf counts (for M6 dataset assembly),
a discard/agreement report, and the real UNRECOGNIZED list for the batch
pattern-review step already agreed (option B, docs/TIER2_LABELING.md §7).

This script does NOT decide anything about UNRECOGNIZED/AMBIGUOUS rows --
it only surfaces them cleanly for review.

Input
-----
- reports/m4_full_reserve_results.csv

Output
------
- reports/m4_full_reserve_summary.md
- reports/m4_full_reserve_unrecognized.csv
- reports/m4_full_reserve_ambiguous.csv
"""

import sys
from pathlib import Path

import pandas as pd

INPUT_CSV = Path("reports/m4_full_reserve_results.csv")
SUMMARY_MD = Path("reports/m4_full_reserve_summary.md")
UNRECOGNIZED_CSV = Path("reports/m4_full_reserve_unrecognized.csv")
AMBIGUOUS_CSV = Path("reports/m4_full_reserve_ambiguous.csv")

LEAF_CLASSES = {"LINKER", "SYNTAX", "SEMANTIC"}
DISCARD_REASONS = {"AGREEMENT_FILTER", "MISSING_DEPENDENCY", "COMPILATION_TIMEOUT"}
REVIEW_FLAGS = {"AMBIGUOUS", "UNRECOGNIZED"}

PILOT_EXTRAPOLATED_LINKER = 375  # docs/TIER2_LABELING.md §5


def main() -> int:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        return 1

    df = pd.read_csv(INPUT_CSV)
    n_total = len(df)
    print(f"Loaded {n_total} rows from {INPUT_CSV}")

    counts = df["classification"].value_counts()
    print("\nFull classification breakdown:")
    print(counts.to_string())
    print(f"\nAs % of {n_total}:")
    print((100 * counts / n_total).round(2).to_string())

    unexpected = set(counts.index) - LEAF_CLASSES - DISCARD_REASONS - REVIEW_FLAGS
    if unexpected:
        print(f"\nWARNING: unexpected classification value(s) found: {unexpected}")

    leaf_counts = counts[counts.index.isin(LEAF_CLASSES)]
    n_labelled = int(leaf_counts.sum())
    print(f"\n--- Tier-2 leaf counts (feed M6 dataset assembly), n={n_labelled} ---")
    print(leaf_counts.to_string())

    discard_counts = counts[counts.index.isin(DISCARD_REASONS)]
    n_discarded = int(discard_counts.sum())
    print(
        f"\n--- Discard breakdown, n={n_discarded} ({100 * n_discarded / n_total:.1f}% of reserve) ---"
    )
    print(discard_counts.to_string())

    agreement_filter_n = int(counts.get("AGREEMENT_FILTER", 0))
    agreement_filter_pct = 100 * agreement_filter_n / n_total
    print(
        f"\nAgreement-filter rate: {agreement_filter_n}/{n_total} = {agreement_filter_pct:.1f}% "
        f"(pilot: 19.6%)"
    )

    review_counts = counts[counts.index.isin(REVIEW_FLAGS)]
    n_review = int(review_counts.sum())
    print(f"\n--- Needs review, n={n_review} ({100 * n_review / n_total:.2f}% of reserve) ---")
    print(review_counts.to_string())

    linker_n = int(counts.get("LINKER", 0))
    print(
        f"\nMeasured LINKER count: {linker_n} "
        f"(pilot extrapolation was ~{PILOT_EXTRAPOLATED_LINKER})"
    )

    unrecognized = df[df["classification"] == "UNRECOGNIZED"][
        ["submission_id", "problem_id", "stderr"]
    ].copy()
    unrecognized["first_error_line"] = unrecognized["stderr"].str.extract(
        r"(^.*: error:.*$)", flags=__import__("re").MULTILINE
    )
    UNRECOGNIZED_CSV.parent.mkdir(parents=True, exist_ok=True)
    unrecognized[["submission_id", "problem_id", "first_error_line"]].to_csv(
        UNRECOGNIZED_CSV, index=False
    )
    print(f"\nWrote {len(unrecognized)} UNRECOGNIZED rows to {UNRECOGNIZED_CSV}")

    ambiguous = df[df["classification"] == "AMBIGUOUS"][
        ["submission_id", "problem_id", "stderr"]
    ].copy()
    ambiguous["first_error_line"] = ambiguous["stderr"].str.extract(
        r"(^.*: error:.*$)", flags=__import__("re").MULTILINE
    )
    ambiguous[["submission_id", "problem_id", "first_error_line"]].to_csv(
        AMBIGUOUS_CSV, index=False
    )
    print(f"Wrote {len(ambiguous)} AMBIGUOUS rows to {AMBIGUOUS_CSV}")

    summary_lines = [
        "# M4 Full-Reserve Summary",
        "",
        f"Total files in COMPILE_ERROR reserve: {n_total}",
        "",
        "## Full classification breakdown",
        counts.to_string(),
        "",
        "## Tier-2 leaf counts (feed M6 dataset assembly)",
        leaf_counts.to_string(),
        f"Total labelled: {n_labelled} ({100 * n_labelled / n_total:.1f}% of reserve)",
        "",
        "## Discard breakdown",
        discard_counts.to_string(),
        f"Agreement-filter rate: {agreement_filter_n}/{n_total} = {agreement_filter_pct:.1f}% (pilot: 19.6%)",
        "",
        "## Needs review (not yet resolved)",
        review_counts.to_string(),
        f"See {UNRECOGNIZED_CSV} and {AMBIGUOUS_CSV} for the full lists.",
        "",
        "## LINKER sanity check",
        f"Measured: {linker_n} (pilot extrapolation: ~{PILOT_EXTRAPOLATED_LINKER})",
    ]
    SUMMARY_MD.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nWrote summary to {SUMMARY_MD}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
