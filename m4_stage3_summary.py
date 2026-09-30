"""
M4 Stage 3 -- Summarize the GCC 6.3.0 vs GCC 16.2.0 recompilation.

Purpose
-------
Turns Stage 2's per-file results into the actual comparison this
remeasurement exists to produce. Two denominators matter here and must
not be conflated:

  - The FULL working sample (498 files) includes 98 files whose original
    outcome was COMPILED_CLEAN -- i.e. the Tier-1 COMPILE_ERROR label
    didn't reproduce under recompilation at all (an "agreement filter"
    discard, per docs/LABELING.md §5's own 19.6% secondary finding).
    These were never real compile failures, so they don't belong in a
    LINKER-rate denominator.
  - The REPRODUCING-FAILURE subset excludes those 98. Arithmetic check:
    5 LINKER / 402 reproducing-failures (500 - 98) = 1.2438%, which
    matches the documented 1.24% almost exactly -- strong evidence this
    is the denominator the original figure actually used.

This script reports both denominators explicitly, so the two are never
silently conflated again. It does NOT make the section-1 contingency
decision (train-anyway vs. merge into SEMANTIC) -- it only produces the
measured numbers that decision depends on.

Inputs
------
- reports/linker_pilot.csv          (full original 500-row pilot -- source
                                      of truth for the "before" side)
- reports/m4_stage2_recompiled.csv  (498 rows -- the "after" side)

Output
------
- reports/m4_stage3_crosstab.csv
- reports/m4_stage3_summary.md
"""

import sys
from pathlib import Path

import pandas as pd

PILOT_CSV = Path("reports/linker_pilot.csv")
STAGE2_CSV = Path("reports/m4_stage2_recompiled.csv")
CROSSTAB_CSV = Path("reports/m4_stage3_crosstab.csv")
SUMMARY_MD = Path("reports/m4_stage3_summary.md")

FULL_RESERVE_SIZE = 30_000

NO_ORIGINAL_LABEL = "AGREEMENT_FILTER_DISCARD (original_outcome=COMPILED_CLEAN)"


def main() -> int:
    if not PILOT_CSV.exists():
        print(f"ERROR: {PILOT_CSV} not found.")
        return 1
    if not STAGE2_CSV.exists():
        print(f"ERROR: Stage 2 output not found at {STAGE2_CSV}. Run Stage 2 first.")
        return 1

    pilot = pd.read_csv(PILOT_CSV)
    df = pd.read_csv(STAGE2_CSV)
    n_working = len(df)
    print(f"Loaded full original pilot: {len(pilot)} rows")
    print(f"Loaded working (Stage 2) sample: {n_working} rows")

    # --- Original (GCC 6.3.0), from the full 500-row source of truth ---
    orig_counts = pilot["classification"].value_counts(dropna=False)
    orig_agreement_filter_n = int(pilot["outcome"].eq("COMPILED_CLEAN").sum())
    orig_reproducing_n = len(pilot) - orig_agreement_filter_n
    orig_linker_n = int(pilot["classification"].eq("LINKER").sum())
    orig_linker_rate = 100 * orig_linker_n / orig_reproducing_n

    print(f"\nFull original pilot (n={len(pilot)}):")
    print(orig_counts.to_string())
    print(
        f"Original LINKER rate (reproducing-failures denominator): "
        f"{orig_linker_n}/{orig_reproducing_n} = {orig_linker_rate:.4f}% "
        f"(documented figure: 1.24%)"
    )

    # --- New (GCC 16.2.0), from the 498-row working sample ---
    df_labeled = df.copy()
    df_labeled["original_classification"] = df_labeled["original_classification"].fillna(
        NO_ORIGINAL_LABEL
    )

    crosstab = pd.crosstab(df_labeled["original_classification"], df_labeled["new_classification"])
    CROSSTAB_CSV.parent.mkdir(parents=True, exist_ok=True)
    crosstab.to_csv(CROSSTAB_CSV)
    print(
        f"\nOriginal (GCC 6.3.0) x New (GCC 16.2.0) classification, full {n_working}-row sample:\n"
    )
    print(crosstab.to_string())

    reproducing_mask = df["original_classification"].notna()
    n_reproducing_working = int(reproducing_mask.sum())
    new_linker_in_reproducing = int(
        (df.loc[reproducing_mask, "new_classification"] == "LINKER").sum()
    )
    new_rate_reproducing = 100 * new_linker_in_reproducing / n_reproducing_working

    new_linker_full = int((df["new_classification"] == "LINKER").sum())
    new_rate_full = 100 * new_linker_full / n_working

    print(
        f"\nNew LINKER rate -- full working sample (WRONG comparison, includes agreement-filter files):"
    )
    print(f"  {new_linker_full}/{n_working} = {new_rate_full:.4f}%")
    print(f"\nNew LINKER rate -- reproducing-failures only (comparable to the original 1.24%):")
    print(f"  {new_linker_in_reproducing}/{n_reproducing_working} = {new_rate_reproducing:.4f}%")

    # --- Transitions, restricted to the reproducing-failure subset ---
    rep_df = df[reproducing_mask]
    still_linker = rep_df[
        (rep_df["original_classification"] == "LINKER") & (rep_df["new_classification"] == "LINKER")
    ]["submission_id"].tolist()
    linker_resolved = rep_df[
        (rep_df["original_classification"] == "LINKER") & (rep_df["new_classification"] != "LINKER")
    ][["submission_id", "new_classification"]]
    new_linker_appearances = rep_df[
        (rep_df["original_classification"] != "LINKER") & (rep_df["new_classification"] == "LINKER")
    ][["submission_id", "original_classification"]]

    print(f"\nStill LINKER under both compilers ({len(still_linker)}): {still_linker}")
    print(f"Was LINKER under GCC 6.3.0, now something else ({len(linker_resolved)}):")
    print(linker_resolved.to_string(index=False) if len(linker_resolved) else "  (none)")
    print(
        f"Was NOT LINKER under GCC 6.3.0, now LINKER under GCC 16.2.0 ({len(new_linker_appearances)}):"
    )
    print(
        new_linker_appearances.to_string(index=False) if len(new_linker_appearances) else "  (none)"
    )

    # --- What happened to the 98 agreement-filter files under GCC 16.2.0 ---
    agreement_filter_df = df[~reproducing_mask]
    af_new_counts = agreement_filter_df["new_classification"].value_counts()
    print(
        f"\nOf the {len(agreement_filter_df)} agreement-filter files "
        f"(originally compiled clean under GCC 6.3.0), new classification under GCC 16.2.0:"
    )
    print(af_new_counts.to_string())

    # --- Extrapolation (labeled estimate only) ---
    est_full_reserve_linker = round(FULL_RESERVE_SIZE * new_rate_reproducing / 100)
    est_orig_full_reserve_linker = round(FULL_RESERVE_SIZE * orig_linker_rate / 100)

    summary_lines = [
        "# M4 Stage 3 -- LINKER Remeasurement Summary",
        "",
        "## Two exclusions from the nominal 500 (Stage 1)",
        "- s198272745: confirmed cross-split dedup drop; ORIGINALLY LABELED LINKER",
        "- s363457044: untraceable exclusion; originally labeled OTHER",
        "",
        "## Denominator note",
        f"Original 1.24% matches {orig_linker_n}/{orig_reproducing_n} = {orig_linker_rate:.4f}% "
        f"(reproducing-failure files only, excluding {orig_agreement_filter_n} agreement-filter "
        f"discards) far more closely than {orig_linker_n}/{len(pilot)}. This denominator is inferred "
        f"from the arithmetic match, not yet confirmed against the docs/LABELING.md §5 text directly.",
        "",
        "## Rates",
        f"- Original (GCC 6.3.0): {orig_linker_n}/{orig_reproducing_n} = {orig_linker_rate:.4f}%",
        f"- New (GCC 16.2.0), same denominator basis: "
        f"{new_linker_in_reproducing}/{n_reproducing_working} = {new_rate_reproducing:.4f}%",
        f"- New (GCC 16.2.0), full working sample -- NOT comparable, listed only to show the "
        f"difference denominator choice makes: {new_linker_full}/{n_working} = {new_rate_full:.4f}%",
        "",
        "## Transitions (reproducing-failure subset only)",
        f"- Still LINKER under both compilers: {still_linker}",
        "- LINKER under GCC 6.3.0, resolved to something else under GCC 16.2.0:",
        linker_resolved.to_string(index=False) if len(linker_resolved) else "  (none)",
        "- Not LINKER under GCC 6.3.0, now LINKER under GCC 16.2.0:",
        (
            new_linker_appearances.to_string(index=False)
            if len(new_linker_appearances)
            else "  (none)"
        ),
        "",
        "## Agreement-filter files (originally compiled clean) under GCC 16.2.0",
        af_new_counts.to_string(),
        "",
        "## Extrapolation to the full 30,000-file COMPILE_ERROR reserve",
        "**ESTIMATE, not a measurement** -- assumes this pilot's rate generalizes uniformly, "
        "which has not been checked, and assumes the reproducing-failure proportion (~80%) "
        "also holds at full-reserve scale.",
        f"- Estimated LINKER count (new rate basis): ~{est_full_reserve_linker}",
        f"- Estimated LINKER count (original rate basis, for reference): ~{est_orig_full_reserve_linker}",
        "",
        "This is the number the section-1 contingency decision (train-anyway-with-caveat "
        "vs. merge LINKER into SEMANTIC) depends on. This script does not make that call.",
    ]
    SUMMARY_MD.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nWrote crosstab to {CROSSTAB_CSV}")
    print(f"Wrote summary to {SUMMARY_MD}")
    print("Stage 3 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
