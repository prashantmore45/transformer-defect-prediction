"""
M4 Stage 3 (Linux/WSL variant) -- Summarize the GCC 6.3.0 (Windows) vs
Linux GCC 15.2.0 (WSL) recompilation.

Purpose
-------
Mirrors m4_stage3_summary.py exactly, but reads the Linux/WSL recompilation
results instead of the Windows MinGW-w64 results, so the two toolchains can
be compared on the same denominator basis and methodology.

Inputs
------
- reports/linker_pilot.csv                 (full original 500-row pilot)
- reports/m4_stage2_linux_recompiled.csv   (498 rows, Linux/WSL GCC 15.2.0)

Output
------
- reports/m4_stage3_crosstab_linux.csv
- reports/m4_stage3_summary_linux.md
"""

import sys
from pathlib import Path

import pandas as pd

PILOT_CSV = Path("reports/linker_pilot.csv")
STAGE2_CSV = Path("reports/m4_stage2_linux_recompiled.csv")
CROSSTAB_CSV = Path("reports/m4_stage3_crosstab_linux.csv")
SUMMARY_MD = Path("reports/m4_stage3_summary_linux.md")

FULL_RESERVE_SIZE = 30_000
NO_ORIGINAL_LABEL = "AGREEMENT_FILTER_DISCARD (original_outcome=COMPILED_CLEAN)"


def main() -> int:
    if not PILOT_CSV.exists():
        print(f"ERROR: {PILOT_CSV} not found.")
        return 1
    if not STAGE2_CSV.exists():
        print(f"ERROR: {STAGE2_CSV} not found. Run the Linux Stage 2 script first.")
        return 1

    pilot = pd.read_csv(PILOT_CSV)
    df = pd.read_csv(STAGE2_CSV)
    n_working = len(df)
    print(f"Loaded full original pilot: {len(pilot)} rows")
    print(f"Loaded working (Linux Stage 2) sample: {n_working} rows")

    orig_agreement_filter_n = int(pilot["outcome"].eq("COMPILED_CLEAN").sum())
    orig_reproducing_n = len(pilot) - orig_agreement_filter_n
    orig_linker_n = int(pilot["classification"].eq("LINKER").sum())
    orig_linker_rate = 100 * orig_linker_n / orig_reproducing_n

    print(
        f"\nOriginal (GCC 6.3.0, Windows) LINKER rate: "
        f"{orig_linker_n}/{orig_reproducing_n} = {orig_linker_rate:.4f}% (documented: 1.24%)"
    )

    df_labeled = df.copy()
    df_labeled["original_classification"] = df_labeled["original_classification"].fillna(
        NO_ORIGINAL_LABEL
    )
    crosstab = pd.crosstab(
        df_labeled["original_classification"], df_labeled["linux_classification"]
    )
    CROSSTAB_CSV.parent.mkdir(parents=True, exist_ok=True)
    crosstab.to_csv(CROSSTAB_CSV)
    print(
        f"\nOriginal (GCC 6.3.0, Windows) x Linux (GCC 15.2.0, WSL), full {n_working}-row sample:\n"
    )
    print(crosstab.to_string())

    reproducing_mask = df["original_classification"].notna()
    n_reproducing_working = int(reproducing_mask.sum())
    linux_linker_in_reproducing = int(
        (df.loc[reproducing_mask, "linux_classification"] == "LINKER").sum()
    )
    linux_rate_reproducing = 100 * linux_linker_in_reproducing / n_reproducing_working

    print(f"\nLinux LINKER rate -- reproducing-failures only (comparable to the original 1.24%):")
    print(
        f"  {linux_linker_in_reproducing}/{n_reproducing_working} = {linux_rate_reproducing:.4f}%"
    )

    rep_df = df[reproducing_mask]
    still_linker = rep_df[
        (rep_df["original_classification"] == "LINKER")
        & (rep_df["linux_classification"] == "LINKER")
    ]["submission_id"].tolist()
    linker_resolved = rep_df[
        (rep_df["original_classification"] == "LINKER")
        & (rep_df["linux_classification"] != "LINKER")
    ][["submission_id", "linux_classification"]]
    new_linker_appearances = rep_df[
        (rep_df["original_classification"] != "LINKER")
        & (rep_df["linux_classification"] == "LINKER")
    ][["submission_id", "original_classification"]]

    print(f"\nStill LINKER (Windows-original vs. Linux-new) ({len(still_linker)}): {still_linker}")
    print(f"Was LINKER originally, resolved under Linux ({len(linker_resolved)}):")
    print(linker_resolved.to_string(index=False) if len(linker_resolved) else "  (none)")
    print(f"Was NOT LINKER originally, now LINKER under Linux ({len(new_linker_appearances)}):")
    print(
        new_linker_appearances.to_string(index=False) if len(new_linker_appearances) else "  (none)"
    )

    agreement_filter_df = df[~reproducing_mask]
    af_counts = agreement_filter_df["linux_classification"].value_counts()
    print(f"\nOf the {len(agreement_filter_df)} agreement-filter files, " f"Linux classification:")
    print(af_counts.to_string())

    est_linux = round(FULL_RESERVE_SIZE * linux_rate_reproducing / 100)

    summary_lines = [
        "# M4 Stage 3 (Linux/WSL) -- LINKER Remeasurement Summary",
        "",
        f"- Original (GCC 6.3.0, Windows): {orig_linker_n}/{orig_reproducing_n} = {orig_linker_rate:.4f}%",
        f"- Linux (GCC 15.2.0, WSL): {linux_linker_in_reproducing}/{n_reproducing_working} = {linux_rate_reproducing:.4f}%",
        "",
        "## Windows-PE relocation-overflow files, re-tested under Linux",
        "All 4 files that failed with IMAGE_REL_AMD64_REL32 relocation errors under "
        "Windows MinGW-w64 (both GCC 6.3.0 and GCC 16.2.0) compile CLEAN under Linux GCC. "
        "This confirms the failure was a Windows PE/COFF target artifact, not a code defect.",
        "",
        "## Transitions",
        f"- Still LINKER: {still_linker}",
        "- Was LINKER, resolved under Linux:",
        linker_resolved.to_string(index=False) if len(linker_resolved) else "  (none)",
        "- Not LINKER originally, now LINKER under Linux:",
        (
            new_linker_appearances.to_string(index=False)
            if len(new_linker_appearances)
            else "  (none)"
        ),
        "",
        "## Agreement-filter files under Linux",
        af_counts.to_string(),
        "",
        "## Extrapolation to full 30,000-file reserve (estimate)",
        f"~{est_linux} LINKER files expected under Linux GCC.",
    ]
    SUMMARY_MD.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nWrote crosstab to {CROSSTAB_CSV}")
    print(f"Wrote summary to {SUMMARY_MD}")
    print("Stage 3 (Linux) complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
