"""
M4 -- Cluster the 4,829 UNRECOGNIZED first-error-lines by normalized
template, to find high-frequency patterns worth adding to tier2.py.

Purpose
-------
4,829 individual lines is too many to review one at a time. This groups
them by a normalized template (file paths, line:col numbers, and quoted
identifiers stripped out) and reports the most frequent templates first,
so pattern-writing effort goes where it has the most payoff.

This script does NOT modify tier2.py or reclassify anything -- it only
surfaces the frequency distribution for review.

Input
-----
- reports/m4_full_reserve_unrecognized.csv

Output
------
- reports/m4_unrecognized_templates.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd

INPUT_CSV = Path("reports/m4_full_reserve_unrecognized.csv")
OUTPUT_CSV = Path("reports/m4_unrecognized_templates.csv")
TOP_N_TO_PRINT = 40

# Strip, in order: the "path:line:col: error:" prefix, quoted identifiers
# (both straight and curly quotes), and any remaining digit runs -- leaves
# just the invariant shape of the message.
PREFIX_RE = re.compile(r"^.*?: error:\s*")
QUOTED_RE = re.compile(r"['\u2018\u2019][^'\u2018\u2019]*['\u2018\u2019]")
DIGITS_RE = re.compile(r"\d+")


def normalize_template(line: str) -> str:
    if not isinstance(line, str) or not line.strip():
        return "(empty)"
    text = PREFIX_RE.sub("", line)
    text = QUOTED_RE.sub("<Q>", text)
    text = DIGITS_RE.sub("<N>", text)
    return text.strip()


def main() -> int:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        return 1

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} UNRECOGNIZED rows")

    df["template"] = df["first_error_line"].apply(normalize_template)
    template_counts = df["template"].value_counts()

    print(f"\n{len(template_counts)} distinct templates found.")
    print(f"\nTop {TOP_N_TO_PRINT} templates by frequency:\n")
    top = template_counts.head(TOP_N_TO_PRINT)
    cumulative = 0
    for template, count in top.items():
        cumulative += count
        pct = 100 * count / len(df)
        example = df.loc[df["template"] == template, "first_error_line"].iloc[0]
        print(f"[{count:5d} | {pct:5.1f}%] {template}")
        print(f"          e.g. {example}")

    print(
        f"\nTop {TOP_N_TO_PRINT} templates cover {cumulative}/{len(df)} "
        f"({100 * cumulative / len(df):.1f}%) of all UNRECOGNIZED files."
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    template_counts.rename("count").to_csv(OUTPUT_CSV)
    print(f"\nWrote full template frequency table to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
