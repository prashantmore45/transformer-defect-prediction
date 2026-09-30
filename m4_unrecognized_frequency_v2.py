"""
M4 -- Cluster the CLEAN, merged UNRECOGNIZED set (1,846 rows) by normalized
template. Same approach as the first frequency pass, but against
post-retry-merge data, and with a fallback extraction for ": fatal error:"
lines (no ": error:" substring) so nothing is miscounted as empty again.

Input
-----
- reports/m4_full_reserve_results_merged.csv

Output
------
- reports/m4_unrecognized_templates_v2.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd

INPUT_CSV = Path("reports/m4_full_reserve_results_merged.csv")
OUTPUT_CSV = Path("reports/m4_unrecognized_templates_v2.csv")
TOP_N_TO_PRINT = 40

ERROR_LINE_RE = re.compile(r"^.*: error:.*$", re.MULTILINE)
FATAL_LINE_RE = re.compile(r"^.*: fatal error:.*$", re.MULTILINE)

PREFIX_RE = re.compile(r"^.*?:\s*(?:error|fatal error):\s*")
QUOTED_RE = re.compile(r"['\u2018\u2019][^'\u2018\u2019]*['\u2018\u2019]")
DIGITS_RE = re.compile(r"\d+")


def extract_line(stderr: str) -> str:
    if not isinstance(stderr, str) or not stderr.strip():
        return ""
    m = ERROR_LINE_RE.search(stderr)
    if m:
        return m.group(0)
    m = FATAL_LINE_RE.search(stderr)
    if m:
        return m.group(0)
    return stderr.strip().splitlines()[0]


def normalize_template(line: str) -> str:
    if not line:
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
    unrec = df[df["classification"] == "UNRECOGNIZED"].copy()
    print(f"Loaded {len(unrec)} UNRECOGNIZED rows from merged data")

    unrec["stderr"] = unrec["stderr"].fillna("")
    n_empty = (unrec["stderr"].str.strip() == "").sum()
    print(f"Genuinely empty stderr: {n_empty} (should be ~0 after the phantom-failure fix)")

    unrec["line"] = unrec["stderr"].apply(extract_line)
    unrec["template"] = unrec["line"].apply(normalize_template)

    template_counts = unrec["template"].value_counts()
    print(f"\n{len(template_counts)} distinct templates found.")
    print(f"\nTop {TOP_N_TO_PRINT} templates by frequency:\n")

    top = template_counts.head(TOP_N_TO_PRINT)
    cumulative = 0
    for template, count in top.items():
        cumulative += count
        pct = 100 * count / len(unrec)
        example = unrec.loc[unrec["template"] == template, "line"].iloc[0]
        print(f"[{count:5d} | {pct:5.1f}%] {template}")
        print(f"          e.g. {example}")

    print(
        f"\nTop {TOP_N_TO_PRINT} templates cover {cumulative}/{len(unrec)} "
        f"({100 * cumulative / len(unrec):.1f}%) of the clean UNRECOGNIZED set."
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    template_counts.rename("count").to_csv(OUTPUT_CSV)
    print(f"\nWrote full template frequency table to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
