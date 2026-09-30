"""
M4 Stage 4 -- Validate SYNTAX/SEMANTIC diagnostic rules against pilot data.

Purpose
-------
Stage 2 already recompiled all 498 pilot files under GCC 16.2.0 and
captured full stderr, including 388 files that failed with something other
than LINKER. This stage reuses that existing data (no new compilation) to
test draft SYNTAX/SEMANTIC pattern lists before they're trusted on the full
30,000-file COMPILE_ERROR reserve.

Classification is applied to the FIRST "error:" line only, not the whole
stderr blob -- a single file can emit many cascading diagnostics after the
real failure, and matching against all of them risks a downstream cascade
line accidentally triggering the wrong bucket.

Four buckets, deliberately -- nothing is defaulted into SYNTAX or SEMANTIC:
  - SYNTAX       : matches a syntax-stage pattern only
  - SEMANTIC     : matches a semantic-stage pattern only
  - AMBIGUOUS    : matches both pattern lists (reviewed by hand, not guessed)
  - UNRECOGNIZED : matches neither (reviewed by hand, patterns extended)

Input
-----
- reports/m4_stage2_recompiled.csv

Output
------
- reports/m4_stage4_syntax_semantic_pilot.csv
- Printed summary + full UNRECOGNIZED / AMBIGUOUS examples for review
"""

import re
import sys
from pathlib import Path

import pandas as pd

INPUT_CSV = Path("reports/m4_stage2_recompiled.csv")
OUTPUT_CSV = Path("reports/m4_stage4_syntax_semantic_pilot.csv")

# --- Draft pattern lists (validate here before trusting at full scale) ---
SYNTAX_PATTERNS = [
    "expected ';'",
    "expected '}'",
    "expected declaration",
    "expected primary-expression before",
    "expected unqualified-id",
    "stray '",  # "stray '<char>' in program"
    "unterminated comment",
    "missing terminating",
]

SEMANTIC_PATTERNS = [
    "was not declared in this scope",
    "no matching function for call to",
    "invalid conversion from",
    "cannot convert",
    "ambiguous overload",
    "call of overloaded",
    "redefinition of",
    "has no member named",
]

FIRST_ERROR_LINE_RE = re.compile(r"^.*: error:.*$", re.MULTILINE)


def extract_first_error_line(stderr: str) -> str:
    if not isinstance(stderr, str) or not stderr.strip():
        return ""
    match = FIRST_ERROR_LINE_RE.search(stderr)
    return match.group(0) if match else stderr.strip().splitlines()[0]


def classify(first_error_line: str) -> str:
    line_lower = first_error_line.lower()
    is_syntax = any(p.lower() in line_lower for p in SYNTAX_PATTERNS)
    is_semantic = any(p.lower() in line_lower for p in SEMANTIC_PATTERNS)

    if is_syntax and is_semantic:
        return "AMBIGUOUS"
    if is_syntax:
        return "SYNTAX"
    if is_semantic:
        return "SEMANTIC"
    return "UNRECOGNIZED"


def main() -> int:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run Stage 2 first.")
        return 1

    df = pd.read_csv(INPUT_CSV)
    other = df[df["new_classification"] == "OTHER_COMPILE_ERROR"].copy()
    print(f"Loaded {len(df)} Stage 2 rows; {len(other)} are OTHER_COMPILE_ERROR (validation set)")

    other["first_error_line"] = other["new_stderr"].apply(extract_first_error_line)
    other["tier2_classification"] = other["first_error_line"].apply(classify)

    counts = other["tier2_classification"].value_counts()
    print("\nTier-2 rule validation results:")
    print(counts.to_string())
    print(f"\nAs % of {len(other)} validation files:")
    print((100 * counts / len(other)).round(1).to_string())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_cols = ["submission_id", "problem_id", "first_error_line", "tier2_classification"]
    other[out_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(other)} rows to {OUTPUT_CSV}")

    unrecognized = other[other["tier2_classification"] == "UNRECOGNIZED"]
    ambiguous = other[other["tier2_classification"] == "AMBIGUOUS"]

    if len(unrecognized) > 0:
        print(f"\n--- UNRECOGNIZED ({len(unrecognized)}) -- first error line per file ---")
        for _, row in unrecognized.iterrows():
            print(f"  {row['submission_id']}: {row['first_error_line']}")

    if len(ambiguous) > 0:
        print(f"\n--- AMBIGUOUS ({len(ambiguous)}) -- matched both pattern lists ---")
        for _, row in ambiguous.iterrows():
            print(f"  {row['submission_id']}: {row['first_error_line']}")

    print("\nStage 4 complete. No compilation was performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
