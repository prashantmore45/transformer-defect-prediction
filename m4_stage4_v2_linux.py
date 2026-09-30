"""
M4 Stage 4 v2 -- Validate expanded SYNTAX/SEMANTIC rules + MISSING_HEADER
bucket against the LINUX (authoritative) recompilation output.

Purpose
-------
Stage 4 v1 ran against the Windows recompilation and left 147/388 files
UNRECOGNIZED. Reviewing those surfaced clear, systematic pattern gaps
(e.g. "does not name a type", "no match for", broad "expected ... before")
and a distinct category -- missing build dependencies (fatal error: ... No
such file or directory) -- that isn't SYNTAX or SEMANTIC at all.

This version:
  1. Runs against reports/m4_stage2_linux_recompiled.csv (the authoritative
     toolchain), not the Windows output.
  2. Adds a MISSING_HEADER bucket, checked first, for missing-dependency
     failures -- kept separate, not forced into SYNTAX/SEMANTIC.
  3. Applies the expanded SYNTAX/SEMANTIC pattern lists.

Five buckets, nothing defaulted:
  - MISSING_HEADER : fatal error, file not found (missing dependency)
  - SYNTAX         : matches a syntax-stage pattern only
  - SEMANTIC       : matches a semantic-stage pattern only
  - AMBIGUOUS      : matches both SYNTAX and SEMANTIC patterns
  - UNRECOGNIZED   : matches nothing (reviewed by hand, patterns extended)

Input
-----
- reports/m4_stage2_linux_recompiled.csv

Output
------
- reports/m4_stage4_v2_linux.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd

INPUT_CSV = Path("reports/m4_stage2_linux_recompiled.csv")
OUTPUT_CSV = Path("reports/m4_stage4_v2_linux.csv")

MISSING_HEADER_MARKERS = ("fatal error", "no such file or directory")

SYNTAX_PATTERNS = [
    "expected ';'",
    "expected '}'",
    "expected declaration",
    "expected primary-expression before",
    "expected unqualified-id",
    "stray '",
    "unterminated comment",
    "missing terminating",
    "invalid preprocessing directive",
]
# Broad rule, checked separately: "expected ... before" covers most
# remaining parser-recovery messages not caught by the specific patterns
# above (e.g. "expected ',' or ';' before 'cout'", "expected ':' before
# '!' token", "expected initializer before 'scanf'").
SYNTAX_BROAD_BOTH = ("expected", "before")

SEMANTIC_PATTERNS = [
    "was not declared in this scope",
    "no matching function for call to",
    "no match for",
    "invalid conversion from",
    "cannot convert",
    "ambiguous overload",
    "call of overloaded",
    "redefinition of",
    "redeclared as different kind of entity",
    "redeclaration of",
    "has no member named",
    "does not name a type",
    "is ambiguous",
    "is not a member of",
    "invalid operands of types",
    "invalid use of member",
    "invalid use of incomplete type",
    "must return 'int'",
    "cannot declare '::main'",
    "is private within this context",
]

FIRST_ERROR_LINE_RE = re.compile(r"^.*: error:.*$", re.MULTILINE)


def extract_first_error_line(stderr: str) -> str:
    if not isinstance(stderr, str) or not stderr.strip():
        return ""
    match = FIRST_ERROR_LINE_RE.search(stderr)
    return match.group(0) if match else stderr.strip().splitlines()[0]


def classify(first_error_line: str, full_stderr: str) -> str:
    line_lower = first_error_line.lower()
    stderr_lower = full_stderr.lower() if isinstance(full_stderr, str) else ""

    # Checked against the FULL stderr, not just the first line -- a fatal
    # "no such file" error is sometimes not literally the first "error:"
    # line matched by our regex if preceded by other output, so check the
    # whole blob for this specific, unambiguous marker pair.
    if all(m in stderr_lower for m in MISSING_HEADER_MARKERS):
        return "MISSING_HEADER"

    is_syntax = any(p.lower() in line_lower for p in SYNTAX_PATTERNS) or (
        SYNTAX_BROAD_BOTH[0] in line_lower and SYNTAX_BROAD_BOTH[1] in line_lower
    )
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
        print(f"ERROR: {INPUT_CSV} not found. Run the Linux Stage 2 script first.")
        return 1

    df = pd.read_csv(INPUT_CSV)
    other = df[df["linux_classification"] == "OTHER_COMPILE_ERROR"].copy()
    print(
        f"Loaded {len(df)} Linux Stage 2 rows; {len(other)} are OTHER_COMPILE_ERROR (validation set)"
    )

    other["first_error_line"] = other["linux_stderr"].apply(extract_first_error_line)
    other["tier2_classification"] = other.apply(
        lambda row: classify(row["first_error_line"], row["linux_stderr"]), axis=1
    )

    counts = other["tier2_classification"].value_counts()
    print("\nTier-2 rule validation results (v2, Linux data):")
    print(counts.to_string())
    print(f"\nAs % of {len(other)} validation files:")
    print((100 * counts / len(other)).round(1).to_string())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_cols = ["submission_id", "problem_id", "first_error_line", "tier2_classification"]
    other[out_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(other)} rows to {OUTPUT_CSV}")

    for bucket in ["MISSING_HEADER", "AMBIGUOUS", "UNRECOGNIZED"]:
        subset = other[other["tier2_classification"] == bucket]
        if len(subset) > 0:
            print(f"\n--- {bucket} ({len(subset)}) -- first error line per file ---")
            for _, row in subset.iterrows():
                print(f"  {row['submission_id']}: {row['first_error_line']}")

    print("\nStage 4 v2 complete. No compilation was performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
