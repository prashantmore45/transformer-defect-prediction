"""
M4 Stage 4 v3 -- Quote normalization, expanded patterns, and an allowlist
split for MISSING_HEADER (genuine external dependency vs. corrupted #include).

Changes from v2
----------------
1. QUOTE NORMALIZATION (the main fix): GCC quotes identifiers with Unicode
   "smart" quotes (U+2018/U+2019, `'...'`), not straight ASCII quotes (').
   Every pattern containing a literal ' silently failed to match real GCC
   output. Both first_error_line and full stderr are normalized (curly ->
   straight) before any pattern check runs, fixing this for every existing
   and future quote-containing pattern at once.

2. New patterns, added only where the v2 UNRECOGNIZED review showed >=2
   real occurrences (a single occurrence isn't a generalizable pattern --
   left in UNRECOGNIZED for manual review instead of over-fitting a rule
   to one file):
     SYNTAX:   "is not valid in an identifier", "a function-definition is
               not allowed here"
     SEMANTIC: "arguments to function", "lvalue required", "jump to label",
               "void value not ignored", "request for member",
               "iso c++ forbids", "declaration does not declare anything",
               "cannot be used as a function"

3. MISSING_HEADER split: a file matching "fatal error" + "no such file or
   directory" is now checked against an allowlist of known genuine external
   / precompiled-header dependencies (atcoder/, boost/, stdafx.h, pch.h,
   iostream.h). If it matches the allowlist, it's a real environment
   dependency -> MISSING_HEADER (candidate for exclusion, not a code
   defect). If it does NOT match the allowlist, it's far more likely a
   corrupted/typo'd #include in the submitted code itself (e.g.
   "bits/stdc..+h", "stack#include <algorithm") -> reclassified as SYNTAX,
   a genuine defect.

Input
-----
- reports/m4_stage2_linux_recompiled.csv

Output
------
- reports/m4_stage4_v3_linux.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd

INPUT_CSV = Path("reports/m4_stage2_linux_recompiled.csv")
OUTPUT_CSV = Path("reports/m4_stage4_v3_linux.csv")

# Known genuine external / precompiled-header dependencies -- NOT code
# defects, plausibly available in the original judge environment.
MISSING_HEADER_ALLOWLIST = [
    "atcoder/",
    "boost/",
    "stdafx.h",
    "pch.h",
    "iostream.h",
]

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
    "is not valid in an identifier",
    "a function-definition is not allowed here",
]
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
    "arguments to function",
    "lvalue required",
    "jump to label",
    "void value not ignored",
    "request for member",
    "iso c++ forbids",
    "declaration does not declare anything",
    "cannot be used as a function",
]

FIRST_ERROR_LINE_RE = re.compile(r"^.*: error:.*$", re.MULTILINE)

QUOTE_MAP = str.maketrans({"\u2018": "'", "\u2019": "'"})


def normalize_quotes(text: str) -> str:
    return text.translate(QUOTE_MAP) if isinstance(text, str) else text


def extract_first_error_line(stderr: str) -> str:
    if not isinstance(stderr, str) or not stderr.strip():
        return ""
    match = FIRST_ERROR_LINE_RE.search(stderr)
    return match.group(0) if match else stderr.strip().splitlines()[0]


def classify(first_error_line: str, full_stderr: str) -> str:
    line = normalize_quotes(first_error_line).lower()
    stderr = normalize_quotes(full_stderr).lower() if isinstance(full_stderr, str) else ""

    if "fatal error" in stderr and "no such file or directory" in stderr:
        if any(known in stderr for known in MISSING_HEADER_ALLOWLIST):
            return "MISSING_HEADER"
        return "SYNTAX"  # corrupted/typo'd #include -- a real code defect

    is_syntax = any(p in line for p in SYNTAX_PATTERNS) or (
        SYNTAX_BROAD_BOTH[0] in line and SYNTAX_BROAD_BOTH[1] in line
    )
    is_semantic = any(p in line for p in SEMANTIC_PATTERNS)

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
    print("\nTier-2 rule validation results (v3, Linux data, quote-normalized):")
    print(counts.to_string())
    print(f"\nAs % of {len(other)} validation files:")
    print((100 * counts / len(other)).round(1).to_string())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_cols = ["submission_id", "problem_id", "first_error_line", "tier2_classification"]
    other[out_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(other)} rows to {OUTPUT_CSV}")

    reclassified_syntax = other[
        (other["tier2_classification"] == "SYNTAX")
        & other["first_error_line"].str.contains("No such file or directory", case=False, na=False)
    ]
    if len(reclassified_syntax) > 0:
        print(f"\n--- Corrupted #include, reclassified SYNTAX ({len(reclassified_syntax)}) ---")
        for _, row in reclassified_syntax.iterrows():
            print(f"  {row['submission_id']}: {row['first_error_line']}")

    for bucket in ["MISSING_HEADER", "AMBIGUOUS", "UNRECOGNIZED"]:
        subset = other[other["tier2_classification"] == bucket]
        if len(subset) > 0:
            print(f"\n--- {bucket} ({len(subset)}) -- first error line per file ---")
            for _, row in subset.iterrows():
                print(f"  {row['submission_id']}: {row['first_error_line']}")

    print("\nStage 4 v3 complete. No compilation was performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
