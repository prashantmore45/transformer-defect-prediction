"""Tier-2 label derivation: g++ recompilation result -> compile-error leaf class.

This module is the single source of truth for the diagnostic-pattern vocabulary
used to split Tier 1's coarse COMPILE_ERROR verdict into SYNTAX, SEMANTIC, and
LINKER. It answers exactly one question, for exactly one recompilation result:

    Given this file's (returncode, stderr) from recompiling with g++, is it a
    real Tier-2 leaf class, a deliberate discard (and if so, why), or a
    diagnostic message we don't yet have a rule for?

Four outcomes, not two — and one important difference from Tier 1
-------------------------------------------------------------------
Tier 1's verdict vocabulary is closed: exactly 12 known strings, enumerated
from a full pass over 8,008,527 rows. An unrecognised verdict there is
necessarily a bug — the vocabulary was exhaustive by construction — so it
raises.

Compiler diagnostic text is not closed vocabulary. It is free text generated
by whichever code path in g++ happened to fire, and no pilot sample, however
careful, can enumerate every message a 30,000-file reserve will produce.
Raising on the first unmatched message would halt a multi-hour batch run over
a single novel phrasing. Instead, an unmatched message returns
`Tier2ReviewFlag.UNRECOGNIZED` — an explicit, named, counted outcome, batch-
reviewed after a run (see docs/TIER2_LABELING.md §7) with the pattern tables
below extended as real gaps are found. This is still "no verdict handled
implicitly": UNRECOGNIZED is never silently merged into SYNTAX or SEMANTIC,
and never silently dropped from the report — it is simply *reviewed on a
different schedule* than a closed-vocabulary lookup could be.

Quote normalisation
--------------------
g++ quotes identifiers with Unicode "smart" quotes (U+2018 `'` / U+2019 `'`),
not the ASCII apostrophe. Every pattern below is matched against
quote-normalised text (curly -> straight) specifically because an earlier
version of this pattern set silently failed to match ~24 real diagnostics for
exactly this reason during M4 pilot validation. Patterns are written using a
straight `'` throughout; normalisation happens once, inside `classify()`.

First-line-only matching
--------------------------
A single failing file can emit many cascading diagnostics as the parser
recovers and gets confused by everything downstream. Classification uses only
the first line containing ": error:" (falling back to ": fatal error:" if no
plain ": error:" line exists) — the same "first diagnostic" scoping
`docs/LABELING.md` §5 and the Master Implementation Plan §1 already specify —
except for the LINKER and MISSING_DEPENDENCY checks, which scan the full
stderr: a link-stage failure is reported by `collect2`/`ld`, tools invoked
after and separately from the compiler proper, so "first error: line" is not
where their signal appears.

Pattern table provenance
--------------------------
The SYNTAX_PATTERNS/SEMANTIC_PATTERNS tables below were built in two passes:
an initial pass validated on a 498-file pilot sample (docs/TIER2_LABELING.md
§7, brought UNRECOGNIZED from 147->63->16 on that sample), and a second pass
(this version) built from frequency analysis of the ACTUAL 30,000-file
reserve's real UNRECOGNIZED residual after the pilot-validated rules were
applied at full scale (1,846 files; top 40 recurring templates covered 67.2%
of that residual). The second pass is evidence from the real target
population, not an extrapolation from a sample — every pattern added in it
was chosen because it recurred >=12 times in the actual reserve, not because
it appeared once.

Scope
-----
This module resolves exactly the three Tier-2 leaves: SYNTAX, SEMANTIC,
LINKER (see `taxonomy.TIER_OF`). It does not compile anything — callers pass
in an already-obtained `(returncode, stderr)` pair, mirroring `tier1.classify`
taking an already-obtained verdict string.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final

from sdp.data.taxonomy import DefectClass


class Tier2DiscardReason(StrEnum):
    """Why a recompilation result is excluded from the Tier-2 labelled set.

    Unlike Tier 1's DiscardReason (a verdict-level judgement), these are
    recompilation-outcome-level: the file was in the COMPILE_ERROR reserve,
    but this particular recompilation doesn't yield a usable Tier-2 label.
    """

    AGREEMENT_FILTER = "AGREEMENT_FILTER"
    """Recompilation succeeded (returncode 0). The Tier-1 COMPILE_ERROR
    verdict didn't reproduce under this toolchain — never a real compile
    failure, so it carries no Tier-2 leaf label. Matches the agreement-filter
    principle already used for Tier 1 (docs/LABELING.md §5)."""

    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    """Fails only because an external header is unavailable in this build
    environment (e.g. the AtCoder library, Boost, an MSVC precompiled-header
    convention) — see MISSING_DEPENDENCY_ALLOWLIST. Reflects a build-
    environment gap, not a defect in the submitted code."""

    COMPILATION_TIMEOUT = "COMPILATION_TIMEOUT"
    """Compilation did not finish within the configured timeout. Cannot be
    classified; recorded and excluded rather than guessed at or retried
    silently."""


class Tier2ReviewFlag(StrEnum):
    """A recompilation result that needs human review before it can be labelled.

    Neither of these is a discard: both are almost certainly real compile
    failures, just not yet resolved to a specific leaf class.
    """

    AMBIGUOUS = "AMBIGUOUS"
    """The first error line matches both a SYNTAX and a SEMANTIC pattern.
    Rather than guess which stage actually fired, this is flagged for the
    pattern tables to be tightened."""

    UNRECOGNIZED = "UNRECOGNIZED"
    """The first error line matches no known pattern. See the module
    docstring: this is an open-vocabulary problem, reviewed in batches,
    not a bug to raise on."""


# --------------------------------------------------------------------------- #
# The diagnostic-pattern vocabulary — the single source of truth
# --------------------------------------------------------------------------- #

LINKER_MARKERS: Final[tuple[str, ...]] = (
    "undefined reference",
    "ld returned",
)
"""Matched against full stderr, not just the first line. "ld returned" (not
"collect2: error: ld returned") is deliberate: an earlier version required
the collect2-prefixed string and silently missed every failure on Windows,
where the linker driver reports itself as "collect2.exe", not "collect2"."""

MISSING_DEPENDENCY_ALLOWLIST: Final[tuple[str, ...]] = (
    "atcoder/",
    "boost/",
    "stdafx.h",
    "pch.h",
    "iostream.h",
)
"""Substrings identifying a genuinely missing external dependency (matched
against full stderr). A "fatal error: ... No such file or directory" NOT
matching this allowlist is treated as a corrupted or typo'd #include in the
submitted code itself (SYNTAX), not a missing dependency — e.g. "bits/stdc++"
(missing ".h") or two concatenated #include directives with no newline
between them. Extending this allowlist is a conscious editorial decision,
same as extending VERDICT_TABLE in tier1.py."""

SYNTAX_PATTERNS: Final[tuple[str, ...]] = (
    # --- pilot-validated (498-file sample) ---
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
    # --- full-reserve frequency evidence (>=12 recurrences each) ---
    "too many decimal points",
    "empty character constant",
    "without a previous",
    "two or more data types",
    "in nested-name-specifier",
    "'#include' expects",
    "arguments, but",
)
"""Structural/lexical-stage failures: the parser could not build a valid
parse tree. See module docstring for the theory this follows."""

SEMANTIC_PATTERNS: Final[tuple[str, ...]] = (
    # --- pilot-validated (498-file sample) ---
    "was not declared in this scope",
    "has not been declared",
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
    "incompatible types",
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
    # --- full-reserve frequency evidence (>=12 recurrences each) ---
    "invalid types",
    "conflicting declaration",
    "return-statement with",
    "cannot bind",
    "unable to find",
    "static assertion failed",
    "not within",
    "invalid type argument",
    "specified with",
    "could not convert",
    "non-scalar type",
    "narrowing conversion",
    "declared void",
    "not an integral constant-expression",
    "brace-enclosed initializer",
    "is not a namespace-name",
    "depend on a template parameter",
    "in a converted constant expression",
    "initializer-string for",
    "no contextual type information",
    "declared for postfix",
    "no type named",
    "cannot resolve overloaded function",
    "have different types",
    "in template argument",
)
"""Name-resolution / type-checking-stage failures: the code parses as valid
C++ grammar but violates a rule about meaning. See module docstring."""

# Broad catch-all pairs, checked as a conjunction (both substrings present),
# for messages whose exact wording varies too much (a digit, a quoted name)
# for a single literal pattern to catch. Each pair is (side, substring_a,
# substring_b).
SYNTAX_BROAD_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("expected", "before"),  # e.g. "expected ',' or ';' before 'cout'"
)
SEMANTIC_BROAD_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("template argument", "is invalid"),  # e.g. "template argument 1 is invalid"
)

_QUOTE_TRANSLATION: Final[dict[int, str]] = str.maketrans({"\u2018": "'", "\u2019": "'"})
_ERROR_LINE_RE: Final[re.Pattern[str]] = re.compile(r"^.*: error:.*$", re.MULTILINE)
_FATAL_ERROR_LINE_RE: Final[re.Pattern[str]] = re.compile(r"^.*: fatal error:.*$", re.MULTILINE)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _normalize_quotes(text: str) -> str:
    """Curly quotes ('/') -> straight ASCII apostrophe. See module docstring."""
    return text.translate(_QUOTE_TRANSLATION)


def _first_error_line(stderr: str) -> str:
    """The first ": error:" line, falling back to ": fatal error:", falling
    back to the first line of stderr if neither exists.

    The fallback to ": fatal error:" matters: GCC's fatal-error format is
    "path:line:col: fatal error: message" — there is no bare ": error:"
    substring in that line (the word "fatal" sits between the colon and
    "error"), so a regex looking only for ": error:" would silently miss it.
    See module docstring for why only the first diagnostic is used at all.
    """
    if not stderr.strip():
        return ""
    match = _ERROR_LINE_RE.search(stderr)
    if match:
        return match.group(0)
    match = _FATAL_ERROR_LINE_RE.search(stderr)
    if match:
        return match.group(0)
    return stderr.strip().splitlines()[0]


_LOCATION_RE: Final[re.Pattern[str]] = re.compile(r":(\d+):(\d+): (?:fatal )?error:")


def first_error_line_number(submission_id: str, stderr: str) -> int | None:
    """1-based line number of the first diagnostic, if it is in the submission's own file.

    Returns None when stderr is empty, when the first diagnostic comes from a
    different file (e.g. a system header, whose line numbers say nothing about
    the submission), or when the line carries no source location. Used only
    for measurement (M6.5); it plays no part in classification.
    """
    line = _first_error_line(stderr)
    if submission_id not in line:
        return None
    match = _LOCATION_RE.search(line)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------- #
# Accessor
# --------------------------------------------------------------------------- #


def classify(
    returncode: int, stderr: str, *, timed_out: bool = False
) -> DefectClass | Tier2DiscardReason | Tier2ReviewFlag:
    """Resolve one recompilation result to a Tier-2 leaf, a discard reason,
    or a review flag.

    Args:
        returncode: exit code from the g++ invocation.
        stderr: captured stderr text (may be empty).
        timed_out: True if compilation was killed by a timeout. When True,
            `returncode` and `stderr` are ignored — a killed process's exit
            code is not meaningful.

    Returns:
        `DefectClass.LINKER | .SYNTAX | .SEMANTIC` for a classified leaf,
        a `Tier2DiscardReason` if this result carries no Tier-2 label,
        or a `Tier2ReviewFlag` if human review is needed before it can be
        labelled. Never raises on unrecognised diagnostic text — see the
        module docstring for why this differs from `tier1.classify`.
    """
    if timed_out:
        return Tier2DiscardReason.COMPILATION_TIMEOUT

    if returncode == 0:
        return Tier2DiscardReason.AGREEMENT_FILTER

    normalized_stderr = _normalize_quotes(stderr).lower()

    if any(marker in normalized_stderr for marker in LINKER_MARKERS):
        return DefectClass.LINKER

    if "fatal error" in normalized_stderr and "no such file or directory" in normalized_stderr:
        if any(known in normalized_stderr for known in MISSING_DEPENDENCY_ALLOWLIST):
            return Tier2DiscardReason.MISSING_DEPENDENCY
        return DefectClass.SYNTAX  # corrupted/typo'd #include: a real defect

    first_line = _normalize_quotes(_first_error_line(stderr)).lower()

    is_syntax = any(p in first_line for p in SYNTAX_PATTERNS) or any(
        a in first_line and b in first_line for a, b in SYNTAX_BROAD_PAIRS
    )
    is_semantic = any(p in first_line for p in SEMANTIC_PATTERNS) or any(
        a in first_line and b in first_line for a, b in SEMANTIC_BROAD_PAIRS
    )

    if is_syntax and is_semantic:
        return Tier2ReviewFlag.AMBIGUOUS
    if is_syntax:
        return DefectClass.SYNTAX
    if is_semantic:
        return DefectClass.SEMANTIC
    return Tier2ReviewFlag.UNRECOGNIZED


def is_labelled(outcome: DefectClass | Tier2DiscardReason | Tier2ReviewFlag) -> bool:
    """True if this outcome is a usable Tier-2 leaf label.

    False for both discards and review flags — a caller building the Tier-2
    manifest should only keep rows where this is True; everything else needs
    either no label (discard) or a decision before it gets one (review flag).
    """
    return isinstance(outcome, DefectClass)
