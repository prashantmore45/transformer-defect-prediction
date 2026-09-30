"""Tests for Tier-2 diagnostic classification.

Four kinds of test live here, mirroring test_tier1.py's structure:

Contract tests
    Assert the shape of the pattern tables themselves (non-empty, no
    duplicate entries, LINKER_MARKERS unaffected by the collect2/collect2.exe
    regression). These fail loudly if a table is accidentally emptied or
    duplicated during editing.

Behavioral tests
    classify() against real diagnostic text captured during the M4 pilot
    (docs/TIER2_LABELING.md), not synthetic strings — including the exact
    Unicode-quote regression that silently broke matching until fixed, so it
    cannot silently return.

Invariant tests
    Structural properties that must hold regardless of table contents: every
    outcome is a recognised type, is_labelled agrees with DefectClass
    membership, timed_out always wins regardless of returncode/stderr.

Cross-module tests
    Verify tier2 and taxonomy agree — classify() only ever returns the exact
    three DefectClass leaves taxonomy.py assigns to tier 2.
"""

from __future__ import annotations

import pytest

from sdp.data import taxonomy as tax
from sdp.data.labeling import tier2
from sdp.data.labeling.tier2 import Tier2DiscardReason, Tier2ReviewFlag
from sdp.data.taxonomy import DefectClass

# --------------------------------------------------------------------------- #
# Contract tests — fail loudly if a pattern table is accidentally damaged
# --------------------------------------------------------------------------- #


def test_pattern_tables_are_non_empty() -> None:
    assert len(tier2.LINKER_MARKERS) > 0
    assert len(tier2.MISSING_DEPENDENCY_ALLOWLIST) > 0
    assert len(tier2.SYNTAX_PATTERNS) > 0
    assert len(tier2.SEMANTIC_PATTERNS) > 0


def test_pattern_tables_have_no_duplicate_entries() -> None:
    for table in (
        tier2.LINKER_MARKERS,
        tier2.MISSING_DEPENDENCY_ALLOWLIST,
        tier2.SYNTAX_PATTERNS,
        tier2.SEMANTIC_PATTERNS,
    ):
        assert len(table) == len(set(table)), table


def test_linker_markers_do_not_require_a_collect2_prefix() -> None:
    """Regression guard for the Windows collect2/collect2.exe undercount bug.

    The fix was matching on "ld returned" alone rather than requiring
    "collect2: error: ld returned" as one literal string. Pin that here so a
    future edit can't silently reintroduce the prefix requirement.
    """
    assert "ld returned" in tier2.LINKER_MARKERS
    assert not any(marker.startswith("collect2") for marker in tier2.LINKER_MARKERS)


def test_no_pattern_appears_in_both_syntax_and_semantic_tables() -> None:
    """A pattern in both tables would make every match AMBIGUOUS by construction."""
    overlap = set(tier2.SYNTAX_PATTERNS) & set(tier2.SEMANTIC_PATTERNS)
    assert not overlap, overlap


def test_broad_pairs_tables_are_non_empty_and_disjoint() -> None:
    assert len(tier2.SYNTAX_BROAD_PAIRS) > 0
    assert len(tier2.SEMANTIC_BROAD_PAIRS) > 0
    overlap = set(tier2.SYNTAX_BROAD_PAIRS) & set(tier2.SEMANTIC_BROAD_PAIRS)
    assert not overlap, overlap


# --------------------------------------------------------------------------- #
# Behavioral tests — real diagnostics captured during the M4 pilot
# --------------------------------------------------------------------------- #


def test_undefined_reference_is_linker() -> None:
    stderr = (
        "/tmp/ccXXXXXX.o: in function `main':\n"
        "main.cpp:(.text+0x10): undefined reference to `foo()'\n"
        "collect2: error: ld returned 1 exit status\n"
    )
    assert tier2.classify(1, stderr) is DefectClass.LINKER


def test_windows_collect2_exe_naming_is_still_linker() -> None:
    """The actual regression: 'collect2.exe:', not 'collect2:', on Windows."""
    stderr = "collect2.exe: error: ld returned 1 exit status\n"
    assert tier2.classify(1, stderr) is DefectClass.LINKER


def test_relocation_overflow_is_linker() -> None:
    """A second, legitimate LINKER cause found during Linux remeasurement:
    large static/global data overflowing a 32-bit relative relocation."""
    stderr = (
        "main.cpp:(.text+0x1f5): relocation truncated to fit: R_X86_64_PC32 "
        "against symbol `mans' defined in .bss section\n"
        "collect2: error: ld returned 1 exit status\n"
    )
    assert tier2.classify(1, stderr) is DefectClass.LINKER


def test_curly_quote_syntax_message_is_syntax() -> None:
    """Regression test for the Unicode-quote bug: GCC uses U+2018/U+2019, not
    the ASCII apostrophe. This exact message silently returned UNRECOGNIZED
    before quote normalisation was added."""
    stderr = "main.cpp:23:18: error: expected \u2019}\u2019 at end of input\n"
    assert tier2.classify(1, stderr) is DefectClass.SYNTAX


def test_stray_character_is_syntax() -> None:
    stderr = "main.cpp:1:5: error: stray '#' in program\n"
    assert tier2.classify(1, stderr) is DefectClass.SYNTAX


def test_expected_before_broad_rule_is_syntax() -> None:
    stderr = "main.cpp:23:5: error: expected ',' or ';' before 'cout'\n"
    assert tier2.classify(1, stderr) is DefectClass.SYNTAX


def test_not_declared_in_scope_is_semantic() -> None:
    stderr = "main.cpp:9:15: error: 'foo' was not declared in this scope\n"
    assert tier2.classify(1, stderr) is DefectClass.SEMANTIC


def test_does_not_name_a_type_is_semantic() -> None:
    stderr = "main.cpp:1:1: error: 'include' does not name a type\n"
    assert tier2.classify(1, stderr) is DefectClass.SEMANTIC


def test_ambiguous_reference_is_semantic() -> None:
    stderr = "main.cpp:17:13: error: reference to 'gcd' is ambiguous\n"
    assert tier2.classify(1, stderr) is DefectClass.SEMANTIC


def test_known_external_dependency_is_missing_dependency_discard() -> None:
    stderr = "main.cpp:2:10: fatal error: atcoder/all: No such file or directory\n"
    assert tier2.classify(1, stderr) is Tier2DiscardReason.MISSING_DEPENDENCY


@pytest.mark.parametrize(
    "known", ["atcoder/all", "boost/multiprecision/cpp_int.hpp", "stdafx.h", "pch.h", "iostream.h"]
)
def test_each_allowlisted_dependency_is_missing_dependency_discard(known: str) -> None:
    stderr = f"main.cpp:1:10: fatal error: {known}: No such file or directory\n"
    assert tier2.classify(1, stderr) is Tier2DiscardReason.MISSING_DEPENDENCY


def test_corrupted_include_not_on_allowlist_is_syntax() -> None:
    """A genuinely typo'd/corrupted #include is a code defect, not a missing
    dependency -- e.g. 'bits/stdc++' (missing '.h'), not an external library."""
    stderr = "main.cpp:1:9: fatal error: bits/stdc++: No such file or directory\n"
    assert tier2.classify(1, stderr) is DefectClass.SYNTAX


def test_clean_compile_is_agreement_filter_discard() -> None:
    assert tier2.classify(0, "") is Tier2DiscardReason.AGREEMENT_FILTER


def test_timeout_is_compilation_timeout_discard_regardless_of_other_args() -> None:
    """timed_out must win even if returncode/stderr look like a clean compile
    or a recognisable error -- a killed process's exit code isn't meaningful."""
    assert tier2.classify(0, "", timed_out=True) is Tier2DiscardReason.COMPILATION_TIMEOUT
    assert (
        tier2.classify(1, "undefined reference to `foo'", timed_out=True)
        is Tier2DiscardReason.COMPILATION_TIMEOUT
    )


def test_ambiguous_when_first_line_matches_both_tables() -> None:
    stderr = "main.cpp:1:1: error: expected 'foo' was not declared in this scope before ';'\n"
    assert tier2.classify(1, stderr) is Tier2ReviewFlag.AMBIGUOUS


def test_unrecognized_when_no_pattern_matches() -> None:
    stderr = "main.cpp:1:1: error: this is not a real gcc diagnostic message\n"
    assert tier2.classify(1, stderr) is Tier2ReviewFlag.UNRECOGNIZED


def test_only_the_first_error_line_is_used() -> None:
    """A second, cascading error must not influence classification."""
    stderr = (
        "main.cpp:1:1: error: stray '#' in program\n"
        "main.cpp:9:15: error: 'foo' was not declared in this scope\n"
    )
    assert tier2.classify(1, stderr) is DefectClass.SYNTAX


# --------------------------------------------------------------------------- #
# Behavioral tests -- full-reserve frequency evidence (2nd pattern-table pass)
# --------------------------------------------------------------------------- #


def test_fatal_error_fallback_when_no_plain_error_line_exists() -> None:
    """Regression test: GCC's fatal-error format is ': fatal error:', which
    contains no bare ': error:' substring. Without the fallback, this would
    extract an empty first-error-line and misclassify as UNRECOGNIZED."""
    stderr = "main.cpp:1:1: fatal error: some non-missing-file fatal condition\n"
    # Not a missing-dependency case (no "no such file or directory"), so this
    # must still reach pattern matching via the fatal-error fallback line,
    # not silently extract "".
    line = tier2._first_error_line(stderr)
    assert line != ""
    assert "fatal error" in line


@pytest.mark.parametrize(
    ("stderr_fragment", "expected"),
    [
        ("error: invalid types 'int[int]' for array subscript", DefectClass.SEMANTIC),
        ("error: conflicting declaration 'double n'", DefectClass.SEMANTIC),
        (
            "error: return-statement with no value, in function returning 'int' [-fpermissive]",
            DefectClass.SEMANTIC,
        ),
        ("error: 'else' without a previous 'if'", DefectClass.SYNTAX),
        ("error: too many decimal points in number", DefectClass.SYNTAX),
        ("error: empty character constant", DefectClass.SYNTAX),
        (
            "error: macro 'mp' requires 3 arguments, but only 2 given",
            DefectClass.SYNTAX,
        ),
        (
            "/usr/include/c++/15/bits/hashtable.h:210:51: error: static assertion failed: "
            "hash function must be copy constructible",
            DefectClass.SEMANTIC,
        ),
        ("error: break statement not within loop or switch", DefectClass.SEMANTIC),
        ("error: continue statement not within a loop", DefectClass.SEMANTIC),
        (
            "error: narrowing conversion of '1.52e+3' from 'double' to 'int' [-Wnarrowing]",
            DefectClass.SEMANTIC,
        ),
        ("error: 'st' is not a namespace-name", DefectClass.SEMANTIC),
        (
            "error: two or more data types in declaration of 'structured binding'",
            DefectClass.SYNTAX,
        ),
        ("error: found ':' in nested-name-specifier, expected '::'", DefectClass.SYNTAX),
    ],
)
def test_full_reserve_frequency_patterns(stderr_fragment: str, expected: DefectClass) -> None:
    stderr = f"main.cpp:1:1: {stderr_fragment}\n"
    assert tier2.classify(1, stderr) is expected


def test_template_argument_is_invalid_broad_pair_is_semantic() -> None:
    """The digit in 'template argument N is invalid' breaks a literal
    substring match -- this must go through SEMANTIC_BROAD_PAIRS."""
    stderr = "main.cpp:1:1: error: template argument 1 is invalid\n"
    assert tier2.classify(1, stderr) is DefectClass.SEMANTIC


# --------------------------------------------------------------------------- #
# Invariant tests
# --------------------------------------------------------------------------- #


def test_every_possible_outcome_is_a_recognised_type() -> None:
    sample_calls = [
        (0, "", False),
        (1, "undefined reference to `foo'", False),
        (1, "main.cpp:1:1: error: stray '#' in program", False),
        (1, "main.cpp:1:1: error: 'x' was not declared in this scope", False),
        (1, "fatal error: atcoder/all: No such file or directory", False),
        (1, "nonsense", False),
        (1, "", True),
    ]
    for returncode, stderr, timed_out in sample_calls:
        outcome = tier2.classify(returncode, stderr, timed_out=timed_out)
        assert isinstance(outcome, (DefectClass, Tier2DiscardReason, Tier2ReviewFlag)), outcome


def test_is_labelled_agrees_with_defect_class_membership() -> None:
    assert tier2.is_labelled(DefectClass.LINKER) is True
    assert tier2.is_labelled(DefectClass.SYNTAX) is True
    assert tier2.is_labelled(DefectClass.SEMANTIC) is True
    assert tier2.is_labelled(Tier2DiscardReason.AGREEMENT_FILTER) is False
    assert tier2.is_labelled(Tier2DiscardReason.MISSING_DEPENDENCY) is False
    assert tier2.is_labelled(Tier2DiscardReason.COMPILATION_TIMEOUT) is False
    assert tier2.is_labelled(Tier2ReviewFlag.AMBIGUOUS) is False
    assert tier2.is_labelled(Tier2ReviewFlag.UNRECOGNIZED) is False


def test_missing_dependency_check_is_case_insensitive_via_normalization() -> None:
    """classify() lowercases before matching -- pin that this is deliberate,
    since GCC's own casing is consistent but defensive normalisation is
    cheap and matches the quote-normalisation precedent."""
    stderr = "main.cpp:1:10: FATAL ERROR: atcoder/all: No Such File Or Directory\n"
    assert tier2.classify(1, stderr) is Tier2DiscardReason.MISSING_DEPENDENCY


# --------------------------------------------------------------------------- #
# Cross-module agreement with taxonomy
# --------------------------------------------------------------------------- #


def test_tier2_only_ever_produces_tier2_leaves() -> None:
    """Every DefectClass classify() can return must be exactly the set of
    leaves taxonomy.py assigns to derivation tier 2 -- no more, no less."""
    tier2_leaves = {leaf for leaf in tax.LEAF_ORDER if tax.TIER_OF[leaf] == 2}
    assert tier2_leaves == {DefectClass.SYNTAX, DefectClass.SEMANTIC, DefectClass.LINKER}


def test_tier2_leaves_all_belong_to_compile_error_parent() -> None:
    tier2_leaves = {leaf for leaf in tax.LEAF_ORDER if tax.TIER_OF[leaf] == 2}
    for leaf in tier2_leaves:
        assert tax.to_coarse(leaf) is tax.CoarseClass.COMPILE_ERROR
