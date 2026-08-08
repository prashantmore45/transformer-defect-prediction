"""Tests for Tier-1 verdict classification.

Three kinds of test live here:

Contract tests
    Hardcode all twelve verdict strings and their outcomes. These duplicate the
    implementation on purpose — the vocabulary was derived from a full pass over
    8,008,527 rows, and changing it should require a conscious decision, not a
    quiet edit. Case sensitivity is pinned explicitly: two verdicts in this
    dataset have irregular capitalisation, and folding case would erase that
    finding.

Invariant tests
    Structural properties that must hold whatever the vocabulary contains: the
    kept/discarded split is exhaustive, every coarse class is reachable, every
    discard reason is used.

Cross-module tests
    Verify that tier1 and taxonomy agree — the verdicts needing Tier 2/3
    refinement are exactly those mapping to non-terminal coarse classes.
"""

from __future__ import annotations

import pytest

from sdp.data import taxonomy as tax
from sdp.data.labeling import tier1
from sdp.data.labeling.tier1 import (
    DiscardReason,
    UnknownVerdictError,
)
from sdp.data.taxonomy import CoarseClass

# --------------------------------------------------------------------------- #
# Contract tests — these SHOULD break if the vocabulary is edited
# --------------------------------------------------------------------------- #

EXPECTED_TABLE: dict[str, str] = {
    "Accepted": "ERROR_FREE",
    "Wrong Answer": "LOGICAL",
    "Compile Error": "COMPILE_ERROR",
    "Runtime Error": "RUNTIME_ERROR",
    "Time Limit Exceeded": "EFFICIENCY_CONSTRAINT",
    "Memory Limit Exceeded": "EFFICIENCY_CONSTRAINT",
    "Output Limit Exceeded": "EFFICIENCY_CONSTRAINT",
    "Query Limit Exceeded": "EFFICIENCY_CONSTRAINT",
    "WA: Presentation Error": "OUTPUT_FORMAT",
    "Judge Not Available": "JUDGE_INFRASTRUCTURE",
    "Internal error": "JUDGE_INFRASTRUCTURE",
    "Judge System Error": "JUDGE_INFRASTRUCTURE",
}


def test_verdict_table_is_frozen() -> None:
    """All twelve verdicts and their outcomes. See module docstring."""
    actual = {v: str(o) for v, o in tier1.VERDICT_TABLE.items()}
    assert actual == EXPECTED_TABLE


def test_vocabulary_counts() -> None:
    assert len(tier1.KNOWN_VERDICTS) == 12
    assert len(tier1.KEPT_VERDICTS) == 4
    assert len(tier1.DISCARDED_VERDICTS) == 8


def test_irregular_verdict_strings_are_preserved_exactly() -> None:
    """Two verdicts have irregular spelling. Both are data-quality findings.

    `WA: Presentation Error` carries a prefix marking it as a Wrong Answer
    subtype; `Internal error` has a lowercase 'e'. Normalising either would
    hide an inconsistency discovered during the Milestone 1 EDA.
    """
    assert "WA: Presentation Error" in tier1.KNOWN_VERDICTS
    assert "Internal error" in tier1.KNOWN_VERDICTS
    assert "Presentation Error" not in tier1.KNOWN_VERDICTS
    assert "Internal Error" not in tier1.KNOWN_VERDICTS


@pytest.mark.parametrize(
    "near_miss",
    [
        "accepted",
        "ACCEPTED",
        "Internal Error",
        "WA: Presentation error",
        "wrong answer",
        "Runtime Errors",
        " Accepted",
        "Accepted ",
    ],
)
def test_matching_is_case_and_whitespace_sensitive(near_miss: str) -> None:
    """Guard against a future `.lower()` or `.strip()` in classify()."""
    with pytest.raises(UnknownVerdictError):
        tier1.classify(near_miss)


# --------------------------------------------------------------------------- #
# Invariant tests — should survive a legitimate vocabulary extension
# --------------------------------------------------------------------------- #


def test_kept_and_discarded_partition_the_vocabulary() -> None:
    """Every known verdict is exactly one of kept or discarded.

    The two sets are built by isinstance filtering. A verdict mapped to a plain
    string instead of an enum member would land in neither set — 'known' but
    unclassifiable, and silently treated as discarded. This sum is the only
    assertion that catches that.
    """
    assert tier1.KEPT_VERDICTS | tier1.DISCARDED_VERDICTS == tier1.KNOWN_VERDICTS
    assert not (tier1.KEPT_VERDICTS & tier1.DISCARDED_VERDICTS)
    assert len(tier1.KEPT_VERDICTS) + len(tier1.DISCARDED_VERDICTS) == len(tier1.KNOWN_VERDICTS)


def test_every_outcome_is_a_recognised_enum_member() -> None:
    for verdict, outcome in tier1.VERDICT_TABLE.items():
        assert isinstance(outcome, (CoarseClass, DiscardReason)), verdict


def test_every_coarse_class_is_reachable() -> None:
    """An unreachable class would be a permanently-empty output slot."""
    reached = set(tier1.VERDICT_TO_CLASS.values())
    assert reached == set(tax.COARSE_ORDER)


def test_every_discard_reason_is_used() -> None:
    """An unused reason is dead documentation."""
    used = set(tier1.VERDICT_TO_DISCARD_REASON.values())
    assert used == set(DiscardReason)


def test_mappings_are_immutable() -> None:
    for mapping in (
        tier1.VERDICT_TABLE,
        tier1.VERDICT_TO_CLASS,
        tier1.VERDICT_TO_DISCARD_REASON,
    ):
        with pytest.raises(TypeError):
            mapping["Accepted"] = None  # type: ignore[index]


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("Accepted", CoarseClass.ERROR_FREE),
        ("Wrong Answer", CoarseClass.LOGICAL),
        ("Compile Error", CoarseClass.COMPILE_ERROR),
        ("Runtime Error", CoarseClass.RUNTIME_ERROR),
    ],
)
def test_coarse_class_of_kept_verdicts(verdict: str, expected: CoarseClass) -> None:
    assert tier1.coarse_class_of(verdict) is expected
    assert tier1.classify(verdict) is expected
    assert tier1.is_kept(verdict) is True


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("Time Limit Exceeded", DiscardReason.EFFICIENCY_CONSTRAINT),
        ("Query Limit Exceeded", DiscardReason.EFFICIENCY_CONSTRAINT),
        ("WA: Presentation Error", DiscardReason.OUTPUT_FORMAT),
        ("Judge System Error", DiscardReason.JUDGE_INFRASTRUCTURE),
        ("Internal error", DiscardReason.JUDGE_INFRASTRUCTURE),
    ],
)
def test_discarded_verdicts_carry_a_reason(verdict: str, expected: DiscardReason) -> None:
    assert tier1.classify(verdict) is expected
    assert tier1.is_kept(verdict) is False


def test_coarse_class_of_rejects_discarded_verdicts() -> None:
    """Asking for the class of a discarded verdict is a caller bug, not a None."""
    with pytest.raises(ValueError) as exc:
        tier1.coarse_class_of("Time Limit Exceeded")
    assert "EFFICIENCY_CONSTRAINT" in str(exc.value)


def test_unknown_verdict_error_is_a_value_error() -> None:
    """`except ValueError` must still catch it."""
    assert issubclass(UnknownVerdictError, ValueError)
    with pytest.raises(ValueError):
        tier1.classify("Nonexistent Verdict")


def test_error_message_names_the_offending_verdict() -> None:
    with pytest.raises(UnknownVerdictError) as exc:
        tier1.classify("Accpeted")
    assert "Accpeted" in str(exc.value)


# --------------------------------------------------------------------------- #
# validate_vocabulary — the guard for the vectorised pandas path
# --------------------------------------------------------------------------- #


def test_validate_accepts_the_full_vocabulary() -> None:
    tier1.validate_vocabulary(tier1.KNOWN_VERDICTS)


def test_validate_accepts_a_subset_and_an_empty_series() -> None:
    tier1.validate_vocabulary(["Accepted", "Runtime Error"])
    tier1.validate_vocabulary([])


def test_validate_reports_every_unknown_not_just_the_first() -> None:
    """Stopping at the first would need N runs to find N typos."""
    with pytest.raises(UnknownVerdictError) as exc:
        tier1.validate_vocabulary(
            ["Accepted", "Runtime Errors", "WA: Presentation error", "Accepted"]
        )
    message = str(exc.value)
    assert "Runtime Errors" in message
    assert "WA: Presentation error" in message
    assert "2 unknown" in message


def test_vectorised_lookup_covers_every_kept_verdict() -> None:
    """Simulates `Series.map(VERDICT_TO_CLASS)` without importing pandas.

    Guards the invariant that makes the vectorised path safe: after
    validate_vocabulary passes, every kept verdict has a class and no lookup
    silently yields a missing value.
    """
    observed = list(tier1.KNOWN_VERDICTS)
    tier1.validate_vocabulary(observed)

    mapped = {v: tier1.VERDICT_TO_CLASS.get(v) for v in observed}
    for verdict, result in mapped.items():
        if verdict in tier1.KEPT_VERDICTS:
            assert result is not None, verdict
        else:
            assert result is None, verdict


# --------------------------------------------------------------------------- #
# Cross-module agreement with taxonomy
# --------------------------------------------------------------------------- #


def test_verdicts_needing_later_tiers_map_to_non_terminal_classes() -> None:
    """Compile and Runtime errors fan out; Accepted and Wrong Answer do not."""
    fans_out = {v for v, c in tier1.VERDICT_TO_CLASS.items() if not tax.is_terminal(c)}
    assert fans_out == {"Compile Error", "Runtime Error"}


def test_terminal_verdicts_are_fully_resolved_at_tier_1() -> None:
    terminal = {v for v, c in tier1.VERDICT_TO_CLASS.items() if tax.is_terminal(c)}
    assert terminal == {"Accepted", "Wrong Answer"}

    resolved = {tier1.coarse_class_of(v).value for v in terminal}
    assert resolved == {leaf.value for leaf in tax.leaves_for_tier(1)}
