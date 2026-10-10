"""Tests for sdp.data.labeling.assemble (M6.2) -- no real corpus needed."""

import pytest

from sdp.data.labeling.assemble import (
    Exclusion,
    is_labelled,
    resolve_leaf,
)
from sdp.data.labeling.tier2 import Tier2DiscardReason, Tier2ReviewFlag
from sdp.data.labeling.tier3 import Tier3Discard, Tier3Label
from sdp.data.taxonomy import DefectClass, TIER_OF


@pytest.mark.parametrize("label", ["ERROR_FREE", "LOGICAL"])
def test_terminal_classes_pass_through(label):
    assert resolve_leaf(label) is DefectClass(label)


@pytest.mark.parametrize("leaf", ["SYNTAX", "SEMANTIC", "LINKER"])
def test_tier2_leaves(leaf):
    assert resolve_leaf("COMPILE_ERROR", tier2=leaf) is DefectClass(leaf)


@pytest.mark.parametrize(
    "reason",
    [r.value for r in Tier2DiscardReason] + [f.value for f in Tier2ReviewFlag],
)
def test_tier2_discards(reason):
    out = resolve_leaf("COMPILE_ERROR", tier2=reason)
    assert out is Exclusion(reason)
    assert not is_labelled(out)


@pytest.mark.parametrize("leaf", ["SIGSEGV", "SIGFPE", "SIGABRT", "NZEC"])
def test_tier3_leaves(leaf):
    assert resolve_leaf("RUNTIME_ERROR", tier3=leaf) is DefectClass(leaf)


@pytest.mark.parametrize("reason", [d.value for d in Tier3Discard])
def test_tier3_discards(reason):
    out = resolve_leaf("RUNTIME_ERROR", tier3=reason)
    assert out is Exclusion(reason)
    assert not is_labelled(out)


def test_all_nine_leaves_are_reachable():
    reached = {
        resolve_leaf("ERROR_FREE"),
        resolve_leaf("LOGICAL"),
        *(
            resolve_leaf("COMPILE_ERROR", tier2=x)
            for x in ("SYNTAX", "SEMANTIC", "LINKER")
        ),
        *(
            resolve_leaf("RUNTIME_ERROR", tier3=x.value)
            for x in Tier3Label
        ),
    }
    assert reached == set(DefectClass)


@pytest.mark.parametrize(
    "args",
    [
        ("ERROR_FREE", "SYNTAX", None),  # terminal with a Tier-2 entry
        ("LOGICAL", None, "NZEC"),  # terminal with a Tier-3 entry
        ("COMPILE_ERROR", None, None),  # missing M4 entry
        ("COMPILE_ERROR", "SYNTAX", "NZEC"),  # both entries
        ("COMPILE_ERROR", None, "NZEC"),  # wrong tier's entry
        ("RUNTIME_ERROR", None, None),  # missing M5 entry
        ("RUNTIME_ERROR", "SYNTAX", "NZEC"),  # both entries
        ("RUNTIME_ERROR", "SYNTAX", None),  # wrong tier's entry
    ],
)
def test_illegal_combinations_raise(args):
    with pytest.raises(ValueError):
        resolve_leaf(*args)


def test_unknown_strings_raise():
    with pytest.raises(ValueError):
        resolve_leaf("NOT_A_CLASS")
    with pytest.raises(ValueError):
        resolve_leaf("COMPILE_ERROR", tier2="SOMETHING_NEW")
    with pytest.raises(ValueError):
        resolve_leaf("RUNTIME_ERROR", tier3="SOMETHING_NEW")


def test_cross_tier_vocabulary_is_not_accepted():
    # A Tier-3 label in the Tier-2 slot (and vice versa) is a wrong file.
    with pytest.raises(ValueError):
        resolve_leaf("COMPILE_ERROR", tier2="SIGSEGV")
    with pytest.raises(ValueError):
        resolve_leaf("RUNTIME_ERROR", tier3="SYNTAX")


def test_nan_is_rejected_not_guessed():
    with pytest.raises(ValueError):
        resolve_leaf("COMPILE_ERROR", tier2=float("nan"))


def test_every_tier3_discard_has_an_exclusion_member():
    assert {d.value for d in Tier3Discard} <= {e.value for e in Exclusion}


def test_every_tier2_outcome_has_an_exclusion_member():
    tier2_strings = {r.value for r in Tier2DiscardReason} | {
        f.value for f in Tier2ReviewFlag
    }
    assert tier2_strings <= {e.value for e in Exclusion}


def test_tier3_labels_match_taxonomy_tier3_leaves():
    assert {m.value for m in Tier3Label} == {
        c.value for c in DefectClass if TIER_OF[c] == 3
    }


def test_every_tier2_leaf_in_taxonomy_is_accepted():
    for leaf in (c for c in DefectClass if TIER_OF[c] == 2):
        assert resolve_leaf("COMPILE_ERROR", tier2=leaf.value) is leaf


def test_exclusion_vocabulary_is_exactly_the_known_set():
    assert {e.value for e in Exclusion} == {
        "AGREEMENT_FILTER",
        "MISSING_DEPENDENCY",
        "COMPILATION_TIMEOUT",
        "AMBIGUOUS",
        "UNRECOGNIZED",
        "EXIT_ZERO",
        "TIMEOUT",
        "RESOURCE_LIMIT",
        "TOOLCHAIN_HARDENING",
        "COMPILE_FAILED",
        "UNRECOGNIZED_SIGNAL",
        "CONFLICTING_DUPLICATE",
    }
