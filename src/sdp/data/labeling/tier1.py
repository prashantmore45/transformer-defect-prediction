"""Tier-1 label derivation: CodeNet judge verdict -> coarse defect class.

This module is the single source of truth for the verdict vocabulary. It answers
exactly one question, for exactly one input string:

    Is this verdict kept (and if so, as which coarse class), deliberately
    discarded (and if so, why), or unrecognised (and therefore an error)?

Three outcomes, not two
----------------------
A two-outcome design (`CoarseClass | None`) conflates "discarded on purpose" with
"never seen before". A verdict IBM adds in a future release would then vanish
silently, surfacing weeks later as an unexplained row-count shortfall — if at
all. Silent data loss is the failure mode that makes results unreproducible, so
unknown verdicts raise instead.

Exact, case-sensitive matching
-----------------------------
Verdict strings are matched byte-for-byte. Case-folding would merge `Internal
error` with a hypothetical `Internal Error`, which is convenient but erases a
data-quality finding from the Milestone 1 EDA. The inconsistent capitalisation
is documented here in code rather than smoothed over.

Two verdict strings are easy to get wrong and are called out explicitly:
`WA: Presentation Error` carries a `WA:` prefix (the judge classifies it as a
Wrong Answer subtype), and `Internal error` has a lowercase 'e'.

Scope
-----
Tier 1 resolves two leaf classes (ERROR_FREE, LOGICAL) and two coarse classes
that fan out later: COMPILE_ERROR splits into SYNTAX/SEMANTIC/LINKER at Tier 2
via g++ diagnostic parsing, and RUNTIME_ERROR splits into
SIGSEGV/SIGFPE/SIGABRT/NZEC at Tier 3 via sandboxed execution.

Known limitation
----------------
`Wrong Answer -> LOGICAL` is the noisiest mapping in the taxonomy. The verdict
fires on genuine logic errors, but also on mishandled edge cases and abandoned
partial attempts. Symmetrically, `Accepted -> ERROR_FREE` means "passed the
judge's test cases", not "correct". Judge verdicts record what happened at
runtime; defect types describe what is wrong with the code. The mapping between
them is many-to-many, and the resulting label noise is irreducible.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Final, Iterable, Mapping

from sdp.data.taxonomy import CoarseClass

# Counts observed in Project CodeNet C++ metadata (8,008,527 rows), recorded in
# notebooks/02_metadata_eda.ipynb. Kept here for provenance only; the tests do
# not assert them, since they describe the data rather than the contract.
#
#   Accepted                4,353,049   Time Limit Exceeded     326,340
#   Wrong Answer            2,571,284   WA: Presentation Error   26,449
#   Compile Error             376,053   Memory Limit Exceeded    14,637
#   Runtime Error             339,670   Output Limit Exceeded       778
#                                       Judge Not Available          94
#                                       Query Limit Exceeded         88
#                                       Internal error               78
#                                       Judge System Error            7


class DiscardReason(StrEnum):
    """Why a known verdict is excluded from the candidate pool.

    The reason records the *rationale for dropping the row*, not the specific
    limit that was breached — the `status` column already preserves that, so
    reports can break down by verdict and by reason independently.
    """

    EFFICIENCY_CONSTRAINT = "EFFICIENCY_CONSTRAINT"
    """A resource or query budget was exceeded. The code may be functionally
    correct; inefficiency is not one of the nine defect classes."""

    OUTPUT_FORMAT = "OUTPUT_FORMAT"
    """Output is correct but formatted differently. The judge's own `WA:` prefix
    marks this as a weaker failure than a genuine wrong answer."""

    JUDGE_INFRASTRUCTURE = "JUDGE_INFRASTRUCTURE"
    """The evaluation itself failed. Says nothing about the submitted code."""


class UnknownVerdictError(ValueError):
    """Raised when a verdict string is not in the documented vocabulary.

    Subclasses ValueError so that `except ValueError` still catches it, while
    allowing callers to distinguish an unrecognised verdict from other bad input.
    """


# --------------------------------------------------------------------------- #
# The verdict vocabulary — the single source of truth
# --------------------------------------------------------------------------- #

VERDICT_TABLE: Final[Mapping[str, CoarseClass | DiscardReason]] = MappingProxyType(
    {
        # --- kept: mapped to a coarse class ---------------------------------
        "Accepted": CoarseClass.ERROR_FREE,
        "Wrong Answer": CoarseClass.LOGICAL,
        "Compile Error": CoarseClass.COMPILE_ERROR,
        "Runtime Error": CoarseClass.RUNTIME_ERROR,
        # --- discarded: budget overruns -------------------------------------
        "Time Limit Exceeded": DiscardReason.EFFICIENCY_CONSTRAINT,
        "Memory Limit Exceeded": DiscardReason.EFFICIENCY_CONSTRAINT,
        "Output Limit Exceeded": DiscardReason.EFFICIENCY_CONSTRAINT,
        "Query Limit Exceeded": DiscardReason.EFFICIENCY_CONSTRAINT,
        # --- discarded: formatting ------------------------------------------
        "WA: Presentation Error": DiscardReason.OUTPUT_FORMAT,
        # --- discarded: judge-side failures ---------------------------------
        "Judge Not Available": DiscardReason.JUDGE_INFRASTRUCTURE,
        "Internal error": DiscardReason.JUDGE_INFRASTRUCTURE,
        "Judge System Error": DiscardReason.JUDGE_INFRASTRUCTURE,
    }
)

VERDICT_TO_CLASS: Final[Mapping[str, CoarseClass]] = MappingProxyType(
    {v: o for v, o in VERDICT_TABLE.items() if isinstance(o, CoarseClass)}
)

VERDICT_TO_DISCARD_REASON: Final[Mapping[str, DiscardReason]] = MappingProxyType(
    {v: o for v, o in VERDICT_TABLE.items() if isinstance(o, DiscardReason)}
)

KEPT_VERDICTS: Final[frozenset[str]] = frozenset(VERDICT_TO_CLASS)
DISCARDED_VERDICTS: Final[frozenset[str]] = frozenset(VERDICT_TO_DISCARD_REASON)
KNOWN_VERDICTS: Final[frozenset[str]] = frozenset(VERDICT_TABLE)


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #


def classify(verdict: str) -> CoarseClass | DiscardReason:
    """Resolve one verdict to a coarse class or a discard reason.

    Raises:
        UnknownVerdictError: the verdict is not in the documented vocabulary.
            This is deliberate — an unrecognised verdict must be a decision,
            never a silently dropped row.
    """
    try:
        return VERDICT_TABLE[verdict]
    except KeyError:
        raise UnknownVerdictError(
            f"unknown verdict {verdict!r}; " f"add it to VERDICT_TABLE with an explicit outcome"
        ) from None


def is_kept(verdict: str) -> bool:
    """True if this verdict contributes a labelled row to the candidate pool.

    Raises UnknownVerdictError on an unrecognised verdict.
    """
    return isinstance(classify(verdict), CoarseClass)


def coarse_class_of(verdict: str) -> CoarseClass:
    """Coarse class for a kept verdict.

    Raises:
        UnknownVerdictError: the verdict is not in the vocabulary.
        ValueError: the verdict is known but deliberately discarded.
    """
    outcome = classify(verdict)
    if isinstance(outcome, DiscardReason):
        raise ValueError(f"verdict {verdict!r} is discarded ({outcome.value}); it has no class")
    return outcome


def validate_vocabulary(verdicts: Iterable[str]) -> None:
    """Assert that every observed verdict is documented. Call before mapping.

    This is the bridge between the pure function above and the vectorised
    pandas path. `Series.map(VERDICT_TO_CLASS)` yields NaN for unrecognised
    keys, which would silently drop rows; calling this first restores the
    raise-on-unknown guarantee at negligible cost.

        validate_vocabulary(df["status"].unique())
        df["coarse_label"] = df["status"].map(VERDICT_TO_CLASS)

    Raises:
        UnknownVerdictError: listing every unrecognised verdict found.
    """
    unknown = sorted(set(verdicts) - KNOWN_VERDICTS)
    if unknown:
        raise UnknownVerdictError(
            f"{len(unknown)} unknown verdict(s): {unknown}; "
            f"add each to VERDICT_TABLE with an explicit outcome"
        )
