"""Unified 9-class assembly -- per-row label resolution (M6.2).

Combines three sources into one leaf label per file:

* the manifest's Tier-1 `coarse_label` (frozen in M2),
* M4's `classification` for COMPILE_ERROR rows,
* M5's `outcome` for RUNTIME_ERROR rows.

Pure functions, no file or process I/O. Deliberately STRICT, unlike tier2/tier3:
those classify open-vocabulary raw output (compiler text, signals) and return
review flags for the unmatched; here every input is a closed vocabulary produced
by our own earlier milestones, so an unknown string or an illegal combination
means a wrong or corrupted file and must raise, as tier1 does.

The accepted vocabularies are derived from the tier modules' own enums and
from taxonomy.TIER_OF, never retyped, so they cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import pandas as pd

from sdp.data.labeling.tier2 import Tier2DiscardReason, Tier2ReviewFlag
from sdp.data.labeling.tier3 import Tier3Discard, Tier3Label
from sdp.data.taxonomy import (
    PARENT_OF,
    TIER_OF,
    CoarseClass,
    DefectClass,
    is_terminal,
    leaf_id,
)


class Exclusion(StrEnum):
    """Every reason a manifest row can be left out of the 9-class corpus."""

    # Tier 2 (M4): mirrors Tier2DiscardReason + Tier2ReviewFlag
    AGREEMENT_FILTER = "AGREEMENT_FILTER"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    COMPILATION_TIMEOUT = "COMPILATION_TIMEOUT"
    AMBIGUOUS = "AMBIGUOUS"  # zero occurrences in the M4 run; kept as known vocabulary
    UNRECOGNIZED = "UNRECOGNIZED"

    # Tier 3 (M5): mirrors Tier3Discard
    EXIT_ZERO = "EXIT_ZERO"
    TIMEOUT = "TIMEOUT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    TOOLCHAIN_HARDENING = "TOOLCHAIN_HARDENING"
    COMPILE_FAILED = "COMPILE_FAILED"
    UNRECOGNIZED_SIGNAL = "UNRECOGNIZED_SIGNAL"

    # Assembly-level exclusion (applied in M6.3, over the whole manifest)
    CONFLICTING_DUPLICATE = "CONFLICTING_DUPLICATE"


AssemblyOutcome = DefectClass | Exclusion


_TIER2_LEAVES: Final = frozenset(c.value for c in DefectClass if TIER_OF[c] == 2)

_TIER2_DISCARDS: Final = frozenset(
    {r.value for r in Tier2DiscardReason} | {f.value for f in Tier2ReviewFlag}
)

_TIER3_LEAVES: Final = frozenset(m.value for m in Tier3Label)

_TIER3_DISCARDS: Final = frozenset(d.value for d in Tier3Discard)


def _check_source_value(name: str, value: object) -> str | None:
    """None passes through; any non-string (e.g. a pandas NaN) is an error."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a str or None, got {type(value).__name__}: {value!r}")
    return value


def _from_tier2(value: str) -> AssemblyOutcome:
    if value in _TIER2_LEAVES:
        return DefectClass(value)
    if value in _TIER2_DISCARDS:
        return Exclusion(value)
    raise ValueError(f"unrecognised Tier-2 classification: {value!r}")


def _from_tier3(value: str) -> AssemblyOutcome:
    if value in _TIER3_LEAVES:
        return DefectClass(value)
    if value in _TIER3_DISCARDS:
        return Exclusion(value)
    raise ValueError(f"unrecognised Tier-3 outcome: {value!r}")


def resolve_leaf(
    coarse_label: str,
    tier2: str | None = None,
    tier3: str | None = None,
) -> AssemblyOutcome:
    """Resolve one manifest row to a 9-class leaf label or a recorded exclusion.

    Args:
        coarse_label: the manifest's Tier-1 label (a CoarseClass name).
        tier2: M4's `classification` string, or None if the row has no M4 entry.
        tier3: M5's `outcome` string, or None if the row has no M5 entry.

    Raises:
        ValueError: unknown coarse label, unknown Tier-2/3 string, a non-string
            source value, or a combination the pipeline can never produce
            (e.g. a RUNTIME_ERROR row that also has an M4 entry).
    """
    coarse = CoarseClass(coarse_label)
    t2 = _check_source_value("tier2", tier2)
    t3 = _check_source_value("tier3", tier3)

    if is_terminal(coarse):
        if t2 is not None or t3 is not None:
            raise ValueError(
                f"terminal class {coarse.value} must have no Tier-2/3 entry "
                f"(tier2={t2!r}, tier3={t3!r})"
            )
        outcome: AssemblyOutcome = DefectClass(coarse.value)

    elif coarse is CoarseClass.COMPILE_ERROR:
        if t2 is None or t3 is not None:
            raise ValueError(
                f"COMPILE_ERROR needs a Tier-2 entry and no Tier-3 entry "
                f"(tier2={t2!r}, tier3={t3!r})"
            )
        outcome = _from_tier2(t2)

    else:  # RUNTIME_ERROR
        if t3 is None or t2 is not None:
            raise ValueError(
                f"RUNTIME_ERROR needs a Tier-3 entry and no Tier-2 entry "
                f"(tier2={t2!r}, tier3={t3!r})"
            )
        outcome = _from_tier3(t3)

    # Defensive: a labelled leaf must belong to the row's own parent class.
    if isinstance(outcome, DefectClass) and PARENT_OF[outcome] is not coarse:
        raise ValueError(f"leaf {outcome.value} is not a child of {coarse.value}")

    return outcome


def is_labelled(outcome: AssemblyOutcome) -> bool:
    """True only for real DefectClass leaves, not exclusions."""
    return isinstance(outcome, DefectClass)


# --------------------------------------------------------------------------- #
# DataFrame layer (M6.3a)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AssemblyResult:
    """Output of `assemble_manifest`.

    `pre_conflict_labelled` is the labelled-row count BEFORE the whole-manifest
    conflicting-duplicate rule; `conflict_removed` is how many labelled rows
    that rule then removed. Reported separately so the effect is measured.
    """

    retained: pd.DataFrame
    excluded: pd.DataFrame
    pre_conflict_labelled: int
    conflict_removed: int


def _require_columns(
    name: str,
    df: pd.DataFrame,
    required: set[str],
) -> None:
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{name} is missing columns: {sorted(missing)}")


def assemble_manifest(
    manifest: pd.DataFrame,
    tier2: pd.DataFrame,
    tier3: pd.DataFrame,
) -> AssemblyResult:
    """Join manifest + M4 + M5 into the unified 9-class corpus.

    Inputs are never mutated. Strict throughout: duplicate ids, orphan tier
    rows, missing tier entries, and unknown strings all raise.
    """
    _require_columns(
        "manifest",
        manifest,
        {"submission_id", "coarse_label", "sha256"},
    )
    _require_columns(
        "tier2",
        tier2,
        {"submission_id", "classification"},
    )
    _require_columns(
        "tier3",
        tier3,
        {"submission_id", "outcome"},
    )

    for name, df in (
        ("manifest", manifest),
        ("tier2", tier2),
        ("tier3", tier3),
    ):
        if df["submission_id"].duplicated().any():
            raise ValueError(f"{name} has duplicate submission_id values")

    known = set(manifest["submission_id"])

    for name, df in (
        ("tier2", tier2),
        ("tier3", tier3),
    ):
        orphans = set(df["submission_id"]) - known
        if orphans:
            raise ValueError(f"{name} has {len(orphans)} ids not in the manifest")

    t2 = dict(zip(tier2["submission_id"], tier2["classification"]))
    t3 = dict(zip(tier3["submission_id"], tier3["outcome"]))

    outcomes = [
        resolve_leaf(
            coarse,
            t2.get(sid),
            t3.get(sid),
        )
        for sid, coarse in zip(
            manifest["submission_id"],
            manifest["coarse_label"],
        )
    ]

    df = manifest.reset_index(drop=True).copy()
    df["_resolved"] = [o.value for o in outcomes]
    df["_labelled"] = [is_labelled(o) for o in outcomes]

    pre_conflict = int(df["_labelled"].sum())

    # Whole-manifest conflict rule: identical source (sha256), more than one
    # distinct resolved outcome (leaf or exclusion reason, across reserves),
    # applied to the group's labelled rows only.
    distinct = df.groupby("sha256")["_resolved"].transform("nunique")
    conflict = df["_labelled"] & (distinct > 1)

    df["exclusion_reason"] = [
        None if labelled else resolved
        for labelled, resolved in zip(
            df["_labelled"],
            df["_resolved"],
        )
    ]

    df.loc[conflict, "exclusion_reason"] = Exclusion.CONFLICTING_DUPLICATE.value

    df["_labelled"] = df["_labelled"] & ~conflict

    df["leaf_label"] = [
        resolved if labelled else None
        for labelled, resolved in zip(
            df["_labelled"],
            df["_resolved"],
        )
    ]

    retained = df[df["_labelled"]].drop(columns=["_resolved", "_labelled", "exclusion_reason"])

    retained = retained.copy()
    retained["leaf_id"] = retained["leaf_label"].map(leaf_id).astype(int)

    if "input_source" in tier3.columns:
        sources = dict(
            zip(
                tier3["submission_id"],
                tier3["input_source"],
            )
        )
        retained["input_source"] = retained["submission_id"].map(sources)

        excluded = (
            df[~df["_labelled"]]
            .drop(columns=["_labelled", "leaf_label"])
            .rename(columns={"_resolved": "resolved_outcome"})
        )

    return AssemblyResult(
        retained=retained.reset_index(drop=True),
        excluded=excluded.reset_index(drop=True),
        pre_conflict_labelled=pre_conflict,
        conflict_removed=int(conflict.sum()),
    )
