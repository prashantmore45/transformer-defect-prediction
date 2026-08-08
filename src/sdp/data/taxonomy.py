"""Defect taxonomy — the single source of truth for class names, IDs, and hierarchy.

This module is a *contract*, not configuration. Every other module in the project
imports its class names and integer IDs from here. Nothing anywhere else may
hardcode a label string or assume an integer mapping.

Two levels are defined:

    Coarse (Tier 1) — derived from CodeNet judge verdicts alone. 4 classes.
    Leaf  (Tiers 1-3) — the full nine-class taxonomy from the base paper.

Only the coarse level is populated as of Milestone 2. The leaf level is declared
now so that no downstream module needs refactoring when Tier 2 (compiler
diagnostic parsing) and Tier 3 (sandboxed execution) are implemented.

Ordering rationale
------------------
Classes are ordered by the *stage of the build/execution pipeline* at which the
defect surfaces:

    nothing wrong -> parse -> semantic analysis -> link -> run -> compare output

This ordering is deliberate. It makes confusion matrices interpretable: adjacent
classes are pipeline-adjacent, so off-diagonal mass near the diagonal indicates
stage confusion, while distant off-diagonal mass indicates a genuine failure.
Alphabetical ordering would destroy this structure.

Stability guarantee
-------------------
Integer IDs are frozen. Changing LEAF_ORDER or COARSE_ORDER after a model has
been trained silently invalidates every checkpoint, every reported metric, and
every confusion matrix. `tests/test_taxonomy.py` pins the exact mapping; if that
test fails, the change is a breaking change and must be versioned.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Final, Mapping

# Stamped into every model artifact bundle so a checkpoint can be matched to the
# taxonomy it was trained against. Bump only if IDs or class membership change.
TAXONOMY_VERSION: Final[str] = "1.0"


class CoarseClass(StrEnum):
    """Tier-1 classes, derivable from the CodeNet `status` field alone."""

    ERROR_FREE = "ERROR_FREE"
    COMPILE_ERROR = "COMPILE_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    LOGICAL = "LOGICAL"


class DefectClass(StrEnum):
    """Full nine-class leaf taxonomy (base paper: Hussain, Yow & Gori, 2025)."""

    ERROR_FREE = "ERROR_FREE"
    SYNTAX = "SYNTAX"
    SEMANTIC = "SEMANTIC"
    LINKER = "LINKER"
    SIGSEGV = "SIGSEGV"
    SIGFPE = "SIGFPE"
    SIGABRT = "SIGABRT"
    NZEC = "NZEC"
    LOGICAL = "LOGICAL"


# --------------------------------------------------------------------------- #
# Canonical ordering. Index position IS the integer class ID.
# --------------------------------------------------------------------------- #

COARSE_ORDER: Final[tuple[CoarseClass, ...]] = (
    CoarseClass.ERROR_FREE,
    CoarseClass.COMPILE_ERROR,
    CoarseClass.RUNTIME_ERROR,
    CoarseClass.LOGICAL,
)

LEAF_ORDER: Final[tuple[DefectClass, ...]] = (
    DefectClass.ERROR_FREE,
    DefectClass.SYNTAX,
    DefectClass.SEMANTIC,
    DefectClass.LINKER,
    DefectClass.SIGSEGV,
    DefectClass.SIGFPE,
    DefectClass.SIGABRT,
    DefectClass.NZEC,
    DefectClass.LOGICAL,
)

NUM_COARSE: Final[int] = len(COARSE_ORDER)
NUM_LEAVES: Final[int] = len(LEAF_ORDER)


# --------------------------------------------------------------------------- #
# Hierarchy
# --------------------------------------------------------------------------- #

# ERROR_FREE and LOGICAL are terminal: they are their own parent. They never
# fan out at Tier 2 or Tier 3.
PARENT_OF: Final[Mapping[DefectClass, CoarseClass]] = MappingProxyType(
    {
        DefectClass.ERROR_FREE: CoarseClass.ERROR_FREE,
        DefectClass.SYNTAX: CoarseClass.COMPILE_ERROR,
        DefectClass.SEMANTIC: CoarseClass.COMPILE_ERROR,
        DefectClass.LINKER: CoarseClass.COMPILE_ERROR,
        DefectClass.SIGSEGV: CoarseClass.RUNTIME_ERROR,
        DefectClass.SIGFPE: CoarseClass.RUNTIME_ERROR,
        DefectClass.SIGABRT: CoarseClass.RUNTIME_ERROR,
        DefectClass.NZEC: CoarseClass.RUNTIME_ERROR,
        DefectClass.LOGICAL: CoarseClass.LOGICAL,
    }
)

CHILDREN_OF: Final[Mapping[CoarseClass, tuple[DefectClass, ...]]] = MappingProxyType(
    {
        parent: tuple(leaf for leaf in LEAF_ORDER if PARENT_OF[leaf] is parent)
        for parent in COARSE_ORDER
    }
)

# Which label-derivation tier resolves each leaf.
#   1 = judge verdict alone
#   2 = recompilation and g++ diagnostic parsing
#   3 = sandboxed execution and exit-signal capture
TIER_OF: Final[Mapping[DefectClass, int]] = MappingProxyType(
    {
        DefectClass.ERROR_FREE: 1,
        DefectClass.LOGICAL: 1,
        DefectClass.SYNTAX: 2,
        DefectClass.SEMANTIC: 2,
        DefectClass.LINKER: 2,
        DefectClass.SIGSEGV: 3,
        DefectClass.SIGFPE: 3,
        DefectClass.SIGABRT: 3,
        DefectClass.NZEC: 3,
    }
)


# --------------------------------------------------------------------------- #
# ID maps (derived from the canonical orderings above)
# --------------------------------------------------------------------------- #

COARSE_TO_ID: Final[Mapping[CoarseClass, int]] = MappingProxyType(
    {cls: idx for idx, cls in enumerate(COARSE_ORDER)}
)
ID_TO_COARSE: Final[Mapping[int, CoarseClass]] = MappingProxyType(
    {idx: cls for cls, idx in COARSE_TO_ID.items()}
)

LEAF_TO_ID: Final[Mapping[DefectClass, int]] = MappingProxyType(
    {cls: idx for idx, cls in enumerate(LEAF_ORDER)}
)
ID_TO_LEAF: Final[Mapping[int, DefectClass]] = MappingProxyType(
    {idx: cls for cls, idx in LEAF_TO_ID.items()}
)


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #


def coarse_id(cls: CoarseClass | str) -> int:
    """Integer ID for a coarse class. Raises ValueError on an unknown name."""
    return COARSE_TO_ID[CoarseClass(cls)]


def leaf_id(cls: DefectClass | str) -> int:
    """Integer ID for a leaf class. Raises ValueError on an unknown name."""
    return LEAF_TO_ID[DefectClass(cls)]


def to_coarse(cls: DefectClass | str) -> CoarseClass:
    """Map a leaf class up to its Tier-1 parent."""
    return PARENT_OF[DefectClass(cls)]


def is_terminal(cls: CoarseClass | str) -> bool:
    """True if this coarse class has no finer subdivision (ERROR_FREE, LOGICAL).

    Terminal classes are fully resolved at Tier 1 — they never fan out at
    Tier 2 or Tier 3, so a hierarchical head predicts them directly.
    """
    return len(CHILDREN_OF[CoarseClass(cls)]) == 1


def leaves_for_tier(max_tier: int) -> tuple[DefectClass, ...]:
    """Leaf classes resolvable using derivation tiers up to and including `max_tier`.

    Used to build the active label set for an experiment. At Milestone 2 only
    tier 1 is implemented, so `leaves_for_tier(1)` returns the two terminal
    classes and the coarse level is used for everything else.
    """
    if max_tier not in (1, 2, 3):
        raise ValueError(f"max_tier must be 1, 2 or 3; got {max_tier!r}")
    return tuple(leaf for leaf in LEAF_ORDER if TIER_OF[leaf] <= max_tier)
