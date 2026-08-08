"""Tests for the defect taxonomy contract.

Two kinds of test live here, and the distinction matters:

Contract tests
    Hardcode the exact expected class ordering and integer IDs. These duplicate
    the implementation *on purpose*. The taxonomy is frozen; any edit to
    LEAF_ORDER, COARSE_ORDER, or the ID maps invalidates every trained
    checkpoint and every reported metric. These tests exist so that such an edit
    fails loudly and forces a conscious TAXONOMY_VERSION bump.

Invariant tests
    Assert structural properties that must hold regardless of the specific class
    names — the hierarchy partitions cleanly, IDs are contiguous, maps are
    mutual inverses. These continue to pass if the taxonomy is legitimately
    extended in future.
"""

from __future__ import annotations

import pytest

from sdp.data import taxonomy as tax
from sdp.data.taxonomy import CoarseClass, DefectClass

# --------------------------------------------------------------------------- #
# Contract tests — these SHOULD break if the taxonomy is edited
# --------------------------------------------------------------------------- #

EXPECTED_LEAF_IDS: dict[str, int] = {
    "ERROR_FREE": 0,
    "SYNTAX": 1,
    "SEMANTIC": 2,
    "LINKER": 3,
    "SIGSEGV": 4,
    "SIGFPE": 5,
    "SIGABRT": 6,
    "NZEC": 7,
    "LOGICAL": 8,
}

EXPECTED_COARSE_IDS: dict[str, int] = {
    "ERROR_FREE": 0,
    "COMPILE_ERROR": 1,
    "RUNTIME_ERROR": 2,
    "LOGICAL": 3,
}


def test_leaf_ids_are_frozen() -> None:
    """Leaf class IDs must never change. See module docstring."""
    actual = {leaf.value: tax.leaf_id(leaf) for leaf in tax.LEAF_ORDER}
    assert actual == EXPECTED_LEAF_IDS


def test_coarse_ids_are_frozen() -> None:
    """Coarse class IDs must never change. See module docstring."""
    actual = {cls.value: tax.coarse_id(cls) for cls in tax.COARSE_ORDER}
    assert actual == EXPECTED_COARSE_IDS


def test_class_counts() -> None:
    assert tax.NUM_LEAVES == 9
    assert tax.NUM_COARSE == 4


def test_leaf_order_follows_pipeline_stages() -> None:
    """Ordering is by build/execution stage, not alphabetical.

    This ordering makes confusion matrices interpretable: pipeline-adjacent
    classes are matrix-adjacent.
    """
    assert [leaf.value for leaf in tax.LEAF_ORDER] == list(EXPECTED_LEAF_IDS)
    # Guard against someone "tidying" it into alphabetical order.
    assert [leaf.value for leaf in tax.LEAF_ORDER] != sorted(EXPECTED_LEAF_IDS)


def test_taxonomy_version_is_a_string() -> None:
    assert isinstance(tax.TAXONOMY_VERSION, str)
    assert tax.TAXONOMY_VERSION


# --------------------------------------------------------------------------- #
# Invariant tests — these should survive a legitimate future extension
# --------------------------------------------------------------------------- #


def test_every_leaf_has_a_parent_and_a_tier() -> None:
    for leaf in tax.LEAF_ORDER:
        assert leaf in tax.PARENT_OF
        assert leaf in tax.TIER_OF


def test_children_partition_the_leaves() -> None:
    """Every leaf belongs to exactly one parent; together they cover all leaves."""
    collected: list[DefectClass] = []
    for parent in tax.COARSE_ORDER:
        collected.extend(tax.CHILDREN_OF[parent])

    assert len(collected) == len(set(collected)), "a leaf appears under two parents"
    assert set(collected) == set(tax.LEAF_ORDER), "leaves are not fully covered"


def test_children_preserve_canonical_order() -> None:
    """Children are listed in LEAF_ORDER order, so grouped plots stay consistent."""
    for parent in tax.COARSE_ORDER:
        children = tax.CHILDREN_OF[parent]
        ids = [tax.leaf_id(child) for child in children]
        assert ids == sorted(ids)


@pytest.mark.parametrize(
    ("id_map", "reverse_map", "order"),
    [
        (tax.LEAF_TO_ID, tax.ID_TO_LEAF, tax.LEAF_ORDER),
        (tax.COARSE_TO_ID, tax.ID_TO_COARSE, tax.COARSE_ORDER),
    ],
)
def test_id_maps_are_contiguous_and_invertible(id_map, reverse_map, order) -> None:
    assert sorted(id_map.values()) == list(range(len(order)))
    for cls, idx in id_map.items():
        assert reverse_map[idx] == cls


def test_mappings_are_immutable() -> None:
    """Read-only views prevent accidental corruption of the contract at runtime."""
    for mapping in (tax.PARENT_OF, tax.CHILDREN_OF, tax.TIER_OF, tax.LEAF_TO_ID):
        with pytest.raises(TypeError):
            mapping[DefectClass.SYNTAX] = None  # type: ignore[index]


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("coarse", "expected"),
    [
        (CoarseClass.ERROR_FREE, True),
        (CoarseClass.LOGICAL, True),
        (CoarseClass.COMPILE_ERROR, False),
        (CoarseClass.RUNTIME_ERROR, False),
    ],
)
def test_is_terminal(coarse: CoarseClass, expected: bool) -> None:
    """Regression: RUNTIME_ERROR has no leaf of the same name and once raised here."""
    assert tax.is_terminal(coarse) is expected


def test_terminal_classes_are_their_own_leaf() -> None:
    for coarse in tax.COARSE_ORDER:
        if tax.is_terminal(coarse):
            assert tax.CHILDREN_OF[coarse] == (DefectClass(coarse.value),)


def test_to_coarse_maps_into_the_coarse_vocabulary() -> None:
    for leaf in tax.LEAF_ORDER:
        assert tax.to_coarse(leaf) in tax.COARSE_ORDER


@pytest.mark.parametrize(
    ("leaf", "expected"),
    [
        ("SYNTAX", CoarseClass.COMPILE_ERROR),
        ("LINKER", CoarseClass.COMPILE_ERROR),
        ("SIGFPE", CoarseClass.RUNTIME_ERROR),
        ("NZEC", CoarseClass.RUNTIME_ERROR),
        ("ERROR_FREE", CoarseClass.ERROR_FREE),
        ("LOGICAL", CoarseClass.LOGICAL),
    ],
)
def test_to_coarse_specific_mappings(leaf: str, expected: CoarseClass) -> None:
    assert tax.to_coarse(leaf) is expected


def test_leaves_for_tier_1_is_the_milestone_2_label_set() -> None:
    """Tier 1 resolves only the two terminal classes; the rest stay coarse."""
    assert tax.leaves_for_tier(1) == (DefectClass.ERROR_FREE, DefectClass.LOGICAL)


def test_leaves_for_tier_2_adds_compile_error_children() -> None:
    assert set(tax.leaves_for_tier(2)) == {
        DefectClass.ERROR_FREE,
        DefectClass.LOGICAL,
        DefectClass.SYNTAX,
        DefectClass.SEMANTIC,
        DefectClass.LINKER,
    }


def test_leaves_for_tier_3_is_the_full_taxonomy() -> None:
    assert set(tax.leaves_for_tier(3)) == set(tax.LEAF_ORDER)


def test_leaves_for_tier_is_monotonic() -> None:
    t1, t2, t3 = (set(tax.leaves_for_tier(n)) for n in (1, 2, 3))
    assert t1 <= t2 <= t3


@pytest.mark.parametrize("bad_tier", [0, 4, -1])
def test_leaves_for_tier_rejects_out_of_range(bad_tier: int) -> None:
    with pytest.raises(ValueError):
        tax.leaves_for_tier(bad_tier)


# --------------------------------------------------------------------------- #
# Error handling — typos must fail loudly, not silently
# --------------------------------------------------------------------------- #


def test_unknown_leaf_name_raises() -> None:
    with pytest.raises(ValueError):
        tax.leaf_id("SYTNAX")


def test_unknown_coarse_name_raises() -> None:
    with pytest.raises(ValueError):
        tax.coarse_id("COMPILE")


def test_coarse_only_name_is_not_a_leaf() -> None:
    """RUNTIME_ERROR is a parent, never a leaf. Confusing the two caused a bug."""
    with pytest.raises(ValueError):
        DefectClass("RUNTIME_ERROR")


def test_string_and_enum_arguments_agree() -> None:
    """StrEnum members compare equal to their values, so pandas filtering works."""
    assert tax.leaf_id("SIGSEGV") == tax.leaf_id(DefectClass.SIGSEGV)
    assert DefectClass.SIGSEGV == "SIGSEGV"
