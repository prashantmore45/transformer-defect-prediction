"""Tests for assemble_manifest (M6.3a) -- small synthetic tables only."""

import pandas as pd
import pytest

from sdp.data.labeling.assemble import assemble_manifest
from sdp.data.taxonomy import LEAF_TO_ID, DefectClass

# sid, coarse_label, sha256, tier2 classification, tier3 outcome
_ROWS = [
    ("a1", "ERROR_FREE", "A", None, None),
    ("a2", "LOGICAL", "B", None, None),
    ("c1", "COMPILE_ERROR", "C", "SEMANTIC", None),
    ("c2", "COMPILE_ERROR", "D", "AGREEMENT_FILTER", None),
    ("r1", "RUNTIME_ERROR", "E", None, "SIGSEGV"),
    ("r2", "RUNTIME_ERROR", "F", None, "EXIT_ZERO"),
    ("r3", "RUNTIME_ERROR", "G", None, "SIGSEGV"),  # conflicts with r4
    ("r4", "RUNTIME_ERROR", "G", None, "EXIT_ZERO"),
    ("r5", "RUNTIME_ERROR", "H", None, "NZEC"),  # same-label duplicate pair
    ("r6", "RUNTIME_ERROR", "H", None, "NZEC"),
    ("a3", "ERROR_FREE", "I", None, None),  # cross-reserve conflict
    ("a4", "LOGICAL", "I", None, None),
    ("r7", "RUNTIME_ERROR", "J", None, "EXIT_ZERO"),  # excluded duplicate pair
    ("r8", "RUNTIME_ERROR", "J", None, "EXIT_ZERO"),
]


def _make():
    manifest = pd.DataFrame(
        {
            "submission_id": [r[0] for r in _ROWS],
            "coarse_label": [r[1] for r in _ROWS],
            "sha256": [r[2] for r in _ROWS],
            "random_split": ["train"] * len(_ROWS),
        }
    )
    tier2 = pd.DataFrame(
        [(r[0], r[3]) for r in _ROWS if r[3]],
        columns=["submission_id", "classification"],
    )
    tier3 = pd.DataFrame(
        [(r[0], r[4], "verified") for r in _ROWS if r[4]],
        columns=["submission_id", "outcome", "input_source"],
    )
    return manifest, tier2, tier3


def _reasons(res):
    return dict(zip(res.excluded["submission_id"], res.excluded["exclusion_reason"]))


def test_counts_before_and_after_conflict_rule():
    res = assemble_manifest(*_make())
    assert res.pre_conflict_labelled == 9
    assert res.conflict_removed == 3
    assert len(res.retained) == 6
    assert len(res.excluded) == 8


def test_retained_and_excluded_partition_the_manifest():
    manifest, t2, t3 = _make()
    res = assemble_manifest(manifest, t2, t3)
    kept = set(res.retained["submission_id"])
    dropped = set(res.excluded["submission_id"])
    assert kept.isdisjoint(dropped)
    assert kept | dropped == set(manifest["submission_id"])


def test_conflict_rule_removes_labelled_rows_only():
    reasons = _reasons(assemble_manifest(*_make()))
    assert reasons["r3"] == "CONFLICTING_DUPLICATE"
    assert reasons["r4"] == "EXIT_ZERO"  # keeps its own, more informative reason
    assert reasons["a3"] == reasons["a4"] == "CONFLICTING_DUPLICATE"  # cross-reserve


def test_same_label_duplicates_are_kept():
    res = assemble_manifest(*_make())
    assert {"r5", "r6"} <= set(res.retained["submission_id"])


def test_duplicate_groups_of_excluded_rows_are_not_conflicts():
    reasons = _reasons(assemble_manifest(*_make()))
    assert reasons["r7"] == reasons["r8"] == "EXIT_ZERO"


def test_leaf_ids_come_from_taxonomy():
    res = assemble_manifest(*_make())
    for row in res.retained.itertuples():
        assert row.leaf_id == LEAF_TO_ID[DefectClass(row.leaf_label)]


def test_manifest_columns_and_input_source_are_carried():
    res = assemble_manifest(*_make())
    assert {"random_split", "sha256", "leaf_label", "leaf_id", "input_source"} <= set(
        res.retained.columns
    )
    by_id = res.retained.set_index("submission_id")
    assert by_id.loc["r1", "input_source"] == "verified"
    assert pd.isna(by_id.loc["a1", "input_source"])


def test_inputs_are_not_mutated():
    manifest, t2, t3 = _make()
    before = (manifest.copy(), t2.copy(), t3.copy())
    assemble_manifest(manifest, t2, t3)
    for after, orig in zip((manifest, t2, t3), before):
        pd.testing.assert_frame_equal(after, orig)


def test_orphan_tier_row_raises():
    manifest, t2, t3 = _make()
    t2 = pd.concat([t2, pd.DataFrame([("zzz", "SYNTAX")], columns=t2.columns)], ignore_index=True)
    with pytest.raises(ValueError):
        assemble_manifest(manifest, t2, t3)


def test_duplicate_submission_id_raises():
    manifest, t2, t3 = _make()
    t2 = pd.concat([t2, t2.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError):
        assemble_manifest(manifest, t2, t3)


def test_missing_tier_entry_raises():
    manifest, t2, t3 = _make()
    t2 = t2[t2["submission_id"] != "c1"]
    with pytest.raises(ValueError):
        assemble_manifest(manifest, t2, t3)


def test_excluded_rows_keep_their_pre_conflict_outcome():
    res = assemble_manifest(*_make())
    resolved = dict(zip(res.excluded["submission_id"], res.excluded["resolved_outcome"]))
    assert resolved["r3"] == "SIGSEGV"  # labelled before the conflict rule
    assert resolved["r4"] == "EXIT_ZERO"
    assert resolved["c2"] == "AGREEMENT_FILTER"
