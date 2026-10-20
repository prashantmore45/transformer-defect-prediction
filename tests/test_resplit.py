"""Tests for sdp.data.resplit (Milestone 8.5)."""

import pandas as pd
import pytest

from sdp.data.resplit import repartition_cache

IDS = [f"s{i}" for i in range(9)]


def make_cache() -> dict[str, pd.DataFrame]:
    """Nine files, stored under a 'problem-level' split of three files each."""
    rows = {
        "submission_id": IDS,
        "input_ids": [[i, i + 1, i + 2] for i in range(9)],
        "label": [i % 3 for i in range(9)],
    }
    frame = pd.DataFrame(rows)
    return {"train": frame.iloc[:3], "val": frame.iloc[3:6], "test": frame.iloc[6:]}


def make_manifest(random_split: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "submission_id": IDS,
            "leaf_id": [i % 3 for i in range(9)],
            "random_split": random_split,
        }
    )


GOOD = ["train", "test", "val", "train", "val", "test", "train", "val", "test"]


def test_each_file_lands_in_the_split_the_manifest_assigns() -> None:
    parts = repartition_cache(make_cache(), make_manifest(GOOD))
    assert parts["train"]["submission_id"].tolist() == ["s0", "s3", "s6"]
    assert parts["val"]["submission_id"].tolist() == ["s2", "s4", "s7"]
    assert parts["test"]["submission_id"].tolist() == ["s1", "s5", "s8"]


def test_rows_are_moved_not_changed() -> None:
    parts = repartition_cache(make_cache(), make_manifest(GOOD))
    moved = pd.concat(parts.values()).set_index("submission_id")
    for index, submission_id in enumerate(IDS):
        assert list(moved.loc[submission_id, "input_ids"]) == [index, index + 1, index + 2]
        assert moved.loc[submission_id, "label"] == index % 3
    assert list(parts["train"].columns) == ["submission_id", "input_ids", "label"]


def test_every_file_appears_exactly_once() -> None:
    parts = repartition_cache(make_cache(), make_manifest(GOOD))
    all_ids = pd.concat(parts.values())["submission_id"]
    assert sorted(all_ids) == sorted(IDS)


def test_duplicate_cache_ids_are_rejected() -> None:
    cache = make_cache()
    cache["val"] = pd.concat([cache["val"], cache["train"].iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        repartition_cache(cache, make_manifest(GOOD))


def test_cache_and_manifest_must_cover_the_same_files() -> None:
    manifest = make_manifest(GOOD).iloc[:-1]
    with pytest.raises(ValueError, match="different files"):
        repartition_cache(make_cache(), manifest)


def test_label_disagreement_is_rejected() -> None:
    manifest = make_manifest(GOOD)
    manifest.loc[0, "leaf_id"] = 2
    with pytest.raises(ValueError, match="labels differ"):
        repartition_cache(make_cache(), manifest)


def test_unknown_split_value_is_rejected() -> None:
    bad = [*GOOD[:-1], "holdout"]
    with pytest.raises(ValueError, match="other than"):
        repartition_cache(make_cache(), make_manifest(bad))


def test_an_empty_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        repartition_cache(make_cache(), make_manifest(["train"] * 6 + ["val"] * 3))


def test_missing_cache_columns_are_rejected() -> None:
    cache = make_cache()
    cache["train"] = cache["train"].drop(columns=["label"])
    with pytest.raises(KeyError, match="label"):
        repartition_cache(cache, make_manifest(GOOD))
