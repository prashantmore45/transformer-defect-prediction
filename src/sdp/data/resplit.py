"""Re-partition the frozen token cache by another split column (Milestone 8.5).

The M6 token cache is stored per PROBLEM-LEVEL split. M8.5 trains a comparison model on the
deliberately unsafe SUBMISSION-LEVEL random split (manifest column ``random_split``), so the
same tokenized files must be partitioned differently. Tokenization does not depend on the
split, so nothing is re-tokenized: rows are moved between splits, never changed.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

SPLITS: tuple[str, ...] = ("train", "val", "test")
CACHE_COLUMNS = ("submission_id", "input_ids", "label")


def repartition_cache(
    frames: Mapping[str, pd.DataFrame],
    manifest: pd.DataFrame,
    split_column: str = "random_split",
) -> dict[str, pd.DataFrame]:
    """Return {'train', 'val', 'test'} caches built from ``frames`` by ``split_column``.

    ``frames`` are the existing cache frames (any number of them); ``manifest`` needs
    ``submission_id``, ``leaf_id`` and ``split_column``. Row order follows the order of
    ``frames`` and is otherwise irrelevant (training shuffles, evaluation is order-free).
    """
    for name, frame in frames.items():
        missing = set(CACHE_COLUMNS) - set(frame.columns)
        if missing:
            raise KeyError(f"cache frame {name!r} is missing columns: {sorted(missing)}")
    combined = pd.concat(list(frames.values()), ignore_index=True)
    if combined["submission_id"].duplicated().any():
        raise ValueError("cache contains duplicate submission_id values")

    needed = manifest[["submission_id", "leaf_id", split_column]]
    if needed["submission_id"].duplicated().any():
        raise ValueError("manifest contains duplicate submission_id values")
    if set(combined["submission_id"]) != set(needed["submission_id"]):
        raise ValueError("cache and manifest cover different files")

    merged = combined.merge(needed, on="submission_id", validate="1:1")
    if not np.array_equal(merged["label"].to_numpy(), merged["leaf_id"].to_numpy()):
        raise ValueError("cached labels differ from the manifest leaf_id")

    partitions: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        part = merged[merged[split_column].astype(str) == split]
        part = part[list(CACHE_COLUMNS)].reset_index(drop=True)
        if part.empty:
            raise ValueError(f"split {split!r} is empty under column {split_column!r}")
        partitions[split] = part
    if sum(len(part) for part in partitions.values()) != len(merged):
        raise ValueError(f"some files have a {split_column!r} value other than {SPLITS}")
    return partitions
