"""Build the token cache for the random-split comparison model (Milestone 8.5).

    python scripts/build_random_split_cache.py

Reads the frozen problem-level cache (data/processed/tokenized_9class/), re-partitions its
rows by the manifest's ``random_split`` column (see sdp.data.resplit) and writes
data/processed/tokenized_9class_random/{train,val,test}.parquet plus a small JSON record.
Nothing is re-tokenized. Refuses to run if any output already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

from sdp.config import PROJECT_ROOT
from sdp.data.dataset import MANIFEST_9CLASS_PATH, TOKENIZED_9CLASS_DIR
from sdp.data.resplit import SPLITS, repartition_cache
from sdp.evaluation.metrics import LEAF_NAMES
from sdp.evaluation.predictions import require_hash

SPLIT_CONFIG_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "split_config.json"
MANIFEST_KEY = "sample_manifest_9class.parquet"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "tokenized_9class_random",
    )
    out_dir: Path = parser.parse_args(argv).out_dir

    targets = [out_dir / f"{split}.parquet" for split in SPLITS]
    record_path = out_dir / "cache_record.json"
    existing = [path for path in (*targets, record_path) if path.exists()]
    if existing:
        for path in existing:
            print(f"STOP: {path} already exists; delete it deliberately to rebuild.")
        return 1

    expected = json.loads(SPLIT_CONFIG_PATH.read_text(encoding="utf-8"))["artifact_hashes"][
        MANIFEST_KEY
    ]
    manifest_sha = require_hash(MANIFEST_9CLASS_PATH, expected, "9-class manifest")
    manifest = pd.read_parquet(
        MANIFEST_9CLASS_PATH, columns=["submission_id", "leaf_id", "random_split"]
    )
    frames = {split: pd.read_parquet(TOKENIZED_9CLASS_DIR / f"{split}.parquet") for split in SPLITS}
    partitions = repartition_cache(frames, manifest, "random_split")

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    hashes: dict[str, str] = {}
    for split, part in partitions.items():
        path = out_dir / f"{split}.parquet"
        part.to_parquet(path, index=False)
        reread = pd.read_parquet(path, columns=["submission_id", "label"])
        if len(reread) != len(part):
            raise SystemExit(f"{path} read back with a different row count")
        hashes[path.name] = sha256_of(path)
        by_class = part["label"].value_counts().reindex(range(len(LEAF_NAMES)), fill_value=0)
        counts[split] = {name: int(by_class[index]) for index, name in enumerate(LEAF_NAMES)}
        print(f"{split}: {len(part)} files")
    print(pd.DataFrame(counts).T.to_string())

    record = {
        "milestone": "M8.5",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_sha256": manifest_sha,
        "source_cache_dir": str(TOKENIZED_9CLASS_DIR),
        "split_column": "random_split",
        "rows": {split: len(part) for split, part in partitions.items()},
        "class_counts": counts,
        "sha256": hashes,
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(record_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
