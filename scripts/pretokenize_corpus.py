"""Build the 9-class pre-tokenized cache and verify the DataLoader pipeline end-to-end.

Run locally (not part of CI: needs the real corpus on disk and, on first run,
network access to the Hugging Face Hub):

    python scripts/pretokenize_corpus.py

Truncation strategy: head-only, max_length=512 - an accepted, documented
limitation given the measured 55.7% truncation rate (see docs/TOKENIZATION.md).

Reads:
    data/processed/splits/sample_manifest_9class.parquet   (M6)
    data/processed/sources/<rel_path>
    reports/m6_class_split_counts.csv                       (verification only)

Writes:
    data/processed/tokenized_9class/{train,val,test}.parquet

Exits non-zero if the cache's per-class, per-split counts differ from the
committed M6 count table.
"""

from __future__ import annotations

import pandas as pd

from sdp.config import PROJECT_ROOT
from sdp.data.dataset import (
    MANIFEST_9CLASS_PATH,
    SOURCES_DIR,
    TOKENIZED_9CLASS_DIR,
    build_dataloaders,
    build_tokenized_cache,
    write_tokenized_cache,
)
from sdp.data.taxonomy import ID_TO_LEAF, LEAF_ORDER
from sdp.data.tokenization import DEFAULT_MAX_LENGTH, load_tokenizer

M6_COUNTS_CSV = PROJECT_ROOT / "reports" / "m6_class_split_counts.csv"


def main() -> None:
    for required in (MANIFEST_9CLASS_PATH, SOURCES_DIR, M6_COUNTS_CSV):
        if not required.exists():
            raise SystemExit(f"not found: {required}")

    manifest = pd.read_parquet(MANIFEST_9CLASS_PATH)
    print(f"Manifest rows: {len(manifest):,}")

    print("Loading microsoft/codebert-base tokenizer...")
    tokenizer = load_tokenizer()

    print(
        f"Tokenizing {len(manifest):,} files "
        f"(head-only truncation, max_length={DEFAULT_MAX_LENGTH})..."
    )
    cache_df = build_tokenized_cache(tokenizer, manifest)

    paths = write_tokenized_cache(cache_df, TOKENIZED_9CLASS_DIR)
    print("\nWrote per-split tokenized caches:")
    for split, path in paths.items():
        n = (cache_df["split"] == split).sum()
        print(f"  {split:6} {path}  ({n:,} rows)")

    print(f"\npad_token_id: {tokenizer.pad_token_id}")
    loaders = build_dataloaders(TOKENIZED_9CLASS_DIR, pad_token_id=tokenizer.pad_token_id)

    print("\n=== DataLoader sanity check (one batch per split) ===")
    for split, loader in loaders.items():
        batch = next(iter(loader))
        print(
            f"  {split:6} input_ids={tuple(batch['input_ids'].shape)} "
            f"attention_mask={tuple(batch['attention_mask'].shape)} "
            f"labels={batch['labels'].tolist()}"
        )

    print("\n=== Per-split, per-class row counts (9-class) ===")
    named = cache_df.assign(leaf=cache_df["label"].map(lambda i: ID_TO_LEAF[i].value))
    actual = (
        named.groupby(["leaf", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=[c.value for c in LEAF_ORDER],
            columns=["train", "val", "test"],
            fill_value=0,
        )
    )
    print(actual.to_string())
    print(f"total rows: {len(cache_df):,}")

    expected = pd.read_csv(M6_COUNTS_CSV, index_col=0).drop(index="TOTAL", columns="total")
    expected = expected.reindex(index=actual.index, columns=actual.columns)
    ok = bool((actual.to_numpy() == expected.to_numpy()).all())
    print(f"\n[{'PASS' if ok else 'FAIL'}] cache counts match {M6_COUNTS_CSV.name}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
