"""Build the pre-tokenized cache and verify the DataLoader pipeline end-to-end.

Run locally, after `scripts/measure_tokenization.py` has already confirmed the
corpus is reachable (not part of CI — needs the real corpus on disk and
network access to the Hugging Face Hub on first run):

    python scripts/pretokenize_corpus.py

Truncation strategy: head-only, max_length=512 — an accepted, documented
limitation given the measured 55.7% truncation rate (see
docs/TOKENIZATION.md), not something this script tries to solve.

Reads:
    data/processed/splits/sample_manifest_hashed.parquet
    data/processed/sources/<rel_path>

Writes:
    data/processed/tokenized/train.parquet
    data/processed/tokenized/val.parquet
    data/processed/tokenized/test.parquet
"""

from __future__ import annotations

import pandas as pd

from sdp.data.dataset import (
    MANIFEST_PATH,
    SOURCES_DIR,
    TOKENIZED_CACHE_DIR,
    build_dataloaders,
    build_tokenized_cache,
    write_tokenized_cache,
)
from sdp.data.taxonomy import COARSE_ORDER
from sdp.data.tokenization import DEFAULT_MAX_LENGTH, load_tokenizer


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"manifest not found: {MANIFEST_PATH}")
    if not SOURCES_DIR.exists():
        raise SystemExit(f"extracted sources not found: {SOURCES_DIR}")

    manifest = pd.read_parquet(MANIFEST_PATH)
    print(f"Manifest rows: {len(manifest):,}")

    print("Loading microsoft/codebert-base tokenizer...")
    tokenizer = load_tokenizer()

    print(
        f"Tokenizing {len(manifest):,} files "
        f"(head-only truncation, max_length={DEFAULT_MAX_LENGTH})..."
    )
    cache_df = build_tokenized_cache(tokenizer, manifest)

    paths = write_tokenized_cache(cache_df, TOKENIZED_CACHE_DIR)
    print("\nWrote per-split tokenized caches:")
    for split, path in paths.items():
        n = (cache_df["split"] == split).sum()
        print(f"  {split:6} {path}  ({n:,} rows)")

    # --- End-to-end sanity check: build the real DataLoaders and pull one
    # batch from each, so this single script run is the evidence for M3's
    # exit criteria ("a DataLoader for each split produces correctly shaped,
    # correctly labelled batches"), not just the cache files existing.
    print(f"\npad_token_id: {tokenizer.pad_token_id}")
    loaders = build_dataloaders(TOKENIZED_CACHE_DIR, pad_token_id=tokenizer.pad_token_id)

    print("\n=== DataLoader sanity check (one batch per split) ===")
    for split, loader in loaders.items():
        batch = next(iter(loader))
        print(
            f"  {split:6} input_ids={tuple(batch['input_ids'].shape)} "
            f"attention_mask={tuple(batch['attention_mask'].shape)} "
            f"labels={batch['labels'].tolist()}"
        )

    print("\n=== Per-split, per-class row counts (Tier-1) ===")
    counts = cache_df.groupby(["split", "label"]).size().unstack(fill_value=0)
    counts.columns = [COARSE_ORDER[i] for i in counts.columns]
    print(counts.to_string())


if __name__ == "__main__":
    main()
