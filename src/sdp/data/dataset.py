"""PyTorch Dataset/DataLoader construction for CodeBERT fine-tuning.

Milestones 3 and 6
------------------
Turns a frozen manifest plus the extracted source files
(`data/processed/sources/`) into three `DataLoader`s — train/val/test — that
respect the M2-frozen split exactly. No re-splitting happens anywhere in this
module; `split` membership is read, never recomputed.

M3 built this pipeline for the 74,850-row Tier-1 manifest
(`sample_manifest_hashed.parquet`, 4 coarse classes). As of M6 it reads the
46,594-row 9-class manifest (`sample_manifest_9class.parquet`) and emits the
9-class leaf IDs from that manifest's `leaf_id` column
(`sdp.data.taxonomy.leaf_id`). This module no longer maps label names itself:
the manifest already carries the integer IDs, computed once from the
taxonomy, so there is no second mapping to keep in sync.

Truncation strategy: head-only, `max_length=512` — decided 2026-09-25 with
the real measured 55.7% truncation rate in hand (see
`docs/TOKENIZATION.md`), not assumed upfront. This is an accepted,
documented limitation, not something this module tries to solve.

Why pre-tokenize instead of tokenizing in `__getitem__`
--------------------------------------------------------
The tokenized ids for a given file never change across training epochs —
only the sampling order does. Re-running BPE tokenization on tens of
thousands of files on every epoch (or, worse, on every `__getitem__` call)
repeats work whose result is already fixed. `scripts/pretokenize_corpus.py`
tokenizes once and caches the result per split; `SourceCodeDataset` only ever
reads that cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from sdp.config import PROJECT_ROOT
from sdp.data.taxonomy import NUM_LEAVES
from sdp.data.tokenization import DEFAULT_MAX_LENGTH, Tokenizer, _decode, head_truncate

SOURCES_DIR = PROJECT_ROOT / "data" / "processed" / "sources"

#: M6 artifacts: the 9-class manifest and its token cache.
MANIFEST_9CLASS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "splits" / "sample_manifest_9class.parquet"
)
TOKENIZED_9CLASS_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized_9class"

#: Manifest column holding the integer class ID (taxonomy.leaf_id).
LABEL_COLUMN = "leaf_id"

# Legacy M3 artifacts (4-class, Tier-1 labels). Kept for reference only; the
# pipeline no longer uses them as of M6.
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "sample_manifest_hashed.parquet"
TOKENIZED_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized"

#: The three frozen M2 splits, in a fixed order — never recomputed here.
SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class Example:
    """One tokenized, head-truncated, integer-labelled training example."""

    submission_id: str
    input_ids: list[int]
    label: int


def tokenize_example(
    tokenizer: Tokenizer,
    submission_id: str,
    source: str,
    label: int,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Example:
    """Tokenize one file's source and apply the chosen head-only truncation.

    `label` is a ready-made integer class ID; this function deliberately knows
    nothing about any taxonomy, so the same code serves any label scheme.

    Reuses `sdp.data.tokenization.head_truncate` — the same tested function
    `scripts/measure_tokenization.py` measured truncation rates against —
    rather than a tokenizer library's own `truncation=True` kwarg, so there
    is exactly one place in the codebase that decides what "truncated"
    means.
    """
    full_ids = tokenizer.encode(source, add_special_tokens=True)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    input_ids = head_truncate(full_ids, max_length=max_length, sep_token_id=sep_id)
    return Example(submission_id=submission_id, input_ids=input_ids, label=int(label))


def _validate_label_column(manifest: pd.DataFrame) -> None:
    """Fail fast, before the slow tokenization loop, on a bad label column."""
    if LABEL_COLUMN not in manifest.columns:
        raise KeyError(f"manifest is missing the '{LABEL_COLUMN}' column")
    labels = manifest[LABEL_COLUMN]
    if not pd.api.types.is_integer_dtype(labels):
        raise TypeError(f"'{LABEL_COLUMN}' must be an integer column, got {labels.dtype}")
    if len(labels) and (labels.min() < 0 or labels.max() >= NUM_LEAVES):
        raise ValueError(
            f"'{LABEL_COLUMN}' values must be in [0, {NUM_LEAVES}); "
            f"found min={labels.min()}, max={labels.max()}"
        )


def build_tokenized_cache(
    tokenizer: Tokenizer,
    manifest: pd.DataFrame,
    sources_dir: Path = SOURCES_DIR,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> pd.DataFrame:
    """Tokenize every row in `manifest`, one row per file in the result.

    `manifest` must have `submission_id`, `rel_path`, `leaf_id`, `split`
    columns - the M6 9-class manifest schema. Reads raw bytes and decodes via
    the same fallback chain (`sdp.data.tokenization._decode`) used for the M1
    measurement pass, so a single non-UTF-8 file can't abort the whole run.
    """
    _validate_label_column(manifest)
    rows = []
    for row in manifest.itertuples():
        path = sources_dir / row.rel_path
        source = _decode(path.read_bytes())
        example = tokenize_example(
            tokenizer, row.submission_id, source, getattr(row, LABEL_COLUMN), max_length
        )
        rows.append(
            {
                "submission_id": example.submission_id,
                "input_ids": example.input_ids,
                "label": example.label,
                "split": row.split,
            }
        )
    return pd.DataFrame(rows)


def write_tokenized_cache(
    cache_df: pd.DataFrame, out_dir: Path = TOKENIZED_9CLASS_DIR
) -> dict[str, Path]:
    """Split `cache_df` by its `split` column and write one parquet per split.

    `split` is dropped from each output file — it's redundant once the file
    itself is scoped to one split, and keeping it would let a bug silently
    mix splits back together downstream without anything failing loudly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, group in cache_df.groupby("split", observed=True):
        path = out_dir / f"{split}.parquet"
        group.drop(columns=["split"]).to_parquet(path, index=False)
        paths[str(split)] = path
    return paths


class SourceCodeDataset(Dataset):
    """One split's pre-tokenized examples, read from a cache parquet.

    Reads `{submission_id, input_ids, label}` written by
    `build_tokenized_cache` / `write_tokenized_cache`. Does no tokenization
    itself — see the module docstring for why that work happens once,
    offline, instead of here.
    """

    _REQUIRED_COLUMNS = {"submission_id", "input_ids", "label"}

    def __init__(self, cache_path: Path):
        self._df = pd.read_parquet(cache_path)
        missing = self._REQUIRED_COLUMNS - set(self._df.columns)
        if missing:
            raise KeyError(f"tokenized cache {cache_path} missing columns: {missing}")

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self._df.iloc[idx]
        return {
            "submission_id": row["submission_id"],
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


def collate_batch(batch: Sequence[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Pad every sequence in `batch` to the batch's own longest sequence.

    Not to a fixed 512: about half of the files already hit the 512 cap after
    truncation (49.6% of the 9-class corpus, see `docs/ASSEMBLY.md`), so many
    batches will need close-to-full-length padding anyway — but for the
    remaining ~50% of files (all shorter than 512, some far shorter), padding
    to the batch max instead of always to 512 still avoids wasted compute. It
    costs nothing extra over fixed padding to implement, so there's no reason
    not to.

    Builds the attention mask here (1 = real token, 0 = padding) since it
    depends on the batch's chosen pad length, not on anything stored
    per-example.
    """
    lengths = [ex["input_ids"].shape[0] for ex in batch]
    max_len = max(lengths)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    submission_ids = []

    for i, ex in enumerate(batch):
        n = lengths[i]
        input_ids[i, :n] = ex["input_ids"]
        attention_mask[i, :n] = 1
        labels[i] = ex["label"]
        submission_ids.append(ex["submission_id"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "submission_ids": submission_ids,
    }


def build_dataloaders(
    cache_dir: Path,
    pad_token_id: int,
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Wire the three frozen splits into `DataLoader`s.

    `train` shuffles per epoch; `val`/`test` do not — evaluation must be
    reproducible run-to-run, and shuffling them would only make debugging
    harder for no benefit. No split is recomputed or rebalanced here: which
    file belongs to which split was decided once, in M2, and is frozen.
    """
    loaders: dict[str, DataLoader] = {}
    for split in SPLITS:
        cache_path = cache_dir / f"{split}.parquet"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"tokenized cache not found: {cache_path} "
                "— run scripts/pretokenize_corpus.py first"
            )
        dataset = SourceCodeDataset(cache_path)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=partial(collate_batch, pad_token_id=pad_token_id),
        )
    return loaders
