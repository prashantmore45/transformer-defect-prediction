"""Tests for `sdp.data.dataset`.

Network-free (stub tokenizer, no Hugging Face download) but does require
`torch` — unlike `test_tokenization.py`, this suite is not runnable in an
environment without torch installed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from sdp.data.dataset import (
    SourceCodeDataset,
    build_dataloaders,
    build_tokenized_cache,
    collate_batch,
    tokenize_example,
    write_tokenized_cache,
)
from sdp.data.taxonomy import NUM_LEAVES, leaf_id


class StubTokenizer:
    """Whitespace tokenizer with the pad/sep ids `dataset.py` needs.

    Same shape as `test_tokenization.py`'s stub, extended with the
    `sep_token_id`/`pad_token_id` attributes a real HF tokenizer exposes and
    that `tokenize_example`/`collate_batch` rely on.
    """

    CLS_ID = 0
    SEP_ID = 1
    PAD_ID = 2

    sep_token_id = SEP_ID
    pad_token_id = PAD_ID

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [i + 3 for i in range(len(text.split()))]
        if add_special_tokens:
            return [self.CLS_ID, *ids, self.SEP_ID]
        return ids


@pytest.fixture
def tokenizer() -> StubTokenizer:
    return StubTokenizer()


def _write_corpus(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    """A tiny 5-file synthetic corpus spanning all three splits."""
    sources = tmp_path / "sources"
    sources.mkdir()
    rows = [
        ("s1", "ERROR_FREE", "s1.cpp", "int main() { return 0; }", "train"),
        ("s2", "LOGICAL", "s2.cpp", "int main() { return 1; }", "train"),
        ("s3", "SYNTAX", "s3.cpp", "int main() { return", "val"),
        ("s4", "SIGSEGV", "s4.cpp", "int main() { int *p=0; return *p; }", "val"),
        ("s5", "ERROR_FREE", "s5.cpp", "int main() { return 0; }", "test"),
    ]
    manifest_rows = []
    for submission_id, leaf_label, filename, code, split in rows:
        (sources / filename).write_text(code)
        manifest_rows.append(
            {
                "submission_id": submission_id,
                "leaf_id": leaf_id(leaf_label),
                "rel_path": filename,
                "split": split,
            }
        )
    return pd.DataFrame(manifest_rows), sources


def test_tokenize_example_keeps_the_given_integer_label(tokenizer: StubTokenizer) -> None:
    example = tokenize_example(tokenizer, "s1", "int main() { return 0; }", 4)
    assert example.submission_id == "s1"
    assert example.input_ids[0] == StubTokenizer.CLS_ID
    assert example.input_ids[-1] == StubTokenizer.SEP_ID
    assert example.label == 4
    assert isinstance(example.label, int)


def test_tokenize_example_applies_head_truncation(tokenizer: StubTokenizer) -> None:
    long_code = " ".join(f"tok{i}" for i in range(1000))
    example = tokenize_example(tokenizer, "s1", long_code, 0, max_length=10)
    assert len(example.input_ids) == 10
    assert example.input_ids[-1] == StubTokenizer.SEP_ID  # forced SEP on cut


def test_build_tokenized_cache_one_row_per_file(tmp_path: Path, tokenizer: StubTokenizer) -> None:
    manifest, sources = _write_corpus(tmp_path)
    cache_df = build_tokenized_cache(tokenizer, manifest, sources_dir=sources)

    assert len(cache_df) == 5
    assert set(cache_df["submission_id"]) == {"s1", "s2", "s3", "s4", "s5"}
    assert set(cache_df.columns) >= {"submission_id", "input_ids", "label", "split"}


def test_write_tokenized_cache_splits_by_split_column(
    tmp_path: Path, tokenizer: StubTokenizer
) -> None:
    manifest, sources = _write_corpus(tmp_path)
    cache_df = build_tokenized_cache(tokenizer, manifest, sources_dir=sources)
    paths = write_tokenized_cache(cache_df, tmp_path / "tokenized")

    assert set(paths) == {"train", "val", "test"}
    train_df = pd.read_parquet(paths["train"])
    assert len(train_df) == 2
    assert "split" not in train_df.columns  # dropped — redundant per-split


def test_source_code_dataset_returns_long_tensors(tmp_path: Path, tokenizer: StubTokenizer) -> None:
    manifest, sources = _write_corpus(tmp_path)
    cache_df = build_tokenized_cache(tokenizer, manifest, sources_dir=sources)
    paths = write_tokenized_cache(cache_df, tmp_path / "tokenized")

    dataset = SourceCodeDataset(paths["train"])
    assert len(dataset) == 2
    item = dataset[0]
    assert isinstance(item["input_ids"], torch.Tensor)
    assert item["input_ids"].dtype == torch.long
    assert isinstance(item["label"], torch.Tensor)
    assert item["label"].dtype == torch.long


def test_source_code_dataset_requires_expected_columns(tmp_path: Path) -> None:
    bad_cache = tmp_path / "bad.parquet"
    pd.DataFrame({"submission_id": ["s1"]}).to_parquet(bad_cache)
    with pytest.raises(KeyError):
        SourceCodeDataset(bad_cache)


def test_collate_batch_pads_to_batch_max_not_fixed_512(tokenizer: StubTokenizer) -> None:
    batch = [
        {"submission_id": "s1", "input_ids": torch.tensor([0, 5, 1]), "label": torch.tensor(0)},
        {
            "submission_id": "s2",
            "input_ids": torch.tensor([0, 5, 6, 7, 1]),
            "label": torch.tensor(1),
        },
    ]
    out = collate_batch(batch, pad_token_id=tokenizer.pad_token_id)

    assert out["input_ids"].shape == (2, 5)
    assert out["attention_mask"].shape == (2, 5)
    assert out["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert out["input_ids"][0, 3:].tolist() == [tokenizer.pad_token_id, tokenizer.pad_token_id]
    assert out["labels"].tolist() == [0, 1]
    assert out["submission_ids"] == ["s1", "s2"]


def test_build_dataloaders_respects_frozen_split(tmp_path: Path, tokenizer: StubTokenizer) -> None:
    manifest, sources = _write_corpus(tmp_path)
    cache_df = build_tokenized_cache(tokenizer, manifest, sources_dir=sources)
    cache_dir = tmp_path / "tokenized"
    write_tokenized_cache(cache_df, cache_dir)

    loaders = build_dataloaders(cache_dir, pad_token_id=tokenizer.pad_token_id, batch_size=2)

    assert set(loaders) == {"train", "val", "test"}

    all_train_ids: set[str] = set()
    for batch in loaders["train"]:
        all_train_ids.update(batch["submission_ids"])
    assert all_train_ids == {"s1", "s2"}

    all_test_ids: set[str] = set()
    for batch in loaders["test"]:
        all_test_ids.update(batch["submission_ids"])
    assert all_test_ids == {"s5"}


def test_build_dataloaders_missing_cache_raises(tmp_path: Path, tokenizer: StubTokenizer) -> None:
    with pytest.raises(FileNotFoundError):
        build_dataloaders(tmp_path / "does_not_exist", pad_token_id=tokenizer.pad_token_id)


def test_build_tokenized_cache_emits_leaf_ids(tmp_path: Path, tokenizer: StubTokenizer) -> None:
    manifest, sources = _write_corpus(tmp_path)
    cache_df = build_tokenized_cache(tokenizer, manifest, sources_dir=sources)
    labels = dict(zip(cache_df["submission_id"], cache_df["label"]))
    assert labels == {
        "s1": leaf_id("ERROR_FREE"),
        "s2": leaf_id("LOGICAL"),
        "s3": leaf_id("SYNTAX"),
        "s4": leaf_id("SIGSEGV"),
        "s5": leaf_id("ERROR_FREE"),
    }


def test_build_tokenized_cache_rejects_out_of_range_labels(
    tmp_path: Path, tokenizer: StubTokenizer
) -> None:
    manifest, sources = _write_corpus(tmp_path)
    manifest.loc[0, "leaf_id"] = NUM_LEAVES
    with pytest.raises(ValueError):
        build_tokenized_cache(tokenizer, manifest, sources_dir=sources)


def test_build_tokenized_cache_requires_leaf_id_column(
    tmp_path: Path, tokenizer: StubTokenizer
) -> None:
    manifest, sources = _write_corpus(tmp_path)
    with pytest.raises(KeyError):
        build_tokenized_cache(tokenizer, manifest.drop(columns="leaf_id"), sources_dir=sources)
