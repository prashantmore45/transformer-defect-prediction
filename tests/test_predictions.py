"""Tests for sdp.evaluation.predictions (Milestone 8.1).

A tiny random encoder stands in for CodeBERT, so nothing here needs a GPU, the network,
or a trained checkpoint. What is tested is the plumbing that decides whether a dump can
be trusted: row order, padding handling, argmax consistency, and every refusal path.
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaModel

from sdp.data.dataset import SourceCodeDataset, collate_batch
from sdp.data.taxonomy import NUM_LEAVES, TAXONOMY_VERSION
from sdp.evaluation.metrics import LEAF_NAMES
from sdp.evaluation.predictions import (
    DUMP_COLUMNS,
    LOGIT_COLUMNS,
    check_label_contract,
    check_reproduction,
    collect_predictions,
    read_dump,
    require_hash,
    sha256_file,
    validate_against_manifest,
    write_dump,
)
from sdp.model.classifier import CodeBERTClassifier

PAD_ID = 1
LENGTHS = [3, 5, 2, 8, 4, 6, 3]  # 7 examples: a batch size of 3 leaves a partial last batch


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_tiny_model() -> CodeBERTClassifier:
    torch.manual_seed(0)
    config = RobertaConfig(
        vocab_size=50,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,  # RoBERTa positions start at pad_id + 1, so leave headroom
        pad_token_id=PAD_ID,
        type_vocab_size=1,
    )
    model = CodeBERTClassifier(RobertaModel(config, add_pooling_layer=False))
    return model.eval()


def write_cache(path: Path) -> list[list[int]]:
    """Write a 7-example token cache; return the token lists. Real tokens are >= 2."""
    rng = np.random.default_rng(0)
    sequences = [rng.integers(2, 50, size=n).tolist() for n in LENGTHS]
    pd.DataFrame(
        {
            "submission_id": [f"s{i}" for i in range(len(LENGTHS))],
            "input_ids": sequences,
            "label": [i % NUM_LEAVES for i in range(len(LENGTHS))],
        }
    ).to_parquet(path, index=False)
    return sequences


def make_loader(path: Path, batch_size: int) -> DataLoader:
    return DataLoader(
        SourceCodeDataset(path),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_batch, pad_token_id=PAD_ID),
    )


def make_frame(y_true: list[int], y_pred: list[int], ids: list[str] | None = None) -> pd.DataFrame:
    """A syntactically valid dump with zero logits (only labels matter for these tests)."""
    frame = pd.DataFrame(
        {
            "submission_id": ids or [f"s{i}" for i in range(len(y_true))],
            "y_true": np.asarray(y_true, dtype=np.int64),
            "y_pred": np.asarray(y_pred, dtype=np.int64),
            "cached_length": np.full(len(y_true), 5, dtype=np.int64),
        }
    )
    for column in LOGIT_COLUMNS:
        frame[column] = np.zeros(len(y_true), dtype=np.float32)
    return frame[list(DUMP_COLUMNS)]


def make_manifest() -> pd.DataFrame:
    """Seven test rows (labels cycling through classes) plus two validation rows."""
    return pd.DataFrame(
        {
            "submission_id": [f"s{i}" for i in range(7)] + ["v0", "v1"],
            "split": ["test"] * 7 + ["val"] * 2,
            "leaf_id": [i % NUM_LEAVES for i in range(7)] + [0, 1],
        }
    )


def good_meta() -> dict:
    return {
        "label_names": list(LEAF_NAMES),
        "num_labels": NUM_LEAVES,
        "taxonomy_version": TAXONOMY_VERSION,
    }


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_require_hash_accepts_uppercase_expected(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    expected = "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
    assert require_hash(path, expected, "thing") == sha256_file(path)


def test_require_hash_rejects_wrong_hash(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="hash mismatch"):
        require_hash(path, "0" * 64, "thing")


# --------------------------------------------------------------------------- #
# Checkpoint label contract
# --------------------------------------------------------------------------- #


def test_label_contract_accepts_matching_meta() -> None:
    check_label_contract(good_meta())  # must not raise


def test_label_contract_rejects_reordered_labels() -> None:
    meta = good_meta()
    meta["label_names"] = list(reversed(meta["label_names"]))
    with pytest.raises(ValueError, match="label order"):
        check_label_contract(meta)


def test_label_contract_rejects_wrong_label_count() -> None:
    meta = good_meta()
    meta["num_labels"] = NUM_LEAVES - 1
    with pytest.raises(ValueError, match="labels"):
        check_label_contract(meta)


def test_label_contract_rejects_wrong_taxonomy_version() -> None:
    meta = good_meta()
    meta["taxonomy_version"] = "0.0-not-real"
    with pytest.raises(ValueError, match="taxonomy version"):
        check_label_contract(meta)


# --------------------------------------------------------------------------- #
# collect_predictions
# --------------------------------------------------------------------------- #


def test_collect_predictions_columns_order_and_argmax(tmp_path: Path) -> None:
    path = tmp_path / "cache.parquet"
    write_cache(path)
    frame = collect_predictions(make_tiny_model(), make_loader(path, 3), torch.device("cpu"), None)

    assert list(frame.columns) == list(DUMP_COLUMNS)
    assert frame["submission_id"].tolist() == [f"s{i}" for i in range(len(LENGTHS))]
    assert frame["y_true"].tolist() == [i % NUM_LEAVES for i in range(len(LENGTHS))]
    assert all(frame[column].dtype == np.float32 for column in LOGIT_COLUMNS)
    # y_pred must be exactly the argmax of the stored logits, in class-ID order.
    assert np.array_equal(
        frame["y_pred"].to_numpy(), frame[list(LOGIT_COLUMNS)].to_numpy().argmax(1)
    )


def test_collect_predictions_counts_real_tokens_only(tmp_path: Path) -> None:
    path = tmp_path / "cache.parquet"
    write_cache(path)
    frame = collect_predictions(make_tiny_model(), make_loader(path, 3), torch.device("cpu"), None)
    # Padding must not be counted: lengths equal the original sequence lengths.
    assert frame["cached_length"].tolist() == LENGTHS


def test_collect_predictions_matches_one_by_one_forward(tmp_path: Path) -> None:
    path = tmp_path / "cache.parquet"
    sequences = write_cache(path)
    model = make_tiny_model()
    frame = collect_predictions(model, make_loader(path, 3), torch.device("cpu"), None)

    with torch.no_grad():
        direct = np.stack(
            [
                model(
                    torch.tensor([ids], dtype=torch.long),
                    torch.ones(1, len(ids), dtype=torch.long),
                )[0].numpy()
                for ids in sequences
            ]
        )
    assert np.allclose(frame[list(LOGIT_COLUMNS)].to_numpy(), direct, atol=1e-4)


def test_collect_predictions_independent_of_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "cache.parquet"
    write_cache(path)
    model = make_tiny_model()
    cpu = torch.device("cpu")
    one = collect_predictions(model, make_loader(path, 1), cpu, None)
    seven = collect_predictions(model, make_loader(path, 7), cpu, None)
    # Padding changes batch composition; it must not change a file's scores.
    assert np.allclose(
        one[list(LOGIT_COLUMNS)].to_numpy(), seven[list(LOGIT_COLUMNS)].to_numpy(), atol=1e-4
    )
    assert one["y_pred"].tolist() == seven["y_pred"].tolist()


# --------------------------------------------------------------------------- #
# validate_against_manifest
# --------------------------------------------------------------------------- #


def good_test_frame() -> pd.DataFrame:
    labels = [i % NUM_LEAVES for i in range(7)]
    return make_frame(labels, labels)


def test_validate_accepts_matching_dump_and_ignores_other_splits() -> None:
    validate_against_manifest(good_test_frame(), make_manifest(), "test")  # must not raise


def test_validate_rejects_wrong_row_count() -> None:
    with pytest.raises(ValueError, match="rows"):
        validate_against_manifest(good_test_frame().iloc[:-1], make_manifest(), "test")


def test_validate_rejects_duplicate_ids() -> None:
    frame = good_test_frame()
    frame.loc[6, "submission_id"] = "s0"
    with pytest.raises(ValueError, match="duplicate"):
        validate_against_manifest(frame, make_manifest(), "test")


def test_validate_rejects_unknown_ids() -> None:
    frame = good_test_frame()
    frame.loc[6, "submission_id"] = "not-in-manifest"
    with pytest.raises(ValueError, match="differ"):
        validate_against_manifest(frame, make_manifest(), "test")


def test_validate_rejects_changed_true_label() -> None:
    frame = good_test_frame()
    frame.loc[3, "y_true"] = (int(frame.loc[3, "y_true"]) + 1) % NUM_LEAVES
    with pytest.raises(ValueError, match="y_true"):
        validate_against_manifest(frame, make_manifest(), "test")


def test_validate_rejects_split_with_no_rows() -> None:
    with pytest.raises(ValueError, match="no rows"):
        validate_against_manifest(good_test_frame(), make_manifest(), "holdout")


# --------------------------------------------------------------------------- #
# check_reproduction
# --------------------------------------------------------------------------- #


def perfect_frame() -> pd.DataFrame:
    labels = list(range(NUM_LEAVES)) * 2  # every class present, so macro-F1 is exactly 1.0
    return make_frame(labels, labels)


def test_reproduction_passes_within_tolerance() -> None:
    result = check_reproduction(perfect_frame(), expected_macro_f1=0.999, tolerance=0.002)
    assert result["passed"] is True
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)


def test_reproduction_fails_outside_tolerance() -> None:
    result = check_reproduction(perfect_frame(), expected_macro_f1=0.9, tolerance=0.002)
    assert result["passed"] is False
    assert result["abs_difference"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# write_dump / read_dump
# --------------------------------------------------------------------------- #


def test_write_then_read_round_trip_leaves_no_temp_file(tmp_path: Path) -> None:
    frame = perfect_frame()
    path = tmp_path / "out" / "dump.parquet"
    write_dump(frame, path)
    back = read_dump(path)
    pd.testing.assert_frame_equal(back, frame)
    assert [p.name for p in path.parent.iterdir()] == ["dump.parquet"]


def test_write_dump_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "dump.parquet"
    write_dump(perfect_frame(), path)
    with pytest.raises(FileExistsError):
        write_dump(perfect_frame(), path)


def test_write_dump_rejects_wrong_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="columns"):
        write_dump(perfect_frame().drop(columns=["cached_length"]), tmp_path / "dump.parquet")
    assert not (tmp_path / "dump.parquet").exists()


def test_read_dump_rejects_unexpected_columns(tmp_path: Path) -> None:
    path = tmp_path / "other.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="unexpected columns"):
        read_dump(path)
