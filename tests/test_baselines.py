"""Tests for sdp.evaluation.baselines."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdp.data.taxonomy import NUM_LEAVES
from sdp.evaluation.baselines import (
    BASELINE_COLUMNS,
    LengthOnlyBaseline,
    MajorityBaseline,
    StratifiedRandomBaseline,
    TfidfLogisticBaseline,
    baseline_frame,
    read_baseline_dump,
    token_documents,
    token_lengths,
    write_baseline_dump,
)
from sdp.evaluation.metrics import macro_f1
from sdp.evaluation.predictions import validate_against_manifest

K = NUM_LEAVES
COUNTS = [10, 20, 100, 1, 5, 2, 3, 4, 30]  # class 2 is the most common; total 175
LABELS = np.repeat(np.arange(K), COUNTS)


def test_token_lengths_counts_each_sequence():
    lengths = token_lengths([[1, 2, 3], np.array([7]), (4, 5)])
    assert lengths.tolist() == [3, 1, 2]
    assert lengths.dtype == np.int64


def test_majority_predicts_most_common_training_class():
    model = MajorityBaseline().fit([0, 1, 1, 2])
    assert model.predict(np.array([10, 20, 30])).tolist() == [1, 1, 1]


def test_majority_requires_fit():
    with pytest.raises(RuntimeError):
        MajorityBaseline().predict(np.array([1, 2]))


def test_length_only_learns_a_clean_length_split():
    lengths = np.r_[np.arange(10, 60), np.arange(400, 450)]
    labels = np.r_[np.zeros(50, dtype=int), np.ones(50, dtype=int)]
    model = LengthOnlyBaseline(min_samples_leaf=1).fit(lengths, labels)
    assert model.predict(np.array([12, 405])).tolist() == [0, 1]


def test_length_only_cannot_separate_classes_when_length_is_constant():
    lengths = np.full(40, 512)
    labels = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    model = LengthOnlyBaseline(min_samples_leaf=1).fit(lengths, labels)
    assert len(set(model.predict(np.array([512, 512, 512])).tolist())) == 1


def test_length_only_requires_fit():
    with pytest.raises(RuntimeError):
        LengthOnlyBaseline().predict(np.array([1, 2]))


def test_length_only_rejects_mismatched_fit_inputs():
    with pytest.raises(ValueError):
        LengthOnlyBaseline().fit(np.array([1, 2, 3]), np.array([0, 1]))


# --------------------------------------------------------------------------- #
# StratifiedRandomBaseline (M8.3)
# --------------------------------------------------------------------------- #


def test_stratified_random_is_reproducible_and_seed_sensitive():
    first = StratifiedRandomBaseline(seed=1).fit(LABELS).predict(np.zeros(200))
    again = StratifiedRandomBaseline(seed=1).fit(LABELS).predict(np.zeros(200))
    other = StratifiedRandomBaseline(seed=2).fit(LABELS).predict(np.zeros(200))
    assert first.dtype == np.int64
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_stratified_random_repeated_predict_calls_agree():
    model = StratifiedRandomBaseline(seed=5).fit(LABELS)
    assert np.array_equal(model.predict(np.zeros(50)), model.predict(np.zeros(50)))


def test_stratified_random_follows_the_training_frequencies():
    predictions = StratifiedRandomBaseline(seed=0).fit(LABELS).predict(np.zeros(20000))
    assert abs((predictions == 2).mean() - 100 / 175) < 0.02


def test_stratified_random_never_predicts_a_class_with_zero_training_count():
    labels = np.repeat(np.arange(K), [10, 20, 100, 0, 5, 2, 3, 4, 30])
    predictions = StratifiedRandomBaseline(seed=0).fit(labels).predict(np.zeros(5000))
    assert 3 not in predictions


def test_stratified_random_requires_fit():
    with pytest.raises(RuntimeError):
        StratifiedRandomBaseline().predict(np.zeros(3))


def test_stratified_random_rejects_bad_labels():
    with pytest.raises(ValueError):
        StratifiedRandomBaseline().fit([])
    with pytest.raises(ValueError):
        StratifiedRandomBaseline().fit([0, K])
    with pytest.raises(TypeError):
        StratifiedRandomBaseline().fit([0.0, 1.0])


def test_chance_level_macro_f1_is_about_one_ninth():
    # If predictions are independent of the truth and follow the class frequencies, each
    # class's F1 equals its share, so macro-F1 is about 1/9 whatever the imbalance.
    rng = np.random.default_rng(0)
    y_true = rng.choice(K, size=100000, p=np.array(COUNTS) / sum(COUNTS))
    y_pred = StratifiedRandomBaseline(seed=3).fit(LABELS).predict(y_true)
    assert abs(macro_f1(y_true, y_pred) - 1 / 9) < 0.01


# --------------------------------------------------------------------------- #
# TfidfLogisticBaseline (M8.3)
# --------------------------------------------------------------------------- #


def make_corpus(n_per_class, seed):
    """Token-ID lists whose only reliable signal is a class-specific token (100, 101, 102)."""
    rng = np.random.default_rng(seed)
    documents, labels = [], []
    for label in (0, 1, 2):
        for _ in range(n_per_class):
            noise = rng.integers(3, 20, size=15).tolist()
            documents.append([0, *noise, 100 + label, 100 + label, 2])
            labels.append(label)
    return token_documents(documents), np.array(labels)


def test_token_documents_join_ids_with_spaces():
    assert token_documents([np.array([0, 5, 2]), [0, 7]]) == ["0 5 2", "0 7"]


def test_tfidf_logistic_recovers_the_signal_token():
    train_docs, y_train = make_corpus(30, seed=1)
    val_docs, y_val = make_corpus(10, seed=2)
    test_docs, y_test = make_corpus(10, seed=3)
    model = TfidfLogisticBaseline(c_grid=(0.5, 5.0), min_df=1)
    model.fit(train_docs, y_train, val_docs, y_val)
    assert np.array_equal(model.predict(test_docs), y_test)
    assert model.predict(test_docs).dtype == np.int64
    assert set(model.validation_scores) == {0.5, 5.0}
    assert model.best_c in model.validation_scores
    assert model.n_features > 0
    assert model.converged
    assert set(model.n_iter_by_c) == {0.5, 5.0}
    assert all(count >= 1 for count in model.n_iter_by_c.values())


def test_a_tie_between_c_values_keeps_the_earliest():
    train_docs, y_train = make_corpus(30, seed=1)
    val_docs, y_val = make_corpus(10, seed=2)
    model = TfidfLogisticBaseline(c_grid=(0.5, 5.0), min_df=1)
    model.fit(train_docs, y_train, val_docs, y_val)
    # Perfectly separable data: both values score the same on validation.
    assert model.validation_scores[0.5] == model.validation_scores[5.0]
    assert model.best_c == 0.5


def test_tfidf_logistic_requires_fit():
    with pytest.raises(RuntimeError):
        TfidfLogisticBaseline().predict(["0 1 2"])
    with pytest.raises(RuntimeError):
        _ = TfidfLogisticBaseline().n_features


def test_tfidf_logistic_rejects_an_empty_grid():
    with pytest.raises(ValueError, match="grid"):
        TfidfLogisticBaseline(c_grid=())


def test_tfidf_logistic_rejects_bad_inputs():
    docs, labels = make_corpus(5, seed=0)
    model = TfidfLogisticBaseline(c_grid=(1.0,), min_df=1)
    with pytest.raises(ValueError, match="one label per document"):
        model.fit(docs, labels[:-1], docs, labels)
    with pytest.raises(ValueError):
        model.fit(docs, labels, docs, np.full(len(docs), K))


# --------------------------------------------------------------------------- #
# Baseline dumps (M8.3)
# --------------------------------------------------------------------------- #


def good_frame():
    labels = [i % K for i in range(7)]
    return baseline_frame([f"s{i}" for i in range(7)], labels, labels[::-1])


def test_baseline_frame_has_the_three_columns_and_int64_labels():
    frame = good_frame()
    assert list(frame.columns) == list(BASELINE_COLUMNS)
    assert frame["y_true"].dtype == np.int64
    assert frame["y_pred"].dtype == np.int64


def test_baseline_frame_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        baseline_frame(["a", "b"], [0, 1], [0])


def test_baseline_dump_round_trip_leaves_no_temp_file(tmp_path: Path):
    path = tmp_path / "out" / "baseline.parquet"
    write_baseline_dump(good_frame(), path)
    pd.testing.assert_frame_equal(read_baseline_dump(path), good_frame())
    assert [p.name for p in path.parent.iterdir()] == ["baseline.parquet"]


def test_baseline_dump_refuses_to_overwrite(tmp_path: Path):
    path = tmp_path / "baseline.parquet"
    write_baseline_dump(good_frame(), path)
    with pytest.raises(FileExistsError):
        write_baseline_dump(good_frame(), path)


def test_baseline_dump_rejects_wrong_columns(tmp_path: Path):
    with pytest.raises(ValueError, match="columns"):
        write_baseline_dump(good_frame().drop(columns=["y_pred"]), tmp_path / "b.parquet")
    other = tmp_path / "other.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(other, index=False)
    with pytest.raises(ValueError, match="unexpected columns"):
        read_baseline_dump(other)


def test_the_m8_1_manifest_check_accepts_a_baseline_dump():
    labels = [i % K for i in range(7)]
    frame = baseline_frame([f"s{i}" for i in range(7)], labels, [0] * 7)
    manifest = pd.DataFrame(
        {
            "submission_id": [f"s{i}" for i in range(7)] + ["v0"],
            "split": ["test"] * 7 + ["val"],
            "leaf_id": [*labels, 0],
        }
    )
    validate_against_manifest(frame, manifest, "test")  # must not raise
