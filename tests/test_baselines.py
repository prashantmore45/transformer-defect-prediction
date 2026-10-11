"""Tests for sdp.evaluation.baselines."""

import numpy as np
import pytest

from sdp.evaluation.baselines import LengthOnlyBaseline, MajorityBaseline, token_lengths


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
