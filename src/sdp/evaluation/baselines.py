"""Reference baselines every CodeBERT result must be compared against.

Milestone 7.1
-------------
Two deliberately weak models:

* ``MajorityBaseline``: always predicts the most common training class. The
  floor: any real model must beat it.
* ``LengthOnlyBaseline``: sees ONLY the capped token length,
  ``min(token_length, 512)``, and nothing about the code's content.

Why the length-only baseline exists
-----------------------------------
Per-class median token lengths differ a lot (docs/ASSEMBLY.md: ERROR_FREE 677,
LOGICAL 641 vs. NZEC 221, SIGFPE 238), so length alone carries class signal that
has nothing to do with defects. CodeBERT can pick up the same shortcut. Its
result is only meaningful relative to this baseline, not relative to majority.

Why the CAPPED length: the cache is head-truncated at 512 tokens, so the model
cannot see a longer length either. The baseline is given exactly the length
information the model has.

Class weights are 'balanced' (full inverse frequency) so the baseline gets its
best shot at macro-F1; that makes it a conservative reference. This setting is
independent of whatever weighting the CodeBERT run uses.

Milestone 8.3 additions
-----------------------
Extends the module for the test-split comparison; nothing above this line changed.

* ``StratifiedRandomBaseline``: draws predictions at the training class frequencies,
  ignoring the input. If predictions are independent of the truth and follow the true
  class frequencies, each class's F1 equals that class's share of the data, so macro-F1
  is about 1/9 = 0.111 whatever the imbalance. That, not the majority baseline's much
  lower value, is the chance level for macro-F1.
* ``TfidfLogisticBaseline``: counts of token IDs and adjacent token pairs, weighted by how
  rare they are across files (TF-IDF), fed to a multinomial logistic regression. It reads
  the same cached 512-token head-truncated IDs that CodeBERT saw, and answers "does a
  transformer beat a bag of words?".
* Baseline dumps: three columns (``submission_id``, ``y_true``, ``y_pred``). The majority
  and random baselines have no meaningful scores, so padding the M8.1 schema with fake
  logits would be misleading. ``sdp.evaluation.predictions.validate_against_manifest``
  accepts these dumps.

Fairness rules: everything is fitted on the training split only (the TF-IDF vocabulary
included); the only tuned setting is the regularisation strength C, chosen on VALIDATION
macro-F1 from a small fixed grid, ties going to the earliest (most regularised) value;
``class_weight='balanced'`` follows the M7.1 convention and is a stronger re-weighting
than the square-root weights used to train CodeBERT, so a CodeBERT win is conservative.
Test predictions are written once and never overwritten.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sdp.data.taxonomy import NUM_LEAVES
from sdp.evaluation.metrics import macro_f1

BASELINE_COLUMNS: tuple[str, ...] = ("submission_id", "y_true", "y_pred")

#: Regularisation strengths tried for logistic regression (smaller C = stronger penalty).
DEFAULT_C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 300_000
LOGREG_MAX_ITER = 1000


def token_lengths(input_ids: Iterable[Sequence[int]]) -> np.ndarray:
    """Length of each token sequence (the cache is already capped at 512)."""
    return np.fromiter((len(ids) for ids in input_ids), dtype=np.int64)


class MajorityBaseline:
    """Predict the most frequent class seen in training (ties: smallest ID)."""

    def __init__(self) -> None:
        self._majority: int | None = None

    def fit(self, labels: object) -> MajorityBaseline:
        arr = np.asarray(labels)
        if arr.size == 0:
            raise ValueError("labels is empty")
        self._majority = int(np.bincount(arr).argmax())
        return self

    def predict(self, lengths: object) -> np.ndarray:
        if self._majority is None:
            raise RuntimeError("fit() must be called before predict()")
        return np.full(len(np.asarray(lengths)), self._majority, dtype=np.int64)


class LengthOnlyBaseline:
    """A shallow decision tree over one feature: the capped token length."""

    def __init__(
        self, max_depth: int = 5, min_samples_leaf: int = 50, random_state: int = 42
    ) -> None:
        self._tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
        )
        self._fitted = False

    @staticmethod
    def _as_feature(lengths: object) -> np.ndarray:
        return np.asarray(lengths, dtype=np.float64).reshape(-1, 1)

    def fit(self, lengths: object, labels: object) -> LengthOnlyBaseline:
        x = self._as_feature(lengths)
        y = np.asarray(labels)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"lengths and labels must have the same length, got {x.shape[0]} and {y.shape[0]}"
            )
        self._tree.fit(x, y)
        self._fitted = True
        return self

    def predict(self, lengths: object) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit() must be called before predict()")
        return self._tree.predict(self._as_feature(lengths)).astype(np.int64)


# --------------------------------------------------------------------------- #
# Milestone 8.3 additions
# --------------------------------------------------------------------------- #


def _check_labels(labels: object) -> np.ndarray:
    """Validate integer class IDs in [0, NUM_LEAVES) and return them as int64."""
    array = np.asarray(labels)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("labels must be a non-empty 1-D array")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"labels must be integer class IDs, got dtype {array.dtype}")
    if array.min() < 0 or array.max() >= NUM_LEAVES:
        raise ValueError(f"labels must be in [0, {NUM_LEAVES})")
    return array.astype(np.int64)


class StratifiedRandomBaseline:
    """Predict at random, with probability equal to each class's training frequency.

    The input is ignored (only its length is used). Predictions are reproducible: the
    random generator is re-created from ``seed`` on every call to ``predict``.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._priors: np.ndarray | None = None

    def fit(self, labels: object) -> StratifiedRandomBaseline:
        counts = np.bincount(_check_labels(labels), minlength=NUM_LEAVES)
        self._priors = counts / counts.sum()
        return self

    def predict(self, lengths: object) -> np.ndarray:
        if self._priors is None:
            raise RuntimeError("fit() must be called before predict()")
        rng = np.random.default_rng(self._seed)
        draws = rng.choice(NUM_LEAVES, size=len(np.asarray(lengths)), p=self._priors)
        return draws.astype(np.int64)


def token_documents(input_ids: Iterable[Sequence[int]]) -> list[str]:
    """Turn each file's token-ID list into a space-separated 'document' of ID strings."""
    return [" ".join(map(str, ids)) for ids in input_ids]


class TfidfLogisticBaseline:
    """TF-IDF over token IDs and adjacent pairs, then multinomial logistic regression.

    ``fit`` takes the VALIDATION split too, because C is chosen on it (one model is fitted
    per value in ``c_grid``; the best validation macro-F1 wins, ties keep the earliest
    value). The validation split is never used to fit the vectorizer or the weights.
    """

    def __init__(
        self,
        c_grid: Sequence[float] = DEFAULT_C_GRID,
        max_iter: int = LOGREG_MAX_ITER,
        min_df: int = TFIDF_MIN_DF,
        max_features: int = TFIDF_MAX_FEATURES,
    ) -> None:
        if len(c_grid) == 0:
            raise ValueError("the C grid is empty")
        self._c_grid = tuple(float(c) for c in c_grid)
        self._max_iter = max_iter
        self._min_df = min_df
        self._max_features = max_features
        self._vectorizer: TfidfVectorizer | None = None
        self._model: LogisticRegression | None = None
        self.best_c: float | None = None
        self.validation_scores: dict[float, float] = {}
        self.n_iter_by_c: dict[float, int] = {}

    def fit(
        self,
        train_documents: Sequence[str],
        train_labels: object,
        val_documents: Sequence[str],
        val_labels: object,
    ) -> TfidfLogisticBaseline:
        y_train = _check_labels(train_labels)
        y_val = _check_labels(val_labels)
        if len(train_documents) != y_train.shape[0] or len(val_documents) != y_val.shape[0]:
            raise ValueError("each split needs one label per document")

        self._vectorizer = TfidfVectorizer(
            token_pattern=r"\S+",
            lowercase=False,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=self._min_df,
            max_features=self._max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        x_train = self._vectorizer.fit_transform(train_documents)  # training files only
        x_val = self._vectorizer.transform(val_documents)

        best_model: LogisticRegression | None = None
        best_score = float("-inf")
        scores: dict[float, float] = {}
        n_iters: dict[float, int] = {}
        for c in self._c_grid:
            model = LogisticRegression(C=c, class_weight="balanced", max_iter=self._max_iter)
            model.fit(x_train, y_train)
            n_iters[c] = int(np.max(model.n_iter_))
            score = macro_f1(y_val, model.predict(x_val).astype(np.int64))
            scores[c] = score
            if best_model is None or score > best_score:  # strict: a tie keeps the earliest C
                best_model, best_score, self.best_c = model, score, c
        self._model = best_model
        self.validation_scores = scores
        self.n_iter_by_c = n_iters
        return self

    def _require_fitted(self) -> tuple[TfidfVectorizer, LogisticRegression]:
        if self._vectorizer is None or self._model is None:
            raise RuntimeError("fit() must be called before predict()")
        return self._vectorizer, self._model

    def predict(self, documents: Sequence[str]) -> np.ndarray:
        vectorizer, model = self._require_fitted()
        return model.predict(vectorizer.transform(documents)).astype(np.int64)

    @property
    def n_features(self) -> int:
        """Size of the TF-IDF vocabulary learned from the training files."""
        vectorizer, _ = self._require_fitted()
        return len(vectorizer.vocabulary_)

    @property
    def n_iter(self) -> int:
        """Solver iterations used by the selected model."""
        _, model = self._require_fitted()
        return int(np.max(model.n_iter_))

    @property
    def converged(self) -> bool:
        """True if the selected model stopped before hitting ``max_iter``."""
        return self.n_iter < self._max_iter


# --------------------------------------------------------------------------- #
# Baseline dumps
# --------------------------------------------------------------------------- #


def baseline_frame(submission_ids: Sequence[str], y_true: object, y_pred: object) -> pd.DataFrame:
    """One row per file in the three-column baseline-dump schema."""
    truth = _check_labels(y_true)
    pred = _check_labels(y_pred)
    if not (len(submission_ids) == truth.shape[0] == pred.shape[0]):
        raise ValueError("submission_ids, y_true and y_pred must have the same length")
    return pd.DataFrame({"submission_id": list(submission_ids), "y_true": truth, "y_pred": pred})[
        list(BASELINE_COLUMNS)
    ]


def write_baseline_dump(frame: pd.DataFrame, path: Path) -> None:
    """Write a baseline dump atomically; refuse to overwrite (one-time rule)."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; dumps are written once and never replaced")
    if list(frame.columns) != list(BASELINE_COLUMNS):
        raise ValueError("baseline dump columns differ from BASELINE_COLUMNS")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def read_baseline_dump(path: Path) -> pd.DataFrame:
    """Read a baseline dump and check its columns."""
    frame = pd.read_parquet(path)
    if list(frame.columns) != list(BASELINE_COLUMNS):
        raise ValueError(f"{path} has unexpected columns: {list(frame.columns)}")
    return frame
