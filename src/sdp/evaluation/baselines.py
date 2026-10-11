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
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.tree import DecisionTreeClassifier


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
                "lengths and labels must have the same length, "
                f"got {x.shape[0]} and {y.shape[0]}"
            )
        self._tree.fit(x, y)
        self._fitted = True
        return self

    def predict(self, lengths: object) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit() must be called before predict()")
        return self._tree.predict(self._as_feature(lengths)).astype(np.int64)
