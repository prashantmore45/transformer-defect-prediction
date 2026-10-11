"""Classification metrics for the 9-class defect taxonomy.

Milestone 7.1
-------------
Pure functions over integer class-ID arrays. They are used for per-epoch
validation during training (M7), for the baselines (M7.1), and for the final
test-set evaluation (M8), so every number in the project comes from one
implementation.

Design rules
------------
* Every class in ``range(num_classes)`` is always scored, even if it never
  appears in the predictions. A class the model never predicts gets F1 = 0 and
  still counts in the macro average. Dropping it would let a model look better
  by ignoring its hardest class.
* Inputs are validated (1-D, non-empty, integer, in range, equal length) and
  rejected loudly instead of producing a plausible-looking wrong number.
* Confusion matrices put TRUE classes on rows and PREDICTED classes on columns,
  in ``LEAF_ORDER``. That order is pipeline-stage order (see
  ``sdp.data.taxonomy``), so near-diagonal mass means stage confusion.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from sdp.data.taxonomy import LEAF_ORDER, NUM_LEAVES

#: Class names in integer-ID order (index == class ID).
LEAF_NAMES: tuple[str, ...] = tuple(cls.value for cls in LEAF_ORDER)


def _validate(y: object, name: str, num_classes: int) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must contain integer class IDs, got dtype {arr.dtype}")
    if arr.min() < 0 or arr.max() >= num_classes:
        raise ValueError(
            f"{name} values must be in [0, {num_classes}); "
            f"found min={arr.min()}, max={arr.max()}"
        )
    return arr


def _validate_pair(
    y_true: object, y_pred: object, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    true = _validate(y_true, "y_true", num_classes)
    pred = _validate(y_pred, "y_pred", num_classes)
    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same length, "
            f"got {true.shape[0]} and {pred.shape[0]}"
        )
    return true, pred


def confusion_matrix(y_true: object, y_pred: object, num_classes: int = NUM_LEAVES) -> np.ndarray:
    """Return a (num_classes, num_classes) matrix; rows = true, columns = predicted."""
    true, pred = _validate_pair(y_true, y_pred, num_classes)
    return sk_metrics.confusion_matrix(true, pred, labels=list(range(num_classes)))


def per_class_report(
    y_true: object, y_pred: object, label_names: Sequence[str] = LEAF_NAMES
) -> pd.DataFrame:
    """One row per class: support, precision, recall, F1 (0 where undefined)."""
    num_classes = len(label_names)
    true, pred = _validate_pair(y_true, y_pred, num_classes)
    precision, recall, f1, support = sk_metrics.precision_recall_fscore_support(
        true, pred, labels=list(range(num_classes)), zero_division=0
    )
    return pd.DataFrame(
        {
            "class": list(label_names),
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def macro_f1(y_true: object, y_pred: object, num_classes: int = NUM_LEAVES) -> float:
    """Unweighted mean of per-class F1 over all ``num_classes`` classes."""
    true, pred = _validate_pair(y_true, y_pred, num_classes)
    _, _, f1, _ = sk_metrics.precision_recall_fscore_support(
        true,
        pred,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )
    return float(f1)


def accuracy(y_true: object, y_pred: object, num_classes: int = NUM_LEAVES) -> float:
    """Fraction of exactly-correct predictions."""
    true, pred = _validate_pair(y_true, y_pred, num_classes)
    return float(sk_metrics.accuracy_score(true, pred))


def summarize(y_true: object, y_pred: object, num_classes: int = NUM_LEAVES) -> dict[str, float]:
    """Headline numbers: accuracy and macro-F1."""
    return {
        "accuracy": accuracy(y_true, y_pred, num_classes),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
    }
