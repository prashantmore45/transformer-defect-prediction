"""Parent-level views of the 9-class results (Milestone 8.2).

Concepts
--------
The nine leaf classes sit under four PARENT classes (Tier 1 of the taxonomy):

    ERROR_FREE | COMPILE_ERROR (SYNTAX, SEMANTIC, LINKER) |
    RUNTIME_ERROR (SIGSEGV, SIGFPE, SIGABRT, NZEC) | LOGICAL

Three numbers describe how the model's mistakes relate to that structure:

* parent accuracy: share of files whose predicted class has the correct parent;
* within-parent error share: of all wrong predictions, the fraction that land in a
  SIBLING class (same parent, different leaf);
* a chance reference for that share: what it would be if every wrong prediction chose
  uniformly among the other K-1 classes, keeping the observed number of errors per true
  class. It is a rough yardstick, not a null model: real errors lean toward frequent
  classes.

A within-parent share clearly above the reference means the model mostly confuses
neighbouring leaves; a share near or below it means errors ignore the hierarchy. M8.6
uses this as evidence on whether a hierarchical head is worth building.

All functions accept one confusion matrix (K, K) or a stack (..., K, K), like
``sdp.evaluation.bootstrap.metrics_from_confusion``, so they work on bootstrap draws.
A ratio whose denominator is zero (for example the error share when there are no errors)
is 0, matching the rule in ``sdp.evaluation.metrics``.
"""

from __future__ import annotations

import numpy as np

from sdp.data.taxonomy import (
    COARSE_ORDER,
    COARSE_TO_ID,
    LEAF_ORDER,
    NUM_COARSE,
    NUM_LEAVES,
    PARENT_OF,
)
from sdp.evaluation.bootstrap import metrics_from_confusion


def parent_ids() -> np.ndarray:
    """Parent class ID for each leaf class ID (shape (NUM_LEAVES,))."""
    return np.array([COARSE_TO_ID[PARENT_OF[leaf]] for leaf in LEAF_ORDER], dtype=np.int64)


def membership_matrix() -> np.ndarray:
    """0/1 matrix (NUM_LEAVES, NUM_COARSE): entry [leaf, parent] is 1 if parent owns leaf."""
    matrix = np.zeros((NUM_LEAVES, NUM_COARSE), dtype=np.int64)
    matrix[np.arange(NUM_LEAVES), parent_ids()] = 1
    return matrix


def parent_blocks() -> list[tuple[str, int, int]]:
    """Contiguous (parent name, start, stop) ranges of leaf IDs, in leaf order.

    Used to draw group separators on the confusion matrix. Raises if a parent's leaves
    are not contiguous in ``LEAF_ORDER``, because separators would then be meaningless.
    """
    ids = parent_ids()
    blocks: list[tuple[str, int, int]] = []
    start = 0
    for index in range(1, NUM_LEAVES + 1):
        if index == NUM_LEAVES or ids[index] != ids[start]:
            blocks.append((COARSE_ORDER[int(ids[start])].value, start, index))
            start = index
    names = [name for name, _, _ in blocks]
    if len(set(names)) != len(names):
        raise ValueError("leaf classes of one parent are not contiguous in LEAF_ORDER")
    return blocks


def _check_leaf_matrices(cm: object) -> np.ndarray:
    matrix = np.asarray(cm)
    if matrix.ndim < 2 or matrix.shape[-2:] != (NUM_LEAVES, NUM_LEAVES):
        raise ValueError(f"expected confusion matrices of shape (..., {NUM_LEAVES}, {NUM_LEAVES})")
    return matrix


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(np.shape(numerator), dtype=np.float64),
        where=denominator > 0,
    )


def aggregate_to_parents(cm: object) -> np.ndarray:
    """Sum leaf confusion matrices into parent confusion matrices (..., 4, 4).

    Rows are true parents, columns are predicted parents, in ``COARSE_ORDER``.
    """
    matrix = _check_leaf_matrices(cm)
    membership = membership_matrix()
    return np.einsum("...ij,ia,jb->...ab", matrix, membership, membership)


def parent_stats(cm: object) -> dict[str, np.ndarray]:
    """Parent accuracy, parent macro-F1, within-parent error share and its chance reference."""
    matrix = np.asarray(_check_leaf_matrices(cm), dtype=np.float64)
    membership = membership_matrix()
    same_parent = membership @ membership.T  # (K, K): 1 where two leaves share a parent
    siblings = same_parent.sum(axis=1) - 1  # other leaves under the same parent, per leaf

    total = matrix.sum(axis=(-2, -1))
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    errors = total - diagonal.sum(axis=-1)
    within_errors = (matrix * same_parent).sum(axis=(-2, -1)) - diagonal.sum(axis=-1)
    errors_per_true_class = matrix.sum(axis=-1) - diagonal  # (..., K)
    expected_within = (errors_per_true_class * siblings).sum(axis=-1) / (NUM_LEAVES - 1)

    parent_cm = aggregate_to_parents(matrix)
    parent_correct = np.trace(parent_cm, axis1=-2, axis2=-1)
    return {
        "parent_accuracy": _ratio(parent_correct, total),
        "parent_macro_f1": metrics_from_confusion(parent_cm)["macro_f1"],
        "within_parent_error_share": _ratio(within_errors, errors),
        "within_parent_error_share_if_uniform": _ratio(expected_within, errors),
    }
