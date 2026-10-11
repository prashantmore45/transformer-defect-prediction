"""Class weights for the imbalanced 9-class loss (Milestone 7.2).

Why weights at all
------------------
The nine classes range from 232 (LINKER) to 9,404 (SEMANTIC) training files.
Plain cross-entropy treats every example equally, so a model can score well by
mostly ignoring the rare classes. A class weight multiplies each example's loss
by a factor that is larger for rarer classes, so mistakes on them cost more.

The strength knob
-----------------
    weight(class) ~ count(class) ** -power

* power = 0.0: no weighting (every weight is 1)
* power = 0.5: square-root weighting (about 6:1 between LINKER and SEMANTIC)
* power = 1.0: full inverse frequency (about 40:1)

The default for this project is 0.5. Full inverse frequency makes one LINKER
example worth about 40 SEMANTIC ones, which risks unstable gradients and
memorizing the 232 LINKER files. That is a judgment, not a measurement; the
power is a config value (``class_weight_power``) so it can be changed without
editing code.

Normalization
-------------
Weights are scaled so the sample-weighted mean is exactly 1: the average
weight a random training example receives is 1, so the overall loss scale
stays comparable to an unweighted run whatever the power. This is also what
scikit-learn calls 'balanced' when power = 1.

Counts must come from the actual training labels, never from assumed numbers.
"""

import numpy as np

from sdp.data.taxonomy import NUM_LEAVES


def count_classes(labels: object, num_classes: int = NUM_LEAVES) -> np.ndarray:
    """Number of examples per class ID, length ``num_classes`` (absent classes get 0)."""
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("labels is empty")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"labels must contain integer class IDs, got dtype {arr.dtype}")
    if arr.min() < 0 or arr.max() >= num_classes:
        raise ValueError(
            f"labels must be in [0, {num_classes}); found min={arr.min()}, max={arr.max()}"
        )
    return np.bincount(arr, minlength=num_classes)


def compute_class_weights(counts: object, power: float) -> np.ndarray:
    """Per-class loss weights ``count ** -power``, scaled to a sample-weighted mean of 1."""
    arr = np.asarray(counts)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"counts must be a non-empty 1-D array, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"counts must be integers, got dtype {arr.dtype}")
    if (arr <= 0).any():
        raise ValueError(
            "every class needs at least one training example to be weighted; "
            f"got counts {arr.tolist()}"
        )
    if not 0.0 <= power <= 1.0:
        raise ValueError(f"power must be in [0, 1], got {power}")

    counts_f = arr.astype(np.float64)
    raw = counts_f ** (-power)
    scale = counts_f.sum() / (counts_f * raw).sum()
    return raw * scale
