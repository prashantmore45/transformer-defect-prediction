"""Working-corpus sampling and archive path construction.

Why not `df.sample(n)`
----------------------
Problem sizes in CodeNet are severely concentrated: the top 100 problems hold
19.6% of candidate submissions and the top 1,000 hold 84.3%, with the largest
single problem contributing 39,830 rows. Uniform random sampling draws in
proportion to problem size, so a 18,000-row sample would be dominated by a few
hundred large problems and would cover a narrow slice of algorithmic tasks.

That is a real confound rather than a cosmetic concern. Submissions to one
problem share input formats, identifier conventions, and algorithmic structure,
so a sample concentrated on few problems invites the model to learn problem
idioms instead of defect signatures — the same failure mode as resubmission-chain
leakage, one level down.

`draw_sample` therefore uses round-robin allocation across problems: one
submission from every problem, then a second from every problem, and so on until
the quota is met. Every problem contributes before any problem contributes twice.

The implementation is vectorised. Rows are shuffled, `cumcount()` numbers them
within each problem, and a *stable* sort on that rank groups all rank-0 rows
first — which is exactly round zero. `kind="stable"` is load-bearing: pandas'
default quicksort would scramble the shuffled problem order within each rank,
making the partial final round irreproducible even with a fixed seed.

Shortfall is reported, never silent
-----------------------------------
If a (split, class) cell holds fewer rows than its quota, every available row is
taken and the deficit is recorded in the returned report. A quietly truncated
class would leave you believing you have thousands of examples when you have
hundreds.

Archive paths
-------------
Both path constants were verified against the full metadata in Milestone 2:
`language` is `C++` and `filename_ext` is `cpp` for all 8,008,527 rows, so the
layout is unambiguous. Paths are built only for sampled rows — materialising
7.64M path strings would cost roughly 600 MB for no information, since every
path is derivable from two columns.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

import numpy as np
import pandas as pd

from sdp.data.splitting import DEFAULT_RATIOS, SPLIT_ORDER, Split
from sdp.data.taxonomy import COARSE_ORDER, CoarseClass

# Verified constants for the CodeNet directory layout (see module docstring).
ARCHIVE_ROOT: Final[str] = "Project_CodeNet/data"
LANGUAGE_DIR: Final[str] = "C++"
FILE_EXT: Final[str] = "cpp"

DEFAULT_SEED: Final[int] = 42

# Working corpus: 75,000 files. Only the two classes that fan out at Tiers 2 and
# 3 carry a reserve; ERROR_FREE and LOGICAL are terminal, so over-sampling them
# buys nothing. COMPILE_ERROR splits three ways (SYNTAX/SEMANTIC/LINKER) and
# RUNTIME_ERROR four (SIGSEGV/SIGFPE/SIGABRT/NZEC).
DEFAULT_CLASS_TOTALS: Final[Mapping[CoarseClass, int]] = MappingProxyType(
    {
        CoarseClass.ERROR_FREE: 10_000,
        CoarseClass.COMPILE_ERROR: 30_000,
        CoarseClass.RUNTIME_ERROR: 25_000,
        CoarseClass.LOGICAL: 10_000,
    }
)


def resolve_quotas(
    class_totals: Mapping[CoarseClass, int] = DEFAULT_CLASS_TOTALS,
    ratios: Mapping[Split, float] = DEFAULT_RATIOS,
) -> dict[tuple[Split, CoarseClass], int]:
    """Split each class total across the three splits.

    Each class is divided independently, so class balance and split proportions
    stay decoupled — changing one cannot perturb the other. Rounding residue is
    absorbed by the final split so that per-class totals are exact.

    Raises:
        ValueError: on a missing or non-positive class total.
    """
    if set(class_totals) != set(COARSE_ORDER):
        raise ValueError(f"class_totals must cover exactly {[c.value for c in COARSE_ORDER]}")
    if any(n <= 0 for n in class_totals.values()):
        raise ValueError(f"all class totals must be positive; got {dict(class_totals)}")

    quotas: dict[tuple[Split, CoarseClass], int] = {}
    for cls in COARSE_ORDER:
        total = class_totals[cls]
        running = 0
        for split in SPLIT_ORDER[:-1]:
            n = int(round(total * ratios[split]))
            quotas[(split, cls)] = n
            running += n
        quotas[(SPLIT_ORDER[-1], cls)] = total - running

    if any(n <= 0 for n in quotas.values()):
        bad = [(s.value, c.value) for (s, c), n in quotas.items() if n <= 0]
        raise ValueError(f"ratios produced empty quotas for: {bad}")
    return quotas


def _round_robin_take(group: pd.DataFrame, n: int, *, seed: int, problem_col: str) -> pd.DataFrame:
    """Take `n` rows, spreading across problems as evenly as possible."""
    if len(group) <= n:
        return group

    shuffled = group.sample(frac=1, random_state=seed)
    rank = shuffled.groupby(problem_col, observed=True).cumcount()
    # Stable sort is required: quicksort would reorder rows within a rank and
    # break reproducibility of the partial final round.
    order = np.argsort(rank.to_numpy(), kind="stable")
    return shuffled.iloc[order[:n]]


def draw_sample(
    df: pd.DataFrame,
    *,
    quotas: Mapping[tuple[Split, CoarseClass], int] | None = None,
    seed: int = DEFAULT_SEED,
    split_col: str = "split",
    label_col: str = "coarse_label",
    problem_col: str = "problem_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw the working corpus from the frozen split manifest.

    Args:
        df: the split manifest; must contain `split_col`, `label_col`,
            `problem_col`.
        quotas: (split, class) -> row count. Defaults to `resolve_quotas()`.
        seed: each cell uses a distinct derived seed, so changing one quota does
            not perturb the selection in other cells.

    Returns:
        (sample, report). `sample` preserves the input columns and index.
        `report` has one row per (split, class) with requested, available, taken
        and shortfall counts, plus the number of distinct problems represented.
    """
    for col in (split_col, label_col, problem_col):
        if col not in df.columns:
            raise KeyError(f"missing required column {col!r}")

    if quotas is None:
        quotas = resolve_quotas()

    chunks: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []

    for cell_index, split in enumerate(SPLIT_ORDER):
        for class_index, cls in enumerate(COARSE_ORDER):
            requested = quotas[(split, cls)]
            cell = df[
                (df[split_col].astype(str) == split.value)
                & (df[label_col].astype(str) == cls.value)
            ]
            # Derived per-cell seed: a change to one quota leaves others intact.
            cell_seed = seed + 1_000 * cell_index + class_index
            taken = _round_robin_take(cell, requested, seed=cell_seed, problem_col=problem_col)
            chunks.append(taken)
            rows.append(
                {
                    "split": split.value,
                    "coarse_label": cls.value,
                    "requested": requested,
                    "available": len(cell),
                    "taken": len(taken),
                    "shortfall": max(0, requested - len(taken)),
                    "problems": taken[problem_col].nunique(),
                    "max_per_problem": (
                        int(taken[problem_col].value_counts().max()) if len(taken) else 0
                    ),
                }
            )

    sample = pd.concat(chunks).sort_index()
    report = pd.DataFrame(rows)
    return sample, report


def archive_paths(
    df: pd.DataFrame,
    *,
    problem_col: str = "problem_id",
    submission_col: str = "submission_id",
) -> pd.Series:
    """Build in-archive member paths for the sampled rows.

    Layout: `Project_CodeNet/data/{problem_id}/C++/{submission_id}.cpp`
    """
    return (
        ARCHIVE_ROOT
        + "/"
        + df[problem_col].astype(str)
        + "/"
        + LANGUAGE_DIR
        + "/"
        + df[submission_col].astype(str)
        + "."
        + FILE_EXT
    )
