"""Fit the M7.1 reference baselines on train and report on VALIDATION.

Deliberately never opens ``test.parquet``: the test split is reserved for the
single final evaluation in M8.

Writes:
    reports/m7_baselines_val.csv          per-class precision/recall/F1 per baseline
    reports/m7_baselines_val_summary.csv  accuracy and macro-F1 per baseline

Exits non-zero if the loaded class counts disagree with
reports/m6_class_split_counts.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.config import PROJECT_ROOT
from sdp.data.dataset import TOKENIZED_9CLASS_DIR
from sdp.data.taxonomy import NUM_LEAVES
from sdp.evaluation.baselines import LengthOnlyBaseline, MajorityBaseline, token_lengths
from sdp.evaluation.metrics import LEAF_NAMES, per_class_report, summarize

REPORTS_DIR = PROJECT_ROOT / "reports"
M6_COUNTS_CSV = REPORTS_DIR / "m6_class_split_counts.csv"


def _load(cache_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(cache_dir / f"{split}.parquet", columns=["input_ids", "label"])
    return token_lengths(df["input_ids"]), df["label"].to_numpy(dtype=np.int64)


def _check_against_m6(train_labels: np.ndarray, val_labels: np.ndarray) -> None:
    expected = pd.read_csv(M6_COUNTS_CSV).set_index("leaf_label")
    for split, labels in (("train", train_labels), ("val", val_labels)):
        got = np.bincount(labels, minlength=NUM_LEAVES)
        want = np.array([int(expected.loc[name, split]) for name in LEAF_NAMES])
        if not np.array_equal(got, want):
            raise SystemExit(
                f"{split} class counts differ from {M6_COUNTS_CSV.name}: "
                f"got {got.tolist()}, expected {want.tolist()}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=TOKENIZED_9CLASS_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    train_len, train_y = _load(args.cache_dir, "train")
    val_len, val_y = _load(args.cache_dir, "val")
    _check_against_m6(train_y, val_y)
    print(f"class counts match {M6_COUNTS_CSV.name} (train {len(train_y)}, val {len(val_y)})")
    print(
        f"share of files at the 512 cap: train {(train_len == 512).mean():.1%}, "
        f"val {(val_len == 512).mean():.1%}"
    )

    baselines = {
        "majority": MajorityBaseline().fit(train_y),
        "length_only": LengthOnlyBaseline().fit(train_len, train_y),
    }

    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for name, model in baselines.items():
        pred = model.predict(val_len)
        report = per_class_report(val_y, pred)
        report.insert(0, "baseline", name)
        frames.append(report)
        summary_rows.append({"baseline": name, **summarize(val_y, pred)})

    per_class = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_class.round(4).to_csv(args.out_dir / "m7_baselines_val.csv", index=False)
    summary.round(4).to_csv(args.out_dir / "m7_baselines_val_summary.csv", index=False)

    table = per_class.pivot(index="class", columns="baseline", values="f1").reindex(LEAF_NAMES)
    table["support"] = frames[0].set_index("class")["support"]
    print("\nValidation F1 per class:")
    print(table.round(3).to_string())
    print("\nValidation summary:")
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
