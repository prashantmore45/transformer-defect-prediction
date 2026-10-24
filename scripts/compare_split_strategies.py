"""Compare the random-split model R with the problem-split model P (Milestone 8.5).

Run after M8.1, M8.2 and scripts/predict_random_split.py:

    python scripts/compare_split_strategies.py

Three outputs, all on TEST sets and all with problem-level bootstrap intervals:

1. HEADLINE: R on its own random test set against P on the problem-level test set. The two
   sets are different files, so each is bootstrapped independently (a conservative interval:
   the sets overlap by about a fifth of their files).
2. PAIRED CONTRAST: the files that are in BOTH test sets. Neither model trained on those
   exact files, but R trained on their problem-mates and P did not, so R minus P on the same
   files isolates the leakage effect (shared draws, as in M8.3).
3. LEAKAGE TIERS of R's test set (disjoint): exact content twin in R's train; same problem
   and same user in train; same problem, other user; problem not in R's train.

Gates: every input's SHA-256 is verified and every dump is checked against the manifest;
P's own test interval must reproduce reports/m8/test_overall.csv (exit code 3 and nothing
written if not).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.config import PROJECT_ROOT
from sdp.data.dataset import MANIFEST_9CLASS_PATH
from sdp.evaluation.bootstrap import DEFAULT_LEVEL, DEFAULT_N_BOOT, DEFAULT_SEED
from sdp.evaluation.metrics import LEAF_NAMES
from sdp.evaluation.paired import (
    IndependentComparison,
    PairedComparison,
    align_systems,
    compare_independent,
    compare_systems,
)
from sdp.evaluation.predictions import read_dump, require_hash, validate_against_manifest
from sdp.evaluation.slices import bootstrap_slices
from sdp.evaluation.split_report import problem_ids_for

R_NAME = "random_split_model"
P_NAME = "problem_split_model"
MIN_SUPPORT = 20
CONSISTENCY_TOLERANCE = 1e-5
TIERS = (
    "exact_twin",
    "same_problem_and_user",
    "same_problem_other_user",
    "problem_not_in_train",
)
TIER_PAIRS = [
    ("exact_twin", "same_problem_other_user"),
    ("same_problem_and_user", "same_problem_other_user"),
    ("same_problem_other_user", "problem_not_in_train"),
]

SPLIT_CONFIG_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "split_config.json"
MANIFEST_KEY = "sample_manifest_9class.parquet"
DUMP_NAME = "test_predictions.parquet"


@dataclass(frozen=True)
class Inputs:
    manifest: pd.DataFrame
    problem_dump: pd.DataFrame
    random_dump: pd.DataFrame
    hashes: dict[str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "m8",
        help="folder with the M8 dumps; outputs go to <dir>/random_split",
    )
    return parser


def load_inputs(reports_dir: Path) -> Inputs:
    random_dir = reports_dir / "random_split"
    m8_1 = json.loads((reports_dir / "m8_1_provenance.json").read_text("utf-8"))["dump_sha256"]
    m8_5 = json.loads((random_dir / "provenance.json").read_text("utf-8"))["dump_sha256"]
    expected_manifest = json.loads(SPLIT_CONFIG_PATH.read_text("utf-8"))["artifact_hashes"][
        MANIFEST_KEY
    ]
    hashes = {
        "manifest": require_hash(MANIFEST_9CLASS_PATH, expected_manifest, "9-class manifest"),
        f"problem_split/{DUMP_NAME}": require_hash(
            reports_dir / DUMP_NAME, m8_1[DUMP_NAME], "problem-split test dump"
        ),
        f"random_split/{DUMP_NAME}": require_hash(
            random_dir / DUMP_NAME, m8_5[DUMP_NAME], "random-split test dump"
        ),
    }
    manifest = pd.read_parquet(
        MANIFEST_9CLASS_PATH,
        columns=[
            "submission_id",
            "problem_id",
            "user_id",
            "sha256",
            "split",
            "random_split",
            "leaf_id",
        ],
    )
    problem_dump = read_dump(reports_dir / DUMP_NAME)
    random_dump = read_dump(random_dir / DUMP_NAME)
    validate_against_manifest(problem_dump, manifest, "test")
    random_view = manifest[["submission_id", "leaf_id"]].assign(split=manifest["random_split"])
    validate_against_manifest(random_dump, random_view, "test")
    return Inputs(manifest, problem_dump, random_dump, hashes)


def check_against_m8_2(headline: IndependentComparison, reports_dir: Path) -> list[str]:
    """Differences between P's interval here and the M8.2 table (empty list = consistent)."""
    stored = pd.read_csv(reports_dir / "test_overall.csv").set_index("metric")
    mine = headline.systems[headline.systems["system"] == P_NAME].set_index("metric")
    problems = []
    for metric in ("accuracy", "macro_f1"):
        for column in ("value", "ci_low", "ci_high"):
            difference = abs(float(mine.loc[metric, column]) - float(stored.loc[metric, column]))
            if difference > CONSISTENCY_TOLERANCE:
                problems.append(f"{metric} {column}: differs from test_overall.csv by {difference}")
    return problems


def paired_on_shared_files(inputs: Inputs) -> tuple[PairedComparison, pd.DataFrame]:
    """R minus P on the files in both test sets, plus per-class differences with file counts."""
    columns = ["submission_id", "y_true", "y_pred"]
    shared = set(inputs.problem_dump["submission_id"]) & set(inputs.random_dump["submission_id"])
    frames = {
        R_NAME: inputs.random_dump.loc[inputs.random_dump["submission_id"].isin(shared), columns],
        P_NAME: inputs.problem_dump.loc[inputs.problem_dump["submission_id"].isin(shared), columns],
    }
    ids, y_true, predictions = align_systems(frames)
    groups = problem_ids_for(pd.DataFrame({"submission_id": ids}), inputs.manifest)
    result = compare_systems(
        y_true,
        predictions,
        groups,
        reference=R_NAME,
        n_boot=DEFAULT_N_BOOT,
        seed=DEFAULT_SEED,
        level=DEFAULT_LEVEL,
    )
    support = dict(
        zip(LEAF_NAMES, np.bincount(y_true, minlength=len(LEAF_NAMES)).tolist(), strict=True)
    )
    per_class = result.per_class_differences.copy()
    per_class["n_files"] = per_class["class"].map(support)
    per_class["low_support"] = per_class["n_files"] < MIN_SUPPORT
    return result, per_class


def leakage_tiers(
    inputs: Inputs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """R's test files split into disjoint leakage tiers relative to R's training set."""
    manifest = inputs.manifest.assign(problem=inputs.manifest["problem_id"].astype(str))
    manifest["key"] = manifest["problem"] + "|" + manifest["user_id"].astype(str)
    train = manifest[manifest["random_split"] == "train"]
    meta = manifest.set_index("submission_id").loc[inputs.random_dump["submission_id"]]
    twin = meta["sha256"].isin(set(train["sha256"])).to_numpy()
    same_user = meta["key"].isin(set(train["key"])).to_numpy()
    same_problem = meta["problem"].isin(set(train["problem"])).to_numpy()
    tier = np.select([twin, same_user, same_problem], list(TIERS[:3]), default=TIERS[3])
    masks = {name: tier == name for name in TIERS}
    return (
        inputs.random_dump["y_true"].to_numpy(),
        inputs.random_dump["y_pred"].to_numpy(),
        meta["problem"].to_numpy(),
        masks,
    )


def add_support_flag(differences: pd.DataFrame, per_class: pd.DataFrame) -> pd.DataFrame:
    """Add min_files (files in the smaller slice, per-class rows only) and low_support."""
    if differences.empty:
        return differences
    support = {
        (name, label): count
        for name, label, count in zip(
            per_class["slice"], per_class["class"], per_class["support"], strict=True
        )
    }
    min_files: list[float] = []
    for first, second, metric in zip(
        differences["first"], differences["second"], differences["metric"], strict=True
    ):
        if metric.startswith("recall:"):
            label = metric.split(":", 1)[1]
            min_files.append(min(support[(first, label)], support[(second, label)]))
        else:
            min_files.append(float("nan"))
    flagged = differences.copy()
    flagged["min_files"] = min_files
    flagged["low_support"] = flagged["min_files"] < MIN_SUPPORT
    return flagged


# --------------------------------------------------------------------------- #
# Printing and writing
# --------------------------------------------------------------------------- #


def interval_text(value: float, low: float, high: float) -> str:
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def excludes_zero(low: float, high: float) -> bool:
    return bool(low > 0 or high < 0)


def print_systems(title: str, systems: pd.DataFrame) -> None:
    print(f"\n{title}")
    for name in systems["system"].unique():
        rows = systems[systems["system"] == name].set_index("metric")
        accuracy, macro = rows.loc["accuracy"], rows.loc["macro_f1"]
        print(
            f"  {name:<20} accuracy "
            f"{interval_text(accuracy['value'], accuracy['ci_low'], accuracy['ci_high'])}   "
            f"macro-F1 {interval_text(macro['value'], macro['ci_low'], macro['ci_high'])}"
        )


def difference_view(
    table: pd.DataFrame, label: object, name_column: str = "metric"
) -> pd.DataFrame:
    rows = list(table.itertuples())
    value_column = "f1_difference" if "f1_difference" in table.columns else "value"
    view = pd.DataFrame(
        {
            "comparison": label,
            name_column: table[name_column].to_numpy(),
            "difference [95% interval]": [
                interval_text(getattr(r, value_column), r.ci_low, r.ci_high) for r in rows
            ],
            "excludes 0": ["yes" if excludes_zero(r.ci_low, r.ci_high) else "" for r in rows],
        }
    )
    if "low_support" in table.columns:
        view["min files"] = [
            ("" if pd.isna(n) else str(int(n)))
            for n in table["min_files" if "min_files" in table.columns else "n_files"]
        ]
        view["note"] = ["LOW n" if flag else "" for flag in table["low_support"]]
    return view


def write_table(table: pd.DataFrame, path: Path) -> None:
    table.round(6).to_csv(path, index=False)
    print(path)


def main(argv: list[str] | None = None) -> int:
    reports_dir: Path = build_parser().parse_args(argv).reports_dir
    out_dir = reports_dir / "random_split"
    inputs = load_inputs(reports_dir)
    random_dump, problem_dump = inputs.random_dump, inputs.problem_dump

    # 1. Headline (independent bootstraps; P keeps the seed it had in M8.2).
    headline = compare_independent(
        (
            random_dump["y_true"].to_numpy(),
            random_dump["y_pred"].to_numpy(),
            problem_ids_for(random_dump, inputs.manifest),
        ),
        (
            problem_dump["y_true"].to_numpy(),
            problem_dump["y_pred"].to_numpy(),
            problem_ids_for(problem_dump, inputs.manifest),
        ),
        names=(R_NAME, P_NAME),
        seeds=(DEFAULT_SEED + 1, DEFAULT_SEED),
        n_boot=DEFAULT_N_BOOT,
        level=DEFAULT_LEVEL,
    )
    problems = check_against_m8_2(headline, reports_dir)
    if problems:
        for problem in problems:
            print(f"STOP: {problem}")
        print("P's interval disagrees with M8.2. Nothing was written.")
        return 3
    print("consistency check passed: P's test interval equals test_overall.csv")

    # 2. Paired contrast on the files in both test sets.
    paired, paired_per_class = paired_on_shared_files(inputs)

    # 3. Leakage tiers of R's test set.
    y_true, y_pred, groups, masks = leakage_tiers(inputs)
    shares = pd.DataFrame(
        {
            "tier": list(masks),
            "n_files": [int(mask.sum()) for mask in masks.values()],
            "share": [float(mask.mean()) for mask in masks.values()],
        }
    )
    present = {name: mask for name, mask in masks.items() if mask.any()}
    pairs = [(a, b) for a, b in TIER_PAIRS if a in present and b in present]
    tiers = bootstrap_slices(
        y_true,
        y_pred,
        groups,
        present,
        pairs=pairs,
        n_boot=DEFAULT_N_BOOT,
        seed=DEFAULT_SEED,
        level=DEFAULT_LEVEL,
        min_class_support=MIN_SUPPORT,
    )
    tier_differences = add_support_flag(tiers.differences, tiers.per_class)
    shared_mask = random_dump["submission_id"].isin(set(problem_dump["submission_id"])).to_numpy()
    difficulty = bootstrap_slices(
        y_true,
        y_pred,
        groups,
        {"shared_with_problem_test": shared_mask, "rest_of_random_test": ~shared_mask},
        pairs=[("shared_with_problem_test", "rest_of_random_test")],
        n_boot=DEFAULT_N_BOOT,
        seed=DEFAULT_SEED,
        level=DEFAULT_LEVEL,
        min_class_support=MIN_SUPPORT,
    )
    difficulty_differences = add_support_flag(difficulty.differences, difficulty.per_class)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(headline.systems, out_dir / "compare_headline_systems.csv")
    write_table(headline.differences, out_dir / "compare_headline_differences.csv")
    write_table(headline.per_class_differences, out_dir / "compare_headline_per_class.csv")
    write_table(paired.systems, out_dir / "compare_paired_systems.csv")
    write_table(paired.differences, out_dir / "compare_paired_differences.csv")
    write_table(paired_per_class, out_dir / "compare_paired_per_class.csv")
    write_table(shares, out_dir / "compare_tier_shares.csv")
    write_table(tiers.overall, out_dir / "compare_tiers_overall.csv")
    write_table(tiers.per_class, out_dir / "compare_tiers_per_class.csv")
    write_table(difficulty.overall, out_dir / "compare_difficulty_overall.csv")
    write_table(difficulty_differences, out_dir / "compare_difficulty_differences.csv")
    if not tier_differences.empty:
        write_table(tier_differences, out_dir / "compare_tiers_differences.csv")
    provenance = {
        "milestone": "M8.5",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bootstrap": {
            "n_boot": DEFAULT_N_BOOT,
            "seed": DEFAULT_SEED,
            "level": DEFAULT_LEVEL,
            "headline": "independent bootstraps, seeds (43 for R, 42 for P)",
            "paired": "shared draws over the problems of the shared files",
        },
        "input_sha256": inputs.hashes,
        "files": {
            "random_test": headline.n_files[R_NAME],
            "problem_test": headline.n_files[P_NAME],
            "shared": paired.n_files,
        },
        "problems": {
            "random_test": headline.n_problems[R_NAME],
            "problem_test": headline.n_problems[P_NAME],
            "shared": paired.n_problems,
        },
        "tier_counts": dict(zip(shares["tier"], shares["n_files"].tolist(), strict=True)),
    }
    provenance_path = out_dir / "compare_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(provenance_path)

    print_systems("HEADLINE: each model on ITS OWN test set", headline.systems)
    print(
        f"  files: {headline.n_files[R_NAME]} vs {headline.n_files[P_NAME]}; "
        f"problems: {headline.n_problems[R_NAME]} vs {headline.n_problems[P_NAME]}"
    )
    print(difference_view(headline.differences, f"{R_NAME} - {P_NAME}").to_string(index=False))

    print_systems(
        f"PAIRED: the same {paired.n_files} files ({paired.n_problems} problems) for both models",
        paired.systems,
    )
    print(difference_view(paired.differences, f"{R_NAME} - {P_NAME}").to_string(index=False))
    print("\nPAIRED per-class F1 difference (R minus P); low-support classes are flagged")
    print(difference_view(paired_per_class, f"{R_NAME} - {P_NAME}", "class").to_string(index=False))

    print("\nLEAKAGE TIERS of the random test set (disjoint)")
    print(shares.round(3).to_string(index=False))
    overall = tiers.overall
    rows = list(overall.itertuples())
    print(
        pd.DataFrame(
            {
                "tier": overall["slice"],
                "files": overall["n_files"],
                "problems": overall["n_problems"],
                "accuracy": [interval_text(r.accuracy, r.accuracy_lo, r.accuracy_hi) for r in rows],
                "macro_recall (classes with enough files)": [
                    interval_text(r.macro_recall, r.macro_recall_lo, r.macro_recall_hi)
                    for r in rows
                ],
                "classes": overall["n_classes_scored"],
            }
        ).to_string(index=False)
    )
    if not tier_differences.empty:
        print("\nTIERS: first minus second")
        labels = (tier_differences["first"] + " - " + tier_differences["second"]).to_numpy()
        print(difference_view(tier_differences, labels).to_string(index=False))
        print("\nDIFFICULTY CONTROL: R on shared files vs the rest of its random test set")
    rows = list(difficulty.overall.itertuples())
    print(
        pd.DataFrame(
            {
                "slice": difficulty.overall["slice"],
                "files": difficulty.overall["n_files"],
                "accuracy": [interval_text(r.accuracy, r.accuracy_lo, r.accuracy_hi) for r in rows],
                "macro_recall": [
                    interval_text(r.macro_recall, r.macro_recall_lo, r.macro_recall_hi)
                    for r in rows
                ],
                "classes": difficulty.overall["n_classes_scored"],
            }
        ).to_string(index=False)
    )
    print(difference_view(difficulty_differences, "shared - rest").to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
