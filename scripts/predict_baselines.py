"""Write the one-time baseline prediction dumps for evaluation (Milestone 8.3).

Run once, after M8.2:

    python scripts/predict_baselines.py

Baselines: majority class, stratified random, length-only (the M7.1 models, unchanged),
and TF-IDF + logistic regression. All are fitted on the TRAIN split only; the logistic
regression's C is the only tuned setting and is chosen on VALIDATION macro-F1.

Order of operations (each step stops the script on failure, before the next one starts):

1. Refuse to start if any output file already exists (one-time rule).
2. Verify the 9-class manifest hash; load train and validation, each checked against the
   manifest (every submission ID and label).
3. Fit the baselines. VALIDATION GATE: the M7.1 baselines must reproduce the numbers stored
   in reports/m7_baselines_val_summary.csv. Failure writes nothing and never reads the test
   split.
4. Only then read the test cache, predict, check against the manifest, and write the
   validation and test dumps plus a provenance file.

Validation numbers are printed (validation is the tuning split). No test numbers are
printed: the first time anyone reads a test number it should be in the comparison table,
next to its interval.
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
import sklearn

from sdp.config import PROJECT_ROOT
from sdp.data.dataset import MANIFEST_9CLASS_PATH, TOKENIZED_9CLASS_DIR
from sdp.evaluation.baselines import (
    DEFAULT_C_GRID,
    LOGREG_MAX_ITER,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    LengthOnlyBaseline,
    MajorityBaseline,
    StratifiedRandomBaseline,
    TfidfLogisticBaseline,
    baseline_frame,
    token_documents,
    token_lengths,
    write_baseline_dump,
)
from sdp.evaluation.metrics import summarize
from sdp.evaluation.predictions import require_hash, sha256_file, validate_against_manifest

SPLIT_CONFIG_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "split_config.json"
MANIFEST_KEY = "sample_manifest_9class.parquet"
M7_SUMMARY_CSV = PROJECT_ROOT / "reports" / "m7_baselines_val_summary.csv"
STRATIFIED_SEED = 42
GATE_TOLERANCE = 1e-4  # the stored CSV is rounded to 4 decimals
BASELINE_NAMES = ("majority", "stratified_random", "length_only", "tfidf_logreg")


@dataclass(frozen=True)
class SplitData:
    """One split's cached inputs, in cache order."""

    submission_ids: list[str]
    labels: np.ndarray
    lengths: np.ndarray
    documents: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=TOKENIZED_9CLASS_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "m8",
        help="where the baseline dumps and the provenance file are written",
    )
    return parser


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}", flush=True)


def dump_path(out_dir: Path, name: str, split: str) -> Path:
    return out_dir / f"baseline_{name}_{split}.parquet"


def load_split(cache_dir: Path, split: str, manifest: pd.DataFrame) -> SplitData:
    """Read one split's token cache and check it against the manifest."""
    frame = pd.read_parquet(
        cache_dir / f"{split}.parquet", columns=["submission_id", "input_ids", "label"]
    )
    labels = frame["label"].to_numpy(dtype=np.int64)
    ids = frame["submission_id"].tolist()
    validate_against_manifest(
        pd.DataFrame({"submission_id": ids, "y_true": labels}), manifest, split
    )
    return SplitData(
        submission_ids=ids,
        labels=labels,
        lengths=token_lengths(frame["input_ids"]),
        documents=token_documents(frame["input_ids"]),
    )


def predict_all(models: dict[str, object], data: SplitData) -> dict[str, np.ndarray]:
    """Every baseline's predictions for one split (text model reads documents, rest lengths)."""
    return {
        "majority": models["majority"].predict(data.lengths),
        "stratified_random": models["stratified_random"].predict(data.lengths),
        "length_only": models["length_only"].predict(data.lengths),
        "tfidf_logreg": models["tfidf_logreg"].predict(data.documents),
    }


def check_m7_reproduction(
    val_labels: np.ndarray, val_predictions: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    """Compare the refitted M7.1 baselines with the numbers stored by M7.1."""
    reference = pd.read_csv(M7_SUMMARY_CSV).set_index("baseline")
    results: list[dict[str, object]] = []
    for name in ("majority", "length_only"):
        got = summarize(val_labels, val_predictions[name])
        for metric in ("accuracy", "macro_f1"):
            stored = float(reference.loc[name, metric])
            difference = abs(got[metric] - stored)
            results.append(
                {
                    "baseline": name,
                    "metric": metric,
                    "got": float(got[metric]),
                    "stored": stored,
                    "abs_difference": float(difference),
                    "passed": bool(difference <= GATE_TOLERANCE),
                }
            )
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir: Path = args.out_dir
    provenance_path = out_dir / "m8_3_provenance.json"

    # Step 1: one-time rule, checked before any slow work.
    outputs = [dump_path(out_dir, n, s) for n in BASELINE_NAMES for s in ("val", "test")]
    existing = [path for path in (*outputs, provenance_path) if path.exists()]
    if existing:
        for path in existing:
            log(f"STOP: {path} already exists; outputs are written once and never replaced.")
        return 1

    started = time.perf_counter()

    # Step 2: manifest and the two splits that may be read before the gate.
    expected_manifest = json.loads(SPLIT_CONFIG_PATH.read_text(encoding="utf-8"))[
        "artifact_hashes"
    ][MANIFEST_KEY]
    manifest_sha = require_hash(MANIFEST_9CLASS_PATH, expected_manifest, "9-class manifest")
    log(f"manifest sha256 {manifest_sha}")
    manifest = pd.read_parquet(MANIFEST_9CLASS_PATH, columns=["submission_id", "split", "leaf_id"])
    train = load_split(args.cache_dir, "train", manifest)
    val = load_split(args.cache_dir, "val", manifest)
    log(f"loaded train ({len(train.labels)} files) and validation ({len(val.labels)} files)")

    # Step 3: fit everything on train; choose C on validation.
    log("fitting majority, stratified random and length-only baselines ...")
    models: dict[str, object] = {
        "majority": MajorityBaseline().fit(train.labels),
        "stratified_random": StratifiedRandomBaseline(seed=STRATIFIED_SEED).fit(train.labels),
        "length_only": LengthOnlyBaseline().fit(train.lengths, train.labels),
    }
    log("fitting TF-IDF + logistic regression (this is the slow step; several minutes) ...")
    tfidf = TfidfLogisticBaseline().fit(train.documents, train.labels, val.documents, val.labels)
    models["tfidf_logreg"] = tfidf
    log(
        f"TF-IDF baseline: best C {tfidf.best_c}, {tfidf.n_features} features, "
        f"{tfidf.n_iter} solver iterations, converged {tfidf.converged}"
    )
    log(f"validation macro-F1 per C: {tfidf.validation_scores}")
    log(f"solver iterations per C: {tfidf.n_iter_by_c}")

    val_predictions = predict_all(models, val)
    gate = check_m7_reproduction(val.labels, val_predictions)
    for row in gate:
        log(
            f"M7.1 gate: {row['baseline']} {row['metric']} {row['got']:.4f} vs stored "
            f"{row['stored']:.4f} (|diff| {row['abs_difference']:.5f})"
        )
    if not all(row["passed"] for row in gate):
        log("STOP: the M7.1 reproduction gate FAILED. Nothing was written; test was not read.")
        return 2
    for name in BASELINE_NAMES:
        summary = summarize(val.labels, val_predictions[name])
        log(
            f"validation  {name:<18} accuracy {summary['accuracy']:.4f}  "
            f"macro-F1 {summary['macro_f1']:.4f}"
        )

    # Step 4: gate passed. Write validation dumps, then read and predict the test split once.
    written: list[Path] = []
    for name in BASELINE_NAMES:
        path = dump_path(out_dir, name, "val")
        write_baseline_dump(
            baseline_frame(val.submission_ids, val.labels, val_predictions[name]), path
        )
        written.append(path)
    log("gate passed; reading the test split ...")
    test = load_split(args.cache_dir, "test", manifest)
    test_predictions = predict_all(models, test)
    for name in BASELINE_NAMES:
        path = dump_path(out_dir, name, "test")
        write_baseline_dump(
            baseline_frame(test.submission_ids, test.labels, test_predictions[name]), path
        )
        written.append(path)

    provenance = {
        "milestone": "M8.3",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "manifest_sha256": manifest_sha,
        "scikit_learn": sklearn.__version__,
        "rows": {"train": len(train.labels), "val": len(val.labels), "test": len(test.labels)},
        "m7_1_reproduction_gate": gate,
        "stratified_random": {"seed": STRATIFIED_SEED},
        "length_only": "LengthOnlyBaseline() defaults: depth 5, min leaf 50, balanced, seed 42",
        "tfidf_logreg": {
            "class_weight": "balanced",
            "ngram_range": list(TFIDF_NGRAM_RANGE),
            "min_df": TFIDF_MIN_DF,
            "max_features": TFIDF_MAX_FEATURES,
            "c_grid": list(DEFAULT_C_GRID),
            "validation_macro_f1_by_c": {str(c): s for c, s in tfidf.validation_scores.items()},
            "best_c": tfidf.best_c,
            "n_features": tfidf.n_features,
            "max_iter": LOGREG_MAX_ITER,
            "n_iter": tfidf.n_iter,
            "n_iter_by_c": {str(c): n for c, n in tfidf.n_iter_by_c.items()},
            "converged": tfidf.converged,
        },
        "dump_sha256": {path.name: sha256_file(path) for path in written},
    }
    with provenance_path.open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    for path in (*written, provenance_path):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
