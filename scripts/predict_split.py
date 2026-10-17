"""Write the one-time prediction dumps for evaluation (Milestone 8.1).

Run once, after M7:

    python scripts/predict_split.py --best-sha256 <full SHA-256 of runs/<run>/best.pt>

Order of operations (each step stops the script on failure, before the next one starts):

1. Verify the checkpoint and manifest SHA-256 hashes.
2. Refuse to start if the test dump or provenance file already exists (one-time rule).
3. Load the model; produce the VALIDATION dump and check that it reproduces the
   validation macro-F1 stored in the checkpoint. Failure writes nothing and never
   reads the test split.
4. Only then run the TEST split, once, and write its dump.

No test metrics are printed here on purpose: M8.2 computes them with problem-level
bootstrap intervals, so the first test number anyone sees comes with its uncertainty.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader

from sdp.config import PROJECT_ROOT
from sdp.data.dataset import (
    MANIFEST_9CLASS_PATH,
    TOKENIZED_9CLASS_DIR,
    SourceCodeDataset,
    collate_batch,
)
from sdp.evaluation.predictions import (
    check_reproduction,
    collect_predictions,
    load_model,
    read_dump,
    require_hash,
    sha256_file,
    validate_against_manifest,
    write_dump,
)
from sdp.training.trainer import autocast_dtype

SPLIT_CONFIG_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "split_config.json"
MANIFEST_KEY = "sample_manifest_9class.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--best-sha256", required=True, help="full expected SHA-256 of best.pt (64 hex characters)"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "codebert-9class-v1",
        help="training run directory that contains best.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "m8",
        help="where the dumps and the provenance file are written",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="run on the CPU if no GPU is found (very slow; off by default)",
    )
    return parser


def make_loader(split: str, pad_token_id: int, batch_size: int) -> DataLoader:
    """One split's loader, never shuffled, reading only that split's token cache."""
    dataset = SourceCodeDataset(TOKENIZED_9CLASS_DIR / f"{split}.parquet")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_batch, pad_token_id=pad_token_id),
    )


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    best_path: Path = args.run_dir / "best.pt"
    out_dir: Path = args.out_dir
    val_path = out_dir / "val_predictions.parquet"
    test_path = out_dir / "test_predictions.parquet"
    provenance_path = out_dir / "m8_1_provenance.json"

    # Step 2 first, because it is free: do not even hash files if the one-time rule is broken.
    for path in (test_path, provenance_path):
        if path.exists():
            log(f"STOP: {path} already exists; the test dump is written once and never replaced.")
            return 1

    # Step 1: hashes.
    expected_manifest = json.loads(SPLIT_CONFIG_PATH.read_text(encoding="utf-8"))[
        "artifact_hashes"
    ][MANIFEST_KEY]
    log("hashing best.pt and the manifest ...")
    best_sha = require_hash(best_path, args.best_sha256, "best.pt")
    manifest_sha = require_hash(MANIFEST_9CLASS_PATH, expected_manifest, "9-class manifest")
    log(f"best.pt     sha256 {best_sha}")
    log(f"manifest    sha256 {manifest_sha}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        log("STOP: no GPU found. Pass --allow-cpu to run anyway (very slow).")
        return 1
    log(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    model, meta = load_model(best_path, device)
    config = meta["config"]
    dtype = autocast_dtype(config["precision"])
    batch_size = int(config["eval_batch_size"])
    pad_token_id = int(model.encoder.config.pad_token_id)
    log(
        f"loaded best epoch {meta['best_epoch']} "
        f"(stored val macro-F1 {meta['val_macro_f1']:.4f}); "
        f"precision {config['precision']}, eval batch {batch_size}, pad id {pad_token_id}"
    )

    manifest = pd.read_parquet(MANIFEST_9CLASS_PATH, columns=["submission_id", "split", "leaf_id"])

    # Step 3: validation gate.
    if val_path.exists():
        log(f"{val_path.name} already exists; re-checking it instead of recomputing")
        val_frame = read_dump(val_path)
        val_is_new = False
    else:
        log("predicting the validation split ...")
        val_frame = collect_predictions(
            model, make_loader("val", pad_token_id, batch_size), device, dtype
        )
        val_is_new = True
    validate_against_manifest(val_frame, manifest, "val")
    gate = check_reproduction(val_frame, float(meta["val_macro_f1"]))
    log(
        f"validation gate: macro-F1 {gate['macro_f1']:.4f} vs stored "
        f"{gate['expected_macro_f1']:.4f} (|diff| {gate['abs_difference']:.5f}, "
        f"tolerance {gate['tolerance']}), accuracy {gate['accuracy']:.4f}"
    )
    if not gate["passed"]:
        log("STOP: the validation gate FAILED. Nothing was written; the test split was not read.")
        return 2
    if val_is_new:
        write_dump(val_frame, val_path)
        log(f"wrote {val_path}")

    # Step 4: the one test pass.
    log("validation gate passed; predicting the test split (once) ...")
    test_frame = collect_predictions(
        model, make_loader("test", pad_token_id, batch_size), device, dtype
    )
    validate_against_manifest(test_frame, manifest, "test")
    write_dump(test_frame, test_path)
    log(f"wrote {test_path} ({len(test_frame)} rows)")

    provenance = {
        "milestone": "M8.1",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "best_pt_sha256": best_sha,
        "manifest_sha256": manifest_sha,
        "best_epoch": int(meta["best_epoch"]),
        "stored_val_macro_f1": float(meta["val_macro_f1"]),
        "validation_gate": gate,
        "precision": config["precision"],
        "eval_batch_size": batch_size,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "rows": {"val": len(val_frame), "test": len(test_frame)},
        "dump_sha256": {
            val_path.name: sha256_file(val_path),
            test_path.name: sha256_file(test_path),
        },
    }
    with provenance_path.open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    log(f"wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
