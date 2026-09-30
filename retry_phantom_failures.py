"""
M4 -- Retry phantom-failure rows (wsl.exe execution failures, not real
compiler diagnostics).

Purpose
-------
3,179 of the 4,829 UNRECOGNIZED rows from the full-reserve run have
genuinely empty stderr, and 3,169 of those share the exact returncode
4294967295 (0xFFFFFFFF) -- not a value g++ itself produces. This is the
signature of wsl.exe failing to launch or complete the command, not g++
running and rejecting the code. These files' true compile outcome was
never actually captured and cannot be recovered by adding classification
patterns -- they need to be recompiled.

This script identifies exactly that subset, re-joins it against the M2
manifest to get rel_path (the results CSV doesn't carry it), and recompiles
using the same compile_one() logic already validated in
run_tier2_full_reserve.py (imported, not duplicated).

Checkpointed the same way as the main run: results are written and flushed
per-row, and a rerun skips whatever this script has already retried.

Input
-----
- reports/m4_full_reserve_results.csv
- data/processed/splits/sample_manifest_hashed.parquet

Output
------
- reports/m4_retry_results.csv (same schema as the main results file)
"""

import csv
import sys
import time
from pathlib import Path

import pandas as pd

# Reuses the exact same validated compile logic -- not duplicated.
import run_tier2_full_reserve as pipeline
from sdp.data.labeling import tier2

RESULTS_CSV = Path("reports/m4_full_reserve_results.csv")
MANIFEST_PARQUET = Path("data/processed/splits/sample_manifest_hashed.parquet")
RETRY_CSV = Path("reports/m4_retry_results.csv")

# The exact signature identified from real data: empty stderr AND not a
# genuine timeout (those are already correctly classified separately).
PHANTOM_RETURNCODES = {4294967295, 2, 15}

OUTPUT_FIELDS = [
    "submission_id",
    "problem_id",
    "returncode",
    "stderr",
    "classification",
    "timed_out",
    "compile_seconds",
]


def load_already_retried() -> set[str]:
    if not RETRY_CSV.exists():
        return set()
    existing = pd.read_csv(RETRY_CSV, usecols=["submission_id"])
    return set(existing["submission_id"].astype(str))


def main() -> int:
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found.")
        return 1
    if not MANIFEST_PARQUET.exists():
        print(f"ERROR: {MANIFEST_PARQUET} not found.")
        return 1

    results = pd.read_csv(RESULTS_CSV)
    results["stderr"] = results["stderr"].fillna("")

    phantom_mask = (
        (results["classification"] == "UNRECOGNIZED")
        & (results["stderr"].str.strip() == "")
        & (results["returncode"].isin(PHANTOM_RETURNCODES))
        & (~results["timed_out"].astype(bool))
    )
    phantoms = results[phantom_mask]
    print(f"Identified {len(phantoms)} phantom-failure rows to retry.")

    other_empty = results[
        (results["classification"] == "UNRECOGNIZED")
        & (results["stderr"].str.strip() == "")
        & (~phantom_mask)
    ]
    if len(other_empty) > 0:
        print(
            f"NOTE: {len(other_empty)} other empty-stderr UNRECOGNIZED rows do NOT "
            f"match the known phantom-failure signature and are left alone: "
            f"{other_empty[['submission_id', 'returncode', 'timed_out']].to_dict('records')}"
        )

    already_retried = load_already_retried()
    if already_retried:
        print(f"Resuming: {len(already_retried)} already retried, skipping those.")

    todo_ids = set(phantoms["submission_id"].astype(str)) - already_retried
    print(f"Remaining to retry: {len(todo_ids)}\n")

    if not todo_ids:
        print("Nothing left to retry.")
        return 0

    manifest = pd.read_parquet(MANIFEST_PARQUET)
    todo = manifest[manifest["submission_id"].astype(str).isin(todo_ids)]
    missing = todo_ids - set(todo["submission_id"].astype(str))
    if missing:
        print(f"WARNING: {len(missing)} submission_id(s) not found in manifest: {missing}")

    pipeline.SCRATCH_DIR.mkdir(exist_ok=True)
    scratch_wsl = pipeline.win_to_wsl_path(str(pipeline.SCRATCH_DIR.resolve()))

    RETRY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RETRY_CSV.exists()
    out_file = open(RETRY_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDS)
    if write_header:
        writer.writeheader()
        out_file.flush()

    start = time.time()
    n_processed = 0
    n_still_phantom = 0

    try:
        for _, row in todo.iterrows():
            returncode, stderr, timed_out, compile_seconds = pipeline.compile_one(
                row["rel_path"], row["submission_id"], scratch_wsl
            )
            outcome = tier2.classify(returncode, stderr, timed_out=timed_out)

            if not stderr.strip() and returncode in PHANTOM_RETURNCODES and not timed_out:
                n_still_phantom += 1

            writer.writerow(
                {
                    "submission_id": row["submission_id"],
                    "problem_id": row["problem_id"],
                    "returncode": returncode,
                    "stderr": stderr,
                    "classification": str(outcome),
                    "timed_out": timed_out,
                    "compile_seconds": round(compile_seconds, 3),
                }
            )
            out_file.flush()
            n_processed += 1

            if n_processed % 100 == 0:
                elapsed = time.time() - start
                rate = n_processed / elapsed
                remaining = len(todo) - n_processed
                eta_min = (remaining / rate / 60) if rate > 0 else 0
                print(
                    f"  {n_processed}/{len(todo)} retried ({rate:.2f} files/s, ETA {eta_min:.0f} min)"
                )

    except KeyboardInterrupt:
        out_file.close()
        print(f"\nStopped after {n_processed} retries this run. Rerun to resume.")
        return 130

    out_file.close()
    try:
        pipeline.SCRATCH_DIR.rmdir()
    except OSError:
        pass

    elapsed = time.time() - start
    print(f"\nDone. Retried {n_processed} files in {elapsed / 60:.1f} min.")
    if n_still_phantom > 0:
        print(
            f"WARNING: {n_still_phantom} file(s) STILL show the phantom-failure "
            f"signature after retry -- these need manual investigation, not another "
            f"automatic retry."
        )
    print(f"Results in {RETRY_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
