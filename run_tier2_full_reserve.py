"""
M4 -- Full-reserve Tier-2 compilation and classification.

Purpose
-------
Runs the actual M4 pipeline: recompiles every file in the frozen 30,000-file
COMPILE_ERROR reserve with Linux GCC (via WSL), classifies each result with
the validated, tested `sdp.data.labeling.tier2.classify()`, and writes one
row per file to a resumable output CSV.

This script is orchestration only -- all classification logic lives in
tier2.py and is unit-tested there (tests/test_tier2.py). This script does
NOT reimplement any pattern matching.

Resumability
-------------
Every result is written and flushed immediately, not batched. On startup,
if the output CSV already exists, already-processed submission_ids are
loaded and skipped. Interrupting this script (Ctrl+C, closed terminal,
sleep) and rerunning the same command resumes from where it left off --
nothing before the interruption is lost or redone.

Runtime
-------
Pilot rate was ~0.53s/file serial. At ~30,000 files, expect roughly 4-5
hours. Progress and an ETA are printed periodically.

Input
-----
- data/processed/splits/sample_manifest_hashed.parquet

Output
------
- reports/m4_full_reserve_results.csv
  One row per file: submission_id, problem_id, returncode, stderr,
  classification, timed_out, compile_seconds

This script does NOT compute the final per-class summary or the
UNRECOGNIZED batch-review list -- that is a separate step once this
finishes (or is interrupted and resumed to completion).
"""

import csv
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

import pandas as pd

from sdp.data.labeling import tier2

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MANIFEST_PARQUET = Path("data/processed/splits/sample_manifest_hashed.parquet")
OUTPUT_CSV = Path("reports/m4_full_reserve_results.csv")
SCRATCH_DIR = Path("build_scratch_linux")

# ASSUMPTION -- flag if wrong: WSL-mounted equivalent of the source root
# already confirmed working in the pilot's Linux recompilation.
WSL_SOURCE_ROOT = "/mnt/d/Dev/Github/transformer-defect-prediction/data/processed/sources"

STD_FLAG = "gnu++17"
TIMEOUT_SECONDS = 30
# Outer safety net only -- NOT the primary timeout mechanism. See compile_one().
PYTHON_OUTER_TIMEOUT_SECONDS = TIMEOUT_SECONDS + 15
# GNU coreutils `timeout` exit codes when the command was killed for timing out.
TIMEOUT_EXIT_CODES = (124, 137)
PROGRESS_EVERY = 200

OUTPUT_FIELDS = [
    "submission_id",
    "problem_id",
    "returncode",
    "stderr",
    "classification",
    "timed_out",
    "compile_seconds",
]


def win_to_wsl_path(win_path: str) -> str:
    p = PureWindowsPath(win_path)
    drive_letter = p.drive.rstrip(":").lower()
    rest_parts = p.parts[1:]
    return f"/mnt/{drive_letter}/" + "/".join(rest_parts)


def load_already_done() -> set[str]:
    if not OUTPUT_CSV.exists():
        return set()
    existing = pd.read_csv(OUTPUT_CSV, usecols=["submission_id"])
    return set(existing["submission_id"].astype(str))


def compile_one(
    rel_path: str, submission_id: str, scratch_wsl: str
) -> tuple[int, str, bool, float]:
    src_wsl = f"{WSL_SOURCE_ROOT}/{rel_path}"
    out_wsl = f"{scratch_wsl}/{submission_id}.out"
    # Timeout enforced INSIDE WSL via `timeout -k 5 <N>`, not by Python across
    # the Windows/WSL process boundary. Python's own subprocess timeout only
    # has a handle on the Windows-side "wsl.exe" launcher -- the actual g++
    # process runs in the WSL Linux VM's own process tree, which "wsl.exe"
    # being killed does not reliably terminate. An orphaned g++ (or its own
    # cc1plus/as/collect2 children) can then keep the stdout/stderr pipes
    # open, hanging Python's read indefinitely -- resistant to Ctrl+C, since
    # Python is blocked in a low-level pipe read, not interpretable code.
    # `timeout`, run inside the same Linux process tree as g++, has real
    # authority to SIGTERM (and SIGKILL after a 5s grace period via -k) it.
    cmd = [
        "wsl",
        "timeout",
        "-k",
        "5",
        str(TIMEOUT_SECONDS),
        "g++",
        f"-std={STD_FLAG}",
        src_wsl,
        "-o",
        out_wsl,
    ]

    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PYTHON_OUTER_TIMEOUT_SECONDS,
        )
        returncode = proc.returncode
        stderr = proc.stderr
        if returncode in TIMEOUT_EXIT_CODES:
            timed_out = True
    except subprocess.TimeoutExpired:
        # The outer safety net fired -- wsl.exe itself didn't return even
        # though the inner `timeout` should have. Rare; logged the same way.
        timed_out = True
        returncode = -1
        stderr = ""
    compile_seconds = time.time() - t0

    out_file = SCRATCH_DIR / f"{submission_id}.out"
    if out_file.exists():
        out_file.unlink()

    return returncode, stderr, timed_out, compile_seconds


def main() -> int:
    check = subprocess.run(["wsl", "g++", "--version"], capture_output=True, text=True)
    if check.returncode != 0:
        print("ERROR: 'wsl g++ --version' failed.")
        print(check.stderr)
        return 1
    print(f"WSL g++ OK: {check.stdout.splitlines()[0]}\n")

    if not MANIFEST_PARQUET.exists():
        print(f"ERROR: {MANIFEST_PARQUET} not found.")
        return 1

    manifest = pd.read_parquet(MANIFEST_PARQUET)
    reserve = manifest[manifest["coarse_label"] == "COMPILE_ERROR"]
    print(f"COMPILE_ERROR reserve: {len(reserve)} rows")

    already_done = load_already_done()
    if already_done:
        print(f"Resuming: {len(already_done)} already processed, skipping those.")

    todo = reserve[~reserve["submission_id"].astype(str).isin(already_done)]
    print(f"Remaining to process: {len(todo)}\n")

    if len(todo) == 0:
        print("Nothing left to do.")
        return 0

    SCRATCH_DIR.mkdir(exist_ok=True)
    scratch_wsl = win_to_wsl_path(str(SCRATCH_DIR.resolve()))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_CSV.exists()
    out_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDS)
    if write_header:
        writer.writeheader()
        out_file.flush()

    start = time.time()
    n_processed = 0

    try:
        for _, row in todo.iterrows():
            returncode, stderr, timed_out, compile_seconds = compile_one(
                row["rel_path"], row["submission_id"], scratch_wsl
            )
            outcome = tier2.classify(returncode, stderr, timed_out=timed_out)

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

            if n_processed % PROGRESS_EVERY == 0:
                elapsed = time.time() - start
                rate = n_processed / elapsed
                remaining = len(todo) - n_processed
                eta_seconds = remaining / rate if rate > 0 else 0
                print(
                    f"  {n_processed}/{len(todo)} done "
                    f"({rate:.2f} files/s, ETA {eta_seconds / 60:.0f} min)"
                )

    except KeyboardInterrupt:
        out_file.close()
        print(
            f"\nStopped after {n_processed} files this run. "
            f"Rerun the same command to resume from here -- nothing is lost."
        )
        return 130

    out_file.close()

    try:
        SCRATCH_DIR.rmdir()
    except OSError:
        pass

    elapsed = time.time() - start
    print(f"\nDone. Processed {n_processed} files in {elapsed / 60:.1f} min this run.")
    print(f"Results in {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
