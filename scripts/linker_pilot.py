"""LINKER-rate feasibility pilot.

Recompiles a random sample of COMPILE_ERROR files from the extracted working
corpus and classifies each failure as LINKER-stage or not, based on stderr
content. This answers a single go/no-go question before Milestone 4: is LINKER
viable as a class at all, or will 30,000 sampled COMPILE_ERROR files yield too
few LINKER examples to train on?

This is NOT Tier 2 label derivation. Tier 2 will need to distinguish SYNTAX from
SEMANTIC as well, which requires parsing g++'s diagnostic categories properly.
This pilot only answers "linker stage or earlier", using two literal substrings
in stderr as the signal.

Known limitation: this pilot uses whichever g++ is on PATH (recorded below).
An outdated compiler may reject valid modern C++ at the parse stage before
reaching the linker, misclassifying some true LINKER failures as earlier-stage
errors. The measured rate is therefore a LOWER BOUND on the true LINKER rate.

Usage:
    python scripts/linker_pilot.py
"""

from __future__ import annotations

import random
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_MANIFEST = PROJECT_ROOT / "data" / "interim" / "sample_manifest.parquet"
SOURCES_ROOT = PROJECT_ROOT / "data" / "processed" / "sources"
REPORT_OUT = PROJECT_ROOT / "reports" / "linker_pilot.csv"

SAMPLE_SIZE = 500
SEED = 42
COMPILE_TIMEOUT_S = 15

LINKER_SIGNALS = ("undefined reference", "collect2: error: ld returned")


def get_compiler_version() -> str:
    result = subprocess.run(["g++", "--version"], capture_output=True, text=True, timeout=10)
    return result.stdout.splitlines()[0] if result.stdout else "unknown"


def classify(stderr: str) -> str:
    return "LINKER" if any(sig in stderr for sig in LINKER_SIGNALS) else "OTHER"


def main() -> None:
    if not SAMPLE_MANIFEST.exists():
        raise SystemExit(f"sample manifest not found: {SAMPLE_MANIFEST}")

    compiler_version = get_compiler_version()
    print(f"Compiler: {compiler_version}")
    print(
        "NOTE: this compiler may be older than the one used to judge these "
        "submissions originally. Measured LINKER rate is a LOWER BOUND.\n"
    )

    sample = pd.read_parquet(SAMPLE_MANIFEST)
    compile_errors = sample[sample["coarse_label"].astype(str) == "COMPILE_ERROR"]
    print(f"COMPILE_ERROR files available in corpus: {len(compile_errors):,}")

    if len(compile_errors) < SAMPLE_SIZE:
        raise SystemExit(
            f"only {len(compile_errors)} COMPILE_ERROR files available; " f"need {SAMPLE_SIZE}"
        )

    picked = compile_errors.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    picked["rel_path"] = picked["archive_path"].str.replace(
        "Project_CodeNet/data/", "", regex=False
    )

    results = []
    start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="linker_pilot_") as tmpdir:
        tmp_out = Path(tmpdir) / "a.out"

        for i, row in enumerate(picked.itertuples(), start=1):
            src_path = SOURCES_ROOT / row.rel_path
            if not src_path.exists():
                results.append(
                    {
                        "submission_id": row.submission_id,
                        "problem_id": row.problem_id,
                        "outcome": "SOURCE_MISSING",
                        "classification": None,
                        "stderr_tail": "",
                    }
                )
                continue

            try:
                proc = subprocess.run(
                    ["g++", "-std=gnu++17", str(src_path), "-o", str(tmp_out)],
                    capture_output=True,
                    text=True,
                    timeout=COMPILE_TIMEOUT_S,
                    errors="replace",
                )
            except subprocess.TimeoutExpired:
                results.append(
                    {
                        "submission_id": row.submission_id,
                        "problem_id": row.problem_id,
                        "outcome": "TIMEOUT",
                        "classification": None,
                        "stderr_tail": "",
                    }
                )
                continue

            if proc.returncode == 0:
                outcome = "COMPILED_CLEAN"
                cls = None
            else:
                outcome = "FAILED"
                cls = classify(proc.stderr)

            results.append(
                {
                    "submission_id": row.submission_id,
                    "problem_id": row.problem_id,
                    "outcome": outcome,
                    "classification": cls,
                    "stderr_tail": proc.stderr[-300:] if proc.stderr else "",
                }
            )

            if tmp_out.exists():
                tmp_out.unlink()

            if i % 50 == 0:
                elapsed = time.monotonic() - start
                print(f"  [{elapsed:5.1f}s] {i}/{SAMPLE_SIZE} compiled")

    elapsed = time.monotonic() - start
    report = pd.DataFrame(results)
    report.to_csv(REPORT_OUT, index=False)

    print(f"\nPilot complete in {elapsed:.1f}s ({elapsed/SAMPLE_SIZE*1000:.0f} ms/file avg)")
    print(f"Report written: {REPORT_OUT}\n")

    outcome_counts = report["outcome"].value_counts()
    print("Outcomes:")
    print(outcome_counts.to_string())

    failed = report[report["outcome"] == "FAILED"]
    n_failed = len(failed)
    if n_failed == 0:
        print("\nNo failures reproduced under this compiler — cannot estimate LINKER rate.")
        return

    cls_counts = failed["classification"].value_counts()
    n_linker = int(cls_counts.get("LINKER", 0))

    print(f"\nOf {n_failed} reproduced failures:")
    print(cls_counts.to_string())
    print(
        f"\nLINKER rate among reproduced COMPILE_ERROR failures: "
        f"{n_linker}/{n_failed} = {n_linker/n_failed*100:.2f}%"
    )
    print(
        f"LINKER rate among the full {SAMPLE_SIZE}-file sample: "
        f"{n_linker}/{SAMPLE_SIZE} = {n_linker/SAMPLE_SIZE*100:.2f}%"
    )

    projected = int(30_000 * n_linker / SAMPLE_SIZE)
    print(
        f"\nProjected LINKER examples in the full 30,000-file COMPILE_ERROR "
        f"reserve: ~{projected:,}"
    )
    print(f"(Recorded compiler: {compiler_version} — treat as a lower bound.)")


if __name__ == "__main__":
    main()
