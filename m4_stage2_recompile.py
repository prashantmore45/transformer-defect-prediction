"""
M4 Stage 2 — Recompile the resolved pilot sample with GCC 16.2.0.

Purpose
-------
Stage 1 resolved 498 of the original 500 pilot submissions to real source
paths on disk. This stage does the actual controlled comparison: recompile
every one of those 498 files with the current toolchain (GCC 16.2.0,
D:\\Dev\\Toolchains\\mingw64) and record what happens, so Stage 3 can compare
against each file's original GCC 6.3.0 classification.

Compiling to a full executable (not `-c` / object-only) is deliberate: a
LINKER error only appears at the link step, so object-only compilation
would silently hide exactly the class we're trying to remeasure.

Inputs
------
- reports/m4_stage1_resolved.csv (498 rows, from Stage 1)

Output
------
- reports/m4_stage2_recompiled.csv
  One row per submission, with:
    submission_id, problem_id, original_classification,
    new_returncode, new_stderr, new_classification,
    compile_seconds, timed_out
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
NEW_GXX = Path(r"D:\Dev\Toolchains\mingw64\bin\g++.exe")
TIMEOUT_SECONDS = 30

INPUT_CSV = Path("reports/m4_stage1_resolved.csv")
OUTPUT_CSV = Path("reports/m4_stage2_recompiled.csv")

LINKER_MARKERS = ("undefined reference", "ld returned")
# NOTE (2026-09-26 fix): previously required the exact substring
# "collect2: error: ld returned", which silently failed to match on
# Windows, where GCC names the linker driver "collect2.exe" --
# "collect2.exe: error: ld returned ..." does NOT contain "collect2: error:
# ld returned" as a substring. Matching on "ld returned" alone is robust to
# the program-name prefix (collect2 vs collect2.exe vs any other variant)
# while still being specific enough that it should only ever appear in a
# genuine link-stage failure message.


def classify(returncode: int, stderr: str, timed_out: bool) -> str:
    if timed_out:
        return "TIMEOUT"
    if returncode == 0:
        return "COMPILES_CLEAN"
    if any(marker in stderr for marker in LINKER_MARKERS):
        return "LINKER"
    return "OTHER_COMPILE_ERROR"


def main() -> int:
    if not NEW_GXX.exists():
        print(f"ERROR: new compiler not found at {NEW_GXX}")
        return 1
    if not INPUT_CSV.exists():
        print(f"ERROR: Stage 1 output not found at {INPUT_CSV}. Run Stage 1 first.")
        return 1

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} resolved rows from Stage 1")

    results = []
    start = time.time()

    with tempfile.TemporaryDirectory(prefix="m4_stage2_") as scratch:
        scratch_dir = Path(scratch)

        for i, row in df.iterrows():
            src = Path(row["resolved_path"])
            out_exe = scratch_dir / f"{row['submission_id']}.exe"

            cmd = [str(NEW_GXX), f"-std={row['std_flag']}", str(src), "-o", str(out_exe)]

            t0 = time.time()
            timed_out = False
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=TIMEOUT_SECONDS,
                )
                returncode = proc.returncode
                stderr = proc.stderr
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode = -1
                stderr = ""
            compile_seconds = time.time() - t0

            new_classification = classify(returncode, stderr, timed_out)

            results.append(
                {
                    "submission_id": row["submission_id"],
                    "problem_id": row["problem_id"],
                    "original_classification": row["original_classification"],
                    "new_returncode": returncode,
                    "new_stderr": stderr,
                    "new_classification": new_classification,
                    "compile_seconds": round(compile_seconds, 3),
                    "timed_out": timed_out,
                }
            )

            # Executable isn't needed after compilation — remove it now
            # rather than letting 498 of them accumulate in the scratch dir.
            if out_exe.exists():
                out_exe.unlink()

            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{len(df)} compiled")

    elapsed = time.time() - start
    print(f"\nRecompiled {len(results)} files in {elapsed:.1f}s")

    out = pd.DataFrame(results)

    n_timeout = out["timed_out"].sum()
    if n_timeout > 0:
        print(f"WARNING: {n_timeout} file(s) timed out after {TIMEOUT_SECONDS}s:")
        print(out.loc[out["timed_out"], "submission_id"].tolist())

    print("\nNew classification counts:")
    print(out["new_classification"].value_counts().to_string())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(out)} rows to {OUTPUT_CSV}")
    print("Stage 2 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
