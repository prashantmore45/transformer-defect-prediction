"""
M4 Stage 2 (Linux/WSL variant) -- Recompile the pilot sample with Linux GCC.

Purpose
-------
The Windows MinGW-w64 recompilation (m4_stage2_recompile.py) surfaced 4
files failing with "relocation truncated to fit: IMAGE_REL_AMD64_REL32" --
a Windows PE/COFF-specific linker limitation, not a property of the C++
source. Since IBM CodeNet submissions were judged on Linux (AtCoder/AIZU),
this script recompiles the same 498-file sample with a Linux GCC (via WSL)
to test whether that failure mode disappears under the actual target
platform the code was originally judged on.

This script runs in the existing Windows Python venv -- it does NOT
require pandas or Python inside WSL. Each compile is shelled out via
`wsl g++ ...`, converting each Windows source path to its WSL-mount
equivalent first.

Inputs
------
- reports/m4_stage1_resolved.csv (498 rows, Windows paths)

Output
------
- reports/m4_stage2_linux_recompiled.csv
  Same shape as m4_stage2_recompiled.csv, kept as a SEPARATE file so the
  Windows and Linux toolchain results can be compared side by side.
"""

import re
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TIMEOUT_SECONDS = 30
INPUT_CSV = Path("reports/m4_stage1_resolved.csv")
OUTPUT_CSV = Path("reports/m4_stage2_linux_recompiled.csv")

# Scratch build directory INSIDE the repo (on D:, which WSL already mounts),
# not the Windows default temp dir -- avoids any doubt about mountability.
SCRATCH_DIR = Path("build_scratch_linux")

LINKER_MARKERS = ("undefined reference", "ld returned")


def win_to_wsl_path(win_path: str) -> str:
    """Convert 'D:\\a\\b\\c.cpp' -> '/mnt/d/a/b/c.cpp'."""
    p = PureWindowsPath(win_path)
    drive_letter = p.drive.rstrip(":").lower()
    rest_parts = p.parts[1:]  # drop the drive part, e.g. 'D:\\'
    return f"/mnt/{drive_letter}/" + "/".join(rest_parts)


def classify(returncode: int, stderr: str, timed_out: bool) -> str:
    if timed_out:
        return "TIMEOUT"
    if returncode == 0:
        return "COMPILES_CLEAN"
    if any(marker in stderr for marker in LINKER_MARKERS):
        return "LINKER"
    return "OTHER_COMPILE_ERROR"


def main() -> int:
    # Sanity check: confirm WSL g++ is actually reachable before compiling anything.
    check = subprocess.run(["wsl", "g++", "--version"], capture_output=True, text=True)
    if check.returncode != 0:
        print("ERROR: 'wsl g++ --version' failed. Is g++ installed in your WSL distro?")
        print(check.stderr)
        return 1
    print(f"WSL g++ version check OK:\n{check.stdout.splitlines()[0]}\n")

    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run Stage 1 first.")
        return 1

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} resolved rows from Stage 1")

    SCRATCH_DIR.mkdir(exist_ok=True)
    scratch_wsl = win_to_wsl_path(str(SCRATCH_DIR.resolve()))

    results = []
    start = time.time()

    for i, row in df.iterrows():
        src_wsl = win_to_wsl_path(row["resolved_path"])
        out_wsl = f"{scratch_wsl}/{row['submission_id']}.out"

        cmd = ["wsl", "g++", f"-std={row['std_flag']}", src_wsl, "-o", out_wsl]

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
                "linux_returncode": returncode,
                "linux_stderr": stderr,
                "linux_classification": new_classification,
                "compile_seconds": round(compile_seconds, 3),
                "timed_out": timed_out,
            }
        )

        out_file = SCRATCH_DIR / f"{row['submission_id']}.out"
        if out_file.exists():
            out_file.unlink()

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(df)} compiled")

    elapsed = time.time() - start
    print(f"\nRecompiled {len(results)} files in {elapsed:.1f}s")

    out = pd.DataFrame(results)

    n_timeout = out["timed_out"].sum()
    if n_timeout > 0:
        print(f"WARNING: {n_timeout} file(s) timed out:")
        print(out.loc[out["timed_out"], "submission_id"].tolist())

    print("\nLinux (WSL GCC) classification counts:")
    print(out["linux_classification"].value_counts().to_string())

    # Specifically check the 4 files that failed with the Windows-only
    # relocation error, to answer the question this script exists for.
    reloc_ids = ["s966843997", "s826249906", "s103319025", "s634326722"]
    check = out[out["submission_id"].isin(reloc_ids)][
        ["submission_id", "linux_classification", "linux_returncode"]
    ]
    print("\nThe 4 Windows-relocation-failure files, under Linux GCC:")
    print(check.to_string(index=False))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(out)} rows to {OUTPUT_CSV}")

    try:
        SCRATCH_DIR.rmdir()
    except OSError:
        pass  # non-empty or in use; harmless, not worth failing the run over

    print("Stage 2 (Linux/WSL) complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
