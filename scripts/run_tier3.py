"""
M5 -- Tier-3 runner (orchestration only).

Runs the RUNTIME_ERROR reserve through scripts/tier3_worker.py (inside WSL),
applies the tested `sdp.data.labeling.tier3.classify()` to the raw results,
and prints a breakdown. No classification logic lives here.

Modes
-----
  Pilot (default): a split-stratified random sample of --sample files (500).
      python scripts/run_tier3.py
  Stability check: re-execute N files from the pilot and count how many
  outcomes change (detects nondeterministic crashes).
      python scripts/run_tier3.py --rerun 200
  Full run: every RUNTIME_ERROR file (only after the pilot is reviewed).
      python scripts/run_tier3.py --sample 0
  Re-summarise without executing anything:
      python scripts/run_tier3.py --summary-only

Resumable: the worker appends one flushed row per file; on startup,
already-processed submission_ids in the raw CSV are skipped (same pattern as
M4). Ctrl+C is safe: the Linux worker is stopped too, and rerunning resumes.

Output
------
  reports/m5_pilot_raw.csv        (or m5_full_reserve_raw.csv with --sample 0)
  reports/m5_pilot_classified.csv (raw + manifest columns + outcome)
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

import pandas as pd

from sdp.data.labeling import tier3

MANIFEST = Path("data/processed/splits/sample_manifest_hashed.parquet")
IO_ROOT = Path(r"D:\Datasets\CodeNet\extracted\Project_CodeNet\derived\input_output")
# Same source root M4 used (confirmed working there).
WSL_SOURCE_ROOT = "/mnt/d/Dev/Github/transformer-defect-prediction/data/processed/sources"
SCRATCH = Path("build_scratch_linux")
WORKER = Path(__file__).resolve().parent / "tier3_worker.py"
ON_WINDOWS = sys.platform == "win32"
WSL = ["wsl"] if ON_WINDOWS else []
PER_FILE_BUDGET_S = 60  # outer safety net per file: compile 30 + run 10 + margin


def to_wsl(path) -> str:
    if not ON_WINDOWS:
        return str(Path(path).resolve())
    p = PureWindowsPath(Path(path).resolve())
    return f"/mnt/{p.drive.rstrip(':').lower()}/" + "/".join(p.parts[1:])


def load_input_info(io_root: Path) -> tuple[set[str], set[str]]:
    """(problems that have a sample input, problems whose sample is unverified)."""
    data = io_root / "data"
    have = {d.name for d in data.iterdir() if d.is_dir()} if data.exists() else set()
    unv_file = io_root / "unverified_accepted_solutions.txt"
    unverified = set()
    if unv_file.exists():  # format not assumed: pull every problem id out of it
        unverified = set(re.findall(r"p\d{5}", unv_file.read_text(errors="ignore")))
    return have, unverified


def build_job(row, source_root, io_root, have, unverified) -> dict:
    pid = row["problem_id"]
    if pid in have:
        source = "sample_unverified" if pid in unverified else "sample_verified"
        input_path = to_wsl(io_root / "data" / pid / "input.txt")
    else:
        source, input_path = "empty_stdin", ""
    return {
        "submission_id": str(row["submission_id"]),
        "src": f"{source_root}/{row['rel_path']}",
        "input": input_path,
        "input_source": source,
    }


def load_done(raw_csv: Path) -> set[str]:
    if not raw_csv.exists():
        return set()
    return set(pd.read_csv(raw_csv, usecols=["submission_id"], dtype=str)["submission_id"])


def stop_linux_worker():
    subprocess.run(WSL + ["pkill", "-f", "tier3_worker.py"], capture_output=True)


def run_batches(jobs: list[dict], raw_csv: Path, batch_size: int) -> bool:
    """Feed jobs to the worker batch by batch. Returns False if interrupted."""
    SCRATCH.mkdir(exist_ok=True)
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    start, done = time.time(), 0
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i : i + batch_size]
        jobs_file = SCRATCH / "tier3_jobs.jsonl"
        jobs_file.write_text("".join(json.dumps(j) + "\n" for j in batch), encoding="utf-8")
        cmd = WSL + [
            "python3", to_wsl(WORKER),
            "--jobs", to_wsl(jobs_file), "--out", to_wsl(raw_csv),
        ]  # fmt: skip
        try:
            subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=PER_FILE_BUDGET_S * len(batch),
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            stop_linux_worker()
            print(f"  batch at {i} hit the outer timeout; rerun to resume.")
            return False
        except KeyboardInterrupt:
            stop_linux_worker()
            print("\nInterrupted. Rerun the same command to resume.")
            return False
        done += len(batch)
        rate = done / (time.time() - start)
        print(f"  {done}/{len(jobs)} done ({rate:.2f} files/s, "
              f"ETA {(len(jobs) - done) / rate / 60:.0f} min)")  # fmt: skip
    return True


def classify_raw(raw_csv: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv, dtype=str, keep_default_na=False)
    raw = raw.drop_duplicates("submission_id", keep="last")
    meta = manifest[["submission_id", "problem_id", "split"]].astype({"submission_id": str})
    df = raw.merge(meta, on="submission_id", how="left")
    df["outcome"] = [
        str(tier3.classify(
            int(r.run_returncode) if r.run_returncode != "" else 0,
            r.run_stderr_tail,
            timed_out=r.run_timed_out == "True",
            compile_ok=r.compile_ok == "True",
        ))
        for r in df.itertuples()
    ]  # fmt: skip
    return df


def print_summary(df: pd.DataFrame):
    n = len(df)
    labelled = df["outcome"].isin([str(x) for x in tier3.Tier3Label]).sum()
    print(f"\nFiles classified: {n}")
    print(f"Reproduced a labelled crash: {labelled} ({100 * labelled / n:.1f}%)")
    print("\nOutcome counts:\n" + df["outcome"].value_counts().to_string())
    print("\nOutcome by split:\n" + pd.crosstab(df["outcome"], df["split"]).to_string())
    print("\nOutcome by input_source:\n"
          + pd.crosstab(df["outcome"], df["input_source"]).to_string())  # fmt: skip
    rate = df.groupby("input_source")["outcome"].apply(
        lambda s: round(100 * s.isin([str(x) for x in tier3.Tier3Label]).mean(), 1)
    )
    print("\nLabelled-crash rate (%) by input_source:\n" + rate.to_string())
    for outcome in ("SIGABRT", "TOOLCHAIN_HARDENING"):
        sub = df[df["outcome"] == outcome]
        if len(sub):
            first = (
                sub["run_stderr_tail"]
                .str.replace(r"0x[0-9a-f]+|\d+", "N", regex=True)
                .str.strip().str.split("\n").str[0].str[:80]
            )  # fmt: skip
            print(f"\n{outcome}: top stderr first lines:\n"
                  + first.value_counts().head(6).to_string())  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=500, help="0 = full reserve")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--rerun", type=int, default=0, help="stability check size")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--io-root", type=Path, default=IO_ROOT)
    ap.add_argument("--source-root", default=WSL_SOURCE_ROOT)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    full = args.sample == 0
    stem = "m5_full_reserve" if full else "m5_pilot"
    raw_csv = args.reports_dir / f"{stem}_raw.csv"
    classified_csv = args.reports_dir / f"{stem}_classified.csv"

    if not args.summary_only:
        # (no argument containing spaces: wsl.exe mangles those)
        check = subprocess.run(WSL + ["python3", "--version"], capture_output=True)
        gpp = subprocess.run(WSL + ["g++", "--version"], capture_output=True, text=True)
        if check.returncode != 0 or gpp.returncode != 0:
            print("ERROR: WSL python3 / g++ not available.")
            return 1

    manifest = pd.read_parquet(args.manifest)
    rt = manifest[manifest["coarse_label"] == "RUNTIME_ERROR"]
    if not full:
        frac = min(1.0, args.sample / len(rt))
        rt = rt.groupby("split", observed=True).sample(frac=frac, random_state=args.seed)
    print(f"RUNTIME_ERROR files in scope: {len(rt)}")
    have, unverified = load_input_info(args.io_root)

    def jobs_for(frame):
        return [build_job(r, args.source_root, args.io_root, have, unverified)
                for _, r in frame.iterrows()]  # fmt: skip

    if args.rerun:
        if not raw_csv.exists():
            print("ERROR: run the pilot first.")
            return 1
        rerun_csv = args.reports_dir / f"{stem}_rerun.csv"
        ids = load_done(raw_csv)
        pick = rt[rt["submission_id"].astype(str).isin(ids)].sample(
            n=min(args.rerun, len(ids)), random_state=args.seed
        )
        todo = pick[~pick["submission_id"].astype(str).isin(load_done(rerun_csv))]
        if len(todo):
            run_batches(jobs_for(todo), rerun_csv, args.batch_size)
        a = classify_raw(raw_csv, manifest).set_index("submission_id")["outcome"]
        b = classify_raw(rerun_csv, manifest).set_index("submission_id")["outcome"]
        common = a.index.intersection(b.index)
        changed = (a[common] != b[common]).sum()
        print(f"\nStability: {changed} of {len(common)} outcomes changed on rerun.")
        print(pd.crosstab(a[common], b[common]).to_string())
        return 0

    if not args.summary_only:
        done = load_done(raw_csv)
        todo = rt[~rt["submission_id"].astype(str).isin(done)]
        print(f"Already done: {len(done)} | remaining: {len(todo)}\n")
        if len(todo) and not run_batches(jobs_for(todo), raw_csv, args.batch_size):
            return 130

    if not raw_csv.exists():
        print("No raw results yet.")
        return 1
    df = classify_raw(raw_csv, manifest)
    df.to_csv(classified_csv, index=False)
    print_summary(df)
    print(f"\nWrote {classified_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
