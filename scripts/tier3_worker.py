"""
M5 -- Tier-3 execution worker. Runs INSIDE WSL/Linux, standard library only.

Why this exists
---------------
Signals (SIGSEGV etc.) are only reported faithfully by Linux itself. Going
through `wsl.exe` per file would flatten them into exit codes and (as M4
showed) produces phantom launch failures. So the Windows side makes ONE
`wsl python3 tier3_worker.py ...` call per batch, and this script does the
compiling and running natively.

It captures RAW facts only (return codes, timeout flags, stderr tail). It
does not decide labels -- sdp.data.labeling.tier3.classify() does that on the
Windows side, so rules can change without re-executing anything.

Sandbox (deliberately simple -- no containers)
----------------------------------------------
* each run happens in a fresh temp directory on the Linux filesystem
* stdin comes from the sample input file (or empty); stdout goes to /dev/null
* RLIMIT_CPU / RLIMIT_AS / RLIMIT_FSIZE bound CPU, memory, and file writes
* RLIMIT_CORE=0 disables core dumps (a crash must not write a huge file)
* wall-clock timeout; on expiry the whole process group is SIGKILLed
  (own session via start_new_session=True)
* RLIMIT_NPROC is NOT used: it counts every process of your Linux user and
  could break your own shell.
Compile flags: M4's (-std=gnu++17, no optimisation flags) plus
-fno-stack-protector (Ubuntu's default turns a stack-buffer overflow into
"stack smashing detected", an abort, instead of a segfault).
libstdc++ assertions (out-of-range v[i] -> abort) could NOT be switched off on
g++ 15.2 by any flag we tried, so tier3.classify() excludes those files
(TOOLCHAIN_HARDENING) by reading stderr instead.
User-written assert() and uncaught exceptions are unaffected (genuine aborts).

Jobs file: one JSON object per line:
    {"submission_id": "...", "src": "/mnt/d/.../x.cpp",
     "input": "/mnt/d/.../input.txt" or "", "input_source": "sample_verified"}
Results are appended to --out (CSV) and flushed per file.
"""

import argparse
import csv
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

STD_FLAG = "gnu++17"
# Neutralise the stack-protector default (see module docstring).
BEHAVIOUR_FLAGS = ["-fno-stack-protector"]
COMPILE_TIMEOUT_S = 30
RUN_WALL_TIMEOUT_S = 10
RUN_CPU_LIMIT_S = 5
RUN_MEMORY_LIMIT_BYTES = 2 * 1024**3
RUN_FILE_SIZE_LIMIT_BYTES = 10 * 1024**2
STDERR_TAIL_CHARS = 1000
COMPILE_STDERR_HEAD_CHARS = 300

OUT_FIELDS = [
    "submission_id",
    "compile_ok",
    "compile_returncode",
    "compile_timed_out",
    "compile_stderr_head",
    "run_returncode",
    "run_timed_out",
    "run_stderr_tail",
    "input_source",
    "run_seconds",
]


def _limits():
    # soft < hard on purpose: with soft == hard Linux sends SIGKILL (-9) instead of
    # SIGXCPU (-24), and the kill would be indistinguishable from other causes.
    resource.setrlimit(resource.RLIMIT_CPU, (RUN_CPU_LIMIT_S, RUN_CPU_LIMIT_S + 1))
    resource.setrlimit(resource.RLIMIT_AS, (RUN_MEMORY_LIMIT_BYTES, RUN_MEMORY_LIMIT_BYTES))
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (RUN_FILE_SIZE_LIMIT_BYTES, RUN_FILE_SIZE_LIMIT_BYTES),
    )
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _run(cmd, *, stdin, stdout, stderr, timeout, cwd, preexec=None):
    """Run cmd in its own session; kill the whole group on timeout.

    Returns (returncode, timed_out). Authority over the process group is real
    here because we are inside Linux (unlike a Windows-side wsl.exe handle).
    """
    proc = subprocess.Popen(
        cmd,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        cwd=cwd,
        start_new_session=True,
        preexec_fn=preexec,
    )
    try:
        proc.wait(timeout=timeout)
        return proc.returncode, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return proc.returncode, True


def _read_tail(path, n_chars):
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - 4 * n_chars))
        data = fh.read()
    return data.decode("utf-8", errors="replace")[-n_chars:]


def process_job(job):
    row = {
        "submission_id": job["submission_id"],
        "compile_ok": False,
        "compile_returncode": "",
        "compile_timed_out": False,
        "compile_stderr_head": "",
        "run_returncode": "",
        "run_timed_out": False,
        "run_stderr_tail": "",
        "input_source": job.get("input_source", ""),
        "run_seconds": "",
    }
    workdir = tempfile.mkdtemp(prefix="tier3_")
    try:
        binary = os.path.join(workdir, "prog.out")
        compile_err = os.path.join(workdir, "compile.err")
        with open(compile_err, "wb") as err_fh:
            rc, timed_out = _run(
                ["g++", f"-std={STD_FLAG}", *BEHAVIOUR_FLAGS, job["src"], "-o", binary],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=err_fh,
                timeout=COMPILE_TIMEOUT_S,
                cwd=workdir,
            )
        row["compile_returncode"] = rc
        row["compile_timed_out"] = timed_out
        row["compile_stderr_head"] = _read_tail(compile_err, 100000)[:COMPILE_STDERR_HEAD_CHARS]
        if rc != 0 or timed_out or not os.path.exists(binary):
            return row
        row["compile_ok"] = True

        run_err = os.path.join(workdir, "run.err")
        stdin_fh = open(job["input"], "rb") if job.get("input") else None
        t0 = time.time()
        try:
            with open(run_err, "wb") as err_fh:
                rc, timed_out = _run(
                    [binary],
                    stdin=stdin_fh if stdin_fh else subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=err_fh,
                    timeout=RUN_WALL_TIMEOUT_S,
                    cwd=workdir,
                    preexec=_limits,
                )
        finally:
            if stdin_fh:
                stdin_fh.close()
        row["run_seconds"] = round(time.time() - t0, 3)
        row["run_returncode"] = rc
        row["run_timed_out"] = timed_out
        row["run_stderr_tail"] = _read_tail(run_err, STDERR_TAIL_CHARS)
        return row
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    write_header = not os.path.exists(args.out)
    with (
        open(args.jobs, encoding="utf-8") as jobs_fh,
        open(args.out, "a", newline="", encoding="utf-8") as out_fh,
    ):
        writer = csv.DictWriter(out_fh, fieldnames=OUT_FIELDS)
        if write_header:
            writer.writeheader()
            out_fh.flush()
        for line in jobs_fh:
            line = line.strip()
            if not line:
                continue
            writer.writerow(process_job(json.loads(line)))
            out_fh.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
