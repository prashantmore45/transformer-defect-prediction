"""Integration tests: synthetic C++ programs -> tier3_worker.py -> classify().

Each program deliberately dies in one specific way, so these tests prove the
whole chain (compile, run with limits, capture return code, classify) on
known ground truth, without needing the real corpus.

Runs natively on Linux, or through `wsl` on Windows. Skipped automatically if
neither a Linux g++ nor WSL is available. Scratch files live inside the repo
(not %TEMP%) because Windows temp paths can contain spaces, which `wsl`
argument passing handles poorly.
"""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import pytest

from sdp.data.labeling.tier3 import Tier3Discard, Tier3Label, classify

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "scripts" / "tier3_worker.py"
SCRATCH = REPO_ROOT / "build_scratch_linux" / "tier3_tests"
ON_WINDOWS = sys.platform == "win32"


def _to_wsl(path: Path) -> str:
    if not ON_WINDOWS:
        return str(path)
    p = PureWindowsPath(path.resolve())
    return f"/mnt/{p.drive.rstrip(':').lower()}/" + "/".join(p.parts[1:])


def _linux_gpp_available() -> bool:
    if ON_WINDOWS:
        if shutil.which("wsl") is None:
            return False
        return subprocess.run(["wsl", "g++", "--version"], capture_output=True).returncode == 0
    return shutil.which("g++") is not None


pytestmark = pytest.mark.skipif(
    not _linux_gpp_available(), reason="needs Linux g++ (native or via WSL)"
)

# name -> (C++ source, stdin text or None, expected outcome)
# `volatile` + no optimisation keep the compiler from removing the crash.
PROGRAMS = {
    "segv_null": (
        "int main(){ volatile int* p = 0; return *p; }",
        None,
        Tier3Label.SIGSEGV,
    ),
    "segv_stack_overflow": (
        "int f(volatile int n){ volatile char buf[4096]; buf[0]=n; return f(n+1)+buf[0]; }"
        " int main(){ return f(0); }",
        None,
        Tier3Label.SIGSEGV,
    ),
    # Pilot v1 regression. On a hardened distro g++ (e.g. Ubuntu g++ 15) an
    # out-of-range vector index aborts via a libstdc++ assertion and must be
    # EXCLUDED, not labelled SIGABRT. On a plain toolchain it is a segfault.
    "vector_far_out_of_range": (
        "#include <vector>\nint main(){ std::vector<int> v(3);"
        " volatile long i = 100000000; v[i] = 1; return 0; }",
        None,
        (Tier3Label.SIGSEGV, Tier3Discard.TOOLCHAIN_HARDENING),
    ),
    # A user-written assert() is a genuine abort and must stay SIGABRT even
    # on a hardened toolchain (its message names the user's file).
    "user_assert": (
        "#include <cassert>\nint main(){ volatile int x = 0; assert(x > 0); }",
        None,
        Tier3Label.SIGABRT,
    ),
    "fpe_div_zero": (
        "int main(){ volatile int a = 1, b = 0; return a / b; }",
        None,
        Tier3Label.SIGFPE,
    ),
    "abort_call": (
        "#include <cstdlib>\nint main(){ abort(); }",
        None,
        Tier3Label.SIGABRT,
    ),
    "abort_uncaught_exception": (
        '#include <stdexcept>\nint main(){ throw std::runtime_error("x"); }',
        None,
        Tier3Label.SIGABRT,
    ),
    "nzec_return_3": ("int main(){ return 3; }", None, Tier3Label.NZEC),
    "exit_zero": ("int main(){ return 0; }", None, Tier3Discard.EXIT_ZERO),
    "timeout_loop": (
        "int main(){ volatile int x = 0; while(true){ x = x + 1; } }",
        None,
        Tier3Discard.TIMEOUT,
    ),
    "resource_bad_alloc": (
        "#include <cstddef>\nint main(){ volatile size_t n = (size_t)8 << 30;"
        " char* p = new char[n]; return p[0]; }",
        None,
        Tier3Discard.RESOURCE_LIMIT,
    ),
    "compile_error": (
        "int main(){ this is not c++ }",
        None,
        Tier3Discard.COMPILE_FAILED,
    ),
    # Proves stdin plumbing: crashes ONLY if the sample input says "0".
    "uses_stdin": (
        "#include <iostream>\nint main(){ int d; std::cin >> d;"
        " volatile int a = 1; return a / d; }",
        "0\n",
        Tier3Label.SIGFPE,
    ),
}


@pytest.fixture(scope="module")
def results():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    jobs_path = SCRATCH / "jobs.jsonl"
    out_path = SCRATCH / "out.csv"
    with open(jobs_path, "w", encoding="utf-8") as fh:
        for name, (source, stdin_text, _) in PROGRAMS.items():
            src = SCRATCH / f"{name}.cpp"
            src.write_text(source, encoding="utf-8")
            inp = ""
            if stdin_text is not None:
                inp_path = SCRATCH / f"{name}.in"
                inp_path.write_text(stdin_text, encoding="utf-8")
                inp = _to_wsl(inp_path)
            fh.write(
                json.dumps(
                    {
                        "submission_id": name,
                        "src": _to_wsl(src),
                        "input": inp,
                        "input_source": "sample_verified" if inp else "empty_stdin",
                    }
                )
                + "\n"
            )

    prefix = ["wsl"] if ON_WINDOWS else []
    cmd = prefix + [
        "python3",
        _to_wsl(WORKER),
        "--jobs",
        _to_wsl(jobs_path),
        "--out",
        _to_wsl(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr

    with open(out_path, newline="", encoding="utf-8") as fh:
        rows = {r["submission_id"]: r for r in csv.DictReader(fh)}
    yield rows
    shutil.rmtree(SCRATCH, ignore_errors=True)


def _classify_row(row):
    return classify(
        int(row["run_returncode"]) if row["run_returncode"] != "" else 0,
        row["run_stderr_tail"],
        timed_out=row["run_timed_out"] == "True",
        compile_ok=row["compile_ok"] == "True",
    )


@pytest.mark.parametrize("name", list(PROGRAMS))
def test_synthetic_program_gets_expected_outcome(results, name):
    expected = PROGRAMS[name][2]
    acceptable = expected if isinstance(expected, tuple) else (expected,)
    assert _classify_row(results[name]) in acceptable


def test_all_programs_were_processed(results):
    assert set(results) == set(PROGRAMS)


def test_input_source_is_recorded(results):
    assert results["uses_stdin"]["input_source"] == "sample_verified"
    assert results["segv_null"]["input_source"] == "empty_stdin"
