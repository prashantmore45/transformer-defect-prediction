# Tier-3 Labelling — Runtime-Error Sub-classification (M5)

Group 42 · `transformer-defect-prediction` · Milestone M5 · Tag `v0.6-tier3`

This document records how the 24,981-file `RUNTIME_ERROR` reserve was turned
into four leaf labels (`SIGSEGV`, `SIGFPE`, `SIGABRT`, `NZEC`) by *executing*
each submission and observing how it fails — and, just as importantly, what
was excluded and why. Every number below is reproducible from
`reports/m5_full_reserve_raw.csv` and `reports/m5_full_reserve_classified.csv`.

---

## 1. Result in one table

| | Files | Share of reserve |
|---|---:|---:|
| **Labelled (4 leaf classes)** | **4,349** | **17.4%** |
| `EXIT_ZERO` (crash did not reproduce) | 16,639 | 66.6% |
| `TOOLCHAIN_HARDENING` (compiler-setting artifact) | 2,395 | 9.6% |
| `COMPILE_FAILED` (Tier-2 mismatch) | 739 | 3.0% |
| `TIMEOUT` | 352 | 1.4% |
| `UNRECOGNIZED_SIGNAL` | 274 | 1.1% |
| `RESOURCE_LIMIT` | 233 | 0.9% |
| **Total** | **24,981** | 100% |

Every file ends in exactly one of these ten outcomes; none is silently dropped.

### Leaf classes by split (problem-level 60/20/20 split from M2)

| Class | Train | Val | Test | Total | Share of labelled |
|---|---:|---:|---:|---:|---:|
| `SIGSEGV` | 1,213 | 474 | 445 | 2,132 | 49.0% |
| `SIGABRT` | 746 | 229 | 219 | 1,194 | 27.5% |
| `SIGFPE` | 391 | 78 | 110 | 579 | 13.3% |
| `NZEC` | 311 | 67 | 66 | 444 | 10.2% |
| **Labelled** | **2,661** | **848** | **840** | **4,349** | 100% |

The labelled rate is similar across splits (train 17.7%, val 17.0%, test
16.8%), so the exclusions did not distort the split proportions.
`NZEC` and `SIGFPE` are the thin classes (66–110 test examples); the model
evaluation must report per-class F1 with that caveat.

---

## 2. Method

For each `RUNTIME_ERROR` file in the frozen manifest:

1. **Recompile** with the Linux g++ used in M4 (`-std=gnu++17`, no
   optimisation flags) plus `-fno-stack-protector` (see §4). A file that does
   not compile is a Tier-2 mismatch → `COMPILE_FAILED`.
2. **Execute** the binary once, with the problem's sample input on stdin
   (CodeNet `derived/input_output/data/<problem>/input.txt`), or an empty stdin
   if the problem has none.
3. **Observe** the return code, whether the wall-clock limit fired, and the
   tail of stderr.
4. **Classify** with the pure function `sdp.data.labeling.tier3.classify()`.

The split of responsibilities mirrors M4: `scripts/tier3_worker.py` (runs
inside WSL/Linux, standard library only) captures raw facts;
`scripts/run_tier3.py` (Windows side) batches the work and applies
`classify()`. Because raw results are stored, classification rules can change
without re-executing 25,000 programs — this was used during the pilot (§4).

### Why execution runs inside WSL, in batches
Signals are only reported faithfully by Linux. Going through `wsl.exe` once per
file would flatten a signal into an exit code (139 for SIGSEGV) and, as M4
showed, produces phantom launch failures. The worker is launched once per batch
of 50 files and uses `subprocess` natively, where a signal death appears as a
negative return code (-11, -8, -6).

### Classification rules (`tier3.classify`)

Rules are applied in this order; order matters because a process the worker
killed itself has a meaningless return code.

| Condition | Outcome |
|---|---|
| compile failed | `COMPILE_FAILED` |
| worker wall-clock timeout | `TIMEOUT` |
| return code 0 | `EXIT_ZERO` |
| return code > 0 | `NZEC` |
| signal 11 | `SIGSEGV` |
| signal 8 | `SIGFPE` |
| signal 6 + `bad_alloc` in stderr | `RESOURCE_LIMIT` |
| signal 6 + libstdc++ assertion or "stack smashing detected" in stderr | `TOOLCHAIN_HARDENING` |
| signal 6, otherwise | `SIGABRT` |
| signal 24 (CPU limit) | `TIMEOUT` |
| signal 25 (file-size limit) | `RESOURCE_LIMIT` |
| any other signal | `UNRECOGNIZED_SIGNAL` |

As in Tier 2, unmatched input returns a review flag rather than raising, so one
odd file cannot stop a 25,000-file run. Signal numbers are Linux literals
because the module is imported by tests on Windows, where several `signal.*`
names do not exist.

### Sandbox (deliberately simple, no containers)
Submissions are short single-file competitive-programming programs run on the
team's own machine, so OS-level limits are sufficient:

| Control | Value |
|---|---|
| Working directory | fresh temp dir on the Linux filesystem (not `/mnt/d`) per file |
| Compile timeout | 30 s |
| Wall-clock run timeout | 10 s, then whole process group `SIGKILL`ed |
| `RLIMIT_CPU` | 5 s soft / 6 s hard (soft < hard so the limit raises `SIGXCPU`, not `SIGKILL`) |
| `RLIMIT_AS` | 2 GiB |
| `RLIMIT_FSIZE` | 10 MB |
| `RLIMIT_CORE` | 0 (no core dumps) |
| stdout | discarded (`/dev/null`) |
| stderr | written to a capped temp file; last 1,000 characters kept |

`RLIMIT_NPROC` was deliberately not used: it counts every process of the Linux
user and could break the host shell. Network isolation (`unshare -n`) was not
added: these programs do not use the network, and the temp directory plus
resource limits already contain what matters.

---

## 3. Input availability and its effect

CodeNet ships one sample input per problem (the first sample from the problem
statement, **not** the judges' hidden tests): 3,867 problems, of which 3,574
are "verified" and 272 "unverified" per the dataset README. In the reserve,
97.0% of files belong to a problem with a sample input.

| `input_source` | Files | Labelled | Rate |
|---|---:|---:|---:|
| `sample_verified` | 23,063 | 4,026 | 17.5% |
| `sample_unverified` | 1,179 | 130 | 11.0% |
| `empty_stdin` | 739 | 193 | 26.1% |

`empty_stdin` has the *highest* rate, which is not a sign of better data: a
program that reads input and receives none works with uninitialised values and
crashes for reasons unrelated to the original judge failure. These 193 files
(4.4% of labelled) are kept but flagged via `input_source`, so evaluation (M8)
can report results with and without them.

---

## 4. Toolchain finding: hardened compiler aborts (important)

The first 500-file pilot (v1) labelled 69 files `SIGABRT`. Inspecting stderr
showed 44 of them were `std::vector::operator[]` / `std::string::operator[]`
assertion failures printed from `/usr/include/c++/…`. The team's compiler is
Ubuntu g++ 15.2.0, whose libstdc++ aborts on an out-of-range container access;
a plain judge build would segfault or corrupt memory silently instead. The
label `SIGABRT` therefore reflected a compiler setting, not the original
failure — a model cannot learn that from source code.

Attempts to switch the check off, all measured on the team's machine, **all
still aborted**: `-U_GLIBCXX_ASSERTIONS`; an `#undef` header forced with
`-include`; `-fno-hardened`; `-U_GLIBCXX_HARDEN`; `-D_GLIBCXX_HARDEN=0`.

Decision: exclude such files as `TOOLCHAIN_HARDENING` (stderr contains both
`/usr/include/c++/` and `Assertion`, or "stack smashing detected") instead of
labelling them. A genuine user-written `assert()` names the submission's own
file, not `/usr/include/c++/`, so it correctly stays `SIGABRT`; this is covered
by tests. `-fno-stack-protector` is still passed so stack overflows segfault as
they would on a plain build.

Effect (same 500 files): `SIGABRT` 69 → 25, `TOOLCHAIN_HARDENING` 46, other
classes essentially unchanged (SIGSEGV 42 → 44, NZEC 10 → 10, SIGFPE 9 → 8).
The v1 pilot outputs are kept as `reports/m5_pilot_v1_*.csv` as evidence.

What the `SIGABRT` class now contains (full run, first stderr line; these six
causes cover 1,111 of the 1,194 files): uncaught
`std::out_of_range` (517), user `assert()` (369), `std::length_error` (107),
`free(): invalid pointer` (52), `std::invalid_argument` (42), plain `abort()`
with no message (24). All of these are genuine abort causes; the remaining 83
files were not individually inspected.

---

## 5. Limitations (state these plainly in the report)

1. **66.6% of files did not reproduce.** The original verdict came from the
   judge's hidden tests; one sample input often does not trigger the failing
   path. Retained files are therefore the crashes that the sample input
   exposes — a measured selection effect, not a defect of the pipeline.
2. **Under-representation of container out-of-range errors.** 2,395 files
   (35.5% of all crashes that reproduced: 2,395 of 6,744) were excluded as
   `TOOLCHAIN_HARDENING`. These are mostly `std::vector` out-of-range accesses,
   probably the most common C++ runtime defect. They are excluded, not
   relabelled; raw-array overruns still appear as `SIGSEGV`. The trained model
   has no example of the container-out-of-range pattern.
3. **Labels are toolchain-conditioned.** They describe behaviour under Ubuntu
   g++ 15.2.0 at `-O0`; an optimised build can change how undefined behaviour
   crashes. Consistent with M4, no `-O2` was used.
4. **Single execution per file.** A stability check (re-executing 200 files)
   found 0 changed outcomes, but it was run on the v1 pilot configuration,
   before the hardening change, which altered only compile flags and
   classification rules.
5. **`UNRECOGNIZED_SIGNAL` (274, 1.1%) has no class.** 262 are `SIGILL`
   (return code -4) and 12 are `SIGBUS` (-7). The taxonomy has no class for
   either, and the cause was not investigated (the pilot's `SIGILL` cases had
   empty stderr). They are reported, not labelled.
6. **Thin classes.** `NZEC` (66 test) and `SIGFPE` (110 test) support only wide
   per-class confidence intervals.
7. **Compile mismatch.** 739 files (3.0%) compile in the original toolchain but
   not under g++ 15.2 (cause not investigated; likely the newer compiler's
stricter rules).

---

## 6. Reproducing

Environment: Windows + WSL, Ubuntu g++ 15.2.0 (`15.2.0-16ubuntu1`), Python 3
in WSL, seed 42 for the pilot sample.

```powershell
python -m pytest tests/test_tier3.py tests/test_tier3_worker.py -q   # 39 tests
python scripts/run_tier3.py                  # 500-file pilot
python scripts/run_tier3.py --rerun 200      # stability check
python scripts/run_tier3.py --sample 0       # full reserve (~5.5 h, resumable)
python scripts/run_tier3.py --sample 0 --summary-only
```

| File | Purpose |
|---|---|
| `src/sdp/data/labeling/tier3.py` | outcome vocabulary + pure `classify()` |
| `scripts/tier3_worker.py` | Linux-side compile + sandboxed run, raw capture |
| `scripts/run_tier3.py` | batching, resume, classification, summary |
| `tests/test_tier3.py` | unit tests for the classifier |
| `tests/test_tier3_worker.py` | synthetic crash programs through the whole chain |
| `reports/m5_full_reserve_raw.csv` | raw return codes / stderr tails per file |
| `reports/m5_full_reserve_classified.csv` | raw + manifest columns + `outcome` |

---

## 7. Deviations from Master Plan v3 §M5

| Plan v3 | What was done | Reason |
|---|---|---|
| Sandbox with `preexec_fn` limits, no container | Same, run inside WSL via a worker script | `wsl.exe` flattens signals; native Linux `subprocess` reports them faithfully |
| Execute against input where available, else empty stdin | Same; `input_source` recorded per file | lets evaluation separate input quality |
| Map signal / non-zero exit to four leaves | Plus five exclusions and a toolchain-artifact rule | our own limits and compiler hardening can create false labels |
| Report actual reproduction rate | 17.4% measured | as required |
| Confirm sandbox with guide | Guide delegated technical decisions to the team | recorded 2026-10-06 |

---

## 8. Viva preparation

- **Why only 17.4% of the reserve?** Because the judge's hidden tests caused
  most of the original crashes and we only have one sample input per problem.
  We measured this rather than assuming it, and it is why the reserve was sized
  at 25,000 in the first place.
- **Why exclude `TOOLCHAIN_HARDENING` instead of labelling it `SIGSEGV`?** We
  cannot know what a plain judge build would have done; labelling it would make
  the class depend on a compiler setting. We tried five flags to disable the
  check; none worked.
- **How do you know a label is real?** Each label is a directly observed OS
  outcome (signal or exit code) under a documented toolchain, produced by a
  tested pure function, with every unlabelled file carrying a recorded reason.
- **Is it safe to run submitted code?** Short single-file programs on the
  team's own machine, each in a disposable directory with CPU, memory, file-size
  and wall-clock limits and process-group kill; no containers because the
  threat model does not need them.
- **Why `SIGABRT` is not just "assertion failed"?** Library assertions are
  excluded; what remains are uncaught exceptions, user assertions, heap
  corruption detected by glibc, and explicit `abort()`.
