# Tier 2 Label Derivation: Compile-Error Sub-classification

**Milestone:** M4 (Master Implementation Plan v3)
**Status:** COMPLETE. All 29,876 files in the `COMPILE_ERROR` reserve have
either a Tier-2 leaf label or a recorded exclusion reason. §1 contingency
decided. §7-8 below record the full-reserve pattern validation and final
results.

## 1. Purpose & Scope

Tier 2 turns the frozen 30,000-file `COMPILE_ERROR` reserve (sized for exactly
this purpose in M2, see `docs/LABELING.md` §2) into three trained-on leaf
labels: `SYNTAX`, `SEMANTIC`, `LINKER`. This follows the diagnostic-string
parsing approach already piloted in `docs/LABELING.md` §5, applied at full
scale, per M4's complexity guardrail (§1 of the Master Implementation Plan):
diagnostic-string parsing on compiler stderr, not a learned classifier.

`docs/LABELING.md` §5's original pilot used an outdated GCC 6.3.0 toolchain
and flagged its 1.24% LINKER rate as a **lower bound** requiring
remeasurement with a current compiler before the full-reserve pipeline could
be trusted. §2–§5 record that remeasurement and two toolchain bugs fixed
along the way. §6 records the §1 contingency decision. §7–§8 record the
production module, the full-reserve run (including a data-integrity issue
found and fixed), and the final results.

## 2. Toolchain

Three compilers were used over the course of this work; one is used for the
actual M4 pipeline.

| | Path / invocation | Version | Role |
|---|---|---|---|
| Original pilot compiler | `C:\MinGW\bin\g++.exe` | GCC 6.3.0 (MinGW.org, Windows) | Historical only — produced the original `docs/LABELING.md` §5 pilot result. |
| Windows current compiler | `D:\Dev\Toolchains\mingw64\bin\g++.exe` | GCC 16.2.0 (MinGW-w64 x86_64 UCRT) | Used for an initial remeasurement, then ruled out for the real pipeline (§4). |
| **Linux compiler (used for M4)** | `wsl g++` (existing Ubuntu WSL distro) | GCC 15.2.0 (Ubuntu 15.2.0-16ubuntu1) | **The compiler used for the full-reserve Tier-2 pipeline.** Matches the Linux-based judge environments (AtCoder/AIZU) IBM CodeNet's submissions were originally judged on. |

Compiler standard: `-std=gnu++17` throughout. WSL was reused from an existing
installation; no new virtualization or container infrastructure was
introduced.

## 3. LINKER Remeasurement Methodology

The original pilot's 1.24% figure came from a fixed 500-file sample
(`reports/linker_pilot.csv`, seed 42). Recompiling that *exact* sample
isolates the compiler as the only variable under test. Two submissions don't
join to the final M2 manifest and are excluded with recorded reasons
(`s198272745`: confirmed cross-split dedup drop; `s363457044`: untraceable
absence). Working pilot sample: n = 498. 98/500 files are "agreement-filter"
discards (Tier-1 `COMPILE_ERROR` didn't reproduce), matching `docs/LABELING.md`
§5's own 19.6% secondary finding almost exactly — strong corroborating
evidence both measurements are sound.

## 4. Two Toolchain Bugs Found and Fixed During Pilot Remeasurement

**Bug 1 — LINKER marker didn't match Windows' program naming.** Windows'
linker driver reports itself as `collect2.exe`, not `collect2`, so a marker
requiring the literal string `"collect2: error: ld returned"` silently missed
genuine LINKER failures. Fixed by matching on the platform-independent token
`"ld returned"`.

**Bug 2 — Windows PE/COFF relocation-overflow is a target-platform artifact,
not a code defect.** Several pilot files failed with `relocation truncated to
fit: IMAGE_REL_AMD64_REL32` — a Windows-specific linker limitation on 32-bit
relative relocations, unrelated to code correctness. `-mcmodel=medium/large`
did not resolve it (expected on Windows PE). Since CodeNet's submissions were
judged on Linux, compiling for Windows PE at all was the wrong target
platform.

**Resolution:** switched to Linux GCC via WSL. All affected files compiled
cleanly under Linux GCC 15.2.0, confirming the failures were Windows-target
artifacts.

## 5. Pilot Remeasurement Results (Linux GCC 15.2.0)

| | Original (GCC 6.3.0, Windows) | Linux (GCC 15.2.0, WSL) |
|---|---|---|
| Reproducing-failure sample size | 402 | 400 |
| LINKER count | 5 | 5 |
| LINKER rate | 1.24% (documented) | **1.25%** |

All 4 files that were LINKER in the original pilot are still LINKER under
Linux GCC — zero flips. One additional file is newly LINKER for a
genuinely different, legitimate reason: an ELF relocation overflow from
large fixed-size global arrays — a real code characteristic, not a
toolchain artifact. LINKER therefore has (at least) two legitimate
underlying causes, both correctly grouped under one flat leaf class.
**On the correct target platform, the LINKER rate matches the original
1.24% figure almost exactly.**

## 6. §1 Contingency Decision

**Decision (finalized by Prashant, 2026-09-26): keep `LINKER` as its own 9th
class. Do not merge it into `SEMANTIC`.**

**Basis:** a stable, reproducible ~1.25% signal with zero flips on the
original files, plus a second independently legitimate cause — a genuinely
distinctive diagnostic pattern, not a noisy artifact. Merging it into
`SEMANTIC` would dilute signal the model could otherwise learn. Class
imbalance is handled via the already-planned M7 class-weighted
cross-entropy loss (no new infrastructure), documented as a stated
limitation in M8/M11.

**Final taxonomy (confirmed, 9 classes):** `ERROR_FREE`, `LOGICAL`, `SYNTAX`,
`SEMANTIC`, `LINKER`, `SIGSEGV`, `SIGFPE`, `SIGABRT`, `NZEC`.

## 7. Production Module and Full-Reserve Pattern Validation

**Module:** `src/sdp/data/labeling/tier2.py`, tested in `tests/test_tier2.py`
(47/47 passing). Mirrors `tier1.py`'s structure (`StrEnum` vocabulary tables,
`Final` tuples, a pure classification function with no I/O). One deliberate
deviation from `tier1.py`: `classify()` returns `Tier2ReviewFlag.UNRECOGNIZED`
for unmatched diagnostics rather than raising, since compiler diagnostic text
is open-vocabulary (unlike Tier 1's closed 12-verdict set) — documented
explicitly in the module docstring as a conscious difference, not an
inconsistency.

**Pattern-table provenance — two evidence passes:**

*Pass 1 (pilot-scale).* Validated against the 498-file pilot sample's 385
`OTHER_COMPILE_ERROR` files. Iterating on real GCC output (including fixing
a Unicode "smart quote" bug — GCC quotes identifiers with `'`/`'`
(U+2018/U+2019), not the ASCII apostrophe, which silently broke ~24 pattern
matches until text was normalized) brought UNRECOGNIZED from 147 → 63 → 16
(4.2%) over three iterations, plus a `MISSING_DEPENDENCY` allowlist splitting
genuine external dependencies (`atcoder/`, `boost/`, `stdafx.h`, `pch.h`,
`iostream.h`) from corrupted/typo'd `#include`s (reclassified `SYNTAX`, a
real defect).

*Pass 2 (full-reserve-scale).* Per the option-B decision (batch-review the
real full-scale UNRECOGNIZED list and extend patterns before falling back to
documented exclusion), the pilot-validated rules were run against the full
29,876-file reserve, and the real residual was clustered by normalized
message template (variable parts — paths, line:col numbers, quoted
identifiers — stripped out) to find high-frequency gaps rather than reading
individual lines. This surfaced ~30 additional patterns, each added only
after recurring **≥12 times** in the actual reserve — evidence from the real
target population, not an extrapolation from a sample. Representative
additions: `SYNTAX` — `"too many decimal points"`, `"empty character
constant"`, `"without a previous"`, `"two or more data types"`; `SEMANTIC` —
`"conflicting declaration"`, `"return-statement with"`, `"cannot bind"`,
`"static assertion failed"`, `"narrowing conversion"`, and ~20 others. A
generalized `*_BROAD_PAIRS` mechanism was added (extending the existing
`"expected"`+`"before"` conjunction rule) for messages whose exact wording
varies too much for a literal substring match, e.g. `"template argument N is
invalid"`.

## 8. Full-Reserve Run and Final Results

**A data-integrity issue found and fixed mid-run.** The full-reserve
orchestration script (`run_tier2_full_reserve.py`) initially enforced its
compile timeout via Python's `subprocess.run(timeout=N)` around a `wsl g++`
invocation. This only has authority over the Windows-side `wsl.exe`
launcher — not the actual `g++` process in WSL's own Linux process tree — so
a killed timeout could leave an orphaned Linux-side process holding the
stdout/stderr pipe open, hanging Python's read indefinitely (resistant to
`Ctrl+C`, since Python was blocked in a low-level pipe read). The run hung
at 26,800/29,876 files. **Fixed** by enforcing the timeout *inside* WSL via
GNU coreutils `timeout -k 5 <N> g++ ...` (real authority within the same
Linux process tree), with Python's own timeout kept only as a larger,
secondary safety net. Checkpointing (every result flushed to disk
immediately, resumable) meant nothing before the hang was lost.

**A second, smaller data-integrity issue, found after the run completed.**
3,179 files showed empty stderr with a returncode (`4294967295`/`0xFFFFFFFF`,
or occasionally `2`/`15`) that `g++` itself never produces — the signature of
`wsl.exe` failing to launch or complete the command under sustained
high-frequency invocation, likely worse immediately after the hang-and-restart
above. These files' true compile outcome was never captured, so no pattern
list could classify them correctly — they were **recompiled** (not
pattern-matched) via a targeted retry script, reusing the same validated
compile logic. All but 1 retried cleanly (323 turned out to be genuine clean
compiles, 1 a genuine timeout — both correctly classified once actually
captured); the retry results were merged into the main dataset, replacing
only the affected rows.

**Final classification, full 29,876-file `COMPILE_ERROR` reserve:**

| Outcome | Count | % of reserve |
|---|---|---|
| SEMANTIC | 15,319 | 51.3% |
| SYNTAX | 6,698 | 22.4% |
| AGREEMENT_FILTER (discard) | 6,446 | 21.6% |
| UNRECOGNIZED (excluded, reviewed) | 535 | 1.79% |
| MISSING_DEPENDENCY (discard) | 394 | 1.32% |
| LINKER | 372 | 1.25% |
| COMPILATION_TIMEOUT (discard) | 112 | 0.37% |
| AMBIGUOUS | 0 | 0% |

**Total Tier-2 labelled: 22,389 files (74.9% of the reserve)** — this is
what feeds M6's unified 9-class dataset assembly.

**Sanity checks, all consistent:**
- Agreement-filter rate: 21.6% vs. the pilot's 19.6% — close.
- LINKER count: 372, landing almost exactly on the pilot's ~375
  extrapolation (§5) — strong corroboration the pilot remeasurement was
  representative.
- UNRECOGNIZED: 1.79%, smaller than the pilot's own 4.2% post-tuning rate —
  expected, since the full-reserve patterns were built from ~60x more real
  evidence than the pilot sample provided.
- Zero AMBIGUOUS: the SYNTAX/SEMANTIC pattern tables remain cleanly
  separated at full scale, not just in the small pilot.

**The remaining 535 UNRECOGNIZED files are excluded from the Tier-2 labelled
set, with the reason recorded as "no matching diagnostic pattern" — this is
a genuine, honestly-reported residual, not a silent drop.** Per the option-B
decision, iteration stopped once the residual reflected a long tail of rare,
mostly one-off messages rather than recurring gaps — the same discipline
already applied to every other exclusion in this project (dedup drops,
agreement-filter discards, missing-dependency files).

**Artifacts:**
`reports/m4_full_reserve_results_final.csv` (authoritative, 29,876 rows) ·
`reports/m4_full_reserve_results_merged.csv` (post phantom-failure fix,
pre-final-pattern-pass) · `reports/m4_retry_results.csv` (the 3,179 retried
files) · `reports/m4_unrecognized_templates_v2.csv` (frequency evidence
behind the pass-2 patterns) · `src/sdp/data/labeling/tier2.py` ·
`tests/test_tier2.py`.

**M4 exit criteria: met.** Every retained `COMPILE_ERROR` file has one of
`SYNTAX`/`SEMANTIC`/`LINKER` or is excluded with a recorded reason; the
LINKER contingency decision is made and documented; per-class counts for the
full reserve are reported.
