# Label Engineering, Deduplication, and Leakage-Safe Splitting

**Milestone 2** — `v0.3-labels`
Group 42 · Design and Implementation of a CodeBERT-Based Framework for Multiclass Software Defect Prediction

This document is the written deliverable for Milestone 2. It records what was
decided, what was measured, and why — as a companion to the code in
`src/sdp/data/` and the tests in `tests/`, and to `data/processed/splits/split_config.json`,
which pins the exact seeds, ratios, and content hash needed to reproduce
everything described here.

---

## 1. Tier-1 label derivation

### 1.1 The verdict vocabulary

Every C++ submission in Project CodeNet carries a `status` field — the judge's
verdict, not a defect label. Milestone 1's EDA found twelve distinct verdict
strings across 8,008,527 rows. Each was assigned an explicit outcome in
`src/sdp/data/labeling/tier1.py`; none are handled implicitly, and an
unrecognised verdict raises rather than being silently dropped.

| Verdict | Count | % of corpus | Outcome |
|---|---:|---:|---|
| `Accepted` | 4,353,049 | 54.36 | → `ERROR_FREE` |
| `Wrong Answer` | 2,571,284 | 32.11 | → `LOGICAL` |
| `Compile Error` | 376,053 | 4.70 | → `COMPILE_ERROR` |
| `Runtime Error` | 339,670 | 4.24 | → `RUNTIME_ERROR` |
| `Time Limit Exceeded` | 326,340 | 4.07 | discarded — `EFFICIENCY_CONSTRAINT` |
| `WA: Presentation Error` | 26,449 | 0.33 | discarded — `OUTPUT_FORMAT` |
| `Memory Limit Exceeded` | 14,637 | 0.18 | discarded — `EFFICIENCY_CONSTRAINT` |
| `Output Limit Exceeded` | 778 | 0.01 | discarded — `EFFICIENCY_CONSTRAINT` |
| `Judge Not Available` | 94 | <0.01 | discarded — `JUDGE_INFRASTRUCTURE` |
| `Query Limit Exceeded` | 88 | <0.01 | discarded — `EFFICIENCY_CONSTRAINT` |
| `Internal error` | 78 | <0.01 | discarded — `JUDGE_INFRASTRUCTURE` |
| `Judge System Error` | 7 | <0.01 | discarded — `JUDGE_INFRASTRUCTURE` |

**Kept: 7,640,056 (95.40%). Discarded: 368,471 (4.60%).**

Two verdict strings have irregular spelling, preserved exactly rather than
normalised, since normalising them would erase a data-quality observation:

- `WA: Presentation Error` — the `WA:` prefix is CodeNet's own; the judge
  classifies presentation errors as a *subtype of Wrong Answer*, which is the
  strongest available argument for discarding it separately from genuine
  `Wrong Answer` rows rather than folding it into `LOGICAL`.
- `Internal error` — lowercase `e`, inconsistent with every other verdict
  string in the vocabulary.

### 1.2 Discard reasons

Three reasons cover all eight discarded verdicts. The reason records *why the
row is excluded from the taxonomy*, not which specific limit was breached —
`status` already preserves that, so any report can be broken down by verdict
and by reason independently.

| Reason | Verdicts | Rationale |
|---|---|---|
| `EFFICIENCY_CONSTRAINT` | TLE, MLE, OLE, Query Limit Exceeded | A resource or query budget was exceeded. The code may be functionally correct; inefficiency is not one of the nine defect classes. |
| `OUTPUT_FORMAT` | WA: Presentation Error | Output is correct but formatted differently — a weaker failure than a genuine wrong answer. |
| `JUDGE_INFRASTRUCTURE` | Judge Not Available, Internal error, Judge System Error | The evaluation itself failed. Says nothing about the submitted code. |

### 1.3 Known limitation: verdict noise in `Wrong Answer → LOGICAL`

`Wrong Answer` means the output disagreed with the judge's test cases. Usually
this reflects a genuine logic error, but it also fires on mishandled edge
cases, subtle output-formatting mismatches, and abandoned partial attempts.
Symmetrically, `Accepted → ERROR_FREE` means the code passed *these* test
cases, not that it is provably correct — a logic bug that happens not to be
exercised by the judge's tests is still labelled `ERROR_FREE`.

This is stated as a limitation, not an oversight. **Judge verdicts record what
happened at runtime against one test set; defect types describe a property of
the code.** The mapping between the two is many-to-many and test-data
dependent, and the resulting label noise is irreducible without re-annotating
by hand — which is precisely the manual-annotation dependency this project
replaces with CodeNet's outcome-derived labels in the first place.

### 1.4 Coarse-to-leaf hierarchy

Tier 1 resolves the four coarse parent classes. Two are terminal
(`ERROR_FREE`, `LOGICAL` — identical to their leaf-level namesakes); two fan
out at later tiers:

```
                    ROOT
      ┌──────────┬────┴─────┬──────────────┐
  ERROR_FREE   LOGICAL   COMPILE_ERROR   RUNTIME_ERROR     ← Tier 1 (this milestone)
                          ┌───┼────┐    ┌──┬──┬───┬──┐
                        SYN SEM LINK  SEGV FPE ABRT NZEC   ← Tiers 2 & 3 (Milestone 4+)
```

`COMPILE_ERROR` splits into SYNTAX / SEMANTIC / LINKER via g++ diagnostic
parsing (Tier 2). `RUNTIME_ERROR` splits into SIGSEGV / SIGFPE / SIGABRT /
NZEC via sandboxed execution (Tier 3, conditional on the guide's assessment of
its complexity-to-marks ratio). Both the coarse and leaf vocabularies are
declared in `src/sdp/data/taxonomy.py` from the start, so no downstream module
requires refactoring when Tiers 2 and 3 are implemented.

---

## 2. Working corpus size and allocation

The candidate pool (7,640,056 rows) is far larger than a BE final-year project
needs or can process on local hardware. A working corpus of **75,000 files**
was sampled, sized so that only the two classes that fan out at later tiers
carry a reserve — `ERROR_FREE` and `LOGICAL` are terminal and gain nothing
from over-sampling.

| Class | Sampled | Rationale |
|---|---:|---|
| `ERROR_FREE` | 10,000 | Terminal class |
| `LOGICAL` | 10,000 | Terminal class |
| `COMPILE_ERROR` | 30,000 | Fans out 3 ways at Tier 2; LINKER is the scarce sub-class (§5) |
| `RUNTIME_ERROR` | 25,000 | Fans out 4 ways at Tier 3; 25.9% truncation rate in M1 EDA implies higher attrition |

Each class total is split 60/20/20 across train/val/test independently of the
split ratios themselves, so class balance and split proportions cannot
perturb each other (`src/sdp/data/sampling.py::resolve_quotas`).

---

## 3. Leakage-safe splitting

### 3.1 The problem

CodeNet contains resubmission chains: a user submits, receives a verdict,
edits a few characters, resubmits. Within the candidate pool:

- **3,992,111** distinct (user, problem) pairs
- **1,364,569 (34.2%)** received more than one submission
- **5,012,514 (65.6%)** of all candidate submissions belong to such a chain

Under a naive submission-level (random) split, each multi-submission chain has
roughly a 50% chance of straddling the train/test boundary. Measured directly
against the frozen split:

| | Chains straddling the boundary |
|---|---:|
| Problem-level split (primary) | **0** |
| Random submission-level split (comparison arm) | **979,741** |

**71.8% of multi-submission chains would have crossed the train/test boundary
under a random split.** Each such pair places near-identical source code —
sometimes with opposite verdicts — on both sides of the evaluation, letting a
model score well by memorising the problem rather than learning defect
signatures. Problem-level splitting eliminates this by construction: every
submission to one `problem_id` is assigned to exactly one split.

### 3.2 Balanced allocation

Problem sizes are severely right-skewed (median 357 submissions per problem,
maximum 39,830; the ten largest problems hold 3.6% of all candidates). Random
problem assignment would therefore give no control over the resulting
*submission* proportions — naive random allocation deviates from the 60/20/20
target by roughly ±1.5 percentage points depending on seed.

`allocate_problems` (`src/sdp/data/splitting.py`) uses longest-processing-
time-first greedy allocation: problems are shuffled with the seed, sorted
largest-first, and each is assigned to whichever split currently holds the
largest deficit against its submission target. Achieved on the real corpus
(seed 42):

| Split | Target rows | Achieved rows | Achieved % | Problems |
|---|---:|---:|---:|---:|
| train | 4,584,033.6 | 4,584,034 | 60.00 | 1,528 |
| val | 1,528,011.2 | 1,528,011 | 20.00 | 1,252 |
| test | 1,528,011.2 | 1,528,011 | 20.00 | 1,252 |

Achieved proportions are exact to the row. Note the asymmetry between problem
counts and submission share: train holds only 1,528 of 4,032 problems (37.9%)
but 60% of all submissions, because the greedy allocation assigns the largest
problems first and needs fewer of them to hit its target.

### 3.3 Class composition across splits

| Class | Corpus-wide | train | val | test | Max deviation |
|---|---:|---:|---:|---:|---:|
| ERROR_FREE | 56.98% | 56.11% | 58.68% | 57.88% | 1.70 pp |
| COMPILE_ERROR | 4.92% | 4.98% | 5.01% | 4.67% | 0.25 pp |
| RUNTIME_ERROR | 4.45% | 4.36% | 4.41% | 4.73% | 0.28 pp |
| LOGICAL | 33.66% | 34.55% | 31.90% | 32.72% | 1.76 pp |

Allocation balances on total submissions only, not per-class. The two
majority classes swing up to ~1.8pp against each other; the two minority
classes stay within 0.3pp. Since sampling (§2) draws a fixed count per class
per split regardless of the split's natural composition, this deviation does
not affect the working corpus — every (split, class) cell had at minimum
**11.9×** the required supply (§4). It does apply to the natural-distribution
test set, which stays within 0.9pp of the corpus-wide reference on every
class.

### 3.4 Split freezing

Seed 42, ratios 60/20/20, method `problem_level_split`. Frozen in
`data/processed/splits/split_config.json` alongside a SHA-256 of the final
manifest. The seed and ratios must not change once model training begins —
re-splitting invalidates every reported metric.

---

## 4. Sampling: diversity over volume

### 4.1 Why not uniform random sampling

Drawing `n` rows uniformly at random from a (split, class) cell draws in
proportion to problem size. Given the concentration in §3.2, a uniform sample
would be dominated by a few hundred large problems, narrowing the working
corpus to a small slice of algorithmic tasks and inviting the model to learn
problem-specific idioms rather than general defect signatures.

### 4.2 Round-robin allocation

`draw_sample` (`src/sdp/data/sampling.py`) takes one submission from every
available problem in a cell, then a second from every problem, and so on
until the quota is met — implemented as a vectorised stable-sort operation,
not a Python loop. Every problem contributes before any problem contributes
twice.

Achieved on the real corpus (seed 42), with zero shortfall in all twelve
(split, class) cells:

| Split | Class | Rows | Problems drawn from | Max per problem |
|---|---|---:|---:|---:|
| train | ERROR_FREE | 6,000 | 1,523 | 4 |
| train | COMPILE_ERROR | 18,000 | 1,314 | 18 |
| train | RUNTIME_ERROR | 15,000 | 1,260 | 16 |
| train | LOGICAL | 6,000 | 1,412 | 5 |
| val | ERROR_FREE | 2,000 | 1,248 | 2 |
| val | COMPILE_ERROR | 6,000 | 1,045 | 7 |
| val | RUNTIME_ERROR | 5,000 | 1,006 | 6 |
| val | LOGICAL | 2,000 | 1,137 | 2 |
| test | ERROR_FREE | 2,000 | 1,246 | 2 |
| test | COMPILE_ERROR | 6,000 | 1,048 | 7 |
| test | RUNTIME_ERROR | 5,000 | 1,013 | 6 |
| test | LOGICAL | 2,000 | 1,151 | 2 |

**All 4,032 candidate problems (100%) are represented in the working corpus.**
No single problem contributes more than 43 of the 75,000 sampled rows
(0.057%), compared with the largest problem's 0.52% share of the raw
candidate pool — round-robin allocation actively flattens the concentration
identified in the Milestone 1 EDA, rather than merely tolerating it.

---

## 5. LINKER-class feasibility pilot

`COMPILE_ERROR` is planned to split into SYNTAX / SEMANTIC / LINKER at
Milestone 4. Before committing 30,000 files to that reserve, a pilot measured
whether LINKER is viable as a trainable class at all.

**Method.** 500 `COMPILE_ERROR` files sampled from the working corpus (seed
42) were recompiled with `g++ -std=gnu++17`, and stderr was checked for
`undefined reference` or `collect2: error: ld returned` as a linker-stage
signal. This is not full Tier 2 classification — SYNTAX and SEMANTIC are not
distinguished — only linker-stage-or-earlier.

**Compiler used:** `g++ (MinGW.org GCC-6.3.0-1) 6.3.0` — the only toolchain
available at pilot time. This compiler is outdated (released 2016,
unmaintained upstream since ~2018) and defaults to an older C++ standard than
the judges that originally evaluated these submissions (many post-2019
AtCoder submissions were judged under GCC 9 with C++17). An older front-end
may reject valid modern syntax at the parse stage before ever reaching the
linker, misclassifying some true LINKER failures as earlier-stage errors. **All
figures below are therefore a lower bound on the true LINKER rate.**

**Results:**

| Outcome | Count | % of sample |
|---|---:|---:|
| Compiled clean (verdict not reproduced) | 98 | 19.6% |
| Failed to compile | 402 | 80.4% |
| — classified LINKER | 5 | 1.0% (of 500) / 1.24% (of 402 reproduced failures) |
| — classified OTHER | 397 | — |

**Projected LINKER examples in the full 30,000-file reserve: ~300** (lower
bound), splitting to roughly 180 train / 60 val / 60 test.

**Interpretation.** LINKER is workable but thin — 180 training examples for
one of nine classes will produce a noisy per-class F1, and with only 60 test
examples each individual misclassification shifts that class's F1 by over a
percentage point. This is recorded as a known limitation rather than deferred
silently. **Decision:** hold the 30,000-file reserve as-is; do not extract
additional files or build targeted LINKER-collection infrastructure now. The
rate will be re-measured with a current GCC toolchain (WinLibs mingw-w64, GCC
14) when Tier 2 is implemented at Milestone 4. If the re-measured rate remains
very low, class weighting or a targeted second extraction pass will be
considered then, using the versioned `corpus_v2` extraction pattern rather
than mutating the frozen `corpus_v1` working corpus.

**Secondary finding.** 19.6% of sampled `COMPILE_ERROR` files compiled
successfully under this different compiler and flag set. This is direct
evidence for the agreement-filter principle planned for Tier 2: recompilation
does not always reproduce the original judge's verdict, and Tier 2's label
derivation must account for this attrition rather than assume every
`Compile Error` row will still fail to compile.

---

## 6. Deduplication and the cross-split content audit

### 6.1 Method

Exact SHA-256 hashing on lightly normalised bytes: CRLF → LF, BOM stripped,
trailing whitespace stripped per line. Nothing further — no comment stripping,
no whitespace collapsing beyond trailing. Comment stripping and similar
transformations are training-time preprocessing decisions (Milestone 3),
applied to train only, after the split is frozen; performing them here would
blur the split/preprocessing boundary the project's leakage-safety argument
depends on.

MinHash-LSH near-duplicate detection was considered and dropped (documented
decision, Milestone 2 planning). Problem-level splitting already eliminates
the dominant near-duplicate vector — resubmission chains — by construction,
since every submission to one problem lands in one split. What remained to
check was (a) rare byte-identical duplicates across *different* problems, and
(b) whether any such duplicate crosses a split boundary. Exact hashing answers
both without a threshold to defend at viva.

### 6.2 Duplicate rate in the raw 75,000-file sample

| Metric | Value |
|---|---:|
| Files hashed | 75,000 |
| Unique content hashes | 73,615 |
| Duplicate groups (size > 1) | 1,070 |
| Files in duplicate groups | 2,455 |
| Redundant files | 1,385 |
| Duplicate rate | 1.85% |

The largest duplicate group (100 files, spread across 87+ problems) is
degenerate content — a single-character file (`"0"`), not a genuine repeated
solution. This reflects a data-quality property of CodeNet itself (empty or
placeholder submissions), not an artefact of sampling.

### 6.3 Cross-split content audit

`cross_split_duplicate_hashes` / `assert_no_cross_split_duplicates`
(`src/sdp/data/dedup.py`) check whether any content hash appears in more than
one split — a stronger, independent claim from `problem_id` disjointness,
since two *different* problems could in principle share byte-identical
submitted code.

**Before exclusion:** 66 distinct hashes (326 rows, 0.435% of the corpus)
appeared in more than one split.

**Investigation.** Of the 66 colliding hashes, 56 carried a single consistent
label across every occurrence (largely reused competitive-programming
boilerplate, e.g. `#include <bits/stdc++.h>\nusing namespace std;` starter
templates that also happen to be complete, correct solutions to more than one
simple problem). **10 hashes carried more than one label** — identical source
code labelled differently depending on which problem it was submitted to.
**Every one of these 10 ambiguous groups involved `RUNTIME_ERROR`** paired
with another class (`LOGICAL`, `COMPILE_ERROR`, or `ERROR_FREE`).

This is not noise; it is direct, concrete evidence for the project's central
labelling argument. `RUNTIME_ERROR` depends on the input a program is run
against — the same source can crash on one problem's test data and complete
successfully (correctly or not) on another's. `COMPILE_ERROR` and `SYNTAX`
type failures are properties of the source text alone and cannot exhibit this
pattern; `RUNTIME_ERROR` can, and in this corpus, does. **Judge verdicts are
properties of a (code, problem, test data) triple, not of code in isolation.**

**Resolution.** All 66 colliding hashes were excluded uniformly (not only the
10 ambiguous ones — a single, simple rule was preferred over one requiring a
label-consistency exception to justify). For each colliding hash, the
occurrence in the split holding the plurality of that hash's rows was
retained; all other occurrences were dropped. **150 of 75,000 rows (0.2%)**
were removed; no (split, class) cell moved by more than 49 rows against its
target (worst case: val/COMPILE_ERROR, 6,000 → 5,951).

**Final working corpus: 74,850 rows.** Re-running the audit on this set:

| Split pair | Colliding content hashes |
|---|---:|
| train & val | 0 |
| train & test | 0 |
| val & test | 0 |

Content-level split disjointness confirmed, independent of and in addition to
the `problem_id`-level disjointness established in §3.

---

## 7. Encoding

A 2,000-file random sample (seed 42) of the extracted working corpus was
checked for UTF-8 decodability, following up on a risk flagged in Milestone 1
(possible Shift-JIS-encoded Japanese comments in AtCoder submissions).

**Result: 2,000 / 2,000 (100%) decoded cleanly as UTF-8**, including all 255
files (12.75%) containing non-ASCII bytes. No file required a Shift-JIS
fallback; none were undecodable.

**Policy:** all source files are decoded as UTF-8. No fallback decoder or
per-file encoding metadata is needed in Tier 2 or the tokenisation pipeline.

---

## 8. Byte-level extraction integrity

Extraction was verified against the 75,000-file sample manifest before any
further processing:

- **File count:** 75,000 / 75,000 extracted; 0 missing, 0 extra relative to
  the sample manifest.
- **Byte totals:** 75,000 / 75,000 files matched `code_size` from the CodeNet
  metadata exactly (100.00%). This confirms `code_size` was recorded in raw
  bytes for this corpus, not characters, and that extraction introduced no
  truncation or encoding corruption.

---

## 9. Environment note

Windows Defender's real-time on-access scanner caches per-file scan results
with limited capacity. Once that cache is exceeded during a long run over many
small files, every subsequent file open blocks on a synchronous scan,
degrading throughput by roughly **40×** (observed: 2,780 files/sec → as low as
61–70 files/sec during the Milestone 2 dedup hashing pass, turning a
sub-30-second operation into 6.5 minutes). This is an environment artefact,
not a code defect, and was resolved by excluding `data/` and the CodeNet
archive directory from real-time scanning:

```powershell
Add-MpPreference -ExclusionPath "<repo>\data"
Add-MpPreference -ExclusionPath "<CodeNet archive directory>"
```

Recorded here so the fix is not rediscovered by every contributor who clones
the repository and runs a full pass over the extracted corpus.

---

## 10. Summary of frozen artefacts

| Artefact | Rows | Location |
|---|---:|---|
| Labelled candidate pool | 7,640,056 | `data/interim/labeled_manifest.parquet` (gitignored) |
| Split manifest (problem-level + random comparison) | 7,640,056 | `data/processed/splits/split_manifest.parquet` (gitignored) |
| Sampled working corpus (pre-dedup) | 75,000 | `data/interim/sample_manifest.parquet` (gitignored) |
| **Final deduplicated working corpus** | **74,850** | `data/processed/splits/sample_manifest_hashed.parquet` (gitignored) |
| Reproducibility record (seeds, ratios, hashes) | — | `data/processed/splits/split_config.json` (**committed**) |
| Discard summary | 8 | `reports/discard_summary.csv` (committed) |
| Sampling report | 12 | `reports/sampling_report.csv` (committed) |
| Dedup report | 7 | `reports/dedup_report.csv` (committed) |
| LINKER pilot raw results | 500 | `reports/linker_pilot.csv` (committed) |

Raw data files are gitignored throughout; every number in this document is
reproducible from `data/raw/codenet_cpp_metadata.parquet` (immutable, from
Milestone 1) plus the seeds and configuration recorded in
`split_config.json`, and independently verifiable via the committed CSV
reports and the SHA-256 hash of the final manifest.

---

## 11. Open items carried to Milestone 3+

- Re-measure LINKER prevalence with a modern GCC (WinLibs, GCC 14) before
  Tier 2 design; decide on class weighting or a targeted `corpus_v2`
  extraction if the rate remains low.
- Tier 2 (compiler-diagnostic parsing) and Tier 3 (sandboxed execution,
  pending the guide's view) remain out of scope until Milestone 4.
- Confirm PS-1/PS-2 presentation dates with the project guide.
- Fix "defect-prone modules before deployment" wording in the synopsis to
  match the actual snippet-level classification task.
