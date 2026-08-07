# Dataset Summary Report — IBM Project CodeNet

**Project:** Design and Implementation of a CodeBERT-Based Framework for
Multiclass Software Defect Prediction
**Group 42** · Sinhgad Institute of Technology · AY 2026–27
**Milestone 1 deliverable** · Prepared 2026-08-07

---

## 1. Source and provenance

| Field | Value |
|---|---|
| Dataset | IBM Project CodeNet, v1.0.0 |
| Citation | Puri, R. et al. (2021), arXiv:2105.12655 |
| Origin | AIZU Online Judge and AtCoder online judging platforms |
| Retrieved | 2026-08-06 |
| Source URL | `https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz` |
| Archive size | 8,343,137,473 bytes (7.76 GiB) |
| Full corpus | 13,916,868 submissions · 4,053 problems · 55 languages |

Problem, submission, and user identifiers are anonymised at source
(`p#####`, `s#########`, `u#########`).

## 2. Licence

| Component | Licence |
|---|---|
| Dataset (source files and metadata) | CDLA-Permissive-2.0 |
| IBM tooling and scripts | Apache 2.0 |

Under CDLA-Permissive-2.0, a recipient may use, modify, and share the data.
The only condition on sharing is that the licence text accompany the shared
data. The agreement imposes no restriction on *Results* — anything produced
by computational use of the data — so models trained on CodeNet carry no
obligation. Attribution is not a legal requirement of version 2.0, but is
provided here as academic practice.

## 3. Scope: C++ only

**8,008,527 C++ submissions** — 57% of the corpus, matching IBM's published
figure exactly. Of these, 4,353,049 are Accepted, also an exact match.
Both figures independently verify the extraction pipeline.

C++ is not chosen for popularity but by necessity: the nine-class taxonomy
requires a compiled language. LINKER errors require a separate link stage;
SIGSEGV requires raw memory access; SIGFPE requires a hardware integer
division trap. None of these can occur in an interpreted language. The
taxonomy determines the language, not the reverse.

## 4. Verdict distribution (C++)

| Verdict | Count | % |
|---|---|---|
| Accepted | 4,353,049 | 54.36 |
| Wrong Answer | 2,571,284 | 32.11 |
| Compile Error | 376,053 | 4.70 |
| Runtime Error | 339,670 | 4.24 |
| Time Limit Exceeded | 326,340 | 4.07 |
| WA: Presentation Error | 26,449 | 0.33 |
| Memory Limit Exceeded | 14,637 | 0.18 |
| Output Limit Exceeded | 778 | 0.01 |
| Judge Not Available | 94 | <0.01 |
| Query Limit Exceeded | 88 | <0.01 |
| Internal error | 78 | <0.01 |
| Judge System Error | 7 | <0.01 |

*Figure: `reports/figures/verdict_distribution_cpp.png`*

**Two discrepancies against IBM's documented status table**, found during
validation: `Query Limit Exceeded` appears in the data but not in the
published table; and the data uses `Internal error` (lowercase 'e') where
the documentation writes `Internal Error`. Filters in this project use an
explicit allow-list of four verified strings, so neither affects results.

## 5. Label derivation strategy

Judge verdicts are **not** defect classes. A verdict records what happened
when the program was judged; a defect class describes what is wrong with
the code. Two of the twelve observed verdicts are umbrella categories that
between them contain seven of the nine target classes.

| Tier | Verdict | Method | Target classes | Candidates |
|---|---|---|---|---|
| 1 | Accepted | metadata filter | ERROR-FREE | 4,353,049 |
| 1 | Wrong Answer | metadata filter | LOGICAL | 2,571,284 |
| 2 | Compile Error | recompile; classify first `g++` diagnostic | SYNTAX, SEMANTIC, LINKER | 376,053 |
| 3 | Runtime Error | sandboxed execution; capture termination signal | SIGSEGV, SIGFPE, SIGABRT, NZEC | 339,670 |

**Candidate pool: 7,640,056 (95.40% of C++ submissions).**

Counts are upper bounds; each reduces under the agreement filter below.

### Agreement filter

A submission is retained only where our re-derivation reproduces the
original judge's outcome — for Tier 2, our compiler must also fail; for
Tier 3, our sandbox must reproduce a crash. Exclusions are counted and
reported. This yields doubly-attested labels at the cost of sample count,
a trade this project accepts deliberately.

### Excluded verdicts (368,471 submissions, 4.60%)

- **Time Limit Exceeded, Memory Limit Exceeded** — performance outcomes,
  not defect types. A correct algorithm too slow for one problem's limit
  is not defective.
- **WA: Presentation Error, Output Limit Exceeded** — formatting outcomes,
  outside the taxonomy.
- **Judge Not Available, Query Limit Exceeded, Internal error, Judge
  System Error** — judge infrastructure artefacts carrying no information
  about the code.

## 6. EDA findings

### Scale and structure
- Candidate submissions: **7,640,056** across **4,032 problems** and 102,866 users
- 4,036 problems have C++ submissions; four contain only excluded verdicts
- Submissions per problem: median 357, mean 1,895, max 39,830 — heavy right skew
- Concentration: top 500 problems hold 59.3%; top 1,000 hold 84.3%
- *Figure: `reports/figures/submissions_per_problem.png`*

### Sequence length
- Median code size 735 B; 95th percentile 3,634 B
- **17.3% exceed the ~512-token limit** (approximated at 1,792 B using
  ~3.5 bytes/token; to be measured exactly with the tokeniser in Milestone 2)
- Truncation rate varies systematically by class: Compile Error 10.5%,
  Accepted 15.8%, Wrong Answer 19.7%, **Runtime Error 25.9%**
- *Figure: `reports/figures/code_size_distribution.png`*

**Threat to validity.** Code length correlates with class, so a model
could exploit length as a shortcut. Milestone 2 will report a
length-only baseline to quantify this signal explicitly.

### Resubmission chains
- 34.2% of (user, problem) pairs contain more than one submission
- **65.6% of candidate submissions belong to a chain**; longest chain 764

Chains are near-duplicate code with differing labels. Under a random
split they appear on both sides of the train/test boundary, allowing
memorisation rather than generalisation.

### Data quality
- All 299,596 `cpu_time` / `memory` nulls fall on Compile Error rows
  (79.7% of that verdict) — programs that never compiled have no runtime
  to measure. Accepted and Runtime Error have zero nulls.
- 171 negative `cpu_time` values, confirming IBM's documented warning
- `code_size` is fully clean: no nulls, no non-positive values
- Consequence: `cpu_time` and `memory` are unusable as features
- 18 distinct `original_language` values, including 8 malformed rows
  (`C  11`, `C ++`, `C ++ 11`)

### Compiler versions (drives Tier 2)
| Value | All candidates | Compile Error only |
|---|---|---|
| C++14 (GCC 5.4.1) | 4,758,434 | 206,685 |
| C++ (GCC 9.2.1) | 1,670,127 | 60,243 |
| C++ (unversioned) | 657,902 | 65,684 |

Two GCC versions dominate. Tier 2 recompilation will use `-std=` flags
derived from `original_language`.

### Provenance and temporal coverage
- AtCoder 6,713,032 (87.9%) · AIZU 927,024 (12.1%)
- Date range 2009-07-10 to 2020-10-01; no undateable rows
- 73% of submissions are from 2019–2020

## 7. Splitting strategy

**60/20/20 at the problem level**, not the submission level.

With 4,032 problems the split is approximately 2,419 / 806 / 807. Because
submissions per problem are heavily skewed, problems will be assigned to
folds greedily by submission count rather than uniformly at random, so
that fold sizes balance by submission rather than by problem.

Both problem-level and random-split results will be reported. The gap
between them quantifies resubmission-chain leakage and is treated as a
finding, not a diagnostic.

## 8. Limitations

1. **Domain mismatch** — competitive programming code is short, single-file,
   and algorithm-centric, with no build systems, dependencies, concurrency,
   or architecture. Generalisation to industrial codebases is unverified.
2. **Snippet-level, not module-level** — this project classifies defect
   types in a code snippet; it does not predict which modules will be
   defective before deployment.
3. **Accepted ≠ defect-free** — Accepted means the visible tests passed,
   not that the program is correct.
4. **Derived labels** — seven of nine classes are produced by our pipeline,
   not supplied by IBM.
5. **Hidden test inputs unavailable** — limits Tier 3 crash reproduction.
6. **Compiler drift** — verdicts issued 2009–2020 by GCC 4.x–9.2;
   re-derivation uses a modern compiler.
7. **Unreliable numeric fields** on exactly the rows of interest.
8. **Resubmission chains** — 65.6% of candidates; mitigated by problem-level
   splitting and IBM's near-duplicate files.
9. **Severe class imbalance** — macro-F1 is therefore the headline metric,
   not accuracy.
10. **Frozen, single-platform corpus** — 88% AtCoder, ending October 2020;
    no post-2020 C++ idioms.

## 9. Reproduction

1. Download `Project_CodeNet.tar.gz` to a location outside this repository.
2. Extract `Project_CodeNet/metadata` only:
   `tar -xzf Project_CodeNet.tar.gz -C <dest> Project_CodeNet/metadata`
3. Run `notebooks/01_metadata_acquisition.ipynb` → `data/raw/*.parquet`
4. Run `notebooks/02_metadata_eda.ipynb` → figures and tables in `reports/`

The archive is retained for Milestone 2, which extracts selected source
files by explicit path list.

## 10. Citation

> Puri, R., Kung, D. S., Janssen, G., Zhang, W., Domeniconi, G., Zolotov, V.,
> Dolby, J., Chen, J., Choudhury, M., Decker, L., Thost, V., Buratti, L.,
> Pujar, S., Ramji, S., Finkler, U., Malaika, S., & Reiss, F. (2021).
> *Project CodeNet: A Large-Scale AI for Code Dataset for Learning a
> Diversity of Coding Tasks.* arXiv:2105.12655.
