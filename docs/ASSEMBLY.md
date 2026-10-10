# Unified 9-Class Dataset Assembly (M6)

M6 merges the three label sources (Tier 1 verdicts, M4 compiler-diagnostic
labels, M5 execution labels) into one frozen 9-class manifest, rebuilds the
token cache from it, and measures how head-only truncation falls on each
class. Every number below comes from a committed script run; the scripts and
reports are listed at the end.

## What M6 produced

| Artifact | Detail |
|---|---|
| `data/processed/splits/sample_manifest_9class.parquet` | 46,594 rows. SHA-256 `ea8b94c84618af17dbf58e79d49b28ddadffe47de6d76d010e2ead2b0aab0b8a`, recorded in `split_config.json` under `artifact_hashes` and `m6_assembly`. Not tracked in git; regenerate with `python scripts/assemble_9class.py --write`. |
| `data/processed/tokenized_9class/{train,val,test}.parquet` | 28,458 / 8,997 / 9,139 rows. Verified against `reports/m6_class_split_counts.csv` by `scripts/pretokenize_corpus.py`. |
| Taxonomy version | 1.0 (class IDs unchanged since M2). |

## How the labels are combined

| Manifest `coarse_label` | Leaf label comes from |
|---|---|
| `ERROR_FREE`, `LOGICAL` | Tier 1 itself (terminal classes, never refined) |
| `COMPILE_ERROR` | M4 `classification` in `reports/m4_full_reserve_results_final.csv` |
| `RUNTIME_ERROR` | M5 `outcome` in `reports/m5_full_reserve_classified.csv` |

`sdp.data.labeling.assemble.resolve_leaf` is deliberately strict, unlike the
open-vocabulary Tier 2/3 classifiers. It raises on an unknown string, a
non-string value (such as a pandas NaN), an illegal combination (for example a
RUNTIME_ERROR row that also has an M4 entry), or a leaf that is not a child of
the row's coarse class. Its accepted vocabularies are derived from the tier
modules' own enums and `taxonomy.TIER_OF`, so they cannot drift apart.
`assemble_manifest` additionally rejects duplicate ids and orphan tier rows.

## Result: per-class, per-split counts (retained)

| Class | train | val | test | total |
|---|---|---|---|---|
| ERROR_FREE | 5,968 | 1,995 | 1,994 | 9,957 |
| SYNTAX | 4,232 | 1,197 | 1,264 | 6,693 |
| SEMANTIC | 9,404 | 2,882 | 2,995 | 15,281 |
| LINKER | 232 | 85 | 54 | 371 |
| SIGSEGV | 1,208 | 472 | 445 | 2,125 |
| SIGFPE | 390 | 77 | 110 | 577 |
| SIGABRT | 741 | 228 | 218 | 1,187 |
| NZEC | 311 | 67 | 66 | 444 |
| LOGICAL | 5,972 | 1,994 | 1,993 | 9,959 |
| **Total** | **28,458** | **8,997** | **9,139** | **46,594** |

Overall the split is 61.1% / 19.3% / 19.6%. The split was made by problem in
M2, so class proportions differ between splits: NZEC is 70.0 / 15.1 / 14.9,
LINKER 62.5 / 22.9 / 14.6, SIGFPE 67.6 / 13.3 / 19.1 (train / val / test, %).

The largest-to-smallest class ratio is about 41:1 (SEMANTIC 15,281 vs LINKER
371). The thinnest test sets are LINKER (54), NZEC (66) and SIGFPE (110), so
per-class F1 for those classes should always be reported together with its
support. Among the 4,333 retained runtime rows, the execution input was
`sample_verified` for 4,014, `empty_stdin` for 191 and `sample_unverified`
for 128.

## What was left out (28,256 of 74,850 rows)

| Reason | Rows | From |
|---|---|---|
| `AGREEMENT_FILTER` | 6,446 | COMPILE_ERROR |
| `UNRECOGNIZED` | 535 | COMPILE_ERROR |
| `MISSING_DEPENDENCY` | 394 | COMPILE_ERROR |
| `COMPILATION_TIMEOUT` | 112 | COMPILE_ERROR |
| `EXIT_ZERO` | 16,639 | RUNTIME_ERROR |
| `TOOLCHAIN_HARDENING` | 2,395 | RUNTIME_ERROR |
| `COMPILE_FAILED` | 739 | RUNTIME_ERROR |
| `TIMEOUT` | 352 | RUNTIME_ERROR |
| `UNRECOGNIZED_SIGNAL` | 274 | RUNTIME_ERROR |
| `RESOURCE_LIMIT` | 233 | RUNTIME_ERROR |
| `CONFLICTING_DUPLICATE` | 137 | ERROR_FREE 42, COMPILE_ERROR 44, RUNTIME_ERROR 16, LOGICAL 35 |

Excluded rows are never relabelled into another class (for example
`AGREEMENT_FILTER` files, which compile cleanly, are not moved to
ERROR_FREE): their true label is unknown. The per-row list is
`reports/m6_excluded_rows.csv` (git-ignored, regenerable).

## The conflicting-duplicate rule

Identical source code (same SHA-256) with more than one distinct resolved
outcome is contradictory evidence, so the group's labelled rows are dropped
as `CONFLICTING_DUPLICATE`. Outcomes include exclusion reasons and span all
three reserves. Rows already excluded keep their own, more informative reason.

* Before the rule, 46,731 rows were labelled (the sum of the Tier 1, M4 and
  M5 closeout counts); the rule removed 137 (0.29%).
* Of those 137 rows, 14 sit in groups with two or more different leaf labels
  and 123 in groups with one leaf label and an excluded twin (for example a
  SIGSEGV file next to an identical file that exited normally). A narrower
  rule that only dropped leaf-versus-leaf conflicts would have removed 14.
* Removed per class: ERROR_FREE 42, LOGICAL 35, SYNTAX 5, SEMANTIC 38,
  LINKER 1, SIGSEGV 7, SIGABRT 7, SIGFPE 2, NZEC 0.
* Same-label duplicates are kept: 429 groups remain, containing 541 extra
  copies (1.2% of retained rows). M2's dedup removed duplicates across splits
  only, and the final cross-split SHA-256 audit shows 0 collisions.

## Truncation by class

Truncated means `token_length > 512` (M3's definition; `token_length`
includes `[CLS]` and `[SEP]`). `scripts/measure_truncation_by_class.py`
first asserts that every cached length equals `min(token_length, 512)`; this
passed for all 46,594 files.

| Class | n | truncated % | median tokens | p90 tokens |
|---|---|---|---|---|
| ERROR_FREE | 9,957 | 61.3 | 677 | 2,198 |
| SYNTAX | 6,693 | 34.2 | 319 | 1,356 |
| SEMANTIC | 15,281 | 44.9 | 446 | 1,586 |
| LINKER | 371 | 53.4 | 579 | 1,849 |
| SIGSEGV | 2,125 | 48.1 | 490 | 1,384 |
| SIGFPE | 577 | 23.6 | 238 | 971 |
| SIGABRT | 1,187 | 44.2 | 430 | 1,940 |
| NZEC | 444 | 25.5 | 221 | 1,003 |
| LOGICAL | 9,959 | 58.8 | 641 | 2,059 |
| **All retained** | **46,594** | **49.6** | **507** | **1,788** |

The 49.6% overall figure is over the 46,594 retained files; the 55.7% in
`docs/TOKENIZATION.md` was measured over the 74,850-file working corpus, a
different population. Median length differs about threefold between classes
(221 for NZEC to 677 for ERROR_FREE), so code length is correlated with the
label. This is why the M8 length-only baseline is required.

## Is the first compile error visible to the model?

Head-only truncation shows the model content tokens 0..509. For SYNTAX and
SEMANTIC files, the first compiler error line was located (from the M4
stderr, g++'s first diagnostic) and classed as fully, partly or not visible,
using exact character offsets from the fast CodeBERT tokenizer. g++'s line
numbers were validated against the echoed source line: 14,963 of 14,963
SEMANTIC and 6,668 of 6,678 SYNTAX echoes match the source line; the ten
SYNTAX differences that were inspected (five of them) involve control
characters that g++ prints as `<U+00XX>`.

Excluded from the percentages and counted: files of at most 2 lines (the
whole program is "line 1"), and files whose first error is not on a line of
the submission's own file.

| | SYNTAX (all) | SYNTAX (missing-include only) | SYNTAX (excl. missing-include) | SEMANTIC |
|---|---|---|---|---|
| files | 6,693 | 573 | 6,120 | 15,281 |
| at most 2 lines | 165 | 15 | 150 | 292 |
| no usable error line | 9 | 2 | 7 | 313 |
| measured | 6,519 | 556 | 5,963 | 14,676 |
| fully visible | 5,408 (83.0%) | 547 (98.4%) | 4,861 (81.5%) | 11,383 (77.6%) |
| partly visible | 37 (0.6%) | 1 (0.2%) | 36 (0.6%) | 122 (0.8%) |
| not visible | 1,074 (16.5%) | 8 (1.4%) | 1,066 (17.9%) | 3,171 (21.6%) |

Across SYNTAX and SEMANTIC, 4,245 of 21,195 measured files (20.0%) have
their first error outside the model's input. Visibility means the evidence
text is present, not that the model uses it. LINKER and the runtime classes
are not measured this way: a link error has no source line, and a crash is a
property of execution, not of a position in the file. Per-file visibility
results are not saved; the M8 error analysis re-runs the measurement.

## Limitations to state in the report

* **Missing-include branch.** 573 of 6,693 SYNTAX labels (8.6%) come from
  g++'s "No such file or directory" fatal error, which labels from the whole
  stderr, not the first error line. The include line is at line 1 for the
  median file.
* **Quoted source lines.** `tier2._first_error_line` can pick a line of the
  submission's own source that quotes compiler-error text. Two SYNTAX rows
  are affected; the frozen M4 labels were left untouched.
* **LINKER composition.** Of 371 LINKER files, 179 (48%) fail with
  "undefined reference to `main'", and 26 (7%) also show the "extra tokens at
  end of #include" warning. Hypothesis, not yet tested: because `main` often
  sits at the end of a long file, outside the 512-token window, "no `main`
  visible" cannot separate these files from ordinary long ones. M8 should
  report LINKER results split into missing-`main` versus other.
* **Verification residue.** 20 measured SYNTAX/SEMANTIC rows (0.09%) could
  not be echo-verified.
* **Parquet hash.** The recorded SHA-256 identifies this exact file. A
  different `pyarrow`/`pandas` version can write different bytes for the same
  rows, so a clean-clone check (M10) should compare row content
  (`submission_id` with `leaf_label`), not only the file hash.

## Reproduce
python scripts/assemble_9class.py # dry run: prints every check
python scripts/assemble_9class.py --write # saves manifest + reports + hash
python scripts/pretokenize_corpus.py # builds and verifies the token cache
python scripts/measure_truncation_by_class.py


Committed reports: `reports/m6_assembly_summary.md`,
`reports/m6_class_split_counts.csv`, `reports/m6_exclusion_counts.csv`,
`reports/m6_truncation_by_class.csv`, `reports/m6_error_visibility.csv`.
Code: `src/sdp/data/labeling/assemble.py`, `src/sdp/data/visibility.py`,
`src/sdp/data/dataset.py`, with tests in `tests/test_assemble.py`,
`tests/test_assemble_manifest.py`, `tests/test_visibility.py` and
`tests/test_dataset.py`.
