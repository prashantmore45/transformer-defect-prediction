# M8 Evaluation Report

Project: CodeBERT-based multiclass software defect prediction (Group 42, SIT, SPPU, AY 2026-27).
Scope: evaluation and error analysis of the M7 model (`best.pt`, tag `v0.8-model`).
Every number below is computed from a table committed under `reports/m8/`; the file is named
next to each result. Intervals are 95% percentile intervals from a bootstrap that resamples
whole problems (2,000 draws, seed 42).

## 1. Summary

| # | Finding | Evidence |
|---|---|---|
| 1 | On a problem-level test split (9,139 files from 1,251 problems) the model reaches macro-F1 **0.355** [0.332, 0.375] and accuracy **0.378** [0.368, 0.389]. Validation gave 0.354 and 0.381, so no selection optimism is visible. | `test_overall.csv`, `val_overall.csv` |
| 2 | It beats a TF-IDF + logistic regression baseline by **+0.091** macro-F1 [0.063, 0.118] (accuracy +0.034 [0.022, 0.047]). The gain is concentrated in short files and the runtime classes, and vanishes above 1,024 tokens. | `compare_test_differences.csv`, `slices/length_overall.csv` |
| 3 | File length alone does not explain the score: a length-only model scores macro-F1 0.077, below the 0.113 chance level. | `compare_test_systems.csv` |
| 4 | Errors mostly cross parent classes: only 7.2% of errors stay inside the correct parent, against 15.8% expected if errors ignored the hierarchy. | `test_parent_stats.csv` |
| 5 | Compile errors beyond the 512-token window are almost never found: SYNTAX recall 0.02 and SEMANTIC 0.15, against 0.28 and 0.38 for visible errors in equally long files. | `slices/visibility_per_class.csv` |
| 6 | A random split would have reported macro-F1 0.443 instead of 0.355 (+0.088 [0.058, 0.117]). On the 1,739 files in both test sets the difference is +0.020 [-0.012, 0.052]. | `random_split/compare_*.csv` |
| 7 | LINKER performance is very weak (3 of 54 test files correct; F1 0.076 [0.000, 0.167]). | `test_per_class.csv`, `test_confusion_9x9.csv` |

Parts B and C give the slice analyses, the leakage comparison, the limitations and the decisions.

## 2. Setup

- **Data and labels.** C++ submissions from IBM Project CodeNet (8,008,527 in the metadata). A
  74,850-file working corpus was labelled into nine classes by what happened when each file was
  judged, compiled and run, not by manual annotation:
  - *Tier 1, judge verdicts:* ERROR_FREE, LOGICAL, COMPILE_ERROR, RUNTIME_ERROR.
  - *Tier 2, compiler diagnostics:* COMPILE_ERROR files were recompiled with g++ 15.2.0 (Linux,
    via WSL) and the first diagnostic was parsed into SYNTAX, SEMANTIC or LINKER; only files
    that still fail on recompilation were kept (22,389 of 29,876 labelled).
  - *Tier 3, execution:* RUNTIME_ERROR files (24,981) were recompiled and run in a temporary
    directory under CPU, memory, file-size and core-dump limits with a wall-clock timeout (OS
    resource limits, no container), using the CodeNet sample input where available and empty
    stdin otherwise. The terminating signal or exit code gave SIGSEGV, SIGFPE, SIGABRT or
    NZEC. Only crashes that reproduced were kept: 4,349 files (17.4%). These labels are tied to
    this toolchain, to -O0 compilation and to the sample input; files whose abort came from the
    toolchain's hardened library (2,395) were excluded.
  - After assembly and removal of 137 conflicting duplicates the frozen dataset has 46,594 files.
- **Split.** Problem-level 60/20/20 split (seed 42): every submission to a problem stays in one
  split. Train 28,458, validation 8,997, test 9,139 files; the test set covers 1,251 problems. No
  exact duplicate of a file (SHA-256 of the normalised source) appears in two splits.
- **Model.** `microsoft/codebert-base` with its pooler layer left out, dropout 0.1 on the [CLS]
  vector and one linear layer to 9 outputs (flat, not hierarchical). Loss: class-weighted
  cross-entropy with weights proportional to count^-0.5. AdamW, learning rate 2e-5, weight decay
  0.01, 10% warmup then linear decay, effective batch 16 (micro-batches of 2 accumulated 8
  times), bf16, seed 42, at most 5 epochs with early stopping on validation macro-F1 (patience
  2). All 5 epochs ran; the best epoch was epoch 4.
- **Input.** The first 512 tokens (510 content tokens plus [CLS] and [SEP]); nothing after that
  is seen. 49.6% of the 9-class dataset and 55.3% of the test files (5,058 of 9,139) are longer
  than that (`reports/m6_truncation_by_class.csv`, `slices/length_overall.csv`).
- **Test support per class.** ERROR_FREE 1,994; SYNTAX 1,264; SEMANTIC 2,995; LINKER 54;
  SIGSEGV 445; SIGFPE 110; SIGABRT 218; NZEC 66; LOGICAL 1,993 (`test_per_class.csv`).

## 3. Evaluation protocol

- The test split was run through the model **once** (`reports/m8/test_predictions.parquet`);
  every later number is computed from that saved dump, never from a new model run.
- Before the test run, the pipeline had to reproduce the validation macro-F1 stored in the
  checkpoint (difference 0.00000); checkpoint, manifest and every dump are hash-verified
  (`m8_1_provenance.json`).
- **Headline metric: macro-F1** (unweighted mean of per-class F1, every class always scored),
  because the classes are heavily imbalanced: SEMANTIC outnumbers LINKER about 41:1 in the full
  dataset and 55:1 in the test split. Accuracy is reported alongside.
- **Intervals:** percentile bootstrap over problems, not files, because files from one problem
  are correlated. Comparisons between systems on the same files are paired (same resampled
  problems), which gives much tighter intervals than comparing two separate intervals.
- The intervals cover test-set sampling only; there is a single training run (seed 42), so
  run-to-run variance is not included.

## 4. Headline results

Test split, 9,139 files (`test_overall.csv`, `test_per_class.csv`):

| class | files | precision | recall | F1 [95% interval] |
|---|---|---|---|---|
| ERROR_FREE | 1,994 | 0.346 | 0.354 | 0.350 [0.332, 0.368] |
| SYNTAX | 1,264 | 0.771 | 0.287 | 0.418 [0.387, 0.449] |
| SEMANTIC | 2,995 | 0.511 | 0.392 | 0.444 [0.425, 0.462] |
| LINKER | 54 | 0.120 | 0.056 | 0.076 [0.000, 0.167] |
| SIGSEGV | 445 | 0.219 | 0.258 | 0.237 [0.197, 0.279] |
| SIGFPE | 110 | 0.254 | 0.291 | 0.271 [0.179, 0.362] |
| SIGABRT | 218 | 0.333 | 0.417 | 0.371 [0.304, 0.430] |
| NZEC | 66 | 0.653 | 0.712 | 0.681 [0.558, 0.775] |
| LOGICAL | 1,993 | 0.279 | 0.463 | 0.348 [0.333, 0.363] |
| **overall** | 9,139 | | accuracy 0.378 [0.368, 0.389] | macro-F1 **0.355** [0.332, 0.375] |

Figures: `reports/m8/test_confusion_percent.png` (9x9 confusion matrix, rows = true class),
`reports/m8/test_per_class_f1.png`.

What the confusion matrix (`test_confusion_9x9.csv`) shows:

- **ERROR_FREE and LOGICAL are close to a coin flip.** Restricted to files of those two classes
  that the model assigned to one of the two, it separates them 55.2% of the time.
- **Predictions pile onto LOGICAL:** 3,314 of 9,139 predictions (36.3%) against 1,993 true
  LOGICAL files (21.8%). Predictions of LOGICAL and ERROR_FREE are also the least confident
  (median margin between the top two probabilities 0.087 and 0.064, against 0.116 over all
  predictions; `slices/confidence_by_predicted_class.csv`). Both act as fallback classes when no
  specific defect is visible.
- **Parent level** (`test_parent_stats.csv`): parent accuracy 0.423 [0.412, 0.434]; the share of
  errors landing inside the correct parent is 0.072 [0.066, 0.080], against 0.158 [0.155, 0.162]
  if every wrong prediction were spread uniformly over the other eight classes.
- **SYNTAX is precise but misses most:** 77% of SYNTAX predictions are right but only 29% of
  SYNTAX files are found. NZEC is the strongest class (F1 0.68) and LINKER the weakest.

## 5. Baselines

All baselines are fitted on the training split; the only tuned setting (TF-IDF's regularisation
strength C) was chosen on validation. Test results (`compare_test_systems.csv`):

| system | accuracy | macro-F1 |
|---|---|---|
| CodeBERT | 0.378 [0.368, 0.389] | **0.355** [0.332, 0.375] |
| TF-IDF + logistic regression | 0.344 [0.334, 0.354] | 0.264 [0.240, 0.288] |
| Stratified random | 0.227 [0.219, 0.235] | 0.113 [0.107, 0.119] |
| Length only | 0.173 [0.166, 0.180] | 0.077 [0.070, 0.084] |
| Majority class (always SEMANTIC) | 0.328 [0.318, 0.337] | 0.055 [0.054, 0.056] |

Figures: `reports/figures/m8/m8_systems_macro_f1.png`, `m8_systems_accuracy.png`.

- Chance level for macro-F1 is about 1/9 = 0.111: a predictor independent of the input that
  follows the class frequencies has F1 equal to each class's share, whatever the imbalance.
  The stratified-random baseline measures 0.113.
- **CodeBERT minus each baseline, paired** (`compare_test_differences.csv`): TF-IDF macro-F1
  +0.091 [0.063, 0.118], accuracy +0.034 [0.022, 0.047]; length-only +0.278; stratified random
  +0.242; majority +0.300 [0.277, 0.321] (accuracy +0.050 [0.037, 0.064]). CodeBERT’s macro-F1 was
  ahead in all 2,000 draws against every baseline. The accuracy gain over the majority class is
  modest; the model's strength is balanced performance across classes, not raw accuracy.
- **Per class, CodeBERT minus TF-IDF** (`compare_test_per_class_differences.csv`;
  `reports/figures/m8/m8_codebert_vs_tfidf_per_class.png`): NZEC +0.368 [0.221, 0.511], SIGFPE
  +0.190 [0.106, 0.272], SIGABRT +0.105, SIGSEGV +0.073, SYNTAX +0.060, LOGICAL +0.055; SEMANTIC
  +0.025 [0.005, 0.044] is marginal. ERROR_FREE (-0.006) and LINKER (-0.050 [-0.206, 0.082])
  show no measurable difference. Nine intervals are reported without correction for multiple
  comparisons.
- **TF-IDF protocol note.** The TF-IDF baseline uses balanced class weights, a stronger
  rebalancing than the square-root weights used for CodeBERT, so CodeBERT's lead is conservative.
  The baseline was revised once after its first run (it hit the solver's iteration limit and its
  best C sat at the edge of the search grid); the decision used validation diagnostics only and
  no baseline test number had been seen (`reports/m8/m8_3_revision_note.md`).


## 6. Slice analyses

Each slice is a subset of the 9,139 test files. All slices are bootstrapped over the same
resampled problems, so differences between slices have paired intervals. Within a slice the
report gives accuracy and per-class recall with file counts; macro-recall is the mean recall
over a fixed set of classes that has at least 20 files in every slice compared. A per-class
difference based on fewer than 20 files in either slice is marked LOW n in the tables and is
not interpreted. Each analysis is also run for the TF-IDF baseline as a control, except
confidence (the baseline has no probabilities). Code: `scripts/analyze_slices.py`; tables:
`reports/m8/slices/`. Before any slice is computed the script checks that a one-slice bootstrap
reproduces the accuracy interval in `test_overall.csv`.

### 6.1 File length

Bins on the full token length (`slices/length_overall.csv`, `slices/length_per_class.csv`;
figure `reports/figures/m8/m8_length_accuracy.png`). Macro-recall is over ERROR_FREE, SYNTAX,
SEMANTIC, SIGSEGV and LOGICAL, the classes with enough files in every bin.

| tokens | files | CodeBERT accuracy | TF-IDF accuracy | CodeBERT macro-recall | TF-IDF macro-recall |
|---|---|---|---|---|---|
| <=128 | 686 | 0.548 [0.512, 0.587] | 0.509 [0.470, 0.546] | 0.514 | 0.442 |
| 129-256 | 1,290 | 0.434 [0.408, 0.462] | 0.371 [0.346, 0.397] | 0.396 | 0.331 |
| 257-512 | 2,105 | 0.401 [0.378, 0.425] | 0.348 [0.326, 0.368] | 0.364 | 0.310 |
| 513-1024 | 2,540 | 0.356 [0.336, 0.376] | 0.320 [0.301, 0.340] | 0.322 | 0.291 |
| >1024 | 2,518 | 0.305 [0.287, 0.323] | 0.305 [0.287, 0.325] | 0.249 | 0.261 |

- Both systems get steadily worse as files get longer, and CodeBERT's lead over TF-IDF shrinks
  to nothing above 1,024 tokens. The model's advantage lives in shorter files.
- Recall of the defect classes collapses for long files while ERROR_FREE recall rises
  (CodeBERT: SYNTAX 0.48 to 0.08, SEMANTIC 0.64 to 0.17, SIGSEGV 0.62 to 0.05, SIGABRT 0.88 to
  0.03, ERROR_FREE 0.18 to 0.48). Long files are also mostly ERROR_FREE or LOGICAL (51% of
  files above 1,024 tokens), so class mix and truncation both contribute; 6.3 separates them.
- **NZEC is a short-file result.** CodeBERT's NZEC recall is 1.00, 0.92, 0.88, 0.50 and 0.00
  across the five bins (13, 13, 16, 16 and 8 files).

### 6.2 Confidence

Quintiles of the model's top probability (`slices/confidence_overall.csv`,
`confidence_calibration.csv`, `confidence_by_predicted_class.csv`).

| quintile | accuracy | mean confidence | confidence minus accuracy |
|---|---|---|---|
| Q1 (least confident) | 0.254 [0.234, 0.274] | 0.311 | +0.057 |
| Q2 | 0.306 [0.284, 0.327] | 0.370 | +0.064 |
| Q3 | 0.342 [0.320, 0.365] | 0.417 | +0.075 |
| Q4 | 0.374 [0.352, 0.396] | 0.484 | +0.110 |
| Q5 (most confident) | 0.614 [0.590, 0.637] | 0.755 | +0.141 |

- Confidence ranks predictions usefully (accuracy rises from 0.25 to 0.61) but the model is
  **overconfident in every quintile**. Class-weighted training, which shifts probability toward
  rare classes, is a plausible contributor; this was not tested. Any confidence shown to a
  user should be labelled a model score, not a calibrated probability.
- Fallback classes (n = predictions of that class; margin = top minus second probability):

| predicted class | n | share | median margin | share with margin < 0.10 | precision |
|---|---|---|---|---|---|
| ERROR_FREE | 2,037 | 22.3% | 0.064 | 67.1% | 0.346 |
| LOGICAL | 3,314 | 36.3% | 0.087 | 55.7% | 0.279 |
| SEMANTIC | 2,297 | 25.1% | 0.212 | 28.9% | 0.511 |
| SYNTAX | 471 | 5.2% | 0.663 | 12.3% | 0.771 |
| SIGFPE | 126 | 1.4% | 0.662 | 7.1% | 0.254 |
| NZEC | 72 | 0.8% | 0.990 | 1.4% | 0.653 |
| all predictions | 9,139 | | 0.116 | 45.3% | 0.378 |

  ERROR_FREE and LOGICAL predictions are the least confident, so **both act as fallback
  classes** chosen when no specific defect is visible (an earlier hypothesis, that LOGICAL
  alone is the default, was only half right). SIGFPE predictions are confident (median margin
  0.66) but right only 25% of the time.

### 6.3 Is the first compile error inside the model's window?

For SYNTAX and SEMANTIC test files, whether the compiler's first error line falls inside the
510 content tokens the model sees. The per-file measurement reuses the M6 code and, summed over
all splits, reproduces `reports/m6_error_visibility.csv` exactly
(`scripts/visibility_per_file.py`). In the test split 4,091 of 4,259 files were measured (100
were too short to say, 68 had no usable error line). Slices: error visible and file fits (2,035
files), error visible but file truncated (1,064), error not fully visible (992, all in
truncated files). Figures: `m8_visibility_recall_codebert.png`, `m8_visibility_recall_tfidf_logreg.png`.

| slice | CodeBERT accuracy | SYNTAX recall | SEMANTIC recall | TF-IDF accuracy |
|---|---|---|---|---|
| visible, file fits | 0.465 [0.441, 0.489] | 0.37 | 0.52 | 0.448 [0.426, 0.470] |
| visible, file truncated | 0.353 [0.322, 0.386] | 0.28 | 0.38 | 0.362 [0.331, 0.394] |
| error beyond the window | 0.119 [0.098, 0.141] | 0.02 | 0.15 | 0.184 [0.160, 0.211] |

- Holding truncation fixed, a visible error is found far more often than a hidden one
  (CodeBERT, visible-truncated minus beyond-window): accuracy +0.234 [0.196, 0.272], SYNTAX
  recall +0.263 [0.204, 0.323], SEMANTIC +0.225 [0.178, 0.275]. TF-IDF shows the same pattern
  (+0.177 [0.140, 0.214]) because it sees the same window. Truncation is a major cause of
  missed compile errors.
- **Where do hidden-error files go?** CodeBERT puts 32.7% under ERROR_FREE and 44.6% under
  LOGICAL (77% together) and 16.6% under COMPILE_ERROR; TF-IDF puts 63% under those two and
  28.6% under COMPILE_ERROR. When the error is not in the window, TF-IDF does better than
  CodeBERT on these files: CodeBERT relies on seeing the error.
- **Truncation is not the only cause.** Even when the error is visible and the file fits, only
  57.1% of these files are predicted under the correct parent (COMPILE_ERROR); 25.5% go to
  LOGICAL and 10.1% to the runtime classes. A visible file that is truncated does worse than one
  that fits (accuracy -0.111 [-0.152, -0.072]), so length matters beyond visibility.
- *Rough size of the opportunity (an estimate, with its assumption):* if hidden-error files
  were recognised as well as visible-truncated ones, about 992 x 0.234 = 232 more files would
  be correct, roughly +2.5 points of overall accuracy.

### 6.4 Author overlap

2,272 of the 9,139 test files (24.9%) come from authors who do not appear in the training split;
6,867 come from authors who do (4,531 test authors, 2,589 also in train)
(`slices/user_overlap_*.csv`). If CodeBERT leaned on author habit, files from seen authors
would be easier.

| slice | files | accuracy | macro-recall (all 9 classes) |
|---|---|---|---|
| seen author | 6,867 | 0.372 [0.361, 0.384] | 0.321 [0.293, 0.350] |
| unseen author | 2,272 | 0.395 [0.376, 0.414] | 0.429 [0.393, 0.464] |

- Seen minus unseen: accuracy -0.023 [-0.045, -0.000], macro-recall **-0.109** [-0.152, -0.066].
  Per class: ERROR_FREE +0.114 [0.067, 0.161]; SYNTAX -0.098; SEMANTIC -0.048; SIGSEGV -0.119;
  SIGFPE -0.295; NZEC -0.314 [-0.528, -0.109] (28 unseen-author NZEC files).
  TF-IDF shows the same direction, weaker (macro-recall -0.043 [-0.091, 0.004]).
- **Files from seen authors are harder, not easier**, so there is no sign that author overlap
  inflates the results, and NZEC's strength is not an author artefact. Why seen-author files
  are harder is unexplained: the groups may differ in other ways, or the model may attach a
  seen author's overall style (mostly accepted code) to their defective files; neither was tested.

### 6.5 Runtime-class input source

Test runtime-class files by the input used to reproduce their crash: 744 with a verified sample
input, 30 with an unverified one, 65 with empty stdin (`slices/runtime_input_*.csv`).

- Verified-input only minus all runtime files: accuracy 0.000 [-0.015, 0.016], macro-recall
  -0.004 [-0.020, 0.009]; every per-class recall difference is within +-0.022. **The flagged
  empty-stdin labels do not affect the results.**
- The empty-stdin slice has 41 SIGSEGV, 16 SIGABRT, 6 SIGFPE and 2 NZEC files, too few for
  per-class conclusions; differences involving it are marked LOW n and are not interpreted
  (the bootstrap interval is badly overconfident at those counts).

### 6.6 Qualitative examples

Fifteen test files drawn at random (seed 42), five per category (`scripts/export_examples.py`;
the output is not committed, see section 9). Short fragments are quoted from CodeNet
submissions, identified by problem and submission ID.

- **NZEC predicted correctly (5 of 5):** each has an explicit non-zero exit from `main`:
  `exit(1)` in a loop (p00195), `return 1;` at the end of `main` (p03089), returning the value
  of `printf(...)` (p03301), `if (n < 300) return 1;` (p03396) and `return n;` (p04029); the
  model's top probability was 0.98 to 1.00. NZEC's strength therefore reflects a locally visible
  cause, not a general ability to predict runtime failures (SIGSEGV recall is 0.26). A
  plausible reason TF-IDF does much worse (NZEC F1 0.31): a bag of tokens cannot tell whether
  `return 1` is in `main` or in a helper function; this was not tested.
- **LINKER files missed (5 of 5):** `int mian()` with no `main` (p03957); `ll f[N][N]` with
  `N = 20010`, about 3.2 GB of static data (p03621); functions or static members declared but
  defined beyond the window (p00607, p03581); a copied standard-library header whose real code
  starts after line 120 (p02926). Top probabilities were 0.20 to 0.46. LINKER needs whole-file
  reasoning, which a 512-token window and 214 training examples do not support.
- **SYNTAX files called LOGICAL (5 of 5):** three errors lie beyond the window (line 90 with the
  window ending at 46; line 53 with 45; a missing final brace at the end of the file) and two
  are visible and missed (a `for` header written `j<w,j+=1`, and `int A[n]B[n];`). In all five
  the top two choices were LOGICAL and ERROR_FREE.
- On reading, the labels agree with the code and compiler output in all fifteen cases (a spot
  check, not an audit).
