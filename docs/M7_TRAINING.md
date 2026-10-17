# M7: CodeBERT fine-tuning (9-class)

**Status:** complete. **Scope:** every number on this page is a VALIDATION result. The test
split was never read during M7; the single test evaluation belongs to M8.

Run `codebert-9class-v1`, trained 2026-10-16 at commit `31d0f0e` with `configs/train.yaml`
(unchanged since commit `1ac5f02`).

## 1. What was trained

- **Model:** `microsoft/codebert-base` encoder without its pooler (124,055,040 parameters),
  dropout 0.1, and one linear layer 768 -> 9 on the [CLS] vector (124,061,961 parameters in
  total). The head is flat, not hierarchical (Master Plan v3, section 1, guardrail 3).
- **Input:** the frozen M6 token cache, head-truncated at 512 tokens (49.6% of the dataset's
  files are truncated; see `docs/ASSEMBLY.md`).
- **Data:** 28,458 training and 8,997 validation files. Train class counts in taxonomy order
  (ERROR_FREE, SYNTAX, SEMANTIC, LINKER, SIGSEGV, SIGFPE, SIGABRT, NZEC, LOGICAL):
  5968, 4232, 9404, 232, 1208, 390, 741, 311, 5972.
- **Loss:** class-weighted cross-entropy with per-sample weights proportional to
  count^-0.5, scaled to a sample-weighted mean of 1. Weights in taxonomy order:
  0.854, 1.015, 0.681, 4.333, 1.899, 3.342, 2.425, 3.743, 0.854. The loss is the weighted sum
  over the samples of one optimizer step divided by their actual number, so every sample
  counts with exactly its class weight (see `src/sdp/training/trainer.py`).
- **Optimization:** AdamW, learning rate 2e-5, weight decay 0.01 (none on biases and
  LayerNorm), 10% linear warmup (889 of 8,895 steps) then linear decay to 0, gradient
  clipping at 1.0, at most 5 epochs, early stopping with patience 2 on validation macro-F1
  (not triggered), seed 42.
- **Batching and precision:** micro-batch 2 with 8 accumulation steps (effective batch 16;
  the final group of each epoch holds 10 samples and is normalized by its own size), bf16
  autocast with the loss computed in float32, no gradient checkpointing.
- **Hardware:** local laptop, NVIDIA RTX 3050 Laptop GPU (4 GB), PyTorch 2.5.1+cu124. The
  configuration was chosen with `scripts/benchmark_gpu.py` (`reports/m7_gpu_*`).

## 2. Deviations from Master Plan v3 (the plan itself is unchanged)

- Training ran locally, not on Colab (decision of 2026-10-10).
- The model class lives in `src/sdp/model/classifier.py`, not `codebert.py`; the `Predictor`
  wrapper is built in M9.
- The GPU benchmark was run before the training loop was written.
- The learning rate is 2e-5. The base paper's reported 1e-9 was judged an error and is
  rejected by the configuration validator (floor 1e-7).

## 3. Run facts

- 2026-10-16, 17:43:55 to 19:29:42 (1 h 46 min), about 24.2 samples/s throughout (benchmark:
  23.6), about 19.6 min of training per epoch plus about 1.5 min for validation and
  checkpointing.
- No non-finite loss. The run ended because the 5-epoch schedule finished; epoch 5 was the
  first epoch without a validation macro-F1 improvement.

## 4. Results (validation)

| Epoch | Train loss | Val loss | Val accuracy | Val macro-F1 |
|---|---|---|---|---|
| 1 | 1.9138 | 1.7953 | 0.3052 | 0.1919 |
| 2 | 1.6595 | 1.6937 | 0.3228 | 0.2899 |
| 3 | 1.4394 | 1.6825 | 0.3532 | 0.3207 |
| 4 (best) | 1.2918 | 1.7210 | 0.3810 | 0.3542 |
| 5 | 1.1748 | 1.7660 | 0.3740 | 0.3454 |

Reference baselines (M7.1, validation): majority class 0.3203 accuracy / 0.0539 macro-F1;
length-only 0.1685 / 0.0775.

Best epoch (4), per class on the validation split:

| Class | Val files | Length-only F1 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| ERROR_FREE | 1995 | 0.356 | 0.351 | 0.357 | 0.354 |
| SYNTAX | 1197 | 0.007 | 0.746 | 0.299 | 0.427 |
| SEMANTIC | 2882 | 0.096 | 0.497 | 0.381 | 0.432 |
| LINKER | 85 | 0.063 | 0.125 | 0.059 | 0.080 |
| SIGSEGV | 472 | 0.099 | 0.283 | 0.333 | 0.306 |
| SIGFPE | 77 | 0.021 | 0.289 | 0.286 | 0.288 |
| SIGABRT | 228 | 0.026 | 0.285 | 0.364 | 0.320 |
| NZEC | 67 | 0.027 | 0.587 | 0.657 | 0.620 |
| LOGICAL | 1994 | 0.004 | 0.293 | 0.475 | 0.362 |

## 5. Observations

Measured:

1. Best validation macro-F1 is 0.3542 (epoch 4): 4.6 times the length-only baseline and 6.6
   times the majority baseline.
2. The model uses code content, not only length: SYNTAX, SEMANTIC, SIGSEGV, SIGFPE, SIGABRT,
   NZEC and LOGICAL rose from near zero to 0.29-0.62 F1 (NZEC 0.027 to 0.620). ERROR_FREE
   did not improve over the length-only score (0.354 against 0.356) and LINKER barely did
   (0.063 to 0.080; 5 of 85 validation files found, precision 0.125).
3. Overfitting began after epoch 3: training loss kept falling (1.44 to 1.17) while
   validation loss rose (1.683 to 1.766). Validation macro-F1 peaked at epoch 4 and fell by
   0.009 at epoch 5. That difference is within the epoch-to-epoch variation of the thin
   classes (LINKER F1 ranged from 0.000 to 0.089 across epochs), so epoch 4 versus epoch 5
   is not a meaningful difference.
4. SYNTAX has high precision (0.746) but low recall (0.299); LOGICAL shows the opposite
   (precision 0.293, recall 0.475).

Hypotheses, not yet tested (to be checked in M8):

- ERROR_FREE and LOGICAL are frequently confused with each other (accepted versus
  wrong-answer code look alike); a confusion matrix will show it.
- Head-only truncation limits SYNTAX and SEMANTIC, whose first compile error lies outside
  the 510-token window for 17.9% and 21.6% of measured files.
- LINKER is limited by data (232 training files), the risk documented at M4.

## 6. Limitations

- The best epoch was selected on the validation split, so its validation score is slightly
  optimistic; the honest figure is the M8 test evaluation.
- One seed. The validation supports of LINKER (85), SIGFPE (77) and NZEC (67) are small, so
  their per-class numbers are noisy.
- Only `class_weight_power` 0.5 was run; no ablation of the weighting strength.
- Labels come from the M4 and M5 pipelines and inherit their documented limits (toolchain,
  sample inputs, verdict noise).

## 7. Artifacts and reproduction

| File | Content |
|---|---|
| `reports/m7_training/metrics.csv` | per-epoch losses, accuracy, macro-F1, nine per-class F1, timing |
| `reports/m7_training/val_report_best.csv` | per-class precision, recall, F1 at the best epoch |
| `reports/m7_training/config.json` | resolved config, class counts and weights, step counts |
| `reports/m7_training/summary.json` | run summary |
| `reports/m7_training/train_log.txt` | full training log |
| `reports/m7_training/best_pt_sha256.txt` | SHA-256 of the best checkpoint |
| `runs/codebert-9class-v1/best.pt` | best weights (epoch 4, about 0.5 GB, git-ignored) with metadata: label names in ID order, taxonomy version 1.0, model name, config, best epoch, validation macro-F1 |

Reproduce at commit `31d0f0e` with `python scripts\train_codebert.py`. Runs are
reproducible from the seed up to GPU nondeterminism.
