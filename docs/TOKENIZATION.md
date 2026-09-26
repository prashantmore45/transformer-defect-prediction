# Tokenization & Training Data Pipeline (M3)

## Tokenizer

`microsoft/codebert-base`'s tokenizer (`transformers.AutoTokenizer`), a
RoBERTa-style BPE tokenizer. Hard architectural limit: **512 tokens** — fixed
by the pretrained positional embedding table. This cannot be exceeded without
retraining/interpolating position embeddings, which is out of scope for this
project regardless of which truncation strategy is chosen.

## Measured statistics (full 74,850-file working corpus, measured 2026-09-25)

`docs/DATASET.md` (M1) used a generic ~3.5 bytes/token approximation to
estimate ~17.3% of files would exceed 512 tokens. That approximation is now
replaced by a direct measurement with the real tokenizer.

| Metric | M1 approximation | **Measured** |
|---|---|---|
| Tokens/byte | ~0.286 (1 / 3.5) | **0.559** |
| Files exceeding 512 tokens | ~17.3% | **55.7%** |

Full distribution (`reports/tokenization_full.csv`, `reports/tokenization_report.csv`):

| | overall | train | val | test |
|---|---|---|---|---|
| n files | 74,850 | 44,978 | 14,941 | 14,931 |
| mean tokens | 975.3 | 928.1 | 1,036.0 | 1,056.7 |
| median tokens | 592 | 543 | 664 | 675 |
| p90 tokens | 1,923 | 1,835 | 2,001 | 2,098 |
| p95 tokens | 2,743.6 | 2,636.2 | 2,774.0 | 2,983.0 |
| p99 tokens | 5,857.6 | 5,629.8 | 5,862.6 | 6,264.5 |
| max tokens | 316,262 | 202,924 | 316,262 | 99,996 |
| truncation rate (>512) | 55.7% | 52.2% | 60.6% | 61.4% |
| mean tokens/byte | 0.559 | 0.560 | 0.557 | 0.559 |

**Why the M1 approximation undershot:** CodeBERT's real BPE tokenizer is
trained on a natural-language + code mix, and C++'s punctuation-dense syntax
(`{`, `;`, `::`, template angle brackets, macro identifiers) splits into more
subword pieces per byte than prose does. Measured density (0.559 tokens/byte,
~1.79 bytes/token) is roughly double what the generic estimate assumed.

**Outlier note:** the corpus maximum (316,262 tokens, in `val`) is far beyond
`p99` (~5,858) — at least one extreme file exists, likely machine-generated
or highly repetitive. It doesn't affect the pipeline (anything over 512 is
truncated identically regardless of how far over it is) but is worth a
one-line EDA mention if raised at viva.

## Decision: truncation strategy — head-only, kept

**Decided 2026-09-25, with the 55.7% truncation-rate measurement in hand —
not assumed upfront.**

Head+tail truncation (keep tokens from both ends of the file) was considered
as an alternative, on the reasoning that competitive-programming C++
conventionally front-loads defect-irrelevant boilerplate (`#include`s,
`typedef`s, macros, fast-IO templates), so head-only truncation risks
systematically keeping the boring part and cutting the part where the actual
logic — and the actual bug — lives.

**Head-only was kept anyway.** Reasoning:
- 512 tokens is a hard limit regardless of strategy — the real fix (a
  code-specific tokenizer, or a long-context model) is out of scope here.
- Head+tail adds a second truncation code path to implement, test, and
  document, without first confirming *where* in a truncated file the
  defect-relevant content actually sits for this specific corpus.
- **The 55.7% truncation rate is an accepted, explicitly documented
  limitation** — reported honestly in the M8 evaluation writeup and the
  report's limitations section, not concealed.
- **Revisit trigger, stated now rather than left vague:** if M8's error
  analysis shows the model systematically underperforms on the *longer*
  files in a way shorter files don't, that's direct evidence for revisiting
  head+tail. Deciding this from evidence rather than upfront applies the same
  discipline the Master Plan already applies to the classification-head
  decision (§1, guardrail 3).

Implementation: `sdp.data.tokenization.head_truncate` — keeps the first
`max_length` tokens of the `[CLS] ... [SEP]`-wrapped sequence; if cut, forces
the final kept token to `[SEP]`.

## Decision: comment/whitespace preprocessing — not applied

**Decided 2026-09-25.** `dedup.py`'s docstring explicitly deferred this
question to M3 ("training-time decisions... applied to train only, after the
split is frozen") without committing to doing it.

Source code is tokenized exactly as extracted, unmodified. Reasoning:
competitive-programming submissions are typically lightly commented, so token
savings would likely be modest — and adding a preprocessing step without
measured justification is exactly the kind of complexity the project's
guardrail argues against. Revisit at M8 if the length-only baseline or error
analysis suggests comment/whitespace noise is actually hurting the model —
not before.

## Pipeline

- `scripts/measure_tokenization.py` — one-shot measurement (produced the
  table above); not part of the training pipeline itself.
- `scripts/pretokenize_corpus.py` — tokenizes the full corpus once
  (head-only truncation, `max_length=512`) and writes a per-split cache:
  `data/processed/tokenized/{train,val,test}.parquet`
  (`submission_id, input_ids, label`). Pre-tokenizing once — rather than
  tokenizing in `Dataset.__getitem__` or once per epoch — avoids repeating
  ~75,000 BPE tokenization calls for ids that never change between epochs;
  only the sampling order does.
- `sdp.data.dataset.SourceCodeDataset` — reads one split's cache; emits
  `{submission_id, input_ids, label}` per row.
- `sdp.data.dataset.collate_batch` — dynamic padding to the batch's own
  longest sequence, not a fixed 512. Given 55.7% of files already hit the
  512 cap after truncation, many batches need close-to-full-length padding
  regardless — but for the remaining ~44% of files (all shorter, some far
  shorter), padding to the batch max still avoids wasted compute versus
  always padding to 512. It costs nothing extra to include.
- `sdp.data.dataset.build_dataloaders` — wires the three frozen M2 splits
  into `DataLoader`s; `train` shuffles per epoch, `val`/`test` do not. No
  split is recomputed anywhere in this module — membership is entirely
  inherited from the M2-frozen `split` column.
- Labels emitted are **Tier-1** (`ERROR_FREE`/`COMPILE_ERROR`/`RUNTIME_ERROR`/
  `LOGICAL`, via `sdp.data.taxonomy.COARSE_TO_ID`). The 9-class leaf labels
  don't exist yet — M6 extends this same pipeline to emit them once M4
  (Tier 2) and M5 (Tier 3) produce the leaf-level labels.

## Verification

`scripts/pretokenize_corpus.py` ends with an end-to-end sanity check: it
builds the real `DataLoader`s and pulls one batch from each split, printing
tensor shapes and the per-split/per-class row counts. This is the evidence
for M3's exit criterion — a `DataLoader` per split producing correctly
shaped, correctly labelled batches — not just the cache files existing.
