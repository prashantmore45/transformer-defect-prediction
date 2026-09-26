"""Tokenization for the CodeBERT fine-tuning pipeline.

Milestone 3 scope
------------------
Two things, and only these two (per the Master Plan's "no unnecessary
complexity" guardrail — this module does not decide labels, splits, or
sampling; those are frozen M2 concerns):

1. Load the real `microsoft/codebert-base` tokenizer and measure the actual
   tokens-per-byte ratio against the frozen corpus, closing the M1 open item
   that used a ~3.5 bytes/token approximation (`docs/DATASET.md`).
2. Provide a documented, deterministic truncation strategy for sequences that
   exceed CodeBERT's 512-token limit.

Why head-only truncation
-------------------------
Three strategies exist: head-only (keep the first N tokens), tail-only, and
head+tail (keep some of each end). Head-only is the default here because:

  - C++ competitive-programming submissions overwhelmingly put `#include`,
    global declarations, and the start of `main()` first — for LOGICAL vs.
    ERROR_FREE discrimination in particular, the early control flow usually
    carries the signal.
  - It is the simplest strategy to implement, test, and explain at a viva.
  - Head+tail is a legitimate alternative (and cheap to add later behind the
    same interface) but is not justified up front without evidence that
    head-only underperforms — consistent with the project's stated
    engineering philosophy of adding complexity only when a measured problem
    calls for it.

This is a documented engineering trade-off, not a claim that it is optimal.

Why the tokenizer dependency is isolated behind a Protocol
-------------------------------------------------------------
`load_tokenizer()` requires network access to the Hugging Face Hub on first
call. Every other function in this module accepts anything satisfying the
narrow `Tokenizer` Protocol below, so the rest of the pipeline — and every
test in `tests/test_tokenization.py` — has zero network dependency. This is
the same Dependency Inversion pattern already used for `Predictor` in
`sdp/model/base.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd

#: microsoft/codebert-base uses RoBERTa's tokenizer and positional embeddings,
#: which cap sequence length at 512 tokens including [CLS]/[SEP].
DEFAULT_MODEL_NAME = "microsoft/codebert-base"
DEFAULT_MAX_LENGTH = 512

#: Decode fallback order. UTF-8 first (majority case per the M2 encoding
#: survey in docs/LABELING.md), then Shift-JIS (CodeNet has AtCoder-sourced
#: submissions with Japanese comments), then a lossy UTF-8 replace as a last
#: resort so a single bad file can never crash the whole measurement pass.
_DECODE_ENCODINGS: tuple[str, ...] = ("utf-8", "shift_jis")


class Tokenizer(Protocol):
    """The minimal interface this module needs from a tokenizer.

    A real `transformers.PreTrainedTokenizerFast` satisfies this. So does any
    lightweight stub used in tests — see `tests/test_tokenization.py`.
    """

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> Tokenizer:
    """Load the CodeBERT tokenizer from the Hugging Face Hub.

    Requires network access on first call; cached locally under
    `~/.cache/huggingface` afterward. Imports `transformers` lazily so that
    nothing else in this module — or any test that injects a stub — pays for
    that dependency.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _decode(raw: bytes) -> str:
    """Decode file bytes, falling back through `_DECODE_ENCODINGS`."""
    for encoding in _DECODE_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class TokenCount:
    """Full (untruncated) tokenization result for one source file."""

    submission_id: str
    byte_length: int
    token_length: int  # includes [CLS]/[SEP]

    @property
    def tokens_per_byte(self) -> float:
        return self.token_length / self.byte_length if self.byte_length else 0.0


def count_tokens(tokenizer: Tokenizer, submission_id: str, source: str) -> TokenCount:
    """Tokenize one file's full source (no truncation) and record its lengths."""
    byte_length = len(source.encode("utf-8"))
    token_ids = tokenizer.encode(source, add_special_tokens=True)
    return TokenCount(
        submission_id=submission_id,
        byte_length=byte_length,
        token_length=len(token_ids),
    )


def measure_corpus(
    tokenizer: Tokenizer,
    files: Iterable[tuple[str, Path]],
) -> pd.DataFrame:
    """Tokenize every `(submission_id, path)` pair; one row per file in the result.

    Reads each file as raw bytes and decodes via `_decode` rather than
    `Path.read_text`, so a single non-UTF-8 file cannot raise and abort the
    whole measurement pass — consistent with the M2 encoding survey already
    finding some Shift-JIS content in the corpus.
    """
    rows = [
        count_tokens(tokenizer, submission_id, _decode(path.read_bytes()))
        for submission_id, path in files
    ]
    return pd.DataFrame(r.__dict__ for r in rows)


def summarize(token_lengths: pd.Series, max_length: int = DEFAULT_MAX_LENGTH) -> dict[str, float]:
    """Headline statistics for a tokens-per-byte / truncation write-up."""
    return {
        "n_files": int(token_lengths.shape[0]),
        "mean_tokens": float(token_lengths.mean()),
        "median_tokens": float(token_lengths.median()),
        "p90_tokens": float(token_lengths.quantile(0.90)),
        "p95_tokens": float(token_lengths.quantile(0.95)),
        "p99_tokens": float(token_lengths.quantile(0.99)),
        "max_tokens": float(token_lengths.max()),
        "truncation_rate": truncation_rate(token_lengths, max_length),
    }


def truncation_rate(token_lengths: pd.Series, max_length: int = DEFAULT_MAX_LENGTH) -> float:
    """Fraction of files whose full (untruncated) token sequence exceeds `max_length`."""
    return float((token_lengths > max_length).mean())


def head_truncate(
    token_ids: list[int],
    max_length: int = DEFAULT_MAX_LENGTH,
    sep_token_id: int | None = None,
) -> list[int]:
    """Head-only truncation: keep the first `max_length` tokens.

    If the sequence is cut and `sep_token_id` is given, the final kept token
    is forced to SEP so a truncated sequence still ends the way CodeBERT
    expects (`[CLS] ... [SEP]`) rather than mid-token.
    """
    if len(token_ids) <= max_length:
        return token_ids
    truncated = token_ids[:max_length]
    if sep_token_id is not None:
        truncated[-1] = sep_token_id
    return truncated
