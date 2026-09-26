"""Tests for `sdp.data.tokenization`.

Deliberately network-free: every test here uses `StubTokenizer` instead of the
real CodeBERT tokenizer, so this suite runs in CI-less local `pytest` without
a Hugging Face Hub download. `load_tokenizer` itself (the one function that
does need the network) is exercised manually via
`scripts/measure_tokenization.py`, not here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sdp.data.tokenization import (
    DEFAULT_MAX_LENGTH,
    TokenCount,
    count_tokens,
    head_truncate,
    measure_corpus,
    summarize,
    truncation_rate,
)


class StubTokenizer:
    """Whitespace tokenizer standing in for CodeBERT's real BPE tokenizer.

    Deterministic and dependency-free: one "token" per whitespace-separated
    chunk, plus [CLS] (id 0) and [SEP] (id 1) when `add_special_tokens=True`.
    Real token *counts* will differ from CodeBERT's BPE, but every code path
    this module exercises (special-token accounting, byte counting,
    truncation) is identical in shape.
    """

    CLS_ID = 0
    SEP_ID = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # +2 offset keeps stub ids disjoint from CLS/SEP so truncation tests
        # can tell "real" tokens apart from special ones.
        ids = [i + 2 for i in range(len(text.split()))]
        if add_special_tokens:
            return [self.CLS_ID, *ids, self.SEP_ID]
        return ids


@pytest.fixture
def tokenizer() -> StubTokenizer:
    return StubTokenizer()


def test_count_tokens_includes_special_tokens(tokenizer: StubTokenizer) -> None:
    result = count_tokens(tokenizer, "sub_1", "int main ( ) { return 0 ; }")
    # 9 whitespace-separated chunks + [CLS] + [SEP]
    assert result.token_length == 11
    assert result.byte_length == len("int main ( ) { return 0 ; }".encode("utf-8"))


def test_tokens_per_byte_is_length_ratio(tokenizer: StubTokenizer) -> None:
    result = count_tokens(tokenizer, "sub_1", "a b c")
    assert result.tokens_per_byte == pytest.approx(result.token_length / result.byte_length)


def test_tokens_per_byte_zero_byte_file_is_zero(tokenizer: StubTokenizer) -> None:
    result = TokenCount(submission_id="empty", byte_length=0, token_length=2)
    assert result.tokens_per_byte == 0.0


def test_measure_corpus_reads_files_from_disk(tmp_path: Path, tokenizer: StubTokenizer) -> None:
    f1 = tmp_path / "a.cpp"
    f1.write_text("int main() { return 0; }")
    f2 = tmp_path / "b.cpp"
    f2.write_text("int x = 1; int y = 2; return x + y;")

    df = measure_corpus(tokenizer, [("sub_a", f1), ("sub_b", f2)])

    assert list(df["submission_id"]) == ["sub_a", "sub_b"]
    assert (df["token_length"] > 0).all()
    assert (df["byte_length"] > 0).all()


def test_measure_corpus_handles_non_utf8_bytes(tmp_path: Path, tokenizer: StubTokenizer) -> None:
    # Shift-JIS bytes for a Japanese comment — must not raise.
    f = tmp_path / "sjis.cpp"
    f.write_bytes("// コメント\nint main() {}".encode("shift_jis"))

    df = measure_corpus(tokenizer, [("sub_sjis", f)])

    assert len(df) == 1
    assert df.loc[0, "token_length"] > 0


def test_truncation_rate_counts_files_over_the_limit() -> None:
    lengths = pd.Series([100, 500, 512, 513, 1000])
    # Only 513 and 1000 strictly exceed 512.
    assert truncation_rate(lengths, max_length=512) == pytest.approx(2 / 5)


def test_truncation_rate_default_matches_codebert_limit() -> None:
    lengths = pd.Series([DEFAULT_MAX_LENGTH, DEFAULT_MAX_LENGTH + 1])
    assert truncation_rate(lengths) == pytest.approx(0.5)


def test_summarize_reports_expected_keys() -> None:
    lengths = pd.Series([10, 20, 30, 600])
    stats = summarize(lengths, max_length=512)

    assert stats["n_files"] == 4
    assert stats["max_tokens"] == 600
    assert stats["truncation_rate"] == pytest.approx(0.25)
    assert set(stats) == {
        "n_files",
        "mean_tokens",
        "median_tokens",
        "p90_tokens",
        "p95_tokens",
        "p99_tokens",
        "max_tokens",
        "truncation_rate",
    }


def test_head_truncate_no_op_when_under_limit() -> None:
    ids = [0, 2, 3, 4, 1]
    assert head_truncate(ids, max_length=10) == ids


def test_head_truncate_keeps_first_n_tokens() -> None:
    ids = list(range(20))
    truncated = head_truncate(ids, max_length=10)
    assert len(truncated) == 10
    assert truncated == ids[:10]


def test_head_truncate_forces_final_token_to_sep_when_cut() -> None:
    ids = list(range(20))
    sep_id = 99
    truncated = head_truncate(ids, max_length=10, sep_token_id=sep_id)
    assert len(truncated) == 10
    assert truncated[-1] == sep_id
    assert truncated[:-1] == ids[:9]


def test_head_truncate_exact_length_is_untouched() -> None:
    ids = list(range(10))
    assert head_truncate(ids, max_length=10, sep_token_id=99) == ids
