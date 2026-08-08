"""Tests for exact-hash deduplication and the cross-split leakage audit.

`test_audit_detects_a_planted_cross_split_duplicate` is the one that matters
most: it proves the audit function can actually catch leakage, not merely that
it returns zero on already-clean data (which could pass vacuously if the
function were broken).
"""

from __future__ import annotations

import pandas as pd
import pytest

from sdp.data.dedup import (
    assert_no_cross_split_duplicates,
    cross_split_duplicate_hashes,
    dedup_report,
    duplicate_groups,
    hash_corpus,
    hash_file,
    normalize_bytes,
)


@pytest.fixture
def corpus(tmp_path):
    """Six files: two byte-identical, one identical-after-normalization,
    three genuinely distinct.
    """
    files = {}

    files["a"] = tmp_path / "a.cpp"
    files["a"].write_bytes(b"int main(){return 0;}\n")

    files["b_dup_of_a"] = tmp_path / "b.cpp"
    files["b_dup_of_a"].write_bytes(b"int main(){return 0;}\n")

    files["c_crlf_of_a"] = tmp_path / "c.cpp"
    files["c_crlf_of_a"].write_bytes(b"int main(){return 0;}\r\n")

    files["d_trailing_ws_of_a"] = tmp_path / "d.cpp"
    files["d_trailing_ws_of_a"].write_bytes(b"int main(){return 0;}   \n")

    files["e_distinct"] = tmp_path / "e.cpp"
    files["e_distinct"].write_bytes(b"int main(){return 1;}\n")

    files["f_distinct"] = tmp_path / "f.cpp"
    files["f_distinct"].write_bytes(b"#include <iostream>\nint main(){}\n")

    return files


# --------------------------------------------------------------------------- #
# normalize_bytes
# --------------------------------------------------------------------------- #


def test_crlf_and_lf_normalize_to_the_same_bytes() -> None:
    assert normalize_bytes(b"a\r\nb\r\n") == normalize_bytes(b"a\nb\n")


def test_bom_is_stripped() -> None:
    assert normalize_bytes(b"\xef\xbb\xbfint main(){}") == normalize_bytes(b"int main(){}")


def test_trailing_whitespace_is_stripped_per_line() -> None:
    assert normalize_bytes(b"a   \nb\t\n") == normalize_bytes(b"a\nb\n")


def test_normalization_does_not_touch_content() -> None:
    """Different code must stay different after normalization."""
    assert normalize_bytes(b"return 0;\n") != normalize_bytes(b"return 1;\n")


def test_leading_whitespace_is_preserved() -> None:
    """Only trailing whitespace is stripped; indentation is real content."""
    assert normalize_bytes(b"    int x;\n") != normalize_bytes(b"int x;\n")


# --------------------------------------------------------------------------- #
# hash_file / hash_corpus
# --------------------------------------------------------------------------- #


def test_byte_identical_files_hash_equal(corpus) -> None:
    assert hash_file(corpus["a"]) == hash_file(corpus["b_dup_of_a"])


def test_crlf_variant_hashes_equal_after_normalization(corpus) -> None:
    assert hash_file(corpus["a"]) == hash_file(corpus["c_crlf_of_a"])


def test_trailing_whitespace_variant_hashes_equal(corpus) -> None:
    assert hash_file(corpus["a"]) == hash_file(corpus["d_trailing_ws_of_a"])


def test_distinct_content_hashes_differ(corpus) -> None:
    assert hash_file(corpus["a"]) != hash_file(corpus["e_distinct"])
    assert hash_file(corpus["a"]) != hash_file(corpus["f_distinct"])


def test_hash_corpus_reports_missing_files_without_raising(tmp_path, corpus) -> None:
    missing = tmp_path / "does_not_exist.cpp"
    result = hash_corpus([corpus["a"], missing])

    assert result["exists"].tolist() == [True, False]
    assert result.loc[result["path"].str.endswith("does_not_exist.cpp"), "sha256"].isna().all()


def test_hash_corpus_relativizes_paths_when_root_given(tmp_path, corpus) -> None:
    result = hash_corpus([corpus["a"]], root=tmp_path)
    assert result.loc[0, "path"] == "a.cpp"


# --------------------------------------------------------------------------- #
# duplicate_groups / dedup_report
# --------------------------------------------------------------------------- #


def test_duplicate_groups_finds_the_four_way_group(corpus) -> None:
    """a, b, c, d all normalize to the same content (plain/CRLF/trailing-ws)."""
    hashes = hash_corpus(list(corpus.values()))
    groups = duplicate_groups(hashes)

    assert len(groups) == 1
    assert groups.loc[0, "count"] == 4


def test_duplicate_groups_returns_empty_for_all_distinct_files(corpus) -> None:
    hashes = hash_corpus([corpus["a"], corpus["e_distinct"], corpus["f_distinct"]])
    groups = duplicate_groups(hashes)
    assert len(groups) == 0


def test_dedup_report_counts_match_the_fixture(corpus) -> None:
    hashes = hash_corpus(list(corpus.values()))
    report = dedup_report(hashes)

    assert report["files_hashed"] == 6
    assert report["files_missing"] == 0
    # a, b, c, d collapse to one hash; e and f are each their own hash -> 3 unique
    assert report["unique_hashes"] == 3
    assert report["redundant_files"] == 3  # 4 files in the group, 1 kept, 3 redundant
    assert report["duplicate_rate_pct"] > 0


def test_dedup_report_on_empty_input() -> None:
    empty = hash_corpus([])
    report = dedup_report(empty)
    assert report["files_hashed"] == 0
    assert report["duplicate_rate_pct"] == 0.0


# --------------------------------------------------------------------------- #
# Cross-split audit — the important one
# --------------------------------------------------------------------------- #


def _split_frame(hash_values: list[str], splits: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"sha256": hash_values, "split": splits})


def test_audit_passes_on_genuinely_disjoint_content() -> None:
    df = _split_frame(["h1", "h2", "h3"], ["train", "val", "test"])
    overlap = cross_split_duplicate_hashes(df)
    assert all(v == 0 for v in overlap.values())
    assert_no_cross_split_duplicates(df)  # must not raise


def test_audit_detects_a_planted_cross_split_duplicate() -> None:
    """The audit must actually catch leakage, not merely pass vacuously.

    Same reasoning as the analogous test in test_splitting.py: an audit that
    always passes on clean fixtures proves nothing about its ability to detect
    a real problem.
    """
    df = _split_frame(["h1", "h2", "h1", "h3"], ["train", "val", "train", "test"])
    # h1 appears in both train rows only -- no leak yet. Add a val row with h1.
    df = pd.concat([df, pd.DataFrame({"sha256": ["h1"], "split": ["val"]})], ignore_index=True)

    overlap = cross_split_duplicate_hashes(df)
    assert overlap["train&val"] == 1

    with pytest.raises(AssertionError) as exc:
        assert_no_cross_split_duplicates(df)
    assert "leaks across splits" in str(exc.value)


def test_audit_ignores_missing_hashes() -> None:
    df = pd.DataFrame({"sha256": ["h1", None, "h2"], "split": ["train", "val", "test"]})
    overlap = cross_split_duplicate_hashes(df)
    assert all(v == 0 for v in overlap.values())


def test_audit_covers_all_three_split_pairs() -> None:
    df = _split_frame(["h1"], ["train"])
    overlap = cross_split_duplicate_hashes(df)
    assert set(overlap.keys()) == {"train&val", "train&test", "val&test"}
