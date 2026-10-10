"""Tests for sdp.data.visibility (M6.5) -- pure functions, no corpus needed."""

from __future__ import annotations

import pytest

from sdp.data.visibility import (
    CONTENT_WINDOW,
    Visibility,
    classify_visibility,
    count_lines,
    echo_status,
    line_span,
)

# Token i covers characters [i, i + 1), so the token end offsets are 1..8.
_ENDS = [1, 2, 3, 4, 5, 6, 7, 8]
_WINDOW = 3  # the "model" sees content tokens 0, 1 and 2 only

_FIRST = "/p/s1.cpp:2:5: error: expected ';' before 'return'"
_STDERR = _FIRST + "\n    2 |     int x = 5\n      |              ^\n"


def test_default_window_is_510() -> None:
    assert CONTENT_WINDOW == 510


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 1), ("a", 1), ("a\n", 2), ("a\nb", 2), ("a\r\nb", 2), ("a\rb", 2)],
)
def test_count_lines(text: str, expected: int) -> None:
    assert count_lines(text) == expected


def test_line_span_handles_lf_and_crlf() -> None:
    text = "ab\ncd\r\nef"
    assert line_span(text, 1) == (0, 2)
    assert line_span(text, 2) == (3, 5)
    assert line_span(text, 3) == (7, 9)


def test_line_span_trailing_newline_gives_empty_last_line() -> None:
    assert line_span("ab\n", 2) == (3, 3)


def test_line_span_beyond_end_is_none() -> None:
    assert line_span("ab\ncd", 3) is None


def test_line_span_rejects_line_zero() -> None:
    with pytest.raises(ValueError):
        line_span("ab", 0)


def test_fully_visible_when_line_fits_window() -> None:
    assert classify_visibility(_ENDS, 0, 2, window=_WINDOW) is Visibility.FULLY_VISIBLE


def test_exact_fit_is_fully_visible() -> None:
    assert classify_visibility(_ENDS, 0, 3, window=_WINDOW) is Visibility.FULLY_VISIBLE


def test_partly_visible_when_line_crosses_window_edge() -> None:
    assert classify_visibility(_ENDS, 2, 6, window=_WINDOW) is Visibility.PARTLY_VISIBLE


def test_not_visible_when_line_starts_at_window_edge() -> None:
    # Three tokens lie before character 3, so this line's first token has
    # index 3 -- one past the last visible index (2).
    assert classify_visibility(_ENDS, 3, 5, window=_WINDOW) is Visibility.NOT_VISIBLE


def test_empty_token_list_is_fully_visible() -> None:
    assert classify_visibility([], 0, 5, window=_WINDOW) is Visibility.FULLY_VISIBLE


def test_classify_rejects_inverted_span() -> None:
    with pytest.raises(ValueError):
        classify_visibility(_ENDS, 5, 2, window=_WINDOW)


def test_echo_match() -> None:
    assert echo_status(_STDERR, _FIRST, 2, "    int x = 5") == "match"


def test_echo_ignores_whitespace_differences() -> None:
    assert echo_status(_STDERR, _FIRST, 2, "int\tx=  5") == "match"


def test_echo_mismatch() -> None:
    assert echo_status(_STDERR, _FIRST, 2, "int y = 6") == "mismatch"


def test_echo_wrong_line_number_is_no_echo() -> None:
    other = _STDERR.replace("    2 |", "    3 |")
    assert echo_status(other, _FIRST, 2, "    int x = 5") == "no_echo"


def test_echo_missing_snippet_is_no_echo() -> None:
    assert echo_status(_FIRST + "\n", _FIRST, 2, "int x = 5") == "no_echo"
