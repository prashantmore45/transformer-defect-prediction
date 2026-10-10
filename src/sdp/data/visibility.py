"""Is a compile error's source line inside the text CodeBERT actually sees? (M6.5)

Background (beginner level)
---------------------------
CodeBERT reads at most 512 tokens. When a file is longer, `head_truncate`
(see `sdp.data.tokenization`) keeps `[CLS]`, the first 510 *content* tokens
and a final `[SEP]`. So the model sees content tokens number 0..509 and
nothing after. If the compiler's first error sits on a line past that point,
the evidence for the label is not in the model's input.

This module holds the small, pure pieces of that measurement so they can be
tested without the corpus, the compiler output or a tokenizer:

* `count_lines` / `line_span` -- locate a 1-based line in the source text.
  Any of "\\r\\n", "\\r" or "\\n" ends a line (the same rule g++ uses).
* `classify_visibility` -- given the end offset of every content token in the
  source text, say whether a line is fully, partly or not visible.
* `echo_status` -- g++ echoes the offending source line under each
  diagnostic ("    12 | int x = 5"). Comparing that echo with the real source
  line confirms the reported line number points where we think it does.

All offsets are character offsets into the same decoded string that was
tokenized (`sdp.data.tokenization._decode`), never byte offsets.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final, Sequence

import numpy as np

from sdp.data.tokenization import DEFAULT_MAX_LENGTH

#: Content tokens the model can see: 512 minus the [CLS] and [SEP] slots.
CONTENT_WINDOW: Final[int] = DEFAULT_MAX_LENGTH - 2

_EOL: Final[re.Pattern[str]] = re.compile(r"\r\n|\r|\n")
_ECHO: Final[re.Pattern[str]] = re.compile(r"^\s*(\d+) \| ?(.*)$")


class Visibility(StrEnum):
    """Where a source line falls relative to the model's input window."""

    FULLY_VISIBLE = "fully_visible"  # every token of the line is inside the window
    PARTLY_VISIBLE = "partly_visible"  # the line starts inside but runs past it
    NOT_VISIBLE = "not_visible"  # the line starts at or after the window's end


def count_lines(text: str) -> int:
    """Number of lines in `text`, counting the empty line after a final newline."""
    return len(_EOL.split(text))


def line_span(text: str, line_no: int) -> tuple[int, int] | None:
    """Character span [start, end) of 1-based line `line_no`, excluding its terminator.

    Returns None if the file has fewer lines than `line_no`. A trailing
    newline produces one empty last line, consistent with `count_lines`.
    """
    if line_no < 1:
        raise ValueError(f"line_no must be >= 1, got {line_no}")
    start = 0
    current = 1
    for match in _EOL.finditer(text):
        if current == line_no:
            return start, match.start()
        start = match.end()
        current += 1
    return (start, len(text)) if current == line_no else None


def classify_visibility(
    token_ends: Sequence[int],
    line_start: int,
    line_end: int,
    window: int = CONTENT_WINDOW,
) -> Visibility:
    """Classify a line against the first `window` content tokens.

    Args:
        token_ends: end character offset of every content token (special
            tokens excluded), in text order.
        line_start / line_end: the line's character span, terminator excluded.
        window: how many content tokens the model sees (510 by default).

    `before` is the number of tokens lying entirely before the line, so the
    line's first token has index `before`. The model sees indexes 0..window-1.
    `through` is the number of tokens that end inside or before the line.
    """
    if line_start < 0 or line_end < line_start:
        raise ValueError(f"invalid line span: ({line_start}, {line_end})")
    ends = np.asarray(token_ends)
    before = int((ends <= line_start).sum())
    through = int((ends <= line_end).sum())
    if before >= window:
        return Visibility.NOT_VISIBLE
    if through <= window:
        return Visibility.FULLY_VISIBLE
    return Visibility.PARTLY_VISIBLE


def _squeeze(text: str) -> str:
    """Remove all whitespace, so tab/space expansion cannot cause a false mismatch."""
    return re.sub(r"\s+", "", text)


def echo_status(stderr: str, first_error_line: str, line_no: int, source_line: str) -> str:
    """Compare g++'s echoed source line with the real source line.

    Returns:
        "match"    -- the echo under the first diagnostic equals the source line
                      (ignoring whitespace);
        "mismatch" -- an echo for this line number exists but differs;
        "no_echo"  -- no echo for this line number follows the diagnostic.
    """
    pos = stderr.find(first_error_line)
    if pos < 0:
        return "no_echo"
    following = stderr[pos + len(first_error_line) :].lstrip("\n").split("\n", 1)[0]
    match = _ECHO.match(following)
    if match is None or int(match.group(1)) != line_no:
        return "no_echo"
    return "match" if _squeeze(match.group(2)) == _squeeze(source_line) else "mismatch"
