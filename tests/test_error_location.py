"""Tests for tier2.first_error_line_number (M6.5 helper)."""

from sdp.data.labeling.tier2 import first_error_line_number

_OWN = "/mnt/d/proj/data/processed/sources/p00000/C++/s123.cpp"


def test_line_of_first_error_in_own_file():
    stderr = (
        f"{_OWN}:2:11: error: expected ';' before 'static'\n"
        "    2 | int x\n"
        f"{_OWN}:9:2: error: expected ';' after class definition\n"
    )
    assert first_error_line_number("s123", stderr) == 2


def test_error_in_a_system_header_gives_none():
    stderr = "/usr/include/c++/13/bits/foo.h:50:5: error: something broke\n"
    assert first_error_line_number("s123", stderr) is None


def test_fatal_error_line_is_parsed():
    stderr = f"{_OWN}:1:10: fatal error: foo.h: No such file or directory\n"
    assert first_error_line_number("s123", stderr) == 1


def test_error_without_a_location_gives_none():
    assert first_error_line_number("s123", "cc1plus: error: something odd\n") is None


def test_empty_stderr_gives_none():
    assert first_error_line_number("s123", "") is None


def test_include_chain_line_does_not_count_as_the_error_line():
    stderr = f"In file included from {_OWN}:1:\n" "/usr/include/foo.h:3:1: error: bad\n"
    assert first_error_line_number("s123", stderr) is None
