"""Unit tests for sdp.data.labeling.tier3 (pure logic, no compiler needed)."""

import pytest

from sdp.data.labeling.tier3 import (
    Tier3Discard,
    Tier3Label,
    classify,
    is_labelled,
)


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [
        (-11, Tier3Label.SIGSEGV),
        (-8, Tier3Label.SIGFPE),
        (-6, Tier3Label.SIGABRT),
        (1, Tier3Label.NZEC),
        (3, Tier3Label.NZEC),
        (255, Tier3Label.NZEC),
    ],
)
def test_leaf_labels(returncode, expected):
    assert classify(returncode) == expected


def test_exit_zero_is_not_reproduced():
    assert classify(0) == Tier3Discard.EXIT_ZERO


def test_bad_alloc_abort_is_resource_limit_not_sigabrt():
    stderr = "terminate called after throwing an instance of 'std::bad_alloc'"
    assert classify(-6, stderr) == Tier3Discard.RESOURCE_LIMIT


def test_other_uncaught_exception_abort_stays_sigabrt():
    stderr = "terminate called after throwing an instance of 'std::out_of_range'"
    assert classify(-6, stderr) == Tier3Label.SIGABRT


def test_bad_alloc_text_ignored_for_other_signals():
    assert classify(-11, "bad_alloc") == Tier3Label.SIGSEGV


def test_cpu_limit_signal_is_timeout():
    assert classify(-24) == Tier3Discard.TIMEOUT


def test_file_size_limit_signal_is_resource_limit():
    assert classify(-25) == Tier3Discard.RESOURCE_LIMIT


@pytest.mark.parametrize("returncode", [-4, -7, -9, -15])
def test_unlabelled_signals_are_flagged_not_raised(returncode):
    assert classify(returncode) == Tier3Discard.UNRECOGNIZED_SIGNAL


def test_timed_out_wins_over_returncode():
    # A process the worker killed has returncode -9; it must be TIMEOUT.
    assert classify(-9, timed_out=True) == Tier3Discard.TIMEOUT
    assert classify(0, timed_out=True) == Tier3Discard.TIMEOUT


def test_compile_failure_wins_over_everything():
    assert classify(-11, timed_out=True, compile_ok=False) == Tier3Discard.COMPILE_FAILED


def test_is_labelled():
    assert all(is_labelled(label) for label in Tier3Label)
    assert not any(is_labelled(d) for d in Tier3Discard)


def test_every_outcome_is_a_plain_string():
    assert str(classify(-11)) == "SIGSEGV"
    assert str(classify(0)) == "EXIT_ZERO"


LIBSTDCXX_ASSERT = (
    "/usr/include/c++/15/bits/stl_vector.h:1263: std::vector<int>::reference "
    "std::vector<int>::operator[](size_type): Assertion '__n < this->size()' failed."
)
USER_ASSERT = (
    "prog.out: /mnt/d/proj/data/sources/p00001/C++/s1.cpp:7: int main(): "
    "Assertion `x > 0' failed."
)


def test_libstdcxx_assertion_abort_is_toolchain_hardening():
    assert classify(-6, LIBSTDCXX_ASSERT) == Tier3Discard.TOOLCHAIN_HARDENING


def test_stack_smashing_abort_is_toolchain_hardening():
    stderr = "*** stack smashing detected ***: terminated"
    assert classify(-6, stderr) == Tier3Discard.TOOLCHAIN_HARDENING


def test_user_written_assert_stays_sigabrt():
    assert classify(-6, USER_ASSERT) == Tier3Label.SIGABRT


def test_hardening_markers_only_matter_for_sigabrt():
    assert classify(-11, LIBSTDCXX_ASSERT) == Tier3Label.SIGSEGV
