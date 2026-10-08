"""
Tier 3 -- runtime-error sub-classification (SIGSEGV / SIGFPE / SIGABRT / NZEC).

What this module does
---------------------
Maps the raw outcome of *running* a compiled RUNTIME_ERROR submission to one
of four leaf labels, or to a recorded exclusion. It is a pure function: no
file access, no subprocess, no compiler. The execution itself lives in
`scripts/tier3_worker.py` (runs inside WSL/Linux); this module only
interprets what the worker captured. That split mirrors M4 (raw results are
stored, classification is re-applied locally), so the rules below can be
changed and re-run without re-executing 25,000 programs.

Background (beginner level)
---------------------------
When a program dies on Linux, the operating system reports one of two things:

* a *signal* -- the OS killed the process. Python reports this as a NEGATIVE
  return code: -11 means "killed by signal 11" (SIGSEGV, illegal memory
  access), -8 is SIGFPE (arithmetic fault, e.g. integer division by zero),
  -6 is SIGABRT (the program called abort(), which an uncaught C++ exception
  also does).
* a *normal exit with a non-zero status* -- e.g. `return 3;`. This is NZEC
  ("non-zero exit code"), kept distinct from the three signal classes.

Conscious differences from tier1.py (documented, not accidental)
-----------------------------------------------------------------
* Like tier2.classify(), unmatched input returns a review flag
  (UNRECOGNIZED_SIGNAL) instead of raising: the set of signals a program can
  die from is larger than the three we label, and one odd file must not
  stop a 25,000-file run.
* Signal numbers are written as Linux literals, not `signal.SIGXCPU` etc.:
  this module is imported by tests on Windows, where `signal` lacks several
  of these names. The worker always runs on Linux, so the numbers are right.

Toolchain-hardening hygiene
---------------------------
Some distro compilers (e.g. Ubuntu g++ 15) abort on an out-of-range
`v[i]` via a libstdc++ assertion, and turn stack overflows into "stack
smashing detected" aborts. A plain judge build would instead segfault or
corrupt memory silently, so the label SIGABRT would reflect the compiler
setting, not the original failure. Pilot v1 found 44 of 69 SIGABRT files were
exactly this. Compiler flags could not switch it off on g++ 15.2 (tried
-U_GLIBCXX_ASSERTIONS, an #undef via -include, -fno-hardened, and
-D_GLIBCXX_HARDEN=0), so such files are EXCLUDED (TOOLCHAIN_HARDENING)
instead of labelled. A genuine user-written assert() names the submission's
own source file, not /usr/include/c++/, so it stays SIGABRT.

Own-limit hygiene
-----------------
Our sandbox limits can themselves kill a program (memory cap -> bad_alloc
-> abort; CPU cap -> SIGXCPU; file-size cap -> SIGXFSZ). Those would look
like real defects if mapped naively, so they are mapped to exclusions
(TIMEOUT / RESOURCE_LIMIT) rather than leaf labels.
"""

from enum import StrEnum
from typing import Final

# Linux signal numbers (see module docstring for why these are literals).
_SIGABRT: Final = 6
_SIGFPE: Final = 8
_SIGSEGV: Final = 11
_SIGXCPU: Final = 24
_SIGXFSZ: Final = 25

# stderr marker printed by libstdc++ when an uncaught std::bad_alloc aborts
# the program. With our RLIMIT_AS cap this means "we ran out of the memory we
# allowed", not necessarily "the program is defective".
_BAD_ALLOC_MARKER: Final = "bad_alloc"

# Library-internal assertion: message is "<path under /usr/include/c++/...>:
# <line>: <function>: Assertion '...' failed." A user assert() instead names
# the submission's own file.
_LIBSTDCXX_PATH_MARKER: Final = "/usr/include/c++/"
_ASSERTION_MARKER: Final = "Assertion"
_STACK_SMASH_MARKER: Final = "stack smashing detected"


class Tier3Label(StrEnum):
    """The four leaf classes derived by M5. Values match the taxonomy names."""

    SIGSEGV = "SIGSEGV"
    SIGFPE = "SIGFPE"
    SIGABRT = "SIGABRT"
    NZEC = "NZEC"


class Tier3Discard(StrEnum):
    """Recorded exclusions. Every unlabelled file gets exactly one reason."""

    EXIT_ZERO = "EXIT_ZERO"  # ran cleanly: the crash did not reproduce
    TIMEOUT = "TIMEOUT"  # wall-clock limit or CPU limit (SIGXCPU)
    RESOURCE_LIMIT = "RESOURCE_LIMIT"  # our memory / file-size cap was hit
    # SIGABRT caused by the compiler's own hardening (libstdc++ assertion or
    # stack protector), not by the submission's failure mode.
    TOOLCHAIN_HARDENING = "TOOLCHAIN_HARDENING"
    COMPILE_FAILED = "COMPILE_FAILED"  # Tier-2 mismatch: does not compile
    UNRECOGNIZED_SIGNAL = "UNRECOGNIZED_SIGNAL"  # a signal we don't label


Tier3Outcome = Tier3Label | Tier3Discard

_SIGNAL_TO_LABEL: Final = {
    _SIGSEGV: Tier3Label.SIGSEGV,
    _SIGFPE: Tier3Label.SIGFPE,
}


def _is_toolchain_hardening(stderr_tail: str) -> bool:
    library_assertion = (
        _LIBSTDCXX_PATH_MARKER in stderr_tail and _ASSERTION_MARKER in stderr_tail
    )
    return library_assertion or _STACK_SMASH_MARKER in stderr_tail


def classify(
    returncode: int,
    stderr_tail: str = "",
    *,
    timed_out: bool = False,
    compile_ok: bool = True,
) -> Tier3Outcome:
    """Map one execution result to a leaf label or a recorded exclusion.

    Args:
        returncode: the binary's return code as Python reports it on Linux
            (negative = killed by that signal number).
        stderr_tail: the last part of the program's stderr (used only to
            recognise our own memory cap via std::bad_alloc).
        timed_out: True if the worker killed the run at the wall-clock limit.
        compile_ok: False if recompilation failed, so nothing was executed.

    Rule order matters and is deliberate: compile failure and timeout are
    decided before the return code is even looked at, because a process we
    killed ourselves has a meaningless (-9) return code.
    """
    if not compile_ok:
        return Tier3Discard.COMPILE_FAILED
    if timed_out:
        return Tier3Discard.TIMEOUT
    if returncode == 0:
        return Tier3Discard.EXIT_ZERO
    if returncode > 0:
        return Tier3Label.NZEC

    signum = -returncode
    if signum == _SIGABRT:
        if _BAD_ALLOC_MARKER in stderr_tail:
            return Tier3Discard.RESOURCE_LIMIT
        if _is_toolchain_hardening(stderr_tail):
            return Tier3Discard.TOOLCHAIN_HARDENING
        return Tier3Label.SIGABRT
    if signum == _SIGXCPU:
        return Tier3Discard.TIMEOUT
    if signum == _SIGXFSZ:
        return Tier3Discard.RESOURCE_LIMIT
    if signum in _SIGNAL_TO_LABEL:
        return _SIGNAL_TO_LABEL[signum]
    return Tier3Discard.UNRECOGNIZED_SIGNAL


def is_labelled(outcome: Tier3Outcome) -> bool:
    """True only for the four real leaf classes (not discards/review flags)."""
    return isinstance(outcome, Tier3Label)
