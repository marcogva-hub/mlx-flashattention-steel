"""Suite-wide guard: no engagement test may rely on a FRAGILE trace position.

volet-A / CC-15.  The dispatch trace records the caller's routing terminal at
``tr[-1]`` — but a *re-entrant* internal call (the process-once
``_auto_warmup_background``, which fires 8 small forward passes on the first
MFA-capable call) prepends records to an open ``capture()``.  Those are now
tagged ``[reentrant]`` (see ``mlx_mfa/_dispatch_trace.py``), but a test that
asserts ``tr[0]`` or ``len(tr) == 1`` instead of ``tr[-1]`` would still be
misled the moment it runs as the first MFA call in a fresh process.

This meta-test greps every test module and FAILS if any of them reaches for the
first trace element or pins an exact trace length — forcing the robust
``tr[-1]`` / ``last_backend()`` / ``last_decision()`` contract everywhere.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).parent
_SELF = Path(__file__).name

# A test file "consumes the trace" if it opens a capture / touches the recorder.
_CONSUMES_TRACE = re.compile(
    r"_dispatch_trace|_dtrace|\.capture\(\)|last_backend|last_decision")

# Fragile patterns: the FIRST element of a trace list, or an exact-length pin
# (``== 1`` / ``< 2``) on a trace list.  We bind to the conventional local names
# (tr / trace / _tr / _trace) so this stays low-false-positive.
_FRAGILE = re.compile(
    r"\b(?:tr|_tr|trace|_trace)\[0\]"
    r"|len\(\s*(?:tr|_tr|trace|_trace)\s*\)\s*(?:==\s*1|<\s*2|<=\s*1)")


def _trace_consumer_files():
    out = []
    for p in sorted(_TESTS_DIR.glob("test_*.py")):
        if p.name == _SELF:
            continue
        txt = p.read_text(encoding="utf-8")
        if _CONSUMES_TRACE.search(txt):
            out.append((p, txt))
    return out


def test_no_test_relies_on_fragile_trace_position():
    """No trace-consuming test may assert tr[0] or len(tr)==1 (CC-15)."""
    offenders = []
    for p, txt in _trace_consumer_files():
        for i, line in enumerate(txt.splitlines(), 1):
            # ignore comments
            code = line.split("#", 1)[0]
            if _FRAGILE.search(code):
                offenders.append(f"{p.name}:{i}: {line.strip()}")
    assert not offenders, (
        "engagement test relies on a fragile trace position — use tr[-1] / "
        "last_backend() instead (a re-entrant warmup record can occupy tr[0]):\n"
        + "\n".join(offenders))


def test_some_test_actually_consumes_the_trace():
    """Anti-vacuity: the guard above is meaningful only if trace tests exist."""
    files = _trace_consumer_files()
    assert len(files) >= 1, "no test consumes _dispatch_trace — guard is vacuous"


def test_guard_regex_bites():
    """Prove the fragile-pattern detector actually matches the bad forms
    (so this guard cannot silently rot into a no-op)."""
    bad = [
        "assert tr[0][0] == 'nax_dense'",
        "assert trace[0] == ('mfa_primitive', 'x')",
        "assert len(tr) == 1",
        "assert len(trace) < 2",
        "if len(_tr) <= 1:",
    ]
    good = [
        "assert tr[-1][0] == 'nax_dense'",
        "assert last_backend() == 'apple_sdpa'",
        "assert trace[-1] == ('sdpa', 'x')",
        "assert len(tr) >= 1",
    ]
    for b in bad:
        assert _FRAGILE.search(b.split("#", 1)[0]), f"detector missed: {b!r}"
    for g in good:
        assert not _FRAGILE.search(g.split("#", 1)[0]), f"detector false-positive: {g!r}"
