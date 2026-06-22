"""Test-only attention-dispatch telemetry — the P-H1 routing-equivalence safety net.

WHY THIS EXISTS (Lot-2 lesson): the suite asserts *correctness*, not *which path
ran*.  A routing regression (wrong backend silently selected, output still correct)
would pass every correctness test — exactly the risk that makes the P-H1 dispatch-tree
refactor unsafe.  This module gives tests a robust, exact signal of which backend the
Python routing layer selected, so a test can assert "this call ran NAX", not infer it
from timing or a coarse byteΔ fingerprint.

DESIGN / COST:
- ``record(backend, reason)`` is a **no-op** (one module-global bool check + return)
  unless a test has opened ``capture()``.  Zero allocation, zero behavior change in
  production — the recording flag is OFF by default and never flipped by library code.
- This module imports **nothing** from ``mlx_mfa`` (no import cycle); ``attention.py``
  imports it at module top as a cheap name binding.
- It records the **Python routing decision** (the dispatch tree P-H1 refactors:
  nax_dense / sdpa-* / mfa_primitive / sage / *_bias / ...).  The downstream C++
  variant pick (V2 vs V3 inside the ``mfa_primitive`` path) is a *separate* layer —
  capture that via the ``MFA_DISABLE_V3`` behavioral toggle (output changes ⟺ V3 ran),
  not via this recorder.

NOT part of the public API (absent from ``mlx_mfa.__all__``).  Test-support only.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

# OFF by default.  Library code never sets this True — only ``capture()`` (tests) does.
_RECORDING: bool = False
_TRACE: list[tuple[str, str]] = []
# Re-entrancy depth.  > 0 while an internal, non-caller dispatch is running
# (background JIT warmup, nested routing).  Records emitted at depth > 0 are
# tagged ``[reentrant]`` so a terminal-record consumer cannot mistake them for
# the caller's own route (volet-A / CC-15).
_REENTRANT_DEPTH: int = 0

# Prefix stamped onto the reason of any record emitted during re-entrant
# internal routing.  Exposed so tests can filter without string-guessing.
REENTRANT_PREFIX: str = "[reentrant] "


def recording() -> bool:
    """True iff a ``capture()`` block is active.

    Lets a caller skip building an expensive trace label when no test is
    listening — ``record()`` is already a no-op when not recording, so the
    label is irrelevant in production.  Keeps which-binary labelling
    zero-overhead off the test path (volet-A / CX-08).
    """
    return _RECORDING


def record(backend: str, reason: str) -> None:
    """Append ``(backend, reason)`` to the active trace — no-op unless recording.

    Called at each routing terminal in ``flash_attention``.  The hot-path cost when
    not recording is a single global read + early return.

    Records emitted while ``reentrant()`` is active (background warmup / nested
    dispatch) get a ``[reentrant]`` reason prefix — they remain observable but
    are unmistakable for the caller's real terminal (which is recorded OUTSIDE
    the re-entrant scope, so ``tr[-1]`` is unaffected).
    """
    if not _RECORDING:
        return
    if _REENTRANT_DEPTH > 0:
        reason = REENTRANT_PREFIX + reason
    _TRACE.append((backend, reason))


@contextmanager
def reentrant():
    """Mark records emitted within as internal re-entrant routing.

    Wrap any internal call path that re-enters the routing layer but is NOT
    the caller's own dispatch decision (e.g. ``_auto_warmup_background``'s
    process-once JIT warmup, which fires 8 small forward passes on the first
    MFA-capable call).  Without this, those records pollute an open
    ``capture()`` and a test asserting ``tr[0]`` / ``len(tr) == 1`` would be
    misled (CC-15).  Nestable; ``tr[-1]`` stays the caller's real terminal.
    """
    global _REENTRANT_DEPTH
    _REENTRANT_DEPTH += 1
    try:
        yield
    finally:
        _REENTRANT_DEPTH -= 1


@contextmanager
def capture():
    """Enable recording for the duration of the block; yields the live trace list.

    Nestable (saves/restores prior state).  Usage::

        with capture() as tr:
            flash_attention(q, k, v, causal=True)
        assert tr[-1][0] == "nax_dense"
    """
    global _RECORDING, _TRACE
    prev_recording, prev_trace = _RECORDING, _TRACE
    _RECORDING, _TRACE = True, []
    try:
        yield _TRACE
    finally:
        _RECORDING, _TRACE = prev_recording, prev_trace


def last_backend() -> Optional[str]:
    """The backend label of the most recent recorded decision, or None."""
    return _TRACE[-1][0] if _TRACE else None


def last_decision() -> Optional[tuple[str, str]]:
    """The most recent ``(backend, reason)`` recorded, or None."""
    return _TRACE[-1] if _TRACE else None
