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


def record(backend: str, reason: str) -> None:
    """Append ``(backend, reason)`` to the active trace — no-op unless recording.

    Called at each routing terminal in ``flash_attention``.  The hot-path cost when
    not recording is a single global read + early return.
    """
    if not _RECORDING:
        return
    _TRACE.append((backend, reason))


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
