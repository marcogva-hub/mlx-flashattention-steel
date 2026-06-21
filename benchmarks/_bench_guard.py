"""Shared phantom-bench guard (audit H7 / H-09).  DEV/BENCH-ONLY (sdist-excluded).

THE PHANTOM-BENCH CLASS (drove the whole V6-backward saga): a bench times
`flash_attention(auto)`/MFA vs SDPA but on an interpreter where `mlx_mfa._ext`
failed to import (the documented Python/ABI mismatch trap), so BOTH arms silently
run SDPA → a fake "parity" ratio is reported.  The library was hardened against
this (`has_nax()`, `measured_speedup`), but the bench scripts were still ungated.

This guard makes the phantom FAIL LOUD (RULE 8) BEFORE any timing, reusing the
library's `has_nax()` as the single source of truth for "is the fast path live".

Two gates:
  - `require_nax_or_die()`  — for NAX/M5-specific benches: dies unless M5+ NAX is
    active.
  - `require_accel_or_die()` — for GENERIC MFA-vs-SDPA benches valid on any
    accelerated host: dies ONLY on the true phantom (`_ext` absent); PASSES on
    M1–M4 where `_ext` is loaded and STEEL (not NAX) provides genuine acceleration
    (so `flash_attention(auto)` ≠ SDPA there too — not a phantom).
"""
from __future__ import annotations

import sys


# Reuse the bench-validity error so the whole bench tree raises one type.
try:
    from benchmarks.bench_validity import FeatureUnavailable  # repo-root import
except Exception:  # pragma: no cover - import-path fallback
    try:
        from bench_validity import FeatureUnavailable  # same-dir import
    except Exception:
        class FeatureUnavailable(RuntimeError):  # last-resort local definition
            pass


def _nax_status() -> tuple[bool, str]:
    """(is_nax_active, reason_code) — never raises; mlx_mfa import failure is itself
    a die-worthy phantom condition."""
    try:
        import mlx_mfa
    except Exception as e:
        return False, f"mlx_mfa-import-failed:{type(e).__name__}"
    try:
        ok, code = mlx_mfa.has_nax(reason=True)
        return bool(ok), str(code)
    except Exception as e:
        return False, f"has_nax-failed:{type(e).__name__}"


def _die(label: str, code: str, need: str) -> "FeatureUnavailable":
    msg = (
        f"[phantom-bench guard] {label or 'this benchmark'} needs {need} but it is "
        f"NOT live (has_nax reason: {code!r}). Refusing to run — both arms would "
        f"silently fall to SDPA and report a fake 'parity' ratio (audit H7/H-09, the "
        f"phantom-bench class). Most common cause: the compiled `mlx_mfa._ext` did "
        f"not import (Python/ABI mismatch). Fix: run in the venv whose Python matches "
        f"the built extension — verify with `python -c 'import mlx_mfa._ext'`."
    )
    raise FeatureUnavailable(msg)


def require_nax_or_die(label: str = "") -> None:
    """Abort unless M5+ NAX acceleration is live.  For NAX/M5-specific benches."""
    ok, code = _nax_status()
    if ok:
        return
    _die(label, code, need="M5+ NAX acceleration")


def require_accel_or_die(label: str = "") -> None:
    """Abort only on the true phantom (`_ext` absent).  PASSES on M5 (NAX) and on
    M1–M4 (`_ext` loaded → STEEL accelerates, so MFA ≠ SDPA — genuine).  Use for
    generic MFA-vs-SDPA benches that are valid on any accelerated host."""
    ok, code = _nax_status()
    if ok or code == "pre-m5-hardware":
        return
    _die(label, code, need="mlx-mfa acceleration (mlx_mfa._ext)")


def nax_active() -> bool:
    """Non-fatal probe (for benches that adapt rather than abort)."""
    return _nax_status()[0]


if __name__ == "__main__":  # quick self-check
    ok, code = _nax_status()
    print(f"nax_active={ok} reason={code!r}", file=sys.stderr)
