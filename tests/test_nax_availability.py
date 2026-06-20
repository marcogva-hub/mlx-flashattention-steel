"""Anti-silent-fallback contract (RULE 8) for the shipped library.

When `mlx_mfa._ext` fails to import, the library routes to SDPA — correct results,
zero acceleration. For an acceleration library that is a silent VALUE failure (and it
caused our multi-session phantom benches: a 3.14 venv importing a 3.11-built `_ext`).
This locks the loud + diagnosable contract:
  - `has_nax()` reports availability (+ reason);
  - the UNEXPECTED fallback (Apple Silicon but `_ext` failed) warns LOUDLY;
  - the EXPECTED fallbacks (non-target platform / pre-M5) stay quiet;
  - opt-in strict mode RAISES;
  - the NAX-present path does NOT warn.

Detection is `_ext`-independent (platform module) so these run on any host — the
`_ext`-absent and platform cases are simulated by monkeypatch.
"""
from __future__ import annotations

import warnings

import pytest

import mlx_mfa
from mlx_mfa import attention as A


@pytest.fixture(autouse=True)
def _reset_fallback_state():
    """Reset the one-time warning flag + availability caches around each test so the
    monkeypatched scenarios are independent."""
    A._warned_unexpected_fallback = False
    yield
    A._warned_unexpected_fallback = False


def test_has_nax_is_exported_and_returns_bool():
    assert hasattr(mlx_mfa, "has_nax") and hasattr(mlx_mfa, "NaxUnavailable")
    assert isinstance(mlx_mfa.has_nax(), bool)
    ok, code = mlx_mfa.has_nax(reason=True)
    assert isinstance(ok, bool) and isinstance(code, str)
    # reason is self-consistent with the bool
    assert (code == "available") == ok


def test_unexpected_fallback_warns_loudly(monkeypatch):
    """Apple Silicon + `_ext` failed → the whole accelerator is down → LOUD warning
    that names the likely cause (ABI/build) and how to check."""
    monkeypatch.setattr(A, "_get_has_nax_cached", lambda: False)
    monkeypatch.setattr(A, "_ext_available", lambda: False)
    monkeypatch.setattr(A, "_is_apple_silicon", lambda: True)
    monkeypatch.delenv("MFA_SILENCE_NAX_WARNING", raising=False)
    monkeypatch.delenv("MFA_REQUIRE_NAX", raising=False)
    A._warned_unexpected_fallback = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        A._warn_if_acceleration_unavailable()
    assert len(w) == 1 and issubclass(w[0].category, RuntimeWarning)
    msg = str(w[0].message)
    assert "_ext" in msg and "SDPA" in msg and "import mlx_mfa._ext" in msg
    # reason code is the unexpected one
    assert A.has_nax(reason=True) == (False, "ext-load-failed")


def test_unexpected_fallback_warns_only_once(monkeypatch):
    monkeypatch.setattr(A, "_get_has_nax_cached", lambda: False)
    monkeypatch.setattr(A, "_ext_available", lambda: False)
    monkeypatch.setattr(A, "_is_apple_silicon", lambda: True)
    monkeypatch.delenv("MFA_SILENCE_NAX_WARNING", raising=False)
    A._warned_unexpected_fallback = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        A._warn_if_acceleration_unavailable()
        A._warn_if_acceleration_unavailable()  # second call: no-op
    assert len(w) == 1


def test_expected_fallback_non_target_platform_is_silent(monkeypatch):
    """Non-Apple-Silicon (Linux / Intel Mac): SDPA fallback is normal → NO warning."""
    monkeypatch.setattr(A, "_get_has_nax_cached", lambda: False)
    monkeypatch.setattr(A, "_ext_available", lambda: False)
    monkeypatch.setattr(A, "_is_apple_silicon", lambda: False)
    A._warned_unexpected_fallback = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        A._warn_if_acceleration_unavailable()
    assert len(w) == 0
    assert A.has_nax(reason=True) == (False, "unsupported-platform")


def test_expected_fallback_pre_m5_is_silent(monkeypatch):
    """M1–M4: `_ext` loaded, STEEL kernels accelerate, only NAX absent → NO warning."""
    monkeypatch.setattr(A, "_get_has_nax_cached", lambda: False)
    monkeypatch.setattr(A, "_ext_available", lambda: True)
    monkeypatch.setattr(A, "_is_apple_silicon", lambda: True)
    A._warned_unexpected_fallback = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        A._warn_if_acceleration_unavailable()
    assert len(w) == 0
    assert A.has_nax(reason=True) == (False, "pre-m5-hardware")


def test_strict_mode_raises(monkeypatch):
    """has_nax(strict=True) and MFA_REQUIRE_NAX=1 both RAISE NaxUnavailable when off."""
    monkeypatch.setattr(A, "_get_has_nax_cached", lambda: False)
    monkeypatch.setattr(A, "_ext_available", lambda: False)
    monkeypatch.setattr(A, "_is_apple_silicon", lambda: True)
    with pytest.raises(mlx_mfa.NaxUnavailable, match=r"_ext|SDPA"):
        A.has_nax(strict=True)
    # env-gated strict mode (the import-time guarantee)
    monkeypatch.setenv("MFA_REQUIRE_NAX", "1")
    A._warned_unexpected_fallback = False
    with pytest.raises(mlx_mfa.NaxUnavailable):
        A._warn_if_acceleration_unavailable()


def test_silence_env_suppresses_warning(monkeypatch):
    monkeypatch.setattr(A, "_get_has_nax_cached", lambda: False)
    monkeypatch.setattr(A, "_ext_available", lambda: False)
    monkeypatch.setattr(A, "_is_apple_silicon", lambda: True)
    monkeypatch.setenv("MFA_SILENCE_NAX_WARNING", "1")
    monkeypatch.delenv("MFA_REQUIRE_NAX", raising=False)
    A._warned_unexpected_fallback = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        A._warn_if_acceleration_unavailable()
    assert len(w) == 0


@pytest.mark.skipif(not mlx_mfa.has_nax(), reason="requires NAX-present host (M5 + _ext)")
def test_nax_present_path_does_not_warn():
    """On a real NAX-present host, importing/checking must be SILENT (no false alarm)."""
    A._warned_unexpected_fallback = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        A._warn_if_acceleration_unavailable()
    assert len(w) == 0
    assert mlx_mfa.has_nax() is True
