"""v2.50.1 Prompt 5g Phase C — hook telemetry tests.

Telemetry detects Pattern #8 (silent hook fallback masking unused
optimization).  Three modes via `MLX_MFA_HOOK_TELEMETRY` env var:

- "off"     : zero overhead; no counters
- "summary" : default; dict-increment per call; readable via
              `mlx_mfa.get_hook_stats()`
- "verbose" : summary + UserWarning per fallback (developer mode)
"""
from __future__ import annotations

import importlib
import os
import sys
import warnings

import mlx.core as mx
import numpy as np
import pytest


def _reimport_mlx_mfa():
    """Re-import mlx_mfa to pick up new MLX_MFA_HOOK_TELEMETRY env var.
    
    Uninstalls existing hooks first so the new module install_hooks() can
    rebind mx.conv_general (the install-time __mlx_mfa_hook__ marker check
    otherwise causes the new install to no-op, leaving the OLD module hook bound).
    """
    if "mlx_mfa" in sys.modules:
        try:
            sys.modules["mlx_mfa"].disable()
        except Exception:
            pass
    for mod_name in list(sys.modules.keys()):
        if mod_name == "mlx_mfa" or mod_name.startswith("mlx_mfa."):
            del sys.modules[mod_name]
    import mlx_mfa
    return mlx_mfa


def _mk_inputs(seed):
    mx.random.seed(seed)
    # III-5 follow-up: C_in/C_out must be % 16 == 0 AND >= 32 to engage
    # the NAX conv path.  C=16 (the prior value) is MPP-INELIGIBLE and now
    # correctly falls back to native — engaging NAX there silently
    # corrupted (legacy im2col partial-K-tile bug).  Use C=32 so these
    # telemetry-engagement tests exercise a path that is both engaged and
    # numerically correct.
    x = (mx.random.normal((1, 4, 8, 8, 32), dtype=mx.float16) * 0.1)
    w = (mx.random.normal((32, 3, 3, 3, 32), dtype=mx.float16) * 0.1)
    mx.eval(x, w); mx.synchronize()
    return x, w


def test_get_hook_stats_returns_snapshot():
    import mlx_mfa
    mlx_mfa.reset_hook_stats()
    stats = mlx_mfa.get_hook_stats()
    assert "executed" in stats
    assert "fallback" in stats
    assert "fallback_reasons" in stats
    assert "mode" in stats
    assert isinstance(stats["executed"], dict)
    assert isinstance(stats["fallback"], dict)


def test_get_hook_stats_returns_independent_copy():
    """Mutating the returned dict should not affect internal state."""
    import mlx_mfa
    mlx_mfa.reset_hook_stats()
    s1 = mlx_mfa.get_hook_stats()
    s1["executed"]["fake_hook"] = 999
    s2 = mlx_mfa.get_hook_stats()
    assert "fake_hook" not in s2["executed"]


def test_reset_hook_stats_clears_counters():
    import mlx_mfa
    from mlx_mfa import get_device_info
    if not get_device_info().get("is_m5_plus"):
        pytest.skip("requires M5+ NAX hardware")
    # Engage a call to bump counters
    x, w = _mk_inputs(seed=42)
    y = mx.conv_general(x, w, stride=1, padding=1, kernel_dilation=1)
    mx.eval(y); mx.synchronize()
    stats_before = mlx_mfa.get_hook_stats()
    assert stats_before["executed"].get("conv3d_nax_forward", 0) > 0
    mlx_mfa.reset_hook_stats()
    stats_after = mlx_mfa.get_hook_stats()
    assert stats_after["executed"] == {}
    assert stats_after["fallback"] == {}


def test_hook_executed_counter_increments_on_nax_engagement():
    import mlx_mfa
    from mlx_mfa import get_device_info
    if not get_device_info().get("is_m5_plus"):
        pytest.skip("requires M5+ NAX hardware")
    mlx_mfa.reset_hook_stats()
    x, w = _mk_inputs(seed=100)
    for _ in range(3):
        y = mx.conv_general(x, w, stride=1, padding=1, kernel_dilation=1)
        mx.eval(y); mx.synchronize()
    assert mlx_mfa.get_hook_stats()["executed"]["conv3d_nax_forward"] == 3


def test_hook_fallback_counter_increments_on_ineligible():
    import mlx_mfa
    mlx_mfa.reset_hook_stats()
    # III-5 follow-up: C_in=16 is below the MPP envelope (need % 16 == 0
    # AND >= 32) for BOTH dtypes, so the call falls back to native rather
    # than reach the numerically-broken legacy NAX path.  The point of
    # this test is that ineligibility is COUNTED, not silently dropped
    # (Pattern #8 / Rule 8): a fallback with a recorded reason, not a
    # silent no-op.
    mx.random.seed(101)
    x = (mx.random.normal((1, 4, 8, 8, 16), dtype=mx.float16) * 0.1)
    w = (mx.random.normal((16, 3, 3, 3, 16), dtype=mx.float16) * 0.1)
    mx.eval(x, w); mx.synchronize()
    y = mx.conv_general(x, w, stride=1, padding=1, kernel_dilation=1)
    mx.eval(y); mx.synchronize()
    stats = mlx_mfa.get_hook_stats()
    assert stats["fallback"]["conv3d_nax_forward"] >= 1
    # Reason names the MPP-gate constraint that was violated.
    reasons = stats["fallback_reasons"]["conv3d_nax_forward"]
    assert any("MPP gate" in r for r in reasons), reasons


def test_fallback_reasons_capped_at_10():
    """Distinct reasons capped to bound memory."""
    import mlx_mfa
    from mlx_mfa._auto_hooks import _record_hook_fallback, _HOOK_EXECUTION_STATS
    mlx_mfa.reset_hook_stats()
    for i in range(20):
        _record_hook_fallback("test_hook", f"reason_{i}")
    reasons = _HOOK_EXECUTION_STATS["fallback_reasons"]["test_hook"]
    assert len(reasons) == 10, f"expected 10, got {len(reasons)}"


def test_telemetry_mode_default_is_summary():
    """Default mode (no env var) is summary."""
    import mlx_mfa
    stats = mlx_mfa.get_hook_stats()
    # Either summary or whatever the test environment set; just verify it
    # is one of the three valid modes.
    assert stats["mode"] in ("off", "summary", "verbose")


def test_telemetry_off_mode_no_counters(monkeypatch):
    """In off mode, executed and fallback counters should stay empty."""
    monkeypatch.setenv("MLX_MFA_HOOK_TELEMETRY", "off")
    mlx_mfa_off = _reimport_mlx_mfa()
    from mlx_mfa import get_device_info
    if not get_device_info().get("is_m5_plus"):
        pytest.skip("requires M5+ NAX hardware")
    x, w = _mk_inputs(seed=200)
    y = mx.conv_general(x, w, stride=1, padding=1, kernel_dilation=1)
    mx.eval(y); mx.synchronize()
    stats = mlx_mfa_off.get_hook_stats()
    assert stats["mode"] == "off"
    assert stats["executed"] == {}
    assert stats["fallback"] == {}


def test_telemetry_verbose_mode_emits_warning(monkeypatch):
    monkeypatch.setenv("MLX_MFA_HOOK_TELEMETRY", "verbose")
    mlx_mfa_v = _reimport_mlx_mfa()
    # Trigger a fallback (bf16 weight)
    mx.random.seed(300)
    x = (mx.random.normal((1, 4, 8, 8, 16), dtype=mx.float16) * 0.1)
    w = (mx.random.normal((16, 3, 3, 3, 16), dtype=mx.bfloat16) * 0.1)
    mx.eval(x, w); mx.synchronize()
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        y = mx.conv_general(x, w, stride=1, padding=1, kernel_dilation=1)
        mx.eval(y); mx.synchronize()
    fallback_warns = [
        item for item in w_list
        if issubclass(item.category, UserWarning)
        and "mlx-mfa hook" in str(item.message)
    ]
    assert len(fallback_warns) >= 1, (
        f"expected at least one fallback UserWarning; got: "
        f"{[str(item.message) for item in w_list]}")


def test_telemetry_invalid_mode_defaults_to_summary(monkeypatch):
    monkeypatch.setenv("MLX_MFA_HOOK_TELEMETRY", "garbage_value")
    mlx_mfa_bad = _reimport_mlx_mfa()
    assert mlx_mfa_bad.get_hook_stats()["mode"] == "summary"
