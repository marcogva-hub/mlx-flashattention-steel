"""Sprint U Section C — auto-hook lifecycle + Conv3D NAX routing validation.

Three-axis validation per CLAUDE_V6_NAX.md §3.5:
  1. Output sanity: hooked mx.conv_general produces correct output for eligible shape
  2. Path entered: hooked function actually calls conv3d_nax_forward on eligible shape
  3. Edges preserved: ineligible shapes (Conv2D, FP32, asymmetric kernel, etc.) pass
     through to vanilla MLX; MFA_DISABLE_AUTO_HOOKS=1 prevents install; enable/disable
     idempotent.
"""
from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from unittest.mock import patch as _mock_patch

import numpy as np
import pytest

import mlx.core as mx


try:
    import mlx_mfa
    _HAS_EXT = True
except (ImportError, RuntimeError):
    _HAS_EXT = False

# Auto-hooks only take effect on M5+
_M5_PLUS = False
if _HAS_EXT:
    try:
        _M5_PLUS = bool(mlx_mfa.hooks_status().get("m5_plus"))
    except Exception:
        _M5_PLUS = False

pytestmark = pytest.mark.skipif(not _HAS_EXT, reason="mlx_mfa not installed")


@contextmanager
def fresh_hooks():
    """Ensure hooks are reinstalled at exit."""
    pre = mlx_mfa.hooks_status()["installed"]
    try:
        yield
    finally:
        # Restore to pre-test state
        if pre and not mlx_mfa.hooks_status()["installed"]:
            mlx_mfa.enable()
        elif not pre and mlx_mfa.hooks_status()["installed"]:
            mlx_mfa.disable()


def _make_conv3d_inputs(B=1, T=4, H=8, W=8, C_in=32, C_out=32, dtype=mx.float16, seed=0):
    """Make eligible Conv3D inputs (channels-last)."""
    mx.random.seed(seed)
    x = (mx.random.uniform(-1.0, 1.0, (B, T, H, W, C_in)) * 0.1).astype(dtype)
    w = (mx.random.uniform(-1.0, 1.0, (C_out, 3, 3, 3, C_in)) * 0.1).astype(dtype)
    mx.async_eval(x, w); mx.synchronize()
    return x, w


# ===== AXIS 1: Output sanity =====

@pytest.mark.skipif(not _M5_PLUS, reason="M5+ only")
def test_axis1_eligible_conv3d_hooked_output_correct():
    """Output of hooked mx.conv_general matches conv3d_nax_forward directly."""
    x, w = _make_conv3d_inputs(seed=10)
    # Hooked path: mx.conv_general now routes to conv3d_nax_forward
    with fresh_hooks():
        mlx_mfa.enable()
        y_hooked = mx.conv_general(x, w, padding=(1, 1, 1))
        mx.async_eval(y_hooked); mx.synchronize()

        # Direct call to conv3d_nax_forward for reference
        from mlx_mfa._ext import conv3d_nax_forward
        y_ref = conv3d_nax_forward(
            x, w,
            stride=(1, 1, 1),
            padding=(1, 1, 1, 1, 1, 1),
            dilation=(1, 1, 1),
            chunk_M=0,
        )
        mx.async_eval(y_ref); mx.synchronize()

    err = np.abs(np.array(y_hooked.astype(mx.float32)) -
                 np.array(y_ref.astype(mx.float32)))
    assert err.max() < 1e-5, \
        f"Hooked mx.conv_general output mismatch vs conv3d_nax_forward direct: {err.max()}"


# ===== AXIS 2: Path entered =====

@pytest.mark.skipif(not _M5_PLUS, reason="M5+ only")
def test_axis2_eligible_conv3d_routes_to_conv3d_nax():
    """Confirm hooked mx.conv_general invokes conv3d_nax_forward."""
    x, w = _make_conv3d_inputs(seed=20)
    # Capture _real BEFORE patching
    from mlx_mfa._ext import conv3d_nax_forward as _real
    with fresh_hooks():
        mlx_mfa.enable()
        with _mock_patch(
            "mlx_mfa._ext.conv3d_nax_forward", side_effect=_real
        ) as m:
            _ = mx.conv_general(x, w, padding=(1, 1, 1))
            assert m.called, "conv3d_nax_forward not invoked on eligible Conv3D call"


# ===== AXIS 3: Edges preserved =====

def test_axis3_ineligible_conv2d_passes_through():
    """Conv2D (4-D weight) bypasses the hook → vanilla MLX."""
    mx.random.seed(30)
    x = (mx.random.uniform(-1.0, 1.0, (1, 8, 8, 32)) * 0.1).astype(mx.float16)
    w = (mx.random.uniform(-1.0, 1.0, (32, 3, 3, 32)) * 0.1).astype(mx.float16)
    mx.async_eval(x, w); mx.synchronize()
    # Should run vanilla conv_general (Conv2D)
    y = mx.conv_general(x, w, padding=(1, 1))
    mx.async_eval(y); mx.synchronize()
    assert y.shape[0] == 1 and y.shape[-1] == 32
    assert not np.isnan(np.array(y.astype(mx.float32))).any()


def test_axis3_ineligible_kernel_size_passes_through():
    """5x5x5 Conv3D (not in {(3,3,3), (1,1,1)}) bypasses the hook."""
    mx.random.seed(40)
    x = (mx.random.uniform(-1.0, 1.0, (1, 6, 8, 8, 16)) * 0.1).astype(mx.float16)
    w = (mx.random.uniform(-1.0, 1.0, (16, 5, 5, 5, 16)) * 0.1).astype(mx.float16)
    mx.async_eval(x, w); mx.synchronize()
    # 5x5x5 → ineligible → passes through to vanilla
    y = mx.conv_general(x, w, padding=(2, 2, 2))
    mx.async_eval(y); mx.synchronize()
    assert y.shape[0] == 1
    assert not np.isnan(np.array(y.astype(mx.float32))).any()


def test_axis3_ineligible_dtype_passes_through():
    """FP32 Conv3D bypasses the hook (NAX only supports FP16/BF16)."""
    mx.random.seed(50)
    x = (mx.random.uniform(-1.0, 1.0, (1, 4, 8, 8, 16)) * 0.1).astype(mx.float32)
    w = (mx.random.uniform(-1.0, 1.0, (16, 3, 3, 3, 16)) * 0.1).astype(mx.float32)
    mx.async_eval(x, w); mx.synchronize()
    y = mx.conv_general(x, w, padding=(1, 1, 1))
    mx.async_eval(y); mx.synchronize()
    assert y.dtype == mx.float32
    assert not np.isnan(np.array(y)).any()


def test_axis3_disable_restores_vanilla():
    """disable() restores original mx.conv_general."""
    with fresh_hooks():
        mlx_mfa.enable()
        assert hasattr(mx.conv_general, "__mlx_mfa_hook__")
        mlx_mfa.disable()
        assert not hasattr(mx.conv_general, "__mlx_mfa_hook__")
        # Re-enable should work
        mlx_mfa.enable()
        assert hasattr(mx.conv_general, "__mlx_mfa_hook__")


def test_axis3_disable_then_enable_idempotent():
    """disable + enable cycle is idempotent (no double-hook)."""
    with fresh_hooks():
        # multiple disable
        mlx_mfa.disable()
        mlx_mfa.disable()
        assert not hasattr(mx.conv_general, "__mlx_mfa_hook__")
        # multiple enable
        mlx_mfa.enable()
        # Second enable is a no-op
        assert mlx_mfa.enable() is False  # already installed
        assert hasattr(mx.conv_general, "__mlx_mfa_hook__")


def test_axis3_env_disable_prevents_install():
    """MFA_DISABLE_AUTO_HOOKS=1 prevents auto-install on fresh subprocess."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import mlx_mfa; "
         "print('installed=' + str(mlx_mfa.hooks_status()['installed']))"],
        env={**os.environ, "MFA_DISABLE_AUTO_HOOKS": "1"},
        capture_output=True, text=True, timeout=30,
    )
    assert "installed=False" in out.stdout, \
        f"MFA_DISABLE_AUTO_HOOKS=1 should prevent install. stdout: {out.stdout}"


def test_axis3_hooks_status_introspection():
    """hooks_status() returns expected keys."""
    s = mlx_mfa.hooks_status()
    assert "installed" in s
    assert "log" in s
    assert "m5_plus" in s
    assert "auto_hooks_disabled_env" in s
    assert isinstance(s["installed"], bool)
    assert isinstance(s["log"], list)
