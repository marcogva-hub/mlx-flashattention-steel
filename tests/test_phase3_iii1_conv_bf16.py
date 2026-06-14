"""Phase III-1 (KD-7) — bf16 conv3d MPP lift locks.

The bf16 MPP convolution2d variant
(__tensorops_impl_convolution2d_op_run_cooperative_dv_bf_dv_bf_f32) was
probed II-2R-style: genuinely implemented at runtime (unlike int8's
header-only case), rel err <= 0.9% (single bf16 store rounding),
99.9-100% bit-identical to mx.conv3d bf16 at the production forms.

Locks:
  1. bf16 MPP-eligible shapes ENGAGE the hook (telemetry executed > 0)
     and match the original op within bf16 rounding.
  2. bf16 shapes OUTSIDE the MPP gate fall back to the original op
     bit-identically (the legacy im2col path is fp16-only — upstream
     MLX utils.h bf16 bug, KD-7).
  3. MFA_DISABLE_CONV3D_MPP=1 makes bf16 ineligible end-to-end (no
     graph-eval crash through the broken legacy path).
  4. Raw C++ API: bf16 reaching the legacy im2col path raises loudly
     at call time (Rule 8 defense-in-depth).
"""
from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx

from mlx_mfa import get_device_info
from mlx_mfa import _auto_hooks as ah

_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))
_skipif_no_nax = pytest.mark.skipif(not _HAS_NAX, reason="MPP requires M5+")


def _mk(T, H, W, C, O, dtype=mx.bfloat16, seed=5):
    mx.random.seed(seed)
    x = (mx.random.normal((1, T, H, W, C)) * 0.5).astype(dtype)
    w = (mx.random.normal((O, 3, 3, 3, C)) * 0.1).astype(dtype)
    mx.eval(x, w)
    return x, w


def _orig_conv3d(x, w):
    fn = ah._ORIGINAL_CONV3D if ah._ORIGINAL_CONV3D is not None else mx.conv3d
    return fn(x, w, stride=(1, 1, 1), padding=(1, 1, 1))


@pytest.fixture(autouse=True)
def _hooks():
    # III-4 F11: restore pre-test hook state in teardown so this file
    # does not leak installed hooks into tests that rely on the
    # unpatched ops (test_conv_nax.py references torch/mx.conv_general).
    was_installed = ah._HOOKS_INSTALLED
    ah.install_hooks()
    yield
    if not was_installed:
        ah.uninstall_hooks()


@_skipif_no_nax
class TestBF16MPPEngagement:
    @pytest.mark.parametrize("cell", [(8, 64, 64, 128, 128),
                                      (8, 32, 32, 256, 256)])
    def test_engages_and_matches(self, cell, monkeypatch):
        T, H, W, C, O = cell
        # III-4 F11: force telemetry on so the engagement assert can
        # never be vacuously skipped (the old if-guard silently passed
        # when MLX_MFA_HOOK_TELEMETRY=off).
        monkeypatch.setattr(ah, "_HOOK_TELEMETRY_MODE", "summary")
        x, w = _mk(T, H, W, C, O)
        before = ah._HOOK_EXECUTION_STATS["executed"]["conv3d_nax_forward"]
        out = mx.conv3d(x, w, stride=(1, 1, 1), padding=(1, 1, 1))
        mx.eval(out)
        after = ah._HOOK_EXECUTION_STATS["executed"]["conv3d_nax_forward"]
        assert after > before, "bf16 MPP cell did not engage the hook"
        ref = _orig_conv3d(x, w)
        mx.eval(ref)
        a = np.asarray(out.astype(mx.float32))
        b = np.asarray(ref.astype(mx.float32))
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-3)
        assert float(rel.max()) < 1e-2, f"bf16 MPP rel err {rel.max():.4f}"

    def test_fp16_path_deterministic(self):
        """III-4 F5: determinism lock for the fp16 path (same call twice
        is bit-identical).  This is NOT a correctness check — fp16 conv
        correctness is locked in test_conv_nax.py against the torch CPU
        FP32 ground truth."""
        x, w = _mk(8, 32, 32, 64, 64, dtype=mx.float16)
        out = mx.conv3d(x, w, stride=(1, 1, 1), padding=(1, 1, 1))
        out2 = mx.conv3d(x, w, stride=(1, 1, 1), padding=(1, 1, 1))
        mx.eval(out, out2)
        assert bool(mx.all(out == out2).item())


@_skipif_no_nax
class TestBF16FallbackGating:
    def test_non_mpp_shape_falls_back_bit_identical(self):
        # C=24 (not %16) -> outside the MPP gate -> original op.
        x, w = _mk(4, 24, 24, 24, 24)
        out = mx.conv3d(x, w, stride=(1, 1, 1), padding=(1, 1, 1))
        ref = _orig_conv3d(x, w)
        mx.eval(out, ref)
        assert bool(mx.all(out == ref).item()), \
            "non-MPP bf16 must fall back to the original op bit-identically"

    def test_env_optout_no_crash(self, monkeypatch):
        monkeypatch.setenv("MFA_DISABLE_CONV3D_MPP", "1")
        x, w = _mk(8, 64, 64, 128, 128)
        out = mx.conv3d(x, w, stride=(1, 1, 1), padding=(1, 1, 1))
        mx.eval(out)  # would raise at graph-eval if routed to legacy bf16
        assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item())

    def test_raw_cpp_legacy_bf16_raises_loudly(self):
        from mlx_mfa._ext import conv3d_nax_forward
        # k=3x3x3 but H=W=20 (not %8) -> MPP branch declines -> legacy
        # path -> must raise the KD-7 message at CALL time.
        x, w = _mk(4, 20, 20, 32, 32)
        with pytest.raises(Exception, match="bf16 is only supported via the MPP"):
            out = conv3d_nax_forward(
                x, w, stride=(1, 1, 1), padding=(1, 1, 1, 1, 1, 1),
                dilation=(1, 1, 1), chunk_M=0)
            mx.eval(out)
