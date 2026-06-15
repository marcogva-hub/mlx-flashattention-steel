"""III-5 follow-up — conv3d NAX small-channel accuracy regression.

Root cause (see docs/v50/campaign-2026-06/phase3/conv-small-channel-fix.md):
`conv3d_nax_forward` has two internal paths — the MPP convolution2d
branch (gated C_in/C_out %16==0 & >=32) and a legacy im2col+matmul2d
fallback.  The legacy matmul2d K-loop reads partial K-tiles past K_FULL
with no tail mask; K = C_in*27 is a multiple of the 32-wide K-tile iff
C_in % 32 == 0, and every such C_in already takes the MPP path — so the
legacy path is reached ONLY by inputs for which it is numerically broken
(fp16 C_in=16 -> ~0.11 MAE/RMS; C_in=31 -> NaN).  The bf16 dispatch was
gated away from it; fp16 was not, so small-channel fp16 silently
corrupted.

This was invisible because every prior conv test used the MPP envelope
(C_in >= 32, % 16 == 0) — a single-shape-class coverage gap (III-4
lesson #10).  This file sweeps the SMALL-channel regime for BOTH dtypes
and asserts the hook output matches an fp32 reference within the dtype
floor, regardless of which internal path is taken.

Fix (III-5): `_conv3d_mpp_eligible` gates BOTH fp16 and bf16, so any shape
outside the MPP envelope falls back to the native op (the production hook
path).

Fix (III-6): the TRUE root cause — the matmul2d unmasked partial-K-tile —
is fixed at the kernel level by zero-padding the contraction K to a K_TILE
multiple (mfa_conv_nax.cpp pad_contraction_k / conv_nax.py _pad_k).  All
three entry points (C++ legacy im2col, C++ 1x1x1 pointwise, the Python
_conv3d_nax_forward_python_legacy orchestrator) are now correct at ALL
C_in.  The production hook still routes small-channel to native — the R.2
bench showed native is 1.7x faster there (orchestration overhead dominates
at tiny C_in; Pattern #6) — so the kernel fix is correctness defence for
raw-API callers, not a perf re-widening.  matmul2d_source now REFUSES a
K that is not K_TILE-aligned (Rule 8), so a future unpadded caller fails
loudly instead of silently corrupting.
"""
from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx

from mlx_mfa import get_device_info
from mlx_mfa import _auto_hooks as ah

_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))
_skipif_no_nax = pytest.mark.skipif(not _HAS_NAX, reason="NAX conv requires M5+")

# Dtype floor for conv3d vs an fp32 reference (single low-precision store
# rounding): fp16 ~1.4e-4, bf16 ~1.1e-3.  The broken legacy path produced
# 0.11 .. 4.2 (or NaN) — far above this bound — so 0.01 is a clean guard.
_MAE_RMS_BOUND = 0.01


def _ref_fp32(x, w):
    """True ground truth: native conv in fp32."""
    fn = ah._ORIGINAL_CONV3D if ah._ORIGINAL_CONV3D is not None else mx.conv3d
    return fn(x.astype(mx.float32), w.astype(mx.float32),
              stride=(1, 1, 1), padding=(1, 1, 1))


def _mae_rms(y, yf):
    yf32 = yf.astype(mx.float32)
    rms = float(mx.sqrt((yf32 ** 2).mean()).item())
    return float(mx.abs(y.astype(mx.float32) - yf32).mean().item()) / max(rms, 1e-6)


@pytest.fixture(autouse=True)
def _hooks():
    # Force a clean (re)install regardless of stale global hook state left
    # by other test files.  install_hooks() early-returns when
    # _HOOKS_INSTALLED is True, which can desync from the actual patch
    # state under cross-file ordering — uninstall first to guarantee the
    # patch is genuinely active for the engagement assertions.
    # Force a clean (re)install so the accuracy/determinism tests exercise
    # the NAX path for eligible shapes regardless of stale global hook
    # state left by other test files (install_hooks() early-returns on a
    # stale _HOOKS_INSTALLED flag).
    was_installed = ah._HOOKS_INSTALLED
    ah.uninstall_hooks()
    ah.install_hooks()
    yield
    if not was_installed:
        ah.uninstall_hooks()


# C_in sweep crossing the MPP envelope boundary in every way:
#   below 32, multiples of 16 (8,16), non-multiples (17,24,31),
#   on the boundary (32), above (48,64,128).
_CIN_SWEEP = [8, 16, 17, 24, 31, 32, 33, 48, 64, 128]
_DTYPES = [mx.float16, mx.bfloat16]


@_skipif_no_nax
class TestConvSmallChannelAccuracy:
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("c_in", _CIN_SWEEP)
    def test_conv3d_hook_matches_fp32(self, c_in, dtype):
        """mx.conv3d via the auto-hook matches fp32 within the dtype floor
        for ALL channel counts — small channels must fall back to the
        native op rather than reach the broken legacy NAX path."""
        c_out = 64
        mx.random.seed(3)
        x = mx.random.normal((1, 8, 8, 8, c_in)).astype(dtype)
        w = mx.random.normal((c_out, 3, 3, 3, c_in)).astype(dtype)
        y = mx.conv3d(x, w, stride=1, padding=1)
        yf = _ref_fp32(x, w)
        mx.eval(y, yf)
        assert not bool(mx.any(mx.isnan(y)).item()), \
            f"NaN in conv3d output (c_in={c_in}, {dtype})"
        err = _mae_rms(y, yf)
        assert err < _MAE_RMS_BOUND, (
            f"conv3d c_in={c_in} {dtype}: MAE/RMS {err:.5f} >= "
            f"{_MAE_RMS_BOUND} — small-channel NAX accuracy regression")

    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("c_in", [8, 16, 31, 32, 64])
    def test_conv3d_hook_deterministic(self, c_in, dtype):
        """Re-running the same conv must be bit-identical (guards against
        the stale-pool-read class that the topk CRITICAL exhibited)."""
        c_out = 64
        mx.random.seed(7)
        x = mx.random.normal((1, 8, 8, 8, c_in)).astype(dtype)
        w = mx.random.normal((c_out, 3, 3, 3, c_in)).astype(dtype)
        y1 = mx.conv3d(x, w, stride=1, padding=1)
        y2 = mx.conv3d(x, w, stride=1, padding=1)
        mx.eval(y1, y2)
        maxdiff = float(mx.abs(y1.astype(mx.float32) - y2.astype(mx.float32)).max().item())
        assert maxdiff == 0.0, \
            f"conv3d c_in={c_in} {dtype} nondeterministic (maxdiff {maxdiff})"

    # The gate predicate is tested directly (pure function — no dependence
    # on installed-hook state or telemetry-mode globals, which leak across
    # test files).  This is the precise lock on the fix: which shapes the
    # NAX conv path is allowed to handle.
    _PAD1 = (1, 1, 1, 1, 1, 1)

    @pytest.mark.parametrize("c_in,c_out,eligible", [
        (16, 64, False),   # the canonical broken cell — must be gated OUT
        (8, 64, False),
        (31, 64, False),   # not % 16 == 0
        (33, 64, False),
        (64, 16, False),   # C_out < 32 (was legacy-correct but no longer routed)
        (32, 32, True),    # boundary — eligible
        (48, 64, True),
        (64, 128, True),
    ])
    def test_mpp_gate_predicate(self, c_in, c_out, eligible):
        """fp16 + bf16: the MPP-eligibility predicate matches the empirical
        correctness boundary (C_in/C_out % 16 == 0 AND >= 32).  A revert of
        the fp16 gate would re-admit C_in=16 (silent corruption)."""
        for dtype in _DTYPES:
            x = mx.zeros((1, 4, 8, 8, c_in), dtype=dtype)
            w = mx.zeros((c_out, 3, 3, 3, c_in), dtype=dtype)
            assert ah._conv3d_mpp_eligible(x, w, self._PAD1) is eligible, (
                f"_conv3d_mpp_eligible c_in={c_in} c_out={c_out} {dtype} "
                f"expected {eligible}")

    def test_pointwise_gated_out(self):
        """1x1x1 pointwise shares the unmasked-K-tail matmul2d kernel, so
        the gate routes it to native regardless of channel count."""
        x = mx.zeros((1, 4, 8, 8, 64), dtype=mx.float16)
        w = mx.zeros((128, 1, 1, 1, 64), dtype=mx.float16)
        assert ah._conv3d_mpp_eligible(x, w, (0, 0, 0, 0, 0, 0)) is False


@_skipif_no_nax
class TestFixedKernelPathsDirect:
    """III-6: the matmul2d K-tail fix makes the raw NAX conv entry points
    correct at ALL C_in — verified DIRECTLY (bypassing the production hook
    gate), each against an INDEPENDENT fp32 reference (never another kernel
    path; see the anti-pattern lesson in audit-framing-inversions.md).

    These cover the C_in values that were broken before III-6 (C_in % 32 !=
    0): the C++ legacy im2col path, the C++ 1x1x1 pointwise path, and the
    Python _conv3d_nax_forward_python_legacy orchestrator.
    """

    # C_in values that route to the (previously-broken) non-MPP paths.
    _BROKEN_CIN = [8, 16, 17, 24, 31, 33, 40]
    _ALIGNED_CIN = [32, 64]  # K-aligned — were always correct, must stay so

    @pytest.mark.parametrize("c_in", _BROKEN_CIN + _ALIGNED_CIN)
    def test_cpp_legacy_im2col_3x3x3_matches_fp32(self, c_in):
        from mlx_mfa._ext import conv3d_nax_forward
        mx.random.seed(3)
        x = mx.random.normal((1, 4, 8, 8, c_in)).astype(mx.float16)
        w = mx.random.normal((64, 3, 3, 3, c_in)).astype(mx.float16)
        y = conv3d_nax_forward(x, w, stride=(1, 1, 1),
                               padding=(1, 1, 1, 1, 1, 1),
                               dilation=(1, 1, 1), chunk_M=0)
        yf = _ref_fp32(x, w)
        mx.eval(y, yf)
        assert not bool(mx.any(mx.isnan(y)).item()), f"NaN c_in={c_in}"
        assert _mae_rms(y, yf) < _MAE_RMS_BOUND, \
            f"C++ legacy im2col c_in={c_in}: MAE/RMS {_mae_rms(y, yf):.5f}"

    @pytest.mark.parametrize("c_in", [16, 31, 48, 64])
    def test_cpp_pointwise_1x1x1_matches_fp32(self, c_in):
        from mlx_mfa._ext import conv3d_nax_forward
        mx.random.seed(3)
        x = mx.random.normal((1, 4, 8, 8, c_in)).astype(mx.float16)
        w = mx.random.normal((128, 1, 1, 1, c_in)).astype(mx.float16)
        y = conv3d_nax_forward(x, w, stride=(1, 1, 1),
                               padding=(0, 0, 0, 0, 0, 0),
                               dilation=(1, 1, 1), chunk_M=0)
        nat = ah._ORIGINAL_CONV3D if ah._ORIGINAL_CONV3D is not None else mx.conv3d
        yf = nat(x.astype(mx.float32), w.astype(mx.float32),
                 stride=(1, 1, 1), padding=(0, 0, 0))
        mx.eval(y, yf)
        assert not bool(mx.any(mx.isnan(y)).item()), f"NaN c_in={c_in}"
        assert _mae_rms(y, yf) < _MAE_RMS_BOUND, \
            f"C++ pointwise c_in={c_in}: MAE/RMS {_mae_rms(y, yf):.5f}"

    @pytest.mark.parametrize("c_in", _BROKEN_CIN + _ALIGNED_CIN)
    def test_python_legacy_orchestrator_matches_fp32(self, c_in):
        from mlx_mfa.conv_nax import _conv3d_nax_forward_python_legacy
        mx.random.seed(3)
        x = (mx.random.normal((1, 4, 8, 8, c_in)) * 0.1).astype(mx.float16)
        w = (mx.random.normal((64, 3, 3, 3, c_in)) * 0.1).astype(mx.float16)
        y = _conv3d_nax_forward_python_legacy(x, w, stride=(1, 1, 1),
                                              padding=(1, 1, 1))
        yf = _ref_fp32(x, w)
        mx.eval(y, yf)
        assert not bool(mx.any(mx.isnan(y)).item()), f"NaN c_in={c_in}"
        assert _mae_rms(y, yf) < _MAE_RMS_BOUND, \
            f"Python legacy c_in={c_in}: MAE/RMS {_mae_rms(y, yf):.5f}"

    def test_matmul2d_source_refuses_unaligned_k(self):
        """Rule-8: the generator must refuse an unpadded K rather than emit
        a kernel that silently reads past the tensor extent."""
        from mlx_mfa.conv_nax import _matmul2d_source
        with pytest.raises(ValueError, match="not a multiple of K_TILE"):
            _matmul2d_source(32, 16, 64)  # K=16 unaligned
        # aligned K is fine
        assert "K_FULL = 32" in _matmul2d_source(32, 32, 64)
