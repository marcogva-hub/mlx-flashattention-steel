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

Fix: `_conv3d_mpp_eligible` now gates BOTH fp16 and bf16, so any shape
outside the MPP envelope falls back to the native op.
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
