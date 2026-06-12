"""Phase II-6 — V34 backward paired-MMA regression locks.

Root cause locked here: every V34 backward generator emits the
S-recompute as a PAIRED 16x32x16 MMA (`for ik += 2` over TK writing
frag_at(iq, ik) and frag_at(iq, ik+1)).  MPP cooperative matmul2d has
no 16x16x16 form, so TK = BK/16 must be even.  The v2.39.1 fused-dKdV
default of BK=16 (TK=1) read 16 K-rows past the tile and wrote one
fragment out of bounds — silent dK/dV corruption that scales
exponentially with score magnitude:

  - input std 0.1 (the old fixtures): error within rmse gates → missed
  - input std 1.0: dV errors up to ~4x the gradient magnitude
  - input std >= 2: inf in dK/dV

Locks:
  1. Default-on D=64-causal backward matches SDPA-vjp at UNIT-scale
     inputs with a PER-ELEMENT max-err bound (rmse alone diluted the
     localized corruption below the old 5e-3 gate).
  2. Gradients stay finite under adversarial-magnitude inputs.
  3. BK % 32 != 0 on any V34 backward Primitive raises loudly
     (guards the MFA_V34BWD*_BK env footgun for all kernels).
"""
from __future__ import annotations

import math
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import flash_attention, get_device_info

_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))
_skipif_no_nax = pytest.mark.skipif(not _HAS_NAX, reason="V34 requires M5+ NAX")


def _grads(fn, q, k, v):
    g = mx.grad(lambda a, b, c: fn(a, b, c).sum(), argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g)
    return g


def _mk(mag, seed, N=4096, Hq=8, Hkv=8, D=64):
    mx.random.seed(seed)
    q = (mx.random.normal((1, Hq, N, D)) * mag).astype(mx.float16)
    k = (mx.random.normal((1, Hkv, N, D)) * mag).astype(mx.float16)
    v = (mx.random.normal((1, Hkv, N, D)) * mag).astype(mx.float16)
    mx.eval(q, k, v)
    return q, k, v


@_skipif_no_nax
class TestUnitScaleCorrectness:
    """Per-element max-err vs SDPA-vjp at input std 1.0 (NOT 0.1)."""

    @pytest.mark.parametrize("seed", [7, 42])
    def test_default_on_matches_sdpa_vjp_elementwise(self, seed):
        q, k, v = _mk(1.0, seed)
        s = 1.0 / math.sqrt(64)
        g = _grads(lambda a, b, c: flash_attention(a, b, c, causal=True), q, k, v)
        gr = _grads(lambda a, b, c: mx.fast.scaled_dot_product_attention(
            a, b, c, scale=s, mask="causal"), q, k, v)
        # fp16 noise floor for these shapes is ~0.008 (measured: split
        # and legacy_fused both land at 0.004-0.008).  The paired-MMA
        # corruption produced 22-130.  Bound set 10x above noise.
        for name, x, y in zip(("dQ", "dK", "dV"), g, gr):
            maxerr = float(mx.max(mx.abs(
                x.astype(mx.float32) - y.astype(mx.float32))).item())
            assert maxerr < 0.1, (
                f"{name} per-element max err {maxerr:.4f} vs SDPA-vjp at "
                f"unit-scale inputs (seed={seed}) — paired-MMA class "
                f"corruption regressed?")


@_skipif_no_nax
class TestAdversarialMagnitudeFinite:
    """Gradients must stay finite at large input magnitudes (fp16 tails)."""

    @pytest.mark.parametrize("mag", [2.0, 12.0])
    def test_grads_finite(self, mag):
        q, k, v = _mk(mag, seed=7)
        g = _grads(lambda a, b, c: flash_attention(a, b, c, causal=True), q, k, v)
        for name, x in zip(("dQ", "dK", "dV"), g):
            n_bad = int(mx.sum(~mx.isfinite(x.astype(mx.float32))).item())
            assert n_bad == 0, (
                f"{name} has {n_bad} non-finite elements at input "
                f"std={mag} — BK=16-class overflow regressed?")


@_skipif_no_nax
class TestBKGuard:
    """BK % 32 != 0 must raise loudly on every V34 backward Primitive."""

    def test_split_dv_bk16_raises(self, monkeypatch):
        monkeypatch.setenv("MFA_V34BWDV_BK", "16")
        from mlx_mfa import _ext
        q, k, v = _mk(1.0, seed=3, N=256)
        L = mx.zeros((1, 8, 256))
        dO = mx.ones_like(q)
        mx.eval(L, dO)
        with pytest.raises(Exception, match="multiple of 32"):
            out = _ext.v6_nax_backward_dv_raw(
                q, k, v, L, dO, 1.0 / 8.0, 4, True)
            mx.eval(out)

    def test_fused_bk16_raises(self, monkeypatch):
        monkeypatch.setenv("MFA_V34BWDF_BK", "16")
        from mlx_mfa import _ext
        q, k, v = _mk(1.0, seed=3, N=256)
        L = mx.zeros((1, 8, 256))
        dO = mx.ones_like(q)
        D0 = mx.zeros((1, 8, 256))
        mx.eval(L, dO, D0)
        with pytest.raises(Exception, match="multiple of 32"):
            out = _ext.v6_nax_backward_fused_dkdv_raw(
                q, k, v, L, dO, D0, 1.0 / 8.0, 4, True)
            mx.eval(*out)


@_skipif_no_nax
class TestAutoRoutesSplit:
    """auto must resolve to the split kernels (II-6 demotion)."""

    def test_auto_matches_forced_split_bitwise(self):
        q, k, v = _mk(1.0, seed=9, N=2048)
        g_auto = _grads(lambda a, b, c: flash_attention(a, b, c, causal=True),
                        q, k, v)
        os.environ["MFA_V34_BWD_KERNEL"] = "split"
        try:
            g_split = _grads(lambda a, b, c: flash_attention(a, b, c, causal=True),
                             q, k, v)
        finally:
            del os.environ["MFA_V34_BWD_KERNEL"]
        for name, x, y in zip(("dQ", "dK", "dV"), g_auto, g_split):
            same = bool(mx.all(x == y).item())
            assert same, f"{name}: auto != forced-split (routing regressed)"
