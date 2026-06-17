"""Phase II-6 — V6NAX backward paired-MMA regression locks.

Root cause locked here: every V6NAX backward generator emits the
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
  3. BK % 32 != 0 on any V6NAX backward Primitive raises loudly
     (guards the MFA_V6BWD*_BK env footgun for all kernels).
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
_skipif_no_nax = pytest.mark.skipif(not _HAS_NAX, reason="V6NAX requires M5+ NAX")


@pytest.fixture(autouse=True)
def _clear_metal_cache():
    """These tests deliberately fill buffers with inf/NaN (adversarial
    magnitudes, guard exceptions).  Clear the Metal buffer pool after
    each test so recycled buffers can't contaminate later lazy-zeros
    tests (known stale-buffer pattern, see MEMORY.md v1.3.0 notes)."""
    yield
    mx.clear_cache()


def _grads(fn, q, k, v):
    g = mx.grad(lambda a, b, c: fn(a, b, c).sum(), argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g)
    return g


def _mk(mag, seed, N=4096, Hq=8, Hkv=8, D=64, dtype=mx.float16):
    mx.random.seed(seed)
    q = (mx.random.normal((1, Hq, N, D)) * mag).astype(dtype)
    k = (mx.random.normal((1, Hkv, N, D)) * mag).astype(dtype)
    v = (mx.random.normal((1, Hkv, N, D)) * mag).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


# III-4 F6: per-element max-err bounds vs SDPA-vjp inside the promoted
# V6NAX envelope.  Measured floors (M5 Max, N=4096 D=64, seeds 7/42):
# fp16 0.004-0.008, bf16 0.016-0.0625 (8 mantissa bits).  Bounds set
# ~10x (fp16) / 4x (bf16) above the measured floor.
_BWD_MAXERR = {mx.float16: 0.1, mx.bfloat16: 0.25}


@_skipif_no_nax
class TestUnitScaleCorrectness:
    """Per-element max-err vs SDPA-vjp at input std 1.0 (NOT 0.1)."""

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    @pytest.mark.parametrize("seed", [7, 42])
    def test_default_on_matches_sdpa_vjp_elementwise(self, seed, dtype):
        q, k, v = _mk(1.0, seed, dtype=dtype)
        s = 1.0 / math.sqrt(64)
        g = _grads(lambda a, b, c: flash_attention(a, b, c, causal=True), q, k, v)
        gr = _grads(lambda a, b, c: mx.fast.scaled_dot_product_attention(
            a, b, c, scale=s, mask="causal"), q, k, v)
        # fp16 noise floor for these shapes is ~0.008 (measured: split
        # and legacy_fused both land at 0.004-0.008).  The paired-MMA
        # corruption produced 22-130.  Bound set 10x above noise.
        # III-4 F6: bf16 floor measured at 0.016-0.0625 -> bound 0.25.
        bound = _BWD_MAXERR[dtype]
        for name, x, y in zip(("dQ", "dK", "dV"), g, gr):
            maxerr = float(mx.max(mx.abs(
                x.astype(mx.float32) - y.astype(mx.float32))).item())
            assert maxerr < bound, (
                f"{name} per-element max err {maxerr:.4f} vs SDPA-vjp at "
                f"unit-scale inputs (seed={seed}, dtype={dtype}) — "
                f"paired-MMA class corruption regressed?")


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
    """BK % 32 != 0 must raise loudly on every V6NAX backward Primitive."""

    def test_split_dv_bk16_raises(self, monkeypatch):
        monkeypatch.setenv("MFA_V6BWDV_BK", "16")
        from mlx_mfa import _ext
        q, k, v = _mk(1.0, seed=3, N=256)
        L = mx.zeros((1, 8, 256))
        dO = mx.ones_like(q)
        mx.eval(L, dO)
        with pytest.raises(Exception, match="multiple of 32"):
            out = _ext.v6_nax_backward_dv_raw(
                q, k, v, L, dO, 1.0 / 8.0, 4, True)
            mx.eval(out)

    def test_fused_bk16_now_legal_and_bk_nonmult16_raises(self, monkeypatch):
        """II-8 addendum item 3 UPDATE: the fused generator gained the
        odd-TK tail (zeroed second K fragment + scratch dest), so BK=16
        is now LEGAL on the fused kernel (validated at unit +
        adversarial scale; measured exactly at parity with split).
        Non-multiples of 16 still raise; split kernels (no tail) still
        reject BK=16 — covered by test_split_dv_bk16_raises above."""
        from mlx_mfa import _ext
        q, k, v = _mk(1.0, seed=3, N=256)
        L = mx.zeros((1, 8, 256))
        dO = mx.ones_like(q)
        D0 = mx.zeros((1, 8, 256))
        mx.eval(L, dO, D0)
        # BK=16: legal post-II-8 (odd-TK tail)
        monkeypatch.setenv("MFA_V6BWDF_BK", "16")
        out = _ext.v6_nax_backward_fused_dkdv_raw(
            q, k, v, L, dO, D0, 1.0 / 8.0, 4, True)
        mx.eval(*out)
        # BK=24 (not a multiple of 16): still loud
        monkeypatch.setenv("MFA_V6BWDF_BK", "24")
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
        os.environ["MFA_V6_BWD_KERNEL"] = "split"
        try:
            g_split = _grads(lambda a, b, c: flash_attention(a, b, c, causal=True),
                             q, k, v)
        finally:
            del os.environ["MFA_V6_BWD_KERNEL"]
        for name, x, y in zip(("dQ", "dK", "dV"), g_auto, g_split):
            same = bool(mx.all(x == y).item())
            assert same, f"{name}: auto != forced-split (routing regressed)"


@_skipif_no_nax
class TestNonCausalPromotionII12:
    """Phase II-12 locks: non-causal D=64 backward default-on via the
    clean split kernel; forward stays bit-SDPA; GQA at H_kv."""

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    def test_unit_scale_elementwise(self, dtype):
        # III-4 F6: bf16 added; measured floors fp16 0.001, bf16 0.0156.
        q, k, v = _mk(1.0, seed=7, dtype=dtype)
        s = 1.0 / math.sqrt(64)
        g = _grads(lambda a, b, c: flash_attention(a, b, c, causal=False), q, k, v)
        gr = _grads(lambda a, b, c: mx.fast.scaled_dot_product_attention(
            a, b, c, scale=s), q, k, v)
        bound = _BWD_MAXERR[dtype]
        for name, x, y in zip(("dQ", "dK", "dV"), g, gr):
            maxerr = float(mx.max(mx.abs(
                x.astype(mx.float32) - y.astype(mx.float32))).item())
            assert maxerr < bound, f"{name} non-causal {dtype} err {maxerr}"

    def test_adversarial_finite(self):
        q, k, v = _mk(12.0, seed=7)
        g = _grads(lambda a, b, c: flash_attention(a, b, c, causal=False), q, k, v)
        for name, x in zip(("dQ", "dK", "dV"), g):
            assert int(mx.sum(~mx.isfinite(x.astype(mx.float32))).item()) == 0, name

    def test_forward_stays_sdpa_bit_identical(self):
        q, k, v = _mk(1.0, seed=1, N=4096)
        o1 = flash_attention(q, k, v, causal=False)
        o2 = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / 8.0)
        mx.eval(o1, o2)
        assert bool(mx.all(o1 == o2).item()), \
            "non-causal forward not bit-SDPA (II-8 carve-out lesson regressed)"

    def test_gqa_hkv_shapes(self):
        q, k, v = _mk(1.0, seed=9, N=2048, Hkv=2)
        g = _grads(lambda a, b, c: flash_attention(a, b, c, causal=False), q, k, v)
        assert tuple(g[1].shape) == (1, 2, 2048, 64), "dK not at H_kv"
        assert tuple(g[2].shape) == (1, 2, 2048, 64), "dV not at H_kv"
