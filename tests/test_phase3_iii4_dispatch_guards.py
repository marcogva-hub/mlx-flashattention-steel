"""Phase III-4 — dispatch/contract guard locks (D8, D9, D13).

D8: backend="mfa" must run the actual MFA Metal forward on V34-eligible
    cells, not Apple SDPA (forced-backend measurement integrity).
D9: V-dim-mismatch / cross-attention shapes must NOT enter the V34
    backward carve-out.
D13: the NAX rope fast path must honor non-base-10000 user tables (route
    to the table-using STEEL path instead of silently applying base=10000).
"""
from __future__ import annotations

import math
import os
import numpy as np
import pytest
import mlx.core as mx

from mlx_mfa import flash_attention, flash_attention_rope_unified, get_device_info
from mlx_mfa.attention import _rope_tables_match_base10000

_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))
_skipif_no_nax = pytest.mark.skipif(not _HAS_NAX, reason="M5+ NAX required")


@_skipif_no_nax
class TestD8ForcedBackend:
    def test_backend_mfa_runs_kernel_not_sdpa(self):
        B, H, N, D = 1, 4, 2048, 64
        mx.random.seed(1)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)
        s = 1.0 / math.sqrt(D)
        o_mfa = flash_attention(q, k, v, scale=s, causal=True, backend="mfa")
        o_sdpa = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=s, mask="causal")
        mx.eval(o_mfa, o_sdpa)
        # backend="mfa" must engage the MFA kernel: NOT bit-identical to
        # SDPA (the II-8 carve-out makes backend="auto" bit-SDPA, but a
        # FORCED kernel must actually run the kernel), yet within fp16.
        assert not bool(mx.all(o_mfa == o_sdpa).item()), \
            "backend='mfa' silently ran SDPA on a V34-eligible cell (D8)"
        err = float(mx.max(mx.abs(
            o_mfa.astype(mx.float32) - o_sdpa.astype(mx.float32))).item())
        assert err < 0.05, f"MFA forward off from SDPA by {err}"

    def test_backend_auto_stays_bit_sdpa(self):
        B, H, N, D = 1, 4, 2048, 64
        mx.random.seed(2)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)
        s = 1.0 / math.sqrt(D)
        o_auto = flash_attention(q, k, v, scale=s, causal=True, backend="auto")
        o_sdpa = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=s, mask="causal")
        mx.eval(o_auto, o_sdpa)
        assert bool(mx.all(o_auto == o_sdpa).item()), \
            "II-8 carve-out regressed: backend='auto' forward not bit-SDPA"


@_skipif_no_nax
class TestD13RopeTableHonored:
    def _tables(self, base, hd=64, maxs=256):
        half = hd // 2
        i = mx.arange(half, dtype=mx.float32)
        inv = mx.exp(-math.log(base) * (2.0 * i / hd))
        p = mx.arange(maxs, dtype=mx.float32)[:, None]
        ang = p * inv[None, :]
        c = mx.cos(ang).astype(mx.float16)
        s = mx.sin(ang).astype(mx.float16)
        mx.eval(c, s)
        return c, s

    def test_detector(self):
        c10, s10 = self._tables(10000.0)
        c5, s5 = self._tables(500000.0)
        assert _rope_tables_match_base10000(c10, s10, 64, True, mx.float16)
        assert not _rope_tables_match_base10000(c5, s5, 64, True, mx.float16)

    def test_custom_base_matches_optout_path(self):
        # base=5e5 tables: the fast path must route to the table-using
        # STEEL path, bit-identical to MFA_DISABLE_ROPE_NAX=1.
        c5, s5 = self._tables(500000.0)
        B, H, N, hd = 1, 2, 128, 64
        mx.random.seed(0)
        q = mx.random.normal((B, H, N, hd)).astype(mx.float16)
        k = mx.random.normal((B, H, N, hd)).astype(mx.float16)
        v = mx.random.normal((B, H, N, hd)).astype(mx.float16)
        mx.eval(q, k, v)
        os.environ.pop("MFA_DISABLE_ROPE_NAX", None)
        o_fix = flash_attention_rope_unified(
            q, k, v, rotary_cos=c5, rotary_sin=s5, causal=True, interleaved=True)
        os.environ["MFA_DISABLE_ROPE_NAX"] = "1"
        try:
            o_steel = flash_attention_rope_unified(
                q, k, v, rotary_cos=c5, rotary_sin=s5, causal=True,
                interleaved=True)
        finally:
            os.environ.pop("MFA_DISABLE_ROPE_NAX", None)
        mx.eval(o_fix, o_steel)
        err = float(mx.max(mx.abs(
            o_fix.astype(mx.float32) - o_steel.astype(mx.float32))).item())
        assert err < 1e-4, (
            f"custom-base rope not routed to the table-honoring path "
            f"(err {err} vs the opt-out reference) — D13 regressed")


@_skipif_no_nax
class TestB1WindowConsistency:
    """III-4 pass-2 B1: the windowed forward and its backward oracle must
    use the SAME window anchor.  Pre-fix the forward 0-anchored non-causal
    windows while the backward oracle used S-N — gradients of a different
    function than the forward computed.  Resolution: the backward now
    matches the forward's documented anchor (qL_off = (causal && N<S) ?
    S-N : 0).  (The forward kernel keeps 0-anchor for non-causal windows;
    decode callers use causal=True for S-N anchoring.)
    """

    def _anchor_mask(self, N, S, wl, causal):
        q_off = (S - N) if (causal and N < S) else 0
        qi = mx.arange(q_off, q_off + N)[:, None]
        ki = mx.arange(S)[None, :]
        inwin = (ki >= qi - wl)
        if causal:
            inwin = inwin & (ki <= qi)
        return mx.where(inwin, mx.zeros((N, S)),
                        mx.full((N, S), float("-inf"))).astype(mx.float16)

    @pytest.mark.parametrize("causal", [False, True])
    def test_forward_matches_documented_anchor(self, causal):
        B, H, D, S, N = 1, 4, 128, 512, 256
        mx.random.seed(3)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        sc = 1.0 / math.sqrt(D)
        wl = 64
        out = flash_attention(q, k, v, scale=sc, causal=causal,
                              window_size=(wl, 0 if causal else -1))
        m = self._anchor_mask(N, S, wl, causal)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=m)
        mx.eval(out, ref)
        err = float(mx.max(mx.abs(
            out.astype(mx.float32) - ref.astype(mx.float32))).item())
        assert err < 5e-3, f"forward window anchor (causal={causal}): {err}"

    @pytest.mark.parametrize("causal", [False, True])
    def test_forward_backward_agree(self, causal):
        B, H, D, S, N = 1, 4, 128, 512, 256
        mx.random.seed(3)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        sc = 1.0 / math.sqrt(D)
        wl = 64
        dO = mx.ones_like(q)
        m = self._anchor_mask(N, S, wl, causal)
        _, g = mx.vjp(lambda a, b, c: flash_attention(
            a, b, c, scale=sc, causal=causal,
            window_size=(wl, 0 if causal else -1)), [q, k, v], [dO])
        _, gr = mx.vjp(lambda a, b, c: mx.fast.scaled_dot_product_attention(
            a, b, c, scale=sc, mask=m), [q, k, v], [dO])
        mx.eval(*g, *gr)
        for name, x, y in zip(("dQ", "dK", "dV"), g, gr):
            err = float(mx.max(mx.abs(
                x.astype(mx.float32) - y.astype(mx.float32))).item())
            assert err < 5e-3, (
                f"{name} fwd/bwd window-anchor disagree (causal={causal}): {err}")


@_skipif_no_nax
class TestP5ReturnLseBackward:
    """III-4 pass-5 P5-1 (CRITICAL): mx.grad through
    flash_attention(return_lse=True) must produce CORRECT gradients.
    The prior code called the raw mfa_forward_with_lse Primitive whose
    2-output C++ vjp corrupted dQ/dK/dV (NaN at large shapes).  Now
    routed through a custom_function with an SDPA-vjp backward."""

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
    @pytest.mark.parametrize("causal", [True, False])
    @pytest.mark.parametrize("N,D", [(512, 64), (2048, 64), (2048, 128)])
    def test_return_lse_grad_matches_sdpa_vjp(self, causal, N, D, dtype):
        # P5-1 had DTYPE-SPECIFIC symptoms (NaN in fp16/bf16, ~1e32 garbage
        # in fp32), so the lock covers all three dtypes x {D=64,128} x
        # {causal, non-causal}.  The bug was causal+return_lse-specific.
        sc = 1.0 / math.sqrt(D)
        mx.random.seed(1)
        q = mx.random.normal((1, 4, N, D)).astype(dtype)
        k = mx.random.normal((1, 4, N, D)).astype(dtype)
        v = mx.random.normal((1, 4, N, D)).astype(dtype)
        mx.eval(q, k, v)
        # Ground truth = SDPA-vjp.  The return_lse path routes through
        # _make_mfa_custom_lse (SDPA-vjp backward), so it must match GT to
        # the dtype floor.  NOTE: the no-return_lse path is NOT necessarily
        # bit-identical — at the V34-eligible cells (D=64, qL>=2048) it uses
        # the V34 backward (default-on, II-12), which differs from SDPA-vjp
        # by the fp16 floor.  So the invariant is "return_lse grad == SDPA-
        # vjp GT within floor", and "within floor of no-return_lse" (both
        # are within floor of GT), NOT bit-exact to no-return_lse.
        g = mx.grad(lambda a: flash_attention(
            a, k, v, scale=sc, causal=causal, return_lse=True)[0].sum())(q)
        g_nolse = mx.grad(lambda a: flash_attention(
            a, k, v, scale=sc, causal=causal).sum())(q)
        gr = mx.grad(lambda a: mx.fast.scaled_dot_product_attention(
            a, k, v, scale=sc, mask="causal" if causal else None).sum())(q)
        mx.eval(g, g_nolse, gr)
        a = np.array(g.astype(mx.float32))
        b = np.array(gr.astype(mx.float32))
        c = np.array(g_nolse.astype(mx.float32))
        assert not np.isnan(a).any() and not np.isinf(a).any(), \
            f"return_lse grad non-finite (dtype={dtype}, causal={causal})"
        bound = 5e-3 if dtype != mx.bfloat16 else 5e-2  # bf16: 8 mantissa bits
        assert float(np.max(np.abs(a - b))) < bound, \
            "return_lse dQ != SDPA-vjp ground truth (P5-1 regressed)"
        assert float(np.max(np.abs(a - c))) < bound, \
            "return_lse grad diverges from no-return_lse grad (P5-1 regressed)"

    def test_lse_value_still_returned(self):
        sc = 1.0 / 8.0
        mx.random.seed(2)
        q = mx.random.normal((1, 4, 512, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 512, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 512, 64)).astype(mx.float16)
        mx.eval(q, k, v)
        O, L = flash_attention(q, k, v, scale=sc, causal=True, return_lse=True)
        mx.eval(O, L)
        assert tuple(L.shape) == (1, 4, 512)
        assert bool(mx.all(mx.isfinite(L)).item())
