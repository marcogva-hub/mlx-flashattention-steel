"""Phase III-2 — TQ paged decode locks.

Two deliverables locked here:

1. The DECODE path: ``TurboQuantPagedInferenceContext.step`` with N_q=1
   routes to gather/dequant kernels + Apple SDPA (mlx_mfa/tq_decode.py)
   by default — §AA.5 inverted the TurboQuant P2-P4 fused-dequant
   premise on M5 (fused attend was 14x dense; this path is 13.8-22.1x
   faster attend-only, 6.0-14.4x full-step).  Opt-out:
   ``MFA_DISABLE_TQ_DECODE_SDPA=1``.

2. The fused-kernel bit-width FIX: the fused TQ kernel's K and V dequant
   emitted the 3-bit bit-planar extraction UNCONDITIONALLY — tq_bits=2/4
   read the pool with the wrong layout and were silently wrong since the
   kernel landed (0.147-0.150 max-abs at unit scale vs ground truth).
   Locked by arbitrating BOTH paths against the Python ground-truth
   dequant (unpack_indices -> centroids -> scales -> sdpa) per the II-6
   lesson: test against ground truth, not another internal path.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import mlx.core as mx

from mlx_mfa.inference import TurboQuantPagedInferenceContext
from mlx_mfa.turboquant import (
    apply_rotation, unpack_indices, unpack_3bit_optimal,
    dequantize_from_indices, _compute_packed_d,
)

Hq, Hkv, D, S0, BS = 8, 2, 128, 512, 64
SCALE = 1.0 / math.sqrt(D)


def _mkctx(bits, mag=1.0, **kw):
    ctx = TurboQuantPagedInferenceContext(
        num_blocks=S0 // BS + 8, block_size=BS, H_kv=Hkv, D=D,
        tq_bits=bits, **kw)
    mx.random.seed(3)
    k0 = (mx.random.normal((1, Hkv, S0, D)) * mag).astype(mx.float16)
    v0 = (mx.random.normal((1, Hkv, S0, D)) * mag).astype(mx.float16)
    q0 = (mx.random.normal((1, Hq, S0, D)) * mag).astype(mx.float16)
    mx.eval(k0, v0, q0)
    mx.eval(ctx.prefill(q0, k0, v0))
    q = (mx.random.normal((1, Hq, 1, D)) * mag).astype(mx.float16)
    mx.eval(q)
    return ctx, q


def _dequant_paged(pool, scales, tbl, bits, S):
    """Python dequant of a TQ-packed paged pool -> [1, Hkv, S, D] fp16."""
    p = pool[tbl]
    s = scales[tbl]
    nbk = p.shape[0]
    if bits == 3:
        # 3-bit uses the bit-planar layout (pack_3bit_optimal), NOT the
        # sequential layout unpack_indices assumes.
        pd = _compute_packed_d(D, 3)
        idx = unpack_3bit_optimal(p.reshape(nbk * BS * Hkv, pd), D)
    else:
        idx = unpack_indices(
            p.reshape(-1), nbk * BS * Hkv * D, bits).reshape(
                nbk * BS * Hkv, D)
    d = dequantize_from_indices(idx, bits)
    d = d * s.reshape(-1)[:, None]
    out = mx.transpose(d.reshape(nbk * BS, Hkv, D)[:S], (1, 0, 2))[None]
    return out.astype(mx.float16)


def _ground_truth(ctx, q_rot, bits):
    """Python dequant -> sdpa (independent of both kernel paths)."""
    table = ctx.get_block_table([0])
    S = ctx.seq_length(0)
    nb = (S + BS - 1) // BS
    tbl = table[0][:nb]
    K = _dequant_paged(ctx._k_pool, ctx._k_scales, tbl, bits, S)
    V = mx.transpose(
        ctx._v_pool_fp16[tbl].reshape(-1, Hkv, D)[:S], (1, 0, 2))[None]
    out = mx.fast.scaled_dot_product_attention(q_rot, K, V, scale=SCALE)
    mx.eval(out)
    return out


def _maxabs(a, b):
    return float(mx.max(mx.abs(
        a.astype(mx.float32) - b.astype(mx.float32))).item())


class TestDecodePathGroundTruth:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_new_path_matches_ground_truth(self, bits):
        ctx, q = _mkctx(bits)
        q_rot = apply_rotation(q.astype(mx.float32), "wht").astype(mx.float16)
        mx.eval(q_rot)
        gt = _ground_truth(ctx, q_rot, bits)
        from mlx_mfa.tq_decode import tq_decode_attend
        table = ctx.get_block_table([0])
        S = ctx.seq_length(0)
        nb = (S + BS - 1) // BS
        out = tq_decode_attend(
            q_rot, ctx._k_pool, ctx._v_pool_fp16, ctx._k_scales,
            ctx._k_centroids, table[0][:nb], S,
            scale=SCALE, block_size=BS, tq_bits=bits)
        mx.eval(out)
        assert _maxabs(out, gt) < 5e-3, f"bits={bits}"

    @pytest.mark.parametrize("bits", [2, 4])
    def test_fused_kernel_bitwidth_fix(self, bits):
        """III-2 regression lock: fused kernel at tq_bits=2/4 must match
        ground truth (was silently wrong — 3-bit-only unpack layout)."""
        ctx, q = _mkctx(bits)
        q_rot = apply_rotation(q.astype(mx.float32), "wht").astype(mx.float16)
        mx.eval(q_rot)
        gt = _ground_truth(ctx, q_rot, bits)
        from mlx_mfa.attention import flash_attention_paged_varlen_turboquant
        cu_q = mx.array([0, 1], dtype=mx.int32)
        table = ctx.get_block_table([0])
        lens = ctx.get_seq_lens([0])
        out = flash_attention_paged_varlen_turboquant(
            q_rot, ctx._k_pool, ctx._v_pool_fp16, table, lens, cu_q,
            ctx._k_centroids, ctx._k_scales, scale=SCALE, causal=True,
            block_size=BS, tq_bits=bits, tq_v_enabled=False,
            tq_wht_enabled=False)
        mx.eval(out)
        assert _maxabs(out, gt) < 5e-3, (
            f"fused kernel tq_bits={bits} deviates from ground truth — "
            f"the III-2 bit-width unpack fix regressed")

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_fused_v_tq_matches_ground_truth(self, bits):
        """III-4 F1: ground-truth lock for the fused V-TQ path
        (tq_v_enabled=True).

        With tq_v=True, V is WHT-rotated then TQ-packed at append time.
        The fused kernel computes P @ V_rot (rotated output space) and
        the wrapper de-rotates the result (WHT is self-inverse).  Ground
        truth must therefore be de-rotated too:
        apply_rotation(sdpa(q_rot, K_deq, V_rot_deq), "wht").
        Measured agreement ~2e-4; bar 5e-3.
        """
        ctx, q = _mkctx(bits, tq_v=True)
        q_rot = apply_rotation(q.astype(mx.float32), "wht").astype(mx.float16)
        mx.eval(q_rot)
        table = ctx.get_block_table([0])
        S = ctx.seq_length(0)
        nb = (S + BS - 1) // BS
        tbl = table[0][:nb]
        K = _dequant_paged(ctx._k_pool, ctx._k_scales, tbl, bits, S)
        V_rot = _dequant_paged(ctx._v_pool_tq, ctx._v_scales, tbl, bits, S)
        out_rot = mx.fast.scaled_dot_product_attention(
            q_rot, K, V_rot, scale=SCALE)
        gt = apply_rotation(
            out_rot.astype(mx.float32), "wht").astype(mx.float16)
        mx.eval(gt)
        from mlx_mfa.attention import flash_attention_paged_varlen_turboquant
        cu_q = mx.array([0, 1], dtype=mx.int32)
        lens = ctx.get_seq_lens([0])
        out = flash_attention_paged_varlen_turboquant(
            q_rot, ctx._k_pool, ctx._v_pool_fp16, table, lens, cu_q,
            ctx._k_centroids, ctx._k_scales, scale=SCALE, causal=True,
            block_size=BS, tq_bits=bits, tq_v_enabled=True,
            tq_wht_enabled=False, v_pool_tq=ctx._v_pool_tq,
            v_centroids=ctx._v_centroids, v_scales=ctx._v_scales)
        mx.eval(out)
        assert _maxabs(out, gt) < 5e-3, f"fused V-TQ bits={bits}"


class TestStepRouting:
    def test_default_routes_new_path_and_matches_optout(self, monkeypatch):
        """tq_v=False -> both paths read fp16 V -> near-exact agreement.
        (At the default tq_v=True the fused path reads QUANTIZED V while
        the new path reads the always-maintained fp16 V pool — the new
        path is deliberately MORE accurate there; see module docstring.)
        """
        monkeypatch.delenv("MFA_DISABLE_TQ_DECODE_SDPA", raising=False)
        ctx, q = _mkctx(3, tq_v=False)
        kn = mx.zeros((1, Hkv, 1, D), dtype=mx.float16)
        vn = mx.zeros((1, Hkv, 1, D), dtype=mx.float16)
        out_new = ctx.step(q, kn, vn)
        mx.eval(out_new)
        ctx2, q2 = _mkctx(3, tq_v=False)
        monkeypatch.setenv("MFA_DISABLE_TQ_DECODE_SDPA", "1")
        out_fused = ctx2.step(q2, kn, vn)
        mx.eval(out_fused)
        assert bool(mx.all(q == q2).item())  # same fixture
        assert _maxabs(out_new, out_fused) < 5e-3

    def test_default_tq_v_true_within_vquant_noise(self, monkeypatch):
        """At the context default (tq_v=True) the two paths differ only
        by V-quantization noise."""
        monkeypatch.delenv("MFA_DISABLE_TQ_DECODE_SDPA", raising=False)
        ctx, q = _mkctx(3)
        kn = mx.zeros((1, Hkv, 1, D), dtype=mx.float16)
        vn = mx.zeros((1, Hkv, 1, D), dtype=mx.float16)
        out_new = ctx.step(q, kn, vn)
        mx.eval(out_new)
        ctx2, q2 = _mkctx(3)
        monkeypatch.setenv("MFA_DISABLE_TQ_DECODE_SDPA", "1")
        out_fused = ctx2.step(q2, kn, vn)
        mx.eval(out_fused)
        assert _maxabs(out_new, out_fused) < 0.15

    def test_multitoken_step_keeps_fused(self, monkeypatch):
        """N_q > 1 must stay on the fused kernel (causal offsets)."""
        monkeypatch.delenv("MFA_DISABLE_TQ_DECODE_SDPA", raising=False)
        ctx, _ = _mkctx(3)
        q = mx.random.normal((1, Hq, 2, D), dtype=mx.float16)
        kn = mx.random.normal((1, Hkv, 2, D), dtype=mx.float16)
        vn = mx.random.normal((1, Hkv, 2, D), dtype=mx.float16)
        mx.eval(q, kn, vn)
        out = ctx.step(q, kn, vn)
        mx.eval(out)
        assert out.shape == (1, Hq, 2, D)
        assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item())

    def test_wht_in_kernel_routing(self, monkeypatch):
        monkeypatch.delenv("MFA_DISABLE_TQ_DECODE_SDPA", raising=False)
        ctx, q = _mkctx(3, tq_v=False, wht_in_kernel=True)
        kn = mx.zeros((1, Hkv, 1, D), dtype=mx.float16)
        vn = mx.zeros((1, Hkv, 1, D), dtype=mx.float16)
        out = ctx.step(q, kn, vn)
        mx.eval(out)
        ctx2, q2 = _mkctx(3, tq_v=False, wht_in_kernel=True)
        monkeypatch.setenv("MFA_DISABLE_TQ_DECODE_SDPA", "1")
        out_f = ctx2.step(q2, kn, vn)
        mx.eval(out_f)
        # In-kernel WHT (fp16) vs Python WHT (fp32) rounding tolerance.
        assert _maxabs(out, out_f) < 0.02

    def test_determinism(self):
        ctx, q = _mkctx(3)
        q_rot = apply_rotation(q.astype(mx.float32), "wht").astype(mx.float16)
        mx.eval(q_rot)
        from mlx_mfa.tq_decode import tq_decode_attend
        table = ctx.get_block_table([0])
        S = ctx.seq_length(0)
        nb = (S + BS - 1) // BS
        runs = []
        for _ in range(5):
            o = tq_decode_attend(
                q_rot, ctx._k_pool, ctx._v_pool_fp16, ctx._k_scales,
                ctx._k_centroids, table[0][:nb], S,
                scale=SCALE, block_size=BS, tq_bits=3)
            mx.eval(o)
            runs.append(o)
        for r in runs[1:]:
            assert bool(mx.all(runs[0] == r).item())

    def test_adversarial_magnitude_finite(self):
        ctx, q = _mkctx(3, mag=8.0)
        kn = mx.zeros((1, Hkv, 1, D), dtype=mx.float16)
        vn = mx.zeros((1, Hkv, 1, D), dtype=mx.float16)
        out = ctx.step(q, kn, vn)
        mx.eval(out)
        assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item())
