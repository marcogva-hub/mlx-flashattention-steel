"""Phase III-4 D7 — block-mask expansion tiling lock (non-divisible N).

The bias-expansion helpers re-derived the tile as ceil(seq / n_tiles),
which RE-TILES the mask whenever seq is not a multiple of the kernel
tile (N=100, BQ=32: validator accepts a 4x4 mask = 32-token tiles, the
helpers expanded it as 25-token tiles) — every SDPA-bias-based sparse
path (M5 per-head fallback, sparse backward closures, no-ext fallback)
silently governed the wrong tokens (measured forward 0.67 / grads up
to 1.1 max-abs vs kernel semantics).

Fix: `_expansion_tile` — legal NAX bt (16/32/64) when seq divides
evenly to one, else the KERNEL tile when the mask matches the
kernel-validated geometry, else legacy ceil.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import mlx.core as mx

from mlx_mfa import flash_attention_sparse
from mlx_mfa.attention import _steel_block_config, _expansion_tile


class TestExpansionTileDerivation:
    def test_non_divisible_uses_kernel_tile(self):
        # N=100, 4 tiles, kernel 32: ceil(100/32)=4 matches -> 32 (not 25).
        assert _expansion_tile(100, 4, 32) == 32

    def test_nax_bt_exact_divide_wins(self):
        assert _expansion_tile(2048, 32, 32) == 64   # bt=64 NAX mask
        assert _expansion_tile(2048, 64, 32) == 32   # bt=32 (== kernel)

    def test_legacy_ceil_without_kernel_tile(self):
        assert _expansion_tile(100, 4, None) == 25


class TestNonDivisibleNSparseSemantics:
    @pytest.mark.parametrize("N", [100, 70])
    def test_forward_and_grads_match_kernel_tiling(self, N):
        B, H, D = 1, 2, 64
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK
        mx.random.seed(4)
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, N, D), dtype=mx.float16)
        v = mx.random.normal((B, H, N, D), dtype=mx.float16)
        mask = (mx.random.uniform(shape=(NQ, NK)) < 0.5) | mx.eye(
            NQ, NK, dtype=mx.bool_)
        mx.eval(q, k, v, mask)
        # token-level reference built with the KERNEL tiling
        mnp = np.asarray(mask)
        tok = np.zeros((N, N), dtype=bool)
        for i in range(N):
            tok[i, :] = mnp[i // BQ, np.arange(N) // BK]
        bias = mx.array(
            np.where(tok, 0.0, -np.inf).astype(np.float32)).astype(mx.float16)
        scale = 1.0 / math.sqrt(D)
        o = flash_attention_sparse(q, k, v, mask, scale=scale)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                                                   mask=bias)
        mx.eval(o, ref)
        assert float(mx.max(mx.abs(
            o.astype(mx.float32) - ref.astype(mx.float32))).item()) < 5e-3
        dO = mx.ones_like(q)
        _, g = mx.vjp(lambda a, b, c: flash_attention_sparse(
            a, b, c, mask, scale=scale), [q, k, v], [dO])
        _, gr = mx.vjp(lambda a, b, c: mx.fast.scaled_dot_product_attention(
            a, b, c, scale=scale, mask=bias), [q, k, v], [dO])
        mx.eval(*g, *gr)
        for name, x, y in zip(("dQ", "dK", "dV"), g, gr):
            err = float(mx.max(mx.abs(
                x.astype(mx.float32) - y.astype(mx.float32))).item())
            assert err < 5e-3, f"{name} max_abs={err:.4f} at N={N}"
