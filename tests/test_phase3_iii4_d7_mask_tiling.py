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


# ── R4 / DOC-05 / NEPF-03 (code review 2026-09, P0 published) ────────────────────
# Phase F made the D=128 mask MAKERS emit 32x32 tiles (_bq_bk) while the STEEL kernel
# geometry is 32x16.  The normalizer split the mask by the COUNT ratio
# ceil(N/16) // ceil(N/32), which floors to 1 for non-32-aligned N (e.g. 7//4) -> the
# mask stayed 4x4 and _expansion_tile re-tiled it to 25/30/31-token key tiles (the D7
# class, reintroduced for D=128).  Fix: split by the TILE ratio (32/16 = 2) then
# truncate to the kernel count; _expansion_tile now refuses the silent re-tile.
from mlx_mfa import make_diagonal_mask, make_sliding_window_mask  # noqa: E402
from mlx_mfa.masks import _bq_bk  # noqa: E402


class TestMakerMasksNonAlignedLargeD:
    @staticmethod
    def _maker_token_bias(mask, N, D, dtype):
        """Token-level bias with the MAKER geometry (the mask's own semantics)."""
        BQm, BKm = _bq_bk(D)
        mnp = np.asarray(mask)
        tok = mnp[np.arange(N)[:, None] // BQm, np.arange(N)[None, :] // BKm]
        return mx.array(np.where(tok, 0.0, -np.inf).astype(np.float32)).astype(dtype)

    @pytest.mark.parametrize("maker", ["diagonal", "sliding"])
    @pytest.mark.parametrize("N", [100, 200, 300, 612])
    @pytest.mark.parametrize("D", [128])   # D=256/512: maker == kernel tile; 96 unsupported
    def test_forward_and_grads_follow_maker_tiles(self, D, N, maker):
        mask = (make_diagonal_mask(N, head_dim=D) if maker == "diagonal"
                else make_sliding_window_mask(N, window_size=32, head_dim=D))
        mx.random.seed(N + D)
        q, k, v = (mx.random.normal((1, 2, N, D)).astype(mx.float16) for _ in range(3))
        scale = 1.0 / math.sqrt(D)
        bias = self._maker_token_bias(mask, N, D, mx.float16)
        o = flash_attention_sparse(q, k, v, mask, scale=scale)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=bias)
        mx.eval(o, ref)
        err = float(mx.max(mx.abs(o.astype(mx.float32) - ref.astype(mx.float32))).item())
        assert err < 5e-3, f"forward max_abs={err:.4f} (D={D}, N={N}, {maker})"
        dO = mx.ones_like(q)
        _, g = mx.vjp(lambda a, b, c: flash_attention_sparse(a, b, c, mask, scale=scale),
                      [q, k, v], [dO])
        _, gr = mx.vjp(lambda a, b, c: mx.fast.scaled_dot_product_attention(
            a, b, c, scale=scale, mask=bias), [q, k, v], [dO])
        mx.eval(*g, *gr)
        for name, x, y in zip(("dQ", "dK", "dV"), g, gr):
            e = float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))).item())
            assert e < 5e-3, f"{name} max_abs={e:.4f} (D={D}, N={N}, {maker})"

    @pytest.mark.parametrize("N", [100, 300])
    def test_per_head_fallback_is_bias_sdpa_byte_identical(self, N):
        """Which-binary: at these (small-mask) shapes the M5 route is the per-head SDPA
        fallback; with the right expansion it IS SDPA+maker-bias, bit for bit."""
        D = 128
        mask = make_diagonal_mask(N, head_dim=D)
        mx.random.seed(7)
        q, k, v = (mx.random.normal((1, 2, N, D)).astype(mx.float16) for _ in range(3))
        scale = 1.0 / math.sqrt(D)
        o = flash_attention_sparse(q, k, v, mask, scale=scale)
        ref = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=self._maker_token_bias(mask, N, D, mx.float16))
        mx.eval(o, ref)
        assert float(mx.max(mx.abs(o.astype(mx.float32) - ref.astype(mx.float32))).item()) == 0.0


class TestExpansionTileNoSilentRetile:
    def test_maker_geometry_resolves_to_legal_tile_not_retile(self):
        # 4 tiles over N=100, kernel tile 16 (ceil(100/16)=7): the mask is the 32x32
        # maker geometry -> 32 (legal), never the derived 25 (the R4/D7 bug).
        assert _expansion_tile(100, 4, 16) == 32
        assert _expansion_tile(4100, 129, 16) == 32   # large non-aligned bt-32 mask

    def test_count_matching_no_legal_tile_raises(self):
        # 5 tiles over N=100: 16->7, 32->4, 64->2 — only a derived 20 would "fit".
        with pytest.raises(ValueError, match="re-tile"):
            _expansion_tile(100, 5, 16)

    def test_kernel_geometry_still_accepted(self):
        assert _expansion_tile(100, 7, 16) == 16
        assert _expansion_tile(612, 20, 32) == 32
