"""Phase III-4 pass-3 F1/F2 — empty/fully-masked query-row contract.

A fully-masked query row (all keys masked out) must follow the II-6
empty-row -> ZEROS contract (the dedicated sparse Metal kernels emit
zeros), NOT propagate NaN.  Two bias-expansion paths fed an all -inf
row into mx.fast.scaled_dot_product_attention (which NaNs there):

  F1: flash_attention_topk(mask=...) reference path.
  F2: lcsa_nax.sparse_attention_dispatch SDPA+bias branch (the NAX
      branch already emitted zeros — the two branches disagreed).
"""
from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx

from mlx_mfa import flash_attention_topk
from mlx_mfa.lcsa_nax import sparse_attention_dispatch


def _masked_block_mask(nq, nk, dead_q_tile):
    bm = np.ones((nq, nk), dtype=bool)
    bm[dead_q_tile, :] = False
    return mx.array(bm)


class TestF1TopkEmptyRow:
    def test_fully_masked_tile_gives_zeros_not_nan(self):
        mx.random.seed(1)
        q = mx.random.normal((1, 1, 64, 64)).astype(mx.float16)
        k = mx.random.normal((1, 1, 64, 64)).astype(mx.float16)
        v = mx.random.normal((1, 1, 64, 64)).astype(mx.float16)
        # tile 0 (rows 0..31) fully masked at BQ=32
        mask = mx.array([[False, False], [True, True]])
        mx.eval(q, k, v, mask)
        out = flash_attention_topk(q, k, v, topk_ratio=0.5, mask=mask)
        mx.eval(out)
        o = np.array(out.astype(mx.float32))
        assert not np.isnan(o).any(), "topk masked-tile produced NaN"
        assert (o[0, 0, :32] == 0).all(), "masked rows must be zeros (II-6)"
        assert np.isfinite(o[0, 0, 32:]).all()
        assert np.abs(o[0, 0, 32:]).max() > 0  # unmasked rows are real


class TestF2DispatchEmptyRow:
    @pytest.mark.parametrize("density,thr", [(0.0, 1.01), (2.0, 0.5)])
    def test_both_branches_zero_masked_rows(self, density, thr):
        """Both the NAX kernel branch (density<thr) and the SDPA+bias
        branch (density>=thr) must emit zeros for a fully-masked row."""
        mx.random.seed(2)
        Q = mx.random.normal((1, 4, 1024, 128)).astype(mx.float16)
        K = mx.random.normal((1, 4, 1024, 128)).astype(mx.float16)
        V = mx.random.normal((1, 4, 1024, 128)).astype(mx.float16)
        bm = _masked_block_mask(1024 // 16, 1024 // 16, dead_q_tile=3)
        mx.eval(Q, K, V, bm)
        out = sparse_attention_dispatch(Q, K, V, bm, block_tile=16,
                                        density=density, density_threshold=thr)
        mx.eval(out)
        o = np.array(out.astype(mx.float32))
        assert not np.isnan(o).any(), f"NaN at density={density}"
        # dead tile 3 = rows 48..63
        assert (o[0, :, 48:64] == 0).all(), "masked rows must be zeros"
        assert np.isfinite(o[0, :, :48]).all()

    def test_branches_agree_on_masked_input(self):
        """The two branches must produce the SAME empty-row behavior
        (both zeros) — pre-fix one NaN'd, the other zeroed."""
        mx.random.seed(2)
        Q = mx.random.normal((1, 4, 1024, 128)).astype(mx.float16)
        K = mx.random.normal((1, 4, 1024, 128)).astype(mx.float16)
        V = mx.random.normal((1, 4, 1024, 128)).astype(mx.float16)
        bm = _masked_block_mask(1024 // 16, 1024 // 16, dead_q_tile=3)
        mx.eval(Q, K, V, bm)
        o_nax = sparse_attention_dispatch(Q, K, V, bm, block_tile=16,
                                          density=0.0, density_threshold=1.01)
        o_sdpa = sparse_attention_dispatch(Q, K, V, bm, block_tile=16,
                                           density=2.0, density_threshold=0.5)
        mx.eval(o_nax, o_sdpa)
        a = np.array(o_nax.astype(mx.float32))[0, :, 48:64]
        b = np.array(o_sdpa.astype(mx.float32))[0, :, 48:64]
        assert (a == 0).all() and (b == 0).all(), \
            "branches disagree on empty-row behavior"
