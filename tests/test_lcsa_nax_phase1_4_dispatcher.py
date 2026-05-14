"""Sprint B Phase 1.4 - sparse_attention_dispatch correctness tests.

Three-axis validation for the density-thresholded dispatcher API per
docs/lcsa-nax/lcsa-nax-phase1_4-results.md.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

import mlx.core as mx

try:
    from mlx_mfa.lcsa_nax import (
        sparse_attention_dispatch,
        sparse_attention_nax,
        _bool_mask_to_float_bias,
        DEFAULT_DENSITY_THRESHOLD,
    )
    _HAS_EXT = True
except (ImportError, RuntimeError):
    _HAS_EXT = False

pytestmark = pytest.mark.skipif(
    not _HAS_EXT,
    reason="Sprint B sparse_attention_nax extension not built",
)


def _make_inputs(B, Hq, Hk, qL, kL, D, seed=0):
    mx.random.seed(seed)
    Q = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(mx.float16)
    K = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(mx.float16)
    V = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(mx.float16)
    mx.async_eval(Q, K, V); mx.synchronize()
    return Q, K, V


def test_dispatch_routes_very_sparse_to_nax():
    """density 0.01 < 0.02 threshold -> Sprint B kernel chosen."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 16
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(11)
    bm = (rng.random((NQ, NK)) < 0.01).astype(np.bool_)
    for q in range(NQ):
        if not bm[q].any():
            bm[q, q % NK] = True
    mask = mx.array(bm)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=11)
    # Both paths should agree
    O_disp = sparse_attention_dispatch(Q, K, V, mask, block_tile=BT)
    O_direct = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O_disp, O_direct); mx.synchronize()
    err = np.abs(np.array(O_disp.astype(mx.float32)) -
                 np.array(O_direct.astype(mx.float32)))
    assert err.max() < 1e-5, "Dispatcher at density 0.01 should route to NAX"


def test_dispatch_routes_moderate_density_to_sdpa():
    """density 0.10 > 0.02 threshold -> SDPA+bias path chosen."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 16
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(21)
    bm = (rng.random((NQ, NK)) < 0.10).astype(np.bool_)
    mask = mx.array(bm)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=21)
    bias = _bool_mask_to_float_bias(mask, BT, qL, kL, mx.float16)
    O_disp = sparse_attention_dispatch(Q, K, V, mask, block_tile=BT,
                                        precomputed_bias=bias)
    O_sdpa = mx.fast.scaled_dot_product_attention(
        Q, K, V, scale=1.0/math.sqrt(D), mask=bias)
    mx.async_eval(O_disp, O_sdpa); mx.synchronize()
    err = np.abs(np.array(O_disp.astype(mx.float32)) -
                 np.array(O_sdpa.astype(mx.float32)))
    assert err.max() < 1e-5, "Dispatcher at density 0.10 should match SDPA+bias"


def test_dispatch_threshold_override():
    """Explicit density_threshold parameter overrides default."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 16
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(31)
    bm = (rng.random((NQ, NK)) < 0.03).astype(np.bool_)
    for q in range(NQ):
        if not bm[q].any():
            bm[q, q % NK] = True
    mask = mx.array(bm)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=31)
    # Force threshold high so density 0.03 routes to NAX
    O_high_thresh = sparse_attention_dispatch(Q, K, V, mask, block_tile=BT,
                                                density_threshold=0.20)
    O_direct_nax = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O_high_thresh, O_direct_nax); mx.synchronize()
    err = np.abs(np.array(O_high_thresh.astype(mx.float32)) -
                 np.array(O_direct_nax.astype(mx.float32)))
    assert err.max() < 1e-5, "Override threshold=0.20 should route 0.03 to NAX"


def test_dispatch_precomputed_density_skips_reduction():
    """Passing density skips the in-dispatcher reduction (correctness check)."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 16
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(41)
    bm = (rng.random((NQ, NK)) < 0.005).astype(np.bool_)
    for q in range(NQ):
        if not bm[q].any():
            bm[q, q % NK] = True
    mask = mx.array(bm)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=41)
    O_with = sparse_attention_dispatch(Q, K, V, mask, block_tile=BT,
                                         density=0.005)
    O_without = sparse_attention_dispatch(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O_with, O_without); mx.synchronize()
    err = np.abs(np.array(O_with.astype(mx.float32)) -
                 np.array(O_without.astype(mx.float32)))
    assert err.max() < 1e-5, "Precomputed density should match auto-computed"


def test_default_threshold_value():
    """v2.50 Sprint 1 (Prompt 1) raised threshold 0.02 → 1.01 based on
    empirical bench showing LCSA NAX wins at all densities on M5+ for
    forward.  See `docs/v50/sprint1-decisions.md`.

    Note: this introduced a backward regression on M5+ for `mx.grad`
    via sparse path (see `docs/v50/sprint1-backward-regression-status.md`).
    The forward perf win is preserved; backward callers must pass
    `density_threshold=0.02` explicitly to override Sprint 1 default OR
    use `backward='steel_sparse'` to route through a kernel with vjp.
    """
    assert DEFAULT_DENSITY_THRESHOLD == 1.01


def test_dispatch_causal_path():
    """causal=True works through both routing paths."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 16
    NQ, NK = qL // BT, kL // BT
    # Causal lower-triangular block mask (Phase 1.2 pattern)
    bm = np.zeros((NQ, NK), dtype=np.bool_)
    for q in range(NQ):
        for k in range(min(q + 1, NK)):
            bm[q, k] = True
    mask = mx.array(bm)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=51)
    # density of causal mask ~0.5; dispatcher routes to SDPA path
    O_disp = sparse_attention_dispatch(
        Q, K, V, mask, block_tile=BT, causal=True)
    O_ref = mx.fast.scaled_dot_product_attention(
        Q, K, V, scale=1.0/math.sqrt(D), mask="causal")
    mx.async_eval(O_disp, O_ref); mx.synchronize()
    err = np.abs(np.array(O_disp.astype(mx.float32)) -
                 np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    assert rmse < 5e-3, f"causal dispatcher RMSE {rmse} too high"
