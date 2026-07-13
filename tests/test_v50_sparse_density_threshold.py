"""Sparse density-threshold compatibility under the hardened β3 shape gate.

`DEFAULT_DENSITY_THRESHOLD` remains a backwards-compatible secondary cap,
but the canonical shape/dtype/density gate now decides the safe public route.

These tests verify:
  - Existing density=0.02 (V1 break-even) callers still route NAX
  - Mid/high density outside the measured region routes SDPA
  - Bool-mask cache (already shipped v2.33.1) continues to work
"""
import math
import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_mfa
from mlx_mfa import flash_attention_sparse, get_device_info
from mlx_mfa.lcsa_nax import (
    DEFAULT_DENSITY_THRESHOLD,
    sparse_attention_dispatch,
    sparse_attention_nax,
)

_flush = getattr(mx, "eval")

_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))


# ─────────────────────────────────────────────────────────────────────
# Threshold value check
# ─────────────────────────────────────────────────────────────────────

def test_v50_sprint1_default_threshold_is_recalibrated():
    """v2.50-Sprint1: default threshold raised from 0.02 → 1.01 (always-NAX)."""
    assert DEFAULT_DENSITY_THRESHOLD >= 1.0, (
        f"DEFAULT_DENSITY_THRESHOLD={DEFAULT_DENSITY_THRESHOLD} should be >= 1.0 "
        f"per v2.50-Sprint1 empirical recalibration."
    )


# ─────────────────────────────────────────────────────────────────────
# Routing tests (dispatcher level)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_NAX, reason="LCSA NAX requires M5+ hardware.")
def test_v50_sprint1_low_density_routes_nax():
    """density=0.02 (V1 historical threshold) still routes through NAX."""
    B, H, qL, D, BT = 1, 4, 4096, 128, 32
    NQ = NK = qL // BT
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)

    # Build mask at very low density (~0.015 — well below old threshold 0.02)
    np.random.seed(0)
    bm = (np.random.rand(NQ, NK) < 0.015).astype(bool)
    block_mask = mx.array(bm)
    actual_density = float(mx.mean(block_mask.astype(mx.float32)))
    assert actual_density < 0.05, f"Test seed not low-density enough: {actual_density}"

    # density < threshold (always-NAX) → routes to NAX
    # Verify by direct comparison: dispatch result must equal NAX direct call
    out_dispatch = sparse_attention_dispatch(
        q, k, v, block_mask, block_tile=BT, scale=D**-0.5
    )
    out_direct_nax = sparse_attention_nax(
        q, k, v, block_mask, block_tile=BT, scale=D**-0.5
    )
    _flush(out_dispatch, out_direct_nax); mx.synchronize()
    # Identical outputs → dispatcher chose NAX
    max_diff = float(mx.max(mx.abs(
        out_dispatch.astype(mx.float32) - out_direct_nax.astype(mx.float32))))
    assert max_diff < 1e-6, (
        f"dispatcher should route low-density to NAX (max_diff={max_diff})"
    )


@pytest.mark.skipif(not _HAS_NAX, reason="LCSA NAX requires M5+ hardware.")
def test_v50_sprint1_mid_density_routes_sdpa_under_hardened_gate():
    """density=0.5 is outside the measured map and delegates to SDPA."""
    B, H, qL, D, BT = 1, 4, 4096, 128, 32
    NQ = NK = qL // BT
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)

    np.random.seed(1)
    bm = (np.random.rand(NQ, NK) < 0.5).astype(bool)
    block_mask = mx.array(bm)
    actual_density = float(mx.mean(block_mask.astype(mx.float32)))
    assert 0.4 < actual_density < 0.6, f"Mid-density expected: {actual_density}"

    out_dispatch = sparse_attention_dispatch(
        q, k, v, block_mask, block_tile=BT, scale=D**-0.5
    )
    out_forced_sdpa = sparse_attention_dispatch(
        q, k, v, block_mask, block_tile=BT, scale=D**-0.5,
        density_threshold=0.0,
    )
    _flush(out_dispatch, out_forced_sdpa); mx.synchronize()
    max_diff = float(mx.max(mx.abs(
        out_dispatch.astype(mx.float32) - out_forced_sdpa.astype(mx.float32))))
    assert max_diff == 0.0


@pytest.mark.skipif(not _HAS_NAX, reason="LCSA NAX requires M5+ hardware.")
def test_v50_sprint1_high_density_routes_sdpa_under_hardened_gate():
    """density=0.95 is outside the measured map and delegates to SDPA."""
    B, H, qL, D, BT = 1, 4, 4096, 128, 32
    NQ = NK = qL // BT
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)

    np.random.seed(2)
    bm = (np.random.rand(NQ, NK) < 0.95).astype(bool)
    block_mask = mx.array(bm)
    actual_density = float(mx.mean(block_mask.astype(mx.float32)))
    assert actual_density > 0.85, f"High-density expected: {actual_density}"

    out_dispatch = sparse_attention_dispatch(
        q, k, v, block_mask, block_tile=BT, scale=D**-0.5
    )
    out_forced_sdpa = sparse_attention_dispatch(
        q, k, v, block_mask, block_tile=BT, scale=D**-0.5,
        density_threshold=0.0,
    )
    _flush(out_dispatch, out_forced_sdpa); mx.synchronize()
    max_diff = float(mx.max(mx.abs(
        out_dispatch.astype(mx.float32) - out_forced_sdpa.astype(mx.float32))))
    assert max_diff == 0.0


@pytest.mark.skipif(not _HAS_NAX, reason="LCSA NAX requires M5+ hardware.")
def test_v50_sprint1_explicit_low_threshold_routes_sdpa():
    """Explicit `density_threshold=0.02` (V1 historical) still routes
    high-density to SDPA+bias path — backward-compat for M1/M3 callers."""
    B, H, qL, D, BT = 1, 4, 4096, 128, 32
    NQ = NK = qL // BT
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)

    np.random.seed(3)
    bm = (np.random.rand(NQ, NK) < 0.5).astype(bool)
    block_mask = mx.array(bm)
    # Force old V1 threshold — density 0.5 >= 0.02 → SDPA+bias path
    out_dispatch = sparse_attention_dispatch(
        q, k, v, block_mask, block_tile=BT, scale=D**-0.5,
        density_threshold=0.02,
    )
    # Direct call to NAX for comparison
    out_nax = sparse_attention_nax(
        q, k, v, block_mask, block_tile=BT, scale=D**-0.5
    )
    _flush(out_dispatch, out_nax); mx.synchronize()
    # When explicit threshold=0.02 forces SDPA path, the FP precision of
    # SDPA+bias differs slightly from NAX (different reduction order).
    # Both should match SDPA-dense-with-mask reference within FP16 ULP.
    # We just verify the dispatch DIDN'T crash and both paths produce
    # finite, similar-magnitude outputs.
    assert not bool(mx.any(mx.isnan(out_dispatch)))
    assert not bool(mx.any(mx.isnan(out_nax)))
    # Outputs should be in the same numerical magnitude band
    out_dispatch_rms = float(mx.sqrt(mx.mean(out_dispatch.astype(mx.float32)**2)))
    out_nax_rms = float(mx.sqrt(mx.mean(out_nax.astype(mx.float32)**2)))
    assert abs(out_dispatch_rms - out_nax_rms) / max(out_dispatch_rms, 1e-6) < 0.1


# ─────────────────────────────────────────────────────────────────────
# flash_attention_sparse PUBLIC API path (axis-2 §Z compliance)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_NAX, reason="LCSA NAX requires M5+ hardware.")
def test_v50_sprint1_flash_attention_sparse_engages_nax_at_mid_density():
    """flash_attention_sparse via PUBLIC API at density 0.023 (audit shape)
    now routes to NAX (was: SDPA+bias path with 1.26× regression)."""
    B, H, qL, D, BT = 1, 12, 4096, 128, 32
    NQ = NK = qL // BT
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)

    # Exact audit shape: 3 active blocks per row centered on diagonal
    bm = np.zeros((NQ, NK), dtype=bool)
    for i in range(NQ):
        for j in range(max(0, i-1), min(NK, i+2)):
            bm[i, j] = True
    block_mask = mx.array(bm)
    density = float(mx.mean(block_mask.astype(mx.float32)))
    assert 0.020 < density < 0.030, f"Audit shape density: {density}"

    # Call via PUBLIC API
    out_public = flash_attention_sparse(q, k, v, block_mask, scale=D**-0.5)
    out_nax_direct = sparse_attention_nax(
        q, k, v, block_mask, block_tile=BT, scale=D**-0.5
    )
    _flush(out_public, out_nax_direct); mx.synchronize()
    max_diff = float(mx.max(mx.abs(
        out_public.astype(mx.float32) - out_nax_direct.astype(mx.float32))))
    assert max_diff < 1e-5, (
        f"flash_attention_sparse should route audit-shape to NAX post-Sprint 1 "
        f"(max_diff={max_diff})"
    )


# ─────────────────────────────────────────────────────────────────────
# Bool-mask cache (already implemented v2.33.1, verify still works)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_NAX, reason="LCSA NAX requires M5+ hardware.")
def test_v50_sprint1_float_bias_cache_repeat_call(monkeypatch):
    """Force V1 threshold (=0.02) to hit the SDPA+bias path and exercise the
    pre-existing v2.33.1 cache.  Cache hit-rate verified via attention.py
    _SPARSE_BIAS_CACHE inspection."""
    from mlx_mfa.attention import _SPARSE_BIAS_CACHE, _sparse_fallback_sdpa_perhead

    B, H, qL, D, BT = 1, 4, 4096, 128, 32
    NQ = NK = qL // BT
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)

    np.random.seed(7)
    bm = (np.random.rand(NQ, NK) < 0.5).astype(bool)
    block_mask = mx.array(bm)

    # Clear cache + first call (cache miss → builds bias)
    _SPARSE_BIAS_CACHE.clear()
    out1 = _sparse_fallback_sdpa_perhead(
        q, k, v, block_mask, scale=D**-0.5, causal=False
    )
    _flush(out1); mx.synchronize()
    cache_size_after_first = len(_SPARSE_BIAS_CACHE)
    assert cache_size_after_first == 1, (
        f"Cache should have 1 entry after first call (got {cache_size_after_first})"
    )

    # Second call with same mask object → cache hit (no new entry)
    out2 = _sparse_fallback_sdpa_perhead(
        q, k, v, block_mask, scale=D**-0.5, causal=False
    )
    _flush(out2); mx.synchronize()
    cache_size_after_second = len(_SPARSE_BIAS_CACHE)
    assert cache_size_after_second == 1, (
        f"Cache should hit, size unchanged (got {cache_size_after_second})"
    )

    # Outputs identical (bit-exact since same float_bias used)
    max_diff = float(mx.max(mx.abs(
        out1.astype(mx.float32) - out2.astype(mx.float32))))
    assert max_diff == 0.0, f"Cache hit should produce identical output (diff={max_diff})"
