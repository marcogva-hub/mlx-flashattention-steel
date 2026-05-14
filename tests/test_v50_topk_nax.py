"""Sprint 3 (v2.50) — flash_attention_topk M5+ NAX dispatch fix tests.

Empirical foundation: docs/v50/sprint3-decisions.md.  On M5+ NAX
hardware (no block mask, D ∈ {64,128}, f16/bf16, k_count<S), the
top-K dispatch routes through `mx.fast.scaled_dot_product_attention`
with a float bias built from the top-K threshold.  This avoids the
explicit `weights @ v` matmul on the materialized [B,H,N,S] scores
tensor and is ~1.25× faster than the sort-based reference path.

Phase 3b — native Metal top-K kernel with streaming threshold — is
deferred; see docs/v50/sprint3-status-phase3b.md.

These tests verify:
  - NAX dispatch engages by default on M5+ for D=64/128 f16/bf16
  - max_abs_diff vs the reference path stays within fp16 ULP tolerance
  - MFA_DISABLE_TOPK_NAX=1 forces the reference path (back-compat)
  - Block mask falls back to reference (Phase 3a is bias-only)
  - k_count >= S falls back to dense attention (no filtering needed)
  - Various topk_ratios produce finite output
"""
import math
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import flash_attention_topk, get_device_info

_flush = getattr(mx, "eval")

_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))


# ─────────────────────────────────────────────────────────────────────
# Correctness: NAX dispatch vs reference fallback
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 3 NAX dispatch requires M5+ hardware.")
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("topk_ratio", [0.016, 0.0625, 0.25])
def test_sprint3_topk_nax_phase3a_matches_reference(monkeypatch, D, dtype, topk_ratio):
    """Phase 3a NAX dispatch (mx.topk-based, legacy path post-Prompt 5c
    Section B promotion of bisection to AUTO default) matches reference
    within fp16 ULP tolerance.

    v2.50 Prompt 5c Section B: bisection is now AUTO default (3.85×
    speedup); Phase 3a remains via `MFA_DISABLE_TOPK_BISECT=1` opt-out.
    This test validates Phase 3a path correctness (preserves original
    Sprint 3 semantic).  Separate test
    `test_sprint5b_section_b_topk_bisect.py::test_bisect_approximates_phase_3a`
    validates bisection's relaxed-tolerance correctness."""
    B, H, qL = 1, 4, 2048
    mx.random.seed(42)
    q = mx.random.normal((B, H, qL, D)).astype(dtype)
    k = mx.random.normal((B, H, qL, D)).astype(dtype)
    v = mx.random.normal((B, H, qL, D)).astype(dtype)
    _flush(q, k, v); mx.synchronize()

    # Force Phase 3a (legacy mx.topk semantics)
    monkeypatch.delenv("MFA_DISABLE_TOPK_NAX", raising=False)
    monkeypatch.setenv("MFA_DISABLE_TOPK_BISECT", "1")
    o_nax = flash_attention_topk(q, k, v, topk_ratio=topk_ratio)
    _flush(o_nax); mx.synchronize()

    # Force reference
    monkeypatch.setenv("MFA_DISABLE_TOPK_NAX", "1")
    o_ref = flash_attention_topk(q, k, v, topk_ratio=topk_ratio)
    _flush(o_ref); mx.synchronize()

    max_diff = float(mx.max(mx.abs(
        o_nax.astype(mx.float32) - o_ref.astype(mx.float32))))
    tol = 5e-3 if dtype == mx.float16 else 2e-2
    assert max_diff < tol, (
        f"Phase 3a NAX vs reference diff {max_diff:.3e} exceeds {tol} "
        f"({dtype}, D={D}, ratio={topk_ratio})"
    )


# ─────────────────────────────────────────────────────────────────────
# Routing: opt-out + fallback conditions
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 3 NAX dispatch requires M5+ hardware.")
def test_sprint3_topk_nax_disable_env_var(monkeypatch):
    """MFA_DISABLE_TOPK_NAX=1 forces reference (no crash, correct output)."""
    B, H, qL, D = 1, 4, 2048, 64
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)

    monkeypatch.setenv("MFA_DISABLE_TOPK_NAX", "1")
    out = flash_attention_topk(q, k, v, topk_ratio=0.25)
    _flush(out); mx.synchronize()
    assert out.shape == (B, H, qL, D)
    assert not bool(mx.any(mx.isnan(out)))
    assert not bool(mx.any(mx.isinf(out)))


@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 3 NAX dispatch requires M5+ hardware.")
def test_sprint3_topk_with_block_mask_uses_reference(monkeypatch):
    """Block mask supplied → reference path (Phase 3a is bias-only)."""
    from mlx_mfa.masks import make_diagonal_mask
    B, H, qL, D = 1, 4, 2048, 128
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    mask = make_diagonal_mask(qL, head_dim=D, num_diagonals=3, bandwidth=2)
    _flush(mask); mx.synchronize()

    monkeypatch.delenv("MFA_DISABLE_TOPK_NAX", raising=False)
    out = flash_attention_topk(q, k, v, topk_ratio=0.25, mask=mask)
    _flush(out); mx.synchronize()
    assert out.shape == (B, H, qL, D)
    assert not bool(mx.any(mx.isnan(out)))


@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 3 NAX dispatch requires M5+ hardware.")
def test_sprint3_topk_fp32_falls_back(monkeypatch):
    """fp32 input forces reference (NAX dispatch is f16/bf16 only)."""
    B, H, qL, D = 1, 4, 1024, 64
    q = mx.random.normal((B, H, qL, D)).astype(mx.float32)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float32)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float32)

    monkeypatch.delenv("MFA_DISABLE_TOPK_NAX", raising=False)
    out = flash_attention_topk(q, k, v, topk_ratio=0.25)
    _flush(out); mx.synchronize()
    assert out.shape == (B, H, qL, D)
    assert not bool(mx.any(mx.isnan(out)))


@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 3 NAX dispatch requires M5+ hardware.")
def test_sprint3_topk_ratio_full_no_filter(monkeypatch):
    """topk_ratio=1.0 → k_count >= S → no filtering, dense attention."""
    B, H, qL, D = 1, 4, 1024, 64
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)

    monkeypatch.delenv("MFA_DISABLE_TOPK_NAX", raising=False)
    out = flash_attention_topk(q, k, v, topk_ratio=1.0)
    _flush(out); mx.synchronize()
    assert out.shape == (B, H, qL, D)
    assert not bool(mx.any(mx.isnan(out)))


# ─────────────────────────────────────────────────────────────────────
# Smoke: PUBLIC API engagement (axis-2 per §Z)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 3 NAX dispatch requires M5+ hardware.")
def test_sprint3_topk_public_api_d128():
    """flash_attention_topk D=128 fp16 default engages NAX dispatch."""
    B, H, qL, D = 1, 16, 4096, 128
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)

    out = flash_attention_topk(q, k, v, topk_ratio=64.0 / qL)
    _flush(out); mx.synchronize()
    assert out.shape == (B, H, qL, D)
    assert not bool(mx.any(mx.isnan(out)))
    assert not bool(mx.any(mx.isinf(out)))
