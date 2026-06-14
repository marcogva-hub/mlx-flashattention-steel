"""Sprint B Phase 1.2 - extended axes: shapes, dtype, mask ndim, causal, asymmetric.

Per design doc S8 Phase 1.2 row, this suite extends the Phase 1.1 scaffold
to cover:
  - 5 additional LCSA shape clusters (mid_seq8k +/- sparse, large_seq16k +/-
    sparse) at BT=32, plus optional BT=64 for sparse variants
  - bfloat16 dtype
  - mask ndim 3 (Hq, NQ, NK) per-head sparsity
  - mask ndim 4 (B, Hq, NQ, NK) per-batch per-head sparsity
  - causal=true within-tile triangular masking
  - asymmetric qL != kL (cross-attention)

Three-axis validation discipline per CLAUDE_V6_NAX S7 maintained for each
new axis: oracle correctness + path entered + edge preserved.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import mlx.core as mx

try:
    from mlx_mfa.lcsa_nax import sparse_attention_nax
    _HAS_EXT = True
except (ImportError, RuntimeError):
    _HAS_EXT = False

pytestmark = pytest.mark.skipif(
    not _HAS_EXT,
    reason="Sprint B sparse_attention_nax extension not built",
)


def _make_inputs(B, Hq, Hk, qL, kL, D, *, dtype=mx.float16, seed=0, mag=1.0):
    # III-4 F7: unit-scale inputs (normal, std 1.0; was uniform*0.1).
    # The II-6 lesson: 0.1-scale fixtures dilute localized kernel
    # corruption below the rmse gates.
    mx.random.seed(seed)
    Q = (mx.random.normal((B, Hq, qL, D)) * mag).astype(dtype)
    K = (mx.random.normal((B, Hk, kL, D)) * mag).astype(dtype)
    V = (mx.random.normal((B, Hk, kL, D)) * mag).astype(dtype)
    mx.async_eval(Q, K, V); mx.synchronize()
    return Q, K, V


def _block_mask_to_float_bias(bm_2d, BT, qL, kL, dtype):
    """Expand 2-D bool block_mask to (qL, kL) float bias (0 / -inf)."""
    bm_np = np.array(bm_2d).astype(bool)
    full = np.repeat(np.repeat(bm_np, BT, axis=0), BT, axis=1)
    bias = np.where(full, 0.0, -np.inf).astype(np.float32)
    if dtype == mx.float16:
        bias = bias.astype(np.float16)
    elif dtype == mx.bfloat16:
        bias = mx.array(bias).astype(mx.bfloat16)
        return bias
    return mx.array(bias)


def _causal_block_mask(NQ, NK):
    """Lower-triangular block mask + diagonal allowed (matches kernel's
    'k_tile <= q_tile' iteration with within-tile triangular)."""
    bm = np.zeros((NQ, NK), dtype=np.bool_)
    for q in range(NQ):
        for k in range(min(q + 1, NK)):
            bm[q, k] = True
    return mx.array(bm)


def _sdpa_with_bias(Q, K, V, bias, scale):
    return mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale, mask=bias)


# ============================================================
# 1. Additional LCSA shape clusters (mid_seq8k + large_seq16k variants)
# ============================================================

@pytest.mark.parametrize(
    "shape_name, qL, kL, density, BT, seed",
    [
        ("lcsa_small_seq4k_sparse",   4096,  4096, 0.07, 32, 11),
        ("lcsa_mid_seq8k",            8192,  8192, 0.12, 32, 12),
        ("lcsa_mid_seq8k_sparse",     8192,  8192, 0.03, 32, 13),
        ("lcsa_large_seq16k",        16384, 16384, 0.12, 32, 14),
        ("lcsa_large_seq16k_sparse", 16384, 16384, 0.03, 32, 15),
    ],
)
def test_axis1_correctness_lcsa_production_shapes(shape_name, qL, kL, density,
                                                   BT, seed):
    """All 5 additional LCSA production shapes - oracle correctness vs SDPA+bias."""
    B, Hq, Hk, D = 1, 8, 8, 128
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(seed)
    bm_np = (rng.random((NQ, NK)) < density).astype(np.bool_)
    # Guarantee no all-False rows for correctness test (each row gets diagonal)
    for q in range(NQ):
        if not bm_np[q].any():
            bm_np[q, min(q, NK - 1)] = True
    mask = mx.array(bm_np)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=seed)
    bias = _block_mask_to_float_bias(mask, BT, qL, kL, mx.float16)
    O = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O); mx.synchronize()
    O_ref = _sdpa_with_bias(Q, K, V, bias, scale=1.0/math.sqrt(D))
    mx.async_eval(O_ref); mx.synchronize()
    err = np.abs(np.array(O.astype(mx.float32)) - np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    # III-4 F7: unit-scale retrofit, measured floor rmse 7.7e-6 (M5 Max).
    assert rmse < 5e-3, f"{shape_name}: RMSE {rmse} too high (density={density})"
    assert not np.isnan(np.array(O.astype(mx.float32))).any(), \
        f"{shape_name}: NaN in output"


# ============================================================
# 2. bfloat16 dtype
# ============================================================

def test_axis1_bf16_correctness_vs_sdpa_bias():
    """bfloat16 dtype - matches SDPA+bias at bf16 noise floor."""
    B, Hq, Hk, qL, kL, D, BT = 1, 8, 8, 4096, 4096, 128, 32
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(99)
    bm_np = (rng.random((NQ, NK)) < 0.20).astype(np.bool_)
    for q in range(NQ):
        if not bm_np[q].any():
            bm_np[q, q] = True
    mask = mx.array(bm_np)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, dtype=mx.bfloat16, seed=100)
    O = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O); mx.synchronize()
    # bf16 oracle via SDPA+bias (cast bias to bf16)
    bias = _block_mask_to_float_bias(mask, BT, qL, kL, mx.bfloat16)
    O_ref = _sdpa_with_bias(Q, K, V, bias, scale=1.0/math.sqrt(D))
    mx.async_eval(O_ref); mx.synchronize()
    err = np.abs(np.array(O.astype(mx.float32)) - np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    # bf16 noise floor is higher (3 mantissa bits less than fp16).
    # III-4 F7: unit-scale retrofit, measured floor rmse 7.6e-5 (M5 Max).
    assert rmse < 2e-2, f"bf16 RMSE {rmse} too high"
    assert O.dtype == mx.bfloat16


def test_axis2_bf16_path_entered():
    """bf16 kernel actually dispatches without OOM/NaN at moderate shape."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 64, 32
    NQ, NK = qL // BT, kL // BT
    mask = mx.ones((NQ, NK), dtype=mx.bool_)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, dtype=mx.bfloat16, seed=200)
    O = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O); mx.synchronize()
    O_np = np.array(O.astype(mx.float32))
    assert not np.isnan(O_np).any() and not np.isinf(O_np).any()
    assert O.shape == (B, Hq, qL, D)


# ============================================================
# 3. 3-D mask (Hq, NQ, NK) per-head sparsity
# ============================================================

def test_axis1_mask_3d_per_head_correctness():
    """3-D mask: different sparsity per head, matches SDPA+per-head bias."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 32
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(31)
    bm_3d = np.zeros((Hq, NQ, NK), dtype=np.bool_)
    for h in range(Hq):
        bm_3d[h] = (rng.random((NQ, NK)) < (0.1 + 0.05 * h)).astype(np.bool_)
        for q in range(NQ):
            if not bm_3d[h, q].any():
                bm_3d[h, q, q] = True
    mask = mx.array(bm_3d)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=32)
    O = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O); mx.synchronize()
    # Build per-head bias for the oracle. Concatenate per-head SDPA calls.
    O_ref_per_head = []
    for h in range(Hq):
        bias_h = _block_mask_to_float_bias(mx.array(bm_3d[h]), BT, qL, kL, mx.float16)
        # SDPA expects (B, H, qL, kL) bias; use (1, 1, qL, kL) and broadcast
        bias_h_4d = mx.expand_dims(mx.expand_dims(bias_h, 0), 0)
        Oh = _sdpa_with_bias(
            Q[:, h:h+1, :, :], K[:, h:h+1, :, :], V[:, h:h+1, :, :],
            bias_h_4d, scale=1.0/math.sqrt(D))
        O_ref_per_head.append(Oh)
    O_ref = mx.concatenate(O_ref_per_head, axis=1)
    mx.async_eval(O_ref); mx.synchronize()
    err = np.abs(np.array(O.astype(mx.float32)) - np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    assert rmse < 5e-3, f"3-D mask RMSE {rmse} too high"


# ============================================================
# 4. 4-D mask (B, Hq, NQ, NK)
# ============================================================

def test_axis1_mask_4d_per_batch_per_head_correctness():
    """4-D mask: independent sparsity per (b, h)."""
    B, Hq, Hk, qL, kL, D, BT = 2, 4, 4, 4096, 4096, 128, 32
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(41)
    bm_4d = np.zeros((B, Hq, NQ, NK), dtype=np.bool_)
    for b in range(B):
        for h in range(Hq):
            bm_4d[b, h] = (rng.random((NQ, NK)) < (0.10 + 0.03 * (b * Hq + h))).astype(np.bool_)
            for q in range(NQ):
                if not bm_4d[b, h, q].any():
                    bm_4d[b, h, q, q] = True
    mask = mx.array(bm_4d)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=42)
    O = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O); mx.synchronize()
    # Oracle: per (b, h) SDPA call
    O_ref_blocks = []
    for b in range(B):
        per_h = []
        for h in range(Hq):
            bias_bh = _block_mask_to_float_bias(
                mx.array(bm_4d[b, h]), BT, qL, kL, mx.float16)
            bias_bh_4d = mx.expand_dims(mx.expand_dims(bias_bh, 0), 0)
            Obh = _sdpa_with_bias(
                Q[b:b+1, h:h+1, :, :], K[b:b+1, h:h+1, :, :], V[b:b+1, h:h+1, :, :],
                bias_bh_4d, scale=1.0/math.sqrt(D))
            per_h.append(Obh)
        O_ref_blocks.append(mx.concatenate(per_h, axis=1))
    O_ref = mx.concatenate(O_ref_blocks, axis=0)
    mx.async_eval(O_ref); mx.synchronize()
    err = np.abs(np.array(O.astype(mx.float32)) - np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    assert rmse < 5e-3, f"4-D mask RMSE {rmse} too high"


# ============================================================
# 5. causal=true (per-tile skip + within-tile triangular)
# ============================================================

def test_axis1_causal_matches_sdpa_causal():
    """causal=true with all-True lower-triangular mask = standard causal SDPA."""
    B, Hq, Hk, qL, kL, D, BT = 1, 8, 8, 4096, 4096, 128, 32
    NQ, NK = qL // BT, kL // BT
    mask = _causal_block_mask(NQ, NK)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=51)
    O = sparse_attention_nax(Q, K, V, mask, block_tile=BT, causal=True)
    mx.async_eval(O); mx.synchronize()
    O_ref = mx.fast.scaled_dot_product_attention(
        Q, K, V, scale=1.0/math.sqrt(D), mask="causal")
    mx.async_eval(O_ref); mx.synchronize()
    err = np.abs(np.array(O.astype(mx.float32)) - np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    # III-4 F7: unit-scale retrofit, measured floor rmse 2.9e-5 (M5 Max).
    assert rmse < 5e-3, f"causal RMSE {rmse} too high vs SDPA causal"


def test_axis3_causal_future_positions_unused():
    """causal=true: changing future K/V values must NOT change earlier output rows."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 32
    NQ, NK = qL // BT, kL // BT
    mask = _causal_block_mask(NQ, NK)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=61)
    O1 = sparse_attention_nax(Q, K, V, mask, block_tile=BT, causal=True)
    mx.async_eval(O1); mx.synchronize()
    # Perturb second half of K and V (positions >= qL/2)
    K_np = np.array(K).copy()
    V_np = np.array(V).copy()
    K_np[:, :, qL // 2:, :] += 1.0  # huge perturbation
    V_np[:, :, qL // 2:, :] += 1.0
    K2 = mx.array(K_np).astype(mx.float16)
    V2 = mx.array(V_np).astype(mx.float16)
    O2 = sparse_attention_nax(Q, K2, V2, mask, block_tile=BT, causal=True)
    mx.async_eval(O2); mx.synchronize()
    # First half (causal: can only see K[:qL/2]) should be IDENTICAL
    diff = np.abs(np.array(O1.astype(mx.float32))[:, :, :qL // 2, :] -
                  np.array(O2.astype(mx.float32))[:, :, :qL // 2, :])
    assert diff.max() < 1e-4, f"Causal violated: first half changed by {diff.max()}"


# ============================================================
# 6. Asymmetric qL != kL (cross-attention pattern)
# ============================================================

def test_axis1_asymmetric_cross_attn_correctness():
    """qL=2048, kL=4096 (cross-attention) - matches SDPA+bias."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 2048, 4096, 128, 32
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(71)
    bm_np = (rng.random((NQ, NK)) < 0.25).astype(np.bool_)
    for q in range(NQ):
        if not bm_np[q].any():
            bm_np[q, q % NK] = True
    mask = mx.array(bm_np)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=72)
    bias = _block_mask_to_float_bias(mask, BT, qL, kL, mx.float16)
    O = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O); mx.synchronize()
    O_ref = _sdpa_with_bias(Q, K, V, bias, scale=1.0/math.sqrt(D))
    mx.async_eval(O_ref); mx.synchronize()
    err = np.abs(np.array(O.astype(mx.float32)) - np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    assert rmse < 5e-3, f"Asymmetric qL=2048 kL=4096 RMSE {rmse} too high"
    assert O.shape == (B, Hq, qL, D)


def test_adversarial_magnitude_finite():
    """III-4 F7: adversarial-magnitude (std 8) inputs must stay finite
    (fp16-overflow guard for the sparse NAX kernel family)."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 32
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(81)
    bm_np = (rng.random((NQ, NK)) < 0.25).astype(np.bool_)
    for q in range(NQ):
        if not bm_np[q].any():
            bm_np[q, q] = True
    mask = mx.array(bm_np)
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=82, mag=8.0)
    O = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O); mx.synchronize()
    assert np.isfinite(np.array(O.astype(mx.float32))).all(), \
        "non-finite output at std-8 inputs"
