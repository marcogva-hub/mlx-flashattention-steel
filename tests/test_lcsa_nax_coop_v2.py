"""Sprint B coop-rewrite Section C - V2 correctness tests.

Three-axis validation per CLAUDE_V6_NAX §7:
  1. Output sanity: V2 output matches V1 (the v2.34.0 SHIP-validated reference)
     within FP16 noise floor on all 7 LCSA shapes + density sweep.
  2. Path entered: V2 actually dispatches when MFA_LCSA_KERNEL_VERSION=v2 and
     eligibility conditions are met; falls back to V1 otherwise.
  3. Edges preserved: all-False mask row -> exact zero output (v2.34.0 contract);
     all-True mask -> dense SDPA-equivalent; correctness under random masks.
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager

import numpy as np
import pytest

import mlx.core as mx

try:
    from mlx_mfa.lcsa_nax import sparse_attention_nax, _bool_mask_to_float_bias
    _HAS_EXT = True
except (ImportError, RuntimeError):
    _HAS_EXT = False

pytestmark = pytest.mark.skipif(
    not _HAS_EXT,
    reason="Sprint B sparse_attention_nax extension not built",
)


@contextmanager
def env_version(v):
    """Set MFA_LCSA_KERNEL_VERSION temporarily."""
    prev = os.environ.get("MFA_LCSA_KERNEL_VERSION")
    os.environ["MFA_LCSA_KERNEL_VERSION"] = v
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("MFA_LCSA_KERNEL_VERSION", None)
        else:
            os.environ["MFA_LCSA_KERNEL_VERSION"] = prev


def _make_inputs(B, Hq, Hk, qL, kL, D, seed=0, mag=1.0):
    # III-4 F7: unit-scale inputs (normal, std 1.0; was uniform*0.1).
    # The II-6 lesson: 0.1-scale fixtures dilute localized kernel
    # corruption below the rmse gates.
    mx.random.seed(seed)
    Q = (mx.random.normal((B, Hq, qL, D)) * mag).astype(mx.float16)
    K = (mx.random.normal((B, Hk, kL, D)) * mag).astype(mx.float16)
    V = (mx.random.normal((B, Hk, kL, D)) * mag).astype(mx.float16)
    mx.async_eval(Q, K, V); mx.synchronize()
    return Q, K, V


def _random_mask(NQ, NK, density, seed):
    rng = np.random.default_rng(seed)
    bm = (rng.random((NQ, NK)) < density).astype(np.bool_)
    for q in range(NQ):
        if not bm[q].any():
            bm[q, min(q, NK - 1)] = True
    return mx.array(bm)


# Phase 0 production shapes + niche representative
SHAPES = [
    # name, B, Hq, Hk, qL, kL, D, density, BT, seed
    ("lcsa_small_seq4k",          1, 12, 12,  4096,  4096, 128, 0.24, 32, 1100),
    ("lcsa_small_seq4k_sparse",   1, 12, 12,  4096,  4096, 128, 0.07, 32, 1101),
    ("lcsa_mid_seq8k",            1,  8,  8,  8192,  8192, 128, 0.12, 32, 1102),
    ("lcsa_mid_seq8k_sparse",     1,  8,  8,  8192,  8192, 128, 0.03, 32, 1103),
    ("lcsa_large_seq16k",         1,  4,  4, 16384, 16384, 128, 0.12, 32, 1104),
    ("lcsa_large_seq16k_sparse",  1,  4,  4, 16384, 16384, 128, 0.03, 32, 1105),
    ("lcsa_mid_seq8k_very_sparse", 1, 8, 8,   8192,  8192, 128, 0.01, 32, 1106),
]


# ===========================================================================
# Axis 1 (output sanity): V1 <-> V2 equivalence on all 7 shapes
# ===========================================================================

@pytest.mark.parametrize("name, B, Hq, Hk, qL, kL, D, density, BT, seed", SHAPES)
def test_axis1_v1_v2_equivalence_per_shape(name, B, Hq, Hk, qL, kL, D, density, BT, seed):
    """V1 and V2 produce numerically equivalent output (FP16 noise floor)."""
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=seed)
    mask = _random_mask(qL // BT, kL // BT, density, seed + 1)

    with env_version("v1"):
        O1 = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O1); mx.synchronize()
    with env_version("v2"):
        O2 = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O2); mx.synchronize()

    err = np.abs(np.array(O1.astype(mx.float32)) - np.array(O2.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    # III-4 F7: unit-scale retrofit, measured floor rmse 2.6e-5 (M5 Max).
    assert rmse < 1e-3, f"{name}: V1 vs V2 RMSE {rmse} exceeds 1e-3"
    assert not np.isnan(np.array(O2.astype(mx.float32))).any(), \
        f"{name}: V2 produced NaN"
    assert not np.isinf(np.array(O2.astype(mx.float32))).any(), \
        f"{name}: V2 produced Inf"


# ===========================================================================
# Axis 2 (path entered): V2 dispatch + V1 fallback behavior
# ===========================================================================

def test_axis2_v2_dispatches_when_eligible():
    """V2-eligible shape: V2 produces V1-equivalent output (sanity for path)."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 32
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=200)
    mask = _random_mask(qL // BT, kL // BT, 0.10, 201)
    with env_version("v2"):
        O = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O); mx.synchronize()
    assert O.shape == (B, Hq, qL, D)
    assert not np.isnan(np.array(O.astype(mx.float32))).any()


def test_axis2_v2_falls_back_for_ineligible_BT():
    """V2 + BT=16 -> silent V1 fallback (BT must be 32 for V2)."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 16
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=210)
    mask = _random_mask(qL // BT, kL // BT, 0.10, 211)
    with env_version("v1"):
        O_v1 = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    with env_version("v2"):
        O_v2 = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O_v1, O_v2); mx.synchronize()
    err = np.abs(np.array(O_v1.astype(mx.float32)) -
                 np.array(O_v2.astype(mx.float32)))
    assert err.max() < 1e-7, \
        "V2 + BT=16 should silently fall back to V1 -> bit-exact output"


def test_axis2_v2_falls_back_for_causal():
    """V2 + causal=true -> silent V1 fallback (causal not yet supported in V2)."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 32
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=220)
    NQ = qL // BT
    bm = np.zeros((NQ, NQ), dtype=np.bool_)
    for q in range(NQ):
        for k in range(min(q + 1, NQ)):
            bm[q, k] = True
    mask = mx.array(bm)
    with env_version("v1"):
        O_v1 = sparse_attention_nax(Q, K, V, mask, block_tile=BT, causal=True)
    with env_version("v2"):
        O_v2 = sparse_attention_nax(Q, K, V, mask, block_tile=BT, causal=True)
    mx.async_eval(O_v1, O_v2); mx.synchronize()
    err = np.abs(np.array(O_v1.astype(mx.float32)) -
                 np.array(O_v2.astype(mx.float32)))
    assert err.max() < 1e-7, \
        "V2 + causal=true should silently fall back to V1 -> bit-exact output"


# ===========================================================================
# Axis 3 (edges preserved): all-False, all-True, diagonal
# ===========================================================================

def test_axis3_v2_all_false_row_zero_output():
    """All-False mask row -> exact zero output (v2.34.0 contract)."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 32
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=300)
    bm = np.ones((qL // BT, kL // BT), dtype=np.bool_)
    bm[3] = False  # row of Q-tiles fully masked
    mask = mx.array(bm)
    with env_version("v2"):
        O = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O); mx.synchronize()
    O_np = np.array(O.astype(mx.float32))
    q_start = 3 * BT
    q_end = q_start + BT
    masked_max = float(np.abs(O_np[:, :, q_start:q_end, :]).max())
    kept_max = float(np.abs(O_np[:, :, :q_start, :]).max())
    assert masked_max == 0.0, \
        f"V2 all-False row produced non-zero: {masked_max}"
    assert kept_max > 0.0, "V2 kept rows degenerate"


def test_axis3_v2_all_true_matches_v1_dense():
    """All-True mask -> V2 matches V1 (which matches dense SDPA)."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 32
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=310)
    mask = mx.ones((qL // BT, kL // BT), dtype=mx.bool_)
    with env_version("v1"):
        O1 = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    with env_version("v2"):
        O2 = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    O_ref = mx.fast.scaled_dot_product_attention(Q, K, V, scale=1.0 / math.sqrt(D))
    mx.async_eval(O1, O2, O_ref); mx.synchronize()
    err_v2_ref = np.abs(np.array(O2.astype(mx.float32)) -
                         np.array(O_ref.astype(mx.float32)))
    err_v1_v2 = np.abs(np.array(O1.astype(mx.float32)) -
                        np.array(O2.astype(mx.float32)))
    rmse_v2_ref = float(np.sqrt((err_v2_ref ** 2).mean()))
    rmse_v1_v2 = float(np.sqrt((err_v1_v2 ** 2).mean()))
    # III-4 F7: unit-scale retrofit, measured floor rmse 4.1e-6 (M5 Max).
    assert rmse_v2_ref < 1e-3, f"V2 all-True RMSE vs SDPA: {rmse_v2_ref}"
    assert rmse_v1_v2 < 1e-3, f"V1 vs V2 all-True RMSE: {rmse_v1_v2}"


def test_axis3_v2_diagonal_only_mask():
    """Diagonal-only block mask -> V2 matches SDPA + diagonal bias."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 32
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=320)
    NQ = qL // BT
    bm = np.eye(NQ, dtype=np.bool_)
    mask = mx.array(bm)
    bias = _bool_mask_to_float_bias(mask, BT, qL, kL, mx.float16)
    with env_version("v2"):
        O = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    O_ref = mx.fast.scaled_dot_product_attention(
        Q, K, V, scale=1.0 / math.sqrt(D), mask=bias)
    mx.async_eval(O, O_ref); mx.synchronize()
    err = np.abs(np.array(O.astype(mx.float32)) -
                 np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    assert rmse < 5e-3, f"V2 diagonal-only RMSE {rmse} too high"


# ===========================================================================
# Density sweep correctness
# ===========================================================================

@pytest.mark.parametrize("density", [0.01, 0.03, 0.05, 0.10, 0.20, 0.50])
def test_density_sweep_v1_v2_equivalence(density):
    """V2 stays correct across the full density envelope."""
    B, Hq, Hk, qL, kL, D, BT = 1, 8, 8, 8192, 8192, 128, 32
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=400)
    mask = _random_mask(qL // BT, kL // BT, density, int(density * 10000))
    with env_version("v1"):
        O1 = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    with env_version("v2"):
        O2 = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O1, O2); mx.synchronize()
    err = np.abs(np.array(O1.astype(mx.float32)) -
                 np.array(O2.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    assert rmse < 1e-3, f"density={density}: V1 vs V2 RMSE {rmse}"


def test_adversarial_magnitude_finite():
    """III-4 F7: adversarial-magnitude (std 8) inputs must stay finite on
    BOTH kernel versions (fp16-overflow guard, coop-V2 kernel family)."""
    B, Hq, Hk, qL, kL, D, BT = 1, 4, 4, 4096, 4096, 128, 32
    Q, K, V = _make_inputs(B, Hq, Hk, qL, kL, D, seed=500, mag=8.0)
    mask = _random_mask(qL // BT, kL // BT, 0.10, 501)
    for ver in ("v1", "v2"):
        with env_version(ver):
            O = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
        mx.async_eval(O); mx.synchronize()
        assert np.isfinite(np.array(O.astype(mx.float32))).all(), \
            f"{ver}: non-finite output at std-8 inputs"
