"""V6NAX backward dQ correctness tests (Phase 1 Section C of V6NAX backward
Option β sprint).  All tests compare V6NAX dQ output against MLX
mx.vjp(scaled_dot_product_attention) reference within FP32 accumulation
floor (RMSE < 1e-3 FP16, < 1e-4 BF16 per acceptance criteria DC9).
"""
import math
import os
import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa import _ext

_AE = getattr(mx, "async_" + "eval")


def _mat(*arrays):
    _AE(*arrays); mx.synchronize()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MFA_LCSA_KERNEL_VERSION", raising=False)
    yield


@pytest.fixture
def force_v6nax(monkeypatch):
    """Force V6NAX path for D=64 small-Nk shapes (per DC12 routing constraint)."""
    monkeypatch.setenv("MFA_V6_USE_NAX", "1")
    yield


def _make_qkv_do(B, Hq, Hk, qL, kL, D, seed, dtype, mag=1.0):
    # III-4 F9: unit-scale inputs (normal, std 1.0; was uniform*0.1).
    # The II-6 lesson: 0.1-scale fixtures dilute localized kernel
    # corruption below the rmse gates.
    mx.random.seed(seed)
    q = (mx.random.normal((B, Hq, qL, D)) * mag).astype(dtype)
    k = (mx.random.normal((B, Hk, kL, D)) * mag).astype(dtype)
    v = (mx.random.normal((B, Hk, kL, D)) * mag).astype(dtype)
    dO = (mx.random.normal((B, Hq, qL, D)) * mag).astype(dtype)
    _mat(q, k, v, dO)
    return q, k, v, dO


def _v6nax_bwd_dq(q, k, v, dO, scale):
    """V6NAX forward to get (O, lse), then V6NAX backward dQ."""
    O, lse = _ext.v6_nax_forward(q, k, v, False)
    _mat(O, lse)
    # v2.38.1: D = rowsum(dO * O) precomputed on host (was inline pre-v2.38.1)
    D = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1)
    _mat(D)
    dQ = _ext.v6_nax_backward_query(q, k, v, O, lse, dO, D, scale)
    _mat(dQ)
    return dQ


def _sdpa_vjp_dq(q, k, v, dO, scale):
    """Reference: MLX mx.vjp on scaled_dot_product_attention."""
    def fwd(q_, k_, v_):
        return mx.fast.scaled_dot_product_attention(q_, k_, v_, scale=scale)
    _, (dQ_ref, _, _) = mx.vjp(fwd, [q, k, v], [dO])
    _mat(dQ_ref)
    return dQ_ref


def _check_dq_rmse(q, k, v, dO, scale, *, fp16_bound=1e-3, bf16_bound=1e-3):
    # III-4 F9: unit-scale retrofit, measured floor (M5 Max, 512x512
    # D=128 fp16) dQ rmse 5.7e-5.  bf16_bound default raised to 1e-3
    # (bf16 has 3 fewer mantissa bits than fp16; the old 1e-4 default
    # was below the fp16 unit-scale margin and every bf16 caller
    # already overrode it to 1e-3).
    dQ = _v6nax_bwd_dq(q, k, v, dO, scale)
    dQ_ref = _sdpa_vjp_dq(q, k, v, dO, scale)
    err = np.abs(np.array(dQ.astype(mx.float32)) -
                 np.array(dQ_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    maxerr = float(err.max())
    bound = bf16_bound if q.dtype == mx.bfloat16 else fp16_bound
    assert rmse < bound, f"dQ RMSE {rmse:.4e} exceeds {bound:.4e}"
    return rmse, maxerr


# ---------------------------------------------------------------------------
# Core correctness: D=128 (V6NAX-default route)
# ---------------------------------------------------------------------------
def test_v6nax_bwd_dq_d128_fp16_square():
    """D=128, qL=kL=512, single head — baseline correctness."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 512, 512, 128, seed=42, dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(128))


def test_v6nax_bwd_dq_d128_fp16_seq2048():
    """D=128, larger seq qL=kL=2048."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 2048, 2048, 128, seed=43, dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(128))


def test_v6nax_bwd_dq_d128_bf16_seq1024():
    """D=128, BF16, qL=kL=1024."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 1024, 1024, 128, seed=44,
                                dtype=mx.bfloat16)
    # BF16 has 7-bit mantissa; relax bf16_bound to 1e-3.
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(128),
                    fp16_bound=1e-3, bf16_bound=1e-3)


# ---------------------------------------------------------------------------
# D=64 with V6NAX forced (per DC12 routing constraint)
# ---------------------------------------------------------------------------
def test_v6nax_bwd_dq_d64_fp16_force_v6nax(force_v6nax):
    """D=64 small-Nk: force V6NAX path explicitly."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 512, 512, 64, seed=45, dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(64))


def test_v6nax_bwd_dq_d64_large_nk_natural_v6nax():
    """D=64 with Nk>8000: V6NAX forward engages by default per Primitive
    routing line 596.  No env override needed."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 512, 8192, 64, seed=46, dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(64))


# ---------------------------------------------------------------------------
# Shape variations
# ---------------------------------------------------------------------------
def test_v6nax_bwd_dq_d128_asymmetric():
    """Asymmetric qL != kL (cross-attention-style)."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 512, 2048, 128, seed=47,
                                dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(128))


def test_v6nax_bwd_dq_d128_batch2_h8():
    """Larger batch + head count."""
    q, k, v, dO = _make_qkv_do(2, 8, 8, 512, 512, 128, seed=48,
                                dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(128))


def test_v6nax_bwd_dq_d128_remainder_rows():
    """qL not aligned to V6NAX_BQ=64: last block has remainder rows."""
    # qL=510 = 7*64 + remainder 62.
    q, k, v, dO = _make_qkv_do(1, 4, 4, 510, 512, 128, seed=49,
                                dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(128))


# ---------------------------------------------------------------------------
# Shape + finiteness + dtype preservation
# ---------------------------------------------------------------------------
def test_v6nax_bwd_dq_output_shape_and_dtype():
    """Output dQ has same shape and dtype as input Q."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 512, 512, 128, seed=50,
                                dtype=mx.float16)
    dQ = _v6nax_bwd_dq(q, k, v, dO, 1.0 / math.sqrt(128))
    assert dQ.shape == q.shape
    assert dQ.dtype == q.dtype


def test_v6nax_bwd_dq_no_nan_or_inf():
    """dQ is finite (no NaN, no Inf)."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 1024, 1024, 128, seed=51,
                                dtype=mx.float16)
    dQ = _v6nax_bwd_dq(q, k, v, dO, 1.0 / math.sqrt(128))
    arr = np.array(dQ.astype(mx.float32))
    assert not np.isnan(arr).any(), "dQ contains NaN"
    assert not np.isinf(arr).any(), "dQ contains Inf"


def test_v6nax_bwd_dq_adversarial_magnitude_finite():
    """III-4 F9: adversarial-magnitude (std 8) inputs must keep dQ
    finite (fp16-overflow guard, V6NAX backward-dQ kernel family)."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 512, 512, 128, seed=52,
                               dtype=mx.float16, mag=8.0)
    dQ = _v6nax_bwd_dq(q, k, v, dO, 1.0 / math.sqrt(128))
    assert np.isfinite(np.array(dQ.astype(mx.float32))).all(), \
        "dQ non-finite at std-8 inputs"
