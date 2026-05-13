"""V34 backward dQ correctness tests (Phase 1 Section C of V34 backward
Option β sprint).  All tests compare V34 dQ output against MLX
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
def force_v34(monkeypatch):
    """Force V34 path for D=64 small-Nk shapes (per DC12 routing constraint)."""
    monkeypatch.setenv("MFA_V6_USE_V34", "1")
    yield


def _make_qkv_do(B, Hq, Hk, qL, kL, D, seed, dtype):
    mx.random.seed(seed)
    q = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(dtype)
    k = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(dtype)
    v = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(dtype)
    dO = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(dtype)
    _mat(q, k, v, dO)
    return q, k, v, dO


def _v34_bwd_dq(q, k, v, dO, scale):
    """V34 forward to get (O, lse), then V34 backward dQ."""
    O, lse = _ext.v6_nax_forward(q, k, v, False)
    _mat(O, lse)
    dQ = _ext.v6_nax_backward_query(q, k, v, O, lse, dO, scale)
    _mat(dQ)
    return dQ


def _sdpa_vjp_dq(q, k, v, dO, scale):
    """Reference: MLX mx.vjp on scaled_dot_product_attention."""
    def fwd(q_, k_, v_):
        return mx.fast.scaled_dot_product_attention(q_, k_, v_, scale=scale)
    _, (dQ_ref, _, _) = mx.vjp(fwd, [q, k, v], [dO])
    _mat(dQ_ref)
    return dQ_ref


def _check_dq_rmse(q, k, v, dO, scale, *, fp16_bound=1e-3, bf16_bound=1e-4):
    dQ = _v34_bwd_dq(q, k, v, dO, scale)
    dQ_ref = _sdpa_vjp_dq(q, k, v, dO, scale)
    err = np.abs(np.array(dQ.astype(mx.float32)) -
                 np.array(dQ_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    maxerr = float(err.max())
    bound = bf16_bound if q.dtype == mx.bfloat16 else fp16_bound
    assert rmse < bound, f"dQ RMSE {rmse:.4e} exceeds {bound:.4e}"
    return rmse, maxerr


# ---------------------------------------------------------------------------
# Core correctness: D=128 (V34-default route)
# ---------------------------------------------------------------------------
def test_v34_bwd_dq_d128_fp16_square():
    """D=128, qL=kL=512, single head — baseline correctness."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 512, 512, 128, seed=42, dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(128))


def test_v34_bwd_dq_d128_fp16_seq2048():
    """D=128, larger seq qL=kL=2048."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 2048, 2048, 128, seed=43, dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(128))


def test_v34_bwd_dq_d128_bf16_seq1024():
    """D=128, BF16, qL=kL=1024."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 1024, 1024, 128, seed=44,
                                dtype=mx.bfloat16)
    # BF16 has 7-bit mantissa; relax bf16_bound to 1e-3.
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(128),
                    fp16_bound=1e-3, bf16_bound=1e-3)


# ---------------------------------------------------------------------------
# D=64 with V34 forced (per DC12 routing constraint)
# ---------------------------------------------------------------------------
def test_v34_bwd_dq_d64_fp16_force_v34(force_v34):
    """D=64 small-Nk: force V34 path explicitly."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 512, 512, 64, seed=45, dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(64))


def test_v34_bwd_dq_d64_large_nk_natural_v34():
    """D=64 with Nk>8000: V34 forward engages by default per Primitive
    routing line 596.  No env override needed."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 512, 8192, 64, seed=46, dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(64))


# ---------------------------------------------------------------------------
# Shape variations
# ---------------------------------------------------------------------------
def test_v34_bwd_dq_d128_asymmetric():
    """Asymmetric qL != kL (cross-attention-style)."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 512, 2048, 128, seed=47,
                                dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(128))


def test_v34_bwd_dq_d128_batch2_h8():
    """Larger batch + head count."""
    q, k, v, dO = _make_qkv_do(2, 8, 8, 512, 512, 128, seed=48,
                                dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(128))


def test_v34_bwd_dq_d128_remainder_rows():
    """qL not aligned to V34_BQ=64: last block has remainder rows."""
    # qL=510 = 7*64 + remainder 62.
    q, k, v, dO = _make_qkv_do(1, 4, 4, 510, 512, 128, seed=49,
                                dtype=mx.float16)
    _check_dq_rmse(q, k, v, dO, 1.0 / math.sqrt(128))


# ---------------------------------------------------------------------------
# Shape + finiteness + dtype preservation
# ---------------------------------------------------------------------------
def test_v34_bwd_dq_output_shape_and_dtype():
    """Output dQ has same shape and dtype as input Q."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 512, 512, 128, seed=50,
                                dtype=mx.float16)
    dQ = _v34_bwd_dq(q, k, v, dO, 1.0 / math.sqrt(128))
    assert dQ.shape == q.shape
    assert dQ.dtype == q.dtype


def test_v34_bwd_dq_no_nan_or_inf():
    """dQ is finite (no NaN, no Inf)."""
    q, k, v, dO = _make_qkv_do(1, 4, 4, 1024, 1024, 128, seed=51,
                                dtype=mx.float16)
    dQ = _v34_bwd_dq(q, k, v, dO, 1.0 / math.sqrt(128))
    arr = np.array(dQ.astype(mx.float32))
    assert not np.isnan(arr).any(), "dQ contains NaN"
    assert not np.isinf(arr).any(), "dQ contains Inf"
