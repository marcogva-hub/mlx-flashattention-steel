"""V34 backward dK/dV correctness tests (Phase 2 Section C).

Compares V34 dK/dV against MLX mx.vjp(scaled_dot_product_attention) reference
within FP32 (dK) / FP16-rounding (dV) accumulation floor.
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
    monkeypatch.setenv("MFA_V6_USE_V34", "1")
    yield


def _make(B, Hq, Hk, qL, kL, D, seed, dtype):
    mx.random.seed(seed)
    q = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(dtype)
    k = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(dtype)
    v = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(dtype)
    dO = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(dtype)
    _mat(q, k, v, dO)
    return q, k, v, dO


def _bwd_kv(q, k, v, dO, scale):
    O, lse = _ext.v6_nax_forward(q, k, v, False)
    _mat(O, lse)
    # v2.38.1: D = rowsum(dO * O) precomputed
    D = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1)
    _mat(D)
    dK, dV = _ext.v6_nax_backward_kv(q, k, v, O, lse, dO, D, scale)
    _mat(dK, dV)
    return dK, dV


def _ref_dkv(q, k, v, dO, scale):
    def fwd(q_, k_, v_):
        return mx.fast.scaled_dot_product_attention(q_, k_, v_, scale=scale)
    _, (_, dK_ref, dV_ref) = mx.vjp(fwd, [q, k, v], [dO])
    _mat(dK_ref, dV_ref)
    return dK_ref, dV_ref


def _check(q, k, v, dO, scale, *, dk_bound=1e-4, dv_bound=1e-3):
    dK, dV = _bwd_kv(q, k, v, dO, scale)
    dK_ref, dV_ref = _ref_dkv(q, k, v, dO, scale)
    err_k = np.abs(np.array(dK.astype(mx.float32)) -
                   np.array(dK_ref.astype(mx.float32)))
    err_v = np.abs(np.array(dV.astype(mx.float32)) -
                   np.array(dV_ref.astype(mx.float32)))
    rmse_k = float(np.sqrt((err_k ** 2).mean()))
    rmse_v = float(np.sqrt((err_v ** 2).mean()))
    assert rmse_k < dk_bound, f"dK RMSE {rmse_k:.4e} exceeds {dk_bound:.4e}"
    assert rmse_v < dv_bound, f"dV RMSE {rmse_v:.4e} exceeds {dv_bound:.4e}"


def test_v34_bwd_kv_d128_fp16_square():
    q, k, v, dO = _make(1, 4, 4, 512, 512, 128, 42, mx.float16)
    _check(q, k, v, dO, 1.0 / math.sqrt(128))


def test_v34_bwd_kv_d128_fp16_seq1024():
    q, k, v, dO = _make(1, 4, 4, 1024, 1024, 128, 43, mx.float16)
    _check(q, k, v, dO, 1.0 / math.sqrt(128))


def test_v34_bwd_kv_d128_bf16():
    q, k, v, dO = _make(1, 4, 4, 512, 512, 128, 44, mx.bfloat16)
    # Looser bounds for bf16 (7-bit mantissa).
    _check(q, k, v, dO, 1.0 / math.sqrt(128), dk_bound=1e-3, dv_bound=2e-3)


def test_v34_bwd_kv_d64_force_v34(force_v34):
    q, k, v, dO = _make(1, 4, 4, 512, 512, 64, 45, mx.float16)
    _check(q, k, v, dO, 1.0 / math.sqrt(64))


def test_v34_bwd_kv_d128_asymmetric():
    q, k, v, dO = _make(1, 4, 4, 512, 2048, 128, 46, mx.float16)
    _check(q, k, v, dO, 1.0 / math.sqrt(128))


def test_v34_bwd_kv_d128_batch2_h8():
    q, k, v, dO = _make(2, 8, 8, 512, 512, 128, 47, mx.float16)
    _check(q, k, v, dO, 1.0 / math.sqrt(128))


def test_v34_bwd_kv_output_shapes():
    q, k, v, dO = _make(1, 4, 4, 512, 256, 128, 48, mx.float16)
    dK, dV = _bwd_kv(q, k, v, dO, 1.0 / math.sqrt(128))
    # dK/dV per Q-head: [B, Hq, kL, D] each.
    assert dK.shape == (1, 4, 256, 128), f"dK shape {dK.shape}"
    assert dV.shape == (1, 4, 256, 128), f"dV shape {dV.shape}"
    assert dK.dtype == mx.float16
    assert dV.dtype == mx.float16


def test_v34_bwd_kv_finiteness():
    q, k, v, dO = _make(1, 4, 4, 1024, 1024, 128, 49, mx.float16)
    dK, dV = _bwd_kv(q, k, v, dO, 1.0 / math.sqrt(128))
    arrK = np.array(dK.astype(mx.float32))
    arrV = np.array(dV.astype(mx.float32))
    assert not np.isnan(arrK).any(), "dK contains NaN"
    assert not np.isinf(arrK).any(), "dK contains Inf"
    assert not np.isnan(arrV).any(), "dV contains NaN"
    assert not np.isinf(arrV).any(), "dV contains Inf"
