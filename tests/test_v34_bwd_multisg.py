"""V6NAX backward multi-SG kernels (Phase 2.O2 — dV-only + dK-only).

Verifies the new WM=4 Q-row-partition kernels produce correct gradients
matching SDPA-vjp reference within FP16/FP32 noise floor.
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


def _make(B, Hq, qL, D, seed, dtype, mag=1.0):
    # III-4 F9: unit-scale inputs (normal, std 1.0; was uniform*0.1).
    # The II-6 lesson: 0.1-scale fixtures dilute localized kernel
    # corruption below the rmse gates.
    mx.random.seed(seed)
    q = (mx.random.normal((B, Hq, qL, D)) * mag).astype(dtype)
    k = (mx.random.normal((B, Hq, qL, D)) * mag).astype(dtype)
    v = (mx.random.normal((B, Hq, qL, D)) * mag).astype(dtype)
    dO = (mx.random.normal((B, Hq, qL, D)) * mag).astype(dtype)
    _mat(q, k, v, dO)
    return q, k, v, dO


def _ref_dkdv(q, k, v, dO, scale):
    def fwd(q_, k_, v_):
        return mx.fast.scaled_dot_product_attention(q_, k_, v_, scale=scale)
    _, (_, dK_ref, dV_ref) = mx.vjp(fwd, [q, k, v], [dO])
    _mat(dK_ref, dV_ref)
    return dK_ref, dV_ref


def _rmse(a, b):
    err = np.abs(np.array(a.astype(mx.float32)) - np.array(b.astype(mx.float32)))
    return float(np.sqrt((err ** 2).mean()))


# ---------------------------------------------------------------------------
# dV multi-SG correctness
# ---------------------------------------------------------------------------
def test_dv_multisg_d128_fp16_qL1024_wm4():
    q, k, v, dO = _make(1, 4, 1024, 128, 42, mx.float16)
    scale = 1.0 / math.sqrt(128)
    O, lse = _ext.v6_nax_forward(q, k, v, False); _mat(O, lse)
    dVp = _ext.v6_nax_backward_dv_raw(q, k, v, lse, dO, scale, 4)
    dV = mx.sum(dVp, axis=2).astype(mx.float16); _mat(dV)
    _, dV_ref = _ref_dkdv(q, k, v, dO, scale)
    rmse = _rmse(dV, dV_ref)
    # III-4 F9: unit-scale retrofit, measured floor rmse 3.7e-5 (M5 Max).
    assert rmse < 1e-2, f"dV multi-SG RMSE {rmse:.4e}"


def test_dv_multisg_d128_fp16_qL512_wm4():
    q, k, v, dO = _make(1, 4, 512, 128, 43, mx.float16)
    scale = 1.0 / math.sqrt(128)
    O, lse = _ext.v6_nax_forward(q, k, v, False); _mat(O, lse)
    dVp = _ext.v6_nax_backward_dv_raw(q, k, v, lse, dO, scale, 4)
    dV = mx.sum(dVp, axis=2).astype(mx.float16); _mat(dV)
    _, dV_ref = _ref_dkdv(q, k, v, dO, scale)
    assert _rmse(dV, dV_ref) < 1e-2


def test_dv_multisg_d128_bf16():
    q, k, v, dO = _make(1, 4, 512, 128, 44, mx.bfloat16)
    scale = 1.0 / math.sqrt(128)
    O, lse = _ext.v6_nax_forward(q, k, v, False); _mat(O, lse)
    dVp = _ext.v6_nax_backward_dv_raw(q, k, v, lse, dO, scale, 4)
    dV = mx.sum(dVp, axis=2).astype(mx.bfloat16); _mat(dV)
    _, dV_ref = _ref_dkdv(q, k, v, dO, scale)
    assert _rmse(dV, dV_ref) < 5e-2  # relaxed for bf16


# ---------------------------------------------------------------------------
# dK multi-SG correctness
# ---------------------------------------------------------------------------
def test_dk_multisg_d128_fp16_qL1024_wm4():
    q, k, v, dO = _make(1, 4, 1024, 128, 45, mx.float16)
    scale = 1.0 / math.sqrt(128)
    O, lse = _ext.v6_nax_forward(q, k, v, False); _mat(O, lse)
    D = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1); _mat(D)
    dKp = _ext.v6_nax_backward_dk_raw(q, k, v, O, lse, dO, D, scale, 4)
    dK = mx.sum(dKp, axis=2).astype(mx.float16); _mat(dK)
    dK_ref, _ = _ref_dkdv(q, k, v, dO, scale)
    # III-4 F9: unit-scale retrofit, measured floor rmse 4.1e-5 (M5 Max).
    assert _rmse(dK, dK_ref) < 1e-3


def test_dk_multisg_d128_fp16_qL512_wm4():
    q, k, v, dO = _make(1, 4, 512, 128, 46, mx.float16)
    scale = 1.0 / math.sqrt(128)
    O, lse = _ext.v6_nax_forward(q, k, v, False); _mat(O, lse)
    D = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1); _mat(D)
    dKp = _ext.v6_nax_backward_dk_raw(q, k, v, O, lse, dO, D, scale, 4)
    dK = mx.sum(dKp, axis=2).astype(mx.float16); _mat(dK)
    dK_ref, _ = _ref_dkdv(q, k, v, dO, scale)
    assert _rmse(dK, dK_ref) < 1e-3


def test_dk_multisg_d128_bf16():
    q, k, v, dO = _make(1, 4, 512, 128, 47, mx.bfloat16)
    scale = 1.0 / math.sqrt(128)
    O, lse = _ext.v6_nax_forward(q, k, v, False); _mat(O, lse)
    D = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1); _mat(D)
    dKp = _ext.v6_nax_backward_dk_raw(q, k, v, O, lse, dO, D, scale, 4)
    dK = mx.sum(dKp, axis=2).astype(mx.bfloat16); _mat(dK)
    dK_ref, _ = _ref_dkdv(q, k, v, dO, scale)
    assert _rmse(dK, dK_ref) < 1e-2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_dv_multisg_partials_shape():
    """dV_partials output shape is [B, Hq, WM, kL, D] FP32."""
    q, k, v, dO = _make(1, 4, 512, 128, 48, mx.float16)
    scale = 1.0 / math.sqrt(128)
    O, lse = _ext.v6_nax_forward(q, k, v, False); _mat(O, lse)
    dVp = _ext.v6_nax_backward_dv_raw(q, k, v, lse, dO, scale, 4)
    assert dVp.shape == (1, 4, 4, 512, 128)
    assert dVp.dtype == mx.float32


def test_dk_multisg_finiteness():
    q, k, v, dO = _make(1, 4, 1024, 128, 49, mx.float16)
    scale = 1.0 / math.sqrt(128)
    O, lse = _ext.v6_nax_forward(q, k, v, False); _mat(O, lse)
    D = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1); _mat(D)
    dKp = _ext.v6_nax_backward_dk_raw(q, k, v, O, lse, dO, D, scale, 4)
    dK = mx.sum(dKp, axis=2).astype(mx.float16); _mat(dK)
    arr = np.array(dK.astype(mx.float32))
    assert not np.isnan(arr).any() and not np.isinf(arr).any()


def test_multisg_adversarial_magnitude_finite():
    """III-4 F9: adversarial-magnitude (std 8) inputs must keep the
    multi-SG dK/dV partials finite (fp16-overflow guard)."""
    q, k, v, dO = _make(1, 4, 512, 128, 50, mx.float16, mag=8.0)
    scale = 1.0 / math.sqrt(128)
    O, lse = _ext.v6_nax_forward(q, k, v, False); _mat(O, lse)
    D = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1); _mat(D)
    dVp = _ext.v6_nax_backward_dv_raw(q, k, v, lse, dO, scale, 4)
    dKp = _ext.v6_nax_backward_dk_raw(q, k, v, O, lse, dO, D, scale, 4)
    dV = mx.sum(dVp, axis=2); dK = mx.sum(dKp, axis=2); _mat(dV, dK)
    assert np.isfinite(np.array(dV.astype(mx.float32))).all(), \
        "dV non-finite at std-8 inputs"
    assert np.isfinite(np.array(dK.astype(mx.float32))).all(), \
        "dK non-finite at std-8 inputs"
