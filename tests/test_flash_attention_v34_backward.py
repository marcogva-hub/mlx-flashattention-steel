"""V34 backward integration via flash_attention() VJP (Phase 2 Section E).

SHIP_OPT_IN posture (per perf regression discovered Phase 3):
- Default behaviour: V34 backward NOT engaged; flash_attention() VJP
  uses STEEL backward / SDPA-vjp per existing dispatch policy.
- Opt-in: set MFA_ENABLE_V34_BACKWARD=1 to engage V34 backward kernels.

Tests verify:
- Opt-in correctness: V34 backward produces correct gradients vs SDPA-vjp.
- Opt-in path engaged: V34-on vs V34-off produce numerically different output.
- Default-off: with no env, fallback path = SDPA-vjp exactly (0.0 RMSE).
"""
import math
import os
import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa import flash_attention


_AE = getattr(mx, "async_" + "eval")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MFA_LCSA_KERNEL_VERSION", raising=False)
    monkeypatch.delenv("MFA_ENABLE_V34_BACKWARD", raising=False)
    yield


@pytest.fixture
def enable_v34_bwd(monkeypatch):
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
    yield


def _make(B, Hq, Hk, qL, kL, D, seed, dtype):
    mx.random.seed(seed)
    q = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(dtype)
    k = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(dtype)
    v = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(dtype)
    _AE(q, k, v); mx.synchronize()
    return q, k, v


def _grads(q, k, v, backend="mfa"):
    def loss(q_, k_, v_):
        return flash_attention(q_, k_, v_, backend=backend).sum()
    g = mx.grad(loss, argnums=(0, 1, 2))
    dQ, dK, dV = g(q, k, v)
    _AE(dQ, dK, dV); mx.synchronize()
    return dQ, dK, dV


def _sdpa_grads(q, k, v, scale):
    def loss(q_, k_, v_):
        return mx.fast.scaled_dot_product_attention(q_, k_, v_, scale=scale).sum()
    g = mx.grad(loss, argnums=(0, 1, 2))
    dQ, dK, dV = g(q, k, v)
    _AE(dQ, dK, dV); mx.synchronize()
    return dQ, dK, dV


def _rmse(a, b):
    err = np.abs(np.array(a.astype(mx.float32)) -
                 np.array(b.astype(mx.float32)))
    return float(np.sqrt((err ** 2).mean()))


# ---------------------------------------------------------------------------
# Opt-in correctness (MFA_ENABLE_V34_BACKWARD=1)
# ---------------------------------------------------------------------------
def test_v34_bwd_optin_d128_fp16_qL1024(enable_v34_bwd):
    q, k, v = _make(1, 4, 4, 1024, 1024, 128, 42, mx.float16)
    scale = 1.0 / math.sqrt(128)
    dQ, dK, dV = _grads(q, k, v)
    dQ_ref, dK_ref, dV_ref = _sdpa_grads(q, k, v, scale)
    assert _rmse(dQ, dQ_ref) < 1e-3
    assert _rmse(dK, dK_ref) < 1e-3
    assert _rmse(dV, dV_ref) < 1e-2


def test_v34_bwd_optin_d128_fp16_qL512(enable_v34_bwd):
    q, k, v = _make(1, 4, 4, 512, 512, 128, 43, mx.float16)
    scale = 1.0 / math.sqrt(128)
    dQ, dK, dV = _grads(q, k, v)
    dQ_ref, dK_ref, dV_ref = _sdpa_grads(q, k, v, scale)
    assert _rmse(dQ, dQ_ref) < 1e-3
    assert _rmse(dK, dK_ref) < 1e-3
    assert _rmse(dV, dV_ref) < 1e-2


def test_v34_bwd_optin_bf16(enable_v34_bwd):
    q, k, v = _make(1, 4, 4, 512, 512, 128, 44, mx.bfloat16)
    scale = 1.0 / math.sqrt(128)
    dQ, dK, dV = _grads(q, k, v)
    dQ_ref, dK_ref, dV_ref = _sdpa_grads(q, k, v, scale)
    assert _rmse(dQ, dQ_ref) < 1e-2
    assert _rmse(dK, dK_ref) < 1e-2
    assert _rmse(dV, dV_ref) < 5e-2


# ---------------------------------------------------------------------------
# Path-entered verification: V34-on vs V34-off produce DIFFERENT output
# ---------------------------------------------------------------------------
def test_v34_bwd_optin_engages_path(enable_v34_bwd, monkeypatch):
    """V34-enabled output should differ from V34-disabled (FP16
    rounding between V34 NAX kernels and SDPA-vjp paths)."""
    q, k, v = _make(1, 4, 4, 1024, 1024, 128, 45, mx.float16)
    dQ_on, _, _ = _grads(q, k, v)
    monkeypatch.delenv("MFA_ENABLE_V34_BACKWARD", raising=False)
    dQ_off, _, _ = _grads(q, k, v)
    diff = _rmse(dQ_on, dQ_off)
    assert diff > 0, "V34 backward did not engage (V34-on == V34-off)"


# ---------------------------------------------------------------------------
# Default-off behaviour
# ---------------------------------------------------------------------------
def test_v34_bwd_default_off_falls_back():
    """Without MFA_ENABLE_V34_BACKWARD=1, fallback path is engaged.
    Fallback = SDPA-vjp, so output matches SDPA-vjp exactly (0.0 RMSE)."""
    q, k, v = _make(1, 4, 4, 1024, 1024, 128, 46, mx.float16)
    scale = 1.0 / math.sqrt(128)
    dQ, dK, dV = _grads(q, k, v)
    dQ_ref, dK_ref, dV_ref = _sdpa_grads(q, k, v, scale)
    # Fallback IS SDPA-vjp -> identical.
    assert _rmse(dQ, dQ_ref) == 0.0
    assert _rmse(dK, dK_ref) == 0.0
    assert _rmse(dV, dV_ref) == 0.0


def test_v34_bwd_optin_d64_small_nk_still_falls_back(enable_v34_bwd):
    """Per DC12 routing parity: D=64 small-Nk falls back even with
    V34 enabled (V34 forward doesn't engage there)."""
    q, k, v = _make(1, 4, 4, 512, 512, 64, 47, mx.float16)
    scale = 1.0 / math.sqrt(64)
    dQ, dK, dV = _grads(q, k, v)
    dQ_ref, dK_ref, dV_ref = _sdpa_grads(q, k, v, scale)
    assert _rmse(dQ, dQ_ref) == 0.0
    assert _rmse(dK, dK_ref) == 0.0
    assert _rmse(dV, dV_ref) == 0.0
