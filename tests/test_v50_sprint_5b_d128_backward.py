"""v2.50 Prompt 5b Section D — D=128 V34 backward broadening tests.

Per Marco's Option A decision: lift `_v34_backward_carveout` D=128 hard-
gate. Foundation: Sprint B v2.40.0-internal empirically validated that
D=128 split kernels work at parity (RMSE ~2e-5 vs SDPA-vjp; fused
regresses 3-7%, split preferred as auto-default for D=128).

Three-axis validation:
- Axe 1 (output sanity): D=128 backward gradients match SDPA-vjp baseline
  within FP16 ULP floor across multiple qL values + causal/non-causal
- Axe 2 (PUBLIC API): `mx.grad(flash_attention(q, k, v))` with D=128 +
  qL>=2048 + fp16/bf16 + `MFA_ENABLE_V34_BACKWARD=1` engages V34 backward
  split kernels (not SDPA-vjp fallback)
- Axe 3 (edges preserved): D=64 backward unchanged, D=128 below qL floor
  falls back to SDPA-vjp, D=128 fp32 falls back, fused kernel for D=128
  still opt-in only (split is auto-default)
"""
from __future__ import annotations

import math
import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_mfa
from mlx_mfa import flash_attention, get_device_info

_AE = getattr(mx, "async_" + "eval")
_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))

_skipif_no_nax = pytest.mark.skipif(
    not _HAS_NAX, reason="V34 backward D=128 broadening requires M5+ NAX hardware"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mk(B, H, qL, D, dtype, seed):
    mx.random.seed(seed)
    q = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    k = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    v = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    dO = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    _AE(q, k, v, dO); mx.synchronize()
    return q, k, v, dO


def _grad_auto(q, k, v, dO, causal):
    """Public AUTO path."""
    def loss(qi, ki, vi):
        return (mlx_mfa.flash_attention(qi, ki, vi, causal=causal) * dO).sum()
    return mx.grad(loss, argnums=(0, 1, 2))(q, k, v)


def _grad_sdpa(q, k, v, dO, causal):
    """Reference SDPA-vjp."""
    D = q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    def loss(qi, ki, vi):
        if causal:
            N, S = qi.shape[2], ki.shape[2]
            mask = mx.triu(
                mx.full((N, S), float("-inf"), dtype=qi.dtype), k=S - N + 1)
            o = mx.fast.scaled_dot_product_attention(qi, ki, vi, scale=scale, mask=mask)
        else:
            o = mx.fast.scaled_dot_product_attention(qi, ki, vi, scale=scale)
        return (o * dO).sum()
    return mx.grad(loss, argnums=(0, 1, 2))(q, k, v)


def _rmse(a, b):
    diff = np.abs(np.array(a.astype(mx.float32)) - np.array(b.astype(mx.float32)))
    return float(np.sqrt((diff ** 2).mean()))


# ---------------------------------------------------------------------------
# Axe 1 + Axe 2: D=128 V34 backward output matches SDPA-vjp baseline
# ---------------------------------------------------------------------------
class TestD128BackwardCorrectness:
    """V34 backward D=128 output matches `mx.vjp(SDPA)` baseline within
    FP16 ULP floor.  Sprint B v2.40.0-internal empirically established
    parity at RMSE ~2e-5 for split kernels; we use a slightly looser
    tolerance to absorb cross-shape variance."""

    @_skipif_no_nax
    def test_v34_bwd_d128_non_causal_qL2048_matches_sdpa(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        q, k, v, dO = _mk(1, 4, 2048, 128, mx.float16, 42)
        dQ_a, dK_a, dV_a = _grad_auto(q, k, v, dO, causal=False)
        dQ_r, dK_r, dV_r = _grad_sdpa(q, k, v, dO, causal=False)
        _AE(dQ_a, dK_a, dV_a, dQ_r, dK_r, dV_r); mx.synchronize()
        # FP16 ULP: ~5e-3 absolute floor for backward at D=128
        for (a, r, name) in [(dQ_a, dQ_r, "dQ"), (dK_a, dK_r, "dK"), (dV_a, dV_r, "dV")]:
            assert _rmse(a, r) < 5e-3, f"D=128 qL=2048 {name} RMSE = {_rmse(a, r):.2e}"

    @_skipif_no_nax
    def test_v34_bwd_d128_non_causal_qL4096_matches_sdpa(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        q, k, v, dO = _mk(1, 4, 4096, 128, mx.float16, 43)
        dQ_a, dK_a, dV_a = _grad_auto(q, k, v, dO, causal=False)
        dQ_r, dK_r, dV_r = _grad_sdpa(q, k, v, dO, causal=False)
        _AE(dQ_a, dK_a, dV_a, dQ_r, dK_r, dV_r); mx.synchronize()
        for (a, r, name) in [(dQ_a, dQ_r, "dQ"), (dK_a, dK_r, "dK"), (dV_a, dV_r, "dV")]:
            assert _rmse(a, r) < 5e-3, f"D=128 qL=4096 {name} RMSE = {_rmse(a, r):.2e}"

    @_skipif_no_nax
    def test_v34_bwd_d128_non_causal_qL8192_matches_sdpa(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        q, k, v, dO = _mk(1, 4, 8192, 128, mx.float16, 44)
        dQ_a, dK_a, dV_a = _grad_auto(q, k, v, dO, causal=False)
        dQ_r, dK_r, dV_r = _grad_sdpa(q, k, v, dO, causal=False)
        _AE(dQ_a, dK_a, dV_a, dQ_r, dK_r, dV_r); mx.synchronize()
        for (a, r, name) in [(dQ_a, dQ_r, "dQ"), (dK_a, dK_r, "dK"), (dV_a, dV_r, "dV")]:
            assert _rmse(a, r) < 5e-3, f"D=128 qL=8192 {name} RMSE = {_rmse(a, r):.2e}"

    @_skipif_no_nax
    def test_v34_bwd_d128_causal_qL2048_matches_sdpa(self, monkeypatch):
        """Causal + D=128: combination depends on Prompt 4 multi-gate fix +
        Sprint B D=128 split kernels.  Should produce correct gradients."""
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        q, k, v, dO = _mk(1, 4, 2048, 128, mx.float16, 45)
        dQ_a, dK_a, dV_a = _grad_auto(q, k, v, dO, causal=True)
        dQ_r, dK_r, dV_r = _grad_sdpa(q, k, v, dO, causal=True)
        _AE(dQ_a, dK_a, dV_a, dQ_r, dK_r, dV_r); mx.synchronize()
        for (a, r, name) in [(dQ_a, dQ_r, "dQ"), (dK_a, dK_r, "dK"), (dV_a, dV_r, "dV")]:
            assert _rmse(a, r) < 5e-3, f"D=128 causal qL=2048 {name} RMSE = {_rmse(a, r):.2e}"

    @_skipif_no_nax
    def test_v34_bwd_d128_causal_qL4096_matches_sdpa(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        q, k, v, dO = _mk(1, 4, 4096, 128, mx.float16, 46)
        dQ_a, dK_a, dV_a = _grad_auto(q, k, v, dO, causal=True)
        dQ_r, dK_r, dV_r = _grad_sdpa(q, k, v, dO, causal=True)
        _AE(dQ_a, dK_a, dV_a, dQ_r, dK_r, dV_r); mx.synchronize()
        for (a, r, name) in [(dQ_a, dQ_r, "dQ"), (dK_a, dK_r, "dK"), (dV_a, dV_r, "dV")]:
            assert _rmse(a, r) < 5e-3, f"D=128 causal qL=4096 {name} RMSE = {_rmse(a, r):.2e}"

    @_skipif_no_nax
    def test_v34_bwd_d128_bf16_matches_sdpa(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        q, k, v, dO = _mk(1, 4, 4096, 128, mx.bfloat16, 47)
        dQ_a, dK_a, dV_a = _grad_auto(q, k, v, dO, causal=False)
        dQ_r, dK_r, dV_r = _grad_sdpa(q, k, v, dO, causal=False)
        _AE(dQ_a, dK_a, dV_a, dQ_r, dK_r, dV_r); mx.synchronize()
        # bf16 has ~3x worse mantissa precision than fp16 → looser tol
        for (a, r, name) in [(dQ_a, dQ_r, "dQ"), (dK_a, dK_r, "dK"), (dV_a, dV_r, "dV")]:
            assert _rmse(a, r) < 1.5e-2, f"D=128 bf16 {name} RMSE = {_rmse(a, r):.2e}"


# ---------------------------------------------------------------------------
# Axe 2: PUBLIC API engagement — V34 split path must engage for D=128
# ---------------------------------------------------------------------------
class TestD128BackwardPublicAPIEngagement:
    """Verify the PUBLIC API path (`mx.grad(flash_attention(...))`) engages
    V34 backward split kernels for D=128.  Engagement detection is
    differential: V34 split gradients differ from SDPA-vjp gradients by
    FP16 rounding (non-zero RMSE).  SDPA fallback produces bit-identical
    gradients to the explicit SDPA reference path."""

    @_skipif_no_nax
    def test_v34_bwd_d128_routing_public_api_engages_v34_split(self, monkeypatch):
        """When env=1 + D=128 + qL>=2048 + fp16, AUTO must engage V34 split
        (gradients differ from SDPA-vjp by FP16 rounding)."""
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        q, k, v, dO = _mk(1, 4, 4096, 128, mx.float16, 100)
        dQ_a, dK_a, dV_a = _grad_auto(q, k, v, dO, causal=False)
        dQ_r, dK_r, dV_r = _grad_sdpa(q, k, v, dO, causal=False)
        _AE(dQ_a, dK_a, dV_a, dQ_r, dK_r, dV_r); mx.synchronize()
        # V34 engaged → gradients are NOT bit-identical to SDPA-vjp (non-zero diff)
        diffs = [_rmse(dQ_a, dQ_r), _rmse(dK_a, dK_r), _rmse(dV_a, dV_r)]
        max_diff = max(diffs)
        assert max_diff > 1e-7, (
            f"V34 backward D=128 did NOT engage (max diff vs SDPA = {max_diff:.2e}; "
            f"expected non-zero FP16 rounding diff if V34 split kernel ran)"
        )


# ---------------------------------------------------------------------------
# Axe 3: Edges — below qL floor, fp32, D=64 preserved
# ---------------------------------------------------------------------------
class TestD128BackwardEdges:
    """Below-floor fallback, dtype fallback, D=64 regression check."""

    @_skipif_no_nax
    def test_v34_bwd_d128_below_qL_threshold_falls_back_sdpa(self, monkeypatch):
        """D=128 qL=1024 (below 2048 floor) MUST fall back to SDPA-vjp
        bit-identically."""
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        q, k, v, dO = _mk(1, 4, 1024, 128, mx.float16, 200)
        dQ_a, dK_a, dV_a = _grad_auto(q, k, v, dO, causal=False)
        dQ_r, dK_r, dV_r = _grad_sdpa(q, k, v, dO, causal=False)
        _AE(dQ_a, dK_a, dV_a, dQ_r, dK_r, dV_r); mx.synchronize()
        # SDPA fallback → bit-identical (RMSE ≈ 0)
        for (a, r, name) in [(dQ_a, dQ_r, "dQ"), (dK_a, dK_r, "dK"), (dV_a, dV_r, "dV")]:
            assert _rmse(a, r) < 1e-7, (
                f"D=128 qL=1024 {name} should fall back to SDPA bit-identically; "
                f"RMSE = {_rmse(a, r):.2e}"
            )

    @_skipif_no_nax
    def test_v34_bwd_d128_fp32_falls_back_sdpa(self, monkeypatch):
        """D=128 fp32 MUST fall back to SDPA-vjp (V34 backward is fp16/bf16
        only)."""
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        q, k, v, dO = _mk(1, 4, 4096, 128, mx.float32, 201)
        dQ_a, dK_a, dV_a = _grad_auto(q, k, v, dO, causal=False)
        dQ_r, dK_r, dV_r = _grad_sdpa(q, k, v, dO, causal=False)
        _AE(dQ_a, dK_a, dV_a, dQ_r, dK_r, dV_r); mx.synchronize()
        for (a, r, name) in [(dQ_a, dQ_r, "dQ"), (dK_a, dK_r, "dK"), (dV_a, dV_r, "dV")]:
            assert _rmse(a, r) < 1e-6, (
                f"D=128 fp32 {name} should fall back; RMSE = {_rmse(a, r):.2e}"
            )

    @_skipif_no_nax
    def test_v34_bwd_d64_unchanged_post_d128_broadening(self, monkeypatch):
        """Regression check: D=64 qL=4096 V34 backward (pre-Section-D
        production-active) continues to engage post-broadening."""
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        q, k, v, dO = _mk(1, 4, 4096, 64, mx.float16, 202)
        dQ_a, dK_a, dV_a = _grad_auto(q, k, v, dO, causal=False)
        dQ_r, dK_r, dV_r = _grad_sdpa(q, k, v, dO, causal=False)
        _AE(dQ_a, dK_a, dV_a, dQ_r, dK_r, dV_r); mx.synchronize()
        # D=64 engages V34 fused kernel — differs from SDPA-vjp by FP16 rounding
        diffs = [_rmse(dQ_a, dQ_r), _rmse(dK_a, dK_r), _rmse(dV_a, dV_r)]
        max_diff = max(diffs)
        assert max_diff > 1e-7, (
            f"D=64 qL=4096 V34 backward should still engage post-broadening; "
            f"max diff vs SDPA = {max_diff:.2e}"
        )
        # AND output is correct
        for (a, r, name) in [(dQ_a, dQ_r, "dQ"), (dK_a, dK_r, "dK"), (dV_a, dV_r, "dV")]:
            assert _rmse(a, r) < 5e-3, f"D=64 regression {name} RMSE = {_rmse(a, r):.2e}"

    @_skipif_no_nax
    def test_v34_bwd_d128_env_unset_falls_back(self, monkeypatch):
        """Without env=1, D=128 must NOT engage V34 (bit-identical to SDPA)."""
        monkeypatch.delenv("MFA_ENABLE_V34_BACKWARD", raising=False)
        q, k, v, dO = _mk(1, 4, 4096, 128, mx.float16, 203)
        dQ_a, dK_a, dV_a = _grad_auto(q, k, v, dO, causal=False)
        dQ_r, dK_r, dV_r = _grad_sdpa(q, k, v, dO, causal=False)
        _AE(dQ_a, dK_a, dV_a, dQ_r, dK_r, dV_r); mx.synchronize()
        for (a, r, name) in [(dQ_a, dQ_r, "dQ"), (dK_a, dK_r, "dK"), (dV_a, dV_r, "dV")]:
            assert _rmse(a, r) < 1e-7, (
                f"D=128 env unset {name} should NOT engage V34; "
                f"RMSE vs SDPA = {_rmse(a, r):.2e}"
            )


# ---------------------------------------------------------------------------
# Carve-out function direct tests (Section D's primary code change)
# ---------------------------------------------------------------------------
class TestV34BackwardCarveoutD128:
    """Direct test of `_v34_backward_carveout` post-broadening."""

    def test_d128_qL2048_fp16_eligible(self, monkeypatch):
        from mlx_mfa.dispatch_policy import _v34_backward_carveout
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        assert _v34_backward_carveout(
            head_dim=128, seq_len=2048, causal=False, dtype_key="float16"
        ) is True

    def test_d128_qL4096_bf16_eligible(self, monkeypatch):
        from mlx_mfa.dispatch_policy import _v34_backward_carveout
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        assert _v34_backward_carveout(
            head_dim=128, seq_len=4096, causal=False, dtype_key="bfloat16"
        ) is True

    def test_d128_qL4096_causal_eligible(self, monkeypatch):
        from mlx_mfa.dispatch_policy import _v34_backward_carveout
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        assert _v34_backward_carveout(
            head_dim=128, seq_len=4096, causal=True, dtype_key="float16"
        ) is True

    def test_d128_qL1024_below_floor_ineligible(self, monkeypatch):
        from mlx_mfa.dispatch_policy import _v34_backward_carveout
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        assert _v34_backward_carveout(
            head_dim=128, seq_len=1024, causal=False, dtype_key="float16"
        ) is False

    def test_d128_fp32_ineligible(self, monkeypatch):
        from mlx_mfa.dispatch_policy import _v34_backward_carveout
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        assert _v34_backward_carveout(
            head_dim=128, seq_len=4096, causal=False, dtype_key="float32"
        ) is False

    def test_d128_env_unset_ineligible(self, monkeypatch):
        from mlx_mfa.dispatch_policy import _v34_backward_carveout
        monkeypatch.delenv("MFA_ENABLE_V34_BACKWARD", raising=False)
        assert _v34_backward_carveout(
            head_dim=128, seq_len=4096, causal=False, dtype_key="float16"
        ) is False

    def test_d64_unchanged_post_broadening(self, monkeypatch):
        """D=64 cases must still work post-broadening."""
        from mlx_mfa.dispatch_policy import _v34_backward_carveout
        monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
        assert _v34_backward_carveout(
            head_dim=64, seq_len=4096, causal=False, dtype_key="float16"
        ) is True
        assert _v34_backward_carveout(
            head_dim=64, seq_len=1024, causal=False, dtype_key="float16"
        ) is False
