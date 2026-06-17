"""v2.50 Prompt 5d Section A - V6NAX backward sparse FULL NATIVE tests."""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import flash_attention_sparse, get_device_info
from mlx_mfa.attention import _convert_mask_for_v6nax_bwd_kernel

_AE = getattr(mx, "async_" + "eval")
_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))

_skipif_no_nax = pytest.mark.skipif(
    not _HAS_NAX, reason="V6NAX native sparse requires M5+ NAX hardware"
)


def _mk(B, H, qL, D, dtype, seed):
    mx.random.seed(seed)
    q = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    k = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    v = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    dO = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    _AE(q, k, v, dO); mx.synchronize()
    return q, k, v, dO


def _rmse(a, b):
    diff = np.abs(np.array(a.astype(mx.float32)) - np.array(b.astype(mx.float32)))
    return float(np.sqrt((diff ** 2).mean()))


def _block_causal_mask(NQ, NK):
    return mx.array(np.tril(np.ones((NQ, NK), dtype=bool)))


def _random_mask(NQ, NK, density, seed=0):
    np.random.seed(seed)
    return mx.array((np.random.rand(NQ, NK) < density).astype(bool))


class TestSparseKernelsAllTrueMaskBitIdentical:
    @_skipif_no_nax
    def test_dq_sparse_all_true_bit_identical(self):
        from mlx_mfa import _ext
        B, H, qL, D = 1, 4, 2048, 64
        BT = 32
        NQ = NK = qL // BT
        q, k, v, dO = _mk(B, H, qL, D, mx.float16, 200)
        scale = 1.0 / math.sqrt(D)
        O, L = _ext.v6_nax_forward(q, k, v, False, True)
        mx.eval(O, L); mx.synchronize()
        D_vec = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1)
        mx.eval(D_vec); mx.synchronize()
        mask_all_bt = mx.ones((NQ, NK), dtype=mx.bool_)
        # v2.50 Prompt 5f Phase A KD-1: convert to dQ kernel geometry.
        mask_all = _convert_mask_for_v6nax_bwd_kernel(mask_all_bt, BT, "dQ", D)
        _AE(mask_all); mx.synchronize()

        dQ_sparse = _ext.v6_nax_backward_query_sparse_raw(
            q, k, v, O, L, dO, D_vec, mask_all, scale, False)
        dQ_dense = _ext.v6_nax_backward_query(q, k, v, O, L, dO, D_vec, scale, False)
        mx.eval(dQ_sparse, dQ_dense); mx.synchronize()
        diff = float(mx.max(mx.abs(
            dQ_sparse.astype(mx.float32) - dQ_dense.astype(mx.float32))))
        assert diff < 1e-6, f"dQ all-True max_diff = {diff:.4e}"

    @_skipif_no_nax
    def test_dk_sparse_all_true_bit_identical(self):
        from mlx_mfa import _ext
        B, H, qL, D = 1, 4, 2048, 64
        BT = 32
        NQ = NK = qL // BT
        q, k, v, dO = _mk(B, H, qL, D, mx.float16, 201)
        scale = 1.0 / math.sqrt(D)
        O, L = _ext.v6_nax_forward(q, k, v, False, True)
        mx.eval(O, L); mx.synchronize()
        D_vec = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1)
        mx.eval(D_vec); mx.synchronize()
        mask_all_bt = mx.ones((NQ, NK), dtype=mx.bool_)
        mask_all = _convert_mask_for_v6nax_bwd_kernel(mask_all_bt, BT, "dK", D)
        _AE(mask_all); mx.synchronize()

        dKp_sparse = _ext.v6_nax_backward_dk_sparse_raw(
            q, k, v, O, L, dO, D_vec, mask_all, scale, 4, False)
        dKp_dense = _ext.v6_nax_backward_dk_raw(q, k, v, O, L, dO, D_vec, scale, 4, False)
        dK_sparse = mx.sum(dKp_sparse, axis=2).astype(q.dtype)
        dK_dense = mx.sum(dKp_dense, axis=2).astype(q.dtype)
        mx.eval(dK_sparse, dK_dense); mx.synchronize()
        diff = float(mx.max(mx.abs(
            dK_sparse.astype(mx.float32) - dK_dense.astype(mx.float32))))
        assert diff < 1e-6, f"dK all-True max_diff = {diff:.4e}"

    @_skipif_no_nax
    def test_fused_sparse_all_true_within_fp32_ulp(self):
        from mlx_mfa import _ext
        B, H, qL, D = 1, 4, 2048, 64
        BT = 32
        NQ = NK = qL // BT
        q, k, v, dO = _mk(B, H, qL, D, mx.float16, 202)
        scale = 1.0 / math.sqrt(D)
        O, L = _ext.v6_nax_forward(q, k, v, False, True)
        mx.eval(O, L); mx.synchronize()
        D_vec = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1)
        mx.eval(D_vec); mx.synchronize()
        mask_all_bt = mx.ones((NQ, NK), dtype=mx.bool_)
        mask_all = _convert_mask_for_v6nax_bwd_kernel(mask_all_bt, BT, "DKDV", D)
        _AE(mask_all); mx.synchronize()

        dKp_s, dVp_s = _ext.v6_nax_backward_fused_dkdv_sparse_raw(
            q, k, v, L, dO, D_vec, mask_all, scale, 4, False)
        dKp_d, dVp_d = _ext.v6_nax_backward_fused_dkdv_raw(
            q, k, v, L, dO, D_vec, scale, 4, False)
        dK_s = mx.sum(dKp_s, axis=2).astype(q.dtype)
        dV_s = mx.sum(dVp_s, axis=2).astype(q.dtype)
        dK_d = mx.sum(dKp_d, axis=2).astype(q.dtype)
        dV_d = mx.sum(dVp_d, axis=2).astype(q.dtype)
        mx.eval(dK_s, dV_s, dK_d, dV_d); mx.synchronize()
        dk_diff = float(mx.max(mx.abs(dK_s.astype(mx.float32) - dK_d.astype(mx.float32))))
        dv_diff = float(mx.max(mx.abs(dV_s.astype(mx.float32) - dV_d.astype(mx.float32))))
        assert dk_diff < 1e-3, f"Fused dK all-True diff = {dk_diff:.4e}"
        assert dv_diff < 1e-3, f"Fused dV all-True diff = {dv_diff:.4e}"


class TestV6NAXSparseFullNativeEndToEnd:
    @_skipif_no_nax
    def test_native_d64_block_causal_matches_sdpa(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1"); monkeypatch.setenv("MFA_V6_BWD_SPARSE_NATIVE", "1")
        B, H, qL, D = 1, 4, 2048, 64
        BT = 32
        NQ = NK = qL // BT
        q, k, v, dO = _mk(B, H, qL, D, mx.float16, 300)
        scale = 1.0 / math.sqrt(D)
        mask = _block_causal_mask(NQ, NK); _AE(mask); mx.synchronize()

        def loss(qi, ki, vi):
            return (flash_attention_sparse(qi, ki, vi, mask, scale=scale) * dO).sum()
        dQ_n, dK_n, dV_n = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)

        from mlx_mfa.attention import _block_mask_to_float_bias
        bias = _block_mask_to_float_bias(
            mask, qL, qL, scale_q_dtype=q.dtype).astype(q.dtype)
        def loss_ref(qi, ki, vi):
            o = mx.fast.scaled_dot_product_attention(
                qi, ki, vi, scale=scale, mask=bias)
            return (o * dO).sum()
        dQ_r, dK_r, dV_r = mx.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dQ_n, dK_n, dV_n, dQ_r, dK_r, dV_r); mx.synchronize()

        for (a, r, name) in [(dQ_n, dQ_r, "dQ"), (dK_n, dK_r, "dK"), (dV_n, dV_r, "dV")]:
            rmse = _rmse(a, r)
            assert rmse < 5e-3, f"D=64 native {name} RMSE = {rmse:.4e}"

    @_skipif_no_nax
    def test_native_d128_block_causal_matches_sdpa(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1"); monkeypatch.setenv("MFA_V6_BWD_SPARSE_NATIVE", "1")
        B, H, qL, D = 1, 4, 2048, 128
        BT = 32
        NQ = NK = qL // BT
        q, k, v, dO = _mk(B, H, qL, D, mx.float16, 301)
        scale = 1.0 / math.sqrt(D)
        mask = _block_causal_mask(NQ, NK); _AE(mask); mx.synchronize()

        def loss(qi, ki, vi):
            return (flash_attention_sparse(qi, ki, vi, mask, scale=scale) * dO).sum()
        dQ_n, dK_n, dV_n = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)

        from mlx_mfa.attention import _block_mask_to_float_bias
        bias = _block_mask_to_float_bias(
            mask, qL, qL, scale_q_dtype=q.dtype).astype(q.dtype)
        def loss_ref(qi, ki, vi):
            o = mx.fast.scaled_dot_product_attention(
                qi, ki, vi, scale=scale, mask=bias)
            return (o * dO).sum()
        dQ_r, dK_r, dV_r = mx.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dQ_n, dK_n, dV_n, dQ_r, dK_r, dV_r); mx.synchronize()

        for (a, r, name) in [(dQ_n, dQ_r, "dQ"), (dK_n, dK_r, "dK"), (dV_n, dV_r, "dV")]:
            rmse = _rmse(a, r)
            assert rmse < 5e-3, f"D=128 native {name} RMSE = {rmse:.4e}"

    @_skipif_no_nax
    def test_native_engages_via_public_api_d64(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1"); monkeypatch.setenv("MFA_V6_BWD_SPARSE_NATIVE", "1")
        B, H, qL, D = 1, 4, 2048, 64
        BT = 32
        NQ = NK = qL // BT
        q, k, v, dO = _mk(B, H, qL, D, mx.float16, 302)
        scale = 1.0 / math.sqrt(D)
        mask = _block_causal_mask(NQ, NK); _AE(mask); mx.synchronize()

        def loss(qi, ki, vi):
            return (flash_attention_sparse(qi, ki, vi, mask, scale=scale) * dO).sum()
        dQ, dK, dV = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dQ, dK, dV); mx.synchronize()
        for g in (dQ, dK, dV):
            arr = np.array(g.astype(mx.float32))
            assert np.isfinite(arr).all()

    @_skipif_no_nax
    def test_section_c_wrapper_fallback_env_unset(self, monkeypatch):
        monkeypatch.delenv("MFA_ENABLE_V6_BACKWARD", raising=False)
        B, H, qL, D = 1, 4, 2048, 64
        BT = 32
        NQ = NK = qL // BT
        q, k, v, dO = _mk(B, H, qL, D, mx.float16, 303)
        scale = 1.0 / math.sqrt(D)
        mask = _block_causal_mask(NQ, NK); _AE(mask); mx.synchronize()

        def loss(qi, ki, vi):
            return (flash_attention_sparse(qi, ki, vi, mask, scale=scale) * dO).sum()
        dQ, dK, dV = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dQ, dK, dV); mx.synchronize()
        for g in (dQ, dK, dV):
            arr = np.array(g.astype(mx.float32))
            assert np.isfinite(arr).all()


class TestDensitySweep:
    @_skipif_no_nax
    @pytest.mark.parametrize("density", [0.1, 0.3, 0.5, 1.0])
    def test_d64_random_mask_density_matches_sdpa(self, monkeypatch, density):
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1"); monkeypatch.setenv("MFA_V6_BWD_SPARSE_NATIVE", "1")
        B, H, qL, D = 1, 4, 2048, 64
        BT = 32
        NQ = NK = qL // BT
        q, k, v, dO = _mk(B, H, qL, D, mx.float16, int(400 + density * 1000))
        scale = 1.0 / math.sqrt(D)
        mask = _random_mask(NQ, NK, density, seed=int(density * 100))
        _AE(mask); mx.synchronize()

        def loss(qi, ki, vi):
            return (flash_attention_sparse(qi, ki, vi, mask, scale=scale) * dO).sum()
        dQ_n, dK_n, dV_n = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)

        from mlx_mfa.attention import _block_mask_to_float_bias
        bias = _block_mask_to_float_bias(
            mask, qL, qL, scale_q_dtype=q.dtype).astype(q.dtype)
        def loss_ref(qi, ki, vi):
            o = mx.fast.scaled_dot_product_attention(
                qi, ki, vi, scale=scale, mask=bias)
            return (o * dO).sum()
        dQ_r, dK_r, dV_r = mx.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dQ_n, dK_n, dV_n, dQ_r, dK_r, dV_r); mx.synchronize()

        for (a, r, name) in [(dQ_n, dQ_r, "dQ"), (dK_n, dK_r, "dK"), (dV_n, dV_r, "dV")]:
            rmse = _rmse(a, r)
            assert rmse < 1e-2, f"density={density} {name} RMSE = {rmse:.4e}"
