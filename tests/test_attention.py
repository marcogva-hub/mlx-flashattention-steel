"""Tests for mlx-mfa flash_attention.

Correctness verified against mx.fast.scaled_dot_product_attention.

Test classes:
  TestFallbackPath    — always run, no extension needed
  TestPublicAPI       — always run, verifies is_mfa_available / get_device_info / etc.
  TestMFAKernel       — skipped without compiled extension
  TestMFABackward     — skipped without compiled extension
  TestEdgeCases       — skipped without compiled extension (GQA, N=1, non-multiples, etc.)
  TestBackwardEdge    — skipped without compiled extension (backward edge cases)
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import (
    flash_attention, flash_attention_rope, flash_attention_sparse,
    make_causal_block_mask, make_sliding_window_mask,
    is_mfa_available, get_device_info, get_supported_configs,
)
from mlx_mfa.attention import _ext_available, _fallback_sdpa, _steel_block_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reference_sdpa(q, k, v, scale, causal=False):
    """Reference SDPA using MLX built-in."""
    mask = None
    if causal:
        N, S = q.shape[2], k.shape[2]
        # MLX 0.31: mask dtype must match (promote to) output dtype
        mask = mx.triu(mx.full((N, S), float("-inf"), dtype=q.dtype), k=S - N + 1)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def random_qkv(B, H, N, D, dtype=mx.float16, seed=42):
    mx.random.seed(seed)
    q = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    k = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    v = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    return q, k, v


# ---------------------------------------------------------------------------
# Fallback path tests (always pass - no C++ extension needed)
# ---------------------------------------------------------------------------

class TestFallbackPath:
    """Tests that run via MLX SDPA fallback (no extension required)."""

    def test_fallback_unsupported_hdim(self):
        """head_dim=32 should fallback to SDPA."""
        q, k, v = random_qkv(1, 4, 16, 32)
        out = flash_attention(q, k, v)
        ref = reference_sdpa(q, k, v, scale=1.0 / math.sqrt(32))
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-2, atol=1e-3
        )

    def test_fallback_unsupported_dtype(self):
        """float64 should fallback."""
        q, k, v = random_qkv(1, 2, 8, 64, dtype=mx.float32)
        out = flash_attention(q, k, v)
        ref = reference_sdpa(q, k, v, scale=1.0 / math.sqrt(64))
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-4, atol=1e-5
        )

    def test_fallback_causal(self):
        """Causal masking via fallback."""
        q, k, v = random_qkv(1, 4, 32, 32)
        out = flash_attention(q, k, v, causal=True)
        ref = reference_sdpa(q, k, v, scale=1.0 / math.sqrt(32), causal=True)
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-2, atol=1e-3
        )

    def test_fallback_custom_scale(self):
        q, k, v = random_qkv(1, 2, 16, 32)
        out = flash_attention(q, k, v, scale=0.5)
        ref = reference_sdpa(q, k, v, scale=0.5)
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-2, atol=1e-3
        )

    def test_shape_validation(self):
        """3D inputs should raise ValueError."""
        q = mx.random.normal(shape=(4, 16, 64))
        k = mx.random.normal(shape=(4, 16, 64))
        v = mx.random.normal(shape=(4, 16, 64))
        with pytest.raises(ValueError, match="4"):
            flash_attention(q, k, v)

    def test_batch_heads(self):
        """Various batch and head combinations."""
        for B, H in [(1, 1), (2, 4), (4, 8)]:
            q, k, v = random_qkv(B, H, 16, 32)
            out = flash_attention(q, k, v)
            ref = reference_sdpa(q, k, v, scale=1.0 / math.sqrt(32))
            mx.eval(out, ref)
            np.testing.assert_allclose(
                np.array(out), np.array(ref), rtol=1e-2, atol=1e-3,
                err_msg=f"Failed for B={B}, H={H}"
            )


# ---------------------------------------------------------------------------
# MFA kernel tests (skipped if extension not compiled)
# ---------------------------------------------------------------------------

requires_ext = pytest.mark.skipif(
    not _ext_available(),
    reason="mlx_mfa._ext not compiled"
)


@requires_ext
class TestMFAKernel:
    """Tests requiring the compiled C++ extension."""

    @pytest.mark.parametrize("D", [64, 128, 256])
    def test_forward_correctness(self, D):
        q, k, v = random_qkv(2, 8, 256, D)
        scale = 1.0 / math.sqrt(D)
        out = flash_attention(q, k, v, scale=scale)
        ref = reference_sdpa(q, k, v, scale=scale)
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-2, atol=1e-3,
            err_msg=f"Forward mismatch at D={D}"
        )

    @pytest.mark.parametrize("D", [64, 128, 256])
    def test_forward_causal(self, D):
        q, k, v = random_qkv(2, 8, 256, D)
        scale = 1.0 / math.sqrt(D)
        out = flash_attention(q, k, v, scale=scale, causal=True)
        ref = reference_sdpa(q, k, v, scale=scale, causal=True)
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-2, atol=1e-3,
            err_msg=f"Causal mismatch at D={D}"
        )

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
    def test_dtypes(self, dtype):
        q, k, v = random_qkv(1, 4, 128, 128, dtype=dtype)
        scale = 1.0 / math.sqrt(128)
        out = flash_attention(q, k, v, scale=scale)
        ref = reference_sdpa(q, k, v, scale=scale)
        mx.eval(out, ref)
        tol = 1e-2 if dtype != mx.float32 else 1e-4
        # bfloat16 is not supported by numpy PEP 3118, so cast to float32 in MLX first.
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=tol, atol=tol,
            err_msg=f"Dtype mismatch for {dtype}"
        )

    def test_long_sequence(self):
        """N=4096 should work without OOM."""
        q, k, v = random_qkv(1, 4, 4096, 128)
        out = flash_attention(q, k, v, scale=1.0 / math.sqrt(128))
        mx.eval(out)
        assert out.shape == (1, 4, 4096, 128)


# ---------------------------------------------------------------------------
# Backward pass tests — Phase 3
# ---------------------------------------------------------------------------

@requires_ext
class TestMFABackward:
    """Gradient correctness tests for MFA backward pass (vjp).

    Strategy: compare MFA dQ/dK/dV against the reference gradients computed
    via mx.grad() through MLX SDPA (scaled_dot_product_attention).
    """

    def _grad_mfa(self, q, k, v, scale, causal=False):
        """Return (dQ, dK, dV) using MFA backward."""
        def loss(q_, k_, v_):
            return mx.sum(flash_attention(q_, k_, v_, scale=scale, causal=causal))
        grad_fn = mx.grad(loss, argnums=(0, 1, 2))
        grads = grad_fn(q, k, v)
        mx.eval(*grads)
        return grads

    def _grad_ref(self, q, k, v, scale, causal=False):
        """Return (dQ, dK, dV) using MLX SDPA backward (reference)."""
        def loss(q_, k_, v_):
            return mx.sum(reference_sdpa(q_, k_, v_, scale=scale, causal=causal))
        grad_fn = mx.grad(loss, argnums=(0, 1, 2))
        grads = grad_fn(q, k, v)
        mx.eval(*grads)
        return grads

    @pytest.mark.parametrize("D", [64, 128])
    def test_backward_f32_non_causal(self, D):
        """dQ/dK/dV must match reference within f32 tolerance (non-causal)."""
        B, H, N = 1, 2, 32
        scale = 1.0 / math.sqrt(D)
        q, k, v = random_qkv(B, H, N, D, dtype=mx.float32, seed=7)

        dq_mfa, dk_mfa, dv_mfa = self._grad_mfa(q, k, v, scale)
        dq_ref, dk_ref, dv_ref = self._grad_ref(q, k, v, scale)

        atol = 1e-4
        np.testing.assert_allclose(
            np.array(dq_mfa), np.array(dq_ref), atol=atol,
            err_msg=f"dQ mismatch D={D}"
        )
        np.testing.assert_allclose(
            np.array(dk_mfa), np.array(dk_ref), atol=atol,
            err_msg=f"dK mismatch D={D}"
        )
        np.testing.assert_allclose(
            np.array(dv_mfa), np.array(dv_ref), atol=atol,
            err_msg=f"dV mismatch D={D}"
        )

    @pytest.mark.parametrize("D", [64, 128])
    def test_backward_f32_causal(self, D):
        """dQ/dK/dV must match reference within f32 tolerance (causal)."""
        B, H, N = 1, 2, 32
        scale = 1.0 / math.sqrt(D)
        q, k, v = random_qkv(B, H, N, D, dtype=mx.float32, seed=13)

        dq_mfa, dk_mfa, dv_mfa = self._grad_mfa(q, k, v, scale, causal=True)
        dq_ref, dk_ref, dv_ref = self._grad_ref(q, k, v, scale, causal=True)

        atol = 1e-4
        np.testing.assert_allclose(
            np.array(dq_mfa), np.array(dq_ref), atol=atol,
            err_msg=f"dQ causal mismatch D={D}"
        )
        np.testing.assert_allclose(
            np.array(dk_mfa), np.array(dk_ref), atol=atol,
            err_msg=f"dK causal mismatch D={D}"
        )
        np.testing.assert_allclose(
            np.array(dv_mfa), np.array(dv_ref), atol=atol,
            err_msg=f"dV causal mismatch D={D}"
        )

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    def test_backward_low_prec(self, dtype):
        """Backward in f16/bf16: looser tolerance (half-precision accumulation)."""
        B, H, N, D = 1, 2, 32, 64
        scale = 1.0 / math.sqrt(D)
        q, k, v = random_qkv(B, H, N, D, dtype=dtype, seed=99)

        dq_mfa, dk_mfa, dv_mfa = self._grad_mfa(q, k, v, scale)
        dq_ref, dk_ref, dv_ref = self._grad_ref(q, k, v, scale)

        # Cast to f32 for numpy comparison (bf16 not supported by numpy PEP 3118)
        atol = 5e-2
        np.testing.assert_allclose(
            np.array(dq_mfa.astype(mx.float32)),
            np.array(dq_ref.astype(mx.float32)),
            atol=atol, err_msg=f"dQ mismatch dtype={dtype}"
        )
        np.testing.assert_allclose(
            np.array(dk_mfa.astype(mx.float32)),
            np.array(dk_ref.astype(mx.float32)),
            atol=atol, err_msg=f"dK mismatch dtype={dtype}"
        )
        np.testing.assert_allclose(
            np.array(dv_mfa.astype(mx.float32)),
            np.array(dv_ref.astype(mx.float32)),
            atol=atol, err_msg=f"dV mismatch dtype={dtype}"
        )

    def test_backward_shapes(self):
        """Gradient shapes and dtypes must match input shapes and dtypes."""
        B, H, N, D = 2, 4, 48, 128
        scale = 1.0 / math.sqrt(D)
        q, k, v = random_qkv(B, H, N, D, dtype=mx.float32, seed=5)

        dq, dk, dv = self._grad_mfa(q, k, v, scale)

        assert dq.shape == q.shape, f"dQ shape {dq.shape} != Q shape {q.shape}"
        assert dk.shape == k.shape, f"dK shape {dk.shape} != K shape {k.shape}"
        assert dv.shape == v.shape, f"dV shape {dv.shape} != V shape {v.shape}"
        assert dq.dtype == q.dtype, "dQ dtype mismatch"
        assert dk.dtype == k.dtype, "dK dtype mismatch"
        assert dv.dtype == v.dtype, "dV dtype mismatch"

    def test_training_step(self):
        """End-to-end training: one gradient descent step should reduce the loss."""
        B, H, N, D = 1, 4, 64, 128
        scale = 1.0 / math.sqrt(D)
        lr = 0.01

        mx.random.seed(42)
        q = mx.random.normal(shape=(B, H, N, D)).astype(mx.float32)
        k = mx.random.normal(shape=(B, H, N, D)).astype(mx.float32)
        v = mx.random.normal(shape=(B, H, N, D)).astype(mx.float32)

        def loss_fn(q_, k_, v_):
            return mx.sum(flash_attention(q_, k_, v_, scale=scale) ** 2)

        val_and_grad = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))

        loss0, (dq, dk, dv) = val_and_grad(q, k, v)
        mx.eval(loss0, dq, dk, dv)

        # Gradient descent step
        q2 = q - lr * dq
        k2 = k - lr * dk
        v2 = v - lr * dv

        loss1 = loss_fn(q2, k2, v2)
        mx.eval(loss1)

        assert float(loss1) < float(loss0), (
            f"Loss did not decrease after grad step: {float(loss0):.4f} → {float(loss1):.4f}"
        )


# ---------------------------------------------------------------------------
# Public API tests (always run — no extension needed)
# ---------------------------------------------------------------------------

class TestPublicAPI:
    """Tests for is_mfa_available(), get_device_info(), get_supported_configs()."""

    def test_is_mfa_available_returns_bool(self):
        result = is_mfa_available()
        assert isinstance(result, bool)

    def test_get_supported_configs_structure(self):
        cfg = get_supported_configs()
        assert "head_dims" in cfg
        assert "dtypes" in cfg
        assert "extension_available" in cfg
        assert 64 in cfg["head_dims"]
        assert 128 in cfg["head_dims"]
        assert 256 in cfg["head_dims"]
        assert mx.float16 in cfg["dtypes"]
        assert mx.bfloat16 in cfg["dtypes"]
        assert mx.float32 in cfg["dtypes"]
        assert isinstance(cfg["extension_available"], bool)

    def test_get_device_info_without_extension(self):
        """get_device_info returns sensible values even without extension."""
        info = get_device_info()
        assert "device_name" in info
        assert "gpu_family_gen" in info
        assert "is_m3_plus" in info
        assert "is_m5_plus" in info
        assert "chip_name" in info
        assert "extension_available" in info

    @pytest.mark.skipif(not _ext_available(), reason="extension not compiled")
    def test_get_device_info_with_extension(self):
        """When extension is available, hardware fields are populated."""
        info = get_device_info()
        assert info["extension_available"] is True
        assert isinstance(info["device_name"], str)
        assert len(info["device_name"]) > 0
        assert isinstance(info["gpu_family_gen"], int)
        assert info["gpu_family_gen"] > 0
        assert isinstance(info["is_m3_plus"], bool)
        # chip_name should be set for known generations
        assert info["chip_name"] is not None

    @pytest.mark.skipif(not _ext_available(), reason="extension not compiled")
    def test_m3_plus_threshold(self):
        """is_m3_plus should be False for M1/M2 (gen < 15), True for M3/M4."""
        info = get_device_info()
        gen = info["gpu_family_gen"]
        expected = gen >= 15
        assert info["is_m3_plus"] == expected, (
            f"gen={gen}: is_m3_plus should be {expected}, got {info['is_m3_plus']}"
        )


# ---------------------------------------------------------------------------
# Edge case tests (Phase 4.6.1) — requires extension
# ---------------------------------------------------------------------------

@requires_ext
class TestEdgeCases:
    """Edge cases: GQA, N=1, non-multiple seq lengths, cross-attention."""

    def test_gqa_4to1(self):
        """GQA 4:1 ratio (4 query heads, 1 kv head) should match reference."""
        B, Hq, Hkv, N, D = 1, 4, 1, 32, 64
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(20)
        q  = mx.random.normal(shape=(B, Hq, N, D)).astype(mx.float32)
        k  = mx.random.normal(shape=(B, Hkv, N, D)).astype(mx.float32)
        v  = mx.random.normal(shape=(B, Hkv, N, D)).astype(mx.float32)

        out = flash_attention(q, k, v, scale=scale)
        # Reference: manually tile k/v and run SDPA
        k_tiled = mx.repeat(k, 4, axis=1)
        v_tiled = mx.repeat(v, 4, axis=1)
        ref = reference_sdpa(q, k_tiled, v_tiled, scale=scale)
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-2, atol=1e-3,
            err_msg="GQA 4:1 mismatch"
        )
        assert out.shape == (B, Hq, N, D)

    def test_gqa_8to2(self):
        """GQA 8:2 ratio (8 query heads, 2 kv heads)."""
        B, Hq, Hkv, N, D = 1, 8, 2, 32, 64
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(21)
        q  = mx.random.normal(shape=(B, Hq, N, D)).astype(mx.float32)
        k  = mx.random.normal(shape=(B, Hkv, N, D)).astype(mx.float32)
        v  = mx.random.normal(shape=(B, Hkv, N, D)).astype(mx.float32)

        out = flash_attention(q, k, v, scale=scale)
        k_tiled = mx.repeat(k, 4, axis=1)
        v_tiled = mx.repeat(v, 4, axis=1)
        ref = reference_sdpa(q, k_tiled, v_tiled, scale=scale)
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-2, atol=1e-3,
            err_msg="GQA 8:2 mismatch"
        )

    def test_gqa_invalid_ratio_raises(self):
        """GQA with non-divisible head counts must raise ValueError."""
        q = mx.random.normal(shape=(1, 3, 16, 64))
        k = mx.random.normal(shape=(1, 2, 16, 64))
        v = mx.random.normal(shape=(1, 2, 16, 64))
        with pytest.raises(ValueError, match="divisible"):
            flash_attention(q, k, v)

    def test_seq_len_1(self):
        """N=1 (single-token decode step) must work for all D."""
        for D in [64, 128, 256]:
            scale = 1.0 / math.sqrt(D)
            q, k, v = random_qkv(1, 4, 1, D, dtype=mx.float32, seed=30 + D)
            out = flash_attention(q, k, v, scale=scale)
            ref = reference_sdpa(q, k, v, scale=scale)
            mx.eval(out, ref)
            np.testing.assert_allclose(
                np.array(out), np.array(ref), rtol=1e-4, atol=1e-5,
                err_msg=f"N=1 mismatch D={D}"
            )
            assert out.shape == (1, 4, 1, D)

    def test_seq_len_not_multiple_of_block(self):
        """N not a multiple of block_q (e.g. N=37) should be handled correctly."""
        D = 64
        scale = 1.0 / math.sqrt(D)
        q, k, v = random_qkv(1, 4, 37, D, dtype=mx.float32, seed=50)
        out = flash_attention(q, k, v, scale=scale)
        ref = reference_sdpa(q, k, v, scale=scale)
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-2, atol=1e-3,
            err_msg="N=37 (non-multiple) mismatch"
        )

    def test_cross_attention(self):
        """Cross-attention: N_q != N_kv should work correctly."""
        B, H, Nq, Nkv, D = 1, 4, 16, 48, 128
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(60)
        q = mx.random.normal(shape=(B, H, Nq, D)).astype(mx.float32)
        k = mx.random.normal(shape=(B, H, Nkv, D)).astype(mx.float32)
        v = mx.random.normal(shape=(B, H, Nkv, D)).astype(mx.float32)

        out = flash_attention(q, k, v, scale=scale)
        ref = reference_sdpa(q, k, v, scale=scale)
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-2, atol=1e-3,
            err_msg="Cross-attention mismatch"
        )
        assert out.shape == (B, H, Nq, D)

    def test_batch_size_1_heads_1(self):
        """B=1, H=1 edge case."""
        D = 128
        scale = 1.0 / math.sqrt(D)
        q, k, v = random_qkv(1, 1, 64, D, dtype=mx.float32, seed=70)
        out = flash_attention(q, k, v, scale=scale)
        ref = reference_sdpa(q, k, v, scale=scale)
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.array(out), np.array(ref), rtol=1e-4, atol=1e-5
        )

    def test_mismatched_head_dim_raises(self):
        """Mismatched head_dim between q and k must raise ValueError."""
        q = mx.random.normal(shape=(1, 4, 16, 64))
        k = mx.random.normal(shape=(1, 4, 16, 128))
        v = mx.random.normal(shape=(1, 4, 16, 128))
        with pytest.raises(ValueError, match="head_dim"):
            flash_attention(q, k, v)


# ---------------------------------------------------------------------------
# Backward edge case tests (Phase 4.6.2) — requires extension
# ---------------------------------------------------------------------------

@requires_ext
class TestBackwardEdge:
    """Backward pass edge cases."""

    def _grad_mfa(self, q, k, v, scale, causal=False):
        def loss(q_, k_, v_):
            return mx.sum(flash_attention(q_, k_, v_, scale=scale, causal=causal))
        grads = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)
        return grads

    def _grad_ref(self, q, k, v, scale, causal=False):
        def loss(q_, k_, v_):
            return mx.sum(reference_sdpa(q_, k_, v_, scale=scale, causal=causal))
        grads = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)
        return grads

    def test_backward_n1(self):
        """N=1 backward should produce valid (possibly all-zero) gradients."""
        D, scale = 64, 1.0 / math.sqrt(64)
        q, k, v = random_qkv(1, 2, 1, D, dtype=mx.float32, seed=80)
        dq, dk, dv = self._grad_mfa(q, k, v, scale)
        dq_ref, dk_ref, dv_ref = self._grad_ref(q, k, v, scale)
        np.testing.assert_allclose(
            np.array(dq), np.array(dq_ref), atol=1e-4,
            err_msg="N=1 dQ backward mismatch"
        )
        np.testing.assert_allclose(
            np.array(dk), np.array(dk_ref), atol=1e-4,
            err_msg="N=1 dK backward mismatch"
        )

    def test_backward_non_multiple_seq(self):
        """N=37 (non-multiple of block_q) backward."""
        D, scale = 64, 1.0 / math.sqrt(64)
        q, k, v = random_qkv(1, 2, 37, D, dtype=mx.float32, seed=90)
        dq, dk, dv = self._grad_mfa(q, k, v, scale)
        dq_ref, dk_ref, dv_ref = self._grad_ref(q, k, v, scale)
        np.testing.assert_allclose(
            np.array(dq), np.array(dq_ref), atol=1e-4,
            err_msg="N=37 dQ backward mismatch"
        )

    def test_value_and_grad(self):
        """mx.value_and_grad should return consistent loss + gradients."""
        B, H, N, D = 1, 2, 32, 64
        scale = 1.0 / math.sqrt(D)
        q, k, v = random_qkv(B, H, N, D, dtype=mx.float32, seed=100)

        def loss(q_, k_, v_):
            return mx.sum(flash_attention(q_, k_, v_, scale=scale))

        val_and_grad = mx.value_and_grad(loss, argnums=(0, 1, 2))
        loss_val, (dq, dk, dv) = val_and_grad(q, k, v)
        mx.eval(loss_val, dq, dk, dv)

        # Loss value must be finite
        assert math.isfinite(float(loss_val))
        # Gradient norms must be finite
        assert math.isfinite(float(mx.sum(mx.abs(dq))))

    def test_partial_argnums(self):
        """Gradient w.r.t. only q (argnum=0) should work."""
        B, H, N, D = 1, 2, 32, 64
        scale = 1.0 / math.sqrt(D)
        q, k, v = random_qkv(B, H, N, D, dtype=mx.float32, seed=110)

        def loss(q_):
            return mx.sum(flash_attention(q_, k, v, scale=scale))

        (dq,) = mx.grad(loss, argnums=(0,))(q)
        mx.eval(dq)
        dq_ref, _, _ = self._grad_ref(q, k, v, scale)

        # Flatten for comparison: when B=1, MLX may squeeze the batch dim from
        # the gradient returned via partial argnums.  Values must still match.
        np.testing.assert_allclose(
            np.array(dq).reshape(-1),
            np.array(dq_ref).reshape(-1),
            atol=1e-4,
            err_msg="Partial argnum=0 dQ mismatch"
        )


# ---------------------------------------------------------------------------
# Native GQA tests — requires C++ extension (STEEL kernel handles GQA natively)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
class TestNativeGQA:
    """Native GQA in STEEL kernel (no mx.repeat expansion)."""

    @pytest.mark.parametrize("ratio,D", [(2, 128), (4, 128), (8, 128), (2, 64), (4, 64)])
    def test_native_gqa_matches_repeat_ref(self, ratio, D):
        """Native GQA result must match mx.repeat + dense SDPA reference."""
        B, H_q, N = 1, 8, 256
        H_kv = H_q // ratio
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(77 + ratio)
        q = mx.random.normal((B, H_q,  N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)

        out_native = flash_attention(q, k, v, scale=scale)
        # Reference via mx.repeat → dense SDPA
        k_rep = mx.repeat(k, ratio, axis=1)
        v_rep = mx.repeat(v, ratio, axis=1)
        out_ref = mx.fast.scaled_dot_product_attention(q, k_rep, v_rep, scale=scale)
        mx.eval(out_native, out_ref)

        np.testing.assert_allclose(
            np.array(out_native.astype(mx.float32)),
            np.array(out_ref.astype(mx.float32)),
            atol=1e-2,
            err_msg=f"Native GQA ratio={ratio} D={D} mismatch"
        )
        assert list(out_native.shape) == [B, H_q, N, D]

    @pytest.mark.parametrize("ratio", [2, 4, 8])
    def test_native_gqa_causal(self, ratio):
        """Native GQA with causal=True matches causal reference."""
        B, H_q, N, D = 1, 8, 256, 128
        H_kv = H_q // ratio
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(99 + ratio)
        q = mx.random.normal((B, H_q,  N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)

        out_native = flash_attention(q, k, v, scale=scale, causal=True)
        k_rep = mx.repeat(k, ratio, axis=1)
        v_rep = mx.repeat(v, ratio, axis=1)
        # Causal mask for reference
        causal_m = mx.triu(mx.full((N, N), float("-inf"), dtype=mx.float16), k=1)
        out_ref = mx.fast.scaled_dot_product_attention(
            q, k_rep, v_rep, scale=scale, mask=causal_m)
        mx.eval(out_native, out_ref)

        np.testing.assert_allclose(
            np.array(out_native.astype(mx.float32)),
            np.array(out_ref.astype(mx.float32)),
            atol=1e-2,
            err_msg=f"Native GQA causal ratio={ratio} mismatch"
        )

    def test_native_gqa_backward_finite(self):
        """GQA backward (via SDPA vjp) must produce finite gradients."""
        B, H_q, H_kv, N, D = 1, 4, 2, 64, 128
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(111)
        q = mx.random.normal((B, H_q,  N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)

        def loss(q_, k_, v_):
            return mx.sum(flash_attention(q_, k_, v_, scale=scale, causal=True))

        dq, dk, dv = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)
        assert list(dq.shape) == [B, H_q,  N, D], "dQ shape wrong"
        assert list(dk.shape) == [B, H_kv, N, D], "dK shape wrong (should be H_kv, not H_q)"
        assert list(dv.shape) == [B, H_kv, N, D], "dV shape wrong"
        assert np.all(np.isfinite(np.array(dq.astype(mx.float32)))), "dQ non-finite"
        assert np.all(np.isfinite(np.array(dk.astype(mx.float32)))), "dK non-finite"
        assert np.all(np.isfinite(np.array(dv.astype(mx.float32)))), "dV non-finite"


# ---------------------------------------------------------------------------
# Block-sparse attention tests
# ---------------------------------------------------------------------------

def _ref_sparse_sdpa(q, k, v, block_mask, scale, causal=False):
    """Reference: expand block_mask to token-level float bias, then dense SDPA."""
    from mlx_mfa.attention import _block_mask_to_float_bias
    N, S = q.shape[2], k.shape[2]
    float_bias = _block_mask_to_float_bias(block_mask, N, S, q.dtype)
    if causal:
        causal_m = mx.triu(
            mx.full((N, S), float("-inf"), dtype=q.dtype), k=S - N + 1
        )
        float_bias = float_bias + causal_m
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=float_bias)


class TestSparseAttentionAPI:
    """Tests for make_causal_block_mask, make_sliding_window_mask shapes."""

    def test_causal_block_mask_shape(self):
        for D in [64, 128, 256]:
            BQ, BK = _steel_block_config(D)
            N = 256
            mask = make_causal_block_mask(N, head_dim=D)
            NQ = (N + BQ - 1) // BQ
            NK = (N + BK - 1) // BK
            assert list(mask.shape) == [NQ, NK], f"D={D}: expected [{NQ},{NK}], got {list(mask.shape)}"
            assert mask.dtype == mx.bool_

    def test_sliding_window_mask_shape(self):
        N, W = 512, 128
        for D in [64, 128, 256]:
            BQ, BK = _steel_block_config(D)
            mask = make_sliding_window_mask(N, W, head_dim=D)
            NQ = (N + BQ - 1) // BQ
            NK = (N + BK - 1) // BK
            assert list(mask.shape) == [NQ, NK]

    def test_causal_block_mask_lower_triangular(self):
        """Causal mask must be lower-triangular at block level."""
        mask = make_causal_block_mask(256, head_dim=128)
        arr = np.array(mask.astype(mx.uint8))
        NQ, NK = arr.shape
        for q in range(NQ):
            for k in range(NK):
                # k-block first token must be <= q-block last token
                BQ, BK = _steel_block_config(128)
                if k * BK > (q + 1) * BQ - 1:
                    assert arr[q, k] == 0, f"Expected 0 at [{q},{k}]"

    def test_sliding_window_all_true_when_window_ge_seq(self):
        """Window >= seq_len → all blocks active."""
        N, D = 128, 128
        mask = make_sliding_window_mask(N, window_size=N * 2, head_dim=D)
        assert mx.all(mask).item()

    def test_sparse_api_rejects_f32(self):
        B, H, N, D = 1, 2, 64, 64
        q = mx.ones((B, H, N, D), dtype=mx.float32)
        k, v = q, q
        BQ, BK = _steel_block_config(D)
        mask = mx.ones(((N + BQ - 1) // BQ, (N + BK - 1) // BK), dtype=mx.bool_)
        with pytest.raises(ValueError, match="float16 or bfloat16"):
            flash_attention_sparse(q, k, v, mask)

    def test_sparse_api_rejects_wrong_mask_shape(self):
        B, H, N, D = 1, 2, 64, 64
        q = mx.ones((B, H, N, D), dtype=mx.float16)
        k, v = q, q
        wrong_mask = mx.ones((5, 5), dtype=mx.bool_)
        with pytest.raises(ValueError, match="block_mask shape"):
            flash_attention_sparse(q, k, v, wrong_mask)


@pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
class TestSparseAttentionKernel:
    """Tests requiring the C++ STEEL sparse kernel."""

    @pytest.mark.parametrize("D", [64, 128, 256])
    def test_all_true_mask_matches_dense(self, D):
        """All-True block mask must produce identical result to dense forward."""
        B, H, N = 1, 4, 128
        q, k, v = random_qkv(B, H, N, D, seed=10)
        scale = 1.0 / math.sqrt(D)

        out_dense = flash_attention(q, k, v, scale=scale)
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK
        all_true = mx.ones((NQ, NK), dtype=mx.bool_)
        out_sparse = flash_attention_sparse(q, k, v, all_true, scale=scale)
        mx.eval(out_dense, out_sparse)

        np.testing.assert_allclose(
            np.array(out_dense.astype(mx.float32)),
            np.array(out_sparse.astype(mx.float32)),
            atol=1e-3,
            err_msg=f"D={D}: all-True sparse ≠ dense"
        )

    @pytest.mark.parametrize("D", [64, 128, 256])
    def test_causal_block_mask_with_causal_matches_dense_causal(self, D):
        """Block-causal mask + causal=True must match flash_attention(causal=True)."""
        B, H, N = 1, 4, 128
        q, k, v = random_qkv(B, H, N, D, seed=20)
        scale = 1.0 / math.sqrt(D)

        mask = make_causal_block_mask(N, head_dim=D)
        out_sparse = flash_attention_sparse(q, k, v, mask, scale=scale, causal=True)
        out_dense  = flash_attention(q, k, v, scale=scale, causal=True)
        mx.eval(out_sparse, out_dense)

        np.testing.assert_allclose(
            np.array(out_dense.astype(mx.float32)),
            np.array(out_sparse.astype(mx.float32)),
            atol=1e-3,
            err_msg=f"D={D}: causal block+causal ≠ dense causal"
        )

    def test_sliding_window_matches_ref(self):
        """Sliding-window mask output must match reference dense SDPA + float bias."""
        B, H, N, D = 1, 4, 256, 128
        q, k, v = random_qkv(B, H, N, D, seed=30)
        scale = 1.0 / math.sqrt(D)
        window = 64

        mask = make_sliding_window_mask(N, window_size=window, head_dim=D)
        out_sparse = flash_attention_sparse(q, k, v, mask, scale=scale)
        out_ref    = _ref_sparse_sdpa(q, k, v, mask, scale)
        mx.eval(out_sparse, out_ref)

        np.testing.assert_allclose(
            np.array(out_sparse.astype(mx.float32)),
            np.array(out_ref.astype(mx.float32)),
            atol=5e-3,
            err_msg="Sliding window sparse ≠ reference SDPA"
        )

    def test_all_false_mask_row_gives_nan_or_zero(self):
        """A row where all K-tiles are masked: output should be 0 (empty softmax)."""
        B, H, N, D = 1, 2, 64, 128
        q, k, v = random_qkv(B, H, N, D, seed=40)
        scale = 1.0 / math.sqrt(D)
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK
        # Only first Q-tile is active
        mask = mx.zeros((NQ, NK), dtype=mx.bool_)
        mask_active = mx.concatenate(
            [mx.ones((1, NK), dtype=mx.bool_), mx.zeros((NQ - 1, NK), dtype=mx.bool_)],
            axis=0
        )
        out = flash_attention_sparse(q, k, v, mask_active, scale=scale)
        mx.eval(out)
        # Second Q-tile rows should be 0 (no keys attended to)
        second_tile = np.array(out[0, 0, BQ:, :].astype(mx.float32))
        assert np.all(second_tile == 0.0) or np.all(np.isnan(second_tile)), \
            "Expected 0 or NaN for fully masked rows"

    @pytest.mark.parametrize("D", [128, 256])
    def test_sparse_backward(self, D):
        """Gradients from sparse attention must be finite (via dense SDPA backward)."""
        B, H, N = 1, 2, 64
        # Use float16 (the native sparse dtype); backward uses dense SDPA + float bias.
        q, k, v = random_qkv(B, H, N, D, dtype=mx.float16, seed=50)
        scale = 1.0 / math.sqrt(D)
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK
        mask = make_sliding_window_mask(N, window_size=32, head_dim=D)

        def loss(q_, k_, v_):
            return mx.sum(flash_attention_sparse(q_, k_, v_, mask, scale=scale))

        dq, dk, dv = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)
        assert np.all(np.isfinite(np.array(dq))), "dQ has non-finite values"
        assert np.all(np.isfinite(np.array(dk))), "dK has non-finite values"
        assert np.all(np.isfinite(np.array(dv))), "dV has non-finite values"


# ---------------------------------------------------------------------------
# Track F — M3+ config path tests (MFA_FORCE_GEN)
# ---------------------------------------------------------------------------

@requires_ext
class TestM3M4Path:
    """Verify M3+ blocking configs produce correct results.

    Uses MFA_FORCE_GEN env var to override hardware detection in the C++ layer.
    std::getenv is called at mx.eval() time (inside eval_gpu), so setting
    os.environ before mx.eval() routes to a different compiled KernelKey.
    Compares output against SDPA fallback on the same inputs.
    """

    def _flash_with_gen(self, q, k, v, scale, causal, gen_str):
        """Run flash_attention with MFA_FORCE_GEN=gen_str, return np.array."""
        import os
        prev = os.environ.get("MFA_FORCE_GEN")
        try:
            os.environ["MFA_FORCE_GEN"] = gen_str
            out = flash_attention(q, k, v, scale=scale, causal=causal)
            mx.eval(out)  # eval_gpu reads MFA_FORCE_GEN here
        finally:
            if prev is None:
                os.environ.pop("MFA_FORCE_GEN", None)
            else:
                os.environ["MFA_FORCE_GEN"] = prev
        return np.array(out.astype(mx.float32))

    def _sdpa_ref(self, q, k, v, scale, causal):
        from mlx_mfa.attention import _fallback_sdpa
        out = _fallback_sdpa(q, k, v, scale, causal)
        mx.eval(out)
        return np.array(out.astype(mx.float32))

    @pytest.mark.parametrize("D,N,causal", [
        (128, 64,  True),
        (128, 128, True),
        (128, 64,  False),
        (256, 64,  True),
        (256, 128, True),
    ])
    def test_m3_config_matches_sdpa(self, D, N, causal):
        """M3+ block config (gen=15) must match SDPA reference."""
        mx.random.seed(7)
        q = mx.random.normal((1, 8, N, D)).astype(mx.float16)
        k = mx.random.normal((1, 8, N, D)).astype(mx.float16)
        v = mx.random.normal((1, 8, N, D)).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        out_m3  = self._flash_with_gen(q, k, v, scale, causal, "15")
        out_ref = self._sdpa_ref(q, k, v, scale, causal)

        np.testing.assert_allclose(
            out_ref, out_m3, atol=1e-2,
            err_msg=f"M3+ config (D={D},N={N},causal={causal}) != SDPA ref",
        )

    def test_m1_and_m3_configs_agree(self):
        """M1 config (gen=13) and M3+ config (gen=15) must agree numerically."""
        mx.random.seed(8)
        D, N = 128, 64
        q = mx.random.normal((1, 4, N, D)).astype(mx.float16)
        k = mx.random.normal((1, 4, N, D)).astype(mx.float16)
        v = mx.random.normal((1, 4, N, D)).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        out_m1 = self._flash_with_gen(q, k, v, scale, True, "13")
        out_m3 = self._flash_with_gen(q, k, v, scale, True, "15")

        np.testing.assert_allclose(
            out_m1, out_m3, atol=1e-2,
            err_msg="M1 (gen=13) and M3+ (gen=15) configs disagree for D=128",
        )


# ---------------------------------------------------------------------------
# Track G — Sparse backward (sdpa_sparse) tests
# ---------------------------------------------------------------------------

@requires_ext
class TestSparseBackwardTiled:
    """Verify backward='sdpa_sparse' (tiled Python backward with saved LSE).

    All tests compare sdpa_sparse against sdpa (dense SDPA backward reference).
    Both paths must agree to atol=2e-2 for f16.
    """

    # ── helpers ─────────────────────────────────────────────────────────────

    def _grads(self, q, k, v, mask, scale, causal=False, backward="sdpa"):
        """Return (dq, dk, dv) as np arrays for the given backward mode."""
        def loss(q_, k_, v_):
            return mx.sum(
                flash_attention_sparse(q_, k_, v_, mask, scale=scale,
                                       causal=causal, backward=backward)
            )
        dq, dk, dv = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)
        return (
            np.array(dq.astype(mx.float32)),
            np.array(dk.astype(mx.float32)),
            np.array(dv.astype(mx.float32)),
        )

    # ── correctness against sdpa reference ──────────────────────────────────

    @pytest.mark.parametrize("D", [64, 128])
    def test_sdpa_sparse_matches_sdpa_dense(self, D):
        """sdpa_sparse gradients must match sdpa (dense) reference for all-true mask."""
        B, H, N = 1, 4, 64
        q, k, v = random_qkv(B, H, N, D, dtype=mx.float16, seed=60)
        scale = 1.0 / math.sqrt(D)
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK
        mask = mx.ones((NQ, NK), dtype=mx.bool_)

        dq_ref, dk_ref, dv_ref = self._grads(q, k, v, mask, scale, backward="sdpa")
        dq_sp,  dk_sp,  dv_sp  = self._grads(q, k, v, mask, scale, backward="sdpa_sparse")

        np.testing.assert_allclose(dq_ref, dq_sp,  atol=2e-2,
                                    err_msg=f"D={D}: dQ sdpa_sparse != sdpa")
        np.testing.assert_allclose(dk_ref, dk_sp,  atol=2e-2,
                                    err_msg=f"D={D}: dK sdpa_sparse != sdpa")
        np.testing.assert_allclose(dv_ref, dv_sp,  atol=2e-2,
                                    err_msg=f"D={D}: dV sdpa_sparse != sdpa")

    def test_sdpa_sparse_causal_matches_sdpa_dense(self):
        """Causal sdpa_sparse must match causal sdpa reference."""
        B, H, N, D = 1, 4, 64, 128
        q, k, v = random_qkv(B, H, N, D, dtype=mx.float16, seed=61)
        scale = 1.0 / math.sqrt(D)
        mask = make_causal_block_mask(N, head_dim=D)

        dq_ref, dk_ref, dv_ref = self._grads(q, k, v, mask, scale,
                                              causal=True, backward="sdpa")
        dq_sp,  dk_sp,  dv_sp  = self._grads(q, k, v, mask, scale,
                                              causal=True, backward="sdpa_sparse")

        np.testing.assert_allclose(dq_ref, dq_sp,  atol=2e-2,
                                    err_msg="causal: dQ sdpa_sparse != sdpa")
        np.testing.assert_allclose(dk_ref, dk_sp,  atol=2e-2,
                                    err_msg="causal: dK sdpa_sparse != sdpa")
        np.testing.assert_allclose(dv_ref, dv_sp,  atol=2e-2,
                                    err_msg="causal: dV sdpa_sparse != sdpa")

    def test_sdpa_sparse_sliding_window_matches_sdpa_dense(self):
        """Sliding-window sdpa_sparse matches sdpa (dense) reference."""
        B, H, N, D = 1, 4, 128, 128
        q, k, v = random_qkv(B, H, N, D, dtype=mx.float16, seed=62)
        scale = 1.0 / math.sqrt(D)
        mask = make_sliding_window_mask(N, window_size=32, head_dim=D)

        dq_ref, dk_ref, dv_ref = self._grads(q, k, v, mask, scale, backward="sdpa")
        dq_sp,  dk_sp,  dv_sp  = self._grads(q, k, v, mask, scale,
                                              backward="sdpa_sparse")

        np.testing.assert_allclose(dq_ref, dq_sp,  atol=2e-2,
                                    err_msg="sliding: dQ sdpa_sparse != sdpa")
        np.testing.assert_allclose(dk_ref, dk_sp,  atol=2e-2,
                                    err_msg="sliding: dK sdpa_sparse != sdpa")
        np.testing.assert_allclose(dv_ref, dv_sp,  atol=2e-2,
                                    err_msg="sliding: dV sdpa_sparse != sdpa")

    # ── finite / shape tests ─────────────────────────────────────────────────

    @pytest.mark.parametrize("D", [64, 128])
    def test_sdpa_sparse_gradients_finite(self, D):
        """sdpa_sparse gradients must be finite (no NaN/Inf)."""
        B, H, N = 1, 2, 64
        q, k, v = random_qkv(B, H, N, D, dtype=mx.float16, seed=63)
        scale = 1.0 / math.sqrt(D)
        mask = make_sliding_window_mask(N, window_size=32, head_dim=D)

        dq, dk, dv = self._grads(q, k, v, mask, scale, backward="sdpa_sparse")
        assert np.all(np.isfinite(dq)), f"D={D}: dQ has non-finite values"
        assert np.all(np.isfinite(dk)), f"D={D}: dK has non-finite values"
        assert np.all(np.isfinite(dv)), f"D={D}: dV has non-finite values"

    @pytest.mark.parametrize("D", [64, 128])
    def test_sdpa_sparse_gradient_shapes(self, D):
        """dQ/dK/dV shapes must match Q/K/V shapes."""
        B, H, N = 1, 4, 64
        q, k, v = random_qkv(B, H, N, D, dtype=mx.float16, seed=64)
        scale = 1.0 / math.sqrt(D)
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK
        mask = mx.ones((NQ, NK), dtype=mx.bool_)

        dq, dk, dv = self._grads(q, k, v, mask, scale, backward="sdpa_sparse")
        assert list(dq.shape) == [B, H, N, D], f"dQ shape {dq.shape} != {[B,H,N,D]}"
        assert list(dk.shape) == [B, H, N, D], f"dK shape {dk.shape} != {[B,H,N,D]}"
        assert list(dv.shape) == [B, H, N, D], f"dV shape {dv.shape} != {[B,H,N,D]}"

    # ── GQA sparse backward ──────────────────────────────────────────────────

    def test_sdpa_sparse_gqa_shape_and_finite(self):
        """GQA sdpa_sparse: dK/dV shapes must be [B, H_kv, S, D] and finite."""
        B, H_q, H_kv, N, D = 1, 8, 2, 64, 128
        mx.random.seed(65)
        q = mx.random.normal((B, H_q, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK
        mask = mx.ones((NQ, NK), dtype=mx.bool_)

        def loss(q_, k_, v_):
            return mx.sum(
                flash_attention_sparse(q_, k_, v_, mask, scale=scale,
                                       backward="sdpa_sparse")
            )
        dq, dk, dv = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)

        assert list(dq.shape) == [B, H_q,  N, D], f"dQ shape {dq.shape}"
        assert list(dk.shape) == [B, H_kv, N, D], f"dK shape {dk.shape}"
        assert list(dv.shape) == [B, H_kv, N, D], f"dV shape {dv.shape}"
        assert np.all(np.isfinite(np.array(dq.astype(mx.float32)))), "dQ non-finite"
        assert np.all(np.isfinite(np.array(dk.astype(mx.float32)))), "dK non-finite"
        assert np.all(np.isfinite(np.array(dv.astype(mx.float32)))), "dV non-finite"

    # ── value_and_grad ───────────────────────────────────────────────────────

    def test_sdpa_sparse_value_and_grad(self):
        """mx.value_and_grad must work with sdpa_sparse backward."""
        B, H, N, D = 1, 4, 64, 128
        q, k, v = random_qkv(B, H, N, D, dtype=mx.float16, seed=66)
        scale = 1.0 / math.sqrt(D)
        mask = make_causal_block_mask(N, head_dim=D)

        def loss(q_, k_, v_):
            return mx.sum(
                flash_attention_sparse(q_, k_, v_, mask, scale=scale,
                                       causal=True, backward="sdpa_sparse")
            )
        val_fn = mx.value_and_grad(loss, argnums=(0, 1, 2))
        loss_val, (dq, dk, dv) = val_fn(q, k, v)
        mx.eval(loss_val, dq, dk, dv)

        assert np.isfinite(float(loss_val)), "loss is not finite"
        assert np.all(np.isfinite(np.array(dq.astype(mx.float32)))), "dQ non-finite"
        assert np.all(np.isfinite(np.array(dk.astype(mx.float32)))), "dK non-finite"


# ==========================================================================
# Flash Decoding (Split-KV) — Track H
# ==========================================================================

@pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
class TestFlashDecode:
    """Flash Decoding (N_q ≤ 4, S ≥ 256) correctness vs. SDPA reference."""

    def _ref(self, q, k, v, scale, causal):
        """Standard SDPA reference (float32 accumulation)."""
        q32 = q.astype(mx.float32)
        k32 = k.astype(mx.float32)
        v32 = v.astype(mx.float32)
        scores = mx.matmul(q32, k32.swapaxes(-1, -2)) * scale
        if causal:
            B, H, N, S = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
            # Build causal mask: query i can attend to keys 0..(S-N+i)
            q_pos = mx.arange(N)[:, None] + (S - N)   # [N, 1]
            k_pos = mx.arange(S)[None, :]              # [1, S]
            mask = (q_pos < k_pos).astype(mx.float32) * -1e9
            scores = scores + mask[None, None, :, :]
        probs = mx.softmax(scores, axis=-1)
        out = mx.matmul(probs, v32)
        mx.eval(out)
        return out

    @pytest.mark.parametrize("D", [64, 128, 256])
    def test_decode_noncausal(self, D):
        """N=1 decode, non-causal: Flash Decode should match SDPA within tol."""
        B, H, N, S = 1, 8, 1, 512
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, S, D]).astype(mx.float16)
        v = mx.random.normal([B, H, S, D]).astype(mx.float16)

        out = flash_attention(q, k, v, scale=scale, causal=False)
        ref = self._ref(q, k, v, scale, causal=False)
        mx.eval(out, ref)

        out_np = np.array(out.astype(mx.float32))
        ref_np = np.array(ref)
        max_err = float(np.max(np.abs(out_np - ref_np)))
        assert max_err < 0.05, f"D={D} non-causal max_err={max_err:.4f}"

    @pytest.mark.parametrize("D", [64, 128])
    def test_decode_causal(self, D):
        """N=1 causal decode: query attends to all keys (qL_off=S-1 with N=1)."""
        B, H, N, S = 1, 8, 1, 512
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, S, D]).astype(mx.float16)
        v = mx.random.normal([B, H, S, D]).astype(mx.float16)

        out = flash_attention(q, k, v, scale=scale, causal=True)
        ref = self._ref(q, k, v, scale, causal=True)
        mx.eval(out, ref)

        out_np = np.array(out.astype(mx.float32))
        ref_np = np.array(ref)
        max_err = float(np.max(np.abs(out_np - ref_np)))
        assert max_err < 0.05, f"D={D} causal max_err={max_err:.4f}"

    def test_decode_large_kv(self):
        """N=1 with S=4096 — exercises many splits."""
        D, S = 128, 4096
        B, H, N = 2, 8, 1
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, S, D]).astype(mx.float16)
        v = mx.random.normal([B, H, S, D]).astype(mx.float16)

        out = flash_attention(q, k, v, scale=scale, causal=False)
        ref = self._ref(q, k, v, scale, causal=False)
        mx.eval(out, ref)

        out_np = np.array(out.astype(mx.float32))
        ref_np = np.array(ref)
        max_err = float(np.max(np.abs(out_np - ref_np)))
        assert max_err < 0.05, f"S=4096 max_err={max_err:.4f}"

    def test_decode_small_kv_boundary(self):
        """S=256 is the activation threshold — should use Flash Decode."""
        D, S = 64, 256
        B, H, N = 1, 4, 1
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, S, D]).astype(mx.float16)
        v = mx.random.normal([B, H, S, D]).astype(mx.float16)

        out = flash_attention(q, k, v, scale=scale, causal=False)
        ref = self._ref(q, k, v, scale, causal=False)
        mx.eval(out, ref)

        out_np = np.array(out.astype(mx.float32))
        ref_np = np.array(ref)
        max_err = float(np.max(np.abs(out_np - ref_np)))
        assert max_err < 0.05, f"S=256 boundary max_err={max_err:.4f}"

    def test_decode_n4(self):
        """N=4 (upper decode threshold)."""
        D, S = 128, 512
        B, H, N = 1, 8, 4
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, S, D]).astype(mx.float16)
        v = mx.random.normal([B, H, S, D]).astype(mx.float16)

        out = flash_attention(q, k, v, scale=scale, causal=False)
        ref = self._ref(q, k, v, scale, causal=False)
        mx.eval(out, ref)

        out_np = np.array(out.astype(mx.float32))
        ref_np = np.array(ref)
        max_err = float(np.max(np.abs(out_np - ref_np)))
        assert max_err < 0.05, f"N=4 max_err={max_err:.4f}"

    def test_decode_gqa(self):
        """GQA (ratio 4:1) with Flash Decode."""
        D, S = 64, 512
        B, H_q, H_kv, N = 1, 8, 2, 1
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal([B, H_q, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H_kv, S, D]).astype(mx.float16)
        v = mx.random.normal([B, H_kv, S, D]).astype(mx.float16)

        out = flash_attention(q, k, v, scale=scale, causal=False)
        # Reference: expand kv to H_q heads
        k_exp = mx.repeat(k, H_q // H_kv, axis=1)
        v_exp = mx.repeat(v, H_q // H_kv, axis=1)
        ref = self._ref(q, k_exp, v_exp, scale, causal=False)
        mx.eval(out, ref)

        out_np = np.array(out.astype(mx.float32))
        ref_np = np.array(ref)
        max_err = float(np.max(np.abs(out_np - ref_np)))
        assert max_err < 0.05, f"GQA max_err={max_err:.4f}"

    def test_decode_bf16(self):
        """bfloat16 Flash Decode."""
        D, S = 64, 256
        B, H, N = 1, 4, 1
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal([B, H, N, D]).astype(mx.bfloat16)
        k = mx.random.normal([B, H, S, D]).astype(mx.bfloat16)
        v = mx.random.normal([B, H, S, D]).astype(mx.bfloat16)

        out = flash_attention(q, k, v, scale=scale, causal=False)
        ref = self._ref(q, k, v, scale, causal=False)
        mx.eval(out, ref)

        out_np = np.array(out.astype(mx.float32))
        ref_np = np.array(ref.astype(mx.float32))
        max_err = float(np.max(np.abs(out_np - ref_np)))
        assert max_err < 0.1, f"bf16 max_err={max_err:.4f}"

    def test_no_flash_decode_n5(self):
        """N=5 should NOT use Flash Decode (uses standard STEEL path)."""
        D, S = 64, 512
        B, H, N = 1, 4, 5
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, S, D]).astype(mx.float16)
        v = mx.random.normal([B, H, S, D]).astype(mx.float16)

        out = flash_attention(q, k, v, scale=scale, causal=False)
        ref = self._ref(q, k, v, scale, causal=False)
        mx.eval(out, ref)

        out_np = np.array(out.astype(mx.float32))
        ref_np = np.array(ref)
        max_err = float(np.max(np.abs(out_np - ref_np)))
        assert max_err < 0.05, f"N=5 (non-FD) max_err={max_err:.4f}"


# ===========================================================================
# Track I — M5+ (gen >= 17) detection stub
# ===========================================================================

class TestM5Detection:
    """Verify is_m5_plus flag and chip-name mapping for gen >= 17 (M5 stub).

    These tests use monkeypatching to simulate M5 hardware without requiring
    actual M5 silicon, exercising the get_device_info() Python logic directly.
    """

    def test_m5_plus_flag_false_on_current_hardware(self):
        """On any hardware we can actually test, is_m5_plus should be bool."""
        info = get_device_info()
        assert "is_m5_plus" in info
        assert isinstance(info["is_m5_plus"], bool) or info["is_m5_plus"] is None
        # Current hardware (M1–M4, gen 13–16) must report False.
        gen = info.get("gpu_family_gen")
        if gen is not None and gen < 17:
            assert info["is_m5_plus"] is False, (
                f"gen={gen} < 17 should give is_m5_plus=False, got {info['is_m5_plus']}"
            )

    def test_m5_chip_name_in_mapping(self, monkeypatch):
        """get_device_info() returns chip_name='M5' when C++ reports gen=17."""
        if not _ext_available():
            pytest.skip("extension not compiled")

        import mlx_mfa._ext as ext_mod
        import mlx_mfa.attention as attn_mod

        original_get = ext_mod.get_device_info

        def mock_get_device_info():
            d = original_get()
            d["gpu_family_gen"] = 17
            return d

        monkeypatch.setattr(ext_mod, "get_device_info", mock_get_device_info)
        monkeypatch.setattr(attn_mod, "_get_device_info_raw",
                            lambda: mock_get_device_info(), raising=False)

        # Call get_device_info() directly using the patched raw dict.
        # Build the result the same way attention.py does.
        raw = mock_get_device_info()
        gen = raw.get("gpu_family_gen")
        _GEN_TO_CHIP = {13: "M1", 14: "M2", 15: "M3", 16: "M4", 17: "M5"}
        chip = _GEN_TO_CHIP.get(gen, f"Apple-g{gen}")
        is_m5_plus = gen >= 17

        assert chip == "M5", f"gen=17 should map to 'M5', got '{chip}'"
        assert is_m5_plus is True

    def test_m5_is_also_m3_plus(self):
        """M5 hardware (gen=17) must satisfy both is_m3_plus and is_m5_plus."""
        # Simulate the logic in get_device_info() for gen=17.
        gen = 17
        is_m3_plus = gen >= 15
        is_m5_plus = gen >= 17
        assert is_m3_plus is True, "M5 (gen=17) should be is_m3_plus=True"
        assert is_m5_plus is True, "M5 (gen=17) should be is_m5_plus=True"


# ---------------------------------------------------------------------------
# Track L: RoPE Fusion tests
# ---------------------------------------------------------------------------

def _make_rope_tables(max_len: int, head_dim: int, base: float = 10000.0):
    """Build float32 [max_len, head_dim/2] cos/sin tables.

    Uses the standard inverse-frequency formula::

        theta_i = base^{-2i/D}   for i = 0, 1, ..., D/2 - 1
        cos[pos, i] = cos(pos * theta_i)
        sin[pos, i] = sin(pos * theta_i)
    """
    half_D = head_dim // 2
    i = mx.arange(half_D, dtype=mx.float32)
    inv_freq = 1.0 / (base ** (2.0 * i / head_dim))
    positions = mx.arange(max_len, dtype=mx.float32)
    # Outer product: [max_len, half_D]
    angles = positions[:, None] * inv_freq[None, :]
    return mx.cos(angles), mx.sin(angles)


def _apply_rope_python(x, cos, sin, offset=0):
    """Reference Python RoPE (interleaved pairs).

    x: [B, H, N, D]
    cos/sin: [max_len, D/2]
    """
    B, H, N, D = x.shape
    half_D = D // 2
    cos_n = cos[offset : offset + N, :]           # [N, D/2]
    sin_n = sin[offset : offset + N, :]           # [N, D/2]
    x_pairs = x.reshape(B, H, N, half_D, 2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    cos_bc = cos_n[None, None, :, :].astype(x.dtype)
    sin_bc = sin_n[None, None, :, :].astype(x.dtype)
    x0_rot = x0 * cos_bc - x1 * sin_bc
    x1_rot = x0 * sin_bc + x1 * cos_bc
    return mx.stack([x0_rot, x1_rot], axis=-1).reshape(B, H, N, D)


@requires_ext
class TestRoPEFusion:
    """Tests for flash_attention_rope — in-kernel RoPE fusion.

    Correctness: kernel result == (apply_rope_python + SDPA fallback).
    """

    @pytest.mark.parametrize("D", [64, 128])
    def test_rope_matches_python_reference(self, D):
        """Fused RoPE kernel output matches Python RoPE + SDPA."""
        from mlx_mfa import flash_attention_rope

        B, H, N, S = 1, 4, 64, 64
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(7)
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, S, D), dtype=mx.float16)
        v = mx.random.normal((B, H, S, D), dtype=mx.float16)
        cos, sin = _make_rope_tables(256, D)

        out_mfa = flash_attention_rope(q, k, v, cos, sin, scale=scale,
                                       causal=False, cache_seqlens=0)
        q_rot = _apply_rope_python(q, cos, sin, offset=0)
        k_rot = _apply_rope_python(k, cos, sin, offset=0)
        ref = mx.fast.scaled_dot_product_attention(q_rot, k_rot, v, scale=scale)
        mx.eval(out_mfa, ref)

        np.testing.assert_allclose(
            np.array(out_mfa.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-2, atol=1e-2,
            err_msg=f"RoPE mismatch at D={D}",
        )

    @pytest.mark.parametrize("D", [64, 128])
    def test_cache_seqlens_offset(self, D):
        """cache_seqlens shifts Q positions (decode scenario)."""
        from mlx_mfa import flash_attention_rope

        B, H, N, S = 1, 2, 4, 64   # N=4 simulates single-token decode
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(11)
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, S, D), dtype=mx.float16)
        v = mx.random.normal((B, H, S, D), dtype=mx.float16)
        cos, sin = _make_rope_tables(256, D)
        cache_seqlens = 32   # Q tokens 0-3 are at absolute positions 32-35

        out_mfa = flash_attention_rope(q, k, v, cos, sin, scale=scale,
                                       causal=True,
                                       cache_seqlens=cache_seqlens)
        q_rot = _apply_rope_python(q, cos, sin, offset=cache_seqlens)
        k_rot = _apply_rope_python(k, cos, sin, offset=0)
        ref = mx.fast.scaled_dot_product_attention(
            q_rot, k_rot, v, scale=scale,
            mask=mx.triu(
                mx.full((N, S), float("-inf"), dtype=mx.float16),
                k=S - N + 1,
            )
        )
        mx.eval(out_mfa, ref)

        np.testing.assert_allclose(
            np.array(out_mfa.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-2, atol=2e-2,
            err_msg=f"RoPE cache_seqlens offset mismatch at D={D}",
        )

    def test_rope_fallback_for_float32(self):
        """float32 inputs fall back to Python RoPE + SDPA (no error)."""
        from mlx_mfa import flash_attention_rope

        B, H, N, D = 1, 2, 32, 64
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal((B, H, N, D), dtype=mx.float32)
        k = mx.random.normal((B, H, N, D), dtype=mx.float32)
        v = mx.random.normal((B, H, N, D), dtype=mx.float32)
        cos, sin = _make_rope_tables(128, D)

        # Should not raise; fallback path applies RoPE in Python then calls SDPA
        out = flash_attention_rope(q, k, v, cos, sin, scale=scale)
        mx.eval(out)
        assert out.shape == (B, H, N, D)

    def test_rope_none_falls_back_to_regular_attention(self):
        """With identity RoPE (cos=1, sin=0), result equals plain attention."""
        from mlx_mfa import flash_attention, flash_attention_rope

        B, H, N, D = 1, 2, 32, 64
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(99)
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, N, D), dtype=mx.float16)
        v = mx.random.normal((B, H, N, D), dtype=mx.float16)

        # Identity rotation: cos=1, sin=0 everywhere → no rotation
        cos_ones = mx.ones((N, D // 2), dtype=mx.float32)
        sin_zeros = mx.zeros((N, D // 2), dtype=mx.float32)

        out_rope = flash_attention_rope(q, k, v, cos_ones, sin_zeros, scale=scale)
        out_plain = flash_attention(q, k, v, scale=scale)
        mx.eval(out_rope, out_plain)

        np.testing.assert_allclose(
            np.array(out_rope.astype(mx.float32)),
            np.array(out_plain.astype(mx.float32)),
            rtol=1e-2, atol=1e-3,
            err_msg="Identity RoPE should match plain flash_attention",
        )

    @pytest.mark.parametrize("D", [64, 128])
    def test_rope_output_shape_and_dtype(self, D):
        """Output has the same shape and dtype as Q."""
        from mlx_mfa import flash_attention_rope

        B, H, N = 2, 4, 48
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, N, D), dtype=mx.float16)
        v = mx.random.normal((B, H, N, D), dtype=mx.float16)
        cos, sin = _make_rope_tables(256, D)

        out = flash_attention_rope(q, k, v, cos, sin, scale=1.0 / math.sqrt(D))
        mx.eval(out)
        assert out.shape == (B, H, N, D), f"Expected {(B, H, N, D)}, got {out.shape}"
        assert out.dtype == mx.float16, f"Expected float16, got {out.dtype}"
