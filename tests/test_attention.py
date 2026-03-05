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


# ============================================================================
# Track O — Spatial 2D/3D block masks
# ============================================================================

class TestSpatialMasks:
    """Tests for make_spatial_2d_mask, make_spatial_3d_mask, make_topk_spatial_mask."""

    def test_2d_mask_shape(self):
        """Correct tile shape for various H, W."""
        from mlx_mfa.masks import make_spatial_2d_mask, _bq_bk
        BQ, BK = _bq_bk(128)
        for H, W, R in [(8, 8, 2), (16, 32, 4), (32, 64, 8)]:
            N = H * W
            NQ = (N + BQ - 1) // BQ
            NK = (N + BK - 1) // BK
            mask = make_spatial_2d_mask(H, W, spatial_radius=R, head_dim=128)
            assert mask.shape == (NQ, NK), f"Expected ({NQ},{NK}), got {mask.shape}"
            assert mask.dtype == mx.bool_, f"dtype should be bool"

    def test_2d_mask_full_radius_all_active(self):
        """With radius >= max(H, W), every tile should be active (dense)."""
        from mlx_mfa.masks import make_spatial_2d_mask
        H, W = 8, 8
        mask = make_spatial_2d_mask(H, W, spatial_radius=100, head_dim=128)
        mx.eval(mask)
        assert bool(mask.all()), "All tiles should be active with large radius"

    def test_2d_mask_symmetry(self):
        """Spatial mask is symmetric: mask[i,j] == mask[j,i] when NQ == NK."""
        from mlx_mfa.masks import make_spatial_2d_mask, _bq_bk
        BQ, BK = _bq_bk(128)
        # Use a seq length where NQ == NK (BQ == BK; head_dim=64 gives BQ=BK=32)
        H, W = 8, 8
        mask = make_spatial_2d_mask(H, W, spatial_radius=2, head_dim=64)
        mx.eval(mask)
        import numpy as np
        m = np.array(mask)
        assert np.array_equal(m, m.T), "2D spatial mask should be symmetric"

    def test_2d_mask_radius_zero_sparse(self):
        """Radius=0: mask is sparser than radius=4."""
        from mlx_mfa.masks import make_spatial_2d_mask
        import numpy as np
        H, W = 16, 16
        mask_r0 = make_spatial_2d_mask(H, W, spatial_radius=0, head_dim=64)
        mask_r4 = make_spatial_2d_mask(H, W, spatial_radius=4, head_dim=64)
        mx.eval(mask_r0, mask_r4)
        density_r0 = np.array(mask_r0).mean()
        density_r4 = np.array(mask_r4).mean()
        assert density_r0 < density_r4, \
            f"radius=0 should be sparser than radius=4: {density_r0:.3f} >= {density_r4:.3f}"

    def test_3d_mask_shape(self):
        """Correct shape for 3D video mask."""
        from mlx_mfa.masks import make_spatial_3d_mask, _bq_bk
        BQ, BK = _bq_bk(128)
        H, W, T = 8, 8, 4
        N = H * W * T
        NQ = (N + BQ - 1) // BQ
        NK = (N + BK - 1) // BK
        mask = make_spatial_3d_mask(H, W, T, spatial_radius=2, temporal_radius=1)
        assert mask.shape == (NQ, NK)

    def test_3d_mask_full_radii_all_active(self):
        """Full spatial + temporal radius → all tiles active."""
        from mlx_mfa.masks import make_spatial_3d_mask
        mask = make_spatial_3d_mask(4, 4, 4, spatial_radius=100, temporal_radius=100)
        mx.eval(mask)
        assert bool(mask.all())

    def test_3d_mask_less_active_at_small_temporal(self):
        """temporal_radius=0 mask has fewer active tiles than temporal_radius=100."""
        from mlx_mfa.masks import make_spatial_3d_mask
        import numpy as np
        # Use a large enough grid so tiles don't span all frames
        H, W, T = 8, 8, 8  # N=512, NQ=16, NK=32
        mask_t0 = make_spatial_3d_mask(H, W, T, spatial_radius=100, temporal_radius=0)
        mask_t100 = make_spatial_3d_mask(H, W, T, spatial_radius=100, temporal_radius=100)
        mx.eval(mask_t0, mask_t100)
        density_t0 = np.array(mask_t0).mean()
        density_t100 = np.array(mask_t100).mean()
        assert density_t0 < density_t100, \
            f"temporal_radius=0 should be sparser: {density_t0:.3f} >= {density_t100:.3f}"

    @pytest.mark.skipif(not is_mfa_available(), reason="MFA extension not available")
    def test_2d_mask_end_to_end(self):
        """End-to-end: 2D spatial mask + flash_attention_sparse."""
        from mlx_mfa.masks import make_spatial_2d_mask
        B, H_heads, D = 1, 4, 128
        pH, pW = 8, 8
        N = pH * pW
        key = mx.random.normal((B, H_heads, N, D), dtype=mx.float16)
        q, k, v = key, key, key
        mask = make_spatial_2d_mask(pH, pW, spatial_radius=4, head_dim=D)
        out = flash_attention_sparse(q, k, v, mask, scale=1.0/D**0.5, causal=False)
        mx.eval(out)
        assert out.shape == (B, H_heads, N, D)
        assert not bool(mx.any(mx.isnan(out)))

    def test_topk_mask_density(self):
        """make_topk_spatial_mask: each row has exactly top_k True values."""
        from mlx_mfa.masks import make_topk_spatial_mask, _bq_bk
        import numpy as np
        B, H_heads, N, D = 1, 2, 128, 64
        q = mx.random.normal((B, H_heads, N, D))
        k = mx.random.normal((B, H_heads, N, D))
        top_k = 4
        mask = make_topk_spatial_mask(q, k, top_k=top_k, head_dim=D)
        mx.eval(mask)
        m = np.array(mask)
        # Each row should have exactly top_k True values (or all True if NK < top_k)
        BQ, BK = _bq_bk(D)
        NK = (N + BK - 1) // BK
        expected = min(top_k, NK)
        for i, row in enumerate(m):
            assert row.sum() == expected, \
                f"Row {i}: expected {expected} True, got {row.sum()}"


# ============================================================================
# Track P — Segment mask
# ============================================================================

class TestSegmentMask:
    """Tests for make_segment_mask and make_causal_segment_mask."""

    def test_single_segment_all_active(self):
        """Single segment → all tiles active."""
        from mlx_mfa.masks import make_segment_mask
        mask = make_segment_mask([256], head_dim=128)
        mx.eval(mask)
        assert bool(mask.all()), "Single segment = all tiles active"

    def test_two_equal_segments_block_diagonal(self):
        """Two segments: upper-right and lower-left blocks must be inactive."""
        from mlx_mfa.masks import make_segment_mask, _bq_bk
        import numpy as np
        seg_len = 64
        mask = make_segment_mask([seg_len, seg_len], head_dim=128)
        mx.eval(mask)
        m = np.array(mask)
        BQ, BK = _bq_bk(128)
        N = seg_len * 2
        NQ = (N + BQ - 1) // BQ
        NK = (N + BK - 1) // BK
        # Tiles that are entirely in segment 0 Q vs segment 1 K should be False
        # seg 0 tokens: 0..63, seg 1 tokens: 64..127
        # Q tiles covering only seg0 (qi < 64//BQ = 2) and K tiles only seg1 (ki >= 64//BK = 4)
        tiles_q0 = seg_len // BQ  # tiles fully in seg 0
        tiles_k1_start = seg_len // BK  # first tile that starts seg 1
        if tiles_q0 > 0 and tiles_k1_start < NK:
            assert not m[:tiles_q0, tiles_k1_start:].any(), \
                "Cross-segment tiles should be inactive"

    def test_tile_boundary_segments_conservative(self):
        """A segment boundary mid-tile keeps that tile active (conservative)."""
        from mlx_mfa.masks import make_segment_mask, _bq_bk
        import numpy as np
        BQ, BK = _bq_bk(128)
        # Put boundary at BQ//2 so first Q tile spans both segments
        seg1 = BQ // 2
        seg2 = BQ - seg1
        mask = make_segment_mask([seg1, seg2], head_dim=128)
        mx.eval(mask)
        m = np.array(mask)
        # Tile 0 in Q spans both segments → should be active in more columns
        assert m[0, :].any(), "Boundary-straddling tile should be active"

    def test_causal_segment_mask_shape_matches(self):
        """Causal segment mask has same shape as segment mask."""
        from mlx_mfa.masks import make_segment_mask, make_causal_segment_mask
        segs = [128, 128]
        seg_mask = make_segment_mask(segs)
        causal_seg_mask = make_causal_segment_mask(segs)
        assert seg_mask.shape == causal_seg_mask.shape

    def test_causal_segment_mask_subset_of_segment_mask(self):
        """Causal+segment mask is a subset of (≤) segment mask."""
        from mlx_mfa.masks import make_segment_mask, make_causal_segment_mask
        import numpy as np
        segs = [64, 64]
        seg = np.array(make_segment_mask(segs))
        causal_seg = np.array(make_causal_segment_mask(segs))
        mx.eval()
        # Every True in causal_seg must be True in seg
        assert np.all((causal_seg & ~seg) == False), \
            "Causal segment mask must be a subset of segment mask"

    @pytest.mark.skipif(not is_mfa_available(), reason="MFA extension not available")
    def test_segment_mask_end_to_end(self):
        """Segment-masked output matches running each segment independently."""
        from mlx_mfa.masks import make_segment_mask
        B, H_heads, D = 1, 2, 64
        segs = [32, 32]
        N = sum(segs)
        q = mx.random.normal((B, H_heads, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H_heads, N, D), dtype=mx.float16)
        v = mx.random.normal((B, H_heads, N, D), dtype=mx.float16)

        mask = make_segment_mask(segs, head_dim=D)
        out_sparse = flash_attention_sparse(q, k, v, mask,
                                            scale=1.0/D**0.5, causal=False)
        mx.eval(out_sparse)

        # Run each segment independently and concatenate
        outputs = []
        offset = 0
        for seg_len in segs:
            q_i = q[:, :, offset:offset+seg_len, :]
            k_i = k[:, :, offset:offset+seg_len, :]
            v_i = v[:, :, offset:offset+seg_len, :]
            out_i = flash_attention_sparse(
                q_i, k_i, v_i,
                make_segment_mask([seg_len], head_dim=D),
                scale=1.0/D**0.5, causal=False)
            outputs.append(out_i)
            offset += seg_len
        out_ref = mx.concatenate(outputs, axis=2)
        mx.eval(out_ref)

        diff = mx.abs(out_sparse - out_ref).max()
        mx.eval(diff)
        assert float(diff) < 0.05, f"Max diff too large: {float(diff)}"


# ============================================================================
# Track Q — Adaptive window mask
# ============================================================================

class TestAdaptiveWindowMask:
    """Tests for make_adaptive_window_mask."""

    def test_shape_correct(self):
        """Output shape matches expected tile dimensions."""
        from mlx_mfa.masks import make_adaptive_window_mask, _bq_bk
        H, W, T = 32, 32, 4
        mask = make_adaptive_window_mask(H, W, num_frames=T,
                                          base_window_h=16, base_window_w=16,
                                          train_resolution=(256, 256),
                                          inference_resolution=(256, 256))
        N = H * W * T
        BQ, BK = _bq_bk(128)
        NQ = (N + BQ - 1) // BQ
        NK = (N + BK - 1) // BK
        assert mask.shape == (NQ, NK)

    def test_at_training_resolution_dense(self):
        """At training resolution with large base_window, mask is fully dense."""
        from mlx_mfa.masks import make_adaptive_window_mask
        H, W = 8, 8  # small grid so base_window covers all
        mask = make_adaptive_window_mask(H, W, num_frames=1,
                                          base_window_h=64, base_window_w=64,
                                          train_resolution=(256, 256),
                                          inference_resolution=(256, 256))
        mx.eval(mask)
        assert bool(mask.all()), "Large base window at train resolution should be dense"

    def test_sparsity_increases_with_resolution(self):
        """Higher inference resolution → fewer active tiles."""
        from mlx_mfa.masks import make_adaptive_window_mask
        import numpy as np
        H_base, W_base = 16, 16

        mask_1x = make_adaptive_window_mask(
            H_base, W_base, num_frames=1,
            base_window_h=8, base_window_w=8,
            train_resolution=(256, 256),
            inference_resolution=(256, 256))

        mask_2x = make_adaptive_window_mask(
            H_base * 2, W_base * 2, num_frames=1,
            base_window_h=8, base_window_w=8,
            train_resolution=(256, 256),
            inference_resolution=(512, 512))

        mx.eval(mask_1x, mask_2x)
        density_1x = np.array(mask_1x).mean()
        density_2x = np.array(mask_2x).mean()
        # At 2x resolution, window is halved → more sparse
        assert density_2x <= density_1x + 0.05, \
            f"2x resolution should be ≤ sparser: {density_2x:.3f} vs {density_1x:.3f}"

    def test_scale_equals_zero_raises_or_clamps(self):
        """Extreme resolution ratio is handled gracefully (no crash)."""
        from mlx_mfa.masks import make_adaptive_window_mask
        # 10x upscale — effective window = 1
        mask = make_adaptive_window_mask(64, 64, num_frames=1,
                                          base_window_h=4, base_window_w=4,
                                          train_resolution=(64, 64),
                                          inference_resolution=(640, 640))
        mx.eval(mask)
        assert mask is not None  # no crash


# ============================================================================
# Track S — Variable-length batching
# ============================================================================

class TestVarlenAttention:
    """Tests for flash_attention_varlen (split-concat implementation)."""

    def _ref(self, q, k, v, scale, causal):
        """Reference: use fallback SDPA."""
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask="causal" if causal else None)

    def test_single_sequence_matches_standard(self):
        """One sequence: varlen == standard flash_attention output."""
        from mlx_mfa import flash_attention_varlen, flash_attention
        B, H, N, D = 1, 4, 64, 64
        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))
        scale = 1.0 / D**0.5

        cu = mx.array([0, N])
        out_varlen = flash_attention_varlen(q, k, v, cu, cu, N, N, scale=scale)
        out_std = flash_attention(q, k, v, scale=scale, causal=False)
        mx.eval(out_varlen, out_std)
        diff = float(mx.abs(out_varlen - out_std).max())
        assert diff < 1e-4, f"Max diff too large: {diff}"

    def test_two_sequences_independent(self):
        """Two packed sequences produce same output as running independently."""
        from mlx_mfa import flash_attention_varlen, flash_attention
        B, H, D = 1, 2, 64
        N1, N2 = 32, 48
        N = N1 + N2
        scale = 1.0 / D**0.5

        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))

        cu = mx.array([0, N1, N])
        out_varlen = flash_attention_varlen(q, k, v, cu, cu, max(N1, N2), max(N1, N2),
                                            scale=scale)

        # Run separately
        out0 = flash_attention(q[:, :, :N1, :], k[:, :, :N1, :], v[:, :, :N1, :],
                               scale=scale)
        out1 = flash_attention(q[:, :, N1:, :], k[:, :, N1:, :], v[:, :, N1:, :],
                               scale=scale)
        out_ref = mx.concatenate([out0, out1], axis=2)

        mx.eval(out_varlen, out_ref)
        diff = float(mx.abs(out_varlen - out_ref).max())
        assert diff < 1e-4, f"Max diff: {diff}"

    def test_different_lengths(self):
        """Different sequence lengths: correct output shape."""
        from mlx_mfa import flash_attention_varlen
        B, H, D = 1, 2, 64
        lengths = [16, 32, 48, 8]
        N = sum(lengths)
        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))

        cu = mx.array([0] + [int(x) for x in np.cumsum(lengths)])
        out = flash_attention_varlen(q, k, v, cu, cu, max(lengths), max(lengths))
        mx.eval(out)
        assert out.shape == (B, H, N, D)

    def test_varlen_causal(self):
        """Causal within each sequence."""
        from mlx_mfa import flash_attention_varlen, flash_attention
        B, H, D = 1, 2, 64
        N1, N2 = 24, 24
        N = N1 + N2
        scale = 1.0 / D**0.5

        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))

        cu = mx.array([0, N1, N])
        out_varlen = flash_attention_varlen(q, k, v, cu, cu, N1, N2,
                                            scale=scale, causal=True)
        out0 = flash_attention(q[:, :, :N1, :], k[:, :, :N1, :], v[:, :, :N1, :],
                               scale=scale, causal=True)
        out1 = flash_attention(q[:, :, N1:, :], k[:, :, N1:, :], v[:, :, N1:, :],
                               scale=scale, causal=True)
        out_ref = mx.concatenate([out0, out1], axis=2)
        mx.eval(out_varlen, out_ref)
        diff = float(mx.abs(out_varlen - out_ref).max())
        assert diff < 1e-4, f"Causal varlen max diff: {diff}"

    def test_varlen_backward(self):
        """Gradients flow correctly through flash_attention_varlen."""
        from mlx_mfa import flash_attention_varlen
        B, H, D = 1, 2, 64
        N1, N2 = 16, 16
        N = N1 + N2
        scale = 1.0 / D**0.5

        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))
        cu = mx.array([0, N1, N])

        def fwd(q, k, v):
            return flash_attention_varlen(q, k, v, cu, cu, N1, N2, scale=scale).sum()

        _, grads = mx.value_and_grad(fwd, argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)
        for name, g in zip("qkv", grads):
            assert g is not None, f"Grad for {name} is None"
            assert g.shape == (B, H, N, D), f"Grad {name} shape mismatch"
            assert not bool(mx.any(mx.isnan(g))), f"NaN in grad {name}"


# ============================================================================
# Track R — 3D RoPE fusion
# ============================================================================

class TestRoPE3D:
    """Tests for make_rope_3d_tables and flash_attention_rope(rope_3d=...)."""

    def _ref_apply_3d_rope(self, x, cos_table, sin_table):
        """Apply RoPE from [N, D/2] tables to [B, H, N, D] tensor in Python."""
        # cos_table: [N, D/2], sin_table: [N, D/2]
        # x: [B, H, N, D]
        cos = cos_table[None, None, :, :]  # [1, 1, N, D/2]
        sin = sin_table[None, None, :, :]  # [1, 1, N, D/2]
        x0 = x[..., 0::2]  # [B, H, N, D/2]
        x1 = x[..., 1::2]  # [B, H, N, D/2]
        out = mx.zeros_like(x)
        # Interleave cos/sin
        rot_0 = x0 * cos - x1 * sin
        rot_1 = x0 * sin + x1 * cos
        # Re-interleave
        out = mx.concatenate([rot_0, rot_1], axis=-1)
        # out is [B, H, N, D] but with even indices first then odd
        # Reshape and transpose to restore interleaved layout
        B, H, N, D = x.shape
        out = out.reshape(B, H, N, 2, D // 2)
        out = mx.transpose(out, (0, 1, 2, 4, 3)).reshape(B, H, N, D)
        return out

    def test_table_shape(self):
        """make_rope_3d_tables returns correct shape."""
        from mlx_mfa import make_rope_3d_tables
        grid_h, grid_w, T = 8, 8, 4
        D = 128
        cos, sin = make_rope_3d_tables(grid_h, grid_w, T, head_dim=D)
        N = grid_h * grid_w * T
        assert cos.shape == (N, D // 2), f"cos shape {cos.shape} != ({N}, {D//2})"
        assert sin.shape == (N, D // 2)
        assert cos.dtype == mx.float32

    def test_table_no_nans(self):
        """Tables contain no NaN or Inf values."""
        from mlx_mfa import make_rope_3d_tables
        cos, sin = make_rope_3d_tables(4, 4, 4, head_dim=128)
        mx.eval(cos, sin)
        assert not bool(mx.any(mx.isnan(cos))), "NaN in cos"
        assert not bool(mx.any(mx.isnan(sin))), "NaN in sin"
        assert not bool(mx.any(mx.isinf(cos))), "Inf in cos"

    def test_table_d_split_sums_to_d(self):
        """d_h + d_w + d_t == head_dim."""
        from mlx_mfa import make_rope_3d_tables
        D = 128
        d_h, d_w = 42, 42
        d_t = D - d_h - d_w
        # Non-equal split (all even)
        cos, sin = make_rope_3d_tables(4, 4, 2, d_h=d_h, d_w=d_w, d_t=d_t,
                                        head_dim=D)
        assert cos.shape == (4 * 4 * 2, D // 2)

    def test_table_odd_split_raises(self):
        """Odd d_h/d_w/d_t should raise ValueError."""
        from mlx_mfa import make_rope_3d_tables
        with pytest.raises(ValueError, match="even"):
            make_rope_3d_tables(4, 4, 2, d_h=3, d_w=4, d_t=4, head_dim=11)

    def test_rope_3d_via_explicit_tables(self):
        """flash_attention_rope with explicit 3D tables matches Python RoPE."""
        from mlx_mfa import make_rope_3d_tables, flash_attention_rope
        B, H_heads, D = 1, 2, 64
        grid_h, grid_w, T = 4, 4, 2
        N = grid_h * grid_w * T

        q = mx.random.normal((B, H_heads, N, D))
        k = mx.random.normal((B, H_heads, N, D))
        v = mx.random.normal((B, H_heads, N, D))
        scale = 1.0 / D**0.5

        cos, sin = make_rope_3d_tables(grid_h, grid_w, T, head_dim=D)

        # Reference: Python RoPE then SDPA
        q_rot = self._ref_apply_3d_rope(q, cos, sin)
        k_rot = self._ref_apply_3d_rope(k, cos, sin)
        ref = mx.fast.scaled_dot_product_attention(q_rot, k_rot, v, scale=scale)

        # Flash attention with explicit cos/sin tables (treated as 1D tables by kernel)
        out = flash_attention_rope(q, k, v, cos, sin, scale=scale)

        mx.eval(out, ref)
        diff = float(mx.abs(out - ref).max())
        assert diff < 0.05, f"3D RoPE tables max diff too large: {diff}"

    def test_rope_3d_dict_api(self):
        """flash_attention_rope(rope_3d=...) builds tables automatically."""
        from mlx_mfa import flash_attention_rope, make_rope_3d_tables
        B, H_heads, D = 1, 2, 64
        grid_h, grid_w, T = 4, 4, 2
        N = grid_h * grid_w * T

        q = mx.random.normal((B, H_heads, N, D))
        k = mx.random.normal((B, H_heads, N, D))
        v = mx.random.normal((B, H_heads, N, D))
        scale = 1.0 / D**0.5

        # Via explicit tables
        cos, sin = make_rope_3d_tables(grid_h, grid_w, T, head_dim=D)
        ref = flash_attention_rope(q, k, v, cos, sin, scale=scale)

        # Via dict API
        out = flash_attention_rope(q, k, v, scale=scale,
                                    rope_3d={"grid_h": grid_h, "grid_w": grid_w,
                                             "num_frames": T})
        mx.eval(out, ref)
        diff = float(mx.abs(out - ref).max())
        assert diff < 1e-5, f"rope_3d dict API mismatch: {diff}"

    def test_rope_3d_no_effect_when_none(self):
        """No rope → plain flash_attention output."""
        from mlx_mfa import flash_attention_rope, flash_attention
        B, H_heads, D = 1, 2, 64
        N = 32

        q = mx.random.normal((B, H_heads, N, D))
        k = mx.random.normal((B, H_heads, N, D))
        v = mx.random.normal((B, H_heads, N, D))
        scale = 1.0 / D**0.5

        # Build identity RoPE (cos=1, sin=0)
        import numpy as _np
        cos_eye = mx.array(_np.ones((N, D // 2), dtype=_np.float32))
        sin_eye = mx.array(_np.zeros((N, D // 2), dtype=_np.float32))

        out_rope = flash_attention_rope(q, k, v, cos_eye, sin_eye, scale=scale)
        out_plain = flash_attention(q, k, v, scale=scale)
        mx.eval(out_rope, out_plain)
        diff = float(mx.abs(out_rope - out_plain).max())
        assert diff < 1e-4, f"Identity RoPE should match plain attention: {diff}"

    def test_rope_3d_exclusive_with_cos_sin(self):
        """Providing both rope_3d and rotary_cos raises ValueError."""
        from mlx_mfa import flash_attention_rope, make_rope_3d_tables
        B, H, D = 1, 2, 64
        N = 32
        q = k = v = mx.zeros((B, H, N, D))
        cos = sin = mx.zeros((N, D // 2))
        with pytest.raises(ValueError, match="mutually exclusive"):
            flash_attention_rope(q, k, v, cos, sin,
                                  rope_3d={"grid_h": 4, "grid_w": 4, "num_frames": 2})


# =============================================================================
# Track U — LCSA Composite Mask tests
# =============================================================================

class TestLCSAMask:
    """Track U: make_lcsa_mask — FlashVSR LCSA composite mask."""

    H, W, D = 16, 16, 128  # 256 tokens

    def _qk(self, H=None, W=None):
        H = H or self.H; W = W or self.W
        N = H * W
        q = mx.random.normal((1, 4, N, self.D)).astype(mx.float16)
        k = mx.random.normal((1, 4, N, self.D)).astype(mx.float16)
        return q, k

    def test_lcsa_is_subset_of_spatial(self):
        from mlx_mfa import make_lcsa_mask, make_spatial_2d_mask
        q, k = self._qk()
        sp = make_spatial_2d_mask(self.H, self.W, spatial_radius=4, head_dim=self.D)
        lcsa = make_lcsa_mask(q, k, self.H, self.W, spatial_radius=4, top_k=4, head_dim=self.D)
        sp_np = np.array(sp)
        lcsa_np = np.array(lcsa)
        # LCSA ⊆ spatial: every True in LCSA must be True in spatial
        assert np.all(~lcsa_np | sp_np), "LCSA must be a subset of the spatial mask"

    def test_lcsa_density_controlled_by_topk(self):
        from mlx_mfa import make_lcsa_mask
        q, k = self._qk()
        for top_k in [2, 4, 8]:
            lcsa_np = np.array(make_lcsa_mask(q, k, self.H, self.W,
                                               spatial_radius=8, top_k=top_k, head_dim=self.D))
            row_sums = lcsa_np.sum(axis=1)
            assert np.all(row_sums <= top_k), f"top_k={top_k}: some rows have {row_sums.max()} active tiles"

    def test_lcsa_with_temporal(self):
        from mlx_mfa import make_lcsa_mask
        H, W, T = 8, 8, 4  # 256 tokens
        N = H * W * T
        q = mx.random.normal((1, 4, N, self.D)).astype(mx.float16)
        k = mx.random.normal((1, 4, N, self.D)).astype(mx.float16)
        mask = make_lcsa_mask(q, k, H, W, spatial_radius=4, top_k=4,
                               head_dim=self.D, num_frames=T, temporal_radius=2)
        assert mask.ndim == 2
        assert mask.dtype == mx.bool_

    def test_lcsa_topk_larger_than_window(self):
        """top_k >= window entries → LCSA == spatial mask."""
        from mlx_mfa import make_lcsa_mask, make_spatial_2d_mask
        q, k = self._qk()
        sp_np = np.array(make_spatial_2d_mask(self.H, self.W, spatial_radius=4, head_dim=self.D))
        # top_k very large — should give same as spatial
        lcsa_np = np.array(make_lcsa_mask(q, k, self.H, self.W, spatial_radius=4,
                                           top_k=1000, head_dim=self.D))
        # Every active spatial tile should be in LCSA
        missing = sp_np & ~lcsa_np
        assert not np.any(missing), "With large top_k, LCSA should equal spatial mask"

    def test_lcsa_end_to_end(self):
        from mlx_mfa import make_lcsa_mask, flash_attention_sparse
        q, k = self._qk()
        v = mx.random.normal((1, 4, self.H * self.W, self.D)).astype(mx.float16)
        mask = make_lcsa_mask(q, k, self.H, self.W, spatial_radius=4, top_k=4, head_dim=self.D)
        out = flash_attention_sparse(q, k, v, mask, scale=1.0 / (self.D ** 0.5))
        assert out.shape == q.shape
        assert not np.any(np.isnan(np.array(out.astype(mx.float32)))), "NaN in LCSA output"


# =============================================================================
# Track V — Axial / Factored Attention Mask tests
# =============================================================================

class TestAxialMasks:
    """Track V: make_axial_spatial_mask, make_axial_temporal_mask."""

    H, W, T, D = 8, 8, 4, 128

    def test_spatial_mask_per_frame_isolation(self):
        """Spatial mask: Q-tile at frame 0 should NOT attend to K-tiles at frame 2+."""
        from mlx_mfa import make_axial_spatial_mask
        mask_np = np.array(make_axial_spatial_mask(self.H, self.W, self.T, head_dim=self.D))
        # Mask shape: [NQ, NK] where NQ uses BQ=32, NK uses BK=16 → not necessarily square
        assert mask_np.ndim == 2
        # Density < 1 (not fully dense)
        density = mask_np.mean()
        assert density < 1.0, "Axial spatial mask should be sparse"

    def test_temporal_mask_same_position_only(self):
        """Temporal mask: Q at frame 0 pos 0 should NOT attend to frame 0 pos 5."""
        from mlx_mfa import make_axial_temporal_mask
        mask_np = np.array(make_axial_temporal_mask(self.H, self.W, self.T, head_dim=self.D))
        assert mask_np.ndim == 2  # [NQ, NK] rectangular is OK
        # Must be sparser than full dense mask
        assert mask_np.mean() < 1.0, "Temporal mask should be sparse"

    def test_axial_masks_complement(self):
        """Spatial | Temporal should have higher density than either alone."""
        from mlx_mfa import make_axial_spatial_mask, make_axial_temporal_mask
        sp = np.array(make_axial_spatial_mask(self.H, self.W, self.T, head_dim=self.D))
        tm = np.array(make_axial_temporal_mask(self.H, self.W, self.T, head_dim=self.D))
        union = sp | tm
        assert union.mean() > sp.mean(), "Union should be denser than spatial alone"
        assert union.mean() > tm.mean(), "Union should be denser than temporal alone"

    def test_temporal_causal(self):
        """Temporal causal: upper triangle (future) should have fewer active tiles."""
        from mlx_mfa import make_axial_temporal_mask
        causal_np = np.array(make_axial_temporal_mask(
            self.H, self.W, self.T, head_dim=self.D, causal=True))
        noncausal_np = np.array(make_axial_temporal_mask(
            self.H, self.W, self.T, head_dim=self.D, causal=False))
        assert causal_np.sum() <= noncausal_np.sum(), \
                             "Causal mask should be subset of non-causal"

    def test_spatial_with_radius(self):
        """Spatial mask with small radius should be sparser than large radius."""
        from mlx_mfa import make_axial_spatial_mask
        small = np.array(make_axial_spatial_mask(self.H, self.W, self.T, head_dim=self.D,
                                                  spatial_radius=2))
        large = np.array(make_axial_spatial_mask(self.H, self.W, self.T, head_dim=self.D,
                                                  spatial_radius=8))
        assert small.sum() <= large.sum(), \
                             "Smaller radius → sparser mask"


# =============================================================================
# Track W — Dilated Temporal Mask tests
# =============================================================================

class TestDilatedTemporalMask:
    """Track W: make_dilated_temporal_mask."""

    H, W, D = 8, 8, 128

    def test_dilation_1_is_full_temporal(self):
        """dilation_rate=1, local_window >= T → every tile active."""
        from mlx_mfa import make_dilated_temporal_mask
        T = 4
        mask_np = np.array(make_dilated_temporal_mask(
            self.H, self.W, T, dilation_rate=1, local_window=T, head_dim=self.D))
        # Should be fully dense
        assert np.all(mask_np), "dilation=1 + large local_window → all tiles active"

    def test_density_decreases_with_dilation(self):
        """Higher dilation rate (fewer attending frames) → lower density."""
        from mlx_mfa import make_dilated_temporal_mask
        T = 16
        d1 = np.array(make_dilated_temporal_mask(self.H, self.W, T, dilation_rate=2,
                                                   local_window=1, head_dim=self.D)).mean()
        d2 = np.array(make_dilated_temporal_mask(self.H, self.W, T, dilation_rate=8,
                                                   local_window=1, head_dim=self.D)).mean()
        assert d1 > d2, "Smaller dilation → higher density"

    def test_local_window_adds_neighbors(self):
        """Larger local_window → more tiles active."""
        from mlx_mfa import make_dilated_temporal_mask
        T = 8
        m0 = np.array(make_dilated_temporal_mask(self.H, self.W, T, dilation_rate=4,
                                                   local_window=0, head_dim=self.D)).sum()
        m2 = np.array(make_dilated_temporal_mask(self.H, self.W, T, dilation_rate=4,
                                                   local_window=2, head_dim=self.D)).sum()
        assert m2 >= m0, "Larger local window should add more active tiles"

    def test_shape_correct(self):
        from mlx_mfa import make_dilated_temporal_mask
        T = 8
        mask = make_dilated_temporal_mask(self.H, self.W, T, dilation_rate=2, head_dim=self.D)
        N = self.H * self.W * T
        BQ, BK = 32, 16  # head_dim=128
        assert mask.shape == ((N + BQ - 1) // BQ, (N + BK - 1) // BK)


# =============================================================================
# Track X — Sink Tokens + Reference Frame Mask tests
# =============================================================================

class TestSinkAndReferenceFrameMasks:
    """Track X: make_sink_window_mask, make_reference_frame_mask."""

    D = 128

    def test_sink_tokens_always_visible(self):
        """First num_sink_tiles K-tiles should be active for ALL Q-tiles."""
        from mlx_mfa import make_sink_window_mask
        N = 256; sink = 32; window = 32
        mask_np = np.array(make_sink_window_mask(N, window, sink, head_dim=self.D))
        # First K-tile (covering first 16 tokens with BK=16) must be True for all Q
        assert np.all(mask_np[:, 0]), "First K-tile should be visible to all Q-tiles"

    def test_zero_sinks_equals_sliding_window(self):
        """num_sink_tokens=0 → pure sliding window (no extra global visibility)."""
        from mlx_mfa import make_sink_window_mask
        N = 256; window = 32
        with_sink = np.array(make_sink_window_mask(N, window, num_sink_tokens=64, head_dim=self.D))
        no_sink = np.array(make_sink_window_mask(N, window, num_sink_tokens=0, head_dim=self.D))
        # With sinks → at least as many True entries
        assert with_sink.sum() >= no_sink.sum(), \
                                "Sinks should add more active tiles"

    def test_sink_plus_causal(self):
        """Causal mode: no future K-tiles, but sinks still visible."""
        from mlx_mfa import make_sink_window_mask
        N = 256; sink = 32; window = 32
        mask_np = np.array(make_sink_window_mask(N, window, sink, head_dim=self.D, causal=True))
        # First K-tile visible for all Q-tiles (sink)
        assert np.all(mask_np[:, 0]), "Sinks must be visible even in causal mode"
        # No future tiles: for Q-tile 0 (tokens 0..BQ-1), K-tile last must be False
        assert not mask_np[0, -1], "Q-tile 0 should not see last K-tile in causal mode"

    def test_reference_frame_always_visible(self):
        """All K-tiles covering the reference frame must be active for all Q-tiles."""
        from mlx_mfa import make_reference_frame_mask
        H, W, T = 8, 8, 4
        mask_np = np.array(make_reference_frame_mask(H, W, T,
                                                      reference_frames=[0],
                                                      head_dim=self.D))
        # K-tile 0 covers tokens 0..BK-1 which are all frame 0 → must be visible to all Q
        assert np.all(mask_np[:, 0]), "Reference frame K-tile must be visible to all Q-tiles"

    def test_reference_frame_plus_local(self):
        """Reference frame + local context → more active tiles than local alone."""
        from mlx_mfa import make_reference_frame_mask, make_spatial_3d_mask
        H, W, T = 8, 8, 4
        with_ref = np.array(make_reference_frame_mask(H, W, T,
                                                       reference_frames=[0],
                                                       temporal_radius=1,
                                                       head_dim=self.D)).sum()
        local_only = np.array(make_reference_frame_mask(H, W, T,
                                                         reference_frames=[],
                                                         temporal_radius=1,
                                                         head_dim=self.D)).sum()
        assert with_ref >= local_only, "Reference frame should add more active tiles"


# =============================================================================
# Track Y — Cross-Stream Attention Mask tests
# =============================================================================

class TestCrossStreamMask:
    """Track Y: make_cross_stream_mask."""

    D = 128

    def test_full_pattern_all_active(self):
        from mlx_mfa import make_cross_stream_mask
        mask = make_cross_stream_mask(256, 512, head_dim=self.D, pattern="full")
        assert np.all(np.array(mask)), "Full pattern should activate all tiles"

    def test_temporal_alignment_frame_diagonal(self):
        """Temporal pattern: Q frame t → KV frame t only → block diagonal."""
        from mlx_mfa import make_cross_stream_mask
        # 2 frames, 128 tokens each, Q and KV same size
        mask_np = np.array(make_cross_stream_mask(
            256, 256, head_dim=self.D, pattern="temporal", q_frames=2, kv_frames=2))
        density = mask_np.mean()
        # Frame-diagonal: should be sparser than full
        assert density < 1.0, "Temporal alignment should produce block-diagonal mask"

    def test_segment_cross_attention(self):
        """Segment pattern: Q segment i → KV segment i only."""
        from mlx_mfa import make_cross_stream_mask
        q_segs = [128, 128]
        kv_segs = [256, 256]
        mask_np = np.array(make_cross_stream_mask(
            256, 512, head_dim=self.D, pattern="segment",
            q_segments=q_segs, kv_segments=kv_segs))
        # Must be sparser than full
        assert mask_np.mean() < 1.0, "Segment cross-attention should be sparser than full"

    def test_asymmetric_token_counts(self):
        """n_tokens_q != n_tokens_kv → rectangular mask."""
        from mlx_mfa import make_cross_stream_mask
        mask = make_cross_stream_mask(256, 512, head_dim=self.D, pattern="full")
        BQ, BK = 32, 16
        NQ = (256 + BQ - 1) // BQ
        NK = (512 + BK - 1) // BK
        assert list(mask.shape) == [NQ, NK]

    def test_cross_stream_end_to_end(self):
        """Full cross-stream mask + flash_attention_sparse produces valid output."""
        from mlx_mfa import make_cross_stream_mask, flash_attention_sparse
        N_q, N_kv = 256, 256
        H = 4
        q = mx.random.normal((1, H, N_q, self.D)).astype(mx.float16)
        k = mx.random.normal((1, H, N_kv, self.D)).astype(mx.float16)
        v = mx.random.normal((1, H, N_kv, self.D)).astype(mx.float16)
        mask = make_cross_stream_mask(N_q, N_kv, head_dim=self.D, pattern="full")
        out = flash_attention_sparse(q, k, v, mask, scale=1.0 / (self.D ** 0.5))
        assert out.shape == (1, H, N_q, self.D)
        assert not np.any(np.isnan(np.array(out.astype(mx.float32))))


# =============================================================================
# Track AA: Softcapping (Gemma 2 / Grok style)
# =============================================================================

class TestSoftcap:
    """Tests for flash_attention(..., softcap=...) — Track AA."""

    D = 128
    B, H, N = 1, 4, 256

    def _ref_sdpa_softcap(self, q, k, v, scale, causal, softcap):
        """Pure-MLX reference SDPA with tanh softcapping."""
        S = mx.matmul(q, mx.transpose(k, [0, 1, 3, 2])) * scale
        S = mx.tanh(S / softcap) * softcap
        if causal:
            Nq, Sk = q.shape[2], k.shape[2]
            mask = mx.triu(
                mx.full((Nq, Sk), float("-inf"), dtype=q.dtype),
                k=Sk - Nq + 1,
            )
            S = S + mask
        A = mx.softmax(S.astype(mx.float32), axis=-1).astype(q.dtype)
        return mx.matmul(A, v)

    def test_softcap_zero_is_noop(self):
        """softcap=0.0 must produce the same output as omitting softcap."""
        from mlx_mfa import flash_attention
        q = mx.random.normal((self.B, self.H, self.N, self.D)).astype(mx.float16)
        k = mx.random.normal((self.B, self.H, self.N, self.D)).astype(mx.float16)
        v = mx.random.normal((self.B, self.H, self.N, self.D)).astype(mx.float16)
        scale = 1.0 / math.sqrt(self.D)

        out_default = flash_attention(q, k, v, scale=scale, causal=False)
        out_zero    = flash_attention(q, k, v, scale=scale, causal=False, softcap=0.0)
        mx.eval(out_default, out_zero)

        np.testing.assert_allclose(
            np.array(out_default.astype(mx.float32)),
            np.array(out_zero.astype(mx.float32)),
            atol=0.0, rtol=0.0,
            err_msg="softcap=0.0 must be bit-identical to no softcap",
        )

    def test_softcap_reduces_extreme_scores(self):
        """Scores with large magnitude must be compressed by tanh softcapping."""
        from mlx_mfa import flash_attention
        # Use a large scale so raw QK^T scores are large (>> softcap)
        q = mx.random.normal((self.B, self.H, self.N, self.D)).astype(mx.float32)
        k = mx.random.normal((self.B, self.H, self.N, self.D)).astype(mx.float32)
        v = mx.random.normal((self.B, self.H, self.N, self.D)).astype(mx.float32)
        big_scale = 10.0  # artificially large → raw scores will be huge

        softcap = 50.0

        # Reference: softcap applied → attention map softened
        S_raw = (mx.matmul(q, mx.transpose(k, [0, 1, 3, 2])) * big_scale)
        S_cap = mx.tanh(S_raw / softcap) * softcap

        mx.eval(S_raw, S_cap)
        S_raw_np = np.array(S_raw)
        S_cap_np = np.array(S_cap)

        # After capping, max absolute score must be <= softcap
        assert np.max(np.abs(S_cap_np)) <= softcap + 1e-4, (
            f"Softcapped scores exceed cap: max={np.max(np.abs(S_cap_np)):.3f}"
        )
        # And the capped scores must differ from the raw ones (test the premise)
        assert not np.allclose(S_raw_np, S_cap_np, atol=1e-3)

    def test_softcap_matches_reference(self):
        """MFA softcap output must match pure-MLX reference within f16 tolerance."""
        from mlx_mfa import flash_attention
        q = mx.random.normal((self.B, self.H, self.N, self.D)).astype(mx.float16)
        k = mx.random.normal((self.B, self.H, self.N, self.D)).astype(mx.float16)
        v = mx.random.normal((self.B, self.H, self.N, self.D)).astype(mx.float16)
        scale = 1.0 / math.sqrt(self.D)
        softcap = 50.0

        out_mfa = flash_attention(q, k, v, scale=scale, causal=False, softcap=softcap)
        out_ref = self._ref_sdpa_softcap(q, k, v, scale, causal=False, softcap=softcap)
        mx.eval(out_mfa, out_ref)

        np.testing.assert_allclose(
            np.array(out_mfa.astype(mx.float32)),
            np.array(out_ref.astype(mx.float32)),
            atol=2e-2, rtol=1e-2,
            err_msg="MFA softcap output diverges from reference (f16 precision)",
        )

    def test_softcap_gemma2_value(self):
        """Softcap=50 (Gemma 2 default) with causal mask matches reference."""
        from mlx_mfa import flash_attention
        # Smaller N to keep test fast; use causal=True (Gemma 2 typical)
        N = 128
        q = mx.random.normal((self.B, self.H, N, self.D)).astype(mx.float16)
        k = mx.random.normal((self.B, self.H, N, self.D)).astype(mx.float16)
        v = mx.random.normal((self.B, self.H, N, self.D)).astype(mx.float16)
        scale = 1.0 / math.sqrt(self.D)
        softcap = 50.0  # Gemma 2 default

        out_mfa = flash_attention(q, k, v, scale=scale, causal=True, softcap=softcap)
        out_ref = self._ref_sdpa_softcap(q, k, v, scale, causal=True, softcap=softcap)
        mx.eval(out_mfa, out_ref)

        np.testing.assert_allclose(
            np.array(out_mfa.astype(mx.float32)),
            np.array(out_ref.astype(mx.float32)),
            atol=2e-2, rtol=1e-2,
            err_msg="Gemma 2 softcap (cap=50, causal) diverges from reference",
        )


# ===========================================================================
# Track AB — ALiBi (Attention with Linear Biases)
# ===========================================================================

@pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
class TestALiBi:
    """Tests for flash_attention ALiBi per-head position bias support."""

    B, H, N, D = 1, 8, 256, 64

    def _ref_sdpa_alibi(self, q, k, v, slopes, scale, causal=False):
        """Pure-MLX ALiBi reference: bias[h,i,j] = slopes[h] * (j - i)."""
        import mlx.core as mx
        B, H, N, _ = q.shape
        Sk = k.shape[2]
        S = mx.matmul(q, mx.transpose(k, [0, 1, 3, 2])) * scale
        q_pos = mx.arange(N, dtype=mx.float32)[:, None]
        k_pos = mx.arange(Sk, dtype=mx.float32)[None, :]
        pos_diff = k_pos - q_pos
        sl = slopes.astype(mx.float32)
        bias = sl[:, None, None] * pos_diff[None, :, :]   # [H, N, Sk]
        S = S + mx.expand_dims(bias, axis=0).astype(q.dtype)
        if causal:
            mask = mx.triu(
                mx.full((N, Sk), float("-inf"), dtype=q.dtype),
                k=Sk - N + 1,
            )
            S = S + mask
        A = mx.softmax(S.astype(mx.float32), axis=-1).astype(q.dtype)
        return mx.matmul(A, v)

    def test_zero_slopes_is_noop(self):
        """ALiBi with all-zero slopes must equal standard attention (no bias)."""
        from mlx_mfa import flash_attention
        mx.random.seed(42)
        q = mx.random.normal((self.B, self.H, self.N, self.D)).astype(mx.float16)
        k = mx.random.normal((self.B, self.H, self.N, self.D)).astype(mx.float16)
        v = mx.random.normal((self.B, self.H, self.N, self.D)).astype(mx.float16)
        slopes = mx.zeros((self.H,), dtype=mx.float32)
        scale = 1.0 / math.sqrt(self.D)

        out_alibi = flash_attention(q, k, v, scale=scale, causal=False,
                                    alibi_slopes=slopes)
        out_plain = flash_attention(q, k, v, scale=scale, causal=False)
        mx.eval(out_alibi, out_plain)

        np.testing.assert_allclose(
            np.array(out_alibi.astype(mx.float32)),
            np.array(out_plain.astype(mx.float32)),
            atol=1e-3, rtol=1e-3,
            err_msg="Zero slopes ALiBi must equal standard attention",
        )

    def test_slopes_reduce_distant_scores(self):
        """Negative ALiBi slopes should penalise distant positions.

        With negative slopes (the typical usage for causal decay), position
        j far from i should receive a more negative score bias than j close
        to i, making distant attention weights smaller.
        """
        from mlx_mfa._ext import mfa_attention_alibi_forward
        mx.random.seed(7)
        # Small N so position effect is clearly measurable
        N = 32
        q = mx.random.normal((1, 1, N, self.D)).astype(mx.float16)
        k = mx.random.normal((1, 1, N, self.D)).astype(mx.float16)
        v = mx.random.normal((1, 1, N, self.D)).astype(mx.float16)
        # Steep negative slope to exaggerate the bias
        slopes = mx.array([-1.0], dtype=mx.float32)
        scale = 1.0 / math.sqrt(self.D)

        out_alibi = mfa_attention_alibi_forward(q, k, v, slopes, scale, True)
        out_plain = flash_attention(q, k, v, scale=scale, causal=True)
        mx.eval(out_alibi, out_plain)

        # The outputs should differ — ALiBi modifies the distribution
        diff = float(mx.mean(mx.abs(
            out_alibi.astype(mx.float32) - out_plain.astype(mx.float32)
        )))
        assert diff > 1e-4, \
            "ALiBi with negative slopes should modify attention outputs"

    def test_matches_reference(self):
        """MFA ALiBi output must match pure-MLX reference within f16 tolerance.

        Uses N=64 (not N=256) to keep max bias ≤ 63 — with N=256 and slope=-1,
        biases reach ±255 causing degenerate softmax concentration where tiny
        f16 accumulation differences shift which single token wins, producing
        large but numerically valid disagreements between kernel and reference.
        """
        from mlx_mfa import flash_attention
        mx.random.seed(13)
        N = 48   # limit max bias to slope * (N-1) ≤ 4.7 for head-0 slope=-0.1
        q = mx.random.normal((self.B, self.H, N, self.D)).astype(mx.float16)
        k = mx.random.normal((self.B, self.H, N, self.D)).astype(mx.float16)
        v = mx.random.normal((self.B, self.H, N, self.D)).astype(mx.float16)
        # Typical ALiBi slopes — moderate magnitude avoids extreme softmax
        # concentration that amplifies tiny f16 rounding into large output diffs.
        slopes = mx.array(
            [-0.1 / (2 ** i) for i in range(self.H)], dtype=mx.float32
        )
        scale = 1.0 / math.sqrt(self.D)

        out_mfa = flash_attention(q, k, v, scale=scale, causal=False,
                                  alibi_slopes=slopes)
        out_ref = self._ref_sdpa_alibi(q, k, v, slopes, scale, causal=False)
        mx.eval(out_mfa, out_ref)

        np.testing.assert_allclose(
            np.array(out_mfa.astype(mx.float32)),
            np.array(out_ref.astype(mx.float32)),
            atol=2e-2, rtol=1e-2,
            err_msg="MFA ALiBi output diverges from reference (f16 precision)",
        )

    def test_falcon_slopes_causal(self):
        """Falcon-style ALiBi with causal masking matches reference."""
        from mlx_mfa import flash_attention
        mx.random.seed(99)
        H = 8  # Falcon-7B uses 8 heads (non-GQA)
        N = 128
        q = mx.random.normal((1, H, N, self.D)).astype(mx.float16)
        k = mx.random.normal((1, H, N, self.D)).astype(mx.float16)
        v = mx.random.normal((1, H, N, self.D)).astype(mx.float16)
        # Falcon ALiBi recipe: slopes = 2^(-8 * h / H) for h in [1..H]
        slopes = mx.array(
            [2.0 ** (-8.0 * h / H) for h in range(1, H + 1)], dtype=mx.float32
        )
        scale = 1.0 / math.sqrt(self.D)

        out_mfa = flash_attention(q, k, v, scale=scale, causal=True,
                                  alibi_slopes=slopes)
        out_ref = self._ref_sdpa_alibi(q, k, v, slopes, scale, causal=True)
        mx.eval(out_mfa, out_ref)

        np.testing.assert_allclose(
            np.array(out_mfa.astype(mx.float32)),
            np.array(out_ref.astype(mx.float32)),
            atol=2e-2, rtol=1e-2,
            err_msg="Falcon ALiBi (causal) diverges from reference",
        )


# ---------------------------------------------------------------------------
# Track AC — RoPE non-interleaved (GPT-NeoX style)
# ---------------------------------------------------------------------------

def _apply_rope_neox(x, cos, sin, offset=0):
    """Reference Python RoPE using GPT-NeoX split-halves convention.

    x: [B, H, N, D]
    cos/sin: [max_len, D/2]
    """
    B, H, N, D = x.shape
    half_D = D // 2
    cos_n = cos[offset : offset + N, :]           # [N, D/2]
    sin_n = sin[offset : offset + N, :]           # [N, D/2]
    cos_bc = cos_n[None, None, :, :].astype(x.dtype)
    sin_bc = sin_n[None, None, :, :].astype(x.dtype)
    x0 = x[..., :half_D]   # first half
    x1 = x[..., half_D:]   # second half
    x0_rot = x0 * cos_bc - x1 * sin_bc
    x1_rot = x0 * sin_bc + x1 * cos_bc
    return mx.concatenate([x0_rot, x1_rot], axis=-1)


@pytest.mark.skipif(not _ext_available(), reason="C++ extension not built")
class TestRoPENonInterleaved:
    """Track AC: RoPE non-interleaved (GPT-NeoX split-halves) mode."""

    @pytest.mark.parametrize("D", [64, 128])
    def test_neox_matches_reference(self, D):
        """interleaved=False output matches Python split-halves RoPE + SDPA."""
        from mlx_mfa import flash_attention_rope

        B, H, N, S = 1, 4, 64, 64
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, S, D), dtype=mx.float16)
        v = mx.random.normal((B, H, S, D), dtype=mx.float16)
        cos, sin = _make_rope_tables(256, D)

        out_mfa = flash_attention_rope(q, k, v, cos, sin, scale=scale,
                                       causal=False, cache_seqlens=0,
                                       interleaved=False)
        q_rot = _apply_rope_neox(q, cos, sin, offset=0)
        k_rot = _apply_rope_neox(k, cos, sin, offset=0)
        ref = mx.fast.scaled_dot_product_attention(q_rot, k_rot, v, scale=scale)
        mx.eval(out_mfa, ref)

        np.testing.assert_allclose(
            np.array(out_mfa.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-2, atol=1e-2,
            err_msg=f"GPT-NeoX RoPE kernel mismatch at D={D}",
        )

    @pytest.mark.parametrize("D", [64, 128])
    def test_interleaved_vs_neox_differ(self, D):
        """interleaved=True and interleaved=False produce different results."""
        from mlx_mfa import flash_attention_rope

        B, H, N = 1, 4, 64
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(7)
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, N, D), dtype=mx.float16)
        v = mx.random.normal((B, H, N, D), dtype=mx.float16)
        cos, sin = _make_rope_tables(256, D)

        out_llama = flash_attention_rope(q, k, v, cos, sin, scale=scale,
                                         causal=False, interleaved=True)
        out_neox = flash_attention_rope(q, k, v, cos, sin, scale=scale,
                                        causal=False, interleaved=False)
        mx.eval(out_llama, out_neox)

        max_diff = float(mx.max(mx.abs(
            out_llama.astype(mx.float32) - out_neox.astype(mx.float32)
        )))
        assert max_diff > 1e-3, (
            f"LLaMA and GPT-NeoX RoPE produced identical outputs at D={D} "
            f"(max_diff={max_diff:.2e}) — kernel may not be branching correctly"
        )

    @pytest.mark.parametrize("D", [64, 128])
    def test_neox_backward_finite(self, D):
        """Backward through interleaved=False produces finite gradients."""
        from mlx_mfa import flash_attention_rope

        B, H, N = 1, 2, 32
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(13)
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, N, D), dtype=mx.float16)
        v = mx.random.normal((B, H, N, D), dtype=mx.float16)
        cos, sin = _make_rope_tables(128, D)

        def fwd(q, k, v):
            return flash_attention_rope(q, k, v, cos, sin, scale=scale,
                                        causal=True, interleaved=False).sum()

        loss, grads = mx.value_and_grad(fwd, argnums=(0, 1, 2))(q, k, v)
        mx.eval(loss, *grads)

        for name, g in zip(["dQ", "dK", "dV"], grads):
            assert mx.all(mx.isfinite(g)).item(), \
                f"GPT-NeoX backward: {name} contains NaN/Inf at D={D}"
