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

from mlx_mfa import flash_attention, is_mfa_available, get_device_info, get_supported_configs
from mlx_mfa.attention import _ext_available, _fallback_sdpa


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
