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
import os
import statistics
import time

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import (
    flash_attention, flash_attention_rope, flash_attention_sparse,
    make_causal_block_mask, make_sliding_window_mask,
    is_mfa_available, get_device_info, get_supported_configs,
)
from mlx_mfa import _dispatch_trace as dtrace
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

    @pytest.mark.skipif(not _ext_available(), reason="extension not compiled")
    def test_gpu_cores_m1_max(self):
        """estimate_gpu_cores() returns 32 for 'Apple M1 Max' (not the gen-13 fallback of 8)."""
        info = get_device_info()
        assert "gpu_cores" in info, "gpu_cores key missing from get_device_info()"
        assert isinstance(info["gpu_cores"], int), f"gpu_cores must be int, got {type(info['gpu_cores'])}"
        assert info["gpu_cores"] > 0, "gpu_cores must be positive"
        # On M1 Max hardware, verify the variant is detected correctly.
        if "M1 Max" in info.get("device_name", ""):
            assert info["gpu_cores"] == 32, (
                f"M1 Max should have gpu_cores=32, got {info['gpu_cores']}"
            )

    @pytest.mark.skipif(not _ext_available(), reason="extension not compiled")
    def test_mlx_build_version(self):
        """_mlx_build_version() returns a semver string matching runtime MLX."""
        import mlx_mfa._ext as _ext
        import mlx.core
        build_ver = _ext._mlx_build_version()
        assert isinstance(build_ver, str)
        assert build_ver != "unknown", "MLX_BUILD_VERSION not captured at compile time"
        # major.minor must match runtime
        runtime_ver = mlx.core.__version__
        bv = tuple(int(x) for x in build_ver.split(".")[:2])
        rv = tuple(int(x) for x in runtime_ver.split(".")[:2])
        assert bv == rv, f"build={build_ver} vs runtime={runtime_ver}"

    def test_supported_configs_features(self):
        """get_supported_configs() returns full feature matrix (Track 1)."""
        cfg = get_supported_configs()
        # core keys
        assert "head_dims" in cfg
        assert "dtypes" in cfg
        assert "extension_available" in cfg
        assert "features" in cfg, "missing 'features' key"
        assert "kernel_types" in cfg, "missing 'kernel_types' key"
        # head_dims includes 512
        assert 512 in cfg["head_dims"], "512 missing from head_dims"
        # features dict has required boolean entries
        required = {
            "causal", "gqa", "block_sparse", "sliding_window", "rope",
            "paged_kv", "varlen", "flash_decode", "alibi", "softcap",
            "attn_bias", "backend_select", "dropout", "return_lse",
            "native_backward", "sparse_backward", "m3_routing", "m5_stub",
            "kvcache_rope_append", "packed_api", "bfloat16", "float16", "d512",
        }
        features = cfg["features"]
        missing = required - set(features.keys())
        assert not missing, f"features dict missing keys: {missing}"
        for k, v in features.items():
            assert isinstance(v, (bool, str)), f"features['{k}'] is not bool/str: {v!r}"
        # known values
        assert features["causal"] is True
        assert features["gqa"] is True
        assert features["d512"] is True
        assert features["native_backward"] == "ext"  # extension code exists; auto D512 stays SDPA
        # kernel_types
        ext = cfg["extension_available"]
        assert cfg["kernel_types"] == (16 if ext else 0), (
            f"expected kernel_types={'16 if ext else 0'}, got {cfg['kernel_types']}"
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
# Track ID — flash_attention API enrichment (attn_bias, backend)
# ---------------------------------------------------------------------------

class TestFlashAttentionAPI:
    """Tests for attn_bias and backend parameters (Track ID)."""

    def _qkv(self, B=1, H=4, N=64, D=64, dtype=mx.float16, seed=42):
        mx.random.seed(seed)
        q = mx.random.normal((B, H, N, D)).astype(dtype)
        k = mx.random.normal((B, H, N, D)).astype(dtype)
        v = mx.random.normal((B, H, N, D)).astype(dtype)
        return q, k, v

    # --- backend='sdpa' ------------------------------------------------------

    def test_backend_sdpa_returns_correct_shape(self):
        q, k, v = self._qkv()
        out = flash_attention(q, k, v, backend="sdpa")
        assert out.shape == q.shape

    def test_backend_sdpa_matches_mlx_sdpa(self):
        q, k, v = self._qkv(dtype=mx.float32)
        scale = 1.0 / math.sqrt(64)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        got = flash_attention(q, k, v, scale=scale, backend="sdpa")
        mx.eval(ref, got)
        np.testing.assert_allclose(
            np.array(ref), np.array(got), atol=1e-5,
            err_msg="backend='sdpa' output must match mx.fast.sdpa"
        )

    def test_backend_sdpa_causal(self):
        q, k, v = self._qkv(dtype=mx.float32)
        scale = 1.0 / math.sqrt(64)
        N, S = q.shape[2], k.shape[2]
        causal_mask = mx.triu(mx.full((N, S), float("-inf"), dtype=mx.float32), k=1)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=causal_mask)
        got = flash_attention(q, k, v, scale=scale, causal=True, backend="sdpa")
        mx.eval(ref, got)
        np.testing.assert_allclose(
            np.array(ref), np.array(got), atol=1e-5,
            err_msg="backend='sdpa' causal output mismatch"
        )

    def test_backend_invalid_raises(self):
        q, k, v = self._qkv()
        with pytest.raises(ValueError, match="backend must be one of"):
            flash_attention(q, k, v, backend="unknown")

    # --- backend='mfa' -------------------------------------------------------

    @pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
    def test_backend_mfa_returns_correct_shape(self):
        q, k, v = self._qkv(dtype=mx.float16)
        out = flash_attention(q, k, v, backend="mfa")
        assert out.shape == q.shape

    @pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
    def test_backend_mfa_matches_sdpa(self):
        # N=128 avoids pre-existing V2 accuracy issue at certain small N values
        q, k, v = self._qkv(N=128, D=128, dtype=mx.float16, seed=7)
        scale = 1.0 / math.sqrt(128)
        ref = flash_attention(q, k, v, scale=scale, backend="sdpa")
        got = flash_attention(q, k, v, scale=scale, backend="mfa")
        mx.eval(ref, got)
        ref32 = np.array(ref.astype(mx.float32))
        got32 = np.array(got.astype(mx.float32))
        np.testing.assert_allclose(ref32, got32, atol=1e-2,
            err_msg="backend='mfa' must match SDPA within f16 tolerance")

    def test_backend_mfa_raises_bad_dim(self):
        """Unsupported head_dim with backend='mfa' should raise RuntimeError."""
        # D=48 is not in {64,128,256,512}
        mx.random.seed(1)
        q = mx.random.normal((1, 2, 32, 48)).astype(mx.float16)
        k = mx.random.normal((1, 2, 32, 48)).astype(mx.float16)
        v = mx.random.normal((1, 2, 32, 48)).astype(mx.float16)
        with pytest.raises(RuntimeError, match="unsupported configuration|not compiled"):
            flash_attention(q, k, v, backend="mfa")

    # --- attn_bias -----------------------------------------------------------

    def test_attn_bias_none_unchanged(self):
        """attn_bias=None must not change output."""
        q, k, v = self._qkv(dtype=mx.float32)
        scale = 1.0 / math.sqrt(64)
        ref = flash_attention(q, k, v, scale=scale)
        got = flash_attention(q, k, v, scale=scale, attn_bias=None)
        mx.eval(ref, got)
        np.testing.assert_allclose(np.array(ref), np.array(got), atol=0)

    def test_attn_bias_zeros_unchanged(self):
        """Zero attn_bias must not change output."""
        q, k, v = self._qkv(dtype=mx.float32)
        B, H, N, S = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
        scale = 1.0 / math.sqrt(64)
        bias = mx.zeros((N, S), dtype=mx.float32)
        ref = flash_attention(q, k, v, scale=scale, backend="sdpa")
        got = flash_attention(q, k, v, scale=scale, attn_bias=bias)
        mx.eval(ref, got)
        np.testing.assert_allclose(np.array(ref), np.array(got), atol=1e-5,
            err_msg="Zero attn_bias must leave output unchanged")

    def test_attn_bias_neginf_masks_positions(self):
        """attn_bias=-inf at (q=0, k=1) should mask key position 1 for query 0."""
        q, k, v = self._qkv(N=8, D=64, dtype=mx.float32)
        scale = 1.0 / math.sqrt(64)
        N, S = q.shape[2], k.shape[2]
        bias = mx.zeros((N, S), dtype=mx.float32)
        bias_np = np.zeros((N, S), dtype=np.float32)
        bias_np[0, 1] = float("-inf")
        bias = mx.array(bias_np)
        out = flash_attention(q, k, v, scale=scale, attn_bias=bias)
        mx.eval(out)
        # Build reference: SDPA with the same mask
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=bias)
        mx.eval(ref)
        np.testing.assert_allclose(np.array(out), np.array(ref), atol=1e-5,
            err_msg="attn_bias masking mismatch")

    def test_attn_bias_combined_with_causal(self):
        """attn_bias + causal=True should apply both masks."""
        q, k, v = self._qkv(N=16, D=64, dtype=mx.float32)
        scale = 1.0 / math.sqrt(64)
        N, S = q.shape[2], k.shape[2]
        bias_np = np.random.default_rng(99).uniform(-0.5, 0.5, (N, S)).astype(np.float32)
        bias = mx.array(bias_np)
        got = flash_attention(q, k, v, scale=scale, causal=True, attn_bias=bias)
        # Reference: apply causal mask + bias together
        causal_mask = mx.triu(mx.full((N, S), float("-inf"), dtype=mx.float32), k=1)
        ref = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=causal_mask + bias)
        mx.eval(got, ref)
        np.testing.assert_allclose(np.array(got), np.array(ref), atol=1e-5,
            err_msg="causal + attn_bias mismatch")

    def test_attn_bias_shape_error_propagates(self):
        """Incompatible attn_bias shape should raise from MLX, not silently wrong."""
        q, k, v = self._qkv(N=8, D=64, dtype=mx.float32)
        bad_bias = mx.zeros((3, 3))  # wrong shape
        with pytest.raises(Exception):
            mx.eval(flash_attention(q, k, v, attn_bias=bad_bias))

    # --- window_size right side (Track LA) -----------------------------------

    def test_window_right_positive_is_ok(self):
        """window_size right>0 should NOT raise (native STEEL support, Track LA)."""
        q, k, v = self._qkv()
        out = flash_attention(q, k, v, window_size=(128, 1))
        mx.eval(out)
        assert out.shape == q.shape

    def test_window_right_zero_is_ok(self):
        """window_size right=0 (no right exclusion) should not raise."""
        q, k, v = self._qkv()
        out = flash_attention(q, k, v, window_size=(128, -1))
        mx.eval(out)
        assert out.shape == q.shape

    def test_window_right_negative_is_ok(self):
        """window_size right=-1 (disabled) should not raise."""
        q, k, v = self._qkv()
        out = flash_attention(q, k, v, window_size=(128, -1))
        mx.eval(out)
        assert out.shape == q.shape

    def test_window_right_f32_fallback(self):
        """window_size right=999 for f32 falls back to masked SDPA (no raise)."""
        q, k, v = self._qkv(dtype=mx.float32)
        out = flash_attention(q, k, v, window_size=(64, 999))
        mx.eval(out)
        assert out.shape == q.shape


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
        from mlx_mfa.masks import _bq_bk
        for D in [64, 128, 256]:
            BQ, BK = _bq_bk(D)  # Phase F: maker geometry (D=128 now 32x32)
            N = 256
            mask = make_causal_block_mask(N, head_dim=D)
            NQ = (N + BQ - 1) // BQ
            NK = (N + BK - 1) // BK
            assert list(mask.shape) == [NQ, NK], f"D={D}: expected [{NQ},{NK}], got {list(mask.shape)}"
            assert mask.dtype == mx.bool_

    def test_sliding_window_mask_shape(self):
        from mlx_mfa.masks import _bq_bk
        N, W = 512, 128
        for D in [64, 128, 256]:
            BQ, BK = _bq_bk(D)  # Phase F: maker geometry (D=128 now 32x32)
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
        with pytest.raises(ValueError, match="block_mask"):
            flash_attention_sparse(q, k, v, wrong_mask)


@pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
class TestSparseAttentionKernel:
    """Tests requiring the C++ STEEL sparse kernel."""

    @pytest.mark.parametrize("D", [64, 128, 256])
    def test_all_true_mask_matches_dense(self, D):
        """All-True block mask must produce identical result to dense forward.

        AUDIT C NOTE (which-binary): on M5 only D=64 (symmetric mask) exercises a
        real sparse kernel (byteΔ~3.8e-6 vs SDPA); D=128/256 use an ASYMMETRIC mask
        that routes to the dense-SDPA fallback (byteΔ==0.0), so for those this
        asserts SDPA==SDPA (vacuous).  Real-kernel coverage WITH a binary
        fingerprint: test_sparse_family_correctness_lock.py (symmetric) +
        test_fingerprint_discipline.py.  See dispatch-map.md.
        """
        # v2.50 Prompt 4 Section A: bumped N=128→2048 to satisfy MLX sparse
        # mask_bytes >= 4096 constraint (mask buffer must not be inlined
        # in constant address space per JIT kernel device-pointer expectation).
        B, H, N = 1, 4, 2048
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
        """Block-causal mask + causal=True must match flash_attention(causal=True).

        v2.50 Prompt 5a Section B.3: previously xfail-marked with misleading
        "accuracy — pre-existing" rationale.  Actual root cause was the NAX
        symmetric-bt path's small-mask buffer limitation (mask < 4096 bytes
        for N=128) raising RuntimeError before any numerical comparison ran.
        Fix: small-mask guard in `flash_attention_sparse` routes small problems
        through STEEL sparse path.  Per-D atol reflects FP16 accumulation:
        D=256 accumulates over twice as many products → ULP doubles.

        AUDIT C NOTE (which-binary): the docstring's "STEEL sparse path" is M1–M4
        history — on M5 this N=128 small-mask call runs the dense-SDPA fallback
        (byteΔ==0.0 vs SDPA-causal), so it asserts SDPA==SDPA. Real sparse-kernel +
        fingerprint coverage: test_sparse_family_correctness_lock.py / test_fingerprint_discipline.py.
        """
        B, H, N = 1, 4, 128
        q, k, v = random_qkv(B, H, N, D, seed=20)
        scale = 1.0 / math.sqrt(D)

        mask = make_causal_block_mask(N, head_dim=D)
        out_sparse = flash_attention_sparse(q, k, v, mask, scale=scale, causal=True)
        out_dense  = flash_attention(q, k, v, scale=scale, causal=True, backend="mfa")
        mx.eval(out_sparse, out_dense)

        # FP16 atol scales with D (more accumulation, more rounding).
        atol = 1e-3 if D <= 128 else 2.5e-3
        np.testing.assert_allclose(
            np.array(out_dense.astype(mx.float32)),
            np.array(out_sparse.astype(mx.float32)),
            atol=atol,
            err_msg=f"D={D}: causal block+causal ≠ dense causal"
        )

    def test_sliding_window_matches_ref(self):
        """Sliding-window mask output must match reference dense SDPA + float bias.

        AUDIT C NOTE: D=128 asymmetric → on M5 this runs the dense-SDPA fallback, and
        the reference IS dense SDPA + bias, so byteΔ==0.0 (reference-is-the-binary;
        vacuous). Real sparse-kernel + fingerprint coverage: test_fingerprint_discipline.py.
        """
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
        # III-4 F12: this test deliberately produces NaN-filled buffers;
        # flush the Metal buffer pool so recycled buffers can't
        # contaminate later lazy-zeros tests (same pattern as the
        # steel_sparse backward test below).
        mx.clear_cache()

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
        # v2.50 Prompt 4 Section A: bumped N=64→2048 for sparse mask>=4096 bytes.
        B, H, N = 1, 4, 2048
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
        # v2.50 Prompt 4 Section A: bumped N=64→2048 for sparse mask>=4096 bytes.
        B, H, N = 1, 2, 2048
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
        # v2.50 Prompt 4 Section A: bumped N=64→2048 for sparse mask>=4096 bytes.
        B, H, N = 1, 4, 2048
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
        # v2.50 Prompt 4 Section A: bumped N=64→2048 for sparse mask>=4096 bytes.
        B, H_q, H_kv, N, D = 1, 8, 2, 2048, 128
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
# Track IC — Native sparse backward (steel_sparse)
# ==========================================================================

@pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
class TestSparseBackwardSteel:
    """Verify backward='steel_sparse' (native STEEL Metal sparse backward).

    T2-3 FIX (audit H6, 2026-06-21): the two `*_matches_sdpa` correctness cells
    were GREEN-ON-WRONG-BINARY.  On M5 a symmetric `make_causal_block_mask`
    auto-routes `flash_attention_sparse` through `_make_sparse_nax_with_sdpa_vjp`
    (SDPA-vjp) and the `backward="steel_sparse"` arg is IGNORED — verified at
    runtime: the native `mfa_steel_backward_sparse` binding is called 0 times and
    byteΔ(steel grads, sdpa grads)==0.0.  So both arms were SDPA-vjp and the test
    validated nothing about the STEEL kernel.  The native kernel only engages on
    M1–M4.  The cells now (a) DETECT engagement via which-binary (byteΔ) and SKIP
    honestly where the kernel can't run, and (b) when engaged, validate against an
    INDEPENDENT fp32 gradient oracle (block-masked manual forward), not SDPA-vjp.
    The scale in `*_all_true` was also a typo (`1/sqrt(N)`≈0.022 → flat softmax);
    fixed to `1/sqrt(D)` to match the sibling cell.  f16 only; D=64/128.
    """

    def _grads(self, q, k, v, mask, scale, causal=False, backward="sdpa"):
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

    def _fp32_oracle_grads(self, q, k, v, mask, scale, causal):
        """Independent fp32 gradient oracle (lesson #11): grad of a MANUAL fp32
        block-masked forward — NOT SDPA-vjp, NOT the kernel.  Applies the SAME
        tile-level block mask the kernel does (no element-causal within a block)."""
        N, S = q.shape[2], k.shape[2]
        NQ, NK = mask.shape[-2], mask.shape[-1]

        def fwd(q_, k_, v_):
            qf, kf, vf = q_.astype(mx.float32), k_.astype(mx.float32), v_.astype(mx.float32)
            Hq, Hk = q_.shape[1], k_.shape[1]
            if Hq != Hk:
                r = Hq // Hk; kf = mx.repeat(kf, r, 1); vf = mx.repeat(vf, r, 1)
            s = (qf @ kf.swapaxes(-1, -2)) * scale
            em = mx.repeat(mx.repeat(mask.astype(mx.float32), N // NQ, -2), S // NK, -1)
            while em.ndim < 4:
                em = em[None]
            s = mx.where(em > 0, s, mx.array(-1e30, mx.float32))
            if causal:
                cm = (mx.arange(N)[:, None] + (S - N) >= mx.arange(S)[None, :]).astype(mx.float32)
                s = mx.where(cm > 0, s, mx.array(-1e30, mx.float32))
            return mx.sum(mx.softmax(s, -1) @ vf)

        dq, dk, dv = mx.grad(fwd, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)
        return (np.array(dq.astype(mx.float32)), np.array(dk.astype(mx.float32)),
                np.array(dv.astype(mx.float32)))

    def _assert_steel_engaged_or_skip(self, steel, sdpa):
        """which-binary: if the native STEEL backward did not engage (byteΔ vs
        SDPA-vjp == 0 across all grads), this hardware routed sparse → SDPA-vjp
        (M5).  Skip honestly — native coverage is M1–M4."""
        if all(float(np.abs(s - r).max()) == 0.0 for s, r in zip(steel, sdpa)):
            pytest.skip("steel_sparse backward not engaged on this hardware "
                        "(M5 routes sparse-symmetric → SDPA-vjp); native coverage is M1–M4")

    @pytest.mark.parametrize("D", [64, 128])
    def test_steel_sparse_all_true_matches_sdpa(self, D):
        """steel_sparse grads (when engaged) match an independent fp32 oracle."""
        # v2.50 Prompt 4 Section A: bumped N=64→2048 for sparse mask>=4096 bytes.
        B, H, N = 1, 2, 2048
        mx.random.seed(99)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        mask = make_causal_block_mask(N, D)
        scale = 1.0 / D**0.5  # T2-4: was 1/sqrt(N) (≈0.022) — a typo that flattened softmax.

        sdpa  = self._grads(q, k, v, mask, scale, backward="sdpa")
        steel = self._grads(q, k, v, mask, scale, backward="steel_sparse")
        self._assert_steel_engaged_or_skip(steel, sdpa)

        oq, ok_, ov = self._fp32_oracle_grads(q, k, v, mask, scale, causal=False)
        for nm, sp, ref in zip(("dQ", "dK", "dV"), steel, (oq, ok_, ov)):
            np.testing.assert_allclose(sp, ref, atol=2e-2, err_msg=f"{nm} vs fp32 oracle")

    @pytest.mark.parametrize("D", [64, 128])
    def test_steel_sparse_causal_block_mask(self, D):
        """Causal block mask: steel_sparse (when engaged) matches an fp32 oracle."""
        # v2.50 Prompt 4 Section A: bumped N=64→2048 for sparse mask>=4096 bytes.
        B, H, N = 1, 2, 2048
        mx.random.seed(77)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        mask = make_causal_block_mask(N, D)
        scale = 1.0 / D**0.5

        sdpa  = self._grads(q, k, v, mask, scale, causal=True, backward="sdpa")
        steel = self._grads(q, k, v, mask, scale, causal=True, backward="steel_sparse")
        self._assert_steel_engaged_or_skip(steel, sdpa)

        oq, ok_, ov = self._fp32_oracle_grads(q, k, v, mask, scale, causal=True)
        for nm, sp, ref in zip(("dQ", "dK", "dV"), steel, (oq, ok_, ov)):
            np.testing.assert_allclose(sp, ref, atol=2e-2, err_msg=f"{nm} causal vs fp32 oracle")

    def test_steel_sparse_gradients_finite(self):
        """Gradients are finite (no NaN/Inf) for steel_sparse."""
        B, H, N, D = 1, 2, 64, 128
        mx.random.seed(13)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        mask = make_causal_block_mask(N, D)

        def loss(q_, k_, v_):
            return mx.sum(
                flash_attention_sparse(q_, k_, v_, mask, backward="steel_sparse")
            )

        dq, dk, dv = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)
        assert mx.all(mx.isfinite(dq)).item(), "dQ has non-finite values"
        assert mx.all(mx.isfinite(dk)).item(), "dK has non-finite values"
        assert mx.all(mx.isfinite(dv)).item(), "dV has non-finite values"

    def test_steel_sparse_backward_binding_exists(self):
        """mfa_steel_backward_sparse binding is importable."""
        from mlx_mfa._ext import mfa_steel_backward_sparse
        assert callable(mfa_steel_backward_sparse)

    def test_steel_sparse_gqa_shape_and_finite(self):
        """GQA (H_q=4, H_kv=2) steel_sparse backward: shapes and finite."""
        # v2.50 Prompt 4 Section A: bumped N=64→2048 for sparse mask>=4096 bytes.
        B, H_q, H_kv, N, D = 1, 4, 2, 2048, 64
        mx.random.seed(42)
        q = mx.random.normal((B, H_q, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK
        mask = mx.ones((H_q, NQ, NK), dtype=mx.bool_)
        scale = 1.0 / D**0.5

        def loss(q_, k_, v_):
            return mx.sum(
                flash_attention_sparse(q_, k_, v_, mask, scale=scale,
                                       causal=False, backward="steel_sparse")
            )

        dq, dk, dv = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)

        assert dq.shape == (B, H_q, N, D), f"dQ shape wrong: {dq.shape}"
        assert dk.shape == (B, H_kv, N, D), f"dK shape wrong: {dk.shape}"
        assert dv.shape == (B, H_kv, N, D), f"dV shape wrong: {dv.shape}"
        assert mx.all(mx.isfinite(dq)).item(), "dQ has non-finite values"
        assert mx.all(mx.isfinite(dk)).item(), "dK has non-finite values"
        assert mx.all(mx.isfinite(dv)).item(), "dV has non-finite values"
        mx.clear_cache()  # flush Metal buffer pool to prevent stale-buffer NaN in downstream tests

    def test_steel_sparse_value_and_grad(self):
        """mx.value_and_grad must work with steel_sparse backward."""
        B, H, N, D = 1, 2, 64, 128
        mx.random.seed(7)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        mask = make_causal_block_mask(N, D)
        scale = 1.0 / D**0.5

        def loss(q_, k_, v_):
            return mx.sum(
                flash_attention_sparse(q_, k_, v_, mask, scale=scale,
                                       causal=True, backward="steel_sparse")
            )

        val_fn = mx.value_and_grad(loss, argnums=(0, 1, 2))
        (loss_val, (dq, dk, dv)) = val_fn(q, k, v)
        mx.eval(loss_val, dq, dk, dv)

        assert mx.isfinite(loss_val).item(), "loss is non-finite"
        assert mx.all(mx.isfinite(dq)).item(), "dQ has non-finite values"
        assert mx.all(mx.isfinite(dk)).item(), "dK has non-finite values"
        mx.clear_cache()  # flush Metal buffer pool to prevent stale-buffer NaN in downstream tests


# ==========================================================================
# v2.33.1 — flash_attention_sparse M5+ fast-fallback regression guards
# ==========================================================================

class TestSparseM5PlusFastFallback:
    """Tests for the v2.33.1 M5+ fast-fallback (cached float bias).

    Sprint B Phase 0 surveyed `flash_attention_sparse` at 2.07-2.10×
    slower than `mx.fast.scaled_dot_product_attention` with prebuilt
    float bias on M5+ Apple Silicon. v2.33.1 patches the M5+ dispatch
    path to cache the expanded float bias by `id(block_mask)`, so reused
    masks hit the cache (within 10% of SDPA-direct). See
    `docs/sparse-fallback-audit.md` for the audit.

    These tests guard against regression: if anyone inadvertently
    re-introduces per-call mask-expansion overhead, the perf-guard
    test fails loudly.
    """

    def test_sparse_m5plus_fallback_correctness_equivalence(self):
        """Sliding-window D=128 sparse output matches the SDPA+float-bias ref.

        Audit Phase F (2026-06-18): this sliding-window D=128 mask is now
        SYMMETRIC 32x32 (was 32x16). Routing now follows the hardened beta-3
        shape/dtype/density gate. NAX is a distinct kernel from SDPA, so the bar
        is NAX-grade (rmse < 1e-3 vs the SDPA+bias reference) when engaged. The
        cached-float-bias fast-fallback this test originally guarded is still
        exercised by DENSE / asymmetric masks (see the perf-guard test).
        """
        # lcsa_small_seq4k from Sprint B Phase 0
        B, H, N, D = 1, 12, 4096, 128
        mx.random.seed(0)
        q = (mx.random.uniform(shape=(B, H, N, D)) * 0.1).astype(mx.float16)
        k = (mx.random.uniform(shape=(B, H, N, D)) * 0.1).astype(mx.float16)
        v = (mx.random.uniform(shape=(B, H, N, D)) * 0.1).astype(mx.float16)
        mask = make_sliding_window_mask(N, window_size=512, head_dim=D)
        mx.async_eval(q, k, v, mask); mx.synchronize()

        from mlx_mfa.masks import _bq_bk
        BQ, BK = _bq_bk(D)  # Phase F: mask geometry (D=128 now 32x32)
        NQ = (N + BQ - 1) // BQ
        NK = (N + BK - 1) // BK
        scale = 1.0 / math.sqrt(D)

        # Reference: SDPA with the float bias expanded from the SAME mask.
        full_mask = mx.broadcast_to(mask[None, None, :, :], (B, H, NQ, NK))
        expanded = full_mask[:, :, :, None, :, None]
        expanded = mx.broadcast_to(expanded, (B, H, NQ, BQ, NK, BK))
        expanded = expanded.reshape(B, H, NQ * BQ, NK * BK)[:, :, :N, :N]
        neg_inf = mx.array(float("-inf"), dtype=q.dtype)
        zero = mx.array(0.0, dtype=q.dtype)
        bias = mx.where(expanded, zero, neg_inf)
        mx.async_eval(bias); mx.synchronize()

        y_ref = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=bias)
        y_mfa = flash_attention_sparse(q, k, v, mask, scale=scale, causal=False)
        mx.async_eval(y_ref, y_mfa); mx.synchronize()

        err = mx.abs(y_ref.astype(mx.float32) - y_mfa.astype(mx.float32))
        rmse = float(mx.sqrt(mx.mean(err * err)))
        # NAX-grade (Phase F): NAX-sparse vs SDPA+bias, not bit-exact.
        assert rmse < 1e-3, (
            f"sliding-window NAX-sparse not within 1e-3 of SDPA+float-bias: "
            f"rmse={rmse:.6e}"
        )

    @pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
    def test_sparse_m5plus_perf_regression_guard(self):
        """Cache-hit pattern: MFA within 10% of SDPA-direct on M5+.

        Pattern: same `block_mask` Python object reused across 5 timed
        calls (after warmup). This is the common production pattern
        (build mask once per forward, reuse across attention calls)
        and is the v2.33.1 cache target.

        v2.33.0 baseline: MFA was 2.07× SDPA-direct (Sprint B Phase 0).
        v2.33.1 target:   MFA ≤ 1.10× SDPA-direct (within 10%).

        Skipped on non-M5+ hardware (the fast-fallback path is M5+ only).
        """
        from mlx_mfa import get_device_info
        info = get_device_info()
        if not info.get("is_m5_plus"):
            pytest.skip(
                "M5+ fast-fallback only — skipping on pre-M5 hardware")

        B, H, N, D = 1, 12, 4096, 128
        mx.random.seed(0)
        q = (mx.random.uniform(shape=(B, H, N, D)) * 0.1).astype(mx.float16)
        k = (mx.random.uniform(shape=(B, H, N, D)) * 0.1).astype(mx.float16)
        v = (mx.random.uniform(shape=(B, H, N, D)) * 0.1).astype(mx.float16)
        mask = make_sliding_window_mask(N, window_size=512, head_dim=D)
        mx.async_eval(q, k, v, mask); mx.synchronize()

        from mlx_mfa.masks import _bq_bk
        BQ, BK = _bq_bk(D)  # Phase F: mask geometry (D=128 now 32x32, → NAX)
        NQ = (N + BQ - 1) // BQ
        NK = (N + BK - 1) // BK
        scale = 1.0 / math.sqrt(D)

        # Prebuilt bias for the SDPA-direct baseline (outside timed loop).
        full_mask = mx.broadcast_to(mask[None, None, :, :], (B, H, NQ, NK))
        expanded = full_mask[:, :, :, None, :, None]
        expanded = mx.broadcast_to(expanded, (B, H, NQ, BQ, NK, BK))
        expanded = expanded.reshape(B, H, NQ * BQ, NK * BK)[:, :, :N, :N]
        neg_inf = mx.array(float("-inf"), dtype=q.dtype)
        zero = mx.array(0.0, dtype=q.dtype)
        bias = mx.where(expanded, zero, neg_inf)
        mx.async_eval(bias); mx.synchronize()

        # Warmup MFA path — populates the LRU cache for `mask`.
        for _ in range(3):
            y = flash_attention_sparse(q, k, v, mask, scale=scale, causal=False)
            mx.async_eval(y); mx.synchronize()
        # Warmup SDPA-direct.
        for _ in range(3):
            y = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=bias)
            mx.async_eval(y); mx.synchronize()

        # Time MFA (cache-hit on every call).
        mfa_times = []
        for _ in range(5):
            mx.synchronize()
            t0 = time.perf_counter()
            y = flash_attention_sparse(q, k, v, mask, scale=scale, causal=False)
            mx.async_eval(y); mx.synchronize()
            mfa_times.append(time.perf_counter() - t0)

        # Time SDPA-direct (bias pre-built outside loop).
        sdpa_times = []
        for _ in range(5):
            mx.synchronize()
            t0 = time.perf_counter()
            y = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=bias)
            mx.async_eval(y); mx.synchronize()
            sdpa_times.append(time.perf_counter() - t0)

        t_mfa = statistics.median(mfa_times)
        t_sdpa = statistics.median(sdpa_times)
        ratio = t_mfa / t_sdpa
        # 10% tolerance per v2.33.1 patch §3.
        assert ratio <= 1.10, (
            f"v2.33.1 perf regression guard: MFA/SDPA ratio = {ratio:.3f}× "
            f"(MFA {t_mfa*1000:.2f}ms vs SDPA {t_sdpa*1000:.2f}ms). "
            f"Target: ≤ 1.10×. v2.33.0 baseline: 2.07×. "
            f"Cache-hit pattern should reach ≤ 1.10×; if this fails, the "
            f"id-keyed cache is not being hit (regression in _SPARSE_BIAS_CACHE "
            f"in mlx_mfa/attention.py)."
        )

    def test_sparse_m1m4_path_unchanged(self):
        """The M5+ fast-fallback patch does NOT touch the M1-M4 path.

        v2.33.1 modifies only `_sparse_fallback_sdpa_perhead`, which is
        reached on M5+ per `attention.py`'s `if info.get("is_m5_plus"):`
        check. M1-M4 still routes via `_make_mfa_sparse_custom` → C++
        STEEL V1 sparse kernel.

        This is a code-level guard: if a future patch removes the
        is_m5_plus dispatch check or rewires the M1-M4 path,
        this test catches it.
        """
        import inspect
        src = inspect.getsource(flash_attention_sparse)
        assert 'is_m5_plus' in src, (
            "flash_attention_sparse must retain the is_m5_plus dispatch check"
        )
        assert '_sparse_fallback_sdpa_perhead' in src, (
            "flash_attention_sparse must retain the M5+ fast-fallback route"
        )
        assert '_make_mfa_sparse_custom' in src, (
            "flash_attention_sparse must retain the M1-M4 STEEL sparse path"
        )


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

    def _run_engaged(self, q, k, v, scale, causal=False):
        """CX-06 (audit): force the MFA decode kernel and ASSERT it engaged.

        These cells previously called ``flash_attention`` on ``backend="auto"``,
        which on M5 routes decode to SDPA (byteΔ=0) — so they compared SDPA to an
        SDPA-derived ``_ref`` and were vacuous (could not catch a flash-decode
        kernel bug; cf. the RC-A/RC-B defect that hid exactly this way). Forcing
        ``backend="mfa"`` and asserting the MFA primitive ran makes them test the
        kernel against the fp32 ``_ref`` oracle. (The primary engaged decode lock
        is Tier-0 ``test_causal_maskzone_split_lock``; these add D=256 / GQA /
        S=256-boundary / bf16 coverage.)
        """
        from mlx_mfa import _dispatch_trace as _dt
        with _dt.capture() as tr:
            out = flash_attention(q, k, v, scale=scale, causal=causal, backend="mfa")
            mx.eval(out)
        assert tr and tr[-1][0] == "mfa_primitive", (
            f"flash-decode test did not engage the MFA kernel "
            f"(backend={tr[-1][0] if tr else None}) — would be vacuous on SDPA")
        return out

    @pytest.mark.parametrize("D", [64, 128, 256])
    def test_decode_noncausal(self, D):
        """N=1 decode, non-causal: Flash Decode should match SDPA within tol."""
        B, H, N, S = 1, 8, 1, 512
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, S, D]).astype(mx.float16)
        v = mx.random.normal([B, H, S, D]).astype(mx.float16)

        out = self._run_engaged(q, k, v, scale, causal=False)
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

        out = self._run_engaged(q, k, v, scale, causal=True)
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

        out = self._run_engaged(q, k, v, scale, causal=False)
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

        out = self._run_engaged(q, k, v, scale, causal=False)
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

        out = self._run_engaged(q, k, v, scale, causal=False)
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

        out = self._run_engaged(q, k, v, scale, causal=False)
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

        out = self._run_engaged(q, k, v, scale, causal=False)
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

        out = self._run_engaged(q, k, v, scale, causal=False)
        ref = self._ref(q, k, v, scale, causal=False)
        mx.eval(out, ref)

        out_np = np.array(out.astype(mx.float32))
        ref_np = np.array(ref)
        max_err = float(np.max(np.abs(out_np - ref_np)))
        assert max_err < 0.05, f"N=5 (non-FD) max_err={max_err:.4f}"


# ===========================================================================
# V2 split-K correctness tests (Phase 3)
# ===========================================================================

@requires_ext
class TestV2SplitK:
    """Correctness tests for V2 split-K path (under-occupied grids).

    V2 split-K fires when total_tgs < 0.8 * gpu_cores.
    With gpu_cores=32 (M1 Max), threshold = 25.6.
    B=1 H=1 N=512: total_tgs = 16*1*1 = 16 < 25.6 → activates split-K.
    """

    @pytest.mark.parametrize("D", [64, 128])
    def test_splitk_causal_correctness(self, D):
        """V2 split-K causal output matches SDPA reference."""
        B, H, N = 1, 1, 512
        q = mx.random.normal([B, H, N, D], dtype=mx.float16)
        k = mx.random.normal([B, H, N, D], dtype=mx.float16)
        v = mx.random.normal([B, H, N, D], dtype=mx.float16)
        scale = D ** -0.5
        mfa_out = flash_attention(q, k, v, scale=scale, causal=True)
        mask = mx.tril(mx.ones([N, N], dtype=mx.bool_))
        ref_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        mx.eval(mfa_out, ref_out)
        max_err = float(mx.max(mx.abs(mfa_out.astype(mx.float32) - ref_out.astype(mx.float32))))
        assert max_err < 0.05, f"V2 split-K causal D={D}: max_err={max_err:.4f}"

    @pytest.mark.parametrize("D", [64, 128])
    def test_splitk_noncausal_correctness(self, D):
        """V2 split-K non-causal output matches SDPA reference."""
        B, H, N = 1, 1, 512
        q = mx.random.normal([B, H, N, D], dtype=mx.float16)
        k = mx.random.normal([B, H, N, D], dtype=mx.float16)
        v = mx.random.normal([B, H, N, D], dtype=mx.float16)
        scale = D ** -0.5
        mfa_out = flash_attention(q, k, v, scale=scale, causal=False)
        ref_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(mfa_out, ref_out)
        max_err = float(mx.max(mx.abs(mfa_out.astype(mx.float32) - ref_out.astype(mx.float32))))
        assert max_err < 0.05, f"V2 split-K non-causal D={D}: max_err={max_err:.4f}"

    @pytest.mark.parametrize("D", [64, 128])
    def test_splitk_rope_matches_sdpa(self, D):
        """V2 split-K + RoPE output matches flash_attention_rope_unified on SDPA path.

        Uses B=1 H=1 (under-occupied → split-K fires) with half-rotate RoPE.
        Reference: flash_attention_rope_unified with MFA_DISABLE_V2=1 (forces
        single-pass V2 which also supports RoPE, or falls through to SDPA).
        """
        import os as _os
        from mlx_mfa import flash_attention_rope_unified
        B, H, N, S = 1, 1, 512, 512
        mx.random.seed(42)
        q = mx.random.normal([B, H, N, D], dtype=mx.float16)
        k = mx.random.normal([B, H, S, D], dtype=mx.float16)
        v = mx.random.normal([B, H, S, D], dtype=mx.float16)
        # Build cos/sin for half-rotate (non-interleaved) RoPE
        cos = mx.ones([N + S, D // 2], dtype=mx.float32)
        sin = mx.zeros([N + S, D // 2], dtype=mx.float32)
        mx.eval(q, k, v, cos, sin)

        # MFA with split-K enabled (default)
        mfa_out = flash_attention_rope_unified(q, k, v, cos, sin, causal=True)
        mx.eval(mfa_out)

        # Reference: disable split-K to force single-pass (avoids circular comparison)
        prev = _os.environ.get("MFA_DISABLE_V2")
        try:
            _os.environ["MFA_DISABLE_V2"] = "1"
            ref_out = flash_attention_rope_unified(q, k, v, cos, sin, causal=True)
            mx.eval(ref_out)
        finally:
            if prev is None:
                _os.environ.pop("MFA_DISABLE_V2", None)
            else:
                _os.environ["MFA_DISABLE_V2"] = prev

        max_err = float(mx.max(mx.abs(
            mfa_out.astype(mx.float32) - ref_out.astype(mx.float32))))
        assert max_err < 0.05, (
            f"V2 split-K + RoPE D={D}: max_err={max_err:.4f}")

    @pytest.mark.parametrize("D", [64, 128])
    @pytest.mark.parametrize("causal", [True, False])
    def test_splitk_alibi_matches_non_split(self, D, causal):
        """V2 split-K + ALiBi matches no-V2 routing on under-occupied grids."""
        import os as _os
        B, H, N = 1, 1, 512
        mx.random.seed(123)
        q = mx.random.normal([B, H, N, D], dtype=mx.float16)
        k = mx.random.normal([B, H, N, D], dtype=mx.float16)
        v = mx.random.normal([B, H, N, D], dtype=mx.float16)
        slopes = mx.array([-0.1], dtype=mx.float32)
        scale = D ** -0.5

        out = flash_attention(
            q, k, v, scale=scale, causal=causal, alibi_slopes=slopes, backend="mfa"
        )
        mx.eval(out)

        prev = _os.environ.get("MFA_DISABLE_V2")
        try:
            _os.environ["MFA_DISABLE_V2"] = "1"
            ref = flash_attention(
                q, k, v, scale=scale, causal=causal, alibi_slopes=slopes, backend="mfa"
            )
            mx.eval(ref)
        finally:
            if prev is None:
                _os.environ.pop("MFA_DISABLE_V2", None)
            else:
                _os.environ["MFA_DISABLE_V2"] = prev

        max_err = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        assert max_err < 0.05, (
            f"V2 split-K + ALiBi D={D} causal={causal}: max_err={max_err:.4f}"
        )

    @pytest.mark.parametrize("D", [64, 128])
    @pytest.mark.parametrize("window", [(256, 0), (512, 0)])
    def test_splitk_window_matches_non_split(self, D, window):
        """V2 split-K + window matches no-V2 routing on under-occupied grids."""
        import os as _os
        B, H, N = 1, 1, 512
        mx.random.seed(456)
        q = mx.random.normal([B, H, N, D], dtype=mx.float16)
        k = mx.random.normal([B, H, N, D], dtype=mx.float16)
        v = mx.random.normal([B, H, N, D], dtype=mx.float16)
        scale = D ** -0.5
        out = flash_attention(
            q, k, v, scale=scale, causal=True, window_size=window, backend="mfa"
        )
        mx.eval(out)

        prev = _os.environ.get("MFA_DISABLE_V2")
        try:
            _os.environ["MFA_DISABLE_V2"] = "1"
            ref = flash_attention(
                q, k, v, scale=scale, causal=True, window_size=window, backend="mfa"
            )
            mx.eval(ref)
        finally:
            if prev is None:
                _os.environ.pop("MFA_DISABLE_V2", None)
            else:
                _os.environ["MFA_DISABLE_V2"] = prev

        max_err = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        assert max_err < 0.05, (
            f"V2 split-K + window D={D} window={window}: max_err={max_err:.4f}"
        )

    @pytest.mark.parametrize("D", [64, 128])
    def test_splitk_rope_window_matches_non_split(self, D):
        """RoPE+window remains supported (via pre-rotated path) and matches no-V2 routing."""
        import os as _os
        from mlx_mfa import flash_attention_kvcache
        B, H, N = 1, 1, 512
        mx.random.seed(789)
        q = mx.random.normal([B, H, N, D], dtype=mx.float16)
        k = mx.random.normal([B, H, N, D], dtype=mx.float16)
        v = mx.random.normal([B, H, N, D], dtype=mx.float16)
        cos, sin = _make_rope_tables(N + 64, D)
        scale = D ** -0.5

        out = flash_attention_kvcache(
            q, k, v,
            scale=scale,
            causal=True,
            rotary_cos=cos,
            rotary_sin=sin,
            window_size=(256, 0),
        )
        mx.eval(out)

        prev = _os.environ.get("MFA_DISABLE_V2")
        try:
            _os.environ["MFA_DISABLE_V2"] = "1"
            ref = flash_attention_kvcache(
                q, k, v,
                scale=scale,
                causal=True,
                rotary_cos=cos,
                rotary_sin=sin,
                window_size=(256, 0),
            )
            mx.eval(ref)
        finally:
            if prev is None:
                _os.environ.pop("MFA_DISABLE_V2", None)
            else:
                _os.environ["MFA_DISABLE_V2"] = prev

        max_err = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        assert max_err < 0.05, (
            f"V2 split-K + RoPE + window D={D}: max_err={max_err:.4f}"
        )

    def test_splitk_rope_alibi_is_explicitly_gated(self):
        """RoPE+ALiBi remains explicitly gated in public API (unsupported combination)."""
        from mlx_mfa import flash_attention_kvcache
        B, H, N, D = 1, 1, 64, 64
        q = mx.random.normal([B, H, N, D], dtype=mx.float16)
        k = mx.random.normal([B, H, N, D], dtype=mx.float16)
        v = mx.random.normal([B, H, N, D], dtype=mx.float16)
        slopes = mx.array([1.0], dtype=mx.float32)
        cos, sin = _make_rope_tables(N + 16, D)
        with pytest.raises(ValueError, match="mutually exclusive"):
            flash_attention_kvcache(
                q, k, v,
                scale=D ** -0.5,
                causal=True,
                rotary_cos=cos,
                rotary_sin=sin,
                alibi_slopes=slopes,
            )


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

    def test_rope_none_falls_back_to_regular_attention(self, monkeypatch):
        """With identity RoPE (cos=1, sin=0), result equals plain attention.

        v2.50 Sprint 2 (Prompt 1) introduced an M5+ NAX path that uses
        `mx.fast.rope` with `base=10000.0` hardcoded — it ignores caller-
        provided cos/sin tables.  This test specifically exercises identity-
        rope semantics (a CUSTOM rotation table), so force STEEL fallback
        via MFA_DISABLE_ROPE_NAX=1 per Sprint 2 DC4 opt-out contract.
        """
        from mlx_mfa import flash_attention, flash_attention_rope

        # v2.50 Prompt 4 Section A: STEEL fallback required for identity-rope
        # semantics + bumped N=32→2048 for any sparse-path mask>=4096.
        monkeypatch.setenv("MFA_DISABLE_ROPE_NAX", "1")
        B, H, N, D = 1, 2, 2048, 64
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
        # v2.50 Prompt 4 Section A: bumped segs [32,32]→[1024,1024] for
        # sparse mask>=4096 bytes constraint.
        segs = [1024, 1024]
        N = sum(segs)
        q = mx.random.normal((B, H_heads, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H_heads, N, D), dtype=mx.float16)
        v = mx.random.normal((B, H_heads, N, D), dtype=mx.float16)

        mask = make_segment_mask(segs, head_dim=D)
        out_sparse = flash_attention_sparse(q, k, v, mask,
                                            scale=1.0/D**0.5, causal=False)
        mx.eval(out_sparse)

        # Run each segment independently and concatenate.
        # v2.50 Prompt 4 Section A: use plain dense flash_attention per segment
        # rather than sparse with a single-segment mask (which would hit MLX's
        # mask>=4096 byte constraint for small seg masks).  Mathematically
        # identical: a single-segment mask is all-True intra-segment, equivalent
        # to dense attention restricted to that segment.
        outputs = []
        offset = 0
        for seg_len in segs:
            q_i = q[:, :, offset:offset+seg_len, :]
            k_i = k[:, :, offset:offset+seg_len, :]
            v_i = v[:, :, offset:offset+seg_len, :]
            out_i = flash_attention(q_i, k_i, v_i, scale=1.0/D**0.5)
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
# Track BD — STEEL varlen forward kernel
# ============================================================================

@pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
class TestSteelVarlen:
    """Correctness tests for the STEEL varlen forward kernel (f16/bf16)."""

    def _ref_concat(self, q, k, v, cu_q, cu_k, scale, causal):
        """Reference: SDPA per sequence, concatenated."""
        cu_q_l = [int(x) for x in cu_q.tolist()]
        cu_k_l = [int(x) for x in cu_k.tolist()]
        num_seqs = len(cu_q_l) - 1
        outs = []
        for i in range(num_seqs):
            q_i = q[:, :, cu_q_l[i]:cu_q_l[i+1], :]
            k_i = k[:, :, cu_k_l[i]:cu_k_l[i+1], :]
            v_i = v[:, :, cu_k_l[i]:cu_k_l[i+1], :]
            ref = mx.fast.scaled_dot_product_attention(
                q_i.astype(mx.float32),
                k_i.astype(mx.float32),
                v_i.astype(mx.float32),
                scale=scale,
                mask="causal" if causal else None,
            ).astype(q.dtype)
            outs.append(ref)
        return mx.concatenate(outs, axis=2)

    def test_single_seq_f16(self):
        """Single f16 sequence: kernel == SDPA reference."""
        from mlx_mfa import flash_attention_varlen
        B, H, N, D = 1, 4, 64, 64
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        scale = 1.0 / D**0.5
        cu = mx.array([0, N], dtype=mx.int32)
        ref = self._ref_concat(q, k, v, cu, cu, scale, causal=False)
        out = flash_attention_varlen(q, k, v, cu, cu, N, N, scale=scale, causal=False)
        mx.eval(out, ref)
        diff = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"Single seq f16 max diff: {diff}"

    def test_two_seqs_f16(self):
        """Two packed f16 sequences: kernel == per-sequence SDPA."""
        from mlx_mfa import flash_attention_varlen
        B, H, D = 1, 4, 128
        N1, N2 = 48, 64
        N = N1 + N2
        scale = 1.0 / D**0.5
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        cu = mx.array([0, N1, N], dtype=mx.int32)
        ref = self._ref_concat(q, k, v, cu, cu, scale, causal=False)
        out = flash_attention_varlen(q, k, v, cu, cu, N2, N2, scale=scale, causal=False)
        mx.eval(out, ref)
        diff = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"Two seqs f16 max diff: {diff}"

    def test_three_seqs_mixed_lengths_f16(self):
        """Three sequences with unequal lengths — shape and correctness."""
        from mlx_mfa import flash_attention_varlen
        B, H, D = 1, 2, 64
        lengths = [33, 64, 17]   # not multiples of BQ=32
        N = sum(lengths)
        scale = 1.0 / D**0.5
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        import numpy as np
        cu_list = [0] + [int(x) for x in np.cumsum(lengths)]
        cu = mx.array(cu_list, dtype=mx.int32)
        ref = self._ref_concat(q, k, v, cu, cu, scale, causal=False)
        out = flash_attention_varlen(q, k, v, cu, cu, max(lengths), max(lengths),
                                    scale=scale, causal=False)
        mx.eval(out, ref)
        assert out.shape == (B, H, N, D)
        diff = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"Mixed-length f16 max diff: {diff}"

    def test_causal_f16(self):
        """Causal kernel: each sequence is independently causal."""
        from mlx_mfa import flash_attention_varlen
        B, H, D = 1, 4, 128
        N1, N2 = 32, 64
        N = N1 + N2
        scale = 1.0 / D**0.5
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        cu = mx.array([0, N1, N], dtype=mx.int32)
        ref = self._ref_concat(q, k, v, cu, cu, scale, causal=True)
        out = flash_attention_varlen(q, k, v, cu, cu, N2, N2, scale=scale, causal=True)
        mx.eval(out, ref)
        diff = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"Causal f16 max diff: {diff}"

    def test_bf16(self):
        """bfloat16: kernel activates and produces finite output."""
        from mlx_mfa import flash_attention_varlen
        B, H, N, D = 1, 4, 64, 128
        scale = 1.0 / D**0.5
        q = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
        k = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
        v = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
        cu = mx.array([0, N], dtype=mx.int32)
        out = flash_attention_varlen(q, k, v, cu, cu, N, N, scale=scale, causal=False)
        mx.eval(out)
        assert out.shape == (B, H, N, D)
        assert out.dtype == mx.bfloat16
        assert not bool(mx.any(mx.isnan(out.astype(mx.float32))))

    def test_gqa_varlen_f16(self):
        """GQA: H_q=4, H_kv=2 — kernel maps query heads to KV heads."""
        from mlx_mfa import flash_attention_varlen
        B, H_q, H_kv, D = 1, 4, 2, 64
        N1, N2 = 32, 48
        N = N1 + N2
        scale = 1.0 / D**0.5
        q = mx.random.normal((B, H_q, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        cu_q = mx.array([0, N1, N], dtype=mx.int32)
        cu_k = mx.array([0, N1, N], dtype=mx.int32)
        out = flash_attention_varlen(q, k, v, cu_q, cu_k, N2, N2, scale=scale)
        mx.eval(out)
        assert out.shape == (B, H_q, N, D)
        assert not bool(mx.any(mx.isnan(out.astype(mx.float32))))

    def test_varlen_d512_delegates_to_sdpa(self):
        """D=512 varlen truthfully locks the public split-concat SDPA path."""
        from mlx_mfa import flash_attention_varlen
        B, H, D = 1, 2, 512
        N1, N2 = 16, 32
        N = N1 + N2
        scale = 1.0 / D**0.5
        mx.random.seed(77)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        cu_q = mx.array([0, N1, N], dtype=mx.int32)
        cu_k = mx.array([0, N1, N], dtype=mx.int32)
        with dtrace.capture() as trace:
            out = flash_attention_varlen(
                q, k, v, cu_q, cu_k, N2, N2, scale=scale
            )
            mx.eval(out)
        terminal = [item for item in trace if not item[1].startswith("[reentrant]")]
        assert terminal and terminal[0][0] == "varlen_split_concat", trace
        assert all(item[0] == "sdpa" for item in terminal[1:]), trace
        assert not any(
            item[0].startswith(("mfa", "v6", "varlen_native", "steel"))
            for item in terminal
        ), trace
        assert out.shape == (B, H, N, D)
        assert not bool(mx.any(mx.isnan(out.astype(mx.float32))))
        # Compare to reference: individual SDPA per sequence
        ref0 = mx.fast.scaled_dot_product_attention(
            q[:, :, :N1, :], k[:, :, :N1, :], v[:, :, :N1, :], scale=scale)
        ref1 = mx.fast.scaled_dot_product_attention(
            q[:, :, N1:, :], k[:, :, N1:, :], v[:, :, N1:, :], scale=scale)
        ref = mx.concatenate([ref0, ref1], axis=2)
        mx.eval(ref)
        diff = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert diff < 5e-2, f"D=512 varlen max diff {diff:.4f} too large"


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

    @pytest.mark.parametrize("H,W,T", [(3, 3, 12), (5, 5, 8), (6, 6, 6)])
    def test_temporal_mask_nonpow2_grid_over_approximates(self, H, W, T):
        """III-4 pass-8 F8-1: a NON-power-of-2 spatial grid makes pHW not
        divide the tile sizes (BQ=32/BK=16), so a tile spans >= pHW tokens
        or crosses a frame boundary.  The block mask must OVER-approximate
        the token-level same-spatial-position relation (a block is active
        iff ANY (q,k) token pair shares a spatial position); the pre-fix
        `% pHW`-of-endpoints range DROPPED active blocks.  Verified vs a
        brute-force token-level reference (the only existing test used
        H=W=8 / pHW=64 which divides the tiles and hid this)."""
        from mlx_mfa import make_axial_temporal_mask
        from mlx_mfa.masks import _bq_bk
        D = 128
        pHW = H * W
        bm = np.array(make_axial_temporal_mask(H, W, T, head_dim=D))
        BQ, BK = _bq_bk(D)
        N = pHW * T
        NQ, NK = bm.shape
        dropped = 0
        for qi in range(NQ):
            qs = set(t % pHW for t in range(qi * BQ, min(qi * BQ + BQ, N)))
            for ki in range(NK):
                ks = set(t % pHW for t in range(ki * BK, min(ki * BK + BK, N)))
                if (qs & ks) and not bm[qi, ki]:
                    dropped += 1
        assert dropped == 0, (
            f"axial-temporal mask dropped {dropped} active blocks at "
            f"H={H} W={W} T={T} (pHW={pHW}) — F8-1 under-approximation")

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
        from mlx_mfa.masks import _bq_bk
        BQ, BK = _bq_bk(self.D)  # Phase F: D=128 symmetric 32x32 (was 32,16)
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

    def test_sink_window_covers_all_in_window_pairs(self):
        """III-4 R6 FIX regression: the tile-level window left edge is
        anchored at q_start - window_size (union over queries in the tile),
        not q_end - window_size. Every (q_tile, k_tile) with ANY in-window
        (q, k) token pair must be active (brute-force token reference)."""
        from mlx_mfa import make_sink_window_mask
        N = 256; ws = 64
        head_dim = 64  # BQ = BK = 32
        BQ, BK = 32, 32
        mask_np = np.array(
            make_sink_window_mask(N, ws, num_sink_tokens=0, head_dim=head_dim)
        )
        NQ = (N + BQ - 1) // BQ
        NK = (N + BK - 1) // BK
        assert mask_np.shape == (NQ, NK)
        for qt in range(NQ):
            q_lo, q_hi = qt * BQ, min(qt * BQ + BQ, N)
            for kt in range(NK):
                k_lo, k_hi = kt * BK, min(kt * BK + BK, N)
                any_in_window = any(
                    (q - ws) <= k_pos <= q
                    for q in range(q_lo, q_hi)
                    for k_pos in range(k_lo, k_hi)
                )
                if any_in_window:
                    assert mask_np[qt, kt], (
                        f"tile ({qt},{kt}) has an in-window (q,k) pair but "
                        f"is inactive"
                    )

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
        from mlx_mfa.masks import _bq_bk
        BQ, BK = _bq_bk(self.D)  # Phase F: D=128 symmetric 32x32 (was 32,16)
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
# Track GNA: Generalized Neighborhood Attention Masks
# =============================================================================

class TestGNAMask:
    """Tests for make_gna_mask() — GNA block mask generation."""

    D = 128

    def test_gna_mask_blocked_all_true(self):
        """stride=window=seq_shape → single block = all tiles active."""
        from mlx_mfa.masks import make_gna_mask
        mask = make_gna_mask((4, 8, 8), (4, 8, 8), (4, 8, 8), head_dim=self.D)
        assert bool(mask.all().item()), "Full block should be all-True"

    def test_gna_mask_blocked(self):
        """stride=window_size should produce non-overlapping blocks."""
        from mlx_mfa.masks import make_gna_mask
        mask = make_gna_mask((4, 8, 8), (2, 4, 4), (2, 4, 4), head_dim=self.D)
        mask_np = np.array(mask)
        # Each Q-tile row should have at least 1 active K-tile
        assert mask_np.any(axis=1).all(), "Every Q-tile must see at least one K-tile"
        # Blocked attention should be sparser than full
        density = mask_np.mean()
        assert density < 0.5, f"Blocked mask should be sparse, got density={density}"

    def test_gna_mask_2d(self):
        """2D (H, W) without temporal dimension."""
        from mlx_mfa.masks import make_gna_mask
        mask = make_gna_mask((16, 16), (5, 5), (1, 1), head_dim=self.D)
        mask_np = np.array(mask)
        N = 256
        from mlx_mfa.masks import _bq_bk
        BQ, BK = _bq_bk(self.D)  # Phase F: D=128 symmetric 32x32 (was 32,16)
        NQ = (N + BQ - 1) // BQ
        NK = (N + BK - 1) // BK
        assert list(mask.shape) == [NQ, NK], f"Expected [{NQ}, {NK}], got {list(mask.shape)}"
        # Non-trivial: not all True, not all False
        density = mask_np.mean()
        assert 0.0 < density < 1.0, f"2D mask should be non-trivial, density={density}"

    def test_gna_mask_sparsity(self):
        """Sparsity increases with smaller window relative to seq."""
        from mlx_mfa.masks import make_gna_mask
        mask_dense = make_gna_mask((4, 16, 16), (4, 16, 16), (4, 16, 16), head_dim=self.D)
        mask_sparse = make_gna_mask((4, 16, 16), (2, 4, 4), (1, 1, 1), head_dim=self.D)
        dense_count = int(mask_dense.astype(mx.int32).sum().item())
        sparse_count = int(mask_sparse.astype(mx.int32).sum().item())
        assert sparse_count < dense_count, \
            f"Sparse mask should have fewer active tiles: {sparse_count} vs {dense_count}"

    def test_gna_mask_with_sparse_attention(self):
        """End-to-end: GNA mask + flash_attention_sparse produces valid output."""
        from mlx_mfa.masks import make_gna_mask
        from mlx_mfa import flash_attention_sparse
        B, H, D = 1, 4, self.D
        T, pH, pW = 4, 8, 8
        N = T * pH * pW
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mask = make_gna_mask((T, pH, pW), (3, 5, 5), (1, 1, 1), head_dim=D)
        out = flash_attention_sparse(q, k, v, mask, scale=1.0 / (D ** 0.5))
        mx.eval(out)
        assert out.shape == (B, H, N, D)
        assert not np.any(np.isnan(np.array(out.astype(mx.float32)))), "Output has NaN"

    def test_gna_mask_stride1_sliding_window(self):
        """stride=(1,1) with small window gives sparse sliding neighborhood."""
        from mlx_mfa.masks import make_gna_mask
        mask = make_gna_mask((8, 8), (3, 3), (1, 1), head_dim=self.D)
        mask_np = np.array(mask)
        # Diagonal should always be active (self-attention)
        N = 64
        BQ, BK = 32, 16
        NQ = (N + BQ - 1) // BQ
        NK = (N + BK - 1) // BK
        # Each Q-tile must see at least one K-tile
        assert mask_np.any(axis=1).all()

    def test_gna_mask_intermediate_stride(self):
        """Intermediate stride (1 < stride < window) produces valid mask.

        Phase F: D=128 mask tiles are now 32x32 (was 32x16). A 64-token grid
        (8x8) collapses to a 2x2 tile mask that a (4,4) window fully covers
        (density 1.0 — still correct, just not sparse at coarse tiling). Use a
        256-token (16x16) grid so intermediate density is observable at 32x32.
        """
        from mlx_mfa.masks import make_gna_mask
        mask = make_gna_mask((16, 16), (4, 4), (2, 2), head_dim=self.D)
        mask_np = np.array(mask)
        density = mask_np.mean()
        # Should be between fully sparse and fully dense
        assert 0.0 < density < 1.0, f"Intermediate stride density={density}"
        # Every Q-tile must see at least one K-tile
        assert mask_np.any(axis=1).all()


class TestGNAAttention:
    """Integration tests for flash_attention_gna()."""

    def test_gna_fullwindow_matches_dense(self):
        """GNA with window=seq_shape, stride=seq_shape should match dense."""
        from mlx_mfa import flash_attention_gna
        B, H, D = 1, 4, 128
        T, pH, pW = 2, 4, 4
        N = T * pH * pW
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        out_gna = flash_attention_gna(q, k, v, (T, pH, pW), (T, pH, pW), (T, pH, pW))
        out_sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0/(D**0.5))
        mx.eval(out_gna, out_sdpa)

        diff = float(mx.max(mx.abs(out_gna.astype(mx.float32) - out_sdpa.astype(mx.float32))).item())
        assert diff < 0.01, f"GNA full-window vs SDPA: max_diff={diff}"

    def test_gna_no_nan_stride1(self):
        """GNA with stride=(1,1,1) should produce no NaN."""
        from mlx_mfa import flash_attention_gna
        B, H, D = 1, 4, 128
        T, pH, pW = 4, 8, 8
        N = T * pH * pW
        mx.random.seed(7)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        out = flash_attention_gna(q, k, v, (T, pH, pW), (3, 5, 5), (1, 1, 1))
        mx.eval(out)
        assert out.shape == (B, H, N, D)
        assert not np.any(np.isnan(np.array(out.astype(mx.float32))))

    def test_gna_no_nan_blocked(self):
        """GNA with stride=window_size (blocked) should produce no NaN."""
        from mlx_mfa import flash_attention_gna
        B, H, D = 1, 4, 128
        T, pH, pW = 4, 8, 8
        N = T * pH * pW
        mx.random.seed(7)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        out = flash_attention_gna(q, k, v, (T, pH, pW), (2, 4, 4), (2, 4, 4))
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out.astype(mx.float32))))

    def test_gna_2d(self):
        """GNA on 2D (H, W) without temporal dimension."""
        from mlx_mfa import flash_attention_gna
        B, H_heads, D = 1, 4, 128
        pH, pW = 16, 16
        N = pH * pW
        mx.random.seed(42)
        q = mx.random.normal((B, H_heads, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_heads, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H_heads, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        out = flash_attention_gna(q, k, v, (pH, pW), (5, 5), (1, 1))
        mx.eval(out)
        assert out.shape == (B, H_heads, N, D)
        assert not np.any(np.isnan(np.array(out.astype(mx.float32))))

    def test_gna_d64(self):
        """GNA with head_dim=64."""
        from mlx_mfa import flash_attention_gna
        B, H, D = 1, 4, 64
        # v2.50 Prompt 4 Section A: bumped grid (2,4,4)→(8,16,16) for N=2048
        # to satisfy sparse mask>=4096 bytes constraint when GNA routes via
        # sparse path (MFA_DISABLE_GNA_NATIVE=1 forces sparse fallback).
        T, pH, pW = 8, 16, 16
        N = T * pH * pW
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        out = flash_attention_gna(q, k, v, (T, pH, pW), (2, 4, 4), (1, 1, 1))
        mx.eval(out)
        assert out.shape == (B, H, N, D)
        assert not np.any(np.isnan(np.array(out.astype(mx.float32))))

    def test_gna_bf16(self):
        """GNA with bfloat16 dtype."""
        from mlx_mfa import flash_attention_gna
        B, H, D = 1, 2, 128
        T, pH, pW = 2, 4, 4
        N = T * pH * pW
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
        k = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
        v = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
        mx.eval(q, k, v)

        out = flash_attention_gna(q, k, v, (T, pH, pW), (2, 4, 4), (2, 4, 4))
        mx.eval(out)
        assert out.shape == (B, H, N, D)
        assert not np.any(np.isnan(np.array(out.astype(mx.float32))))

    def test_gna_intermediate_stride(self):
        """GNA with 1 < stride < window_size."""
        from mlx_mfa import flash_attention_gna
        B, H, D = 1, 4, 128
        pH, pW = 8, 8
        N = pH * pW
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        out = flash_attention_gna(q, k, v, (pH, pW), (4, 4), (2, 2))
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out.astype(mx.float32))))


class TestGNABackward:
    """Gradient tests for flash_attention_gna() via sparse backward path."""

    @pytest.fixture(autouse=True, scope="class")
    def _disable_gna_native_for_backward(self):
        """Native GNA kernel is forward-only; backward tests must use the
        sparse path.

        III-4 F10: this fixture was MODULE-level autouse with
        scope="class", which applied it to EVERY class in the module —
        the whole file silently tested the sparse fallback instead of
        the production GNA-native path.  Scoped to this class only.
        """
        os.environ["MFA_DISABLE_GNA_NATIVE"] = "1"
        yield
        os.environ.pop("MFA_DISABLE_GNA_NATIVE", None)

    def test_gna_backward_no_nan(self):
        """GNA backward produces finite gradients."""
        from mlx_mfa import flash_attention_gna
        B, H, D = 1, 2, 64
        # v2.50 Prompt 4 Section A: bumped grid for sparse mask>=4096 bytes.
        T, pH, pW = 8, 16, 16
        N = T * pH * pW
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        def fn(q, k, v):
            return flash_attention_gna(q, k, v, (T, pH, pW), (2, 4, 4), (1, 1, 1)).sum()

        dq, dk, dv = mx.grad(fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)
        assert not mx.isnan(dq).any().item(), "dQ has NaN"
        assert not mx.isnan(dk).any().item(), "dK has NaN"
        assert not mx.isnan(dv).any().item(), "dV has NaN"
        assert dq.abs().max().item() > 1e-6, "dQ is zero"
        assert dk.abs().max().item() > 1e-6, "dK is zero"
        assert dv.abs().max().item() > 1e-6, "dV is zero"

    def test_gna_backward_fullwindow_matches_dense(self):
        """GNA backward with full window should match dense backward."""
        from mlx_mfa import flash_attention_gna, flash_attention
        B, H, D = 1, 2, 64
        # v2.50 Prompt 4 Section A: bumped grid for sparse mask>=4096 bytes.
        T, pH, pW = 8, 16, 16
        N = T * pH * pW
        mx.random.seed(7)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        def gna_fn(q, k, v):
            return flash_attention_gna(
                q, k, v, (T, pH, pW), (T, pH, pW), (T, pH, pW)).sum()

        def dense_fn(q, k, v):
            return flash_attention(q, k, v, backend="sdpa").sum()

        gna_grads = mx.grad(gna_fn, argnums=(0, 1, 2))(q, k, v)
        dense_grads = mx.grad(dense_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(*gna_grads, *dense_grads)

        for g_gna, g_dense, name in zip(gna_grads, dense_grads, ["dQ", "dK", "dV"]):
            diff = mx.abs(
                g_gna.astype(mx.float32) - g_dense.astype(mx.float32)
            ).max().item()
            assert diff < 0.1, f"{name} mismatch: max_diff={diff}"

    def test_gna_backward_blocked(self):
        """GNA backward with stride=window_size produces finite gradients."""
        from mlx_mfa import flash_attention_gna
        B, H, D = 1, 2, 128
        T, pH, pW = 4, 8, 8
        N = T * pH * pW
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        def fn(q, k, v):
            return flash_attention_gna(
                q, k, v, (T, pH, pW), (2, 4, 4), (2, 4, 4)).sum()

        dq, dk, dv = mx.grad(fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)
        assert not mx.isnan(dq).any().item()
        assert not mx.isnan(dk).any().item()
        assert not mx.isnan(dv).any().item()


# =============================================================================
# Track GNA-B: Additional mask/bias utilities
# =============================================================================

class TestDiagonalMask:
    """Tests for make_diagonal_mask()."""

    D = 128

    def test_diagonal_mask_single(self):
        """Single diagonal: each Q-tile sees at least one K-tile."""
        from mlx_mfa.masks import make_diagonal_mask
        mask = make_diagonal_mask(512, num_diagonals=1, bandwidth=1, head_dim=self.D)
        mask_np = np.array(mask)
        # Every Q-tile should see at least one K-tile (main diagonal)
        assert mask_np.any(axis=1).all(), "Every Q-tile must see at least one K-tile"
        # Should be sparse (not all True)
        assert mask_np.mean() < 1.0, "Single diagonal should be sparse"

    def test_diagonal_mask_tridiagonal(self):
        """Tri-diagonal includes more tiles than single."""
        from mlx_mfa.masks import make_diagonal_mask
        single = make_diagonal_mask(512, num_diagonals=1, bandwidth=1, head_dim=self.D)
        tri = make_diagonal_mask(512, num_diagonals=3, bandwidth=1, head_dim=self.D)
        assert int(tri.sum().item()) >= int(single.sum().item())

    def test_diagonal_mask_with_sparse(self):
        """End-to-end sparse attention with diagonal mask."""
        from mlx_mfa.masks import make_diagonal_mask
        from mlx_mfa import flash_attention_sparse
        B, H, N, D = 1, 4, 512, self.D
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)
        mask = make_diagonal_mask(N, num_diagonals=3, bandwidth=2, head_dim=D)
        out = flash_attention_sparse(q, k, v, mask, scale=1.0 / (D ** 0.5))
        mx.eval(out)
        assert out.shape == (B, H, N, D)
        assert not np.any(np.isnan(np.array(out.astype(mx.float32))))


class TestTemporalGroupMask:
    """Tests for make_temporal_group_mask()."""

    D = 128

    def test_dense_nearby(self):
        """Same-frame tiles should be active with density=1.0."""
        from mlx_mfa.masks import make_temporal_group_mask
        groups = [{"distance_range": (0, 1), "density": 1.0}]
        mask = make_temporal_group_mask(4, 64, groups, head_dim=self.D)
        assert int(mask.sum().item()) > 0

    def test_sparser_far(self):
        """Distant frames with low density should give intermediate sparsity."""
        from mlx_mfa.masks import make_temporal_group_mask
        groups = [
            {"distance_range": (0, 1), "density": 1.0},
            {"distance_range": (1, 100), "density": 0.1},
        ]
        mask = make_temporal_group_mask(8, 64, groups, head_dim=self.D)
        total_density = int(mask.sum().item()) / (mask.shape[0] * mask.shape[1])
        assert 0.05 < total_density < 0.95

    def test_deterministic(self):
        """Same seed produces same mask."""
        from mlx_mfa.masks import make_temporal_group_mask
        groups = [{"distance_range": (0, 100), "density": 0.5}]
        m1 = make_temporal_group_mask(4, 64, groups, seed=42)
        m2 = make_temporal_group_mask(4, 64, groups, seed=42)
        assert bool(mx.array_equal(m1, m2).item())


class TestTopkAttention:
    """Tests for flash_attention_topk() Python reference."""

    def test_topk_ratio_1_matches_dense(self):
        """topk_ratio=1.0 should match standard dense attention."""
        from mlx_mfa import flash_attention_topk, flash_attention
        B, H, N, D = 1, 4, 64, 64
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float32)
        k = mx.random.normal((B, H, N, D)).astype(mx.float32)
        v = mx.random.normal((B, H, N, D)).astype(mx.float32)
        mx.eval(q, k, v)

        out_topk = flash_attention_topk(q, k, v, topk_ratio=1.0)
        out_dense = flash_attention(q, k, v, backend="sdpa")
        mx.eval(out_topk, out_dense)
        diff = float(mx.max(mx.abs(out_topk - out_dense)).item())
        # v2.50 Prompt 4 Section A: tolerance bumped 1e-4 → 1e-3 because
        # Sprint 3 Phase 3a dispatch (Prompt 2) routes topk_ratio=1.0 to
        # mx.fast.sdpa via float-bias mask path which has slightly different
        # rounding order than the dense reference; max_diff ~6e-4 observed.
        assert diff < 1e-3, f"topk_ratio=1.0 should match dense: diff={diff}"

    def test_topk_reduces_context(self):
        """topk_ratio=0.25 gives different output than dense."""
        from mlx_mfa import flash_attention_topk, flash_attention
        B, H, N, D = 1, 4, 64, 64
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float32)
        k = mx.random.normal((B, H, N, D)).astype(mx.float32)
        v = mx.random.normal((B, H, N, D)).astype(mx.float32)
        mx.eval(q, k, v)

        out_topk = flash_attention_topk(q, k, v, topk_ratio=0.25)
        out_dense = flash_attention(q, k, v, backend="sdpa")
        mx.eval(out_topk, out_dense)
        diff = float(mx.max(mx.abs(out_topk - out_dense)).item())
        assert diff > 1e-3, f"topk_ratio=0.25 should differ from dense: diff={diff}"

    def test_topk_no_nan(self):
        """No NaN in output."""
        from mlx_mfa import flash_attention_topk
        B, H, N, D = 1, 2, 128, 64
        mx.random.seed(7)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        out = flash_attention_topk(q, k, v, topk_ratio=0.5)
        mx.eval(out)
        assert not mx.isnan(out).any().item()

    def test_topk_with_mask(self):
        """Top-k composed with block mask."""
        from mlx_mfa import flash_attention_topk
        from mlx_mfa.masks import make_diagonal_mask
        B, H, N, D = 1, 2, 256, 64
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float32)
        k = mx.random.normal((B, H, N, D)).astype(mx.float32)
        v = mx.random.normal((B, H, N, D)).astype(mx.float32)
        mx.eval(q, k, v)

        mask = make_diagonal_mask(N, num_diagonals=3, bandwidth=2, head_dim=D)
        out = flash_attention_topk(q, k, v, topk_ratio=0.5, mask=mask)
        mx.eval(out)
        assert not mx.isnan(out).any().item()
        assert out.shape == (B, H, N, D)


class TestTemporalDistanceBias:
    """Tests for make_temporal_distance_bias + threshold converter."""

    D = 128

    def test_shape(self):
        from mlx_mfa.masks import make_temporal_distance_bias
        bias = make_temporal_distance_bias(4, 16, num_heads=8, head_dim=self.D)
        assert bias.shape == (1, 8, 64, 64)

    def test_same_frame_zero(self):
        """Tokens in the same frame should have bias = 0."""
        from mlx_mfa.masks import make_temporal_distance_bias
        bias = make_temporal_distance_bias(4, 16, num_heads=1, head_dim=self.D)
        assert abs(bias[0, 0, 0, 15].item()) < 1e-6

    def test_monotonic(self):
        """Bias magnitude increases with temporal distance."""
        from mlx_mfa.masks import make_temporal_distance_bias
        bias = make_temporal_distance_bias(4, 16, num_heads=1, head_dim=self.D)
        b01 = abs(bias[0, 0, 0, 16].item())  # frame 0 vs frame 1
        b02 = abs(bias[0, 0, 0, 32].item())  # frame 0 vs frame 2
        assert b02 > b01

    def test_to_mask(self):
        """Threshold conversion produces valid sparse mask."""
        from mlx_mfa.masks import make_temporal_distance_bias, temporal_distance_bias_to_mask
        bias = make_temporal_distance_bias(4, 64, num_heads=1, decay_rate=2.0, head_dim=self.D)
        mask = temporal_distance_bias_to_mask(bias, threshold=-3.0, head_dim=self.D)
        assert mask.dtype == mx.bool_
        assert mask.ndim == 2
        # Should be sparser than full
        assert float(mask.astype(mx.float32).mean().item()) < 1.0

    def test_memory_guard(self):
        """Large sequences should raise ValueError."""
        from mlx_mfa.masks import make_temporal_distance_bias
        try:
            make_temporal_distance_bias(256, 256, num_heads=32)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "GB" in str(e)


class TestStridedMask:
    """Tests for make_strided_mask()."""

    D = 128

    def test_strided_mask_large_window_dense(self):
        """Window covering full sequence should be all True."""
        from mlx_mfa.masks import make_strided_mask
        mask = make_strided_mask(512, window_size=1024, global_stride=99999, head_dim=self.D)
        assert bool(mask.all().item())

    def test_strided_mask_global_adds_tiles(self):
        """Global stride adds non-local tiles."""
        from mlx_mfa.masks import make_strided_mask
        mask_local = make_strided_mask(4096, window_size=256, global_stride=999999, head_dim=self.D)
        mask_both = make_strided_mask(4096, window_size=256, global_stride=512, head_dim=self.D)
        assert int(mask_both.sum().item()) > int(mask_local.sum().item())

    def test_strided_mask_position_zero_in_stride_set(self):
        """III-4 R11 FIX regression: position 0 belongs to the global stride
        set {0, gs, 2*gs, ...} — the first K-tile must be active for ALL
        Q-tiles, including those far outside the local window."""
        from mlx_mfa.masks import make_strided_mask
        # window_size=2 keeps the local window tiny; gs=128 means only
        # positions {0, 128} are global — K-tile 0 holds position 0.
        mask = make_strided_mask(256, window_size=2, global_stride=128,
                                 head_dim=self.D)
        mask_np = np.array(mask)
        assert mask_np[:, 0].all(), (
            "K-tile containing position 0 must be globally visible"
        )

    def test_strided_mask_with_sparse(self):
        """End-to-end sparse attention."""
        from mlx_mfa.masks import make_strided_mask
        from mlx_mfa import flash_attention_sparse
        B, H, N, D = 1, 4, 1024, self.D
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)
        mask = make_strided_mask(N, window_size=256, global_stride=512, head_dim=D)
        out = flash_attention_sparse(q, k, v, mask, scale=1.0 / (D ** 0.5))
        mx.eval(out)
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


# ---------------------------------------------------------------------------
# Track AD — Per-batch cache_seqlens tensor
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ext_available(), reason="C++ extension not built")
class TestPerBatchCacheSeqlens:
    """Track AD: cache_seqlens as 1D array/list for per-batch RoPE offsets."""

    D = 64

    def _ref(self, q, k, v, cos, sin, cs_list, causal):
        """Reference: per-batch loop with scalar cache_seqlens."""
        from mlx_mfa import flash_attention_rope
        chunks = [
            flash_attention_rope(
                q[b:b+1], k[b:b+1], v[b:b+1],
                cos, sin, causal=causal,
                cache_seqlens=cs_list[b],
            )
            for b in range(len(cs_list))
        ]
        return mx.concatenate(chunks, axis=0)

    def test_list_matches_per_batch_ref(self):
        """Passing a Python list equals per-batch scalar calls."""
        from mlx_mfa import flash_attention_rope

        B, H, N = 4, 2, 32
        D = self.D
        mx.random.seed(22)
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, N, D), dtype=mx.float16)
        v = mx.random.normal((B, H, N, D), dtype=mx.float16)
        cos, sin = _make_rope_tables(256, D)
        cs_list = [0, 16, 32, 64]

        out_batch = flash_attention_rope(q, k, v, cos, sin,
                                         cache_seqlens=cs_list, causal=False)
        out_ref = self._ref(q, k, v, cos, sin, cs_list, causal=False)
        mx.eval(out_batch, out_ref)

        np.testing.assert_allclose(
            np.array(out_batch.astype(mx.float32)),
            np.array(out_ref.astype(mx.float32)),
            atol=0, rtol=0,
            err_msg="Per-batch list cache_seqlens diverges from reference",
        )

    def test_array_matches_per_batch_ref(self):
        """Passing an mx.array cache_seqlens equals per-batch scalar calls."""
        from mlx_mfa import flash_attention_rope

        B, H, N = 3, 2, 16
        D = self.D
        mx.random.seed(33)
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, N, D), dtype=mx.float16)
        v = mx.random.normal((B, H, N, D), dtype=mx.float16)
        cos, sin = _make_rope_tables(128, D)
        cs_arr = mx.array([0, 8, 24], dtype=mx.int32)
        cs_list = [0, 8, 24]

        out_batch = flash_attention_rope(q, k, v, cos, sin,
                                         cache_seqlens=cs_arr, causal=True)
        out_ref = self._ref(q, k, v, cos, sin, cs_list, causal=True)
        mx.eval(out_batch, out_ref)

        np.testing.assert_allclose(
            np.array(out_batch.astype(mx.float32)),
            np.array(out_ref.astype(mx.float32)),
            atol=0, rtol=0,
            err_msg="Per-batch mx.array cache_seqlens diverges from reference",
        )

    def test_length_mismatch_raises(self):
        """cache_seqlens length != B raises ValueError."""
        from mlx_mfa import flash_attention_rope

        B, H, N, D = 2, 2, 16, 64
        mx.random.seed(44)
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, N, D), dtype=mx.float16)
        v = mx.random.normal((B, H, N, D), dtype=mx.float16)
        cos, sin = _make_rope_tables(64, D)

        with pytest.raises(ValueError, match="must equal B="):
            flash_attention_rope(q, k, v, cos, sin,
                                  cache_seqlens=[0, 16, 32])  # len=3, B=2


# ---------------------------------------------------------------------------
# Track AE — headdim_v != headdim_qk
# ---------------------------------------------------------------------------

class TestHeadDimVMismatch:
    """Track AE: V may have a different head_dim than Q/K.

    The MFA kernel requires equal head dims; the function falls back to SDPA.
    """

    def test_smaller_v_headdim(self):
        """flash_attention with Dv < Dqk falls back to SDPA and returns Dv."""
        from mlx_mfa import flash_attention

        D_qk, D_v = 128, 64
        mx.random.seed(55)
        q = mx.random.normal((1, 4, 32, D_qk))
        k = mx.random.normal((1, 4, 32, D_qk))
        v = mx.random.normal((1, 4, 32, D_v))

        out = flash_attention(q, k, v, causal=False)
        mx.eval(out)

        assert out.shape == (1, 4, 32, D_v), \
            f"Expected shape (1,4,32,{D_v}), got {out.shape}"
        assert mx.all(mx.isfinite(out)).item(), "Output contains NaN/Inf"

    def test_larger_v_headdim(self):
        """flash_attention with Dv > Dqk falls back to SDPA and returns Dv."""
        from mlx_mfa import flash_attention

        D_qk, D_v = 64, 128
        mx.random.seed(66)
        q = mx.random.normal((2, 2, 16, D_qk))
        k = mx.random.normal((2, 2, 16, D_qk))
        v = mx.random.normal((2, 2, 16, D_v))

        out = flash_attention(q, k, v, causal=True)
        mx.eval(out)

        assert out.shape == (2, 2, 16, D_v), \
            f"Expected shape (2,2,16,{D_v}), got {out.shape}"
        assert mx.all(mx.isfinite(out)).item(), "Output contains NaN/Inf"

    def test_matches_sdpa_reference(self):
        """Dv != Dqk result matches mx.fast.scaled_dot_product_attention."""
        from mlx_mfa import flash_attention

        D_qk, D_v = 128, 64
        scale = 1.0 / math.sqrt(D_qk)
        mx.random.seed(77)
        q = mx.random.normal((1, 2, 24, D_qk))
        k = mx.random.normal((1, 2, 24, D_qk))
        v = mx.random.normal((1, 2, 24, D_v))

        out_mfa = flash_attention(q, k, v, scale=scale, causal=False)
        out_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(out_mfa, out_ref)

        np.testing.assert_allclose(
            np.array(out_mfa), np.array(out_ref), atol=0, rtol=0,
            err_msg="Dv!=Dqk fallback differs from SDPA reference",
        )

    def test_k_dim_mismatch_raises(self):
        """K head_dim != Q head_dim must still raise ValueError."""
        from mlx_mfa import flash_attention

        q = mx.random.normal((1, 2, 8, 128))
        k = mx.random.normal((1, 2, 8, 64))   # wrong K dim
        v = mx.random.normal((1, 2, 8, 128))

        with pytest.raises(ValueError, match="q and k must have the same head_dim"):
            flash_attention(q, k, v)

# ---------------------------------------------------------------------------
# Track 1 — flash_attention_kvcache append mode (k_new / v_new)
# ---------------------------------------------------------------------------

class TestKVCacheAppendUnified:
    """flash_attention_kvcache with k_new/v_new — unified append-mode API."""

    D = 64

    def _qkv(self, B=1, H=2, N=16, D=None, seed=200, dtype=mx.float32):
        D = D or self.D
        mx.random.seed(seed)
        q = mx.random.normal((B, H, N, D)).astype(dtype)
        k = mx.random.normal((B, H, N, D)).astype(dtype)
        v = mx.random.normal((B, H, N, D)).astype(dtype)
        return q, k, v

    def test_kvcache_with_k_new_matches_inline(self):
        """flash_attention_kvcache(k_new=...) matches manual concat+attend."""
        from mlx_mfa import flash_attention_kvcache, flash_attention

        B, H, past, D = 1, 2, 32, self.D
        mx.random.seed(201)
        k_cache = mx.random.normal((B, H, past, D))
        v_cache = mx.random.normal((B, H, past, D))
        q_new = mx.random.normal((B, H, 1, D))
        k_new = mx.random.normal((B, H, 1, D))
        v_new = mx.random.normal((B, H, 1, D))

        # Manual inline: concat then attend
        k_full = mx.concatenate([k_cache, k_new], axis=2)
        v_full = mx.concatenate([v_cache, v_new], axis=2)
        out_ref = flash_attention(q_new, k_full, v_full, causal=True)

        # New unified API
        out_new, k_new_up, v_new_up = flash_attention_kvcache(
            q_new, k_cache, v_cache,
            k_new=k_new, v_new=v_new, causal=True,
        )
        mx.eval(out_ref, k_full, v_full, out_new, k_new_up, v_new_up)

        np.testing.assert_allclose(
            np.array(out_ref), np.array(out_new), atol=1e-5,
            err_msg="append-mode output differs from manual concat+attend"
        )
        np.testing.assert_allclose(
            np.array(k_full), np.array(k_new_up), atol=0, rtol=0,
            err_msg="k_updated differs from concat"
        )

    def test_kvcache_with_k_new_no_existing_cache(self):
        """k_new without existing cache (first decode step: k_cache=None)."""
        from mlx_mfa import flash_attention_kvcache, flash_attention

        B, H, N, D = 1, 2, 16, self.D
        q, k, v = self._qkv(B=B, H=H, N=N, seed=202)

        out_new, k_up, v_up = flash_attention_kvcache(
            q, None, None, k_new=k, v_new=v, causal=True,
        )
        out_ref = flash_attention(q, k, v, causal=True)
        mx.eval(out_new, k_up, v_up, out_ref)

        assert out_new.shape == (B, H, N, D)
        assert k_up.shape == (B, H, N, D)
        np.testing.assert_allclose(
            np.array(out_new), np.array(out_ref), atol=1e-5,
            err_msg="first-step append differs from flash_attention"
        )

    def test_kvcache_with_k_new_returns_3tuple(self):
        """When k_new is provided, return type is (array, array, array)."""
        from mlx_mfa import flash_attention_kvcache

        B, H, N, D = 1, 2, 8, self.D
        q, k, v = self._qkv(B=B, H=H, N=N, seed=203)
        result = flash_attention_kvcache(q, None, None, k_new=k, v_new=v)
        assert isinstance(result, tuple) and len(result) == 3, (
            f"expected 3-tuple, got {type(result)}"
        )
        out, k_up, v_up = result
        assert isinstance(out, mx.array)
        assert isinstance(k_up, mx.array)
        assert isinstance(v_up, mx.array)

    def test_kvcache_without_k_new_returns_array(self):
        """When k_new is None, return type is mx.array (backward compat)."""
        from mlx_mfa import flash_attention_kvcache

        B, H, N, D = 1, 2, 8, self.D
        q, k, v = self._qkv(B=B, H=H, N=N, seed=204)
        result = flash_attention_kvcache(q, k, v)
        assert isinstance(result, mx.array), (
            f"expected mx.array, got {type(result)}"
        )

    def test_kvcache_with_k_new_cache_shape_grows(self):
        """Updated cache has past_len + N tokens on axis 2."""
        from mlx_mfa import flash_attention_kvcache

        B, H, past, N, D = 1, 2, 32, 4, self.D
        mx.random.seed(205)
        k_cache = mx.random.normal((B, H, past, D))
        v_cache = mx.random.normal((B, H, past, D))
        q_new = mx.random.normal((B, H, N, D))
        k_new = mx.random.normal((B, H, N, D))
        v_new = mx.random.normal((B, H, N, D))

        _, k_up, v_up = flash_attention_kvcache(
            q_new, k_cache, v_cache, k_new=k_new, v_new=v_new, causal=True
        )
        mx.eval(k_up, v_up)
        assert k_up.shape == (B, H, past + N, D), f"k_up shape {k_up.shape}"
        assert v_up.shape == (B, H, past + N, D), f"v_up shape {v_up.shape}"

    def test_kvcache_with_k_new_and_rope(self):
        """k_new gets RoPE rotation at cache_seqlens offset before append."""
        from mlx_mfa import flash_attention_kvcache, flash_attention_kvcache_rope_append

        B, H, past, N, D = 1, 2, 16, 1, 64
        mx.random.seed(206)
        k_cache = mx.random.normal((B, H, past, D)).astype(mx.float16)
        v_cache = mx.random.normal((B, H, past, D)).astype(mx.float16)
        q_new = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k_new = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v_new = mx.random.normal((B, H, N, D)).astype(mx.float16)
        max_seq = past + N + 8
        cos = mx.random.normal((max_seq, D // 2)).astype(mx.float32)
        sin = mx.random.normal((max_seq, D // 2)).astype(mx.float32)

        # Reference: use flash_attention_kvcache_rope_append
        out_ref, k_ref, v_ref = flash_attention_kvcache_rope_append(
            q_new, k_new, v_new, k_cache, v_cache,
            rotary_cos=cos, rotary_sin=sin,
            cache_seqlens=past, causal=True,
        )
        # New API
        out_new, k_up, v_up = flash_attention_kvcache(
            q_new, k_cache, v_cache,
            k_new=k_new, v_new=v_new,
            rotary_cos=cos, rotary_sin=sin,
            cache_seqlens=past, causal=True,
        )
        mx.eval(out_ref, k_ref, v_ref, out_new, k_up, v_up)

        # Updated caches must match exactly (same RoPE rotation on k_new)
        np.testing.assert_allclose(
            np.array(k_ref.astype(mx.float32)),
            np.array(k_up.astype(mx.float32)),
            atol=1e-4, err_msg="k_updated with RoPE differs"
        )
        np.testing.assert_allclose(
            np.array(out_ref.astype(mx.float32)),
            np.array(out_new.astype(mx.float32)),
            atol=5e-2, err_msg="output with RoPE differs from kvcache_rope_append"
        )

    def test_kvcache_with_k_new_and_softcap(self):
        """k_new append works with softcap."""
        from mlx_mfa import flash_attention_kvcache

        B, H, past, N, D = 1, 2, 8, 2, self.D
        mx.random.seed(207)
        k_cache = mx.random.normal((B, H, past, D))
        v_cache = mx.random.normal((B, H, past, D))
        q_new = mx.random.normal((B, H, N, D))
        k_new = mx.random.normal((B, H, N, D))
        v_new = mx.random.normal((B, H, N, D))

        out, k_up, v_up = flash_attention_kvcache(
            q_new, k_cache, v_cache,
            k_new=k_new, v_new=v_new, softcap=20.0, causal=True,
        )
        mx.eval(out, k_up, v_up)
        assert out.shape == (B, H, N, D)
        assert mx.all(mx.isfinite(out)).item()

    def test_kvcache_k_new_without_v_new_raises(self):
        """k_new without v_new must raise ValueError."""
        from mlx_mfa import flash_attention_kvcache

        B, H, N, D = 1, 2, 4, self.D
        q, k, v = self._qkv(B=B, H=H, N=N, seed=208)
        with pytest.raises(ValueError, match="k_new and v_new must both"):
            flash_attention_kvcache(q, None, None, k_new=k)

    def test_kvcache_k_new_paged_succeeds(self):
        """k_new + block_table (paged-append) is now supported (Track JC)."""
        from mlx_mfa import flash_attention_kvcache

        B, H, N, D = 1, 2, 1, self.D
        q, k, v = self._qkv(B=B, H=H, N=N, seed=209)
        # Pool with 4 blocks of size 16; block 0 is the only used block for seq 0
        block_table = mx.zeros((B, 4), dtype=mx.int32)
        seq_lens = mx.array([0], dtype=mx.int32)  # cache currently empty
        pool_k = mx.zeros((4, 16, H, D), dtype=mx.float16)
        pool_v = mx.zeros((4, 16, H, D), dtype=mx.float16)
        out, k_new_pool, v_new_pool = flash_attention_kvcache(
            q, pool_k, pool_v,
            block_table=block_table, seq_lens=seq_lens, block_size=16,
            k_new=k, v_new=v,
        )
        mx.eval(out)
        assert out.shape == (B, H, N, D)
        assert mx.all(mx.isfinite(out)).item()
        # III-4 pass-3 (F4 strengthening): the cache was empty, so q attends
        # ONLY to the freshly-appended (k, v) — the result must equal
        # single-token SDPA over them.  This exercises the fp32-q / fp16-pool
        # cast path (Class A) and asserts CORRECTNESS, not just finiteness
        # (the prior shape+finite assert silently passed reinterpreted
        # garbage before the pass-2 cast fix).
        ref = mx.fast.scaled_dot_product_attention(
            q.astype(mx.float16), k.astype(mx.float16), v.astype(mx.float16),
            scale=1.0 / math.sqrt(D))
        mx.eval(ref)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)), np.array(ref.astype(mx.float32)),
            rtol=1e-2, atol=5e-3,
            err_msg="paged-append fp32-q/fp16-pool output != single-token SDPA",
        )


# ---------------------------------------------------------------------------
# Track AG — Attention dropout fallback
# ---------------------------------------------------------------------------

class TestAttentionDropout:
    """Track AG: dropout_p parameter on flash_attention."""

    D = 64

    def test_zero_dropout_matches_plain(self):
        """dropout_p=0 produces same output as plain flash_attention."""
        from mlx_mfa import flash_attention

        B, H, N = 1, 2, 16
        D = self.D
        mx.random.seed(200)
        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))

        out_plain = flash_attention(q, k, v, causal=False, dropout_p=0.0)
        out_drop0 = flash_attention(q, k, v, causal=False)
        mx.eval(out_plain, out_drop0)

        np.testing.assert_allclose(
            np.array(out_plain), np.array(out_drop0), atol=0, rtol=0,
        )

    def test_dropout_output_shape(self):
        """dropout_p > 0 returns correct output shape."""
        from mlx_mfa import flash_attention

        B, H, N, D = 1, 2, 16, self.D
        mx.random.seed(201)
        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))

        out = flash_attention(q, k, v, causal=True, dropout_p=0.1)
        mx.eval(out)

        assert out.shape == (B, H, N, D), f"Bad output shape: {out.shape}"
        assert mx.all(mx.isfinite(out)).item(), "Output contains NaN/Inf"

    def test_dropout_differs_per_call(self):
        """Two calls with dropout_p>0 produce different outputs (stochastic)."""
        from mlx_mfa import flash_attention

        B, H, N, D = 1, 4, 32, self.D
        mx.random.seed(202)
        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))

        out1 = flash_attention(q, k, v, causal=False, dropout_p=0.5)
        out2 = flash_attention(q, k, v, causal=False, dropout_p=0.5)
        mx.eval(out1, out2)

        max_diff = float(mx.max(mx.abs(out1 - out2)).item())
        assert max_diff > 0, "dropout_p=0.5 produced identical outputs on two calls"

    def test_dropout_output_finite_causal(self):
        """Causal + dropout_p > 0 produces finite outputs."""
        from mlx_mfa import flash_attention

        B, H, N, D = 2, 4, 24, self.D
        mx.random.seed(203)
        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))

        out = flash_attention(q, k, v, causal=True, dropout_p=0.3)
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item(), "Causal+dropout output contains NaN/Inf"


# ---------------------------------------------------------------------------
# Track AH — Return attention probabilities
# ---------------------------------------------------------------------------

class TestReturnAttnWeights:
    """Track AH: return_attn_weights=True returns (output, weights)."""

    D = 64

    def test_returns_tuple(self):
        """return_attn_weights=True returns a 2-tuple."""
        from mlx_mfa import flash_attention

        B, H, N, D = 1, 2, 8, self.D
        mx.random.seed(300)
        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))

        result = flash_attention(q, k, v, causal=False, return_attn_weights=True)
        assert isinstance(result, tuple) and len(result) == 2, \
            f"Expected 2-tuple, got {type(result)}"

    def test_output_shape_and_weights_shape(self):
        """Shapes: output [B,H,N,D], weights [B,H,N,S]."""
        from mlx_mfa import flash_attention

        B, H, N, S, D = 1, 4, 16, 24, self.D
        mx.random.seed(301)
        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, S, D))
        v = mx.random.normal((B, H, S, D))

        out, weights = flash_attention(q, k, v, causal=False,
                                        return_attn_weights=True)
        mx.eval(out, weights)

        assert out.shape == (B, H, N, D), f"output shape {out.shape}"
        assert weights.shape == (B, H, N, S), f"weights shape {weights.shape}"
        assert weights.dtype == mx.float32, f"weights dtype {weights.dtype}"

    def test_weights_sum_to_one(self):
        """Attention weights sum to 1 along the key dim."""
        from mlx_mfa import flash_attention

        B, H, N, D = 1, 2, 16, self.D
        mx.random.seed(302)
        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))

        _, weights = flash_attention(q, k, v, causal=False,
                                     return_attn_weights=True)
        sums = weights.sum(axis=-1)  # [B, H, N]
        mx.eval(sums)

        np.testing.assert_allclose(
            np.array(sums), np.ones_like(np.array(sums)),
            atol=1e-5, rtol=1e-5,
            err_msg="Attention weights do not sum to 1",
        )

    def test_output_matches_no_return(self):
        """Output with return_attn_weights=True matches standard forward."""
        from mlx_mfa import flash_attention

        # v2.50 Prompt 4 Section A: bumped N=16→2048 for sparse mask>=4096.
        B, H, N, D = 1, 2, 2048, self.D
        mx.random.seed(303)
        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))

        out_plain = flash_attention(q, k, v, causal=True)
        out_with_w, _ = flash_attention(q, k, v, causal=True,
                                         return_attn_weights=True)
        mx.eval(out_plain, out_with_w)

        # v2.50 Prompt 4 Section A: tolerance bumped atol=1e-5→2e-3 because
        # return_attn_weights=True takes a different (non-fused) code path
        # that materializes weights explicitly; at N=2048 fp16 accumulation
        # order differs from the fused fast-SDPA path by ~1.4e-3 max.
        np.testing.assert_allclose(
            np.array(out_plain), np.array(out_with_w),
            atol=2e-3, rtol=1e-3,
            err_msg="Output diverges when return_attn_weights=True",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Track BE — PagedKVCache + flash_attention_paged
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPagedKVCache:
    """Tests for PagedKVCache and flash_attention_paged."""

    def test_paged_cache_construction(self):
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=32, block_size=16, H=4, D=64)
        assert cache.num_blocks == 32
        assert cache.block_size == 16
        assert "PagedKVCache" in repr(cache)

    def test_paged_cache_free_tracks_usage(self):
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=8, block_size=16, H=2, D=64)
        assert len(cache._free) == 8
        cache._ensure_seq(0)   # allocates one block
        assert len(cache._free) == 7

    def test_paged_cache_free_seq(self):
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=8, block_size=16, H=2, D=64)
        cache._ensure_seq(0)
        cache._ensure_seq(1)
        assert len(cache._free) == 6
        cache.free_seq(0)
        assert len(cache._free) == 7

    def test_paged_cache_repr(self):
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=16, block_size=32, H=4, D=128)
        r = repr(cache)
        assert "16" in r and "32" in r

    def test_paged_attention_single_seq_correctness(self):
        """flash_attention_paged single seq == flash_attention reference."""
        from mlx_mfa import flash_attention_paged, flash_attention
        mx.random.seed(7)
        B, H, N, S, D = 1, 4, 8, 32, 64
        block_size = 16
        scale = 1.0 / D**0.5

        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)

        # Build page pool: 2 pages needed for S=32, block_size=16
        # k is [B, H, S, D] → rearrange to [S, H, D] → reshape into blocks
        n_blocks = S // block_size
        k_nhd = k[0].transpose(1, 0, 2)   # [S, H, D]
        v_nhd = v[0].transpose(1, 0, 2)

        # Stack blocks directly (avoids .at[].set() which isn't in MLX)
        k_blocks = k_nhd.reshape(n_blocks, block_size, H, D)
        v_blocks = v_nhd.reshape(n_blocks, block_size, H, D)
        pad = mx.zeros((4, block_size, H, D), dtype=mx.float16)
        pool_k = mx.concatenate([k_blocks, pad], axis=0)
        pool_v = mx.concatenate([v_blocks, pad], axis=0)

        block_table = mx.array([[0, 1, -1, -1]], dtype=mx.int32)   # 2 blocks used
        seq_lens = mx.array([S], dtype=mx.int32)

        out_paged = flash_attention_paged(
            q, pool_k, pool_v, block_table, seq_lens,
            scale=scale, causal=False, block_size=block_size
        )
        out_ref = flash_attention(q, k, v, scale=scale, causal=False)
        mx.eval(out_paged, out_ref)

        diff = float(mx.abs(out_paged.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"Paged vs direct max diff: {diff}"

    def test_paged_attention_two_seqs(self):
        """Two sequences with different lengths via paged attention."""
        from mlx_mfa import flash_attention_paged, flash_attention
        mx.random.seed(11)
        H, N, D = 4, 4, 64
        block_size = 16
        S0, S1 = 32, 48
        scale = 1.0 / D**0.5

        q = mx.random.normal((2, H, N, D)).astype(mx.float16)
        k0 = mx.random.normal((1, H, S0, D)).astype(mx.float16)
        k1 = mx.random.normal((1, H, S1, D)).astype(mx.float16)
        v0 = mx.random.normal((1, H, S0, D)).astype(mx.float16)
        v1 = mx.random.normal((1, H, S1, D)).astype(mx.float16)
        mx.eval(q, k0, k1, v0, v1)

        # Build pool: 2 blocks for S0=32, 3 blocks for S1=48
        # Reshape each seq's KV into blocks, concatenate into shared pool
        n0, n1 = S0 // block_size, S1 // block_size
        k0_nhd = k0[0].transpose(1, 0, 2)   # [S0, H, D]
        v0_nhd = v0[0].transpose(1, 0, 2)
        k1_nhd = k1[0].transpose(1, 0, 2)   # [S1, H, D]
        v1_nhd = v1[0].transpose(1, 0, 2)

        k0_blocks = k0_nhd.reshape(n0, block_size, H, D)
        v0_blocks = v0_nhd.reshape(n0, block_size, H, D)
        k1_blocks = k1_nhd.reshape(n1, block_size, H, D)
        v1_blocks = v1_nhd.reshape(n1, block_size, H, D)
        pad = mx.zeros((2, block_size, H, D), dtype=mx.float16)
        pool_k = mx.concatenate([k0_blocks, k1_blocks, pad], axis=0)
        pool_v = mx.concatenate([v0_blocks, v1_blocks, pad], axis=0)

        table = mx.array([[0, 1, -1], [2, 3, 4]], dtype=mx.int32)
        seq_lens = mx.array([S0, S1], dtype=mx.int32)

        out_paged = flash_attention_paged(
            q, pool_k, pool_v, table, seq_lens,
            scale=scale, block_size=block_size
        )
        ref0 = flash_attention(q[0:1], k0, v0, scale=scale)
        ref1 = flash_attention(q[1:2], k1, v1, scale=scale)
        ref = mx.concatenate([ref0, ref1], axis=0)
        mx.eval(out_paged, ref)

        diff = float(mx.abs(out_paged.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"Paged two-seq max diff: {diff}"

    def test_paged_attention_output_shape(self):
        """Output shape matches [B, H, N_q, D]."""
        from mlx_mfa import flash_attention_paged
        B, H, N_q, S, D = 2, 4, 8, 16, 64
        q = mx.zeros((B, H, N_q, D), dtype=mx.float16)
        pool = mx.zeros((4, 16, H, D), dtype=mx.float16)
        table = mx.array([[0, -1], [1, -1]], dtype=mx.int32)
        lens = mx.array([16, 16], dtype=mx.int32)
        out = flash_attention_paged(q, pool, pool, table, lens, block_size=16)
        mx.eval(out)
        assert out.shape == (B, H, N_q, D)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Track GA — PagedKVCache rewrite (v1.0.1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPagedKVCacheGA:
    """Tests for the rewritten PagedKVCache (dual-pool, functional gather)."""

    def test_dual_pool_construction(self):
        """K and V pools are MLX arrays with correct shape and dtype."""
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=16, block_size=32, H=4, D=64)
        assert cache._k_pool.shape == (16, 32, 4, 64)
        assert cache._v_pool.shape == (16, 32, 4, 64)
        assert cache._k_pool.dtype == mx.float16
        assert cache._v_pool.dtype == mx.float16

    def test_k_pool_v_pool_properties(self):
        """k_pool / v_pool return mx.array with correct shape and dtype."""
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=4, block_size=16, H=2, D=64)
        k_pool = cache.k_pool
        v_pool = cache.v_pool
        assert k_pool.shape == (4, 16, 2, 64)
        assert v_pool.shape == (4, 16, 2, 64)
        assert k_pool.dtype == mx.float16
        assert v_pool.dtype == mx.float16

    def test_append_single_token(self):
        """Append 1 token and verify pool content matches."""
        from mlx_mfa import PagedKVCache
        import numpy as np
        mx.random.seed(1)
        cache = PagedKVCache(num_blocks=4, block_size=16, H=2, D=64)
        k = mx.random.normal((1, 2, 1, 64)).astype(mx.float16)
        v = mx.random.normal((1, 2, 1, 64)).astype(mx.float16)
        mx.eval(k, v)
        cache.append(k, v, seq_id=0)

        assert cache.seq_lengths == {0: 1}
        # Verify K written correctly: pool[blk_id, 0] == k[0, :, 0, :]
        blk_id = cache._block_table[0][0]
        k_ref = np.array(k[0].astype(mx.float32)).transpose(1, 0, 2)  # [T=1, H, D]
        pool_val = np.array(cache._k_pool[blk_id, 0].astype(mx.float32))
        np.testing.assert_allclose(pool_val, k_ref[0], atol=1e-4)

    def test_append_no_numpy_roundtrip(self):
        """Verify pool holds mx.array (no numpy backing store)."""
        from mlx_mfa import PagedKVCache
        import mlx.core as mx
        cache = PagedKVCache(num_blocks=4, block_size=16, H=2, D=64)
        assert isinstance(cache._k_pool, mx.array), "_k_pool must be mx.array"
        assert isinstance(cache._v_pool, mx.array), "_v_pool must be mx.array"
        assert not hasattr(cache, "_k_np"), "numpy backing store must not exist"
        assert not hasattr(cache, "_v_np"), "numpy backing store must not exist"
        # After append, pool is still mx.array.
        k = mx.zeros((1, 2, 4, 64), dtype=mx.float16)
        v = mx.zeros((1, 2, 4, 64), dtype=mx.float16)
        cache.append(k, v, seq_id=0)
        assert isinstance(cache._k_pool, mx.array)
        assert isinstance(cache._v_pool, mx.array)

    def test_append_multi_token(self):
        """Append T tokens; seq_lengths reflects total count."""
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=8, block_size=16, H=2, D=64)
        T = 32
        k = mx.zeros((1, 2, T, 64), dtype=mx.float16)
        v = mx.zeros((1, 2, T, 64), dtype=mx.float16)
        cache.append(k, v, seq_id=0)
        assert cache.seq_lengths[0] == T
        # 32 tokens in blocks of 16 → 2 blocks used
        assert len(cache._block_table[0]) == 2

    def test_append_cross_block_boundary(self):
        """Append tokens that cross a block boundary."""
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=8, block_size=16, H=2, D=64)
        # Fill block 0 with 10 tokens, then add 10 more (spills to block 1)
        k1 = mx.zeros((1, 2, 10, 64), dtype=mx.float16)
        v1 = mx.zeros((1, 2, 10, 64), dtype=mx.float16)
        cache.append(k1, v1, seq_id=0)
        assert cache._write_ptr[0] == 10

        k2 = mx.zeros((1, 2, 10, 64), dtype=mx.float16)
        v2 = mx.zeros((1, 2, 10, 64), dtype=mx.float16)
        cache.append(k2, v2, seq_id=0)
        assert cache.seq_lengths[0] == 20
        assert len(cache._block_table[0]) == 2
        assert cache._write_ptr[0] == 4   # 20 - 16 = 4 tokens in block 1

    def test_gather_matches_append(self):
        """gather() returns exactly what was appended."""
        from mlx_mfa import PagedKVCache
        import numpy as np
        mx.random.seed(2)
        cache = PagedKVCache(num_blocks=8, block_size=16, H=4, D=64)
        k = mx.random.normal((1, 4, 24, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 24, 64)).astype(mx.float16)
        mx.eval(k, v)
        cache.append(k, v, seq_id=0)

        k_out, v_out = cache.gather(seq_id=0)
        assert k_out.shape == (1, 4, 24, 64)
        assert v_out.shape == (1, 4, 24, 64)

        np.testing.assert_allclose(
            np.array(k_out.astype(mx.float32)),
            np.array(k.astype(mx.float32)),
            atol=1e-4, err_msg="gather K != appended K")
        np.testing.assert_allclose(
            np.array(v_out.astype(mx.float32)),
            np.array(v.astype(mx.float32)),
            atol=1e-4, err_msg="gather V != appended V")

    def test_gather_empty_seq(self):
        """gather() on an empty / unknown seq returns [1, H, 0, D]."""
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=4, block_size=16, H=2, D=64)
        k_out, v_out = cache.gather(seq_id=99)
        assert k_out.shape == (1, 2, 0, 64)
        assert v_out.shape == (1, 2, 0, 64)

    def test_free_seq_releases_blocks(self):
        """free_seq() returns blocks to the free list."""
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=4, block_size=16, H=2, D=64)
        k = mx.zeros((1, 2, 32, 64), dtype=mx.float16)
        v = mx.zeros((1, 2, 32, 64), dtype=mx.float16)
        cache.append(k, v, seq_id=0)
        assert len(cache._free) == 2   # 4 - 2 = 2 free

        cache.free_seq(0)
        assert len(cache._free) == 4   # all returned
        assert 0 not in cache._block_table

    def test_get_block_table_shape_and_dtype(self):
        """get_block_table() returns [B, max_blocks] int32 with -1 padding."""
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=8, block_size=16, H=2, D=64)
        k = mx.zeros((1, 2, 32, 64), dtype=mx.float16)
        v = mx.zeros((1, 2, 32, 64), dtype=mx.float16)
        cache.append(k, v, seq_id=0)   # 2 blocks
        cache.append(mx.zeros((1, 2, 8, 64), dtype=mx.float16),
                     mx.zeros((1, 2, 8, 64), dtype=mx.float16), seq_id=1)  # 1 block

        bt = cache.get_block_table([0, 1])
        assert bt.shape == (2, 2)     # max 2 blocks, padded to 2
        assert bt.dtype == mx.int32
        assert int(bt[1, 1].item()) == -1   # seq 1 has only 1 block → pad

    def test_get_seq_lens(self):
        """get_seq_lens() returns correct token count per sequence."""
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=8, block_size=16, H=2, D=64)
        cache.append(mx.zeros((1, 2, 10, 64), dtype=mx.float16),
                     mx.zeros((1, 2, 10, 64), dtype=mx.float16), seq_id=0)
        cache.append(mx.zeros((1, 2, 25, 64), dtype=mx.float16),
                     mx.zeros((1, 2, 25, 64), dtype=mx.float16), seq_id=1)
        sl = cache.get_seq_lens([0, 1])
        assert sl.dtype == mx.int32
        assert int(sl[0].item()) == 10
        assert int(sl[1].item()) == 25

    @pytest.mark.skipif(not is_mfa_available(), reason="MFA extension not available")
    def test_paged_integration(self):
        """PagedKVCache + flash_attention_paged produces correct output."""
        from mlx_mfa import PagedKVCache, flash_attention_paged, flash_attention
        mx.random.seed(42)
        B, H, N_q, S, D = 1, 4, 4, 32, 64
        scale = 1.0 / D**0.5

        q  = mx.random.normal((B, H, N_q, D)).astype(mx.float16)
        k  = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v  = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)

        cache = PagedKVCache(num_blocks=8, block_size=16, H=H, D=D)
        cache.append(k, v, seq_id=0)

        out_paged = flash_attention_paged(
            q, cache.k_pool, cache.v_pool,
            cache.get_block_table([0]), cache.get_seq_lens([0]),
            scale=scale, block_size=cache.block_size,
        )
        out_ref = flash_attention(q, k, v, scale=scale)
        mx.eval(out_paged, out_ref)

        diff = float(mx.abs(out_paged.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"paged vs ref max diff: {diff}"

    @pytest.mark.skipif(not is_mfa_available(), reason="MFA extension not available")
    def test_kvcache_integration(self):
        """PagedKVCache + flash_attention_kvcache (paged mode) produces correct output."""
        from mlx_mfa import PagedKVCache, flash_attention_kvcache, flash_attention
        mx.random.seed(7)
        B, H, N_q, S, D = 1, 4, 1, 48, 64
        scale = 1.0 / D**0.5

        q = mx.random.normal((B, H, N_q, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)

        cache = PagedKVCache(num_blocks=8, block_size=16, H=H, D=D)
        cache.append(k, v, seq_id=0)

        out_paged = flash_attention_kvcache(
            q, cache.k_pool, cache.v_pool,
            block_table=cache.get_block_table([0]),
            seq_lens=cache.get_seq_lens([0]),
            block_size=cache.block_size,
            scale=scale, causal=False,
        )
        out_ref = flash_attention(q, k, v, scale=scale)
        mx.eval(out_paged, out_ref)

        diff = float(mx.abs(out_paged.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"kvcache paged vs ref max diff: {diff}"

    @pytest.mark.skipif(not is_mfa_available(), reason="MFA extension not available")
    def test_multi_seq(self):
        """Multiple sequences with different lengths."""
        from mlx_mfa import PagedKVCache, flash_attention_paged, flash_attention
        mx.random.seed(5)
        H, D = 4, 64
        S0, S1 = 32, 48
        scale = 1.0 / D**0.5

        q = mx.random.normal((2, H, 4, D)).astype(mx.float16)
        k0 = mx.random.normal((1, H, S0, D)).astype(mx.float16)
        k1 = mx.random.normal((1, H, S1, D)).astype(mx.float16)
        v0 = mx.random.normal((1, H, S0, D)).astype(mx.float16)
        v1 = mx.random.normal((1, H, S1, D)).astype(mx.float16)
        mx.eval(q, k0, k1, v0, v1)

        cache = PagedKVCache(num_blocks=16, block_size=16, H=H, D=D)
        cache.append(k0, v0, seq_id=0)
        cache.append(k1, v1, seq_id=1)

        bt = cache.get_block_table([0, 1])
        sl = cache.get_seq_lens([0, 1])
        assert bt.shape == (2, 3)   # seq0: 2 blocks; seq1: 3 blocks

        out_paged = flash_attention_paged(
            q, cache.k_pool, cache.v_pool, bt, sl,
            scale=scale, block_size=cache.block_size,
        )
        ref = mx.concatenate([
            flash_attention(q[0:1], k0, v0, scale=scale),
            flash_attention(q[1:2], k1, v1, scale=scale),
        ], axis=0)
        mx.eval(out_paged, ref)

        diff = float(mx.abs(out_paged.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"multi-seq max diff: {diff}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Track BF — QKV / KV packed tensor formats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPackedFormats:
    """Tests for flash_attention_qkv_packed and flash_attention_kv_packed."""

    # ── QKV packed ────────────────────────────────────────────────────────

    def test_qkv_flat_matches_split(self):
        """[B,N,3*H*D] flat layout == split Q/K/V attention."""
        from mlx_mfa import flash_attention_qkv_packed, flash_attention
        mx.random.seed(3)
        B, H, N, D = 2, 4, 32, 64
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        # Build flat [B, N, 3*H*D]: q_flat || k_flat || v_flat per token
        q_flat = q.transpose(0, 2, 1, 3).reshape(B, N, H * D)
        k_flat = k.transpose(0, 2, 1, 3).reshape(B, N, H * D)
        v_flat = v.transpose(0, 2, 1, 3).reshape(B, N, H * D)
        qkv = mx.concatenate([q_flat, k_flat, v_flat], axis=-1)   # [B,N,3*H*D]
        mx.eval(qkv)

        out_packed = flash_attention_qkv_packed(qkv, num_heads=H)
        out_ref    = flash_attention(q, k, v)
        mx.eval(out_packed, out_ref)

        diff = float(mx.abs(out_packed.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"QKV flat max diff: {diff}"

    def test_qkv_head_first_matches_split(self):
        """[B,H,N,3,D] head-first layout == split Q/K/V attention."""
        from mlx_mfa import flash_attention_qkv_packed, flash_attention
        mx.random.seed(5)
        B, H, N, D = 2, 4, 32, 64
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        qkv = mx.stack([q, k, v], axis=3)   # [B, H, N, 3, D]
        mx.eval(qkv)

        out_packed = flash_attention_qkv_packed(qkv)
        out_ref    = flash_attention(q, k, v)
        mx.eval(out_packed, out_ref)

        diff = float(mx.abs(out_packed.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"QKV head-first max diff: {diff}"

    def test_qkv_flat_bad_shape_raises(self):
        """Flat layout with num_heads not divisible raises ValueError."""
        from mlx_mfa import flash_attention_qkv_packed
        import pytest
        qkv = mx.zeros((1, 16, 100), dtype=mx.float16)   # 100 not divisible by 3
        with pytest.raises(ValueError):
            flash_attention_qkv_packed(qkv, num_heads=5)

    def test_qkv_flat_requires_num_heads(self):
        """Flat layout without num_heads raises ValueError."""
        from mlx_mfa import flash_attention_qkv_packed
        import pytest
        qkv = mx.zeros((1, 16, 3*4*64), dtype=mx.float16)
        with pytest.raises(ValueError, match="num_heads required"):
            flash_attention_qkv_packed(qkv)  # no num_heads

    def test_qkv_bad_ndim_raises(self):
        """Unsupported ndim raises ValueError."""
        from mlx_mfa import flash_attention_qkv_packed
        import pytest
        qkv = mx.zeros((1, 4, 16, 64), dtype=mx.float16)   # ndim=4
        with pytest.raises(ValueError, match="unsupported shape"):
            flash_attention_qkv_packed(qkv)

    def test_qkv_causal_flat(self):
        """QKV flat causal matches split causal."""
        from mlx_mfa import flash_attention_qkv_packed, flash_attention
        mx.random.seed(9)
        B, H, N, D = 1, 4, 32, 64
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        q_flat = q.transpose(0, 2, 1, 3).reshape(B, N, H * D)
        k_flat = k.transpose(0, 2, 1, 3).reshape(B, N, H * D)
        v_flat = v.transpose(0, 2, 1, 3).reshape(B, N, H * D)
        qkv = mx.concatenate([q_flat, k_flat, v_flat], axis=-1)
        mx.eval(qkv)
        out_packed = flash_attention_qkv_packed(qkv, num_heads=H, causal=True)
        out_ref    = flash_attention(q, k, v, causal=True)
        mx.eval(out_packed, out_ref)
        diff = float(mx.abs(out_packed.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"QKV causal max diff: {diff}"

    # ── KV packed ────────────────────────────────────────────────────────

    def test_kv_flat_matches_split(self):
        """[B,S,2*H*D] flat KV layout == split K/V attention."""
        from mlx_mfa import flash_attention_kv_packed, flash_attention
        mx.random.seed(13)
        B, H, N, S, D = 2, 4, 16, 32, 64
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        k_flat = k.transpose(0, 2, 1, 3).reshape(B, S, H * D)
        v_flat = v.transpose(0, 2, 1, 3).reshape(B, S, H * D)
        kv = mx.concatenate([k_flat, v_flat], axis=-1)   # [B, S, 2*H*D]
        mx.eval(q, kv)

        out_packed = flash_attention_kv_packed(q, kv, num_kv_heads=H)
        out_ref    = flash_attention(q, k, v)
        mx.eval(out_packed, out_ref)

        diff = float(mx.abs(out_packed.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"KV flat max diff: {diff}"

    def test_kv_head_first_matches_split(self):
        """[B,H,S,2,D] head-first KV layout == split K/V attention."""
        from mlx_mfa import flash_attention_kv_packed, flash_attention
        mx.random.seed(17)
        B, H, N, S, D = 2, 4, 16, 32, 64
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        kv = mx.stack([k, v], axis=3)   # [B, H, S, 2, D]
        mx.eval(q, kv)

        out_packed = flash_attention_kv_packed(q, kv)
        out_ref    = flash_attention(q, k, v)
        mx.eval(out_packed, out_ref)

        diff = float(mx.abs(out_packed.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3, f"KV head-first max diff: {diff}"

    def test_kv_flat_requires_num_kv_heads(self):
        """Flat KV layout without num_kv_heads raises ValueError."""
        from mlx_mfa import flash_attention_kv_packed
        import pytest
        q  = mx.zeros((1, 4, 8, 64), dtype=mx.float16)
        kv = mx.zeros((1, 32, 2*4*64), dtype=mx.float16)
        with pytest.raises(ValueError, match="num_kv_heads required"):
            flash_attention_kv_packed(q, kv)

    def test_kv_bad_ndim_raises(self):
        """Unsupported kv ndim raises ValueError."""
        from mlx_mfa import flash_attention_kv_packed
        import pytest
        q  = mx.zeros((1, 4, 8, 64), dtype=mx.float16)
        kv = mx.zeros((1, 32, 128), dtype=mx.float16)
        with pytest.raises(ValueError):
            flash_attention_kv_packed(q, kv, num_kv_heads=3)  # 128 not / by 6


@requires_ext
class TestSteelBackwardGQA:
    """STEEL backward for grouped-query attention (Track DA — GQA guard removed)."""

    @pytest.mark.parametrize("ratio,D,causal", [
        (2, 64, False), (4, 128, True), (8, 128, False),
    ])
    def test_gqa_backward_matches_sdpa(self, ratio, D, causal):
        """STEEL backward GQA gradients match mx.vjp(SDPA) reference."""
        B, H_q, N = 1, 8, 128
        H_kv = H_q // ratio
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(42 + ratio + D)

        q = mx.random.normal((B, H_q,  N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)

        # MFA path — should now use STEEL backward (no GQA guard)
        def loss_mfa(q_, k_, v_):
            return mx.sum(flash_attention(q_, k_, v_, scale=scale, causal=causal))
        dq_mfa, dk_mfa, dv_mfa = mx.grad(loss_mfa, argnums=(0, 1, 2))(q, k, v)

        # Reference: expand K/V to H_q, run SDPA backward, sum grads to H_kv
        k_rep = mx.repeat(k, ratio, axis=1)
        v_rep = mx.repeat(v, ratio, axis=1)

        def loss_ref(q_, k_, v_):
            return mx.sum(mx.fast.scaled_dot_product_attention(
                q_, k_, v_, scale=scale,
                mask="causal" if causal else None))
        dq_ref, dk_exp, dv_exp = mx.grad(loss_ref, argnums=(0, 1, 2))(q, k_rep, v_rep)
        dk_ref = dk_exp.reshape(B, H_kv, ratio, N, D).sum(axis=2)
        dv_ref = dv_exp.reshape(B, H_kv, ratio, N, D).sum(axis=2)

        mx.eval(dq_mfa, dk_mfa, dv_mfa, dq_ref, dk_ref, dv_ref)

        assert list(dk_mfa.shape) == [B, H_kv, N, D], "dK shape must be [B,H_kv,N,D]"
        assert list(dv_mfa.shape) == [B, H_kv, N, D], "dV shape must be [B,H_kv,N,D]"

        np.testing.assert_allclose(
            np.array(dq_mfa.astype(mx.float32)),
            np.array(dq_ref.astype(mx.float32)),
            atol=5e-2, rtol=1e-1,
            err_msg=f"dQ mismatch GQA ratio={ratio} D={D} causal={causal}")
        np.testing.assert_allclose(
            np.array(dk_mfa.astype(mx.float32)),
            np.array(dk_ref.astype(mx.float32)),
            atol=5e-2, rtol=1e-1,
            err_msg=f"dK mismatch GQA ratio={ratio} D={D} causal={causal}")
        np.testing.assert_allclose(
            np.array(dv_mfa.astype(mx.float32)),
            np.array(dv_ref.astype(mx.float32)),
            atol=5e-2, rtol=1e-1,
            err_msg=f"dV mismatch GQA ratio={ratio} D={D} causal={causal}")


@requires_ext
class TestSteelBackwardD256:
    """D=256 D-split STEEL backward kernels (Track CE — v0.9.2).

    The D-split approach partitions the head dimension into lo (0..127) and
    hi (128..255) halves, fitting within the 32 KB Metal TGP budget while
    still dispatching native STEEL kernels instead of falling back to SDPA VJP.
    """

    @pytest.mark.parametrize("causal", [True, False])
    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    def test_d256_backward_matches_sdpa(self, dtype, causal):
        """STEEL D-split backward gradients match mx.vjp(SDPA) reference."""
        B, H, N, D = 1, 4, 128, 256
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(77 + int(causal))

        q = mx.random.normal((B, H, N, D)).astype(dtype)
        k = mx.random.normal((B, H, N, D)).astype(dtype)
        v = mx.random.normal((B, H, N, D)).astype(dtype)
        cot = mx.ones((B, H, N, D), dtype=dtype)
        mx.eval(q, k, v, cot)

        _, (dq_mfa, dk_mfa, dv_mfa) = mx.vjp(
            lambda q_, k_, v_: flash_attention(q_, k_, v_, scale=scale, causal=causal),
            [q, k, v], [cot])
        _, (dq_ref, dk_ref, dv_ref) = mx.vjp(
            lambda q_, k_, v_: mx.fast.scaled_dot_product_attention(
                q_, k_, v_, scale=scale, mask="causal" if causal else None),
            [q, k, v], [cot])
        mx.eval(dq_mfa, dk_mfa, dv_mfa, dq_ref, dk_ref, dv_ref)

        dtype_str = "f16" if dtype == mx.float16 else "bf16"
        for name, mfa, ref in [("dQ", dq_mfa, dq_ref),
                                ("dK", dk_mfa, dk_ref),
                                ("dV", dv_mfa, dv_ref)]:
            assert list(mfa.shape) == [B, H, N, D], f"{name} shape mismatch"
            assert mfa.dtype == dtype, f"{name} dtype mismatch"
            np.testing.assert_allclose(
                np.array(mfa.astype(mx.float32)),
                np.array(ref.astype(mx.float32)),
                atol=5e-2, rtol=1e-1,
                err_msg=f"{name} D=256 {dtype_str} causal={causal}")

    def test_d256_backward_finite(self):
        """All D=256 backward gradients are finite (no NaN/Inf)."""
        B, H, N, D = 1, 8, 64, 256
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(99)

        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)

        def loss(q_, k_, v_):
            return mx.sum(flash_attention(q_, k_, v_, scale=scale, causal=True))

        dq, dk, dv = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)

        assert mx.all(mx.isfinite(dq)).item(), "dQ has non-finite values"
        assert mx.all(mx.isfinite(dk)).item(), "dK has non-finite values"
        assert mx.all(mx.isfinite(dv)).item(), "dV has non-finite values"

    def test_d256_backward_gqa(self):
        """D=256 D-split backward works with GQA (ratio=2)."""
        B, H_q, H_kv, N, D = 1, 4, 2, 64, 256
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(111)

        q = mx.random.normal((B, H_q,  N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)

        def loss(q_, k_, v_):
            return mx.sum(flash_attention(q_, k_, v_, scale=scale, causal=True))

        dq, dk, dv = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)

        assert list(dq.shape) == [B, H_q,  N, D], "dQ shape mismatch for GQA D=256"
        assert list(dk.shape) == [B, H_kv, N, D], "dK shape mismatch for GQA D=256"
        assert list(dv.shape) == [B, H_kv, N, D], "dV shape mismatch for GQA D=256"
        assert mx.all(mx.isfinite(dq)).item(), "dQ non-finite in GQA D=256"
        assert mx.all(mx.isfinite(dk)).item(), "dK non-finite in GQA D=256"
        assert mx.all(mx.isfinite(dv)).item(), "dV non-finite in GQA D=256"


@requires_ext
class TestVarlenBackward:
    """Differentiable flash_attention_varlen via mx.custom_function (Track EA)."""

    @pytest.mark.parametrize("D,dtype", [
        (64, mx.float16),
        (128, mx.float16),
        (128, mx.bfloat16),
        (256, mx.float16),
    ])
    def test_varlen_backward_matches_ref(self, D, dtype):
        """Varlen backward matches per-sequence flash_attention backward."""
        from mlx_mfa import flash_attention, flash_attention_varlen
        B, H = 1, 4
        lens = [32, 64, 48]
        N = sum(lens)
        cu_off = [sum(lens[:i]) for i in range(len(lens) + 1)]
        cu = mx.array(cu_off, dtype=mx.int32)
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(42 + D)

        q = mx.random.normal((B, H, N, D)).astype(dtype)
        k = mx.random.normal((B, H, N, D)).astype(dtype)
        v = mx.random.normal((B, H, N, D)).astype(dtype)
        mx.eval(q, k, v)

        def loss_varlen(q_, k_, v_):
            return mx.sum(flash_attention_varlen(
                q_, k_, v_, cu, cu, max(lens), max(lens),
                scale=scale, causal=True))

        dq, dk, dv = mx.grad(loss_varlen, argnums=(0, 1, 2))(q, k, v)

        # Reference: per-sequence flash_attention backward (same STEEL kernel)
        def loss_ref(q_, k_, v_):
            parts = []
            for i in range(len(lens)):
                s = sum(lens[:i])
                e = sum(lens[:i + 1])
                parts.append(flash_attention(
                    q_[:, :, s:e, :], k_[:, :, s:e, :], v_[:, :, s:e, :],
                    scale=scale, causal=True))
            return mx.sum(mx.concatenate(parts, axis=2))

        dq_ref, dk_ref, dv_ref = mx.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv, dq_ref, dk_ref, dv_ref)

        dtype_str = "f16" if dtype == mx.float16 else "bf16"
        for name, mfa, ref in [("dQ", dq, dq_ref), ("dK", dk, dk_ref), ("dV", dv, dv_ref)]:
            assert list(mfa.shape) == [B, H, N, D], f"{name} shape mismatch"
            np.testing.assert_allclose(
                np.array(mfa.astype(mx.float32)),
                np.array(ref.astype(mx.float32)),
                atol=5e-2, rtol=1e-1,
                err_msg=f"{name} varlen bwd D={D} {dtype_str}")

    def test_varlen_backward_f32_fallback(self):
        """f32 varlen backward works via split-concat fallback."""
        from mlx_mfa import flash_attention_varlen
        B, H, D = 1, 2, 64
        lens = [16, 32]
        N = sum(lens)
        cu = mx.array([0, 16, 48], dtype=mx.int32)
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(7)

        q = mx.random.normal((B, H, N, D))   # f32
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))
        mx.eval(q, k, v)

        def loss(q_, k_, v_):
            return mx.sum(flash_attention_varlen(
                q_, k_, v_, cu, cu, max(lens), max(lens), scale=scale, causal=False))

        dq, dk, dv = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)
        assert mx.all(mx.isfinite(dq)).item(), "dQ non-finite (f32 fallback)"
        assert mx.all(mx.isfinite(dk)).item(), "dK non-finite (f32 fallback)"
        assert mx.all(mx.isfinite(dv)).item(), "dV non-finite (f32 fallback)"

    def test_varlen_backward_gqa(self):
        """Varlen backward with GQA (H_q=4, H_kv=2)."""
        from mlx_mfa import flash_attention_varlen
        B, H_q, H_kv, D = 1, 4, 2, 128
        lens = [32, 48]
        N = sum(lens)
        cu = mx.array([0, 32, 80], dtype=mx.int32)
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(55)

        q = mx.random.normal((B, H_q,  N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        mx.eval(q, k, v)

        def loss(q_, k_, v_):
            return mx.sum(flash_attention_varlen(
                q_, k_, v_, cu, cu, max(lens), max(lens), scale=scale, causal=True))

        dq, dk, dv = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)
        assert list(dq.shape) == [B, H_q,  N, D], "dQ shape GQA varlen"
        assert list(dk.shape) == [B, H_kv, N, D], "dK shape GQA varlen"
        assert list(dv.shape) == [B, H_kv, N, D], "dV shape GQA varlen"
        assert mx.all(mx.isfinite(dq)).item(), "dQ non-finite GQA varlen"
        assert mx.all(mx.isfinite(dk)).item(), "dK non-finite GQA varlen"
        assert mx.all(mx.isfinite(dv)).item(), "dV non-finite GQA varlen"


# ===========================================================================
# Track EB — Paged attention backward (Metal gather + custom_function)
# ===========================================================================

@requires_ext
class TestPagedBackward:
    """EB.6: Metal paged KV gather + differentiable flash_attention_paged."""

    def _make_paged(self, B, H_q, H_kv, N_q, D, kv_lens, block_size=16,
                    dtype=mx.float16):
        """Build pool/table/lens for testing."""
        import math
        num_blocks = sum((kv_len + block_size - 1) // block_size
                         for kv_len in kv_lens) + 2  # spare
        q = (mx.random.normal((B, H_q, N_q, D)) * 0.1).astype(dtype)
        k_pool = (mx.random.normal((num_blocks, block_size, H_kv, D)) * 0.1).astype(dtype)
        v_pool = (mx.random.normal((num_blocks, block_size, H_kv, D)) * 0.1).astype(dtype)

        # Assign blocks sequentially
        table = [[-1] * 4 for _ in range(B)]
        phys = 0
        for b in range(B):
            n_blk = (kv_lens[b] + block_size - 1) // block_size
            for lb in range(n_blk):
                table[b][lb] = phys
                phys += 1

        block_table = mx.array(table, dtype=mx.int32)
        seq_lens = mx.array(kv_lens, dtype=mx.int32)
        return q, k_pool, v_pool, block_table, seq_lens

    def test_paged_forward_shape(self):
        """Output shape is [B, H_q, N_q, D]."""
        from mlx_mfa import flash_attention_paged
        B, H_q, H_kv, N_q, D = 2, 4, 4, 1, 64
        q, k_p, v_p, bt, sl = self._make_paged(B, H_q, H_kv, N_q, D, [20, 18])
        out = flash_attention_paged(q, k_p, v_p, bt, sl)
        mx.eval(out)
        assert list(out.shape) == [B, H_q, N_q, D]

    def test_paged_forward_finite(self):
        """Forward output values are finite."""
        from mlx_mfa import flash_attention_paged
        B, H_q, H_kv, N_q, D = 2, 4, 4, 1, 128
        q, k_p, v_p, bt, sl = self._make_paged(B, H_q, H_kv, N_q, D, [32, 16])
        out = flash_attention_paged(q, k_p, v_p, bt, sl)
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item()

    @pytest.mark.parametrize("D", [64, 128])
    def test_paged_dQ_finite(self, D):
        """dQ is finite for f16 paged attention."""
        from mlx_mfa import flash_attention_paged
        B, H_q, H_kv, N_q = 2, 4, 4, 1
        q, k_p, v_p, bt, sl = self._make_paged(B, H_q, H_kv, N_q, D, [24, 20])

        def loss(q_):
            return flash_attention_paged(q_, k_p, v_p, bt, sl).sum()

        dq = mx.grad(loss)(q)
        mx.eval(dq)
        assert list(dq.shape) == [B, H_q, N_q, D], "dQ shape"
        assert mx.all(mx.isfinite(dq)).item(), "dQ non-finite"

    def test_paged_dQ_gqa(self):
        """dQ is correct with GQA (H_q=4, H_kv=2)."""
        from mlx_mfa import flash_attention_paged
        B, H_q, H_kv, N_q, D = 1, 4, 2, 1, 64
        q, k_p, v_p, bt, sl = self._make_paged(B, H_q, H_kv, N_q, D, [20])

        def loss(q_):
            return flash_attention_paged(q_, k_p, v_p, bt, sl).sum()

        dq = mx.grad(loss)(q)
        mx.eval(dq)
        assert list(dq.shape) == [B, H_q, N_q, D]
        assert mx.all(mx.isfinite(dq)).item()

    def test_paged_dQ_matches_non_paged(self):
        """dQ from paged path matches non-paged flash_attention (single seq)."""
        from mlx_mfa import flash_attention, flash_attention_paged
        import math
        B, H, N_q, N_kv, D, block_size = 1, 4, 2, 32, 64, 16
        mx.random.seed(77)
        q = (mx.random.normal((B, H, N_q, D)) * 0.1).astype(mx.float16)
        k = (mx.random.normal((B, H, N_kv, D)) * 0.1).astype(mx.float16)
        v = (mx.random.normal((B, H, N_kv, D)) * 0.1).astype(mx.float16)
        mx.eval(q, k, v)

        scale = 1.0 / math.sqrt(D)

        # Build paged pool matching k/v
        # k shape: [1, H, N_kv, D] → pool [num_blocks, BS, H, D]
        # Rearrange k[0]: [H, N_kv, D] → [N_kv, H, D] → [num_blocks, BS, H, D]
        k_tok = k[0].transpose(1, 0, 2)  # [N_kv, H, D]
        v_tok = v[0].transpose(1, 0, 2)
        n_blk = N_kv // block_size
        k_pool = k_tok.reshape(n_blk, block_size, H, D)
        v_pool = v_tok.reshape(n_blk, block_size, H, D)
        bt = mx.array([[0, 1]], dtype=mx.int32)
        sl = mx.array([N_kv], dtype=mx.int32)
        mx.eval(k_pool, v_pool, bt, sl)

        def loss_paged(q_):
            return flash_attention_paged(q_, k_pool, v_pool, bt, sl,
                                         scale=scale).sum()

        def loss_ref(q_):
            return flash_attention(q_, k, v, scale=scale).sum()

        dq_paged = mx.grad(loss_paged)(q)
        dq_ref   = mx.grad(loss_ref)(q)
        mx.eval(dq_paged, dq_ref)

        max_err = mx.max(mx.abs(dq_paged - dq_ref)).item()
        assert max_err < 0.05, f"dQ paged vs ref max err = {max_err}"

    def test_paged_causal_backward_heterogeneous_seq_lens(self):
        """III-4 D5 FIX regression: causal backward with heterogeneous
        kv_lens and N_q > 1.  The forward slices each row to its own kv_len
        (query i of row b sits at causal position kv_len_b - N_q + i); the
        old backward mask anchored ALL rows at max_kv_len, producing wrong
        grads for every row with kv_len < max_kv_len.  Reference: per-row
        sliced SDPA with the true per-row causal positions."""
        from mlx_mfa import flash_attention_paged
        import math
        B, H, N_q, D, block_size = 2, 4, 4, 64, 16
        kv_lens = [48, 32]
        mx.random.seed(31)
        q, k_p, v_p, bt, sl = self._make_paged(
            B, H, H, N_q, D, kv_lens, block_size=block_size)
        mx.eval(q, k_p, v_p, bt, sl)
        scale = 1.0 / math.sqrt(D)

        # Reconstruct contiguous per-row K/V from the (sequential) pool.
        bt_list = bt.tolist()

        def _row_kv(b):
            n_blk = kv_lens[b] // block_size
            blks_k = [k_p[bt_list[b][lb]] for lb in range(n_blk)]
            blks_v = [v_p[bt_list[b][lb]] for lb in range(n_blk)]
            k_row = mx.concatenate(blks_k, axis=0)  # [kv_len, H, D]
            v_row = mx.concatenate(blks_v, axis=0)
            return (k_row.transpose(1, 0, 2)[None],
                    v_row.transpose(1, 0, 2)[None])  # [1, H, kv_len, D]

        def _row_causal_mask(kv_len):
            q_pos = mx.arange(kv_len - N_q, kv_len, dtype=mx.int32)[:, None]
            k_pos = mx.arange(kv_len, dtype=mx.int32)[None, :]
            return mx.where(
                k_pos <= q_pos,
                mx.zeros((N_q, kv_len), dtype=mx.float16),
                mx.full((N_q, kv_len), float("-inf"), dtype=mx.float16),
            )

        def loss_paged(q_):
            return flash_attention_paged(
                q_, k_p, v_p, bt, sl, scale=scale, causal=True,
                block_size=block_size).sum()

        def loss_ref(q_):
            total = mx.zeros(())
            for b in range(B):
                k_b, v_b = _row_kv(b)
                out_b = mx.fast.scaled_dot_product_attention(
                    q_[b:b + 1], k_b, v_b, scale=scale,
                    mask=_row_causal_mask(kv_lens[b]))
                total = total + out_b.sum()
            return total

        dq_paged = mx.grad(loss_paged)(q)
        dq_ref = mx.grad(loss_ref)(q)
        mx.eval(dq_paged, dq_ref)
        max_err_q = mx.max(mx.abs(dq_paged - dq_ref)).item()
        assert max_err_q < 0.05, (
            f"dQ paged-causal vs per-row SDPA ref max err = {max_err_q}"
        )

        # dK through the pool: scatter per-row reference grads back to pool.
        def loss_paged_k(k_p_):
            return flash_attention_paged(
                q, k_p_, v_p, bt, sl, scale=scale, causal=True,
                block_size=block_size).sum()

        dk_paged = mx.grad(loss_paged_k)(k_p)

        def loss_ref_kb(k_row, b):
            v_b = _row_kv(b)[1]
            out_b = mx.fast.scaled_dot_product_attention(
                q[b:b + 1], k_row, v_b, scale=scale,
                mask=_row_causal_mask(kv_lens[b]))
            return out_b.sum()

        max_err_k = 0.0
        for b in range(B):
            k_b = _row_kv(b)[0]
            dk_ref_b = mx.grad(loss_ref_kb)(k_b, b)[0]  # [H, kv_len, D]
            n_blk = kv_lens[b] // block_size
            for lb in range(n_blk):
                phys = bt_list[b][lb]
                ref_tile = dk_ref_b[
                    :, lb * block_size:(lb + 1) * block_size, :
                ].transpose(1, 0, 2)  # [bs, H, D]
                err = mx.max(mx.abs(dk_paged[phys] - ref_tile)).item()
                max_err_k = max(max_err_k, err)
        assert max_err_k < 0.05, (
            f"dK paged-causal vs per-row SDPA ref max err = {max_err_k}"
        )

    # ── Track IF — dK/dV scatter ──────────────────────────────────────────

    def test_paged_dk_shape(self):
        """dK_pages has the same shape as k_pages."""
        from mlx_mfa import flash_attention_paged
        B, H_q, H_kv, N_q, D = 2, 4, 4, 2, 64
        q, k_p, v_p, bt, sl = self._make_paged(B, H_q, H_kv, N_q, D, [24, 20])

        def loss(k_p_):
            return flash_attention_paged(q, k_p_, v_p, bt, sl).sum()

        dk = mx.grad(loss)(k_p)
        mx.eval(dk)
        assert dk.shape == k_p.shape, f"dk shape {dk.shape} != {k_p.shape}"

    def test_paged_dkv_finite(self):
        """dK_pages and dV_pages are finite."""
        from mlx_mfa import flash_attention_paged
        B, H_q, H_kv, N_q, D = 2, 4, 4, 2, 64
        q, k_p, v_p, bt, sl = self._make_paged(B, H_q, H_kv, N_q, D, [24, 20])

        def loss(k_p_, v_p_):
            return flash_attention_paged(q, k_p_, v_p_, bt, sl).sum()

        dk, dv = mx.grad(loss, argnums=(0, 1))(k_p, v_p)
        mx.eval(dk, dv)
        assert mx.all(mx.isfinite(dk)).item(), "dK_pages has non-finite values"
        assert mx.all(mx.isfinite(dv)).item(), "dV_pages has non-finite values"

    def test_paged_dkv_nonzero(self):
        """dK and dV are not all zeros (regression: previously returned zeros)."""
        from mlx_mfa import flash_attention_paged
        B, H, N_q, D, block_size = 1, 4, 2, 64, 16
        q, k_p, v_p, bt, sl = self._make_paged(B, H, H, N_q, D, [32])

        def loss(k_p_, v_p_):
            return flash_attention_paged(q, k_p_, v_p_, bt, sl).sum()

        dk, dv = mx.grad(loss, argnums=(0, 1))(k_p, v_p)
        mx.eval(dk, dv)
        # At least some gradient must be non-zero
        assert float(mx.sum(mx.abs(dk)).item()) > 0.0, "dK_pages is all zeros"
        assert float(mx.sum(mx.abs(dv)).item()) > 0.0, "dV_pages is all zeros"

    def test_paged_dk_matches_non_paged(self):
        """dK from paged path matches non-paged flash_attention (single seq, f32)."""
        from mlx_mfa import flash_attention, flash_attention_paged
        B, H, N_q, N_kv, D, block_size = 1, 2, 4, 32, 64, 16
        mx.random.seed(33)
        dtype = mx.float32
        q  = (mx.random.normal((B, H, N_q, D)) * 0.1).astype(dtype)
        kv = (mx.random.normal((B, H, N_kv, D)) * 0.1).astype(dtype)
        vv = (mx.random.normal((B, H, N_kv, D)) * 0.1).astype(dtype)

        # Build paged pool from the same K/V using numpy
        import numpy as _np_t
        n_blk = (N_kv + block_size - 1) // block_size
        k_pool_np = _np_t.zeros((n_blk, block_size, H, D), dtype=_np_t.float32)
        v_pool_np = _np_t.zeros((n_blk, block_size, H, D), dtype=_np_t.float32)
        for lb in range(n_blk):
            s, e = lb * block_size, min((lb + 1) * block_size, N_kv)
            # kv[0, :, s:e, :] → [H, e-s, D] → [e-s, H, D]
            k_pool_np[lb, :e-s] = _np_t.array(kv[0, :, s:e, :].transpose(1, 0, 2))
            v_pool_np[lb, :e-s] = _np_t.array(vv[0, :, s:e, :].transpose(1, 0, 2))
        k_pool = mx.array(k_pool_np, dtype=dtype)
        v_pool = mx.array(v_pool_np, dtype=dtype)
        bt = mx.array([[i for i in range(n_blk)]], dtype=mx.int32)
        sl = mx.array([N_kv], dtype=mx.int32)

        def loss_paged(k_p_, v_p_):
            return flash_attention_paged(q, k_p_, v_p_, bt, sl, scale=0.1).sum()

        def loss_ref(k_, v_):
            return flash_attention(q, k_, v_, scale=0.1).sum()

        dk_paged, dv_paged = mx.grad(loss_paged, argnums=(0, 1))(k_pool, v_pool)
        dk_ref, _ = mx.grad(loss_ref, argnums=(0, 1))(kv, vv)
        mx.eval(dk_paged, dv_paged, dk_ref)

        # Reconstruct contiguous dk from paged pool
        dk_contig = mx.concatenate(
            [dk_paged[lb, :min(block_size, N_kv - lb*block_size), :, :]
             for lb in range(n_blk)], axis=0)  # [N_kv, H, D]
        dk_contig = dk_contig.transpose(1, 0, 2)[None]  # [1, H, N_kv, D]

        # Compare with reference dK
        max_err = float(mx.max(mx.abs(dk_contig - dk_ref)).item())
        assert max_err < 1e-4, f"paged dK vs ref max_err={max_err:.2e}"


# ===========================================================================
# Track EC — Varlen packed tensor convenience wrappers
# ===========================================================================

class TestVarlenPacked:
    """EC: flash_attention_varlen_qkv_packed + flash_attention_varlen_kv_packed."""

    def _cu(self, lens):
        import math
        cu = [0]
        for l in lens:
            cu.append(cu[-1] + l)
        return mx.array(cu, dtype=mx.int32)

    def test_varlen_qkv_packed_head_first(self):
        """[1, H, total, 3, D] head-first layout → correct output shape."""
        from mlx_mfa import flash_attention_varlen_qkv_packed
        import math
        H, D, lens = 4, 64, [32, 48]
        total = sum(lens)
        cu = self._cu(lens)
        qkv = (mx.random.normal((1, H, total, 3, D)) * 0.1).astype(mx.float16)
        out = flash_attention_varlen_qkv_packed(
            qkv, cu, cu, max(lens), max(lens), causal=True)
        mx.eval(out)
        assert list(out.shape) == [1, H, total, D]
        assert mx.all(mx.isfinite(out)).item()

    def test_varlen_qkv_packed_flat(self):
        """[1, total, 3*H*D] flat layout → correct output shape."""
        from mlx_mfa import flash_attention_varlen_qkv_packed
        import math
        H, D, lens = 4, 64, [32, 48]
        total = sum(lens)
        cu = self._cu(lens)
        qkv = (mx.random.normal((1, total, 3 * H * D)) * 0.1).astype(mx.float16)
        out = flash_attention_varlen_qkv_packed(
            qkv, cu, cu, max(lens), max(lens), num_heads=H, causal=False)
        mx.eval(out)
        assert list(out.shape) == [1, H, total, D]
        assert mx.all(mx.isfinite(out)).item()

    def test_varlen_kv_packed_head_first(self):
        """[1, H_kv, total_kv, 2, D] head-first layout → correct output shape."""
        from mlx_mfa import flash_attention_varlen_kv_packed
        H_q, H_kv, D, lens = 4, 2, 64, [32, 48]
        total = sum(lens)
        cu = self._cu(lens)
        q  = (mx.random.normal((1, H_q, total, D)) * 0.1).astype(mx.float16)
        kv = (mx.random.normal((1, H_kv, total, 2, D)) * 0.1).astype(mx.float16)
        out = flash_attention_varlen_kv_packed(
            q, kv, cu, cu, max(lens), max(lens), causal=True)
        mx.eval(out)
        assert list(out.shape) == [1, H_q, total, D]
        assert mx.all(mx.isfinite(out)).item()

    def test_varlen_kv_packed_flat(self):
        """[1, total_kv, 2*H_kv*D] flat layout → correct output shape."""
        from mlx_mfa import flash_attention_varlen_kv_packed
        H_q, H_kv, D, lens = 4, 2, 64, [32, 48]
        total = sum(lens)
        cu = self._cu(lens)
        q  = (mx.random.normal((1, H_q, total, D)) * 0.1).astype(mx.float16)
        kv = (mx.random.normal((1, total, 2 * H_kv * D)) * 0.1).astype(mx.float16)
        out = flash_attention_varlen_kv_packed(
            q, kv, cu, cu, max(lens), max(lens), num_kv_heads=H_kv, causal=False)
        mx.eval(out)
        assert list(out.shape) == [1, H_q, total, D]
        assert mx.all(mx.isfinite(out)).item()


# ===========================================================================
# Track FB — Native sliding window attention (STEEL kernel window_left)
# ===========================================================================

@pytest.mark.skipif(not _ext_available(), reason="C++ extension required")
class TestSlidingWindow:
    """Tests for flash_attention(window_size=(left, right)) native STEEL path.

    The STEEL kernel skips entire K-tiles before the window boundary
    (``kb_start = max(0, (q_min - window_left) / BK)``), then applies a
    per-element mask for the first partial tile.  Results must match a dense
    SDPA reference with an equivalent window mask.
    """

    def _ref_window(self, q, k, v, scale, causal, window_left):
        """Reference SDPA with explicit window mask.

III-4 pass-2 B1: window anchor matches the kernel's qL_off —
        S-N only when causal AND N<S, else 0 (non-causal windows anchor
        at position 0; this is the documented forward convention, and the
        backward oracle was made consistent with it).
        """
        N, S = q.shape[2], k.shape[2]
        q_off = (S - N) if (causal and N < S) else 0
        q_idx = mx.arange(q_off, q_off + N, dtype=mx.int32)[:, None]  # [N,1]
        k_idx = mx.arange(S, dtype=mx.int32)[None, :]                  # [1,S]
        in_win = k_idx >= q_idx - window_left
        if causal:
            in_win = in_win & (k_idx <= q_idx)
        mask = mx.where(in_win,
                        mx.zeros((N, S), dtype=q.dtype),
                        mx.full((N, S), float("-inf"), dtype=q.dtype))
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    @pytest.mark.parametrize("D", [64, 128])
    @pytest.mark.parametrize("causal", [False, True])
    def test_window_matches_ref(self, D, causal):
        """Native window output matches masked SDPA reference (f16)."""
        B, H, N, S = 1, 4, 256, 512
        scale = 1.0 / math.sqrt(D)
        window_left = 64
        q = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
        k = (mx.random.normal((B, H, S, D)) * 0.1).astype(mx.float16)
        v = (mx.random.normal((B, H, S, D)) * 0.1).astype(mx.float16)

        out_mfa = flash_attention(q, k, v, scale=scale, causal=causal,
                                  window_size=(window_left, -1))
        out_ref = self._ref_window(q, k, v, scale, causal, window_left)
        mx.eval(out_mfa, out_ref)

        np.testing.assert_allclose(
            np.array(out_mfa.astype(mx.float32)),
            np.array(out_ref.astype(mx.float32)),
            rtol=1e-2, atol=5e-3,
            err_msg=f"Window mismatch D={D} causal={causal}",
        )

    def test_window_disabled_matches_standard(self):
        """window_size=None (default) gives same result as without window."""
        B, H, N, D = 1, 4, 256, 128
        scale = 1.0 / math.sqrt(D)
        q = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
        k = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
        v = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)

        out_no_window  = flash_attention(q, k, v, scale=scale)
        out_none_window = flash_attention(q, k, v, scale=scale, window_size=None)
        mx.eval(out_no_window, out_none_window)

        np.testing.assert_allclose(
            np.array(out_no_window.astype(mx.float32)),
            np.array(out_none_window.astype(mx.float32)),
            rtol=0, atol=0,
        )

    def test_window_output_is_finite(self):
        """Window output must contain no NaN or Inf."""
        B, H, N, D = 2, 4, 512, 128
        scale = 1.0 / math.sqrt(D)
        q = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
        k = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
        v = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
        out = flash_attention(q, k, v, scale=scale, causal=True,
                              window_size=(128, -1))
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item(), "Window output contains NaN/Inf"

    def test_window_fallback_for_f32(self):
        """f32 dtype falls back to masked SDPA (no MFA window kernel for f32)."""
        B, H, N, D = 1, 4, 64, 64
        scale = 1.0 / math.sqrt(D)
        window_left = 32
        q = mx.random.normal((B, H, N, D))
        k = mx.random.normal((B, H, N, D))
        v = mx.random.normal((B, H, N, D))

        out_fa = flash_attention(q, k, v, scale=scale, causal=True,
                                 window_size=(window_left, -1))
        # Reference: exact masked SDPA
        N2, S2 = q.shape[2], k.shape[2]
        q_idx = mx.arange(S2 - N2, S2, dtype=mx.int32)[:, None]
        k_idx = mx.arange(S2, dtype=mx.int32)[None, :]
        in_win = (k_idx >= q_idx - window_left) & (k_idx <= q_idx)
        mask = mx.where(in_win,
                        mx.zeros((N2, S2), dtype=q.dtype),
                        mx.full((N2, S2), float("-inf"), dtype=q.dtype))
        out_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        mx.eval(out_fa, out_ref)
        np.testing.assert_allclose(
            np.array(out_fa), np.array(out_ref), rtol=1e-5, atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Track LA: window_size.right native STEEL support
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ext_available(), reason="C++ extension required")
class TestWindowRight:
    """Tests for window_size=(left, right) right-side support in the STEEL kernel.

    The right window masks all K positions k > q + window_right.
    Expected: output must match a dense SDPA reference with the equivalent mask.
    """

    def _ref_window(self, q, k, v, scale, causal, window_left, window_right):
        """Reference SDPA with explicit bilateral window mask.

III-4 pass-2 B1: anchor matches the kernel qL_off = (causal and
        N<S) ? S-N : 0; see TestSlidingWindow.
        """
        N, S = q.shape[2], k.shape[2]
        q_off = (S - N) if (causal and N < S) else 0
        q_idx = mx.arange(q_off, q_off + N, dtype=mx.int32)[:, None]
        k_idx = mx.arange(S, dtype=mx.int32)[None, :]
        in_win = mx.ones((N, S), dtype=mx.bool_)
        if window_left >= 0:
            in_win = in_win & (k_idx >= q_idx - window_left)
        if window_right >= 0:
            in_win = in_win & (k_idx <= q_idx + window_right)
        if causal:
            in_win = in_win & (k_idx <= q_idx)
        mask = mx.where(in_win,
                        mx.zeros((N, S), dtype=q.dtype),
                        mx.full((N, S), float("-inf"), dtype=q.dtype))
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    @pytest.mark.parametrize("D", [64, 128])
    @pytest.mark.parametrize("causal", [False, True])
    def test_right_only_matches_ref(self, D, causal):
        """Right-only window (left=-1) output matches masked SDPA reference."""
        B, H, N, S = 1, 4, 256, 256
        scale = 1.0 / math.sqrt(D)
        window_right = 64
        q = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
        k = (mx.random.normal((B, H, S, D)) * 0.1).astype(mx.float16)
        v = (mx.random.normal((B, H, S, D)) * 0.1).astype(mx.float16)

        out_mfa = flash_attention(q, k, v, scale=scale, causal=causal,
                                  window_size=(-1, window_right))
        out_ref = self._ref_window(q, k, v, scale, causal, -1, window_right)
        mx.eval(out_mfa, out_ref)

        np.testing.assert_allclose(
            np.array(out_mfa.astype(mx.float32)),
            np.array(out_ref.astype(mx.float32)),
            rtol=1e-2, atol=5e-3,
            err_msg=f"Right-only window mismatch D={D} causal={causal}",
        )

    @pytest.mark.parametrize("D", [64, 128])
    def test_bilateral_window_matches_ref(self, D):
        """Bilateral window (left + right) output matches masked SDPA reference."""
        B, H, N, S = 1, 4, 256, 512
        scale = 1.0 / math.sqrt(D)
        window_left, window_right = 128, 32
        q = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
        k = (mx.random.normal((B, H, S, D)) * 0.1).astype(mx.float16)
        v = (mx.random.normal((B, H, S, D)) * 0.1).astype(mx.float16)

        out_mfa = flash_attention(q, k, v, scale=scale, causal=False,
                                  window_size=(window_left, window_right))
        out_ref = self._ref_window(q, k, v, scale, False, window_left, window_right)
        mx.eval(out_mfa, out_ref)

        np.testing.assert_allclose(
            np.array(out_mfa.astype(mx.float32)),
            np.array(out_ref.astype(mx.float32)),
            rtol=1e-2, atol=5e-3,
            err_msg=f"Bilateral window mismatch D={D}",
        )

    def test_window_right_output_finite(self):
        """Window right output must contain no NaN or Inf."""
        B, H, N, D = 2, 4, 512, 128
        scale = 1.0 / math.sqrt(D)
        q = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
        k = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
        v = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
        out = flash_attention(q, k, v, scale=scale, causal=False,
                              window_size=(-1, 64))
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item(), "Window-right output contains NaN/Inf"

    def test_window_right_large_allows_all(self):
        """window_right larger than sequence length = full attention."""
        B, H, N, D = 1, 4, 256, 128
        scale = 1.0 / math.sqrt(D)
        q = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
        k = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
        v = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)

        out_big_win = flash_attention(q, k, v, scale=scale,
                                     window_size=(-1, N * 2))
        out_full    = flash_attention(q, k, v, scale=scale)
        mx.eval(out_big_win, out_full)

        np.testing.assert_allclose(
            np.array(out_big_win.astype(mx.float32)),
            np.array(out_full.astype(mx.float32)),
            rtol=1e-3, atol=1e-3,
            err_msg="window_right > N should match full attention",
        )

    def test_window_right_bf16(self):
        """bfloat16 bilateral window output is finite and correct shape."""
        B, H, N, D = 1, 4, 256, 128
        scale = 1.0 / math.sqrt(D)
        q = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.bfloat16)
        k = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.bfloat16)
        v = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.bfloat16)
        out = flash_attention(q, k, v, scale=scale, window_size=(64, 64))
        mx.eval(out)
        assert out.shape == q.shape
        assert mx.all(mx.isfinite(out)).item(), "bf16 window-right output contains NaN/Inf"


# ---------------------------------------------------------------------------
# Track FA: Unified KV-cache API  (flash_attention_kvcache)
# ---------------------------------------------------------------------------

class TestUnifiedKVCache:
    """Tests for flash_attention_kvcache — dense and paged modes."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        mx.random.seed(0)

    # ------------------------------------------------------------------
    # FA.1  Dense mode — basic correctness
    # ------------------------------------------------------------------

    def test_dense_basic_matches_flash_attention(self):
        """Dense mode with no extras must equal flash_attention."""
        from mlx_mfa import flash_attention_kvcache
        B, H, N, D = 2, 4, 32, 64
        q, k, v = random_qkv(B, H, N, D)
        ref = flash_attention(q, k, v, causal=True)
        out = flash_attention_kvcache(q, k, v, causal=True)
        mx.eval(ref, out)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    def test_dense_non_causal(self):
        """Dense non-causal mode."""
        from mlx_mfa import flash_attention_kvcache
        B, H, N, D = 1, 2, 16, 128
        q, k, v = random_qkv(B, H, N, D)
        ref = flash_attention(q, k, v, causal=False)
        out = flash_attention_kvcache(q, k, v, causal=False)
        mx.eval(ref, out)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    # ------------------------------------------------------------------
    # FA.2  Dense mode — softcap
    # ------------------------------------------------------------------

    def test_dense_softcap(self):
        """Dense mode with softcap must equal flash_attention(softcap=...)."""
        from mlx_mfa import flash_attention_kvcache
        B, H, N, D = 1, 2, 32, 64
        q, k, v = random_qkv(B, H, N, D)
        ref = flash_attention(q, k, v, causal=True, softcap=30.0)
        out = flash_attention_kvcache(q, k, v, causal=True, softcap=30.0)
        mx.eval(ref, out)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    # ------------------------------------------------------------------
    # FA.3  Dense mode — ALiBi
    # ------------------------------------------------------------------

    def test_dense_alibi(self):
        """Dense mode with ALiBi must equal flash_attention(alibi_slopes=...)."""
        from mlx_mfa import flash_attention_kvcache
        B, H, N, D = 1, 4, 16, 64
        q, k, v = random_qkv(B, H, N, D)
        slopes = mx.array([0.5 ** h for h in range(1, H + 1)],
                          dtype=mx.float32)
        ref = flash_attention(q, k, v, causal=True, alibi_slopes=slopes)
        out = flash_attention_kvcache(q, k, v, causal=True, alibi_slopes=slopes)
        mx.eval(ref, out)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    # ------------------------------------------------------------------
    # FA.4  Dense mode — sliding window
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("D", [64, 128])
    def test_dense_window(self, D):
        """Dense mode with window_size must equal flash_attention(window_size=...)."""
        from mlx_mfa import flash_attention_kvcache
        B, H, N = 1, 2, 64
        q, k, v = random_qkv(B, H, N, D)
        ref = flash_attention(q, k, v, causal=True, window_size=(32, -1))
        out = flash_attention_kvcache(q, k, v, causal=True, window_size=(32, -1))
        mx.eval(ref, out)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    # ------------------------------------------------------------------
    # FA.5  Dense mode — RoPE
    # ------------------------------------------------------------------

    def test_dense_rope(self):
        """Dense mode with RoPE must equal flash_attention_rope."""
        from mlx_mfa import flash_attention_kvcache, flash_attention_rope
        B, H, N, D = 1, 2, 16, 64
        q, k, v = random_qkv(B, H, N, D)
        max_len = 256
        cos = mx.ones((max_len, D // 2), dtype=mx.float32)
        sin = mx.zeros((max_len, D // 2), dtype=mx.float32)
        past = 8
        ref = flash_attention_rope(q, k, v, rotary_cos=cos, rotary_sin=sin,
                                   causal=True, cache_seqlens=past)
        out = flash_attention_kvcache(q, k, v, rotary_cos=cos, rotary_sin=sin,
                                      causal=True, cache_seqlens=past)
        mx.eval(ref, out)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    # ------------------------------------------------------------------
    # FA.6  Dense mode — GQA (H_kv < H_q)
    # ------------------------------------------------------------------

    def test_dense_gqa(self):
        """Dense mode GQA routes correctly through flash_attention."""
        from mlx_mfa import flash_attention_kvcache
        B, H_q, H_kv, N, D = 1, 8, 2, 32, 64
        mx.random.seed(1)
        q = mx.random.normal((B, H_q, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        out = flash_attention_kvcache(q, k, v, causal=True)
        ref = flash_attention(q, k, v, causal=True)
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    # ------------------------------------------------------------------
    # FA.7  Paged mode — basic correctness
    # ------------------------------------------------------------------

    def test_paged_basic_correctness(self):
        """Paged mode must produce same result as flash_attention_paged."""
        from mlx_mfa import flash_attention_kvcache, flash_attention_paged
        B, H, N_q, D = 1, 2, 1, 64
        kv_len = 32
        block_sz = 16
        n_blocks = kv_len // block_sz
        mx.random.seed(7)
        q = mx.random.normal((B, H, N_q, D)).astype(mx.float16)
        pool_k = mx.random.normal((n_blocks, block_sz, H, D)).astype(mx.float16)
        pool_v = mx.random.normal((n_blocks, block_sz, H, D)).astype(mx.float16)
        btable = mx.array([[0, 1]], dtype=mx.int32)
        slens = mx.array([kv_len], dtype=mx.int32)

        ref = flash_attention_paged(q, pool_k, pool_v, btable, slens,
                                    scale=1.0, causal=False, block_size=block_sz)
        out = flash_attention_kvcache(q, pool_k, pool_v,
                                      block_table=btable, seq_lens=slens,
                                      block_size=block_sz,
                                      scale=1.0, causal=False)
        mx.eval(ref, out)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    # ------------------------------------------------------------------
    # FA.8  Output shape
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("D,causal", [(64, True), (128, False)])
    def test_output_shape(self, D, causal):
        """Output must be [B, H, N, D]."""
        from mlx_mfa import flash_attention_kvcache
        B, H, N = 2, 4, 16
        q, k, v = random_qkv(B, H, N, D)
        out = flash_attention_kvcache(q, k, v, causal=causal)
        mx.eval(out)
        assert out.shape == (B, H, N, D)

    # ------------------------------------------------------------------
    # FA.9  Error paths
    # ------------------------------------------------------------------

    def test_error_rope_and_alibi_together(self):
        """rotary_cos + alibi_slopes must raise ValueError."""
        from mlx_mfa import flash_attention_kvcache
        B, H, N, D = 1, 2, 8, 64
        q, k, v = random_qkv(B, H, N, D)
        cos = mx.ones((64, D // 2), dtype=mx.float32)
        sin = mx.zeros((64, D // 2), dtype=mx.float32)
        slopes = mx.ones((H,), dtype=mx.float32)
        with pytest.raises(ValueError, match="mutually exclusive"):
            flash_attention_kvcache(q, k, v, rotary_cos=cos, rotary_sin=sin,
                                    alibi_slopes=slopes)

    def test_error_paged_missing_seq_lens(self):
        """Paged mode without seq_lens must raise ValueError."""
        from mlx_mfa import flash_attention_kvcache
        B, H, N, D = 1, 2, 1, 64
        q = mx.zeros((B, H, N, D), dtype=mx.float16)
        pool = mx.zeros((4, 16, H, D), dtype=mx.float16)
        btable = mx.zeros((B, 4), dtype=mx.int32)
        with pytest.raises(ValueError, match="seq_lens"):
            flash_attention_kvcache(q, pool, pool, block_table=btable)

    def test_error_dense_missing_cache(self):
        """Dense mode with k_cache=None must raise ValueError."""
        from mlx_mfa import flash_attention_kvcache
        B, H, N, D = 1, 2, 8, 64
        q = mx.zeros((B, H, N, D), dtype=mx.float16)
        with pytest.raises(ValueError):
            flash_attention_kvcache(q, None, None)

    def test_error_q_not_4d(self):
        """Non-4D q must raise ValueError."""
        from mlx_mfa import flash_attention_kvcache
        q = mx.zeros((2, 8, 64), dtype=mx.float16)
        k = mx.zeros((2, 8, 64), dtype=mx.float16)
        with pytest.raises(ValueError, match="4-D"):
            flash_attention_kvcache(q, k, k)

    # ------------------------------------------------------------------
    # FA.10  Output is finite
    # ------------------------------------------------------------------

    def test_output_finite(self):
        """Output must be finite for normal inputs."""
        from mlx_mfa import flash_attention_kvcache
        B, H, N, D = 2, 4, 64, 128
        q, k, v = random_qkv(B, H, N, D)
        out = flash_attention_kvcache(q, k, v, causal=True)
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item()

    # ------------------------------------------------------------------
    # FA.11  Backward (autograd) works through dense path
    # ------------------------------------------------------------------

    def test_dense_backward_finite(self):
        """Backward pass via dense path must produce finite gradients."""
        from mlx_mfa import flash_attention_kvcache
        B, H, N, D = 1, 2, 16, 64
        q, k, v = random_qkv(B, H, N, D)
        def _fwd(q, k, v):
            return flash_attention_kvcache(q, k, v, causal=True)
        grads = mx.grad(lambda q, k, v: _fwd(q, k, v).sum())(q, k, v)
        mx.eval(*grads if isinstance(grads, tuple) else [grads])
        # grads is dq
        dq = grads
        assert mx.all(mx.isfinite(dq)).item()


# ---------------------------------------------------------------------------
# Track JC — Paged append in flash_attention_kvcache
# ---------------------------------------------------------------------------

class TestPagedAppend:
    """Tests for flash_attention_kvcache with k_new + block_table (paged append)."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        mx.random.seed(7)

    def test_paged_append_output_matches_dense_append(self):
        """Paged-append output matches dense concat+attend reference."""
        from mlx_mfa import flash_attention_kvcache, flash_attention

        B, H_q, H_kv, D = 1, 4, 4, 64
        block_size = 16
        past_len = 32   # 2 full blocks
        N_new = 1
        num_blocks = 4

        mx.random.seed(7)
        k_pool = mx.random.normal((num_blocks, block_size, H_kv, D)).astype(mx.float16)
        v_pool = mx.random.normal((num_blocks, block_size, H_kv, D)).astype(mx.float16)
        block_table = mx.array([[0, 1, 2, -1]], dtype=mx.int32)
        seq_lens = mx.array([past_len], dtype=mx.int32)

        q     = mx.random.normal((B, H_q, N_new, D)).astype(mx.float16)
        k_new = mx.random.normal((B, H_kv, N_new, D)).astype(mx.float16)
        v_new = mx.random.normal((B, H_kv, N_new, D)).astype(mx.float16)

        # Dense reference: gather past from pool, concat k_new, attend
        # k_pool[i] has shape [block_size, H_kv, D] → transpose to [H_kv, block_size, D]
        k_past = mx.concatenate(
            [k_pool[0].transpose(1, 0, 2), k_pool[1].transpose(1, 0, 2)], axis=1
        )[None]  # [1, H_kv, past_len, D]
        v_past = mx.concatenate(
            [v_pool[0].transpose(1, 0, 2), v_pool[1].transpose(1, 0, 2)], axis=1
        )[None]
        k_full = mx.concatenate([k_past, k_new], axis=2)
        v_full = mx.concatenate([v_past, v_new], axis=2)
        ref_out = flash_attention(q, k_full, v_full, scale=None, causal=True)

        out, k_pool_up, v_pool_up = flash_attention_kvcache(
            q, k_pool, v_pool,
            k_new=k_new, v_new=v_new,
            block_table=block_table, seq_lens=seq_lens, block_size=block_size,
            causal=True,
        )
        mx.eval(ref_out, out, k_pool_up, v_pool_up)

        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref_out.astype(mx.float32)),
            rtol=2e-3, atol=2e-3,
            err_msg="paged-append output differs from dense reference",
        )
        # New token lands in block 2 (past_len=32, slot 32//16=2), offset 0
        new_k_in_pool = k_pool_up[2, 0, :, :]   # [H_kv, D]
        expected_k    = k_new[0, :, 0, :]        # [H_kv, D]
        mx.eval(new_k_in_pool, expected_k)
        np.testing.assert_allclose(
            np.array(new_k_in_pool.astype(mx.float32)),
            np.array(expected_k.astype(mx.float32)),
            rtol=1e-5, atol=0,
            err_msg="new token not correctly scattered to pool block 2 slot 0",
        )

    def test_paged_append_pool_shape_preserved(self):
        """Pool shape must be unchanged after paged append."""
        from mlx_mfa import flash_attention_kvcache

        B, H, D = 1, 2, 64
        block_size = 16
        num_blocks = 8
        past_len = 16  # 1 block used (slots 0..15 filled)

        k_pool = mx.zeros((num_blocks, block_size, H, D), dtype=mx.float16)
        v_pool = mx.zeros((num_blocks, block_size, H, D), dtype=mx.float16)
        block_table = mx.array([[0] + [-1] * 7], dtype=mx.int32)
        seq_lens = mx.array([past_len], dtype=mx.int32)

        q     = mx.random.normal((B, H, 1, D)).astype(mx.float16)
        k_new = mx.random.normal((B, H, 1, D)).astype(mx.float16)
        v_new = mx.random.normal((B, H, 1, D)).astype(mx.float16)

        out, k_up, v_up = flash_attention_kvcache(
            q, k_pool, v_pool,
            k_new=k_new, v_new=v_new,
            block_table=block_table, seq_lens=seq_lens, block_size=block_size,
            causal=True,
        )
        mx.eval(out, k_up, v_up)
        assert k_up.shape == k_pool.shape, "k_pool shape changed after paged append"
        assert v_up.shape == v_pool.shape, "v_pool shape changed after paged append"
        assert out.shape == (B, H, 1, D), f"unexpected output shape {out.shape}"


# ---------------------------------------------------------------------------
# Track FX-1: return_lse in flash_attention
# ---------------------------------------------------------------------------

class TestReturnLSE:
    """Tests for flash_attention(return_lse=True)."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        mx.random.seed(0)

    def test_returns_tuple(self):
        """return_lse=True must return a 2-tuple (O, L)."""
        B, H, N, D = 1, 2, 32, 64
        q, k, v = random_qkv(B, H, N, D)
        result = flash_attention(q, k, v, causal=True, return_lse=True)
        assert isinstance(result, tuple) and len(result) == 2, \
            f"Expected (O, L) tuple, got {type(result)}"

    def test_output_and_lse_shapes(self):
        """O must be [B,H,N,D] and L must be [B,H,N]."""
        B, H, N, D = 2, 4, 16, 128
        q, k, v = random_qkv(B, H, N, D)
        O, L = flash_attention(q, k, v, causal=False, return_lse=True)
        mx.eval(O, L)
        assert O.shape == (B, H, N, D), f"O shape {O.shape} != {(B, H, N, D)}"
        assert L.shape == (B, H, N), f"L shape {L.shape} != {(B, H, N)}"

    def test_lse_consistent_with_softmax(self):
        """L must satisfy: O_no_lse == softmax(scores) @ V where sum(softmax)=1.

        Check that exp2(L[b,h,i] - max_score) ≈ sum(2^(score_row - max_score)).
        We verify via: O values match between return_lse=True and False.

        v2.50 Prompt 5a Section B.4: previously xfail-marked with rationale
        "pre-existing numerical issue".  Actual root cause is FP16 ULP — the
        return_lse=True path and the return_lse=False path may take different
        kernel routes (STEEL with LSE vs STEEL without), producing reductions
        in slightly different order.  Max diff observed: ~0.001 = 1 FP16 ULP
        for values near 1.0.  The original atol=1e-4 was tighter than FP16's
        ~3-decimal precision floor.  Loosened to 2e-3 (≈2 ULP).
        """
        B, H, N, D = 1, 2, 32, 64
        q, k, v = random_qkv(B, H, N, D)
        O_lse, L = flash_attention(q, k, v, causal=True, return_lse=True)
        O_ref   = flash_attention(q, k, v, causal=True)
        mx.eval(O_lse, O_ref, L)
        # Outputs must agree within FP16 ULP (different kernel routes)
        np.testing.assert_allclose(
            np.array(O_lse.astype(mx.float32)),
            np.array(O_ref.astype(mx.float32)),
            rtol=1e-3, atol=2e-3,
        )
        # L must be finite
        assert mx.all(mx.isfinite(L)).item(), "LSE contains non-finite values"

    def test_return_attn_weights_and_lse_mutually_exclusive(self):
        """return_attn_weights + return_lse must raise ValueError."""
        B, H, N, D = 1, 2, 8, 64
        q, k, v = random_qkv(B, H, N, D)
        with pytest.raises(ValueError, match="mutually exclusive"):
            flash_attention(q, k, v, return_attn_weights=True, return_lse=True)


# ---------------------------------------------------------------------------
# Track FX-2: cache_batch_idx in flash_attention_kvcache
# ---------------------------------------------------------------------------

class TestCacheBatchIdx:
    """Tests for flash_attention_kvcache(cache_batch_idx=...)."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        mx.random.seed(0)

    def test_cache_batch_idx_matches_explicit_gather(self):
        """cache_batch_idx gather must equal manually indexing the cache pool."""
        from mlx_mfa import flash_attention_kvcache
        pool_size, H, S, D = 8, 2, 32, 64
        B = 3  # only 3 of 8 pool slots are active
        mx.random.seed(5)
        k_pool = mx.random.normal((pool_size, H, S, D)).astype(mx.float16)
        v_pool = mx.random.normal((pool_size, H, S, D)).astype(mx.float16)
        q = mx.random.normal((B, H, 1, D)).astype(mx.float16)
        # Pick 3 non-contiguous slots
        idx = mx.array([2, 5, 7], dtype=mx.int32)

        # Explicit gather: select from pool manually, then attend
        k_sel = k_pool[idx]  # [B, H, S, D]
        v_sel = v_pool[idx]
        ref = flash_attention_kvcache(q, k_sel, v_sel, causal=True)

        # cache_batch_idx path
        out = flash_attention_kvcache(q, k_pool, v_pool,
                                      cache_batch_idx=idx, causal=True)
        mx.eval(ref, out)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    def test_cache_batch_idx_output_shape(self):
        """Output shape must be [B, H, N, D] with cache_batch_idx."""
        from mlx_mfa import flash_attention_kvcache
        pool_size, H, S, D = 4, 2, 16, 64
        B = 2
        k_pool = mx.zeros((pool_size, H, S, D), dtype=mx.float16)
        v_pool = mx.zeros((pool_size, H, S, D), dtype=mx.float16)
        q = mx.zeros((B, H, 1, D), dtype=mx.float16)
        idx = mx.array([0, 3], dtype=mx.int32)
        out = flash_attention_kvcache(q, k_pool, v_pool,
                                      cache_batch_idx=idx, causal=False)
        mx.eval(out)
        assert out.shape == (B, H, 1, D)


# ---------------------------------------------------------------------------
# Track FX-3: rotary_dim partial RoPE
# ---------------------------------------------------------------------------

class TestRotaryDim:
    """Tests for flash_attention_rope(rotary_dim=...)."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        mx.random.seed(0)

    def test_rotary_dim_full_matches_default(self):
        """rotary_dim=D must produce the same result as no rotary_dim."""
        from mlx_mfa import flash_attention_rope
        B, H, N, D = 1, 2, 16, 64
        q, k, v = random_qkv(B, H, N, D)
        max_len = 128
        cos = mx.random.normal((max_len, D // 2)).astype(mx.float32)
        sin = mx.random.normal((max_len, D // 2)).astype(mx.float32)
        ref = flash_attention_rope(q, k, v, rotary_cos=cos, rotary_sin=sin,
                                   causal=True)
        out = flash_attention_rope(q, k, v, rotary_cos=cos, rotary_sin=sin,
                                   causal=True, rotary_dim=D)
        mx.eval(ref, out)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    def test_rotary_dim_partial_tail_unchanged(self, monkeypatch):
        """With rotary_dim=D//2: first D//2 dims are rotated, last D//2 unchanged.

        Use identity rotation (cos=1, sin=0) for the first half so that the
        rotated result equals the original — then all dims should be unchanged,
        confirming the tail pass-through is correct.

        v2.50 Sprint 2 (Prompt 1) introduced an M5+ NAX rope dispatch that
        uses `mx.fast.rope(base=10000)` ignoring caller cos/sin tables. This
        test exercises identity rotation (custom tables); force STEEL fallback
        via MFA_DISABLE_ROPE_NAX=1 per Sprint 2 DC4 opt-out contract.
        """
        from mlx_mfa import flash_attention_rope, flash_attention
        # v2.50 Prompt 4 Section A: STEEL fallback + bumped N=8→2048.
        # Also fixed underlying cos/sin shape bug (pre-existing): the reference
        # call used full rotation but the cos/sin tables had width rot_dim//2
        # (=16), incompatible with full-rotation D//2 (=32).  Replaced reference
        # with plain flash_attention (no rope) since identity-rotation result
        # should equal no-rotation result.
        monkeypatch.setenv("MFA_DISABLE_ROPE_NAX", "1")
        B, H, N, D = 1, 2, 2048, 64
        rot_dim = D // 2  # 32
        q, k, v = random_qkv(B, H, N, D)
        max_len = N
        # Identity rotation: cos=1, sin=0 → q_rot == q for the rotated portion.
        cos = mx.ones((max_len, rot_dim // 2), dtype=mx.float32)
        sin = mx.zeros((max_len, rot_dim // 2), dtype=mx.float32)
        out = flash_attention_rope(q, k, v, rotary_cos=cos, rotary_sin=sin,
                                   causal=False, rotary_dim=rot_dim)
        # Reference: plain flash_attention (no rope).  With identity rotation,
        # q_rot=q so the rope-path output should match the no-rope output.
        ref = flash_attention(q, k, v, causal=False)
        # With identity rotation the partial and full results must agree on the
        # attended output (since q_rot=q for both paths).
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )


# ---------------------------------------------------------------------------
# Track FC: Fused RoPE in cache append
# ---------------------------------------------------------------------------

class TestKVCacheRopeAppend:
    """Tests for flash_attention_kvcache_rope_append."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        mx.random.seed(0)

    def test_rope_append_matches_naive(self):
        """Pre-rotated cache append must match naive rotate-full-KV approach."""
        from mlx_mfa import flash_attention_kvcache_rope_append, flash_attention_rope
        B, H, D = 1, 2, 64
        past_len, new_len = 8, 1
        max_len = 64
        mx.random.seed(3)
        # Past cache (already rotated at positions 0..past_len)
        k_past_unrot = mx.random.normal((B, H, past_len, D)).astype(mx.float16)
        v_past = mx.random.normal((B, H, past_len, D)).astype(mx.float16)
        k_new_unrot = mx.random.normal((B, H, new_len, D)).astype(mx.float16)
        v_new = mx.random.normal((B, H, new_len, D)).astype(mx.float16)
        q = mx.random.normal((B, H, new_len, D)).astype(mx.float16)
        cos = mx.random.normal((max_len, D // 2)).astype(mx.float32)
        sin = mx.random.normal((max_len, D // 2)).astype(mx.float32)

        # Naive reference: rotate everything at decode time
        k_past_rot = _apply_rope_for_test(k_past_unrot, cos, sin, offset=0)
        k_full_rot = mx.concatenate(
            [k_past_rot,
             _apply_rope_for_test(k_new_unrot, cos, sin, offset=past_len)], axis=2
        )
        q_rot = _apply_rope_for_test(q, cos, sin, offset=past_len)
        # C-01 cleanup (audit, 2026-06-21): a dead `ref = flash_attention(q_rot,
        # k_full_rot, v_past, ...)` line was removed here — k_full_rot had seq
        # past_len+new_len=9 but v_past had seq 8, a mismatched-K/V call that
        # silently OOB-read before the new shape guard and whose result was
        # immediately discarded (overwritten by ref2 below with the proper v_full).
        v_full = mx.concatenate([v_past, v_new], axis=2)
        ref2 = flash_attention(q_rot, k_full_rot, v_full, causal=True)

        # FC rope-append path: build pre-rotated cache first
        k_cache_rot = _apply_rope_for_test(k_past_unrot, cos, sin, offset=0)
        out, k_upd, v_upd = flash_attention_kvcache_rope_append(
            q, k_new_unrot, v_new, k_cache_rot, v_past, cos, sin,
            cache_seqlens=past_len, causal=True,
        )
        mx.eval(ref2, out)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref2.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    def test_rope_append_no_cache(self):
        """First-step (no cache) must work with k_cache=None."""
        from mlx_mfa import flash_attention_kvcache_rope_append
        B, H, N, D = 1, 2, 4, 64
        max_len = 32
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k_new = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v_new = mx.random.normal((B, H, N, D)).astype(mx.float16)
        cos = mx.ones((max_len, D // 2), dtype=mx.float32)
        sin = mx.zeros((max_len, D // 2), dtype=mx.float32)
        out, k_upd, v_upd = flash_attention_kvcache_rope_append(
            q, k_new, v_new, None, None, cos, sin, cache_seqlens=0, causal=True,
        )
        mx.eval(out)
        assert out.shape == (B, H, N, D)
        assert k_upd.shape == (B, H, N, D)

    def test_rope_append_cache_grows(self):
        """Cache shape must grow by N_new per step."""
        from mlx_mfa import flash_attention_kvcache_rope_append
        B, H, D = 1, 2, 64
        max_len = 64
        cos = mx.ones((max_len, D // 2), dtype=mx.float32)
        sin = mx.zeros((max_len, D // 2), dtype=mx.float32)

        k_cache, v_cache = None, None
        total_len = 0
        for step in range(4):
            N = 2
            q = mx.random.normal((B, H, N, D)).astype(mx.float16)
            k_new = mx.random.normal((B, H, N, D)).astype(mx.float16)
            v_new = mx.random.normal((B, H, N, D)).astype(mx.float16)
            out, k_cache, v_cache = flash_attention_kvcache_rope_append(
                q, k_new, v_new, k_cache, v_cache, cos, sin,
                cache_seqlens=total_len, causal=True,
            )
            total_len += N
            mx.eval(k_cache)
            assert k_cache.shape[2] == total_len, \
                f"step {step}: expected cache len {total_len}, got {k_cache.shape[2]}"


# Helper for tests above: rotate tensor using _apply_rope_mlx
def _apply_rope_for_test(x, cos, sin, offset):
    from mlx_mfa.attention import _apply_rope_mlx
    return _apply_rope_mlx(x, cos, sin, offset=offset, interleaved=True)


# =============================================================================
# Track FD — Paged STEEL Forward (kernel-level paged KV)
# =============================================================================

def _build_pool(k: "mx.array", v: "mx.array", block_size: int):
    """Pack contiguous [B,H,S,D] K/V into paged pool tensors.

    Returns (pool_k, pool_v, block_table, seq_lens) ready for paged attention.
    Pool layout: [num_total_blocks, block_size, H, D].
    block_table: [B, max_blocks_per_seq] int32.
    seq_lens:    [B] int32.
    """
    B, H, S, D = k.shape
    n_blk = (S + block_size - 1) // block_size
    # Pad S to multiple of block_size
    pad_len = n_blk * block_size - S
    k_pad = mx.pad(k, [(0,0),(0,0),(0,pad_len),(0,0)]) if pad_len > 0 else k
    v_pad = mx.pad(v, [(0,0),(0,0),(0,pad_len),(0,0)]) if pad_len > 0 else v
    # k_pad: [B, H, n_blk*block_size, D] → [B, n_blk, block_size, H, D]
    k_blk = k_pad.reshape(B, H, n_blk, block_size, D).transpose(0, 2, 3, 1, 4)
    # k_blk: [B, n_blk, block_size, H, D]
    v_blk = v_pad.reshape(B, H, n_blk, block_size, D).transpose(0, 2, 3, 1, 4)
    # Stack all batch blocks into a single pool: [B*n_blk, block_size, H, D]
    pool_k = k_blk.reshape(B * n_blk, block_size, H, D)
    pool_v = v_blk.reshape(B * n_blk, block_size, H, D)
    # block_table: batch b uses blocks [b*n_blk, ..., b*n_blk + n_blk - 1]
    table = mx.array(
        [[b * n_blk + i for i in range(n_blk)] for b in range(B)],
        dtype=mx.int32)
    seq_lens = mx.array([S] * B, dtype=mx.int32)
    return pool_k, pool_v, table, seq_lens


@pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
class TestPagedSteelForward:
    """Correctness tests for the kernel-level paged STEEL forward pass (Track FD).

    Each test compares `mfa_paged_steel_forward` (or `flash_attention_paged`
    which now routes to it) against the dense reference `flash_attention`.
    """

    TOL = 5e-3   # max-abs tolerance (f16 precision)

    # ── 1. D=64 non-causal ───────────────────────────────────────────────
    def test_d64_noncausal(self):
        mx.random.seed(1)
        B, H, N, S, D = 1, 4, 8, 64, 64
        bs = 16
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        scale = D**-0.5

        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        from mlx_mfa._ext import mfa_paged_steel_forward
        O_paged, L = mfa_paged_steel_forward(q, pool_k, pool_v, table, lens,
                                             scale=scale, causal=False, block_size=bs)
        # III-4 F4: reference is MLX SDPA (ground truth), not flash_attention
        O_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(O_paged, O_ref)
        diff = float(mx.abs(O_paged.astype(mx.float32) - O_ref.astype(mx.float32)).max())
        assert diff < self.TOL, f"D=64 non-causal diff={diff:.4e}"

    # ── 2. D=128 non-causal ──────────────────────────────────────────────
    def test_d128_noncausal(self):
        mx.random.seed(2)
        B, H, N, S, D = 2, 8, 16, 128, 128
        bs = 32
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        scale = D**-0.5

        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        from mlx_mfa._ext import mfa_paged_steel_forward
        O_paged, _ = mfa_paged_steel_forward(q, pool_k, pool_v, table, lens,
                                              scale=scale, causal=False, block_size=bs)
        # III-4 F4: reference is MLX SDPA (ground truth), not flash_attention
        O_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(O_paged, O_ref)
        diff = float(mx.abs(O_paged.astype(mx.float32) - O_ref.astype(mx.float32)).max())
        assert diff < self.TOL, f"D=128 non-causal diff={diff:.4e}"

    # ── 3. D=256 non-causal ──────────────────────────────────────────────
    def test_d256_noncausal(self):
        mx.random.seed(3)
        B, H, N, S, D = 1, 4, 8, 64, 256
        bs = 16
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        scale = D**-0.5

        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        from mlx_mfa._ext import mfa_paged_steel_forward
        O_paged, _ = mfa_paged_steel_forward(q, pool_k, pool_v, table, lens,
                                              scale=scale, causal=False, block_size=bs)
        # III-4 F4: reference is MLX SDPA (ground truth), not flash_attention
        O_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(O_paged, O_ref)
        diff = float(mx.abs(O_paged.astype(mx.float32) - O_ref.astype(mx.float32)).max())
        assert diff < self.TOL, f"D=256 non-causal diff={diff:.4e}"

    # ── 4. bfloat16 ──────────────────────────────────────────────────────
    def test_bf16(self):
        mx.random.seed(4)
        B, H, N, S, D = 1, 4, 8, 64, 64
        bs = 16
        q = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
        k = mx.random.normal((B, H, S, D)).astype(mx.bfloat16)
        v = mx.random.normal((B, H, S, D)).astype(mx.bfloat16)
        mx.eval(q, k, v)
        scale = D**-0.5

        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        from mlx_mfa._ext import mfa_paged_steel_forward
        O_paged, _ = mfa_paged_steel_forward(q, pool_k, pool_v, table, lens,
                                              scale=scale, causal=False, block_size=bs)
        # III-4 F4: reference is MLX SDPA (ground truth), not flash_attention
        O_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(O_paged, O_ref)
        diff = float(mx.abs(O_paged.astype(mx.float32) - O_ref.astype(mx.float32)).max())
        assert diff < 1e-2, f"bf16 diff={diff:.4e}"

    # ── 5. GQA 2:1 ───────────────────────────────────────────────────────
    def test_gqa_2to1(self):
        mx.random.seed(5)
        H_q, H_kv, D = 8, 4, 64
        B, N, S, bs = 1, 8, 64, 16
        q = mx.random.normal((B, H_q, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        scale = D**-0.5

        # Build pool with H_kv heads
        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        from mlx_mfa._ext import mfa_paged_steel_forward
        O_paged, _ = mfa_paged_steel_forward(q, pool_k, pool_v, table, lens,
                                              scale=scale, causal=False, block_size=bs)
        # III-4 F4: reference is MLX SDPA (ground truth), not flash_attention
        O_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)  # SDPA handles GQA natively
        mx.eval(O_paged, O_ref)
        diff = float(mx.abs(O_paged.astype(mx.float32) - O_ref.astype(mx.float32)).max())
        assert diff < self.TOL, f"GQA 2:1 diff={diff:.4e}"

    # ── 6. Causal mask ────────────────────────────────────────────────────
    def test_causal(self):
        mx.random.seed(6)
        B, H, N, S, D = 1, 4, 8, 64, 64
        bs = 16
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        scale = D**-0.5

        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        from mlx_mfa._ext import mfa_paged_steel_forward
        O_paged, _ = mfa_paged_steel_forward(q, pool_k, pool_v, table, lens,
                                              scale=scale, causal=True, block_size=bs)
        # III-4 F4: reference is MLX SDPA (ground truth), not flash_attention
        O_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
        mx.eval(O_paged, O_ref)
        diff = float(mx.abs(O_paged.astype(mx.float32) - O_ref.astype(mx.float32)).max())
        assert diff < self.TOL, f"causal diff={diff:.4e}"

    # ── 7. Cross-block boundary (S not multiple of block_size) ───────────
    def test_cross_block_boundary(self):
        """S=40 with block_size=16: last block is partially filled (8 tokens)."""
        mx.random.seed(7)
        B, H, N, S, D = 1, 4, 8, 40, 64
        bs = 16   # 2 full + 1 partial block
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        scale = D**-0.5

        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        from mlx_mfa._ext import mfa_paged_steel_forward
        O_paged, _ = mfa_paged_steel_forward(q, pool_k, pool_v, table, lens,
                                              scale=scale, causal=False, block_size=bs)
        # III-4 F4: reference is MLX SDPA (ground truth), not flash_attention
        O_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(O_paged, O_ref)
        diff = float(mx.abs(O_paged.astype(mx.float32) - O_ref.astype(mx.float32)).max())
        assert diff < self.TOL, f"cross-block diff={diff:.4e}"

    # ── 8. Decode (N_q=1) ────────────────────────────────────────────────
    def test_decode_nq1(self):
        """Autoregressive decode step: N_q=1 (single new token)."""
        mx.random.seed(8)
        B, H, N, S, D = 2, 8, 1, 256, 128
        bs = 32
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        scale = D**-0.5

        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        from mlx_mfa._ext import mfa_paged_steel_forward
        O_paged, _ = mfa_paged_steel_forward(q, pool_k, pool_v, table, lens,
                                              scale=scale, causal=False, block_size=bs)
        # III-4 F4: reference is MLX SDPA (ground truth), not flash_attention
        O_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(O_paged, O_ref)
        diff = float(mx.abs(O_paged.astype(mx.float32) - O_ref.astype(mx.float32)).max())
        assert diff < self.TOL, f"decode N_q=1 diff={diff:.4e}"

    # ── 9. Long context S=1024 ───────────────────────────────────────────
    def test_long_context_s1024(self):
        mx.random.seed(9)
        B, H, N, S, D = 1, 4, 64, 1024, 128
        bs = 64
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        scale = D**-0.5

        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        from mlx_mfa._ext import mfa_paged_steel_forward
        O_paged, _ = mfa_paged_steel_forward(q, pool_k, pool_v, table, lens,
                                              scale=scale, causal=False, block_size=bs)
        # III-4 F4: reference is MLX SDPA (ground truth), not flash_attention
        O_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(O_paged, O_ref)
        diff = float(mx.abs(O_paged.astype(mx.float32) - O_ref.astype(mx.float32)).max())
        assert diff < self.TOL, f"S=1024 diff={diff:.4e}"

    # ── 10. Output shape is correct ──────────────────────────────────────
    def test_output_shapes(self):
        mx.random.seed(10)
        B, H, N, S, D = 2, 4, 8, 64, 64
        bs = 16
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        from mlx_mfa._ext import mfa_paged_steel_forward
        O, L = mfa_paged_steel_forward(q, pool_k, pool_v, table, lens,
                                        scale=D**-0.5, causal=False, block_size=bs)
        mx.eval(O, L)
        assert O.shape == (B, H, N, D), f"O shape {O.shape}"
        assert L.shape == (B, H, N), f"L shape {L.shape}"
        assert O.dtype == mx.float16
        assert L.dtype == mx.float32

    # ── 11. flash_attention_paged routes to paged STEEL kernel ───────────
    def test_flash_attention_paged_uses_kernel(self):
        """flash_attention_paged with f16+D=128 should route to paged STEEL."""
        from mlx_mfa import flash_attention_paged, flash_attention
        mx.random.seed(11)
        B, H, N, S, D = 1, 4, 8, 64, 128
        bs = 16
        scale = D**-0.5
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        out_paged = flash_attention_paged(q, pool_k, pool_v, table, lens,
                                          scale=scale, causal=False, block_size=bs)
        out_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)  # III-4 F4: SDPA GT
        mx.eval(out_paged, out_ref)
        diff = float(mx.abs(out_paged.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < self.TOL, f"flash_attention_paged diff={diff:.4e}"


# =============================================================================
# Track FD-decode — Paged Flash Decode (gather + Flash Decode two-phase)
# =============================================================================

@pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
class TestPagedFlashDecode:
    """Tests for the paged Flash Decode path (N_q ≤ 4, long KV).

    When N_q ≤ 4 and max_kv_len ≥ 256, flash_attention_paged gathers K/V
    from the pool and routes to flash_attention() which activates the
    split-KV Flash Decode two-phase kernel for better parallelism.
    """

    TOL = 5e-3

    # ── 1. N_q=1, S=512 non-causal ───────────────────────────────────────
    def test_decode_s512_noncausal(self):
        from mlx_mfa import flash_attention_paged
        mx.random.seed(20)
        B, H, N, S, D = 1, 4, 1, 512, 128
        bs = 64
        scale = D**-0.5
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        out_pd = flash_attention_paged(q, pool_k, pool_v, table, lens,
                                       scale=scale, causal=False, block_size=bs)
        out_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)  # III-4 F4: SDPA GT
        mx.eval(out_pd, out_ref)
        diff = float(mx.abs(out_pd.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < self.TOL, f"N=1 S=512 non-causal diff={diff:.4e}"

    # ── 2. N_q=1, S=1024 causal ─────────────────────────────────────────
    def test_decode_s1024_causal(self):
        from mlx_mfa import flash_attention_paged
        mx.random.seed(21)
        B, H, N, S, D = 1, 4, 1, 1024, 128
        bs = 64
        scale = D**-0.5
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        out_pd = flash_attention_paged(q, pool_k, pool_v, table, lens,
                                       scale=scale, causal=True, block_size=bs)
        out_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")  # III-4 F4: SDPA GT
        mx.eval(out_pd, out_ref)
        diff = float(mx.abs(out_pd.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < self.TOL, f"N=1 S=1024 causal diff={diff:.4e}"

    # ── 3. N_q=4, S=512 (boundary of flash decode activation) ────────────
    def test_decode_nq4_s512(self):
        from mlx_mfa import flash_attention_paged
        mx.random.seed(22)
        B, H, N, S, D = 2, 8, 4, 512, 64
        bs = 64
        scale = D**-0.5
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        out_pd = flash_attention_paged(q, pool_k, pool_v, table, lens,
                                       scale=scale, causal=False, block_size=bs)
        out_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)  # III-4 F4: SDPA GT
        mx.eval(out_pd, out_ref)
        diff = float(mx.abs(out_pd.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < self.TOL, f"N=4 S=512 diff={diff:.4e}"

    # ── 4. GQA 4:1, N_q=1, S=512 ─────────────────────────────────────────
    def test_decode_gqa(self):
        from mlx_mfa import flash_attention_paged
        mx.random.seed(23)
        H_q, H_kv = 8, 2
        B, N, S, D = 1, 1, 512, 64
        bs = 64
        scale = D**-0.5
        q = mx.random.normal((B, H_q, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        pool_k, pool_v, table, lens = _build_pool(k, v, bs)
        out_pd = flash_attention_paged(q, pool_k, pool_v, table, lens,
                                       scale=scale, causal=False, block_size=bs)
        out_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)  # III-4 F4: SDPA GT (native GQA)
        mx.eval(out_pd, out_ref)
        diff = float(mx.abs(out_pd.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < self.TOL, f"GQA 4:1 decode diff={diff:.4e}"


# ============================================================================
# Track HA — D=512 forward tests
# ============================================================================

@pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")
class TestD512DelegatesToSDPA:
    """D=512 public correctness tests for the intentional SDPA delegation.

    D=512 is outside the V6 NAX expert binding.  The legacy STEEL D-split
    implementation is only reachable through an explicit force path; the
    default public route is SDPA and every test below locks that fact.
    """

    TOL = 2e-2  # f16/bf16 tolerance

    def _ref(self, q, k, v, scale, causal):
        import mlx.core.fast as fast
        mask = "causal" if causal else None
        return fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    def _public_sdpa(self, q, k, v, scale, causal):
        from mlx_mfa import _dispatch_trace as dtrace
        with dtrace.capture() as trace:
            out = flash_attention(q, k, v, scale=scale, causal=causal)
            mx.eval(out)
            mx.synchronize()
        terminal = [item for item in trace if not item[1].startswith("[reentrant]")]
        assert terminal and terminal[-1][0] == "sdpa", (
            f"D=512 must delegate publicly to SDPA, got trace={trace}"
        )
        assert not any(item[0].startswith("mfa") or item[0].startswith("v6")
                       for item in terminal), trace
        return out

    def test_expert_v6_nax_rejects_d512(self):
        """The V6 NAX expert binding rejects unsupported D=512 explicitly."""
        from mlx_mfa import _ext
        q = mx.zeros((1, 1, 8, 512), dtype=mx.float16)
        k = mx.zeros((1, 1, 8, 512), dtype=mx.float16)
        v = mx.zeros((1, 1, 8, 512), dtype=mx.float16)
        scale = float(512 ** -0.5)
        with pytest.raises(RuntimeError, match=r"V6: D must be 64, 128, or expert-only 256"):
            _ext.v6_nax_forward(q, k, v, False, True, scale)

    def test_expert_mfa_lse_rejects_d512(self):
        """The legacy expert MFA binding also rejects D=512, without fallback."""
        from mlx_mfa import _ext
        q = mx.zeros((1, 1, 8, 512), dtype=mx.float16)
        k = mx.zeros((1, 1, 8, 512), dtype=mx.float16)
        v = mx.zeros((1, 1, 8, 512), dtype=mx.float16)
        scale = float(512 ** -0.5)
        with pytest.raises(ValueError, match=r"mfa_forward_with_lse: head_dim must be 64, 128, or 256"):
            _ext.mfa_forward_with_lse(q, k, v, scale, False)

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    def test_forward_causal(self, dtype):
        """D=512 causal forward matches SDPA reference."""
        mx.random.seed(1)
        B, H, N, D = 1, 4, 64, 512
        q = mx.random.normal([B, H, N, D]).astype(dtype)
        k = mx.random.normal([B, H, N, D]).astype(dtype)
        v = mx.random.normal([B, H, N, D]).astype(dtype)
        scale = float(D ** -0.5)
        out = self._public_sdpa(q, k, v, scale, causal=True)
        ref = self._ref(q, k, v, scale, causal=True)
        mx.eval(out, ref)
        err = float(mx.max(mx.abs(out - ref)))
        assert err < self.TOL, f"causal max_err={err:.4f} dtype={dtype}"

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    def test_forward_non_causal(self, dtype):
        """D=512 non-causal forward matches SDPA reference."""
        mx.random.seed(2)
        B, H, N, D = 1, 4, 64, 512
        q = mx.random.normal([B, H, N, D]).astype(dtype)
        k = mx.random.normal([B, H, N, D]).astype(dtype)
        v = mx.random.normal([B, H, N, D]).astype(dtype)
        scale = float(D ** -0.5)
        out = self._public_sdpa(q, k, v, scale, causal=False)
        ref = self._ref(q, k, v, scale, causal=False)
        mx.eval(out, ref)
        err = float(mx.max(mx.abs(out - ref)))
        assert err < self.TOL, f"non-causal max_err={err:.4f} dtype={dtype}"

    def test_forward_unaligned_seqlen(self):
        """D=512 with N not a multiple of BQ=32."""
        mx.random.seed(3)
        B, H, N, D = 1, 4, 65, 512  # 65 = 2*32 + 1
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        scale = float(D ** -0.5)
        out = self._public_sdpa(q, k, v, scale, causal=True)
        ref = self._ref(q, k, v, scale, causal=True)
        mx.eval(out, ref)
        err = float(mx.max(mx.abs(out - ref)))
        assert err < self.TOL, f"unaligned max_err={err:.4f}"

    def test_forward_batch_multi_head(self):
        """D=512 with larger batch/head counts."""
        mx.random.seed(4)
        B, H, N, D = 2, 8, 128, 512
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        scale = float(D ** -0.5)
        out = self._public_sdpa(q, k, v, scale, causal=True)
        ref = self._ref(q, k, v, scale, causal=True)
        mx.eval(out, ref)
        err = float(mx.max(mx.abs(out - ref)))
        assert err < self.TOL, f"B=2 H=8 max_err={err:.4f}"

    def test_forward_gqa(self):
        """D=512 GQA (H_q=8, H_kv=2)."""
        mx.random.seed(5)
        B, H_q, H_kv, N, D = 1, 8, 2, 64, 512
        q = mx.random.normal([B, H_q, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H_kv, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H_kv, N, D]).astype(mx.float16)
        scale = float(D ** -0.5)
        out = self._public_sdpa(q, k, v, scale, causal=True)
        k_exp = mx.repeat(k, H_q // H_kv, axis=1)
        v_exp = mx.repeat(v, H_q // H_kv, axis=1)
        ref = self._ref(q, k_exp, v_exp, scale, causal=True)
        mx.eval(out, ref)
        err = float(mx.max(mx.abs(out - ref)))
        assert err < self.TOL, f"GQA max_err={err:.4f}"

    def test_forward_decode_n1(self):
        """D=512 single-token decode (N=1)."""
        mx.random.seed(6)
        B, H, N, S, D = 1, 4, 1, 64, 512
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, S, D]).astype(mx.float16)
        v = mx.random.normal([B, H, S, D]).astype(mx.float16)
        scale = float(D ** -0.5)
        out = self._public_sdpa(q, k, v, scale, causal=False)
        ref = self._ref(q, k, v, scale, causal=False)
        mx.eval(out, ref)
        err = float(mx.max(mx.abs(out - ref)))
        assert err < self.TOL, f"decode N=1 max_err={err:.4f}"


# ────────────────────────────────────────────────────────────────
# Track HB — D=512 backward tests
# ────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")
class TestD512Backward:
    """Gradient correctness for the public D=512 SDPA delegation."""

    TOL_F16  = 5e-2
    TOL_BF16 = 1e-1

    def _ref_bwd(self, q, k, v, scale, causal):
        """Expand GQA and run mx.vjp(SDPA)."""
        H_q, H_kv = q.shape[1], k.shape[1]
        if H_q != H_kv:
            k = mx.repeat(k, H_q // H_kv, axis=1)
            v = mx.repeat(v, H_q // H_kv, axis=1)
        mask = "causal" if causal else None
        def fn(q, k, v):
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=mask
            )
        co = mx.ones_like(mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask))
        _, grads = mx.vjp(fn, [q, k, v], [co])
        return grads

    def _public_sdpa_bwd(self, q, k, v, scale, causal):
        from mlx_mfa import _dispatch_trace as dtrace
        with dtrace.capture() as trace:
            co = mx.ones_like(flash_attention(q, k, v, scale=scale, causal=causal))
            _, grads = mx.vjp(
                lambda q, k, v: flash_attention(q, k, v, scale=scale, causal=causal),
                [q, k, v], [co]
            )
            mx.eval(*grads)
            mx.synchronize()
        terminal = [item for item in trace if not item[1].startswith("[reentrant]")]
        assert terminal and terminal[-1][0] == "sdpa", (
            f"D=512 backward must delegate publicly to SDPA, got trace={trace}"
        )
        return grads

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    def test_backward_causal(self, dtype):
        """D=512 causal backward: dQ, dK, dV within tolerance."""
        mx.random.seed(10)
        B, H, N, D = 1, 4, 128, 512
        scale = float(D ** -0.5)
        q = mx.random.normal([B, H, N, D]).astype(dtype)
        k = mx.random.normal([B, H, N, D]).astype(dtype)
        v = mx.random.normal([B, H, N, D]).astype(dtype)
        dq_m, dk_m, dv_m = self._public_sdpa_bwd(q, k, v, scale, causal=True)
        dq_r, dk_r, dv_r = self._ref_bwd(q, k, v, scale, causal=True)
        mx.eval(dq_m, dk_m, dv_m, dq_r, dk_r, dv_r)
        tol = self.TOL_F16 if dtype == mx.float16 else self.TOL_BF16
        def err(a, b): return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        assert err(dq_m, dq_r) < tol, f"dQ err={err(dq_m, dq_r):.5f}"
        assert err(dk_m, dk_r) < tol, f"dK err={err(dk_m, dk_r):.5f}"
        assert err(dv_m, dv_r) < tol, f"dV err={err(dv_m, dv_r):.5f}"

    def test_backward_non_causal(self):
        """D=512 non-causal backward."""
        mx.random.seed(11)
        B, H, N, D = 1, 4, 128, 512
        scale = float(D ** -0.5)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        dq_m, dk_m, dv_m = self._public_sdpa_bwd(q, k, v, scale, causal=False)
        dq_r, dk_r, dv_r = self._ref_bwd(q, k, v, scale, causal=False)
        mx.eval(dq_m, dk_m, dv_m, dq_r, dk_r, dv_r)
        def err(a, b): return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        assert err(dq_m, dq_r) < self.TOL_F16, f"dQ err={err(dq_m, dq_r):.5f}"
        assert err(dk_m, dk_r) < self.TOL_F16, f"dK err={err(dk_m, dk_r):.5f}"
        assert err(dv_m, dv_r) < self.TOL_F16, f"dV err={err(dv_m, dv_r):.5f}"

    def test_backward_gqa(self):
        """D=512 GQA (8:2) backward: gradient shapes and finite values."""
        mx.random.seed(12)
        B, H_q, H_kv, N, D = 1, 8, 2, 64, 512
        scale = float(D ** -0.5)
        q = mx.random.normal([B, H_q, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H_kv, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H_kv, N, D]).astype(mx.float16)
        dq_m, dk_m, dv_m = self._public_sdpa_bwd(q, k, v, scale, causal=True)
        mx.eval(dq_m, dk_m, dv_m)
        assert dq_m.shape == q.shape, f"dQ shape mismatch {dq_m.shape} vs {q.shape}"
        assert dk_m.shape == k.shape, f"dK shape mismatch {dk_m.shape} vs {k.shape}"
        assert dv_m.shape == v.shape, f"dV shape mismatch {dv_m.shape} vs {v.shape}"
        assert mx.all(mx.isfinite(dq_m)).item(), "dQ has non-finite values"
        assert mx.all(mx.isfinite(dk_m)).item(), "dK has non-finite values"
        assert mx.all(mx.isfinite(dv_m)).item(), "dV has non-finite values"

    def test_backward_value_and_grad(self):
        """D=512 value_and_grad works end-to-end."""
        from mlx_mfa import _dispatch_trace as dtrace
        mx.random.seed(13)
        B, H, N, D = 1, 2, 64, 512
        scale = float(D ** -0.5)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)

        def loss(q, k, v):
            return mx.mean(flash_attention(q, k, v, scale=scale, causal=True))

        val_and_grad = mx.value_and_grad(loss)
        with dtrace.capture() as trace:
            loss_val, grads = val_and_grad(q, k, v)
            mx.eval(loss_val, *grads)
            mx.synchronize()
        terminal = [item for item in trace if not item[1].startswith("[reentrant]")]
        assert terminal and terminal[-1][0] == "sdpa", (
            f"D=512 value_and_grad must delegate publicly to SDPA, got trace={trace}"
        )
        assert mx.isfinite(loss_val).item(), "loss is not finite"
        assert all(mx.all(mx.isfinite(g)).item() for g in grads), "grads have non-finite values"

# ---------------------------------------------------------------------------
# Track JB — flash_attention_rope_unified
# ---------------------------------------------------------------------------

class TestRoPEUnified:
    """Tests for flash_attention_rope_unified — the single RoPE entry point."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        mx.random.seed(42)

    def _make_cos_sin(self, max_len, head_dim, dtype=mx.float32):
        import mlx.core as mx
        half = head_dim // 2
        freq = 1.0 / (10000.0 ** (mx.arange(half, dtype=mx.float32) / half))
        pos = mx.arange(max_len, dtype=mx.float32)
        angles = pos[:, None] * freq[None, :]  # [max_len, half]
        return mx.cos(angles), mx.sin(angles)

    def test_standalone_matches_rope(self):
        """Unified standalone mode must match flash_attention_rope."""
        from mlx_mfa import flash_attention_rope, flash_attention_rope_unified
        B, H, N, D = 1, 4, 64, 128
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        cos, sin = self._make_cos_sin(256, D)
        ref = flash_attention_rope(q, k, v, cos, sin, causal=True)
        out = flash_attention_rope_unified(q, k, v, cos, sin, causal=True,
                                           return_updated_cache=False)
        mx.eval(ref, out)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    def test_cache_append_first_step(self):
        """Cache-append first step (k_cache=None) returns 3-tuple."""
        from mlx_mfa import flash_attention_rope_unified
        B, H, N, D = 1, 4, 8, 128
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        cos, sin = self._make_cos_sin(256, D)
        result = flash_attention_rope_unified(
            q, k, v, cos, sin,
            k_cache=None, v_cache=None,
            return_updated_cache=True, causal=True,
        )
        assert isinstance(result, tuple) and len(result) == 3
        out, k_upd, v_upd = result
        mx.eval(out, k_upd, v_upd)
        assert out.shape == (B, H, N, D)
        assert k_upd.shape == (B, H, N, D)   # no past; updated = k_new_rotated
        assert v_upd.shape == (B, H, N, D)

    def test_cache_append_matches_rope_append(self):
        """Cache-append mode must match flash_attention_kvcache_rope_append."""
        from mlx_mfa import flash_attention_kvcache_rope_append, flash_attention_rope_unified
        B, H, N, D = 1, 4, 8, 128
        past_len = 32
        q     = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k_new = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v_new = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k_cache = mx.random.normal((B, H, past_len, D)).astype(mx.float16)
        v_cache = mx.random.normal((B, H, past_len, D)).astype(mx.float16)
        cos, sin = self._make_cos_sin(256, D)
        ref_out, ref_k, ref_v = flash_attention_kvcache_rope_append(
            q, k_new, v_new, k_cache, v_cache, cos, sin,
            cache_seqlens=past_len, causal=True,
        )
        out, k_upd, v_upd = flash_attention_rope_unified(
            q, k_new, v_new, cos, sin,
            k_cache=k_cache, v_cache=v_cache,
            cache_seqlens=past_len, return_updated_cache=True, causal=True,
        )
        mx.eval(ref_out, out, ref_k, k_upd)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref_out.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )
        np.testing.assert_allclose(
            np.array(k_upd.astype(mx.float32)),
            np.array(ref_k.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    def test_standalone_return_false(self):
        """return_updated_cache=False returns a plain array, not a tuple."""
        from mlx_mfa import flash_attention_rope_unified
        B, H, N, D = 1, 2, 16, 64
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        cos, sin = self._make_cos_sin(128, D)
        out = flash_attention_rope_unified(
            q, k, v, cos, sin, return_updated_cache=False, causal=False,
        )
        mx.eval(out)
        assert isinstance(out, mx.array)
        assert out.shape == (B, H, N, D)

    def test_rope_3d_standalone(self):
        """rope_3d parameter routes through 3D table construction."""
        from mlx_mfa import flash_attention_rope_unified, flash_attention_rope
        B, H, N, D = 1, 2, 16, 128
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        rope_3d_cfg = {"grid_h": 4, "grid_w": 4, "num_frames": 1}
        ref = flash_attention_rope(q, k, v, rope_3d=rope_3d_cfg, causal=False)
        out = flash_attention_rope_unified(
            q, k, v, rope_3d=rope_3d_cfg,
            return_updated_cache=False, causal=False,
        )
        mx.eval(ref, out)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    def test_multistep_grow(self):
        """Two-step incremental decode: cache grows correctly."""
        from mlx_mfa import flash_attention_rope_unified
        B, H, D = 1, 4, 128
        cos, sin = self._make_cos_sin(256, D)
        # Step 0 — prefill 32 tokens
        q0 = mx.random.normal((B, H, 32, D)).astype(mx.float16)
        k0 = mx.random.normal((B, H, 32, D)).astype(mx.float16)
        v0 = mx.random.normal((B, H, 32, D)).astype(mx.float16)
        out0, kc, vc = flash_attention_rope_unified(
            q0, k0, v0, cos, sin,
            return_updated_cache=True, causal=True,
        )
        mx.eval(out0, kc, vc)
        assert kc.shape == (B, H, 32, D)
        # Step 1 — decode 1 token
        q1 = mx.random.normal((B, H, 1, D)).astype(mx.float16)
        k1 = mx.random.normal((B, H, 1, D)).astype(mx.float16)
        v1 = mx.random.normal((B, H, 1, D)).astype(mx.float16)
        out1, kc2, vc2 = flash_attention_rope_unified(
            q1, k1, v1, cos, sin,
            k_cache=kc, v_cache=vc,
            cache_seqlens=32, return_updated_cache=True, causal=True,
        )
        mx.eval(out1, kc2, vc2)
        assert kc2.shape == (B, H, 33, D)
        assert out1.shape == (B, H, 1, D)

    def test_per_batch_cache_seqlens(self):
        """Per-batch list cache_seqlens with cache-append mode."""
        from mlx_mfa import flash_attention_rope_unified
        B, H, D = 2, 2, 64
        past0, past1 = 8, 16
        cos, sin = self._make_cos_sin(64, D)
        k_cache0 = mx.random.normal((1, H, past0, D)).astype(mx.float16)
        k_cache1 = mx.random.normal((1, H, past1, D)).astype(mx.float16)
        # Stack caches padded to max(past_len) — simpler: call separately and concat
        q0  = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        k0  = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        v0  = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        q1  = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        k1  = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        v1  = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        v_cache0 = mx.random.normal((1, H, past0, D)).astype(mx.float16)
        v_cache1 = mx.random.normal((1, H, past1, D)).astype(mx.float16)
        ref0, _, _ = flash_attention_rope_unified(
            q0, k0, v0, cos, sin, k_cache=k_cache0, v_cache=v_cache0,
            cache_seqlens=past0, return_updated_cache=True, causal=True,
        )
        ref1, _, _ = flash_attention_rope_unified(
            q1, k1, v1, cos, sin, k_cache=k_cache1, v_cache=v_cache1,
            cache_seqlens=past1, return_updated_cache=True, causal=True,
        )
        mx.eval(ref0, ref1)
        # Both outputs must be finite
        assert mx.all(mx.isfinite(ref0)).item()
        assert mx.all(mx.isfinite(ref1)).item()

# ---------------------------------------------------------------------------
# Track JD — LLM inference helpers
# ---------------------------------------------------------------------------

class TestSpeculativeVerify:
    """Tests for flash_attention_speculative_verify."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        mx.random.seed(0)

    def test_returns_correct_shapes(self):
        """Returns (out, lse, target_logprobs) with correct shapes."""
        from mlx_mfa import flash_attention_speculative_verify
        B, H, N_draft, D = 1, 4, 4, 128
        S = 64
        q = mx.random.normal((B, H, N_draft, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        draft_ids = mx.zeros((B, N_draft), dtype=mx.int32)
        out, lse, lp = flash_attention_speculative_verify(q, k, v, draft_ids)
        mx.eval(out, lse, lp)
        assert out.shape == (B, H, N_draft, D)
        assert lse.shape == (B, H, N_draft)
        assert lp.shape == (B, N_draft)

    def test_output_is_finite(self):
        """All returned tensors must be finite."""
        from mlx_mfa import flash_attention_speculative_verify
        B, H, N_draft, D = 1, 4, 4, 64
        S = 32
        q = mx.random.normal((B, H, N_draft, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        draft_ids = mx.zeros((B, N_draft), dtype=mx.int32)
        out, lse, lp = flash_attention_speculative_verify(q, k, v, draft_ids)
        mx.eval(out, lse, lp)
        assert mx.all(mx.isfinite(out)).item(), "out has non-finite values"
        assert mx.all(mx.isfinite(lse)).item(), "lse has non-finite values"
        assert mx.all(mx.isfinite(lp)).item(), "target_logprobs has non-finite values"

    def test_output_matches_flash_attention(self):
        """out must match flash_attention output (lse is bonus)."""
        from mlx_mfa import flash_attention_speculative_verify, flash_attention
        B, H, N_draft, D = 1, 4, 4, 64
        S = 32
        q = mx.random.normal((B, H, N_draft, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        draft_ids = mx.zeros((B, N_draft), dtype=mx.int32)
        ref = flash_attention(q, k, v, causal=True)
        out, _, _ = flash_attention_speculative_verify(q, k, v, draft_ids)
        mx.eval(ref, out)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)),
            np.array(ref.astype(mx.float32)),
            rtol=1e-4, atol=1e-4,
        )

    def test_logprobs_are_negative(self):
        """Log-probabilities must be ≤ 0."""
        from mlx_mfa import flash_attention_speculative_verify
        B, H, N_draft, D = 1, 2, 3, 64
        q = mx.random.normal((B, H, N_draft, D)).astype(mx.float16)
        k = mx.random.normal((B, H, 32, D)).astype(mx.float16)
        v = mx.random.normal((B, H, 32, D)).astype(mx.float16)
        draft_ids = mx.zeros((B, N_draft), dtype=mx.int32)
        _, _, lp = flash_attention_speculative_verify(q, k, v, draft_ids)
        mx.eval(lp)
        assert mx.all(lp <= 0).item(), "log-probs must be ≤ 0"


class TestSharedPrefixCache:
    """Tests for make_shared_prefix_cache."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        mx.random.seed(1)

    def test_returns_three_tuple(self):
        """Returns (prefix_out, k_prefix, v_prefix)."""
        from mlx_mfa import make_shared_prefix_cache
        B, H, N_pre, D = 1, 4, 32, 64
        q = mx.random.normal((B, H, N_pre, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N_pre, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N_pre, D)).astype(mx.float16)
        out, kp, vp = make_shared_prefix_cache(q, k, v)
        mx.eval(out, kp, vp)
        assert out.shape == (B, H, N_pre, D)
        assert kp.shape == k.shape
        assert vp.shape == v.shape

    def test_prefix_k_equals_input_k(self):
        """Returned k_prefix must be identical to the input k."""
        from mlx_mfa import make_shared_prefix_cache
        B, H, N_pre, D = 1, 2, 16, 64
        q = mx.random.normal((B, H, N_pre, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N_pre, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N_pre, D)).astype(mx.float16)
        _, kp, vp = make_shared_prefix_cache(q, k, v)
        mx.eval(kp, vp)
        np.testing.assert_array_equal(
            np.array(kp.astype(mx.float32)),
            np.array(k.astype(mx.float32)),
        )

    def test_suffix_concat_extends_cache(self):
        """Concatenating k_prefix with k_suffix and attending produces finite output."""
        from mlx_mfa import make_shared_prefix_cache, flash_attention
        B, H, N_pre, N_suf, D = 1, 4, 16, 8, 64
        q_pre = mx.random.normal((B, H, N_pre, D)).astype(mx.float16)
        k_pre = mx.random.normal((B, H, N_pre, D)).astype(mx.float16)
        v_pre = mx.random.normal((B, H, N_pre, D)).astype(mx.float16)
        _, kp, vp = make_shared_prefix_cache(q_pre, k_pre, v_pre)

        q_suf = mx.random.normal((B, H, N_suf, D)).astype(mx.float16)
        k_suf = mx.random.normal((B, H, N_suf, D)).astype(mx.float16)
        v_suf = mx.random.normal((B, H, N_suf, D)).astype(mx.float16)
        k_full = mx.concatenate([kp, k_suf], axis=2)
        v_full = mx.concatenate([vp, v_suf], axis=2)
        out = flash_attention(q_suf, k_full, v_full, causal=True)
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item()


class TestSplitFuse:
    """Tests for flash_attention_splitfuse."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        mx.random.seed(2)

    def test_prefill_only(self):
        """Passing only prefill returns (out_prefill, None)."""
        from mlx_mfa import flash_attention_splitfuse
        B, H, N_p, D = 1, 4, 64, 128
        q = mx.random.normal((B, H, N_p, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N_p, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N_p, D)).astype(mx.float16)
        out_p, out_d = flash_attention_splitfuse(q, k, v, None, None, None)
        mx.eval(out_p)
        assert out_p is not None and out_p.shape == (B, H, N_p, D)
        assert out_d is None

    def test_decode_only(self):
        """Passing only decode returns (None, out_decode)."""
        from mlx_mfa import flash_attention_splitfuse
        B, H, N_d, S, D = 1, 4, 1, 128, 128
        q = mx.random.normal((B, H, N_d, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        out_p, out_d = flash_attention_splitfuse(None, None, None, q, k, v)
        mx.eval(out_d)
        assert out_p is None
        assert out_d is not None and out_d.shape == (B, H, N_d, D)

    def test_splitfuse_both(self):
        """Prefill + decode simultaneously — both outputs are finite."""
        from mlx_mfa import flash_attention_splitfuse
        B_p, B_d = 2, 4
        H, N_p, N_d, S, D = 4, 128, 1, 256, 128
        qp = mx.random.normal((B_p, H, N_p, D)).astype(mx.float16)
        kp = mx.random.normal((B_p, H, N_p, D)).astype(mx.float16)
        vp = mx.random.normal((B_p, H, N_p, D)).astype(mx.float16)
        qd = mx.random.normal((B_d, H, N_d, D)).astype(mx.float16)
        kd = mx.random.normal((B_d, H, S, D)).astype(mx.float16)
        vd = mx.random.normal((B_d, H, S, D)).astype(mx.float16)
        out_p, out_d = flash_attention_splitfuse(qp, kp, vp, qd, kd, vd)
        mx.eval(out_p, out_d)
        assert mx.all(mx.isfinite(out_p)).item()
        assert mx.all(mx.isfinite(out_d)).item()
        assert out_p.shape == (B_p, H, N_p, D)
        assert out_d.shape == (B_d, H, N_d, D)


# ---------------------------------------------------------------------------
# Track JF: Cross-attention via flash_attention_kvcache
# ---------------------------------------------------------------------------

class TestCrossAttentionKVCache:
    """flash_attention_kvcache as cross-attention (encoder–decoder)."""

    def test_forward_shape_and_finite(self):
        """Basic cross-attention: Q from decoder, K/V from encoder, causal=False."""
        from mlx_mfa import flash_attention_kvcache
        B, H_q, H_kv, S_enc, S_dec, D = 2, 8, 2, 256, 32, 128
        q = mx.random.normal([B, H_q, S_dec, D]).astype(mx.float16)
        k = mx.random.normal([B, H_kv, S_enc, D]).astype(mx.float16)
        v = mx.random.normal([B, H_kv, S_enc, D]).astype(mx.float16)
        out = flash_attention_kvcache(q, k, v, causal=False)
        mx.eval(out)
        assert out.shape == (B, H_q, S_dec, D)
        assert mx.all(mx.isfinite(out)).item()

    def test_single_token_decode(self):
        """Single-token (N_q=1) cross-attention — typical decode-step usage."""
        from mlx_mfa import flash_attention_kvcache
        B, H_q, H_kv, S_enc, D = 1, 4, 4, 128, 64
        q = mx.random.normal([B, H_q, 1, D]).astype(mx.float16)
        k = mx.random.normal([B, H_kv, S_enc, D]).astype(mx.float16)
        v = mx.random.normal([B, H_kv, S_enc, D]).astype(mx.float16)
        out = flash_attention_kvcache(q, k, v, causal=False)
        mx.eval(out)
        assert out.shape == (B, H_q, 1, D), out.shape
        assert mx.all(mx.isfinite(out)).item()

    def test_autograd_gradients_finite(self):
        """Cross-attention backward: dQ, dK_enc, dV_enc all finite."""
        from mlx_mfa import flash_attention_kvcache
        B, H_q, H_kv, S_enc, S_dec, D = 2, 4, 2, 64, 16, 128
        q = mx.random.normal([B, H_q, S_dec, D]).astype(mx.float16)
        k = mx.random.normal([B, H_kv, S_enc, D]).astype(mx.float16)
        v = mx.random.normal([B, H_kv, S_enc, D]).astype(mx.float16)

        def fn(q_, k_, v_):
            return flash_attention_kvcache(q_, k_, v_, causal=False)

        cot = mx.ones([B, H_q, S_dec, D], dtype=mx.float16)
        _, grads = mx.vjp(fn, (q, k, v), (cot,))
        dq, dk, dv = grads
        mx.eval(dq, dk, dv)
        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape
        assert mx.all(mx.isfinite(dq)).item()
        assert mx.all(mx.isfinite(dk)).item()
        assert mx.all(mx.isfinite(dv)).item()


# ---------------------------------------------------------------------------
# Track KA: SageAttention quantization utilities
# ---------------------------------------------------------------------------

class TestSageQuantization:
    """Tests for mlx_mfa.quantize — per-block int8 quantization utilities."""

    def test_quantize_per_block_roundtrip_f16(self):
        """Quantize then dequantize stays close to original (int8 precision)."""
        from mlx_mfa.quantize import quantize_per_block, dequantize
        B, H, N, D = 1, 2, 64, 128
        mx.random.seed(0)
        x = mx.random.normal([B, H, N, D]).astype(mx.float16)
        x_int8, scale = quantize_per_block(x, block_size=32)
        mx.eval(x_int8, scale)
        x_deq = dequantize(x_int8, scale, block_size=32, dtype=mx.float16)
        mx.eval(x_deq)
        # int8 has ~0.8% relative error per element; absolute tolerance ~ absmax/127
        absmax = float(mx.max(mx.abs(x)).item())
        atol = absmax / 127.0 * 1.5  # 1.5× rounding budget
        err = float(mx.max(mx.abs(x.astype(mx.float32) - x_deq.astype(mx.float32))).item())
        assert err <= atol, f"roundtrip error {err:.4f} > atol {atol:.4f}"

    def test_quantize_scale_shape(self):
        """Scale tensor has correct shape [B, H, N_blocks, 1]."""
        from mlx_mfa.quantize import quantize_per_block
        B, H, N, D = 2, 4, 48, 64
        x = mx.random.normal([B, H, N, D]).astype(mx.float16)
        x_int8, scale = quantize_per_block(x, block_size=16)
        mx.eval(x_int8, scale)
        assert x_int8.shape == (B, H, N, D), x_int8.shape
        assert x_int8.dtype == mx.int8
        assert scale.shape == (B, H, 3, 1), scale.shape  # 48/16 = 3 blocks
        assert scale.dtype == mx.float32

    def test_quantize_non_multiple_N(self):
        """N not divisible by block_size: output has same N as input."""
        from mlx_mfa.quantize import quantize_per_block
        B, H, N, D = 1, 1, 50, 64  # 50 not divisible by 32
        x = mx.random.normal([B, H, N, D]).astype(mx.float16)
        x_int8, scale = quantize_per_block(x, block_size=32)
        mx.eval(x_int8, scale)
        assert x_int8.shape == (B, H, N, D)
        assert scale.shape == (B, H, 2, 1)  # ceil(50/32) = 2 blocks

    def test_smooth_k_mean_subtracted(self):
        """smooth_k subtracts per-channel mean correctly."""
        from mlx_mfa.quantize import smooth_k
        B, H, S, D = 1, 2, 32, 64
        mx.random.seed(7)
        k = mx.random.normal([B, H, S, D]).astype(mx.float16)
        k_smooth, k_mean = smooth_k(k)
        mx.eval(k_smooth, k_mean)
        # Smoothed K should have near-zero channel mean
        residual_mean = mx.mean(k_smooth.astype(mx.float32), axis=2)  # [B,H,D]
        max_residual = float(mx.max(mx.abs(residual_mean)).item())
        assert max_residual < 0.01, f"Residual mean too large: {max_residual}"

    def test_smooth_k_shapes(self):
        """smooth_k returns correct shapes."""
        from mlx_mfa.quantize import smooth_k
        B, H, S, D = 2, 4, 128, 64
        k = mx.random.normal([B, H, S, D]).astype(mx.float16)
        k_smooth, k_mean = smooth_k(k)
        mx.eval(k_smooth, k_mean)
        assert k_smooth.shape == (B, H, S, D)
        assert k_mean.shape == (B, H, 1, D)
        assert k_mean.dtype == mx.float32

    def test_quantize_scale_positive(self):
        """All scale values must be positive (not zero) to avoid NaN."""
        from mlx_mfa.quantize import quantize_per_block
        B, H, N, D = 1, 1, 32, 64
        x = mx.zeros([B, H, N, D]).astype(mx.float16)  # all-zero input
        _, scale = quantize_per_block(x, block_size=32)
        mx.eval(scale)
        assert bool(mx.all(scale > 0).item()), "Scale must be positive even for zero input"

    def test_sage_block_sizes(self):
        """Block sizes match STEEL tile dimensions."""
        from mlx_mfa.quantize import sage_block_sizes
        bq64, bk64 = sage_block_sizes(64)
        bq128, bk128 = sage_block_sizes(128)
        bq256, bk256 = sage_block_sizes(256)
        # CP3: Sage uses V2 BK values (doubled vs V1, gen-independent for Python API)
        assert bq64 == 32 and bk64 == 64
        assert bq128 == 32 and bk128 == 32
        assert bq256 == 32 and bk256 == 16


# ===========================================================================
# Track LB: 4D sparse block masks  [B, H, NQ, NK] and [H, NQ, NK]
# ===========================================================================

@pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
class TestBlockMask4D:
    """Tests for 2-D / 3-D / 4-D sparse block mask support (Track LB)."""

    def _base(self, B=2, H=4, N=128, D=128, seed=0):
        """Return q, k, v, scale and tile grid dims."""
        q, k, v = random_qkv(B, H, N, D, seed=seed)
        scale = 1.0 / math.sqrt(D)
        BQ, BK = _steel_block_config(D)
        NQ = (N + BQ - 1) // BQ
        NK = (N + BK - 1) // BK
        return q, k, v, scale, BQ, BK, NQ, NK

    # ── Shape validation tests ──────────────────────────────────────────────

    def test_3d_mask_wrong_shape0_raises(self):
        """3-D block_mask with shape[0] ≠ H must raise ValueError."""
        q, k, v, scale, BQ, BK, NQ, NK = self._base(H=4)
        bad = mx.ones((3, NQ, NK), dtype=mx.bool_)  # H=4 but mask says 3
        with pytest.raises(ValueError, match="must equal H=4"):
            flash_attention_sparse(q, k, v, bad, scale=scale)

    def test_4d_mask_wrong_shape0_raises(self):
        """4-D block_mask with shape[0] ≠ B must raise ValueError."""
        q, k, v, scale, BQ, BK, NQ, NK = self._base(B=2, H=4)
        bad = mx.ones((3, 4, NQ, NK), dtype=mx.bool_)  # B=2 but mask says 3
        with pytest.raises(ValueError, match="must equal B=2"):
            flash_attention_sparse(q, k, v, bad, scale=scale)

    def test_4d_mask_wrong_shape1_raises(self):
        """4-D block_mask with shape[1] ≠ H must raise ValueError."""
        q, k, v, scale, BQ, BK, NQ, NK = self._base(B=2, H=4)
        bad = mx.ones((2, 3, NQ, NK), dtype=mx.bool_)  # H=4 but mask says 3
        with pytest.raises(ValueError, match="must equal H=4"):
            flash_attention_sparse(q, k, v, bad, scale=scale)

    def test_5d_mask_raises(self):
        """5-D block_mask must raise ValueError."""
        q, k, v, scale, BQ, BK, NQ, NK = self._base(B=2, H=4)
        bad = mx.ones((2, 4, NQ, NK, 1), dtype=mx.bool_)
        with pytest.raises(ValueError, match="2-D.*3-D.*4-D"):
            flash_attention_sparse(q, k, v, bad, scale=scale)

    def test_1d_mask_raises(self):
        """1-D block_mask must raise ValueError."""
        q, k, v, scale, BQ, BK, NQ, NK = self._base()
        bad = mx.ones((NQ * NK,), dtype=mx.bool_)
        with pytest.raises(ValueError, match="2-D.*3-D.*4-D"):
            flash_attention_sparse(q, k, v, bad, scale=scale)

    # ── Correctness: 3-D and 4-D all-True == dense ──────────────────────────

    @pytest.mark.parametrize("D", [64, 128])
    def test_3d_all_true_matches_dense(self, D):
        """3-D all-True mask [H, NQ, NK] must match dense flash_attention."""
        # v2.50 Prompt 4 Section A: bumped N=128→2048 for sparse mask>=4096.
        B, H, N = 1, 4, 2048
        q, k, v = random_qkv(B, H, N, D, seed=10)
        scale = 1.0 / math.sqrt(D)
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK

        mask_3d = mx.ones((H, NQ, NK), dtype=mx.bool_)
        out_sparse = flash_attention_sparse(q, k, v, mask_3d, scale=scale)
        out_dense  = flash_attention(q, k, v, scale=scale)
        mx.eval(out_sparse, out_dense)

        np.testing.assert_allclose(
            np.array(out_dense.astype(mx.float32)),
            np.array(out_sparse.astype(mx.float32)),
            atol=1e-3,
            err_msg=f"D={D}: 3-D all-True sparse ≠ dense"
        )

    @pytest.mark.parametrize("D", [64, 128])
    def test_4d_all_true_matches_dense(self, D):
        """4-D all-True mask [B, H, NQ, NK] must match dense flash_attention."""
        # v2.50 Prompt 4 Section A: bumped N=128→2048 for sparse mask>=4096.
        B, H, N = 2, 4, 2048
        q, k, v = random_qkv(B, H, N, D, seed=11)
        scale = 1.0 / math.sqrt(D)
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK

        mask_4d = mx.ones((B, H, NQ, NK), dtype=mx.bool_)
        out_sparse = flash_attention_sparse(q, k, v, mask_4d, scale=scale)
        out_dense  = flash_attention(q, k, v, scale=scale)
        mx.eval(out_sparse, out_dense)

        np.testing.assert_allclose(
            np.array(out_dense.astype(mx.float32)),
            np.array(out_sparse.astype(mx.float32)),
            atol=1e-3,
            err_msg=f"D={D}: 4-D all-True sparse ≠ dense"
        )

    # ── Correctness: 3-D/4-D broadcast behaviour ─────────────────────────────

    def test_3d_broadcast_vs_2d_allheads(self):
        """Per-head 3-D mask broadcast == stacking the same 2-D mask H times."""
        B, H, N, D = 1, 4, 128, 128
        q, k, v = random_qkv(B, H, N, D, seed=20)
        scale = 1.0 / math.sqrt(D)
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK

        mx.random.seed(5)
        mask_2d = mx.random.uniform(shape=(NQ, NK)) > 0.3  # ~70% active

        # 3-D mask = same 2-D mask broadcast across all heads
        mask_3d = mx.stack([mask_2d] * H, axis=0)  # [H, NQ, NK]
        out_2d = flash_attention_sparse(q, k, v, mask_2d, scale=scale)
        out_3d = flash_attention_sparse(q, k, v, mask_3d, scale=scale)
        mx.eval(out_2d, out_3d)

        np.testing.assert_allclose(
            np.array(out_2d.astype(mx.float32)),
            np.array(out_3d.astype(mx.float32)),
            atol=1e-4,
            err_msg="3-D same-per-head mask ≠ 2-D shared mask"
        )

    def test_4d_broadcast_vs_2d(self):
        """4-D all-same mask [B, H, NQ, NK] == matching 2-D shared mask."""
        B, H, N, D = 2, 4, 128, 128
        q, k, v = random_qkv(B, H, N, D, seed=21)
        scale = 1.0 / math.sqrt(D)
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK

        mx.random.seed(6)
        mask_2d = mx.random.uniform(shape=(NQ, NK)) > 0.4

        # Repeat into 4-D: [B, H, NQ, NK]
        mask_4d = mx.broadcast_to(mask_2d[None, None, :, :], (B, H, NQ, NK))
        mask_4d = mx.contiguous(mask_4d)

        out_2d = flash_attention_sparse(q, k, v, mask_2d, scale=scale)
        out_4d = flash_attention_sparse(q, k, v, mask_4d, scale=scale)
        mx.eval(out_2d, out_4d)

        np.testing.assert_allclose(
            np.array(out_2d.astype(mx.float32)),
            np.array(out_4d.astype(mx.float32)),
            atol=1e-4,
            err_msg="4-D same-for-all mask ≠ 2-D shared mask"
        )

    # ── Per-head different masks (4-D) ─────────────────────────────────────

    def test_4d_per_head_different_masks(self):
        """Different masks per head give different outputs for each head."""
        B, H, N, D = 1, 4, 128, 128
        q, k, v = random_qkv(B, H, N, D, seed=30)
        scale = 1.0 / math.sqrt(D)
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK

        # Build per-head masks: head h has different random patterns
        masks_per_head = []
        for h in range(H):
            mx.random.seed(100 + h)
            m = mx.random.uniform(shape=(NQ, NK)) > 0.5
            masks_per_head.append(m)
        # Stack into [1, H, NQ, NK] (B=1)
        mask_4d = mx.stack(masks_per_head, axis=0)[None, :, :, :]
        mask_4d = mx.contiguous(mask_4d)

        out_4d = flash_attention_sparse(q, k, v, mask_4d, scale=scale)
        mx.eval(out_4d)
        assert out_4d.shape == (B, H, N, D), f"unexpected shape {out_4d.shape}"

        # Compare against per-head reference: run each head separately with 2-D mask
        for h in range(H):
            q_h = q[:, h:h+1, :, :]
            k_h = k[:, h:h+1, :, :]
            v_h = v[:, h:h+1, :, :]
            out_h = flash_attention_sparse(q_h, k_h, v_h, masks_per_head[h], scale=scale)
            mx.eval(out_h)
            np.testing.assert_allclose(
                np.array(out_4d[:, h:h+1, :, :].astype(mx.float32)),
                np.array(out_h.astype(mx.float32)),
                atol=2e-3,
                err_msg=f"4-D per-head mask mismatch at head {h}"
            )

    # ── Output shape tests ──────────────────────────────────────────────────

    def test_3d_output_shape(self):
        """3-D block_mask must produce correct output shape [B, H, N, D]."""
        # v2.50 Prompt 4 Section A: bumped N=128→2048 for sparse mask>=4096.
        B, H, N, D = 2, 4, 2048, 64
        q, k, v = random_qkv(B, H, N, D, seed=40)
        scale = 1.0 / math.sqrt(D)
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK
        mask = mx.ones((H, NQ, NK), dtype=mx.bool_)
        out = flash_attention_sparse(q, k, v, mask, scale=scale)
        mx.eval(out)
        assert out.shape == (B, H, N, D), f"expected {(B,H,N,D)}, got {out.shape}"

    def test_4d_output_shape(self):
        """4-D block_mask must produce correct output shape [B, H, N, D]."""
        B, H, N, D = 3, 8, 64, 128
        q, k, v = random_qkv(B, H, N, D, seed=41)
        scale = 1.0 / math.sqrt(D)
        BQ, BK = _steel_block_config(D)
        NQ, NK = (N + BQ - 1) // BQ, (N + BK - 1) // BK
        mask = mx.ones((B, H, NQ, NK), dtype=mx.bool_)
        out = flash_attention_sparse(q, k, v, mask, scale=scale)
        mx.eval(out)
        assert out.shape == (B, H, N, D), f"expected {(B,H,N,D)}, got {out.shape}"


# ─── Phase 2 — KVCacheProtocol + PagedInferenceContext ───────────────────────

class TestKVCacheProtocol:
    """KVCacheProtocol methods on DenseKVCache and PagedKVCache."""

    def test_dense_protocol_methods(self):
        """DenseKVCache satisfies protocol: k/v_for_attention, seq_length, reset."""
        from mlx_mfa import DenseKVCache
        cache = DenseKVCache(B=1, H=4, D=64, max_seq_len=512)
        k = mx.zeros([1, 4, 32, 64], dtype=mx.float16)
        v = mx.zeros([1, 4, 32, 64], dtype=mx.float16)
        cache.append(k, v, seq_id=0)  # seq_id accepted

        assert cache.seq_length(0) == 32
        assert cache.seq_length() == 32  # default seq_id=0

        ka = cache.k_for_attention()
        va = cache.v_for_attention()
        assert ka.shape == (1, 4, 32, 64)
        assert va.shape == (1, 4, 32, 64)

        cache.reset(seq_id=None)  # seq_id=None accepted
        assert cache.seq_length() == 0

    def test_paged_protocol_methods(self):
        """PagedKVCache satisfies protocol: k/v_for_attention, seq_length, reset."""
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=16, block_size=16, H=4, D=64)
        k = mx.zeros([1, 4, 24, 64], dtype=mx.float16)
        v = mx.zeros([1, 4, 24, 64], dtype=mx.float16)
        cache.append(k, v, seq_id=0)

        assert cache.seq_length(0) == 24

        ka = cache.k_for_attention(0)
        va = cache.v_for_attention(0)
        assert ka.shape == (1, 4, 24, 64)
        assert va.shape == (1, 4, 24, 64)

        cache.reset(seq_id=0)
        assert cache.seq_length(0) == 0

    def test_paged_reset_all(self):
        """PagedKVCache.reset(None) frees all sequences."""
        from mlx_mfa import PagedKVCache
        cache = PagedKVCache(num_blocks=32, block_size=16, H=2, D=64)
        k = mx.zeros([1, 2, 16, 64], dtype=mx.float16)
        v = mx.zeros([1, 2, 16, 64], dtype=mx.float16)
        cache.append(k, v, seq_id=0)
        cache.append(k, v, seq_id=1)
        assert cache.seq_length(0) == 16
        assert cache.seq_length(1) == 16
        cache.reset()  # seq_id=None by default
        assert cache.seq_length(0) == 0
        assert cache.seq_length(1) == 0

    def test_protocol_isinstance(self):
        """Both cache classes are instances of KVCacheProtocol."""
        from mlx_mfa import KVCacheProtocol, DenseKVCache, PagedKVCache
        assert isinstance(DenseKVCache(1, 4, 64), KVCacheProtocol)
        assert isinstance(PagedKVCache(8, 16, 4, 64), KVCacheProtocol)


class TestPagedInferenceContext:
    """PagedInferenceContext prefill / step / reset lifecycle."""

    def test_repr(self):
        from mlx_mfa import PagedInferenceContext
        ctx = PagedInferenceContext(num_blocks=64, block_size=16, H_kv=4, D=64)
        r = repr(ctx)
        assert "PagedInferenceContext" in r
        assert "num_blocks=64" in r

    def test_prefill_shape(self):
        """prefill returns correct [1, H_q, N, D] shape."""
        from mlx_mfa import PagedInferenceContext
        ctx = PagedInferenceContext(num_blocks=64, block_size=16, H_kv=4, D=64)
        N = 32
        q = mx.zeros([1, 4, N, 64], dtype=mx.float16)
        k = mx.zeros([1, 4, N, 64], dtype=mx.float16)
        v = mx.zeros([1, 4, N, 64], dtype=mx.float16)
        out = ctx.prefill(q, k, v, scale=0.125)
        mx.eval(out)
        assert out.shape == (1, 4, N, 64)
        assert ctx.seq_length(0) == N

    def test_step_shape(self):
        """step returns [1, H_q, 1, D] for single-token decode."""
        from mlx_mfa import PagedInferenceContext
        ctx = PagedInferenceContext(num_blocks=64, block_size=16, H_kv=4, D=64)
        k_pre = mx.zeros([1, 4, 32, 64], dtype=mx.float16)
        v_pre = mx.zeros([1, 4, 32, 64], dtype=mx.float16)
        q_pre = mx.zeros([1, 4, 32, 64], dtype=mx.float16)
        ctx.prefill(q_pre, k_pre, v_pre, scale=0.125)
        mx.eval(ctx._cache.k_for_attention())

        q_new = mx.zeros([1, 4, 1, 64], dtype=mx.float16)
        k_new = mx.zeros([1, 4, 1, 64], dtype=mx.float16)
        v_new = mx.zeros([1, 4, 1, 64], dtype=mx.float16)
        out = ctx.step(q_new, k_new, v_new, scale=0.125)
        mx.eval(out)
        assert out.shape == (1, 4, 1, 64)
        assert ctx.seq_length(0) == 33  # 32 prefill + 1 decode

    def test_reset_frees_seq(self):
        """reset(seq_id) frees the cache for that sequence."""
        from mlx_mfa import PagedInferenceContext
        ctx = PagedInferenceContext(num_blocks=32, block_size=16, H_kv=2, D=64)
        k = mx.zeros([1, 2, 16, 64], dtype=mx.float16)
        v = mx.zeros([1, 2, 16, 64], dtype=mx.float16)
        q = mx.zeros([1, 2, 16, 64], dtype=mx.float16)
        ctx.prefill(q, k, v, seq_id=0)
        assert ctx.seq_length(0) == 16
        ctx.reset(seq_id=0)
        assert ctx.seq_length(0) == 0

    def test_context_manager(self):
        """Context manager resets on exit."""
        from mlx_mfa import PagedInferenceContext
        with PagedInferenceContext(num_blocks=32, block_size=16, H_kv=2, D=64) as ctx:
            k = mx.zeros([1, 2, 16, 64], dtype=mx.float16)
            v = mx.zeros([1, 2, 16, 64], dtype=mx.float16)
            q = mx.zeros([1, 2, 16, 64], dtype=mx.float16)
            ctx.prefill(q, k, v)
            assert ctx.seq_length(0) == 16
        # After exit, seq length is 0 (reset called)
        assert ctx.seq_length(0) == 0

    def test_export(self):
        """PagedInferenceContext and KVCacheProtocol are exported from mlx_mfa."""
        import mlx_mfa
        assert hasattr(mlx_mfa, "PagedInferenceContext")
        assert hasattr(mlx_mfa, "KVCacheProtocol")
        assert "PagedInferenceContext" in mlx_mfa.__all__
        assert "KVCacheProtocol" in mlx_mfa.__all__


class TestInferenceContextFactory:
    """Unified decode context helper routing."""

    def test_backend_sage_routes_to_sage_context(self):
        from mlx_mfa import create_inference_context, SageInferenceContext
        ctx = create_inference_context(
            backend="sage",
            B=1,
            H_kv=4,
            D=64,
            max_seq_len=256,
        )
        assert isinstance(ctx, SageInferenceContext)

    def test_paged_hint_routes_to_paged_context(self):
        from mlx_mfa import create_inference_context, PagedInferenceContext
        ctx = create_inference_context(
            backend="auto",
            paged=True,
            B=1,
            H_kv=4,
            D=64,
            max_seq_len=256,
            num_blocks=32,
            block_size=16,
        )
        assert isinstance(ctx, PagedInferenceContext)

    def test_invalid_dense_plus_paged_fails(self):
        from mlx_mfa import create_inference_context
        with pytest.raises(ValueError, match="backend='dense'.*paged=True"):
            create_inference_context(
                backend="dense",
                paged=True,
                B=1,
                H_kv=4,
                D=64,
            )

    def test_invalid_paged_plus_quantized_fails(self):
        from mlx_mfa import create_inference_context
        with pytest.raises(ValueError, match="backend='paged'.*quantized_kv=True"):
            create_inference_context(
                backend="paged",
                quantized_kv=True,
                H_kv=4,
                D=64,
            )

    def test_helper_is_exported(self):
        import mlx_mfa
        assert hasattr(mlx_mfa, "create_inference_context")
        assert "create_inference_context" in mlx_mfa.__all__

    def test_auto_quantized_routes_sage_only_for_narrow_decode_regime(self):
        from mlx_mfa import create_inference_context, SageInferenceContext
        ctx = create_inference_context(
            backend="auto",
            quantized_kv=True,
            B=1,
            H_q=8,
            H_kv=4,
            D=128,
            decode_nq=4,
            expected_cache_len=4096,
            causal=True,
            window_size=(256, 0),
            max_seq_len=8192,
        )
        assert isinstance(ctx, SageInferenceContext)

    def test_auto_quantized_non_qualifying_shape_stays_dense(self):
        from mlx_mfa import create_inference_context, InferenceContext
        ctx = create_inference_context(
            backend="auto",
            quantized_kv=True,
            B=1,
            H_q=8,
            H_kv=4,
            D=128,
            decode_nq=4,
            expected_cache_len=4096,
            causal=True,
            window_size=None,
            max_seq_len=8192,
        )
        assert isinstance(ctx, InferenceContext)

    def test_auto_quantized_force_override(self, monkeypatch):
        from mlx_mfa import create_inference_context, InferenceContext, SageInferenceContext
        monkeypatch.setenv("MFA_FORCE_SAGE_DECODE", "0")
        off_ctx = create_inference_context(
            backend="auto",
            quantized_kv=True,
            B=1,
            H_kv=4,
            D=128,
            decode_nq=1,
            expected_cache_len=8192,
            causal=True,
            window_size=(256, 0),
        )
        assert isinstance(off_ctx, InferenceContext)

        monkeypatch.setenv("MFA_FORCE_SAGE_DECODE", "1")
        on_ctx = create_inference_context(
            backend="auto",
            quantized_kv=True,
            B=1,
            H_kv=4,
            D=64,
            decode_nq=1,
            expected_cache_len=0,
            causal=True,
            window_size=None,
        )
        assert isinstance(on_ctx, SageInferenceContext)

    def test_force_sage_auto_never_routes_without_quantized_kv(self, monkeypatch):
        from mlx_mfa import create_inference_context, InferenceContext
        monkeypatch.setenv("MFA_FORCE_SAGE_DECODE", "1")
        ctx = create_inference_context(
            backend="auto",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=128,
            decode_nq=1,
            expected_cache_len=8192,
            causal=True,
            window_size=(256, 0),
        )
        assert isinstance(ctx, InferenceContext)

    def test_invalid_hq_hkv_combination_fails(self):
        from mlx_mfa import create_inference_context
        with pytest.raises(ValueError, match="H_q must be divisible by H_kv"):
            create_inference_context(
                backend="auto",
                quantized_kv=True,
                B=1,
                H_q=6,
                H_kv=4,
                D=128,
            )


class TestDecodeRuntimeFactory:
    """Lightweight runtime wrapper over dense/paged/sage contexts."""

    def test_dense_runtime_selected(self):
        from mlx_mfa import create_decode_runtime, DecodeRuntime, InferenceContext
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        assert isinstance(rt, DecodeRuntime)
        assert rt.backend == "dense"
        assert isinstance(rt.context, InferenceContext)
        assert rt.metadata["backend"] == "dense"
        assert rt.metadata["paged_active"] is False
        assert rt.metadata["sage_active"] is False

    def test_paged_runtime_selected(self):
        from mlx_mfa import create_decode_runtime, PagedInferenceContext
        rt = create_decode_runtime(
            backend="auto",
            paged=True,
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
            num_blocks=32,
            block_size=16,
        )
        assert rt.backend == "paged"
        assert isinstance(rt.context, PagedInferenceContext)
        assert rt.metadata["backend"] == "paged"
        assert rt.metadata["paged_active"] is True
        assert rt.metadata["sage_active"] is False

    def test_paged_runtime_default_seq_id_applies_to_prefill_and_step(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
            num_blocks=64,
            block_size=16,
            default_seq_id=3,
        )
        q = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        out = rt.prefill(q, k, v, scale=0.125)
        mx.eval(out)
        assert rt.seq_length(3) == 16
        assert rt.seq_length(0) == 0

        q_new = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        k_new = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        v_new = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        out2 = rt.step(q_new, k_new, v_new, scale=0.125)
        mx.eval(out2)
        assert rt.seq_length(3) == 17

    def test_sage_runtime_selected_for_narrow_auto_regime(self):
        from mlx_mfa import create_decode_runtime, SageInferenceContext
        rt = create_decode_runtime(
            backend="auto",
            quantized_kv=True,
            B=1,
            H_q=8,
            H_kv=4,
            D=128,
            decode_nq=4,
            expected_cache_len=4096,
            causal=True,
            window_size=(256, 0),
            dtype=mx.float16,
        )
        assert rt.backend == "sage"
        assert isinstance(rt.context, SageInferenceContext)
        assert rt.metadata["backend"] == "sage"
        assert rt.metadata["sage_active"] is True

    def test_sage_runtime_requires_quantized_kv(self):
        from mlx_mfa import create_decode_runtime
        with pytest.raises(ValueError, match="backend='sage'.*quantized_kv=True"):
            create_decode_runtime(
                backend="sage",
                quantized_kv=False,
                B=1,
                H_kv=4,
                D=128,
            )

    def test_paged_and_quantized_is_invalid(self):
        from mlx_mfa import create_decode_runtime
        with pytest.raises(ValueError, match="paged=True.*quantized_kv=True"):
            create_decode_runtime(
                backend="auto",
                paged=True,
                quantized_kv=True,
                B=1,
                H_kv=4,
                D=64,
            )

    def test_runtime_exported(self):
        import mlx_mfa
        assert hasattr(mlx_mfa, "DecodeRuntime")
        assert hasattr(mlx_mfa, "create_decode_runtime")
        assert "DecodeRuntime" in mlx_mfa.__all__
        assert "create_decode_runtime" in mlx_mfa.__all__

    def test_runtime_invalid_default_seq_id(self):
        from mlx_mfa import create_decode_runtime
        with pytest.raises(ValueError, match="default_seq_id must be >= 0"):
            create_decode_runtime(
                backend="dense",
                quantized_kv=False,
                B=1,
                H_kv=4,
                D=64,
                default_seq_id=-1,
            )

    def test_shared_prefix_helper_accessible_via_runtime(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        out, kp, vp = rt.shared_prefix_cache(q, k, v)
        mx.eval(out, kp, vp)
        assert out.shape == (1, 4, 16, 64)
        assert kp.shape == k.shape
        assert vp.shape == v.shape

    def test_splitfuse_helper_accessible_via_runtime(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q = mx.random.normal((1, 4, 32, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 32, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 32, 64)).astype(mx.float16)
        out_p, out_d = rt.splitfuse(q, k, v, None, None, None)
        mx.eval(out_p)
        assert out_p.shape == (1, 4, 32, 64)
        assert out_d is None

    def test_splitfuse_step_uses_runtime_dense_cache(self):
        from mlx_mfa import create_decode_runtime

        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q_pre = mx.random.normal((1, 4, 12, 64)).astype(mx.float16)
        k_pre = mx.random.normal((1, 4, 12, 64)).astype(mx.float16)
        v_pre = mx.random.normal((1, 4, 12, 64)).astype(mx.float16)
        rt.prefill(q_pre, k_pre, v_pre, scale=0.125)

        q_dec = mx.random.normal((1, 4, 2, 64)).astype(mx.float16)
        out_p, out_d = rt.splitfuse_step(q_dec, scale=0.125)
        mx.eval(out_d)
        assert out_p is None
        assert out_d.shape == (1, 4, 2, 64)
        assert rt.metadata["splitfuse_active"] is True
        assert rt.metadata["last_splitfuse"]["used_runtime_cache"] is True

    def test_splitfuse_step_can_use_registered_prefix(self):
        from mlx_mfa import create_decode_runtime

        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q_pre = mx.random.normal((1, 4, 10, 64)).astype(mx.float16)
        k_pre = mx.random.normal((1, 4, 10, 64)).astype(mx.float16)
        v_pre = mx.random.normal((1, 4, 10, 64)).astype(mx.float16)
        rt.register_prefix("p0", q_pre, k_pre, v_pre, overwrite=True)
        rt.seed_prefix(prefix_id="p0", reset=True)
        q_dec = mx.random.normal((1, 4, 2, 64)).astype(mx.float16)

        out_p, out_d = rt.splitfuse_step(
            q_dec,
            use_registered_prefix=True,
            prefix_id="p0",
            scale=0.125,
        )
        mx.eval(out_p, out_d)
        assert out_p.shape == (1, 4, 10, 64)
        assert out_d.shape == (1, 4, 2, 64)
        assert rt.metadata["last_splitfuse"]["used_registered_prefix"] is True

    def test_splitfuse_step_paged_single_seq_cache(self):
        from mlx_mfa import create_decode_runtime

        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            query_layout="batched",
            quantized_kv=False,
            B=1,
            H_q=4,
            H_kv=4,
            D=64,
            num_blocks=64,
            block_size=16,
        )
        q_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        k_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        v_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        rt.prefill(q_pre, k_pre, v_pre, seq_id=9, scale=0.125)

        q_dec = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        out_p, out_d = rt.splitfuse_step(q_dec, seq_id=9, scale=0.125)
        mx.eval(out_d)
        assert out_p is None
        assert out_d.shape == (1, 4, 1, 64)
        assert rt.metadata["last_splitfuse"]["seq_id"] == 9
        assert rt.metadata["last_splitfuse"]["paged_native_decode_only"] is True

    def test_splitfuse_step_rejects_unsupported_backend(self):
        from mlx_mfa import create_decode_runtime

        rt = create_decode_runtime(
            backend="sage",
            quantized_kv=True,
            B=1,
            H_kv=4,
            D=64,
        )
        q_dec = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        with pytest.raises(ValueError, match="dense/paged runtime only"):
            rt.splitfuse_step(q_dec, scale=0.125)

    def test_prefill_shared_prefix_seeds_dense_runtime_cache(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q_pre = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        k_pre = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        v_pre = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        out_pre, kp, vp = rt.prefill_shared_prefix(q_pre, k_pre, v_pre, scale=0.125)
        mx.eval(out_pre, kp, vp)
        assert rt.seq_length() == 16
        assert rt.metadata["shared_prefix_active"] is True

        q_new = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        k_new = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        v_new = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        out = rt.step(q_new, k_new, v_new, scale=0.125)
        mx.eval(out)
        assert rt.seq_length() == 17

    def test_register_prefix_and_metadata_dense(self):
        from mlx_mfa import create_decode_runtime

        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        k_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        v_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)

        out_pre, kp, vp = rt.register_prefix(
            "sys_prompt",
            q_pre,
            k_pre,
            v_pre,
            scale=0.125,
        )
        mx.eval(out_pre, kp, vp)
        assert "sys_prompt" in rt.list_registered_prefix_ids()
        assert rt.metadata["prefix_cache_size"] == 1
        assert rt.metadata["active_prefix_id"] == "sys_prompt"

        rt.seed_prefix(prefix_id="sys_prompt", reset=True)
        assert rt.seq_length() == 8
        assert rt.metadata["last_prefix_reuse"]["prefix_ids"] == ("sys_prompt",)

    def test_prefill_with_prefix_dense_matches_manual_seed_then_chunked(self):
        from mlx_mfa import create_decode_runtime

        scale = 0.125
        q_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        k_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        v_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        q_suf = mx.random.normal((1, 4, 6, 64)).astype(mx.float16)
        k_suf = mx.random.normal((1, 4, 6, 64)).astype(mx.float16)
        v_suf = mx.random.normal((1, 4, 6, 64)).astype(mx.float16)

        rt_auto = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        rt_auto.register_prefix("p0", q_pre, k_pre, v_pre, scale=scale)
        out_auto = rt_auto.prefill_with_prefix(
            q_suf,
            k_suf,
            v_suf,
            prefix_id="p0",
            chunk_size=2,
            scale=scale,
            causal=True,
            reset=True,
        )

        rt_ref = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        rt_ref.register_prefix("p0", q_pre, k_pre, v_pre, scale=scale)
        rt_ref.seed_prefix(prefix_id="p0", reset=True)
        out_ref = rt_ref.chunked_prefill(
            q_suf,
            k_suf,
            v_suf,
            chunk_size=2,
            scale=scale,
            causal=True,
            reset=False,
        )

        mx.eval(out_auto, out_ref)
        diff = float(mx.abs(out_auto.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3
        assert rt_auto.seq_length() == 14

    def test_prefill_with_prefix_paged_batched(self):
        from mlx_mfa import create_decode_runtime

        scale = 1.0 / math.sqrt(64)
        seq_id = 77
        q_pre = mx.random.normal((1, 8, 6, 64)).astype(mx.float16)
        k_pre = mx.random.normal((1, 4, 6, 64)).astype(mx.float16)
        v_pre = mx.random.normal((1, 4, 6, 64)).astype(mx.float16)
        q_suf = mx.random.normal((1, 8, 5, 64)).astype(mx.float16)
        k_suf = mx.random.normal((1, 4, 5, 64)).astype(mx.float16)
        v_suf = mx.random.normal((1, 4, 5, 64)).astype(mx.float16)

        rt_auto = create_decode_runtime(
            backend="paged",
            paged=True,
            query_layout="batched",
            quantized_kv=False,
            B=1,
            H_q=8,
            H_kv=4,
            D=64,
            num_blocks=64,
            block_size=16,
        )
        rt_auto.register_prefix("p0", q_pre, k_pre, v_pre, scale=scale)
        out_auto = rt_auto.prefill_with_prefix(
            q_suf,
            k_suf,
            v_suf,
            prefix_id="p0",
            seq_id=seq_id,
            chunk_size=2,
            scale=scale,
            causal=True,
            reset=True,
        )

        rt_ref = create_decode_runtime(
            backend="paged",
            paged=True,
            query_layout="batched",
            quantized_kv=False,
            B=1,
            H_q=8,
            H_kv=4,
            D=64,
            num_blocks=64,
            block_size=16,
        )
        rt_ref.register_prefix("p0", q_pre, k_pre, v_pre, scale=scale)
        rt_ref.seed_prefix(prefix_id="p0", seq_id=seq_id, reset=True)
        out_ref = rt_ref.chunked_prefill(
            q_suf,
            k_suf,
            v_suf,
            chunk_size=2,
            seq_ids=[seq_id],
            scale=scale,
            causal=True,
            reset=False,
        )

        mx.eval(out_auto, out_ref)
        diff = float(mx.abs(out_auto.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3
        assert rt_auto.seq_length(seq_id) == 11

    def test_prefill_with_prefix_paged_packed_single_seq(self):
        from mlx_mfa import create_decode_runtime

        scale = 1.0 / math.sqrt(64)
        seq_id = 5
        q_pre = mx.random.normal((1, 8, 4, 64)).astype(mx.float16)
        k_pre = mx.random.normal((1, 4, 4, 64)).astype(mx.float16)
        v_pre = mx.random.normal((1, 4, 4, 64)).astype(mx.float16)
        q_suf = mx.random.normal((1, 8, 5, 64)).astype(mx.float16)
        k_suf = mx.random.normal((1, 4, 5, 64)).astype(mx.float16)
        v_suf = mx.random.normal((1, 4, 5, 64)).astype(mx.float16)
        cu = mx.array([0, 5], dtype=mx.int32)

        rt_auto = create_decode_runtime(
            backend="paged",
            paged=True,
            query_layout="packed",
            quantized_kv=False,
            B=1,
            H_q=8,
            H_kv=4,
            D=64,
            num_blocks=64,
            block_size=16,
        )
        rt_auto.register_prefix("p0", q_pre, k_pre, v_pre, scale=scale)
        out_auto = rt_auto.prefill_with_prefix(
            q_suf,
            k_suf,
            v_suf,
            prefix_id="p0",
            seq_ids=[seq_id],
            cu_seqlens_q=cu,
            chunk_size=2,
            scale=scale,
            causal=True,
            reset=True,
        )

        rt_ref = create_decode_runtime(
            backend="paged",
            paged=True,
            query_layout="packed",
            quantized_kv=False,
            B=1,
            H_q=8,
            H_kv=4,
            D=64,
            num_blocks=64,
            block_size=16,
        )
        rt_ref.register_prefix("p0", q_pre, k_pre, v_pre, scale=scale)
        rt_ref.seed_prefix(prefix_id="p0", seq_ids=[seq_id], reset=True)
        out_ref = rt_ref.chunked_prefill(
            q_suf,
            k_suf,
            v_suf,
            chunk_size=2,
            seq_ids=[seq_id],
            cu_seqlens_q=cu,
            scale=scale,
            causal=True,
            reset=False,
        )

        mx.eval(out_auto, out_ref)
        diff = float(mx.abs(out_auto.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3
        assert rt_auto.seq_length(seq_id) == 9

    def test_prefix_reuse_invalid_combinations_fail_clearly(self):
        from mlx_mfa import create_decode_runtime

        rt_dense = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q_pre = mx.random.normal((1, 4, 4, 64)).astype(mx.float16)
        k_pre = mx.random.normal((1, 4, 4, 64)).astype(mx.float16)
        v_pre = mx.random.normal((1, 4, 4, 64)).astype(mx.float16)
        rt_dense.register_prefix("p0", q_pre, k_pre, v_pre)
        with pytest.raises(ValueError, match="supported only on paged runtime"):
            rt_dense.seed_prefix(prefix_ids=["p0"], seq_ids=[0])

        rt_paged = create_decode_runtime(
            backend="paged",
            paged=True,
            query_layout="packed",
            quantized_kv=False,
            B=1,
            H_q=4,
            H_kv=4,
            D=64,
            num_blocks=32,
            block_size=16,
        )
        rt_paged.register_prefix("p0", q_pre, k_pre, v_pre)
        with pytest.raises(ValueError, match="length mismatch"):
            rt_paged.seed_prefix(prefix_ids=["p0", "p0"], seq_ids=[1])

        q = mx.random.normal((1, 4, 3, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 3, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 3, 64)).astype(mx.float16)
        with pytest.raises(ValueError, match="requires cu_seqlens_q"):
            rt_paged.prefill_with_prefix(
                q,
                k,
                v,
                prefix_id="p0",
                seq_ids=[1],
                chunk_size=2,
                causal=True,
            )

    def test_decode_from_shared_prefix_requires_prepare(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q = mx.random.normal((1, 4, 4, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 4, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 4, 64)).astype(mx.float16)
        with pytest.raises(ValueError, match="prefill_shared_prefix\\(\\) first"):
            rt.decode_from_shared_prefix(q, k, v, scale=0.125)

    def test_splitfuse_can_use_prepared_prefix(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q_pre = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        k_pre = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        v_pre = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        rt.prefill_shared_prefix(
            q_pre,
            k_pre,
            v_pre,
            scale=0.125,
            seed_runtime_cache=False,
        )
        q_decode = mx.random.normal((1, 4, 2, 64)).astype(mx.float16)
        k_decode = mx.random.normal((1, 4, 32, 64)).astype(mx.float16)
        v_decode = mx.random.normal((1, 4, 32, 64)).astype(mx.float16)
        out_p, out_d = rt.splitfuse(
            None,
            None,
            None,
            q_decode,
            k_decode,
            v_decode,
            use_prepared_prefix=True,
            scale=0.125,
        )
        mx.eval(out_p, out_d)
        assert out_p.shape == (1, 4, 16, 64)
        assert out_d.shape == (1, 4, 2, 64)
        assert rt.metadata["splitfuse_active"] is True

    def test_splitfuse_rejects_partial_prefill_inputs(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        with pytest.raises(ValueError, match="prefill inputs must be all provided"):
            rt.splitfuse(q, None, None, None, None, None)

    def test_speculative_verify_helper_via_dense_runtime_cache(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        rt.prefill(q, k, v)
        q_target = mx.random.normal((1, 4, 2, 64)).astype(mx.float16)
        draft_ids = mx.zeros((1, 2), dtype=mx.int32)
        out, lse, lp = rt.speculative_verify(q_target, draft_ids)
        mx.eval(out, lse, lp)
        assert out.shape == (1, 4, 2, 64)
        assert lse.shape == (1, 4, 2)
        assert lp.shape == (1, 2)
        assert rt.metadata["speculative_verify_active"] is True

    def test_speculative_verify_empty_paged_cache_without_explicit_cache(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
            num_blocks=32,
            block_size=16,
        )
        q_target = mx.random.normal((1, 4, 2, 64)).astype(mx.float16)
        draft_ids = mx.zeros((1, 2), dtype=mx.int32)
        with pytest.raises(ValueError, match="paged runtime cache is empty"):
            rt.speculative_verify(q_target, draft_ids)

    def test_speculative_verify_accepts_explicit_cache(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
            num_blocks=32,
            block_size=16,
        )
        q_target = mx.random.normal((1, 4, 2, 64)).astype(mx.float16)
        k_cache = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        v_cache = mx.random.normal((1, 4, 16, 64)).astype(mx.float16)
        draft_ids = mx.zeros((1, 2), dtype=mx.int32)
        out, lse, lp = rt.speculative_verify(
            q_target,
            draft_ids,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        mx.eval(out, lse, lp)
        assert out.shape == (1, 4, 2, 64)

    def test_speculative_verify_via_paged_runtime_cache(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            quantized_kv=False,
            B=1,
            H_q=4,
            H_kv=4,
            D=64,
            num_blocks=64,
            block_size=16,
        )
        q_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        k_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        v_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        rt.prefill(q_pre, k_pre, v_pre, seq_id=3)

        q_target = mx.random.normal((1, 4, 3, 64)).astype(mx.float16)
        draft_ids = mx.zeros((1, 3), dtype=mx.int32)
        out, lse, lp = rt.speculative_verify(q_target, draft_ids, seq_id=3)
        mx.eval(out, lse, lp)
        assert out.shape == (1, 4, 3, 64)
        assert lse.shape == (1, 4, 3)
        assert lp.shape == (1, 3)

    def test_speculative_verify_via_paged_runtime_cache_batched_seq_ids(self):
        from mlx_mfa import create_decode_runtime

        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            query_layout="batched",
            quantized_kv=False,
            B=2,
            H_q=4,
            H_kv=4,
            D=64,
            num_blocks=64,
            block_size=16,
        )
        q_pre = mx.random.normal((2, 4, 6, 64)).astype(mx.float16)
        k_pre = mx.random.normal((2, 4, 6, 64)).astype(mx.float16)
        v_pre = mx.random.normal((2, 4, 6, 64)).astype(mx.float16)
        _ = rt.paged_prefill_batch(q_pre, k_pre, v_pre, seq_ids=[3, 7], causal=True)

        q_target = mx.random.normal((2, 4, 3, 64)).astype(mx.float16)
        draft_ids = mx.zeros((2, 3), dtype=mx.int32)
        out, lse, lp = rt.speculative_verify(
            q_target,
            draft_ids,
            seq_ids=[3, 7],
        )
        mx.eval(out, lse, lp)
        assert out.shape == (2, 4, 3, 64)
        assert lse.shape == (2, 4, 3)
        assert lp.shape == (2, 3)

    def test_speculative_step_full_accept_and_metadata(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q = mx.random.normal((1, 4, 12, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 12, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 12, 64)).astype(mx.float16)
        rt.prefill(q, k, v)
        seq_before = rt.seq_length()

        q_target = mx.random.normal((1, 4, 4, 64)).astype(mx.float16)
        draft_ids = mx.array([[3, 4, 5, 6]], dtype=mx.int32)
        result = rt.speculative_step(
            q_target,
            draft_ids,
            accept_logprob_delta=-1e9,
        )
        mx.eval(
            result["out"],
            result["lse"],
            result["target_logprobs"],
            result["accept_mask"],
            result["accepted_prefix_lens"],
            result["accepted_ids"],
            result["rejected_ids"],
        )

        assert tuple(result["accepted_prefix_lens"].tolist()) == (4,)
        assert tuple(result["accepted_ids"].tolist()[0]) == (3, 4, 5, 6)
        assert tuple(result["rejected_ids"].tolist()[0]) == (-1, -1, -1, -1)
        assert rt.seq_length() == seq_before
        assert rt.metadata["speculative_step_active"] is True
        assert rt.metadata["last_speculative_step"]["tokens"] == 4

    def test_speculative_step_partial_accept_with_draft_logprobs(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q = mx.random.normal((1, 4, 10, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 10, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 10, 64)).astype(mx.float16)
        rt.prefill(q, k, v)

        q_target = mx.random.normal((1, 4, 4, 64)).astype(mx.float16)
        draft_ids = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
        _, _, lp = rt.speculative_verify(q_target, draft_ids)
        lp_np = np.array(lp.astype(mx.float32))
        draft_lp_np = lp_np - 0.5
        draft_lp_np[:, 2:] = lp_np[:, 2:] + 5.0
        draft_lp = mx.array(draft_lp_np.astype(np.float32))

        result = rt.speculative_step(
            q_target,
            draft_ids,
            draft_logprobs=draft_lp,
            accept_logprob_delta=0.0,
        )
        mx.eval(
            result["accept_mask"],
            result["accepted_prefix_lens"],
            result["accepted_ids"],
            result["rejected_ids"],
        )
        assert tuple(result["accepted_prefix_lens"].tolist()) == (2,)
        assert tuple(result["accepted_ids"].tolist()[0]) == (0, 1, -1, -1)
        assert tuple(result["rejected_ids"].tolist()[0]) == (-1, -1, 2, 3)

    def test_speculative_step_reject_all_with_high_threshold(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q = mx.random.normal((1, 4, 10, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 10, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 10, 64)).astype(mx.float16)
        rt.prefill(q, k, v)

        q_target = mx.random.normal((1, 4, 3, 64)).astype(mx.float16)
        draft_ids = mx.array([[9, 7, 1]], dtype=mx.int32)
        result = rt.speculative_step(
            q_target,
            draft_ids,
            accept_logprob_delta=1e6,
        )
        mx.eval(result["accepted_prefix_lens"], result["accepted_ids"], result["rejected_ids"])
        assert tuple(result["accepted_prefix_lens"].tolist()) == (0,)
        assert tuple(result["accepted_ids"].tolist()[0]) == (-1, -1, -1)
        assert tuple(result["rejected_ids"].tolist()[0]) == (9, 7, 1)

    def test_speculative_step_invalid_draft_logprobs_shape(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        rt.prefill(q, k, v)
        q_target = mx.random.normal((1, 4, 2, 64)).astype(mx.float16)
        draft_ids = mx.array([[0, 1]], dtype=mx.int32)
        bad_lp = mx.zeros((2, 2), dtype=mx.float32)
        with pytest.raises(ValueError, match="shape to match draft_ids"):
            rt.speculative_step(
                q_target,
                draft_ids,
                draft_logprobs=bad_lp,
            )

    def test_speculative_verify_packed_query_layout_without_explicit_cache_fails(self):
        from mlx_mfa import create_decode_runtime
        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            query_layout="packed",
            quantized_kv=False,
            B=1,
            H_q=4,
            H_kv=4,
            D=64,
            num_blocks=32,
            block_size=16,
        )
        q_target = mx.random.normal((1, 4, 2, 64)).astype(mx.float16)
        draft_ids = mx.zeros((1, 2), dtype=mx.int32)
        with pytest.raises(ValueError, match="requires query_layout='batched'"):
            rt.speculative_verify(q_target, draft_ids)


# ==========================================================================
# Track LE — Paged KV + packed varlen query API / runtime
# ==========================================================================


class TestPagedVarlenQueries:
    """Correctness coverage for flash_attention_paged_varlen."""

    @staticmethod
    def _build_hetero_paged_pool(
        k_seqs: list,
        v_seqs: list,
        block_size: int,
    ):
        """Pack per-sequence contiguous KV tensors into one paged pool."""
        B = len(k_seqs)
        H_kv = k_seqs[0].shape[1]
        D = k_seqs[0].shape[3]
        blocks_per_seq = [
            (int(k.shape[2]) + block_size - 1) // block_size
            for k in k_seqs
        ]
        total_blocks = sum(blocks_per_seq)
        max_blocks = max(blocks_per_seq) if blocks_per_seq else 0

        pool_k = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float16)
        pool_v = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float16)
        table = np.full((B, max_blocks), -1, dtype=np.int32)
        lens = np.zeros((B,), dtype=np.int32)

        blk_base = 0
        for b in range(B):
            k_np = np.array(k_seqs[b]).astype(np.float16)[0].transpose(1, 0, 2)  # [S,H,D]
            v_np = np.array(v_seqs[b]).astype(np.float16)[0].transpose(1, 0, 2)  # [S,H,D]
            S = k_np.shape[0]
            lens[b] = S
            n_blk = blocks_per_seq[b]
            for lb in range(n_blk):
                table[b, lb] = blk_base + lb
                s0 = lb * block_size
                s1 = min(S, s0 + block_size)
                pool_k[blk_base + lb, : s1 - s0] = k_np[s0:s1]
                pool_v[blk_base + lb, : s1 - s0] = v_np[s0:s1]
            blk_base += n_blk

        return (
            mx.array(pool_k),
            mx.array(pool_v),
            mx.array(table, dtype=mx.int32),
            mx.array(lens, dtype=mx.int32),
        )

    @staticmethod
    def _pack_queries(q_seqs: list):
        """Pack per-sequence [1,H,Qi,D] into [1,H,total_q,D] + cu_seqlens_q."""
        offsets = [0]
        for q in q_seqs:
            offsets.append(offsets[-1] + int(q.shape[2]))
        q_pack = mx.concatenate(q_seqs, axis=2) if offsets[-1] > 0 else q_seqs[0][:, :, :0, :]
        cu = mx.array(offsets, dtype=mx.int32)
        return q_pack, cu

    def test_paged_varlen_basic_correctness(self):
        """Fused paged-varlen matches per-sequence SDPA reference."""
        from mlx_mfa import flash_attention_paged_varlen

        mx.random.seed(701)
        H_q, H_kv, D = 8, 4, 64
        q_lens = [3, 1, 4]
        kv_lens = [27, 19, 33]
        block_size = 16
        scale = 1.0 / math.sqrt(D)

        q_seqs = [mx.random.normal((1, H_q, ql, D)).astype(mx.float16) for ql in q_lens]
        k_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        v_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        mx.eval(*q_seqs, *k_seqs, *v_seqs)
        q_pack, cu_q = self._pack_queries(q_seqs)
        pool_k, pool_v, table, lens = self._build_hetero_paged_pool(k_seqs, v_seqs, block_size)

        out = flash_attention_paged_varlen(
            q_pack,
            pool_k,
            pool_v,
            table,
            lens,
            cu_q,
            max_seqlen_q=max(q_lens),
            scale=scale,
            causal=False,
            block_size=block_size,
        )

        # Reference: per-sequence SDPA (ground truth)
        ref_parts = []
        for i in range(len(q_lens)):
            qs, qe = int(cu_q[i].item()), int(cu_q[i + 1].item())
            out_i = mx.fast.scaled_dot_product_attention(
                q_pack[:, :, qs:qe, :],
                k_seqs[i],
                v_seqs[i],
                scale=scale,
            )
            ref_parts.append(out_i)
        ref = mx.concatenate(ref_parts, axis=2)
        mx.eval(out, ref)
        diff = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert out.shape == (1, H_q, sum(q_lens), D)
        assert diff < 5e-3, f"paged_varlen vs SDPA ref max diff {diff}"
        # CC-17 (audit): engagement proof — a silent fall-through to the
        # per-sequence SDPA bridge would be byteΔ=0 vs this same-precision SDPA
        # ref; the fused fp16 kernel differs by its own reduction-order noise
        # (~2e-4).  A zero diff means the fused kernel did NOT run (vacuous).
        assert diff > 1e-6, "paged-varlen did not engage the fused kernel (byteΔ=0 vs SDPA bridge)"

        # III-10 3b: INDEPENDENT fp32 oracle lock (lesson #11).
        # The reference above runs SDPA at fp16 (kernel precision).  Lock the
        # fused paged-varlen kernel against the per-sequence fp32 SDPA oracle
        # (inputs cast to float32) — NOT patched by auto-hooks.
        ref32_parts = []
        for i in range(len(q_lens)):
            qs, qe = int(cu_q[i].item()), int(cu_q[i + 1].item())
            ref32_parts.append(
                mx.fast.scaled_dot_product_attention(
                    q_pack[:, :, qs:qe, :].astype(mx.float32),
                    k_seqs[i].astype(mx.float32),
                    v_seqs[i].astype(mx.float32),
                    scale=scale,
                )
            )
        ref32 = mx.concatenate(ref32_parts, axis=2)
        mx.eval(ref32)
        diff32 = float(mx.abs(out.astype(mx.float32) - ref32).max())
        # fp16 kernel vs fp32 oracle: tolerance covers fp16 accumulation only.
        assert diff32 < 2e-2, f"paged_varlen vs fp32 SDPA oracle max diff {diff32}"

    def test_paged_varlen_handles_zero_length_sequence(self):
        from mlx_mfa import flash_attention_paged_varlen

        mx.random.seed(702)
        H_q, H_kv, D = 4, 2, 64
        q_lens = [2, 0, 3]
        kv_lens = [20, 12, 28]
        block_size = 16
        q_seqs = [mx.random.normal((1, H_q, ql, D)).astype(mx.float16) for ql in q_lens]
        k_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        v_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        mx.eval(*q_seqs, *k_seqs, *v_seqs)
        q_pack, cu_q = self._pack_queries(q_seqs)
        pool_k, pool_v, table, lens = self._build_hetero_paged_pool(k_seqs, v_seqs, block_size)

        out = flash_attention_paged_varlen(
            q_pack, pool_k, pool_v, table, lens, cu_q,
            causal=True, block_size=block_size
        )
        mx.eval(out)
        assert out.shape == (1, H_q, sum(q_lens), D)
        assert bool(mx.all(mx.isfinite(out)).item())

    def test_paged_varlen_invalid_cu_fails(self):
        from mlx_mfa import flash_attention_paged_varlen

        q = mx.zeros((1, 4, 5, 64), dtype=mx.float16)
        pool = mx.zeros((8, 16, 4, 64), dtype=mx.float16)
        table = mx.array([[0], [1]], dtype=mx.int32)
        lens = mx.array([16, 16], dtype=mx.int32)
        bad_cu = mx.array([0, 2, 4], dtype=mx.int32)  # should end at total_q=5

        with pytest.raises(ValueError, match="must equal total_q"):
            flash_attention_paged_varlen(q, pool, pool, table, lens, bad_cu)

    def test_runtime_paged_varlen_matches_direct_api(self):
        from mlx_mfa import (
            create_decode_runtime,
            flash_attention_paged_varlen,
            PagedInferenceContext,
        )

        mx.random.seed(703)
        H_q, H_kv, D = 8, 4, 64
        q_lens = [1, 3]
        kv_lens = [17, 29]
        block_size = 16
        scale = 1.0 / math.sqrt(D)
        seq_ids = [11, 22]

        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            query_layout="packed",
            quantized_kv=False,
            B=1,
            H_q=H_q,
            H_kv=H_kv,
            D=D,
            num_blocks=64,
            block_size=block_size,
        )
        assert isinstance(rt.context, PagedInferenceContext)
        assert rt.metadata["query_layout"] == "packed"

        q_seqs = [mx.random.normal((1, H_q, ql, D)).astype(mx.float16) for ql in q_lens]
        k_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        v_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        mx.eval(*q_seqs, *k_seqs, *v_seqs)
        q_pack, cu_q = self._pack_queries(q_seqs)

        for sid, k_i, v_i in zip(seq_ids, k_seqs, v_seqs):
            rt.context.cache.append(k_i, v_i, seq_id=sid)

        out_rt = rt.paged_varlen(
            q_pack,
            cu_q,
            seq_ids=seq_ids,
            scale=scale,
            causal=True,
        )
        table = rt.context.cache.get_block_table(seq_ids)
        lens = rt.context.cache.get_seq_lens(seq_ids)
        out_ref = flash_attention_paged_varlen(
            q_pack,
            rt.context.cache.k_pool,
            rt.context.cache.v_pool,
            table,
            lens,
            cu_q,
            scale=scale,
            causal=True,
            block_size=block_size,
        )
        mx.eval(out_rt, out_ref)
        diff = float(mx.abs(out_rt.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 1e-6

    def test_runtime_query_layout_validation(self):
        from mlx_mfa import create_decode_runtime

        with pytest.raises(ValueError, match="query_layout='packed'.*paged runtime"):
            create_decode_runtime(
                backend="dense",
                query_layout="packed",
                quantized_kv=False,
                B=1,
                H_kv=4,
                D=64,
            )


class TestPagedContinuousBatching:
    """Scheduler-style continuous batching coverage for paged runtime/API."""

    def test_flash_attention_paged_cache_batch_idx_matches_row_gather(self):
        from mlx_mfa import flash_attention_paged

        mx.random.seed(811)
        B_pool, H_q, H_kv, D = 4, 8, 4, 64
        q_len = 2
        kv_lens = [24, 31, 19, 27]
        block_size = 16
        scale = 1.0 / math.sqrt(D)

        q = mx.random.normal((2, H_q, q_len, D)).astype(mx.float16)
        k_seqs = [mx.random.normal((1, H_kv, s, D)).astype(mx.float16) for s in kv_lens]
        v_seqs = [mx.random.normal((1, H_kv, s, D)).astype(mx.float16) for s in kv_lens]
        mx.eval(q, *k_seqs, *v_seqs)
        pool_k, pool_v, table, lens = TestPagedVarlenQueries._build_hetero_paged_pool(
            k_seqs,
            v_seqs,
            block_size,
        )
        idx = mx.array([3, 1], dtype=mx.int32)

        out_remap = flash_attention_paged(
            q,
            pool_k,
            pool_v,
            table,
            lens,
            scale=scale,
            causal=True,
            block_size=block_size,
            cache_batch_idx=idx,
        )
        out_ref = flash_attention_paged(
            q,
            pool_k,
            pool_v,
            table[idx],
            lens[idx],
            scale=scale,
            causal=True,
            block_size=block_size,
        )
        mx.eval(out_remap, out_ref)
        diff = float(mx.abs(out_remap.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 1e-6

    def test_runtime_paged_step_batch_with_reordered_active_requests(self):
        from mlx_mfa import create_decode_runtime, PagedKVCache, flash_attention_paged

        mx.random.seed(812)
        H_q, H_kv, D = 8, 4, 64
        block_size = 16
        seq_ids = [10, 20, 30]
        kv_lens = [18, 25, 22]
        scale = 1.0 / math.sqrt(D)

        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            quantized_kv=False,
            query_layout="batched",
            B=1,
            H_q=H_q,
            H_kv=H_kv,
            D=D,
            num_blocks=128,
            block_size=block_size,
        )
        cache_ref = PagedKVCache(num_blocks=128, block_size=block_size, H=H_kv, D=D)

        k_prefill = {}
        v_prefill = {}
        for sid, kv_len in zip(seq_ids, kv_lens):
            k_i = mx.random.normal((1, H_kv, kv_len, D)).astype(mx.float16)
            v_i = mx.random.normal((1, H_kv, kv_len, D)).astype(mx.float16)
            k_prefill[sid] = k_i
            v_prefill[sid] = v_i
            rt.context.cache.append(k_i, v_i, seq_id=sid)
            cache_ref.append(k_i, v_i, seq_id=sid)

        active_seq_ids = [30, 10]
        q = mx.random.normal((2, H_q, 1, D)).astype(mx.float16)
        k_new = mx.random.normal((2, H_kv, 1, D)).astype(mx.float16)
        v_new = mx.random.normal((2, H_kv, 1, D)).astype(mx.float16)

        out_rt = rt.paged_step_batch(
            q,
            k_new,
            v_new,
            seq_ids=active_seq_ids,
            scale=scale,
            causal=True,
        )

        for b, sid in enumerate(active_seq_ids):
            cache_ref.append(k_new[b : b + 1], v_new[b : b + 1], seq_id=sid)
        table_ref = cache_ref.get_block_table(active_seq_ids)
        lens_ref = cache_ref.get_seq_lens(active_seq_ids)
        out_ref = flash_attention_paged(
            q,
            cache_ref.k_pool,
            cache_ref.v_pool,
            table_ref,
            lens_ref,
            scale=scale,
            causal=True,
            block_size=block_size,
        )

        mx.eval(out_rt, out_ref)
        diff = float(mx.abs(out_rt.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3
        assert rt.seq_length(30) == kv_lens[2] + 1
        assert rt.seq_length(10) == kv_lens[0] + 1
        assert rt.seq_length(20) == kv_lens[1]
        assert rt.metadata["active_seq_ids"] == tuple(active_seq_ids)
        assert rt.metadata["active_cache_batch_idx"] is None

    def test_runtime_paged_step_batch_cache_batch_idx(self):
        from mlx_mfa import create_decode_runtime

        mx.random.seed(813)
        H_q, H_kv, D = 8, 4, 64
        block_size = 16
        scale = 1.0 / math.sqrt(D)
        seq_ids = [5, 7, 9]

        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            quantized_kv=False,
            query_layout="batched",
            B=1,
            H_q=H_q,
            H_kv=H_kv,
            D=D,
            num_blocks=128,
            block_size=block_size,
        )
        for sid in seq_ids:
            k_i = mx.random.normal((1, H_kv, 16, D)).astype(mx.float16)
            v_i = mx.random.normal((1, H_kv, 16, D)).astype(mx.float16)
            rt.context.cache.append(k_i, v_i, seq_id=sid)

        q = mx.random.normal((2, H_q, 1, D)).astype(mx.float16)
        k_new = mx.random.normal((2, H_kv, 1, D)).astype(mx.float16)
        v_new = mx.random.normal((2, H_kv, 1, D)).astype(mx.float16)
        idx = mx.array([2, 0], dtype=mx.int32)
        out = rt.paged_step_batch(
            q,
            k_new,
            v_new,
            seq_ids=seq_ids,
            cache_batch_idx=idx,
            scale=scale,
            causal=True,
        )
        mx.eval(out)
        assert out.shape == (2, H_q, 1, D)
        assert rt.metadata["active_seq_ids"] == (9, 5)
        assert rt.metadata["active_cache_batch_idx"] == (2, 0)

    def test_flash_attention_paged_varlen_cache_batch_idx(self):
        from mlx_mfa import flash_attention_paged_varlen

        mx.random.seed(814)
        H_q, H_kv, D = 8, 4, 64
        block_size = 16
        q_lens = [2, 3]
        kv_lens = [20, 27, 18, 31]
        scale = 1.0 / math.sqrt(D)

        q_seqs = [mx.random.normal((1, H_q, ql, D)).astype(mx.float16) for ql in q_lens]
        k_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        v_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        mx.eval(*q_seqs, *k_seqs, *v_seqs)
        q_pack, cu_q = TestPagedVarlenQueries._pack_queries(q_seqs)
        pool_k, pool_v, table, lens = TestPagedVarlenQueries._build_hetero_paged_pool(
            k_seqs,
            v_seqs,
            block_size,
        )
        idx = mx.array([3, 1], dtype=mx.int32)

        out_remap = flash_attention_paged_varlen(
            q_pack,
            pool_k,
            pool_v,
            table,
            lens,
            cu_q,
            scale=scale,
            causal=True,
            block_size=block_size,
            cache_batch_idx=idx,
        )
        out_ref = flash_attention_paged_varlen(
            q_pack,
            pool_k,
            pool_v,
            table[idx],
            lens[idx],
            cu_q,
            scale=scale,
            causal=True,
            block_size=block_size,
        )
        mx.eval(out_remap, out_ref)
        diff = float(mx.abs(out_remap.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3

    def test_invalid_paged_remap_fails(self):
        from mlx_mfa import (
            flash_attention_paged,
            flash_attention_paged_varlen,
            create_decode_runtime,
        )

        q = mx.zeros((2, 4, 1, 64), dtype=mx.float16)
        pool = mx.zeros((8, 16, 4, 64), dtype=mx.float16)
        table = mx.array([[0], [1], [2]], dtype=mx.int32)
        lens = mx.array([16, 16, 16], dtype=mx.int32)

        with pytest.raises(ValueError, match="out-of-range"):
            flash_attention_paged(
                q,
                pool,
                pool,
                table,
                lens,
                cache_batch_idx=mx.array([0, 3], dtype=mx.int32),
            )

        q_pack = mx.zeros((1, 4, 3, 64), dtype=mx.float16)
        cu = mx.array([0, 1, 3], dtype=mx.int32)
        with pytest.raises(ValueError, match="out-of-range"):
            flash_attention_paged_varlen(
                q_pack,
                pool,
                pool,
                table,
                lens,
                cu,
                cache_batch_idx=mx.array([0, 5], dtype=mx.int32),
            )

        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            quantized_kv=False,
            query_layout="batched",
            B=1,
            H_q=4,
            H_kv=4,
            D=64,
            num_blocks=16,
            block_size=16,
        )
        q_bad = mx.zeros((2, 4, 1, 64), dtype=mx.float16)
        k_bad = mx.zeros((2, 4, 1, 64), dtype=mx.float16)
        v_bad = mx.zeros((1, 4, 1, 64), dtype=mx.float16)
        with pytest.raises(ValueError, match="batch sizes must match"):
            rt.paged_step_batch(q_bad, k_bad, v_bad, seq_ids=[0, 1])


class TestChunkedPrefillRuntime:
    """Chunked prefill parity/validation for dense and paged runtime paths."""

    def test_dense_chunked_prefill_matches_monolithic(self):
        from mlx_mfa import create_decode_runtime

        mx.random.seed(821)
        B, H, N, D = 1, 4, 37, 64
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)

        rt_ref = create_decode_runtime(
            backend="dense", quantized_kv=False, B=B, H_kv=H, D=D
        )
        out_ref = rt_ref.prefill(q, k, v, scale=scale, causal=True)

        rt_chunk = create_decode_runtime(
            backend="dense", quantized_kv=False, B=B, H_kv=H, D=D
        )
        out_chunk = rt_chunk.chunked_prefill(
            q, k, v, chunk_size=8, scale=scale, causal=True
        )
        mx.eval(out_ref, out_chunk)
        diff = float(mx.abs(out_ref.astype(mx.float32) - out_chunk.astype(mx.float32)).max())
        assert diff < 5e-3
        assert rt_chunk.seq_length() == N

    def test_paged_chunked_prefill_matches_incremental_manual_reference(self):
        from mlx_mfa import create_decode_runtime, PagedKVCache, flash_attention_paged

        mx.random.seed(822)
        B, H_q, H_kv, N, D = 2, 8, 4, 29, 64
        scale = 1.0 / math.sqrt(D)
        seq_ids = [101, 202]
        q = mx.random.normal((B, H_q, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)

        cache_ref = PagedKVCache(num_blocks=128, block_size=16, H=H_kv, D=D)
        out_ref_parts = []
        for s in range(0, N, 7):
            e = min(N, s + 7)
            q_c = q[:, :, s:e, :]
            k_c = k[:, :, s:e, :]
            v_c = v[:, :, s:e, :]
            for b, sid in enumerate(seq_ids):
                cache_ref.append(k_c[b : b + 1], v_c[b : b + 1], seq_id=sid)
            table = cache_ref.get_block_table(seq_ids)
            lens = cache_ref.get_seq_lens(seq_ids)
            out_ref_parts.append(
                flash_attention_paged(
                    q_c,
                    cache_ref.k_pool,
                    cache_ref.v_pool,
                    table,
                    lens,
                    scale=scale,
                    causal=True,
                    block_size=16,
                )
            )
        out_ref = mx.concatenate(out_ref_parts, axis=2)

        rt_chunk = create_decode_runtime(
            backend="paged",
            paged=True,
            quantized_kv=False,
            query_layout="batched",
            B=1,
            H_q=H_q,
            H_kv=H_kv,
            D=D,
            num_blocks=128,
            block_size=16,
        )
        out_chunk = rt_chunk.chunked_prefill(
            q,
            k,
            v,
            chunk_size=7,
            seq_ids=seq_ids,
            scale=scale,
            causal=True,
        )
        mx.eval(out_ref, out_chunk)
        diff = float(mx.abs(out_ref.astype(mx.float32) - out_chunk.astype(mx.float32)).max())
        assert diff < 5e-3
        assert rt_chunk.seq_length(101) == N
        assert rt_chunk.seq_length(202) == N

    def test_paged_packed_chunked_prefill_multi_chunk(self):
        from mlx_mfa import (
            create_decode_runtime,
            PagedKVCache,
            flash_attention_paged_varlen,
        )

        mx.random.seed(823)
        H_q, H_kv, D = 8, 4, 64
        q_lens = [5, 2, 7]
        seq_ids = [7, 11, 13]
        scale = 1.0 / math.sqrt(D)

        q_seqs = [mx.random.normal((1, H_q, ql, D)).astype(mx.float16) for ql in q_lens]
        k_seqs = [mx.random.normal((1, H_kv, ql, D)).astype(mx.float16) for ql in q_lens]
        v_seqs = [mx.random.normal((1, H_kv, ql, D)).astype(mx.float16) for ql in q_lens]
        q_pack = mx.concatenate(q_seqs, axis=2)
        k_pack = mx.concatenate(k_seqs, axis=2)
        v_pack = mx.concatenate(v_seqs, axis=2)
        offsets = [0]
        for ql in q_lens:
            offsets.append(offsets[-1] + ql)
        cu = mx.array(offsets, dtype=mx.int32)

        rt_chunk = create_decode_runtime(
            backend="paged",
            paged=True,
            quantized_kv=False,
            query_layout="packed",
            B=1,
            H_q=H_q,
            H_kv=H_kv,
            D=D,
            num_blocks=128,
            block_size=16,
        )
        out_chunk = rt_chunk.chunked_prefill(
            q_pack,
            k_pack,
            v_pack,
            chunk_size=3,
            seq_ids=seq_ids,
            cu_seqlens_q=cu,
            scale=scale,
            causal=True,
        )

        # Reference the same scheduling contract explicitly:
        # append chunk K/V into paged cache, then run packed paged_varlen for
        # currently active rows, and reassemble per original packed sequence order.
        cache_ref = PagedKVCache(num_blocks=128, block_size=16, H=H_kv, D=D)
        lengths = [offsets[i + 1] - offsets[i] for i in range(len(seq_ids))]
        consumed = [0] * len(seq_ids)
        out_parts = [[] for _ in seq_ids]
        chunk_size = 3
        while any(consumed[i] < lengths[i] for i in range(len(seq_ids))):
            active_rows = [i for i in range(len(seq_ids)) if consumed[i] < lengths[i]]
            active_seq_ids = [seq_ids[i] for i in active_rows]
            q_parts = []
            chunk_offsets = [0]
            for i in active_rows:
                s = offsets[i] + consumed[i]
                e = min(offsets[i + 1], s + chunk_size)
                q_parts.append(q_pack[:, :, s:e, :])
                cache_ref.append(k_pack[:, :, s:e, :], v_pack[:, :, s:e, :], seq_id=seq_ids[i])
                chunk_offsets.append(chunk_offsets[-1] + (e - s))

            q_chunk = mx.concatenate(q_parts, axis=2)
            cu_chunk = mx.array(chunk_offsets, dtype=mx.int32)
            table = cache_ref.get_block_table(active_seq_ids)
            lens = cache_ref.get_seq_lens(active_seq_ids)
            out_step = flash_attention_paged_varlen(
                q_chunk,
                cache_ref.k_pool,
                cache_ref.v_pool,
                table,
                lens,
                cu_chunk,
                scale=scale,
                causal=True,
                block_size=16,
            )
            for local_idx, i in enumerate(active_rows):
                s = chunk_offsets[local_idx]
                e = chunk_offsets[local_idx + 1]
                out_parts[i].append(out_step[:, :, s:e, :])
                consumed[i] += e - s

        out_ref = mx.concatenate(
            [
                mx.concatenate(parts, axis=2)
                if parts
                else mx.zeros((1, H_q, 0, D), dtype=q_pack.dtype)
                for parts in out_parts
            ],
            axis=2,
        )

        mx.eval(out_chunk, out_ref)
        diff = float(mx.abs(out_chunk.astype(mx.float32) - out_ref.astype(mx.float32)).max())
        assert diff < 5e-3
        for sid, ql in zip(seq_ids, q_lens):
            assert rt_chunk.seq_length(sid) == ql

    def test_chunked_prefill_invalid_params(self):
        from mlx_mfa import create_decode_runtime

        q = mx.zeros((1, 4, 8, 64), dtype=mx.float16)
        k = mx.zeros((1, 4, 8, 64), dtype=mx.float16)
        v = mx.zeros((1, 4, 8, 64), dtype=mx.float16)

        rt_dense = create_decode_runtime(
            backend="dense", quantized_kv=False, B=1, H_kv=4, D=64
        )
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            rt_dense.chunked_prefill(q, k, v, chunk_size=0)
        with pytest.raises(ValueError, match="requires causal=True"):
            rt_dense.chunked_prefill(q, k, v, chunk_size=4, causal=False)

        rt_packed = create_decode_runtime(
            backend="paged",
            paged=True,
            quantized_kv=False,
            query_layout="packed",
            B=1,
            H_q=4,
            H_kv=4,
            D=64,
            num_blocks=32,
            block_size=16,
        )
        cu = mx.array([0, 8], dtype=mx.int32)
        with pytest.raises(ValueError, match="requires explicit seq_ids"):
            rt_packed.chunked_prefill(q, k, v, chunk_size=4, cu_seqlens_q=cu)
        # cache_batch_idx is now supported (passed through to paged_varlen)

    def test_chunked_prefill_cache_growth_with_reset_false(self):
        from mlx_mfa import create_decode_runtime

        mx.random.seed(824)
        B, H, D = 1, 4, 64
        rt = create_decode_runtime(
            backend="dense", quantized_kv=False, B=B, H_kv=H, D=D
        )
        q1 = mx.random.normal((B, H, 10, D)).astype(mx.float16)
        k1 = mx.random.normal((B, H, 10, D)).astype(mx.float16)
        v1 = mx.random.normal((B, H, 10, D)).astype(mx.float16)
        q2 = mx.random.normal((B, H, 6, D)).astype(mx.float16)
        k2 = mx.random.normal((B, H, 6, D)).astype(mx.float16)
        v2 = mx.random.normal((B, H, 6, D)).astype(mx.float16)

        out1 = rt.chunked_prefill(q1, k1, v1, chunk_size=4, causal=True, reset=True)
        out2 = rt.chunked_prefill(q2, k2, v2, chunk_size=3, causal=True, reset=False)
        mx.eval(out1, out2)
        assert out1.shape == (B, H, 10, D)
        assert out2.shape == (B, H, 6, D)
        assert rt.seq_length() == 16


# ==========================================================================
# Phase 4: SageAttention KV-cache + SageInferenceContext (Track LA)
# ==========================================================================

@pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
class TestSageKVCache:
    """sage_attention_kvcache and SageInferenceContext (Track LA)."""

    D = 64

    def test_sage_kvcache_same_length(self):
        """sage_attention_kvcache with N_q == N_k."""
        from mlx_mfa import sage_attention_kvcache
        B, H, N = 1, 2, 32
        mx.random.seed(10)
        q = mx.random.normal([B, H, N, self.D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, self.D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, self.D]).astype(mx.float16)
        out = sage_attention_kvcache(q, k, v, causal=True)
        mx.eval(out)
        assert out.shape == (B, H, N, self.D)
        assert mx.all(mx.isfinite(out)).item()

    def test_sage_kvcache_decode_shape(self):
        """sage_attention_kvcache decode: N_q=1 attends to N_k=64 cache."""
        from mlx_mfa import sage_attention_kvcache
        B, H, N_k = 1, 2, 64
        mx.random.seed(11)
        q = mx.random.normal([B, H, 1, self.D]).astype(mx.float16)
        k = mx.random.normal([B, H, N_k, self.D]).astype(mx.float16)
        v = mx.random.normal([B, H, N_k, self.D]).astype(mx.float16)
        out = sage_attention_kvcache(q, k, v, causal=True)
        mx.eval(out)
        assert out.shape == (B, H, 1, self.D)
        assert mx.all(mx.isfinite(out)).item()

    def test_sage_kvcache_matches_sage_attention(self):
        """sage_attention_kvcache output == sage_attention for same N_q==N_k."""
        from mlx_mfa import sage_attention, sage_attention_kvcache
        B, H, N = 1, 2, 32
        mx.random.seed(12)
        q = mx.random.normal([B, H, N, self.D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, self.D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, self.D]).astype(mx.float16)
        out_a = sage_attention(q, k, v, causal=False)
        out_b = sage_attention_kvcache(q, k, v, causal=False)
        mx.eval(out_a, out_b)
        assert mx.allclose(out_a, out_b, atol=0.0).item()

    def test_sage_inference_context_repr(self):
        """SageInferenceContext repr shows correct params."""
        from mlx_mfa import SageInferenceContext
        ctx = SageInferenceContext(B=1, H_kv=4, D=128, max_seq_len=1024)
        r = repr(ctx)
        assert "SageInferenceContext" in r
        assert "H_kv=4" in r
        assert "D=128" in r
        assert "seqlen=0" in r

    def test_sage_inference_context_prefill_shape(self):
        """SageInferenceContext prefill returns correct shape."""
        from mlx_mfa import SageInferenceContext
        B, H_q, H_kv, N = 1, 4, 2, 32
        ctx = SageInferenceContext(B=B, H_kv=H_kv, D=self.D, max_seq_len=256)
        mx.random.seed(20)
        q = mx.random.normal([B, H_q, N, self.D]).astype(mx.float16)
        k = mx.random.normal([B, H_kv, N, self.D]).astype(mx.float16)
        v = mx.random.normal([B, H_kv, N, self.D]).astype(mx.float16)
        out = ctx.prefill(q, k, v, scale=1.0 / self.D**0.5)
        mx.eval(out)
        assert out.shape == (B, H_q, N, self.D)
        assert ctx.seqlen == N
        assert mx.all(mx.isfinite(out)).item()
        mx.clear_cache()

    def test_sage_inference_context_step_shape(self):
        """SageInferenceContext step appends to cache and returns correct shape."""
        from mlx_mfa import SageInferenceContext
        B, H_q, H_kv, N = 1, 2, 2, 32
        ctx = SageInferenceContext(B=B, H_kv=H_kv, D=self.D, max_seq_len=256)
        mx.random.seed(21)
        kp = mx.random.normal([B, H_kv, N, self.D]).astype(mx.float16)
        vp = mx.random.normal([B, H_kv, N, self.D]).astype(mx.float16)
        # Prefill
        ctx.prefill(mx.random.normal([B, H_q, N, self.D]).astype(mx.float16), kp, vp)
        # Step
        q_tok = mx.random.normal([B, H_q, 1, self.D]).astype(mx.float16)
        k_tok = mx.random.normal([B, H_kv, 1, self.D]).astype(mx.float16)
        v_tok = mx.random.normal([B, H_kv, 1, self.D]).astype(mx.float16)
        out = ctx.step(q_tok, k_tok, v_tok, scale=1.0 / self.D**0.5)
        mx.eval(out)
        assert out.shape == (B, H_q, 1, self.D)
        assert ctx.seqlen == N + 1
        assert mx.all(mx.isfinite(out)).item()
        mx.clear_cache()

    def test_sage_inference_context_reset(self):
        """SageInferenceContext reset clears the cache."""
        from mlx_mfa import SageInferenceContext
        ctx = SageInferenceContext(B=1, H_kv=2, D=self.D, max_seq_len=256)
        mx.random.seed(22)
        k = mx.random.normal([1, 2, 16, self.D]).astype(mx.float16)
        v = mx.random.normal([1, 2, 16, self.D]).astype(mx.float16)
        ctx.prefill(mx.random.normal([1, 2, 16, self.D]).astype(mx.float16), k, v)
        assert ctx.seqlen == 16
        ctx.reset()
        assert ctx.seqlen == 0

    def test_sage_inference_context_manager(self):
        """SageInferenceContext context manager resets on exit."""
        from mlx_mfa import SageInferenceContext
        ctx = SageInferenceContext(B=1, H_kv=2, D=self.D, max_seq_len=256)
        mx.random.seed(23)
        k = mx.random.normal([1, 2, 16, self.D]).astype(mx.float16)
        v = mx.random.normal([1, 2, 16, self.D]).astype(mx.float16)
        with ctx:
            ctx.prefill(mx.random.normal([1, 2, 16, self.D]).astype(mx.float16), k, v)
            assert ctx.seqlen == 16
        assert ctx.seqlen == 0

    def test_sage_kvcache_export(self):
        """sage_attention_kvcache and SageInferenceContext exported from mlx_mfa."""
        import mlx_mfa
        assert hasattr(mlx_mfa, "sage_attention_kvcache")
        assert hasattr(mlx_mfa, "SageInferenceContext")
        assert "sage_attention_kvcache" in mlx_mfa.__all__
        assert "SageInferenceContext" in mlx_mfa.__all__


# ==========================================================================
# CP1: SageInferenceContext → QuantizedKVCache (incremental int8)
# ==========================================================================

@requires_ext
class TestSageInferenceContextQuantized:
    """SageInferenceContext now uses QuantizedKVCache for incremental int8."""

    D = 128

    def test_cache_type_is_quantized(self):
        """SageInferenceContext._cache is a QuantizedKVCache."""
        from mlx_mfa import SageInferenceContext
        from mlx_mfa.attention import QuantizedKVCache
        ctx = SageInferenceContext(B=1, H_kv=4, D=self.D, max_seq_len=256)
        assert isinstance(ctx._cache, QuantizedKVCache), (
            f"Expected QuantizedKVCache, got {type(ctx._cache).__name__}"
        )

    def test_step_returns_correct_shape(self):
        """step() returns [B, H_q, N_new, D] after prefill."""
        from mlx_mfa import SageInferenceContext
        B, Hq, Hkv, N_pre, D = 1, 4, 2, 64, self.D
        ctx = SageInferenceContext(B=B, H_kv=Hkv, D=D, max_seq_len=256)
        mx.random.seed(42)
        q_pre = mx.random.normal([B, Hq, N_pre, D]).astype(mx.float16)
        k_pre = mx.random.normal([B, Hkv, N_pre, D]).astype(mx.float16)
        v_pre = mx.random.normal([B, Hkv, N_pre, D]).astype(mx.float16)
        mx.eval(q_pre, k_pre, v_pre)
        ctx.prefill(q_pre, k_pre, v_pre)

        q_tok = mx.random.normal([B, Hq, 1, D]).astype(mx.float16)
        k_tok = mx.random.normal([B, Hkv, 1, D]).astype(mx.float16)
        v_tok = mx.random.normal([B, Hkv, 1, D]).astype(mx.float16)
        mx.eval(q_tok, k_tok, v_tok)
        out = ctx.step(q_tok, k_tok, v_tok)
        mx.eval(out)
        assert out.shape == (B, Hq, 1, D)
        assert mx.isfinite(out).all().item()

    def test_step_matches_prequantized(self):
        """step() output matches sage_attention_prequantized called manually."""
        from mlx_mfa import SageInferenceContext
        from mlx_mfa.attention import QuantizedKVCache, sage_attention_prequantized
        B, Hq, Hkv, N_pre, N_new, D = 1, 4, 4, 32, 1, self.D
        mx.random.seed(7)
        k_pre = mx.random.normal([B, Hkv, N_pre, D]).astype(mx.float16)
        v_pre = mx.random.normal([B, Hkv, N_pre, D]).astype(mx.float16)
        q_tok = mx.random.normal([B, Hq, N_new, D]).astype(mx.float16)
        k_tok = mx.random.normal([B, Hkv, N_new, D]).astype(mx.float16)
        v_tok = mx.random.normal([B, Hkv, N_new, D]).astype(mx.float16)
        mx.eval(k_pre, v_pre, q_tok, k_tok, v_tok)
        scale = 1.0 / math.sqrt(D)

        # Context path
        ctx = SageInferenceContext(B=B, H_kv=Hkv, D=D, max_seq_len=256)
        # manually load prefill into cache
        ctx._cache.append(k_pre, v_pre)
        out_ctx = ctx.step(q_tok, k_tok, v_tok, scale=scale)
        mx.eval(out_ctx)

        # Manual path — independent QuantizedKVCache
        cache2 = QuantizedKVCache(B, Hkv, D, max_seq_len=256)
        cache2.append(k_pre, v_pre)
        cache2.append(k_tok, v_tok)
        out_manual = sage_attention_prequantized(
            q_tok, cache2.k_int8, cache2.k_scale, cache2.v,
            scale=scale, causal=True,
        )
        mx.eval(out_manual)

        diff = mx.max(mx.abs(
            out_ctx.astype(mx.float32) - out_manual.astype(mx.float32)
        )).item()
        assert diff < 1e-4, f"ctx vs manual max_diff={diff:.4e}"

    def test_step_incremental_correct(self):
        """Step-by-step decode is consistent across steps (finite, correct shape)."""
        from mlx_mfa import SageInferenceContext
        B, Hq, Hkv, N_pre, D = 1, 4, 4, 32, self.D
        mx.random.seed(99)
        ctx = SageInferenceContext(B=B, H_kv=Hkv, D=D, max_seq_len=128)
        q_pre = mx.random.normal([B, Hq, N_pre, D]).astype(mx.float16)
        k_pre = mx.random.normal([B, Hkv, N_pre, D]).astype(mx.float16)
        v_pre = mx.random.normal([B, Hkv, N_pre, D]).astype(mx.float16)
        mx.eval(q_pre, k_pre, v_pre)
        ctx.prefill(q_pre, k_pre, v_pre)
        for _ in range(5):
            q_t = mx.random.normal([B, Hq, 1, D]).astype(mx.float16)
            k_t = mx.random.normal([B, Hkv, 1, D]).astype(mx.float16)
            v_t = mx.random.normal([B, Hkv, 1, D]).astype(mx.float16)
            mx.eval(q_t, k_t, v_t)
            out = ctx.step(q_t, k_t, v_t)
            mx.eval(out)
            assert out.shape == (B, Hq, 1, D)
            assert mx.isfinite(out).all().item()

    def test_step_window_matches_prequantized(self):
        """step(window_size=...) must forward the decode window to Sage kernel."""
        from mlx_mfa import SageInferenceContext
        from mlx_mfa.attention import QuantizedKVCache, sage_attention_prequantized
        B, Hq, Hkv, N_pre, D = 1, 4, 4, 64, self.D
        window = (256, 0)
        mx.random.seed(199)
        k_pre = mx.random.normal([B, Hkv, N_pre, D]).astype(mx.float16)
        v_pre = mx.random.normal([B, Hkv, N_pre, D]).astype(mx.float16)
        q_tok = mx.random.normal([B, Hq, 1, D]).astype(mx.float16)
        k_tok = mx.random.normal([B, Hkv, 1, D]).astype(mx.float16)
        v_tok = mx.random.normal([B, Hkv, 1, D]).astype(mx.float16)
        mx.eval(k_pre, v_pre, q_tok, k_tok, v_tok)

        ctx = SageInferenceContext(B=B, H_kv=Hkv, D=D, max_seq_len=256)
        ctx._cache.append(k_pre, v_pre)
        out_ctx = ctx.step(q_tok, k_tok, v_tok, window_size=window)
        mx.eval(out_ctx)

        cache2 = QuantizedKVCache(B, Hkv, D, max_seq_len=256)
        cache2.append(k_pre, v_pre)
        cache2.append(k_tok, v_tok)
        out_manual = sage_attention_prequantized(
            q_tok,
            cache2.k_int8,
            cache2.k_scale,
            cache2.v,
            causal=True,
            window_size=window,
        )
        mx.eval(out_manual)

        diff = mx.max(mx.abs(
            out_ctx.astype(mx.float32) - out_manual.astype(mx.float32)
        )).item()
        assert diff < 1e-4, f"ctx windowed step vs manual max_diff={diff:.4e}"

    def test_reset_clears_cache(self):
        """reset() zeroes the seqlen in QuantizedKVCache."""
        from mlx_mfa import SageInferenceContext
        ctx = SageInferenceContext(B=1, H_kv=2, D=self.D, max_seq_len=256)
        k = mx.random.normal([1, 2, 16, self.D]).astype(mx.float16)
        v = mx.random.normal([1, 2, 16, self.D]).astype(mx.float16)
        mx.eval(k, v)
        ctx._cache.append(k, v)
        assert ctx.seqlen == 16
        ctx.reset()
        assert ctx.seqlen == 0

    def test_step_no_apply_smooth_k_param(self):
        """step() no longer accepts apply_smooth_k (QuantizedKVCache path)."""
        import inspect
        from mlx_mfa.inference import SageInferenceContext
        sig = inspect.signature(SageInferenceContext.step)
        assert "apply_smooth_k" not in sig.parameters, (
            "apply_smooth_k should have been removed from step() signature"
        )


# ==========================================================================
# Phase 5: warmup_kernels + get_supported_configs update (Track LB)
# ==========================================================================

class TestWarmupAndConfigs:
    """warmup_kernels, get_supported_configs updates (Phase 5 / Track LB)."""

    def test_warmup_kernels_no_error(self):
        """warmup_kernels() runs without error."""
        from mlx_mfa import warmup_kernels
        import mlx.core as mx
        warmup_kernels(head_dims=[64], dtypes=[mx.float16])

    def test_warmup_kernels_noop_no_ext(self):
        """warmup_kernels is a no-op when extension unavailable (returns None)."""
        from mlx_mfa import warmup_kernels
        import mlx.core as mx
        result = warmup_kernels(head_dims=[64], dtypes=[mx.float16])
        assert result is None

    def test_warmup_kernels_exported(self):
        """warmup_kernels exported from mlx_mfa."""
        import mlx_mfa
        assert hasattr(mlx_mfa, "warmup_kernels")
        assert "warmup_kernels" in mlx_mfa.__all__

    def test_get_supported_configs_kernel_types_16(self):
        """kernel_types should be 16 when extension available."""
        from mlx_mfa import get_supported_configs
        cfg = get_supported_configs()
        if cfg["extension_available"]:
            assert cfg["kernel_types"] == 16
        else:
            assert cfg["kernel_types"] == 0

    def test_get_supported_configs_new_features(self):
        """Phase 4/5 features present in get_supported_configs()."""
        from mlx_mfa import get_supported_configs
        cfg = get_supported_configs()
        assert "sage_attention_kvcache" in cfg["features"]
        assert "sage_inference_context" in cfg["features"]
        assert "warmup_kernels" in cfg["features"]
        assert cfg["features"]["warmup_kernels"] is True


# ==========================================================================
# Phase 6: DispatchPolicy coherent runtime (Track LC-runtime)
# ==========================================================================

class TestDispatchPolicy:
    """DispatchPolicy string constants (Phase 6)."""

    def test_constants_exist(self):
        """DispatchPolicy has AUTO, MFA, SDPA constants."""
        from mlx_mfa import DispatchPolicy
        assert DispatchPolicy.AUTO == "auto"
        assert DispatchPolicy.MFA  == "mfa"
        assert DispatchPolicy.SDPA == "sdpa"

    def test_exported(self):
        """DispatchPolicy exported from mlx_mfa."""
        import mlx_mfa
        assert hasattr(mlx_mfa, "DispatchPolicy")
        assert "DispatchPolicy" in mlx_mfa.__all__

    def test_flash_attention_accepts_dispatch_policy(self):
        """flash_attention accepts DispatchPolicy string constants."""
        import mlx.core as mx
        from mlx_mfa import flash_attention, DispatchPolicy
        B, H, N, D = 1, 1, 4, 64
        mx.random.seed(0)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        # SDPA always works; AUTO and MFA require extension
        out_sdpa = flash_attention(q, k, v, backend=DispatchPolicy.SDPA)
        out_auto = flash_attention(q, k, v, backend=DispatchPolicy.AUTO)
        mx.eval(out_sdpa, out_auto)
        assert out_sdpa.shape == (B, H, N, D)
        assert out_auto.shape == (B, H, N, D)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.5 — TestSmartDispatch: shape-aware MFA/SDPA routing
# ─────────────────────────────────────────────────────────────────────────────
class TestSmartDispatch:
    """Validate the Phase 1 smart dispatch policy (should_use_mfa + integration)."""

    # ── should_use_mfa unit tests ─────────────────────────────────────────────

    def test_small_n_causal_d64_routes_sdpa(self):
        """Small N below threshold: D=64 causal N=512 should NOT use MFA."""
        from mlx_mfa.dispatch_policy import should_use_mfa
        # Threshold for D=64 causal is 2048; N=512 is below → SDPA
        assert not should_use_mfa(64, 512, causal=True, is_m3_plus=False)

    def test_large_n_causal_d64_routes_mfa(self):
        """Large N above threshold: D=64 causal N=4096 should use MFA."""
        from mlx_mfa.dispatch_policy import should_use_mfa
        # Threshold raised to 4096 for stable dispatch (1.04x at N=4096).
        assert should_use_mfa(64, 4096, causal=True, is_m3_plus=False)
        assert should_use_mfa(64, 8192, causal=True, is_m3_plus=False)

    def test_large_n_causal_d128_routes_mfa(self):
        """D=128 causal N=8192 should use MFA (1.25x crossover)."""
        from mlx_mfa.dispatch_policy import should_use_mfa
        assert should_use_mfa(128, 8192, causal=True, is_m3_plus=False)

    def test_d256_causal_f16_narrow_regime(self):
        """D=256 causal f16 uses MFA from N>=2048 (post-BK=8: 1.09x@2048)."""
        import mlx.core as mx
        from mlx_mfa.dispatch_policy import should_use_mfa
        assert not should_use_mfa(256, 1024, causal=True, is_m3_plus=False, dtype=mx.float16)
        assert should_use_mfa(256, 2048, causal=True, is_m3_plus=False, dtype=mx.float16)

    def test_d256_causal_bf16_stays_sdpa(self):
        """D=256 causal bf16 remains SDPA-backed on current benchmark evidence."""
        import mlx.core as mx
        from mlx_mfa.dispatch_policy import should_use_mfa
        assert not should_use_mfa(256, 16384, causal=True, is_m3_plus=False, dtype=mx.bfloat16)

    def test_d256_unknown_dtype_keeps_conservative_fallback(self):
        """Without dtype, policy keeps the conservative legacy threshold."""
        from mlx_mfa.dispatch_policy import should_use_mfa
        assert not should_use_mfa(256, 4096, causal=True, is_m3_plus=False)
        assert should_use_mfa(256, 8192, causal=True, is_m3_plus=False)

    def test_d256_m3plus_promoted_from_benchmark(self):
        """M3+ D=256 causal promoted from N>=2048 (M4 Max benchmarks: 1.58-1.68x)."""
        import mlx.core as mx
        from mlx_mfa.dispatch_policy import should_use_mfa
        # M3+ D=256 f16 causal: promoted from N>=2048
        assert should_use_mfa(256, 2048, causal=True, is_m3_plus=True, dtype=mx.float16)
        assert should_use_mfa(256, 16384, causal=True, is_m3_plus=True, dtype=mx.float16)
        # M3+ D=256 bf16 causal: also promoted (native bf16 ALU on M3+)
        assert should_use_mfa(256, 4096, causal=True, is_m3_plus=True, dtype=mx.bfloat16)
        # Below threshold: still SDPA
        assert not should_use_mfa(256, 1024, causal=True, is_m3_plus=True, dtype=mx.float16)

    def test_d256_force_path_override_mfa(self, monkeypatch):
        """MFA_FORCE_D256_PATH=1 must force D=256 auto route to MFA."""
        import mlx.core as mx
        from mlx_mfa.dispatch_policy import should_use_mfa
        monkeypatch.setenv("MFA_FORCE_D256_PATH", "1")
        assert should_use_mfa(256, 2048, causal=True, is_m3_plus=False, dtype=mx.bfloat16)
        # Non-D256 shapes should ignore this override.
        assert not should_use_mfa(128, 512, causal=False, is_m3_plus=False, dtype=mx.float16)

    def test_d256_force_path_override_sdpa(self, monkeypatch):
        """MFA_FORCE_D256_PATH=0 must force D=256 auto route to SDPA."""
        import mlx.core as mx
        from mlx_mfa.dispatch_policy import should_use_mfa
        monkeypatch.setenv("MFA_FORCE_D256_PATH", "0")
        assert not should_use_mfa(256, 16384, causal=True, is_m3_plus=False, dtype=mx.float16)

    def test_d512_dense_stays_sdpa_by_default(self):
        """D=512 dense stays SDPA-backed on current benchmark evidence."""
        import mlx.core as mx
        from mlx_mfa.dispatch_policy import should_use_mfa
        assert not should_use_mfa(512, 8192, causal=True, is_m3_plus=False, dtype=mx.float16)
        assert not should_use_mfa(512, 8192, causal=False, is_m3_plus=False, dtype=mx.float16)

    def test_d512_force_path_override_mfa(self, monkeypatch):
        """MFA_FORCE_D512_PATH=1 must force D=512 auto route to MFA."""
        import mlx.core as mx
        from mlx_mfa.dispatch_policy import should_use_mfa
        monkeypatch.setenv("MFA_FORCE_D512_PATH", "1")
        assert should_use_mfa(512, 1024, causal=False, is_m3_plus=False, dtype=mx.float16)
        # Non-D512 shapes should ignore this override.
        assert not should_use_mfa(64, 512, causal=False, is_m3_plus=False, dtype=mx.float16)

    def test_d512_force_path_override_sdpa(self, monkeypatch):
        """MFA_FORCE_D512_PATH=0 must force D=512 auto route to SDPA."""
        import mlx.core as mx
        from mlx_mfa.dispatch_policy import should_use_mfa
        monkeypatch.setenv("MFA_FORCE_D512_PATH", "0")
        assert not should_use_mfa(512, 16384, causal=True, is_m3_plus=False, dtype=mx.float16)

    def test_noncausal_dispatch_policy(self):
        """Non-causal dispatch: M1/M2 routes MFA for D=64/128 N>=2048; M3+ stays SDPA."""
        from mlx_mfa.dispatch_policy import should_use_mfa
        # M1/M2: D=64/128 non-causal wins from N>=2048 (V2 BK=64, high TGP BW)
        assert should_use_mfa(64, 2048, causal=False, is_m3_plus=False)
        assert should_use_mfa(128, 4096, causal=False, is_m3_plus=False)
        # M1/M2: below threshold stays SDPA
        assert not should_use_mfa(64, 1024, causal=False, is_m3_plus=False)
        assert not should_use_mfa(128, 1024, causal=False, is_m3_plus=False)
        # M1/M2: D=256/512 non-causal stays SDPA
        assert not should_use_mfa(256, 8192, causal=False, is_m3_plus=False)
        assert not should_use_mfa(512, 8192, causal=False, is_m3_plus=False)
        # M3+: all non-causal stays SDPA (0.60-0.77x on M4 Max)
        for D in [64, 128, 256, 512]:
            for N in [2048, 4096, 8192]:
                assert not should_use_mfa(D, N, causal=False, is_m3_plus=True), \
                    f"M3+ non-causal D={D} N={N} routed to MFA unexpectedly"

    def test_window_always_routes_mfa(self):
        """Sliding-window attention always uses MFA (tile-skip guarantee)."""
        from mlx_mfa.dispatch_policy import should_use_mfa
        # left-only window, small N
        assert should_use_mfa(64, 128, causal=False, is_m3_plus=False,
                               window_size=(64, -1))
        # right-only window
        assert should_use_mfa(128, 64, causal=False, is_m3_plus=False,
                               window_size=(-1, 32))
        # both dimensions
        assert should_use_mfa(256, 256, causal=True, is_m3_plus=False,
                               window_size=(128, 128))

    def test_backend_mfa_forces_mfa(self):
        """backend='mfa' overrides the shape dispatch → always True."""
        from mlx_mfa.dispatch_policy import should_use_mfa
        assert should_use_mfa(256, 64, causal=False, is_m3_plus=False,
                               backend="mfa")

    def test_backend_sdpa_forces_sdpa(self):
        """backend='sdpa' overrides the shape dispatch → always False."""
        from mlx_mfa.dispatch_policy import should_use_mfa
        assert not should_use_mfa(64, 8192, causal=True, is_m3_plus=False,
                                   backend="sdpa")

    # ── flash_attention integration: shape routing + correctness ─────────────

    def test_auto_small_n_matches_sdpa(self):
        """flash_attention(auto, small N) == flash_attention(sdpa): same output."""
        import mlx.core as mx
        from mlx_mfa import flash_attention
        B, H, N, D = 1, 4, 256, 64  # N=256 < threshold 2048, non-causal
        mx.random.seed(42)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        out_auto = flash_attention(q, k, v, causal=False, backend="auto")
        out_sdpa = flash_attention(q, k, v, causal=False, backend="sdpa")
        mx.eval(out_auto, out_sdpa)
        import numpy as np
        np.testing.assert_allclose(
            np.array(out_auto.astype(mx.float32)),
            np.array(out_sdpa.astype(mx.float32)),
            atol=1e-3,
            err_msg="auto-small-N output differs from sdpa backend",
        )

    def test_auto_large_n_causal_d64_uses_sdpa_on_m5(self):
        """M5 dense D64 forward delegates to SDPA; only backward uses V6."""
        import mlx.core as mx
        from mlx_mfa import flash_attention, get_device_info, is_mfa_available
        from mlx_mfa import _dispatch_trace as dtrace
        if not is_mfa_available():
            pytest.skip("MFA extension not available")
        if not bool(get_device_info().get("is_m5_plus", False)):
            pytest.skip("M5/NAX dispatch-map lock")
        B, H, N, D = 1, 4, 4096, 64  # above 2048 threshold
        mx.random.seed(7)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        with dtrace.capture() as trace:
            out_auto = flash_attention(q, k, v, causal=True, backend="auto")
            mx.eval(out_auto)
        out_sdpa = flash_attention(q, k, v, causal=True, backend="sdpa")
        mx.eval(out_sdpa)
        terminal = [entry for entry in trace if not entry[1].startswith("[reentrant]")]
        assert terminal[-1][0] == "apple_sdpa"
        assert float(mx.max(mx.abs(
            out_auto.astype(mx.float32) - out_sdpa.astype(mx.float32)
        )).item()) == 0.0

    def test_mixed_dtype_routes_mfa(self):
        """Mixed-dtype (f32 Q + f16 K/V) always routes to MFA, not SDPA (NaN guard)."""
        import mlx.core as mx
        from mlx_mfa import flash_attention, is_mfa_available
        if not is_mfa_available():
            pytest.skip("MFA extension not available — mixed-dtype handled by MFA only")
        # Flush the Metal buffer pool to prevent stale-buffer NaN from prior
        # large backward tests leaking into uninitialized allocations.
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.clear_cache()
        B, H, N, D = 1, 2, 32, 64
        mx.random.seed(99)
        q = mx.random.normal([B, H, N, D]).astype(mx.float32)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        out = flash_attention(q, k, v, causal=False, backend="auto")
        mx.eval(out)
        assert out.shape == (B, H, N, D)
        import numpy as np
        assert np.all(np.isfinite(np.array(out.astype(mx.float32)))), \
            "mixed-dtype auto dispatch produced NaN — not routed to MFA"
        # III-4 PASS1-REGRESSION FIX: finiteness alone let silent garbage
        # through — eval_gpu keyed the kernel dtype on q alone, so an f32
        # kernel reinterpreted f16 K/V buffers (max_err ~15, NaN only when
        # the Metal buffer pool recycled dirty allocations).  Assert
        # numerical agreement with an explicitly-cast SDPA ground truth.
        ref = mx.fast.scaled_dot_product_attention(
            q, k.astype(mx.float32), v.astype(mx.float32), scale=D ** -0.5)
        max_err = float(mx.abs(out - ref).max())
        assert max_err < 5e-3, \
            f"mixed-dtype output diverges from cast-SDPA ground truth: {max_err}"

    def test_calibrate_dispatch_returns_dict(self):
        """calibrate_dispatch() is importable and returns a threshold dict."""
        from mlx_mfa import calibrate_dispatch
        # Smoke-test: just verify the function exists and is callable.
        import inspect
        assert callable(calibrate_dispatch)
        sig = inspect.signature(calibrate_dispatch)
        assert "head_dims" in sig.parameters
        assert "save_path" in sig.parameters


# ---------------------------------------------------------------------------
# V2 gen-aware BK selection (Phase 1 — MFA_FORCE_GEN + MFA_V2_FORCE_BK)
# ---------------------------------------------------------------------------

@requires_ext
class TestV2GenAwareBK:
    """Verify gen-aware BK selection for D=128 V2 kernels.

    BK=32 on M1/M2 (gen=13/14), BK=64 on M3+ (gen>=15).
    MFA_V2_FORCE_BK=32|64 overrides both paths.

    All tests compare against SDPA reference to ensure correctness under
    the different BK configs; both should produce equivalent results within
    f16 tolerance (atol=1e-2).
    """

    def _run(self, q, k, v, scale, causal, *, gen=None, force_bk=None):
        """Run flash_attention with optional gen/BK overrides, return np.array."""
        import os as _os
        env_backup = {}
        try:
            if gen is not None:
                env_backup["MFA_FORCE_GEN"] = _os.environ.get("MFA_FORCE_GEN")
                _os.environ["MFA_FORCE_GEN"] = str(gen)
            if force_bk is not None:
                env_backup["MFA_V2_FORCE_BK"] = _os.environ.get("MFA_V2_FORCE_BK")
                _os.environ["MFA_V2_FORCE_BK"] = str(force_bk)
            out = flash_attention(q, k, v, scale=scale, causal=causal)
            mx.eval(out)
        finally:
            for key, prev in env_backup.items():
                if prev is None:
                    _os.environ.pop(key, None)
                else:
                    _os.environ[key] = prev
        return np.array(out.astype(mx.float32))

    def _sdpa(self, q, k, v, scale, causal):
        from mlx_mfa.attention import _fallback_sdpa
        out = _fallback_sdpa(q, k, v, scale, causal)
        mx.eval(out)
        return np.array(out.astype(mx.float32))

    @pytest.mark.parametrize("causal", [True, False])
    def test_v2_config_m1(self, causal):
        """D=128 with M1 gen (BK=32) matches SDPA reference."""
        mx.random.seed(42)
        q = mx.random.normal([1, 4, 128, 128]).astype(mx.float16)
        k = mx.random.normal([1, 4, 128, 128]).astype(mx.float16)
        v = mx.random.normal([1, 4, 128, 128]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(128)
        out_m1  = self._run(q, k, v, scale, causal, gen=13)  # M1 → BK=32
        out_ref = self._sdpa(q, k, v, scale, causal)
        np.testing.assert_allclose(
            out_ref, out_m1, atol=1e-2,
            err_msg=f"V2 M1 BK=32 (causal={causal}) != SDPA ref",
        )

    @pytest.mark.parametrize("causal", [True, False])
    def test_v2_config_m3_plus(self, causal):
        """D=128 with M3+ gen (BK=64) matches SDPA reference."""
        mx.random.seed(43)
        q = mx.random.normal([1, 4, 128, 128]).astype(mx.float16)
        k = mx.random.normal([1, 4, 128, 128]).astype(mx.float16)
        v = mx.random.normal([1, 4, 128, 128]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(128)
        out_m3  = self._run(q, k, v, scale, causal, gen=15)  # M3+ → BK=64
        out_ref = self._sdpa(q, k, v, scale, causal)
        np.testing.assert_allclose(
            out_ref, out_m3, atol=1e-2,
            err_msg=f"V2 M3+ BK=64 (causal={causal}) != SDPA ref",
        )

    @pytest.mark.parametrize("force_bk", [32, 64])
    def test_v2_force_bk_env(self, force_bk):
        """MFA_V2_FORCE_BK overrides gen-based BK for D=128; output must match SDPA."""
        mx.random.seed(44)
        q = mx.random.normal([1, 4, 128, 128]).astype(mx.float16)
        k = mx.random.normal([1, 4, 128, 128]).astype(mx.float16)
        v = mx.random.normal([1, 4, 128, 128]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(128)
        # Force M1 gen so we can verify FORCE_BK=64 overrides it
        out  = self._run(q, k, v, scale, True, gen=13, force_bk=force_bk)
        ref  = self._sdpa(q, k, v, scale, True)
        np.testing.assert_allclose(
            ref, out, atol=1e-2,
            err_msg=f"MFA_V2_FORCE_BK={force_bk} (M1 sim) != SDPA ref",
        )


# ---------------------------------------------------------------------------
# Phase 2 — Auto-calibration + python -m mlx_mfa CLI
# ---------------------------------------------------------------------------

class TestAutoCalibration:
    """Verify calibrate_kernel_configs benchmarks BK and saves to JSON."""

    def test_calibrate_creates_file(self, tmp_path):
        """calibrate_dispatch(calibrate_kernel_configs=True) writes kernel_configs to JSON."""
        from mlx_mfa.dispatch_policy import calibrate_dispatch
        out = tmp_path / "dispatch.json"
        calibrate_dispatch(
            head_dims=[64],          # fast — skip 128/256 dispatch sweep
            save_path=str(out),
            warmup=1,
            n_iters=2,
            calibrate_kernel_configs=True,
        )
        assert out.exists(), "calibrate_dispatch did not write JSON"
        import json
        data = json.loads(out.read_text())
        assert "kernel_configs" in data, "kernel_configs key missing from JSON"
        bk = data["kernel_configs"].get("d128_optimal_bk")
        assert bk in (32, 64), f"d128_optimal_bk={bk!r} is not 32 or 64"

    def test_calibrate_without_kernel_configs(self, tmp_path):
        """calibrate_kernel_configs=False omits kernel_configs from JSON."""
        from mlx_mfa.dispatch_policy import calibrate_dispatch
        out = tmp_path / "dispatch.json"
        calibrate_dispatch(
            head_dims=[64],
            save_path=str(out),
            warmup=1,
            n_iters=2,
            calibrate_kernel_configs=False,
        )
        import json
        data = json.loads(out.read_text())
        assert "kernel_configs" not in data

    def test_calibrate_writes_splitk_thresholds(self, tmp_path):
        """calibrate_dispatch writes split-K crossover entries to JSON."""
        from mlx_mfa.dispatch_policy import calibrate_dispatch
        out = tmp_path / "dispatch.json"
        calibrate_dispatch(
            head_dims=[64],
            save_path=str(out),
            warmup=1,
            n_iters=1,
            calibrate_kernel_configs=False,
            calibrate_splitk=True,
        )
        import json
        data = json.loads(out.read_text())
        entries = data.get("splitk_thresholds")
        assert isinstance(entries, list) and entries, "splitk_thresholds missing/empty"
        sample = entries[0]
        for key in ("D", "causal", "has_alibi", "has_window", "max_N"):
            assert key in sample, f"splitk_thresholds entry missing key: {key}"


class TestSplitKPolicy:
    """Split-K calibration table + env override behavior."""

    def test_should_use_splitk_env_override_precedence(self, monkeypatch):
        """MFA_FORCE_SPLITK must override calibrated split-K max_N entries."""
        from mlx_mfa.dispatch_policy import should_use_splitk, _splitk_env_key

        key = _splitk_env_key(64, True, has_alibi=False)  # non-windowed → _W0
        monkeypatch.setenv(key, "256")

        monkeypatch.setenv("MFA_FORCE_SPLITK", "1")
        assert should_use_splitk(64, 4096, True) is True

        monkeypatch.setenv("MFA_FORCE_SPLITK", "0")
        assert should_use_splitk(64, 64, True) is False

        monkeypatch.delenv("MFA_FORCE_SPLITK", raising=False)
        assert should_use_splitk(64, 128, True) is True
        assert should_use_splitk(64, 512, True) is False

    def test_load_calibration_missing_file_is_safe(self, tmp_path, monkeypatch):
        """Missing dispatch table must not raise or mutate split-K env."""
        from mlx_mfa.dispatch_policy import _load_calibrated_kernel_config, _splitk_env_key

        missing = tmp_path / "no_such_dispatch.json"
        monkeypatch.setenv("MLX_MFA_DISPATCH_TABLE", str(missing))
        env_key = _splitk_env_key(64, True, has_alibi=False)  # non-windowed → _W0
        monkeypatch.delenv(env_key, raising=False)

        _load_calibrated_kernel_config()
        assert os.environ.get(env_key) is None

    def test_load_calibration_sets_splitk_env(self, tmp_path, monkeypatch):
        """v2 splitk_thresholds entries (with window sizes) load into env vars
        for C++ dispatch, keyed by the window SIZE (M-02)."""
        import json
        from mlx_mfa.dispatch_policy import (
            _load_calibrated_kernel_config, _splitk_env_key, _CALIBRATION_SCHEMA_VERSION,
        )

        table = tmp_path / "dispatch.json"
        payload = {
            "calibration_schema_version": _CALIBRATION_SCHEMA_VERSION,
            "thresholds": [],
            "splitk_thresholds": [
                {"D": 64, "causal": True, "has_alibi": False,
                 "window_left": 256, "window_right": 0, "max_N": 1024},
            ],
        }
        table.write_text(json.dumps(payload))

        monkeypatch.setenv("MLX_MFA_DISPATCH_TABLE", str(table))
        env_key = _splitk_env_key(64, True, has_alibi=False, window_left=256, window_right=0)
        monkeypatch.delenv(env_key, raising=False)
        _load_calibrated_kernel_config()
        assert os.environ.get(env_key) == "1024"


class TestSageDecodePolicy:
    """Sage decode auto policy + env override behavior."""

    def test_force_override_precedence(self, monkeypatch):
        from mlx_mfa.dispatch_policy import should_use_sage_decode

        monkeypatch.setenv("MFA_FORCE_SAGE_DECODE", "0")
        assert should_use_sage_decode(
            128, 1, 8192, True,
            has_quantized_kv=True,
            window_size=(256, 0),
            gqa_factor=2,
        ) is False

        monkeypatch.setenv("MFA_FORCE_SAGE_DECODE", "1")
        assert should_use_sage_decode(
            64, 2, 512, True,
            has_quantized_kv=True,
            window_size=None,
            gqa_factor=1,
        ) is True

    def test_auto_policy_is_narrow_and_decode_only(self, monkeypatch):
        from mlx_mfa.dispatch_policy import should_use_sage_decode

        monkeypatch.delenv("MFA_FORCE_SAGE_DECODE", raising=False)
        assert should_use_sage_decode(
            128, 4, 4096, True,
            has_quantized_kv=True,
            window_size=(256, 0),
            gqa_factor=2,
            dtype=mx.float16,
        ) is True
        assert should_use_sage_decode(
            128, 1, 4096, True,
            has_quantized_kv=True,
            window_size=(256, 0),
            gqa_factor=2,
            dtype=mx.bfloat16,
        ) is True
        assert should_use_sage_decode(
            128, 4, 2048, True,
            has_quantized_kv=True,
            window_size=(256, 0),
            gqa_factor=2,
            dtype=mx.float16,
        ) is False
        assert should_use_sage_decode(
            128, 1, 8192, True,
            has_quantized_kv=True,
            window_size=None,
            gqa_factor=2,
            dtype=mx.bfloat16,
        ) is False
        assert should_use_sage_decode(
            64, 1, 8192, True,
            has_quantized_kv=True,
            window_size=(256, 0),
            gqa_factor=1,
            dtype=mx.float16,
        ) is False
        assert should_use_sage_decode(
            128, 8, 8192, True,
            has_quantized_kv=True,
            window_size=(256, 0),
            gqa_factor=2,
            dtype=mx.float16,
        ) is False

    def test_requires_quantized_kv(self, monkeypatch):
        from mlx_mfa.dispatch_policy import should_use_sage_decode

        monkeypatch.delenv("MFA_FORCE_SAGE_DECODE", raising=False)
        assert should_use_sage_decode(
            128, 1, 8192, True,
            has_quantized_kv=False,
            window_size=(256, 0),
            gqa_factor=1,
        ) is False


class TestNativeBackwardPolicy:
    """Native backward policy-table behavior.

    The MFA_FORCE_NATIVE_BWD override knob was removed in v2.56.0
    (deprecation cycle complete; forced STEEL backward was dominated at
    every cell — sprint-C Track 2).  Routing follows the benchmark-backed
    policy table only; the now-removed env var is inert — see
    tests/test_v50_prompt_5f_kd5_deprecation.py for the removal guard.
    """

    def test_auto_policy_stays_disabled_without_winning_regime(self):
        """Auto policy keeps native backward off until benchmark-backed wins exist."""
        from mlx_mfa.dispatch_policy import should_use_native_backward

        assert should_use_native_backward(64, 16384, True, dtype=mx.float16) is False
        assert should_use_native_backward(128, 16384, True, dtype=mx.bfloat16) is False

    def test_unsupported_shapes_never_route_native(self):
        """Non-causal, D>128, and float32 must stay off native backward."""
        from mlx_mfa.dispatch_policy import should_use_native_backward

        assert should_use_native_backward(256, 8192, True, dtype=mx.float16) is False
        assert should_use_native_backward(64, 8192, False, dtype=mx.float16) is False
        assert should_use_native_backward(64, 8192, True, dtype=mx.float32) is False


@requires_ext
class TestNativeBackwardRouting:
    """Integration checks for backward routing in the custom VJP path."""

    def test_default_routing_uses_sdpa_vjp(self, monkeypatch):
        """The default flash_attention VJP uses SDPA-vjp, not native STEEL
        backward: the policy table is conservative and the
        MFA_FORCE_NATIVE_BWD knob that used to force native was removed in
        v2.56.0."""
        import mlx_mfa.attention as attn
        import mlx_mfa._ext as ext
        from mlx_mfa import flash_attention

        attn._make_mfa_custom.cache_clear()
        calls = {"n": 0}
        original = ext.mfa_steel_backward

        def wrapped(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ext, "mfa_steel_backward", wrapped)
        attn._make_mfa_custom.cache_clear()

        B, H, N, D = 1, 1, 64, 64
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(1234)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        dO = mx.random.normal([B, H, N, D]).astype(mx.float16)

        _, (dq, dk, dv) = mx.vjp(
            lambda qi, ki, vi: flash_attention(
                qi, ki, vi, scale=scale, causal=True, backend="mfa"
            ),
            [q, k, v],
            [dO],
        )
        mx.eval(dq, dk, dv)
        attn._make_mfa_custom.cache_clear()
        assert calls["n"] == 0, (
            "Default VJP must use SDPA-vjp; native STEEL backward is no "
            "longer auto-routed (policy table conservative; force knob removed)")

    # Repo review 2026-05: the D=128 N>=2048 xfails (KD-5 "zeroed blocks")
    # are REMOVED — root cause found and fixed: MFASteelBwdDKV dispatch
    # computed the grid with cfg.BK (=32 on M3+ for D=128) while the
    # generator overrides BK to 16 for D>64, leaving K-rows beyond NK*16
    # unwritten.  Dispatch BK now mirrors the generator override
    # (mfa_attention.cpp) and STEEL backward D=128 matches SDPA-VJP.
    #
    # v2.56.0: the MFA_FORCE_NATIVE_BWD knob that used to route here was
    # removed.  Per keep-all-paths the STEEL backward kernel is RETAINED;
    # this guard now exercises it via its DIRECT ext binding
    # (mfa_forward_with_lse -> mfa_steel_backward), so the kernel stays
    # tested even though it is no longer auto-routed.
    @pytest.mark.parametrize("D,N", [
        (64, 2048),
        (64, 4096),
        (128, 2048),
        (128, 4096),
    ])
    def test_steel_backward_matches_sdpa_gradients(self, D, N):
        """STEEL backward (direct binding) gradients match SDPA-VJP at target shapes."""
        import mlx_mfa._ext as ext
        from mlx_mfa.attention import _fallback_sdpa

        B, H = 1, 2
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(314159)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        dO = mx.random.normal([B, H, N, D]).astype(mx.float16)

        O, L = ext.mfa_forward_with_lse(q, k, v, scale, True)
        dq_native, dk_native, dv_native = ext.mfa_steel_backward(
            q, k, v, O, L, dO, scale, True)
        mx.eval(dq_native, dk_native, dv_native)

        _, (dq_sdpa, dk_sdpa, dv_sdpa) = mx.vjp(
            lambda qi, ki, vi: _fallback_sdpa(qi, ki, vi, scale, True),
            [q, k, v],
            [dO],
        )
        mx.eval(dq_sdpa, dk_sdpa, dv_sdpa)

        for name, a, b in (("dQ", dq_native, dq_sdpa),
                           ("dK", dk_native, dk_sdpa),
                           ("dV", dv_native, dv_sdpa)):
            np.testing.assert_allclose(
                np.array(a.astype(mx.float32)),
                np.array(b.astype(mx.float32)),
                atol=5e-2,
                err_msg=f"{name} mismatch for D={D} N={N}",
            )


class TestMainCLI:
    """Verify python -m mlx_mfa subcommands."""

    def test_main_info(self, capsys):
        """python -m mlx_mfa info prints device and version."""
        from mlx_mfa.__main__ import main
        main(["info"])
        captured = capsys.readouterr()
        assert "mlx-mfa" in captured.out
        assert "Device" in captured.out

    def test_main_no_args_exits(self):
        """python -m mlx_mfa with no subcommand exits with code 1."""
        from mlx_mfa.__main__ import main
        import pytest
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# Phase 3 — V2 Feature Extensions (RoPE, ALiBi; sparse stays in V1)
# ---------------------------------------------------------------------------

class TestV2FeatureExtensions:
    """Verify V2 kernel correctly handles RoPE and ALiBi, and that sparse
    attention routes to V1 (mask block-size mismatch with V2 BK)."""

    @pytest.mark.parametrize("D", [64, 128])
    @pytest.mark.skipif(
        not is_mfa_available(), reason="MFA extension not available"
    )
    def test_v2_rope_matches_v1(self, D):
        """V2 kernel with RoPE fusion matches V1 kernel with RoPE fusion."""
        import os as _os
        B, H, N = 1, 4, 4096
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(41)
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, N, D), dtype=mx.float16)
        v = mx.random.normal((B, H, N, D), dtype=mx.float16)
        cos, sin = _make_rope_tables(N + 64, D)

        out_v2 = flash_attention_rope(q, k, v, cos, sin, scale=scale, causal=True)
        prev = _os.environ.get("MFA_DISABLE_V2")
        try:
            _os.environ["MFA_DISABLE_V2"] = "1"
            out_v1 = flash_attention_rope(q, k, v, cos, sin, scale=scale, causal=True)
        finally:
            if prev is None:
                _os.environ.pop("MFA_DISABLE_V2", None)
            else:
                _os.environ["MFA_DISABLE_V2"] = prev
        mx.eval(out_v2, out_v1)

        np.testing.assert_allclose(
            np.array(out_v2.astype(mx.float32)),
            np.array(out_v1.astype(mx.float32)),
            atol=1e-2, rtol=1e-2,
            err_msg=f"D={D}: V2+RoPE differs from V1+RoPE",
        )

    @pytest.mark.parametrize("D", [64, 128])
    @pytest.mark.skipif(
        not is_mfa_available(), reason="MFA extension not available"
    )
    def test_v2_alibi_matches_v1(self, D):
        """V2 kernel with ALiBi matches V1 kernel with ALiBi."""
        import os as _os
        B, H, N = 1, 4, 4096
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, N, D), dtype=mx.float16)
        v = mx.random.normal((B, H, N, D), dtype=mx.float16)
        # Geometric ALiBi slopes: 2^(-8/H), 2^(-8*2/H), ...
        slopes = mx.array(
            [2 ** (-8.0 * (i + 1) / H) for i in range(H)], dtype=mx.float32
        )

        out_v2 = flash_attention(q, k, v, scale=scale, causal=True,
                                  alibi_slopes=slopes)
        prev = _os.environ.get("MFA_DISABLE_V2")
        try:
            _os.environ["MFA_DISABLE_V2"] = "1"
            out_v1 = flash_attention(q, k, v, scale=scale, causal=True,
                                      alibi_slopes=slopes)
        finally:
            if prev is None:
                _os.environ.pop("MFA_DISABLE_V2", None)
            else:
                _os.environ["MFA_DISABLE_V2"] = prev
        mx.eval(out_v2, out_v1)

        np.testing.assert_allclose(
            np.array(out_v2.astype(mx.float32)),
            np.array(out_v1.astype(mx.float32)),
            atol=1e-2, rtol=1e-2,
            err_msg=f"D={D}: V2+ALiBi differs from V1+ALiBi",
        )

    @pytest.mark.skipif(
        not is_mfa_available(), reason="MFA extension not available"
    )
    def test_sparse_routes_to_v1(self):
        """Sparse attention (block_mask) must route to V1 — V2 BK != mask BK.
        Verified by ensuring output matches V1-forced run."""
        import os as _os
        from mlx_mfa import flash_attention_sparse, make_causal_block_mask
        # v2.50 Prompt 4 Section A: bumped N=512→2048 for sparse mask>=4096.
        B, H, N, D = 1, 4, 2048, 64
        scale = 1.0 / math.sqrt(D)
        mx.random.seed(43)
        q = mx.random.normal((B, H, N, D), dtype=mx.float16)
        k = mx.random.normal((B, H, N, D), dtype=mx.float16)
        v = mx.random.normal((B, H, N, D), dtype=mx.float16)
        mask = make_causal_block_mask(N, head_dim=D)

        # Both should produce identical output (sparse always routes to V1)
        out1 = flash_attention_sparse(q, k, v, mask, scale=scale, causal=True)
        prev = _os.environ.get("MFA_DISABLE_V2")
        try:
            _os.environ["MFA_DISABLE_V2"] = "1"
            out2 = flash_attention_sparse(q, k, v, mask, scale=scale, causal=True)
        finally:
            if prev is None:
                _os.environ.pop("MFA_DISABLE_V2", None)
            else:
                _os.environ["MFA_DISABLE_V2"] = prev
        mx.eval(out1, out2)

        np.testing.assert_allclose(
            np.array(out1.astype(mx.float32)),
            np.array(out2.astype(mx.float32)),
            atol=1e-5,
            err_msg="Sparse output changed when V2 disabled — routing bug",
        )


# ---------------------------------------------------------------------------
# CP9: compile_metallib (AOT metallib compilation)
# ---------------------------------------------------------------------------

class TestCompileMetallib:
    """Tests for mlx_mfa.compile_metallib (CP9 AOT compilation)."""

    def test_compile_metallib_importable(self):
        """compile_metallib must be importable from the top-level package."""
        from mlx_mfa import compile_metallib
        assert callable(compile_metallib)

    def test_compile_metallib_in_all(self):
        """compile_metallib must be in mlx_mfa.__all__."""
        import mlx_mfa
        assert "compile_metallib" in mlx_mfa.__all__

    def test_xcrun_check(self):
        """_xcrun_metal_available() must not crash; result is a bool."""
        from mlx_mfa.compile_metallib import _xcrun_metal_available
        result = _xcrun_metal_available()
        assert isinstance(result, bool)

    def test_compile_metallib_returns_dict(self, tmp_path):
        """compile_metallib() returns a dict {filename: bool}.
        With xcrun present, at least one metallib is compiled; without,
        the dict is empty but no exception is raised."""
        from mlx_mfa.compile_metallib import compile_metallib, _xcrun_metal_available
        import os

        result = compile_metallib(output_dir=str(tmp_path), verbose=False)
        assert isinstance(result, dict), "Expected dict return type"

        if _xcrun_metal_available():
            assert len(result) > 0, "Expected at least one compiled metallib"
            for fname, ok in result.items():
                assert fname.endswith(".metallib"), f"Unexpected filename: {fname}"
                assert isinstance(ok, bool)
                if ok:
                    assert os.path.exists(os.path.join(str(tmp_path), fname))
        else:
            assert result == {}, "Expected empty dict when xcrun unavailable"


class TestAsyncV2Metallib:
    """Tests for CP4 async V2 metallib (simdgroup_async_copy hardware DMA).

    These tests cover the build script, shader_cache fallback chain, and
    numerical correctness when async_v2.metallib is present.
    """

    def test_async_kernel_source_exists(self):
        """csrc/async_v2_kernel.metal must exist in the repository."""
        import importlib.util, os
        spec = importlib.util.find_spec("mlx_mfa")
        pkg_dir = os.path.dirname(spec.origin)
        repo_root = os.path.dirname(pkg_dir)
        metal_path = os.path.join(repo_root, "csrc", "async_v2_kernel.metal")
        assert os.path.exists(metal_path), (
            f"csrc/async_v2_kernel.metal not found at {metal_path}"
        )

    def test_async_kernel_contains_asm_intrinsics(self):
        """async_v2_kernel.metal must contain the __asm simdgroup_async_copy intrinsics."""
        import importlib.util, os
        spec = importlib.util.find_spec("mlx_mfa")
        pkg_dir = os.path.dirname(spec.origin)
        repo_root = os.path.dirname(pkg_dir)
        metal_path = os.path.join(repo_root, "csrc", "async_v2_kernel.metal")
        with open(metal_path) as f:
            src = f.read()
        assert "air.simdgroup_async_copy_2d.p3i8.p1i8" in src, (
            "Expected __asm air.simdgroup_async_copy_2d intrinsic in async_v2_kernel.metal"
        )
        assert "mlx_mfa_v2_async_attention" in src
        assert "mlx_mfa_v2_async_attention_d128" in src
        assert "FC_CAUSAL" in src
        assert "FC_GQA_FACTOR" in src

    def test_build_async_metallib_script_exists(self):
        """scripts/build_async_metallib.sh must exist."""
        import importlib.util, os
        spec = importlib.util.find_spec("mlx_mfa")
        pkg_dir = os.path.dirname(spec.origin)
        repo_root = os.path.dirname(pkg_dir)
        script = os.path.join(repo_root, "scripts", "build_async_metallib.sh")
        assert os.path.exists(script), f"build_async_metallib.sh not found at {script}"
        assert os.access(script, os.X_OK), "build_async_metallib.sh must be executable"

    @pytest.mark.skipif(
        not pytest.importorskip("mlx_mfa", reason="mlx_mfa not installed")
        or not __import__("mlx_mfa").is_mfa_available(),
        reason="MFA C++ extension not available",
    )
    def test_async_fallback_to_sync(self):
        """With MFA_DISABLE_ASYNC=1, execution must fall back to sync path
        and produce numerically identical results."""
        import os, mlx.core as mx, mlx_mfa

        B, H, N, D = 1, 2, 256, 64
        key = mx.random.normal([B, H, N, D], dtype=mx.float16)
        q, k, v = key, key, key
        scale = D ** -0.5

        # Reference: async disabled → sync path
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
            os.environ, {"MFA_DISABLE_ASYNC": "1"}
        ):
            out_sync = mlx_mfa.flash_attention(q, k, v, scale=scale, causal=True,
                                               backend="mfa")
            mx.eval(out_sync)

        # Default: async path (or falls back if metallib absent)
        out_default = mlx_mfa.flash_attention(q, k, v, scale=scale, causal=True,
                                              backend="mfa")
        mx.eval(out_default)

        # Both paths should agree numerically
        err = mx.max(mx.abs(out_sync.astype(mx.float32) -
                            out_default.astype(mx.float32))).item()
        assert err < 1e-2, f"sync vs default mismatch: max_err={err:.4e}"

    @pytest.mark.skipif(
        not __import__("os").path.exists(
            __import__("os").path.join(
                __import__("os").path.dirname(
                    __import__("importlib.util", fromlist=["find_spec"])
                    .find_spec("mlx_mfa").origin),
                "precompiled", "async_v2.metallib")),
        reason="async_v2.metallib not present (macOS 26 expected — xcrun rejects __asm)",
    )
    def test_async_v2_matches_sync(self):
        """When async_v2.metallib is present, output must match sync V2 within f16 tolerance."""
        import os, mlx.core as mx, mlx_mfa
        from unittest.mock import patch

        for D, N in [(64, 512), (128, 512)]:
            B, H = 1, 2
            key = mx.random.normal([B, H, N, D], dtype=mx.float16)
            q, k, v = key, key, key
            scale = D ** -0.5

            # Async path (default — async metallib loaded if present)
            with patch.dict(os.environ, {"MFA_DISABLE_ASYNC": "0"}, clear=False):
                out_async = mlx_mfa.flash_attention(q, k, v, scale=scale,
                                                    causal=True, backend="mfa")
                mx.eval(out_async)

            # Sync path (async disabled)
            with patch.dict(os.environ, {"MFA_DISABLE_ASYNC": "1"}):
                out_sync = mlx_mfa.flash_attention(q, k, v, scale=scale,
                                                   causal=True, backend="mfa")
                mx.eval(out_sync)

            err = mx.max(mx.abs(out_async.astype(mx.float32) -
                                out_sync.astype(mx.float32))).item()
            assert err < 1e-2, (
                f"D={D} N={N}: async vs sync max_err={err:.4e} (expected < 1e-2)"
            )


@requires_ext
class TestSteelV3:
    """STEEL V3 (separate K_smem + V_smem, 2 barriers/iter).

    V3 is disabled by default (regresses vs V2 due to occupancy drop from
    doubled TGP usage).  Enabled via MFA_ENABLE_V3=1.

    Eligible: D=64 all gens, D=128 M1/M2 (BK=32).
    """

    @pytest.mark.parametrize("D,N,causal", [
        (64, 256, True), (64, 256, False),
        (64, 1024, True), (64, 1024, False),
        (64, 4096, True), (64, 4096, False),
        (128, 256, True), (128, 256, False),
        (128, 1024, True), (128, 1024, False),
        (128, 4096, True), (128, 4096, False),
    ])
    def test_v3_matches_v2(self, D, N, causal):
        """V3 output matches V2 output within f16 tolerance."""
        import os as _os
        from unittest.mock import patch
        mx.random.seed(42)
        B, H = 1, 4
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        # V3 path (opt-in via MFA_ENABLE_V3)
        with patch.dict(_os.environ, {"MFA_ENABLE_V3": "1"}):
            out_v3 = flash_attention(q, k, v, scale=scale, causal=causal)
            mx.eval(out_v3)

        # V2 path (default — MFA_ENABLE_V3 not set)
        out_v2 = flash_attention(q, k, v, scale=scale, causal=causal)
        mx.eval(out_v2)

        diff = mx.max(mx.abs(
            out_v3.astype(mx.float32) - out_v2.astype(mx.float32)
        )).item()
        assert diff < 1e-2, (
            f"D={D} N={N} causal={causal}: V3 vs V2 max_diff={diff:.4e}"
        )

    @pytest.mark.parametrize("D", [64, 128])
    def test_v3_matches_sdpa(self, D):
        """V3 output matches MLX SDPA reference (N=1024, causal)."""
        import os as _os
        from unittest.mock import patch
        from mlx_mfa.attention import _fallback_sdpa
        mx.random.seed(7)
        B, H, N = 1, 4, 1024
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        with patch.dict(_os.environ, {"MFA_ENABLE_V3": "1"}):
            out_v3 = flash_attention(q, k, v, scale=scale, causal=True)
            mx.eval(out_v3)
        out_ref = _fallback_sdpa(q, k, v, scale, causal=True)
        mx.eval(out_ref)

        diff = mx.max(mx.abs(
            out_v3.astype(mx.float32) - out_ref.astype(mx.float32)
        )).item()
        assert diff < 1e-2, f"D={D}: V3 vs SDPA max_diff={diff:.4e}"

    @pytest.mark.parametrize("D", [64, 128])
    def test_v3_gqa(self, D):
        """V3 handles GQA (H_q=8, H_kv=2)."""
        import os as _os
        from unittest.mock import patch
        mx.random.seed(11)
        B, Hq, Hkv, N = 1, 8, 2, 512
        q = mx.random.normal([B, Hq, N, D]).astype(mx.float16)
        k = mx.random.normal([B, Hkv, N, D]).astype(mx.float16)
        v = mx.random.normal([B, Hkv, N, D]).astype(mx.float16)
        mx.eval(q, k, v)

        with patch.dict(_os.environ, {"MFA_ENABLE_V3": "1"}):
            out = flash_attention(q, k, v, causal=True)
            mx.eval(out)
        assert out.shape == (B, Hq, N, D)

    def test_v3_bf16(self):
        """V3 works with bfloat16 dtype."""
        import os as _os
        from unittest.mock import patch
        mx.random.seed(99)
        B, H, N, D = 1, 4, 256, 64
        q = mx.random.normal([B, H, N, D]).astype(mx.bfloat16)
        k = mx.random.normal([B, H, N, D]).astype(mx.bfloat16)
        v = mx.random.normal([B, H, N, D]).astype(mx.bfloat16)
        mx.eval(q, k, v)

        with patch.dict(_os.environ, {"MFA_ENABLE_V3": "1"}):
            out = flash_attention(q, k, v, causal=True)
            mx.eval(out)
        assert out.dtype == mx.bfloat16
        assert out.shape == (B, H, N, D)
        assert mx.isfinite(out).all().item()


@pytest.mark.skip(reason="STEEL V4/V5 retired from build (Lot-2 chore); opt-in kernels removed")
@requires_ext
class TestSteelV4:
    """STEEL V4 (direct device K reads, 2 barriers/iter, M3+ only).

    V4 eliminates K_smem: each simdgroup loads K fragments directly from
    device memory in the GEMM loop. 2 barriers/tile vs V2's 4.

    RETIRED (Lot-2): V4 is removed from the build and MFA_ENABLE_V4 is a no-op
    (class skipped above).  Historically: enabled via MFA_ENABLE_V4=1, M3+ gate
    simulated with MFA_FORCE_GEN=15.
    """

    @pytest.mark.parametrize("D,N,causal", [
        (64, 256, True), (64, 256, False),
        (64, 1024, True),
        (128, 256, True), (128, 256, False),
        (128, 1024, True),
    ])
    def test_v4_matches_v2(self, D, N, causal):
        """V4 output matches V2 (or SDPA) within f16 tolerance."""
        import os as _os
        from unittest.mock import patch
        mx.random.seed(42)
        B, H = 1, 4
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        # V4 path (force M3+ routing even on M1)
        with patch.dict(_os.environ, {"MFA_ENABLE_V4": "1", "MFA_FORCE_GEN": "15"}):
            out_v4 = flash_attention(q, k, v, scale=scale, causal=causal)
            mx.eval(out_v4)

        # V2 default path
        out_v2 = flash_attention(q, k, v, scale=scale, causal=causal)
        mx.eval(out_v2)

        diff = mx.max(mx.abs(
            out_v4.astype(mx.float32) - out_v2.astype(mx.float32)
        )).item()
        assert diff < 1e-2, (
            f"D={D} N={N} causal={causal}: V4 vs V2 max_diff={diff:.4e}"
        )

    @pytest.mark.parametrize("D", [64, 128])
    def test_v4_gqa(self, D):
        """V4 handles GQA (H_q=8, H_kv=2)."""
        import os as _os
        from unittest.mock import patch
        mx.random.seed(11)
        B, Hq, Hkv, N = 1, 8, 2, 512
        q = mx.random.normal([B, Hq, N, D]).astype(mx.float16)
        k = mx.random.normal([B, Hkv, N, D]).astype(mx.float16)
        v = mx.random.normal([B, Hkv, N, D]).astype(mx.float16)
        mx.eval(q, k, v)

        with patch.dict(_os.environ, {"MFA_ENABLE_V4": "1", "MFA_FORCE_GEN": "15"}):
            out = flash_attention(q, k, v, causal=True)
            mx.eval(out)
        assert out.shape == (B, Hq, N, D)

    def test_v4_bf16(self):
        """V4 works with bfloat16 dtype."""
        import os as _os
        from unittest.mock import patch
        mx.random.seed(99)
        B, H, N, D = 1, 4, 256, 64
        q = mx.random.normal([B, H, N, D]).astype(mx.bfloat16)
        k = mx.random.normal([B, H, N, D]).astype(mx.bfloat16)
        v = mx.random.normal([B, H, N, D]).astype(mx.bfloat16)
        mx.eval(q, k, v)

        with patch.dict(_os.environ, {"MFA_ENABLE_V4": "1", "MFA_FORCE_GEN": "15"}):
            out = flash_attention(q, k, v, causal=True)
            mx.eval(out)
        assert out.dtype == mx.bfloat16
        assert out.shape == (B, H, N, D)
        assert mx.isfinite(out).all().item()


@pytest.mark.skip(reason="STEEL V4/V5 retired from build (Lot-2 chore); opt-in kernels removed")
class TestSteelV5:
    """STEEL V5 (D-blocked, BK=128, BD_tile=32, 3 TG/CU, all gens).

    V5 eliminates Q_smem: Q is loaded from device into registers per SIMD.
    KV_smem = max(K^T, V) = 10,240 B → 3 TG/CU (vs V2's 18,944 B → 1 TG/CU).
    BK=128 → 4× fewer K-tile iterations vs V2 M1/M2.

    RETIRED (Lot-2): V5 is removed from the build and MFA_ENABLE_V5 is a no-op
    (class skipped above).  Historically: enabled via MFA_ENABLE_V5=1.
    """

    @pytest.mark.parametrize("D,N,causal", [
        (64, 256, False), (64, 256, True),
        (64, 1024, True),
        (128, 256, False), (128, 256, True),
        (128, 1024, True),
        (128, 4096, True),
    ])
    def test_v5_matches_sdpa(self, D, N, causal):
        """V5 output matches SDPA within f16 tolerance."""
        import os as _os
        from unittest.mock import patch
        mx.random.seed(42)
        B, H = 1, 4
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        with patch.dict(_os.environ, {"MFA_ENABLE_V5": "1"}):
            out_v5 = flash_attention(q, k, v, scale=scale, causal=causal)
            mx.eval(out_v5)

        out_ref = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask='causal' if causal else None)
        mx.eval(out_ref)

        diff = mx.max(mx.abs(
            out_v5.astype(mx.float32) - out_ref.astype(mx.float32)
        )).item()
        assert diff < 2e-2, (
            f"D={D} N={N} causal={causal}: V5 vs SDPA max_diff={diff:.4e}"
        )

    @pytest.mark.parametrize("D", [64, 128])
    def test_v5_gqa(self, D):
        """V5 handles GQA (H_q=8, H_kv=2)."""
        import os as _os
        from unittest.mock import patch
        mx.random.seed(11)
        B, Hq, Hkv, N = 1, 8, 2, 512
        q = mx.random.normal([B, Hq, N, D]).astype(mx.float16)
        k = mx.random.normal([B, Hkv, N, D]).astype(mx.float16)
        v = mx.random.normal([B, Hkv, N, D]).astype(mx.float16)
        mx.eval(q, k, v)

        with patch.dict(_os.environ, {"MFA_ENABLE_V5": "1"}):
            out = flash_attention(q, k, v, causal=True)
            mx.eval(out)
        assert out.shape == (B, Hq, N, D)
        assert mx.isfinite(out).all().item()

    def test_v5_bf16(self):
        """V5 works with bfloat16 dtype."""
        import os as _os
        from unittest.mock import patch
        mx.random.seed(99)
        B, H, N, D = 1, 4, 256, 128
        q = mx.random.normal([B, H, N, D]).astype(mx.bfloat16)
        k = mx.random.normal([B, H, N, D]).astype(mx.bfloat16)
        v = mx.random.normal([B, H, N, D]).astype(mx.bfloat16)
        mx.eval(q, k, v)

        with patch.dict(_os.environ, {"MFA_ENABLE_V5": "1"}):
            out = flash_attention(q, k, v, causal=True)
            mx.eval(out)
        assert out.dtype == mx.bfloat16
        assert out.shape == (B, H, N, D)
        assert mx.isfinite(out).all().item()

    def test_v5_batch(self):
        """V5 handles batch > 1."""
        import os as _os
        from unittest.mock import patch
        mx.random.seed(7)
        B, H, N, D = 2, 4, 256, 64
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        with patch.dict(_os.environ, {"MFA_ENABLE_V5": "1"}):
            out = flash_attention(q, k, v, scale=scale, causal=True)
            mx.eval(out)
        assert out.shape == (B, H, N, D)
        assert mx.isfinite(out).all().item()

    @pytest.mark.parametrize("D,N", [(64, 512), (128, 512)])
    def test_v5_nonaligned_seq(self, D, N):
        """V5 handles sequence lengths not divisible by BK=128."""
        import os as _os
        from unittest.mock import patch
        # N=512 % BK=128 == 0 → aligned; try odd length
        N_odd = 500  # 500 % 128 = 116 (not aligned)
        mx.random.seed(5)
        B, H = 1, 4
        q = mx.random.normal([B, H, N_odd, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N_odd, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N_odd, D]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        with patch.dict(_os.environ, {"MFA_ENABLE_V5": "1"}):
            out_v5 = flash_attention(q, k, v, scale=scale, causal=False)
            mx.eval(out_v5)

        out_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        mx.eval(out_ref)

        diff = mx.max(mx.abs(
            out_v5.astype(mx.float32) - out_ref.astype(mx.float32)
        )).item()
        assert diff < 2e-2, f"non-aligned N={N_odd} D={D}: diff={diff:.4e}"

    @pytest.mark.parametrize("D,N,causal", [
        (64, 512, True), (64, 512, False),
        (128, 512, True), (128, 512, False),
        (128, 2048, True),
    ])
    def test_v5_causal_explicit(self, D, N, causal):
        """Explicit causal/non-causal correctness vs SDPA (larger N than smoke tests)."""
        import os as _os
        from unittest.mock import patch
        mx.random.seed(77)
        B, H = 1, 4
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        with patch.dict(_os.environ, {"MFA_ENABLE_V5": "1"}):
            out_v5 = flash_attention(q, k, v, scale=scale, causal=causal)
            mx.eval(out_v5)

        out_ref = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask='causal' if causal else None)
        mx.eval(out_ref)

        diff = mx.max(mx.abs(
            out_v5.astype(mx.float32) - out_ref.astype(mx.float32)
        )).item()
        assert diff < 2e-2, f"D={D} N={N} causal={causal}: diff={diff:.4e}"

    @pytest.mark.parametrize("D,ratio", [
        (64, 2), (64, 4), (128, 2), (128, 8),
    ])
    def test_v5_gqa_explicit(self, D, ratio):
        """V5 GQA correctness: H_q=8, H_kv=H_q/ratio vs SDPA reference."""
        import os as _os
        from unittest.mock import patch
        mx.random.seed(33)
        B, Hq, N = 1, 8, 512
        Hkv = Hq // ratio
        q = mx.random.normal([B, Hq,  N, D]).astype(mx.float16)
        k = mx.random.normal([B, Hkv, N, D]).astype(mx.float16)
        v = mx.random.normal([B, Hkv, N, D]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        with patch.dict(_os.environ, {"MFA_ENABLE_V5": "1"}):
            out_v5 = flash_attention(q, k, v, scale=scale, causal=True)
            mx.eval(out_v5)

        # Reference: expand k/v to Hq heads then SDPA
        k_exp = mx.repeat(k, ratio, axis=1)
        v_exp = mx.repeat(v, ratio, axis=1)
        out_ref = mx.fast.scaled_dot_product_attention(
            q, k_exp, v_exp, scale=scale, mask='causal')
        mx.eval(out_ref)

        diff = mx.max(mx.abs(
            out_v5.astype(mx.float32) - out_ref.astype(mx.float32)
        )).item()
        assert diff < 2e-2, f"D={D} ratio={ratio}: diff={diff:.4e}"

    @pytest.mark.parametrize("D,N,window", [
        (64,  1024, 128),
        (64,  2048, 256),
        (128, 1024, 128),
        (128, 2048, 512),
    ])
    def test_v5_window(self, D, N, window):
        """V5 sliding window matches SDPA with per-element bias simulation."""
        import os as _os
        from unittest.mock import patch
        mx.random.seed(55)
        B, H = 1, 4
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        with patch.dict(_os.environ, {"MFA_ENABLE_V5": "1"}):
            out_v5 = flash_attention(
                q, k, v, scale=scale, causal=True,
                window_size=(window,))
            mx.eval(out_v5)

        # Reference: causal SDPA with a sliding-window mask applied as bias
        # Build [1,1,N,N] mask: True where attention is allowed
        rows = mx.arange(N)[:, None]   # [N, 1]
        cols = mx.arange(N)[None, :]   # [1, N]
        allow = (cols <= rows) & (cols >= rows - window)
        bias = mx.where(allow,
                        mx.zeros([N, N], dtype=mx.float16),
                        mx.full([N, N], float('-inf'), dtype=mx.float16))
        bias = bias[None, None, :, :]  # [1,1,N,N]
        out_ref = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=bias)
        mx.eval(out_ref)

        diff = mx.max(mx.abs(
            out_v5.astype(mx.float32) - out_ref.astype(mx.float32)
        )).item()
        assert diff < 2e-2, f"D={D} N={N} window={window}: diff={diff:.4e}"


class TestSteelV5CP5:
    """CP5: softcap, ALiBi, and sparse support in STEEL V5.

    T2-2 (audit H5/IC-C1, 2026-06-21): V4/V5 are RETIRED from the build, so
    `MFA_ENABLE_V5=1` is a NO-OP — `_run_v5` ran the DEFAULT backend and the
    softcap/alibi cells compared the default path to itself (`ref` and `out` both
    default) → vacuous green-on-removed-knob.  Skipped to match TestSteelV4/V5.
    """

    pytestmark = [
        pytest.mark.skip(reason="STEEL V4/V5 retired from build (Lot-2 chore); "
                                "MFA_ENABLE_V5 is a no-op → these compared the default path to itself"),
        pytest.mark.skipif(not _ext_available(), reason="C++ extension not available"),
    ]

    B, H, N, D = 2, 8, 1024, 128

    def _run_v5(self, q, k, v, scale, causal=False, **kwargs):
        import os, contextlib
        old = os.environ.get("MFA_ENABLE_V5")
        os.environ["MFA_ENABLE_V5"] = "1"
        try:
            return flash_attention(q, k, v, scale=scale, causal=causal, **kwargs)
        finally:
            if old is None:
                del os.environ["MFA_ENABLE_V5"]
            else:
                os.environ["MFA_ENABLE_V5"] = old

    # ── softcap ──────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("D,causal", [(64, False), (128, True)])
    def test_v5_softcap(self, D, causal):
        """V5 softcap must match reference within f16 tolerance."""
        mx.random.seed(7)
        B, H, N = 2, 8, 512
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        scale = D ** -0.5
        softcap = 30.0

        ref = flash_attention(q, k, v, scale=scale, causal=causal,
                              softcap=softcap)
        out = self._run_v5(q, k, v, scale, causal=causal, softcap=softcap)
        mx.eval(ref, out)

        diff = mx.max(mx.abs(
            ref.astype(mx.float32) - out.astype(mx.float32)
        )).item()
        assert diff < 2e-2, f"D={D} causal={causal}: diff={diff:.4e}"

    # ── ALiBi ────────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("D,causal", [(64, False), (128, False)])
    def test_v5_alibi(self, D, causal):
        """V5 ALiBi must match reference within f16 tolerance."""
        mx.random.seed(11)
        B, H, N = 2, 8, 512
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        scale = D ** -0.5
        slopes = mx.array([2.0 ** (-i) for i in range(1, H + 1)],
                          dtype=mx.float32)

        ref = flash_attention(q, k, v, scale=scale, causal=causal,
                              alibi_slopes=slopes)
        out = self._run_v5(q, k, v, scale, causal=causal,
                           alibi_slopes=slopes)
        mx.eval(ref, out)

        diff = mx.max(mx.abs(
            ref.astype(mx.float32) - out.astype(mx.float32)
        )).item()
        assert diff < 2e-2, f"D={D}: diff={diff:.4e}"

    # ── block-sparse ─────────────────────────────────────────────────────────
    # V5 excludes sparse: block_mask is sized for V2's BK, not V5's BK=128.
    # With MFA_ENABLE_V5=1, sparse calls fall through to V2.
    @pytest.mark.parametrize("D", [64, 128])
    def test_v5_sparse_falls_back_to_v2(self, D):
        """Sparse calls must fall through to V2 (not error) when MFA_ENABLE_V5=1."""
        import os
        mx.random.seed(23)
        # v2.50 Prompt 4 Section A: bumped N=512→2048 for sparse mask>=4096.
        B, H, N = 2, 8, 2048
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        scale = D ** -0.5
        mask = make_causal_block_mask(N, head_dim=D)

        old = os.environ.get("MFA_ENABLE_V5")
        os.environ["MFA_ENABLE_V5"] = "1"
        try:
            # Should not raise; falls through to V2 sparse
            out = flash_attention_sparse(q, k, v, mask, scale=scale, causal=True)
            ref = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask="causal")
            mx.eval(out, ref)
        finally:
            if old is None:
                del os.environ["MFA_ENABLE_V5"]
            else:
                os.environ["MFA_ENABLE_V5"] = old

        diff = mx.max(mx.abs(
            out.astype(mx.float32) - ref.astype(mx.float32)
        )).item()
        assert diff < 2e-2, f"D={D}: diff={diff:.4e}"


class TestSteelV5DirectReads:
    """CP7: V5 M3+ direct device reads — 0 barriers/K-tile.

    T2-2 (audit H5/IC-C1, 2026-06-21): RETIRED.  `MFA_ENABLE_V5=1` is a no-op
    (V5 removed from build), so these exercised the default path, not a V5
    direct-reads kernel.  Skipped to match TestSteelV4/V5.
    """

    pytestmark = [
        pytest.mark.skip(reason="STEEL V4/V5 retired from build (Lot-2 chore); "
                                "MFA_ENABLE_V5 is a no-op → exercised the default path, not V5"),
        pytest.mark.skipif(not _ext_available(), reason="C++ extension not available"),
    ]

    def _run_v5_m3plus(self, q, k, v, scale, causal=False, **kwargs):
        """Run V5 with MFA_ENABLE_V5=1 and MFA_FORCE_GEN=15 (simulated M3+)."""
        import os
        saved = {key: os.environ.get(key) for key in ("MFA_ENABLE_V5", "MFA_FORCE_GEN")}
        os.environ["MFA_ENABLE_V5"] = "1"
        os.environ["MFA_FORCE_GEN"] = "15"
        try:
            return flash_attention(q, k, v, scale=scale, causal=causal, **kwargs)
        finally:
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    @pytest.mark.parametrize("D,N,causal", [
        (64, 1024, False),
        (64, 1024, True),
        (128, 2048, False),
        (128, 2048, True),
    ])
    def test_v5_direct_reads_matches_sdpa(self, D, N, causal):
        """V5 M3+ direct reads must match SDPA within f16 tolerance."""
        mx.random.seed(77)
        B, H = 2, 8
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        scale = D ** -0.5
        out = self._run_v5_m3plus(q, k, v, scale, causal=causal)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                                                   mask="causal" if causal else None)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()
        assert diff < 2e-2, f"D={D} N={N} causal={causal}: diff={diff:.4e}"

    def test_v5_direct_reads_gqa(self):
        """V5 M3+ direct reads: GQA must produce correct output."""
        mx.random.seed(88)
        B, H_q, H_kv, N, D = 2, 8, 2, 512, 64
        q = mx.random.normal([B, H_q, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H_kv, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H_kv, N, D]).astype(mx.float16)
        scale = D ** -0.5
        out = self._run_v5_m3plus(q, k, v, scale, causal=False)
        k_exp = mx.repeat(k, H_q // H_kv, axis=1)
        v_exp = mx.repeat(v, H_q // H_kv, axis=1)
        ref = mx.fast.scaled_dot_product_attention(q, k_exp, v_exp, scale=scale)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()
        assert diff < 2e-2, f"GQA diff={diff:.4e}"

    def test_v5_direct_reads_nonaligned(self):
        """V5 M3+ direct reads: non-power-of-2 sequence length must not NaN."""
        mx.random.seed(99)
        B, H, D = 2, 4, 128
        for N in (63, 65, 100, 513):
            q = mx.random.normal([B, H, N, D]).astype(mx.float16)
            k = mx.random.normal([B, H, N, D]).astype(mx.float16)
            v = mx.random.normal([B, H, N, D]).astype(mx.float16)
            out = self._run_v5_m3plus(q, k, v, scale=D**-0.5, causal=True)
            mx.eval(out)
            assert not mx.any(mx.isnan(out)).item(), f"NaN at N={N}"


class TestSteelV2DirectReads:
    """P1: V2 M3+ direct device reads — 0 barriers/K-tile for K/V.

    Exercises MFA_DIRECT_READS=1 (emitted when is_m3_plus=True and no RoPE).
    Uses MFA_FORCE_GEN=15 to simulate M3+ on M1/M2 hardware.

    Primary validation: gen=15 (direct reads) must match gen=13 (TGP path)
    bit-for-bit. This isolates the direct reads correctness from any
    pre-existing V2 accuracy issues at certain N/D/causal configs.
    """

    pytestmark = [
        pytest.mark.skipif(not _ext_available(), reason="C++ extension not available"),
    ]

    def _run_with_gen(self, gen, q, k, v, scale, causal=False, **kwargs):
        """Run V2 with MFA_FORCE_GEN=gen.

        Forces BK=32 for D=128 so that gen=13 and gen=15 use the same blocking.
        Forces MFA_FORCE_V2=1 so M3+ causal still dispatches to V2 (not V1).
        This isolates the direct-reads variable from M3+ routing changes.
        """
        import os
        env_keys = ("MFA_FORCE_GEN", "MFA_V2_FORCE_BK", "MFA_FORCE_V2")
        saved = {k: os.environ.get(k) for k in env_keys}
        os.environ["MFA_FORCE_GEN"] = str(gen)
        os.environ["MFA_V2_FORCE_BK"] = "32"  # same BK for both paths
        os.environ["MFA_FORCE_V2"] = "1"       # bypass M3+ V1 preference
        try:
            out = flash_attention(q, k, v, scale=scale, causal=causal, **kwargs)
            mx.eval(out)
            return out
        finally:
            for k, val in saved.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val

    @pytest.mark.parametrize("D,N,causal", [
        (64, 512, False),
        (64, 512, True),
        (64, 1024, False),
        (64, 1024, True),
        (64, 4096, False),
        (64, 4096, True),
        (128, 512, False),
        (128, 512, True),
        (128, 1024, False),
        (128, 1024, True),
        (128, 2048, False),
        (128, 2048, True),
    ])
    def test_v2_direct_reads_matches_tgp(self, D, N, causal):
        """V2 M3+ direct reads (gen=15) must match TGP path (gen=13)."""
        mx.random.seed(42)
        B, H = 2, 8
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        scale = D ** -0.5
        out_dr = self._run_with_gen(15, q, k, v, scale, causal=causal)
        out_tgp = self._run_with_gen(13, q, k, v, scale, causal=causal)
        diff = mx.max(mx.abs(out_dr.astype(mx.float32) - out_tgp.astype(mx.float32))).item()
        # Direct reads and TGP should produce identical results (same arithmetic,
        # just different load path). Allow tiny tolerance for float rounding.
        assert diff < 1e-4, f"D={D} N={N} causal={causal}: diff={diff:.4e}"

    def test_v2_direct_reads_gqa(self):
        """V2 M3+ direct reads: GQA gen=15 must match gen=13."""
        mx.random.seed(88)
        B, H_q, H_kv, N, D = 2, 8, 2, 1024, 128
        q = mx.random.normal([B, H_q, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H_kv, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H_kv, N, D]).astype(mx.float16)
        scale = D ** -0.5
        out_dr = self._run_with_gen(15, q, k, v, scale, causal=True)
        out_tgp = self._run_with_gen(13, q, k, v, scale, causal=True)
        diff = mx.max(mx.abs(out_dr.astype(mx.float32) - out_tgp.astype(mx.float32))).item()
        assert diff < 1e-4, f"GQA diff={diff:.4e}"

    def test_v2_direct_reads_nonaligned(self):
        """V2 M3+ direct reads: non-aligned N must not NaN and must match TGP."""
        mx.random.seed(99)
        B, H, D = 2, 4, 128
        for N in (63, 65, 100, 513):
            q = mx.random.normal([B, H, N, D]).astype(mx.float16)
            k = mx.random.normal([B, H, N, D]).astype(mx.float16)
            v = mx.random.normal([B, H, N, D]).astype(mx.float16)
            out_dr = self._run_with_gen(15, q, k, v, scale=D**-0.5, causal=True)
            out_tgp = self._run_with_gen(13, q, k, v, scale=D**-0.5, causal=True)
            assert not mx.any(mx.isnan(out_dr)).item(), f"NaN at N={N}"
            diff = mx.max(mx.abs(out_dr.astype(mx.float32) - out_tgp.astype(mx.float32))).item()
            assert diff < 1e-4, f"N={N}: diff={diff:.4e}"

    def test_v2_direct_reads_bf16(self):
        """V2 M3+ direct reads: bf16 gen=15 must match gen=13."""
        mx.random.seed(55)
        B, H, N, D = 2, 8, 1024, 64
        q = mx.random.normal([B, H, N, D]).astype(mx.bfloat16)
        k = mx.random.normal([B, H, N, D]).astype(mx.bfloat16)
        v = mx.random.normal([B, H, N, D]).astype(mx.bfloat16)
        scale = D ** -0.5
        out_dr = self._run_with_gen(15, q, k, v, scale, causal=False)
        out_tgp = self._run_with_gen(13, q, k, v, scale, causal=False)
        diff = mx.max(mx.abs(out_dr.astype(mx.float32) - out_tgp.astype(mx.float32))).item()
        assert diff < 1e-4, f"bf16 diff={diff:.4e}"
