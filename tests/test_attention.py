"""Tests for mlx-mfa flash_attention.

Correctness verified against mx.fast.scaled_dot_product_attention.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import flash_attention
from mlx_mfa.attention import _ext_available, _fallback_sdpa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reference_sdpa(q, k, v, scale, causal=False):
    """Reference SDPA using MLX built-in."""
    mask = None
    if causal:
        N, S = q.shape[2], k.shape[2]
        mask = mx.triu(mx.full((N, S), float("-inf")), k=S - N + 1)
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
        with pytest.raises(ValueError, match="4D"):
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
        np.testing.assert_allclose(
            np.array(out, dtype=np.float32),
            np.array(ref, dtype=np.float32),
            rtol=tol, atol=tol,
            err_msg=f"Dtype mismatch for {dtype}"
        )

    def test_long_sequence(self):
        """N=4096 should work without OOM."""
        q, k, v = random_qkv(1, 4, 4096, 128)
        out = flash_attention(q, k, v, scale=1.0 / math.sqrt(128))
        mx.eval(out)
        assert out.shape == (1, 4, 4096, 128)
