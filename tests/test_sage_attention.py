"""Tests for SageAttention (Track KC).

Test classes:
  TestQuantizeUtils     — always run; mlx_mfa.quantize utilities
  TestSageAPI           — always run; sage_attention() interface
  TestSageKernel        — skipped without C++ extension; numerical correctness
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import (
    sage_attention,
    quantize_per_block,
    dequantize,
    smooth_k,
    sage_output_correction,
    sage_block_sizes,
    get_supported_configs,
)
from mlx_mfa.attention import _ext_available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reference_sdpa(q, k, v, scale, causal=False):
    """Reference using MLX built-in SDPA."""
    mask = None
    if causal:
        N, S = q.shape[2], k.shape[2]
        mask = mx.triu(mx.full((N, S), float("-inf"), dtype=q.dtype), k=S - N + 1)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def rand_qkv(B, H, N, D, H_kv=None, dtype=mx.float16, seed=7):
    """Return fp16 q/k/v tensors with controlled randomness."""
    mx.random.seed(seed)
    H_kv = H_kv or H
    q = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    k = mx.random.normal(shape=(B, H_kv, N, D)).astype(dtype)
    v = mx.random.normal(shape=(B, H_kv, N, D)).astype(dtype)
    return q, k, v


# ---------------------------------------------------------------------------
# TestQuantizeUtils — always runs
# ---------------------------------------------------------------------------

class TestQuantizeUtils:
    """Unit tests for mlx_mfa.quantize utilities."""

    def test_quantize_roundtrip_shape(self):
        q = mx.random.normal(shape=(2, 4, 64, 128)).astype(mx.float16)
        BQ, _ = sage_block_sizes(128)
        q_int8, q_scale = quantize_per_block(q, BQ)
        assert q_int8.shape == (2, 4, 64, 128)
        assert q_int8.dtype == mx.int8
        NQ = math.ceil(64 / BQ)
        assert q_scale.shape == (2, 4, NQ, 1)
        assert q_scale.dtype == mx.float32

    def test_quantize_roundtrip_accuracy(self):
        """dequantize(quantize(x)) ~= x with small error."""
        mx.random.seed(99)
        x = mx.random.normal(shape=(1, 1, 32, 64)).astype(mx.float16)
        x_int8, x_scale = quantize_per_block(x, 32)
        x_deq = dequantize(x_int8, x_scale, 32)
        diff = np.abs(np.array(x.astype(mx.float32)) -
                      np.array(x_deq.astype(mx.float32)))
        absmax = float(mx.max(mx.abs(x.astype(mx.float32))))
        assert diff.max() < 0.02 * absmax, f"roundtrip error too large: {diff.max():.4f}"

    def test_quantize_non_multiple_seq(self):
        """Sequence length not a multiple of block_size should still work."""
        x = mx.random.normal(shape=(1, 2, 50, 64)).astype(mx.float16)
        x_int8, x_scale = quantize_per_block(x, 32)
        assert x_int8.shape == (1, 2, 50, 64)
        assert x_scale.shape == (1, 2, 2, 1)  # ceil(50/32)=2 blocks

    def test_smooth_k_shape(self):
        k = mx.random.normal(shape=(2, 4, 128, 64)).astype(mx.float16)
        k_smooth, k_mean = smooth_k(k)
        assert k_smooth.shape == k.shape
        assert k_mean.shape == (2, 4, 1, 64)
        assert k_mean.dtype == mx.float32

    def test_smooth_k_zero_mean(self):
        """k_smooth should have mean ~0 per channel."""
        k = mx.random.normal(shape=(1, 1, 64, 32)).astype(mx.float16)
        k_smooth, _ = smooth_k(k)
        mean_after = float(mx.mean(mx.abs(
            mx.mean(k_smooth.astype(mx.float32), axis=2)
        )))
        assert mean_after < 1e-3, f"channel mean not zeroed: {mean_after}"

    def test_sage_block_sizes(self):
        # V2 tile sizes: BK doubled vs V1 to match select_steel_v2_block_config
        assert sage_block_sizes(64)  == (32, 64)   # V2: was (32,32) in V1
        assert sage_block_sizes(128) == (32, 32)   # V2: was (32,16) in V1
        assert sage_block_sizes(256) == (16, 32)   # V2: was (32,16) in V1

    def test_dequantize_shape(self):
        B, H, N, D = 1, 2, 48, 64
        x = mx.zeros(shape=(B, H, N, D), dtype=mx.int8)
        scale = mx.ones(shape=(B, H, 2, 1), dtype=mx.float32)
        out = dequantize(x, scale, block_size=32)
        assert out.shape == (B, H, N, D)


# ---------------------------------------------------------------------------
# TestSageAPI — always runs
# ---------------------------------------------------------------------------

class TestSageAPI:
    """Interface tests for sage_attention()."""

    def test_output_shape(self):
        q, k, v = rand_qkv(1, 4, 128, 64)
        out = sage_attention(q, k, v)
        mx.eval(out)
        assert out.shape == (1, 4, 128, 64)

    def test_output_dtype_fp16(self):
        q, k, v = rand_qkv(1, 2, 64, 64)
        out = sage_attention(q, k, v)
        mx.eval(out)
        assert out.dtype == mx.float16

    def test_output_dtype_bf16(self):
        q, k, v = rand_qkv(1, 2, 64, 64, dtype=mx.bfloat16)
        out = sage_attention(q, k, v)
        mx.eval(out)
        assert out.dtype == mx.bfloat16

    def test_no_nans_basic(self):
        q, k, v = rand_qkv(1, 4, 256, 128)
        out = sage_attention(q, k, v, causal=False)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out.astype(mx.float32))))

    def test_no_nans_causal(self):
        q, k, v = rand_qkv(1, 4, 256, 128)
        out = sage_attention(q, k, v, causal=True)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out.astype(mx.float32))))

    def test_no_smooth_k(self):
        q, k, v = rand_qkv(1, 2, 64, 64)
        out = sage_attention(q, k, v, apply_smooth_k=False)
        mx.eval(out)
        assert out.shape == (1, 2, 64, 64)

    def test_get_supported_configs_sage(self):
        cfg = get_supported_configs()
        assert "sage_attention" in cfg["features"]
        if _ext_available():
            assert cfg["features"]["sage_attention"] is True
            assert cfg["kernel_types"] == 16


# ---------------------------------------------------------------------------
# TestSageKernel — extension required
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ext_available(), reason="C++ extension not available")
class TestSageKernel:
    """Numerical correctness of the SageAttention Metal kernel."""

    # Tolerance for int8 attention: max absolute error <= TOL.
    TOL = 0.30

    def _check(self, q, k, v, causal=False, atol=0.30):
        """Compare sage_attention vs reference fp16 SDPA."""
        scale = 1.0 / math.sqrt(q.shape[-1])
        ref = reference_sdpa(q, k, v, scale=scale, causal=causal)
        out = sage_attention(q, k, v, scale=scale, causal=causal)
        mx.eval(ref, out)
        ref_np = np.array(ref.astype(mx.float32))
        out_np = np.array(out.astype(mx.float32))
        max_err = np.abs(ref_np - out_np).max()
        assert max_err <= atol, (
            f"max_err={max_err:.4f} > atol={atol:.4f}  "
            f"shape={q.shape} causal={causal}"
        )

    def test_d64_noncausal(self):
        q, k, v = rand_qkv(1, 4, 256, 64, seed=1)
        self._check(q, k, v, causal=False)

    def test_d64_causal(self):
        q, k, v = rand_qkv(1, 4, 256, 64, seed=2)
        # Causal: wider tolerance due to int8 variance and Metal non-determinism
        self._check(q, k, v, causal=True, atol=0.50)

    def test_d128_noncausal(self):
        q, k, v = rand_qkv(1, 4, 256, 128, seed=3)
        self._check(q, k, v, causal=False)

    def test_d128_causal(self):
        q, k, v = rand_qkv(1, 4, 256, 128, seed=4)
        # Causal: wider tolerance due to int8 variance and Metal non-determinism
        self._check(q, k, v, causal=True, atol=0.50)

    def test_d128_longer_seq(self):
        """Longer sequence stresses the K-loop."""
        q, k, v = rand_qkv(1, 4, 1024, 128, seed=5)
        self._check(q, k, v, causal=False)

    def test_gqa_2to1(self):
        """GQA ratio 2:1 (H_q=4, H_kv=2)."""
        mx.random.seed(10)
        B, H, H_kv, N, D = 1, 4, 2, 256, 64
        q = mx.random.normal(shape=(B, H, N, D)).astype(mx.float16)
        k = mx.random.normal(shape=(B, H_kv, N, D)).astype(mx.float16)
        v = mx.random.normal(shape=(B, H_kv, N, D)).astype(mx.float16)
        k_exp = mx.repeat(k, 2, axis=1)
        v_exp = mx.repeat(v, 2, axis=1)
        scale = 1.0 / math.sqrt(D)
        ref = reference_sdpa(q, k_exp, v_exp, scale=scale)
        out = sage_attention(q, k, v, scale=scale)
        mx.eval(ref, out)
        max_err = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        assert max_err <= self.TOL, f"GQA max_err={max_err:.4f}"

    def test_batch_gt1(self):
        q, k, v = rand_qkv(2, 4, 256, 64, seed=6)
        self._check(q, k, v, causal=False)

    def test_no_smooth_correctness(self):
        """Without K smoothing, output should be finite."""
        q, k, v = rand_qkv(1, 4, 256, 128, seed=7)
        out = sage_attention(q, k, v, apply_smooth_k=False)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out.astype(mx.float32))))

    def test_output_finite_d256(self):
        """D=256 path should produce finite output."""
        q, k, v = rand_qkv(1, 2, 256, 256, seed=8)
        out = sage_attention(q, k, v, causal=True)
        mx.eval(out)
        out_np = np.array(out.astype(mx.float32))
        assert np.all(np.isfinite(out_np))
