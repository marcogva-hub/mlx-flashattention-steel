"""Tests for native Metal attn_bias kernel (modes 1/2)."""

import unittest
import mlx.core as mx
import numpy as np

from mlx_mfa.attention import _classify_bias_shape, flash_attention


def _sdpa_ref(q, k, v, scale, causal=False, attn_bias=None):
    """Reference SDPA with explicit attn_bias addition."""
    mask = attn_bias
    if causal:
        N, S = q.shape[2], k.shape[2]
        causal_mask = mx.triu(
            mx.full((N, S), float("-inf"), dtype=q.dtype), k=S - N + 1
        )
        mask = causal_mask + mask if mask is not None else causal_mask
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


# ───────────────────────────────────────────────────────────────────
# Shape classification
# ───────────────────────────────────────────────────────────────────
class TestClassifyBiasShape(unittest.TestCase):
    """Unit tests for _classify_bias_shape()."""

    def _make(self, B, H, N, S):
        q = mx.zeros((B, H, N, 64))
        k = mx.zeros((B, H, S, 64))
        return q, k

    def test_mode1_broadcast(self):
        q, k = self._make(2, 8, 128, 256)
        bias = mx.zeros((1, 1, 1, 256))
        self.assertEqual(_classify_bias_shape(bias, q, k), 1)

    def test_mode2_per_head(self):
        q, k = self._make(2, 8, 128, 256)
        bias = mx.zeros((1, 8, 1, 256))
        self.assertEqual(_classify_bias_shape(bias, q, k), 2)

    def test_mode3_per_head_full(self):
        q, k = self._make(2, 8, 128, 256)
        bias = mx.zeros((1, 8, 128, 256))
        self.assertEqual(_classify_bias_shape(bias, q, k), 3)

    def test_mode0_full(self):
        q, k = self._make(2, 8, 128, 256)
        bias = mx.zeros((2, 8, 128, 256))
        self.assertEqual(_classify_bias_shape(bias, q, k), 0)

    def test_wrong_nkv(self):
        q, k = self._make(2, 8, 128, 256)
        bias = mx.zeros((1, 1, 1, 999))
        self.assertEqual(_classify_bias_shape(bias, q, k), -1)

    def test_wrong_ndim(self):
        q, k = self._make(2, 8, 128, 256)
        bias = mx.zeros((256,))
        self.assertEqual(_classify_bias_shape(bias, q, k), -1)


# ───────────────────────────────────────────────────────────────────
# Mode 1: [1,1,1,Nkv] per-KV broadcast bias
# ───────────────────────────────────────────────────────────────────
class TestBiasMode1(unittest.TestCase):
    """Native Metal kernel — mode 1 bias."""

    def _run(self, B, H, N, S, D, causal=False, dtype=mx.float16, atol=5e-2):
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(dtype)
        k = mx.random.normal((B, H, S, D)).astype(dtype)
        v = mx.random.normal((B, H, S, D)).astype(dtype)
        bias = mx.random.normal((1, 1, 1, S)).astype(dtype) * 0.1
        scale = 1.0 / (D ** 0.5)

        out = flash_attention(q, k, v, scale=scale, causal=causal, attn_bias=bias)
        ref = _sdpa_ref(q, k, v, scale=scale, causal=causal, attn_bias=bias)
        mx.eval(out, ref)

        out_np = np.array(out.astype(mx.float32))
        ref_np = np.array(ref.astype(mx.float32))
        max_err = float(np.max(np.abs(out_np - ref_np)))
        self.assertLess(max_err, atol, f"max_err={max_err}")
        self.assertEqual(out.shape, (B, H, N, D))

    def test_d64_non_causal(self):
        self._run(2, 8, 128, 128, 64, causal=False)

    def test_d128_causal(self):
        self._run(2, 8, 128, 128, 128, causal=True)

    def test_cross_attention(self):
        """N_q != N_kv (cross-attention). Wider tolerance for longer KV."""
        self._run(1, 4, 32, 256, 128, causal=False, atol=0.1)

    def test_bf16(self):
        self._run(1, 4, 64, 64, 128, dtype=mx.bfloat16)


# ───────────────────────────────────────────────────────────────────
# Mode 2: [1,H,1,Nkv] per-head per-KV bias
# ───────────────────────────────────────────────────────────────────
class TestBiasMode2(unittest.TestCase):
    """Native Metal kernel — mode 2 bias."""

    def _run(self, B, H, N, S, D, causal=False, H_kv=None):
        dtype = mx.float16
        mx.random.seed(42)
        H_kv = H_kv or H
        q = mx.random.normal((B, H, N, D)).astype(dtype)
        k = mx.random.normal((B, H_kv, S, D)).astype(dtype)
        v = mx.random.normal((B, H_kv, S, D)).astype(dtype)
        bias = mx.random.normal((1, H, 1, S)).astype(dtype) * 0.1
        scale = 1.0 / (D ** 0.5)

        out = flash_attention(q, k, v, scale=scale, causal=causal, attn_bias=bias)
        ref = _sdpa_ref(q, k, v, scale=scale, causal=causal, attn_bias=bias)
        mx.eval(out, ref)

        out_np = np.array(out.astype(mx.float32))
        ref_np = np.array(ref.astype(mx.float32))
        max_err = float(np.max(np.abs(out_np - ref_np)))
        self.assertLess(max_err, 5e-2, f"max_err={max_err}")

    def test_d64_non_causal(self):
        self._run(2, 8, 128, 128, 64)

    def test_d128_causal(self):
        self._run(2, 8, 128, 128, 128, causal=True)

    def test_gqa(self):
        """GQA: H=8, H_kv=2."""
        self._run(1, 8, 64, 64, 128, H_kv=2)


# ───────────────────────────────────────────────────────────────────
# Edge cases
# ───────────────────────────────────────────────────────────────────
class TestBiasEdgeCases(unittest.TestCase):

    def test_mode3_falls_back(self):
        """Mode 3 should still produce correct output via SDPA fallback."""
        B, H, N, S, D = 1, 4, 32, 64, 128
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        bias = mx.random.normal((1, H, N, S)).astype(mx.float16) * 0.1
        scale = 1.0 / (D ** 0.5)

        out = flash_attention(q, k, v, scale=scale, attn_bias=bias)
        ref = _sdpa_ref(q, k, v, scale=scale, attn_bias=bias)
        mx.eval(out, ref)
        max_err = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        self.assertLess(max_err, 1e-2)

    def test_zero_bias(self):
        """Zero bias should match no-bias output."""
        B, H, N, S, D = 1, 4, 64, 64, 128
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        scale = 1.0 / (D ** 0.5)

        bias = mx.zeros((1, 1, 1, S), dtype=mx.float16)
        out_bias = flash_attention(q, k, v, scale=scale, attn_bias=bias)
        out_none = flash_attention(q, k, v, scale=scale)
        mx.eval(out_bias, out_none)

        max_err = float(mx.max(mx.abs(
            out_bias.astype(mx.float32) - out_none.astype(mx.float32)
        )))
        self.assertLess(max_err, 1e-3)

    def test_output_shape(self):
        B, H, N, S, D = 2, 4, 64, 128, 64
        q = mx.zeros((B, H, N, D), dtype=mx.float16)
        k = mx.zeros((B, H, S, D), dtype=mx.float16)
        v = mx.zeros((B, H, S, D), dtype=mx.float16)
        bias = mx.zeros((1, 1, 1, S), dtype=mx.float16)
        out = flash_attention(q, k, v, attn_bias=bias)
        mx.eval(out)
        self.assertEqual(out.shape, (B, H, N, D))


# ───────────────────────────────────────────────────────────────────
# Application: token merging bias
# ───────────────────────────────────────────────────────────────────
class TestTokenMergingBias(unittest.TestCase):

    def test_suppress_last_quarter(self):
        """Suppress the last quarter of keys via negative bias (N=S)."""
        B, H, N, D = 1, 4, 128, 128
        S = N  # N=S avoids online-softmax precision cliff at N<S
        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, S, D)).astype(mx.float16)
        v = mx.random.normal((B, H, S, D)).astype(mx.float16)
        scale = 1.0 / (D ** 0.5)

        # Bias: 0 for first 3/4, -10 for last 1/4
        bias_np = np.zeros((1, 1, 1, S), dtype=np.float16)
        bias_np[0, 0, 0, 3 * S // 4 :] = -10.0
        bias = mx.array(bias_np)

        out = flash_attention(q, k, v, scale=scale, attn_bias=bias)
        ref = _sdpa_ref(q, k, v, scale=scale, attn_bias=bias)
        mx.eval(out, ref)

        max_err = float(mx.max(mx.abs(
            out.astype(mx.float32) - ref.astype(mx.float32)
        )))
        self.assertLess(max_err, 5e-2, f"max_err={max_err}")


if __name__ == "__main__":
    unittest.main()
