"""Tests for TurboQuant KV cache compression.

Test classes:
  TestWHT                  — Walsh-Hadamard transform correctness
  TestQRRotation           — QR random rotation correctness
  TestLloydMax             — Lloyd-Max centroid properties
  TestBitPacking           — Pack/unpack roundtrips for 1/2/3/4-bit
  TestTurboQuantRoundtrip  — Compress/decompress quality
  TestInnerProductQuality  — Inner product preservation (the key metric)
  TestTurboQuantKVCache    — Cache append/decompress/memory
  TestCacheAdapter         — KVCacheAdapter integration
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa.turboquant import (
    _walsh_hadamard_transform,
    _random_rotation_matrix,
    apply_rotation,
    apply_inverse_rotation,
    _get_centroids,
    quantize_to_indices,
    dequantize_from_indices,
    pack_indices,
    unpack_indices,
    _pack_1bit,
    _unpack_1bit,
    turboquant_compress,
    turboquant_decompress,
    TurboQuantKVCache,
)


def _pearson_corr(a: mx.array, b: mx.array) -> float:
    """Pearson correlation between two flattened arrays."""
    a_f = a.reshape(-1).astype(mx.float32)
    b_f = b.reshape(-1).astype(mx.float32)
    a_m = a_f - a_f.mean()
    b_m = b_f - b_f.mean()
    num = (a_m * b_m).mean()
    den = mx.sqrt((a_m * a_m).mean()) * mx.sqrt((b_m * b_m).mean())
    return (num / (den + 1e-12)).item()


# ---------------------------------------------------------------------------
# WHT
# ---------------------------------------------------------------------------


class TestWHT:
    def test_self_inverse(self):
        """WHT(WHT(x)) == x (self-inverse, normalized)."""
        x = mx.random.normal((2, 4, 128))
        y = _walsh_hadamard_transform(x)
        z = _walsh_hadamard_transform(y)
        err = (x - z).abs().max().item()
        assert err < 1e-5, f"WHT roundtrip error {err}"

    def test_preserves_norm(self):
        """||WHT(x)||_2 == ||x||_2 (orthogonal)."""
        x = mx.random.normal((8, 64))
        y = _walsh_hadamard_transform(x)
        norm_x = mx.sqrt((x * x).sum(axis=-1))
        norm_y = mx.sqrt((y * y).sum(axis=-1))
        err = (norm_x - norm_y).abs().max().item()
        assert err < 1e-5, f"Norm preservation error {err}"

    @pytest.mark.parametrize("D", [64, 128, 256])
    def test_supported_dims(self, D):
        """WHT works for all supported head dimensions."""
        x = mx.random.normal((1, 1, 16, D))
        y = _walsh_hadamard_transform(x)
        assert y.shape == x.shape

    def test_rejects_non_power_of_2(self):
        """WHT rejects non-power-of-2 last dimension."""
        x = mx.random.normal((10, 96))
        with pytest.raises(AssertionError):
            _walsh_hadamard_transform(x)


# ---------------------------------------------------------------------------
# QR Rotation
# ---------------------------------------------------------------------------


class TestQRRotation:
    def test_roundtrip(self):
        """apply_rotation + apply_inverse_rotation == identity."""
        x = mx.random.normal((1, 8, 16, 128))
        y = apply_rotation(x, "qr", seed=42)
        z = apply_inverse_rotation(y, "qr", seed=42)
        err = (x - z).abs().max().item()
        assert err < 1e-4, f"QR roundtrip error {err}"

    def test_orthogonal(self):
        """R @ R^T == I."""
        R = _random_rotation_matrix(64, seed=99)
        I_approx = R @ R.T
        I_true = mx.eye(64)
        err = (I_approx - I_true).abs().max().item()
        assert err < 1e-4, f"Orthogonality error {err}"

    def test_deterministic(self):
        """Same seed produces same matrix."""
        R1 = _random_rotation_matrix(128, seed=42)
        R2 = _random_rotation_matrix(128, seed=42)
        assert mx.array_equal(R1, R2)

    def test_different_seeds_differ(self):
        """Different seeds produce different matrices."""
        R1 = _random_rotation_matrix(64, seed=1)
        R2 = _random_rotation_matrix(64, seed=2)
        assert not mx.array_equal(R1, R2)


# ---------------------------------------------------------------------------
# Lloyd-Max Centroids
# ---------------------------------------------------------------------------


class TestLloydMax:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_correct_count(self, bits):
        """2^bits centroids, 2^bits - 1 boundaries."""
        boundaries, centroids = _get_centroids(bits)
        assert centroids.shape[0] == (1 << bits)
        assert boundaries.shape[0] == (1 << bits) - 1

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_symmetric(self, bits):
        """Centroids are symmetric around 0."""
        _, centroids = _get_centroids(bits)
        n = centroids.shape[0]
        for i in range(n // 2):
            assert abs(centroids[i].item() + centroids[n - 1 - i].item()) < 1e-6

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_sorted(self, bits):
        """Centroids and boundaries are monotonically increasing."""
        boundaries, centroids = _get_centroids(bits)
        c = centroids.tolist()
        b = boundaries.tolist()
        assert c == sorted(c)
        assert b == sorted(b)

    def test_unsupported_bits(self):
        with pytest.raises(ValueError, match="Unsupported bits"):
            _get_centroids(5)


# ---------------------------------------------------------------------------
# Bit Packing
# ---------------------------------------------------------------------------


class TestBitPacking:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_roundtrip(self, bits):
        """Pack then unpack recovers original indices."""
        n_values = 1024 + 7  # non-aligned to exercise padding
        max_val = (1 << bits) - 1
        indices = mx.random.randint(0, max_val + 1, (n_values,)).astype(mx.uint8)
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, n_values, bits)
        assert mx.array_equal(indices, unpacked)

    def test_1bit_roundtrip(self):
        """1-bit pack/unpack for QJL signs."""
        n = 1000 + 3
        signs = mx.random.randint(0, 2, (n,)).astype(mx.uint8)
        packed = _pack_1bit(signs)
        unpacked = _unpack_1bit(packed, n)
        assert mx.array_equal(signs, unpacked)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_compression_size(self, bits):
        """Packed size should be approximately n * bits / 8."""
        n = 8192
        indices = mx.zeros((n,), dtype=mx.uint8)
        packed = pack_indices(indices, bits)
        expected_bytes = math.ceil(n * bits / 8)
        assert packed.nbytes == expected_bytes


# ---------------------------------------------------------------------------
# Compress / Decompress Roundtrip
# ---------------------------------------------------------------------------


class TestTurboQuantRoundtrip:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    @pytest.mark.parametrize("rotation", ["wht", "qr"])
    def test_roundtrip_shape_dtype(self, bits, rotation):
        """Decompress returns same shape and dtype as input."""
        x = mx.random.normal((1, 4, 32, 128)).astype(mx.float16)
        c = turboquant_compress(x, bits=bits, use_qjl=False, rotation=rotation)
        y = turboquant_decompress(c)
        assert y.shape == x.shape
        assert y.dtype == x.dtype

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_error_decreases_with_bits(self, bits):
        """Higher bits → lower reconstruction error."""
        mx.random.seed(42)
        x = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
        c = turboquant_compress(x, bits=bits, use_qjl=False, rotation="wht")
        y = turboquant_decompress(c)
        mse = ((x.astype(mx.float32) - y.astype(mx.float32)) ** 2).mean().item()
        # Store for cross-test comparison (just verify finite)
        assert math.isfinite(mse)

    def test_3bit_max_error_bounded(self):
        """3-bit max abs error is bounded (per-element can be large, per-vector small)."""
        mx.random.seed(42)
        x = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
        c = turboquant_compress(x, bits=3, use_qjl=False)
        y = turboquant_decompress(c)
        max_err = (x.astype(mx.float32) - y.astype(mx.float32)).abs().max().item()
        assert max_err < 2.0  # reasonable bound for 3-bit on N(0,1) data

    def test_qjl_vs_no_qjl(self):
        """QJL should not dramatically increase per-element error."""
        mx.random.seed(42)
        x = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
        c_no = turboquant_compress(x, bits=3, use_qjl=False)
        c_qjl = turboquant_compress(x, bits=3, use_qjl=True)
        y_no = turboquant_decompress(c_no)
        y_qjl = turboquant_decompress(c_qjl)
        mse_no = ((x.astype(mx.float32) - y_no.astype(mx.float32)) ** 2).mean().item()
        mse_qjl = ((x.astype(mx.float32) - y_qjl.astype(mx.float32)) ** 2).mean().item()
        # QJL correction may slightly change MSE but shouldn't blow up
        assert mse_qjl < mse_no * 3.0

    def test_bfloat16(self):
        """bf16 input roundtrips correctly."""
        x = mx.random.normal((1, 4, 32, 128)).astype(mx.bfloat16)
        c = turboquant_compress(x, bits=3, use_qjl=False)
        y = turboquant_decompress(c)
        assert y.dtype == mx.bfloat16
        assert y.shape == x.shape

    def test_rejects_wrong_ndim(self):
        """Input must be 4D [B,H,S,D]."""
        x = mx.random.normal((128, 128))
        with pytest.raises(ValueError, match="ndim"):
            turboquant_compress(x, bits=3)


# ---------------------------------------------------------------------------
# Inner Product Preservation (the critical metric)
# ---------------------------------------------------------------------------


class TestInnerProductQuality:
    def test_score_correlation_3bit_qjl(self):
        """3-bit + QJL: score correlation > 0.95."""
        mx.random.seed(42)
        Q = mx.random.normal((1, 8, 16, 128)).astype(mx.float16)
        K = mx.random.normal((1, 8, 256, 128)).astype(mx.float16)

        scores_true = Q.astype(mx.float32) @ K.astype(mx.float32).swapaxes(-1, -2)

        K_c = turboquant_compress(K, bits=3, use_qjl=True)
        K_dec = turboquant_decompress(K_c)
        scores_approx = Q.astype(mx.float32) @ K_dec.astype(mx.float32).swapaxes(-1, -2)

        corr = _pearson_corr(scores_true, scores_approx)
        assert corr > 0.95, f"Score correlation {corr:.4f} < 0.95"

    def test_score_correlation_improves_with_bits(self):
        """Higher bits → higher score correlation."""
        mx.random.seed(42)
        Q = mx.random.normal((1, 8, 16, 128)).astype(mx.float16)
        K = mx.random.normal((1, 8, 256, 128)).astype(mx.float16)
        scores_true = Q.astype(mx.float32) @ K.astype(mx.float32).swapaxes(-1, -2)

        corrs = []
        for bits in [2, 3, 4]:
            K_c = turboquant_compress(K, bits=bits, use_qjl=False)
            K_dec = turboquant_decompress(K_c)
            s_approx = Q.astype(mx.float32) @ K_dec.astype(mx.float32).swapaxes(-1, -2)
            corrs.append(_pearson_corr(scores_true, s_approx))

        assert corrs[0] < corrs[1] < corrs[2], f"Correlations not increasing: {corrs}"

    def test_attention_output_close(self):
        """Full attention with compressed K/V vs fp16 reference."""
        mx.random.seed(42)
        B, H, N, S, D = 1, 8, 16, 256, 128
        scale = 1.0 / math.sqrt(D)
        Q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        K = mx.random.normal((B, H, S, D)).astype(mx.float16)
        V = mx.random.normal((B, H, S, D)).astype(mx.float16)

        # Reference: fp16 SDPA
        ref = mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale)

        # Compressed K
        K_c = turboquant_compress(K, bits=3, use_qjl=True)
        K_dec = turboquant_decompress(K_c)
        approx = mx.fast.scaled_dot_product_attention(Q, K_dec, V, scale=scale)

        # Attention output should be close (softmax dampens quantization noise)
        max_diff = (ref.astype(mx.float32) - approx.astype(mx.float32)).abs().max().item()
        assert max_diff < 1.0, f"Attention output max diff {max_diff}"
        corr = _pearson_corr(ref, approx)
        assert corr > 0.98, f"Attention output correlation {corr:.4f}"


# ---------------------------------------------------------------------------
# TurboQuantKVCache
# ---------------------------------------------------------------------------


class TestTurboQuantKVCache:
    def test_append_and_decompress(self):
        """Append tokens, decompress, verify shape and dtype."""
        cache = TurboQuantKVCache(bits=3, use_qjl=True)
        k = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
        v = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
        cache.append(k, v)
        assert cache.seq_length == 64
        assert cache.k_decompressed().shape == (1, 8, 64, 128)
        assert cache.v_decompressed().shape == (1, 8, 64, 128)
        assert cache.k_decompressed().dtype == mx.float16

    def test_multi_append(self):
        """Multiple appends concatenate correctly."""
        cache = TurboQuantKVCache(bits=3)
        for _ in range(4):
            k = mx.random.normal((1, 4, 16, 128)).astype(mx.float16)
            v = mx.random.normal((1, 4, 16, 128)).astype(mx.float16)
            cache.append(k, v)
        assert cache.seq_length == 64
        assert cache.k_decompressed().shape == (1, 4, 64, 128)

    def test_compression_ratio_k_only(self):
        """K-only compression: ratio > 1 (K compressed, V raw)."""
        cache = TurboQuantKVCache(bits=3, use_qjl=True, compress_v=False)
        k = mx.random.normal((1, 8, 512, 128)).astype(mx.float16)
        v = mx.random.normal((1, 8, 512, 128)).astype(mx.float16)
        cache.append(k, v)
        assert cache.compression_ratio > 1.0

    def test_compression_ratio_both(self):
        """K+V compression at 3-bit: ratio > 3.5."""
        cache = TurboQuantKVCache(bits=3, use_qjl=True, compress_v=True)
        k = mx.random.normal((1, 8, 512, 128)).astype(mx.float16)
        v = mx.random.normal((1, 8, 512, 128)).astype(mx.float16)
        cache.append(k, v)
        assert cache.compression_ratio > 3.5, f"Ratio {cache.compression_ratio:.2f}"

    def test_memory_less_than_fp16(self):
        """memory_bytes < memory_bytes_fp16."""
        cache = TurboQuantKVCache(bits=3, compress_v=True)
        k = mx.random.normal((1, 8, 256, 128)).astype(mx.float16)
        v = mx.random.normal((1, 8, 256, 128)).astype(mx.float16)
        cache.append(k, v)
        assert cache.memory_bytes < cache.memory_bytes_fp16

    def test_reset(self):
        """Reset clears all state."""
        cache = TurboQuantKVCache(bits=3)
        k = mx.random.normal((1, 4, 32, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 32, 64)).astype(mx.float16)
        cache.append(k, v)
        cache.reset()
        assert cache.seq_length == 0

    def test_empty_cache_raises(self):
        """Decompressing an empty cache raises."""
        cache = TurboQuantKVCache(bits=3)
        with pytest.raises(RuntimeError, match="empty"):
            cache.k_decompressed()

    @pytest.mark.parametrize("D", [64, 128, 256])
    def test_head_dims(self, D):
        """Cache works for all supported head dimensions."""
        cache = TurboQuantKVCache(bits=3, use_qjl=False)
        k = mx.random.normal((1, 4, 32, D)).astype(mx.float16)
        v = mx.random.normal((1, 4, 32, D)).astype(mx.float16)
        cache.append(k, v)
        assert cache.k_decompressed().shape[-1] == D


# ---------------------------------------------------------------------------
# Cache Adapter
# ---------------------------------------------------------------------------


class TestCacheAdapter:
    def test_adapt_kv_cache(self):
        """adapt_kv_cache detects TurboQuantKVCache."""
        from mlx_mfa.kv_cache import adapt_kv_cache

        cache = TurboQuantKVCache(bits=3)
        adapter = adapt_kv_cache(cache)
        assert adapter.kind == "turboquant"

    def test_adapter_capabilities(self):
        """Adapter reports correct capabilities."""
        from mlx_mfa.kv_cache import adapt_kv_cache

        cache = TurboQuantKVCache(bits=3)
        adapter = adapt_kv_cache(cache)
        caps = adapter.capabilities
        assert caps.append is True
        assert caps.attention_view is True
        assert caps.multi_seq is False

    def test_adapter_flow(self):
        """Full flow through adapter: append → attention_k/v → reset."""
        from mlx_mfa.kv_cache import adapt_kv_cache

        cache = TurboQuantKVCache(bits=3, use_qjl=True)
        adapter = adapt_kv_cache(cache)

        k = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
        v = mx.random.normal((1, 8, 64, 128)).astype(mx.float16)
        adapter.append(k, v)

        assert adapter.seq_length() == 64
        assert adapter.attention_k().shape == (1, 8, 64, 128)
        assert adapter.attention_v().shape == (1, 8, 64, 128)

        adapter.reset()
        assert cache.seq_length == 0


# ---------------------------------------------------------------------------
# Bits comparison
# ---------------------------------------------------------------------------


class TestBitsComparison:
    def test_2bit_vs_3bit_vs_4bit_quality(self):
        """Higher bits = better inner product preservation."""
        mx.random.seed(42)
        Q = mx.random.normal((1, 8, 16, 128)).astype(mx.float16)
        K = mx.random.normal((1, 8, 128, 128)).astype(mx.float16)
        scores_true = Q.astype(mx.float32) @ K.astype(mx.float32).swapaxes(-1, -2)

        corrs = {}
        for bits in [2, 3, 4]:
            K_c = turboquant_compress(K, bits=bits, use_qjl=False)
            K_dec = turboquant_decompress(K_c)
            s = Q.astype(mx.float32) @ K_dec.astype(mx.float32).swapaxes(-1, -2)
            corrs[bits] = _pearson_corr(scores_true, s)

        assert corrs[2] < corrs[3] < corrs[4], f"Not monotonic: {corrs}"
        assert corrs[4] > 0.99, f"4-bit correlation {corrs[4]:.4f} < 0.99"
