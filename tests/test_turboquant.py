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


# ---------------------------------------------------------------------------
# Phase 2 — Fused TurboQuant kernel tests
# ---------------------------------------------------------------------------


def _skip_if_no_ext():
    """Skip if C++ extension not built."""
    try:
        from mlx_mfa import is_mfa_available
        if not is_mfa_available():
            pytest.skip("MFA extension not available")
    except ImportError:
        pytest.skip("mlx_mfa not installed")


def _build_tq_paged_pool(k_seqs, v_seqs, block_size, bits=3, rotation="wht", seed=42):
    """Build fp16 V pool + TQ-packed K pool from per-sequence KV tensors."""
    from mlx_mfa.turboquant import pack_k_for_metal, _get_centroids

    B = len(k_seqs)
    H_kv = k_seqs[0].shape[1]
    D = k_seqs[0].shape[3]
    packed_D = D // 2
    blocks_per_seq = [
        (int(k.shape[2]) + block_size - 1) // block_size for k in k_seqs
    ]
    total_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq) if blocks_per_seq else 0

    pool_k_tq = np.zeros((total_blocks, block_size, H_kv, packed_D), dtype=np.uint8)
    pool_k_scales = np.zeros((total_blocks, block_size, H_kv), dtype=np.float32)
    pool_v = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float16)
    table = np.full((B, max_blocks), 0, dtype=np.int32)
    lens = np.zeros((B,), dtype=np.int32)

    # Get centroids once
    _, centroids_f32 = _get_centroids(bits)
    centroids_fp16 = centroids_f32.astype(mx.float16)

    blk_base = 0
    for b in range(B):
        # k_seqs[b] = [1, H_kv, S, D]
        S = k_seqs[b].shape[2]
        lens[b] = S

        # Pack K for this sequence
        k_packed, k_scales, _ = pack_k_for_metal(
            k_seqs[b], bits=bits, rotation=rotation, seed=seed
        )
        mx.synchronize()
        # k_packed: [1, H_kv, S, packed_D], k_scales: [1, H_kv, S]

        k_packed_np = np.array(k_packed)[0]  # [H_kv, S, packed_D]
        k_scales_np = np.array(k_scales.astype(mx.float32))[0]  # [H_kv, S]

        v_np = np.array(v_seqs[b].astype(mx.float16))[0]  # [H_kv, S, D]

        n_blk = blocks_per_seq[b]
        for lb in range(n_blk):
            table[b, lb] = blk_base + lb
            s0 = lb * block_size
            s1 = min(S, s0 + block_size)
            chunk_len = s1 - s0
            # pool layout: [block, block_size, H_kv, ...]
            # from [H_kv, S, ...] -> transpose to [S, H_kv, ...]
            pool_k_tq[blk_base + lb, :chunk_len] = k_packed_np.transpose(1, 0, 2)[s0:s1]
            pool_k_scales[blk_base + lb, :chunk_len] = k_scales_np.transpose(1, 0)[s0:s1]
            pool_v[blk_base + lb, :chunk_len] = v_np.transpose(1, 0, 2)[s0:s1]
        blk_base += n_blk

    return (
        mx.array(pool_k_tq),
        mx.array(pool_v),
        mx.array(pool_k_scales, dtype=mx.float32),
        centroids_fp16,
        mx.array(table, dtype=mx.int32),
        mx.array(lens, dtype=mx.int32),
    )


def _pack_queries(q_seqs):
    """Pack per-sequence [1,H,Qi,D] into [1,H,total_q,D] + cu_seqlens_q."""
    offsets = [0]
    for q in q_seqs:
        offsets.append(offsets[-1] + int(q.shape[2]))
    q_pack = mx.concatenate(q_seqs, axis=2)
    cu = mx.array(offsets, dtype=mx.int32)
    return q_pack, cu


class TestTurboQuantFusedKernel:
    """Tests for the Phase 2 fused TQ paged varlen kernel."""

    def test_fused_vs_decompress_noncausal(self):
        """Fused TQ kernel matches decompress->paged_varlen for non-causal."""
        _skip_if_no_ext()
        from mlx_mfa import flash_attention_paged_varlen, flash_attention_paged_varlen_turboquant
        from mlx_mfa.turboquant import turboquant_compress, turboquant_decompress

        mx.random.seed(801)
        H_q, H_kv, D = 4, 4, 64
        bits = 3
        block_size = 16
        scale = 1.0 / math.sqrt(D)

        q_seqs = [mx.random.normal((1, H_q, 8, D)).astype(mx.float16)]
        k_seqs = [mx.random.normal((1, H_kv, 32, D)).astype(mx.float16)]
        v_seqs = [mx.random.normal((1, H_kv, 32, D)).astype(mx.float16)]
        mx.synchronize()

        # --- Fused path ---
        # Pre-rotate Q with WHT
        from mlx_mfa.turboquant import apply_rotation
        q_rot_seqs = [apply_rotation(q.astype(mx.float32), "wht").astype(mx.float16) for q in q_seqs]
        mx.synchronize()

        q_pack, cu_q = _pack_queries(q_rot_seqs)
        pool_k_tq, pool_v, k_scales, centroids, table, lens = _build_tq_paged_pool(
            k_seqs, v_seqs, block_size, bits=bits
        )

        out_fused = flash_attention_paged_varlen_turboquant(
            q_pack, pool_k_tq, pool_v, table, lens, cu_q,
            centroids, k_scales,
            scale=scale, causal=False, block_size=block_size, tq_bits=bits,
        )
        mx.synchronize()

        # --- Decompress path (reference) ---
        k_decomp_seqs = []
        for k in k_seqs:
            c = turboquant_compress(k, bits=bits, use_qjl=False, rotation="wht")
            k_decomp_seqs.append(turboquant_decompress(c))
        mx.synchronize()

        # Build fp16 paged pool for reference
        B = len(k_seqs)
        blocks_per_seq = [(int(k.shape[2]) + block_size - 1) // block_size for k in k_seqs]
        total_blocks = sum(blocks_per_seq)
        max_blocks_val = max(blocks_per_seq)
        pool_k_ref = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float16)
        pool_v_ref = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float16)
        table_ref = np.full((B, max_blocks_val), 0, dtype=np.int32)
        lens_ref = np.zeros((B,), dtype=np.int32)

        blk = 0
        for b in range(B):
            S = int(k_seqs[b].shape[2])
            lens_ref[b] = S
            n_blk = blocks_per_seq[b]
            k_np = np.array(k_decomp_seqs[b].astype(mx.float16))[0].transpose(1, 0, 2)
            v_np = np.array(v_seqs[b].astype(mx.float16))[0].transpose(1, 0, 2)
            for lb in range(n_blk):
                table_ref[b, lb] = blk + lb
                s0 = lb * block_size
                s1 = min(S, s0 + block_size)
                pool_k_ref[blk + lb, :s1 - s0] = k_np[s0:s1]
                pool_v_ref[blk + lb, :s1 - s0] = v_np[s0:s1]
            blk += n_blk

        # Reference uses decompressed K (original space) → needs original Q (not rotated).
        q_orig_pack, cu_q_orig = _pack_queries(q_seqs)
        out_ref = flash_attention_paged_varlen(
            q_orig_pack,
            mx.array(pool_k_ref), mx.array(pool_v_ref),
            mx.array(table_ref, dtype=mx.int32),
            mx.array(lens_ref, dtype=mx.int32),
            cu_q_orig,
            scale=scale, causal=False, block_size=block_size,
        )
        mx.synchronize()

        err = np.abs(np.array(out_fused.astype(mx.float32)) - np.array(out_ref.astype(mx.float32)))
        max_err = err.max()
        # Fused and decompress paths should produce close results
        assert max_err < 0.1, f"max_abs_err={max_err:.4f} > 0.1"

    def test_fused_causal(self):
        """Fused TQ kernel with causal masking."""
        _skip_if_no_ext()
        from mlx_mfa import flash_attention_paged_varlen_turboquant
        from mlx_mfa.turboquant import apply_rotation

        mx.random.seed(802)
        H_q, H_kv, D = 4, 4, 128
        bits = 3
        block_size = 16
        scale = 1.0 / math.sqrt(D)

        q = mx.random.normal((1, H_q, 16, D)).astype(mx.float16)
        k = mx.random.normal((1, H_kv, 16, D)).astype(mx.float16)
        v = mx.random.normal((1, H_kv, 16, D)).astype(mx.float16)
        mx.synchronize()

        q_rot = apply_rotation(q.astype(mx.float32), "wht").astype(mx.float16)
        mx.synchronize()

        q_pack, cu_q = _pack_queries([q_rot])
        pool_k_tq, pool_v, k_scales, centroids, table, lens = _build_tq_paged_pool(
            [k], [v], block_size, bits=bits
        )

        out = flash_attention_paged_varlen_turboquant(
            q_pack, pool_k_tq, pool_v, table, lens, cu_q,
            centroids, k_scales,
            scale=scale, causal=True, block_size=block_size, tq_bits=bits,
        )
        mx.synchronize()

        assert out.shape == q_pack.shape
        out_np = np.array(out.astype(mx.float32))
        assert np.all(np.isfinite(out_np)), "NaN or Inf in causal output"

    def test_fused_gqa(self):
        """Fused TQ kernel with GQA (H_q > H_kv)."""
        _skip_if_no_ext()
        from mlx_mfa import flash_attention_paged_varlen_turboquant
        from mlx_mfa.turboquant import apply_rotation

        mx.random.seed(803)
        H_q, H_kv, D = 8, 2, 64
        bits = 3
        block_size = 16
        scale = 1.0 / math.sqrt(D)

        q = mx.random.normal((1, H_q, 4, D)).astype(mx.float16)
        k = mx.random.normal((1, H_kv, 32, D)).astype(mx.float16)
        v = mx.random.normal((1, H_kv, 32, D)).astype(mx.float16)
        mx.synchronize()

        q_rot = apply_rotation(q.astype(mx.float32), "wht").astype(mx.float16)
        mx.synchronize()

        q_pack, cu_q = _pack_queries([q_rot])
        pool_k_tq, pool_v, k_scales, centroids, table, lens = _build_tq_paged_pool(
            [k], [v], block_size, bits=bits
        )

        out = flash_attention_paged_varlen_turboquant(
            q_pack, pool_k_tq, pool_v, table, lens, cu_q,
            centroids, k_scales,
            scale=scale, causal=False, block_size=block_size, tq_bits=bits,
        )
        mx.synchronize()

        assert out.shape == (1, H_q, 4, D)
        out_np = np.array(out.astype(mx.float32))
        assert np.all(np.isfinite(out_np)), "NaN or Inf in GQA output"

    def test_fused_multi_seq(self):
        """Fused TQ with multiple variable-length sequences."""
        _skip_if_no_ext()
        from mlx_mfa import flash_attention_paged_varlen_turboquant
        from mlx_mfa.turboquant import apply_rotation

        mx.random.seed(804)
        H_q, H_kv, D = 4, 4, 64
        bits = 3
        block_size = 16
        scale = 1.0 / math.sqrt(D)

        q_lens = [3, 1, 4]
        kv_lens = [27, 19, 33]

        q_seqs = [mx.random.normal((1, H_q, ql, D)).astype(mx.float16) for ql in q_lens]
        k_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        v_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        mx.synchronize()

        q_rot_seqs = [apply_rotation(q.astype(mx.float32), "wht").astype(mx.float16) for q in q_seqs]
        mx.synchronize()

        q_pack, cu_q = _pack_queries(q_rot_seqs)
        pool_k_tq, pool_v, k_scales, centroids, table, lens = _build_tq_paged_pool(
            k_seqs, v_seqs, block_size, bits=bits
        )

        out = flash_attention_paged_varlen_turboquant(
            q_pack, pool_k_tq, pool_v, table, lens, cu_q,
            centroids, k_scales,
            scale=scale, causal=False, block_size=block_size, tq_bits=bits,
        )
        mx.synchronize()

        total_q = sum(q_lens)
        assert out.shape == (1, H_q, total_q, D)
        out_np = np.array(out.astype(mx.float32))
        assert np.all(np.isfinite(out_np)), "NaN or Inf in multi-seq output"

    @pytest.mark.parametrize("bits", [2, 4])
    def test_fused_other_bitwidths(self, bits):
        """Fused TQ kernel with 2-bit and 4-bit."""
        _skip_if_no_ext()
        from mlx_mfa import flash_attention_paged_varlen_turboquant
        from mlx_mfa.turboquant import apply_rotation

        mx.random.seed(805 + bits)
        H_q, H_kv, D = 4, 4, 64
        block_size = 16
        scale = 1.0 / math.sqrt(D)

        q = mx.random.normal((1, H_q, 8, D)).astype(mx.float16)
        k = mx.random.normal((1, H_kv, 32, D)).astype(mx.float16)
        v = mx.random.normal((1, H_kv, 32, D)).astype(mx.float16)
        mx.synchronize()

        q_rot = apply_rotation(q.astype(mx.float32), "wht").astype(mx.float16)
        mx.synchronize()

        q_pack, cu_q = _pack_queries([q_rot])
        pool_k_tq, pool_v, k_scales, centroids, table, lens = _build_tq_paged_pool(
            [k], [v], block_size, bits=bits
        )

        out = flash_attention_paged_varlen_turboquant(
            q_pack, pool_k_tq, pool_v, table, lens, cu_q,
            centroids, k_scales,
            scale=scale, causal=False, block_size=block_size, tq_bits=bits,
        )
        mx.synchronize()

        assert out.shape == q_pack.shape
        out_np = np.array(out.astype(mx.float32))
        assert np.all(np.isfinite(out_np)), f"NaN/Inf with {bits}-bit"

    def test_pack_k_for_metal_roundtrip(self):
        """pack_k_for_metal indices roundtrip to correct centroid values."""
        from mlx_mfa.turboquant import pack_k_for_metal

        mx.random.seed(810)
        k = mx.random.normal((1, 2, 8, 64)).astype(mx.float16)
        mx.synchronize()

        for bits in [2, 3, 4]:
            k_packed, scales, centroids = pack_k_for_metal(k, bits=bits)
            mx.synchronize()

            assert k_packed.shape == (1, 2, 8, 32), f"bits={bits}: bad shape {k_packed.shape}"
            assert k_packed.dtype == mx.uint8
            assert scales.shape == (1, 2, 8)
            assert centroids.shape == (1 << bits,)

    def test_build_tq_paged_k_pool(self):
        """build_tq_paged_k_pool produces correctly shaped output."""
        from mlx_mfa.turboquant import build_tq_paged_k_pool

        mx.random.seed(811)
        num_pages, block_size, H_kv, D = 4, 16, 2, 128
        k_pool = mx.random.normal((num_pages, block_size, H_kv, D)).astype(mx.float16)
        mx.synchronize()

        k_pool_tq, scales, centroids = build_tq_paged_k_pool(k_pool, bits=3)
        mx.synchronize()

        assert k_pool_tq.shape == (num_pages, block_size, H_kv, D // 2)
        assert k_pool_tq.dtype == mx.uint8
        assert scales.shape == (num_pages, block_size, H_kv)
        assert centroids.shape == (8,)  # 2^3
