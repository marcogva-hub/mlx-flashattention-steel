from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_mfa import LocalHostKVStoreAdapter


class TestLocalHostKVStoreAdapter:
    def test_put_fetch_roundtrip(self):
        ad = LocalHostKVStoreAdapter()
        k = mx.random.normal((1, 2, 5, 16)).astype(mx.float16)
        v = mx.random.normal((1, 2, 5, 16)).astype(mx.float16)
        ad.put(3, k, v, meta={"tag": "rt"})
        k2, v2 = ad.fetch(3)
        mx.eval(k2, v2)
        assert k2.shape == (1, 2, 5, 16)
        assert v2.shape == (1, 2, 5, 16)
        assert ad.seq_length(3) == 5
        assert ad.has(3) is True

    def test_prefetch_and_evict(self):
        ad = LocalHostKVStoreAdapter()
        k = mx.random.normal((1, 1, 2, 8)).astype(mx.float16)
        v = mx.random.normal((1, 1, 2, 8)).astype(mx.float16)
        ad.put(9, k, v)
        ad.prefetch(9)
        assert ad.state["last_prefetch"]["seq_id"] == 9
        ad.evict(9)
        assert ad.has(9) is False

    def test_fetch_missing_raises(self):
        ad = LocalHostKVStoreAdapter()
        with pytest.raises(KeyError):
            ad.fetch(42)

    def test_prefetch_missing_raises(self):
        ad = LocalHostKVStoreAdapter()
        with pytest.raises(KeyError):
            ad.prefetch(42)

    def test_preserves_dtype_f16(self):
        """f16 K/V roundtrip preserves values exactly (zero-copy)."""
        ad = LocalHostKVStoreAdapter()
        k = mx.random.normal((1, 4, 32, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 32, 64)).astype(mx.float16)
        ad.put(0, k, v)
        k2, v2 = ad.fetch(0)
        assert k2.dtype == mx.float16
        assert mx.array_equal(k, k2).item()
        assert mx.array_equal(v, v2).item()

    def test_preserves_dtype_bf16(self):
        """bf16 roundtrip preserves dtype (zero-copy)."""
        ad = LocalHostKVStoreAdapter()
        k = mx.random.normal((1, 4, 32, 64)).astype(mx.bfloat16)
        v = mx.random.normal((1, 4, 32, 64)).astype(mx.bfloat16)
        ad.put(0, k, v)
        k2, v2 = ad.fetch(0)
        assert k2.dtype == mx.bfloat16

    def test_multiple_seqs(self):
        """Multiple sequences stored and retrieved independently."""
        ad = LocalHostKVStoreAdapter()
        for sid in range(5):
            k = mx.full((1, 2, 16, 64), float(sid), dtype=mx.float16)
            v = mx.full((1, 2, 16, 64), float(sid + 10), dtype=mx.float16)
            ad.put(sid, k, v)
        assert ad.offloaded_seq_ids == (0, 1, 2, 3, 4)
        k3, v3 = ad.fetch(3)
        assert k3[0, 0, 0, 0].item() == 3.0

    def test_evict_clears(self):
        """Evict removes the record, fetch raises."""
        ad = LocalHostKVStoreAdapter()
        k = mx.zeros((1, 2, 8, 64), dtype=mx.float16)
        v = mx.zeros((1, 2, 8, 64), dtype=mx.float16)
        ad.put(0, k, v)
        ad.evict(0)
        assert not ad.has(0)
        with pytest.raises(KeyError):
            ad.fetch(0)

    def test_overwrite(self):
        """Overwriting a seq_id replaces the previous record."""
        ad = LocalHostKVStoreAdapter()
        k1 = mx.ones((1, 2, 8, 64), dtype=mx.float16)
        v1 = mx.ones((1, 2, 8, 64), dtype=mx.float16)
        ad.put(0, k1, v1)
        k2 = mx.full((1, 2, 16, 64), 2.0, dtype=mx.float16)
        v2 = mx.full((1, 2, 16, 64), 2.0, dtype=mx.float16)
        ad.put(0, k2, v2)
        assert ad.seq_length(0) == 16
        k_out, _ = ad.fetch(0)
        assert k_out[0, 0, 0, 0].item() == 2.0
