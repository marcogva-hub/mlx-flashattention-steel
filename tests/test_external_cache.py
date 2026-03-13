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
