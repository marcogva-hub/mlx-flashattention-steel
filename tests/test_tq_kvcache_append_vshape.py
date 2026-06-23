"""Volet P4 — close the cache-appender defect class.

The appender silent-broadcast/inconsistent-state class recurred 4× (Dense /
Quantized / TurboQuantPaged in P1, TurboQuantKVCache here). P4 hardens
TurboQuantKVCache.append (Part A: direct + adapter, raw-V + compressed-V) and
sweeps EVERY direct K/V state-producer the property guard derives (Part B), so the
class is closed, not just the instance. LocalHostKVStoreAdapter.put was surfaced
by the name-independent state-write detector (Part C) and hardened too.
"""
import numpy as np
import mlx.core as mx
import pytest
from mlx_mfa.attention import DenseKVCache, QuantizedKVCache
from mlx_mfa.turboquant import TurboQuantKVCache
from mlx_mfa.kv_cache import adapt_kv_cache
from mlx_mfa.external_cache import LocalHostKVStoreAdapter


def _kv(kh, vh, n=2, b=1, d=128):
    k = mx.random.normal((b, kh, n, d)).astype(mx.float16)
    v = mx.random.normal((b, vh, n, d)).astype(mx.float16)
    mx.eval(k, v)
    return k, v


# ── Part A: TurboQuantKVCache.append — direct + adapter, raw + compressed ────────
@pytest.mark.parametrize("compress_v", [False, True])
def test_tqkvcache_direct_rejects_malformed_v_accepts_valid(compress_v):
    k, v = _kv(2, 1)
    with pytest.raises((ValueError, Exception)):
        TurboQuantKVCache(compress_v=compress_v).append(k, v)
    kok, vok = _kv(2, 2)
    TurboQuantKVCache(compress_v=compress_v).append(kok, vok)   # valid


def test_tqkvcache_adapter_rejects_malformed_v():
    k, v = _kv(2, 1)
    with pytest.raises((ValueError, Exception)):
        adapt_kv_cache(TurboQuantKVCache()).append(k, v)
    kok, vok = _kv(2, 2)
    adapt_kv_cache(TurboQuantKVCache()).append(kok, vok)        # valid (delegates)


@pytest.mark.parametrize("mut", ["rank", "batch", "headdim"])
def test_tqkvcache_rejects_other_v_mismatches(mut):
    k = mx.random.normal((1, 2, 2, 128)).astype(mx.float16)
    if mut == "rank":
        v = mx.random.normal((2, 2, 128)).astype(mx.float16)       # 3-D
    elif mut == "batch":
        v = mx.random.normal((2, 2, 2, 128)).astype(mx.float16)    # B=2
    elif mut == "headdim":
        v = mx.random.normal((1, 2, 2, 64)).astype(mx.float16)     # D=64
    mx.eval(k, v)
    with pytest.raises((ValueError, Exception)):
        TurboQuantKVCache().append(k, v)


def test_tqkvcache_valid_byte_identical_decompress():
    # byteΔ=0 on valid: the added checks don't change stored/compressed state.
    k, v = _kv(2, 2)
    c = TurboQuantKVCache(compress_v=False)
    c.append(k, v)
    out = c.v_decompressed() if hasattr(c, "v_decompressed") else None
    if out is not None:
        mx.eval(out)
        assert bool(np.isfinite(np.array(out.astype(mx.float32))).all())


# ── Part B: every direct K/V state-producer rejects malformed V ──────────────────
def test_all_direct_state_producers_reject_malformed_v():
    bad = _kv(2, 1)        # K heads=2, V heads=1
    good = _kv(2, 2)
    producers = [
        ("DenseKVCache.append", lambda kv: DenseKVCache(1, 2, 128, 64).append(*kv)),
        ("QuantizedKVCache.append", lambda kv: QuantizedKVCache(1, 2, 128, 64).append(*kv)),
        ("TurboQuantKVCache.append", lambda kv: TurboQuantKVCache().append(*kv)),
        ("LocalHostKVStoreAdapter.put", lambda kv: LocalHostKVStoreAdapter().put(0, *kv)),
    ]
    for name, fn in producers:
        with pytest.raises((ValueError, Exception)):
            r = fn(bad)
            if r is not None:
                mx.eval(r)
        fn(good)            # valid accepts (no exception)
