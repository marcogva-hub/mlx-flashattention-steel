"""Volet P1 Part B — cache appenders must reject a malformed V head count
(silent-broadcast HIGH).

DenseKVCache.append / QuantizedKVCache.append / TurboQuantPagedInferenceContext
.append wrote v_new via a slice-assign / pool-pack that SILENTLY BROADCAST a
mismatched V head count (1-head V into a 2-head cache → no raise). This bypasses
the function-surface Q/K/V mutual-shape contract. Now a V whose head count
mismatches K / the cache must raise loudly; valid shapes still accept.
"""
import sys
import numpy as np
import mlx.core as mx
import pytest
from mlx_mfa.attention import DenseKVCache, QuantizedKVCache

B, H, D = 1, 2, 128


def _kv(vh, n=1):
    k = mx.random.normal((B, H, n, D)).astype(mx.float16)
    v = mx.random.normal((B, vh, n, D)).astype(mx.float16)
    mx.eval(k, v)
    return k, v


def test_dense_rejects_v_head_mismatch_accepts_valid():
    c = DenseKVCache(B, H, D, 256)
    with pytest.raises((ValueError, Exception)):
        k, v = _kv(1); c.append(k, v); mx.eval(c.k, c.v)
    c2 = DenseKVCache(B, H, D, 256)
    k, v = _kv(H); c2.append(k, v); mx.eval(c2.k, c2.v)   # valid


def test_quant_rejects_v_head_mismatch_accepts_valid():
    c = QuantizedKVCache(B, H, D, 256)
    with pytest.raises((ValueError, Exception)):
        k, v = _kv(1); c.append(k, v)
    c2 = QuantizedKVCache(B, H, D, 256)
    k, v = _kv(H); c2.append(k, v)                         # valid


def _tq_ctx():
    sys.path.insert(0, "tests")
    import test_phase3_iii2_tq_decode as T
    ctx, _q = T._mkctx(3)
    return ctx


def test_tq_append_rejects_v_head_mismatch_accepts_valid():
    ctx = _tq_ctx()
    kbad = mx.random.normal((1, 2, 1, 128)).astype(mx.float16)
    vbad = mx.random.normal((1, 1, 1, 128)).astype(mx.float16)
    mx.eval(kbad, vbad)
    with pytest.raises((ValueError, Exception)):
        ctx.append(kbad, vbad)
    kok = mx.random.normal((1, 2, 1, 128)).astype(mx.float16)
    vok = mx.random.normal((1, 2, 1, 128)).astype(mx.float16)
    mx.eval(kok, vok)
    ctx.append(kok, vok)                                   # valid


def test_dense_also_rejects_batch_and_dim_mismatch():
    c = DenseKVCache(B, H, D, 256)
    k = mx.random.normal((B, H, 1, D)).astype(mx.float16)
    v_baddim = mx.random.normal((B, H, 1, D // 2)).astype(mx.float16)
    mx.eval(k, v_baddim)
    with pytest.raises((ValueError, Exception)):
        c.append(k, v_baddim); mx.eval(c.k, c.v)
