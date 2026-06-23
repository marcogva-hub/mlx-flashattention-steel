"""Volet P6 — dtype axis of the shared K/V persistence contract.

The last axis gap (re-run #3): `assert_kv_persist_compat` enforced
rank/batch/heads/token/D but not dtype, so a fixed-dtype cache silently CAST a
mismatched append (fp16 cache + bf16 append → silent precision change), and a
K-fp16/V-bf16 pair was accepted. The dtype axis now lives in the one shared
helper: K↔V dtype consistency (always) + an accepted-input-dtype set per surface
(storage caches → their single storage dtype; quantizing/host → fp16/bf16). No
silent cast.
"""
import sys
import numpy as np
import mlx.core as mx
import pytest
from mlx_mfa.attention import DenseKVCache, QuantizedKVCache, PagedKVCache
from mlx_mfa.turboquant import TurboQuantKVCache
from mlx_mfa.external_cache import LocalHostKVStoreAdapter

sys.path.insert(0, "tests")
F16, BF16 = mx.float16, mx.bfloat16


def _kv(dk, dv, b=1, h=2, n=1, d=128):
    k = mx.zeros((b, h, n, d), dk)
    v = mx.zeros((b, h, n, d), dv)
    mx.eval(k, v)
    return k, v


def _dense(dk, dv, cache_dtype=F16):
    DenseKVCache(1, 2, 128, 64, dtype=cache_dtype).append(*_kv(dk, dv))


def _quant(dk, dv, cache_dtype=F16):
    QuantizedKVCache(1, 2, 128, 64, dtype=cache_dtype).append(*_kv(dk, dv))


def _paged(dk, dv, cache_dtype=F16):
    PagedKVCache(8, 16, 2, 128, dtype=cache_dtype).append(*_kv(dk, dv))


def _tqkv(dk, dv):
    TurboQuantKVCache().append(*_kv(dk, dv))


def _host(dk, dv):
    LocalHostKVStoreAdapter().put(0, *_kv(dk, dv))


def _tqp(dk, dv):
    import test_phase3_iii2_tq_decode as T
    ctx, _ = T._mkctx(3)
    ctx.append(*_kv(dk, dv))


_ALL = [("dense", _dense), ("quant", _quant), ("paged", _paged),
        ("tqkv", _tqkv), ("host", _host), ("tqp", _tqp)]


# ── K↔V dtype consistency: all 6 surfaces raise on K fp16 / V bf16 ───────────────
@pytest.mark.parametrize("name,fn", _ALL, ids=[s[0] for s in _ALL])
def test_kv_dtype_mismatch_raises(name, fn):
    with pytest.raises((ValueError, Exception)):
        fn(F16, BF16)


# ── storage caches: bf16 append into an fp16 cache raises (no silent cast) ───────
@pytest.mark.parametrize("name,fn", [("dense", _dense), ("quant", _quant), ("paged", _paged)],
                         ids=["dense", "quant", "paged"])
def test_storage_cache_rejects_mismatched_append(name, fn):
    with pytest.raises((ValueError, Exception)):
        fn(BF16, BF16, cache_dtype=F16)          # bf16 append into fp16 cache
    fn(F16, F16, cache_dtype=F16)                # matched → accepts


# ── quantizing / host: accept BOTH fp16 and bf16 (no regression) ────────────────
@pytest.mark.parametrize("dt", [F16, BF16])
def test_quantizing_and_host_accept_fp16_bf16(dt):
    _tqkv(dt, dt)
    _host(dt, dt)


# ── shape axes intact (no regression from the dtype addition) ───────────────────
def test_shape_axes_still_enforced():
    # batch mismatch still raises with matched dtype
    with pytest.raises((ValueError, Exception)):
        DenseKVCache(1, 2, 128, 64, dtype=F16).append(
            mx.zeros((1, 2, 1, 128), F16), mx.zeros((2, 2, 1, 128), F16))
    # head mismatch still raises
    with pytest.raises((ValueError, Exception)):
        DenseKVCache(1, 2, 128, 64, dtype=F16).append(
            mx.zeros((1, 2, 1, 128), F16), mx.zeros((1, 1, 1, 128), F16))


# ── D-contract unchanged: register_prefix (attention call) accepts asym D_v ──────
def test_register_prefix_asym_dv_unchanged():
    import mlx_mfa
    q = mx.random.normal((1, 8, 16, 128)).astype(F16)
    k = mx.random.normal((1, 2, 16, 128)).astype(F16)
    v = mx.random.normal((1, 2, 16, 64)).astype(F16)
    mx.eval(q, k, v)
    o = mlx_mfa.make_shared_prefix_cache(q, k, v)
    mx.eval(o[0])
    assert bool(np.isfinite(np.array(o[0].astype(mx.float32))).all())
