"""Volet P5 — shared K/V persistence-validation contract (batch axis + D rule).

Re-run #2 found the paged appenders enforced only a SUBSET of axes (missed
batch): `PagedKVCache.append` indexed `k[0]` without a batch check (batch>1
silently sliced to [0]); `TurboQuantPagedInferenceContext.append` validated
rank/heads/D/token but not batch. Root cause = per-site partial contracts. P5
defines the contract ONCE (`_persist_validate.assert_kv_persist_compat`,
rank/batch/heads/token/D) and routes EVERY persistence surface through it, so no
surface can enforce a partial subset.

D-contract: aligned with the function surface — caches are single-`D` buffers so
persistence requires `D_v == D_k`; the asymmetric-`D_v` allowance is for the dense
*attention call* (`make_shared_prefix_cache`), which is NOT a persistence surface
and keeps accepting it.
"""
import sys
import numpy as np
import mlx.core as mx
import pytest
from mlx_mfa.attention import DenseKVCache, QuantizedKVCache, PagedKVCache
from mlx_mfa.turboquant import TurboQuantKVCache
from mlx_mfa.kv_cache import adapt_kv_cache
from mlx_mfa.external_cache import LocalHostKVStoreAdapter

sys.path.insert(0, "tests")


def _b(batch, h=2, n=1, d=128):
    a = mx.zeros((batch, h, n, d), dtype=mx.float16)
    mx.eval(a)
    return a


# ── batch axis: every persistence surface raises on batch-mismatch, accepts valid
def _dense(kb, vb):
    DenseKVCache(1, 2, 128, 64).append(_b(kb), _b(vb))


def _quant(kb, vb):
    QuantizedKVCache(1, 2, 128, 64).append(_b(kb), _b(vb))


def _tqkv(kb, vb):
    TurboQuantKVCache().append(_b(kb), _b(vb))


def _host(kb, vb):
    LocalHostKVStoreAdapter().put(0, _b(kb), _b(vb))


def _paged(kb, vb):
    PagedKVCache(8, 16, 2, 128).append(_b(kb), _b(vb))


def _tqp(kb, vb):
    import test_phase3_iii2_tq_decode as T
    ctx, _ = T._mkctx(3)
    ctx.append(_b(kb), _b(vb))


_SURFACES = [("dense", _dense), ("quant", _quant), ("tqkv", _tqkv),
             ("host", _host), ("paged", _paged), ("tqp", _tqp)]


@pytest.mark.parametrize("name,fn", _SURFACES, ids=[s[0] for s in _SURFACES])
def test_batch_mismatch_raises(name, fn):
    with pytest.raises((ValueError, Exception)):
        fn(1, 2)            # K batch=1, V batch=2


@pytest.mark.parametrize("name,fn", _SURFACES, ids=[s[0] for s in _SURFACES])
def test_valid_batch_accepts(name, fn):
    fn(1, 1)                # matched batch=1 — valid


def test_paged_batch_gt1_raises_not_silent_slice():
    # the HIGH #1 case: a paged append given batch>1 must RAISE, not silently
    # slice [0] and drop the rest.
    with pytest.raises((ValueError, Exception)):
        _paged(2, 2)


# ── single-source: every persistence surface routes through the shared helper ───
def test_single_source_helper():
    import pathlib
    root = pathlib.Path(__file__).parent.parent / "mlx_mfa"
    callers = set()
    for f in root.glob("*.py"):
        if f.name == "_persist_validate.py":
            continue
        if "assert_kv_persist_compat(" in f.read_text():
            callers.add(f.name)
    for needed in ("attention.py", "inference.py", "turboquant.py", "external_cache.py"):
        assert needed in callers, f"{needed} does not route through the shared helper"


# ── D-contract aligned with the function surface ────────────────────────────────
def test_dense_attention_call_allows_asym_dv():
    # make_shared_prefix_cache is an ATTENTION CALL (not persistence) → asym D_v ok
    import mlx_mfa
    q = mx.random.normal((1, 8, 16, 128)).astype(mx.float16)
    k = mx.random.normal((1, 2, 16, 128)).astype(mx.float16)
    v = mx.random.normal((1, 2, 16, 64)).astype(mx.float16)   # D_v=64 != D_k=128
    mx.eval(q, k, v)
    o = mlx_mfa.make_shared_prefix_cache(q, k, v)
    mx.eval(o[0])
    assert bool(np.isfinite(np.array(o[0].astype(mx.float32))).all())


def test_persistence_requires_dv_eq_dk():
    # single-D cache buffers → persistence rejects D_v != D_k
    k = mx.zeros((1, 2, 1, 128), mx.float16)
    v = mx.zeros((1, 2, 1, 64), mx.float16)
    mx.eval(k, v)
    with pytest.raises((ValueError, Exception)):
        DenseKVCache(1, 2, 128, 64).append(k, v)
