"""NEPB-05 (code review 2026-09, P1 PUBLISHED) — backward through a data-dependent mask.

mx.grad through flash_attention_sparse raised "[async_eval] Not allowed inside a graph
transformation" whenever the block mask was derived from the differentiated inputs
(dynamic top-k selection — exactly what sla_attention does); mx.stop_gradient did not
help.  The sparse-bias cache helpers (and sparse_attention_dispatch's density read)
called mx.async_eval to materialize + cache.  Fix: materialization is attempted;
inside a transformation the helpers skip the cache (inserting a traced array into a
global cache would also leak the tracer) and return the graph-safe array.  Only that
exact MLX error is handled — anything else still propagates (Rule 8).
"""
from __future__ import annotations

import mlx.core as mx
import pytest

import mlx_mfa.attention as att
from mlx_mfa import is_mfa_available
from mlx_mfa.attention import flash_attention_sparse
from mlx_mfa.lcsa_nax import sparse_attention_dispatch

pytestmark = pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")


def _dyn_mask(q, k, nq, D):
    qp = q.reshape(q.shape[0], q.shape[1], nq, 32, D).mean(3)
    kp = k.reshape(k.shape[0], k.shape[1], nq, 32, D).mean(3)
    s = (qp @ mx.swapaxes(kp, -1, -2))[0, 0]
    return (s > mx.mean(s)) | mx.eye(nq, dtype=mx.bool_)


def _inputs(N, D=64):
    mx.random.seed(N)
    return tuple(mx.random.normal((1, 1, N, D)).astype(mx.float16) for _ in range(3))


def _err(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


@pytest.mark.parametrize("stop_grad", [False, True])
@pytest.mark.parametrize("N", [512, 8192])      # per-head path / default gate cell
def test_grad_with_input_dependent_mask(N, stop_grad):
    D = 64
    q, k, v = _inputs(N, D)
    nq = N // 32

    def f_dyn(q, k, v):
        m = _dyn_mask(q, k, nq, D)
        if stop_grad:
            m = mx.stop_gradient(m)
        return flash_attention_sparse(q, k, v, m).astype(mx.float32).sum()

    m_const = _dyn_mask(q, k, nq, D)
    mx.eval(m_const)

    def f_const(q, k, v):
        return flash_attention_sparse(q, k, v, m_const).astype(mx.float32).sum()

    g_dyn = mx.grad(f_dyn, argnums=(0, 1, 2))(q, k, v)
    g_ref = mx.grad(f_const, argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g_dyn, *g_ref)
    for a, b in zip(g_dyn, g_ref):
        assert bool(mx.all(mx.isfinite(a.astype(mx.float32))).item())
        assert _err(a, b) < 1e-3


def test_no_cache_insertion_under_transformation():
    N, D = 512, 64
    q, k, v = _inputs(N, D)
    att._SPARSE_BIAS_CACHE.clear()
    att._SPARSE_SANITIZED_BIAS_CACHE.clear()
    mx.eval(*mx.grad(lambda q, k, v: flash_attention_sparse(
        q, k, v, _dyn_mask(q, k, N // 32, D)).astype(mx.float32).sum(),
        argnums=(0, 1, 2))(q, k, v))
    assert len(att._SPARSE_BIAS_CACHE) == 0 and len(att._SPARSE_SANITIZED_BIAS_CACHE) == 0
    m = _dyn_mask(q, k, N // 32, D)
    mx.eval(m)
    mx.eval(flash_attention_sparse(q, k, v, m))        # plain call: caching still works
    assert len(att._SPARSE_BIAS_CACHE) == 1


def test_dispatcher_grad_with_input_dependent_mask():
    N, D = 512, 64
    q, k, v = _inputs(N, D)
    g = mx.grad(lambda q, k, v: sparse_attention_dispatch(
        q, k, v, _dyn_mask(q, k, N // 32, D), block_tile=32).astype(mx.float32).sum(),
        argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g)
    assert all(bool(mx.all(mx.isfinite(x.astype(mx.float32))).item()) for x in g)
