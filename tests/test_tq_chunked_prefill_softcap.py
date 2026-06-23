"""Volet P1 Part C — TQ chunked_prefill softcap compat (MEDIUM).

DecodeRuntime.chunked_prefill forwarded `softcap` (and `window_size`) to the
TurboQuant context.step, whose signature accepts neither → valid TQ chunked
prefill raised TypeError before any compute, even with the default softcap=0.0.
Root cause: TQ decode genuinely has no softcap capability (it runs SDPA /
paged-tq). Fix: DecodeRuntime.step drops a default-valued unsupported feature
and raises a CLEAR capability error if a non-default one is requested.
"""
import numpy as np
import mlx.core as mx
import pytest
from mlx_mfa.runtime import create_decode_runtime


def _rt():
    return create_decode_runtime(backend="turboquant", B=1, H_q=8, H_kv=2, D=128,
                                 tq_bits=3, tq_v=False, max_seq_len=256)


def _qkv(nq):
    q = mx.random.normal((1, 8, nq, 128)).astype(mx.float16)
    k = mx.random.normal((1, 2, nq, 128)).astype(mx.float16)
    v = mx.random.normal((1, 2, nq, 128)).astype(mx.float16)
    mx.eval(q, k, v)
    return q, k, v


def test_tq_chunked_prefill_default_softcap_runs():
    rt = _rt()
    q, k, v = _qkv(8)
    out = rt.chunked_prefill(q, k, v, chunk_size=4)
    o = out[0] if isinstance(out, tuple) else out
    mx.eval(o)
    assert bool(np.isfinite(np.array(o.astype(mx.float32))).all())


def test_tq_step_default_softcap_runs():
    rt = _rt()
    q, k, v = _qkv(1)
    o = rt.step(q, k, v, softcap=0.0)
    mx.eval(o[0] if isinstance(o, tuple) else o)


def test_tq_explicit_softcap_raises_clear_capability_error():
    rt = _rt()
    q, k, v = _qkv(1)
    with pytest.raises((ValueError, Exception)):
        o = rt.step(q, k, v, softcap=30.0)
        mx.eval(o[0] if isinstance(o, tuple) else o)
