"""IV-D1 regression guard: TQ decode append-eval deferral is bit-identical.

The tq_v=False decode branch defers append()'s per-step eager mx.eval (the gather
reads the pools as graph-inputs, so eval(o) materializes them).  This is a pure
materialization-ORDERING change — it must produce results bit-identical to the eager
path.  Guards the inverse-add_temporary stale-read class (IV-D1).
"""
from __future__ import annotations
import math
import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import TurboQuantPagedInferenceContext, get_device_info

_HAS_NAX = bool(get_device_info().get("is_m5_plus", False))
pytestmark = pytest.mark.skipif(not _HAS_NAX, reason="TQ decode gather path requires M5+ NAX")

S0, KSTEPS, Hq, Hkv, D, bs, bits = 256, 40, 8, 2, 128, 64, 3


def _mkctx():
    ctx = TurboQuantPagedInferenceContext(num_blocks=(S0 + KSTEPS) // bs + 16,
                                          block_size=bs, H_kv=Hkv, D=D,
                                          tq_bits=bits, tq_v=False)
    mx.random.seed(7)
    pq = (mx.random.uniform(-1, 1, (1, Hq, S0, D)) * 0.1).astype(mx.float16)
    pk = (mx.random.uniform(-1, 1, (1, Hkv, S0, D)) * 0.1).astype(mx.float16)
    pv = (mx.random.uniform(-1, 1, (1, Hkv, S0, D)) * 0.1).astype(mx.float16)
    ctx.prefill(pq, pk, pv)
    return ctx


def _step_inputs(i):
    mx.random.seed(1000 + i)
    sq = (mx.random.uniform(-1, 1, (1, Hq, 1, D)) * 0.1).astype(mx.float16)
    sk = (mx.random.uniform(-1, 1, (1, Hkv, 1, D)) * 0.1).astype(mx.float16)
    sv = (mx.random.uniform(-1, 1, (1, Hkv, 1, D)) * 0.1).astype(mx.float16)
    mx.eval(sq, sk, sv)
    return sq, sk, sv


def _force_eager(ctx):
    """Wrap append so the defer flag step() passes is overridden to eager — the
    only difference from the deferred ctx is the append-eval timing (faithful A/B,
    same step() rotation + decode path)."""
    orig = ctx.append
    def eager(k, v, *, seq_id=0, defer_pool_materialize=False):
        return orig(k, v, seq_id=seq_id, defer_pool_materialize=False)
    ctx.append = eager
    return ctx


def test_deferred_append_bit_identical_to_eager():
    """Many-step decode via step(): deferred == forced-eager, bit-for-bit, under churn."""
    scale = 1.0 / math.sqrt(D)
    ctx_def = _mkctx()              # real step(): deferred on tq_v=False decode branch
    ctx_eager = _force_eager(_mkctx())  # step() but append forced eager
    for i in range(KSTEPS):
        sq, sk, sv = _step_inputs(i)
        o_def = ctx_def.step(sq, sk, sv, scale=scale)
        # churn between the lazy step and its eval (inverse-add_temporary trigger)
        c = mx.random.uniform(0, 1, (512, 512)).astype(mx.float16)
        mx.eval(c @ c.T)
        o_eager = ctx_eager.step(sq, sk, sv, scale=scale)
        mx.eval(o_def, o_eager)
        d = float(np.max(np.abs(np.array(o_def.astype(mx.float32))
                                - np.array(o_eager.astype(mx.float32)))))
        assert d == 0.0, f"step {i}: deferred vs eager diff={d:.3e} (must be bit-identical)"
        assert bool(mx.isfinite(o_def).all().item())


def _mkctx_tqv(tq_v):
    ctx = TurboQuantPagedInferenceContext(num_blocks=(S0 + KSTEPS) // bs + 16,
                                          block_size=bs, H_kv=Hkv, D=D,
                                          tq_bits=bits, tq_v=tq_v)
    mx.random.seed(7)
    pq = (mx.random.uniform(-1, 1, (1, Hq, S0, D)) * 0.1).astype(mx.float16)
    pk = (mx.random.uniform(-1, 1, (1, Hkv, S0, D)) * 0.1).astype(mx.float16)
    pv = (mx.random.uniform(-1, 1, (1, Hkv, S0, D)) * 0.1).astype(mx.float16)
    ctx.prefill(pq, pk, pv)
    return ctx


def test_tqv_true_combined_eval_bit_identical_to_eager():
    """IV-D2: tq_v=True default decode — deferred+combined-eval == forced-eager, bit-for-bit."""
    scale = 1.0 / math.sqrt(D)
    ctx_def = _mkctx_tqv(True)              # combined eval folds packed-V (deferred append)
    ctx_eager = _force_eager(_mkctx_tqv(True))  # append eager + combined eval (same result)
    for i in range(KSTEPS):
        sq, sk, sv = _step_inputs(i)
        o_def = ctx_def.step(sq, sk, sv, scale=scale)
        c = mx.random.uniform(0, 1, (512, 512)).astype(mx.float16)
        mx.eval(c @ c.T)
        o_eager = ctx_eager.step(sq, sk, sv, scale=scale)
        mx.eval(o_def, o_eager)
        d = float(np.max(np.abs(np.array(o_def.astype(mx.float32))
                                - np.array(o_eager.astype(mx.float32)))))
        assert d == 0.0, f"step {i}: tq_v=True deferred vs eager diff={d:.3e}"
        assert bool(mx.isfinite(o_def).all().item())


def test_fused_read_after_decode_sees_materialized_packed_v():
    """IV-D2 edge: a fused read after decode steps must see the packed-V the combined
    evals materialized — bit-identical to the all-eager path (no stale read)."""
    import os
    scale = 1.0 / math.sqrt(D)
    ctx_def = _mkctx_tqv(True)
    ctx_eager = _force_eager(_mkctx_tqv(True))
    for i in range(20):
        sq, sk, sv = _step_inputs(i)
        ctx_def.step(sq, sk, sv, scale=scale)
        ctx_eager.step(sq, sk, sv, scale=scale)
    # now force the fused path (reads packed-V raw) on both
    os.environ["MFA_DISABLE_TQ_DECODE_SDPA"] = "1"
    try:
        fq, fk, fv = _step_inputs(9999)
        od = ctx_def.step(fq, fk, fv, scale=scale)
        oe = ctx_eager.step(fq, fk, fv, scale=scale)
        mx.eval(od, oe)
    finally:
        os.environ.pop("MFA_DISABLE_TQ_DECODE_SDPA", None)
    d = float(np.max(np.abs(np.array(od.astype(mx.float32)) - np.array(oe.astype(mx.float32)))))
    assert bool(mx.isfinite(od).all().item())
    assert d == 0.0, f"fused-read-after-decode diff={d:.3e} (stale packed-V read?)"
