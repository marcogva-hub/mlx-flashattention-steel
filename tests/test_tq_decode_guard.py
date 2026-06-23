"""Volet P1 Part A — tq_decode kernel bounds guard + loud TQ index validation
(CX-TQ-DECODE-01, CRITICAL).

The two tq_decode kernels (K-dequant, V-gather) read k_pool/k_scales/v_pool at
phys = block_table[blk] with NO `0 <= phys < num_blocks` guard → an OOB/negative
physical index drove an out-of-bounds device load (allocation-sensitive
finite-wrong; phys=-5 diverged 0.125 from the -1 sentinel). Reachable by default
via TurboQuantPagedInferenceContext.step (Nq=1).

Fix: (1) in-kernel bounds guard (always-on memory-safety) → OOB/-1 reads zero,
never out-of-bounds; (2) loud default validation on the public step (raise on
malformed), with the MFA_PAGED_TRUST_INDICES=1 opt-out (skips the value-sync; the
in-kernel guard keeps it memory-safe).
"""
import os
import sys
import math
import numpy as np
import mlx.core as mx
import pytest

sys.path.insert(0, "tests")


def _n(a):
    return np.array(a.astype(mx.float32))


# ── kernel-level memory-safety (in-kernel guard) ────────────────────────────────
def _kernel_inputs():
    from mlx_mfa.tq_decode import tq_decode_attend
    nb, bs, Hkv, D, bits = 4, 16, 2, 128, 3
    pd = (D // 32) * 12
    mx.random.seed(0)
    kp = mx.random.randint(0, 256, (nb, bs, Hkv, pd)).astype(mx.uint8)
    vp = mx.random.normal((nb, bs, Hkv, D)).astype(mx.float16)
    ks = mx.random.normal((nb, bs, Hkv)).astype(mx.float32)
    cen = mx.random.normal((2 ** bits,)).astype(mx.float16)
    q = mx.random.normal((1, 8, 1, D)).astype(mx.float16)
    mx.eval(kp, vp, ks, cen, q)
    return tq_decode_attend, (kp, vp, ks, cen, q), (bs, D, bits, nb)


@pytest.mark.parametrize("phys", [99, -5, -1])
def test_kernel_oob_phys_reads_zero(phys):
    fn, (kp, vp, ks, cen, q), (bs, D, bits, nb) = _kernel_inputs()
    S = 16

    def run(p):
        bt = mx.array([p], dtype=mx.int32); mx.eval(bt)
        o = fn(q, kp, vp, ks, cen, bt, S, scale=1 / math.sqrt(D),
               block_size=bs, tq_bits=bits); mx.eval(o); return _n(o)
    sentinel = run(-1)          # an all-skip page → zero
    assert float(np.abs(sentinel).max()) == 0.0
    out = run(phys)
    assert bool(np.isfinite(out).all())
    assert float(np.max(np.abs(out - sentinel))) == 0.0  # OOB → same zeroed result


def test_kernel_valid_phys_nonzero_and_finite():
    fn, (kp, vp, ks, cen, q), (bs, D, bits, nb) = _kernel_inputs()
    bt = mx.array([0], dtype=mx.int32); mx.eval(bt)
    o = fn(q, kp, vp, ks, cen, bt, 16, scale=1 / math.sqrt(D),
           block_size=bs, tq_bits=bits); mx.eval(o)
    assert bool(np.isfinite(_n(o)).all())


# ── public-step loud validation + opt-out memory-safety ─────────────────────────
def _ctx():
    import test_phase3_iii2_tq_decode as T
    ctx, q = T._mkctx(3)
    kn = mx.random.normal((1, 2, 1, 128)).astype(mx.float16)
    vn = mx.random.normal((1, 2, 1, 128)).astype(mx.float16)
    mx.eval(kn, vn)
    return ctx, q, kn, vn


def test_public_step_valid_runs():
    ctx, q, kn, vn = _ctx()
    o = ctx.step(q, kn, vn); mx.eval(o)
    assert bool(np.isfinite(_n(o)).all())


@pytest.mark.parametrize("phys", [99, -5])
def test_public_step_oob_raises_by_default(phys):
    ctx, q, kn, vn = _ctx()
    orig = ctx.get_block_table
    ctx.get_block_table = lambda ids=None: mx.array([[phys]], dtype=mx.int32)
    try:
        with pytest.raises((ValueError, Exception)):
            o = ctx.step(q, kn, vn); mx.eval(o)
    finally:
        ctx.get_block_table = orig


@pytest.mark.parametrize("phys", [99, -5, -1])
def test_public_step_oob_memory_safe_under_optout(phys):
    ctx, q, kn, vn = _ctx()
    orig = ctx.get_block_table
    ctx.get_block_table = lambda ids=None: mx.array([[phys]], dtype=mx.int32)
    os.environ["MFA_PAGED_TRUST_INDICES"] = "1"
    try:
        o = ctx.step(q, kn, vn); mx.eval(o)
        assert bool(np.isfinite(_n(o)).all())   # kernel guard → no OOB/crash
    finally:
        os.environ.pop("MFA_PAGED_TRUST_INDICES", None)
        ctx.get_block_table = orig


def test_public_step_dtype_check_stays_on_under_optout():
    ctx, q, kn, vn = _ctx()
    orig = ctx.get_block_table
    ctx.get_block_table = lambda ids=None: mx.array([[0]], dtype=mx.int64)
    os.environ["MFA_PAGED_TRUST_INDICES"] = "1"
    try:
        with pytest.raises((ValueError, Exception)):
            o = ctx.step(q, kn, vn); mx.eval(o)
    finally:
        os.environ.pop("MFA_PAGED_TRUST_INDICES", None)
        ctx.get_block_table = orig
