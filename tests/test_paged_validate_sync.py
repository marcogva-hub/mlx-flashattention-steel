"""Pre-release perf-audit: paged block_table/seq_lens value-range check.

`_validate_paged_block_table` range-checked block_table/seq_lens VALUES with four
separate `mx.min/max(...).item()` device syncs — measured +0.69 ms on a 0.26 ms
Nq=1 decode (3.7×). Fix: batch the reductions into ONE sync (default, reject
still bites) + an `MFA_PAGED_TRUST_INDICES=1` opt-out for hot loops (skips the
sync; the paged gather kernel still bounds-guards `phys < num_blocks`, so OOB
stays memory-safe). This locks: default reject still bites, opt-out skips the
raise but stays memory-safe (finite), and byteΔ=0 between the two on valid input.
"""
import os
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa

B, Hq, Hk, D, BS = 1, 8, 2, 128, 16
S = 512
NB = S // BS
SC = 1.0 / math.sqrt(D)


def _mk(bt=None, sl=None):
    mx.random.seed(0)
    kp = mx.random.normal((NB + 2, BS, Hk, D)).astype(mx.float16)
    vp = mx.random.normal((NB + 2, BS, Hk, D)).astype(mx.float16)
    q = mx.random.normal((B, Hq, 4, D)).astype(mx.float16)
    bt = bt if bt is not None else mx.array([list(range(NB))], dtype=mx.int32)
    sl = sl if sl is not None else mx.array([S], dtype=mx.int32)
    mx.eval(kp, vp, q, bt, sl)
    return q, kp, vp, bt, sl


def _call(a):
    return mlx_mfa.flash_attention_paged(*a, scale=SC, causal=True, block_size=BS)


def _n(x):
    return np.array(x.astype(mx.float32))


def test_default_reject_still_bites():
    # default (validate on): OOB page / negative seq_lens / int64 metadata raise.
    with pytest.raises((ValueError, Exception)):
        mx.eval(_call(_mk(bt=mx.array([[99] + list(range(NB - 1))], dtype=mx.int32))))
    with pytest.raises((ValueError, Exception)):
        mx.eval(_call(_mk(sl=mx.array([-5], dtype=mx.int32))))
    with pytest.raises((ValueError, Exception)):
        mx.eval(_call(_mk(bt=mx.array([list(range(NB))], dtype=mx.int64))))


def test_byte_identical_default_vs_optout_on_valid():
    o_def = _call(_mk()); mx.eval(o_def)
    os.environ["MFA_PAGED_TRUST_INDICES"] = "1"
    try:
        o_opt = _call(_mk()); mx.eval(o_opt)
    finally:
        os.environ.pop("MFA_PAGED_TRUST_INDICES", None)
    assert float(np.max(np.abs(_n(o_def) - _n(o_opt)))) == 0.0


def test_optout_skips_raise_but_is_memory_safe():
    # opt-out: OOB block_table no longer raises early, but the in-kernel
    # `phys < num_blocks` guard keeps it memory-safe → finite output, no crash.
    os.environ["MFA_PAGED_TRUST_INDICES"] = "1"
    try:
        o = _call(_mk(bt=mx.array([[99] + list(range(NB - 1))], dtype=mx.int32)))
        mx.eval(o)
        assert bool(np.isfinite(_n(o)).all())
    finally:
        os.environ.pop("MFA_PAGED_TRUST_INDICES", None)


def test_optout_does_not_skip_metadata_checks():
    # the cheap metadata checks (dtype/rank/cardinality) stay on even with the
    # value-check opted out — int64 block_table still raises.
    os.environ["MFA_PAGED_TRUST_INDICES"] = "1"
    try:
        with pytest.raises((ValueError, Exception)):
            mx.eval(_call(_mk(bt=mx.array([list(range(NB))], dtype=mx.int64))))
    finally:
        os.environ.pop("MFA_PAGED_TRUST_INDICES", None)


def test_minus_one_padding_valid():
    o = _call(_mk(bt=mx.array([list(range(NB - 1)) + [-1]], dtype=mx.int32)))
    mx.eval(o)
    assert bool(np.isfinite(_n(o)).all())
