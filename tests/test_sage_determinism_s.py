"""Volet S — Sage forward determinism (CX-R6-02).

mfa_sage_forward shares one threadgroup buffer (KV_smem) for K then V each
K-tile, but lacked a barrier between iteration kb's P@V (reading KV_smem as V) and
iteration kb+1's K cooperative load (overwriting KV_smem as K). A fast simdgroup
raced a slow one's V-read → NONDETERMINISTIC output for identical inputs (silent-
wrong on a public path). Manifested only at N≥512 (multi-tile = cross-iteration
reuse); single-tile (N<512) had no reuse so looked fine. Root cause was a missing
start-of-loop barrier (the STEEL forward this kernel derived from has it).

This locks determinism: N≥512, both MHA and GQA, f16/bf16 — identical inputs must
give byte-identical output over N runs. Bite: remove the barrier → these fail.
"""
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa


def _runs(Hq, Hk, N, D, dt, smooth, nruns=20):
    outs = []
    for _ in range(nruns):
        mx.random.seed(0)
        q = mx.random.normal((1, Hq, N, D)).astype(dt)
        k = mx.random.normal((1, Hk, N, D)).astype(dt)
        v = mx.random.normal((1, Hk, N, D)).astype(dt)
        mx.eval(q, k, v)
        o = mlx_mfa.sage_attention(q, k, v, scale=1.0 / math.sqrt(D),
                                   apply_smooth_k=smooth)
        mx.eval(o)
        outs.append(np.array(o.astype(mx.float32)))
    return max(float(np.max(np.abs(outs[0] - outs[i]))) for i in range(1, nruns))


@pytest.mark.parametrize("Hq,Hk", [(8, 2), (16, 2), (8, 1), (12, 4), (4, 4), (2, 2)])
@pytest.mark.parametrize("N", [512, 1024])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
def test_sage_deterministic_multitile(Hq, Hk, N, D, dt):
    md = _runs(Hq, Hk, N, D, dt, smooth=True)
    assert md == 0.0, (
        f"sage Hq{Hq}/Hk{Hk} N{N} D{D} {dt}: nondeterministic, max pairwise "
        f"byteΔ={md:.3e} over 20 identical-input runs (KV_smem reuse race).")


@pytest.mark.parametrize("smooth", [True, False])
def test_sage_deterministic_smooth_independent(smooth):
    # The race is in mfa_sage_forward, independent of apply_smooth_k.
    assert _runs(8, 2, 1024, 128, mx.float16, smooth=smooth) == 0.0


def test_sage_gqa_consistent_oracle_relerr():
    """Post-fix, relerr vs fp64 oracle is a tight distribution (the int8 floor),
    not the 0.011-0.278 spread the race produced."""
    def f64(a):
        return np.array(a.astype(mx.float32)).astype(np.float64)
    Hq, Hk, N, D = 8, 2, 512, 128
    g = Hq // Hk
    errs = []
    for _ in range(8):
        mx.random.seed(0)
        q = mx.random.normal((1, Hq, N, D)).astype(mx.float16)
        k = mx.random.normal((1, Hk, N, D)).astype(mx.float16)
        v = mx.random.normal((1, Hk, N, D)).astype(mx.float16)
        mx.eval(q, k, v)
        o = mlx_mfa.sage_attention(q, k, v, scale=1.0 / math.sqrt(D))
        mx.eval(o)
        qf, kf, vf = f64(q), f64(k), f64(v)
        kk, vv = np.repeat(kf, g, 1), np.repeat(vf, g, 1)
        s = np.einsum("bhnd,bhmd->bhnm", qf, kk) / math.sqrt(D)
        s -= s.max(-1, keepdims=True)
        e = np.exp(s)
        p = e / e.sum(-1, keepdims=True)
        ref = np.einsum("bhnm,bhmd->bhnd", p, vv)
        errs.append(float(np.max(np.abs(f64(o) - ref)) / (np.max(np.abs(ref)) + 1e-9)))
    assert max(errs) - min(errs) < 1e-4, f"relerr spread {min(errs):.3e}..{max(errs):.3e} (race?)"
    assert max(errs) < 5e-2, f"relerr {max(errs):.3e} exceeds int8 sage floor"
