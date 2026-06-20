"""Lock: the dense fused dKdV backward at BK=16 (TK=1) is NUMERICALLY CORRECT.

History (corrected 2026-06-20): II-6 found the fused dKdV at BK=16 corrupts — the paired
16×32×16 MMA (`for ik+=2` writing `frag_at(iq,ik+1)`) has TK=BK/16=1 → the unpaired second
fragment writes OOB (dV err ≈ 35.9 at unit scale). **Phase II-8 item 3 fixed it** with a TK=1
odd-tail scratch path (NAAttentionKernel.cpp ~6069/6216: load `tail_lim ≤ 16` K-rows, run the
paired MMA into the real `frag_at(iq,ik)` + a throwaway `scratch` fragment, consume only the
real one). The dense fused dispatch passes `generator_handles_odd_tk=true`
(mfa_v6_nax_primitive.cpp:2036) so the BK%32 guard admits BK=16 ONLY through that validated
tail. A later session (c8fb3f4) mis-documented BK=16 as "still corrupt / not worth fixing"
WITHOUT testing the tail — this lock corrects the record and prevents regression.

Disposition (Point 1, 2026-06-20): BK=16-fused is correct AND ~1.00–1.03× vs split@BK=32
(parity-to-noise; far below split's 2.18–3.06× vs SDPA-vjp) → **split stays the default**;
BK=16-fused is correct-but-not-advantageous. This test locks correctness only.

Realistic scale (≈1.0) — the 0.1-scale fixtures hid the original corruption.
"""
from __future__ import annotations

import math
import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_mfa


def _has_nax() -> bool:
    try:
        return bool(mlx_mfa.has_nax())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_nax(), reason="V6 NAX fused backward requires M5 + _ext")


def _fp32_grads(q, k, v, sc, causal):
    N = q.shape[2]
    def f(q, k, v):
        qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
        s = (qf @ kf.swapaxes(-1, -2)) * sc
        if causal:
            s = s + mx.triu(mx.full((N, N), -1e30, dtype=mx.float32), k=1)
        return (mx.softmax(s, -1) @ vf).sum()
    return mx.grad(f, argnums=(0, 1, 2))(q, k, v)


@pytest.mark.parametrize("causal", [False, True])
def test_fused_bk16_oracle_correct_unit_scale(causal):
    """BK=16 (TK=1 tail) dense fused dKdV — dq/dk/dv finite + within the fp16-bwd floor
    of an independent fp32 oracle at UNIT scale. Pre-II-8-tail this gave dV ≈ 35.9 → the
    <0.5 asserts BITE on the corrupt path."""
    snap = {k: os.environ.get(k) for k in
            ("MFA_ENABLE_V6_BACKWARD", "MFA_V6_BWD_KERNEL", "MFA_V6BWDF_BK")}
    os.environ["MFA_ENABLE_V6_BACKWARD"] = "1"
    os.environ["MFA_V6_BWD_KERNEL"] = "fused"     # force the fused dKdV kernel (D=64)
    os.environ["MFA_V6BWDF_BK"] = "16"            # the TK=1 odd-tail path
    try:
        B, H, N, D = 1, 4, 2048, 64
        mx.random.seed(0)
        f = lambda: (mx.random.normal((B, H, N, D)) * 1.0).astype(mx.float16)
        q, k, v = f(), f(), f()
        mx.eval(q, k, v)
        sc = 1.0 / math.sqrt(D)
        dq, dk, dv = mx.grad(
            lambda q, k, v: mlx_mfa.flash_attention(q, k, v, scale=sc, causal=causal).sum(),
            argnums=(0, 1, 2))(q, k, v)
        mx.eval(dq, dk, dv)
        for name, g in (("dq", dq), ("dk", dk), ("dv", dv)):
            assert bool(mx.all(mx.isfinite(g)).item()), f"{name} non-finite at BK=16 unit scale"
        rq, rk, rv = _fp32_grads(q, k, v, sc, causal)
        mx.eval(rq, rk, rv)
        err = lambda a, b: float(np.abs(np.array(a.astype(mx.float32)) - np.array(b)).max())
        assert err(dq, rq) < 0.5, f"dq err {err(dq, rq):.3f} (BK=16 TK=1 regression?)"
        assert err(dk, rk) < 0.5, f"dk err {err(dk, rk):.3f} (BK=16 TK=1 regression?)"
        # dV is the fragment the OOB corruption hit (≈35.9 pre-tail). The tightest cell.
        assert err(dv, rv) < 0.5, f"dv err {err(dv, rv):.3f} (BK=16 OOB 2nd-frag regression?)"
    finally:
        for kk, vv in snap.items():
            if vv is None:
                os.environ.pop(kk, None)
            else:
                os.environ[kk] = vv
