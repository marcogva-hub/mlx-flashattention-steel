"""II-6 regression lock — the fused V6NAX backward paired-MMA out-of-bounds.

ROOT CAUSE (source-verified, NAAttentionKernel.cpp ~2885 + mfa_v6_nax_primitive.cpp):
the V6NAX QK / dK-dV recompute emits a PAIRED 16x32x16 cooperative MMA
(`for (ik = 0; ik < TK; ik += 2)` writing `frag_at(iq, ik+1)`), so TK = BK/16 must be
EVEN. At BK=16 → TK=1 the K-load reads 16 rows past the tile and the second output
fragment lands out of bounds → dK/dV corruption. It is SCALE-DEPENDENT (OOB garbage
~exp(score·scale)) → invisible at the 0.1-scale promotion fixtures that originally
shipped it; dV row errors reach ~35.9 (ref ~8) at std 1.0.

STRUCTURAL: the MPP cooperative matmul has no 16x16x16 op (header static_assert requires
one dim = 32) — so BK=16 cannot be expressed as the paired MMA. The II-6 disposition is a
loud `BK % 32 == 0` guard (forward + all 8 backward pipelines); a true BK=16 (TK=1) variant
is a deferred M-effort with a WITHDRAWN premise (its "1.01-1.12x vs split" was corrupt math).

This lock (a) proves the guard bites on BK=16 — the mitigation — and (b) checks the
production backward is finite + oracle-correct at UNIT scale (≈1.0), the realistic
magnitude the 0.1-scale fixture failed to exercise.
"""
from __future__ import annotations

import math
import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_mfa

def _is_m5_plus() -> bool:
    try:
        return bool(mlx_mfa.get_device_info().get("is_m5_plus", False))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _is_m5_plus(), reason="V6NAX requires M5+ NAX")


def _restore(name, prev):
    if prev is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = prev


def test_forward_bk16_guard_raises():
    """The BK%32 guard (II-6) must REJECT an invalid BK=16 tile override — without
    it, the paired 16x32x16 MMA reads past the tile (the corruption). This is the
    shipped mitigation; removing the guard re-opens the OOB."""
    mx.random.seed(0)
    q = (mx.random.normal((1, 8, 2048, 128)) * 0.1).astype(mx.float16)
    k, v = q + 0.0, q + 0.0
    mx.eval(q, k, v)
    sc = 1.0 / math.sqrt(128)
    prev = os.environ.get("MFA_V6_NAX_BK")
    os.environ["MFA_V6_NAX_BK"] = "16"
    try:
        with pytest.raises(RuntimeError, match=r"multiple of 32|16x32x16|paired"):
            o = mlx_mfa.flash_attention(q, k, v, scale=sc, causal=False)
            mx.eval(o)
    finally:
        _restore("MFA_V6_NAX_BK", prev)


def test_forward_default_bk_ok():
    """The valid default tile (BK=32) compiles + runs — the guard is not over-broad."""
    mx.random.seed(0)
    q = (mx.random.normal((1, 8, 2048, 128)) * 0.1).astype(mx.float16)
    k, v = q + 0.0, q + 0.0
    mx.eval(q, k, v)
    o = mlx_mfa.flash_attention(q, k, v, scale=1.0 / math.sqrt(128), causal=False)
    mx.eval(o)
    assert o.shape == (1, 8, 2048, 128)
    assert bool(mx.all(mx.isfinite(o)).item())


@pytest.mark.parametrize("causal", [False, True])
def test_backward_unit_scale_finite_and_correct(causal):
    """De-vacuity: validate the production backward at UNIT scale (≈1.0) — the
    magnitude that EXPOSED II-6 (the 0.1-scale fixture suppressed it). Grads must be
    finite and match an independent fp32 oracle within the fp16-backward floor.
    (Whatever path the default routes to — split / SDPA-vjp — must be correct here.)"""
    B, H, N, D = 1, 4, 2048, 64
    mx.random.seed(0)
    f = lambda: (mx.random.normal((B, H, N, D)) * 1.0).astype(mx.float16)  # std≈1.0
    q, k, v = f(), f(), f()
    mx.eval(q, k, v)
    sc = 1.0 / math.sqrt(D)

    dq, dk, dv = mx.grad(
        lambda q, k, v: mlx_mfa.flash_attention(q, k, v, scale=sc, causal=causal).sum(),
        argnums=(0, 1, 2))(q, k, v)
    mx.eval(dq, dk, dv)
    for name, g in (("dq", dq), ("dk", dk), ("dv", dv)):
        assert bool(mx.all(mx.isfinite(g)).item()), f"{name} non-finite at unit scale"

    def f32(q, k, v):
        qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
        s = (qf @ kf.swapaxes(-1, -2)) * sc
        if causal:
            s = s + mx.triu(mx.full((N, N), -1e30, dtype=mx.float32), k=1)
        return (mx.softmax(s, -1) @ vf).sum()
    rq, rk, rv = mx.grad(f32, argnums=(0, 1, 2))(q, k, v)
    mx.eval(rq, rk, rv)
    err = lambda a, b: float(np.abs(np.array(a.astype(mx.float32)) - np.array(b)).max())
    # Unit-scale fp16 backward floor (dV the loosest). Pre-II-6 fused-BK16 gave dV≈35.9 here.
    assert err(dq, rq) < 0.5, f"dq err {err(dq, rq):.3f} (II-6 corruption?)"
    assert err(dk, rk) < 0.5, f"dk err {err(dk, rk):.3f} (II-6 corruption?)"
    assert err(dv, rv) < 0.5, f"dv err {err(dv, rv):.3f} (II-6 corruption: OOB dV)"
