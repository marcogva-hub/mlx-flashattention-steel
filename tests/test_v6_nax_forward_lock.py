"""Standalone forward correctness LOCK for the dense NAX matmul2d forward
`v6_nax_forward` (audit E-addendum / B-gap, 2026-06-18).

`v6_nax_forward` (csrc/mfa_steel_fwd_v6_nax.cpp — 10 matmul2d/cooperative-tensor
hits, the ONLY dense forward on the competitive NAX primitive) was previously
validated only INDIRECTLY, via B3's backward gradients consuming its (O, L).
This module gives it a STANDALONE forward check vs an INDEPENDENT manual fp32
oracle (strict lesson #11 — NOT SDPA, NOT another mlx-mfa kernel), closing the
B-gap (the dense NAX forward had no standalone forward spec/lock).

Constraints (archaeology, current master):
  - Binding `v6_nax_forward(q, k, v, causal, force_v6nax=False)` has NO scale
    parameter — the host bakes scale = 1/sqrt(D) (attention.py ~5018). So this
    kernel is DEFAULT-SCALE ONLY; the oracle uses 1/sqrt(D) accordingly.
  - Forward-eligible shapes (binding DC12): D=128 any N; D=64 only Nk > 8000.
  - M5+ NAX hardware only (matmul2d cooperative tensors).
"""
from __future__ import annotations
import math

import numpy as np
import mlx.core as mx
import pytest

try:
    from mlx_mfa._ext import v6_nax_forward
    _HAVE = True
except Exception:
    _HAVE = False

try:
    from mlx_mfa import get_device_info
    _IS_M5 = bool(get_device_info().get("is_m5_plus"))
except Exception:
    _IS_M5 = False

pytestmark = pytest.mark.skipif(
    not (_HAVE and _IS_M5),
    reason="v6_nax_forward dense NAX kernel: M5+ NAX hardware + extension required",
)

# T2-1 de-vacuity (audit, 2026-06-21): these correctness cells ran ONLY at the
# 0.1 input scale — a regime where fp16 softmax is near-uniform and small kernel
# divergences are suppressed (the lock comment records a prior simdgroup fallback
# giving D=64 N=4096 err≈512 vs fp32, a divergence only realistic-scale inputs
# expose).  Every correctness cell now ALSO runs at realistic unit scale
# (std≈1.0, normal), validated vs the SAME independent fp32 oracle.  Toy keeps
# the original ABSOLUTE bound (< 2e-2); unit uses a scale-invariant RELATIVE
# bound.  A unit-scale failure is a BUG-DISCOVERY signal — confirm which-binary
# (byteΔ vs SDPA proves the NAX kernel ran, not a silent SDPA fallback) and do
# NOT loosen without confirming the kernel is correct.
#   Empirically (this hardware, MLX 0.31.2, M5 Max): all cells give unit-scale
#   rel-err ≲ 7e-4, byteΔ-vs-SDPA ≳ 1e-4 (NAX path confirmed engaged).  The cells
#   all have N==S, so the causal (S-N) term is dormant (S-N=0) — no flipped-sign
#   latent bug is exercisable here.
_REL_TOL = 5e-2
_MAG = {"mode": "toy"}


@pytest.fixture(autouse=True, params=["toy", "unit"])
def _regime(request):
    """Run every correctness cell at BOTH 0.1 (toy) and std≈1.0 (unit) input
    scale (T2-1).  Scale-independent cells (the pure-NAX Δ==0 equivalence) pass
    identically at both — their bite is preserved."""
    _MAG["mode"] = request.param
    yield
    _MAG["mode"] = "toy"


def _gen(shape):
    """Inputs at the active magnitude.  unit → std≈1.0 normal (realistic, the
    regime that materially peaks softmax); else → the original 0.1-uniform toy."""
    if _MAG["mode"] == "unit":
        return mx.random.normal(shape).astype(mx.float16)
    return (mx.random.uniform(-1, 1, shape) * 0.1).astype(mx.float16)


def _check(O, ref, label):
    """Toy: original ABS bound (< 2e-2).  Unit: scale-invariant RELATIVE bound."""
    err = float(np.abs(np.array(O.astype(mx.float32)) - np.array(ref)).max())
    assert np.isfinite(err), f"{label}: non-finite err"
    if _MAG["mode"] == "unit":
        rel = err / (float(np.abs(np.array(ref)).max()) + 1e-6)
        assert rel < _REL_TOL, f"{label}: unit-scale rel_err={rel:.3e} exceeds {_REL_TOL} (abs={err:.3e})"
    else:
        assert err < 2e-2, f"{label}: toy-scale max_err={err:.3e} exceeds 2e-2"


def _fp32_oracle(q, k, v, causal, D):
    """Independent manual fp32 FA-2 forward (lesson #11), default scale 1/sqrt(D)."""
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    Hq, Hk = q.shape[1], k.shape[1]
    if Hq != Hk:  # GQA: repeat KV heads
        kf = mx.repeat(kf, Hq // Hk, axis=1)
        vf = mx.repeat(vf, Hq // Hk, axis=1)
    s = (qf @ kf.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(D))
    if causal:
        N, S = q.shape[2], k.shape[2]
        cm = mx.arange(S)[None, :] > (mx.arange(N)[:, None] + (S - N))
        s = mx.where(cm[None, None], mx.array(-1e30, mx.float32), s)
    o = mx.softmax(s, axis=-1) @ vf
    mx.eval(o)
    return o


# (D, N, Hq, Hk, causal) — forward-eligible shapes only (D=64 needs Nk>8000).
_CELLS = [
    (128, 2048, 8, 8, False), (128, 2048, 8, 8, True),
    (128, 4096, 8, 8, False), (128, 4096, 8, 8, True),
    (64, 8192, 8, 8, False), (64, 8192, 8, 8, True),
    (128, 2048, 8, 2, False), (128, 2048, 8, 2, True),  # GQA
]


@pytest.mark.parametrize("D,N,Hq,Hk,causal", _CELLS)
def test_v6_nax_forward_matches_fp32(D, N, Hq, Hk, causal):
    mx.random.seed(13)
    q = _gen((1, Hq, N, D))
    k = _gen((1, Hk, N, D))
    v = _gen((1, Hk, N, D))
    mx.eval(q, k, v)
    O, L = v6_nax_forward(q, k, v, causal, True)  # force_v6nax=True
    mx.eval(O)
    ref = _fp32_oracle(q, k, v, causal, D)
    _check(O, ref, f"v6_nax_forward D={D} N={N} Hq={Hq} Hk={Hk} causal={causal} ({_MAG['mode']})")


@pytest.mark.parametrize("D,N", [(64, 2048), (64, 4096), (128, 2048)])
@pytest.mark.parametrize("causal", [False, True])
def test_v6_forward_is_pure_nax(D, N, causal):
    """Audit F-3: V6 is PURE NAX — the simdgroup-within-V6 fallback (a diverged,
    D=64-BROKEN duplicate: D=64 N=4096 gave max-abs-err ≈ 512 vs fp32) is removed.

    `force_v6nax=False` (the value that USED to select the broken simdgroup at
    D=64-small-N) now produces the SAME output as `force_v6nax=True` (both NAX) and
    matches the fp32 oracle.  Drift-back: if a simdgroup fallback reappears, the
    False path diverges from the True path (Δ>0) and/or from fp32 → this FAILS."""
    mx.random.seed(4)
    q = _gen((1, 8, N, D))
    k = _gen((1, 8, N, D))
    v = _gen((1, 8, N, D))
    mx.eval(q, k, v)
    O_false, _ = v6_nax_forward(q, k, v, causal, False)  # was simdgroup at D=64
    O_true, _ = v6_nax_forward(q, k, v, causal, True)     # always NAX
    mx.eval(O_false, O_true)
    # Δ==0 equivalence is scale-INDEPENDENT — it must hold at both toy and unit.
    drift = float(mx.max(mx.abs(O_false.astype(mx.float32) - O_true.astype(mx.float32))).item())
    assert drift == 0.0, (
        f"V6 not pure NAX (D={D} N={N} c={causal} {_MAG['mode']}): force_v6nax=False "
        f"diverged from True by Δ={drift:.3e} — the simdgroup-within-V6 fallback reappeared")
    ref = _fp32_oracle(q, k, v, causal, D)
    _check(O_false, ref, f"V6 pure-NAX path (D={D} N={N} c={causal} {_MAG['mode']})")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("scale", [0.05, 0.30])
def test_v6_nax_forward_respects_custom_scale(causal, scale):
    """Audit F-2 (Change 3): the binding now accepts a custom scale and the kernel
    FULLY respects it (scale plumbed through descriptor.scale + baked #define +
    cache-keyed).  Output must match the fp32 oracle AT THE CUSTOM SCALE — and the
    custom scale must be distinguishable from the default (larger inputs so QK is
    O(1+); a stale default-scale pipeline would diverge from the custom oracle)."""
    D, N = 128, 2048
    mx.random.seed(11)
    # larger inputs so the scale materially changes softmax (distinguishability)
    q = mx.random.uniform(-1, 1, (1, 8, N, D)).astype(mx.float16)
    k = mx.random.uniform(-1, 1, (1, 8, N, D)).astype(mx.float16)
    v = mx.random.uniform(-1, 1, (1, 8, N, D)).astype(mx.float16)
    mx.eval(q, k, v)
    O, _ = v6_nax_forward(q, k, v, causal, True, scale)
    mx.eval(O)
    ref = _fp32_oracle_scaled(q, k, v, causal, scale)
    err = float(np.abs(np.array(O.astype(mx.float32)) - np.array(ref)).max())
    assert np.isfinite(err) and err < 2e-2, (
        f"v6_nax_forward custom scale={scale} causal={causal}: err={err:.3e} vs fp32")
    # distinguishability guard: the custom-scale output must NOT match the default
    # oracle (else the cache returned a stale default-scale pipeline).
    default = 1.0 / math.sqrt(D)
    if abs(scale - default) > 0.05:
        ref_def = _fp32_oracle_scaled(q, k, v, causal, default)
        sep = float(np.abs(np.array(O.astype(mx.float32)) - np.array(ref_def)).max())
        assert sep > 0.01, (
            f"custom scale {scale} produced the DEFAULT-scale output (Δ={sep:.4f}) "
            f"— cache-key collision: a distinct scale reused the wrong baked pipeline")


def _fp32_oracle_scaled(q, k, v, causal, scale):
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    Hq, Hk = q.shape[1], k.shape[1]
    if Hq != Hk:
        kf = mx.repeat(kf, Hq // Hk, axis=1)
        vf = mx.repeat(vf, Hq // Hk, axis=1)
    s = (qf @ kf.transpose(0, 1, 3, 2)) * scale
    if causal:
        N, S = q.shape[2], k.shape[2]
        cm = mx.arange(S)[None, :] > (mx.arange(N)[:, None] + (S - N))
        s = mx.where(cm[None, None], mx.array(-1e30, mx.float32), s)
    o = mx.softmax(s, axis=-1) @ vf
    mx.eval(o)
    return o


def test_v6_nax_forward_default_scale_sentinel():
    """F-2 (Change 3): the <=0 scale sentinel (or the no-scale call) resolves to the
    default 1/sqrt(D).  (Pre-F-2 this kernel was default-scale-ONLY; F-2 plumbed a
    real scale arg — see test_v6_nax_forward_respects_custom_scale — but the default
    behavior is preserved for existing callers, e.g. the backward-recompute.)"""
    D, N = 128, 2048
    mx.random.seed(1)
    q = _gen((1, 4, N, D))
    k = _gen((1, 4, N, D))
    v = _gen((1, 4, N, D))
    mx.eval(q, k, v)
    # no scale arg → default sentinel → 1/sqrt(D)
    O, _ = v6_nax_forward(q, k, v, False, True)
    mx.eval(O)
    ref_default = _fp32_oracle(q, k, v, False, D)  # scale = 1/sqrt(128)
    _check(O, ref_default, f"v6_nax_forward default sentinel ({_MAG['mode']})")
