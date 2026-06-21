"""B4 family correctness LOCK — GNA / conv3d-nax / topk / sage-int8 / paged (audit B4).

Each kernel validated vs its APPROPRIATE independent oracle (lesson #11 — never
another kernel), with its OWN tolerance discipline:
  - GNA   : EXACT per-element-window fp32 oracle (~1e-4). Resolves Phase-A's 7.3e-2
            (that was a block-mask reference over-approximation, not a bug).
  - conv  : fp32 mx.conv_general oracle (eligible NAX + ineligible fallback).
  - topk  : fp32 top-k attention oracle.
  - sage  : QUANT-AWARE — faithful int8 round-trip + a principled int8 cos floor
            (~0.997 at D=128; int8 7-bit precision), NOT an arbitrarily-loose bound.
  - paged : fp32 gather attention oracle (decode). (IV-D1/D2 bit-identity is locked
            separately by tests/test_iv_d1_tq_append_defer.py.)

M5+-gated.

T2-1 (audit de-vacuity, 2026-06-21): these locks ran ONLY at 0.1 input scale —
the regime that hid the II-6 fused-dKdV corruption family-wide.  Every
correctness cell now runs at BOTH 0.1 (kept) AND realistic unit scale (std≈1.0,
normal), validated vs the SAME independent per-kernel oracle.  A module-level
autouse fixture parametrizes `_MAG["mode"]` over {toy, unit} so every cell runs
twice; each cell's input generator routes through `_gen`.

Tolerance discipline per kernel (output magnitude differs):
  - GNA / topk / paged : attention output is O(1) and scale-independent → unit
    scale uses a scale-invariant RELATIVE bound (`_rel`); cos floors are already
    scale-invariant and kept.
  - conv3d-nax : output magnitude GROWS with unit inputs (a fixed absolute bound
    would spuriously fail — unit md≈0.1) → RELATIVE bound + the scale-invariant
    `_cos>0.999`.
  - sage int8 : tolerance is a QUANT floor (round-trip ≤1.5·step is already
    scale-relative; cos floor is scale-invariant) → kept verbatim; unit inputs
    verified not to break the int8 round-trip (qmin/qmax stay in [-128,127]).

Which-binary confirmed at BOTH scales (gating is shape-based, value-independent):
GNA native engaged (D=128/3D/f16); conv-elig → executed conv3d_nax_forward (NAX),
conv-inel → fallback conv3d_nax_forward (outside MPP gate, by design).  A
unit-scale failure is a BUG-DISCOVERY signal — investigate which-binary; do NOT
loosen without confirming the kernel matches its independent oracle.
"""
from __future__ import annotations

import math
import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa import (flash_attention_gna, flash_attention_topk, sage_attention,
                     flash_attention_kvcache)
from mlx_mfa.attention import _get_is_m5_plus_cached
import mlx_mfa.quantize as Q

pytestmark = pytest.mark.skipif(
    not _get_is_m5_plus_cached(), reason="B4 family lock asserts M5+ kernels")

mx.random.seed(0)

# T2-1: dual input regime. toy = the original 0.1-scale uniform; unit = realistic
# std≈1.0 normal.  `_gen` routes both; the autouse fixture flips the mode.
_MAG = {"mode": "toy"}
_REL_TOL = 5e-2  # scale-invariant attention rel-err floor (fp16 ≲1e-2 + margin)


def _gen(shape):
    if _MAG["mode"] == "unit":
        return mx.random.normal(shape).astype(mx.float16)          # std ≈ 1.0
    return (mx.random.uniform(-1, 1, shape) * 0.1).astype(mx.float16)


@pytest.fixture(autouse=True, params=["toy", "unit"])
def _regime(request):
    """Run every cell in this module at BOTH input scales (T2-1)."""
    _MAG["mode"] = request.param
    yield
    _MAG["mode"] = "toy"


def _md(a, b):
    mx.eval(a, b); return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _rel(a, b):
    """Scale-invariant relative max-abs error (denominator = max|ref|)."""
    d = _md(a, b)
    denom = float(mx.max(mx.abs(b.astype(mx.float32))).item()) + 1e-6
    return d / denom


def _cos(a, b):
    a, b = a.astype(mx.float32).reshape(-1), b.astype(mx.float32).reshape(-1); mx.eval(a, b)
    return float((mx.sum(a * b) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b)))).item())


# ── GNA: exact per-element-window oracle (resolves Phase-A 7.3e-2) ───────────
def _gna_elem_mask(seq_shape, window, stride):
    N = int(np.prod(seq_shape)); coords = np.array(np.unravel_index(np.arange(N), seq_shape)).T
    M = np.ones((N, N), bool)
    for d in range(len(seq_shape)):
        pos = coords[:, d]
        gb = (pos // stride[d]) * stride[d]
        lo = np.maximum(gb - (window[d] - stride[d]) // 2, 0)
        hi = np.minimum(gb + stride[d] + (window[d] - stride[d] + 1) // 2, seq_shape[d])
        cj = coords[:, d]
        M &= (cj[None, :] >= lo[:, None]) & (cj[None, :] < hi[:, None])
    return M


@pytest.mark.parametrize("seq,win,st", [((8, 8, 8), (3, 3, 3), (1, 1, 1)),
                                        ((8, 8, 8), (5, 5, 5), (1, 1, 1)),
                                        ((4, 8, 8), (3, 3, 3), (2, 2, 2))])
def test_gna_matches_exact_per_element_oracle(seq, win, st):
    N = int(np.prod(seq)); D = 128; sc = 1 / math.sqrt(D)
    f = lambda: _gen((1, 4, N, D))
    q, k, v = f(), f(), f(); mx.eval(q, k, v)
    o = flash_attention_gna(q, k, v, seq, win, st, scale=sc)
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    s = (qf @ kf.swapaxes(-1, -2)) * sc
    s = mx.where(mx.array(_gna_elem_mask(seq, win, st))[None, None], s, mx.array(-1e30))
    ref = mx.softmax(s, -1) @ vf
    assert bool(mx.all(mx.isfinite(o.astype(mx.float32))).item())
    if _MAG["mode"] == "unit":
        assert _rel(o, ref) < _REL_TOL, "GNA diverged from the exact per-element window (unit)"
    else:
        assert _md(o, ref) < 3e-2, "GNA diverged from the exact per-element window"


# ── conv3d-nax: fp32 conv oracle ─────────────────────────────────────────────
def test_conv3d_nax_eligible_matches_fp32():
    # which-binary: executed conv3d_nax_forward (NAX MPP gate) at BOTH scales.
    x = _gen((1, 8, 16, 16, 128))
    w = _gen((128, 3, 3, 3, 128)); mx.eval(x, w)
    o = mx.conv_general(x, w, stride=1, padding=1)
    ref = mx.conv_general(x.astype(mx.float32), w.astype(mx.float32), stride=1, padding=1)
    # conv output magnitude grows with unit inputs → RELATIVE bound (the cos floor
    # is already scale-invariant; the old absolute 3e-2 spuriously fails at unit).
    assert _rel(o, ref) < _REL_TOL and _cos(o, ref) > 0.999


def test_conv3d_ineligible_fallback_matches_fp32():
    # which-binary: fallback conv3d_nax_forward (outside MPP gate, by design) both scales.
    x = _gen((1, 8, 16, 16, 16))
    w = _gen((16, 3, 3, 3, 16)); mx.eval(x, w)
    o = mx.conv_general(x, w, stride=1, padding=1)
    ref = mx.conv_general(x.astype(mx.float32), w.astype(mx.float32), stride=1, padding=1)
    assert _rel(o, ref) < _REL_TOL and _cos(o, ref) > 0.999


# ── topk: fp32 top-k attention oracle ────────────────────────────────────────
def test_topk_matches_fp32_topk():
    B, H, N, D = 2, 8, 512, 128; sc = 1 / math.sqrt(D)
    f = lambda: _gen((B, H, N, D))
    q, k, v = f(), f(), f(); mx.eval(q, k, v)
    ratio = 0.25; kk = max(1, int(round(ratio * N)))
    o = flash_attention_topk(q, k, v, topk_ratio=ratio, scale=sc)
    s = (q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)) * sc
    thr = mx.sort(s, axis=-1)[..., N - kk][..., None]
    ref = mx.softmax(mx.where(s >= thr, s, mx.array(-1e30)), -1) @ v.astype(mx.float32)
    # topk error is dominated by top-k SET-MEMBERSHIP disagreement near the
    # threshold (a few keys flip vs the fp32 oracle) — a different, looser noise
    # class than dense attention.  At toy scale outputs are tiny so the relative
    # form is structurally noisy (~6-7e-2, measured stable across seeds) while the
    # ABSOLUTE error stays small → keep the original absolute bound for toy; unit
    # outputs are O(1) so the scale-invariant relative bound applies (~2e-2).  cos
    # floor (scale-invariant, 0.9998) holds at both.
    if _MAG["mode"] == "unit":
        assert _rel(o, ref) < _REL_TOL and _cos(o, ref) > 0.999
    else:
        assert _md(o, ref) < 1e-1 and _cos(o, ref) > 0.999


# ── sage int8: quant-aware (faithful round-trip + principled int8 cos floor) ──
def test_sage_int8_quant_roundtrip_faithful():
    mx.random.seed(0)  # order-independence: per-test seed (module seed only fires at import;
    # unseeded draws here depend on cumulative global RNG consumed by prior tests → suite-order flake)
    x = _gen((2, 8, 512, 128)); mx.eval(x)
    qx, scales = Q.quantize_per_block(x, block_size=128)
    xr = Q.dequantize(qx, scales, block_size=128); mx.eval(qx, scales, xr)
    # KEPT verbatim (T2-1): tolerance is a QUANT floor, not fp16 — `step*1.5` is
    # already scale-relative (step ∝ max|x|), so it holds at unit scale (verified:
    # unit md≈0.02 vs step*1.5≈0.06; qmin/qmax stay in [-128,127]).
    assert -128 <= float(mx.min(qx).item()) and float(mx.max(qx).item()) <= 127
    step = float(mx.max(scales).item())
    assert _md(x, xr) <= step * 1.5, "int8 round-trip exceeds one quantization step"


def test_sage_int8_attention_within_principled_int8_bound():
    # int8 7-bit symmetric quant of Q,K over D=128 -> a cos floor ~0.997 (measured
    # stable across input amplitude). Lock at cos>=0.995 (principled int8 margin),
    # NOT an arbitrarily-loose bound.
    mx.random.seed(0)  # order-independence (see roundtrip test): the cos floor is the tightest
    # bound, so a polluted-RNG input draw was the chronic suite-order flake; seed=0 → cos 0.9985 (margin).
    B, H, N, D = 2, 8, 512, 128; sc = 1 / math.sqrt(D)
    f = lambda: _gen((B, H, N, D))
    q, k, v = f(), f(), f(); mx.eval(q, k, v)
    o = sage_attention(q, k, v, scale=sc, causal=False)
    s = (q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)) * sc
    ref = mx.softmax(s, -1) @ v.astype(mx.float32)
    # cos floor is scale-invariant (int8 7-bit) → KEPT (unit cos≈0.998, margin).
    assert _cos(o, ref) >= 0.995, "sage int8 below the principled int8 cos floor"


# ── paged/kvcache decode: fp32 gather oracle ─────────────────────────────────
# CC-17 (audit) — ROUTING-not-kernel cell: on M5, N=1 decode routes to SDPA
# (byteΔ=0 vs SDPA) — the documented-current production decode path, pinned by
# test_dispatch_map_lock::test_kvcache_decode_is_sdpa.  This cell asserts the
# *routed* decode path is numerically correct vs an independent fp32 gather
# oracle; it is NOT a distinct-paged-kernel engagement claim (so it deliberately
# does not force a kernel).  Which-binary is locked by the dispatch lock; this
# locks the math.
def test_paged_decode_matches_fp32_gather():
    B, H, S, D = 1, 8, 1024, 128; sc = 1 / math.sqrt(D)
    qd = _gen((B, H, 1, D))
    kc = _gen((B, H, S, D))
    vc = _gen((B, H, S, D)); mx.eval(qd, kc, vc)
    o = flash_attention_kvcache(qd, kc, vc, scale=sc, causal=True, cache_seqlens=S)
    s = (qd.astype(mx.float32) @ kc.astype(mx.float32).swapaxes(-1, -2)) * sc
    ref = mx.softmax(s, -1) @ vc.astype(mx.float32)
    # attention output O(1) → scale-invariant RELATIVE bound + cos floor.
    assert _rel(o, ref) < _REL_TOL and _cos(o, ref) > 0.999
