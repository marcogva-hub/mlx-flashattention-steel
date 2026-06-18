"""conv3d-NAX per-axis (causal) pad lock (feature/conv3d-nax-asym-pad-m5, M5 Max, 2026-06-18).

Tier-1 #3 found the NAX conv kernel beats mx.conv 1.3-2.7× on VAE-channel 3×3×3 convs (compute-
bound) but 0% of production VAE convs reached it: every VAE conv3d is CAUSAL
(InflatedCausalConv3d / CogVideoXCausalConv3d) → after the VAE's upstream manual time-pad it calls
mx.conv_general with pad=(0,1,1), while the MPP hook required symmetric pad=(1,1,1).

This feature generalizes the MPP path to per-axis "same"-style pad (temporal {0,1}, spatial 1) —
causality is handled upstream, so the kernel only needed to accept pad_T=0 (the kt time-loop's
`-pT_left` offset + T_out=T-2). The symmetric (1,1,1) path is UNCHANGED (keep-all-paths).

Locks: (1) causal (0,1,1) routes to NAX (not the silent fallback); (2) NAX causal output matches an
independent fp32 conv reference (the tight bar — conv output feeds the decoder, drift compounds,
Lesson #11); (3) symmetric (1,1,1) still routes NAX + correct (regression); (4) unsupported pads
fall back (Rule 8, no mis-gather).
"""
from __future__ import annotations
import os
import numpy as np
import mlx.core as mx
import pytest

import mlx_mfa
from mlx_mfa import get_hook_stats
from mlx_mfa.attention import _get_is_m5_plus_cached

pytestmark = pytest.mark.skipif(
    not _get_is_m5_plus_cached(),
    reason="conv3d-NAX asym-pad lock asserts M5+ MPP conv")

_ec = lambda s: s["executed"].get("conv3d_nax_forward", 0)
_fc = lambda s: s["fallback"].get("conv3d_nax_forward", 0)


def _delta(a, b):
    mx.eval(a, b)
    return float(np.abs(np.array(a.astype(mx.float32)) - np.array(b.astype(mx.float32))).max())


def _xw(C, T, H, W, seed=0):
    mx.random.seed(seed)
    x = (mx.random.uniform(-1, 1, (1, T, H, W, C)) * 0.3).astype(mx.float16)
    w = (mx.random.uniform(-1, 1, (C, 3, 3, 3, C)) * 0.1).astype(mx.float16)
    mx.eval(x, w)
    return x, w


def _route_and_out(x, w, pad):
    be, bf = _ec(get_hook_stats()), _fc(get_hook_stats())
    o = mx.conv_general(x, w, stride=1, padding=pad)
    mx.eval(o)
    de = _ec(get_hook_stats()) - be
    return ("NAX" if de > 0 else "fallback"), o


@pytest.mark.parametrize("C,T,H,W", [(512, 5, 32, 32), (256, 5, 64, 64), (128, 5, 64, 64)])
def test_causal_pad_routes_nax_and_correct(C, T, H, W):
    """Causal (0,1,1) 3×3×3 routes to the NAX MPP conv (not the silent fallback) AND matches an
    independent fp32 conv reference within the fp16 floor (Lesson #11 — conv output, tight bar)."""
    x, w = _xw(C, T, H, W)
    route, o = _route_and_out(x, w, (0, 1, 1))
    assert route == "NAX", f"causal (0,1,1) C={C} did NOT route to NAX (got {route}) — gate regressed"
    assert o.shape[1] == T - 2, f"causal T_out wrong: {o.shape[1]} != {T - 2}"
    ref = mx.conv_general(x.astype(mx.float32), w.astype(mx.float32), stride=1, padding=(0, 1, 1))
    err = _delta(o, ref)
    assert err < 1e-2, f"causal NAX conv wrong vs fp32 ref (Δ={err:.2e}) — drift would compound in the decoder"
    assert bool(np.isfinite(np.array(o.astype(mx.float32))).all())


def test_symmetric_pad_unchanged():
    """Keep-all-paths: symmetric (1,1,1) still routes NAX, T_out==T, correct vs fp32 (the existing
    path's emitted source is byte-identical — pT_left defaults to 1)."""
    C, T, H, W = 256, 6, 32, 32
    x, w = _xw(C, T, H, W)
    route, o = _route_and_out(x, w, (1, 1, 1))
    assert route == "NAX", "symmetric (1,1,1) no longer routes NAX — regression"
    assert o.shape[1] == T, f"symmetric T_out drifted: {o.shape[1]} != {T}"
    ref = mx.conv_general(x.astype(mx.float32), w.astype(mx.float32), stride=1, padding=(1, 1, 1))
    assert _delta(o, ref) < 1e-2


@pytest.mark.parametrize("pad", [(2, 1, 1), (0, 2, 2), (1, 1, 2)])
def test_unsupported_pad_falls_back(pad):
    """Rule 8: pads outside the supported set (temporal pad 2, spatial H/W != 1) must NOT route to
    NAX — they fall back to mx.conv (no silent mis-gather through the MPP kernel)."""
    C, T, H, W = 256, 6, 32, 32
    x, w = _xw(C, T, H, W)
    route, _ = _route_and_out(x, w, pad)
    assert route == "fallback", f"unsupported pad {pad} routed to NAX — Rule 8 mis-gather risk"


# ── Adversarial anti-corruption matrix (hardening pass) ──────────────────────
# The invariant: a NAX-routed conv output MUST equal the fp32 reference within the
# conv floor; anything NAX can't do correctly falls back / raises — NEVER silently
# routes NAX with a wrong output (the corruption this sweep exists to catch). The
# fp32 check runs REGARDLESS of routing, so a mis-gather is caught even if the gate
# wrongly admitted the pad.

def _conv_and_fp32(C, k, pad, stride=1, dilation=1, groups=1):
    mx.random.seed(0)
    T, H, W = 6, 32, 32
    x = (mx.random.uniform(-1, 1, (1, T, H, W, C)) * 0.3).astype(mx.float16)
    w = (mx.random.uniform(-1, 1, (C, *k, C // groups)) * 0.1).astype(mx.float16)
    mx.eval(x, w)
    be = _ec(get_hook_stats())
    try:
        o = mx.conv_general(x, w, stride=stride, padding=pad,
                            kernel_dilation=dilation, groups=groups)
        mx.eval(o)
    except Exception:
        return "raise", None  # MLX/Rule-8 raise — not a silent NAX mis-gather
    route = "NAX" if _ec(get_hook_stats()) - be > 0 else "fallback"
    ref = mx.conv_general(x.astype(mx.float32), w.astype(mx.float32), stride=stride,
                          padding=pad, kernel_dilation=dilation, groups=groups)
    return route, _delta(o, ref)


@pytest.mark.parametrize("pad", [
    (1, 1, 1), (0, 1, 1),                                   # supported (must route NAX)
    (2, 1, 1), (0, 2, 1), (1, 0, 1), (0, 1, 0), (1, 1, 0), (0, 0, 1), (2, 2, 2),  # unsupported
])
def test_adversarial_pad_no_corruption(pad):
    """Every pad: a NAX route must be fp32-correct; unsupported pads must fall back. No NAX-with-
    wrong-output (corruption) under any pad."""
    route, err = _conv_and_fp32(256, (3, 3, 3), pad)
    if route == "NAX":
        assert err is not None and err < 1e-2, f"CORRUPTION: pad {pad} routed NAX with err={err}"
        assert pad in ((1, 1, 1), (0, 1, 1)), f"pad {pad} unexpectedly routed NAX"
    else:
        assert pad not in ((1, 1, 1), (0, 1, 1)), f"supported pad {pad} did NOT route NAX"


@pytest.mark.parametrize("label,k,kw", [
    ("dilation2", (3, 3, 3), {"dilation": 2}),
    ("stride2", (3, 3, 3), {"stride": 2}),
    ("k(1,3,3)", (1, 3, 3), {}),
    ("k(3,1,1)", (3, 1, 1), {}),
    ("k(5,3,3)", (5, 3, 3), {}),
])
def test_adversarial_config_no_silent_nax(label, k, kw):
    """Adversarial configs (non-1 dilation/stride, non-3×3×3 kernels) must NOT silently take the
    3×3×3 NAX path with a wrong result — fall back / raise, or NAX only if fp32-correct."""
    pad = (1, 1, 1) if k == (3, 3, 3) else (0, (k[1] - 1) // 2, (k[2] - 1) // 2)
    route, err = _conv_and_fp32(256, k, pad, **kw)
    if route == "NAX":
        assert err is not None and err < 1e-2, f"CORRUPTION: {label} routed NAX with err={err}"
