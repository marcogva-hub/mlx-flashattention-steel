"""Runtime dispatch-map regression LOCK (audit Phase A, 2026-06-17).

The authoritative, machine-readable map of WHICH KERNEL ACTUALLY RUNS for each
public entry × decision-boundary input class on M5/26.6 — established by RUNTIME
FINGERPRINT (byte-identity vs a known reference; density signature; hook
telemetry), never by source-tracing (the lesson of four which-binary inversions).

Fingerprint semantics:
  - byteΔ vs the SDPA reference == 0.0  => the path IS that reference kernel
    (the SDPA fallback is literally `mx.fast.sdpa` → bit-identical).
  - byteΔ ~1e-6 (small, nonzero)        => a DIFFERENT real kernel (same math,
    different rounding) — e.g. STEEL, the NAX sparse kernel.
  - hook telemetry executed[X] > 0      => the hooked kernel X ran (conv).

This test asserts the CURRENT verified reality. Audit Phase F (2026-06-18) FIXED
two of the original gotchas — D=128 built-in-maker sparse now routes to NAX
(symmetric 32×32), and D=64 sparse routes to V2 (not the slow V1). The residual
SDPA routes (asymmetric/custom, small, dense-via-gate) + sparse-backward-dense
are locked as "expected, documented" so UNINTENTIONAL drift fails CI. Do NOT
weaken to make a reroute pass silently — update the map entry with a comment when
intended.

Gated to M5+ (the guards under test are M5-specific). On non-M5 the routes differ;
the test skips rather than asserting M5 reality elsewhere.
"""
from __future__ import annotations

import math
import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa import (
    flash_attention, flash_attention_sparse, flash_attention_gna,
    flash_attention_kvcache, make_gna_mask, get_hook_stats,
)
from mlx_mfa.attention import _get_is_m5_plus_cached, _steel_block_config

pytestmark = pytest.mark.skipif(
    not _get_is_m5_plus_cached(),
    reason="dispatch-map lock asserts M5+ runtime routes; non-M5 routes differ",
)

mx.random.seed(0)

# T2-1 (audit de-vacuity, 2026-06-21): this file is PRIMARILY a which-binary lock
# — almost every cell asserts a byteΔ FINGERPRINT vs an SDPA reference (==0.0 ⇒
# IS the SDPA fallback; >0 ⇒ a different REAL kernel). Those fingerprint
# assertions are SCALE-INDEPENDENT (a path that emits identical bytes at 0.1 emits
# identical bytes at std≈1.0 — same kernel, same input identity) and stay EXACT.
# The audit runs every cell at BOTH the original 0.1 toy scale (kept) AND a
# realistic unit scale (std≈1.0, normal) — the regime that hid the II-6 fused-dKdV
# corruption — so the routing lock is proven robust at the production input
# magnitude. The few cells that ALSO carry a loose `< 3e-2` upper sanity cap on a
# real-kernel-vs-SDPA byteΔ keep that cap EXACT: it is a routing sanity bound
# (wrong-kernel ⇒ huge Δ), not an oracle-math tolerance — verified to hold at unit
# scale (matmul cells ≲3e-4). A flip at unit scale (a ==0.0 becoming >0, or a >0
# becoming ==0.0) would be a SCALE-DEPENDENT ROUTE = a real audit finding, NOT a
# tolerance to relax. (This file has no independent-oracle math cells to convert
# to relative tolerance — its locks are byte-fingerprint by construction.)
_MAG = {"mode": "toy"}


def _gen(shape):
    if _MAG["mode"] == "unit":
        return mx.random.normal(shape).astype(mx.float16)          # std ≈ 1.0
    return (mx.random.uniform(-1, 1, shape) * 0.1).astype(mx.float16)


@pytest.fixture(autouse=True, params=["toy", "unit"])
def _regime(request):
    """Run EVERY cell at both input magnitudes (T2-1). byteΔ fingerprints are
    scale-independent → the routing lock is strengthened, not duplicated-noise."""
    _MAG["mode"] = request.param
    yield
    _MAG["mode"] = "toy"


def _qkv(B, H, N, D, Hk=None):
    Hk = Hk or H
    f = lambda h: _gen((B, h, N, D))
    q, k, v = f(H), f(Hk), f(Hk)
    mx.eval(q, k, v)
    return q, k, v


def _delta(a, b):
    mx.eval(a, b)
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _block_bias(bm, qL, kL):
    NQ, NK = bm.shape[-2], bm.shape[-1]
    em = mx.repeat(mx.repeat(bm.astype(mx.float32), qL // NQ, axis=-2), kL // NK, axis=-1)
    b = mx.where(em > 0, mx.array(0.0), mx.array(-1e9))
    while b.ndim < 4:
        b = b[None]
    return b.astype(mx.float16)


# ── dense ──────────────────────────────────────────────────────────────────
def test_dense_auto_D128_is_nax_not_sdpa():
    """Audit F-2 (Change 3): flash_attention(backend=auto) dense D=128 → the NAX
    matmul2d forward (`v6_nax_forward`), NOT Apple SDPA.  Δ vs SDPA must be a small
    real-kernel difference (~1e-6), NOT 0.0.  Drift-back to SDPA (Δ==0) fails CI —
    the parity-to-modest-win D=128 dense route would have silently regressed."""
    B, H, N, D = 1, 4, 2048, 128
    q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
    o = flash_attention(q, k, v, scale=sc, causal=False)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)
    d = _delta(o, ref)
    assert 1e-7 < d < 3e-2, (
        f"D=128 dense auto is not the NAX kernel (Δ={d}: 0.0 ⇒ drifted back to "
        f"SDPA — the F-2 dense-NAX route regressed; large ⇒ wrong kernel)")


def test_dense_auto_D64_is_sdpa():
    """F-2: dense D=64 auto stays Apple SDPA (byte-identical) — NAX LOSES at D=64
    (1.17-1.22×), so it must NOT route there.  Δ==0 ⇒ SDPA (correct)."""
    B, H, N, D = 1, 4, 4096, 64
    q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
    o = flash_attention(q, k, v, scale=sc, causal=False)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)
    assert _delta(o, ref) == 0.0, "D=64 dense auto drifted OFF SDPA (NAX loses at D=64 — must stay SDPA)"


def test_dense_auto_D64_causal_largeN_is_sdpa():
    """M2 (audit 2026-06-21): D=64 CAUSAL large-N (B·H>=4, N>=4096) is SDPA on the
    M5/NAX tier (byteΔ==0), NOT a real MFA primitive.  should_use_mfa(D=64,causal,
    has_nax=True)→False; the Python trace records 'mfa_primitive' (the V6 carveout
    flips use_mfa) but the C++ M5 guard routes to SDPA.  The dispatch-map cell was
    relabeled to match; this locks the runtime.  (The 'real primitive byteΔ>0'
    only exists on the M3/M4 tier where has_nax=False — unreachable on this host.)"""
    B, H, N, D = 2, 4, 4096, 64
    q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
    o = flash_attention(q, k, v, scale=sc, causal=True)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask="causal")
    assert _delta(o, ref) == 0.0, (
        "D=64 causal large-N drifted OFF SDPA — if this is the M3/M4 tier that's "
        "expected (real primitive); on M5/NAX it must stay SDPA. Update the map.")


def test_dense_D128_backend_sdpa_and_optout_stay_sdpa():
    """F-2: backend='sdpa' and MFA_DISABLE_V6_DENSE=1 keep D=128 dense on SDPA
    (keep-all-paths; the NAX route is auto-only + opt-outable)."""
    import os
    B, H, N, D = 1, 4, 2048, 128
    q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)
    assert _delta(flash_attention(q, k, v, scale=sc, backend="sdpa"), ref) == 0.0, \
        "backend='sdpa' D=128 must stay SDPA"
    os.environ["MFA_DISABLE_V6_DENSE"] = "1"
    try:
        assert _delta(flash_attention(q, k, v, scale=sc), ref) == 0.0, \
            "MFA_DISABLE_V6_DENSE=1 must force D=128 dense back to SDPA"
    finally:
        os.environ.pop("MFA_DISABLE_V6_DENSE", None)


def test_dense_mfa_is_real_steel_not_sdpa():
    """flash_attention(backend=mfa) dense → real STEEL kernel (NOT a silent SDPA fallback)."""
    B, H, N, D = 1, 4, 2048, 128
    q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
    o = flash_attention(q, k, v, scale=sc, causal=False, backend="mfa")
    sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)
    d = _delta(o, sdpa)
    assert d > 1e-7, f"backend=mfa is byte-identical to SDPA (Δ={d}) — STEEL path lost"
    assert d < 3e-2, f"backend=mfa output wrong vs SDPA (Δ={d})"


# ── sparse forward (the cartography + increment-0 findings, locked) ──────────
def test_sparse_D128_asymmetric_is_silent_sdpa_fallback():
    """D=128 + ASYMMETRIC / custom mask (bt_q != bt_k) → dense SDPA fallback.
    Phase F (2026-06-18) FIXED the built-in makers (now symmetric 32×32 → NAX,
    see test_sparse_D128_symmetric_is_real_nax_sparse); only asymmetric/custom
    masks remain on the SDPA fallback. This cell builds an asymmetric 32×16 mask
    explicitly to lock that residual route."""
    B, H, N, D = 1, 4, 2048, 128
    q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
    BQ, BK = _steel_block_config(D)  # (32, 16) → asymmetric (NOT a built-in maker)
    NQ, NK = N // BQ, N // BK
    m = np.zeros((NQ, NK), bool); m[:, :NK // 4] = True; m = mx.array(m)
    o = flash_attention_sparse(q, k, v, m, scale=sc, causal=False)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=_block_bias(m, N, N))
    assert _delta(o, ref) == 0.0, "D=128 asymmetric sparse no longer the SDPA fallback (drift? update map)"


def test_sparse_D128_symmetric_is_real_nax_sparse():
    """D=128 + symmetric mask → real NAX sparse kernel (the reachable win)."""
    B, H, N, D = 1, 4, 2048, 128
    q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
    NB = N // 32  # bt=32 symmetric
    m = np.zeros((NB, NB), bool); m[:, :NB // 4] = True; m = mx.array(m)
    o = flash_attention_sparse(q, k, v, m, scale=sc, causal=False)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=_block_bias(m, N, N))
    d = _delta(o, ref)
    assert 1e-7 < d < 3e-2, f"D=128 symmetric not the real sparse kernel (Δ={d}: 0=SDPA-fallback, big=wrong)"


def test_sparse_D64_is_real_sparse():
    """D=64 default (symmetric) → real sparse kernel (slow, but real — see report)."""
    B, H, N, D = 1, 4, 2048, 64
    q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
    NB = N // 32
    m = np.zeros((NB, NB), bool); m[:, :NB // 4] = True; m = mx.array(m)
    o = flash_attention_sparse(q, k, v, m, scale=sc, causal=False)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=_block_bias(m, N, N))
    d = _delta(o, ref)
    assert 1e-7 < d < 3e-2, f"D=64 sparse drifted (Δ={d})"


# ── GNA / decode / backward ─────────────────────────────────────────────────
def test_gna_runs_native_not_sdpa_fallback():
    """flash_attention_gna (D=128 3D f16) → native GNA kernel (Δ≠0 vs block-bias SDPA)."""
    ss = (8, 8, 8); Ng = 512; D = 128
    q, k, v = _qkv(1, 4, Ng, D); sc = 1 / math.sqrt(D)
    o = flash_attention_gna(q, k, v, ss, (3, 3, 3), (1, 1, 1), scale=sc)
    gm = make_gna_mask(ss, (3, 3, 3), (1, 1, 1), head_dim=D)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=_block_bias(gm, Ng, Ng))
    assert _delta(o, ref) != 0.0, "GNA collapsed to the SDPA fallback (native kernel lost)"


def test_kvcache_decode_is_sdpa():
    """flash_attention_kvcache decode (N_q=1) → Apple SDPA (gather + SDPA)."""
    B, H, S, D = 1, 8, 1024, 128
    qd = _gen((B, H, 1, D))
    kc = _gen((B, H, S, D))
    vc = _gen((B, H, S, D)); mx.eval(qd, kc, vc)
    o = flash_attention_kvcache(qd, kc, vc, scale=1 / math.sqrt(D), causal=True, cache_seqlens=S)
    ref = mx.fast.scaled_dot_product_attention(qd, kc, vc, scale=1 / math.sqrt(D))
    assert _delta(o, ref) == 0.0, "kvcache decode no longer the SDPA path"


def test_dense_backward_is_sdpa_vjp():
    """mx.grad(flash_attention) dense → SDPA-vjp (byte-identical gradient)."""
    B, H, N, D = 1, 4, 1024, 128
    q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
    gd = mx.grad(lambda x: mx.sum(flash_attention(x, k, v, scale=sc, causal=False)))(q)
    gs = mx.grad(lambda x: mx.sum(mx.fast.scaled_dot_product_attention(x, k, v, scale=sc)))(q)
    assert _delta(gd, gs) == 0.0, "dense backward drifted off SDPA-vjp"


def test_sparse_backward_default_is_dense_sdpa_vjp():
    """GOTCHA (documented): sparse backward DEFAULT (no MFA_ENABLE_V6_BACKWARD) → dense
    SDPA-vjp dQ (the sparse forward win does NOT carry to the backward by default)."""
    B, H, N, D = 1, 4, 2048, 128
    q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
    NB = N // 32
    m = np.zeros((NB, NB), bool); m[:, :NB // 4] = True; m = mx.array(m)
    bias = _block_bias(m, N, N)
    gsp = mx.grad(lambda x: mx.sum(flash_attention_sparse(x, k, v, m, scale=sc, causal=False)))(q)
    gref = mx.grad(lambda x: mx.sum(mx.fast.scaled_dot_product_attention(x, k, v, scale=sc, mask=bias)))(q)
    assert _delta(gsp, gref) == 0.0, "sparse backward default no longer SDPA-vjp (update map if intended)"


# ── conv auto-hook (telemetry fingerprint) ──────────────────────────────────
def test_conv3d_nax_hook_eligible_executes():
    """Conv3D eligible (C%16==0 & ≥32, HW%8, B=1, pad=1, f16) → NAX conv kernel executes."""
    s0 = get_hook_stats().get("executed", {}).get("conv3d_nax_forward", 0)
    x = (mx.random.uniform(-1, 1, (1, 8, 16, 16, 128)) * 0.1).astype(mx.float16)
    w = (mx.random.uniform(-1, 1, (128, 3, 3, 3, 128)) * 0.1).astype(mx.float16); mx.eval(x, w)
    o = mx.conv_general(x, w, stride=1, padding=1); mx.eval(o)
    s1 = get_hook_stats().get("executed", {}).get("conv3d_nax_forward", 0)
    assert s1 > s0, "conv3d NAX hook did not execute for an eligible shape"


def test_conv3d_nax_hook_ineligible_falls_back():
    """Conv3D ineligible (C<32) → falls back to mx.conv_general (telemetry fallback)."""
    s0 = get_hook_stats().get("fallback", {}).get("conv3d_nax_forward", 0)
    x = (mx.random.uniform(-1, 1, (1, 8, 16, 16, 16)) * 0.1).astype(mx.float16)
    w = (mx.random.uniform(-1, 1, (16, 3, 3, 3, 16)) * 0.1).astype(mx.float16); mx.eval(x, w)
    o = mx.conv_general(x, w, stride=1, padding=1); mx.eval(o)
    s1 = get_hook_stats().get("fallback", {}).get("conv3d_nax_forward", 0)
    assert s1 > s0, "conv3d NAX ineligible shape did not record a fallback"
