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

This test asserts the CURRENT verified reality, INCLUDING the documented gotchas
(D=128-sparse→SDPA, D=64-sparse-slow, sparse-backward-dense), locked as
"expected, documented" so UNINTENTIONAL drift fails CI. Phase F will deliberately
update this map + test together when it fixes the routing. Do NOT weaken to make
a reroute pass silently — update the map entry with a comment when intended.

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


def _qkv(B, H, N, D, Hk=None):
    Hk = Hk or H
    f = lambda h: (mx.random.uniform(-1, 1, (B, h, N, D)) * 0.1).astype(mx.float16)
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
def test_dense_auto_is_sdpa():
    """flash_attention(backend=auto) dense → Apple SDPA (byte-identical)."""
    B, H, N, D = 1, 4, 2048, 128
    q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
    o = flash_attention(q, k, v, scale=sc, causal=False)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)
    assert _delta(o, ref) == 0.0, "dense auto drifted off the SDPA path"


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
    """GOTCHA (documented): D=128 + asymmetric mask (every built-in maker) → dense SDPA.
    Locked as current reality; Phase F will reroute + update this."""
    B, H, N, D = 1, 4, 2048, 128
    q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
    BQ, BK = _steel_block_config(D)  # (32, 16) → asymmetric
    NQ, NK = N // BQ, N // BK
    m = np.zeros((NQ, NK), bool); m[:, :NK // 4] = True; m = mx.array(m)
    o = flash_attention_sparse(q, k, v, m, scale=sc, causal=False)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=_block_bias(m, N, N))
    assert _delta(o, ref) == 0.0, "D=128 asymmetric sparse no longer the SDPA fallback (Phase F reroute? update map)"


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
    qd = (mx.random.uniform(-1, 1, (B, H, 1, D)) * 0.1).astype(mx.float16)
    kc = (mx.random.uniform(-1, 1, (B, H, S, D)) * 0.1).astype(mx.float16)
    vc = (mx.random.uniform(-1, 1, (B, H, S, D)) * 0.1).astype(mx.float16); mx.eval(qd, kc, vc)
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
