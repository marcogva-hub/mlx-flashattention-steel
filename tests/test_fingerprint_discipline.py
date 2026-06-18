"""Fingerprint-assertion discipline (audit Phase C) — make green-on-wrong-binary
STRUCTURALLY catchable.

The campaign's original sin: a test asserts the MATH (Δ vs reference small)
without asserting the BINARY (which kernel ran), so it stays green even when the
path silently fell back to SDPA (the four V6-NAX sprints + the pre-existing
D=128 sparse tests). This module locks the BINARY:

1. **Wrong-binary instances, documented + locked** — the pre-existing D=128
   asymmetric sparse-forward tests run dense SDPA on M5 (byteΔ==0.0 vs SDPA), so
   their "sparse ≈ SDPA-reference" assertions are vacuous (SDPA vs SDPA). We lock
   that CURRENT reality (Δ==0.0): when Phase F reroutes D=128 onto the real sparse
   kernel, these flip to Δ>0 and FAIL, forcing a deliberate update. (See
   phase-C-test-audit-report.md for the full ledger.)

2. **Positive fingerprint demo** — a symmetric-mask D=128 sparse call asserts
   byteΔ>0 vs SDPA (a real, distinct kernel ran). If it ever drifts to the SDPA
   fallback, byteΔ→0 and this FAILS. This is the discipline every kernel-claiming
   test should carry (the 42 B1–B4 lock cells already do).

M5+-gated.
"""
from __future__ import annotations

import math
import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa import (flash_attention_sparse, flash_attention, sage_attention,
                     flash_attention_topk, make_causal_block_mask, make_sliding_window_mask)
from mlx_mfa.attention import _get_is_m5_plus_cached, _steel_block_config

pytestmark = pytest.mark.skipif(
    not _get_is_m5_plus_cached(), reason="fingerprint discipline asserts M5+ dispatch")

mx.random.seed(10)


def _qkv(B, H, N, D):
    f = lambda: (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
    q, k, v = f(), f(), f(); mx.eval(q, k, v); return q, k, v


def _delta(a, b):
    mx.eval(a, b); return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _bias(m, N, S):
    NQ, NK = m.shape[-2], m.shape[-1]
    em = mx.repeat(mx.repeat(m.astype(mx.float32), N // NQ, -2), S // NK, -1)
    b = mx.where(em > 0, mx.array(0.0), mx.array(-1e9))
    while b.ndim < 4:
        b = b[None]
    return b.astype(mx.float16)


# ── 1. D=128 SDPA edge cases: the THREE residual routes that stay on SDPA ──────
class TestKnownWrongBinaryLockedAsSDPA:
    """Audit Phase F (2026-06-18) FIXED the common D=128 sparse case: built-in
    maker masks are now symmetric 32×32 and route to the real NAX-sparse kernel
    (see TestFingerprintDisciplineDemo + test_large_sliding_window_maker_is_nax).
    These cells lock the THREE residual routes that CORRECTLY stay on SDPA — not
    the old silent-fallback bug, but deliberate routing:
      (a) ASYMMETRIC / custom masks (bt_q != bt_k) — skip the symmetric auto-route;
      (b) SMALL masks (mask_bytes < 4096) — NAX device-pointer lowering excludes them;
      (c) DENSE symmetric masks (density >= the 0.78 ceiling) — NAX loses, gate → SDPA.
    Byte-identity (Δ==0.0 vs SDPA) is the fingerprint. A flip means a routing
    drift that must be re-examined."""

    def test_all_true_asymmetric_d128_is_sdpa_not_sparse(self):
        # (a) ASYMMETRIC mask (32x16, _steel_block_config) — not a symmetric maker;
        # bt_q=32 != bt_k=16 skips the NAX auto-route → SDPA fallback (unchanged).
        B, H, N, D = 1, 4, 2048, 128
        q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
        BQ, BK = _steel_block_config(D); NQ, NK = N // BQ, N // BK
        at = mx.ones((NQ, NK), mx.bool_)
        o = flash_attention_sparse(q, k, v, at, scale=sc)
        sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)
        assert _delta(o, sdpa) == 0.0, (
            "D=128 asymmetric all-true sparse no longer byte-identical to SDPA — "
            "the asymmetric path drifted off SDPA. Update this lock."
        )

    def test_small_sliding_window_d128_is_sdpa_not_sparse(self):
        # (b) SMALL symmetric mask: N=256 → 8x8 bool = 64 bytes < 4096 → the NAX
        # small-mask guard falls through to the SDPA fallback (Phase F unchanged).
        B, H, N, D = 1, 4, 256, 128
        q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
        m = make_sliding_window_mask(N, window_size=64, head_dim=D)  # symmetric 8x8
        o = flash_attention_sparse(q, k, v, m, scale=sc)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=_bias(m, N, N))
        assert _delta(o, ref) == 0.0, (
            "small (mask_bytes<4096) D=128 sparse rerouted off SDPA — update lock")

    def test_dense_symmetric_d128_is_sdpa_via_density_gate(self):
        # (c) DENSE symmetric mask (density >= 0.78 ceiling): the Phase-F density
        # gate routes it to the SDPA fallback (NAX loses when near-dense).
        B, H, N, D = 1, 4, 2048, 128
        q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
        NB = N // 32
        dm = np.random.default_rng(0).random((NB, NB)) < 0.90  # d~0.90 >= 0.78
        m = mx.array(dm)
        o = flash_attention_sparse(q, k, v, m, scale=sc)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=_bias(m, N, N))
        assert _delta(o, ref) == 0.0, (
            "dense symmetric D=128 sparse did NOT hit the density gate → SDPA "
            "(Phase F regression: gate threshold or routing changed)")


# ── 2. positive fingerprint demo: symmetric D=128 sparse IS a real kernel ─────
class TestFingerprintDisciplineDemo:
    def test_symmetric_d128_sparse_is_real_kernel_not_sdpa(self):
        """The CORRECT way to test the sparse kernel at D=128: a symmetric mask.
        Asserts a real distinct kernel ran (byteΔ>0 vs SDPA). Drift to the SDPA
        fallback would make byteΔ→0 and FAIL this — green-on-wrong-binary caught."""
        B, H, N, D = 1, 4, 2048, 128
        q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
        NB = N // 32  # symmetric bt=32 → engages the real V-selected sparse kernel
        m = np.zeros((NB, NB), bool); m[:, :NB // 4] = True; m = mx.array(m)
        o = flash_attention_sparse(q, k, v, m, scale=sc)
        sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=_bias(m, N, N))
        d = _delta(o, sdpa)
        # correctness (vs SDPA math) AND binary (distinct kernel, not the fallback)
        assert d < 3e-2, f"symmetric sparse wrong vs SDPA math (Δ={d})"
        assert d > 0.0, "symmetric D=128 sparse is byte-identical to SDPA — it drifted to the fallback (WRONG BINARY)"

    def test_large_sliding_window_maker_is_nax_not_sdpa(self):
        """Audit Phase F: the built-in D=128 sliding-window MAKER mask (large,
        symmetric 32x32, low density) now routes to the real NAX-sparse kernel
        — NOT the old silent SDPA fallback. byteΔ>0 vs SDPA proves the fix; a
        drift back to SDPA (byteΔ→0) re-opens the gotcha-1 loss and FAILS here."""
        B, H, N, D = 1, 4, 2048, 128
        q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
        m = make_sliding_window_mask(N, window_size=256, head_dim=D)  # symmetric 64x64, d~0.25
        o = flash_attention_sparse(q, k, v, m, scale=sc)
        sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=_bias(m, N, N))
        d = _delta(o, sdpa)
        assert d < 3e-2, f"sliding-window NAX-sparse wrong vs SDPA math (Δ={d})"
        assert d > 0.0, (
            "large D=128 sliding-window MAKER mask is byte-identical to SDPA — "
            "Phase-F maker→NAX routing drifted back to the SDPA fallback (WRONG BINARY)")


# ── 3. expert-path fingerprints (audit C2: the expert subset runs its CLAIMED binary) ──
class TestExpertPathsRunClaimedBinary:
    """C2: every expert-binary-claiming path verified to run a real DISTINCT kernel
    (byteΔ>0 vs the SDPA reference), not a silent fallback. A drift to SDPA flips
    byteΔ→0 and FAILS — green-on-wrong-binary caught for the expert paths too."""

    def _qkv(self, B, H, N, D):
        f = lambda: (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
        q, k, v = f(), f(), f(); mx.eval(q, k, v); return q, k, v

    def test_backend_mfa_is_steel_not_sdpa(self):
        q, k, v = self._qkv(2, 8, 4096, 128); sc = 1 / math.sqrt(128)
        o = flash_attention(q, k, v, scale=sc, causal=False, backend="mfa")
        s = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)
        assert _delta(o, s) > 0.0, "backend=mfa byte-identical to SDPA — STEEL path lost"

    def test_d64_backward_default_on_is_native(self):
        q, k, v = self._qkv(2, 8, 4096, 64); sc = 1 / math.sqrt(64)
        dO = (mx.random.uniform(-1, 1, q.shape) * 0.1).astype(mx.float16); mx.eval(dO)
        _, gk = mx.vjp(lambda a, b, c: flash_attention(a, b, c, scale=sc, causal=True), (q, k, v), (dO,))
        _, gs = mx.vjp(lambda a, b, c: mx.fast.scaled_dot_product_attention(a, b, c, scale=sc, mask="causal"), (q, k, v), (dO,))
        assert _delta(gk[0], gs[0]) > 0.0, "D=64 default-on backward reverted to SDPA-vjp (claim 'native' wrong)"

    def test_d128_backward_optin_is_native(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        q, k, v = self._qkv(2, 8, 2048, 128); sc = 1 / math.sqrt(128)
        dO = (mx.random.uniform(-1, 1, q.shape) * 0.1).astype(mx.float16); mx.eval(dO)
        _, gk = mx.vjp(lambda a, b, c: flash_attention(a, b, c, scale=sc, causal=False), (q, k, v), (dO,))
        _, gs = mx.vjp(lambda a, b, c: mx.fast.scaled_dot_product_attention(a, b, c, scale=sc), (q, k, v), (dO,))
        assert _delta(gk[0], gs[0]) > 0.0, "D=128 opt-in backward stayed SDPA-vjp (claim 'native' wrong)"

    def test_sage_is_int8_kernel_not_sdpa(self):
        q, k, v = self._qkv(2, 8, 512, 128); sc = 1 / math.sqrt(128)
        o = sage_attention(q, k, v, scale=sc, causal=False)
        s = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)
        assert _delta(o, s) > 0.0, "sage byte-identical to SDPA — int8 path lost"

    def test_topk_is_own_path_not_dense_sdpa(self):
        q, k, v = self._qkv(2, 8, 512, 128); sc = 1 / math.sqrt(128)
        o = flash_attention_topk(q, k, v, topk_ratio=0.25, scale=sc)
        s = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)
        assert _delta(o, s) > 0.0, "topk byte-identical to dense SDPA — top-k selection lost"
