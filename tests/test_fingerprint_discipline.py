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

from mlx_mfa import flash_attention_sparse, make_causal_block_mask, make_sliding_window_mask
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


# ── 1. wrong-binary instances: locked as "runs SDPA on M5" (flip = Phase-F drift) ──
class TestKnownWrongBinaryLockedAsSDPA:
    """These pre-existing-style sparse-forward calls run dense SDPA on M5 (the
    asymmetric-mask → _sparse_fallback_sdpa_perhead route). Locked so the vacuous
    state is explicit and a Phase-F reroute is forced to update it."""

    def test_all_true_d128_is_sdpa_not_sparse(self):
        B, H, N, D = 1, 4, 2048, 128
        q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
        BQ, BK = _steel_block_config(D); NQ, NK = N // BQ, N // BK
        at = mx.ones((NQ, NK), mx.bool_)
        o = flash_attention_sparse(q, k, v, at, scale=sc)
        sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)
        assert _delta(o, sdpa) == 0.0, (
            "D=128 all-true sparse no longer byte-identical to SDPA — the asymmetric "
            "path was rerouted to a real kernel (Phase F?). Update this lock + the "
            "pre-existing test's claim."
        )

    def test_sliding_window_d128_is_sdpa_not_sparse(self):
        B, H, N, D = 1, 4, 256, 128
        q, k, v = _qkv(B, H, N, D); sc = 1 / math.sqrt(D)
        m = make_sliding_window_mask(N, window_size=64, head_dim=D)
        o = flash_attention_sparse(q, k, v, m, scale=sc)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=_bias(m, N, N))
        assert _delta(o, ref) == 0.0, "D=128 sliding-window sparse rerouted off SDPA — update lock"


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
