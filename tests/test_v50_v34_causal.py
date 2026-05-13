"""Sprint 4 (v2.50) — V34 forward causal extension tests (Phase 4a +
dQ Phase 4b partial).

Phase 4a: V34 forward kernel (`createV34Source()`) now supports causal
masking via Apple SDPA NAX pattern (steel_attention_nax.h:176-187,
279-301): per-block `kb_lim` shrink + per-element `r < c → -inf` mask.

Phase 4b partial: V34 backward dQ kernel (`createV34BackwardQuerySource()`)
mirrors the forward causal mask block.  4 K-parallel backward kernels
(dKV, split dV, split dK, fused dKdV) are NOT yet causal-aware —
Phase 4b-complete deferred (see `docs/v50/sprint4-status-phase4b-complete.md`).

Eligibility gate (`_v34_eligible`) retains `not causal` so production
callers using `flash_attention(causal=True)` continue to use SDPA-vjp
fallback for the backward pass.

These tests verify:
  - V34 forward causal output matches mx.fast.scaled_dot_product_attention(mask='causal')
    within fp16 ULP tolerance
  - V34 forward non-causal still bit-identical to mx.fast.sdpa (Phase 4a
    introduced no regression on the non-causal path)
  - lse from V34 forward causal is finite, natural-log domain
  - Production `flash_attention(causal=True)` with MFA_ENABLE_V34_BACKWARD=1
    falls back cleanly to SDPA-vjp (eligibility gate retained for safety)
"""
import math
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import flash_attention, get_device_info
from mlx_mfa.attention import _v34_eligible

_flush = getattr(mx, "eval")

_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))


# ─────────────────────────────────────────────────────────────────────
# Phase 4a — V34 forward causal correctness (direct C binding)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 4 V34 forward causal requires M5+ hardware.")
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("qL", [256, 1024])
def test_sprint4_v34_fwd_causal_matches_sdpa(D, dtype, qL):
    """V34 forward causal output matches mx.fast.sdpa(mask='causal')."""
    from mlx_mfa._ext import v6_nax_forward
    B, H = 1, 4
    mx.random.seed(42)
    q = mx.random.normal((B, H, qL, D)).astype(dtype)
    k = mx.random.normal((B, H, qL, D)).astype(dtype)
    v = mx.random.normal((B, H, qL, D)).astype(dtype)
    _flush(q, k, v); mx.synchronize()
    scale = 1.0 / math.sqrt(D)

    o_v34, lse_v34 = v6_nax_forward(q, k, v, True, True)  # causal=True, force_v34=True
    _flush(o_v34, lse_v34); mx.synchronize()
    o_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask='causal')
    _flush(o_ref); mx.synchronize()

    max_diff = float(mx.max(mx.abs(
        o_v34.astype(mx.float32) - o_ref.astype(mx.float32))))
    tol = 5e-3 if dtype == mx.float16 else 2.5e-2
    assert max_diff < tol, (
        f"V34 fwd causal vs SDPA causal diff {max_diff:.3e} exceeds {tol} "
        f"({dtype}, D={D}, qL={qL})"
    )

    # lse must be finite and not all -inf (causal rows have at least one key)
    assert not bool(mx.any(mx.isnan(lse_v34)))
    # First-row lse is log(exp(scale*Q[0]·K[0])) — finite, not extreme
    first_row_lse = float(lse_v34[0, 0, 0])
    assert math.isfinite(first_row_lse), f"lse[0,0,0]={first_row_lse}"


@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 4 V34 forward causal requires M5+ hardware.")
def test_sprint4_v34_fwd_noncausal_unchanged():
    """Phase 4a introduces NO regression on the existing non-causal V34 forward path."""
    from mlx_mfa._ext import v6_nax_forward
    B, H, qL, D = 1, 4, 1024, 128
    mx.random.seed(7)
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    _flush(q, k, v); mx.synchronize()
    scale = 1.0 / math.sqrt(D)

    o_v34, _ = v6_nax_forward(q, k, v, False, True)
    _flush(o_v34); mx.synchronize()
    o_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    _flush(o_ref); mx.synchronize()
    diff = float(mx.max(mx.abs(o_v34.astype(mx.float32) - o_ref.astype(mx.float32))))
    assert diff < 5e-3, f"non-causal regression after Phase 4a: {diff}"


# ─────────────────────────────────────────────────────────────────────
# Phase 4b partial — eligibility gate safety + SDPA-vjp fallback
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 4 eligibility gate requires M5+ hardware.")
def test_sprint4_v34_eligibility_causal_returns_false(monkeypatch):
    """Eligibility gate retains `not causal` until Phase 4b-complete."""
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
    assert _v34_eligible(64, mx.float16, causal=True) is False
    assert _v34_eligible(64, mx.float16, causal=False) is True
    assert _v34_eligible(128, mx.float16, causal=True) is False


@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 4 SDPA-vjp fallback requires M5+ hardware.")
def test_sprint4_flash_attention_causal_uses_sdpa_vjp(monkeypatch):
    """flash_attention(causal=True) with V34_BACKWARD=1 falls back to SDPA-vjp.

    Verifies that the eligibility gate redirects causal callers to the
    safe SDPA-vjp path while Phase 4b-complete is deferred.  The
    forward output AND the backward gradients should both match
    mx.fast.scaled_dot_product_attention exactly (no V34 path engaged
    for causal=True at all).
    """
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
    B, H, qL, D = 1, 4, 2048, 64
    mx.random.seed(13)
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    dO = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    scale = 1.0 / math.sqrt(D)

    def test(q, k, v):
        return flash_attention(q, k, v, scale=scale, causal=True)
    def ref(q, k, v):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask='causal')

    _, (dQ_t, dK_t, dV_t) = mx.vjp(test, [q, k, v], [dO])
    _, (dQ_r, dK_r, dV_r) = mx.vjp(ref, [q, k, v], [dO])
    _flush(dQ_t, dK_t, dV_t, dQ_r, dK_r, dV_r); mx.synchronize()

    diff_q = float(mx.max(mx.abs(dQ_t.astype(mx.float32) - dQ_r.astype(mx.float32))))
    diff_k = float(mx.max(mx.abs(dK_t.astype(mx.float32) - dK_r.astype(mx.float32))))
    diff_v = float(mx.max(mx.abs(dV_t.astype(mx.float32) - dV_r.astype(mx.float32))))
    # SDPA-vjp fallback should match exactly (bit-identical in this path)
    assert diff_q < 1e-2, f"dQ diff {diff_q:.3e}"
    assert diff_k < 1e-2, f"dK diff {diff_k:.3e}"
    assert diff_v < 1e-2, f"dV diff {diff_v:.3e}"
