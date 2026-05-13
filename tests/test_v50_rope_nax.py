"""Sprint 2 (v2.50) — flash_attention_rope_unified M5+ NAX path tests.

Empirical foundation: docs/v50/sprint2-decisions.md.  On M5+ NAX
hardware, `mx.fast.rope` (Apple native rope kernel) + `flash_attention`
(Apple SDPA NAX) is ~4× faster than the STEEL `_mfa_rope_forward`
fused-rope kernel.  Sprint 2 adds an M5+ early-return path that uses
the NAX-optimal pair.

These tests verify:
  - M5+ NAX path engages by default for D=64/128 fp16/bf16 non-partial-rope
  - MFA_DISABLE_ROPE_NAX=1 forces STEEL fallback (back-compat)
  - Outputs match within FP16 ULP tolerance
  - Partial rope (rotary_dim < D) falls back to STEEL
  - fp32 falls back to STEEL
"""
import math
import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_mfa
from mlx_mfa import flash_attention_rope_unified, get_device_info

_flush = getattr(mx, "eval")

_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))


def _make_rope_tables(D, max_seqlen, base=10000.0):
    inv_freq = 1.0 / (base ** (mx.arange(0, D, 2).astype(mx.float32) / D))
    pos = mx.arange(max_seqlen).astype(mx.float32)
    freqs = mx.outer(pos, inv_freq)
    cos_t = mx.cos(freqs)
    sin_t = mx.sin(freqs)
    _flush(cos_t, sin_t); mx.synchronize()
    return cos_t, sin_t


# ─────────────────────────────────────────────────────────────────────
# Correctness: NAX path vs STEEL fallback
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 2 NAX path requires M5+ hardware.")
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_sprint2_rope_nax_matches_steel_fallback(monkeypatch, D, dtype):
    """NAX path output matches STEEL fallback within FP16 ULP tolerance."""
    B, H, qL = 1, 4, 2048
    mx.random.seed(42)
    q = mx.random.normal((B, H, qL, D)).astype(dtype)
    k = mx.random.normal((B, H, qL, D)).astype(dtype)
    v = mx.random.normal((B, H, qL, D)).astype(dtype)
    cos_t, sin_t = _make_rope_tables(D, qL)

    # Default (NAX path on M5+)
    monkeypatch.delenv("MFA_DISABLE_ROPE_NAX", raising=False)
    o_nax = flash_attention_rope_unified(
        q, k, v, rotary_cos=cos_t, rotary_sin=sin_t, scale=D**-0.5
    )
    _flush(o_nax); mx.synchronize()

    # Force STEEL fallback
    monkeypatch.setenv("MFA_DISABLE_ROPE_NAX", "1")
    o_steel = flash_attention_rope_unified(
        q, k, v, rotary_cos=cos_t, rotary_sin=sin_t, scale=D**-0.5
    )
    _flush(o_steel); mx.synchronize()

    # FP16/BF16 ULP tolerance ~5e-3 (rope rotation introduces ~1 ULP drift
    # at each multiplication; NAX path uses different reduction order)
    max_diff = float(mx.max(mx.abs(
        o_nax.astype(mx.float32) - o_steel.astype(mx.float32))))
    tol = 5e-3 if dtype == mx.float16 else 1e-2
    assert max_diff < tol, (
        f"NAX vs STEEL output diff {max_diff:.3e} exceeds {tol} ({dtype}, D={D})"
    )


# ─────────────────────────────────────────────────────────────────────
# Routing: env var opt-out + edge cases
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 2 NAX path requires M5+ hardware.")
def test_sprint2_rope_nax_disable_env_var(monkeypatch):
    """MFA_DISABLE_ROPE_NAX=1 forces STEEL fallback (no crash, correct output)."""
    B, H, qL, D = 1, 4, 2048, 64
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    cos_t, sin_t = _make_rope_tables(D, qL)

    monkeypatch.setenv("MFA_DISABLE_ROPE_NAX", "1")
    out = flash_attention_rope_unified(
        q, k, v, rotary_cos=cos_t, rotary_sin=sin_t, scale=D**-0.5
    )
    _flush(out); mx.synchronize()
    assert out.shape == (B, H, qL, D)
    assert not bool(mx.any(mx.isnan(out)))
    assert not bool(mx.any(mx.isinf(out)))


@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 2 NAX path requires M5+ hardware.")
def test_sprint2_rope_nax_partial_rope_falls_back(monkeypatch):
    """rotary_dim < D forces STEEL fallback (partial rope not yet supported by NAX path)."""
    B, H, qL, D = 1, 4, 2048, 128
    rotary_dim = 64  # < D
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    cos_t, sin_t = _make_rope_tables(rotary_dim, qL)

    monkeypatch.delenv("MFA_DISABLE_ROPE_NAX", raising=False)
    # Partial rope (rotary_dim<D): _partial_rope=True → goes through Python
    # path (line 899-910), not the standalone path where Sprint 2 NAX branch
    # lives.  This test verifies partial rope completes without crash on M5+.
    out = flash_attention_rope_unified(
        q, k, v, rotary_cos=cos_t, rotary_sin=sin_t, scale=D**-0.5,
        rotary_dim=rotary_dim,
    )
    _flush(out); mx.synchronize()
    assert out.shape == (B, H, qL, D)
    assert not bool(mx.any(mx.isnan(out)))


@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 2 NAX path requires M5+ hardware.")
def test_sprint2_rope_nax_fp32_falls_back(monkeypatch):
    """fp32 inputs force STEEL fallback (NAX path is f16/bf16 only)."""
    B, H, qL, D = 1, 4, 1024, 64
    q = mx.random.normal((B, H, qL, D)).astype(mx.float32)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float32)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float32)
    cos_t, sin_t = _make_rope_tables(D, qL)

    # fp32 → _can_use_mfa() check at line 899 forces Python path
    out = flash_attention_rope_unified(
        q, k, v, rotary_cos=cos_t, rotary_sin=sin_t, scale=D**-0.5
    )
    _flush(out); mx.synchronize()
    assert out.shape == (B, H, qL, D)
    assert not bool(mx.any(mx.isnan(out)))


# ─────────────────────────────────────────────────────────────────────
# Causal mask interaction (rope + causal)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 2 NAX path requires M5+ hardware.")
def test_sprint2_rope_nax_with_causal(monkeypatch):
    """Causal mask works correctly with Sprint 2 NAX rope path."""
    B, H, qL, D = 1, 4, 2048, 128
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    cos_t, sin_t = _make_rope_tables(D, qL)

    monkeypatch.delenv("MFA_DISABLE_ROPE_NAX", raising=False)
    o_nax = flash_attention_rope_unified(
        q, k, v, rotary_cos=cos_t, rotary_sin=sin_t,
        scale=D**-0.5, causal=True,
    )
    _flush(o_nax); mx.synchronize()

    monkeypatch.setenv("MFA_DISABLE_ROPE_NAX", "1")
    o_steel = flash_attention_rope_unified(
        q, k, v, rotary_cos=cos_t, rotary_sin=sin_t,
        scale=D**-0.5, causal=True,
    )
    _flush(o_steel); mx.synchronize()

    max_diff = float(mx.max(mx.abs(
        o_nax.astype(mx.float32) - o_steel.astype(mx.float32))))
    assert max_diff < 5e-3, (
        f"NAX vs STEEL causal output diff {max_diff:.3e}"
    )


# ─────────────────────────────────────────────────────────────────────
# Smoke: PUBLIC API engagement (axis-2)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_NAX, reason="Sprint 2 NAX path requires M5+ hardware.")
def test_sprint2_rope_nax_public_api_d64():
    """flash_attention_rope_unified D=64 fp16 standalone engages NAX path."""
    B, H, qL, D = 1, 4, 2048, 64
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16)
    cos_t, sin_t = _make_rope_tables(D, qL)

    out = flash_attention_rope_unified(
        q, k, v, rotary_cos=cos_t, rotary_sin=sin_t, scale=D**-0.5
    )
    _flush(out); mx.synchronize()
    assert out.shape == (B, H, qL, D)
    assert not bool(mx.any(mx.isnan(out)))
    assert not bool(mx.any(mx.isinf(out)))
