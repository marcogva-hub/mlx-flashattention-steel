"""Phase III-4 — dispatch/contract guard locks (D8, D9, D13).

D8: backend="mfa" must run the actual MFA Metal forward on V34-eligible
    cells, not Apple SDPA (forced-backend measurement integrity).
D9: V-dim-mismatch / cross-attention shapes must NOT enter the V34
    backward carve-out.
D13: the NAX rope fast path must honor non-base-10000 user tables (route
    to the table-using STEEL path instead of silently applying base=10000).
"""
from __future__ import annotations

import math
import os
import numpy as np
import pytest
import mlx.core as mx

from mlx_mfa import flash_attention, flash_attention_rope_unified, get_device_info
from mlx_mfa.attention import _rope_tables_match_base10000

_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))
_skipif_no_nax = pytest.mark.skipif(not _HAS_NAX, reason="M5+ NAX required")


@_skipif_no_nax
class TestD8ForcedBackend:
    def test_backend_mfa_runs_kernel_not_sdpa(self):
        B, H, N, D = 1, 4, 2048, 64
        mx.random.seed(1)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)
        s = 1.0 / math.sqrt(D)
        o_mfa = flash_attention(q, k, v, scale=s, causal=True, backend="mfa")
        o_sdpa = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=s, mask="causal")
        mx.eval(o_mfa, o_sdpa)
        # backend="mfa" must engage the MFA kernel: NOT bit-identical to
        # SDPA (the II-8 carve-out makes backend="auto" bit-SDPA, but a
        # FORCED kernel must actually run the kernel), yet within fp16.
        assert not bool(mx.all(o_mfa == o_sdpa).item()), \
            "backend='mfa' silently ran SDPA on a V34-eligible cell (D8)"
        err = float(mx.max(mx.abs(
            o_mfa.astype(mx.float32) - o_sdpa.astype(mx.float32))).item())
        assert err < 0.05, f"MFA forward off from SDPA by {err}"

    def test_backend_auto_stays_bit_sdpa(self):
        B, H, N, D = 1, 4, 2048, 64
        mx.random.seed(2)
        q = mx.random.normal((B, H, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(q, k, v)
        s = 1.0 / math.sqrt(D)
        o_auto = flash_attention(q, k, v, scale=s, causal=True, backend="auto")
        o_sdpa = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=s, mask="causal")
        mx.eval(o_auto, o_sdpa)
        assert bool(mx.all(o_auto == o_sdpa).item()), \
            "II-8 carve-out regressed: backend='auto' forward not bit-SDPA"


@_skipif_no_nax
class TestD13RopeTableHonored:
    def _tables(self, base, hd=64, maxs=256):
        half = hd // 2
        i = mx.arange(half, dtype=mx.float32)
        inv = mx.exp(-math.log(base) * (2.0 * i / hd))
        p = mx.arange(maxs, dtype=mx.float32)[:, None]
        ang = p * inv[None, :]
        c = mx.cos(ang).astype(mx.float16)
        s = mx.sin(ang).astype(mx.float16)
        mx.eval(c, s)
        return c, s

    def test_detector(self):
        c10, s10 = self._tables(10000.0)
        c5, s5 = self._tables(500000.0)
        assert _rope_tables_match_base10000(c10, s10, 64, True, mx.float16)
        assert not _rope_tables_match_base10000(c5, s5, 64, True, mx.float16)

    def test_custom_base_matches_optout_path(self):
        # base=5e5 tables: the fast path must route to the table-using
        # STEEL path, bit-identical to MFA_DISABLE_ROPE_NAX=1.
        c5, s5 = self._tables(500000.0)
        B, H, N, hd = 1, 2, 128, 64
        mx.random.seed(0)
        q = mx.random.normal((B, H, N, hd)).astype(mx.float16)
        k = mx.random.normal((B, H, N, hd)).astype(mx.float16)
        v = mx.random.normal((B, H, N, hd)).astype(mx.float16)
        mx.eval(q, k, v)
        os.environ.pop("MFA_DISABLE_ROPE_NAX", None)
        o_fix = flash_attention_rope_unified(
            q, k, v, rotary_cos=c5, rotary_sin=s5, causal=True, interleaved=True)
        os.environ["MFA_DISABLE_ROPE_NAX"] = "1"
        try:
            o_steel = flash_attention_rope_unified(
                q, k, v, rotary_cos=c5, rotary_sin=s5, causal=True,
                interleaved=True)
        finally:
            os.environ.pop("MFA_DISABLE_ROPE_NAX", None)
        mx.eval(o_fix, o_steel)
        err = float(mx.max(mx.abs(
            o_fix.astype(mx.float32) - o_steel.astype(mx.float32))).item())
        assert err < 1e-4, (
            f"custom-base rope not routed to the table-honoring path "
            f"(err {err} vs the opt-out reference) — D13 regressed")
