"""Phase II-0 — V6NAX backward D=64 causal default-on promotion tests.

Marco-approved promotion (Phase-I Track 2: 2.2-2.6x vs SDPA-vjp).
Default-on envelope: D=64, causal, qL >= 2048, fp16/bf16, M5+ NAX.
Opt-out: MFA_DISABLE_V6_BACKWARD=1.  Broader envelope (D=64 non-causal,
D=128) remains opt-in via MFA_ENABLE_V6_BACKWARD=1.

Also locks the GQA gradient-shape fix this promotion surfaced: V6NAX
backward kernels emit Hq-shaped dK/dV; the orchestrator now group-sums
to [B, H_kv, S, D] (latent bug in the opt-in path since v2.37.0).
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import flash_attention, get_device_info
from mlx_mfa.attention import _v6nax_eligible

_eval_force = mx.eval
_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))
_skipif_no_nax = pytest.mark.skipif(not _HAS_NAX, reason="V6NAX requires M5+ NAX")


def _mk(B, Hq, Hkv, N, D, seed):
    # Phase II-6: fixture magnitude raised 0.1 -> 1.0.  The 0.1-scale
    # fixtures exponentially suppressed the paired-MMA BK=16 corruption
    # (P = exp2(wrong-S - L) errors shrink with score magnitude) and
    # let the II-0 promotion gate pass over a corrupt fused kernel.
    # Unit scale is the realistic activation envelope; the II-6 lock
    # file additionally tests std 2.0 and 12.0.
    mx.random.seed(seed)
    q = mx.random.normal((B, Hq, N, D)).astype(mx.float16)
    k = mx.random.normal((B, Hkv, N, D)).astype(mx.float16)
    v = mx.random.normal((B, Hkv, N, D)).astype(mx.float16)
    _eval_force(q, k, v)
    return q, k, v


def _grads(fn, q, k, v):
    dO = mx.ones_like(q)
    _, g = mx.vjp(fn, [q, k, v], [dO])
    _eval_force(*g)
    return g


def _rmse(a, b):
    """Hybrid error: min(absolute RMSE, relative RMSE).  Absolute alone
    fails on group-summed MQA gradients (magnitudes ~ratio x larger);
    relative alone fails on near-zero-magnitude gradients (cancellation
    inflates the ratio).  A true wrong-kernel/wrong-shape result fails
    BOTH by orders of magnitude."""
    a32 = np.asarray(a.astype(mx.float32)); b32 = np.asarray(b.astype(mx.float32))
    abs_rmse = float(np.sqrt(((a32 - b32) ** 2).mean()))
    den = max(1e-6, float(np.sqrt((b32 ** 2).mean())))
    return min(abs_rmse, abs_rmse / den)


class TestDefaultOnEligibility:
    def test_d64_causal_default_on_with_seq(self, monkeypatch):
        monkeypatch.delenv("MFA_ENABLE_V6_BACKWARD", raising=False)
        monkeypatch.delenv("MFA_DISABLE_V6_BACKWARD", raising=False)
        if not _HAS_NAX:
            pytest.skip("NAX required")
        assert _v6nax_eligible(64, mx.float16, causal=True, seq_len=4096) is True

    def test_opt_out_respected(self, monkeypatch):
        monkeypatch.setenv("MFA_DISABLE_V6_BACKWARD", "1")
        assert _v6nax_eligible(64, mx.float16, causal=True, seq_len=4096) is False

    def test_below_seq_floor_stays_off(self, monkeypatch):
        monkeypatch.delenv("MFA_ENABLE_V6_BACKWARD", raising=False)
        assert _v6nax_eligible(64, mx.float16, causal=True, seq_len=1024) is False

    def test_non_causal_default_on_ii12(self, monkeypatch):
        """Phase II-12: non-causal D=64 is NOW default-on (1.72-2.01x via
        the clean split kernel; same envelope + opt-out as causal)."""
        monkeypatch.delenv("MFA_ENABLE_V6_BACKWARD", raising=False)
        monkeypatch.delenv("MFA_DISABLE_V6_BACKWARD", raising=False)
        assert _v6nax_eligible(64, mx.float16, False, seq_len=4096) is True
        monkeypatch.setenv("MFA_DISABLE_V6_BACKWARD", "1")
        assert _v6nax_eligible(64, mx.float16, False, seq_len=4096) is False

    def test_d128_not_widened(self, monkeypatch):
        monkeypatch.delenv("MFA_ENABLE_V6_BACKWARD", raising=False)
        assert _v6nax_eligible(128, mx.float16, causal=True, seq_len=4096) is False

    def test_legacy_no_seq_unchanged(self, monkeypatch):
        # Calls without seq_len keep the env-opt-in behavior (test truth table).
        monkeypatch.delenv("MFA_ENABLE_V6_BACKWARD", raising=False)
        assert _v6nax_eligible(64, mx.float16, causal=True) is False


class TestPromotedCellCorrectness:
    @_skipif_no_nax
    @pytest.mark.parametrize("N", [2048, 4096])
    def test_mha_matches_sdpa_vjp(self, N):
        q, k, v = _mk(1, 8, 8, N, 64, seed=21)
        s = 1.0 / math.sqrt(64)
        g = _grads(lambda a, b, c: flash_attention(a, b, c, scale=s, causal=True), q, k, v)
        gr = _grads(lambda a, b, c: mx.fast.scaled_dot_product_attention(
            a, b, c, scale=s, mask="causal"), q, k, v)
        for name, x, y in zip(("dQ", "dK", "dV"), g, gr):
            assert _rmse(x, y) < 5e-3, f"{name} N={N}"

    @_skipif_no_nax
    @pytest.mark.parametrize("Hkv", [2, 1])  # GQA ratio 4, MQA
    def test_gqa_mqa_shapes_and_values(self, Hkv):
        """Locks the Phase II-0 GQA gradient-shape fix: dK/dV must be
        [B, H_kv, S, D] (group-summed), matching SDPA-vjp."""
        q, k, v = _mk(1, 8, Hkv, 4096, 64, seed=22)
        s = 1.0 / math.sqrt(64)
        g = _grads(lambda a, b, c: flash_attention(a, b, c, scale=s, causal=True), q, k, v)
        gr = _grads(lambda a, b, c: mx.fast.scaled_dot_product_attention(
            a, b, c, scale=s, mask="causal"), q, k, v)
        for name, x, y in zip(("dQ", "dK", "dV"), g, gr):
            assert tuple(x.shape) == tuple(y.shape), (
                f"{name} shape {tuple(x.shape)} != SDPA-vjp {tuple(y.shape)} "
                f"(Hkv={Hkv}) — GQA group-sum reduction regressed")
            assert _rmse(x, y) < 5e-3, f"{name} Hkv={Hkv}"
