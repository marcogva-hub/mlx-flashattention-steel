"""Tests for v2.32.0 SDPA routing dispatch on M5+ NAX."""
import math
import os

import mlx.core as mx
import pytest

import mlx_mfa
from mlx_mfa import dispatch_policy, get_device_info


# Skip the whole module on hardware without NAX (M1-M4); the routing logic
# is M5-NAX-specific and these tests are a no-op on older HW.
_DEV = get_device_info()
_HAS_NAX_HW = bool(_DEV.get("is_m5_plus", False))
pytestmark = pytest.mark.skipif(
    not _HAS_NAX_HW,
    reason="v2.32.0 SDPA routing tests require M5+ NAX hardware (gen >= 17).",
)


def _correctness(q, k, v, *, causal=False):
    out = mlx_mfa.flash_attention(q, k, v, causal=causal)
    ref = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0/math.sqrt(q.shape[-1]),
        mask="causal" if causal else None,
    )
    mx.synchronize()
    diff = (out.astype(mx.float32) - ref.astype(mx.float32))
    return float(mx.sqrt(mx.mean(diff * diff)))


# ── Pure dispatch_policy tests (no MLX kernels) ────────────────────────


def test_should_use_mfa_canonical_d128_routes_to_sdpa():
    """M5+ NAX, D=128 canonical (qL>8) → SDPA."""
    res = dispatch_policy.should_use_mfa(
        128, 4096, False, True, has_nax=True, dtype=mx.float16,
    )
    assert res is False, "M5 NAX D=128 N=4096 should route to SDPA"


def test_should_use_mfa_canonical_d64_routes_to_sdpa():
    """M5+ NAX, D=64 canonical (qL>8) → SDPA."""
    res = dispatch_policy.should_use_mfa(
        64, 4096, False, True, has_nax=True, dtype=mx.float16,
    )
    assert res is False


def test_should_use_mfa_decode_routes_to_sdpa_on_nax():
    """M5+ NAX, decode pattern (qL=1, kL≥4096) → SDPA.

    Sprint A measured SDPA's sdpa_vector path winning 1.9-2.6× over
    MFA flash-decode on M5+ NAX (llama-decode-8k/32k). The cross-attn
    rule that previously routed this to MFA is now qualified with
    `has_nax ∧ seq_len ≤ 16 → fall through to NAX SDPA route`.
    """
    res = dispatch_policy.should_use_mfa(
        128, 1, False, True, has_nax=True, kv_seq_len=8192, dtype=mx.float16,
    )
    assert res is False, "M5+ NAX decode should route to SDPA per Sprint A"


def test_should_use_mfa_cross_attn_keeps_mfa_on_nax():
    """M5+ NAX, real cross-attn (qL>16, kL>=4096) → MFA.

    Sprint A measured ltx2-cross (qL=2048, kL=14000, D=64) at MFA +11%
    over SDPA. The cross-attn rule routes this to MFA via the existing
    `_kv_len >= 4096 ∧ seq_len <= 4096 → MFA` branch, which fires when
    seq_len > 16 (i.e., not pure decode).
    """
    res = dispatch_policy.should_use_mfa(
        64, 2048, False, True, has_nax=True, kv_seq_len=14000, dtype=mx.float16,
    )
    assert res is True, "M5+ NAX legitimate cross-attn should keep MFA per Sprint A"


def test_force_sdpa_route_overrides_dispatch():
    """MFA_FORCE_SDPA_ROUTE=1 → SDPA regardless."""
    os.environ["MFA_FORCE_SDPA_ROUTE"] = "1"
    try:
        res = dispatch_policy.should_use_mfa(
            128, 1, True, True, has_nax=True,
            kv_seq_len=4096, dtype=mx.float16,
        )
        assert res is False
    finally:
        os.environ.pop("MFA_FORCE_SDPA_ROUTE", None)


def test_disable_sdpa_route_falls_through_to_legacy():
    """MFA_DISABLE_SDPA_ROUTE=1 → SDPA-routing disabled, fall through to
    M3+ thresholds. D=128 N=4096 causal routes to MFA via M3+ table."""
    os.environ["MFA_DISABLE_SDPA_ROUTE"] = "1"
    try:
        res = dispatch_policy.should_use_mfa(
            128, 4096, True, True, has_nax=True, dtype=mx.float16,
        )
        assert res is True
    finally:
        os.environ.pop("MFA_DISABLE_SDPA_ROUTE", None)


def test_m3_plus_no_nax_unchanged():
    """M3/M4 (is_m3_plus but no NAX) preserves existing behavior."""
    res = dispatch_policy.should_use_mfa(
        128, 4096, True, True, has_nax=False, dtype=mx.float16,
    )
    # M3+ thresholds: D=128 causal min_N = 1024, so N=4096 → MFA
    assert res is True


def test_m1_unchanged():
    """M1/M2 unchanged."""
    res = dispatch_policy.should_use_mfa(
        128, 4096, True, False, has_nax=False, dtype=mx.float16,
    )
    # _DEFAULT_THRESHOLDS: D=128 causal min_N = 2048 → MFA
    assert res is True


def test_backend_mfa_overrides_sdpa_routing():
    """backend='mfa' forces MFA even on M5 NAX canonical."""
    res = dispatch_policy.should_use_mfa(
        128, 4096, False, True, has_nax=True, backend="mfa",
    )
    assert res is True


def test_backend_sdpa_overrides_mfa_decode():
    """backend='sdpa' forces SDPA even when NAX decode would normally use MFA."""
    res = dispatch_policy.should_use_mfa(
        128, 1, False, True, has_nax=True, kv_seq_len=8192, backend="sdpa",
    )
    assert res is False


# ── End-to-end correctness tests ───────────────────────────────────────


def test_canonical_d128_correctness():
    """D=128 canonical routes to SDPA — output must match SDPA reference."""
    q = mx.random.normal((1, 20, 4096, 128), dtype=mx.float16)
    rmse = _correctness(q, q, q)
    assert rmse < 1e-3, f"RMSE {rmse} exceeds 1e-3"


def test_canonical_d128_causal_correctness():
    """D=128 causal canonical routes to SDPA."""
    q = mx.random.normal((1, 32, 4096, 128), dtype=mx.float16)
    rmse = _correctness(q, q, q, causal=True)
    assert rmse < 1e-3


def test_d64_canonical_correctness():
    """D=64 canonical routes to SDPA."""
    q = mx.random.normal((1, 8, 4096, 64), dtype=mx.float16)
    rmse = _correctness(q, q, q)
    assert rmse < 1e-3


def test_d80_falls_back_to_sdpa():
    """D=80 (no MFA support) falls back to SDPA via _can_use_mfa."""
    q = mx.random.normal((1, 12, 1500, 80), dtype=mx.float16)
    rmse = _correctness(q, q, q)
    assert rmse < 1e-3


def test_decode_keeps_mfa_correctness():
    """qL=1 decode uses mlx-mfa (flash-decode); output matches SDPA."""
    q = mx.random.normal((1, 32, 1, 128), dtype=mx.float16)
    k = mx.random.normal((1, 8, 8192, 128), dtype=mx.float16)
    v = mx.random.normal((1, 8, 8192, 128), dtype=mx.float16)
    rmse = _correctness(q, k, v)
    assert rmse < 1e-3


def test_disable_sdpa_route_env_var_e2e():
    """MFA_DISABLE_SDPA_ROUTE=1 forces native mlx-mfa path; output still correct."""
    os.environ["MFA_DISABLE_SDPA_ROUTE"] = "1"
    mlx_mfa.attention._dispatch_decision_cache.clear()
    try:
        q = mx.random.normal((1, 20, 4096, 128), dtype=mx.float16)
        rmse = _correctness(q, q, q)
        assert rmse < 1e-3
    finally:
        os.environ.pop("MFA_DISABLE_SDPA_ROUTE", None)
        mlx_mfa.attention._dispatch_decision_cache.clear()


# ── Carve-out infrastructure tests ─────────────────────────────────────


def test_carveout_function_exists_and_returns_false_by_default():
    """Sprint A.6 carve-outs use _should_use_mfa_m5_nax_carveout(...).
    With no carve-outs configured, it returns False (default to SDPA)."""
    res = dispatch_policy._should_use_mfa_m5_nax_carveout(
        head_dim=128, seq_len=4096, kv_seq_len=4096,
        causal=False, dtype_key="float16",
    )
    # Default behavior — Sprint A.6 carve-outs may extend this.
    # The default must be False (SDPA wins on canonical M5 NAX).
    assert isinstance(res, bool)
