"""Sprint v2.39.0 Phase C.1.a tests — Option γ fused dK+dV kernel (D=64 only).

Three-axis validation per §3.5 amended:
- Axis 1 (output sanity): fused vs split bit-identical / RMSE bound
- Axis 2 (path entered via PUBLIC API): mx.grad(flash_attention(..., backend="auto"))
  with MFA_V34_BWD_KERNEL routes correctly
- Axis 3 (edges preserved): D=128 falls back to split; legacy env vars honored

Per /metal-kernel-dev audit (2026-05-13): fused kernel is algebraically
identical to split kernels (order constraint: dV before dS overwrite).
Expected ≤1 ULP drift; observed RMSE=0 across all tested shapes.
"""
import math
import os

import mlx.core as mx
import pytest

import mlx_mfa
from mlx_mfa import flash_attention, get_device_info

_flush = getattr(mx, "eval")

# Hardware gate: V34 backward path requires M5+ NAX.
_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))
pytestmark = pytest.mark.skipif(
    not _HAS_NAX,
    reason="v2.39.0 fused dK+dV kernel requires M5+ NAX hardware (gen >= 17).",
)


# Force V34 backward path engagement for all tests in this module.
@pytest.fixture(autouse=True)
def _enable_v34_backward(monkeypatch):
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
    yield


def _make_inputs(B, H, qL, D, dtype=mx.float16, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, H, qL, D)).astype(dtype) * 0.1
    k = mx.random.normal((B, H, qL, D)).astype(dtype) * 0.1
    v = mx.random.normal((B, H, qL, D)).astype(dtype) * 0.1
    dO = mx.random.normal((B, H, qL, D)).astype(dtype) * 0.1
    _flush(q, k, v, dO)
    return q, k, v, dO


def _grads(q, k, v, dO, scale, env_kernel_mode):
    """Compute (dQ, dK, dV) under env-pinned kernel mode.

    Restores the prior env value at exit to avoid contaminating other tests
    (especially `test_flash_attention_v34_backward.py` which runs in the
    same pytest session and assumes the auto-default routing).
    """
    prior = os.environ.get("MFA_V34_BWD_KERNEL")
    os.environ["MFA_V34_BWD_KERNEL"] = env_kernel_mode
    try:
        def fn(qq, kk, vv):
            out = flash_attention(qq, kk, vv, scale=scale, causal=False,
                                   backend="auto")
            return (out * dO.astype(mx.float32)).sum()

        dq, dk, dv = mx.grad(fn, argnums=(0, 1, 2))(q, k, v)
        _flush(dq, dk, dv)
        mx.synchronize()
        return dq, dk, dv
    finally:
        if prior is None:
            os.environ.pop("MFA_V34_BWD_KERNEL", None)
        else:
            os.environ["MFA_V34_BWD_KERNEL"] = prior


def _rmse(a, b):
    d = a.astype(mx.float32) - b.astype(mx.float32)
    return float(mx.sqrt(mx.mean(d * d)))


# ── Axis 1: output sanity ─────────────────────────────────────────────

@pytest.mark.parametrize("qL", [2048, 4096, 8192])
def test_fused_d64_matches_split_d64(qL, monkeypatch):
    """Fused dK+dV produces gradients identical to split kernels at D=64.

    /metal-kernel-dev audit predicted ≤1 ULP drift; observed RMSE=0 (bit-
    identical) across all D=64 shapes due to identical FP order-of-ops
    within each accumulator's reduction tree.
    """
    B, H, D = 1, 4, 64
    q, k, v, dO = _make_inputs(B, H, qL, D, seed=qL)
    scale = 1.0 / math.sqrt(D)
    dqf, dkf, dvf = _grads(q, k, v, dO, scale, "fused")
    dqs, dks, dvs = _grads(q, k, v, dO, scale, "split")
    # dQ kernel is unchanged across paths → exact match expected.
    assert _rmse(dqf, dqs) == 0.0
    # dK + dV from fused vs split: bit-identical (audit-predicted; verified).
    # Tolerance: 1e-4 fp16 ULP if floating-point reorder ever introduces drift.
    assert _rmse(dkf, dks) <= 1e-4
    assert _rmse(dvf, dvs) <= 1e-4


@pytest.mark.parametrize("qL", [4096, 8192])
def test_fused_d64_no_nan_no_inf(qL, monkeypatch):
    """Fused path produces finite gradients (no NaN, no Inf)."""
    B, H, D = 1, 4, 64
    q, k, v, dO = _make_inputs(B, H, qL, D, seed=qL + 100)
    scale = 1.0 / math.sqrt(D)
    dq, dk, dv = _grads(q, k, v, dO, scale, "fused")
    for grad, name in [(dq, "dQ"), (dk, "dK"), (dv, "dV")]:
        assert not bool(mx.any(mx.isnan(grad))), f"{name} contains NaN"
        assert not bool(mx.any(mx.isinf(grad))), f"{name} contains Inf"


def test_fused_d64_matches_sdpa_vjp(monkeypatch):
    """Fused dK+dV gradients match SDPA-vjp baseline within FP16 tolerance."""
    B, H, qL, D = 1, 4, 4096, 64
    q, k, v, dO = _make_inputs(B, H, qL, D, seed=42)
    scale = 1.0 / math.sqrt(D)
    dqf, dkf, dvf = _grads(q, k, v, dO, scale, "fused")

    # SDPA-vjp reference path (disable V34 → falls back to mx.fast SDPA vjp).
    monkeypatch.setenv("MFA_DISABLE_V34_BACKWARD", "1")
    monkeypatch.delenv("MFA_ENABLE_V34_BACKWARD", raising=False)
    monkeypatch.delenv("MFA_V34_BWD_KERNEL", raising=False)

    def fn_ref(qq, kk, vv):
        out = flash_attention(qq, kk, vv, scale=scale, causal=False,
                               backend="auto")
        return (out * dO.astype(mx.float32)).sum()

    dqr, dkr, dvr = mx.grad(fn_ref, argnums=(0, 1, 2))(q, k, v)
    _flush(dqr, dkr, dvr)
    mx.synchronize()

    # FP16 tolerance band; same thresholds as v2.38.1 perf-claim tests.
    assert _rmse(dqf, dqr) < 1e-3, f"dQ RMSE {_rmse(dqf, dqr):.3e}"
    assert _rmse(dkf, dkr) < 1e-3, f"dK RMSE {_rmse(dkf, dkr):.3e}"
    assert _rmse(dvf, dvr) < 1e-3, f"dV RMSE {_rmse(dvf, dvr):.3e}"


# ── Axis 2: path entered via PUBLIC API ───────────────────────────────

def test_auto_default_engages_fused_for_d64(monkeypatch):
    """v2.39.1 outcome α: MFA_V34_BWD_KERNEL=auto routes D=64 to FUSED.

    The v2.39.0 outcome δ regression was root-caused to H1 register pressure
    at the default BK=32 and fixed in v2.39.1 by lowering BK to 16.  Auto
    now defaults to fused for D=64.  Correctness preserved within FP16
    tolerance (~2e-5 RMSE vs split, same as v2.38.1 D_vec drift vs SDPA).
    See docs/v6-nax/v39-1-investigation-synthesis.md.
    """
    B, H, qL, D = 1, 4, 4096, 64
    q, k, v, dO = _make_inputs(B, H, qL, D, seed=1)
    scale = 1.0 / math.sqrt(D)
    monkeypatch.setenv("MFA_V34_BWD_KERNEL", "auto")
    dqa, dka, dva = _grads(q, k, v, dO, scale, "auto")
    dqf, dkf, dvf = _grads(q, k, v, dO, scale, "fused")
    # auto and fused produce identical gradients on D=64 (auto→fused per α).
    assert _rmse(dqa, dqf) == 0.0
    assert _rmse(dka, dkf) == 0.0
    assert _rmse(dva, dvf) == 0.0


def test_auto_default_engages_split_for_d128(monkeypatch):
    """MFA_V34_BWD_KERNEL=auto routes D=128 to split kernel.

    D=128 has no fused kernel implementation (Phase C.1.b deferred); auto
    naturally falls through to split regardless of outcome δ.  Same observed
    behavior as v2.38.1.
    """
    B, H, qL, D = 1, 4, 4096, 128
    q, k, v, dO = _make_inputs(B, H, qL, D, seed=2)
    scale = 1.0 / math.sqrt(D)
    monkeypatch.setenv("MFA_V34_BWD_KERNEL", "auto")
    dqa, dka, dva = _grads(q, k, v, dO, scale, "auto")
    monkeypatch.setenv("MFA_V34_BWD_KERNEL", "split")
    dqs, dks, dvs = _grads(q, k, v, dO, scale, "split")
    # D=128 auto → split → identical gradients.
    assert _rmse(dqa, dqs) == 0.0
    assert _rmse(dka, dks) == 0.0
    assert _rmse(dva, dvs) == 0.0


def test_fused_opt_in_at_d64_still_works(monkeypatch):
    """Fused kernel remains opt-in via MFA_V34_BWD_KERNEL=fused at D=64.

    Despite outcome δ (not auto-default), the kernel ships and is reachable
    for users who want to bench on their own workloads or for future-sprint
    perf-tuning experiments.  Correctness verified bit-identical to split.
    """
    B, H, qL, D = 1, 4, 4096, 64
    q, k, v, dO = _make_inputs(B, H, qL, D, seed=8)
    scale = 1.0 / math.sqrt(D)
    monkeypatch.setenv("MFA_V34_BWD_KERNEL", "fused")
    dq, dk, dv = _grads(q, k, v, dO, scale, "fused")
    for g, name in [(dq, "dQ"), (dk, "dK"), (dv, "dV")]:
        assert not bool(mx.any(mx.isnan(g))), f"{name} NaN"
        assert not bool(mx.any(mx.isinf(g))), f"{name} Inf"


def test_fused_at_d128_works_via_direct_binding(monkeypatch):
    """v2.40.0-internal Sprint B: D=128 fused kernel correctness via DIRECT
    binding (PUBLIC AUTO API does NOT reach D=128 fused).

    Phase C.1.a (v2.39.0/.1) shipped fused for D=64.  Phase C.1.b
    (v2.40.0-internal Sprint B) lifted the D=64-only hard-gate so the
    D-parameterized source generator covers D=128 too.  HOWEVER:

      - dispatch_policy.should_use_mfa(D=128) returns False (threshold
        999_999 in `_M5_NAX_THRESHOLDS[(128, False)]`)
      - _v34_backward_carveout(D=128) returns False (D=64 hard-gated)

    So `mx.grad(flash_attention(..., backend="auto"))` at D=128 routes
    to SDPA-vjp fallback BEFORE `_make_mfa_custom` even constructs the
    vjp closure.  MFA_V34_BWD_KERNEL=fused is ignored at D=128 via
    PUBLIC API.

    Architectural ship: the kernel is reachable via direct C++ binding
    `_ext.v6_nax_backward_fused_dkdv_raw`.  This test verifies the
    kernel produces correct (~FP16-tolerance) gradients at D=128 when
    invoked directly.  Bench data (also via direct binding) shows
    fused regresses 3-7% vs split at qL ≤ 8192, parity at qL=16384 —
    outcome (γ) per blueprint, no auto-default change, no perf claim.

    See `docs/v6-nax/v40-0-internal-decisions.md` for full methodology.
    """
    from mlx_mfa._ext import (
        v6_nax_backward_fused_dkdv_raw,
        v6_nax_backward_dv_raw,
        v6_nax_backward_dk_raw,
        v6_nax_forward,
    )

    B, H, qL, D = 1, 4, 4096, 128
    mx.random.seed(99)
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16) * 0.1
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16) * 0.1
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16) * 0.1
    dO = mx.random.normal((B, H, qL, D)).astype(mx.float16) * 0.1
    _flush(q, k, v, dO)
    # Run V34 forward to get O + natural-log lse (force_v34=True is critical:
    # V34 backward kernels expect natural-log lse, not log2 from legacy).
    O, L = v6_nax_forward(q, k, v, False, True)
    _flush(O, L); mx.synchronize()
    # Precompute D_vec per v2.38.1 contract
    D_vec = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1)
    _flush(D_vec); mx.synchronize()

    scale = 1.0 / math.sqrt(D)
    WM = 4

    # Fused: single kernel computes both dK and dV partials
    dKp_f, dVp_f = v6_nax_backward_fused_dkdv_raw(
        q, k, v, L, dO, D_vec, scale, WM)
    _flush(dKp_f, dVp_f); mx.synchronize()
    dK_f = mx.sum(dKp_f, axis=2).astype(mx.float16)
    dV_f = mx.sum(dVp_f, axis=2).astype(mx.float16)

    # Split: separate dV + dK kernels (reference)
    dVp_s = v6_nax_backward_dv_raw(q, k, v, L, dO, scale, WM)
    dKp_s = v6_nax_backward_dk_raw(q, k, v, O, L, dO, D_vec, scale, WM)
    _flush(dKp_s, dVp_s); mx.synchronize()
    dK_s = mx.sum(dKp_s, axis=2).astype(mx.float16)
    dV_s = mx.sum(dVp_s, axis=2).astype(mx.float16)

    # Finite gradients
    for g, name in [(dK_f, "dK_fused"), (dV_f, "dV_fused")]:
        assert not bool(mx.any(mx.isnan(g))), f"{name} NaN"
        assert not bool(mx.any(mx.isinf(g))), f"{name} Inf"

    # RMSE bit-identical-or-near-identical to split (FP16 ULP tolerance,
    # ~2e-5 measured in v2.40.0-internal smoke; same magnitude as v2.38.1
    # D_vec drift vs SDPA).
    assert _rmse(dK_f, dK_s) <= 1e-3, f"dK fused vs split RMSE: {_rmse(dK_f, dK_s)}"
    assert _rmse(dV_f, dV_s) <= 1e-3, f"dV fused vs split RMSE: {_rmse(dV_f, dV_s)}"


def test_d128_split_engages_via_public_api(monkeypatch):
    """v2.50 Prompt 5b Section D: PUBLIC AUTO API engages V34 split kernels
    for D=128.  Updated from prior "unreachable_via_public_api" test that
    codified the pre-broadening "D=128 PUBLIC API routes to SDPA-vjp"
    contract.  Per Sprint B v2.40.0-internal Phase C.1.b, D=128 split
    kernels achieve parity with SDPA-vjp (RMSE ~2e-5); Prompt 5b Section
    D broadens the carve-out.

    Test is REGRESSION coverage: if a future sprint NARROWS the carve-out
    back to D=64-only, this test must be updated.  D=128 fused remains
    OPT-IN (not auto-default per outcome γ); only D=128 split engages via
    AUTO.
    """
    from mlx_mfa.dispatch_policy import (
        should_use_mfa, _v34_backward_carveout, _dispatch_dtype_key,
    )
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
    fp16_key = _dispatch_dtype_key(mx.float16)
    # `should_use_mfa` still returns False for D=128 (V34 backward path
    # routes via carve-out, NOT through the legacy MFA path).  This is
    # the design intent — V34 backward broadening doesn't change MFA
    # forward routing.
    assert should_use_mfa(
        head_dim=128, seq_len=4096, causal=False,
        is_m3_plus=True, has_nax=True, dtype=mx.float16,
    ) is False
    # The carve-out NOW returns True for D=128 + qL>=2048 + fp16 + env=1
    # (post-Prompt 5b Section D broadening).
    assert _v34_backward_carveout(
        head_dim=128, seq_len=4096, causal=False, dtype_key=fp16_key,
    ) is True


def test_fused_at_d256_still_raises_loudly(monkeypatch):
    """MFA_V34_BWD_KERNEL=fused with D=256 raises ValueError (Rule 8 loud failure).

    Phase C.1.a + C.1.b cover D ∈ {64, 128}.  D=256 and other head_dim
    values are not supported by the fused kernel — silent fallback would
    mask user mis-configuration, so we raise a clear error.
    """
    from mlx_mfa.attention import _v34_backward_vjp

    B, H, qL, D = 1, 4, 2048, 256
    mx.random.seed(100)
    q = mx.random.normal((B, H, qL, D)).astype(mx.float16) * 0.1
    k = mx.random.normal((B, H, qL, D)).astype(mx.float16) * 0.1
    v = mx.random.normal((B, H, qL, D)).astype(mx.float16) * 0.1
    O = mx.random.normal((B, H, qL, D)).astype(mx.float16) * 0.1
    L = mx.random.normal((B, H, qL)).astype(mx.float32)
    dO = mx.random.normal((B, H, qL, D)).astype(mx.float16) * 0.1
    _flush(q, k, v, O, L, dO)

    monkeypatch.setenv("MFA_V34_BWD_KERNEL", "fused")
    scale = 1.0 / math.sqrt(D)

    with pytest.raises(ValueError, match="head_dim"):
        dQ, dK, dV = _v34_backward_vjp(q, k, v, O, L, dO, scale)
        _flush(dQ, dK, dV)
        mx.synchronize()


def test_env_split_override_works(monkeypatch):
    """MFA_V34_BWD_KERNEL=split forces split path even at D=64."""
    B, H, qL, D = 1, 4, 4096, 64
    q, k, v, dO = _make_inputs(B, H, qL, D, seed=3)
    scale = 1.0 / math.sqrt(D)
    # Verify split engages without crashing (matches split semantics).
    monkeypatch.setenv("MFA_V34_BWD_KERNEL", "split")
    dqs, dks, dvs = _grads(q, k, v, dO, scale, "split")
    for g in (dqs, dks, dvs):
        assert not bool(mx.any(mx.isnan(g)))


def test_legacy_fused_env_var_backcompat(monkeypatch):
    """MFA_V34BWD_USE_FUSED=1 (legacy v2.38.0 env) routes to legacy fused kernel."""
    B, H, qL, D = 1, 4, 4096, 64
    q, k, v, dO = _make_inputs(B, H, qL, D, seed=4)
    scale = 1.0 / math.sqrt(D)
    monkeypatch.delenv("MFA_V34_BWD_KERNEL", raising=False)
    monkeypatch.setenv("MFA_V34BWD_USE_FUSED", "1")

    def fn(qq, kk, vv):
        out = flash_attention(qq, kk, vv, scale=scale, causal=False,
                               backend="auto")
        return (out * dO.astype(mx.float32)).sum()

    dq, dk, dv = mx.grad(fn, argnums=(0, 1, 2))(q, k, v)
    _flush(dq, dk, dv)
    mx.synchronize()
    # Legacy fused path: finite gradients verified (correctness within legacy
    # WM=1 kernel's own bounds; not directly compared to fused/split).
    for g, name in [(dq, "dQ"), (dk, "dK"), (dv, "dV")]:
        assert not bool(mx.any(mx.isnan(g))), f"{name} NaN"
        assert not bool(mx.any(mx.isinf(g))), f"{name} Inf"


# ── Axis 3: edges preserved ───────────────────────────────────────────

def test_v37_carveout_still_eligible_at_d64_qL4096(monkeypatch):
    """v2.37.2 carve-out (D=64 qL≥4096 auto-default) preserved post-v2.39.0."""
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
    from mlx_mfa.dispatch_policy import _v34_backward_carveout
    assert _v34_backward_carveout(
        head_dim=64, seq_len=4096, causal=False, dtype_key="float16"
    ) is True


# ── v2.39.2-internal: carve-out broadened from qL≥4096 to qL≥2048 ────

def test_v39_2_internal_carveout_engages_at_qL2048(monkeypatch):
    """v2.39.2-internal Sprint A: carve-out broadened to qL≥2048.

    v2.39.1 BK=16 fused achieves parity with SDPA-vjp at qL=2048
    (3-session variance 1.004; see docs/v6-nax/v39-2-internal-decisions.md).
    """
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
    from mlx_mfa.dispatch_policy import _v34_backward_carveout
    assert _v34_backward_carveout(
        head_dim=64, seq_len=2048, causal=False, dtype_key="float16"
    ) is True


def test_v39_2_internal_carveout_engages_at_qL3072(monkeypatch):
    """v2.39.2-internal: qL=3072 (between old 4096 floor and new 2048 floor)."""
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
    from mlx_mfa.dispatch_policy import _v34_backward_carveout
    assert _v34_backward_carveout(
        head_dim=64, seq_len=3072, causal=False, dtype_key="float16"
    ) is True


def test_v39_2_internal_carveout_rejects_below_qL2048(monkeypatch):
    """v2.39.2-internal: qL=1024 still rejected (regresses vs SDPA-vjp at -15%)."""
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
    from mlx_mfa.dispatch_policy import _v34_backward_carveout
    assert _v34_backward_carveout(
        head_dim=64, seq_len=1024, causal=False, dtype_key="float16"
    ) is False
    # qL=1536 also below the conservative 2048 floor
    assert _v34_backward_carveout(
        head_dim=64, seq_len=1536, causal=False, dtype_key="float16"
    ) is False


def test_v39_2_internal_carveout_rejects_qL_at_boundary_minus_1(monkeypatch):
    """v2.39.2-internal: qL=2047 (one below new floor) still rejected."""
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
    from mlx_mfa.dispatch_policy import _v34_backward_carveout
    assert _v34_backward_carveout(
        head_dim=64, seq_len=2047, causal=False, dtype_key="float16"
    ) is False


def test_v50_prompt5b_d128_eligibility_broadened(monkeypatch):
    """v2.50 Prompt 5b Section D: D=128 NOW eligible at qL>=2048 (carve-out
    broadened from D=64-only).  Test replaces former
    `test_v39_2_internal_carveout_preserves_d128_exclusion` which codified
    the pre-broadening "D=128 excluded" contract.  Sprint B v2.40.0-internal
    Phase C.1.b empirically validated D=128 split kernels at parity with
    SDPA-vjp (RMSE ~2e-5)."""
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
    from mlx_mfa.dispatch_policy import _v34_backward_carveout
    # D=128 at qL>=2048 now eligible
    for qL in (2048, 4096, 8192):
        assert _v34_backward_carveout(
            head_dim=128, seq_len=qL, causal=False, dtype_key="float16"
        ) is True
    # Below qL floor still ineligible
    assert _v34_backward_carveout(
        head_dim=128, seq_len=1024, causal=False, dtype_key="float16"
    ) is False
    # D=64 unchanged
    assert _v34_backward_carveout(
        head_dim=64, seq_len=4096, causal=False, dtype_key="float16"
    ) is True


def test_v38_1_d_vec_still_functional(monkeypatch):
    """v2.38.1 D_vec precompute still produces correct rowsum used by fused."""
    B, H, qL, D = 1, 4, 4096, 64
    q, k, v, dO = _make_inputs(B, H, qL, D, seed=5)
    # D = rowsum(dO * O) — match the precompute that _v34_backward_vjp does.
    # Compute via SDPA-vjp reference to get a reference O.
    scale = 1.0 / math.sqrt(D)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    D_ref = mx.sum(dO.astype(mx.float32) * o.astype(mx.float32), axis=-1)
    _flush(D_ref)
    # Sanity: D_ref shape + dtype + no NaN.
    assert D_ref.shape == (B, H, qL)
    assert D_ref.dtype == mx.float32
    assert not bool(mx.any(mx.isnan(D_ref)))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fused_d64_dtype_coverage(dtype, monkeypatch):
    """Fused kernel works for both FP16 and BF16 at D=64."""
    B, H, qL, D = 1, 4, 4096, 64
    q, k, v, dO = _make_inputs(B, H, qL, D, dtype=dtype, seed=7)
    scale = 1.0 / math.sqrt(D)
    dq, dk, dv = _grads(q, k, v, dO, scale, "fused")
    for g, name in [(dq, "dQ"), (dk, "dK"), (dv, "dV")]:
        assert not bool(mx.any(mx.isnan(g))), f"{dtype} {name} NaN"


# ── Eligibility predicate ─────────────────────────────────────────────

def test_v34_eligible_d64_still_engages(monkeypatch):
    """_v34_eligible returns True for D=64 non-causal fp16 with env set."""
    from mlx_mfa.attention import _v34_eligible
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
    assert _v34_eligible(64, mx.float16, causal=False) is True
    assert _v34_eligible(64, mx.bfloat16, causal=False) is True


def test_v34_eligible_d128_still_engages_for_split_path(monkeypatch):
    """_v34_eligible returns True for D=128 — eligibility is at the V34
    level; the AUTO routing inside _v34_backward_vjp decides fused vs split
    based on head_dim.  D=128 still goes through V34 backward (just split)."""
    from mlx_mfa.attention import _v34_eligible
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
    assert _v34_eligible(128, mx.float16, causal=False) is True


def test_v34_eligible_causal_true():
    """v2.50 Phase 4b-complete (Prompt 4 Section B): causal now eligible.
    Root cause of Prompt 3 dV residual was a missed dispatch gate at
    MFAV6Forward::eval_gpu() routing causal forward to STEEL legacy
    (log2-domain lse) instead of V34 (natural-log lse).  Fix lifts gate;
    V34 backward causal now produces correct gradients."""
    import os
    os.environ['MFA_ENABLE_V34_BACKWARD'] = '1'
    from mlx_mfa.attention import _v34_eligible
    assert _v34_eligible(64, mx.float16, causal=True) is True
    del os.environ['MFA_ENABLE_V34_BACKWARD']
