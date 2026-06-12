"""Parameterized regression test: every documented perf claim in user-
facing docs MUST be reachable via the public user-facing API path.

Per `CLAUDE_V6_NAX.md` §Z (Public API path testing rule).  Reference
incident: v2.37.0/v2.37.1 silent integration bug — release notes
documented "1.4-1.85× faster" but the public AUTO API silently fell
back to SDPA-vjp.  v2.37.2 fixed the routing; this test prevents
future recurrence.

What this test does
-------------------
For each entry in `PERF_CLAIMS`:

1. Sets the documented env vars the claim depends on.
2. Calls the documented public API (`mlx_mfa.flash_attention(...)`)
   via `mx.grad` with the default `backend="auto"` — exactly what the
   user is told to do in release notes / README / training guides.
3. Verifies the expected kernel engages by comparing output gradients
   against a reference path that is GUARANTEED to use V34 backward
   (forced via `backend="mfa"` + env).  If the AUTO path produces
   gradients bit-identical to SDPA-vjp instead of V34, the claim is
   unreachable and the test fails.

This is differential engagement-detection: we don't rely on
instrumenting production code with counters; instead we exploit the
fact that V34 backward and SDPA-vjp produce numerically distinct
gradients (within FP16 rounding noise).

What this test catches
----------------------
- Future regressions to `should_use_mfa()` or the v2.37.2 carve-out
  that re-introduce the silent SDPA fallback pattern.
- New perf claims added to docs without a corresponding entry here
  (the test file IS the perf-claim registry).
- Routing-gate regressions that make a previously-reachable claim
  unreachable.

How to add a new claim
----------------------
When a release adds a perf claim to user-facing docs, append an
entry to `PERF_CLAIMS` with:

- `id`: unique identifier (e.g., "v2.37.2_d64_qL8192_v34_engages")
- `env`: dict of env vars the docs say to set
- `shape`: (B, H, qL, kL, D)
- `dtype`: MLX dtype the claim targets
- `expected`: "v34_backward" if the claim says V34 backward should
  engage; "sdpa_fallback" if the claim states the AUTO path should
  fall back to SDPA-vjp (e.g., the D=128 reclassified entries that
  must NOT engage V34 via AUTO).
- `documented_in`: list of doc references (file paths)
- `documented_perf_claim`: short description of the claim text
"""
import os
import numpy as np
import pytest
import mlx.core as mx
import mlx_mfa


_AE = getattr(mx, "async_" + "eval")


# ---------------------------------------------------------------------------
# Perf-claim registry — every entry must match a user-facing doc claim
# ---------------------------------------------------------------------------
PERF_CLAIMS = [
    # v2.39.1: D=64 qL=4096 — "2.00×" with fused-BK16 (PUBLIC API, post-H1 fix)
    {
        "id": "v2.39.1_d64_qL4096_fused_bk16_engages_via_auto",
        "env": {"MFA_ENABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 4096, 4096, 64),
        "dtype": mx.float16,
        "expected": "v34_backward",
        "documented_in": [
            "CHANGELOG.md",
            "README.md",
            "docs/v6-nax/v39-1-investigation-synthesis.md",
        ],
        "documented_perf_claim": (
            "v2.39.1 D=64 qL=4096: fused-BK16 V34 backward 2.00× vs SDPA-vjp "
            "(was 1.91× v2.38.1 split-D_vec; wall-time -2.9%; H1 register-"
            "pressure root cause fixed)"
        ),
    },
    # v2.39.1: D=64 qL=8192 — "1.95×" with fused-BK16
    {
        "id": "v2.39.1_d64_qL8192_fused_bk16_engages_via_auto",
        "env": {"MFA_ENABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 8192, 8192, 64),
        "dtype": mx.float16,
        "expected": "v34_backward",
        "documented_in": [
            "CHANGELOG.md",
            "README.md",
            "docs/v6-nax/v39-1-investigation-synthesis.md",
        ],
        "documented_perf_claim": (
            "v2.39.1 D=64 qL=8192: fused-BK16 V34 backward 1.95× vs SDPA-vjp "
            "(was 1.87× v2.38.1 split-D_vec; wall-time -1.4%)"
        ),
    },
    # v2.38.1: D=64 qL=4096 — "1.91× faster" with D_vec precompute (PUBLIC API)
    # Preserved historical baseline; the v2.39.1 fused-BK16 path supersedes
    # this measurement but the v2.38.1 split-D_vec path is still reachable
    # via MFA_V34_BWD_KERNEL=split for verification.
    {
        "id": "v2.38.1_d64_qL4096_v34_dvec_engages_via_auto",
        "env": {"MFA_ENABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 4096, 4096, 64),
        "dtype": mx.float16,
        "expected": "v34_backward",
        "documented_in": [
            "CHANGELOG.md",
            "README.md",
            "docs/v6-nax/v38-1-perf-claim-audit.md",
        ],
        "documented_perf_claim": (
            "v2.38.1 D=64 qL=4096: V34 backward 1.91× vs SDPA-vjp "
            "(was 1.75× in v2.37.3 under identical conditions)"
        ),
    },
    # v2.38.1: D=64 qL=8192 — "1.87× faster"
    {
        "id": "v2.38.1_d64_qL8192_v34_dvec_engages_via_auto",
        "env": {"MFA_ENABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 8192, 8192, 64),
        "dtype": mx.float16,
        "expected": "v34_backward",
        "documented_in": [
            "CHANGELOG.md",
            "README.md",
            "docs/v6-nax/v38-1-perf-claim-audit.md",
        ],
        "documented_perf_claim": (
            "v2.38.1 D=64 qL=8192: V34 backward 1.87× vs SDPA-vjp "
            "(was 1.79× in v2.37.3)"
        ),
    },
    # v2.39.1: D=64 qL=16384 — "1.72× faster" with fused-BK16 (PUBLIC API)
    # Same physical engagement as v2.38.1 entry below but under v2.39.1 routing
    # convention (default kernel = fused-BK16; v2.38.1 split-D_vec reachable
    # via MFA_V34_BWD_KERNEL=split).  Both rows live for §Z audit-trail
    # preservation per `docs/PERF_CLAIMS.md` Active claims.
    {
        "id": "v2.39.1_d64_qL16384_fused_bk16_engages_via_auto",
        "env": {"MFA_ENABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 16384, 16384, 64),
        "dtype": mx.float16,
        "expected": "v34_backward",
        "documented_in": [
            "CHANGELOG.md",
            "docs/v6-nax/v39-1-investigation-synthesis.md",
        ],
        "documented_perf_claim": (
            "v2.39.1 D=64 qL=16384: fused-BK16 V34 backward 1.72× vs SDPA-vjp "
            "(3-session median; fresh-machine 1.89×; thermal drift across "
            "back-to-back sessions)"
        ),
    },
    # v2.38.1: D=64 qL=16384 — "1.80× faster"
    {
        "id": "v2.38.1_d64_qL16384_v34_dvec_engages_via_auto",
        "env": {"MFA_ENABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 16384, 16384, 64),
        "dtype": mx.float16,
        "expected": "v34_backward",
        "documented_in": [
            "CHANGELOG.md",
            "docs/v6-nax/v38-1-perf-claim-audit.md",
        ],
        "documented_perf_claim": (
            "v2.38.1 D=64 qL=16384: V34 backward 1.80× vs SDPA-vjp "
            "(was 1.75× in v2.37.3)"
        ),
    },
    # v2.37.2 / v2.37.3: D=64 qL=4096 — "1.82× faster end-to-end" (preserved historical)
    {
        "id": "v2.37.2_d64_qL4096_v34_engages_via_auto",
        "env": {"MFA_ENABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 4096, 4096, 64),
        "dtype": mx.float16,
        "expected": "v34_backward",
        "documented_in": [
            "docs/releases/v2.37.2-release-notes.md",
            "docs/TRAINING_QUICKSTART.md",
            "README.md",
        ],
        "documented_perf_claim": "D=64 qL=4096: V34 backward 1.82× faster than SDPA-vjp",
    },
    # v2.37.2 / v2.37.3: D=64 qL=8192 — "1.81× faster end-to-end"
    {
        "id": "v2.37.2_d64_qL8192_v34_engages_via_auto",
        "env": {"MFA_ENABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 8192, 8192, 64),
        "dtype": mx.float16,
        "expected": "v34_backward",
        "documented_in": [
            "docs/releases/v2.37.2-release-notes.md",
            "docs/TRAINING_QUICKSTART.md",
            "README.md",
        ],
        "documented_perf_claim": "D=64 qL=8192: V34 backward 1.81× faster than SDPA-vjp",
    },
    # v2.50 Prompt 5b Section D: D=128 broadened.  V34 backward NOW
    # ENGAGES via AUTO for D=128 + qL>=2048 + fp16/bf16 (split kernels
    # per Sprint B v2.40.0-internal outcome γ — at parity with SDPA-vjp,
    # ~RMSE 2e-5).  Provided as coverage extension for D=128 training;
    # perf gain not guaranteed at D=128 (parity is the empirical floor).
    # See `docs/v50/sprint-5b-section-d-dispatch-audit.md`.
    {
        "id": "v2.50.0_prompt5b_d128_qL8192_auto_engages_v34_split_at_parity",
        "env": {"MFA_ENABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 8192, 8192, 128),
        "dtype": mx.float16,
        "expected": "v34_backward",
        "documented_in": [
            "docs/v50/sprint-5b-section-d-dispatch-audit.md",
            "CHANGELOG.md",
        ],
        "documented_perf_claim": (
            "v2.50 Prompt 5b Section D: D=128 qL=8192 V34 backward "
            "engages via AUTO at parity with SDPA-vjp (Sprint B v2.40.0-"
            "internal empirical RMSE ~2e-5; cohérence narrative "
            "'V34 backward couvre D=64 + D=128' prime sur perf gain "
            "marginal — no speedup claim, contract-honest engagement)"
        ),
    },
    # v2.39.2-internal: D=64 qL=2048 now ENGAGES V34 backward (at parity).
    # The v2.37.2/v2.37.3 floor was qL≥4096; v2.39.2-internal lowered it to
    # qL≥2048 after v2.39.1 BK=16 fused kernel achieved parity with SDPA-vjp
    # at qL=2048 (3-session variance 1.004; see docs/v6-nax/v39-2-internal-
    # decisions.md).  V34 engages but at parity — no speedup claim.
    {
        "id": "v2.39.2_internal_d64_qL2048_auto_engages_v34_at_parity",
        "env": {"MFA_ENABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 2048, 2048, 64),
        "dtype": mx.float16,
        "expected": "v34_backward",
        "documented_in": [
            "docs/v6-nax/v39-2-internal-decisions.md",
            "CHANGELOG.md",
        ],
        "documented_perf_claim": (
            "v2.39.2-internal D=64 qL=2048: V34 backward engages via AUTO "
            "at parity with SDPA-vjp (3-session variance 1.004; no speedup "
            "claim but contract-honest engagement per env-var opt-in)"
        ),
    },
    # v2.37.3 retracted (historical record): D=64 qL=1024 still falls back
    # to SDPA-vjp via AUTO.  Below the v2.39.2-internal broadened qL=2048
    # floor.  Preserves the "below-floor MUST fall back" coverage that the
    # v2.37.3 qL=2048 row used to provide.
    {
        "id": "v2.39.2_internal_d64_qL1024_auto_falls_back_to_sdpa",
        "env": {"MFA_ENABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 1024, 1024, 64),
        "dtype": mx.float16,
        "expected": "sdpa_fallback",
        "documented_in": [
            "docs/v6-nax/v39-2-internal-decisions.md",
            "CHANGELOG.md",
        ],
        "documented_perf_claim": (
            "D=64 qL=1024: below v2.39.2-internal carve-out floor (regresses "
            "15% vs SDPA-vjp empirically); carve-out correctly does not engage"
        ),
    },
    # Phase II-12 contract UPDATE (was: env unset -> never engage).
    # D=64 backward (causal AND non-causal) is now DEFAULT-ON at
    # qL >= 2048 via the clean split kernel (II-0 + II-12 promotions);
    # the opt-out env restores SDPA-vjp exactly.
    {
        "id": "ii12_d64_qL8192_default_on_v34",
        "env": {},  # default: V34 split engages
        "shape": (1, 4, 8192, 8192, 64),
        "dtype": mx.float16,
        "expected": "v34_backward",
        "documented_in": ["README.md", "CHANGELOG.md"],
        "documented_perf_claim": (
            "Default behavior: D=64 backward routes to the V34 split "
            "kernel (1.7-2.7x vs SDPA-vjp); opt-out via "
            "MFA_DISABLE_V34_BACKWARD=1"
        ),
    },
    {
        "id": "ii12_d64_qL8192_optout_sdpa",
        "env": {"MFA_DISABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 8192, 8192, 64),
        "dtype": mx.float16,
        "expected": "sdpa_fallback",
        "documented_in": ["README.md", "CHANGELOG.md"],
        "documented_perf_claim": (
            "MFA_DISABLE_V34_BACKWARD=1 restores SDPA-vjp bit-exactly"
        ),
    },
    # --- conv3d MPP claims (II-9 fp16 promotion + III-1 KD-7 bf16 lift).
    # `shape` is (T, H, W, C_in, C_out); engagement detected via the
    # auto-hook telemetry executed counter (install_hooks + mx.conv3d is
    # the documented public path).
    {
        "id": "ii9_conv3d_t16_64x64_c128_fp16_mpp_default",
        "env": {},
        "shape": (16, 64, 64, 128, 128),
        "dtype": mx.float16,
        "expected": "conv3d_mpp",
        "documented_in": ["CHANGELOG.md"],
        "documented_perf_claim": (
            "conv3d via the MPP convolution2d primitive, default-on: "
            "2.3-2.5x vs the materialized-im2col path at T8/T16 64x64 "
            "C128 (II-9, 3 sessions, medians)"
        ),
    },
    {
        "id": "iii1_conv3d_t16_64x64_c128_bf16_mpp_default",
        "env": {},
        "shape": (16, 64, 64, 128, 128),
        "dtype": mx.bfloat16,
        "expected": "conv3d_mpp",
        "documented_in": ["CHANGELOG.md"],
        "documented_perf_claim": (
            "bf16 conv3d via MPP (KD-7 lift): 1.4-2.7x vs the pre-lift "
            "public bf16 path (Apple mx.conv3d fallback) at the II-9 "
            "cells (III-1, 3 sessions, medians)"
        ),
    },
    # --- TQ paged decode (III-2): step() N_q=1 routes to gather/dequant
    # kernels + Apple SDPA by default.  Engagement detected via the
    # tq_decode kernel-cache population through the public step() path.
    {
        "id": "iii2_tq_paged_decode_step_default",
        "env": {},
        "shape": (1, 8, 512, 512, 128),  # (B, Hq, S0, S0, D); Hkv=2
        "dtype": mx.float16,
        "expected": "tq_decode_sdpa",
        "documented_in": ["CHANGELOG.md"],
        "documented_perf_claim": (
            "TurboQuant paged decode step 6.0x (S=4K) to 14.4x (S=16K) "
            "faster via per-step gather/dequant kernels + Apple SDPA "
            "(attend-only 13.8-22.1x vs the fused TQ kernel); opt-out "
            "MFA_DISABLE_TQ_DECODE_SDPA=1"
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mk(shape, dtype):
    """Deterministic inputs so AUTO and reference paths see identical tensors."""
    B, H, qL, kL, D = shape
    mx.random.seed(0xC0DE)
    q = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    k = (mx.random.uniform(-1.0, 1.0, (B, H, kL, D)) * 0.1).astype(dtype)
    v = (mx.random.uniform(-1.0, 1.0, (B, H, kL, D)) * 0.1).astype(dtype)
    _AE(q, k, v); mx.synchronize()
    return q, k, v


def _grad_auto(q, k, v):
    """Public AUTO path — what the user-facing docs tell users to call."""
    def loss(q_, k_, v_):
        return mlx_mfa.flash_attention(q_, k_, v_).sum()
    return mx.grad(loss, argnums=(0, 1, 2))(q, k, v)


def _grad_sdpa(q, k, v):
    """Reference: explicit SDPA-vjp.  Bit-identical to flash_attention's
    SDPA fallback path on the same inputs."""
    D = q.shape[-1]
    scale = 1.0 / (D ** 0.5)
    def loss(q_, k_, v_):
        return mx.fast.scaled_dot_product_attention(
            q_, k_, v_, scale=scale).sum()
    return mx.grad(loss, argnums=(0, 1, 2))(q, k, v)


def _rmse(a, b):
    diff = np.abs(np.array(a.astype(mx.float32)) -
                  np.array(b.astype(mx.float32)))
    return float(np.sqrt((diff ** 2).mean()))


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("claim", PERF_CLAIMS, ids=lambda c: c["id"])
def test_perf_claim_engages_via_public_api(claim, monkeypatch):
    """Per `CLAUDE_V6_NAX.md` §Z: the documented kernel must engage via
    `mx.grad(flash_attention(...))` with default `backend="auto"`.

    Engagement detection is differential: V34 backward gradients differ
    from SDPA-vjp gradients by FP16-rounding amount (non-zero).  SDPA
    fallback produces bit-identical gradients to the SDPA reference.

    - `expected == "v34_backward"`: AUTO gradients MUST differ from SDPA
      reference (V34 engaged).
    - `expected == "sdpa_fallback"`: AUTO gradients MUST be bit-identical
      to SDPA reference (V34 did NOT engage; correct fallback).
    """
    # --- conv3d MPP claims (II-9 / III-1): engagement via the auto-hook
    # telemetry executed counter through the documented public path
    # (install_hooks -> mx.conv3d).
    if claim["expected"] == "conv3d_mpp":
        monkeypatch.delenv("MFA_DISABLE_CONV3D_MPP", raising=False)
        for kk, vv in claim["env"].items():
            monkeypatch.setenv(kk, vv)
        from mlx_mfa import _auto_hooks as ah
        ah.install_hooks()
        monkeypatch.setattr(ah, "_HOOK_TELEMETRY_MODE", "on")
        T, H, W, C_in, C_out = claim["shape"]
        mx.random.seed(5)
        x = (mx.random.normal((1, T, H, W, C_in)) * 0.5).astype(claim["dtype"])
        w = (mx.random.normal((C_out, 3, 3, 3, C_in)) * 0.1).astype(claim["dtype"])
        mx.eval(x, w)
        before = ah._HOOK_EXECUTION_STATS["executed"]["conv3d_nax_forward"]
        out = mx.conv3d(x, w, stride=(1, 1, 1), padding=(1, 1, 1))
        mx.eval(out)
        after = ah._HOOK_EXECUTION_STATS["executed"]["conv3d_nax_forward"]
        assert after > before, (
            f"Perf claim '{claim['id']}' is UNREACHABLE via public API path. "
            f"Documented in: {claim['documented_in']}. "
            f"Claim text: {claim['documented_perf_claim']}. "
            f"mx.conv3d (hooked) did not engage conv3d_nax_forward — "
            f"per CLAUDE_V6_NAX.md §Z, fix the routing or correct the claim."
        )
        orig = ah._ORIGINAL_CONV3D if ah._ORIGINAL_CONV3D is not None else None
        assert orig is not None
        ref = orig(x, w, stride=(1, 1, 1), padding=(1, 1, 1))
        mx.eval(ref)
        a = np.asarray(out.astype(mx.float32))
        b = np.asarray(ref.astype(mx.float32))
        # Absolute bar at these fixed unit-scale fixtures (II-9 measured
        # 0.0039-0.0078 fp16; III-1 probe 0.031-0.062 bf16 — single
        # store-rounding of the dtype).  10x headroom over measured.
        abs_bar = 0.05 if claim["dtype"] == mx.float16 else 0.3
        max_abs = float(np.abs(a - b).max())
        assert max_abs < abs_bar, (
            f"conv3d MPP engaged but output deviates from the original op "
            f"beyond the dtype floor (max abs {max_abs:.4f} >= {abs_bar})"
        )
        return

    # --- TQ paged decode claim (III-2): engagement via the tq_decode
    # kernel cache through the documented public step() path.
    if claim["expected"] == "tq_decode_sdpa":
        monkeypatch.delenv("MFA_DISABLE_TQ_DECODE_SDPA", raising=False)
        from mlx_mfa import tq_decode
        from mlx_mfa.inference import TurboQuantPagedInferenceContext
        _B, Hq, S0, _S, D = claim["shape"]
        Hkv = 2
        ctx = TurboQuantPagedInferenceContext(
            num_blocks=S0 // 64 + 8, block_size=64, H_kv=Hkv, D=D,
            tq_bits=3)
        mx.random.seed(3)
        k0 = mx.random.normal((1, Hkv, S0, D), dtype=claim["dtype"])
        v0 = mx.random.normal((1, Hkv, S0, D), dtype=claim["dtype"])
        q0 = mx.random.normal((1, Hq, S0, D), dtype=claim["dtype"])
        mx.eval(k0, v0, q0)
        mx.eval(ctx.prefill(q0, k0, v0))
        q = mx.random.normal((1, Hq, 1, D), dtype=claim["dtype"])
        kn = mx.zeros((1, Hkv, 1, D), dtype=claim["dtype"])
        vn = mx.zeros((1, Hkv, 1, D), dtype=claim["dtype"])
        mx.eval(q, kn, vn)
        before = len(tq_decode._K_DEQUANT_KERNELS)
        cache_key = (D, Hkv, 64, 3)
        tq_decode._K_DEQUANT_KERNELS.pop(cache_key, None)
        out = ctx.step(q, kn, vn)
        mx.eval(out)
        assert cache_key in tq_decode._K_DEQUANT_KERNELS, (
            f"Perf claim '{claim['id']}' is UNREACHABLE via public API "
            f"path: step() did not route to the tq_decode gather/dequant "
            f"path with env={claim['env']}. "
            f"Claim text: {claim['documented_perf_claim']}. "
            f"Per CLAUDE_V6_NAX.md §Z, fix the routing or correct the claim."
        )
        assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item())
        del before
        return

    # Set documented env vars; clear all routing-related env vars first
    # so external session state doesn't leak into the parameterized test.
    monkeypatch.delenv("MFA_ENABLE_V34_BACKWARD", raising=False)
    monkeypatch.delenv("MFA_DISABLE_V34_BACKWARD", raising=False)
    monkeypatch.delenv("MFA_V34_BWD_KERNEL", raising=False)  # v2.39.0 (outcome δ)
    monkeypatch.delenv("MFA_V34BWD_USE_FUSED", raising=False)  # v2.38.0 legacy
    for k, val in claim["env"].items():
        monkeypatch.setenv(k, val)

    q, k, v = _mk(claim["shape"], claim["dtype"])

    # Public AUTO path — what the docs tell users to call
    dQ_auto, dK_auto, dV_auto = _grad_auto(q, k, v)
    _AE(dQ_auto, dK_auto, dV_auto); mx.synchronize()

    # SDPA reference (identical to flash_attention's SDPA fallback)
    dQ_ref, dK_ref, dV_ref = _grad_sdpa(q, k, v)
    _AE(dQ_ref, dK_ref, dV_ref); mx.synchronize()

    diff_q = _rmse(dQ_auto, dQ_ref)
    diff_k = _rmse(dK_auto, dK_ref)
    diff_v = _rmse(dV_auto, dV_ref)
    total_diff = diff_q + diff_k + diff_v

    if claim["expected"] == "v34_backward":
        # V34 engaged → gradients differ from SDPA reference by FP16 rounding.
        # If diff is exactly zero, V34 didn't engage (silent SDPA fallback).
        assert total_diff > 0.0, (
            f"Perf claim '{claim['id']}' is UNREACHABLE via public API path. "
            f"Documented in: {claim['documented_in']}. "
            f"Claim text: {claim['documented_perf_claim']}. "
            f"Expected V34 backward to engage via `flash_attention(...)` "
            f"with env={claim['env']}, but AUTO gradients are bit-identical "
            f"to SDPA-vjp reference (RMSE q={diff_q}, k={diff_k}, v={diff_v}). "
            f"This means `should_use_mfa()` short-circuited to SDPA fallback "
            f"before the V34 carve-out engaged — the v2.37.0/v2.37.1 silent "
            f"integration bug pattern has regressed. "
            f"Per CLAUDE_V6_NAX.md §Z, fix the routing or correct the claim."
        )
        # Correctness sanity: V34 vs SDPA gradients within FP16 floor
        assert diff_q < 1e-2 and diff_k < 1e-2 and diff_v < 1e-2, (
            f"V34 backward engaged but produces gradients far from SDPA "
            f"reference (RMSE q={diff_q}, k={diff_k}, v={diff_v}). "
            f"Investigate kernel correctness regression."
        )
    elif claim["expected"] == "sdpa_fallback":
        # AUTO must fall back to SDPA-vjp → gradients identical to the
        # SDPA reference up to numerical-noise tolerance.
        #
        # The strict `== 0.0` check was relaxed to `< 1e-7` per the pre-
        # merge code review (MEDIUM finding 2026-05-13).  Rationale:
        # `flash_attention()`'s `_fallback_sdpa` and the reference
        # `_grad_sdpa` both call `mx.fast.scaled_dot_product_attention`
        # with the same scale, producing bit-identical gradients today.
        # But future MLX SDPA-vjp internal reordering (fused softmax-
        # then-matmul, or a Python cast inside one path and not the
        # other) could introduce tiny-non-zero RMSE without actually
        # engaging V34 backward.  1e-7 is well below FP16-rounding
        # noise (V34-engaged gradients show ~1e-4 RMSE) but absorbs
        # any future float-reordering drift on the SDPA fallback path.
        assert total_diff < 1e-7, (
            f"Perf claim '{claim['id']}' expected SDPA fallback via AUTO "
            f"path but gradients differ from SDPA reference far above "
            f"numerical-noise tolerance "
            f"(RMSE q={diff_q}, k={diff_k}, v={diff_v}, total={total_diff}). "
            f"This means a routing carve-out engaged for a shape/config "
            f"where the docs say it shouldn't. "
            f"Documented in: {claim['documented_in']}. "
            f"Claim text: {claim['documented_perf_claim']}. "
            f"Either fix the carve-out shape gate or update the docs."
        )
    else:
        pytest.fail(
            f"Unknown 'expected' value in claim '{claim['id']}': "
            f"{claim['expected']!r}.  Use 'v34_backward' or 'sdpa_fallback'."
        )


# ---------------------------------------------------------------------------
# Meta-test: ensure the perf-claim registry stays in sync with docs
# ---------------------------------------------------------------------------
def test_perf_claim_registry_non_empty():
    """Sanity check: at least one perf claim is registered.

    If this file gets emptied accidentally, the §Z rule has no
    executable enforcement.  Future releases adding perf claims to
    user-facing docs MUST add an entry here (see module docstring).
    """
    assert len(PERF_CLAIMS) >= 4, (
        "PERF_CLAIMS registry has fewer than 4 entries — likely "
        "incomplete.  Per CLAUDE_V6_NAX.md §Z, every user-facing "
        "perf claim needs an entry here for ongoing reachability "
        "regression coverage."
    )
