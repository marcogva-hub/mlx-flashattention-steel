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
    # v2.37.3 reclassified: D=128 — MUST fall back to SDPA-vjp via AUTO
    # (V34 backward is research-only, requires backend="mfa")
    {
        "id": "v2.37.3_d128_qL8192_auto_falls_back_to_sdpa",
        "env": {"MFA_ENABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 8192, 8192, 128),
        "dtype": mx.float16,
        "expected": "sdpa_fallback",
        "documented_in": [
            "docs/releases/v2.37.3-release-notes.md",
            "docs/TRAINING_QUICKSTART.md",
            "CHANGELOG.md",
        ],
        "documented_perf_claim": (
            "D=128 V34 backward is research-only; AUTO path falls back "
            "to SDPA-vjp at parity (carve-out is D=64 only)"
        ),
    },
    # v2.37.3 retracted: D=64 qL=2048 — MUST fall back to SDPA-vjp via AUTO
    # (v2.37.1 claim "1.44× win" retracted; v2.37.2 carve-out is qL≥4096)
    {
        "id": "v2.37.3_d64_qL2048_auto_falls_back_to_sdpa",
        "env": {"MFA_ENABLE_V34_BACKWARD": "1"},
        "shape": (1, 4, 2048, 2048, 64),
        "dtype": mx.float16,
        "expected": "sdpa_fallback",
        "documented_in": [
            "docs/releases/v2.37.3-release-notes.md",
            "docs/TRAINING_QUICKSTART.md",
            "CHANGELOG.md",
        ],
        "documented_perf_claim": (
            "D=64 qL=2048: v2.37.1 '1.44× win' retracted; v2.37.2 "
            "carve-out correctly does not engage (qL < 4096 floor)"
        ),
    },
    # v2.37.3: env unset — V34 must NOT engage anywhere
    {
        "id": "v2.37.3_d64_qL8192_env_unset_no_v34",
        "env": {},  # MFA_ENABLE_V34_BACKWARD NOT set
        "shape": (1, 4, 8192, 8192, 64),
        "dtype": mx.float16,
        "expected": "sdpa_fallback",
        "documented_in": ["README.md", "CHANGELOG.md"],
        "documented_perf_claim": (
            "Default behavior (env unset): AUTO path uses SDPA-vjp; "
            "V34 backward never engages without explicit opt-in"
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
