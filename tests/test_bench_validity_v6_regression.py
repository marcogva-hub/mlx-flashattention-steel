"""Regression lock for the `measured_speedup` helper — the V6-backward case that
exposed the false-"vacuous" + the bite-proofs that must survive the fix.

CONTEXT (2026-06-20): the V6-backward D=64 env-toggle bench false-raised "vacuous"
under Python 3.14 — NOT a helper bug and NOT a dispatch-decision-cache bug. Root
cause (source+runtime verified): `mlx_mfa._ext` is built only for CPython 3.11, so
under 3.14 it fails to import → `has_nax=False` → V6 never engages → BOTH arms run
SDPA → byteΔ=0 → the helper CORRECTLY flagged "same code path". The fix makes that
benign environment mismatch a precise `FeatureUnavailable` (via `require=`) instead
of a misleading "check the toggle", and clears caches between arms so no caching
layer can bleed across arms.

This lock runs in the UNIT venv (3.11, where `_ext` exists). It proves:
  1. the V6-backward D=64 env-toggle now runs CORRECTLY through the helper —
     engagement byteΔ>0, oracle-correct, and reproduces the hand-verified
     D=64 win (2.16-3.05×) within tolerance;
  2. a genuinely-identical pair STILL raises VacuousBenchmark (bite-proof);
  3. a feature-unavailable predicate raises FeatureUnavailable (the real-cause path).

Dev-only helper (`benchmarks/`, sdist-excluded); imported from repo root.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import mlx.core as mx
import pytest

import mlx_mfa

# benchmarks/ is sdist-excluded; add repo root so the dev-only helper imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.bench_validity import (  # noqa: E402
    measured_speedup,
    VacuousBenchmark,
    FeatureUnavailable,
)


def _has_nax() -> bool:
    try:
        return bool(mlx_mfa.attention._get_has_nax_cached())
    except Exception:
        return False


def _mk(N, D, seed):
    mx.random.seed(seed)
    f = lambda: (mx.random.normal((1, 4, N, D)) * 1.0).astype(mx.float16)
    q, k, v = f(), f(), f()
    mx.eval(q, k, v)
    return q, k, v


def _full_bwd(q, k, v, sc, causal):
    return mx.grad(lambda q, k, v: mlx_mfa.flash_attention(q, k, v, scale=sc, causal=causal).sum(),
                   argnums=(0, 1, 2))(q, k, v)


def _fp32_oracle(q, k, v, sc, causal):
    N = q.shape[2]
    def f(q, k, v):
        qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
        s = (qf @ kf.swapaxes(-1, -2)) * sc
        if causal:
            s = s + mx.triu(mx.full((N, N), -1e30, dtype=mx.float32), k=1)
        return (mx.softmax(s, -1) @ vf).sum()
    return mx.grad(f, argnums=(0, 1, 2))(q, k, v)


@pytest.mark.skipif(not _has_nax(), reason="V6 NAX backward requires M5+ and a loaded mlx_mfa._ext")
@pytest.mark.parametrize("causal", [False, True])
def test_v6_toggle_engages_and_reproduces_win(causal):
    """The exact case that false-raised: D=64 (≥2048) default-on (split-V6) vs
    MFA_DISABLE_V6_BACKWARD=1 (SDPA-vjp). Must run CORRECTLY through the helper —
    engaged (byteΔ>noise), oracle-correct, and a real win in the verified band."""
    N, D = 4096, 64
    sc = 1.0 / math.sqrt(D)
    qt, kt, vt = _mk(N, D, 7)
    qb, kb, vb = _mk(N, D, 7)  # distinct objects (defeat graph cache)

    def test_arm():       # split-V6 (default-on)
        os.environ.pop("MFA_DISABLE_V6_BACKWARD", None)
        return _full_bwd(qt, kt, vt, sc, causal)

    def baseline_arm():   # SDPA-vjp
        os.environ["MFA_DISABLE_V6_BACKWARD"] = "1"
        return _full_bwd(qb, kb, vb, sc, causal)

    try:
        r = measured_speedup(
            test_arm, baseline_arm,
            test_label=f"split-V6 D={D} causal={causal}", baseline_label="SDPA-vjp",
            oracle=lambda: _fp32_oracle(qt, kt, vt, sc, causal),
            oracle_tol=0.5,
            require=lambda: _has_nax(),
            warmup=4, iters=10,
        )
    finally:
        os.environ.pop("MFA_DISABLE_V6_BACKWARD", None)

    # engaged (different code paths) — the whole point.
    assert r.byte_delta > r.noise_floor, f"not engaged: byteΔ={r.byte_delta} ≤ noise {r.noise_floor}"
    # oracle-correct (different AND right).
    assert r.oracle_max_abs is not None and r.oracle_max_abs <= 0.5
    # reproduces the hand-verified D=64 win (nc ~2.16×, causal ~2.77× @qL4096).
    # Lower bound is generous (timing variance under CI load); upper bound rejects
    # an absurd ratio. The structural locks above (engaged + correct) are the
    # non-flaky guarantees; this asserts a genuine speedup was measured.
    assert 1.4 <= r.ratio <= 6.0, f"D=64 causal={causal} ratio {r.ratio:.2f} outside verified band"


def test_identical_pair_still_raises_vacuous():
    """Bite-proof: two arms on the SAME path (identical computation) must still
    RAISE — the fix must not weaken the engagement check. Uses distinct input
    objects but the SAME (no-toggle) path, so byteΔ ≈ 0."""
    mx.random.seed(0)
    a = mx.random.normal((256, 256)).astype(mx.float16)
    b = mx.random.normal((256, 256)).astype(mx.float16)
    mx.eval(a, b)
    # Both arms: identical matmul path, distinct inputs but same values → byteΔ=0.
    a2, b2 = a + 0.0, b + 0.0
    mx.eval(a2, b2)
    with pytest.raises(VacuousBenchmark):
        measured_speedup(
            lambda: a @ b, lambda: a2 @ b2,
            test_label="matmul", baseline_label="matmul (identical)",
            warmup=2, iters=4,
        )


def test_feature_unavailable_raises_clearly():
    """The REAL 3.14 cause, simulated: a `require` predicate returning False (e.g.
    `mlx_mfa._ext` didn't load → has_nax False) raises FeatureUnavailable BEFORE
    timing — distinct from VacuousBenchmark, with an actionable message."""
    with pytest.raises(FeatureUnavailable, match=r"_ext|interpreter|extension"):
        measured_speedup(
            lambda: mx.zeros((4, 4)), lambda: mx.ones((4, 4)),
            test_label="feature X", baseline_label="fallback",
            require=lambda: False, require_label="feature X (mlx_mfa._ext)",
            warmup=1, iters=2,
        )
