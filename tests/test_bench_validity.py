"""Locks the engagement-asserting bench helper's core logic:
  - a genuine pair (different outputs) → valid stamped ratio;
  - an identical pair (the V6-vs-V6 vacuous failure) → RAISES.

The helper lives in `benchmarks/` (dev-only, sdist-excluded). This test imports it
from the repo root; if `benchmarks/` isn't importable (e.g. running from an installed
wheel/sdist where it is intentionally absent), the test skips.
"""
from __future__ import annotations

import math

import mlx.core as mx
import pytest

bench_validity = pytest.importorskip(
    "benchmarks.bench_validity",
    reason="benchmarks/ is dev-only (sdist-excluded) — not importable here",
)
measured_speedup = bench_validity.measured_speedup
VacuousBenchmark = bench_validity.VacuousBenchmark
IncorrectTestArm = bench_validity.IncorrectTestArm


def _x():
    mx.random.seed(0)
    a = (mx.random.normal((256, 256)) * 0.1).astype(mx.float32)
    mx.eval(a)
    return a


def test_genuine_pair_returns_validated_ratio():
    """Two genuinely-different deterministic paths → a valid, stamped ratio."""
    x = _x()
    res = measured_speedup(
        lambda: x * 2.0,                 # test arm
        lambda: x * 3.0 + 0.0,           # baseline arm (different output)
        test_label="x*2", baseline_label="x*3",
        warmup=2, iters=5,
    )
    assert res.ratio > 0
    assert res.byte_delta > res.noise_floor          # engagement proven
    assert res.mlx_version == mx.__version__         # stamped
    assert res.hardware and res.date                 # stamped
    assert "byteΔ" in res.engagement_evidence or "trace" in res.engagement_evidence


def test_identical_pair_raises_vacuous():
    """The closure's exact failure shape: both arms the SAME path → must RAISE,
    never return ~1.0×."""
    x = _x()
    with pytest.raises(VacuousBenchmark):
        measured_speedup(
            lambda: x * 2.0,
            lambda: x * 2.0,             # identical → byteΔ == noise floor
            test_label="x*2", baseline_label="x*2(again)",
            warmup=2, iters=5,
        )


def test_different_but_incorrect_raises():
    """A test arm that is a different path but WRONG vs the fp32 oracle → RAISE
    (Lesson #11: different AND right, not different-because-broken)."""
    x = _x()
    with pytest.raises(IncorrectTestArm):
        measured_speedup(
            lambda: x * 2.0 + 999.0,     # different + wrong
            lambda: x * 3.0,
            test_label="wrong", baseline_label="x*3",
            oracle=lambda: x * 2.0,      # truth is x*2
            oracle_tol=1e-3,
            warmup=2, iters=5,
        )
