"""Engagement-asserting benchmark helper — kills the *vacuous benchmark* class.

WHY (recurred twice): the V6-backward "parity" bench toggled `MFA_ENABLE_V6_BACKWARD`
— a no-op for default-ON D=64 (gated by `MFA_DISABLE_V6_BACKWARD`) — so BOTH arms ran
V6 → ratio 1.0 mislabelled "vs SDPA-vjp parity"; the real win is 2.5–5.8×. A speedup
ratio is INVALID unless (1) the two arms are proven *genuinely different code paths*
(engagement) and (2) the arm-under-test is *oracle-correct* (Lesson #11 — different AND
right). This is the perf analogue of the test-suite's which-binary discipline
(CLAUDE.md RULE 16 #3/#8). Toggle semantics are a minefield — never assume a direction;
prove engagement empirically.

DEV/BENCH-ONLY. Lives under `benchmarks/` (sdist-excluded) — NOT shipped in the runtime
package. Import from repo root: `from benchmarks.bench_validity import measured_speedup`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

import mlx.core as mx


class VacuousBenchmark(RuntimeError):
    """Raised when the two arms are the SAME code path (byteΔ ≤ noise floor)."""


class IncorrectTestArm(RuntimeError):
    """Raised when the test arm is different from baseline but wrong vs the fp32 oracle."""


@dataclass
class SpeedupResult:
    ratio: float                       # baseline_ms / test_ms (>1 = test faster)
    test_ms: float
    baseline_ms: float
    engagement_evidence: str
    byte_delta: float
    noise_floor: float
    oracle_max_abs: Optional[float]
    mlx_version: str
    hardware: str
    date: str
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (f"{self.ratio:.2f}× (test {self.test_ms:.3f} ms / baseline "
                f"{self.baseline_ms:.3f} ms) | engagement: {self.engagement_evidence} "
                f"| MLX {self.mlx_version} | {self.hardware} | {self.date}")


def _as_list(out) -> list:
    """Normalize an arm's output (array OR tuple/list of arrays) to a flat list."""
    if isinstance(out, (tuple, list)):
        return list(out)
    return [out]


def _byte_delta(a, b) -> float:
    """Max abs elementwise diff between two arm outputs (array or sequence)."""
    la, lb = _as_list(a), _as_list(b)
    if len(la) != len(lb):
        return float("inf")
    import numpy as np
    return max(
        float(np.abs(np.array(x.astype(mx.float32)) - np.array(y.astype(mx.float32))).max())
        for x, y in zip(la, lb)
    )


def _median_ms(fn: Callable, warmup: int, iters: int) -> tuple[float, object]:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    out = None
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3, out


def _hardware() -> str:
    try:
        import mlx_mfa
        info = mlx_mfa.get_device_info()
        return str(info.get("chip_name") or info.get("device_name") or "unknown")
    except Exception:
        return "unknown"


def measured_speedup(
    test_arm: Callable[[], object],
    baseline_arm: Callable[[], object],
    *,
    test_label: str,
    baseline_label: str,
    oracle: Optional[Callable[[], object]] = None,
    oracle_tol: Optional[float] = None,
    expect_trace: Optional[dict] = None,
    warmup: int = 8,
    iters: int = 40,
    eps: float = 1e-9,
) -> SpeedupResult:
    """Run two arms, PROVE they are different code paths + the test arm is correct,
    then return a stamped validated speedup. Raises (RULE 8) on any violation.

    Each arm is a zero-arg callable that sets up its EXACT path (env, backend, …) and
    returns its output — no assumed toggle direction. `expect_trace={"test": "...",
    "baseline": "..."}` uses `mlx_mfa._dispatch_trace` telemetry when the path is
    instrumented; otherwise falls back to byteΔ (and notes the trace gap).
    """
    # 1) time + capture outputs
    test_ms, test_out = _median_ms(test_arm, warmup, iters)
    base_ms, base_out = _median_ms(baseline_arm, warmup, iters)

    # 2) engagement — preferred: dispatch-trace; fallback: byteΔ vs same-kernel noise.
    evidence = None
    trace_ok = False
    if expect_trace is not None:
        try:
            from mlx_mfa import _dispatch_trace as _dt
            with _dt.capture() as tr_t:
                mx.eval(test_arm())
            t_last = tr_t[-1][0] if tr_t else None
            with _dt.capture() as tr_b:
                mx.eval(baseline_arm())
            b_last = tr_b[-1][0] if tr_b else None
            if t_last is not None and b_last is not None:
                if t_last != expect_trace.get("test") or b_last != expect_trace.get("baseline"):
                    raise VacuousBenchmark(
                        f"trace mismatch: test arm ran {t_last!r} (expected "
                        f"{expect_trace.get('test')!r}), baseline ran {b_last!r} "
                        f"(expected {expect_trace.get('baseline')!r})")
                if t_last == b_last:
                    raise VacuousBenchmark(
                        f"vacuous bench: BOTH arms ran {t_last!r} — same code path")
                evidence = f"trace: test={t_last}, baseline={b_last}"
                trace_ok = True
            else:
                evidence = ("byteΔ (trace UNAVAILABLE — path not instrumented; "
                            "FLAG: backward/vjp terminals need _dispatch_trace.record)")
        except VacuousBenchmark:
            raise
        except Exception as e:  # telemetry import/use failed — fall back, don't hide
            evidence = f"byteΔ (trace error: {type(e).__name__}; fell back)"

    # byteΔ fallback / corroboration — calibrate the same-kernel noise floor by
    # running the test arm twice (deterministic kernels → ~0; atomics → small).
    bd = _byte_delta(test_out, base_out)
    _, test_out2 = _median_ms(test_arm, 0, 3)
    noise = _byte_delta(test_out, test_out2)
    if not trace_ok:
        if bd <= noise + eps:
            raise VacuousBenchmark(
                f"vacuous bench: byteΔ(test,baseline)={bd:.2e} ≤ same-kernel noise "
                f"floor {noise:.2e} — the two arms are the SAME code path "
                f"({test_label} vs {baseline_label}). Check the toggle direction "
                f"(ENABLE vs DISABLE, default-on vs opt-in).")
        evidence = (evidence or "byteΔ") + f" (byteΔ={bd:.2e} > noise {noise:.2e})"

    # 3) oracle correctness (Lesson #11) — different AND right.
    oracle_max = None
    if oracle is not None:
        if oracle_tol is None:
            raise ValueError("oracle supplied without oracle_tol")
        oracle_max = _byte_delta(test_out, oracle())
        if oracle_max > oracle_tol:
            raise IncorrectTestArm(
                f"test arm {test_label!r} is a different path but INCORRECT: "
                f"max|Δ| vs fp32 oracle = {oracle_max:.2e} > tol {oracle_tol:.2e}")

    # 4) stamp + 5) return
    return SpeedupResult(
        ratio=base_ms / test_ms,
        test_ms=test_ms,
        baseline_ms=base_ms,
        engagement_evidence=evidence or "byteΔ",
        byte_delta=bd,
        noise_floor=noise,
        oracle_max_abs=oracle_max,
        mlx_version=mx.__version__,
        hardware=_hardware(),
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        extra={"test_label": test_label, "baseline_label": baseline_label},
    )
