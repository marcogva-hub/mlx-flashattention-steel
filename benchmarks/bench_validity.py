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


class FeatureUnavailable(RuntimeError):
    """Raised when the feature-under-test cannot engage in THIS interpreter.

    The canonical case (and the actual root cause of the 3.14 V6-backward
    false-"vacuous"): the compiled extension `mlx_mfa._ext` is built for one
    CPython ABI (e.g. 3.11) and the bench runs under another (3.14) where the
    import fails → `has_nax=False` → the NAX path silently never engages →
    BOTH arms fall to SDPA → byteΔ=0. That is NOT a toggle-direction bug and
    NOT a cache bug — the feature simply isn't present. Surfacing it as its
    own error (vs a misleading "check the toggle") is the fix that prevents
    re-debugging a benign environment mismatch as a code defect.
    """


def _default_reset_caches() -> None:
    """Best-effort: clear mlx-mfa's dispatch decision cache + lru_cache factories
    so each arm re-decides its dispatch path from a clean slate (defeats any
    cross-arm cache bleed — present or future). Cheap: called once per arm setup,
    NOT per timed iteration. No-op if mlx_mfa isn't importable.

    NOTE (source-verified 2026-06-20): for the V6-backward knobs this is
    belt-and-suspenders — `should_use_mfa` reads no env, the 6 forward-affecting
    vars ARE in the decision-cache key, and `MFA_DISABLE/ENABLE_V6_BACKWARD` are
    read LIVE by the (uncached) carve-out + the vjp's `_v6nax_eligible`. It is
    kept so the helper is a trustworthy GENERAL toggle-bench: an arm that toggles
    an env var NOT in the decision-cache key (or a future one) cannot silently
    reuse the other arm's cached decision.
    """
    try:
        from mlx_mfa import attention as _A
    except Exception:
        return
    try:
        _A._dispatch_decision_cache.clear()
    except Exception:
        pass
    for _name in dir(_A):
        _obj = getattr(_A, _name, None)
        _cc = getattr(_obj, "cache_clear", None)
        if callable(_cc):
            try:
                _cc()
            except Exception:
                pass


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
    require: Optional[Callable[[], bool]] = None,
    require_label: str = "",
    reset_caches: Optional[Callable[[], None]] = _default_reset_caches,
    warmup: int = 8,
    iters: int = 40,
    eps: float = 1e-9,
) -> SpeedupResult:
    """Run two arms, PROVE they are different code paths + the test arm is correct,
    then return a stamped validated speedup. Raises (RULE 8) on any violation.

    Each arm is a zero-arg callable that sets up its EXACT path (env, backend, …) and
    returns its output — no assumed toggle direction. **Each arm MUST own DISTINCT
    input objects** (defeat MLX's input-identity graph cache) — the helper times each
    arm's intended path at steady state, but cannot create the arms' inputs for them.

    `expect_trace={"test": "...", "baseline": "..."}` uses `mlx_mfa._dispatch_trace`
    telemetry when the path is instrumented; otherwise falls back to byteΔ.

    `require` (optional): a zero-arg predicate asserting the feature-under-test can
    engage in THIS interpreter (e.g. `lambda: mlx_mfa.attention._get_has_nax_cached()`).
    If it returns False the helper raises `FeatureUnavailable` BEFORE timing — turning
    the silent "both arms fell to the fallback → byteΔ=0 → looks vacuous" trap (the
    3.14 missing-`_ext` case) into a precise, actionable error.

    `reset_caches` (default: clear mlx-mfa dispatch/decision/lru caches): invoked once
    before each arm so neither arm can reuse the other's cached dispatch decision. Pass
    `lambda: None` to disable.
    """
    # 0) precondition — is the feature even present in this interpreter? Surfaced as
    # its own error so a benign environment mismatch (wrong-ABI extension → has_nax
    # False → both arms hit the fallback) is never re-debugged as a code/cache defect.
    if require is not None and not require():
        raise FeatureUnavailable(
            f"feature-under-test {require_label or test_label!r} cannot engage in this "
            f"interpreter — its `require` predicate returned False. Most common cause: "
            f"the compiled extension `mlx_mfa._ext` did not import (ABI mismatch — e.g. "
            f"a 3.11-built `_ext` under Python "
            f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}). "
            f"Check `python -c 'import mlx_mfa._ext'` and run in the venv whose Python "
            f"matches the built extension. (NOT a toggle-direction or cache bug.)")

    def _reset():
        if reset_caches is not None:
            reset_caches()

    # 1) time + capture outputs — reset caches before each arm so each re-decides its
    # own dispatch path from a clean slate (no cross-arm cache bleed).
    _reset()
    test_ms, test_out = _median_ms(test_arm, warmup, iters)
    _reset()
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
    _reset()
    _, test_out2 = _median_ms(test_arm, 0, 3)
    noise = _byte_delta(test_out, test_out2)
    if not trace_ok:
        if bd <= noise + eps:
            raise VacuousBenchmark(
                f"vacuous bench: byteΔ(test,baseline)={bd:.2e} ≤ same-kernel noise "
                f"floor {noise:.2e} — the two arms are the SAME code path "
                f"({test_label} vs {baseline_label}). TWO causes to check, in order: "
                f"(1) the feature-under-test is UNAVAILABLE in this interpreter "
                f"(both arms fell to the same fallback) — verify the compiled "
                f"extension loaded (`import mlx_mfa._ext`; check has_nax) and pass "
                f"`require=` to assert it up front; (2) wrong toggle DIRECTION "
                f"(ENABLE vs DISABLE, default-on vs opt-in) — both arms set the same "
                f"path. (1) is the silent one — the 3.14 missing-`_ext` trap.")
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
