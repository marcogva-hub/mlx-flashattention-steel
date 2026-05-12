# Canonical re-bench — decisions (DC1-DC10)

## DC1 — Adopt canonical protocol for sub-1.5ms regime

**Decision**: replace §4-strict-with-warmup-during-cooldown attempts
with the canonical Apple Silicon protocol (10 warmup + 100 continuous,
ratio analysis) for sub-1.5ms kernel measurement.

**Rationale**: two REGRESSION sprints (mx.matmul v2.36.0, matched-
workload 2026-05-12) + 6-source web research convergence
(`canonical-protocol.md`) confirmed userspace warmup-during-cooldown
cannot defeat M5 Max GPU power-state variance. The canonical protocol
sidesteps the problem by never cooling down — continuous back-to-back
iterations keep the GPU in a steady state, and ratio analysis cancels
absolute timing variance.

[VERIFIED via 6 web research sources + 2 REGRESSION verdicts]. [HIGH].

## DC2 — Ratio analysis as primary verdict criterion

**Decision**: cross-session **ratio range** (V2_p50 / SDPA_p50) is the
CONFIDENT/BOUNDARY/HIGH_VARIANCE flag input. Absolute timing range is
recorded but informational only.

**Rationale**: V2 and SDPA share the same per-session GPU power-state
baseline when measured back-to-back. Cross-session, the absolute
baseline shifts but the ratio is invariant (modulo true kernel-shape
sensitivity). Per `canonical-protocol.md`, this is the canonical Apple
Silicon community pattern (Feng et al. and TristanBilot/mlx-benchmark
both use ratio-style comparisons).

[HIGH].

## DC3 — Stats: p50, p95, p99, mean, min, max

**Decision**: record all six. p50 used for verdict; others informational
for distribution shape.

**Rationale**: p95/p99 surface tail-latency that p50 hides — relevant
for production characterization (some users care about p99 not p50).
mean and min/max give distribution shape at a glance.

[HIGH].

## DC4 — 5s inter-shape settle (NOT §4 cooldown)

**Decision**: between consecutive shapes within a session, sleep 5s.
NOT a §4 cooldown (60s); just enough to let the previous shape's
output array deallocate cleanly before the next shape's input
allocation.

**Rationale**: 5s is below the < 100ms downclock threshold scale
(actually above by orders of magnitude — let me reconsider) — wait,
5s exceeds the < 100ms threshold so GPU power state will decay. But
the canonical protocol's first 10 iterations are warmup precisely
to recover from any power-state transition; the warmup absorbs the
decay. Net: 5s is the smallest sleep that ensures clean shape
boundary AND lets the warmup do its job.

Alternative considered: 0s (no sleep). Risk: input/output buffer
collisions, harder-to-debug failures. Trade-off favored 5s for
operational robustness.

[MEDIUM] confidence on the exact 5s number; protocol is robust to
2-15s range.

## DC5 — Single subprocess per session (vs §4-strict's same)

**Decision**: 3 sessions C1/C2/C3, each its own subprocess. Same as
§4-strict.

**Rationale**: cross-session range is the variance metric. Three
subprocess-isolated sessions is the minimum credible sample for range
estimation. Same convention as v2.36.0 and matched-workload sprints —
cross-protocol comparison stays clean.

[HIGH].

## DC6 — Order: V2 first, then SDPA (per shape)

**Decision**: within each shape, run V2 (10 warmup + 100 timed) first,
then SDPA (10 warmup + 100 timed). No gap between V2 and SDPA.

**Rationale**: per `canonical-protocol.md` ratio analysis, V2 and SDPA
must be measured under the same per-session power-state. Back-to-back
ordering guarantees this. Order V2-first because V2 is the variable
under test; SDPA-first would risk SDPA's specific warmup pattern
biasing V2's first iterations.

[HIGH].

## DC7 — Smoke gate per session per shape

**Decision**: before timing each shape, compute V2(Q,K,V,mask) vs
SDPA(Q,K,V,float_bias) and check RMSE < 1e-3.

**Rationale**: three-axis rule axis-1 (output sanity). Catches kernel
regressions and numerical surprises before they corrupt timing data.
Cost: ~50ms per shape per session, negligible.

[HIGH].

## DC8 — Calibration approach: empirical inflection, not pre-set

**Decision**: `_V2_DEFAULT_WORK_THRESHOLD` is determined POST-bench
from cross-session ratio range data, not prescribed pre-bench.

**Rationale**: pre-setting the threshold biases toward whatever was
expected. The empirical inflection between CONFIDENT and HIGH_VARIANCE
shapes is the natural threshold. If the inflection is sharp (e.g.,
all small shapes HIGH, all large shapes CONFIDENT), threshold lands
between them. If diffuse, set threshold conservatively (favoring
SHIP_OPT_IN over false-positive SHIP_BROAD).

[HIGH].

## DC9 — Threshold expression: qL × kL × D (work product)

**Decision**: threshold expressed as `qL * kL * D >= T` rather than
direct wall-clock estimate.

**Rationale**: work-product is independent of measurement noise and
exactly computable at dispatch time without prior timing data. Wall-
clock estimate would require either a calibration table or per-shape
profiling. Work-product captures the dominant factor in V2's
computational cost (each (q,k) pair does O(D) work).

Edge case: density also affects wall-clock (sparse skips tiles). For
v2.36.1 calibration, the 7-shape set densities span 0.01–0.24, all
significantly skipping. We accept that work-product is an upper bound
on actual work and may classify a very-sparse small shape as
"V2-eligible" when its wall-clock is actually sub-threshold. This
errs on the side of activating V2 — acceptable because V2's ratio
gains scale with density (sparser = more skip = bigger win).

[MEDIUM] — refinement to include density as a factor is a future
improvement if calibration data shows misclassification.

## DC10 — C++ binding extension: per-call `kernel_version` param

**Decision**: extend `_ext.sparse_attention_forward` to accept an
explicit `kernel_version: str` parameter that overrides the env-var-
based logic when non-empty.

**Rationale**: env-var manipulation around the Python call is thread-
unsafe AND interacts unpredictably with MLX's lazy evaluation timing.
An explicit per-call param is the clean solution. Backward compatible:
default empty string falls back to env-var-or-default logic. Existing
callers (including all tests pre-v2.36.1) work unchanged.

C++ rebuild required for v2.36.1 anyway (version bump in
`mlx_mfa/__init__.py` doesn't strictly require rebuild but the wheel
needs to be rebuilt for PyPI). Adding this param is zero marginal cost.

[HIGH].
