# M5 Max — GPU Counter Availability via Public Metal API

**Date:** 2026-05-04
**Hardware:** Apple M5 Max (40 GPU cores, applegpu_g17s)
**Method:** `bench/v6_counter_discovery.mm` enumerates `MTLDevice.counterSets`

---

## TL;DR — Apple exposes only timestamps

The public `MTLCounterSet` API on M5 Max exposes **exactly ONE counter
set with ONE counter**:

```
Counter Set #0: timestamp (1 counters)
  [0] GPUTimestamp

Sample buffer support: OK (storage mode shared and private)
```

**ALU active %, occupancy, register spill, memory limiter, stall counters,
and NAX-specific counters are NOT exposed via the public Metal API.** They
are accessible only through Xcode's Metal Debugger GUI (which itself
crashes WindowServer when replaying V6 NAX traces — verified 2026-05-04).

---

## Discovery script + raw output

`bench/v6_counter_discovery.mm`:

```objc
NSArray<id<MTLCounterSet>>* counterSets = device.counterSets;
for (id<MTLCounterSet> set in counterSets) {
  for (id<MTLCounter> counter in set.counters) {
    // ...
  }
}
```

Run on M5 Max (build with `clang++ -fobjc-arc -framework Metal ...`):

```
Device: Apple M5 Max
supportsFamily(Apple9): 1
maxThreadgroupMemoryLength: 32768 B (32.0 KB)

Counter sets exposed by device: 1
  Counter Set #0: timestamp (1 counters)
    [0] GPUTimestamp

Sample buffer support test:
  timestamp [shared]: OK
  timestamp [private]: OK
```

---

## What this means for V6 NAX profiling

### What we CAN do

- **Measure precise per-kernel GPU time** via `MTLCounterSampleBuffer`
  with timestamp counters. This is more accurate than CPU-side
  `time.perf_counter()` measurement because it excludes CPU dispatch
  overhead and waiting time.
- **Compare V6 vs SDPA kernel-only GPU time** — verifies the kernel-only
  ratio (1.22-1.34× from CPU-side profiling) at GPU-timeline precision.
- **Measure inter-kernel idle time** in a multi-dispatch sequence by
  capturing timestamps between encoders (relevant for chunked-K and
  multi-pass workloads).

### What we CANNOT do (without Xcode)

- ALU utilization (% of theoretical TFLOPS)
- Threadgroup occupancy (% of max)
- Register pressure / spill events
- Memory limiter %
- Stall reason breakdown
- NAX-specific cycles or utilization

These are exactly the counters needed to validate Hypothesis 1 (MPP
abstraction-layer overhead) and Hypothesis 2 (register spill) from
`v6-vs-sdpa-profiling-analysis.md`.

---

## Plan B — decision based on accumulated evidence

Per the user's contingency plan:

> "Si après l'étape 1 il s'avère que les counters d'intérêt ne sont pas
> exposés par MTLCounterSampleBuffer sur le M5 Max [...] on bascule sur
> Plan B : décider sur la base des hypothèses empiriques accumulées."

The accumulated evidence (across Phase 0-3 of the V6 NAX work):

1. **Coverage validated** (`v6-tile-coverage-results-v2.md`): V6 writes
   every output cell at FP32-accumulator precision. No silent corruption.
2. **Tile-tuning exhausted** (`optimization-campaign-report.md`): 245
   configs swept, 10 axes exercised. The dispatch table is at the
   per-axis optimum.
3. **CPU-side profiling** (`v6-vs-sdpa-profiling-analysis.md`):
   transposes contributed 0.7-12% of full-time; kernel-only V6/SDPA
   = 1.22-1.34×. After BHND elimination of transposes, V6 kernel-only
   is still 22-34% slower than SDPA.
4. **Architectural reading** (`apple-sdpa-nax-analysis.md`): Apple
   uses `metal_simdgroup_matrix` (low-level), V6 uses
   `mpp::tensor_ops::matmul2d` (one abstraction layer up).
5. **BHND migration** (`bhnd-migration-report.md`): unconditional 4×
   memory savings + 2.5-15.2% time gains across shapes. Closes much
   of the SDPA gap (FlashVSR 1.52× → 1.13×).

**Plan B decision matrix**:

| Hypothesis | Status from evidence | Action |
|------------|---------------------|--------|
| MPP abstraction overhead | LIKELY (consistent with all observations) | Sprint 3: simdgroup_matrix rewrite, expected gain 5-15% |
| Register spill | UNVERIFIED but consistent with axis 2 (BLOCK_D) regression pattern | Address as part of Sprint 3 if needed |
| Suboptimal tile size | INVALIDATED (245-config sweep converged) | None |
| Bandwidth-bound | INVALIDATED (roofline AI > 1500) | None |

**Recommended sequencing**:
1. ✅ BHND migration — DONE (Sprint 2A, commit 346479e)
2. Sprint 2B (Chunked-K) — being prototyped now to test if kernel-side
   streaming benefit is captured at the wrapper level
3. Sprint 3: simdgroup_matrix rewrite — committed conditionally on
   2-3 weeks budget. The expected upside (5-15%) is meaningful but
   the development cost is substantial. Consider deferring until M6
   hardware exposes more counters or until other workloads benefit
   from the rewrite (e.g., V6 backward pass for training).

---

## Timing-only profiler (still useful)

Even without ALU/occupancy/spill counters, an `MTLCounterSampleBuffer`
with timestamp counters provides:

- Sub-microsecond GPU-timeline timing (vs CPU-side `perf_counter` which
  includes encode/dispatch overhead)
- Per-encoder accurate timing for multi-dispatch sequences
- Removes CPU-GPU sync overhead from measurements

This is useful infrastructure but limited compared to what was hoped.
Implementation deferred unless concrete profiling needs justify ~1
day of binding work.

---

## Reusable artifact

`bench/v6_counter_discovery.mm` — runs anytime to verify available
counters on the current macOS / Metal version. If Apple ships more
counters in a future macOS update, this script will detect them.
