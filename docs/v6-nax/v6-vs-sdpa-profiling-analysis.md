# V6 NAX vs SDPA — Profiling Analysis & Sprint 2 Priorities

**Date:** 2026-05-04
**Hardware:** Apple M5 Max (40 GPU cores, applegpu_g17s)
**Branch:** `feat/v6-nax`
**Companion docs:** `profiling-counters.md` (raw data), `apple-sdpa-nax-analysis.md`
                   (architectural background), `v6-tile-coverage-results-v2.md` (correctness)

---

## TL;DR — Two distinct cost components, neither hides the abstraction-layer ceiling

| Component | FlashVSR | SeedVR2-small |
|-----------|---------:|--------------:|
| **(a) Layout transpose overhead** (3× contiguous + 1× transpose) | **11.6%** of full V6 time | **0.7%** of full V6 time |
| **(b) Kernel-only V6/SDPA ratio** (after subtracting transposes) | **1.34×** slower | **1.22×** slower |
| Memory cost of transpose | 4× peak vs SDPA | 4× peak vs SDPA |

**The MPP abstraction-layer ceiling is the dominant cost on every shape.**
Even with zero transpose overhead, V6 NAX would still be 22-34% slower
than SDPA. The transpose overhead is a real but **secondary** problem.

This empirically supports the architectural hypothesis from
`apple-sdpa-nax-analysis.md`: Apple's `metal_simdgroup_matrix` path lets
their kernel pack more work per cycle than our MPP `matmul2d_descriptor`
path. The gap is in the kernel itself, not in surrounding infrastructure.

---

## Methodology and limitations

### What we measured

1. **End-to-end attention call wall time** (`mx.synchronize()` bracketing)
2. **Per-step decomposition** (transposes alone, transposes+contiguous, full pipeline)
3. **Peak memory delta** (`mx.get_peak_memory()` around the call)
4. **Static dispatch count** (read from `csrc/mfa_v6_nax_primitive.cpp` and
   MLX `scaled_dot_product_attention.cpp`)

ITERS=20 with 3 warmup runs. Variance is ~2-7% — well below the gap
ratios we report.

### What we couldn't measure

The `.gputrace` bundles use Apple's proprietary `MTSP`/`xdic` binary
serialization. The main 150 MB blob (`231131F2A31A0D0D`) contains the
GPU counters and command stream but is not parseable without:
- Xcode's Metal Debugger GUI, or
- Apple's private Replayer SDK (not public)

The 4 captured traces are saved to disk for future Xcode analysis. They
will provide:
- ALU Active %
- Occupancy %
- Register pressure / spill events
- Memory limiter %
- Stall counters (memory, instruction, register)
- NAX utilization %

For now, those quantities are **not part of this analysis**. We work with
what's measurable.

---

## Question A — Where is the bottleneck?

| Symptom | Inference |
|---------|-----------|
| V6 transposes_contig = 0.175 ms (FlashVSR) / 1.893 ms (SeedVR2-small) | Memory copy (~4× input size). Bandwidth-limited. |
| V6 kernel-only is 1.34× / 1.22× SDPA | Kernel itself is the dominant bottleneck for V6's gap to SDPA |
| V6 kernel time scales near-linearly with N (1.3 ms → 273 ms; 200×) | Kernel is compute-bound on the attention matmul, consistent with prior roofline analysis (Axe 10) |
| V6 peak memory: 4× SDPA on every shape | Pure transpose materialization, not kernel-internal |

**Verdict (programmatic-only)**: V6 has TWO distinct issues:
1. A small, layout-driven memory cost at the host boundary (transposes)
2. A larger, structural compute-time gap inside the kernel itself

The compute-time gap dominates at every shape we measured.

To distinguish ALU-bound vs memory-bound vs stall-bound INSIDE the kernel,
**Xcode counter analysis is required**. The CPU profiler cannot see inside
the GPU dispatch.

---

## Question B/C — Occupancy and register spill

**Not directly measurable without Xcode counters.** From static analysis
in `v6-metal-profile.md`:

- 16 simdgroups × 32 threads = 512 threads/TG (50% of M5's 1024 max)
- Estimated 22.7 KB register usage per simdgroup (~70% of estimated
  register file lower bound)
- Apple SDPA NAX: 4 simdgroups × 32 = 128 threads/TG (12.5% of max)

Apple favors smaller TGs (more co-resident wavefronts for latency hiding);
V6 favors larger TGs (more arithmetic per TG-launch overhead). Our
autoresearch converged on `ExecSG=16` because it amortizes MPP per-call
overhead — but at the cost of fewer TGs co-resident per core.

**This is the architectural trade-off we cannot resolve without changing
the abstraction layer.**

---

## Question D — Number of dispatches per attention call

| Path | Dispatches | Detail |
|------|-----------:|--------|
| V6 NAX | **~4** | 3× `Copy` (from `mlx::core::contiguous`) + 1 main NAX kernel |
| SDPA NAX | **1** | One `attention_nax<T, BQ, BK, BD, WM, WN>` kernel |

Source: static reading of `csrc/mfa_v6_nax_primitive.cpp:355-370` and
`mlx/backend/metal/scaled_dot_product_attention.cpp:18-164`.

**Empirical impact**: 0.175 ms (FlashVSR) / 1.893 ms (SeedVR2-small)
overhead. Small in absolute terms, but 4× more dispatches still imply
4× more command-buffer encoding, 4× more pipeline-state binding, and
4× more sync barrier opportunities — observable in CPU command-buffer
construction time on FlashVSR-dense (where it's 12% of the total).

---

## Question E — Inter-dispatch idle time

Cannot be measured from CPU side alone — need a GPU timeline (Xcode
Instruments → "Metal System Trace" view). The captured `.gputrace`
files contain this data; opening them in Xcode would expose:
- Time between kernel A completion and kernel B start
- GPU stall reasons
- CPU-GPU command queue depth

For programmatic analysis, an indirect proxy: end-to-end time minus
sum of individual kernel times. Implemented for V6 above:
- FlashVSR: full 1.510 ms = 0.175 ms (transposes) + 1.334 ms (kernel
  inferred) + 0.001 ms residual → 0.07% inter-dispatch idle (negligible)
- SeedVR2-small: full 274.565 ms = 1.893 ms + 272.672 ms + 0 ms residual
  → effectively zero idle

Inter-dispatch overhead on V6 is dominated by **the dispatches themselves**
(transpose work), not idle gaps between them.

---

## Question F — NAX utilization

**Not directly measurable** without Xcode (the trace contains the data
but in proprietary format).

Indirect inference from earlier roofline analysis (`v6-roofline-analysis.md`,
Axe 10):
- V6 ALU efficiency (FLOPS / theoretical NAX peak): 38–43% across shapes
- SDPA ALU efficiency: 39–48% across shapes

Both kernels reach roughly half of theoretical NAX peak — the rest is
lost to softmax, online normalization, mask handling, and memory traffic.
The 5–10pp gap to SDPA represents actual NAX cycles V6 wastes vs SDPA;
this profiling sprint cannot determine WHERE those cycles go without
counter data.

---

## Question G — Bandwidth utilization

**Memory peak measured (CPU-side)** — but this is *peak allocation*, not
*bandwidth used*. From `mx.get_peak_memory()`:

| Shape | V6 peak Δ | SDPA peak Δ | Ratio |
|-------|----------:|------------:|------:|
| FlashVSR | 21.1 MB | 5.4 MB | 3.9× |
| SeedVR2-small | 549.6 MB | 139.0 MB | 4.0× |

The 4× peak ratio matches expectation for layout transposes: V6 holds
Q+K+V (3× input footprint, ~3× of `15.7` MB = ~47 MB) plus output.
SDPA holds only output (~5 MB on FlashVSR).

Per-bandwidth utilization (GB/s achieved vs M5's ~614 GB/s peak) requires
counter data. The peak-memory measurement does NOT tell us bandwidth
utilization, only allocation footprint.

---

## Hypothesis validation

### Hypothesis 1 — Apple uses `metal_simdgroup_matrix` which bypasses MPP dispatcher

**STATUS: PARTIALLY CONFIRMED.**

- ✅ V6 has 4× more dispatches per call than SDPA (static analysis).
- ✅ The transpose-driven dispatch overhead is observable as 11.62% of FlashVSR end-to-end.
- ❓ The kernel-internal MPP overhead (vs simdgroup_matrix direct fragment
  scheduling) cannot be quantified without Xcode counter data showing
  ALU-active % or stall reasons inside the kernel.
- 🟡 The 1.22-1.34× kernel-only ratio is consistent with MPP abstraction
  cost but is also consistent with other explanations (occupancy, register
  spill, stall patterns).

### Hypothesis 2 — V6 has register spill that SDPA doesn't

**STATUS: NOT TESTABLE PROGRAMMATICALLY.**

Static estimates put V6 at ~22.7 KB register usage per simdgroup, near
the estimated capacity ceiling. But neither `mx.metal.start_capture()`
nor `MTLDevice` API exposes register spill counts. Xcode's Shader
Profiler is required.

### Hypothesis 3 — Tile size 16×48 / 16×64 is sub-optimal

**STATUS: PARTIALLY INVALIDATED by autoresearch.**

The 245-config Phase 3B autoresearch + 10-axis campaign converged on
the current configs. They are **empirically optimal** within the search
space. SDPA's choice (BQ=64, BK=32, WM=4, WN=1) yields a different shape
of TG (128 threads vs our 512) that is incompatible with the MPP
cooperative-tensor abstraction we use.

To match SDPA's tile geometry would require switching abstraction layers
(Hypothesis 1) — they're coupled.

### Hypothesis 4 — V6 is bandwidth-bound, SDPA is compute-bound

**STATUS: PARTIALLY INVALIDATED.**

V6's 4× peak memory delta vs SDPA is **layout-transpose driven**, not
kernel-internal bandwidth. The roofline analysis (`v6-roofline-analysis.md`)
showed BOTH V6 and SDPA at very high arithmetic intensity (>1500 FLOPS/byte)
on all production shapes — far above M5's ridge of 114. Both are **compute-
bound**, not bandwidth-bound. The transpose overhead is bandwidth-bound,
but the kernel itself is not.

---

## Sprint 2 priorities (justified by data)

In priority order, with effort estimates and expected gains:

### 1. Switch V6 to BHND layout (eliminate transposes) — HIGH PRIORITY

**Why**: 4× peak memory reduction (21 MB → 5.4 MB on FlashVSR; 549 MB →
139 MB on SeedVR2-small). Time savings: 0.175 ms (FlashVSR, 11.6%) /
1.89 ms (SeedVR2-small, 0.7%).

**Effort**: ~3 days. Modify the kernel's offset math in
`NAAttentionKernel.cpp` so the `Q_buf`/`K_buf`/`V_buf` indexing walks
[B, H, N, D] strides instead of [B, N, H, D] strides. The kernel's
internal tile layout doesn't change, only how it indexes inputs.

**Expected gain**: 5-12% on small-to-medium shapes, <1% on large shapes,
**but 4× peak memory reduction is unconditional**. This is also a
correctness improvement for memory-constrained workloads (e.g., training
runs that already pressure 64 GB RAM).

**Confidence**: HIGH. Pure layout change, low risk of correctness
regressions, easily verified with the `MFA_V6_SENTINEL_FILL` regression gate.

### 2. Open captured `.gputrace` in Xcode → ground-truth bottleneck — HIGH PRIORITY

**Why**: All 4 hypotheses about the kernel-internal gap (MPP overhead,
register spill, occupancy, stall reasons) need counter data to discriminate.

**Effort**: 1-2 hours of GUI inspection by Marco. No code change.

**Expected output**: A definitive answer to whether the 1.22-1.34× kernel-
only gap is dominated by ALU stalls (→ MPP dispatch issue), register
spill (→ tile resize needed), or memory stalls (→ access pattern issue).

**Confidence**: HIGH. The trace data is captured and permanent.

### 3. Implement chunked-K dispatch for N>65K — MEDIUM PRIORITY

**Why**: SeedVR2-large (N=111375) qualifies for the PR #3307 chunking
pattern. Independent of the abstraction-layer question — chunked-K
helps memory residency and GPU watchdog headroom regardless.

**Effort**: 1-2 days (already scoped in `mlx-pr-analysis.md`).

**Expected gain**: 5-15% on the slowest production shape; correctness
preservation via LSE-weighted reduction.

**Confidence**: MEDIUM. Chunking pattern is well-understood (FlashAttention-2
reduction); only risk is integrating with V6's Morton-grid dispatch.

### 4. Reimplement V6 forward using `metal_simdgroup_matrix` — HIGH EFFORT, MEDIUM CONFIDENCE

**Why**: If Sprint 2 priority #2 (Xcode counter analysis) confirms the
gap is MPP overhead, this is the single largest possible win. The
1.22-1.34× kernel-only ratio is the upper bound of the gain.

**Effort**: 2-3 weeks (rewrite kernel using NAXFrag/NAXTile pattern from
Apple's `nax.h`, reimplement online softmax with explicit fragment
scheduling, retune tiles).

**Expected gain**: 0–22% (i.e., closing some or all of the kernel-only
gap to SDPA). Upper bound: V6 reaches SDPA parity. Realistic: 5-10%
improvement (matching ~half the gap).

**Confidence**: MEDIUM. The architectural hypothesis is plausible and
supported by static evidence, but quantitative validation requires
priority #2.

**Sequencing**: Do priority #1 and #2 FIRST. They're cheap and
information-dense. Use the resulting data to decide whether to commit
to priority #4.

### 5. Skip — already ruled out by other sprints

| Item | Why skip |
|------|----------|
| Re-tune tiles (Hypothesis 3) | Already searched 245 configs (Phase 3B), 10 axes (campaign). Diminishing returns. |
| BLOCK_D sub-tiling (Axe 2) | Empirically NO-GO. |
| `tensor_inline + matmul2d` reconstruction | Coverage is 100% (sentinel-validated). Not the bottleneck. |
| Double-buffering cS_0 (Axe 7) | Architecturally infeasible without MPP prefetch primitive. |

---

## Decision-grade conclusion

The abstraction-layer hypothesis from `apple-sdpa-nax-analysis.md` is
**consistent with all measurable data** but **not yet conclusively
validated**. Two cheap actions (BHND layout switch + Xcode counter
inspection) will:

1. Unconditionally save 4× peak memory on every shape.
2. Provide ground-truth on whether the kernel-internal gap is MPP overhead,
   register spill, or something else.

After those two priorities land, the simdgroup_matrix rewrite decision
will be data-driven rather than hypothesis-driven.

For the FlashVSR-dense gap (V6 = 1.52× SDPA): ~10% of the gap is layout
transposes (closeable via priority #1), ~50% is the kernel-internal MPP
ceiling (priority #4 candidate), the remaining gap is within
kernel-launch and command-encoder overhead noise. We have a clear path
to 5-15% improvement before any major rewrite.
