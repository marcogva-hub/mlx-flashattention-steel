# V6 NAX Metal Profiling — First-Pass

**Date:** 2026-05-03
**Hardware:** Apple M5 Max (40 GPU cores, 64 NAX, 137 GB unified)
**Capture:** `docs/v6-nax/captures/v6_flashvsr_dense.gputrace`

---

## TL;DR

- **MLX exposes `mx.metal.start_capture(path)`** — programmatic GPU traces
  work. The `.gputrace` bundle is openable in Xcode Instruments. We
  successfully captured one (FlashVSR-dense, B=1 H=10 N=4096 D=64).
- **No programmatic GPU counter API** in MLX 0.31.x — depth analysis
  (occupancy, register usage, bandwidth) requires opening the .gputrace
  in Instruments.
- **Static analysis of register pressure** suggests V6's cooperative-
  tensor approach uses ~22-32 KB of register file per simdgroup at
  D=128 BK=48 ExecSG=16, near the M5 NAX register file capacity.
- **Occupancy at 16 simdgroups × 32 threads = 512 threads/TG**: M5
  supports up to 1024 threads/TG, so we are at 50% of the per-TG max.
  The autoresearch-converged `ExecSG=16` likely amortizes MPP fixed
  overhead more than it sacrifices occupancy.

---

## What MLX exposes

`mlx.metal` API surface (verified):
```python
['clear_cache', 'device_info', 'get_active_memory', 'get_cache_memory',
 'get_peak_memory', 'is_available', 'reset_peak_memory', 'set_cache_limit',
 'set_memory_limit', 'set_wired_limit', 'start_capture', 'stop_capture']
```

GPU counter APIs (`MTLCounterSampleBuffer`, `MTLCounterSet`) are NOT
exposed by MLX — would require a custom binding to read counters
programmatically.

`device_info()` returns a small dict (no register-file size, no
core count, no SLC/L2 size):
```json
{
  "device_name": "Apple M5 Max",
  "max_recommended_working_set_size": 115448725504,
  "memory_size": 137438953472,
  "architecture": "applegpu_g17s",
  "max_buffer_length": 86586540032,
  "resource_limit": 499000
}
```

For deeper limits, query Metal directly via Obj-C++ (see
`m5-threadgroup-memory.md`).

---

## GPU trace capture (verified working)

```python
import os
os.environ["MTL_CAPTURE_ENABLED"] = "1"  # required
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward

# allocate q, k, v
# warmup compile (run once before capture)

mx.metal.start_capture("docs/v6-nax/captures/v6.gputrace")
o, _ = v6_nax_forward(q, k, v, False)
mx.async_eval(o)
mx.synchronize()
mx.metal.stop_capture()
```

Output: `.gputrace` directory bundle (~3-30 MB depending on dispatch
size). Opens in Xcode Instruments → "Metal System Trace" view, exposing:
- Kernel timing breakdown (per-encoder, per-pipeline)
- ALU utilization (% of theoretical FLOPS achieved)
- Memory bandwidth (reads, writes, MMU traffic)
- Register pressure (spilled vs in-register)
- Threadgroup memory occupancy
- Warp execution time histogram

**Limitation**: requires opening Xcode Instruments GUI. Cannot extract
counter data from `.gputrace` bundle programmatically without parsing
the internal `Metadata.plist` (undocumented format).

---

## Static analysis: occupancy

For our **best D=128 config** (R=16, C=48, SG=16, used by SeedVR2-small,
CogVideoX, SeedVR2-large):

| Quantity | Value | Source |
|----------|-------|--------|
| BLOCK_R (BQ) | 16 | dispatch table |
| BLOCK_C (BK) | 48 | dispatch table |
| BLOCK_D (BD) | 128 | head_dim |
| executionSIMDGroups | 16 | dispatch table |
| Threads per TG | 16 × 32 = 512 | constant |
| Max threads per TG (M5) | 1024 | hardware probe |
| TG memory per TG | 16 × 48 × 16 × 2B = 24,576 B (24 KB) | `mfa_v6_nax_primitive.cpp:308` |
| Max TG memory per TG (M5) | 32,768 B (32 KB) | hardware probe |

**Threads-per-TG occupancy**: 512 / 1024 = **50%** of per-TG ceiling.
**TG memory occupancy**: 24576 / 32768 = **75%** of per-TG ceiling.

**Per-core simultaneous TGs**: M5 NAX cores can co-host multiple TGs as
long as cumulative resources fit. Apple's SDPA (4 simdgroups, 128 threads)
allows ~8 TGs co-resident per core. Our V6 (16 simdgroups, 512 threads)
allows only ~2 TGs co-resident. Apple has more wavefronts per core for
latency hiding.

But: per-TG arithmetic intensity is much higher in V6 (more MMAs per TG
boundary cost). The autoresearch found this trade-off favors fewer
larger TGs on M5 NAX, contrary to Apple's choice.

---

## Static analysis: register pressure

Kernel cooperative_tensor allocations per simdgroup
(from `csrc/mfa/v6_nax/NAAttentionKernel.cpp:766-794`):

| Tensor | Shape (per simdgroup) | Bytes (FP32) |
|--------|-----------------------|--------------|
| `cS_0` (QK accum) | BR × BC = 16 × 48 = 768 | 3072 |
| `cM` (row max) | BR = 16 | 64 |
| `cL` (row LSE) | BR = 16 | 64 |
| `correction` | BR = 16 | 64 |
| `cM_0_new` | BR = 16 | 64 |
| `cP` (post-softmax) | BR × BC = 768 | 3072 |
| `cO_0` (P@V output) | BR × BD = 16 × 128 = 2048 | 8192 |
| `Otile.frag_at(0,0..n)` | BR × BD = 2048 | 8192 |

**Per simdgroup**: ~22.7 KB (5688 FP32 values + small accumulators).
**Per TG (16 simdgroups)**: ~363 KB.

M5 NAX register file size is not publicly documented but A-series prior
arts suggest 32–64 KB per simdgroup execution unit. **At 22.7 KB/simdgroup
we are using 70% of the lower bound estimate.** This is likely the actual
ceiling — register pressure forced our autoresearch toward smaller
ExecSG (we tested 24 and 32 and they regressed).

Cross-check: Axe 2 (BLOCK_D sub-tiling) showed catastrophic regression
with smaller BD — consistent with register spill, since smaller BD
forces more iterations, each of which needs to keep cS_0/cP in registers
longer between MMAs.

---

## Bottleneck hypothesis (from indirect evidence)

Without a parsed `.gputrace`, our best inference from autoresearch +
static analysis is:

**Bottleneck #1 — MPP cooperative_tensor scheduling overhead (suspected)**
- Apple's `simdgroup_matrix` path (steel_attention_nax) does not use MPP.
  We do. The 5–7pp gap to SDPA at parity tile config is most plausibly
  explained by the abstraction layer.
- **Validation path**: re-run V6 with `MFA_V6_BYPASS_TGP=0` (Path B,
  default) and time the full forward. Compare to a hand-rolled
  simdgroup_matrix prototype (Sprint 2 work).

**Bottleneck #2 — Memory transpose overhead (estimated)**
- We transpose Q/K/V from `[B,H,N,D]` → `[B,N,H,D]` before the kernel
  and O back after. For SeedVR2-large that is 4 transposes × 570 MB ≈
  2.3 GB extra memory traffic, ~6 ms at 400 GB/s.
- 6 ms / 4700 ms total = ~0.13% — negligible. Not the bottleneck.

**Bottleneck #3 — Register pressure ceiling (suspected)**
- 22.7 KB/simdgroup is high but does not appear to spill (axes 1 + 2
  show stable behavior with no anomalous slowdowns at the chosen tile).
- The Axe 2 result that smaller BD regresses confirms: the kernel is
  *register-residency-sensitive* — smaller BD forces shorter loop
  iterations that do not amortize register-restoration cost.

**Not bottlenecks (ruled out)**:
- Threadgroup memory: at 24 KB / 32 KB we have 25% headroom, not
  saturated. Dropping to 8 KB does not help (Axe 2 confirms).
- Bandwidth: roofline analysis (Axe 10) showed AI > 1500 FLOPS/byte,
  far above M5 ridge of 114. Compute-bound.
- Dispatch overhead: 2 µs measured (Axe 9). Not relevant for kernels
  taking 1.4–4700 ms.

---

## Recommendation: profile in Instruments before Sprint 2

Open the captured `.gputrace`:
```
open docs/v6-nax/captures/v6_flashvsr_dense.gputrace
```

Or generate fresh ones for each production shape:
```
.venv/bin/python bench/v6_capture_traces.py
```
(Script not yet written — would loop FlashVSR/SeedVR2-small/CogVideoX
shapes, each with a corresponding `.gputrace` output.)

In Instruments → Metal System Trace, look for:
1. **GPU active time / wall time ratio**: should be > 0.85; lower means
   GPU stalled.
2. **ALU utilization on attention kernel**: V6 currently estimated at
   38–43% (Axe 10). Confirm via Instruments. Apple SDPA likely 45–50%.
3. **L2 cache hit rate**: Morton-order grid should give > 85%. If lower,
   our Morton-decode is suboptimal.
4. **Register spill events**: should be zero for V6. If non-zero, our
   tile choice is too aggressive.
5. **TG memory bank conflicts**: P-staging into TGP is a candidate; if
   conflicts > 0.1%, we should investigate padding.

This profiling pass is a Sprint 2 prerequisite — without quantitative
counters, we can only theorize about the bottlenecks.

---

## Captured trace inventory

| File | Shape | Size | Notes |
|------|-------|-----:|-------|
| `captures/v6_flashvsr_dense.gputrace` | B=1 H=10 N=4096 D=64 | ~3-10 MB | Verified working |
