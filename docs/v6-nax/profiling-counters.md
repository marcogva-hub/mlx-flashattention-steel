# V6 NAX vs SDPA — Profiling Data Extraction

**Date:** 2026-05-04
**Hardware:** Apple M5 Max (40 GPU cores, applegpu_g17s)
**Branch:** `feat/v6-nax`
**Captures:** 4 `.gputrace` bundles + CPU-side profiling

---

## What was captured

| Trace bundle | Size | Shape |
|--------------|------|-------|
| `docs/v6-nax/captures/v6_flashvsr.gputrace` | 221 MB | FlashVSR-dense (B=1, H=10, N=4096, D=64) |
| `docs/v6-nax/captures/sdpa_flashvsr.gputrace` | 221 MB | FlashVSR-dense (same shape, SDPA NAX) |
| `docs/v6-nax/captures/v6_seedvr2_small.gputrace` | 1.9 GB | SeedVR2-small (B=1, H=20, N=26730, D=128) |
| `docs/v6-nax/captures/sdpa_seedvr2_small.gputrace` | 2.0 GB | SeedVR2-small (same shape, SDPA NAX) |
| `docs/v6-nax/captures/v6_flashvsr_dense.gputrace` | 216 MB | (older capture from investigation sprint, redundant) |

All captures use `mx.metal.start_capture(path)` / `mx.metal.stop_capture()`
with 3 warmup iterations to eliminate compile/cache cold-start. Capture
script: `bench/v6_capture_traces.py` (inline in this doc — captured via
ad-hoc script).

Total disk usage: ~4.4 GB. Excluded from git via `.gitignore`.

---

## Bundle internal structure

`.gputrace` is a directory bundle. Inspection of the existing
`v6_flashvsr_dense.gputrace`:

```
metadata                                 1136 B  Apple binary plist
capture                                  1776 B  MTSP binary (proprietary)
index                                    14 KB   xdic binary (proprietary)
device-resources-0xNN                     8 KB   MTSP binary
delta-device-resources-0xNN              56 B    MTSP binary
unused-device-resources-0xNN             56 KB   MTSP binary
store0                                   28 KB   binary
unsorted-capture                         16 KB   binary
startup-{0,1}-platform                   76 B    binary
231131F2A31A0D0D                         150 MB  main trace blob (binary)
347C9007F0F59FC6 / F862D52962569A65      ~150 KB various binary blobs
MTLBuffer-{32..72}-0                     48 MB   captured GPU buffer contents
```

### What is human-readable

Only `metadata` (binary plist) is decodable via `plutil -convert json`:

```json
{
  "DYCaptureSession.unusedFunctionCount": 0,
  "DYCaptureSession.graphics_api": 1,
  "DYCaptureSession.unusedCommandQueueCount": 0,
  "(uuid)": "7C3D2B63-AD53-49EF-92DA-B39E27C56164",
  "DYCaptureSession.boundaryLess": false,
  "DYCaptureSession.capture_version": 0,
  "DYCaptureSession.unusedRenderPipelineStateCount": 0,
  "DYCaptureSession.deviceId": 1,
  "DYCaptureEngine.launch_dictionary": {},
  "DYCaptureSession.nativePointerSize": 8,
  "DYCaptureSession.unusedBufferCount": 0,
  "DYCaptureSession.unusedTextureCount": 0,
  "DYCaptureEngine.linked_on_apex_or_later": 1,
  "DYCaptureSession.unusedSamplerStateCount": 0,
  "DYCaptureSession.library_link_time_versions": {
    "System": 88866816, "Metal": -1, "Foundation": -1,
    "CoreFoundation": 264765951, "UIKit": -1, "AppKit": -1
  },
  "DYCaptureSession.unusedLibraryCount": 0,
  "DYCaptureSession.unusedComputePipelineStateCount": 0,
  "DYCaptureEngine.captured_frames_count": 1,
  "DYCaptureSession.unusedDepthStencilStateCount": 0,
  "DYCaptureSession.interpose_feature_version": 65538,
  "DYCaptureSession.interpose_patch_version": "0"
}
```

This gives session metadata but **no GPU counter data**.

### What is NOT human-readable

- **`capture` / `index` / `device-resources*`**: Apple's proprietary `MTSP`
  and `xdic` binary serialization formats. The first 4 bytes of `capture`
  are `MTSP`, of `index` are `xdic`. Apple has not publicly documented
  these formats.
- **`231131F2A31A0D0D`**: 150 MB main trace blob. `strings` extraction
  yields garbage (no human-readable text). This is where the GPU counter
  data and command stream are stored.

### Conclusion: counter extraction requires Xcode GUI

GPU counters (ALU active %, occupancy, register pressure, memory limiter,
stall reasons, NAX cycles) are inside the proprietary binary blobs. They
are accessible only by:

1. Opening the `.gputrace` in Xcode (Metal Debugger)
2. Programmatically via Apple's private Metal Replayer SDK (not public)
3. Using `MTLCounterSampleBuffer` API directly during capture (not exposed
   by MLX's `mx.metal.start_capture`)

For full counter analysis, opening the captured traces in Xcode is the
required path. The traces are saved on disk for that future GUI analysis.

---

## What CAN be measured programmatically

### CPU-side timing breakdown

Source: `bench/v6_cpu_profile.py`, ITERS=20, p50 reported.

| Shape | full_v6 (ms) | transp+contig (ms) | kernel_only (ms) | sdpa (ms) | V6/SDPA |
|-------|------------:|-------------------:|-----------------:|----------:|--------:|
| FlashVSR-dense | 1.510 | 0.175 (11.62%) | 1.334 | 0.995 | **1.517** |
| SeedVR2-small  | 274.565 | 1.893 (0.69%) | 272.672 | 222.729 | **1.233** |

Where:
- `full_v6` = end-to-end `v6_nax_forward(q, k, v, False)` Python call
- `transposes_contig` = 3 transposes + 3 `contiguous()` calls (BHND→BNHD)
- `kernel_only` = `full_v6 − transposes_contig` (implied kernel + return transpose)
- `sdpa` = `mx.fast.scaled_dot_product_attention(q, k, v, scale=...)`

**Key derivable ratios**:
- **Transpose overhead share**: 11.62% (FlashVSR) / 0.69% (SeedVR2-small)
- **Kernel-only V6/SDPA ratio**: 1.341× (FlashVSR) / 1.224× (SeedVR2-small)

### Peak memory delta

Source: `bench/v6_cpu_profile.py` (mx.{reset,get}_peak_memory).

| Shape | Q/K/V base | V6 peak Δ | SDPA peak Δ | V6 extra |
|-------|-----------:|----------:|------------:|---------:|
| FlashVSR-dense | 15.7 MB | 21.1 MB | 5.4 MB | **+15.7 MB (3.9× SDPA)** |
| SeedVR2-small  | 410.6 MB | 549.6 MB | 139.0 MB | **+410.6 MB (4.0× SDPA)** |

V6 peak memory delta is consistently ~4× SDPA's. The extra equals the
size of Q+K+V (one full copy each via `contiguous()`). This is the cost
of the BHND→BNHD layout transpose.

### Stdev (run-to-run variance)

| Shape | full_v6 stdev | sdpa stdev |
|-------|--------------:|-----------:|
| FlashVSR-dense | 0.110 ms | 0.081 ms |
| SeedVR2-small  | 5.111 ms | 4.653 ms |

Variance is low (≤2% for SeedVR2-small, ≤7% for FlashVSR). The 1.22-1.52×
V6/SDPA ratio is well above noise — the gap is real, not stdev.

---

## Static analysis: dispatch count per attention call

From source code reading (no GPU instrumentation needed):

### V6 NAX path (`csrc/mfa_v6_nax_primitive.cpp:355-370`)

```cpp
auto q_bnhd = mlx::core::transpose(q, ...);   // creates strided view (no dispatch)
auto k_bnhd = mlx::core::transpose(k, ...);   // creates strided view (no dispatch)
auto v_bnhd = mlx::core::transpose(v, ...);   // creates strided view (no dispatch)
auto qc = mlx::core::contiguous(q_bnhd, ...); // 1 dispatch (Copy primitive)
auto kc = mlx::core::contiguous(k_bnhd, ...); // 1 dispatch
auto vc = mlx::core::contiguous(v_bnhd, ...); // 1 dispatch
// ... v6_nax_forward primitive eval_gpu ...   // 1 dispatch (the main NAX kernel)
// transpose output BNHD→BHND on return         // strided view (no dispatch in current code)
```

**Total V6 dispatches: ~4 per attention call** (3 contiguous + 1 main kernel).

### SDPA path (`mlx/backend/metal/scaled_dot_product_attention.cpp:18-164`)

```cpp
void sdpa_full_self_attention_nax(...) {
  ...
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);  // 1 dispatch
}
```

Note: 5 occurrences of `dispatch_threadgroups` in the file, but those cover
**different code paths** (vector decode, full attention, etc.). The full
NAX path used for our shapes runs exactly **1 dispatch per call**.

**Static dispatch ratio**: V6 issues ~4× more GPU dispatches than SDPA per
attention call. Empirically, the `transposes_contig` time is 0.175 ms
(FlashVSR) / 1.89 ms (SeedVR2-small) — small in absolute terms.

---

## Summary table

| Quantity | V6 NAX | SDPA | Source |
|----------|-------:|-----:|--------|
| **Dispatches per call** | ~4 | 1 | Static analysis |
| **End-to-end time (FlashVSR)** | 1.510 ms | 0.995 ms | CPU timing |
| **Kernel-only time (FlashVSR)** | 1.334 ms | 0.995 ms | CPU timing (subtracted) |
| **End-to-end time (SeedVR2-small)** | 274.6 ms | 222.7 ms | CPU timing |
| **Kernel-only time (SeedVR2-small)** | 272.7 ms | 222.7 ms | CPU timing (subtracted) |
| **Peak memory delta (FlashVSR)** | 21.1 MB | 5.4 MB | mx.get_peak_memory |
| **Peak memory delta (SeedVR2-small)** | 549.6 MB | 139.0 MB | mx.get_peak_memory |
| ALU Active % | ? | ? | Xcode GUI required |
| Occupancy % | ? | ? | Xcode GUI required |
| Register pressure | ? | ? | Xcode GUI required |
| Memory limiter % | ? | ? | Xcode GUI required |
| Stall counters | ? | ? | Xcode GUI required |
| NAX utilization % | ? | ? | Xcode GUI required |

The first 7 rows are firmly measured. Rows marked `?` need a future
Xcode session on the captured traces.

---

## Reusable artifacts

- `docs/v6-nax/captures/{v6,sdpa}_{flashvsr,seedvr2_small}.gputrace`
  (gitignored, ready for Xcode inspection)
- `bench/v6_cpu_profile.py` (reusable CPU profiler)
- `docs/v6-nax/profiling-counters.json` (raw timing/memory data)
- `docs/v6-nax/profiling-counters.md` (this file)

The next step is the analytical synthesis — see
`docs/v6-nax/v6-vs-sdpa-profiling-analysis.md`.
