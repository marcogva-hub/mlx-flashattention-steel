# Conv2D/3D NAX — Phase 0 Survey Report

**Date**: 2026-05-11
**Sprint family**: C (Conv2D / Conv3D NAX)
**Hardware**: M5 Max 128GB, macOS 26.4, iStat performance fan profile.
**Branch**: `experiment/conv-nax-phase0-survey` (off `feat/conv-nax` off `feat/v6-nax`).

## 1. Executive summary

Sprint C targets MLX 0.31.2's conv stack, which **does not use NAX on M5+**. Every backend in `mlx/backend/metal/conv.cpp` (depthwise, implicit_gemm 2D, implicit_gemm 2D general, winograd 2D, implicit_gemm 3D, explicit_gemm ND) routes through `steel/gemm/mma.h` which uses **legacy `metal::simdgroup_matrix<T, 8, 8>` MMA** — the same 8×8 hardware Sprint 3's microbench proved was 14-50× slower than NAX on large matmuls. There is zero `is_nax_available()`, zero `MTLGPUFamilyApple9` check, zero `mpp::tensor_ops::*` usage in the conv path. NAX is completely unused for convolution.

**The target workload is dramatically Conv3D-skewed.** SeedVR2 VAE decoder profiling (Marco's prior `phase0_profiling.py` work, results in `~/code/SeedVR2_VAE_Flash-VAED/results/phase0/`) shows: **Conv3d_3x3x3 = 91.94% of FLOPs, Conv3d_1x1x1 = 7.23% of FLOPs, attention = 0.76%, everything else < 0.1%.** Conv3D dominates 99.17% of decoder compute. Sprint C's primary target is Conv3D, not Conv2D.

**Apple's NAX surface exposes `mpp::tensor_ops::convolution2d`** (verified in `/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/MPPTensorOpsConvolution2d.h`). Conv2D has a first-class, NAX-aware descriptor with full configurability (strides, dilations, groups, relaxed_precision, multiply vs multiply_accumulate, NHWC activation + HWIO weights layout, cooperative_tensor destination). **There is NO `convolution3d` primitive.** Conv3D must be routed via either (a) per-temporal-slice Conv2D loops, (b) implicit GEMM (im2col-3D → matmul2d), or (c) hand-rolled `NAXFrag` MMA.

**Recommendation: Option F — wrap `mpp::tensor_ops::convolution2d` for Conv2D shapes + implicit-GEMM via `mpp::tensor_ops::matmul2d` for Conv3D shapes.** The Conv2D wrapper is structurally analogous to V6 NAX wrapping `matmul2d` for attention. The Conv3D path leverages our existing NAX GEMM infrastructure (`csrc/v34_compile.mm`-style binding). This is a hybrid Option A/E variant that the prompt's A/B/C/D enumeration doesn't cleanly cover — see §10 for full rationale. Phase 1.0 design takes this as input.

The theoretical NAX bound on the 6 representative VAE Conv3D shapes ranges from 7.6 ms (mid_block 512→512) to 207.5 ms (up3 resnet 256→128 at 1×17×512×512). All shapes are heavily compute-bound (AI 1700–6600 FLOPs/byte vs M5 Max ridge ≈95) — no shape sits in the bandwidth-bound regime, so the bottleneck is NAX utilization, not memory traffic.

## 2. MLX 0.31.2 state on conv NAX

Reading the MLX 0.31.2 source at `~/code/mlx-source/`:

### 2.1 Conv ops in the public API
`mlx/ops.cpp` exposes (lines 4099, 4112, 4134, 4184, 4256):
- `conv1d(...)` → delegates to `conv_general`
- `conv2d(...)` → delegates to `conv_general`
- `conv3d(...)` → delegates to `conv_general`
- `conv_general(...)` → builds a `Convolution` primitive

All paths land at `Convolution::eval_gpu` in `mlx/backend/metal/conv.cpp:1283`.

### 2.2 Dispatch tree (Metal backend)
`Convolution::eval_gpu` dispatches by output rank:
- 5D → `conv_3D_gpu` (line 1210)
- 4D → `conv_2D_gpu` (line 1165)
- 3D → `conv_1D_gpu` (line 1078)

`conv_2D_gpu` → `dispatch_conv_2D_gpu` (line 970), which selects from FIVE backends:

| Backend | Trigger condition | Source file |
|---|---|---|
| `depthwise_conv_2D_gpu` (line 908) | groups>1, depthwise pattern, kernel≤7, stride≤2 | `steel/conv/...` |
| `winograd_conv_2D_gpu` (line 714) | 3×3, stride=1, dilation=1, big channels (C+O≥256) and big input (N×H×W≥4096) | `steel/conv/...` |
| `implicit_gemm_conv_2D_gpu` (line 191) | (C≤4 or C%16==0) and (O≤16 or O%16==0), most-friendly channel counts | `steel/conv/conv.h`, `steel_conv.h` |
| `implicit_gemm_conv_2D_general_gpu` (line 324) | (C%16==0 and O%16==0) or out_large, general dilation/stride | `steel_conv_general.h` |
| `explicit_gemm_conv_ND_gpu` (line 34) | fallback: explicit im2col + GEMM | classic legacy |

`conv_3D_gpu` is simpler: `dispatch_conv_3D_gpu` (line 671) → either `implicit_gemm_conv_3D_gpu` (line 503) or `pad_and_slice_conv_3D_gpu` (line 624, which itself wraps implicit_gemm). Only one main 3D backend: `steel_conv_3d.h`.

### 2.3 NAX presence in MLX 0.31.2 conv: zero

Grep results from `~/code/mlx-source/`:
- `grep -rn "is_nax_available\|nax_available\|FamilyApple9\|FamilyApple10" mlx/backend/metal/conv.cpp mlx/backend/metal/kernels/steel/conv/` → **empty**
- `find mlx -name "*conv*nax*" -o -name "*nax*conv*"` → **empty**
- `grep -rn "NAXFrag\|NAXTile\|matmul2d_descriptor\|tensor_ops" mlx/backend/metal/kernels/steel/conv/` → **empty**

Every Steel conv kernel includes `mlx/backend/metal/kernels/steel/gemm/mma.h` (verified):
- `mlx/backend/metal/kernels/steel/conv/conv.h:10`: `#include "mlx/backend/metal/kernels/steel/gemm/mma.h"`
- `mlx/backend/metal/kernels/steel/conv/kernels/steel_conv.metal:7`: same
- `mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_3d.metal:7`: same
- `mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_general.metal:7`: same

`mma.h` itself (`mlx/backend/metal/kernels/steel/gemm/mma.h:6-46`) uses `<metal_simdgroup_matrix>` and types `metal::simdgroup_matrix<T, 8, 8>` — the legacy 8×8 MMA primitives shipped on Apple GPUs since A14. NO `mpp::tensor_ops`, NO `NAXFrag/NAXTile`, NO Apple9+/Apple10 conditional path.

Compare `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h:3` which DOES include `nax.h` (NAXFrag/NAXTile abstractions). The attention stack has both a NAX-aware and a legacy path. The conv stack is legacy-only.

### 2.4 Implication

MLX 0.31.2's full conv stack on M5+ runs on legacy 8×8 simdgroup MMA hardware. NAX is unused. Sprint C's headroom is the gap between legacy 8×8 throughput and NAX throughput — Sprint 3's microbench measured this gap as 14-50× on naïve large matmuls. Real-world conv with grid fill, overhead, and non-matmul ops will see a smaller speedup, but the headroom is substantial and the opportunity is real.

## 3. mlx-mfa current state on conv

Verified by grep across `csrc/`, `mlx_mfa/`, `tests/`, `bench/`:
- No files contain `conv2d`, `conv3d`, `conv_general`, or `Convolution` (substring or function name)
- No filenames contain `conv`

**Zero conv presence in mlx-mfa.** Sprint C is greenfield: no scaffolding to extend, no tests to maintain compatibility with, no user-facing conv API to preserve.

Implication for Phase 1.0: the design doc has freedom to pick the cleanest Primitive structure without legacy-compat constraints. The natural model is the V6 NAX Primitive (`csrc/mfa_v6_nax_primitive.cpp`) — same source-gen + cache key + nanobind binding pattern.

## 4. Apple NAX conv surface

### 4.1 MPP framework headers present
`/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/`:
- `MPPTensorOpsMatMul2d.h` (used by V6 NAX)
- **`MPPTensorOpsConvolution2d.h`** (NEW for Sprint C)
- `MPPTensorOpsAvailability.h`
- `MPPTensorOpsBase.h`
- `MPPTensorOpsTraits.h`
- `MPPTensorOpsTypes.h`
- `MPPTensorOpsUtility.h`

No `MPPTensorOpsConvolution3d.h` exists. **Conv3D is not a first-class MPP primitive on M5 / macOS 26.4.**

### 4.2 `convolution2d_descriptor` capabilities

Per `MPPTensorOpsConvolution2d.h:56-101`:
```cpp
struct convolution2d_descriptor {
    int4 destination_dimensions;  // .x=C_out, .y=W, .z=H, .w=N (NHWO out)
    int4 source_dimensions;       // .x=C_in, .y=W, .z=H, .w=N (NHWC in)
    int2 kernel_dimensions;       // .y=KH, .x=KW
    convolution2d_activation_layout activation_layout;  // NHWC only currently
    convolution2d_weights_layout weights_layout;        // HWIO only currently
    int2 strides;
    int2 dilations;
    int groups;                   // ==1 only currently
    bool relaxed_precision;
    mode conv2d_mode;             // multiply | multiply_accumulate
};
```

Static constraints (header asserts):
- Activation layout: NHWC only (matches MLX's conv2d default)
- Weights layout: HWIO only
- Groups: 1 only (no depthwise / grouped conv support — IMPORTANT limitation for any future grouped-conv VAE)
- Half and float destination types both supported (`MPPTensorOpsConvolution2dImpl.h` template overloads at lines 2187, 2267, 2307, 2387, 2427, ...)

### 4.3 Execution surface
The `convolution2d` op template takes a `Scope` parameter analogous to `matmul2d`. Per `MPPTensorOpsConvolution2d.h:25-46` comment: "Currently only scope supported by convolution2d op is full threadgroup." So unlike `matmul2d` which supports `execution_simdgroup`, `convolution2d` requires `execution_threadgroup` scope. This is a tighter scope; we cannot dispatch a single conv from a single simdgroup the way Sprint 3's microbench did for matmul.

### 4.4 Cooperative tensor destination
Same pattern as `matmul2d`: `get_destination_cooperative_tensor<>()` returns a register-resident cooperative tensor that can be post-processed (bias add, activation) before writing to device memory. This is the V6 NAX pattern; we already know how to use it.

### 4.5 Implication

For Conv2D shapes that fit within MPP's static constraints (NHWC, HWIO, groups=1), wrapping `mpp::tensor_ops::convolution2d` is the natural path. The wrapper looks structurally identical to V6 NAX's wrap of `matmul2d`.

For Conv3D shapes (which dominate the VAE workload at 99% of FLOPs), there is no equivalent MPP primitive. Three sub-options:
1. **Per-temporal-slice Conv2D**: outer T-axis loop, inner conv2d call per temporal slice. Simple. Loses temporal-axis NAX parallelism. Per-call NAX overhead × T.
2. **Implicit GEMM via matmul2d**: build a 3D im2col tensor (expansion ~K_t × K_h × K_w = 27× for 3×3×3), then dispatch a single big `matmul2d`. Reuses existing V34 NAX-matmul expertise. Memory overhead from im2col, but the matmul itself sees maximum NAX utilization.
3. **Hand-rolled `NAXFrag`**: low-level conv3d kernel using NAX fragment primitives directly, bypassing both `convolution2d` (which doesn't exist for 3D) and `matmul2d`. Most flexible, highest implementation cost.

§10 selects the recommended path.

## 5. Turbo-VAED-Cog / SeedVR2 VAE bottleneck shape inventory

### 5.1 Op-type FLOPs breakdown
From `~/code/SeedVR2_VAE_Flash-VAED/results/phase0/architecture_map.json`:

| Op type | Param count | Total FLOPs | FLOPs % |
|---|---:|---:|---:|
| **Conv3d_3x3x3** | 144,898,179 | 1.044e14 | **91.94%** |
| **Conv3d_1x1x1** | 4,629,888 | 8.212e12 | **7.23%** |
| Attention_Matmul | 0 | 8.611e11 | 0.76% |
| GroupNorm | 23,040 | 3.564e10 | 0.03% |
| Linear_Attention | 1,050,624 | 4.295e10 | 0.04% |

Conv3D dominates 99.17% of decoder FLOPs. Attention is negligible (0.76%) — Sprint A's V34 backward research had marginal pipeline impact even at theoretical max gain. **Sprint C is the larger ROI sprint by ≥130×.**

### 5.2 Top wall-clock contributors

From `profiling_baseline.json` (PyTorch MPS / FP16, M1 Max system per profiling, scaled here for relative ranking on M5 Max):

| Rank | Block | MPS time | % of total |
|---:|---|---:|---:|
| 1 | up_blocks.2 (aggregate) | 39,775 ms | 25.8% |
| 2 | up_blocks.1 (aggregate) | 27,339 ms | 17.7% |
| 3 | up_blocks.3 (aggregate) | 27,245 ms | 17.7% |
| 4 | up_blocks.1.upsamplers.0 | 15,651 ms | 10.2% |
| 5 | up_blocks.2.upsamplers.0 | 14,241 ms | 9.2% |
| 6 | up_blocks.2.resnets.0 | 11,560 ms | 7.5% |
| 7 | up_blocks.3.resnets.0 | 11,299 ms | 7.3% |
| 8 | up_blocks.3.resnets.[1,2] | 15,016 ms (combined) | 9.7% |
| 9 | up_blocks.2.resnets.[1,2] | 14,432 ms (combined) | 9.4% |

Total decoder wall-clock: 153,980 ms per call (on the MPS baseline). The up_blocks dominate — every up_block is essentially a stack of Conv3d 3×3×3 resnets plus upsamplers (themselves Conv3D).

### 5.3 Representative shape inventory for Sprint C targeting

Drawn from the up_blocks resnet shapes (the inner Conv3D ops):

| Label | Input shape (N,C,T,H,W) | C_out | Kernel | Stride | Notes |
|---|---|---:|---|---|---|
| up2_resnet_256to256_T17_HW256_k3 | (1, 256, 17, 256, 256) | 256 | 3×3×3 | 1 | mid-spatial, peak-channels-mid |
| up3_resnet_128to128_T17_HW512_k3 | (1, 128, 17, 512, 512) | 128 | 3×3×3 | 1 | largest spatial, mid-channels |
| up3_resnet0_256to128_T17_HW512_k3 | (1, 256, 17, 512, 512) | 128 | 3×3×3 | 1 | channel reduction, max spatial |
| up1_resnet_512to512_T9_HW128_k3 | (1, 512,  9, 128, 128) | 512 | 3×3×3 | 1 | small-spatial, max-channels |
| mid_resnet_512to512_T5_HW64_k3 | (1, 512,  5,  64,  64) | 512 | 3×3×3 | 1 | smallest spatial |
| up2_resnet0_512to256_T17_HW256_k3 | (1, 512, 17, 256, 256) | 256 | 3×3×3 | 1 | mid-spatial channel reduce |

These 6 shapes cover the dominant 99%+ of Conv3D wall-clock. Phase 1 implementation targets all 6.

## 6. Theoretical NAX matmul bound

Conv3D 3×3×3 as implicit GEMM: M = N×T×H×W output positions, K = KT×KH×KW×C_in = 27×C_in, N_out = C_out.

NAX peak: 38 TFLOPS at FP16 (Apple advertised). HBM: ~400 GB/s.

| Shape | M | K | N_out | GFLOPs | compute_ms@38TF | bw_ms@400GBs | min_ms | AI (FLOPs/byte) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| up2_resnet_256to256_T17_HW256_k3 | 1,114,112 | 6,912 | 256 | 3,943 | **103.8** | 2.86 | **103.8** | 3,445 |
| up3_resnet_128to128_T17_HW512_k3 | 4,456,448 | 3,456 | 128 | 3,943 | **103.8** | 5.71 | **103.8** | 1,727 |
| up3_resnet0_256to128_T17_HW512_k3 | 4,456,448 | 6,912 | 128 | 7,886 | **207.5** | 8.56 | **207.5** | 2,303 |
| up1_resnet_512to512_T9_HW128_k3 | 147,456 | 13,824 | 512 | 2,087 | **54.9** | 0.79 | **54.9** | 6,602 |
| mid_resnet_512to512_T5_HW64_k3 | 20,480 | 13,824 | 512 | 290 | **7.6** | 0.14 | **7.6** | 5,168 |
| up2_resnet0_512to256_T17_HW256_k3 | 1,114,112 | 13,824 | 256 | 7,886 | **207.5** | 4.30 | **207.5** | 4,589 |

Total GFLOPs (one decoder forward, summed across the 6 shapes × call frequency = 3 resnets per up_block + 1 channel-reducer + ...): ~30,000-40,000 GFLOPs equivalent. Theoretical floor at 38 TFLOPS NAX: ~800-1100 ms total decoder wall-clock if every conv hit theoretical peak.

**Key observation: every shape is compute-bound by 18-385× over the bandwidth limit.** M5 Max's ridge point (38 TF / 400 GB/s = 95 FLOPs/byte) is well below the AI of all 6 shapes (1,727 - 6,602). The bottleneck is NAX utilization, not memory traffic. This is exactly the regime where switching from legacy 8×8 MMA (~5-10 TFLOPS peak depending on workload) to NAX (38 TFLOPS peak) should produce ~4× speedup on the kernel itself.

## 7. ROI ranking

[PENDING — fills in after baseline bench completes at ~13:05]

## 8. Bench data summary

[PENDING — fills in after baseline bench completes. Source: `docs/conv-nax/conv-nax-phase0-baseline-data.json`, 3 sessions × 6 shapes × 5 runs each, §4-compliant cooldowns.]

## 9. Open questions / data gaps

- **Apple's published NAX peak for FP16 conv specifically**: the 38 TFLOPS figure is from MPP `matmul2d` benchmarks; whether `convolution2d` achieves the same density is unknown. If the conv primitive has more setup overhead or different scheduling, the practical peak might be lower. Phase 1.0 design should include a microbench analogous to Sprint 3's MPP-vs-simdgroup, but applied to `convolution2d` to measure actual achievable TFLOPS.
- **Conv3D im2col memory pressure**: a 3×3×3 implicit-GEMM materializes a (N×T×H×W) × (27×C_in) im2col tensor. For up3_resnet0 (N=1, T=17, H=W=512), that's 4,456,448 × 6,912 × 2 bytes = **61.6 GB** of im2col data — exceeds M5 Max RAM budget. Phase 1.0 design must include tiling: process the conv in batches of output positions rather than materializing the full im2col. The standard approach for very large conv is "implicit im2col" — the matmul kernel computes the im2col addressing on-the-fly without materializing the full expansion. MPP `matmul2d` doesn't natively know about im2col addressing, so we need to either (a) chunk the matmul into smaller batches that DO fit, or (b) bind a custom address-mapping tensor accessor (more complex). The `mpp::tensor_ops::convolution2d` primitive presumably handles this internally for 2D — we should investigate whether Apple's implementation supports very large output dimensions efficiently.
- **MLX's `mx.conv_general` flip handling for Conv3D**: the `flip` parameter (used for ConvTranspose) is in the dispatch path but unused on this VAE (no transposed conv in CogVideoX decoder). Phase 1 should still verify the NAX primitive handles flip correctly, or document the flip-false-only restriction.
- **NHWC vs NCHW layout cost**: MLX uses NHWC for activations (matches MPP), but PyTorch uses NCHW. If users want to integrate `mlx_mfa.conv2d_nax` with PyTorch code via MLX-PyTorch interop, layout conversions add overhead. Document the layout requirement clearly.
- **Conv1D coverage**: the audit identified `conv_1D_gpu` exists in MLX, but no Conv1D shapes appear in the VAE workload. Phase 1 likely defers Conv1D unless an integration target surfaces.
- **Backward conv**: out of scope for Phase 1 (inference-only target). Document as future work.

## 10. Recommended algorithmic approach

### Conv2D (the simpler subset of the work)
**Option F1: Wrap `mpp::tensor_ops::convolution2d` directly.** Analogous to V6 NAX's wrap of `mpp::tensor_ops::matmul2d`. Use cooperative_tensor destination + manual writeback to device memory (same MPP API constraints that Sprint 3's microbench discovered: device-pointer destination is rejected by static_assert; cooperative_tensor is the only valid destination type).

### Conv3D (the dominant subset, 99% of target FLOPs)
**Option F2: Implicit GEMM via `mpp::tensor_ops::matmul2d` with chunked tiling.** Conv3D = matmul (N×T×H_chunk×W_chunk × 27×C_in) @ (27×C_in × C_out). Process the conv in chunks of output positions; each chunk fits in memory and runs as a single NAX matmul.

Why not Option F-alt (per-temporal-slice 2D conv): MPP `convolution2d` accumulates over (KH × KW × C_in). A per-temporal-slice approach loses the KT dimension's contribution. To preserve KT, we'd need to sum (KT) separate 2D conv results — that's 3× the matmul work for a 3×3×3 kernel. The implicit GEMM via matmul2d is structurally cleaner because it folds KT into K from the start.

Why not Option F-alt2 (hand-rolled NAXFrag): we don't have a measurement-grounded reason to believe a hand-rolled kernel beats `matmul2d`'s built-in scheduling. Sprint 3's microbench showed bare `simdgroup_matrix` (the equivalent for legacy hardware) was 14-50× SLOWER than MPP. The NAX path likely repeats this pattern: MPP's primitive is the natural starting point, not something we beat with naïve code.

### Combined: Option F (= F1 for 2D + F2 for 3D)
**This is the recommended approach.** Phase 1.0 design produces:
1. `NAConv2DKernel` (analogous to `NAAttentionKernel`): source-gen wrapper around `convolution2d_descriptor`. Cache key on (N, C_in, H, W, C_out, KH, KW, stride, pad, dtype).
2. `NAConv3DKernel`: source-gen for the implicit-GEMM-via-matmul2d path. Cache key on (N, C_in, T, H, W, C_out, KT, KH, KW, stride, pad, dtype) + chunk size.
3. Common dispatch surface `mlx_mfa.conv2d_nax(...)` and `mlx_mfa.conv3d_nax(...)` that routes by input rank.

### Why not Option D (defer / shelve)
The theoretical headroom is substantial. Conv3D 91.94% of FLOPs in target workload runs on legacy 8×8 MMA hardware. If we route through NAX matmul (38 TF) instead of legacy MMA (~5-10 TF effective), the headroom is 4-8× on the conv kernel itself. The actual implementation gain will be less due to im2col addressing overhead and grid-fill effects, but a 1.5-2.5× wall-clock improvement on dominant VAE shapes is plausible and would impact VSR pipeline end-to-end timing materially.

## 11. Phase 1.0 design doc scope

Phase 1.0 design doc (separate prompt after this survey) should specify:
- **Primitive class structure**: `MFAConv2DForward` and `MFAConv3DForward` parallel to `MFAV6Forward`. Per-Primitive Params struct, eval_gpu, vjp (NYI initially).
- **Source-gen approach**: a `NAConv2DKernel` class that generates MSL source via `CodeWriter` substitution, same pattern as `NAAttentionKernel`. Cache key in `csrc/mfa_conv_nax_primitive.cpp`.
- **Tile shapes per dominant cluster**: needs Phase 0 baseline data to inform initial defaults. Likely BR (output positions per tile) and BK (im2col contraction) tunable via env vars. Initial defaults from Apple's `convolution2d` examples + Sprint 3 learnings (BR=16 plays well with matmul2d on small destinations).
- **Validation strategy**:
  - Bit-exactness vs PyTorch CPU FP32 oracle: build a `numpy`-FP32 ground-truth comparison on smaller representative shapes (e.g., tile 64×64×8 versions of the production shapes).
  - Sentinel-fill correctness gate (same pattern as V6 NAX, env var `MFA_CONV_SENTINEL_FILL=1` — 100% coverage = every output position written).
  - Cross-validation against MLX's existing `mx.conv_general` on the same shapes.
- **Sub-phase breakdown**: 
  - 1.0 — design doc
  - 1.1 — Conv2D forward Primitive scaffolding + first dispatch on smallest representative shape
  - 1.2 — Conv2D full shape coverage + tile autoresearch
  - 1.3 — Conv3D forward Primitive (implicit-GEMM-via-matmul2d) on smallest representative shape
  - 1.4 — Conv3D full shape coverage + chunk tuning
  - 1.5 — Perf sweep (analogous to Sprint A Phase 1.5) on all 6 production shapes
- **Risks register**:
  - **im2col memory pressure on largest shapes**: up3 resnet0 256→128 at T17×512×512 expands to 61.6 GB. Phase 1.4 chunking must keep peak working set under ~16 GB.
  - **MPP `convolution2d` execution scope constraint** (full threadgroup only, not simdgroup). May limit how many simdgroups we can use per dispatch; need to verify per-shape.
  - **NHWC vs NCHW layout incompatibility with downstream code**: documented in §9, no fix in Phase 1.
  - **Conv1D out of scope**: re-evaluate if a workload surfaces.

## 12. Sign-off

**Sprint C Phase 0 verdict**: proceed to Phase 1.0 design with Option F (hybrid `mpp::tensor_ops::convolution2d` wrap for Conv2D + implicit-GEMM-via-`matmul2d` for Conv3D).

The opportunity is well-documented:
1. MLX 0.31.2's conv stack runs entirely on legacy 8×8 MMA — zero NAX usage (verified by source audit).
2. Apple's MPP exposes `convolution2d` as a first-class NAX-aware primitive.
3. The target workload is 99.17% Conv3D-bound (verified from Marco's existing VAE profiling).
4. The 6 representative Conv3D shapes are all heavily compute-bound (AI 1,727-6,602 vs ridge 95), placing the bottleneck squarely on NAX utilization, not memory traffic.
5. Theoretical headroom is 4-8× on the kernel itself, plausibly 1.5-2.5× wall-clock end-to-end on real VSR pipelines.

The Phase 0 baseline bench (running at the time of writing) will quantify the exact gap between MLX's current implementation and theoretical max — see §7 and §8 after the bench completes.

**Next concrete step Marco takes**: review Phase 0 survey, kick off Phase 1.0 design prompt. Phase 1.0 produces the design doc that Phase 1.1 implementation works from. Phase 1.1 ships the first NAX Conv2D primitive scaffolding (smallest shape, correctness-validated).

If between this Phase 0 and Phase 1.0, the baseline bench (§7-§8) surfaces MLX achieving already ≥ 60% of theoretical NAX bound on the dominant shapes: the case for Sprint C weakens significantly. Re-evaluate the recommendation; potentially pivot to Sprint B (block-sparse / LCSA NAX) per the original A→C→B order. This is a defensible Phase 0 outcome.

