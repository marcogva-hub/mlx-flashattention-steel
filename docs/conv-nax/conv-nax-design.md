# Conv3D NAX — Sprint C Phase 1.0 Design

**Date**: 2026-05-11
**Sprint family**: C (Conv3D NAX — kernel-level acceleration of SeedVR2 VAE)
**Branch**: `experiment/conv-nax-phase1_0_design` (off `experiment/conv-nax-phase0-survey` off `feat/conv-nax` off `feat/v6-nax`).
**Status**: design document; no kernel code written.
**Foundation**: Sprint C Phase 0 survey (`docs/conv-nax/survey-report.md` + companion JSON artifacts); Option F recommended.

---

## 1. Strategic context

**Target**: SeedVR2 VAE decoder (undistilled). The 6 representative shapes from `survey-report.md` §5.3 cover the up_blocks resnet families that account for ~99% of decoder wall-clock. **Conv3D 3×3×3 = 91.94% of decoder FLOPs, Conv3D 1×1×1 = 7.23%** per `architecture_map.json` op_type_breakdown — the workload is overwhelmingly Conv3D-bound.

**Phase 0 verdict (Option F)**: wrap `mpp::tensor_ops::convolution2d` for Conv2D + implicit-GEMM via `mpp::tensor_ops::matmul2d` for Conv3D. Phase 1.0 makes two refinements to Option F: (a) Conv2D is deferred to a future mini-sprint when a Conv2D workload surfaces (§10 — SeedVR2 VAE is 0% Conv2D); (b) the Conv3D implicit-GEMM specifically uses **materialized chunked im2col** (Option α in §2.2 below) rather than implicit on-the-fly im2col (Option β) or MPS Graph delegation (Option γ).

**Headroom mechanism**: MLX 0.31.2's conv stack on M5+ runs entirely on legacy `metal::simdgroup_matrix<T, 8, 8>` MMA hardware (verified by survey §2.3 grep of `mlx-source/`). NAX FP16 peak is ~38 TFLOPS theoretical vs legacy 8×8 simdgroup at ~5-10 TFLOPS effective. Baseline measurement (`baseline-summary.json`, 3 §4-compliant sessions): MLX achieves **38-40% of theoretical NAX peak** at consistent 2.52-2.67× ratio across all 6 shapes, 0.1-4.5% cross-session variance.

Quantitative ROI per SeedVR2 VAE decoder forward (per `baseline-summary.json`):
- Baseline: 2,643 ms (6 kernels × call counts)
- Theoretical NAX min: 1,033 ms
- Headroom at peak: 1,610 ms (60.9% reduction)
- Realistic target at 70%-of-peak NAX utilization: 1,127 ms savings (42.6% reduction)

Sprint C value proposition: deliver the Conv3D NAX path that pulls 42.6% wall-clock out of the dominant VAE in Marco's production VSR pipelines. The work is complementary to (not competing with) Flash-VAED's architecture-level distillation — both axes layer multiplicatively (§11).

## 2. Algorithm specification

### 2.1 Implicit GEMM formulation

For Conv3D with stride=1, pad=1, dilation=1 (the SeedVR2 VAE case):

**Tensor shapes (MLX NDHWC convention)**:
- Input X: `[N, T, H, W, C_in]`
- Weight K: `[C_out, K_T, K_H, K_W, C_in]` (analogous to MLX's `mx.conv_general` weight layout)
- Output Y: `[N, T_out, H_out, W_out, C_out]` with T_out = T, H_out = H, W_out = W

**Implicit GEMM mapping**:
- Output position index `m ∈ [0, M)` with `M = N · T · H · W`
- Inner contraction index `k ∈ [0, K_total)` with `K_total = K_T · K_H · K_W · C_in = 27 · C_in` for 3×3×3
- Output channel index `n ∈ [0, N_out)` with `N_out = C_out`

**Im2col tensor `A[m, k]`** (the materialized intermediate):
```
m → (batch, t_out, h_out, w_out)            via m = batch·T·H·W + t_out·H·W + h_out·W + w_out
k → (k_t, k_h, k_w, c_in)                   via k = ((k_t·K_H + k_h)·K_W + k_w)·C_in + c_in
t_in = t_out·stride_T - pad_T + k_t·dil_T   (=t_out + k_t - 1 in SeedVR2 case)
h_in = h_out·stride_H - pad_H + k_h·dil_H   (=h_out + k_h - 1)
w_in = w_out·stride_W - pad_W + k_w·dil_W   (=w_out + k_w - 1)
A[m, k] = (in_bounds(t_in, h_in, w_in)) ? X[batch, t_in, h_in, w_in, c_in] : 0
```

**Weight matrix B[k, n]**: reinterpret the `[C_out, K_T, K_H, K_W, C_in]` weight tensor as a `[K_total, C_out]` matrix. The natural memory layout — assuming weights stored as `[C_out, K_T, K_H, K_W, C_in]` row-major — gives B such that `B[k, n] = K[n, k_t, k_h, k_w, c_in]` where `(k_t, k_h, k_w, c_in)` is the inverse mapping of `k`. To get a `matmul2d`-friendly `[K_total × N_out]` matrix, transpose the weight once at primitive-construction time (or pre-pack at module init for production). The transpose is a one-time cost amortized across many forward passes.

**Matmul call**: `Y[m, n] = Σ_k A[m, k] · B[k, n]`. Direct `matmul2d` on `[M × K_total]` and `[K_total × N_out]`.

**Output reshape**: Y written contiguously as `[M × N_out]` is reshaped to `[N, T, H, W, C_out]` via stride manipulation only (no copy).

**Boundary handling**: the `in_bounds` predicate in the im2col indexing handles padding. The im2col writer kernel emits 0 for out-of-bounds positions (zero padding). Other padding modes (replicate, reflect) are out of scope for SeedVR2 VAE which uses zero padding throughout.

**Causal Conv3D note**: SeedVR2 / CogVideoX VAEs use causal Conv3D in the temporal dimension (pad applied only on the left, future-direction T positions not visible). Per `vae_cogvideox.py:166-173`: `pad_t = k_t - 1; F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_t, 0))` — asymmetric T padding (left = K_T - 1, right = 0). Phase 1.1's im2col kernel must support asymmetric pad_T explicitly; symmetric pad in H/W remains the default.

### 2.2 Im2col materialization strategy

Three options for how to handle the im2col tensor, evaluated against the SeedVR2 VAE workload:

#### Option α — Materialized chunked im2col [RECOMMENDED]
Build the im2col tensor `A[chunk_M × K_total]` for chunks of M output positions; dispatch one `matmul2d(A_chunk, B)` per chunk; write chunk results into Y.

**Pros**:
- Simple, debuggable, leverages MLX memory allocator.
- Per-chunk matmul is a clean `matmul2d` call — we already have NAX matmul expertise from V6 NAX (`csrc/mfa_v6_nax_primitive.cpp` source-gen pattern carries over directly).
- Sentinel-fill validation gate (per V6 NAX `MFA_V6_SENTINEL_FILL`) directly applies to detect addressing bugs in the im2col writer.
- Failure modes are isolated: an im2col bug shows as a wrong intermediate; a matmul issue shows in the matmul step. Three oracles (PyTorch CPU, MLX `conv_general`, sentinel) each catch different classes (§7).

**Cons**:
- Extra memory traffic: im2col tensor is written (im2col kernel) then read (matmul2d) = 2× of `M · K_total · 2 bytes`. For up3_resnet0 at chunk_M=297K, K_total=6912: 4.1 GB written + 4.1 GB read = 8.2 GB per chunk-pair, ~20 ms at 400 GB/s.
- Per-chunk dispatch overhead: each chunk pays one im2col-kernel-launch + one matmul2d-kernel-launch. At ~few µs per launch and ~16 chunks max, total ~50-100 µs overhead. Negligible vs the 200-500 ms per-shape matmul time.

**Estimated wall-clock per shape**:
`im2col_time + matmul2d_time` per chunk × n_chunks. For up3_resnet0_256to128 (M=4.46M, K=6912, N=128, chunk_M=297K, n_chunks=15):
- Per-chunk im2col: 4.1 GB / 400 GB/s = 10.3 ms (BW-bound).
- Per-chunk matmul2d compute: M_chunk · K · N · 2 FLOPs = 297K · 6912 · 128 · 2 = 526 GFLOPs at 70% of 38 TFLOPS = 19.8 ms.
- Per-chunk total ≈ 30 ms; 15 chunks ≈ 450 ms (no overlap).
- With double-buffered ping-pong (im2col kernel + previous matmul overlap): per chunk ≈ max(10.3, 19.8) = 19.8 ms; 15 chunks ≈ 297 ms.

Phase 0 theoretical min (no im2col cost) for this shape is 207.5 ms. Option α's estimated 297 ms is 43% slower than theory but **44% faster than MLX baseline** (529.9 ms) — already a meaningful win, before any tile tuning. Phase 1.5 perf sweep will refine these estimates with measured numbers.

#### Option β — Implicit on-the-fly im2col [REJECTED for Phase 1]
Custom matmul kernel that computes im2col addressing inside the inner loop. Each thread reads its `A[m, k]` directly from X using the inverse mapping, never materializing A.

**Pros**:
- Minimal memory traffic: only X (input) is read once, output written once. No im2col intermediate.
- Single kernel dispatch per shape — no chunking overhead.

**Cons**:
- Cannot use `mpp::tensor_ops::matmul2d` directly: the primitive expects already-laid-out matrix inputs at known addresses. To apply implicit im2col, we'd need a hand-rolled matmul kernel using `NAXFrag`/`NAXTile` — substantially more code (Sprint 3's microbench `docs/v6-nax/mpp-overhead-analysis.md` proved bare `simdgroup_matrix` rewrites are ~50× harder to make competitive than wrapping MPP).
- Validation difficulty: addressing logic and matmul logic are entangled. Sentinel-fill only catches macro coverage; subtle addressing bugs become matmul-level numerical noise.
- Apple may not have invested NAX-aware optimizations into the lower-level path; we'd be writing a NAX-driving kernel from scratch.

**Decision**: defer Option β until Option α ships AND a Phase 1.5 measurement shows meaningful gap between Option α achieved TFLOPS and theoretical. If Option α reaches 70% of theoretical at 38 TFLOPS = 26.6 TFLOPS sustained, the residual headroom from killing im2col memory traffic is small (~10 ms on the largest shapes). Not worth the implementation cost for Phase 1.

#### Option γ — MPS Graph fallback [REJECTED]
Hand the conv work to `MPSGraph`, hoping Apple's framework has NAX-aware conv3d kernels behind it.

**Pros**:
- Zero implementation work.

**Cons**:
- No control: cannot tune, cannot validate, cannot improve.
- No evidence Apple ships NAX conv3d in MPS Graph (Apple's MPP exposes `convolution2d` natively per survey §4, but no `convolution3d` — same constraint likely applies to MPSGraph's higher-level wrapping).
- Conflicts with Sprint C charter: this is a kernel-level optimization sprint, not a framework-delegation sprint.

**Decision**: REJECTED. No further consideration.

#### Final recommendation: Option α

Materialized chunked im2col with `mpp::tensor_ops::matmul2d` for each chunk's matmul. The implementation cost is modest (Phase 1.x ~16-27h CC time per §8 estimate); the perf win is bounded but meaningful (~40-44% wall-clock reduction on the largest shapes); the validation surface is well-understood from V6 NAX precedent.

### 2.3 Per-shape chunk_M sizing

**Target working-set budget**: chunk_M × K_total × dtype_bytes ≤ 4 GB per chunk. With double-buffered ping-pong, peak working set is ~8 GB — well under the 16 GB practical ceiling (M5 Max 128 GB total, but MLX runtime + Vivid-VR / VSR pipeline activations occupy ~30-40 GB during full pipeline runs).

| Shape | C_in | K_total = 27·C_in | im2col bytes/row | max chunk_M @ 4 GB | actual M | n_chunks |
|---|---:|---:|---:|---:|---:|---:|
| up2_resnet_256to256_T17_HW256_k3 | 256 | 6,912 | 13.5 KB | 297,000 | 1,114,112 | 4 |
| up3_resnet_128to128_T17_HW512_k3 | 128 | 3,456 | 6.75 KB | 594,000 | 4,456,448 | 8 |
| up3_resnet0_256to128_T17_HW512_k3 | 256 | 6,912 | 13.5 KB | 297,000 | 4,456,448 | 15 |
| up1_resnet_512to512_T9_HW128_k3 | 512 | 13,824 | 27 KB | 148,000 | 147,456 | 1 |
| mid_resnet_512to512_T5_HW64_k3 | 512 | 13,824 | 27 KB | 148,000 | 20,480 | 1 |
| up2_resnet0_512to256_T17_HW256_k3 | 512 | 13,824 | 27 KB | 148,000 | 1,114,112 | 8 |

**Per-chunk matmul saturation check**: smallest non-trivial case is up2_resnet_256to256 at chunk_M = 278,528 (= ceil(1,114,112 / 4)). Per-chunk matmul = 278,528 × 6,912 × 256 × 2 FLOPs ≈ 985 GFLOPs. At 70% of NAX peak (26.6 TFLOPS), this is 37 ms of compute per chunk — large enough to saturate NAX and amortize per-dispatch overhead.

**Smaller-than-chunk_M heuristic**: when M < chunk_M (single-chunk shapes), the heuristic shrinks chunk_M to M to avoid over-allocating the im2col buffer. up1_resnet (M=147,456 < 148,000) and mid_resnet (M=20,480) fit single-chunk; no chunking work needed at runtime.

**Auto-select algorithm**: `chunk_M = min(M, floor(4 GB / (K_total × dtype_bytes)))`, rounded down to a multiple of 32 (matmul2d's preferred M-tile boundary granularity per V6 NAX learnings — avoids tile remainder overhead, see `mfa_v6_nax_primitive.cpp:138-149`). Phase 1.3 measures chunk_M sweet spot empirically; the formula above is the initial heuristic.

**Env var override**: `MFA_CONV3D_CHUNK_M=N` for autoresearch; default = auto. Pattern lifted from V6 NAX `MFA_V6_BLOCK_R` / `BLOCK_C` / `BLOCK_D` env vars at `csrc/mfa_v6_nax_primitive.cpp:155-161`.

## 3. Sub-phase 0 microbench requirement

**Phase 0 made a critical assumption**: the 38 TFLOPS NAX FP16 peak figure (from Apple's published `mpp::tensor_ops::matmul2d` benchmarks) is the relevant theoretical bound. The actual sustained TFLOPS on the implicit-GEMM-shape matmuls (large M, modest K, modest N) is **unknown** — Apple's published peak typically uses balanced (M=N=K=4096+) square matmuls; our workload is heavily M-skewed (M up to 4.5M, K up to 13.8K, N down to 128).

**Phase 1.1 must begin with a sub-phase 0 microbench** that measures actual sustained TFLOPS at production shape ranges, mirroring Sprint 3's MPP-vs-simdgroup microbench methodology (`docs/v6-nax/mpp-overhead-analysis.md` precedent + `bench/mpp_vs_simdgroup_microbench.py` as structural template).

**Microbench grid** (24 cells minimum, all M×K×N combinations):
- M ∈ {20K, 150K, 300K, 600K, 1.1M, 4.5M} (six values, span 200× — covers all six shapes' chunk_M and full-M cases)
- K ∈ {3,456, 6,912, 13,824} (three values, all production K_total instances)
- N ∈ {128, 256, 512} (three values, all production N_out instances)
- Pruned subset acceptable if 4.5M × 13,824 × 512 OOMs (32 GB just for inputs); skip cells that exceed budget.

Apply Sprint A's §4-compliant cooldowns (90s round / 60s shape / 180s initial) per CLAUDE_V6_NAX.md. Three sessions sequential for cross-session variance per Sprint A precedent. Apply `mx.fast.metal_kernel`-style harness wrapping `mpp::tensor_ops::matmul2d` with cooperative_tensor destination + manual store loop (Sprint 3's harness is the structural template; reuse its MPP-API workarounds — `(device half*)A` cast for input const-stripping, cooperative_tensor destination only).

**Microbench output**: `docs/conv-nax/matmul2d-sustained-tflops.json` with `{(M, K, N): {median_us, sustained_tflops, fraction_of_38_peak}}`.

**Decision gate**:
- If sustained TFLOPS ≥ 30 (= 79% of advertised peak) on the dominant shapes: Phase 0's 42.6% realistic reduction target stands; proceed with Phase 1.1 as designed.
- If sustained TFLOPS 20-30 (= 53-79% of peak): Phase 0 ROI revises downward to ~28-38% reduction. Still ship-worthy, but R1 revision of this design doc to update §1 ROI numbers, §6 tile defaults, §8 sub-phase 1.5 ship/shelve thresholds.
- If sustained TFLOPS < 20 (< 53% of peak): the Sprint C value proposition collapses to the 14-17% range. Consider Option β reconsideration, or pivot recommendation in Phase 1.5 final.

**This microbench is the single most important measurement in Sprint C**. Its outcome calibrates every subsequent Phase 1.x sub-phase. Phase 1.1 cannot begin meaningful primitive implementation work without it.

## 4. Primitive class structure

`MFAConv3DForward` extends `mlx::core::Primitive`, structurally parallel to `MFAV6Forward` at `csrc/mfa_v6_nax_primitive.cpp:213-389`.

### 4.1 Class declaration (proposed)

```cpp
namespace mlx_mfa {

class MFAConv3DForward : public mlx::core::Primitive {
 public:
  struct Params {
    // Shape (N, C_in, T, H, W) input; (C_out, K_T, K_H, K_W) kernel.
    int N, C_in, T, H, W;
    int C_out, K_T, K_H, K_W;
    // Stride / dilation / padding per spatial axis.
    int stride_T, stride_H, stride_W;
    int dilation_T, dilation_H, dilation_W;
    int pad_T_lo, pad_T_hi;     // asymmetric for causal Conv3D
    int pad_H_lo, pad_H_hi;     // typically symmetric (pad_H_lo == pad_H_hi)
    int pad_W_lo, pad_W_hi;
    // ConvTranspose flip — out of scope for SeedVR2 VAE (always false).
    bool flip;
    // Tunable: 0 = auto-select per §2.3 heuristic.
    int chunk_M;
  };

  MFAConv3DForward(mlx::core::Stream stream, Params params);

  const char* name() const override { return "MFAConv3DForward"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("Conv3D NAX is GPU only");
  }

  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;

  std::vector<mlx::core::array> vjp(...) override {
    throw std::runtime_error("Conv3D NAX vjp NYI");
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override;

  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override;

 private:
  Params params_;
};

}  // namespace mlx_mfa
```

### 4.2 `eval_gpu` execution sequence

```
1. Sanity asserts (Sprint A pattern at csrc/v6_nax_compile.mm:46-64):
   - inputs[0].shape() == [N, T, H, W, C_in]  (NDHWC layout, MLX convention)
   - inputs[1].shape() == [C_out, K_T, K_H, K_W, C_in]
   - dtype == FP16 (initial; BF16 later if motivated)
   - chunk_M sanity: chunk_M × K_total × 2 ≤ 4 GB working budget
   - stride/dilation/pad bounds non-negative
   - output shape matches Params (consistency check)
2. Resolve chunk_M from heuristic (§2.3) if params_.chunk_M == 0.
3. Allocate per-call scratch:
   - im2col buffer A0: [chunk_M × K_total] FP16 (~4 GB max)
   - im2col buffer A1: [chunk_M × K_total] FP16 (ping-pong, ~4 GB)
   - pre-packed weight matrix B: [K_total × C_out] FP16 (small, ≤ ~20 MB)
     (If weights pre-packed at Primitive construction, this is reused
      across forward calls — see §4.3.)
4. Output buffer Y: [N × T × H × W × C_out] FP16, allocated upfront.
5. For each chunk c ∈ [0, n_chunks):
   a. m_start = c × chunk_M
   b. m_end = min(m_start + chunk_M, M)
   c. Dispatch `conv3d_im2col_chunk_kernel(X, A_ping, m_start, m_end, params)`
      to populate the im2col buffer for this chunk.
   d. Dispatch `matmul2d` on `(A_ping[m_end-m_start × K_total]) @ (B[K_total × C_out])`
      with cooperative_tensor destination + manual store loop to write
      directly into Y[m_start*C_out : m_end*C_out].
   e. Swap A_ping ↔ A_pong for next iteration (double-buffer).
6. Done. Y is in NDHWC layout, contiguous, ready for next op.
```

The im2col kernel and matmul2d on the previous chunk can overlap in the GPU command queue if MLX schedules them across separate command buffers. Phase 1.3 measures whether the overlap actually materializes and tunes if not.

### 4.3 Weight pre-packing

Two pre-packing options:

**Option (a)** — pre-pack once per Primitive instance at construction time, store as `static thread_local` on the Primitive. Each `eval_gpu` reuses the pre-packed B. Pros: zero per-call pre-pack cost. Cons: requires Primitive to own the pre-packed buffer, complicating the MLX Primitive contract.

**Option (b)** — pre-pack at module init time, store on the user-facing `mlx_mfa.conv3d_nax` wrapper. The wrapper holds the pre-packed B as a Python-side attribute on the conv-layer object; passes B as the second `inputs` array on each call. Pros: clean Primitive (stateless), explicit data flow. Cons: pre-pack code lives in Python.

**Recommendation**: Option (b). Mirrors the typical PyTorch / MLX module pattern (weights live on `nn.Module`-like objects). The pre-pack itself is a one-time `mlx::core::transpose(W, [4, 0, 1, 2, 3])` (or equivalent) at module load, ~milliseconds for the largest weight tensor.

### 4.4 Im2col kernel source-gen

The im2col kernel is small (~50-100 lines of MSL) but **must be source-generated per shape** because:
- Chunk M boundary affects the indexing math.
- Stride / dilation / pad / kernel size are best inlined for performance (no runtime branching in the hot path).
- Asymmetric pad_T for causal Conv3D needs a separate branch.

Source-gen pattern: mirror `csrc/mfa/v6_nax/NAAttentionKernel.cpp::createSource()`. Use `CodeWriter::SetValue` for template substitution. Cache by `ConvKey` (§5).

Per-shape MSL kernel does:
```
each thread loads one (m_chunk, k) cell of A:
  m_global = m_start + m_chunk
  decode m_global into (batch, t_out, h_out, w_out)
  decode k into (k_t, k_h, k_w, c_in)
  compute (t_in, h_in, w_in) via the indexing formula
  if in_bounds: A[m_chunk * K_total + k] = X[batch, t_in, h_in, w_in, c_in]
  else: A[m_chunk * K_total + k] = 0
```

Threadgroup size: 256 threads per TG, each thread handles one `(m_chunk, k)` cell. Grid: `((chunk_M_size × K_total + 255) / 256, 1, 1)`. Simple, fully parallel, no synchronization needed within or across threadgroups.

## 5. Cache key design

**Avoid Sprint A's three-separate-maps tech debt.** The Sprint A backward pipeline cache evolved into three separate `std::unordered_map`s (one each for dQ, dKV, and combined) with Direction discriminator deferred to v2.34.0+ — a maintenance burden documented in Sprint A's review report. Sprint C uses a single unified key from day 1.

### 5.1 Unified ConvKey

```cpp
struct ConvKey {
  enum class Kind : uint8_t {
    Conv3DIm2colKernel,   // im2col writer kernel
    Conv3DMatmul,         // matmul2d wrapping for one chunk
    // Future expansion:
    Conv2DDirect,         // mpp::tensor_ops::convolution2d wrap (§10)
  };
  Kind kind;
  // Convolution-defining params (shared across all kinds).
  int C_in, C_out;
  int K_T, K_H, K_W;
  int stride_T, stride_H, stride_W;
  int dilation_T, dilation_H, dilation_W;
  int pad_T_lo, pad_T_hi;
  int pad_H_lo, pad_H_hi;
  int pad_W_lo, pad_W_hi;
  // Per-kind specialization parameters.
  int chunk_M;            // Conv3DIm2colKernel: chunk boundary; Conv3DMatmul: M dim.
  int N, T, H, W;         // input dims for im2col kernel (full output extents needed
                          // because the kernel hard-codes them for unrolling).
  mlx::core::Dtype dtype;
  bool flip;
};

struct ConvKeyHash {
  size_t operator()(const ConvKey& k) const {
    // Use std::hash on a packed representation. Fields are small,
    // total ≤ ~100 bytes; combine via XOR-shift mix.
    size_t h = 0;
    h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.kind));
    h ^= std::hash<int>{}(k.C_in)  << 1;
    h ^= std::hash<int>{}(k.C_out) << 2;
    h ^= std::hash<int>{}(k.K_T)   << 3;
    h ^= std::hash<int>{}(k.K_H)   << 4;
    h ^= std::hash<int>{}(k.K_W)   << 5;
    // ... continue for remaining fields ...
    h ^= std::hash<int>{}(k.chunk_M) << 13;
    return h;
  }
};

namespace mlx_mfa {
  // Single global pipeline cache. Single mutex. Single point of management.
  std::mutex conv_mtx;
  std::unordered_map<ConvKey, void*, ConvKeyHash> conv_pipelines;
}
```

### 5.2 Why unified

Sprint A's three-maps pattern emerged organically because dQ and dKV were initially developed as separate primitives, and a combined cache was added later for the wrapper. The cache management code touched all three maps for every operation; iterators and locking became error-prone.

For Sprint C, the natural data axis is **the convolution itself** (defined by its shape + kernel + stride + ...) — and within that, the kernel kind (im2col writer vs matmul2d wrapper). Encoding kind as an `enum class` field of the key (not as a separate map) keeps:
- One cache, one mutex, one allocation path.
- Easy iteration / introspection / eviction (e.g., for unit tests resetting state).
- Future expansion (Conv2DDirect, Conv1D, ...) is a single line added to the enum.

### 5.3 Cache lifetime

Same pattern as V6 NAX (`csrc/mfa_v6_nax_primitive.cpp:97-98`): static global `std::mutex` + `std::unordered_map`, lifetime = process. No eviction policy needed — pipeline pointers are ~bytes each; even with hundreds of unique shapes, memory cost is negligible.

If unit tests need a clean cache: provide `mlx_mfa::conv_clear_cache()` helper that flushes the map under the mutex. Pattern from V6 NAX `clear_cache` in attention path.

## 6. Tile shapes per cluster

For the matmul2d call inside each chunk: `[chunk_M × K_total] @ [K_total × N_out]`. MPP's `matmul2d` exposes tile shape config via the `matmul2d_descriptor`; the survey §4 confirmed MPP determines its own internal scheduling but we control destination tile granularity (the cooperative tensor's row/col split).

### 6.1 Tile selection rationale

From V6 NAX learnings (`csrc/mfa_v6_nax_primitive.cpp:127-149`):
- M-axis (here output positions × C_out, written as Y[m, n]): prefer 16-row tile granularity. Larger M tiles per simdgroup increase register pressure; smaller wastes simdgroup parallelism.
- N-axis (here C_out): prefer 32-64 column tile granularity matching matmul2d's MMA-frag sizes.
- K-axis (here K_total contraction): full reduction in matmul2d's internal traversal; no tunable knob from our wrapper.

Empirically, V6 NAX converged on M_tile=16, with N_tile chosen per shape based on whether the output dim is bandwidth- or compute-bound. Sprint C's Conv3D shapes have N = C_out ∈ {128, 256, 512} — all comfortably above the 32-element MMA frag size. Tile choice is determined by N relative to BC traversal block.

### 6.2 Initial tile recommendations per cluster

Cluster the 6 shapes by structural similarity:

**Cluster 1 (top ROI, ranks 1-4 from §7 of survey)**: large-spatial T=17, HW ∈ {256, 512}, K ∈ {3,456, 6,912}, N ∈ {128, 256}. These are the chunked shapes with M ≫ chunk_M (≥ 4 chunks each).

| Sub-cluster | Shapes | M_tile | N_tile | exec_sg |
|---|---|---:|---:|---:|
| 1a: N=128 | up3_resnet, up3_resnet0 | 16 | 128 | 8 |
| 1b: N=256 | up2_resnet, up2_resnet0 | 16 | 64 | 16 |

Rationale: at N=128, we can pack the full N tile into a single matmul2d cooperative_tensor row (128 ≤ MMA-frag-tile column dimension on M5 NAX); at N=256, the cooperative_tensor row needs to split across two MMA frags, so smaller N_tile + more simdgroups keeps utilization high.

**Cluster 2 (smaller spatial, max channels)**: T ∈ {5, 9}, HW ∈ {64, 128}, K=13,824, N=512. Single-chunk or 3-chunk shapes.

| Sub-cluster | Shapes | M_tile | N_tile | exec_sg |
|---|---|---:|---:|---:|
| 2a: N=512 | up1_resnet, mid_resnet | 16 | 64 | 16 |

Rationale: N=512 splits across multiple cooperative_tensor row tiles regardless; pick the 64-column tile that V6 NAX found optimal on D=128 attention shapes (analogous traversal pattern).

**Cluster 3 (residual)**: up2_resnet0_512to256 (N=256, K=13,824, M=1.1M, 8 chunks). Inherits Cluster 1b tile.

### 6.3 Im2col tile structure (separate)

The im2col addressing kernel uses a different tile structure (purely a memory-rearrangement op, no MMA):
- One thread per `(m_chunk, k)` output cell.
- Threadgroup size: 256 threads.
- Grid: `((chunk_M × K_total + 255) / 256, 1, 1)`.
- No simdgroup-matrix abstractions used; pure scatter-store from device memory.

Phase 1.3 may discover that wider threadgroups (512 threads) reduce dispatch overhead on shapes with many im2col cells. Tunable via env var `MFA_CONV3D_IM2COL_TG_SIZE` — defer measurement to Phase 1.3.

### 6.4 Env var override grid (Phase 1.3 autoresearch)

| Env var | Knob | Initial default | Range to sweep |
|---|---|---|---|
| `MFA_CONV3D_CHUNK_M` | chunk_M per call | auto (§2.3) | 32K, 64K, 128K, 256K, 512K |
| `MFA_CONV3D_M_TILE` | matmul2d M tile | 16 | 16, 32 |
| `MFA_CONV3D_N_TILE` | matmul2d N tile | per-cluster | 32, 64, 128 |
| `MFA_CONV3D_EXEC_SG` | matmul2d simdgroups | per-cluster | 4, 8, 16 |
| `MFA_CONV3D_IM2COL_TG_SIZE` | im2col threadgroup size | 256 | 128, 256, 512 |

Autoresearch budget: 5 knobs × ~3 values each × 6 shapes × 3 sessions × ~5s/measurement = ~6 hours. Manageable within Phase 1.3 8h estimate.

## 7. Validation strategy

[skeleton — fills in Commit 4]

## 8. Sub-phase breakdown

[skeleton — fills in Commit 4]

## 9. Risks register

[skeleton — fills in Commit 4]

## 10. Conv2D inclusion / deferral

[skeleton — fills in Commit 5]

## 11. Relation to Flash-VAED

[skeleton — fills in Commit 5]

## 12. Open questions / R1 revision targets

[skeleton — fills in Commit 5]

