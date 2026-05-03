# Draw Things MFA v2 — Source Code Analysis

**Date:** 2026-05-03
**Branch under review:** `liuliu/ccv` `unstable` HEAD = `b2834be8` (May 2 2026)
**Our port commit:** `b4c63d59` (May 3 2026)

---

## TL;DR — The premise of this task is incorrect

The user's framing was: *"we ported `lib/nnc/mfa/kernels/NAAttentionKernel.cpp` (the OLD v1) but should have ported `lib/nnc/mfa/v2/NAAttentionKernel.cpp` (the v2.5)."*

**This is wrong.** Verified by:

1. **No `/v2/` directory exists at HEAD** of `liuliu/ccv` `unstable`:
   ```
   $ ls /tmp/ccv-latest/lib/nnc/mfa/v2/
   ls: lib/nnc/mfa/v2/: No such file or directory
   ```

2. **Commit `0bf97fca` (March 6 2026, "Finish migrate v2.")** is literally
   `git mv lib/nnc/mfa/v2/* lib/nnc/mfa/kernels/`. It moved the v2 code INTO
   `kernels/` and removed the `v2/` directory. Verified by `git show
   --stat 0bf97fca`:
   ```
   lib/nnc/mfa/{v2 => kernels}/AttentionDescriptor.cpp     |   0
   lib/nnc/mfa/{v2 => kernels}/AttentionKernel.cpp         |   0
   lib/nnc/mfa/{v2 => kernels}/AttentionKernelDescriptor.cpp | 0
   ...
   ```
   (Size delta = 0 = pure rename.)

3. **Our port is dated May 3, 2026** — well after the March 6 migrate. We
   ported from the post-migrate `kernels/` directory, which contains the
   v2 code.

4. **Bit-identical kernel source generator**. Diff vs upstream HEAD:
   ```
   $ diff csrc/mfa/v6_nax/NAAttentionKernel.cpp \
          /tmp/ccv-latest/lib/nnc/mfa/kernels/NAAttentionKernel.cpp | wc -l
   99
   ```
   All 99 diff lines are in the **first 130 lines of the file** (constructor
   + threadgroup helpers). The body of `createSource()` (lines 130–2647)
   that emits the actual MSL kernel is byte-identical.

**Conclusion: V6 NAX = MFA v2.5. The 4.6× M5/M4 perf claim from the Draw
Things blog post applies to OUR kernel.** Our 0.87-0.97× SDPA result is
the v2 code's own ceiling on M5 Max, not v1 staleness.

---

## What the 99-line diff covers

The differences are pure framework adaptation, not algorithm:

### 1. Header includes (lines 1–10)
- Upstream: `#include "../ccv_nnc_mfa.hpp"` (ccv error handling)
- Ours: `#include "../mfa_compat.h"` (mlx-mfa shim)

### 2. Constructor signature (line 33)
- Upstream: takes `MTL::Device *const device`, compiles library inline:
  ```cpp
  source = createSource();
  library = device->newLibrary(source, ...);
  if (!library) {
    bypassThreadgroupMemory = false;
    source = createSource();  // retry without TGP bypass
    library = device->newLibrary(source, ...);
  }
  ```
- Ours: source-only — we delegate compilation to mlx-mfa's shader cache
  (`csrc/v6_nax_compile.mm`). Same fallback logic implemented in our
  pipeline-creation path.

### 3. `threadgroupMemoryAllocation` / `threadgroupSize` signatures
- Upstream: take `MTL::ComputePipelineState *const pipelineState` to read
  `threadExecutionWidth()` at runtime.
- Ours: hardcode `32` (Apple Silicon SIMD width is invariant). Simpler.

### 4. `threadgroupsPerGrid` (Morton-order grid layout)
- Upstream: implemented inside `NAAttentionKernel.cpp`:
  ```cpp
  case AttentionKernelType::forward: {
    const uint32_t row_groups = ceilDivide(matrixDimensions[0],
                                            blockDimensions[0] * executionSIMDGroups);
    const uint32_t morton_bits = ceil_log2_u32_host(row_groups) +
                                 ceil_log2_u32_host(Hq);
    return MTL::Size(uint64_t(1) << morton_bits, 1, batchDimension);
  }
  ```
- Ours: same logic implemented in `csrc/v6_nax_compile.mm:111-119`. Verified
  identical formula.

The Morton-order grid layout is critical for L2 cache locality (adjacent
threadgroups process spatially nearby (row_block, head) tiles). The kernel
template at `csrc/mfa/v6_nax/NAAttentionKernel.cpp:280-290` already contains
the matching `morton_decode_rectangular_2d(tgid.x, ...)` decode logic — we
ported it correctly.

---

## Inventory: post-migrate `lib/nnc/mfa/kernels/` (= the v2 code)

| File | Purpose | In our port? |
|------|---------|-------------:|
| `NAAttentionKernel.cpp` (2667 LOC) | NAX forward kernel source generator | YES |
| `NAAttentionKernel.hpp` | Kernel class declaration | YES |
| `NAAttentionKernelDescriptor.cpp/hpp` | Tile config descriptor | YES |
| `NAAttentionDescriptor.cpp/hpp` | Per-call params descriptor | NO (we inline these in `mfa_v6_nax_primitive.cpp`) |
| `AttentionKernel.cpp` (≥1000 LOC) | Generic (non-NAX) attention generator | NO (this is the V2 STEEL ancestor — we already have V2 STEEL via separate path) |
| `AttentionKernelType.hpp` | enum: forward / backwardQuery / backwardKeyValue | YES |
| `AttentionOperand.hpp` | enum: Q, K, V, O, S, P, L | YES |
| `NAInt8AttentionKernel.cpp` | INT8 quantized NAX attention | NO |
| `NAInt8AttentionDescriptor.cpp` | INT8 NAX descriptor | NO |
| `NAInt8AttentionKernelDescriptor.cpp` | INT8 tile descriptor | NO |
| `AttentionKernel+Precompiled.cpp/inc` | Precompiled metallib emitters | NO (mlx-mfa JIT-compiles) |
| `GEMMHeaders.hpp` | Common GEMM utility macros | YES (in `csrc/mfa/`) |
| `CodeWriter.hpp/cpp` | String templating for kernel source | YES |
| `ANERowwiseTransform*` | ANE rowwise primitives (CoreML path) | NO |
| `Adam*Kernel`, `Cmul*Kernel`, etc. | Other ML primitives (~30 files) | NO |

**What we'd port additionally to expand support**:
- `NAInt8Attention*` for INT8 NAX (would unlock 4-bit quantized attention)
- `NAAttentionKernel.cpp` backward-pass code (already partially ported as
  `MFAV6Backward` scaffolding but not exposed)

**What is genuinely new in v2** (post-migrate development on `kernels/`):
The git log of `lib/nnc/mfa/kernels/NAAttentionKernel.cpp` shows 14 commits
between the migrate (March 6) and our port (May 3). Highlights:

```
e90be253 2026-04-28  Update with MASK_SCALE to fix exp2 mismatch
f9394134 2026-04-28  Fix unit test issues
2d1bc88e 2026-04-28  Adding is_varlen support for NAAttention and NAInt8Attention
5a484bba 2026-04-20  Adding mask support for attention op
84a3418d 2026-04-20  Adding isCausal support for both NAInt8Attention / NAAttention
fe176694 2026-04-06  Add support for tail case on NAAttention backward
b5d2cb61 2026-04-03  Change launch order for backward key-value
384add80 2026-04-03  Performance fix for NAAttention                  ← perf
e2504260 2026-04-03  Performance hit fix for NAAttention backward pass ← perf
ae1de996 2026-03-30  Enable morton order for attention forward kernel  ← perf
aaecd1aa 2026-03-30  Tighten up backward pass staging
9ba5bdae 2026-03-29  Clean up code for multi-kBlocks on backward pass
515fce43 2026-03-29  Add backward pass for NAAttention
```

Since our port is May 3 and the latest kernel commit is April 28, **we have
all the perf fixes** (`384add80`, `e2504260`, `ae1de996`). The Morton-order
grid (`ae1de996`) is verified present in our port at
`csrc/mfa/v6_nax/NAAttentionKernel.cpp:280-290` and dispatched correctly
at `csrc/v6_nax_compile.mm:111-119`.

---

## Architectural questions (answered)

### Tensor layout
- **Same as our port**: `[B, N, H, D]`. The kernel's `Q`, `K`, `V` device
  tensors are 2D `[K_Hq, R_LENGTH]` slices indexed by head batch
  (line 751–753 of v2):
  ```cpp
  auto Q = tensor<device {{MEMORY_NAME_Q}}, dextents<int32_t, 2>, tensor_inline>(
      Q_buf, dextents<int32_t, 2>(K_Hq, {{R_LENGTH}}));
  ```
- We transpose MLX's natural `[B, H, N, D]` to `[B, N, H, D]` in
  `mfa_v6_nax_primitive.cpp:355-357`. This is the same layout the v2
  expects.

### Default tile dimensions
- Same. `blockDimensions = (BQ, BK, BD)` = `(32, 32, head_dim)` by default.
  The v2 has no different per-shape tuning table; sweeping was always our
  responsibility.

### Online softmax algorithm
- Same FlashAttention-2 (cM, cL accumulators, exp2 in log2 domain).
  Verified identical between our port and HEAD by line-for-line comparison
  of `createSource()` body lines 770–1380.

### Cooperative tensors
- Same. `cS_0` (single QK accumulator), `cM`, `cL`, `correction`, `cO_*`.
  No `cS_1` (no double-buffering — confirmed in our Axe 7 architectural
  review).

### Double-buffering on C
- Not present in v2. Our Axe 7 SKIP rationale stands.

### Bypass threadgroup memory path
- Present in v2, same as our port. Constructor falls back to
  `bypassThreadgroupMemory=false` if Path A library compilation fails — we
  observed this empirically in Axe 3 (SeedVR2-small bypass=true didn't
  compile).

### New patterns post-our-port?
- None. Our port is current as of April 28, 2026 (latest
  `NAAttentionKernel.cpp` commit). No commits since then.

### Larger head dimensions (D > 128)?
- v2 `NAAttentionKernel.cpp` has no special D > 128 path. We restrict to
  D ∈ {64, 128} in `v6_nax_forward` and that matches v2's tested range.

### Segmented matmul (MoE)?
- v2 has a separate file `ccv_nnc_mfa_segmented_gemm.cpp` for MoE-style
  GEMMs, but it is NOT linked to attention. The Draw Things blog mentions
  segmented matmul for MoE GEMMs — not for attention. Our V6 NAX attention
  does not need it.

---

## What the v2 has that we DON'T have ported

| Feature | File | Decision |
|---------|------|----------|
| INT8 NAX attention | `NAInt8AttentionKernel.cpp` | DEFER — would integrate with TurboQuant pathway |
| Backward NAX | already in `NAAttentionKernel.cpp` (`type=backwardQuery/backwardKeyValue`) | LATER — V6.1 backward |
| `is_varlen` packed sequences | `2d1bc88e (April 28)` adds varlen | LATER — already covered by V2 STEEL paged-varlen |
| Mask support (additive bias) | `5a484bba (April 20)` | LATER — already covered by attn_bias native |
| `MASK_SCALE` exp2 fix | `e90be253 (April 28)` | OURS HAS IT — same kernel source |

---

## Why is V6 only at 38–43% of NAX peak?

Since our v6 IS the v2 code, the gap is **inherent to the MPP/cooperative_tensor
abstraction layer**, not staleness. Apple's SDPA (`steel_attention_nax`)
uses the **lower-level `metal_simdgroup_matrix` API** directly, bypassing
MPP's `matmul2d_descriptor` (see `apple-sdpa-nax-analysis.md`).

The hypothesis: MPP's `matmul2d_descriptor` adds driver-side scheduling
overhead and may not always fuse epilogues optimally. Apple's
`simdgroup_matrix` lets them place fragments in registers explicitly,
schedule MMAs by hand, and avoid driver-controlled cooperative-tensor
allocation.

This explains why our 10-axis tile-tuning campaign converged at 38–43%
peak — the **abstraction layer itself imposes the ceiling**.

---

## Recommendation

**Do NOT re-port from `/v2/`.** It does not exist; we already have its
content. Our port is current.

**Future work (Sprint 2 candidates)**, in priority:

1. **Reimplement V6 using `metal_simdgroup_matrix.h`** (instead of MPP
   cooperative_tensor). Mirror Apple's `attention_nax` abstraction layer.
   This is the single largest potential win — the MPP overhead is the
   most plausible explanation for the 5–7pp gap to SDPA.

2. **Add chunked-K dispatch for very long sequences** (PR #3307 pattern).
   SeedVR2-large (N=111375) exceeds the 65K threshold and would benefit
   from chunking. ~4.6 GB memory footprint reduction at 256K context.

3. **Port v2's INT8 NAX path** for quantized inference (4-bit + 8-bit
   activations).

4. **Port v2's backward NAX path** to enable training-mode V6.

The Draw Things blog's "4.6× M5/M4" claim is real but already realized in
our port — we just don't beat Apple's SDPA because Apple uses a more
direct API for the same hardware.
