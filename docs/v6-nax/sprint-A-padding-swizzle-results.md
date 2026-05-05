# Sprint A — ld_padding + swizzle investigation + tgmem-allocation cleanup

**Date:** 2026-05-05 (v2.30 deferred-sprints session)
**Status:** **A.1 (tgmem cleanup) shipped. A.2 (swizzle) skipped on Apple-evidence. A.3 (ld_padding) skipped on architectural grounds.**

## A.0 — Investigation findings (the brief was over-spec'd)

The v2.29.0 retrospective brief proposed sweeping `ld_padding` (bank-conflict
padding on Q/K/V) and `swizzle_log` (block-reorder for cache locality)
because Apple's GEMM kernel uses both. Investigating Apple's *attention*
kernel (the one we're competing with — `steel_attention_nax.h`) reveals:

```
$ grep -n "swizzle\|tid\.x\|tid\.y" steel_attention_nax.h
102:  ulong3 tidl{tid.x, tid.y, tid.z};
108:  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
180:    int q_max = (tid.x + 1) * BQ + params->qL_off;
189:  const bool is_last_bq = int(tid.x) == (params->NQ_aligned);
282:      const int base_row = tid.x * BQ + params->qL_off + tm;
310:      const int base_row = tid.x * BQ + tm;
```

Apple's NAX attention kernel uses **raw `tid.x` / `tid.y` / `tid.z`** —
*no* swizzle. Apple themselves don't apply swizzle to attention. The
swizzle pattern is in their GEMM (`steel/gemm/transforms.h`) and conv
kernels but not attention.

The reason is the access pattern: GEMM has cross-threadgroup tile
overlap on both A and B operands (which swizzle exploits for L2 reuse).
Attention has *no* cross-threadgroup overlap on Q (each Q-tile is
private to its threadgroup) and the K/V are streamed sequentially —
already cache-friendly without swizzle. **A.2 dropped on Apple-evidence.**

For ld_padding: in V6 NAX single-Otile, Q/K/V live in *device tensors*
(via MPP `tensor<device T, dextents<int32_t,2>, tensor_inline>`),
**not** threadgroup-staged buffers. Bank-conflict padding applies to
threadgroup memory; without threadgroup-staged Q/K/V, padding has
nothing to apply to. Apple's NAX attention kernel similarly uses
device tensors. **A.3 dropped on architectural grounds.**

## A.1 — Real win: skip tgmem allocation when single-Otile + bypass

Investigating threadgroup memory usage in the merged single-Otile path
revealed that `NAAttentionKernel::threadgroupMemoryAllocation()` returns
`BQ × BK × executionSIMDGroups × sizeof(O)` bytes for the forward path
*regardless* of whether the kernel actually needs threadgroup memory.

For default v2.29.0 D=128 N≥50k config (BQ=16, BK=32, SG=16, FP16):
**16 × 32 × 16 × 2 = 16 KB wasted per threadgroup.**

Apple's threadgroup memory limit is 32 KB on M5 Max. Wasting 16 KB
halves the threadgroup co-residency floor on the GPU.

### Fix

`csrc/mfa/v6_nax/NAAttentionKernel.cpp:56-66`:

```cpp
unsigned short NAAttentionKernel::threadgroupMemoryAllocation() const noexcept {
  if (type.value == AttentionKernelType::forward) {
    // Sprint A.1 — single-Otile + bypass forward kernel never uses P_buf.
    if (singleOtileMode && bypassThreadgroupMemory) {
      return 0;
    }
    // ... pre-existing path ...
  }
  // ...
}
```

3-LOC change. Both flags are already set together in single-Otile dispatch
(single-Otile forces bypass on by design).

### Bench (multi-run 5×, M5 Max)

| Shape | Pre-fix (v2.29.0 + S3.6) | Post-fix | Δ |
|---|---:|---:|---:|
| FlashVSR-dense (D=64)        | 1.11 ms   | 1.13 ms   | +1.57 % (noise) |
| LTX2-cross (D=64)            | 1.59 ms   | 1.56 ms   | -2.08 % (noise) |
| SeedVR2-small (D=128 small N)| 290.01 ms | 290.05 ms | +0.01 % (noise) |
| **CogVideoX (D=128 large N)**    | 3349 ms   | **3227 ms**   | **-3.64 %** |
| **SeedVR2-large (D=128 long N)** | 7244 ms   | **7062 ms**   | **-2.52 %** |

The two long-N D=128 shapes show consistent (if modest) wins. The wins
are below the multi-run shipping threshold (15% delta per Marco's rules)
but the direction is consistent and the change is a pure code-quality
fix (don't allocate what you don't use).

### Decision: ship the cleanup

Even at 0% gain, allocating unused memory is a bug. The 2-3.6% gains
on the production-priority long-N shapes are bonus. **Shipped.**

## Files

- `csrc/mfa/v6_nax/NAAttentionKernel.cpp` — 3-LOC tgmem allocation skip
- `bench/v6_sprint_a_tgmem_fix_bench.py` — multi-run bench
- `docs/v6-nax/sprint-A-tgmem-fix-bench.json` — raw timings
- `docs/v6-nax/sprint-A-padding-swizzle-results.md` — this file

## What didn't ship and why

- **A.2 swizzle** — Apple's `steel_attention_nax.h` doesn't use swizzle.
  Replicating GEMM-style swizzle would take ~50-100 LOC of source-gen
  modification with zero expected gain (Apple's empirical evidence
  applies to our access pattern too).
- **A.3 ld_padding** — V6 NAX uses device tensors for Q/K/V, not
  threadgroup-staged. Bank-conflict padding is for threadgroup memory.
  Architecturally inapplicable.

## Lessons logged

- **Read the target before implementing.** The brief assumed Apple's
  GEMM optimizations apply to attention; checking `steel_attention_nax.h`
  shows they don't. 5 minutes of grep saved 4-8 hours of dead-end
  source-gen extension.
- **Code-quality fixes ship even at 0% gain.** Allocating unused memory
  is a bug regardless of whether the GPU's occupancy floor exposes it
  on this specific hardware.
