# Sprint E — Threadgroup memory + max_total_threads sweep

**Status:** **Investigated. Implementation deferred.**

## Investigation

The pipeline state in `csrc/v6_nax_compile.mm:57-58` is created via the
basic `newComputePipelineStateWithFunction:error:`:

```objc
id<MTLComputePipelineState> pipeline =
    [device newComputePipelineStateWithFunction:function error:&error];
```

This uses Metal defaults for `maxTotalThreadsPerThreadgroup` (typically
1024 on M5 Max). To explicitly set it, the API requires
`MTLComputePipelineDescriptor`:

```objc
MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
desc.computeFunction = function;
desc.maxTotalThreadsPerThreadgroup = N;  // explicit cap
id<MTLComputePipelineState> pipeline =
    [device newComputePipelineStateWithDescriptor:desc options:0
                                       reflection:nil error:&error];
```

Effort: ~30-50 LOC refactor of `v6_nax_compile.mm` plus env var or
parameter plumbing.

## Why this is low-priority for v2.30

The `max_total_threads_per_threadgroup` attribute lets the compiler
emit *fewer* registers per thread when threads-per-threadgroup is
guaranteed to be small (allowing more threadgroup co-residency at the
cost of more registers per thread). For our use case:

- Default v2.29.0 + S3.6 D=128 N≥50k config: `executionSIMDGroups=16`
  → 512 threads per threadgroup. Far below the 1024 default cap.
- Explicit cap to 512 may help — it tells the compiler "these threadgroups
  will be at most 512 threads, optimize for higher per-thread register
  pressure / lower threadgroup count".

But the **single-Otile** path is **register-pressure-bound**, not
threadgroup-co-residency-bound (Sprint A.1's measurements showed
allocating an unused 16KB tgmem only saved 2-3.6 % on the slowest
shapes). Increasing per-thread register count via lower
`max_total_threads_per_threadgroup` is unlikely to help further;
single-Otile already uses few registers per thread by design (no kBlocks
split).

The `num_simdgroups_per_tg` axis is what the existing `MFA_V6_EXEC_SG`
env var controls — densely swept in S3.1 and S3.6 already.

## Implementation sketch (for future sprint)

```cpp
// v6_nax_compile.mm — extend signature with max_threads_hint
void* v6_nax_compile_with_constants(
    const std::string& source, const std::string& function_name,
    void* raw_device,
    uint32_t R, uint32_t C, uint32_t Q_bs, uint32_t K_bs,
    uint32_t V_bs, uint32_t O_bs,
    uint16_t max_threads_per_tg /* 0 = default */) {
  // ... function specialization (unchanged) ...
  if (max_threads_per_tg > 0) {
    MTLComputePipelineDescriptor* desc = [...];
    desc.computeFunction = function;
    desc.maxTotalThreadsPerThreadgroup = max_threads_per_tg;
    pipeline = [device newComputePipelineStateWithDescriptor:desc
                                                     options:0
                                                  reflection:nil
                                                       error:&error];
  } else {
    pipeline = [device newComputePipelineStateWithFunction:function
                                                     error:&error];
  }
  // ...
}
```

Plus env var `MFA_V6_MAX_THREADS_PER_TG` and pipeline cache key bit.

## Decision

Sprint E pipeline-attribute exploration is deferred until a focused
profiling investigation identifies threadgroup co-residency as the
specific bottleneck on a specific shape. As of v2.29.0 + Sprint A.1
+ Sprint B + (forthcoming) Sprint C, the bottleneck has not been
quantitatively localized to threadgroup attributes.

## What this section produces

- This document
- No bench script
- No JSON data
- No code change
