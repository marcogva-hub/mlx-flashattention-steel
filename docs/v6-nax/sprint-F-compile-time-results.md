# Sprint F — compile-time vs runtime function constants

**Status:** **Investigated. Mostly already-implemented; remaining items have**
**low expected gain.**

## Current state — what's already compile-time

The V6 NAX kernel has **two** layers of compile-time specialization:

### Layer 1: source-generator template substitution

`csrc/mfa/v6_nax/NAAttentionKernel.cpp` substitutes `{{TOKEN}}` placeholders
into the MSL source before compilation. These become **compile-time
constants** in the generated source:

- `{{HEAD_DIMENSION}}` → e.g. `128` (literal int)
- `{{BLOCK_DIMENSIONS_PARALLELIZATION}}` → e.g. `16` (BQ)
- `{{BLOCK_DIMENSIONS_TRAVERSAL}}` → e.g. `32` (BK)
- `{{BLOCK_DIMENSIONS_HEAD}}` → e.g. `128` (BD)
- `{{H_HK_RATIO}}` → e.g. `/ 4` (GQA ratio)
- ... ~30 such tokens ...

Each unique combination produces a **separate compiled kernel** — caching
keyed on the shape parameters (see `V6Key` in `mfa_v6_nax_primitive.cpp`).

### Layer 2: Metal function constants

`csrc/v6_nax_compile.mm:38-46` sets these as Metal *function constants*
(`MTLFunctionConstantValues`), specialized at function-creation time:

- `R` (sequence length) at index 0
- `C` (KV length) at index 1
- `Q_bs`, `K_bs`, `V_bs`, `O_bs` (batch strides) at indices 2-5

These are compile-time within the Metal compilation but runtime from the
host. Effectively they're shape-specialized at function creation, not
dispatch.

## What COULD be moved to compile-time but isn't

Looking at the kernel source, the runtime-only quantities are:

1. **`tgid.x`, `tgid.y`, `tgid.z`** — threadgroup position. Inherently
   per-dispatch. Cannot be compile-time.
2. **`thread_index_in_threadgroup`, `simdgroup_index_in_threadgroup`** —
   per-thread/-simdgroup. Cannot be compile-time.
3. **Scalar inputs** like `qL_off` for causal attention. Currently
   computed at runtime. Could be a function constant if pre-computed
   on host.

Item 3 is the only candidate. But:

- `qL_off = N_kv - N_q` for square causal; already computed once at
  setup and used as `causal_column_offset`. The kernel reads it from a
  threadgroup-local `int` constant inside the source — already compile-time
  via R, C function constants.

So **there's nothing left to promote to compile-time** that isn't
already.

## What COULD be moved to runtime but isn't

- `BQ`, `BK`, `BD`, `executionSIMDGroups`, GQA ratio: **could** be
  function constants instead of source substitutions. This would unify
  the cache (one kernel for all tile sizes, function-constant-specialized).
  But:
  - Metal Performance Primitives `matmul2d_descriptor` requires
    **constexpr** dimensions for `cooperative_tensor` allocations.
    Moving BQ/BK to function constants breaks the matmul2d static_asserts.
  - Conversely, function constants can't appear in template parameters
    (`matmul2d<descriptor, execution_simdgroups<1>>`).
  - So tile dimensions **must** be source-time substitutions, not
    function constants.

Other items like `bypassThreadgroupMemory` and `singleOtileMode` flags
are **structurally compile-time** (they select different code paths in
the source generator). Promoting them to function constants would require
generating both code paths in every kernel and dispatching at runtime —
larger compiled binaries, no perf benefit.

## Decision

The V6 NAX kernel's compile-time/runtime split is already at its
natural boundary:
- Tile dimensions, GQA ratio, axis flags: **source-time substitution**
  (Layer 1) because MPP requires constexpr.
- Sequence lengths, batch strides: **Metal function constants**
  (Layer 2) for shape-class specialization.
- Thread positions, per-call shapes: **runtime**.

There's no remaining axis to swap between these layers. **Nothing
implemented, no test required.**

## What this section produces

- This document (architectural justification)
- No bench script
- No JSON data
- No code change

## Future-work item

If, during a future sprint, a specific runtime computation in the kernel
is identified as a measurable bottleneck (e.g. via Metal counters or
profiling), promote it to a function constant individually. As of
v2.30.0, no such bottleneck has been identified.
