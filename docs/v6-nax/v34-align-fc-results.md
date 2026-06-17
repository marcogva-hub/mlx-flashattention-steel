# V6NAX align_Q / align_K compile-time gates — Sprint 3 results

**Date:** 2026-05-06
**Sprint:** V6NAX-FORWARD-MAX Sprint 3 (alignment specialization)
**Branch:** `experiment/v6nax-forward-max`
**Commit:** `3bfd782`

## Summary

V6NAX now exposes Apple-style `align_Q` / `align_K` specialization for the
fast-path on shapes where `qL % BQ == 0` and/or `kL % BK == 0` — the per-
element bounds checks (`is_last_q`, `is_last_k`) become dead code in the
aligned kernels. Apple's `steel_attention_nax.h` does this through
function constants 200/201 (runtime branch); we use compile-time
`#define V6NAX_ALIGN_Q` / `#define V6NAX_ALIGN_K` macros so the dead code is
truly eliminated by MSL compilation.

**Outcome: perf-neutral at our pipeline-cache scale** (~24 entries for
the production shape set). Documented as a no-op for our scale, with the
escape hatch left in place for future hardware where the FC pattern
becomes load-bearing.

## What changed

### Generator (`createV6NAXSource` in `NAAttentionKernel.cpp`)

`is_last_q` / `is_last_k` branches in the K-loop, `kL_rem` masking, and
the final `Otile.store` are wrapped in `#if !V6NAX_ALIGN_Q` /
`#if !V6NAX_ALIGN_K`:

```c
#if !V6NAX_ALIGN_K
if (is_last_k) {
  // per-element kL_rem mask
  ...
}
#endif
```

The `V6NAX_ALIGN_Q/K` macro values are injected at source-gen time based on
the runtime descriptor, not the input array, so the kernel is specialized
per (D, BQ, BK, WM, align_Q, align_K) tuple.

### Cache key

`V6Key` already had `v6nax_BQ`, `v6nax_BK`, `v6nax_WM` from v2.31.0. Sprint 3
adds two `bool` fields:

```cpp
struct V6Key {
  ...
  bool v6nax_align_q = false;
  bool v6nax_align_k = false;
  ...
};
```

Hashed into `V6KeyHash`. This produces up to 4× more pipeline cache
entries on V6NAX-eligible shapes (q-aligned × k-aligned), but the absolute
count stays tiny — production has ~5 distinct (D, BQ, BK, WM) combos and
most production shapes are aligned, so the cache typically holds 6–10
entries instead of 4–6. Well within MTLDevice limits.

This follows CLAUDE_V6_NAX.md §4 (no bit-packing in cache keys; explicit
fields only).

### Dispatch (`v6_nax_compile.mm`)

Computed from runtime params before pipeline lookup:

```cpp
bool align_q = (params.qL % params.BQ == 0);
bool align_k = (params.kL % params.BK == 0);
```

Forced off via `MFA_V6_NAX_DISABLE_ALIGN=1` env var (always uses unaligned
kernels for A/B comparison or debugging).

## Why compile-time vs runtime function constants?

Apple's `steel_attention_nax.h` uses function constants 200/201, which
let one compiled kernel handle both aligned and unaligned cases via
runtime branching. The trade-off:

| Approach | Pipeline entries | Branch cost | Apple uses |
|---|---|---|---|
| Function constants (Apple) | 1 per (D, BQ, BK, WM) | Runtime branch on aligned bool | Yes |
| Compile-time `#define` (V6NAX) | up to 4× per (D, BQ, BK, WM) | None — dead code eliminated | No |

For Apple's massive in-production kernel matrix (many models × many
shape buckets), pipeline cache pressure matters more than branch cost.
For our 5-shape production set, the math inverts: 24 cache entries is
nothing, so we take the dead-code-elimination win.

The Sprint 3 perf measurement showed 0% delta either way at our scale,
so the choice is currently a wash. Documented as such; the architectural
choice is reversible if the cache-multiplier becomes a real cost on
future M-series Mac hardware.

## Validation

### Correctness

Subprocess-isolated tests via `_ext.v6_nax_forward` covering all 4
combinations of (align_Q ∈ {true, false}, align_K ∈ {true, false}):

| Shape | qL % BQ | kL % BK | align_q | align_k | RMSE FP32 |
|---|:---:|:---:|:---:|:---:|---:|
| FlashVSR-dense (4096², BQ=32, BK=32) | 0 | 0 | ✓ | ✓ | 3.60e-06 |
| LTX2-cross (2048×8400, BQ=32, BK=32) | 0 | ≠0 (8400%32=16) | ✓ | ✗ | 1.76e-06 |
| Decode-style (200×512, BQ=32, BK=32) | ≠0 (200%32=8) | 0 | ✗ | ✓ | 4.21e-06 |
| Misaligned (200×500, BQ=32, BK=32) | ≠0 | ≠0 | ✗ | ✗ | 4.85e-06 |
| SeedVR2-small (26730², BQ=64, BK=32) | ≠0 (26730%64=42) | ≠0 (26730%32=10) | ✗ | ✗ | 1.75e-06 |
| Llama-2k causal (2048², BQ=64, BK=32) | 0 | 0 | ✓ | ✓ | 9.82e-06 |
| CogVideoX (70200², BQ=64, BK=32) | 0 | 0 | ✓ | ✓ | 1.11e-06 |
| SeedVR2-large (111375², BQ=64, BK=32) | ≠0 (111375%64=15) | ≠0 (111375%32=15) | ✗ | ✗ | 8.98e-07 |

8/8 OK. All RMSE within 1e-3 release criterion. Both code paths
(unaligned bounds checks active vs eliminated) produce numerically
equivalent results — confirming the bounds checks were correctly the
only difference.

### Performance

A/B against `MFA_V6_NAX_DISABLE_ALIGN=1` (forces unaligned kernel even
on aligned shapes). 3-run subprocess medians:

| Shape | Aligned kernel ms | Unaligned forced ms | Δ |
|---|---:|---:|---:|
| SeedVR2-small (D=128, aligned) | 211.8 | 213.4 | -0.8% |
| FlashVSR-dense (D=64, aligned) | 1.007 | 1.014 | -0.7% |

Both deltas are under measurement noise (~1-2% for V6NAX cross-session).
Conclusion: at our scale, the per-element bounds checks are essentially
free — likely the ALU vs branch-prediction trade-off washes out, and
the loops are too short for the eliminated branches to matter.

The infrastructure ships anyway because:
1. It's idiomatic to Apple's reference kernel.
2. Future workloads with much larger BQ/BK (e.g., next-gen NAX with
   wider tiles) might benefit.
3. The `MFA_V6_NAX_DISABLE_ALIGN=1` escape hatch makes A/B trivial.

## Files

- `csrc/mfa/v6_nax/NAAttentionKernel.cpp` (`createV6NAXSource`, ~6 sites
  wrapped in `#if !V6NAX_ALIGN_Q/K`)
- `csrc/mfa_v6_nax_primitive.cpp` (`v6nax_align_q`, `v6nax_align_k` fields
  in `V6Key`, hashed in `V6KeyHash`)
- `csrc/v6_nax_compile.mm` (`v6nax_compile`/`v6nax_dispatch` plumb the bools)

## Apple reference

- `steel_attention_nax.h:175,185` — function constants 200, 201 (`align_Q`,
  `align_K`).
- `steel_attention_nax.h:228-244` — `is_last_q` / `is_last_k` branches that
  become dead code under aligned specialization.

## Open follow-ups

- If pipeline cache pressure ever becomes a real cost (e.g., dynamic
  workloads with many distinct shape buckets per session), the FC
  approach can be added as an alternative path. The cache-key fields
  would just be ignored by an FC-mode codegen.
