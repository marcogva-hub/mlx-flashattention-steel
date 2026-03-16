# M3+ GPU Optimizations — Design

**Date**: 2026-03-16
**Target version**: v2.11.0
**Context**: M4 Max benchmarks show MFA parity/regression vs SDPA at D=64/128.
Root cause: M3+ reduced threadgroup memory bandwidth, hardcoded M1-era configs.

## Priority Order

### P2: Fix V2 arch_gen hardcoded to 13
- `mfa_steel_fwd_v2.cpp` lines 151, 813, 1494: `arch_gen = key.is_m3_plus ? 15 : 13`
- Add `#define ARCHITECTURE_GEN` to V2 Metal source (currently missing)
- Add pragma unroll conditional on arch_gen >= 15
- Expected impact: 3-8%

### P3: V5 BQ=32 on M3+
- `mfa_steel_fwd_v5.hpp` line 55: BQ=16 WM=2 → BQ=32 WM=4
- Dynamic register allocation on M3+ makes BQ=32 viable
- Expected impact: 10-15% on V5

### P4: V2 BQ64 bug
- Diagnose why V2 BQ64 produces ~0.15ms (no-op kernel)
- Fix grid dispatch or gate out unsupported config

### P1: Direct device reads in V2 (M3+)
- Port V5's `MFA_DIRECT_READS` pattern into V2 generator
- M3+: K/V from device pointers, skip KV_smem, eliminate barriers
- Q stays in threadgroup (loaded once, reused N/BK times)
- Gate with `#if MFA_DIRECT_READS` / `#if !MFA_DIRECT_READS`
- Expected impact: 15-25% on D=64/128 causal

## Testing
- 755 existing tests must pass after each change
- A/B bench on M1 Max (no regression)
- MFA_FORCE_GEN=15 for M3+ path testing on M1

## Non-goals
- Mixed-precision softmax (P5) — future version
- Texture buffer for K/V (P6) — requires MLX allocator changes
