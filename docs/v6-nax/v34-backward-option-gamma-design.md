# V34 backward Option γ — fused dK+dV with TGP cross-SG reduction (design)

Design analysis for the next-attempt sprint at V34 backward perf parity
vs SDPA-vjp.  Documented after v2.37.0 ship as Option β reached its
architectural floor at 2.4× SDPA-vjp.

## Motivation

V34 backward two-kernel split (Phase 2.O2 / v2.37.0) trades 2×
softmax replication for register relief.  Split totals at qL=8192 D=128:
- dV kernel: 9.2ms (light: QK^T + softmax + P^T@dO)
- dK kernel: 21.0ms (heavy: + dO@V^T + dS + dK GEMM)
- Sum: 30.2ms

Fused dK+dV could share the softmax compute (~4-5ms savings), bringing
combined to ~22-25ms.  But requires holding BOTH accumulators
simultaneously, which exceeds the M5 register file at WM=4 with the
two-kernel split's BK=32.

## Register budget analysis

Per-SG register state for fused dK+dV WM=4 Q-row partition:

| Component | Size per SG |
|---|---:|
| dK_accum (FP32) | BK × D × 4 = 16 KB at BK=32, 8 KB at BK=16 |
| dV_accum (FP32) | Same as dK_accum |
| Stile (FP32, BQ/WM × BK) | 16 × BK × 4 / 32 lanes |
| dPtile (FP32) | Same as Stile |
| Qtile transient (FP16, BQ/WM × D) | 16 × D × 2 / 32 lanes |
| dOtile transient (FP16) | Same as Qtile |
| K/V single-frag loads | ~512 bytes |

For BQ=64 BK=32 D=128 WM=4:
- dK + dV: 32 KB
- Stile + dPtile: 4 KB
- Q + dO retained: 16 KB
- Total: ~52 KB → spill (M5 limit ~32 KB per SG)

For BQ=64 BK=16 D=128 WM=4:
- dK + dV: 16 KB
- Stile + dPtile: 2 KB
- Q + dO retained: 16 KB
- Total: ~34 KB → over by ~2 KB, likely spill

For BQ=64 BK=16 D=128 WM=4 + WITHOUT retaining Q/dO:
- dK + dV: 16 KB
- Stile + dPtile: 2 KB
- Transient single-frag loads: ~2 KB peak
- Total: ~20 KB → fits comfortably

The "WITHOUT retain" path reloads Q + dO per inner-iter, costing ~3×
device-memory bandwidth on those tiles.  Worth investigating empirically
whether the bandwidth overhead < softmax replication savings.

## TGP cross-SG reduction

Each SG produces FULL BK × D dK + dV partials (per-SG Q-row
contributions to all BK rows).  At end of Q-loop:
- Per-SG dK + dV: 8 KB × 2 = 16 KB at BK=16
- 4 SGs total: 64 KB needs reduction

Cannot fit 64 KB in 32 KB TGP at once.  Streaming row-by-row:
- Per row r: 4 SGs × D × FP32 = 4 × 128 × 4 = 2 KB in TGP
- Iterate BK rows: BK=16 → 16 row iterations per dispatch

Per iteration: 4 SGs write to TGP, barrier, SG0 reads + sums + writes to
device, barrier.  Overhead ~50-100µs per K-tile.

For qL=8192 NK=512: 512 × 100µs = 51ms total reduction overhead.
HIGHER than the savings from softmax fusion.

So TGP streaming reduction is NOT viable on M5 for fused dK+dV.

## Alternative: per-SG-slot device write + mx.sum

Same pattern as Phase 2.O2 dV/dK split — each SG writes its partial to
a unique slot in dK_partials [B, Hq, WM, kL, D] FP32 + dV_partials
[same shape] FP32.  Python wrapper reduces via mx.sum + cast.

Memory overhead: 2× (dK + dV) × WM intermediate buffers.  At qL=8192
B=1 Hq=4 D=128: 2 × 4 × 4 × 8192 × 128 × 4B = 135 MB temporary.  Big
but feasible on M5 Max 128GB unified memory.

mx.sum cost: ~0.5ms × 2 (dK + dV) = 1ms additional Python-side cost.

This is the SAME pattern as the split kernels.  The fused kernel
would just do less work overall (one Q-loop instead of two).

Expected fused dK+dV perf at BK=16 WM=4 (without Q/dO retain):
- Per Q-iter: ~2-3× current dV iter cost (because adds dP + dS + dK
  GEMM on top of dV's work)
- Total dK+dV: ~2-3× current dV = 18-28ms at qL=8192
- vs current split: 30.2ms

Estimated savings: 2-12ms (7-40% speedup).  Final V34 backward:
~36-46ms vs current 49ms.  Closer to SDPA-vjp 20ms but still ~2× away.

## Verdict

Fused dK+dV with per-SG-slot output + Python mx.sum reduction is
implementation-feasible and may deliver 7-40% speedup over the
current Phase 2.O2 two-kernel split.  Worth a follow-up sprint of
~1-2 days CC.

NOT a path to perf parity with SDPA-vjp (still ~2× slower estimated).
True parity requires reverse-engineering Apple's SDPA-vjp algorithm
(likely a different mathematical formulation or fused-pass design
specific to M5 hardware capabilities).

## Implementation plan for Option γ sprint

### Phase γ.1: Source-gen fused kernel
- Write `createV34BackwardDKDVFusedSource()` ~700 LOC
- Q-row partition WM=4 BQ=64 BK=16
- Per Q-iter: load Q + dO (transient, not retained), compute D, S, P,
  dP, dS, dV_accum += P^T@dO, dK_accum += dS^T@Q
- Per-SG store dK + dV partials to unique slots

### Phase γ.2: Primitive + binding + dispatch helper

### Phase γ.3: Integration via flash_attention VJP
- Add MFA_V34BWD_USE_FUSED_GAMMA=1 env var
- Default off until perf validated

### Phase γ.4: Bench + correctness tests
- Canonical methodology if applicable
- Three-axis validation
- Compare vs current split: must beat 30.2ms at qL=8192 to be worth shipping

### Phase γ.5: Ship as SHIP_OPT_IN if validated; otherwise document falsification

## References

- `docs/v6-nax/v34-backward-status.md` — full sprint timeline + Option β floor
- `docs/v6-nax/v34-backward-decisions.md` — DC0-DC13 design rationale
- CHANGELOG.md [2.37.0] — Option β ship + perf data
- This document: Option γ design analysis for follow-up sprint
