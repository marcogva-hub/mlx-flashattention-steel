# D=256 Design Track (Pass 2) — Targeted Strategy Evaluation

Date: 2026-03-12  
Branch: `codex/d256-design-track`

## Strategy tested

1. **Separate BK selector for D=256/512 D-split family**
   - Added `select_steel_v2_dsplit_block_config(is_m3_plus)`
   - Decoupled from global `MFA_V2_FORCE_BK` (D=128 calibration override)
   - Added D=256-only override: `MFA_V2_FORCE_BK_D256=32|64`

## Why this strategy

D=256 path previously reused D=128 BK selection plumbing. This allowed a
D=128 calibration override (`MFA_V2_FORCE_BK=64`) to unintentionally force BK=64
for D=256 D-split on M1/M2, which is usually a regression.

## Benchmark evidence (separate processes, M1 Max, f16, causal, B=2 H=8)

### Before (pre-change behavior, global BK=64 effectively applied to D=256)

| N | SDPA / MFA ratio |
|---|---:|
| 4096 | 0.84x |
| 8192 | 0.84x |
| 16384 | 0.92x |

### After (post-change)

`MFA_V2_FORCE_BK=64` only (no D256 override):

| N | SDPA / MFA ratio |
|---|---:|
| 4096 | 1.01x |
| 8192 | 1.06x |
| 16384 | 1.12x |

`MFA_V2_FORCE_BK=64` + `MFA_V2_FORCE_BK_D256=64`:

| N | SDPA / MFA ratio |
|---|---:|
| 4096 | 0.85x |
| 8192 | 0.80x |
| 16384 | 0.86x |

Interpretation: the D=256 policy is now isolated. Global D=128 tuning no longer
accidentally degrades large-D behavior.

## Decision

Keep this strategy. It is narrow, low-risk, and materially improves D=256 policy
robustness without broad claims.
