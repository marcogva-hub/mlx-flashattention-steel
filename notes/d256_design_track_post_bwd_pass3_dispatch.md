# D=256 Design Track (Post-Backward) — Auto Dispatch Tightening

Date: 2026-03-12  
Branch: `codex/d256-design-track-post-bwd`

## Policy status

D=256 auto routing remains narrow and benchmark-backed:
- `f16` + `causal=True` + `N>=4096` (M1/M2): MFA eligible
- `bf16`: SDPA default
- non-causal: SDPA default

Added debug override for D=256 auto path only:
- `MFA_FORCE_D256_PATH=1|mfa` -> force MFA
- `MFA_FORCE_D256_PATH=0|sdpa` -> force SDPA

## Spot benchmark (separate process, M1 Max, B=2 H=8, D=256, causal, N=4096)

| dtype | auto policy | SDPA/auto |
|---|---|---:|
| f16 | MFA | 1.054x |
| bf16 | SDPA | 0.973x |

## Override checks

- `MFA_FORCE_D256_PATH=0` -> `should_use_mfa(... D=256 ...) == False`
- `MFA_FORCE_D256_PATH=1` -> `should_use_mfa(... D=256 ...) == True`
