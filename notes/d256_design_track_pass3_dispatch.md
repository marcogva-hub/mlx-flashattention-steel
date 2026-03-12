# D=256 Design Track (Pass 3) — Auto Dispatch Narrowing

Date: 2026-03-12  
Branch: `codex/d256-design-track`

## Policy change

`should_use_mfa()` now accepts `dtype` and applies a D=256-specific narrow rule
for dense causal on M1/M2:

- `dtype=float16`: MFA from `N >= 4096`
- `dtype=bfloat16`: keep SDPA
- no dtype provided: keep conservative fallback (`N >= 8192`)
- M3+ remains conservative until measured on real M3/M4 hardware

## Validation benchmark (separate process, M1 Max, B=2 H=8, causal=True)

| Dtype | N | Policy | SDPA (ms) | Auto (ms) | SDPA/Auto |
|---|---:|---|---:|---:|---:|
| f16 | 4096 | MFA | 38.49 | 38.24 | 1.01x |
| f16 | 8192 | MFA | 153.08 | 144.26 | 1.06x |
| f16 | 16384 | MFA | 653.99 | 564.79 | 1.16x |
| bf16 | 4096 | SDPA | 43.97 | 46.05 | 0.95x |
| bf16 | 8192 | SDPA | 177.33 | 176.88 | 1.00x |
| bf16 | 16384 | SDPA | 728.42 | 735.21 | 0.99x |

## Decision

Promote D=256 only in this narrow regime (`f16`, causal, M1/M2, `N>=4096`).
Keep bf16 and all non-causal D=256 on SDPA by default.
