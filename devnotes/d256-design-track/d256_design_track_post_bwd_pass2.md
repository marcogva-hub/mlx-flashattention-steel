# D=256 Design Track (Post-Backward) — Targeted Strategy Evaluation

Date: 2026-03-12  
Branch: `codex/d256-design-track-post-bwd`

## Candidate strategies evaluated

1. **Earlier D=256 f16 causal promotion** (`N>=2048` instead of `N>=4096`).
2. **D=256 BK variant check** (`BK=32` vs `BK=64`) for D-split behavior consistency.

## Benchmarks (separate processes)

### Production-like profile (`B=2 H=8`, f16, causal)

| N | SDPA/MFA |
|---:|---:|
| 2048 | 0.92x–1.00x (unstable / parity) |
| 4096 | 1.02x–1.06x |
| 8192 | 1.06x |
| 16384 | 1.14x–1.17x |

Interpretation: `N=2048` is not robustly winning for production-like shapes.
`N>=4096` remains the narrow evidence-backed promotion boundary.

### bf16 check (`B=2 H=8`, causal)

| N | SDPA/MFA |
|---:|---:|
| 2048 | 0.66x |
| 4096 | 0.69x |
| 8192 | 0.70x |
| 16384 | 0.75x |

Interpretation: bf16 remains clearly SDPA territory.

### BK variant sanity (D-split family)

- `BK=64` remains a losing choice on M1 Max for D=256.
- Existing separated D=256 config path (defaulting to BK=32 on M1/M2) remains correct.

## Decision

No additional promotion beyond current policy.
Keep D=256 auto-dispatch narrow and benchmark-backed:
- promote only dense causal f16 at `N>=4096` on M1/M2,
- keep bf16/non-causal on SDPA,
- keep D=512 out of scope for this pass.
