# D=256 Design Track (Post-Backward) — Evidence Refresh

Date: 2026-03-12  
Branch: `codex/d256-design-track-post-bwd`  
Script: `benchmarks/bench_d256_design_matrix.py`  
Command: `/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python benchmarks/bench_d256_design_matrix.py --warmup 1 --iters 1 --output devnotes/d256_design_matrix_post_bwd_latest.json`

## Summary

- Device: Apple M1 Max (`is_m3_plus=false`)
- Cases: 32
- Classification (best MFA route vs SDPA):
  - `maybe_win`: 8
  - `neutral`: 0
  - `losing`: 24

## Region map (best MFA route vs SDPA)

| Profile | Dtype | Causal | N range | Outcome |
|---|---|---|---|---|
| `prod_b2h8` | `f16` | `False` | 2048..16384 | losing (`0.51x..0.55x`) |
| `prod_b2h8` | `f16` | `True` | 2048..16384 | maybe-win (`1.04x..1.19x`) |
| `prod_b2h8` | `bf16` | `False` | 2048..16384 | losing (`0.44x..0.48x`) |
| `prod_b2h8` | `bf16` | `True` | 2048..16384 | losing (`0.69x..0.89x`) |
| `under_b1h1` | `f16` | `False` | 2048..16384 | losing (`0.54x..0.95x`) |
| `under_b1h1` | `f16` | `True` | 2048..16384 | maybe-win (`1.10x..1.63x`) |
| `under_b1h1` | `bf16` | `False` | 2048..16384 | losing (`0.53x..0.85x`) |
| `under_b1h1` | `bf16` | `True` | 2048..16384 | losing (`0.72x..0.87x`) |

## Path notes

- D=256 split-K force toggles were benchmarked and remain non-composable / non-applicable
  in the current D=256 route design.
- Winning signal remains concentrated in **dense causal f16**, consistent with the current
  narrow production regime.
- bf16 remains SDPA territory in this matrix.
