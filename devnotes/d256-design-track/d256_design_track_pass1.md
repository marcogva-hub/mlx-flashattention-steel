# D=256 Design Track (Pass 1) — Matrix Results

Date: 2026-03-12  
Branch: `codex/d256-design-track`  
Script: `benchmarks/bench_d256_design_matrix.py`  
Command: `/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python benchmarks/bench_d256_design_matrix.py --warmup 1 --iters 1`

## Summary

- Device: Apple M1 Max (`is_m3_plus=false`)
- Cases: 32
- Classification (best MFA route vs SDPA):
  - `maybe_win`: 6
  - `neutral`: 3
  - `losing`: 23

## Region table (best MFA ratio vs SDPA)

| Profile | Dtype | Causal | N range | Outcome |
|---|---|---|---|---|
| `prod_b2h8` | `f16` | `False` | 2048..16384 | losing (`0.52x..0.56x`) |
| `prod_b2h8` | `f16` | `True` | 2048 | losing (`0.90x`) |
| `prod_b2h8` | `f16` | `True` | 4096..16384 | maybe-win (`1.04x..1.15x`) |
| `prod_b2h8` | `bf16` | `False` | 2048..16384 | losing (`0.41x..0.46x`) |
| `prod_b2h8` | `bf16` | `True` | 2048..8192 | losing (`0.67x..0.87x`) |
| `prod_b2h8` | `bf16` | `True` | 16384 | neutral (`0.96x`) |
| `under_b1h1` | `f16` | `False` | 2048..16384 | losing (`0.54x..0.94x`) |
| `under_b1h1` | `f16` | `True` | 2048 | losing (`0.87x`) |
| `under_b1h1` | `f16` | `True` | 4096..16384 | maybe-win (`1.23x..1.27x`) |
| `under_b1h1` | `bf16` | `False` | 2048..16384 | losing (`0.34x..0.60x`) |
| `under_b1h1` | `bf16` | `True` | 2048, 8192 | losing (`0.83x..0.91x`) |
| `under_b1h1` | `bf16` | `True` | 4096, 16384 | neutral (`0.95x..0.96x`) |

## Route hints from this pass

- D=256 split-K force toggles are effectively no-op in these runs (as expected by current eligibility logic).
- Strongest narrow signal is `f16 + causal + long N`.
- For winning rows, `MFA_V2_FORCE_BK=32` often matched or beat default BK on M1 Max.

## Immediate implication for next tasks

- Keep D=256 policy narrow and explicit.
- Evaluate at most two candidates:
  1. D=256 causal `f16` threshold tuning around `N>=4096`.
  2. D=256 BK selection (`default` vs forced `BK=32`) in causal long-N.
