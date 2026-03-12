# D=256 Dispatch Decision (2026-03-12)

## Setup
- Device: Apple M1 Max (gen 13)
- Dtype: `float16`
- Shape family: `B=2, H=8, D=256`
- Benchmark script: `benchmarks/bench_d256_decision.py`
- Raw JSON: `notes/d256_decision_latest.json`

## Measured Results

| N | causal | SDPA ms | V2 D-split ms | V2/SDPA |
|---:|:------:|--------:|--------------:|--------:|
| 4096  | ✅ | 36.55 | 37.35 | 0.98x |
| 8192  | ✅ | 143.24 | 141.78 | 1.01x |
| 16384 | ✅ | 685.77 | 578.13 | 1.19x |
| 4096  | ❌ | 36.56 | 66.66 | 0.55x |
| 8192  | ❌ | 144.52 | 267.52 | 0.54x |
| 16384 | ❌ | 611.60 | 1108.43 | 0.55x |

Additional focused reruns (higher iterations) confirmed:
- `N=1024 causal`: `0.73x` (V2 loses)
- `N=2048 causal`: `0.94x` (V2 loses)
- `N=8192 causal`: `1.11x` (V2 wins)
- `N=16384 causal`: `1.14x` (V2 wins)

## Decision
- Promote a **narrow winning regime** in auto dispatch:
  - `D=256`, `causal=True`, `N >= 8192` => MFA V2 D-split
- Keep SDPA default for:
  - `D=256`, `causal=True`, `N < 8192`
  - all `D=256`, `causal=False`
  - all `D=512` (unchanged)

This keeps the promotion evidence-based and limits risk outside the measured win region.
