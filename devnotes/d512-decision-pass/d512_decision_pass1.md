# D=512 Decision Pass — Matrix Refresh (Post Runtime-Unification)

Date: 2026-03-12  
Device: Apple M1 Max (gen=13, M3+=false, GPU cores=32)  
Version: mlx-mfa v2.9.2

Benchmark command:

```bash
.venv/bin/python benchmarks/bench_d512_decision_matrix.py \
  --warmup 2 --iters 5 \
  --output devnotes/d512_decision_matrix_latest.json
```

Scope:
- D=512 only
- N in {1024, 2048, 4096, 8192}
- causal in {False, True}
- dtype in {f16, bf16}
- profiles:
  - `prod_b2h8` (production-like)
  - `under_b1h1` (under-occupied)
- compared routes:
  - `sdpa`
  - `mfa_v1` (`MFA_DISABLE_V2=1`)
  - `mfa_v2_dsplit` (default MFA route for D=512)
  - `mfa_v5_optin` (`MFA_ENABLE_V5=1`)
  - `auto`

## Summary

Classification counts (best MFA route vs SDPA):
- maybe_win: 0
- no_win: 0
- losing: 32

Global maxima:
- max(best MFA/SDPA): **0.813x**
- max(V2 D-split/SDPA): **0.749x**
- auto routed to MFA: **0 / 32 rows**

Per-family best observed MFA/SDPA ratios:

| Profile | Dtype | Causal | Best MFA/SDPA |
|---|---|---:|---:|
| prod_b2h8 | f16 | False | 0.393x |
| prod_b2h8 | f16 | True  | 0.733x |
| prod_b2h8 | bf16 | False | 0.291x |
| prod_b2h8 | bf16 | True  | 0.590x |
| under_b1h1 | f16 | False | 0.813x |
| under_b1h1 | f16 | True  | 0.687x |
| under_b1h1 | bf16 | False | 0.658x |
| under_b1h1 | bf16 | True  | 0.678x |

## Decision signal from Task 1

- No D=512 regime beat SDPA in this matrix.
- Current `backend="auto"` behavior (stay on SDPA for dense D=512) matches benchmark evidence.
- `MFA_ENABLE_V5=1` does not provide a benchmark-backed D=512 win; D=512 remains a conservative SDPA family for production auto-dispatch.

## Task 3 narrow candidate check

Candidate tried: force D-split BK=64 (`MFA_V2_FORCE_BK_D256=64`) for D=512.

Focused checks (warmup=3, iters=10, separate processes):

| Shape | SDPA ms | V2 default ms | V2 BK64 ms | Best MFA/SDPA |
|---|---:|---:|---:|---:|
| B=2 H=8 N=4096 causal f16 | 74.0 | 100.9 | 98.9 | 0.75x |
| B=2 H=8 N=8192 causal f16 | 288.0 | 387.4 | 373.1 | 0.77x |
| B=1 H=1 N=1024 non-causal f16 | 1.38 | 1.59 | 1.89 | 0.87x |
| B=1 H=1 N=2048 non-causal f16 | 1.84 | 3.67 | 3.92 | 0.50x |

Conclusion: BK override improves some rows slightly but never reaches parity.
No narrow benchmark-backed D=512 win was found.
