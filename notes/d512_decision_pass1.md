# D=512 Decision Pass — Matrix Refresh (Post Runtime-Unification)

Date: 2026-03-12  
Device: Apple M1 Max (gen=13, M3+=false, GPU cores=32)  
Version: mlx-mfa v2.9.2

Benchmark command:

```bash
.venv/bin/python benchmarks/bench_d512_decision_matrix.py \
  --warmup 2 --iters 5 \
  --output notes/d512_decision_matrix_latest.json
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
