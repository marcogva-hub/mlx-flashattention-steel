# Paged / Shared-Prefix / Splitfuse Runtime Matrix (v2.9.2)

Date: 2026-03-12  
Device: Apple M1 Max (gen=13, GPU cores=32)  
Command:

```bash
.venv/bin/python benchmarks/bench_paged_sharedprefix_matrix.py \
  --warmup 3 --iters 10 \
  --output devnotes/paged_sharedprefix_matrix_latest.json
```

Profile:
- `B=1, H_q=8, H_kv=4` (GQA 2:1)
- `D in {64, 128}`
- decode `N_q in {1,2,4}`
- cache `N_cache in {1024,2048,4096,8192,16384}`
- causal decode paths

## Summary counts

| Family | clear_win | maybe_win | no_win | losing |
|---|---:|---:|---:|---:|
| paged_step (`flash_attention_paged` vs dense `flash_attention_kvcache`) | 0 | 1 | 1 | 28 |
| paged_setup (paged runtime prefill vs dense runtime prefill) | 0 | 0 | 0 | 10 |
| shared_prefix (`make_shared_prefix_cache` reuse flow) | 4 | 0 | 3 | 1 |
| splitfuse (`flash_attention_splitfuse` vs separate prefill+decode calls) | 3 | 0 | 0 | 5 |

## Key observations

1. **Paged decode is not benchmark-backed for auto promotion in this matrix.**
   - `paged_step` is mostly losing (`28/30` losing, best only `1.04x`).
   - `paged_setup` is always losing (`10/10`).
   - The single `1.04x` row is weak and near noise floor.

2. **Shared-prefix shows selective wins when prefix reuse is real.**
   - Strongest rows are `D=128`, `N_prefix=2048`, reuse `2-4` (`1.42x-1.62x`).
   - Some rows are neutral/losing, so it should remain an explicit optimization.

3. **Splitfuse is mixed and shape-sensitive.**
   - Some `N_q=4` rows show small wins (~`1.05x-1.08x`).
   - Several rows lose, especially with `D=128` and shorter prefill.

## Policy signal for this pass

- Keep paged decode **explicit-only** in auto mode (no broad auto-route).
- Continue exposing shared-prefix and splitfuse via runtime helpers, with docs
  emphasizing that wins are workload-dependent.
