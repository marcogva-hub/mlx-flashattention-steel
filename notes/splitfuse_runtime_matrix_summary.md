# Splitfuse Runtime Integration Matrix Summary

Source artifact: `notes/splitfuse_runtime_matrix_latest.json`  
Script: `benchmarks/bench_splitfuse_runtime.py`

## Measured Rows (M1 Max)

| D | helper ms | runtime splitfuse ms | runtime splitfuse_step ms | runtime splitfuse_step + prefix ms |
|---:|---:|---:|---:|---:|
| 64  | 0.774 | 0.774 | 0.680 | 0.976 |
| 128 | 0.788 | 0.826 | 0.739 | 1.058 |

## Interpretation

- Runtime-integrated `splitfuse_step(...)` is at/near helper parity in this
  focused matrix.
- Prefix-enabled splitfuse path adds measurable overhead vs cache-only step,
  as expected from additional prefill-side work.
- Primary outcome is runtime integration clarity and metadata visibility, not a
  broad throughput-promotion claim.
