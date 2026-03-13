# Runtime Unification Microbenchmark (Task 5)

Date: 2026-03-12
Script: `benchmarks/bench_runtime_decode_overhead.py`
Artifact: `devnotes/runtime_unification_overhead_latest.json`

## Setup

- Separate-process run
- Dense decode path (production default)
- Shape: `B=1, H_q=4, H_kv=4, N_pre=64, D=64, steps=32, dtype=f16`

## Result

- Legacy context decode-loop mean: `23.02 ms`
- Unified runtime decode-loop mean: `22.81 ms`
- Ratio (`unified/legacy`): `0.991x`

Factory construction overhead:

- Legacy `create_inference_context`: `306.17 us`
- Unified `create_decode_runtime`: `311.94 us`
- Ratio (`unified/legacy`): `1.019x`

## Conclusion

- No decode-loop regression from runtime unification on this short-loop benchmark.
- Runtime wrapper adds negligible factory-time overhead while reducing
  duplicated routing/validation branches via shared inference helpers.
