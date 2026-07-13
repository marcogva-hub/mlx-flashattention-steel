# Benchmarking mlx-mfa

This guide defines the evidence required before a performance result is treated as current.

## Measurement contract

A comparative result is admissible only when all of the following are recorded:

1. both arms use the intended dtype and input geometry;
2. dispatch telemetry identifies two different terminal paths;
3. the candidate output agrees with an independent oracle;
4. each arm runs in a fresh process for five sustained sessions in both arm orders;
5. sub-millisecond cells use twenty dispatches per sample;
6. the report names MLX, mlx-mfa, macOS, and the Apple GPU generation;
7. an A-vs-A null experiment establishes the resolution floor for small effects.

Ratios without these fields are historical observations, not active claims.

## Standard commands

Use the repository environment explicitly:

```bash
/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python benchmarks/bench_final_routed_spots.py --help
/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python benchmarks/bench_sparse_gate_remap.py --help
```

The hardened spot harness refuses to emit a ratio if either arm lacks a terminal fingerprint. Benchmark JSON belongs under `benchmarks/results/`; reports should link the exact files used.

## Correctness first

Forward attention comparisons use an fp32 score/softmax oracle. Sparse and GNA comparisons must include the same mask or neighborhood relation in the oracle. Backward comparisons validate `dQ`, `dK`, and `dV` separately. Packed-varlen comparisons validate every segment as well as the concatenated output.

The default acceptance threshold is cosine similarity at least `0.999`. A stricter threshold or max-absolute bound may be required by a family lock.

## Reading a ratio

This project reports `baseline_time / candidate_time`; values above one favor mlx-mfa. Absolute milliseconds remain mandatory because a ratio can hide a thermally invalid or unexpectedly dispatched arm.

Do not transfer a result across dtype, sequence length, head count, mask family, operating-system build, or kernel terminal unless the new cell was measured.

## Current hardened evidence

The compact current table lives in [RESULTS.md](../../RESULTS.md). Dispatch conditions are independent of performance prose and are defined in [dispatch-map.md](dispatch-map.md).

## Thermal and process hygiene

Before a GPU run, verify memory pressure, other Python jobs, and GPU activity. Run benchmark arms in the foreground and isolate sessions by process. Discard a session when its absolute timings show sustained thermal drift; record every exclusion.

## Adding a benchmark

A new harness must emit machine-readable JSON containing shape, dtype, arm order, absolute samples, summary statistics, correctness metrics, terminal fingerprints, runtime versions, and hardware stamp. It must fail loudly when an engagement or correctness field is missing.
