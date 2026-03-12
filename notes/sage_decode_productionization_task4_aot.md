# Sage Decode AOT Decision (Task 4)

Date: 2026-03-12

## Decision

Defer Sage-specific AOT metallib coverage in this pass.

## Why

1. Benchmark-backed Sage auto regime is intentionally tiny (2 selected rows in the
   current matrix, both wins, but very narrow conditions).
2. Sage kernel key space includes multiple compile-time knobs (`D`, dtype,
   `causal`, `window`, `gqa_factor`, block config). Adding broad AOT now would
   grow artifacts and maintenance faster than practical benefit.
3. Existing AOT path already covers the primary production default (STEEL V2 and
   V2 D-split). Sage remains a specialized backend and still has JIT fallback.

## Revisit trigger

Revisit Sage AOT when either of the following is true:

- benchmark-backed Sage winning regimes expand beyond the current narrow decode
  windowed cases, or
- production telemetry shows cold-start JIT overhead is material for repeated
  short-lived Sage decode workloads.
