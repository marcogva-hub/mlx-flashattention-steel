# Selective AOT Evaluation (Experimental Triage Pass)

Date: 2026-03-12
Device: Apple M1 Max

## Goal
Evaluate whether selective AOT expansion should be added for advanced runtime kernels.

Candidates evaluated (subprocess-isolated cold-start probes):
- `sage_decode_d128_gqa2`
- `paged_gather_d128`
- `paged_steel_d128`

## Method
Used `benchmarks/bench_experimental_triage.py --subprocess-mode coldstart`.
Compared:
1. JIT-only mode (`MFA_DISABLE_PRECOMPILED=1`)
2. Precompiled mode after `python -m mlx_mfa.compile_metallib --force`

## Results

| Candidate | JIT first-call (ms) | AOT first-call (ms) | Verdict |
|---|---:|---:|---|
| `sage_decode_d128_gqa2` | 6.33 | 152.74 | AOT regressed cold-start strongly |
| `paged_gather_d128` | 4.96 | 83.73 | AOT regressed cold-start strongly |
| `paged_steel_d128` | 121.72 | 198.81 | AOT regressed cold-start |

## Decision
Selective AOT expansion is **deferred** for this pass.

Rationale:
- Current precompiled loading path did not improve cold-start for targeted advanced kernels on this hardware.
- Shipping this AOT expansion would increase complexity while regressing startup latency.
- Keep advanced-kernel paths on JIT for now and revisit if loader behavior or artifact strategy changes.
