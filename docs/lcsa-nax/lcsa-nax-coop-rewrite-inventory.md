# Sprint B follow-on (coop rewrite) — inventory

## Foundation

- master tip: `3a5751f` (Sprint B §4-validated)
- v2.34.0 live on PyPI + GitHub
- 33/33 LCSA + integration tests pass (pre-rewrite baseline)
- V1 architecture: per-thread-Q-row FA-2 (register math, no matmul2d)

## New artifacts planned this sprint

| File | Section | Purpose |
|---|---|---|
| `docs/lcsa-nax/lcsa-nax-design.md` §13 | A | Architecture v2 design |
| `docs/lcsa-nax/lcsa-nax-coop-rewrite-decisions.md` | A | DC0-DC8 decisions log |
| `docs/lcsa-nax/lcsa-nax-coop-rewrite-inventory.md` | A | This file |
| `csrc/mfa_sparse_attention.cpp` (edit) | B | Add `sparse_kernel_source_v2()` + dispatch |
| `csrc/mfa_sparse_attention.hpp` (edit) | B | Optional: expose version-selector helper |
| `mlx_mfa/lcsa_nax.py` (edit) | B | Optional: expose kernel-version env knob |
| `tests/test_lcsa_nax_coop_v2.py` | C | V1↔V2 equivalence + three-axis V2 |
| `bench/lcsa_nax_coop_rewrite_harness.py` | D | §4-strict 3-session sweep harness |
| `bench/lcsa_nax_coop_density_sweep.py` | D | Density 0.01→0.50 characterization |
| `docs/lcsa-nax/lcsa-nax-coop-rewrite-results.md` | D | §4-validated analysis output |
| `docs/lcsa-nax/lcsa-nax-coop-rewrite-data.json` | D | Raw bench data |
| `docs/lcsa-nax/coop-runlog-S{1,2,3}.txt` | D | Per-session stdout |
| `devnotes/SESSION_LOG.md` entry | A/F | Sprint log |

## Pre-existing artifacts referenced (unchanged)

| File | Used for |
|---|---|
| `csrc/mfa/v6_nax/NAAttentionKernel.cpp:2307-3671` | V34 forward source-gen reference (`createV34Source()`) |
| `csrc/mfa_v6_nax_primitive.cpp` | V34 forward dispatch pattern reference |
| `csrc/mfa_sparse_attention.{hpp,cpp}` (V1 unchanged) | Fallback path |
| `mlx_mfa/lcsa_nax.py` | Python API (no surface change) |
| `mlx_mfa/integrations/flashvsr_lcsa.py` | Patcher (transparent to V1/V2 switch) |
| `docs/lcsa-nax/lcsa-nax-phase1_5-ship-verdict.md` | v2.34.0 §4-validated baseline |
| `docs/lcsa-nax/lcsa-nax-rebench-results.md` | Variance source diagnostic motivating rewrite |
| `bench/lcsa_nax_phase1_5_harness.py` | §4-compliant harness template |
| `bench/lcsa_nax_rebench_analysis.py` | Cross-session analysis template |

## Shape inventory (Section D)

Same 7 shapes as Sprint B §4 rebench + density sweep on niche shape:

| Shape | qL | kL | D | density | BT (V1) | BT (V2) | class |
|---|---:|---:|---:|---:|---:|---:|:--|
| lcsa_small_seq4k          |  4096 |  4096 | 128 | 0.24 | 16 | 32 | moderate |
| lcsa_small_seq4k_sparse   |  4096 |  4096 | 128 | 0.07 | 16 | 32 | moderate |
| lcsa_mid_seq8k            |  8192 |  8192 | 128 | 0.12 | 16 | 32 | moderate |
| lcsa_mid_seq8k_sparse     |  8192 |  8192 | 128 | 0.03 | 16 | 32 | boundary |
| lcsa_large_seq16k         | 16384 | 16384 | 128 | 0.12 | 16 | 32 | moderate |
| lcsa_large_seq16k_sparse  | 16384 | 16384 | 128 | 0.03 | 16 | 32 | boundary |
| lcsa_mid_seq8k_very_sparse|  8192 |  8192 | 128 | 0.01 | 16 | 32 | niche |

Density sweep on `lcsa_mid_seq8k` (D=128, qL=kL=8192): density ∈ {0.01,
0.03, 0.05, 0.10, 0.20, 0.50}. Maps the new break-even point.

## Hardware + environment

- M5 Max 128GB, macOS 26.5, iStat performance fan profile
- MLX: 0.31.2, mlx_mfa: 2.34.0 (pre-rewrite baseline)
- Python: 3.11.14 (.venv)

## Sprint phases this session vs deferred

| Section | Status (this session) | Notes |
|---|---|---|
| A — Design + decisions + inventory | EXECUTING | This commit batch |
| B-scaffold — Primitive dispatch + cache key + stub source-gen (compiles, no-op V2) | EXECUTING | Following commit batch |
| B-kernel-body — Lift V34 cooperative-tensor pattern + sparse outer loop | DEFERRED | Multi-hour focused work; next session |
| C — V1↔V2 equivalence + three-axis V2 tests | DEFERRED | Depends on B-kernel-body |
| D — §4 perf sweep + density sweep + ship/shelve verdict | DEFERRED | Depends on C |
| E (cond.) — v2.35.0 release flow | DEFERRED | Depends on D verdict |
