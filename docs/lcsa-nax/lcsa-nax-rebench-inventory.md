# Sprint B §4 re-bench — inventory

## Foundation

- master tip: `c0c77ee` (Sprint B closed)
- v2.34.0 live on PyPI + GitHub
- Pre-bench LCSA + integration test suite: 33/33 pass

## New artifacts created in this sprint

| File | Purpose |
|---|---|
| `bench/lcsa_nax_phase1_5_harness.py` | §4-strict harness (Sprint C structural pattern) |
| `bench/lcsa_nax_rebench_analysis.py` | Cross-session analysis tool |
| `docs/lcsa-nax/lcsa-nax-rebench-decisions.md` | Decisions log |
| `docs/lcsa-nax/lcsa-nax-rebench-inventory.md` | This file |
| `docs/lcsa-nax/lcsa-nax-rebench-results.md` | §4-validated analysis output |
| `docs/lcsa-nax/lcsa-nax-rebench-data.json` | 3-session raw data |
| `docs/lcsa-nax/rebench-runlog-S{1,2,3}.txt` | Per-session stdout |

## Shape inventory (7 shapes)

| Shape | qL | kL | D | density | BT | class |
|---|---:|---:|---:|---:|---:|:--|
| lcsa_small_seq4k          |  4096 |  4096 | 128 | 0.24 | 16 | moderate |
| lcsa_small_seq4k_sparse   |  4096 |  4096 | 128 | 0.07 | 16 | moderate |
| lcsa_mid_seq8k            |  8192 |  8192 | 128 | 0.12 | 16 | moderate |
| lcsa_mid_seq8k_sparse     |  8192 |  8192 | 128 | 0.03 | 16 | boundary |
| lcsa_large_seq16k         | 16384 | 16384 | 128 | 0.12 | 16 | moderate |
| lcsa_large_seq16k_sparse  | 16384 | 16384 | 128 | 0.03 | 16 | boundary |
| lcsa_mid_seq8k_very_sparse|  8192 |  8192 | 128 | 0.01 | 16 | niche |

## Pre-existing artifacts referenced (unchanged)

| File | Used for |
|---|---|
| `docs/lcsa-nax/lcsa-nax-phase1_5-ship-verdict.md` | Single-session baseline to compare against |
| `docs/lcsa-nax/lcsa-nax-phase1_4-dispatcher-sweep.json` | Source of single-session ratios |
| `bench/lcsa_nax_phase1_4_dispatcher_sweep.py` | De-facto Phase 1.5 harness it replaces (kept for archive) |
| `bench/conv_nax_phase1_5_harness.py` | Sprint C §4 structural reference |
| `bench/conv_nax_phase1_5_analysis.py` | Sprint C analysis reference |

## Hardware + environment

- Hardware: M5 Max 128GB, macOS 26.4, iStat performance fan profile
- MLX: 0.31.2
- `mlx_mfa`: 2.34.0 (pre-rebench)
- Python: 3.11.14 (.venv)
