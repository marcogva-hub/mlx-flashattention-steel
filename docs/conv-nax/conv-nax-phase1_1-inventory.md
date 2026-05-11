# Phase 1.1 — File + Binding Inventory

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_1`
**Scope:** Sub-phase 0 (microbench) + sub-phase B (Conv3D scaffolding + mid_resnet correctness)

## Files added

| Path | Lines | Purpose |
|------|------:|---------|
| `bench/conv_nax_matmul2d_microbench.py`            | 343 | Phase 1.1 sub-phase 0 microbench v2 (per-tile + smoke gate). |
| `mlx_mfa/conv_nax.py`                              | 402 | Python orchestrator: im2col + matmul2d JIT chain, sanity asserts. |
| `tests/test_conv_nax.py`                           | 218 | Phase 1.1 sub-phase B mid_resnet correctness tests (4 tests). |
| `docs/conv-nax/conv-nax-phase1_1-microbench-blocker.md` | 200 | v1 methodology blocker diagnostic (historical). |
| `docs/conv-nax/conv-nax-phase1_1-inventory.md`     | this | This file. |
| `docs/conv-nax/conv-nax-phase1_1-decisions.md`     | TBD | D-numbered Phase 1.1 decisions. |
| `docs/conv-nax/conv-nax-phase1_1-results.md`       | TBD | Correctness + microbench gate verdict. |
| `docs/conv-nax/conv-nax-phase1_1-data.json`        | TBD | Raw correctness reproduction data. |
| `docs/conv-nax/conv-nax-phase1_1-matmul2d-microbench-v2.json` | TBD | 3-session microbench TFLOPS. |
| `docs/conv-nax/conv-nax-phase1_1-microbench-v2-runlog.txt` | TBD | 3-session run log. |

## Files unchanged (production code)

No `csrc/` files touched. No existing `mlx_mfa/` module modified. No
`tests/` other than `test_conv_nax.py` added. Sprint A V6 NAX
infrastructure untouched.

## Bindings inventory

- `mlx_mfa.conv_nax.conv3d_nax_forward(x, w, stride, padding, dilation)` —
  the public Phase 1.1 API. Python orchestrator (no C++ binding).
- No `_ext.conv3d_nax_forward` C++ binding in this Phase 1.1.
  **Deferred to Phase 1.5+ post-ship-verdict.** See decisions.md D3.

## JIT kernel inventory

| Kernel name | Source location | Dispatch |
|-------------|-----------------|----------|
| `im2col3d_*` | `mlx_mfa/conv_nax.py:_im2col3d_source()` | 1 thread per (m, k) element |
| `conv3d_matmul2d_*` | `mlx_mfa/conv_nax.py:_matmul2d_source()` | 1 TG per (M_TILE x N_TILE) output tile |
| `matmul2d_v2_*` | `bench/conv_nax_matmul2d_microbench.py:kernel_source()` | Same dispatch as above; microbench standalone |

Tile config (validated): `M_TILE=N_TILE=K_TILE=32`, `EXEC_SIMDGROUPS=1`,
`TG_THREADS=32`. Matches V6 NAX's BQ=BK=BD=32.

## Commits on branch (chronological)

1. `5e57430` — defective v1 harness + blocker diagnostic (historical context)
2. `edd9683` — session log [CLAUDE] BLOCKED entry (historical)
3. `2a02997` — bench v2: per-tile descriptor + smoke gate
4. `318c978` — tile config (32,32,32,sg=1) matches V6 NAX, >30 TF gate
5. `0de39f8` — feat conv-nax: im2col + matmul2d JIT chain (rightT bug fixed)
6. (this commit) — Phase 1.1 tests + 5 deliverables

## Validation status

- microbench v2 smoke gate: **PASS** (rel_err = 0 on K=64; rel_err 2.5e-5 on K=13824)
- Tile-config verification: **PASS** ((32,32,32,sg=1) chosen)
- Single-session prod_smoke: **44.92 TF mid_resnet, median dominant 37.91 TF** (above 30 TF gate)
- 3-session §4-compliant bench: **in progress at write time**; verdict in results.md.
- mid_resnet correctness tests (4): **all PASS** (see results.md for numbers).
- 3-session bit-exact reproduction: **PASS** (rmse=1.0580762755e-03 identical).
