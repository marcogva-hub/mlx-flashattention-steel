# Phase 1.4 — File + Test Inventory

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_4` (branched from `experiment/conv-nax-phase1_3` tip)
**Scope:** 1×1×1 fast path (skip im2col entirely; pointwise matmul on reshaped input)

## Files modified

| Path | Δ lines | Purpose |
|------|--------:|---------|
| `mlx_mfa/conv_nax.py`     | +88  | 1×1×1 detection + `_dispatch_1x1x1_fast_path` + `_make_pointwise_matmul_kernel` + env-var escape hatch + single-chunk slice optimization |
| `tests/test_conv_nax.py`  | +165 | 5 new Phase 1.4 tests |

## Files added (deliverables)

| Path | Purpose |
|------|---------|
| `docs/conv-nax/conv-nax-phase1_4-inventory.md` | This file |
| `docs/conv-nax/conv-nax-phase1_4-decisions.md` | D27-D29 |
| `docs/conv-nax/conv-nax-phase1_4-results.md`   | Perf + correctness |
| `docs/conv-nax/conv-nax-phase1_4-data.json`    | Timing data + tests inventory |

## Public API additions

```python
# Environment variable (escape hatch -- mostly for tests + diagnostics)
import os
os.environ["MFA_CONV_NAX_NO_FAST_PATH"] = "1"  # forces general path
# os.environ.pop("MFA_CONV_NAX_NO_FAST_PATH", None)  # default: fast path on

# No new public functions -- conv3d_nax_forward auto-detects 1×1×1.
```

## Tests inventory

**20 total** (4 Phase 1.1 + 7 Phase 1.2 + 4 Phase 1.3 + 5 Phase 1.4):

Phase 1.4 additions:
- `test_conv3d_nax_1x1x1_finite_shape_dtype` — shape preserved, no NaN/Inf
- `test_conv3d_nax_1x1x1_vs_torch_cpu_fp32` — rel < 1e-3 vs PyTorch FP32 oracle
- `test_conv3d_nax_1x1x1_vs_mlx_conv_general` — rel < 1e-4 vs MLX baseline
- `test_conv3d_nax_1x1x1_faster_than_general_path` — fast wall-clock < general wall-clock
- `test_conv3d_nax_1x1x1_fast_equals_general` — bit-exact rmse=0 vs general path

## Commits on branch (chronological)

1. `6d8e6a6` — feat+test(conv-nax): Phase 1.4 -- 1×1×1 fast path (5 new tests)
2. (next) — docs: Phase 1.4 deliverables

## Validation status

- Phase 1.1+1.2+1.3 tests: 15/15 PASS unchanged
- Phase 1.4 tests: 5/5 PASS new
- Bit-exact identity vs general path: rmse=0
- Perf: 15% wall-clock speedup at ONE_BY_ONE_CFG shape (B=1,T=5,H=64,W=64,C=512)
- Regression: 951 total tests pass; 6 pre-existing failures unchanged
