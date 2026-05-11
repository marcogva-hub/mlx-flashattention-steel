# Phase 1.4 — Results

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_4`

## Summary

| Goal | Status |
|------|:------:|
| 1×1×1 detection (strict K=1,1,1 + no padding/stride extras) | ✓ implemented |
| Fast path: skip im2col, dispatch matmul on reshaped input | ✓ implemented |
| Env-var escape hatch `MFA_CONV_NAX_NO_FAST_PATH=1` | ✓ implemented |
| 5 new tests, bit-exact identity to general path | ✓ 5/5 PASS |
| Measurable wall-clock speedup vs general path | ✓ ~15% (small shape) |

## Correctness validation

All 5 Phase 1.4 tests PASS. Configuration: B=1, T=5, H=64, W=64,
C_in=512, C_out=512, K=1×1×1.

| Test | Oracle | Bar | Result |
|------|--------|----|:------:|
| finite_shape_dtype     | shape (1,5,64,64,512), no NaN/Inf | — | PASS |
| vs_torch_cpu_fp32      | PyTorch FP32 conv3d (perm channels-first) | rel < 1e-3 | PASS |
| vs_mlx_conv_general    | MLX baseline same dtype regime | rel < 1e-4 | PASS |
| faster_than_general    | wall-clock comparison | t_fast < t_general | PASS |
| fast_equals_general    | bit-exact rmse against forced general | rmse == 0 | PASS |

## Wall-clock comparison

Measured on M5 Max via `time_path()` in `test_conv3d_nax_1x1x1_faster_than_general_path`,
15 runs each after warmup, per-call mx.synchronize:

| Path | Median (ms) | Min (ms) | Max (ms) |
|------|------------:|---------:|---------:|
| Fast (skip im2col)            | 0.672 | 0.629 | 1.319 |
| General (im2col + matmul)     | 0.791 | 0.768 | 0.875 |

**Speedup: ~15%** at this shape. The general path's im2col kernel
writes M*K = 20480 × 512 = 10.5M f16 elements (21 MB) — small but
non-zero overhead.

For larger 1×1×1 shapes (more M), the speedup ratio is expected to
grow because im2col cost scales with M*K while matmul cost scales
with M*K*N. Phase 1.5 perf sweep will measure this if 1×1×1 shapes
are included.

## Bit-exact identity vs general path

Per `test_conv3d_nax_1x1x1_fast_equals_general`:

```
fast vs mlx        rmse=0.000019 mag=1.5400 rel=1.2539e-05
general vs mlx     rmse=0.000019 mag=1.5400 rel=1.2539e-05
fast vs general    rmse=0.000000 mag=1.5400 rel=0.0000e+00
```

Both paths produce **identical output**. The fast path's reshape +
matmul is mathematically equivalent to general's im2col + matmul
when im2col is the identity (which it is for 1×1×1 no-pad no-stride).

## Regression scan

- Phase 1.1+1.2+1.3 tests: 15/15 PASS unchanged
- Phase 1.4 tests: 5/5 PASS new
- Existing mlx-mfa suite: 931 PASS + 20 conv_nax = 951 total
- 6 pre-existing failures unchanged

## Detection coverage

The fast path activates ONLY for STRICT 1×1×1 (D27):
- K_T=K_H=K_W=1
- All paddings = 0
- All strides = 1

Cases routing through general path:
- K_T=1, K_H=K_W=3 (Phase 1.2 K_T=1 routing test): general path ✓
- K_T=K_H=K_W=1 with stride=2: would route general (correct,
  conservative; not tested explicitly but exercises the strict gate)
- K_T=K_H=K_W=1 with padding>0: same as above

## Items for Phase 1.5

- Perf sweep across 6 production shapes (mid_resnet, up1_resnet,
  up2_resnet0_chunk_cap, up3_resnet_chunk_cap, up2_resnet_full,
  up2_resnet0_peakflops).
- Pre-flight correctness gate.
- Variance handling per Sprint A §B.7.
- `ship-shelve-decision.md` per Sprint A precedent.
- Note: 1×1×1 fast path adds a 7th category of shape behavior to
  consider in the perf sweep. Either include 1×1×1 in the sweep
  (and apply ship/shelve to it separately) or defer 1×1×1 to a
  follow-up sprint focused on pointwise-conv workloads.

## Files in this phase

```
mlx_mfa/conv_nax.py                       modified (+88 lines net)
tests/test_conv_nax.py                    modified (+165 lines, 5 new tests)
docs/conv-nax/conv-nax-phase1_4-inventory.md
docs/conv-nax/conv-nax-phase1_4-decisions.md  D27-D29
docs/conv-nax/conv-nax-phase1_4-results.md    (this file)
docs/conv-nax/conv-nax-phase1_4-data.json
devnotes/SESSION_LOG.md                   [CLAUDE] entry on close commit
```
