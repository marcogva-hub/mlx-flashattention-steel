# Phase 1.3 — Results

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_3`

## Summary

| Goal | Status |
|------|:------:|
| Working-set instrumentation (`estimate_working_set`) | ✓ shipped |
| 16 GB hard gate at sanity-assert time | ✓ enforced |
| Per-chunk eval bounds peak GPU memory | ✓ 9× reduction validated |
| All 6 production shapes fit < 16 GB peak | ✓ verified |
| 4 new tests + 11 inherited = 15/15 PASS | ✓ all PASS |

## Working-set per production shape

All 6 shapes from design §3.1 with the chunking heuristic + per-chunk eval:

| Shape | M | K | N | chunks | per_chunk_im2col | total_peak_est | < 16 GB |
|-------|--:|--:|--:|-------:|------------------:|---------------:|:-------:|
| mid_resnet              |  20,480  | 13824 | 512 |  1 | 0.57 GB | 0.59 GB | ✓ |
| up1_resnet              | 147,456  | 13824 | 512 |  3 | 1.36 GB | 1.51 GB | ✓ |
| up2_resnet0_chunk_cap   | 297,000  | 13824 | 256 |  5 | 1.64 GB | 1.80 GB | ✓ |
| up3_resnet_chunk_cap    | 594,000  |  3456 | 128 |  3 | 1.37 GB | 1.52 GB | ✓ |
| up2_resnet_full         | 1,114,112 |  6912 | 256 |  9 | 1.71 GB | 2.28 GB | ✓ |
| up2_resnet0_peakflops   | 1,114,112 | 13824 | 256 | 17 | 1.81 GB | 2.38 GB | ✓ |

## Lazy-accumulation → per-chunk eval

Phase 1.2 was vulnerable to MLX lazy-eval accumulation:

| Shape | Without per-chunk eval | With per-chunk eval | Reduction |
|-------|----------------------:|--------------------:|----------:|
| 17-chunk (M=1.1M K=13824 N=256) | 32.29 GB peak | 3.53 GB peak | 9× |

Without per-chunk eval, MLX accumulates all 17 chunks' im2col buffers
in the lazy graph before the final `mx.concatenate` triggers evaluation
→ 17 × 1.81 = 30.8 GB + concat = 31 GB peak.

With per-chunk eval (D23), MLX realizes each chunk's output before the
next chunk's im2col is allocated → bounded to ~1 chunk's transient
+ accumulated outputs.

The 3.53 GB observed peak is within the 16 GB hard gate. Estimator
predicted 2.38 GB — observed 3.53 GB. The ~1.15 GB delta is MLX
allocator overhead (see D25).

## Correctness validation

### Phase 1.1 + 1.2 + 1.3 test results

```
tests/test_conv_nax.py: 15 tests, all PASS in 4.18s

Phase 1.1 (4): mid_resnet finite_shape_dtype, vs_torch_cpu_fp32,
               vs_mlx_conv_general, sentinel_coverage
Phase 1.2 (7): up1_resnet × 4, causal_pad_t × 2, kt1_routing
Phase 1.3 (4): working_set_all_production_shapes_within_gate,
               working_set_chunk_plan_correctness,
               working_set_oversize_rejected_by_sanity,
               multi_chunk_correctness_5chunks
```

### Large-shape end-to-end probe

```
Shape: B=1, T=17, H=256, W=256, C_in=512, C_out=256, 3×3×3 same pad
  M = 1,114,112  K = 13,824  N = 256
  17 chunks of 61920 rows × 13824 K each = 1.81 GB im2col per chunk

Working-set estimator: 2.38 GB total peak prediction
Observed:               3.53 GB total peak (mx.get_peak_memory)
NaN/Inf check:          0 / 0
Correctness vs mx.conv_general f16: rel_err = 3.38e-5 (PASS, bar 1e-4)
```

## Regression scan

- Phase 1.1 + 1.2 + 1.3 tests: 15/15 PASS
- Existing mlx-mfa suite: 931 PASS + 15 conv_nax = 946 total
- 6 pre-existing failures unchanged (verified across phases)

## Items for Phase 1.4

- 1×1×1 fast path (skip im2col, direct matmul on reshape'd input)
  per D26 in this Phase's decisions doc.
- 4 new tests per prompt §E.2.

## Items for Phase 1.5

- Perf sweep across 6 production shapes.
- Pre-flight correctness gate on the actual harness path.
- Variance handling per Sprint A §B.7.
- `ship-shelve-decision.md` final actionable conclusion.

## Files in this phase

```
mlx_mfa/conv_nax.py                      modified (+96 lines net)
tests/test_conv_nax.py                   modified (+106 lines, 4 new tests)
docs/conv-nax/conv-nax-phase1_3-inventory.md
docs/conv-nax/conv-nax-phase1_3-decisions.md  D23-D26
docs/conv-nax/conv-nax-phase1_3-results.md    (this file)
docs/conv-nax/conv-nax-phase1_3-data.json
devnotes/SESSION_LOG.md                  [CLAUDE] entry on close commit
```
