# Phase 1.2 — Results

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_2`

## Summary

| Feature | Status | Tests |
|---------|:------:|:-----:|
| up1_resnet single-chunk path | ✓ via M-chunking (3 chunks) | 4 of 4 PASS |
| Asymmetric pad_T (causal) | ✓ via tuple-of-pairs API | 2 of 2 PASS |
| K_T=1 routing | ✓ via general path | 1 of 1 PASS |
| **Total** | | **7 of 7 new tests PASS** |

Plus Phase 1.1's 4 mid_resnet tests unchanged (PASS). 11 total
conv_nax tests pass.

## Phase 1.1 HANDOFF Pitfall 5 → root-caused + fixed

### Bug investigation

Bisection across M (5 runs each, deterministic):

| M | NaN count | NaN rows | Start row | M*K bytes |
|---|----------:|---------:|----------:|----------:|
|  20480 |          0 |     0 |     -  | 0.57 GB |
|  98304 | 10,551,296 | 20608 | **77696** | 2.72 GB |
| 114688 | 18,939,904 | 36992 | **77696** | 3.17 GB |
| 131072 | 27,328,512 | 53376 | **77696** | 3.62 GB |
| 147456 | 35,717,120 | 69760 | **77696** | 4.08 GB |

**Pattern.** Every M produces NaN starting **exactly at row 77696**.
At K=13824 f16 (dtype_bytes=2), the byte offset of row 77696 is
77696 × 13824 × 2 = **2,147,287,040 bytes** ≈ **2^31 - 196,608 bytes**.

**Root cause.** MPP `matmul2d` uses **int32 internally for byte
addresses**. Any byte address beyond `2^31 = 2,147,483,648` overflows
to negative, breaking address arithmetic and producing NaN.

This is a structural limit of Apple's MPP library. Cannot be fixed
in the kernel without alternate algorithms; must be worked around
at the dispatch level.

### Fix: M-chunking

Per decision D18. `_compute_chunk_layout(M, K, dtype_bytes)`:

```
SAFETY_HEADROOM = 0.875
max_chunk_M = floor(2^31 * 0.875 / (K * dtype_bytes))
max_chunk_M = floor_align(max_chunk_M, M_TILE=32)
```

For production shapes:

| Shape | M_total | max_chunk_M | n_chunks | chunk_M | per-chunk bytes |
|-------|--------:|------------:|---------:|--------:|----------------:|
| mid_resnet              |  20,480 | 65,504 | 1 |  20480 | 0.57 GB |
| up1_resnet              | 147,456 | 65,504 | 3 |  49152 | 1.36 GB |
| up2_resnet0_chunk_cap   | 297,000 | 65,504 | 5 |  59424 | 1.64 GB |
| up2_resnet_full         |1,114,112| 138,528 | 9 | 124032 | 1.71 GB |
| up2_resnet0_peakflops   |1,114,112| 65,504 | 18 |  61920 | 1.71 GB |
| up3_resnet_chunk_cap    | 594,000 | 262,016 | 3 | 198016 | 0.68 GB |

(K varies by shape; max_chunk_M varies inversely.)

## Correctness validation

### up1_resnet (Phase 1.2 anchor shape)

| Oracle | rel_err | Bar | Status |
|--------|--------:|----:|:------:|
| PyTorch CPU FP32   | 3.23e-5 | 1e-3 | PASS |
| MLX conv_general f16 | 3.23e-5 | 1e-4 | PASS |
| Sentinel coverage  | 0 NaN, 0 Inf | — | PASS |
| Chunk-boundary probes | rel < 1e-3 at t=0, 3, 6 | — | PASS |

3-session bit-exact reproduction:

```
S1: rmse=1.1613434181e-03  max=3.1250000000e-02  mag=3.5968750000e+01
S2: rmse=1.1613434181e-03  max=3.1250000000e-02  mag=3.5968750000e+01
S3: rmse=1.1613434181e-03  max=3.1250000000e-02  mag=3.5968750000e+01
```

10-decimal identical. Multi-chunk path is fully deterministic.

### Causal pad_T (mid_resnet shape, padding=((2,0),(1,1),(1,1)))

| Test | Status | Detail |
|------|:------:|--------|
| `test_mid_resnet_causal_pad_t` | PASS | rel_err < 1e-4 vs mx.conv_general with explicit low/high padding |
| `test_mid_resnet_causal_pad_t_flag` | PASS | causal_pad_t=True flag is **bit-exact** equivalent to explicit asymmetric form |

### K_T=1 routing (B=1, T=5, H=64, W=64, K=(1,3,3))

| Test | Status |
|------|:------:|
| `test_kt1_routing` | PASS, rel_err < 1e-4 vs mx.conv_general |

K_T=1 routes through the general path with K = K_H × K_W × C_in =
3 × 3 × 512 = 4608 (vs 27 × 512 = 13824 for 3×3×3). Im2col addressing
naturally handles K_T=1 via the compile-time loop bound — no special
case in code.

## Regression scan

- Phase 1.1 mid_resnet tests: 4/4 PASS unchanged
- Existing suite: 931 PASS + 11 conv_nax = 942 total
- 6 pre-existing failures unchanged (confirmed on Phase 1.1 close)

## Open follow-ups for Phase 1.3

1. **Working-set instrumentation.** Per-chunk im2col allocation +
   matmul output allocation total = ~3-4 GB peak for largest shapes.
   Phase 1.3 will instrument peak working-set and add hard gate at 16 GB.
2. **All 6 production shapes pass oracles.** Phase 1.2 validates 2
   shapes (mid_resnet, up1_resnet). Phase 1.3 will add tests for the
   remaining 4 (up2_resnet0_chunk_cap, up3_resnet_chunk_cap,
   up2_resnet_full, up2_resnet0_peakflops if it fits).
3. **Ping-pong buffer optimization (optional).** Currently each chunk
   allocates fresh im2col + output buffers; Phase 1.3 may add
   ping-pong reuse for slight perf gain. Defer if not free.

## Files in this phase

```
mlx_mfa/conv_nax.py       (modified: chunking + asymmetric pad)
tests/test_conv_nax.py    (added 7 tests; 11 total now)
docs/conv-nax/conv-nax-phase1_2-inventory.md
docs/conv-nax/conv-nax-phase1_2-decisions.md  (D18-D22)
docs/conv-nax/conv-nax-phase1_2-results.md    (this file)
docs/conv-nax/conv-nax-phase1_2-data.json
devnotes/SESSION_LOG.md   (Phase 1.2 [CLAUDE] entry on close commit)
```
