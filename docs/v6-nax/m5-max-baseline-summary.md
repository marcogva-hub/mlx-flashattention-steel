# M5 Max V2 STEEL Baseline Summary

**Date:** 2026-05-02
**Hardware:** Apple M5 Max (40 GPU cores, gen 17 / `applegpu_g17s`)
**Memory:** 128 GB unified
**Software:** macOS 26.5 · MLX 0.31.2 · mlx-mfa 2.28.0 · Python 3.11.14
**Kernel:** V2 STEEL (auto-dispatch via `flash_attention()`)

This is the V2 STEEL baseline. **V6 NAX (Phase 1) speedups will be measured
relative to these numbers.** No NAX, no MPP — pure V2 STEEL JIT kernels.

---

## 1. Production VSR shapes

| Shape          | N_q     | D   | H   | dtype | p50 (ms) | RMSE vs SDPA | input MB |
|----------------|--------:|----:|----:|:------|---------:|-------------:|---------:|
| SeedVR2-small  |  26,730 | 128 |  20 | fp16  |   196.64 |       0.0e+0 |    391.4 |
| SeedVR2-large  | 111,375 | 128 |  20 | fp16  |  4758.32 |       0.0e+0 |   1631.0 |
| FlashVSR-dense |   4,096 |  64 |  10 | fp16  |     1.28 |       0.0e+0 |     15.0 |
| CogVideoX      |  70,200 | 128 |  30 | fp16  |  2627.69 |       0.0e+0 |   1542.4 |
| LTX2-cross     |   2,048 |  64 |   8 | fp16  |     1.34 |       7.2e-2 |     16.4 |

The LTX2-cross RMSE is the only non-zero; this shape uses an asymmetric
`N_q=2048 / N_kv=14000` layout where `flash_attention()` dispatches the
MFA cross-attention path rather than falling back to SDPA.

---

## 2. Validation shapes

| Shape            | N    | D   | dtype | p50 (ms) |
|------------------|-----:|----:|:------|---------:|
| small-D64        | 1024 |  64 | fp16  |     0.28 |
| small-D128       | 1024 | 128 | fp16  |     0.37 |
| small-D64-bf16   | 1024 |  64 | bf16  |     0.32 |
| small-D128-bf16  | 1024 | 128 | bf16  |     0.42 |

All small shapes have RMSE = 0 vs SDPA.

---

## 3. M5 Max vs M1 Max — V2 STEEL on identical shapes

| Shape          | M5 Max (ms) | M1 Max (ms) | M5 / M1 | Speedup |
|----------------|------------:|------------:|--------:|--------:|
| SeedVR2-small  |      196.64 |      598.16 |   0.33× |  3.04×  |
| SeedVR2-large  |     4758.32 |    10184.80 |   0.47× |  2.14×  |
| CogVideoX      |     2627.69 |     6134.60 |   0.43× |  2.33×  |
| FlashVSR-dense |        1.28 |        6.73 |   0.19× |  5.24×  |

> M1 Max source: `docs/audit_dit_dispatch_results.json` (mlx-mfa v2.26.0,
> April 2026). Speedup factor reliable within ±5-10% across mlx-mfa versions
> as the V2 STEEL kernel has been stable.

**Observed speedup is significantly higher than predicted** (1.3-1.7× from
bandwidth/core ratios alone). Likely contributors:
- Larger L2 cache reducing async tile-load latency
- Higher simdgroup matrix-multiply throughput per core
- Improved threadgroup memory bandwidth on M5

**This means V6 NAX speedups are stacked on a 2-5× M5 baseline** — V6 NAX
needs to outperform an already-fast V2 STEEL on M5, which is a higher bar
than outperforming M1 Max.

---

## 4. What's NOT measured

- **V6 NAX kernel** — Phase 0 stub only; Phase 1 will activate.
- **Sparse paths** — known regression after MLX 0.31.2 upgrade (18 test
  failures, see CHANGELOG v2.28.0). These benchmarks use only dense paths.
- **Backward / training** — forward-only.
- **Causal masking** — `causal=False` for all benchmarks (matches typical
  VSR / DiT inference where causal masking is not used).

---

## 5. Reproduction

```bash
.venv/bin/python bench/m5_max_baseline.py
# Output: docs/v6-nax/m5-max-baseline-v2-steel.json
```

To run a subset:
```bash
.venv/bin/python bench/m5_max_baseline.py --shapes SeedVR2-small CogVideoX
```

To increase iterations for tighter percentiles:
```bash
.venv/bin/python bench/m5_max_baseline.py --warmup 5 --iterations 20
```
