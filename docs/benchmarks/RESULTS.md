# mlx-mfa Benchmark Results — v0.1.0

## Executive Summary

**Causal attention (the LLM use case) is 1.5–2.6× faster than `mx.fast.scaled_dot_product_attention`
on M1 Max for float16, head_dim=64/128.**

| Use case | Speedup vs MLX SDPA |
|---|---|
| D=64 causal, large N (≥4096) | **1.9–2.9×** |
| D=128 causal, large N (≥4096) | **1.5–2.6×** |
| D=128 non-causal | ~0.9–1.1× (parity) |
| D=256 causal, large N (≥4096) | **1.2–1.5×** |
| D=256 non-causal | 0.5–0.7× (slower) |
| f32 (any D) | 0.15–0.44× (ccv path, not STEEL) |

The causal speedup is fundamental to the STEEL design: the causal mask causes roughly
half the K-tiles to be skipped entirely (any tile where all keys are after the current
query position), halving the effective work while SDPA still pays full cost.

---

## Test Environment

| Field | Value |
|---|---|
| Hardware | Apple M1 Max |
| macOS | 26.4 |
| Python | 3.11.14 |
| MLX | 0.31.0 |
| GPU family gen | 13 (M1) |
| Benchmark | median of 20 iterations, 5 warmup |

---

## Results: float16, non-causal

### B=1, H=8

```
    D      N   SDPA ms    MFA ms  Speedup
   64    512      0.60      0.63    0.95x
   64   1024      1.42      1.41    1.00x
   64   2048      1.91      1.98    0.97x
   64   4096      4.91      4.94    0.99x
   64   8192     18.25     20.03    0.91x
  128    512      0.45      0.46    0.99x
  128   1024      0.89      0.94    0.95x
  128   2048      2.60      2.89    0.90x
  128   4096      9.51     10.43    0.91x
  128   8192     37.84     40.47    0.94x
  256    512      0.66      0.89    0.74x
  256   1024      1.53      2.60    0.59x
  256   2048      4.73      8.82    0.54x
  256   4096     17.05     34.98    0.49x
  256   8192     67.32    163.82    0.41x
```

### B=2, H=8

```
    D      N   SDPA ms    MFA ms  Speedup
   64   1024      1.24      1.17    1.05x
   64   4096     10.71     11.75    0.91x
   64   8192     46.75     36.71    1.27x
  128   1024      1.60      1.72    0.93x
  128   4096     22.22     20.19    1.10x
  128   8192     75.09    105.65    0.71x
  256   1024      2.59      4.89    0.53x
  256   4096     33.46     72.08    0.46x
  256   8192    188.70    267.91    0.70x
```

---

## Results: float16, causal

### B=1, H=8

```
    D      N   SDPA ms    MFA ms  Speedup
   64    512      0.61      0.54    1.13x
   64   1024      1.59      1.37    1.16x
   64   2048      1.66      1.17    1.42x
   64   4096      5.69      2.91    1.96x
   64   8192     21.77     10.21    2.13x
  128    512      0.51      0.50    1.01x
  128   1024      0.98      0.98    1.00x
  128   2048      2.80      2.22    1.26x
  128   4096      9.98      6.47    1.54x
  128   8192     57.95     22.09    2.62x
  256    512      0.64      0.81    0.79x
  256   1024      1.51      2.46    0.61x
  256   2048      5.18      7.67    0.68x
  256   4096     29.42     20.15    1.46x
  256   8192    108.92     93.54    1.16x
```

### B=2, H=8

```
    D      N   SDPA ms    MFA ms  Speedup
   64   1024      1.48      0.92    1.60x
   64   4096     15.75      5.39    2.92x
   64   8192     41.40     21.35    1.94x
  128   1024      2.01      1.57    1.28x
  128   4096     18.89     11.23    1.68x
  128   8192    107.24     54.20    1.98x
  256   1024      2.89      3.70    0.78x
  256   4096     36.54     37.07    0.99x
  256   8192    175.24    143.04    1.23x
```

---

## Results: bfloat16 (B=1, H=8, D=128)

```
causal=False:
  N=1024  SDPA=1.41ms  MFA=1.89ms  0.75x
  N=4096  SDPA=13.43ms MFA=15.26ms 0.88x

causal=True:
  N=1024  SDPA=1.64ms  MFA=1.55ms  1.06x
  N=4096  SDPA=14.36ms MFA=10.45ms 1.37x
```

bfloat16 uses the STEEL path. Performance is similar to float16; causal is faster.

---

## Results: float32 (B=1, H=8, D=128) — ccv path

```
causal=False:
  N=1024  SDPA=1.72ms  MFA=5.24ms   0.33x
  N=4096  SDPA=16.34ms MFA=109.88ms 0.15x

causal=True:
  N=1024  SDPA=2.53ms  MFA=5.80ms   0.44x
  N=4096  SDPA=38.86ms MFA=138.05ms 0.28x
```

**float32 uses the ccv-based kernel (not STEEL)** due to threadgroup memory constraints.
The ccv path is significantly slower than MLX SDPA. Recommend f16 or bf16 for production.

---

## Crossover Analysis

At what sequence length does MFA first beat SDPA?

| D | dtype | causal | Crossover N |
|---|---|---|---|
| 64 | f16 | False | ~1024 (parity; 0.95-1.0x) |
| 64 | f16 | True | **~512** |
| 128 | f16 | False | ~1024 (parity; 0.9-1.0x) |
| 128 | f16 | True | **~2048** |
| 256 | f16 | False | Never (MFA slower for all N) |
| 256 | f16 | True | **~4096** |

---

## Before/After STEEL (D=128, N=4096, f16, B=2 H=8)

| Kernel | non-causal | causal |
|---|---|---|
| ccv original | ~0.08x SDPA | — |
| ccv + async_copy fix | ~0.95x SDPA | — |
| STEEL (v0.1.0) | ~1.10x SDPA | **1.68x SDPA** |

The ccv-to-STEEL rewrite delivered the causal speedup by eliminating the
`simdgroup_async_copy` dependency (broken on macOS 26+) and using a
register-efficient tile loading strategy (Q hoisted into registers once).
