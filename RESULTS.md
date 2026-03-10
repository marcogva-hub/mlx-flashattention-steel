# mlx-mfa Benchmark Results

> **Version: v2.4.0** (STEEL V2 + Gen-Aware BK + Auto-Calibration + RoPE/ALiBi in V2)
> Device: Apple M1 Max (gen=13, 32 cores)
> B=2 H=8, warmup=8, iters=20. Values are means; GPU variance ±5–15%.
> Full data: [docs/benchmarks/RESULTS.md](docs/benchmarks/RESULTS.md)

---

## Dense Forward Pass (B=2 H=8, f16/bf16)

SDPA = `mx.fast.scaled_dot_product_attention` (explicit upper-triangular mask).
V1 = STEEL V1 (`MFA_DISABLE_V2=1`). V2 = STEEL V2 (default path, v2.4.0).

| Config | SDPA ms | V1 ms | V2 ms | V1/SDPA | V2/SDPA | V2/V1 |
|--------|--------:|------:|------:|--------:|--------:|------:|
| D=64  N=2048  f16 causal | 3.02 | 3.01 | 1.88 | 1.00× | **1.61×** | 1.60× |
| D=64  N=4096  f16 causal | 10.93 | 8.80 | 6.57 | 1.24× | **1.66×** ★ | 1.34× |
| D=64  N=8192  f16 causal | 42.69 | 24.62 | 19.82 | 1.73× ★ | **2.15×** ★ | 1.24× |
| D=64  N=8192  f16 non-causal | 35.97 | 39.86 | 37.36 | 0.90× | 0.96× | 1.07× |
| D=128 N=2048  f16 causal | 5.21 | 6.19 | 3.49 | 0.84× | **1.49×** | 1.77× |
| D=128 N=4096  f16 causal | 18.96 | 17.58 | 12.14 | 1.08× | **1.56×** ★ | 1.45× |
| D=128 N=8192  f16 causal | 74.87 | 51.95 | 43.69 | 1.44× | **1.71×** ★ | 1.19× |
| D=128 N=16384 f16 causal | 299.34 | 188.17 | 167.78 | 1.59× ★ | **1.78×** ★ | 1.12× |
| D=128 N=4096  bf16 causal | 26.73 | 29.89 | 19.21 | 0.89× | **1.39×** | 1.56× |
| D=128 N=8192  f16 non-causal | 73.36 | 85.87 | 81.86 | 0.85× | 0.90× | 1.05× |
| D=256 N=4096  f16 causal | 37.21 | 49.54 | 50.34 | 0.75× | 0.74× | 0.98× |
| D=256 N=8192  f16 causal | 146.43 | 157.44 | 157.34 | 0.93× | 0.93× | 1.00× |
| D=256 N=4096  f16 non-causal | 33.63 | 68.80 | 70.58 | 0.49× | 0.48× | 0.97× |

★ = ≥1.5× speedup.

**Key wins**: D=64 N=8192 causal **2.15×**, D=128 N=4096+ causal **1.56–1.78×**.
D=256 dense routes to V1 (V2 BK mismatch with 3D blocking).

---

## Sliding Window (B=2 H=8, f16, causal)

| Config | SDPA ms | MFA ms | MFA/SDPA |
|--------|--------:|-------:|---------:|
| D=64  N=4096 win=512  | 11.52 | 2.81 | **4.1×** |
| D=64  N=8192 win=512  | 42.06 | 3.15 | **13.3×** |
| D=128 N=4096 win=512  | 19.13 | 4.22 | **4.5×** |
| D=128 N=8192 win=512  | 74.63 | 6.34 | **11.8×** |
| D=128 N=4096 win=256  | 18.86 | 1.95 | **9.7×** |
| D=128 N=8192 win=256  | 74.56 | 3.69 | **20.2×** |

Window masking skips ~(N−win)/N fraction of K-tiles, giving super-linear speedup.

---

Regenerate: `python benchmarks/bench_v2_final.py --warmup 8 --iters 20 --save`
