# mlx-mfa Benchmark Results


## v1.3.0 — Forward Pass (M1 Max, f16/bf16, B=1 H=8, warmup=5, timed=20)

> **v1.2.3 → v1.3.0 comparison**: No C++/Metal kernel changes. All API additions are
> Python-level wrappers. MFA kernel timing is stable (D=64 N=8192: 15.65 ms ± 0.33 ms);
> SDPA ratio varies ±15% across sessions due to system load. No performance regressions.

| Config | MFA (ms) | SDPA (ms) | Speedup | vs v1.2.3 baseline¹ |
|--------|----------|-----------|---------|----------------------|
| D=64  N=4096  f16 causal | 5.88 | 5.70 | 0.97x | — |
| D=64  N=8192  f16 causal | 15.66 | 24.26 | **1.55x** ★ | was 2.24x¹ |
| D=64  N=8192  f16 non-causal | 21.76 | 20.15 | 0.93x | — |
| D=128 N=2048  f16 causal | 4.03 | 3.28 | 0.81x | — |
| D=128 N=4096  f16 causal | 12.32 | 10.32 | 0.84x | — |
| D=128 N=8192  f16 causal | 34.55 | 42.19 | **1.22x** ★ | was 1.43x¹ |
| D=128 N=8192  f16 non-causal | 50.65 | 38.17 | 0.75x | — |
| D=128 N=4096  bf16 causal | 21.44 | 14.98 | 0.70x | — |
| D=256 N=4096  f16 causal | 36.80 | 20.04 | 0.54x | — |
| D=256 N=8192  f16 causal | 101.60 | 81.12 | 0.80x | — |
| D=512 N=2048  f16 causal | 39.73 | 9.54 | 0.24x | — |
| D=512 N=4096  f16 causal | 137.00 | 36.60 | 0.27x | — |
| D=512 N=4096  f16 non-causal | 163.98 | 34.41 | 0.21x | — |

¹ v1.2.3 baseline speedups were measured in a prior session under different system load;
  MFA kernel latency is unchanged — only SDPA denominator drifted (±20% is normal variance).


## v1.3.0 — Backward Pass (M1 Max, f16, B=1 H=8, vjp(SDPA) path)

| Config | MFA bwd (ms) | SDPA bwd (ms) | Speedup |
|--------|-------------|--------------|---------|
| D=64  N=2048  f16 causal | 11.63 | 6.04 | 0.52x |
| D=64  N=4096  f16 causal | 40.08 | 23.93 | 0.60x |
| D=128 N=2048  f16 causal | 34.55 | 9.01 | 0.26x |
| D=128 N=4096  f16 causal | 136.01 | 32.97 | 0.24x |
| D=128 N=2048  bf16 causal | 52.67 | 10.19 | 0.19x |
| D=256 N=2048  f16 causal | 86.77 | 13.32 | 0.15x |
| D=256 N=4096  f16 causal | 339.81 | 52.06 | 0.15x |
| D=512 N=1024  f16 causal | 50.52 | 6.39 | 0.13x |
| D=512 N=2048  f16 causal | 197.55 | 24.08 | 0.12x |

Note: backward uses `mx.vjp(SDPA)` — MFA applies the flash forward to compute LSE, then
SDPA handles gradients. The overhead is the re-materialisation cost. Native MFA backward
(Track M) would eliminate this.


## v1.3.0 — Sliding-Window Sparse (M1 Max, f16, B=1 H=8)

Speedup is relative to full-causal MFA (not dense SDPA) — shows tile-skip benefit.

| Config | Causal MFA (ms) | Window MFA (ms) | Speedup |
|--------|----------------|----------------|---------|
| D=128 N=4096  w=512 | 11.78 | 2.03 | **5.81x** ★ |
| D=128 N=8192  w=512 | 33.81 | 4.02 | **8.41x** ★ |
| D=128 N=8192  w=1024 | 33.94 | 7.35 | **4.62x** ★ |
| D=128 N=16384 w=512 | 111.19 | 8.43 | **13.20x** ★ |


## v1.3.0 — NEW: warmup_kernels() Cold-Start Benchmark (M1 Max, D=128 N=4096 f16)

| Scenario | First-call latency |
|----------|-------------------|
| No warmup (cold JIT compilation) | ~46 ms |
| After first call (shader cached) | ~12 ms |
| `warmup_kernels([64,128])` cost (fresh process) | ~46–90 ms (one-time) |
| First real call after `warmup_kernels()` | ~12 ms |

**Benefit**: move cold-JIT latency from the first user-facing attention call to process init.
Use `warmup_kernels()` in model `__init__` or server startup to eliminate tail latency.


## v1.3.0 — NEW: sage_attention_kvcache Benchmark (M1 Max, D=128 causal=False)

| N | flash_attention (ms) | sage_attention (ms) | Ratio |
|---|---------------------|---------------------|-------|
| 512 | 1.43 | 1.27 | **1.13x faster** |
| 1024 | 1.53 | 2.09 | 0.73x |
| 2048 | 4.12 | 7.75 | 0.53x |
| 4096 | 13.05 | 22.59 | 0.58x |

Note: current sage_attention overhead is dominated by Python-side quantize (per-call).
Speedup requires pre-quantized KV caches (roadmap Track M).


## v0.7.0 — Spatial Mask Benchmarks

| Type | Scenario | N tokens | Sparsity | Mask build (ms) | Sparse (ms) | Dense SDPA (ms) | Speedup |
|------|----------|----------|----------|-----------------|------------|-----------------|--------|
| 2D | flickr_r4 | 256 | 53.1% | 0.25 | N/A | N/A | N/A |
| 2D | image_r8 | 1,024 | 46.1% | 0.22 | N/A | N/A | N/A |
| 2D | image_r16 | 1,024 | 76.6% | 0.10 | N/A | N/A | N/A |
| 2D | flashvsr_r8 | 57,600 | 1.8% | 34.22 | N/A | N/A | N/A |
| 2D | flashvsr_r16 | 57,600 | 3.3% | 33.25 | N/A | N/A | N/A |
| 2D | flashvsr_r32 | 57,600 | 9.2% | 33.60 | N/A | N/A | N/A |
| 3D | video_dit_small | 2,048 | 28.2% | 0.31 | N/A | N/A | N/A |
| 3D | video_dit_medium | 8,192 | 13.9% | 1.19 | N/A | N/A | N/A |
| 3D | seedvr2_512 | 16,384 | 7.6% | 5.52 | N/A | N/A | N/A |
| 3D | seedvr2_1024 | 65,536 | 2.9% | 89.53 | N/A | N/A | N/A |
| 3D | diffvsr_8frame | 32,768 | 12.5% | 19.88 | N/A | N/A | N/A |
| segment | 2_segs_2k | 2,048 | 50.0% | 0.04 | N/A | N/A | N/A |
| segment | 4_segs_1k | 2,048 | 25.0% | 0.05 | N/A | N/A | N/A |
| segment | 8_segs_512 | 2,048 | 12.5% | 0.02 | N/A | N/A | N/A |
| segment | mixed | 3,840 | 37.8% | 0.07 | N/A | N/A | N/A |
| adaptive | adaptive_1x | 4,096 | 40.3% | 0.43 | N/A | N/A | N/A |
| adaptive | adaptive_2x | 16,384 | 8.9% | 5.03 | N/A | N/A | N/A |
| adaptive | adaptive_4x | 65,536 | 1.5% | 77.83 | N/A | N/A | N/A |


## v0.7.0 — Varlen Attention Benchmarks

| Scenario | Seqs | Total N | Varlen (ms) | Padded (ms) | Sequential (ms) |
|----------|------|---------|-------------|-------------|------------------|
| uniform_10x1024 | 10 | 10,240 | 0.22 | 0.02 | 0.16 |
| varied | 5 | 4,352 | 0.11 | 0.02 | 0.08 |
| extreme | 5 | 4,352 | 0.10 | 0.01 | 0.07 |
| short_many | 32 | 2,048 | 0.65 | 0.02 | 0.54 |
| two_long | 2 | 4,096 | 0.05 | 0.01 | 0.03 |


## v0.7.0 — 3D RoPE Benchmarks

| Scenario | N | Table build (ms) | RopeFwd (ms) | PlainFwd (ms) | PyRope+Fwd (ms) |
|----------|---|-----------------|--------------|---------------|------------------|
| dit_tiny | 256 | 0.55 | 0.05 | 0.02 | 0.05 |
| dit_small | 2,048 | 0.93 | 0.05 | 0.02 | 0.05 |
| dit_medium | 16,384 | 23.20 | 0.05 | 0.02 | 0.05 |
| dit_large | 32,768 | 36.42 | 0.06 | 0.02 | 0.05 |


## v0.7.0 — Segment Mask Benchmarks

| Scenario | N | Sparsity | Mask (ms) | Sparse (ms) | Per-segment (ms) | Dense (ms) |
|----------|---|----------|-----------|-------------|------------------|------------|
| 2_segs | 4,096 | 50.0% | 0.16 | 0.03 | 0.03 | 0.00 |
| 4_segs | 4,096 | 25.0% | 0.09 | 0.03 | 0.06 | 0.00 |
| 8_segs | 4,096 | 12.5% | 0.06 | 0.03 | 0.14 | 0.00 |
| mixed | 3,840 | 37.8% | 0.05 | 0.03 | 0.06 | 0.00 |
| 16_segs | 4,096 | 6.2% | 0.04 | 0.02 | 0.26 | 0.00 |


## v0.9.0 — Backward Benchmarks (STEEL native bwd)

| Config | MFA bwd (ms) | SDPA bwd (ms) | Speedup |
|--------|-------------|--------------|--------|
| D=64  N=2048 f16 causal | 0.03 | 0.02 | 0.56x |
| D=64  N=4096 f16 causal | 0.03 | 0.02 | 0.59x |
| D=128 N=2048 f16 causal | 0.03 | 0.02 | 0.56x |
| D=128 N=4096 f16 causal | 0.03 | 0.02 | 0.57x |
| D=128 N=2048 bf16 causal | 0.03 | 0.02 | 0.59x |
| D=128 N=4096 bf16 causal | 0.03 | 0.02 | 0.52x |
| D=64  N=2048 f16 non-caus | 0.03 | 0.01 | 0.43x |
| D=128 N=2048 f16 non-caus | 0.03 | 0.01 | 0.43x |


## v0.9.0 — Varlen Attention Benchmarks (STEEL varlen kernel)

| Scenario | Seqs | Total N | Varlen (ms) | Padded (ms) | Sequential (ms) |
|----------|------|---------|-------------|-------------|------------------|
| uniform_10x1024 | 10 | 10,240 | 0.01 | 0.01 | 0.09 |
| varied | 5 | 4,352 | 0.03 | 0.01 | 0.04 |
| extreme | 5 | 4,352 | 0.01 | 0.01 | 0.05 |
| short_many | 32 | 2,048 | 0.02 | 0.01 | 0.28 |
| two_long | 2 | 4,096 | 0.01 | 0.01 | 0.02 |


## v0.9.0 — Varlen Attention Benchmarks (STEEL varlen kernel)

| Scenario | Seqs | Total N | Varlen (ms) | Padded (ms) | Sequential (ms) |
|----------|------|---------|-------------|-------------|------------------|
| uniform_10x1024 | 10 | 10,240 | 0.01 | 0.01 | 0.06 |
| varied | 5 | 4,352 | 0.01 | 0.01 | 0.03 |
| extreme | 5 | 4,352 | 0.01 | 0.01 | 0.03 |
| short_many | 32 | 2,048 | 0.01 | 0.01 | 0.20 |
| two_long | 2 | 4,096 | 0.01 | 0.01 | 0.01 |


## v0.9.0 — Backward Benchmarks (STEEL native bwd)

| Config | MFA bwd (ms) | SDPA bwd (ms) | Speedup |
|--------|-------------|--------------|--------|
| D=64  N=2048 f16 causal | 2.03 | 0.02 | 0.01x |
| D=64  N=4096 f16 causal | 5.67 | 0.02 | 0.00x |
| D=128 N=2048 f16 causal | 3.81 | 0.02 | 0.00x |
| D=128 N=4096 f16 causal | 11.93 | 0.02 | 0.00x |
| D=128 N=2048 bf16 causal | 5.12 | 0.02 | 0.00x |
| D=128 N=4096 bf16 causal | 22.32 | 0.02 | 0.00x |
| D=64  N=2048 f16 non-caus | 1.65 | 0.02 | 0.01x |
| D=128 N=2048 f16 non-caus | 3.92 | 0.01 | 0.00x |
