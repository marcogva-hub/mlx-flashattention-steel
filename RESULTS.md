# mlx-mfa Benchmark Results


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
