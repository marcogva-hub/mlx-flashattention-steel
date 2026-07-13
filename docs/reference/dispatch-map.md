# Runtime Dispatch Map

**Stamp:** 2026-07-13, M5 Max, MLX 0.31.2, macOS 27 beta. Routes marked β3 require stable-macOS revalidation.

Terminal names come from runtime dispatch tracing and are the authoritative engagement vocabulary.

## Dense forward

| Public input | Terminal |
|---|---|
| Plain self-attention, D=128, f16/bf16, NAX available, compatible features | `nax_dense` |
| D=64 plain forward | `sdpa` unless a decode carveout applies |
| D=512 | `sdpa` |
| fp32 or an unsupported feature combination | `sdpa` |
| forced native STEEL-compatible call | `mfa_primitive` |

## Decode β3 carveouts

| Exact envelope | Terminal |
|---|---|
| qL=8, D=64, GQA=8, non-causal, f16/bf16, 4096<=kL<=65536 | `mfa_primitive` |
| qL=16, D=64, GQA in {4,8,16}, non-causal, f16/bf16, 16384<=kL<=65536 | `mfa_primitive` |
| Every adjacent cell | `sdpa` |

## Sparse β3 gate

`v6nax_sparse` requires self-attention, effective BT32, D in {64,128}, and one row from the tables below. BT64 is expanded 2x2 and then tested against the same gate.

### Non-causal

| Dtype | N | B·H | D | Density ceiling |
|---|---:|---:|---:|---:|
| fp16 | 8192 | 1,4,12 | 64,128 | 0.30 |
| fp16 | 4096..8192 | 12 | 128 | 0.30 |
| fp16 | 4096..8192 | 12 | 64 | 0.25 |
| fp16 | 4096..8192 | 4 | 128 | 0.05 |
| bf16 | 4096..8192 | 12 | 128 | 0.30 |

### Causal

| Dtype | N | B·H | D | Density ceiling |
|---|---:|---:|---:|---:|
| fp16 | 4096 | 4 | 128 | 0.10 |
| fp16 | 4096 | 12 | 128 | 0.30 |
| fp16 | 8192 | 12 | 64,128 | 0.30 |
| bf16 | 4096 | 4 | 128 | 0.10 |

Unlisted sparse cells use `sdpa` or `scalar_fallback`. BT values outside {32,64} cannot reach V6 NAX.

## GNA

| Envelope | Terminal |
|---|---|
| 3-D, f16/bf16, D=128, N>=2048 | `gna_v6nax` |
| 3-D, f16/bf16, D=64, N>=4096 | `gna_v6nax` |
| 3-D, f16/bf16, D=128 below NAX threshold | `gna_steel` |
| native disabled or unsupported dimensionality | sparse fallback |

`MFA_DISABLE_GNA_NATIVE=1` is the escape to the sparse path.

## Packed varlen

| Envelope | Terminal |
|---|---|
| `MFA_ENABLE_VARLEN_NAX=1`, B=1, D=128, f16/bf16, GQA in {2,4,8}, segments in {20,24}, total Q=K in 35018..35250, identical Q/K cumulative offsets | `varlen_v6nax` |
| f16/bf16, D<=256, no block mask, outside V6 gate | `varlen_native` |
| fp32, D=512, block mask, or unsupported shape | `varlen_split_concat` |
| causal segment with qL>kL | `varlen_split_concat` plus per-segment `varlen_sdpa` |

The V6 route fixes BQ=32, BK=32, and WM=2 through one generation-and-dispatch configuration.

## Backward

| Envelope | Terminal |
|---|---|
| D=64, qL>=2048, f16/bf16, not disabled | `v6_split_backward` |
| D=128 with `MFA_ENABLE_V6_BACKWARD=1` | `v6_split_backward` |
| eligible STEEL policy cell | `steel_backward` |
| other cells | `sdpa_vjp` |

## Transparent hooks

Conv3D terminals are observed through hook telemetry rather than the attention dispatch trace. Spatial pad-and-slice and general pad-and-slice are opt-in β3 envelopes. An absent opt-in preserves MLX behavior.
