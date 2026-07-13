# Verified results

This document contains only measurements that satisfy the current comparison
contract: same input dtype, distinct terminal fingerprints, fp32-oracle
correction, process-isolated arms, five sessions in both orders, and 20
dispatches per sample for sub-millisecond work. Ratios are baseline time divided
by mlx-mfa time, so values above one favor mlx-mfa.

All active rows below were measured on Apple M5 Max, macOS 27 beta and MLX
0.31.2. They are beta-3 indicative and must be revalidated on stable macOS.

## Sparse routing map

Date: 2026-07-13. Evidence:
`benchmarks/results/sparse_gate_remap_null.json`,
`benchmarks/results/sparse_gate_remap_map.json` and
`benchmarks/results/sparse_gate_remap_causal_d128_n8192_bh12.json`.

The A-vs-A null experiment established a 7.53% decision floor. A cell is a win
or loss only when both execution orders exceed that floor in the same
direction.

| Population | Win | Loss | Noise |
|---|---:|---:|---:|
| complete map | 61 | 55 | 6 |
| fp16 base cells at N8192 | 36 | 0 | 0 |
| fp16 base cells at N2048 | 0 | 36 | 0 |

This result invalidates a density-only route. Head load, sequence length,
dimension, causal mode and dtype all participate in the current gate.

The separately measured causal dominance cell was D128, N8192, B*H=12,
fp16 and actual block density 0.2999878:

| Order | `v6nax_sparse` | masked SDPA | SDPA / native | Native cosine vs fp32 |
|---|---:|---:|---:|---:|
| native first | 2.7868 ms | 10.8222 ms | 3.8833x | 0.99999994 |
| SDPA first | 2.8139 ms | 10.8293 ms | 3.8485x | 0.99999994 |

## Hardened route spot-checks

Date: 2026-07-13. Schema:
`mlx-mfa.final-routed-spots.v2`. Evidence is stored under
`benchmarks/results/audit_remediation_*`.

| Route and cell | Order A | Order B | Terminal pair | Correction floor |
|---|---:|---:|---|---:|
| GNA D128, N4096, fp16, 3D 1x7x7 | 1.1795x | 1.2407x | `gna_v6nax` / `sdpa` | cos >= 0.99999994 |
| former broad sparse D128, N4096 sliding cell | 0.7670x | 0.7866x | `v6nax_sparse` / `sdpa` | cos >= 0.99999988 |

The sparse loss is retained here because it motivated the measured gate
contraction. That cell is no longer evidence for a broad default route.

## Decode carveout

Date: 2026-07-12. Evidence:
`benchmarks/results/decode_edge_edge_*_20260712_1942*.json`.

For qL=8, kL=4096, D64, GQA=8, non-causal fp16, the terminal pair was
`mfa_primitive` / `sdpa`. SDPA/MFA measured 1.2099x with MFA first and 1.2458x
with SDPA first. Both paths had cosine above 0.99999989 against the fp32 oracle.
The full carveout uses only bounds locked in the dispatch map.

## SeedVR2 VAE opt-in study

Date: 2026-07-12. These are isolated-process production-weight measurements,
not default routing claims. The target combined the measured Conv3D spatial
pad/slice envelope with fused GroupNorm+SiLU calls.

| Process order | Baseline encode+decode | Target encode+decode | Baseline / target | Output checks |
|---|---:|---:|---:|---|
| baseline then target | 3216.98 ms | 2255.59 ms | 1.4262x | encode cos 0.99999990; decode cos 1.0 |
| target then baseline | 3134.16 ms | 2218.43 ms | 1.4128x | same captured-output contract |

Evidence:
`benchmarks/results/seedvr2_vae_cumulative_{A,B}_{baseline,target}.json`.
The Conv3D spatial-pad route remains opt-in and beta-3 indicative; the fused
GroupNorm probe is not a public default surface.

## Excluded historical numbers

Published changelog entries preserve earlier measurements as history. They are
not copied into this current-results page because their harnesses did not prove
both binaries under the contract above. A historical number must be labeled
`historical, pre-hardened-harness` if discussed elsewhere; it cannot justify a
current routing decision.

## Reproduction rules

1. Use the exact public API for the path being claimed.
2. Record MLX, mlx-mfa, macOS, hardware, dtype and absolute milliseconds.
3. Capture the terminal of both arms and reject a missing fingerprint.
4. Compare both outputs to an independent fp32 oracle before timing ratios.
5. Run each arm in a fresh process for five sessions and reverse arm order.
6. Calibrate A-vs-A noise before interpreting a fine-grained result.
