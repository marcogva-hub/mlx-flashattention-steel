# Verified results

This document contains only measurements that satisfy the current comparison
contract: same input dtype, distinct terminal fingerprints, fp32-oracle
correction, process-isolated arms, five sessions in both orders, and 20
dispatches per sample for sub-millisecond work. Ratios are baseline time divided
by mlx-mfa time, so values above one favor mlx-mfa.

All active rows below were measured on Apple M5 Max, macOS 27 beta and MLX
0.31.2. They are beta-3 indicative and must be revalidated on stable macOS.

**Regulator-controlled revalidation (2026-07-30, MLX 0.31.2 + 0.32.0).** The 07-12/13
rows were taken under the macOS 27 beta thermal-regulation bug (FB23754032: GPU clock
reduced before the fan margin is consulted, bistable). They were revalidated under a
remediated fan curve (full frequency held under sustained load: die 74–81 °C → GPU
1518–1618 MHz, fans ~5764 rpm, under-load frequency gate passed on every block). Under
the controlled regulator **no routing classification flips**: the sparse gate's whole
engaging region stays a win (61/61 cells across the July map ∩ the routing predicate,
plus three N6144 evidence-gap cells) on both MLX 0.31.2 and 0.32.0. Evidence:
`benchmarks/results/reval_*`. Rows annotated *regulator-controlled* below carry this
provenance; unmeasured rows keep their beta-3 label.

## Sparse routing map

Date: 2026-07-13. Evidence:
`benchmarks/results/sparse_gate_remap_null.json`,
`benchmarks/results/sparse_gate_remap_map.json` and
`benchmarks/results/sparse_gate_remap_causal_d128_n8192_bh12.json`.

The A-vs-A null experiment established a 7.53% decision floor. A cell is a win
or loss only when both execution orders exceed that floor in the same
direction.

*Regulator-controlled (2026-07-30):* the engaging region of the gate — the July
map intersected with the routing predicate `_nax_sparse_route_viable`
(`mlx_mfa/lcsa_nax.py`), 58 cells, plus three N6144 evidence-gap cells that route
without a July measurement — was re-mapped and is **61/61 win, zero flip** on both
MLX 0.31.2 and 0.32.0. The gate is B·H-dependent, not min-N: it correctly delegates
the July losers (B·H=1 below N=8192) to SDPA, so those cells have no native path to
benchmark. The N=8192 sparse null floor of the day is **10.43% (0.31.2) / 11.80%
(0.32.0)** — it *rose* from the July 7.53% rather than falling, so the "throttling
inflated the floors" reading is demonstrated for the dense floor only, not the sparse
floor. Evidence `benchmarks/results/reval_C_*`, `reval_sparse_floor_*`,
`reval_E_sparse_n8192`.

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

*Regulator-controlled (2026-07-30):* re-measured at **3.83x / 3.86x** across the two
orders (`v6nax_sparse` engaged, cosine 0.99999994), confirming the 3.85–3.88x range —
a heavy-workload ratio the throttling did not distort. Evidence
`benchmarks/results/reval_B_causal_*`.

## Hardened route spot-checks

Date: 2026-07-13. Schema:
`mlx-mfa.final-routed-spots.v2`. Evidence is stored under
`benchmarks/results/audit_remediation_*`.

| Route and cell | Order A | Order B | Terminal pair | Correction floor |
|---|---:|---:|---|---:|
| GNA D128, N4096, fp16, 3D 1x7x7 | 2.39x | 2.44x | `gna_v6nax` / `sdpa` | cos >= 0.99999994 |
| former broad sparse D128, N4096 sliding cell | (delegates) | (delegates) | `sdpa` / `sdpa` | — |

*Regulator-controlled (2026-07-30):* the GNA row was re-measured at **2.39x / 2.44x**
(stable cold≈warm; `gna_v6nax` engaged). The July **1.1795x / 1.2407x** is retained as a
historical, *throttle-compressed* value: this sub-millisecond window (~0.26 ms) was the
one place the July regulator compressed a ratio — a 1×7×7 window (~1.2% density) should
beat full SDPA by far more than 1.2x, and under the controlled regulator it does (2.4x).
The direction is favorable (native wins more); it is not a routing risk. Evidence
`benchmarks/results/reval_A_gna_*`.

The former broad sparse sliding cell (July 0.7670x / 0.7866x, a loss) **delegates to SDPA
today** — this B·H=1 / N=4096 shape is outside the engaging region of the current gate
(`d3836d3`), so there is no native path to benchmark. This is expected delegation, not a
flip. The loss is retained as the motivation for the measured gate contraction; it is no
longer evidence for a broad default route.

## Decode carveout

Date: 2026-07-12. Evidence:
`benchmarks/results/decode_edge_edge_*_20260712_1942*.json`.

For qL=8, kL=4096, D64, GQA=8, non-causal fp16, the terminal pair was
`mfa_primitive` / `sdpa`. SDPA/MFA measured 1.2099x with MFA first and 1.2458x
with SDPA first. Both paths had cosine above 0.99999989 against the fp32 oracle.
The full carveout uses only bounds locked in the dispatch map.

*Regulator-controlled (2026-07-30):* the qL=8/kL=4096 reference cell was re-measured at
**1.25x / 1.27x** across the two orders (`mfa_primitive` engaged), confirming the
1.21–1.25x range. *New evidence (out of the revalidation scope, longer kL):* at the same
qL=8/D64/GQA=8 the SDPA/MFA ratio grows with kL — kL=8192 **1.39x**, kL=16384 **1.58x**,
kL=32768 **1.62x** (all `mfa_primitive` engaged). qL=4 is below the carveout bound and
delegates to SDPA (expected). Evidence `benchmarks/results/reval_decode/`.

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

## Regulator-controlled null floors and new evidence (2026-07-30)

Null floors are cross-process A-vs-A dispersions; they are the decision thresholds, not
performance claims. Measured same-build (`_ext` SHA `f219d573` on 0.31.2, `dee8b29e` on
0.32.0; the two arms' gate fingerprints are byte-identical across the MLX versions).

| Null floor | July | 0.31.2 (day) | 0.32.0 (day) |
|---|---:|---:|---:|
| dense B1/H8/N4096/D128 fp16 — conservative (transients included) | 27.4% | 7.44% | 1.08% |
| dense — clean regime (transients excluded) | — | 0.99% | 0.79–1.08% |
| sparse N8192 sliding/128/d0.15 | 7.53% (2-cell) | 10.43% | 11.80% |
| sparse frontier N4096/B·H12/D128 d0.29 | — | 2.79% | 1.49% |

The dense floor is reported dual, like the frontier: 7.44% is the *conservative* threshold
used for every verdict this campaign (so no verdict is re-judged), and 0.99% is the clean
regime after excluding non-recurring cross-process transients. Transient census (all floor
runs): none is correlated with die temperature (the floor benches are light, the machine is
cold, the under-load frequency gate passed) and none recurred on re-measure — e.g. the
frontier run showed one 20-process transient at +47% that vanished on re-measure (45.64% →
2.79%). The floor differences between the two MLX versions are a function of each run's
transient census (machine state), not the MLX version: sentinels and ratios are concordant
and the gate fingerprints are byte-identical, which is what makes the findings
MLX-version-independent. Evidence `benchmarks/results/reval_dense_null`, `reval_E_dense_null`,
`reval_sparse_floor_*`, `reval_E_*`, and the campaign telemetry `reval_macmon_telemetry.jsonl`.

*New evidence (out of the revalidation scope):* the three N6144 cells that the gate routes
without a July measurement are wins — random d0.05/B·H4/D128 **2.65x**, d0.25/B·H12/D64
**1.95x**, d0.30/B·H12/D128 **2.19x** (`v6nax_sparse` engaged). Evidence
`benchmarks/results/reval_C_*_n6144`.

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
