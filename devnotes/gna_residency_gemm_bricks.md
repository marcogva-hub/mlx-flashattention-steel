# GNA Residency and MLX GEMM Brick Audit

Date: 2026-07-13. β3-indicative. Branch: `explore/gna-residency`.

This is a source audit plus an isolated GNA probe. No public threshold was
changed, no default was changed, and no cross-kernel adoption was made.

## §AA.5: premise validation

The requested MLX source path did not exist initially. I cloned the official
MLX repository read-only to `/tmp/mlx/mlx-main` and audited commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e`. Its `LICENSE` is MIT, copyright
Apple Inc. [VERIFIED]

The relevant MLX implementation is:

- `steel/gemm/loader.h:14-134` — `BlockLoader`: vectorized aligned reads,
  threadgroup destination, and safe-tail loads.
- `steel/gemm/loader.h:142-260` — `BlockLoaderT`: arbitrary destination
  strides, including transposed K staging.
- `steel/gemm/gemm_nax.h:26-128` — NAX GEMM loop, `SK=32`, direct
  `NAXTile.load/load_safe`, barriers between K iterations, no double buffer.
- `steel/gemm/kernels/steel_gemm_fused_nax.h:97-215` — grid remap via
  `params->swizzle_log`, NAX accumulator, safe output tails and optional
  AXPBY epilogue.
- `steel/gemm/transforms.h:61-69` — `BlockSwizzle` mapping.
- `steel/attn/kernels/steel_attention_nax.h:83-429` — attention NAX uses the
  same direct NAX fragment loads and online softmax; it does not use the GEMM
  `BlockLoader` staging path. [VERIFIED]

The local GNA NAX source confirms a different constraint:
`csrc/mfa_gna_nax.cpp:270-450` launches one threadgroup per Q tile/head and
loads K/V fragments directly with `NAXTile.load`. Threadgroup memory is not
shared between neighboring Q threadgroups. Therefore a K/V tile cannot be
staged once and consumed by multiple Q threadgroups without changing the
work decomposition and the live accumulator footprint. [VERIFIED]

## Brick × kernel inventory

| Brick | MLX behavior | GNA NAX | Sparse V6 NAX | Dense V6 NAX | Conv3D NAX | V6 NAX linear | Packed varlen NAX | Adoptability |
|---|---|---|---|---|---|---|---|---|
| NAX fragment/MMA | `NAXTile` + `tile_matmad_nax` | Same cooperative-tensor family | Same local NAX helper copy | Same local NAX helper copy | Uses MPP `matmul2d` | Same local NAX helper | Same local NAX helper | Already adopted in all NAX attention/linear paths; no new delta. |
| Vectorized loader | `BlockLoader` uses aligned vector reads into threadgroup memory | Direct fragment loads | Direct fragment loads | Direct fragment loads | MPP operand loads | Direct fragment loads | Direct fragment loads | Not drop-in: NAX fragments are the current load unit; a new staged work decomposition is required. |
| Strided/transposed staging | `BlockLoaderT` supports arbitrary destination strides, including K transpose | No threadgroup K/V staging | No threadgroup K/V staging | No threadgroup K/V staging | Matrix operands are MPP tensors | No staging | No staging | Applicable to the STEEL GNA path, already represented by local `MFABlockLoaderT`; not a GNA NAX drop-in. |
| Safe tails | `load_safe` zero-fills partial A/B tiles | `load/load_rows` with explicit bounds | Gate/BT alignment plus NAX loads; tails are handled by the sparse generator contract | `load_rows/load_safe` in the generator | K padding before MPP | `store_safe`/input checks | Segment-local tail checks and locks | Correctness feature, not a residency optimization. |
| Grid swizzle | `BlockSwizzle` is active for fused GEMM and encoded by `swizzle_log` | Absent before this run | Absent | Absent | Separate MPP grids | Absent | Segment tile scheduler, not GEMM swizzle | Adaptable as an opt-in walk-order probe; measured below. |
| Double buffering | `gemm_nax.h` has no double buffer | None | None | None | None in the relevant MPP wrapper | None | None | No MLX evidence for a missing NAX double buffer; remains closed. |
| Epilogue transforms | Add/AXPBY after NAX accumulation | Softmax/mask, not GEMM epilogue | Mask/softmax/PV | Softmax/PV/LSE | Slice/store epilogue | Bias/GELU already fused locally | Segment softmax | Not a K/V residency lever. |
| Tile specialization | Static MLX GEMM set `(64/64/256)`, `(64/128/64/256)`, `(128/128/64/256/512)` | Runtime `BQ/BK/WM` env sweep | Fixed generator contract | Parametric generator | Fixed MPP tile | Runtime `BM/BN/BK/WM/WN` | Parametric BQ/BK/WM | Adoptable as tuning input, not a missing shared brick. |

The important negative result is structural: MLX's GEMM loader is a
threadgroup staging primitive, while GNA NAX is currently a direct-fragment
attention kernel. Copying the loader alone would not create inter-Q-tile
residency. [DEDUCED from the verified source structure]

## GNA probes

### Probe A: MLX-style grid swizzle

The new opt-in `MFA_GNA_NAX_SWIZZLE_LOG={1,2}` applies the MIT MLX mapping to
the GNA grid. The JIT name includes `_swz{log}` so the two binaries cannot
alias. The launch dimensions are remapped in **thread units**, not group
units; a first probe caught this distinction because it produced cos `0.143`
with incomplete Q coverage. That faulty measurement was discarded and the
grid was fixed before collecting ratios. [VERIFIED]

All timed cells used the public `flash_attention_gna` path. The trace for every
arm was `gna_v6nax`, and all oracle cosines were at least `0.9999999424`.
The numerator below is baseline median ms divided by candidate median ms;
`>1` means the candidate is faster. Each order is a fresh foreground process,
five sessions, 20 dispatches per measured sample, with three warmups.

#### Swizzle log 1

| dtype | D | window | baseline-first | candidate-first | geometric summary |
|---|---:|---|---:|---:|---:|
| fp16 | 64 | 1×7×7 | 1.050× | 1.302× | 1.169× |
| fp16 | 64 | 3×11×11 | 1.189× | 0.868× | 1.016× |
| fp16 | 128 | 1×7×7 | 1.052× | 1.018× | 1.035× |
| fp16 | 128 | 3×11×11 | 0.911× | 0.896× | 0.903× |
| bf16 | 64 | 1×7×7 | 1.027× | 1.014× | 1.020× |
| bf16 | 64 | 3×11×11 | 0.787× | 0.950× | 0.865× |
| bf16 | 128 | 1×7×7 | 0.913× | 0.923× | 0.918× |
| bf16 | 128 | 3×11×11 | 0.881× | 0.865× | 0.873× |

The isolated D64/fp16 small-window signal does not reproduce as a general
GNA win; large-window and D128 cells are neutral-to-negative. [VERIFIED]

#### Swizzle log 2 autoresearch round

`log=2` did not open a regime. The geometric summaries were `0.988×` at best
(D64/fp16 small) and as low as `0.820×` (D128/bf16 large); large windows were
negative for both dtypes at both D values. [VERIFIED]

### Probe B: Q-tile aggregation / K/V amortization

The source-supported candidate was `BQ=64, WM=4` for D64 and `BQ=128,
WM=4` for D128, with `BK=32`. This keeps one K/V load per larger Q tile and
tests the residency hypothesis without adding a new memory object.

| dtype | D | window | baseline-first | candidate-first | geometric summary |
|---|---:|---|---:|---:|---:|
| fp16 | 64 | 1×7×7 | 1.324× | 0.985× | 1.142× |
| fp16 | 64 | 3×11×11 | 0.935× | 0.970× | 0.952× |
| fp16 | 128 | 1×7×7 | 0.590× | 0.507× | 0.547× |
| fp16 | 128 | 3×11×11 | 0.349× | 0.331× | 0.340× |
| bf16 | 64 | 1×7×7 | 1.006× | 0.969× | 0.987× |
| bf16 | 64 | 3×11×11 | 1.105× | 0.996× | 1.049× |
| bf16 | 128 | 1×7×7 | 0.587× | 0.608× | 0.597× |
| bf16 | 128 | 3×11×11 | 0.335× | 0.330× | 0.332× |

The large-Q tile loses decisively at D128. The extra live NAX accumulator and
partial-MMA work dominate the K/V-load amortization. D64 is not a stable win
across order and dtype. [VERIFIED]

Raw artifacts:

- `benchmarks/results/gna_residency_swizzle_orderA.json`
- `benchmarks/results/gna_residency_swizzle_orderB.json`
- `benchmarks/results/gna_residency_swizzle2_orderA.json`
- `benchmarks/results/gna_residency_swizzle2_orderB.json`
- `benchmarks/results/gna_residency_bq_orderA.json`
- `benchmarks/results/gna_residency_bq_orderB.json`

## Step 3: cross-kernel adoption backlog

No cross-kernel brick is adopted in this branch.

- **Sparse and dense NAX:** flag a future source-generator cleanup to share
  the Apple NAX helper rather than maintain local copies. This is structural
  debt, not a measured performance lever.
- **Linear:** the local `v6_nax_linear` already has shape-specific tiles and a
  fused bias/GELU epilogue; a future comparison against MLX's complete static
  shape set should be a separate FFN sprint.
- **Conv3D:** its MPP path is not the same as attention's K/V residency problem;
  do not transplant attention loaders without a Conv3D-specific measurement.
- **Packed varlen:** its segment scheduler and tail contract are different;
  MLX GEMM swizzle is not directly applicable. The current `tile_offsets`
  mapping must remain the source of truth.
- **GNA STEEL:** it already uses the local equivalent of MLX `BlockLoaderT`
  for threadgroup K/V staging. A future experiment could compare its loader
  details directly, but it would be a STEEL-vs-NAX architectural comparison,
  not a drop-in NAX brick adoption.

## Verdict

The audit found no adoptable MLX GEMM brick that fixes the dominant GNA issue.
The two measurable GNA-specific hypotheses were closed:

1. Grid swizzle is correct but does not provide a stable gain. Keep the probe
   out of default routing.
2. Larger Q tiles amortize a theoretical K/V load but lose to register/live-
   fragment pressure, especially D128. Keep the current measured tile policy.

The cumulative gain from this run is **1.000× by adoption**: no candidate is
promoted. The previously measured GNA roof utilization remains the reference
range `4.1–6.6%` from `devnotes/gna_tuning.md`; this run did not remeasure a
roofline, so there is no new roof percentage to claim. [VERIFIED for prior
report; current-run roof percentage NOT MEASURED]

The routed GNA kernel is conserved and was not changed in its default
configuration. Focused conservation/correctness locks pass (`16 passed`),
including D64/D128 f16/bf16 byte equality between the default walk and the
swizzle probe. The release-gate lock plus the full suite pass (`3492 passed,
93 skipped, 3 warnings`). [VERIFIED]

## Red-team

- The result is a grid-walk and tile-size probe, not a hardware cache-counter
  proof; M5 exposes no usable occupancy/L2 counters in this environment.
- The benchmark uses N=4096, shape `4×32×32`, one head, and the two requested
  windows. It does not justify a new default threshold for other GQA/head,
  N, or stride regimes.
- The first invalid swizzle grid was caught by the oracle gate and is excluded
  from all tables. This demonstrates why grid dimensions must be audited in
  thread units at the `metal_kernel` boundary.
- `gna_v6nax` trace proves public path engagement, but does not expose the
  private JIT name. The source-level `_swz` key discriminator plus the env
  configuration establish candidate separation; a future public JIT-name hook
  would strengthen this evidence.
- No swizzle or BQ candidate is recommended for routing. No trans-kernel
  adoption was forced from source similarity alone.

## Stamp

| Item | Value |
|---|---|
| Python | `/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python` 3.11.14 |
| MLX | 0.31.2 |
| mlx-mfa | 2.61.0 |
| Hardware | Apple M5 Max, 40 GPU cores |
| OS | macOS 27.0 beta, build 26A5378j |
| Metal | 32023.918 |
| MLX source | `/tmp/mlx/mlx-main`, commit `4367c73b60541ddd5a266ce4644fd93d20223b6e`, MIT |
| Measurement | foreground; 5 sessions; 20 dispatches/sample; two fresh process orders |

## Skill invocations

| Gate | Result |
|---|---|
| `metal-kernel-dev` | Reviewed NAX fragment loads, threadgroup sizing, register/live-tile risk, and the swizzle grid-unit correction. |
| `benchmark-harness-builder` | Used oracle-before-ratio, public dispatch trace, sustained sessions, two orders, and raw JSON artifacts. |
