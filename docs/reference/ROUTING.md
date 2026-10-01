# Runtime routing inventory — "for my shapes, what routes where?"

**Reading rule.** The eligibility *predicate at the source* is the authority — this document is
an **inventory generated from that source and the 2026-07-30 regulator-controlled revalidation
evidence** (Apple M5 Max, macOS 27 beta, MLX 0.31.2, `_ext` build of commit `558a191`). Every
routing condition cites `file:line`; every measured ratio cites its evidence JSON by path and is
`sdpa / native` median across both process orders (values > 1 favour mlx-mfa). The locked terminal
map is [`docs/reference/dispatch-map.md`](dispatch-map.md); the measurement contract and
null floors are [`RESULTS.md`](../../RESULTS.md). Beta-3 indicative — revalidate on stable macOS.

Terminals: `nax_dense` (dense NAX matmul2d), `v6nax_sparse` (sparse NAX), `mfa_primitive`
(STEEL-family primitive / decode), `gna_v6nax` / `gna_steel` (neighborhood), `varlen_v6nax`
(packed varlen), `sdpa` (MLX fallback).

---

## 1. Dense forward — `flash_attention(..., backend="auto")`

**2.64 (decision D1, 2026-10-01): dense D=128 delegates to SDPA.** The 2.63 `nax_dense` route ran
on 10/10 production D=128 shapes, never byte-identical to SDPA, 5–11 % slower on 5/10
(`devnotes/production_shapes_2026-10.md` §1, archived).

| Eligibility (source) | Terminal | Evidence |
|---|---|---|
| D=128, f16/bf16, plain self-attn, no bias/window/dropout/attn-weights, matching dtype/seqlen | **`sdpa`** — `_fallback_sdpa` = `mx.fast.scaled_dot_product_attention`, byte-identical | production shapes (2.64 M5 gate `prod_dense_*_auto_sdpa` = 0.0) |
| … inside a `dispatch_policy.DENSE_TILE_TABLE` row (non-causal, `Hq == Hk`, exact B·H) | **`nax_dense`**, tile **32·32·2** | each row ≥ 2× its noise floor vs SDPA at every measured point (inline evidence; tile_vs_sdpa 2026-10-01) |
| … with `MFA_ENABLE_V6_DENSE=1`, N ≥ `MFA_V6_DENSE_MIN_N` (default 2048) | `nax_dense` (default tile BQ64/BK32/WM4) | the 2.63 route, explicit |
| D=64 plain forward | `sdpa` (unless a decode carveout applies) | — |
| D=512 · fp32 · unsupported feature combo | `sdpa` | — |

- Predicate: `mlx_mfa/attention.py` `_select_dense_backend` → `("sdpa" | "nax_dense", reason, tile)`;
  table + guard + priors: `mlx_mfa/dispatch_policy.py` (`DENSE_TILE_CANDIDATES` →
  `DENSE_TILE_TABLE`, `DENSE_TILE_PRIORS`, `calibrate_dense_tile_priors`).  The tile is passed per
  call through the `v6_nax_forward` binding into the primitive (never via env at lazy-eval time).
- `MFA_DISABLE_V6_DENSE=1` → `sdpa` everywhere (wins over the table and the knob).
- Verbose: `MLX_MFA_VERBOSE_DISPATCH=1` prints `terminal=<backend>` at the terminal that runs;
  policy predicates print `policy:` lines only (the 2.63 "-> SDPA optimal" line preceded a
  `nax_dense` terminal).

## 2. Sparse gate — `flash_attention_sparse(...)`

**2.64 (decision D3): the former extended envelope is the default law.** Authoritative predicate:
`mlx_mfa/lcsa_nax.py` `_nax_sparse_route_viable` (capacity, then the law); near-dense:
`_quasi_dense_nax_viable`; exact BT64→32 expansion: `_expand_bt64_exact` (before `auto_pad`).

- Capacity: `block_tile == 32`, f16/bf16, D ∈ {64,128}, V matching Q/K, mask ≥ 4096 B.
- Non-causal law: both lengths in [`SPARSE_NAX_MIN_N`=2048, `SPARSE_NAX_MAX_N`=200000], `qL ≠ kL`
  allowed (B5; kernel-documented, exact on the FlashVSR shapes), any B·H (coverage
  `SPARSE_NAX_MEASURED_BH_COVERAGE` = {1,4,12,16,32,40,56}), density ≤ 0.50; below N=8192 the
  measured lower ceilings hold (D128 B·H4 0.05, D64 B·H12 0.25). Ceiling 0.50 (Marco 2026-10-01; 2.63:
  0.30): Volet A sliding "d0.50" cells (measured block density 0.44, B1H40 D128, N 16384–144288)
  1.61–2.04× vs dense SDPA (M5 Max, MLX 0.31.2, 2.63-era kernel, 2026-08-12); 2.64 re-probe through the
  public API vs SDPA + bool mask at density 0.42–0.48: 1.10–2.29× on 7/8 engaged cells, B·H1 N2048 D128
  d0.48 0.91× (sub-ms, in-process n=6 — indicative; M5 Max, MLX 0.31.2, mlx-mfa 2.64.0, 2026-10-01).
  Causal cells and the legacy 2.63 policy keep 0.30.
- Near-dense (B6): density ≥ `MFA_SPARSE_D_DENSE_CUTOFF` (0.85) → `v6nax_sparse` whenever the kernel
  can serve the call (non-causal, aligned or `auto_pad`, law N bounds), else `sdpa` + bool mask.
- Causal: unchanged 2.63 cells, `qL == kL` (U2).
- Fallbacks: SDPA with a **bool** keep-mask (B2), empty rows → zeros per element row, size guard
  `MFA_SPARSE_FALLBACK_MAX_BYTES` (4 GiB; B1: rescue to the NAX kernel or refuse, never allocate).
- `MFA_SPARSE_NAX_LEGACY_POLICY=1` → the 2.63 default policy (one release; not the 2.63 extended opt-in). `MFA_SPARSE_NAX_EXTENDED` →
  deprecated no-op (warns).
- Evidence: Volet A (16k–144k B1H40), DAY-3 Block 1 (B·H16, 204 cells, 0 loss), production shapes
  Phase 2 (N 200 000, B·H 56: engaged, row-correct; NAX time 0.10–0.19× of dense SDPA — M5 Max,
  MLX 0.31.2, macOS 27.2, mlx-mfa 2.63.0 kernels, 2026-09-30, `devnotes/production_shapes_2026-10.md` §2).

- *New evidence (out of scope):* N6144 cells the gate routes without a July datum — random d0.05
  B·H4 D128 **2.65×**, d0.30 B·H12 D128 **2.19×**, d0.25 B·H12 D64 **1.95×**
  (`reval_C_*_n6144_*`).

**2.63 delegations → `sdpa` — since 2.64 only under `MFA_SPARSE_NAX_LEGACY_POLICY=1`** (the `return False` branches of `_route_viable_263`; historical line refs; `lcsa_nax.py:309`
"measured-loss or unmeasured cells delegate to SDPA"):
- **N=2048 (all)** — `qL` outside `[4096, 8192]` (range gate `:394-395`; `SPARSE_NAX_MIN_N=4096` `:336`).
- **B·H=1 below N=8192** — B·H=1 routes only via the N=8192 all-cells branch (`:391-393`); at
  N∈[4096,8192) the per-region branches (`:396-401`) admit only bh ∈ {4,12}, so bh=1 → `return False` (`:402`).
- **density above the region ceiling** — every cell with `density >` its ceiling.
- **sliding B·H=1 N=4096** — outside the region (commit `d3836d3` contraction); July 0.767/0.787
  loss retained as the *motivation* for the contraction, not a route.

## 3. Decode carveouts — `flash_attention(...)` narrow envelopes

Predicate: `mlx_mfa/dispatch_policy.py:168` (`_M5_NAX_DECODE_EDGE_MAX_KV_LEN = 65536`), `:170-173`
(per-qL kL floors + GQA sets), `:214` (`_m5_nax_decode_edge_carveout`).

| Exact envelope | Terminal | Measured 30/07 |
|---|---|---|
| qL=8, D=64, GQA=8, non-causal, f16/bf16, **4096 ≤ kL ≤ 65536** | **`mfa_primitive`** | **1.25× / 1.27×** at kL=4096 — `benchmarks/results/reval_decode/` (cf. RESULTS.md §Decode). *New evidence:* kL=8192 **1.39×**, 16384 **1.58×**, 32768 **1.62×** |
| qL=16, D=64, GQA ∈ {4,8,16}, non-causal, f16/bf16, **16384 ≤ kL ≤ 65536** | `mfa_primitive` | consolidation remeasure (`:172-173`) |
| every adjacent cell (qL=4, kL below floor, …) | `sdpa` | — |

## 4. Backward — `mx.grad(flash_attention(...))`

Predicate: `mlx_mfa/dispatch_policy.py:416` (`_v6nax_backward_carveout`); **active body**
`D==64 and seq_len >= 2048 and dtype ∈ {float16,bfloat16} and not MFA_DISABLE_V6_BACKWARD` →
**default-on** (the docstring's `MFA_ENABLE_V6_BACKWARD=1` is stale; the body is disable-only).

| Eligibility | Terminal | Measured 30/07 |
|---|---|---|
| D=64, qL ≥ 2048, f16/bf16 (causal & non-causal) | V6NAX split backward (Apple-SDPA fwd carveout) | **2.50× / 2.77×** (D64 B·H4 N4096 causal, sdpa-vjp / v6) — `benchmarks/results/reval_A_bwd_{v6,sdpa}_order{A,B}.json` |
| D=128 backward | SDPA-vjp | (V6NAX D128 bwd measured slower — excluded, `:490`) |

## 5. Packed varlen — `flash_attention_varlen*(...)` (opt-in)

Predicate: `mlx_mfa/attention.py:6650` (`_varlen_v6nax_eligible`), gated by
`:6659` **`MFA_ENABLE_VARLEN_NAX`** (opt-in, default-off); **D=128 only** (`:6665`
`q.shape[-1] != 128 → return False`) → `:6902` `v6_nax_varlen_forward`.

| Eligibility | Terminal | Measured 30/07 |
|---|---|---|
| `MFA_ENABLE_VARLEN_NAX=1`, packed QKV, **D=128**, tile BQ32/BK32/WM2 | **`varlen_v6nax`** | median **1.329× / 1.344×** across **16 geometries** — `benchmarks/results/reval_A_varlen_order{A,B}.json` |
| default (opt-in off) | STEEL varlen / `sdpa` | — |

## 6. GNA — `flash_attention_gna(...)`

Predicate: `mlx_mfa/attention.py:167-168` (`_GNA_NAX_D128_MIN_N = 2048`, `_GNA_NAX_D64_MIN_N = 4096`).

| Envelope | Terminal | Measured 30/07 |
|---|---|---|
| 3-D, f16/bf16, D=128, N ≥ 2048 | **`gna_v6nax`** | **2.39× / 2.44×** (D128 N4096 fp16, 3D 1×7×7) — `benchmarks/results/reval_A_gna_public_order{A,B}.json`; **2.45×** `reval_E_gna_*` |
| 3-D, f16/bf16, D=64, N ≥ 4096 | `gna_v6nax` | — |
| D=128 below the NAX threshold | `gna_steel` | — |
| native disabled / unsupported dim | sparse fallback | — |

## 7. Conv3D spatial-pad (SeedVR2 VAE, opt-in — not an attention surface)

The `mx.conv_general` hook (`mlx_mfa/_auto_hooks.py`) rescues specific 3-D VAE convs whose ONLY
MPP-ineligibility axis is H/W % 8: it zero-pads H/W to the next multiple of 8, runs
`conv3d_nax_forward`, and slices back (exact for the original output extent). Opt-in
`MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE=1` (default-off).

| Eligibility (`_auto_hooks.py:324-353`) | Terminal | Measured |
|---|---|---|
| fp16, B=1, 3×3×3, stride 1, pad (0,0,1,1,1,1), H/W not %8, input `(T,H,W,C_in,C_out)` ∈ allowlist | `conv3d_nax_spatial_pad_slice` | see below |
| any other conv | native `mx.conv_general` | — |

Allowlist `_CONV3D_SPATIAL_PAD_FAMILIES` (`_auto_hooks.py:300`), keyed on the conv **input**
`(T,H,W,C_in,C_out)`:
- **108×132** (family #1): `(4,108,132,512,512)`, `(5,108,132,512,512)` — SeedVR2 /debug/117 A/B
  under fp16 measured **+38.78 s (9.3% E2E)**, engagement 2257, SSIM = fp16 class.
- **54×66** (family #2, added 2.62.1): `(3,54,66,512,512)`, `(4,54,66,512,512)` — census-authoritative
  input-T (SeedVR2 /debug/116; input-T = output-T + 2, one level deeper than 108×132 so {3,4} not
  {4,5}). mfa-side isolated microbench (fresh process × two orders,
  `benchmarks/bench_conv3d_spatial_pad.py` → `benchmarks/results/conv3d_spatial_pad_family2_order{A,B}.json`):
  native/spatial-pad **1.06–1.37× (input-T3 dominant)**, **2.15–2.49× (input-T4 boundary)**, correctness
  cos ≈ 1.0 vs native. E2E confirmation deferred to SeedVR2 /debug/118 (~8 s predicted).

Not rescued (remain native `mx.conv_general`): stride-2 downsamples (no MPP-NAX for stride≠1),
channel-tail convs (C_in/C_out < 32), bf16 (fp16-only gate — locked inert).

## 8. Knobs & opt-ins

Full registry: [`ENV_VARS.md`](../../ENV_VARS.md). Status of the routing knobs referenced above:
`MFA_ENABLE_V6_DENSE` (2.64: explicit dense NAX), `MFA_V6_DENSE_MIN_N` (its threshold, default 2048), `MFA_DISABLE_V6_DENSE` (opt-out), `MFA_DISABLE_V6_BACKWARD`
(opt-out; D64 bwd default-on), `MFA_ENABLE_VARLEN_NAX` (opt-in, default-off), `MFA_ENABLE_CONV3D_*`
(conv opt-ins, default-off). The sparse gate's DEFAULT is the 2.64 law (§2);
`MFA_SPARSE_NAX_LEGACY_POLICY=1` restores the 2.63 default policy for one release, `MFA_SPARSE_NAX_EXTENDED`
is a deprecated no-op, `MFA_SPARSE_D_DENSE_CUTOFF` (default 0.85) sets the near-dense threshold, and
`MFA_SPARSE_FALLBACK_MAX_BYTES` (4 GiB) bounds the SDPA fallback's mask.
