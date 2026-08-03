# Runtime routing inventory — "for my shapes, what routes where?"

**Reading rule.** The eligibility *predicate at the source* is the authority — this document is
an **inventory generated from that source and the 2026-07-30 regulator-controlled revalidation
evidence** (Apple M5 Max, macOS 27 beta, MLX 0.31.2, `_ext` build of commit `558a191`). Every
routing condition cites `file:line`; every measured ratio cites its evidence JSON by path and is
`sdpa / native` median across both process orders (values > 1 favour mlx-mfa). The locked terminal
map is [`docs/dispatch-map.md`](dispatch-map.md); the measurement contract and
null floors are [`RESULTS.md`](../../RESULTS.md). Beta-3 indicative — revalidate on stable macOS.

Terminals: `nax_dense` (dense NAX matmul2d), `v6nax_sparse` (sparse NAX), `mfa_primitive`
(STEEL-family primitive / decode), `gna_v6nax` / `gna_steel` (neighborhood), `varlen_v6nax`
(packed varlen), `sdpa` (MLX fallback).

---

## 1. Dense forward — `flash_attention(..., backend="auto")`

| Eligibility (source) | Terminal | Measured 30/07 |
|---|---|---|
| D=128, f16/bf16, N ≥ `MFA_V6_DENSE_MIN_N` (default **2048**), plain self-attn, no bias/window/dropout/attn-weights, matching dtype/seqlen | **`nax_dense`** | parity-or-win **1.067× / 1.071×** (D128 N4096 fp16) — `benchmarks/results/reval_A_dense_public_order{A,B}.json`; **1.08×** — `reval_E_dense_*` |
| D=64 plain forward | `sdpa` (unless a decode carveout applies) | — |
| D=512 · fp32 · unsupported feature combo | `sdpa` | — |

- Predicate: `mlx_mfa/attention.py:52` (`_V6_DENSE_MIN_N_DEFAULT = 2048`), `:521` (reads
  `MFA_V6_DENSE_MIN_N`), `:537` (`return ("nax_dense", "auto D128 N>=v6_min_n")`). Below N=2048,
  D=64/512, fp32 → `sdpa` (`docs/dispatch-map.md:11-14`).
- Kernel default tile BQ64/BK32/WM4 (`csrc/mfa_v6_nax_primitive.cpp`; scale baked, cache-keyed).
- Opt-out: `MFA_DISABLE_V6_DENSE=1` → `sdpa`. Keep-all-paths: `MFA_V6_DENSE_MIN_N=0` forces NAX at all N.

## 2. Sparse gate — `flash_attention_sparse(...)`

Authoritative predicate: `mlx_mfa/lcsa_nax.py:350-402` (`_nax_sparse_route_viable`), constants
`:336-345`. Routes to **`v6nax_sparse`** ONLY inside the β3-measured region (unmeasured B·H and
causal cells are deliberately **not** interpolated). Common gates (`:355-366`): `block_tile ∈ {32}`,
dtype f16/bf16, D ∈ {64,128}, `qL == kL`, `qL ≤ 8192`. `bh = B·H`.

### Non-causal (`lcsa_nax.py:385-402`)
| N | B·H | D | density ≤ | source |
|---:|---:|---:|---:|---|
| 8192 | 1, 4, 12 | 64, 128 | 0.30 | `:391-393` (won all 36 fp16 cells) |
| 4096–8192 | 12 | 128 | 0.30 | `:396-397` |
| 4096–8192 | 12 | 64 | 0.25 | `:398-399` (`_D64_BH12_DENSITY_CEILING`) |
| 4096–8192 | 4 | 128 | 0.05 | `:400-401` (`_D128_BH4_DENSITY_CEILING`) |
| 4096–8192 | 12 | 128 | 0.30 | bf16 only region (`:385-388`) |

### Causal (`lcsa_nax.py:370-380`)
| N | B·H | D | density ≤ | source |
|---:|---:|---:|---:|---|
| 4096 | 4 | 128 | 0.10 | `:376` (`_CAUSAL_BH4_DENSITY_CEILING`) |
| 4096 | 12 | 128 | 0.30 | `:378` |
| 8192 | 12 | 64, 128 | 0.30 | `:380` |
| 4096 (bf16) | 4 | 128 | 0.10 | `:372-374` |

**Measured 30/07 by region** (`benchmarks/results/reval_C_*`, `reval_B_causal_*`):
- Non-causal N8192 B·H12 D128 wins **up to 8.23×** — density-dependent (sliding-window and low
  random density win biggest, higher block density less): sw128 **8.23× / 8.22×**
  `reval_C_sw128_nc_fp16_b1_h12_n8192_d128_*`; sw256 7.25/7.30; random d0.05 6.70/6.70; sw512
  6.33/5.38; d0.15 4.88/3.97; d0.30 2.50/2.94.
- N8192 B·H12 D64 up to **4.58×**; B·H4 D128 up to **4.15×** (`reval_C_*_b1_h4_*`, `*_d64_*`).
- Causal D128 N8192 B·H12 d0.30 **3.83× / 3.86×** — `reval_B_causal_*` (re-measure of the July
  locked **3.8833× / 3.8485×**; confirms the 3.85–3.88× band).
- *New evidence (out of scope):* N6144 cells the gate routes without a July datum — random d0.05
  B·H4 D128 **2.65×**, d0.30 B·H12 D128 **2.19×**, d0.25 B·H12 D64 **1.95×**
  (`reval_C_*_n6144_*`).

**Explicit delegations → `sdpa`** (the `return False` branches; `lcsa_nax.py:309`
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

## 7. Knobs & opt-ins

Full registry: [`ENV_VARS.md`](../../ENV_VARS.md). Status of the routing knobs referenced above:
`MFA_V6_DENSE_MIN_N` (default 2048), `MFA_DISABLE_V6_DENSE` (opt-out), `MFA_DISABLE_V6_BACKWARD`
(opt-out; D64 bwd default-on), `MFA_ENABLE_VARLEN_NAX` (opt-in, default-off), `MFA_ENABLE_CONV3D_*`
(conv opt-ins, default-off). The sparse gate has no env override — it is the `_nax_sparse_route_viable`
predicate alone.
