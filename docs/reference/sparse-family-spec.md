# Sparse / LCSA Family — Verified Per-Kernel Spec (audit B1, durable reference)

RUNTIME-verified on M5/26.6. Correctness vs an INDEPENDENT manual fp32 oracle (not SDPA, not another
kernel — lesson #11), locked by `tests/test_sparse_family_correctness_lock.py` (13 cells) +
`tests/test_dispatch_map_lock.py`. Labels: [V]erified / [D]educed.

## 1. The sparse forward — TWO kernels, work-product-selected

`flash_attention_sparse` (symmetric mask) → `_sparse_nax_with_sdpa_vjp` → `sparse_attention_dispatch`
→ `sparse_attention_nax` → `_ext.sparse_attention_forward(kernel_version=decide_auto_version(...))`.
**`decide_auto_version` (since Phase F, 2026-06-18): route by V2-CAPABILITY — `D in {64, 128}` → "v2",
else (e.g. D=256) → "v1"** (env `MFA_LCSA_KERNEL_VERSION` overrides). The old `qL*kL*D >= 2_147_483_648`
(= 4096·4096·128) work-product gate is **RETIRED** — Phase E measured V1-scalar is never fastest, and the
threshold only mis-routed D=64 (always < 2³¹) and D=128 small-N to the slow V1. The C++
`sparse_attention_forward` still falls v2→v1 internally when V2 is ineligible (causal / block_tile≠32),
so V1 remains the genuine fallback, never the default for a V2-capable shape.
[V — `mlx_mfa/lcsa_nax.py:59-106`]

| | **V2 — `sparse_kernel_source_v2`** | **V1 — `sparse_kernel_source`** |
|---|---|---|
| MMA form | **NAX `matmul2d`** cooperative-tensor (`BaseNAXFrag::mma`, desc 16,32,16) | **per-thread SCALAR** (`float q_vec[cD]`, scalar `acc += p[kc]*V`) |
| Selected when (Phase F) | D∈{64,128} default (V2-capable head dims); fp16/bf16, block_tile==32, !causal | only the genuine fallback — D∉{64,128}, or V2 ineligible (causal / block_tile≠32) falls back here in C++ |
| Speed (D=128 N=4096 d=0.25) | **1.20 ms** (env-toggle: default==v2) | **49.3 ms** (~41× slower) [V] |
| Algorithm | block-sparse flash: matmul2d QK^T + online softmax + mask-gated block skip + matmul2d P@V | block-sparse flash: scalar QK^T + online softmax + block skip + scalar P@V |
| Paper-fidelity | **faithful block-sparse FlashAttention-2** (no "inspired-by" deviation found) [V] | faithful; scalar (unvectorized) |

**Both compute the same masked attention** — fp32-oracle-verified across banded, scattered,
density→{min,1.0}, all-masked query-block, causal, GQA, mask-ndim 2/3/4: **max_abs_err ≤ 7.9e-5,
all finite** (V2) / ≤ 7.9e-5 (V1). [V — locked tests]

**Constraints:** D∈{64,128}; fp16/bf16; block_tile∈{16,32,64}; symmetric mask (`bt_q==bt_k`) to
engage on M5+ (asymmetric → SDPA fallback, §4); mask total bytes ≥ 4096 (MLX inlines smaller buffers);
mask-ndim 2/3/4; GQA `Hq%Hk==0`. V2 eligibility (C++ `sparse_attention_forward`) =
`D∈{64,128} && block_tile==32 && (fp16 || bf16) && !causal`; otherwise transparently falls back to V1.
**bf16 now routes to V2** (the old `is_f16`-only gate was lifted — gotcha 4 fix; previously bf16 silently
fell to the ~50× slower V1 scalar). [V — `csrc/mfa_sparse_attention.cpp:1046-1054`]

**Mask dtype contract (path-dependent, documented current behavior):** callers should
provide a boolean block mask. The direct V6NAX entries pass it to C++, where non-`bool`
is rejected. The residual STEEL path converts accepted masks to contiguous `uint8`, and
the SDPA fallback expands truth values into an additive bias; on those fallback paths
zero means inactive and nonzero means active. Sparse-backward conversion also casts to
`bool`. Numeric-mask acceptance by a fallback is therefore not a portable V6NAX
contract. [V — `csrc/mfa_sparse_attention.cpp`, `mlx_mfa/attention.py`]

## 2. The mask machinery

The built-in mask-makers derive their block size from `masks.py::_bq_bk(D)`, which since **Phase F**
(2026-06-18) is **DECOUPLED** from STEEL's `_steel_block_config`: `_bq_bk(64)=(32,32)`,
`_bq_bk(128)=(32,32)` **symmetric** (was the asymmetric 32×16 that mirrored STEEL — the old root of the
§4 fallback), `_bq_bk(256)=(32,16)` (NAX-sparse unsupported at D=256). So **both D=64 AND D=128 makers
now emit symmetric masks → auto-route to the NAX V2 kernel.** `make_causal_block_mask` /
`make_sliding_window_mask` / `make_strided_mask` / `make_lcsa_mask` all emit at this granularity.
(STEEL's own dense `_steel_block_config` still uses 32×16 at D=128 — intentionally separate.)
[V — `mlx_mfa/masks.py:36-56`]

## 3. Mask-faithfulness (Phase F — SHIPPED 2026-06-18)

A **D=128 symmetric 32×32** convention faithfully represents the patterns vs the old 32×16:
**byte-identical.** Comparing 32×32 (D=64-conv) vs 32×16 (D=128-conv) block masks of the
SAME pattern, fp32 attention is **Δ=0.0e+00** for sliding-window, causal, AND strided; both differ
from the exact element-level pattern by the same 1.1e-1 (the inherent, granularity-independent
block-sparse approximation). [V] **Phase F SHIPPED this**: the built-in D=128 makers now emit
symmetric 32×32 (`masks.py::_bq_bk(128)=(32,32)`), so the auto-route (`bt_q==bt_k`) sends them to the
real NAX V2 kernel with zero correctness cost. The masks were **regenerated at 32×32**, NOT OR-merged
from 32×16 (superset → denser → not faithful). See `dispatch-map.md` §Phase F.

## 4. SDPA-fallback paths (part of the sparse entry's runtime behavior)

- `_sparse_fallback_sdpa_perhead`: M5+ **asymmetric / custom** mask (`bt_q≠bt_k`) → dense Apple SDPA +
  per-head block-expanded bias. byte-identical to `mx.fast.sdpa` (Δ=0.0). Reason: the asymmetric STEEL
  kernel is disabled by the `(long)p->NK` compiler miscompile (`.doc-archive/docs/v6-nax/sparse-bug-investigation.md`).
  [V] **Gotcha 1 status:** since Phase F the built-in D=128 makers emit symmetric 32×32 → NAX (they no
  longer hit this fallback); the residual SDPA path now only applies to genuinely asymmetric/custom masks
  or masks <4096 bytes, or near-dense masks (density ≥ ceiling 0.78) — all intentional. See `dispatch-map.md`.
- `_sparse_fallback_sdpa`: ndim-3/4 cross-head-union fallback. [V]

## 5. Sparse backward

- **Default** (no env): `_sparse_nax_with_sdpa_vjp` → **dense SDPA-vjp** (dQ Δ=0.0 vs SDPA-vjp). The
  sparse forward win does NOT carry to the backward. [V] **Gotcha 3.**
- **Opt-in** `MFA_ENABLE_V6_BACKWARD=1` + D∈{64,128} + N≥2048 + ndim==2 + **bt≥64**: hybrid (dV native
  + dQ/dK SDPA-vjp; dQ Δ=0.0 by design). Declined-on-perf (Pattern #6). [V]
- **Opt-in** `MFA_V6_BWD_SPARSE_NATIVE=1`: full-native. Declined-on-perf (Pattern #6, native < SDPA-vjp). [D — source + cartography]

## 6. Routing status (dispatch-map cells, after Phase F + bf16 fix)

D=128 symmetric (incl. built-in makers) → V2 matmul2d (the 1.7–4.2× win, **gotcha 1 FIXED**); D=128
asymmetric/custom or <4096-byte or density≥0.78 → SDPA fallback (intentional); D=64 → V2 always
(**gotcha 2 FIXED** — `decide_auto_version` retired the 2³¹ gate, ~9× vs old V1); bf16 symmetric → V2
(**gotcha 4 FIXED**, was the ~50× V1 scalar); backward → SDPA-vjp (gotcha 3, by design). V1 scalar is
now only the genuine D∉{64,128} fallback. See `dispatch-map.md`.

## 7. Comment sweep (B1, comment-only fixes)
- `mfa_sparse_attention.cpp:13` "Phase 1.3 *will* swap … matmul2d" → corrected (the swap is DONE in V2).
- `lcsa_nax.py:21` "Phase 1.5 *will* introduce sparse_attention_dispatch" → corrected (it exists).
- "V1 kernel only at PoC stage" (lcsa_nax.py:183, cpp:1101) → corrected (production; V1 is the LSE path).
