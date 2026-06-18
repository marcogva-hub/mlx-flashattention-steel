# Sparse / LCSA Family — Verified Per-Kernel Spec (audit B1, durable reference)

RUNTIME-verified on M5/26.6. Correctness vs an INDEPENDENT manual fp32 oracle (not SDPA, not another
kernel — lesson #11), locked by `tests/test_sparse_family_correctness_lock.py` (13 cells) +
`tests/test_dispatch_map_lock.py`. Labels: [V]erified / [D]educed.

## 1. The sparse forward — TWO kernels, work-product-selected

`flash_attention_sparse` (symmetric mask) → `_sparse_nax_with_sdpa_vjp` → `sparse_attention_dispatch`
→ `sparse_attention_nax` → `_ext.sparse_attention_forward(kernel_version=decide_auto_version(...))`.
**`decide_auto_version`: `qL*kL*D >= 2_147_483_648` (= 4096·4096·128) → "v2", else "v1"** (env
`MFA_LCSA_KERNEL_VERSION` overrides). [V — read + env-toggle]

| | **V2 — `sparse_kernel_source_v2`** | **V1 — `sparse_kernel_source`** |
|---|---|---|
| MMA form | **NAX `matmul2d`** cooperative-tensor (`BaseNAXFrag::mma`, desc 16,32,16) | **per-thread SCALAR** (`float q_vec[cD]`, scalar `acc += p[kc]*V`) |
| Selected when | work ≥ 2.147e9 (e.g. D=128 N≥4096) | work < 2.147e9 (D=64 any N≤8k; D=128 N<4096) |
| Speed (D=128 N=4096 d=0.25) | **1.20 ms** (env-toggle: default==v2) | **49.3 ms** (~41× slower) [V] |
| Algorithm | block-sparse flash: matmul2d QK^T + online softmax + mask-gated block skip + matmul2d P@V | block-sparse flash: scalar QK^T + online softmax + block skip + scalar P@V |
| Paper-fidelity | **faithful block-sparse FlashAttention-2** (no "inspired-by" deviation found) [V] | faithful; scalar (unvectorized) |

**Both compute the same masked attention** — fp32-oracle-verified across banded, scattered,
density→{min,1.0}, all-masked query-block, causal, GQA, mask-ndim 2/3/4: **max_abs_err ≤ 7.9e-5,
all finite** (V2) / ≤ 7.9e-5 (V1). [V — locked tests]

**Constraints:** D∈{64,128}; fp16/bf16; block_tile∈{16,32,64}; symmetric mask (`bt_q==bt_k`) to
engage on M5+ (asymmetric → SDPA fallback, §4); mask total bytes ≥ 4096 (MLX inlines smaller buffers);
mask-ndim 2/3/4; GQA `Hq%Hk==0`. V2 additionally requires `block_tile==32` and `!causal` (causal →
V1). [V]

## 2. The mask machinery

`_steel_block_config(D)`: D=64 → (BQ=32, BK=32) **symmetric**; D=128 → (BQ=32, BK=16) **asymmetric**.
[V] Every mask-maker derives its block size from this, so **D=64 masks are symmetric, D=128 masks are
asymmetric** — the root of the §4 fallback. `make_causal_block_mask` / `make_sliding_window_mask` /
`make_strided_mask` / `make_lcsa_mask` all emit at this granularity. [V]

## 3. Mask-faithfulness (the Phase-F premise — established, not implemented)

Would a **D=128 symmetric 32×32** convention faithfully represent the patterns vs the current 32×16?
**YES — byte-identical.** Comparing the 32×32 (D=64-conv) vs 32×16 (D=128-conv) block masks of the
SAME pattern, fp32 attention is **Δ=0.0e+00** for sliding-window, causal, AND strided; both differ
from the exact element-level pattern by the same 1.1e-1 (the inherent, granularity-independent
block-sparse approximation). [V] **⇒ Phase F can route D=128 onto the symmetric V2 kernel by
regenerating masks at 32×32 with zero correctness cost** (for these patterns). Caveat: strided with
stride>BK and LCSA's top-k component were not separately isolated here — [D] likely also faithful;
verify in Phase F. **Do NOT OR-merge an existing 32×16 mask to 32×32** (superset → denser → not
faithful); regenerate at 32×32.

## 4. SDPA-fallback paths (part of the sparse entry's runtime behavior)

- `_sparse_fallback_sdpa_perhead` (`attention.py:3239`): M5+ **asymmetric** mask → dense Apple SDPA +
  per-head block-expanded bias. byte-identical to `mx.fast.sdpa` (Δ=0.0). Reason: the asymmetric STEEL
  kernel is disabled by the `(long)p->NK` compiler miscompile (`docs/v6-nax/sparse-bug-investigation.md`).
  [V] **Gotcha 1** (loses the 1.7–4.2× sparse win at D=128).
- `_sparse_fallback_sdpa`: ndim-3/4 cross-head-union fallback. [V]

## 5. Sparse backward

- **Default** (no env): `_sparse_nax_with_sdpa_vjp` → **dense SDPA-vjp** (dQ Δ=0.0 vs SDPA-vjp). The
  sparse forward win does NOT carry to the backward. [V] **Gotcha 3.**
- **Opt-in** `MFA_ENABLE_V6_BACKWARD=1` + D∈{64,128} + N≥2048 + ndim==2 + **bt≥64**: hybrid (dV native
  + dQ/dK SDPA-vjp; dQ Δ=0.0 by design). Declined-on-perf (Pattern #6). [V]
- **Opt-in** `MFA_V6_BWD_SPARSE_NATIVE=1`: full-native. Declined-on-perf (Pattern #6, native < SDPA-vjp). [D — source + cartography]

## 6. Routing status (dispatch-map cells)

D=128 symmetric → V2 matmul2d (routed-as-intended, the win); D=128 asymmetric / built-in makers →
SDPA fallback (silent, gotcha 1); D=64 / D=128-small-N → V1 scalar (real but ~40× slower, gotcha 2);
backward → SDPA-vjp (gotcha 3). See `dispatch-map.md`.

## 7. Comment sweep (B1, comment-only fixes)
- `mfa_sparse_attention.cpp:13` "Phase 1.3 *will* swap … matmul2d" → corrected (the swap is DONE in V2).
- `lcsa_nax.py:21` "Phase 1.5 *will* introduce sparse_attention_dispatch" → corrected (it exists).
- "V1 kernel only at PoC stage" (lcsa_nax.py:183, cpp:1101) → corrected (production; V1 is the LSE path).
