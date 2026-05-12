# Sprint B Phase 1.0 — Decisions Companion

**Date:** 2026-05-12
**Branch:** `experiment/lcsa-nax-phase1_0_design`

D-numbered decisions made during Sprint B Phase 1.0. Numbering continues
from Sprint A's last D-number (Sprint A's decisions used D1-D??, Sprint
B starts fresh at S-B D-numbers to avoid collision with Sprint C/D
numbering).

To distinguish: Sprint C/D used D-numbers like D1-D36. Sprint B uses
**B-D1 through B-D10**.

---

## B-D1 — Algorithm: Option α (block-skip dispatch via dense matmul2d)

Phase 0 §10 selected Option α with rationale anchored on Apple MPP NAX
being dense-only. Sprint B Phase 1.0 ratifies — no algorithm change.

Alternatives considered + rejected in Phase 0:
- Option β (CSR layout): premature for FlashVSR's density range
- Option γ (custom NAX MMA): ~10× implementation effort, marginal upside
- Option δ (defer/shelve): rejected per Phase 0 §12 PROCEED verdict

---

## B-D2 — C++ Primitive via `mlx::core::fast::metal_kernel` from Phase 1.1

Inherit Sprint D D33: skip the "Python orchestrator first, migrate
later" pattern. C++ Primitive ships directly in Phase 1.1.

Rationale:
1. Sprint C/D proved the `fast::metal_kernel` pattern works for
   dispatch-level orchestration (Conv3D NAX).
2. Avoids the ~3-4h migration work that Sprint D ran for Conv3D.
3. Bench numbers from Phase 1.5 are directly meaningful (no
   Python overhead to account for).

---

## B-D3 — Unified `SparseAttnKey` cache (single `unordered_map`)

Inherit Sprint C D3: no per-Kind separate maps. Single
`unordered_map<SparseAttnKey, void*, …>` with `Kind` enum
discriminator. Bounded LRU at 32 entries.

---

## B-D4 — Block-tile (BT) defaults: 32 for density > 0.10, else 64

Phase 1.0 design §6 establishes preliminary defaults. Phase 1.3
autoresearch refines per shape cluster. Env var `MFA_LCSA_BT` for
override.

Rationale:
- BT=32 maximizes sparsity exploitation at modest density (0.12-0.24)
- BT=64 reduces dispatch overhead at very-sparse density (0.03-0.07)
- BT=16 / 128 reserved for autoresearch exploration

---

## B-D5 — Sub-phase 0 microbench: targeted re-bench at per-tile shapes

Sprint C's microbench covered large-M shapes (M=20k-1M, K=3.5k-14k).
Sprint B per-tile matmul shapes (M=16-128, K=64-128, N=16-128) aren't
covered → run targeted re-bench at Phase 1.1.

Gate: median sustained TFLOPS on dominant per-tile shape ≥ 5 TF
(relaxed from Sprint C's 30 TF gate). Small per-tile shapes can't
hit peak; the gate is "meaningful throughput, not peak."

---

## B-D6 — Three-axis validation mandatory (`CLAUDE_V6_NAX` rule)

Every test pack in Phase 1.x must cover three axes:
1. Output sanity (RMSE oracle + sentinel-fill)
2. Path entered (perf A/B verifies NAX-native path beats fallback)
3. Edges preserved (all-False row → 0 / NaN; all-True → dense
   equivalence)

Per `docs/proposed-claude-v6-nax-updates.md`. Codified institutional
rule (Sprint C/D/v2.33.1 arc evidence).

---

## B-D7 — All-False mask row: output = 0 (not NaN), matching v2.33.1

Phase 1.0 design §2 specifies: when all K-tiles for a Q-tile are
masked, the online softmax denominator stays at `denorm_min` (initial
small ε), `O_partial / l_running` ≈ 0. Output = 0 for that Q-tile.

This **matches v2.33.1's `_sparse_fallback_sdpa_perhead` behavior**
(MLX SDPA with all-`-inf` mask row → softmax of all `-inf` → NaN
→ 0 × NaN = NaN; then NaN / 0 = NaN in IEEE 754 → MLX may sanitize to 0).

Three-axis validation D6 requires explicit test for this edge case.
Phase 1.1 includes `test_sparse_nax_all_false_row_zero_output`.

---

## B-D8 — Causal handling: per-tile diagonal correction (not block-mask
modification)

When `causal=True`, the block_mask may include diagonal-and-below
blocks, but within-block (BT × BT) causal masking is still needed.
Sprint B Phase 1.2 design: apply causal correction inside the kernel
at the per-tile S = Q @ K^T step:

```msl
if (causal && k_tile_idx == q_tile_idx) {
    // Within-block causal: zero out upper triangle of S
    apply_within_tile_causal(S);
}
```

Alternative considered + rejected: pre-process block_mask to AND with
a causal block_mask. Rejected because it doesn't handle within-block
causal masking (only block-level).

---

## B-D9 — `sparse_attention_forward` is additive to `flash_attention_sparse`

Sprint B doesn't replace the v2.33.1 path. Both coexist:
- `mlx_mfa.sparse_attention.sparse_attention_forward()` (new in v2.34.0):
  NAX-native block-skip, M5+ head_dim ∈ {64,128} bool block_mask.
- `mlx_mfa.flash_attention_sparse()` (unchanged): dispatcher routes
  between Sprint B path, v2.33.1 cached SDPA fallback, M1-M4 STEEL V1.

User-facing API surface unchanged. Power users can call
`sparse_attention_forward` directly for explicit dispatch control.

---

## B-D10 — Phase 2 integration via Sprint D-style patchers

If Phase 1.5 SHIP-DEFAULT verdict:
- `mlx_mfa.integrations.flashvsr.patch_flashvsr_lcsa(model)`
- `mlx_mfa.integrations.sparkvsr.patch_sparkvsr_sliding_window(model)` (if applicable)

Mirror `patch_seedvr2_vae` pattern (Sprint D). Use `__class__` swap
(D34 lesson — instance-level `__call__` is no-op for `nn.Module`).

Phase 2 deliverables: patcher + 4 tests per integration + A/B bench.

---

## Sign-off

Phase 1.0 design locked at 2026-05-12. Phase 1.1 begins from
`experiment/lcsa-nax-phase1_1` branched off `feat/lcsa-nax`
(post-design-merge).
