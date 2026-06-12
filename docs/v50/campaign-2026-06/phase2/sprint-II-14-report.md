# Sprint II-14 — Buffer-pool residual: root cause + structural class fix

**Date:** 2026-06-12 · **Status:** COMPLETE · **Version:** 2.50.1 (unchanged)

## Verdict

The "buffer-pool stale-value residual" was **not a buffer-pool bug**. It is
**intermittent single-lane cooperative-tensor fragment loss** caused by a
data-dependent branch (`if (!tile_active) continue;`) inside the main loop
of the V34 **sparse** backward kernels, executing around live cooperative
accumulators (dK/dV/dQ `_accum` tiles that persist across all iterations).
Fixed structurally in all 4 sparse generators by replacing the in-loop
branch with a **compacted active-tile list + uniform counted loop** (zero
in-loop branching — the dense twin's execution shape). Post-fix: **0
nondeterministic runs across every ladder rung** (pre-fix: ~2/5 standalone).

## R.1 — Stabilized repro

- Permanent stress-gated tripwire: `tests/test_v50_sprint_5e_ii14_pool_tripwire.py`
  (`MFA_POOL_STRESS=1` + M5 gated). Victim config: fused dKdV sparse,
  B=1 H=4 qL=2048 D=64, **all-true** mask, raw-partials bitwise compare
  sparse×2 / dense×2 / sparse-vs-dense, ×5 amplification.
- Pre-fix standalone fire rate ≈ 2/5 per script run; tripwire 8/8 detection.
- **Suppression inversion** [VERIFIED]: full-suite context *suppresses* the
  bug (~0 in-suite over 47 stressed suite runs); standalone fires reliably.
  The earlier 47-clean-suite-runs evidence was therefore misleading.

## R.2 — Exact mechanism

- Divergent values are **zeros** (accumulator reverts to `.clear()` state
  for single lanes in random (head, simdgroup, K-tile) regions) — never
  −inf/NaN/stale content. Content-independent → **not** pool data reuse.
  The canary correlation in II-8 was coincidental timing.
- Full dense-vs-sparse generator diff isolated the **mask `continue`** as
  the only functional difference. Removing/reshaping the branch is both
  necessary and sufficient: body-wrap variant (branch kept, body braced)
  reduced 2/5 → 1/30; active-list variant (branch eliminated) → 0/60.
- Fires even with all-true masks (branch never *taken*) → the hazard is
  the compiled divergent-control-flow structure around the cooperative
  accumulator, not the skip execution itself. [VERIFIED empirically;
  compiler-level attribution DEDUCED]

## R.3 — Structural class fix (all 4 sparse generators)

`csrc/mfa/v6_nax/NAAttentionKernel.cpp`: thread 0 compacts active tile
indices into a 1024-entry threadgroup `ushort` list (one barrier), then a
uniform counted loop iterates only active tiles:

| Kernel | Loop | List | Notes |
|---|---|---|---|
| fused dKdV sparse | Q-loop | active-qb | root-cause victim; validated first |
| dV sparse | Q-loop | active-qb | partial last block always-active (unchanged semantics) |
| dK sparse | Q-loop | active-qb | same recipe |
| dQ sparse | K-loop | active-kb | K/V rebased per active tile (was incremental advance) |

Host guards (`csrc/mfa_v6_nax_primitive.cpp`, all 4 sparse dispatch sites):
`ceil(qL/BQ) > 1024` (resp. `ceil(kL/BK)` for dQ) throws loudly (Rule 8).
1024 entries cover qL ≤ 32768 at BQ=32 / 65536 at BQ=64 — beyond every
supported envelope.

**Cost**: common path (dense kernels) untouched — zero cost by
construction. Sparse path: one single-thread scan of ≤1024 mask bytes per
TG + one barrier; loop trip count is now exactly the active count.

## R.4 — Validation ladder (all clean)

| Rung | Result |
|---|---|
| Fused victim determinism script | **0/60** nondet (pre-fix 2/5) |
| All-true sparse == dense (bitwise, raw partials) | True |
| Diag-mask: finite, deterministic, ≠ all-true | True |
| Split kernels (dQ/dK/dV × D64/D128 × all-true/tridiag) ×30 | **0/29 nondet each**, finite |
| Tripwire standalone ×30 consecutive | **30/30 clean** |
| Pool-poison canary standalone ×30 consecutive | **30/30 clean** |
| Stressed full suite (MFA_POOL_STRESS=1) ×3 | see close-out |
| Sparse-backward + tripwire test files (29 tests) | pass |
| Perf spot (M5 Max, B=1 H=8 N=4096): tridiag vs all-true | dQ 7.3×/14.6×, fused 8.5×, dK 11.5×, dV 9.7× faster — skip benefit intact |

## Skill invocations (§AA.2)

| Checkpoint | Skill | Outcome |
|---|---|---|
| Kernel-debugging methodology | sentinel-write/localization per `docs/methodology/kernel-debugging.md` | raw-partials bitwise diff localized single-lane loss |
| Pre-merge review | `/mlx-code-review` discipline applied to generator diff | dense-vs-sparse diff isolation |
| Bench methodology | §AA.4 (30-iter medians, warmup) | perf spot table above |

## Ledger disposition

II-8 addendum item 2 (buffer-pool bug MANDATORY-clean): **CLOSED** — root
cause found, class fixed structurally, canary at zero. The pool-poison
canary remains in-tree as a permanent stress gate alongside the new
tripwire.
