# v2.50 Sprint 4 Phase 4b-complete — 4 K-parallel backward kernels — DEFERRED

**Sprint date**: 2026-05-13
**Status**: **DEFERRED to dedicated future session**
**Phase 4a status**: SHIPPED (V6NAX forward causal)
**Phase 4b status**: SHIPPED partial (dQ kernel causal mask only)

## TL;DR

V6NAX backward causal end-to-end requires 4 additional K-parallel
backward kernels (dKV legacy fused, split dV, split dK, fused dKdV)
to each add their own per-element causal mask block.  Phase 4b's
prediction in the Sprint 4 Prompt 1 STATUS doc ("backward kernels
likely need NO source changes") was empirically FALSIFIED in Sprint 4
Prompt 2.

Per §AA.1 failure-mode handling, halting Phase 4b-complete in this
session + documenting per-kernel design + deferring to a dedicated
future session.  Phase 4a + Phase 4b dQ infrastructure preserved on
master; production eligibility gate retains `not causal` clause for
safety.

## Why Phase 4a + Phase 4b dQ alone is insufficient

When `_v6nax_eligible(causal=True)` lifts to allow V6NAX backward causal,
the V6NAX backward dispatcher (`_v6nax_backward_vjp`) calls 3 kernels:

1. **dQ** (Q-parallel): now has causal mask via Phase 4b partial ✓
2. **split dV** (K-parallel) or **fused dKdV** (K-parallel): NO causal mask
3. **split dK** (K-parallel) or **fused dKdV** (K-parallel): NO causal mask

The K-parallel kernels recompute S = Q @ K^T from scratch (no saved
score tensor) and use the causal-masked lse from forward to compute
P = exp(S - lse).  Without their own causal mask, P[r,c] for c>r is
NOT zero — it's a finite positive number because lse only sums over
c<=r positions, so exp(S[r,c] - lse[r]) for c>r is unbounded.

Empirical falsification (Sprint 4 Prompt 2 dev, B=1 H=4 qL=2048 D=64 f16):
- Without K-parallel kernel masking:
  - dQ max_diff = 2144 (vs SDPA-vjp reference; would be small if Phase 4b dQ alone sufficed)
  - dK max_diff = 163
  - dV max_diff = 136
- These are 6-10 orders of magnitude above acceptable tolerance.

## Phase 4b-complete design sketch

For each of the 4 K-parallel backward kernels, add to the source generator:

### 1. Macro injection

```cpp
ss << "#define V6NAXBWD<X>_CAUSAL " << (isCausal ? 1 : 0) << "\n";
```

Where `<X>` is the kernel-specific prefix: `KV` (legacy fused), `V`
(split dV), `K` (split dK), `FUSED` (new fused dKdV).

### 2. Params struct extension

```cpp
struct V6NAXBwd<X>Params {
  ...
  int qL_rem, kL_rem;
  int qL_off;  // v2.50 Sprint 4 Phase 4b-complete
  ...
};
```

Mirror in host-side struct in `v6_nax_compile.mm`.

### 3. Per-element causal mask block

Inserted in each kernel's Q-loop body, AFTER the existing last-K
length mask and BEFORE `P = exp2(S - lse_log2)`:

```cpp
#if V6NAXBWD<X>_CAUSAL
{
  constexpr auto neg_inf = Limits<float>::finite_min;
  const short2 sc_c = s_q_t::NAXFrag_t::get_coord();
  // K-parallel kernels: K is the parallel dim (tid.x), Q is in loop (qb).
  const int base_row = qb * V6NAXBWD<X>_BQ + params.qL_off;
  // For split kernels with WM>1: K is partitioned across simd groups.
  // base_col must include the simd_group_id offset.
  // For legacy fused (WM=1): no SG partition needed.
  const int base_col = int(tid.x) * V6NAXBWD<X>_BK
                       + simd_group_id * (V6NAXBWD<X>_BK / V6NAXBWD<X>_WM);
  for (iq, ik, ii, jj):
    r = base_row + iq*16 + ii*kFragRowsJump + sc_c.y;
    c = base_col + ik*16 + jj + sc_c.x;
    Stile.frag(iq,ik)[loc] = (r < c) ? neg_inf : Stile.frag(iq,ik)[loc];
}
#endif
```

### 4. Loop-bound optimization (perf, not correctness)

For K-parallel kernels with causal, the Q-loop bound can be
tightened: only Q-tiles where `q_min <= K_block_max` contribute
(Q-tiles entirely above the diagonal have all rows masked).
Computing:
```cpp
const int K_max = (int(tid.x) + 1) * V6NAXBWD<X>_BK
                  + simd_group_id * (V6NAXBWD<X>_BK / V6NAXBWD<X>_WM);
int qb_min = metal::max(0, (K_max - 1 - params.qL_off) / V6NAXBWD<X>_BQ);
```
saves wasted Q-tile iterations.  Skip for the correctness-only first
pass; add in a follow-up perf sprint.

## Per-kernel implementation notes

### Legacy fused dKV (WM=1)
- Simplest: no simd_group partition for K.
- `base_col = int(tid.x) * V6NAXBWDKV_BK`
- Mask block placement: line ~4600 (before `Stile.row_bin_op<ExpSubOp>(lse_log2)`)

### Split dV (WM=4 default)
- K rows of dV_accum partitioned across simd groups.
- `base_col = int(tid.x) * V6NAXBWDV_BK + simd_group_id * (V6NAXBWDV_BK / V6NAXBWDV_WM)`
- Need to verify which simd groups see which K cols by reading the Q-load + K-load offsets carefully.

### Split dK (WM=4 default)
- Same as split dV structurally.
- `base_col = int(tid.x) * V6NAXBWDK_BK + simd_group_id * (V6NAXBWDK_BK / V6NAXBWDK_WM)`

### Fused dKdV (newer, replaces legacy fused dKV when MFA_V6BWD_USE_FUSED=1 NOT set)
- Same K-parallel pattern.
- Need to confirm if it uses simd_group K-partition like split kernels.

## Estimated Phase 4b-complete effort (dedicated session)

| Component | CC time |
|---|---|
| Read 4 K-parallel kernels + confirm simd_group partition for each | 30 min |
| `/metal-kernel-dev` pre-impl: verify mask block doesn't change register budget | 15 min |
| Implement legacy fused dKV causal mask (simplest, WM=1) | 30 min |
| Test legacy fused dKV vs SDPA-vjp causal via `MFA_V6BWD_USE_FUSED=1` | 15 min |
| Implement split dV causal mask + test | 30 min |
| Implement split dK causal mask + test | 30 min |
| Implement fused dKdV causal mask + test | 30 min |
| Lift `_v6nax_eligible` and `_v6nax_backward_carveout` causal gates | 5 min |
| Three-axis validation: full flash_attention(causal=True) vjp vs SDPA-vjp | 30 min |
| `/mlx-debug-forensics` corruption audit | 30 min |
| `/mlx-mfa-bench-methodology` 3-session perf bench | 30 min |
| `/mlx-code-review` + decisions doc + CHANGELOG update | 30 min |
| **Total** | **~5h CC** |

This is the upper end of the audit's L (~3-6h) estimate.

## What's still preserved

- **Phase 4a SHIPPED at Sprint 4 branch** (will merge to master with
  this STATUS doc): V6NAX forward causal kernel + non-causal regression
  verified.
- **Phase 4b dQ SHIPPED**: dQ kernel causal mask block (compiled
  only when `V6NAXBWD_CAUSAL=1`, which fires only when the eligibility
  gate is lifted — currently it isn't).
- **Test count post-merge**: 1140 passing (1129 baseline + 11 Sprint 4).
- **Production safety**: `_v6nax_eligible(causal=True)` returns False;
  `_v6nax_backward_carveout(causal=True)` returns False.  Callers using
  `flash_attention(causal=True)` with `MFA_ENABLE_V6_BACKWARD=1`
  cleanly fall back to SDPA-vjp (bit-identical, verified in test
  `test_sprint4_flash_attention_causal_uses_sdpa_vjp`).

## Recommended next steps for Phase 4b-complete (dedicated future session)

1. Read each of the 4 K-parallel backward kernels carefully + identify
   the simd_group offset pattern for K columns in each.  Particularly
   for WM>1 split kernels, the base_col must include the SG slice.
2. `/metal-kernel-dev` pre-impl for register-budget impact (small —
   the mask block uses existing fragment storage, no new accumulators).
3. Implement the 4 mask blocks one kernel at a time, testing each
   independently before proceeding (use `MFA_V6BWD_USE_FUSED=0/1` +
   targeted `_v6nax_eligible(causal=True)` override for testing).
4. Lift the eligibility gates only after all 4 kernels pass three-axis
   validation.
5. Run full mx.vjp(flash_attention(causal=True)) vs SDPA-vjp causal
   correctness sweep across D ∈ {64, 128} × dtype ∈ {f16, bf16}
   × qL ∈ {1024, 2048, 4096, 8192}.

## Files in this Phase 4b-complete deferral

Only this STATUS doc + sprint4-decisions.md + 11 tests covering Phase
4a + the SDPA-vjp fallback path.  No code changes for Phase 4b-complete
(the 4 K-parallel kernels).  Sprint 4 branch
(`feat/v50-sprint4-v6nax-causal`) ships Phase 4a + Phase 4b dQ
infrastructure ONLY.

## Master state post-Sprint-4 (Phase 4a + dQ infrastructure)

- `master`: ~`<Sprint 4 merge commit>`
- Sprint 4 Phase 4a complete: V6NAX forward causal extension.
- Sprint 4 Phase 4b partial: dQ kernel causal mask (infrastructure).
- Sprint 4 Phase 4b-complete deferred with corrected scope (~5h CC).
- CHANGELOG `[Unreleased — for v2.50]`: 4 entries (Sprint 1, 2, 3, 4).
- Tests: 1140 passing.
- No version bump, no tag, no PyPI publication per internal-mode
  contract.
