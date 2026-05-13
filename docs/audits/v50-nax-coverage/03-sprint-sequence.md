# v2.50 Sprint Sequence Recommendation

**Source**: `docs/HARDWARE_SUPPORT.md` NAX-opportunities matrix
**Audit date**: 2026-05-13
**Constraint**: master accumulates internally; no PyPI release between
v2.39.1 and v2.50 final bundle

## Tier 1 — Must-have for v2.50 "production complete"

Three sprints, ~8 hours CC total estimated.  All ship as accumulating
internal sprints (no version bump, no tag, no PyPI between any of them).

### Sprint 1: `flash_attention_sparse` density threshold + bool-mask cache

**Effort**: S (~1h CC realistic)
**Files**: `mlx_mfa/attention.py` (single function `_sparse_fallback_sdpa_perhead` + LCSA dispatcher routing)
**Mandate**:
1. Add density-threshold check to `lcsa_nax.sparse_attention_dispatch` —
   route to dense SDPA when active-block ratio < empirical threshold
   (TBD via bench, expect ~30-40% as the inflexion).
2. Apply `docs/sparse-fallback-audit.md` Layer 1 (bool-mask substitution,
   ~30 LOC, no float-bias conversion) — saves ~1.3ms per call.
3. Apply Layer 2 (LRU mask expansion cache keyed by `id(block_mask) +
   shape + dtype`, bounded to 8 entries) — saves ~2ms on cache hit.
**§AA gates**: `/mlx-mfa-bench-methodology` for threshold calibration;
`/mlx-mfa-perf-audit` for the "X% improvement at sparse density Y" claim
if shipped in CHANGELOG.
**Expected outcome**: closes the 1.20-1.26× gap to ~1.0× or better on
both symmetric-low-density and asymmetric paths.

### Sprint 2: Fused RoPE NAX in `flash_attention_rope_unified`

**Effort**: S/M (~2h CC realistic)
**Files**: `csrc/mfa/v6_nax/NAAttentionKernel.cpp` (V34 forward source generator
already has rope_q_base + rope_cos_stride params per ~line 2741), routing
in `mlx_mfa/attention.py::flash_attention_rope_unified`.
**Mandate**:
1. Wire `flash_attention_rope_unified` to invoke V34 forward NAX kernel
   (which already supports rope params) instead of host-side rope +
   separate flash_attention call.
2. Verify rope_q_base + rope_cos_stride params are exercised correctly
   (V34 forward currently emits rope code only when explicitly invoked
   from `_v34_backward_vjp` for forward-fusion).
3. Eligibility: D=64, D=128, fp16/bf16, non-causal (rope is typically
   not used with causal LLM training — verify use cases).
**§AA gates**: `/metal-kernel-dev` pre-flight for the new dispatch path;
`/mlx-debug-forensics` for bit-identity vs current rope+attend output;
`/mlx-mfa-bench-methodology` 3-session for the perf delta claim.
**Expected outcome**: closes the 1.54× rope overhead → ~1.05-1.10× SDPA
(rope is structural extra arithmetic; can't be eliminated entirely but
the host-call elimination is the major win).

### Sprint 3: Top-K native Metal kernel

**Effort**: L (~5h CC realistic — new kernel)
**Files**:
- New `csrc/mfa/v6_nax/NATopkKernel.cpp` (or inline into NAAttentionKernel.cpp)
- New `MFAV34TopkForward` Primitive in `csrc/mfa_v6_nax_primitive.cpp`
- New `v6_nax_topk_forward_raw` binding in `csrc/bindings.cpp`
- Routing in `mlx_mfa/attention.py::flash_attention_topk`
- Tests in `tests/test_flash_attention_topk.py` (new file)
**Mandate**:
1. Design kernel: per-query Top-K block selection via per-block max-score,
   then dense attention over selected blocks (block-sparse with dynamic
   selection mask).
2. M5+ NAX dispatch via V34 forward template; D=64/D=128 eligibility.
3. `/metal-kernel-dev` pre-flight on register budget (TopK selection has
   transient state per query — careful with persistent registers).
4. Three-axis validation: bit-identical to Python reference at high K
   ratios; perf wins at low K ratios (sparsity benefit).
**§AA gates**: `/metal-kernel-dev` pre-impl + post-register-budget audit;
`/mlx-debug-forensics` corruption audit; `/mlx-mfa-bench-methodology`
3-session perf characterization; `/mlx-mfa-perf-audit` per perf claim.
**Expected outcome**: closes the 17× regression at typical K ratios
(K/qL ~0.02-0.10); function becomes usable at production scales.

## Tier 2 — Training-side high value

### Sprint 4: V34 backward causal NAX

**Effort**: M (~2h CC realistic)
**Files**: extend `createV34BackwardQuerySource()` + `createV34BackwardDKSource()`
+ `createV34BackwardFusedDKDVSource()` in NAAttentionKernel.cpp to support
causal masking; update `_v34_backward_carveout` to include causal.
**Mandate**: causal V34 backward for mlx-lm training workloads (currently
the carve-out is non-causal only; causal LLM training falls back to
SDPA-vjp which is ~1.9× slower per v2.38.1 baseline at D=64 qL=4096).
**Expected outcome**: extend v2.38.1 / v2.39.1 perf claim to causal —
1.91× / 1.95× / 1.80× speedups at D=64 qL=4096/8192/16384 should hold
for causal too (causal kernel work is ~5-10% more than non-causal).

### Sprint 5: V34 backward block-sparse NAX

**Effort**: M (~2h CC realistic)
**Files**: extend V34 backward source generators to support block-sparse
mask (mask buffer + per-block early-exit).
**Mandate**: VSR training (FlashVSR/STCDiT) typically uses block-sparse
attention for memory savings; backward currently falls back to SDPA-vjp.
**Expected outcome**: closes the training-side sparse-backward gap.

## Tier 3 — Deferred post-v2.50

The following NAX-opportunities are deferred per audit's
"breadth-not-depth" mandate + effort/gain trade-off:

- **Paged-NAX variants** (B.15, B.16, B.17, B.22): XL effort each (~6-12h),
  marginal gain over STEEL paged (5-15%), and structurally dependent on
  Apple adding paged-NAX upstream.  STEEL paged is "good enough" for v2.50.
- **Sage fused-quantize NAX** (B.4): L effort, narrow workload value (long-
  context int8 KV training), prequantized variants likely already (A).
- **D ∉ {64, 128} backward** (memory roadmap): no specific user demand
  surfaced; defer until empirical workload pressure.

## Total v2.50 effort

| Tier | Sprints | Effort estimate | Calendar (1 focused session per sprint) |
|---|---|---|---|
| Tier 1 | 3 | ~8h CC | 3 sessions |
| Tier 2 | 2 | ~4h CC | 2 sessions |
| Tier 3 | deferred | — | — |
| **Total Tier 1+2** | **5 sprints** | **~12h CC** | **~5 focused sessions** |

Sequencing: Tier 1 strictly before Tier 2 (Sprint 1-3 fix forward-path
breadth; Sprint 4-5 are backward-side optimization on top of the working
forward path).  Within tiers, sprints are independent and can be reordered
per Marco's priority.

## v2.50 release strategy

Once all Tier 1+2 sprints have landed in master (accumulating under
`[Unreleased — for v2.50]` CHANGELOG section):

1. Multi-SoT version bump 2.39.1 → 2.50.0 (pyproject.toml + __init__.py +
   README.md banner)
2. Rename `[Unreleased — for v2.50]` to `[2.50.0] — <date> — Production complete`
3. `/mlx-mfa-release-audit` canonical pre-tag gate (skip-bumped version
   check now applies — must pass before tag)
4. `/mlx-mfa-perf-audit` for each perf claim in the consolidated
   CHANGELOG (Sprint 1-5 deltas)
5. `/mlx-code-review` pre-merge audit
6. Tag v2.50.0, build wheel + sdist, PyPI upload, GitHub release
7. Update PERF_CLAIMS.md registry with v2.50 entries
8. Update SPRINT_HISTORY.md with sprint A→B→C + v2.50 series consolidation
