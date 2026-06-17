# Campaign 2026-06 — CLOSE-OUT

**Sprints**: B (codification) → A (cache-key audit) → C (discovery) + repeat
sweep.  **Dates**: 2026-06-11 → 2026-06-12.  **Baseline**: post-2026-05
whole-repo review (`c83a18e`).

## Total correctness fixes (B + A + C)

| Source | Count | Items |
|---|---|---|
| Sprint A cache-key audit | 6 | A-1 cfg_axis_flags uint8 truncation (CRITICAL) · A-1b EXEC_SG ghost-knob removal · A-2/A-3 STEEL-bwd is_equivalent has_block_mask · A-4 TQ is_equivalent tq_wht_enabled (HIGH) · A-5 MLX_MFA_DISPATCH_TABLE runtime contract · A-8 legacy conv bf16 type-pun guard |
| Sprint C Track 0 | 4 doc-level | MFA_TOPK_BISECT ghost corrected · MFA_REQUIRE_MSL4 sentinel corrected · DISABLE_ALIGN stale row removed · 10 undocumented knobs documented |
| Sprint B | 0 code (5 institutional) | Pattern #9 · gate #9 · §AA.7 · #6 note · KD-5 ledger |

## Optimizations promoted (net perf delta from campaign start)

| Path | Delta | Validation |
|---|---|---|
| D=256 causal forward, M5 auto path | **1.38×** (6.46→4.67ms @ N=4096; ~1.40× @ N=8192) | 9-cell grid, diff=0.0 |
| TQ paged decode step | ~2% (1.35→1.32 ms/step) + per-token numpy upload removed (block-table cache) | 300-step bit-identical |
| Paged RoPE decode (M5) | per-step mx.compile churn eliminated | targeted tests green |
| conv3d Python wrapper | padding parse memoized | 30/30 conv tests |
| Headline D≤128 paths | unchanged (0.97-1.01× vs SDPA, diff=0.0) | monotonically non-worse confirmed |

## Track 0 knob ledger

79 LIVE · 1 removed (EXEC_SG, Sprint A) · 1 ghost-corrected
(MFA_TOPK_BISECT) · 2 docs-corrected · 10 documented.  Historical
invalidation: v2.30 EXEC_SG sweeps (note added in-doc); no tuning
default ever consumed those numbers.

## Track 6 outcome

**FULL C++ migration** (not C++-only-with-Python-scoped): all 11 key
structs tie()-migrated; shared `mfa_key_tie.hpp`; CI-static loud-failure
semantics (zero release-build overhead — documented decision); perf
<1% verified; static invariant test green throughout.  Python-side keys
need no migration (tuple keys ARE the tie pattern by language
semantics) — recorded, not escalated.

## Marco-gated items (awaiting sign-off; defaults UNCHANGED)

1. **V6NAX backward D=64 causal auto-promotion** — 2.2-2.6× training
   speedup measured (matrix in sprint-C-report).  RECOMMENDED.
2. **MFA_FORCE_NATIVE_BWD disposition** — keep deprecated; rationale
   now "superseded by V6NAX/SDPA-vjp at every cell", not "broken"
   (KD-5 fixed; correctness verified at the formerly-zeroed cells).
3. **Kernel-sprint candidates** (per §AA.5 workflow): Sage-NAX int8
   (premise: MPP int8 verified in headers; needs MSL4-path microbench,
   kill <1.3×) · Top-K streaming Approach 5 (CONFIRMATION on record,
   ~6h) · conv3d small-K retune (cheap sweep).
4. CacheKey Python-side formalization beyond tuple keys: assessed
   unnecessary; no action proposed.

## Declined-with-evidence ledger (do not re-tread blindly)

V3/V4/V5 promotion (3-4× behind SDPA on M5, fresh numbers — the
re-bench Sprint B requested is DONE) · dense-decode early-exit
(3.4µs/call) · TQ searchsorted (no primitive) · FA-3 warp-spec /
FlashDecoding++ / vAttention / NSA / online-softmax numerics /
3D-Winograd (Track 3 table) · 4 micro/cold Track-1 items · P5 rope-lru
(2026-05, reconfirmed obsolete via the mx.fast.rope paged path).

## Completion criterion

- **Fresh cache-key pass**: the Sprint A audit + Track 6 structural
  enforcement + static invariant test → zero new findings on the final
  sweep.  ✅
- **Fresh discovery pass**: Tracks 0-4 complete; every candidate
  promoted, declined-with-evidence, or Marco-gated; Track 5 re-survey
  surfaced zero new actionable autonomous candidates.  ✅
- Suite: **1366 passed, 0 xfailed, 0 xpassed, 0 flakes × 3 runs.**
  Headline paths diff vs SDPA = 0.0.

**Statement**: to the best of this analysis capability, mlx-mfa is
fully audited and fully optimized *within its autonomous scope*.  What
remains is exclusively Marco-gated: the V6NAX-D64 backward promotion
decision, the MFA_FORCE_NATIVE_BWD disposition, and three
premise-validated kernel-sprint candidates.
