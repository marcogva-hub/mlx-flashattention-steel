# Audit framing inversions catalogue — v2.50 onwards

**Purpose**: empirically-validated catalogue of cases where the v2.50-NAX-
coverage audit's framing was inverted, partially inverted, or scope-
underestimated by the implementation sprint's investigation.  Each
entry records the audit's prescription, the empirical finding that
contradicted/refined it, and the actual shipped resolution.

This doc operationalises §AA.5 (premise validation discipline) in
`CLAUDE_V6_NAX.md`.  Update on every sprint that surfaces a new
inversion or confirms an audit prediction.

## Pattern types

| Type | Definition | Resolution |
|---|---|---|
| **FULL_INVERSION** | Audit prescribed kernel build; Apple primitive recovers ≥80% of the regression via dispatch fix | Ship dispatch fix; cancel kernel sprint |
| **PARTIAL_INVERSION** | Apple primitive recovers some (1.2×-3×) of the regression; kernel work needed for remainder | Ship dispatch fix as Phase X.a; defer/scope kernel as Phase X.b |
| **SCOPE_CORRECTION** | Audit's effort estimate was 1.5-2× off (under or over); audit's prescription is correct but bigger/smaller | Implement what's possible in budget; STATUS doc for deferred portion |
| **CONFIRMATION** | Audit's prescription empirically correct at estimated effort | Proceed with kernel implementation |

## Catalogue

### v2.50 Sprint 1 — Sparse density threshold — FULL_INVERSION

**Audit prescription** (from `02-consolidated-bench-results.md` G3):
> S effort (~30min) — bool-mask substitution + float-bias cache.

**Empirical finding** (Sprint 1, 2026-05-13):
- Bool-mask substitution **FALSIFIED** on MLX 0.31: bool 1.085× SLOWER
  than float bias in current MLX (the audit's reference doc
  `sparse-fallback-audit.md` was from a 2025 MLX version with
  different bool-mask perf characteristics).
- Float-bias cache (Layer 2) already shipped v2.33.1.
- Real bottleneck: `lcsa_nax.sparse_attention_dispatch::DEFAULT_DENSITY_THRESHOLD
  = 0.02` was calibrated for V1 STEEL on M1/M3; the audit shape
  (density 0.023) was being routed to SDPA+bias instead of NAX direct.

**Resolution**: One-line change `0.02 → 1.01`.  LCSA NAX wins at all
densities on M5+ (verified empirically across 0.016 → 1.0 sweep).

**Shipped at master `be30352`** (Sprint 1 merged).

**Speedup**: ~6× at audit shape (2.97ms → 0.38ms).

**Effort**: ~1h CC (mostly the empirical sweep + framing investigation).

### v2.50 Sprint 2 — Fused-RoPE NAX kernel — FULL_INVERSION

**Audit prescription** (G7):
> S/M effort (~1-2h) — build fused-RoPE NAX kernel.  "Host-side RoPE
> preprocessing overhead — needs fused RoPE NAX kernel".

**Empirical finding** (Sprint 2, 2026-05-13):
- The slow path on M5+ is the **STEEL `_mfa_rope_forward` fused-rope
  kernel** (pre-NAX design, uses simdgroup_matrix + Python rope buffer
  marshaling).
- `mx.fast.rope` (Apple native rope Metal kernel) + `flash_attention`
  (Apple SDPA NAX) composes to a 4× faster path than STEEL fused-rope.

**Component decomposition** (qL=4096 D=128 fp16):
| Path | Latency |
|---|---|
| STEEL fused-rope (current) | 8.09 ms |
| `mx.fast.rope + flash_attention` | 1.99 ms |
| Baseline `flash_attention` (no rope) | ~3.1 ms |

**Resolution**: ~40 LOC dispatch swap in
`flash_attention_rope_unified` standalone path.  M5+ NAX path = `mx.fast.rope
+ flash_attention`; M1-M4 / partial-rope / fp32 paths preserved.

**Shipped at master `4601505`** (Sprint 2 merged).

**Speedup**: ~4× at audit shape (8.09ms → 1.99ms, -75% wall time).

**Effort**: ~1.5h CC.

### v2.50 Sprint 3 — Native top-K Metal kernel — PARTIAL_INVERSION

**Audit prescription** (G5):
> L effort (~3-6h) — new top-K-fused Metal kernel, Primitive + binding,
> three-axis test scaffold, routing.

**Empirical finding** (Sprint 3, 2026-05-13):
- Apple primitives (`mx.topk`, `mx.partition`, `mx.argpartition`, `mx.fast.sdpa`)
  enable a dispatch-only fix recovering **1.25× speedup** (55.6 → 44.4 ms,
  -20% wall time) via mask-then-flash.
- Component decomposition: `mx.sort` over [B,H,N,S] = 1GB tensor is
  ~33ms; `weights@v` final matmul is ~1.6ms.  `mx.partition`/`mx.topk`
  have the SAME cost as `mx.sort` in MLX 0.31 → dispatch fix saves
  the matmul (~11ms) but not the threshold-finding (~33ms).
- Audit's L estimate **CONFIRMED** for the remaining 14× gap vs dense
  SDPA — no primitive composition reaches it.

**Resolution**: Phase 3a SHIPPED (~50 LOC dispatch fix, 1.25× speedup).
Phase 3b (native streaming top-K kernel) DEFERRED with design sketch +
~6h CC estimate.

**Shipped at master `408e1b3`** (Sprint 3 Phase 3a merged).

**Effort**: ~2h CC (premise check + dispatch fix + tests + docs).

### v2.50 Sprint 4 — V34 causal extension — SCOPE_CORRECTION + Phase 4b prediction FALSIFIED

**Audit prescription** (Sprint 4 mandate):
> Sprint 4: V34 backward causal NAX — M effort (~1-2h CC).  Extend
> backward source generators to support causal masking.

**Prompt 1 Sprint 4 scope-discovery finding** (`sprint4-status.md`, 2026-05-13):
- V34 forward gates on `!isCausal` (`NAAttentionKernel.cpp:171`) → V34
  forward causal extension is an unaccounted prerequisite (Phase 4a).
- Halted with corrected scope L (~5h CC) and STATUS doc.

**Prompt 2 Sprint 4 empirical finding** (`sprint4-decisions.md`, 2026-05-14):
- Phase 4a (V34 forward causal) implemented at ~2h CC ✓.
- Phase 4b prediction in Prompt 1 STATUS doc — "backward kernels
  likely need NO source changes because the FA-2 backward pattern
  handles causal via lse-encoded masking automatically" — **FALSIFIED**.
- Direct test: V34 backward dQ via `mx.vjp(flash_attention(causal=True))`
  with V34 forward emitting causal-masked lse → dQ max_diff = 2144
  vs SDPA-vjp reference (6+ orders of magnitude above tolerance).
- Root cause: V34 backward recomputes S = Q@K^T from scratch; causal-
  masked lse alone doesn't zero P[r,c] for c>r because lse only sums
  c<=r positions.
- Resolution: dQ kernel needs its own causal mask (Phase 4b partial
  shipped ~50 LOC); 4 K-parallel backward kernels (dKV, split dV,
  split dK, fused dKdV) each need their own causal mask block
  (Phase 4b-complete, deferred ~3h CC).

**Shipped at master `<Sprint 4 merge>`** (Sprint 4 Phase 4a + dQ
infrastructure).

**Effort**: ~3h CC for what shipped; remaining ~3-5h for Phase 4b-complete.

**Aggregate audit estimate vs reality**:
- Audit: M (~1-2h CC)
- Reality: L (~5-6h CC) when accounting for Phase 4a prerequisite +
  Phase 4b-complete K-parallel kernels.
- Underestimate factor: ~2-3×.

### v2.50 Sprint 5 — V34 backward block-sparse — Premise check NOT YET DONE; DEFERRED

**Audit prescription** (Sprint 5 mandate):
> M effort (~2h CC) — extend V34 backward source generators to support
> block-sparse mask (mask buffer + per-block early-exit).

**Sprint 5 status (`sprint5-status.md`, 2026-05-14)**:
- HALTED with dependency on Phase 4b-complete (4 K-parallel kernels
  need updates for BOTH causal AND block-sparse).
- Recommended bundling: Phase 4b-complete + Sprint 5 in one dedicated
  session (~5-6h CC total vs ~7-8h independent due to duplication).
- §AA.5 premise check NOT YET DONE — audit's "training-side sparse-
  backward gap" is asserted but not empirically measured.  Next
  session should bench current `mx.vjp(flash_attention_sparse(...))`
  before committing to L kernel work.

## Recurring patterns observed

1. **Apple primitive coverage is broader than audit framing assumes.**
   Sprints 1+2 both found that existing primitives (`mx.fast.rope`,
   `lcsa_nax.sparse_attention_dispatch`) deliver the win.  Always
   check primitives before kernel work.

2. **MLX version drift falsifies historical reference docs.**
   Sprint 1's bool-mask substitution FALSIFIED on MLX 0.31 even though
   it was correct on a 2025 MLX version.  Reference docs older than
   ~6 months should be empirically re-verified.

3. **Audit's effort estimates lack implementation-time investigation.**
   Sprint 4 was estimated M (~2h) but actually requires L (~5-6h)
   because the audit didn't notice V34 forward causal as a prerequisite.
   Sprint 3's L estimate for native kernel was CONFIRMED but the
   audit missed that a dispatch fix would deliver 1.25× independently.

4. **Component decomposition reveals the real bottleneck.**
   Sprint 3's audit framing implied "build a top-K kernel"; component
   bench showed the bottleneck is the materialized score tensor + sort,
   not the algorithmic top-K operation.

5. **Backward kernel dependencies cascade.**
   Sprint 4 surfaced that 4 K-parallel backward kernels need causal
   mask blocks; Sprint 5 (block-sparse) extends the SAME 4 kernels.
   Bundle these in one session to avoid infrastructure duplication.

6. **Incomplete-fix dispatch-chain pattern (Pattern #5 — v2.50 Prompt 4).**

   When a kernel takes inputs from N upstream sites in a dispatch
   chain (forward → backward → routing → fallback), a "fix" that
   addresses M < N of those sites silently leaves the kernel consuming
   incompatible inputs from the unfixed remainder.  Each site reads
   correct in isolation; the residual only manifests as numerical
   drift in the final output, with no localised stack-trace pointing
   at the culprit.

   **Empirical case** (v2.50 Prompt 4 Section B — dV residual):
   V34 backward dV kernel consumes `lse` produced by the forward.  Two
   eligibility gates routed forward to V34 (natural-log lse), but a
   THIRD gate in `MFAV6Forward::eval_gpu()` routed *causal* forward
   to STEEL legacy (log2-domain lse).  The dV kernel decoded
   `exp(score - lse)` correctly for non-causal (gates 1+2 fixed) but
   produced ~0.4 dV residual for causal (gate 3 still routed to STEEL,
   yielding log2 lse interpreted as natural log).

   **Detection technique**: sentinel writes (see
   `docs/methodology/kernel-debugging.md`).  Inject a uniquely-valued
   constant into the dispatch-active code path; observe via
   `mx.eval`-then-print whether the sentinel reaches the output.
   Absence of the sentinel proves a different code path is active.
   This is faster than gradient bisection and more precise than
   "which kernel was called" debugger inspection.

   **Multi-gate audit requirement**: see `CLAUDE_V6_NAX.md` §AA.5.x
   amendment.  Before declaring any kernel-input compatibility fix
   complete, enumerate ALL dispatch sites that produce that input and
   verify each one was patched to the new convention.  Single-site
   fixes for multi-site inputs are insufficient.

   **Cross-references**:
   - `docs/v6-nax/v50-prompt4-sectionb-dv-residual-RESOLVED.md`
     (full investigation log)
   - `docs/methodology/kernel-debugging.md` §2 (sentinel writes)
   - `CLAUDE_V6_NAX.md` §AA.5.x (multi-gate audit amendment)

7. **Pattern #6 — Apple primitive M5+ optimization level falsifies custom-kernel speedup projections (v2.50 Prompt 5d).**

   When a sprint projects "X× speedup via custom NAX kernel" based on
   theoretical pattern reasoning (e.g., sparse-skip should win at low
   density), and Apple SDPA NAX is in the comparison path on M5+
   hardware, the projection must be **empirically validated** before
   committing to the custom kernel implementation.  M5+ Apple SDPA NAX
   is sufficiently optimized that custom V34-style NAX kernels —
   even with algorithmically-superior optimizations like sparse-skip
   or top-K filtering — cannot outpace it at audit-relevant shapes.

   **Empirical case** (v2.50 Prompt 5d Section A v3):
   Sprint 5 native sparse backward was projected to deliver 10×
   speedup at density 0.1 (FlashVSR-typical) via 4 native V34 NAX
   backward kernels.  Implementation completed (3 new kernels + dV
   PoC, all math-correct).  Bench at VSR shape (B=1 H=12 qL=4096
   D=128 fp16) shows:

   | Density | SDPA-vjp | V34 hybrid | V34 full native |
   |---|---|---|---|
   | 0.1 | 17.41 ms | 34.84 ms | 22.58 ms (0.77× SDPA) |
   | 1.0 | 16.93 ms | 175.09 ms | 181.07 ms (0.09× SDPA) |

   V34 native is 0.09×–0.77× SDPA-vjp dense at all tested densities.
   The projected 10× speedup does not materialize because the
   projection assumed V34 dense kernels were at parity with SDPA-vjp
   on M5+ (Sprint B v2.40.0-internal validated parity-or-slight-
   regression for D=128 dense; sparse extension inherits that
   overhead).

   **Inversion verdict**: Sprint 5 sparse projection FALSIFIED at VSR
   audit shape.  Production routing reverts to Prompt 5c hybrid (NAX
   sparse forward preserves Sprint 1 forward win + SDPA-vjp backward
   leverages Apple SDPA NAX optimization).

   **Sister pattern to Pattern #2** (Sprint 2 `mx.fast.rope` discovery):
   Apple primitive coverage was broader than audit framing assumed,
   eliminating the perceived gap before kernel work.  Pattern #6 is
   the inverse: custom kernel was implemented BUT empirical bench
   confirms Apple primitive is still optimal.

   **Rule for future sprints**: empirical bench is MANDATORY before
   extending custom kernel coverage when Apple SDPA NAX is in the
   comparison path on M5+.  Use the `/mlx-mfa-bench-methodology` skill
   3-session protocol at the audit-target shape BEFORE committing to
   implementation.  This extends `/mlx-mfa-apple-primitives-coverage`
   §AA.5 premise-check protocol — coverage check alone is insufficient
   on M5+ where Apple primitives have been heavily optimized.

   **Cross-references**:
   - `docs/v50/section-a-v3-empirical-verification.md` (full bench data)
   - `docs/v50/section-b-v3-approach-5-empirical-skip-decision.md`
     (Approach 5 deferred per Scenario 3 inference)
   - `docs/v50/sprint-5d-section-a-status.md` (Section A native code)

8. **Misleading xfail rationales conceal real bugs (Section B).**

   Three of six xfail decorations investigated in v2.50 Prompt 5a
   Section B used high-level conceptual rationales ("accuracy",
   "API compatibility") when the actual root cause was either:
   (a) overly tight tolerance below the FP16 ULP floor, or (b) a
   `RuntimeError` in a code path the test inadvertently exercised
   (e.g., NAX small-mask buffer rejection).  Future contributors
   investigating xfails were forced to re-discover the empirical
   failure mode each time.

   **Discipline**: `pytest.mark.xfail(reason=...)` must include the
   actual observed failure mode (e.g., `max_diff = 0.30 vs atol 5e-2`
   or `raises RuntimeError: mask < 4096 bytes`), not just a category.
   See `docs/v50/sprint-prompt5a-sectionB-xfails-status.md` Pattern
   observations section.

## Pattern #8 — Silent hook fallback masking unused optimization path (v2.50.1 Prompt 5g finding)

**Symptom**: production code ships with "optimization path enabled"
(hook installed, NAX kernel compiled, dispatch policy set) but the
optimization NEVER executes in production.  Users see correct outputs
via baseline fallback so the bug is invisible from output inspection
alone.  Performance matches baseline expectations within noise.

**Mechanism**: auto-hooks patching MLX primitives globally can fail
when input contracts mismatch (dtype, shape, kwargs, etc.).  In
mlx-mfa's case, the failure raised `RuntimeError` at graph-evaluation
time, which user pipelines silently absorbed via downstream `try/except`
wrappers — masquerading as "code works" while the optimization was
never engaged.  The two failure modes produce identical user-visible
behavior:

1. Hook code path: NAX kernel dispatched
2. Actual execution path: MLX baseline (user's try/except caught the
   hook's exception)
3. User observation: correct output, baseline performance
4. Detection: only possible via instrumentation (hook telemetry from
   Phase C) OR via post-hoc bench comparison if baseline expectations
   are documented

**Concrete case observed**: `_auto_hooks.py::conv3d_nax_forward`
(introduced v2.36.0, root cause fixed v2.50.1 Prompt 5g Phase A).
The hook enforced `weight.dtype in {fp16, bf16}` but missed the more
fundamental requirement that the **C++ NAX kernel** required
`x.dtype == w.dtype`.  MLX baseline `mx.conv_general` accepts mismatched
dtypes via automatic promotion.  VSR VAE encoders pass fp32 input +
fp16 weights — this raised `RuntimeError: conv_nax: x.dtype != w.dtype`
on every call -> silent absorption by user pipelines -> M5 Neural Engine
fixed-function Conv3D acceleration **NEVER executed in any production
inference run between v2.36.0 and v2.50.0**.

While validating the fix, a **second independent bug** surfaced: the
bf16 NAX kernel path triggers a Metal shader compilation failure in
MLX upstream `utils.h:502` (im2col helper `half` vs `bfloat16_t` type
mismatch).  This had also been broken since v2.36.0 — zero user reports
because no production workload exercised the bf16 path (KD-7 tracks
this).

**Detection requires**:

1. **Hooks log/warn on fallback path engagement** (Pattern #8 telemetry
   introduced Prompt 5g Phase C — `mlx_mfa.get_hook_stats()`).
2. **Integration tests verify hook actually executes** for representative
   input patterns (Phase D smoke tests across user models).
3. **Audit scope includes stable-but-unverified production-critical
   code**, not just modifications since last release (Prompt 5e audit
   scope limitation that contributed to this bug surviving — see SS-AA
   amendment).
4. **Cross-version bench comparison**: if v(N+1) "added optimization"
   shows no perf delta vs v(N), investigate whether the optimization
   actually engages — the v2.36.0 release notes mentioned NAX Conv3D
   speedups that were never user-visible because the path was unreachable.

**Prevention** (institutionalized in Prompt 5g):

- `/mlx-code-review` skill amended: flag hook patches with input
  contract stricter than the patched primitive's baseline contract.
- `/mlx-mfa-perf-audit` skill amended: require "hook actually executes"
  verification via telemetry stats for any perf claim related to
  hooked primitives.
- `/mlx-mfa-release-audit` skill amended: 8th check added — auto-hooks
  compatibility contract preserved vs MLX baseline.
- `CLAUDE_V6_NAX.md` SS-AA scope discipline amended: audit scope must
  include stable-but-unverified production-critical code.

**Sister patterns**:

- Pattern #2 (`mx.fast.rope` discovery) — Apple primitive beats custom
  kernel; mlx-mfa wasn't using the Apple primitive even though it was
  available.
- Pattern #5 (incomplete-fix dispatch-chain) — dispatch policy
  decisions that propagate through multiple code paths can silently
  break when any link in the chain has a stale assumption.
- Pattern #6 (Apple SDPA NAX optimization level) — empirical bench
  falsifies projection; the "expected" speedup never materializes.
- **Pattern #8 distinguishes** by: optimization path EXISTS but never
  EXECUTES.  Unlike Pattern #6 ("wrong projection"), Pattern #8 is
  "silent non-engagement" — the code is correct but unreachable.

**Generalizable lesson**: when patching global primitives via hooks,
the **compatibility contract with the patched primitive is part of
the public API**.  Tightening any input check creates a silent break
that masquerades as performance regression at best, or correctness
bug at worst.  Audit scope for hooks must include the patched
primitive's documented contract behavior, not just the eligibility
heuristic the hook author had in mind when writing it.

**Cross-references**:
- `docs/v50/known-debt-v2.50.md` KD-6 (resolved) + KD-7 (open bf16 path)
- `docs/v50/prompt-5g-section-b-hooks-inventory.md` (full audit verdict)
- `docs/v50/prompt-5g-section-d-smoke-test-findings.md` (portfolio engagement validation)
- `docs/HOOK_TELEMETRY.md` (detection infrastructure)
- `mlx_mfa/_auto_hooks.py::_patched_conv_general` (fix implementation)

## Doc maintenance

- Add a new entry to the Catalogue on every sprint that surfaces a
  framing inversion or confirms a prediction.
- Update the "Recurring patterns" section quarterly or when a new
  meta-pattern emerges.
- Cross-reference each entry from the sprint's decisions/status doc.
