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

   **2026-05 external application**: the whole-repo review correctly
   applied this pattern by DECLINING V3/V4/V5 STEEL-variant perf
   promotion despite all 22 accuracy variants passing on M5 Max —
   recognizing that M1-era perf verdicts (V3 0.77–0.88× V2, V5
   0.60–0.90× V2) require a dedicated M5 bench campaign per §AA.4
   before any dispatch change.  The accuracy xfail markers were
   removed (they now guard regressions as real passes); perf promotion
   was correctly NOT attempted on stale evidence.  Re-bench of
   V3/V4/V5 on M5 is a tracked campaign candidate (Sprint C scope).

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

## Pattern #9 — Generator/dispatch hardcoded-constant mismatch (silent partial-write) (2026-05 whole-repo review, KD-5 root cause)

**Symptom**: output correct on a PREFIX of the K/N dimension, zeroed
(or stale) beyond a threshold equal to `NK · BK_source`, where
`BK_source` is the value hardcoded in the MSL generator preamble —
NOT the `cfg.BK` used to compute the threadgroup launch count.  In
the KD-5 instance: dK/dV zeroed for rows ≥ 1024 at N=2048, D=128 on
M3+ (cfg.BK=32), correct on M1 (cfg.BK=16).

**Mechanism**: a dimension constant (BK/BN/BD/tile size) is computed
one way on the dispatch side (`eval_gpu`, from `select_steel_block_config`)
and hardcoded another way in the source generator (C++/MSL preamble
override such as `const int BK = (BD <= 64) ? cfg.BK : 16`).  When the
two diverge for some (D, dtype, causal, hardware-gen) cell, the launch
grid and the kernel's internal striding disagree:

1. Dispatch launches `NK = ceil(S / BK_dispatch)` threadgroups.
2. The compiled kernel processes `BK_source` rows per threadgroup at
   `BK_source`-row strides.
3. Rows beyond `NK · BK_source` are never written — silent partial
   write that LOOKS like a fundamental kernel limitation.

**Concrete case observed**: `MFASteelBwdDKV::eval_gpu` used `cfg.BK`
(= 32 on M3+ at D=128) for NK / params / cache key, while
`generate_steel_backward_dkv_source` hardcodes `BK = (BD <= 64) ?
cfg.BK : 16`.  Carried for multiple weeks as "deprecated STEEL backward
debt slated for v2.51 removal" (KD-5); fixed by one expression — the
dispatch BK now mirrors the generator override.  Both xfails removed;
all 4 TestNativeBackwardRouting shapes assert against SDPA-VJP and pass.

**Why it hid**: the divergence only manifests on hardware/config cells
where the two values differ.  On the cell where they coincide (M1,
BK=16 both sides) behavior is correct, so the bug masquerades as
hardware-specific and gets filed as deprecated debt.  Compounding
factor: the KD entry recorded the SYMPTOM ("STEEL backward zeroes
blocks at D=128 N≥2048") without the MECHANISM, which let it sit as
accepted debt — see the KD-ledger lesson in `known-debt-v2.50.md` KD-5.

**Prevention rule**: for every kernel with dimension constants, the
dispatch-side and source-side values MUST be asserted equal per
(D, dtype, causal) cell of the dispatch table.  Enforced by
`/mlx-mfa-release-audit` gate #9 (source/dispatch block-dimension
consistency) and §AA.7 in `CLAUDE_V6_NAX.md`.

**Generalization**: any constant that appears in BOTH the launch-grid
computation AND the kernel source is a mismatch candidate.  Enumerate
them per kernel: BK, BN, BD, tile rows/cols, SIMD group counts (WM/WN),
threadgroup memory sizes.  Generator-side conditional overrides
(`(cond) ? cfg.X : <literal>`) are the highest-risk construct — grep
for reassignments of cfg-derived names inside generators.

**Lesson linkage**:
- Reinforces **Pattern #1** (Apple-primitive/audit framing): never
  accept "fundamentally broken" as a permanent verdict without
  re-deriving the mechanism.
- Sister to **Pattern #5** (incomplete-fix dispatch-chain): both are
  multi-site consistency failures where one link holds a stale value.
- Sister to **Pattern #7** (misleading xfail rationales): the xfail
  text encoded the wrong theory ("16×BQ tile-loop termination bug"),
  steering later readers away from the true mechanism.

**Cross-references**:
- `csrc/mfa_attention.cpp` MFASteelBwdDKV dispatch (fix site, comment block)
- `csrc/mfa_steel_bwd.cpp` generate_steel_backward_dkv_source (override site)
- `docs/v50/known-debt-v2.50.md` KD-5 (resolution entry)
- `docs/v50/repo-review-2026-05-report.md` (discovery context)
- `~/.claude/skills/jit-kernel-cache-audit` Audit 1 (generic procedure)

## Doc maintenance

- Add a new entry to the Catalogue on every sprint that surfaces a
  framing inversion or confirms a prediction.
- Update the "Recurring patterns" section quarterly or when a new
  meta-pattern emerges.
- Cross-reference each entry from the sprint's decisions/status doc.

---

## Pattern #9 — three exhibits (as of the III-4 audit)

The generator/dispatch dimension-mismatch class now has THREE recorded
exhibits, all enforced by `/mlx-mfa-release-audit` gate #9:
1. **KD-5** (2026-05) — `MFASteelBwdDKV` dispatched `cfg.BK`=32 while the
   generator hardcoded BK=16 for D>64 (above).
2. **v2.39.1 fused-backward BK=16** — the fused dKdV default `BK=16`
   (TK=1) vs the paired-MMA `ik += 2` requirement (TK even).
3. **MFA_V6_V34_BK forward override** (II-8) — an unguarded env-var
   forward override of BK, found during the Phase II-8 sweep.
Gate #9 is now a programmatic test (`tests/test_phase2_ii8_gate9_parity.py`)
asserting every paired-MMA emission site's BK is `% 32`-guarded.

## III-4 institutional lessons (Phase III-4 audit, 2026-06)

The 9-pass repeat-until-clean whole-repo audit added three meta-lessons,
each from a real bug it surfaced:

8. **MLX `grid` is total THREADS, not threadgroups.** A kernel indexed
   by `threadgroup_position_in_grid.X` and reducing cooperatively over
   its threads needs `grid.X = n_items_X × threadgroup.X`. The top-K
   bisection kernel (a promoted AUTO-default) used `grid.x = N`, launching
   only `N/256` threadgroups → it wrote thresholds for the first ~8 query
   rows per head and the rest read **stale Metal-pool memory**. A CRITICAL
   that passed tests because recycled buffers usually held benign zeros.
   *Rule*: for every `mx.fast.metal_kernel`, assert the grid matches the
   indexing mode (threadgroup-indexed → `n × tg_size`; thread-indexed →
   `n` rounded up with an overshoot guard) AND that 100% of the output is
   written (an undercount + a partially-written output = stale reads).

9. **`isinstance(child, dict)` before `nn.Module` is a silent-no-op trap.**
   `nn.Module` IS a `dict` subclass (`issubclass(nn.Linear, dict)` is
   True). A tree walk that branches on `dict` first treats a direct
   `nn.Linear` attribute (the most common model structure) as a container,
   iterates its weight/bias arrays, and replaces nothing — while reporting
   success. `quantize_model` was a silent no-op on direct-attribute models
   for this reason. *Rule*: check `nn.Module` before `dict`/`list`.

10. **Single-shape-class test suites hide entire bug families.** Every
    III-4 bug that prior suites missed shared a "the only test used one
    shape class" root: power-of-2 spatial grids (axial-temporal mask),
    `nn.Sequential` (svdquant), 0.1-scale fixtures (II-6 paired-MMA),
    same-dtype inputs (dtype-reinterpret), never-gradded feature combos
    (`return_lse` backward). The audit's value came from deliberately
    breaking each assumption — non-pow2 grids, direct attributes, unit +
    adversarial scale, mixed dtypes, gradients through every feature.
    *Rule*: a regression test must vary the dimension the bug class lives
    in, not just exercise the happy-path shape.

## III-6 institutional lesson (conv K-tail root-cause fix, 2026-06)

11. **A low-precision kernel is validated against an INDEPENDENT
    higher-precision reference — never against another kernel path.** The
    conv3d small-channel corruption survived all 9 III-4 passes because
    `test_fp16_still_works` compared the Python legacy GEMM against
    `mx.conv_general` — but under installed hooks `mx.conv_general` routed
    to the *same* broken matmul2d kernel. Two instances of one bug compared
    equal → green. The trap generalizes: any test asserting
    `kernel_A(x) ≈ kernel_B(x)` is blind to a bug both share — and "B" is
    deceptively easy to route back into "A" via a hook, a shared helper, or
    a fallback. *Rule*: the reference for a low-precision kernel must be (a)
    an independent implementation (fp32 native / a PyTorch CPU oracle / a
    different vendor primitive such as Apple SDPA for attention) AND (b)
    computed at higher precision (fp32) so its own error floor is far below
    the bug magnitude. When the reference is a hookable op, pin it to the
    UNHOOKED original (`_ORIGINAL_CONV3D`), never the bare patched symbol.
    *Sweep result (III-6)*: the only active instance was
    `test_fp16_still_works` (fixed III-5 → C_in=32 + III-6 root-cause fix);
    all other low-precision kernel tests already validate against an
    independent fp32/SDPA/PyTorch reference. The root cause itself — the
    matmul2d unmasked partial-K-tile — was fixed at the kernel level (K
    zero-padded to a K_TILE multiple); `matmul2d_source` now refuses an
    unaligned K (Rule 8) so a future unpadded caller fails loudly.

## III-7 targeted sweep (hunt the conv3d bug's siblings, 2026-06)

A dedicated three-class sweep (one sub-agent per class) hunted hidden
siblings of the conv3d bug across the whole kernel surface, every suspect
confirmed against an INDEPENDENT fp32 reference:

- **Class A (non-independent reference)** — CLEAN. ~50 numerical-validation
  tests classified; 7 non-independent-reference categories re-probed vs
  fp32, all at the dtype floor. The only active instance was the III-5/6
  conv one. Several passing-by-luck tests (cpp-vs-python-legacy migration,
  topk-vs-fallback, LCSA-v1-vs-v2, context-vs-base) were strengthened to an
  independent fp32 anchor — note the trick: **fp32 `mx.conv_general` is
  inherently independent** because NAX only handles fp16/bf16, so an fp32
  conv can never be hooked to the kernel under test regardless of hook
  state.
- **Class B (unmasked partial-tile read)** — CLEAN. The conv matmul2d
  K-tail was the *only* unmasked partial-tile read; every other tiled
  kernel (V6 NAX, STEEL V1/decode, sage, GNA, paged-varlen ±TQ) masks its
  tail (`load_safe` + `-inf` score mask). The sparse-attention alignment
  gate (`mfa_sparse_attention.cpp:966`) is the model of the *right* pattern
  — it drops remainder branches but enforces `qL/kL % block_tile == 0` with
  a documented, load-bearing Rule-8 raise. No latent unguarded gate found.
- **Class C (single-shape coverage)** — found **2 `quantize_model`
  findings** (same lesson-#9 silent-failure family the III-4 F7-1 fix only
  partially closed): a bare top-level `nn.Linear` was a silent no-op (the
  child-walker never tests the top-level module; can't setattr-replace
  `self`), and a group-misaligned `in_features` raised mid-walk after
  partial mutation. Both fixed (top-level → Rule-8 raise; default predicate
  skips group-misaligned cleanly; a pre-validation pass makes a custom
  predicate's misalignment fail atomically before any mutation). All other
  accepted-but-untested regimes (partial-N attention at fp16/bf16, non-cubic
  GNA, non-aligned paged/TQ seq) probed CLEAN vs fp32 — coverage gaps, now
  locked with parametrized regression tests.

**New mechanism variant** (Class B.2 cousin): a **perf-motivated dispatch
gate can mask a latent correctness bug on a forced expert path**. The
`backend="mfa"` non-causal D∈{64,128} forward diverges from fp32 SDPA
(MAE ~0.12) on M5+/NAX, but `dispatch_policy` routes all non-causal dense
to SDPA *for speed* (documented: "non-causal dense routes remain
conservative SDPA"), so the broken kernel is never auto-selected and no
test forces it — the exact "only reached when [a gate sends real traffic]
elsewhere" structure of the conv3d bug, here masked by a perf gate rather
than a correctness gate. Reachable only via the expert `backend="mfa"`
escape hatch. Flagged to Marco (disposition: Rule-8 raise on the forced
path vs kernel investigation vs leave-documented) — not the default path,
fix has expert-API-contract implications.

## III-8/8c methodological lesson — cooperative-MMA kernels can't be debugged by register dumps

12. **For cooperative-MMA (simdgroup-matrix) Metal kernels, mid-kernel
    register dumps of an MMA tile are UNRELIABLE — diagnose via end-to-end
    output vs an independent fp32 reference + static source differential.**
    Chasing the V2 single-pass non-causal bug (III-8), two register-dump
    methods (`store_contiguous` and faithful element-wise `frag_at`→memory)
    both produced wrong-looking Stile/P values — but a control proved them
    unreliable: **dumping the *causal* Stile (whose end-to-end output is
    correct) also yielded wrong-looking scores**, and the two methods
    disagreed. The cooperative simdgroup-matrix fragment lives in a
    lane-distributed register layout that a naive per-lane write does not
    serialize faithfully (and an early `return` to dump perturbs state).
    Two conclusions were drawn from these dumps and then **retracted**
    ("it's Q@K^T / the scores").
    *Rule*: (a) the trustworthy signal for a cooperative-MMA kernel is the
    **full-pipeline output O vs an independent fp32 reference**, not
    mid-kernel register state; (b) a probe that depends on an unproven
    component is **confounded, not conclusive** — the one-hot-V `O=P` trick
    is reliable only when P@V is already proven correct (it was for causal,
    NOT for the non-causal path under suspicion); (c) when only a few
    source lines differ between a correct and a buggy variant, **static
    causal-vs-non-causal differential reading + git archaeology** localize
    far more reliably than instrumentation. To read an MMA fragment
    mid-kernel when truly necessary, use a layout-correct `simd_shuffle`
    gather into a known lane order, or a minimal standalone reproduction of
    the MMA outside the full kernel — never a naive `frag_at`→memory dump.

## III-8e methodological win — known-answer uniform-P probe (companion to #12)

13. **To debug a cooperative-MMA attention kernel, read the EFFECTIVE
    attention through the correct full pipeline with KNOWN-ANSWER inputs —
    do not serialize the MMA fragment.** Lesson #12 said mid-kernel
    register dumps are unfaithful (re-proven in III-8d via the mandatory
    `col≤row` self-check). The technique that finally cracked the V2
    non-causal mechanism (III-8e, after 4 sprints of failed register reads):
    - **Q=0** ⇒ scores all 0 ⇒ **uniform P** regardless of any Q@K^T bug ⇒
      isolates softmax-sum + P@V from the scores.
    - **V[j,0]=j** (ramp) ⇒ `O[i,0]=Σ_j P[i,j]·j` = mean of the attended key
      set ⇒ reveals WHICH keys are attended.
    - **V=1[j∈S]** ⇒ `O[i,0]` = P-mass on set S ⇒ confirms the attended set;
      **V=ones ⇒ O=1.0** self-checks that P is normalized.
    Each probe reads through the *correct* O=P@V pipeline (no
    fragment-serialization, no one-hot-O=P confound), validated vs fp32.
    This pinned the mechanism reliably: non-causal single-pass attends only
    `(qb+1)·BQ` keys (the causal `q_max` bound leaking in) — keys ≥ q_max are
    truncated, not miscomputed. *Rule*: when an MMA-internal value resists
    faithful serialization, design known-answer inputs that make the
    pipeline OUTPUT reveal the internal quantity, and isolate stages by
    zeroing the upstream ones (Q=0 ⇒ uniform P). The output is always
    faithful; the fragment registers are not.

## III-8 root-cause lesson — confirm WHICH BINARY runs before debugging source (#14)

14. **Before debugging a kernel's SOURCE, confirm the source is the binary
    that actually executes. A precompiled / AOT / async metallib loaded ahead
    of the JIT path bypasses the source entirely — and every source-level
    diagnostic run against it is inert.** The `backend="mfa"` non-causal
    divergence consumed five sprints (III-7 → III-8e) of register dumps,
    static causal-vs-non-causal differential reading, and oracle-bisection of
    `generate_steel_v2_source` — **all inert**, because `shader_cache.mm`
    calls `try_async_pipeline()` *first* and, for `SteelForwardV2` keys on
    macOS 26, served a pipeline built from the precompiled `async_v2.metallib`.
    That metallib uses `simdgroup_async_copy` (hardware DMA), which Apple
    removed from the macOS-26 runtime AIR compiler; its broken DMA loads only
    `~(qb+1)·BQ` keys per Q-tile → the exact "q-dependent key truncation"
    III-8e's uniform-P oracle measured. **The behavioral mechanism was
    correctly characterized; the LOCATION was mis-attributed to JIT codegen.**
    The tell, in hindsight: III-8e concluded "the `(qb+1)·BQ` signature
    matches NO obvious source line" — a signature that fits *no* source line
    is itself evidence the source isn't running.
    *Rule*: when investigating a kernel bug, the FIRST step (≤10 min, before
    any source reading) is a **which-binary check**:
    - (a) a **sentinel write** in the suspect JIT source — if it never appears
      in the output, the JIT source is not the running binary (§AA.5.x,
      kernel-debugging.md);
    - (b) toggle any env that switches the dispatch/compile path
      (`MFA_DISABLE_ASYNC`, AOT-vs-JIT, forced backend) — if it changes the
      output, the bug is in path *selection*, not in the source you're reading;
    - (c) enumerate every loader of that kernel key — `try_async_pipeline`,
      precompiled-metallib, JIT — and note which wins for the failing key,
      dtype, OS, and arch. The one that *wins* is the one to debug.
    A precompiled metallib is invisible to source grep, git blame, and every
    `generate_*_source` edit. Confirm execution provenance first; debug source
    second.

    ⚠ **CORRECTION (III-9): the async metallib was NOT the root cause either.**
    III-8 concluded "async metallib broken on macOS 26 = root cause" and shipped
    a gate — but that was a *second* wrong turn. Applying THIS lesson properly
    in III-9 (instrument the dispatch to see which path actually runs) showed
    `backend="mfa"` non-causal D∈{64,128} dispatches to the V2 **split-K** path,
    and the real bug was a **scratch-buffer lifetime error**: `pO`/`pL` freed at
    encode time while MLX's lazy execution left the kernels pending, so a
    concurrent allocation reused the pool memory and corrupted the not-yet-run
    reduce. Fix: `enc.add_temporary` (tie scratch to command-buffer completion)
    in `mfa_attention.cpp`. The async gate is independently defensible
    (`simdgroup_async_copy` is genuinely broken on macOS 26) but inert for this
    bug. **Meta-meta-lesson**: lesson #14's "which binary runs" must be applied
    to the *actual failing dispatch* (here: split-K), not the first plausible
    candidate (the async single-pass path) — verify the path with a dispatch
    trace, not by assuming. Full write-up: `backend-mfa-noncausal-divergence.md
    § Resolution (III-9 — CORRECTED)`.

## III-12b perf-claim lesson — state numerator, denominator, direction, absolute (no bare ratios)

15. **A perf claim must state numerator, denominator, direction, AND an absolute (ms) — never a
    bare ratio.** The TQ paged-decode flagship claim shipped in the README as "6–14× faster" with
    NO baseline. That bare ratio was the new gather/dequant+SDPA path vs the *old fused TQ kernel*
    (a real, correctly-directional 6–23× win, with ms absolutes in the III-2 report) — but with the
    baseline dropped, a reader naturally assumes "vs fp16/SDPA", which is wrong (the new path is
    ~1.4–3× *slower* than fp16 dense; its value is 4–5× KV memory). The ambiguity bit twice: it
    misled readers, AND it misled Sprint III-12, which "re-measured" the claim by benching the
    *fused kernel vs fp16* — neither arm of the claim — and concluded (wrongly) that the claim was
    "inverted". III-12b recovered the true baseline from the record (PERF_CLAIMS.md + the III-2
    report) and re-confirmed the claim on 26.6.
    *Rule*: every documented perf number reads "X ms vs Y ms → Z× faster/slower than <named
    baseline>". A `Z×` alone is ambiguous by construction — it cannot encode direction or baseline,
    and that is exactly how a memory-feature shipped as a flagship "speedup" and how a re-measure
    benched the wrong thing. Corollary (compounds #14): before "re-measuring" a claim, recover the
    claim's exact path + baseline + metric from the record — measuring a plausible-but-different
    comparison produces a confident wrong verdict.
