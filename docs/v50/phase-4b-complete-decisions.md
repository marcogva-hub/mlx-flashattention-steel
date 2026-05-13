# v2.50 Phase 4b-complete — V34 backward causal — PARTIAL

**Sprint date**: 2026-05-14 (Prompt 3 Section B)
**Branch**: `feat/v50-phase4b-complete-prompt3`
**Master tip pre-Section-B**: `48372f1` (post-Prompt-3 Phase 3b halt merge)

## TL;DR

Phase 4b-complete (Prompt 3) executed two distinct phases of work:

### Phase 4b-complete.A — CRITICAL bug fix (SHIPPED)

While validating my Prompt 2 Phase 4b dQ infrastructure, discovered
that `compile_v34_backward_pipeline` (Sprint v2.40.x consolidation
helper) **hardcoded `isCausal=false`** at line 852.  This meant:
- All 5 V34 backward source generators were compiled with
  `descriptor.isCausal == false`
- All `#define V34BWD*_CAUSAL` macros evaluated to 0
- All `#if V34BWD*_CAUSAL { mask block }` were COMPILED OUT
- Prompt 2's Phase 4b dQ work was effectively a **silent no-op in
  production**

This was a latent bug introduced in v2.40.x Sprint C consolidation.

**Fix**: plumbed `isCausal` through:
- `compile_v34_backward_pipeline` signature: added `bool isCausal = false` parameter
- 5 `V34Bwd*Key` structs: added `bool causal` field + hash
- 5 `MFAV34Bwd*` Primitive classes: added `causal_` member + ctor parameter
- 5 raw helpers (`v6_nax_backward_query`, etc.): added `bool causal` parameter
- 5 nanobind bindings: added `nb::arg("causal") = false`
- Python `_v34_backward_vjp(... , causal=False)`: pass causal through

**Empirical impact**:
- dQ kernel causal mask now actually fires
- dQ RMSE at D=64 qL=2048 f16 causal: **8.7e-6** (vs 1e-3 bound — TIGHT pass)
- dK RMSE: **1.19e-5** (vs 1e-4 bound — close)
- dV RMSE: **2.7e-3** (vs 1e-3 bound — 2.7× over, remaining bug below)

This isCausal-plumbing fix alone resolves Prompt 2's "Phase 4b dQ
infrastructure" → "Phase 4b dQ actually working".

### Phase 4b-complete.B — K-parallel kernels (PARTIAL, gated off)

Added per-element causal mask blocks to all 4 K-parallel backward
kernels (dV split, dK split, dKV legacy fused, dKdV fused) mirroring
the dQ Phase 4b pattern.  Each kernel now has:
- `#define V34BWD*_CAUSAL` macro
- `int qL_off` field in device-side params struct (matching host-side)
- Per-element causal mask block before `Stile.row_bin_op<ExpSubOp>(lse_log2)`

The mask blocks are structurally identical to the V34 forward causal
mask (Prompt 2 Phase 4a) which is empirically correct.  But empirical
validation in this session showed:

| Metric | Result | Status |
|---|---|---|
| dV magnitude (V34 causal vs SDPA causal) | 10× SMALLER | structural under-counting |
| dV RMSE | 2.7e-3 (vs 1e-3 bound) | 2.7× over |
| dV correlation with SDPA | 0.73 | not perfect alignment |
| dV ratio (V34/SDPA) | median 0.039 across heads | consistent scaling factor |
| All 3 kernel modes (fused/split/legacy) | identical output | not a kernel-routing bug |

The structural ~25× under-counting (ratio 0.039 ≈ 1/25) is consistent
across all 4 K-parallel kernels.  Possible root causes (not yet
investigated to root cause):
- Off-by-one in fragment coordinate calculation for K-parallel kernels
  (despite the mask block being structurally identical to forward's)
- Subtle interaction between `sg_q_offset` Q-row partition and the
  per-fragment row/col walk
- `qb` Q-loop not iterating all blocks (boundary condition)

The remaining work is a focused kernel-debug session (~1-2h CC):
1. Add a debug-write to the dV kernel that emits the per-(r, c)
   mask decisions to a buffer; verify the mask matches expected
2. Compare dV-with-mask vs dV-without-mask vs SDPA-causal to
   isolate whether the mask is over-masking or under-counting
3. Possibly involves SIMD lane coordinate semantics specific to
   K-parallel kernels that differ from Q-parallel (e.g., fragment
   transpose semantics for P^T)

**Production decision**: `_v34_eligible(causal=True)` returns False
and `_v34_backward_carveout(causal=True)` returns False — production
callers using `flash_attention(causal=True)` continue to use SDPA-vjp
fallback (bit-identical, safe).  The Phase 4b-complete.B infrastructure
ships compiled-in but never activates until the residual is resolved.

## DC1 — The compile_v34_backward_pipeline isCausal bug

**Pre-Prompt-3 (master)**:
```cpp
NAAttentionKernelDescriptor desc(
    blockDims, (unsigned short)D, ...,
    AttentionKernelType::forward,
    /*scale=*/scale,
    /*bypassThreadgroupMemory=*/false,
    /*isCausal=*/false, /*masked=*/false);  // ← HARDCODED FALSE
desc.useV34 = true;
NAAttentionKernel ker(desc);
std::string src = source_gen_fn(ker);  // <-- isCausal accessed here
```

Source generators (e.g., `createV34BackwardQuerySource()`) read
`isCausal` to produce the macro:
```cpp
ss << "#define V34BWD_CAUSAL " << (isCausal ? 1 : 0) << "\n";
```

With `isCausal` hardcoded to false, the macro was always `#define
V34BWD_CAUSAL 0`, and the `#if V34BWD_CAUSAL` mask block was always
excluded from compilation.

**Detection**: validation test for Phase 4b dQ showed dQ matching
SDPA-vjp causal at RMSE ~3e-4 (close to tolerance).  Suspecting an
issue, I dumped the generated source with
`MFA_V34BWDF_DUMP_SOURCE=1` and grep'd `V34BWDF_CAUSAL` — found it
was 0 despite the test passing `causal=True` through Python.

**Fix**: Threaded `isCausal` through:
1. `compile_v34_backward_pipeline` accepts `bool isCausal` parameter
2. Each `V34Bwd*Key` cache key includes `bool causal` field
3. Each `MFAV34Bwd*` Primitive stores `bool causal_` and passes it
4. Each raw helper (`v6_nax_backward_query`, etc.) accepts `bool causal`
5. Each binding exposes `nb::arg("causal") = false`
6. Python `_v34_backward_vjp` accepts `causal` parameter and threads
   to each binding call
7. Python `_make_mfa_custom._backward` passes `causal` to `_v34_backward_vjp`

**Impact**: Phase 4b dQ now actually fires; dQ RMSE drops from
"not validated in production" to 8.7e-6 (TIGHT).

## DC2 — Why the dV residual was not resolved this session

I committed substantial effort to root-causing the K-parallel kernel
dV residual:
- Verified mask block is compiled in (V34BWDF_CAUSAL = 1 in source dump)
- Verified mask predicate `(r < c) ? neg_inf : fg[loc]` is structurally
  identical to V34 forward's (which works perfectly)
- Verified all 3 kernel modes (fused, split, legacy) produce the same
  RMSE — eliminates kernel-routing as the cause
- Verified non-causal regression test passes (RMSE 6e-6 — unchanged)
- Component analysis showed errors concentrate at K=0 (low-K, many Q
  contributors) and decay to noise at K=qL-1 (one Q contributor)
- Correlation analysis showed V34 dV vs SDPA dV correlation 0.73, ratio
  ~0.039 — structural ~25× under-counting

The remaining investigation requires either:
- Deep debugging session with per-element kernel sentinel writes
- Comparison against a known-correct reference implementation at the
  per-Q-row level
- Possibly involves fragment transpose semantics for `dV += P^T @ dO`
  vs `dQ += dS @ K` that I haven't fully understood

Per §AA.1 halt protocol, this exceeds the safe envelope for an
autonomous session that also needs to evaluate Section C feasibility.
The Phase 4b-complete.B infrastructure ships compiled-in (zero
production risk because eligibility gates prevent causal activation).

## DC3 — Strategic value of Phase 4b-complete.A alone

Even WITHOUT Phase 4b-complete.B resolution, this session's work
delivers:
1. **Critical latent bug fixed**: the `compile_v34_backward_pipeline`
   hardcoded `isCausal=false` would have continued to silently
   no-op any future causal kernel work
2. **dQ kernel now works**: V34 backward dQ for causal is now ready
   for production use whenever the dispatch gate is lifted; just
   needs the K-parallel kernel residual fixed
3. **Infrastructure shipped**: the entire causal plumbing chain
   (Primitive + Key + Helper + Binding + Python) is in place; the
   K-parallel kernel fix is a localized debugging task

## Skill invocations log (per §AA.2)

| Phase | Skill | Status |
|---|---|---|
| 4b.0 §AA.5 premise validation | (no skill — STATUS doc read + structural review) | done |
| 4b.1 dV split causal block | manual edit | done |
| 4b.2 dK split causal block | manual edit | done |
| 4b.3 dKV legacy fused causal block | manual edit | done |
| 4b.4 fused dKdV causal block | manual edit | done |
| 4b.5 CRITICAL isCausal plumbing fix | manual investigation + multi-file edit | done — root cause found and fixed |
| 4b.6 lift eligibility gates | (reverted per §AA.1 — dV residual unresolved) | reverted |
| 4b.7 three-axis validation | (test suite — 1140 pass, zero new regressions) | partial — dV residual remains |
| 4b.8 corruption audit | `/mlx-debug-forensics` (manual coord + magnitude analysis) | done — confirms structural under-counting, not kernel routing |
| 4b.9 perf bench | not run (residual unresolved) | N/A |
| 4b.10 pre-merge | `/mlx-code-review` | (pending review of partial-state merge) |

## Files changed

| File | Net LOC | Purpose |
|---|---|---|
| `csrc/mfa/v6_nax/NAAttentionKernel.cpp` | +180 | 4 K-parallel kernel CAUSAL macros + qL_off fields + per-element mask blocks |
| `csrc/mfa_v6_nax_primitive.cpp` | +50 | 5 cache keys + 5 Primitive constructors + 5 compile call sites + 5 helper signatures threaded with causal |
| `csrc/bindings.cpp` | +20 | 5 forward decls + 5 binding lambdas extended with causal |
| `mlx_mfa/attention.py` | +15 | `_v34_backward_vjp` accepts/threads causal; eligibility gate retained on causal |
| `mlx_mfa/dispatch_policy.py` | +5 | `_v34_backward_carveout` retains `not causal` constraint |
| `docs/v50/phase-4b-complete-decisions.md` | +200 (new) | this doc |

## Recommended next steps (focused future session)

1. Bench V34 backward causal at qL=64 with eligibility gate bypassed
   (single Q-block; eliminates qb-loop boundary effects from the
   investigation)
2. Add a debug-write to the dV kernel inside `#if V34BWDF_CAUSAL`
   that emits per-element (r, c, fg[loc] pre-mask, fg[loc] post-mask)
   to a 4D fp32 buffer; compare against a Python reference
3. Verify fragment transpose semantics for `dV_accum += P^T @ dO`
   — possibly `NAXFrag::mma` with `transpose_a=true` reads from
   different row/col indices than I assume
4. Once residual resolved, lift eligibility gates (one-line revert in
   `_v34_eligible` and `_v34_backward_carveout`)
5. Re-run full three-axis validation + cross-session §AA.4 bench

Estimated effort: ~2-3h CC dedicated debug session.

## Master state post-Phase-4b-complete (Prompt 3 Section B)

- Critical isCausal-plumbing fix LANDED
- Phase 4b dQ kernel now WORKS in causal mode (was silent no-op pre-fix)
- 4 K-parallel kernels have causal infrastructure compiled-in (gated off)
- Production behavior unchanged: causal callers continue using SDPA-vjp
- CHANGELOG entry: Phase 4b-complete partial (dQ works; K-parallel pending)
