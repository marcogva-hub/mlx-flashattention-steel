# Proposed addition to `CLAUDE_V6_NAX.md` — three-axis validation rule

**Status:** PROPOSAL — Marco's discretion to commit, refine wording,
or reject.
**Source:** Release-flow validation report §F (CC observation across
Sprint C → D → v2.33.1 arc).
**Date:** 2026-05-12

---

## Background

Three consecutive silent-bug catches across the Sprint C → D → v2.33.1
arc each surfaced a distinct class of validation gap. Each was caught
by a distinct gate class. Mainline correctness alone — the kind of
"run the test suite and see green" check — was insufficient in all three
cases.

| # | Sprint phase | Silent bug | Caught by gate | Class |
|---|--------------|-----------|----------------|-------|
| 1 | Sprint C Phase 1.1 | Microbench reported 101 TFLOPS on M5 Max (physically impossible — NAX FP16 peak is ~38 TF). Methodology bug: full matrix dims passed to descriptor instead of per-tile dims; only ONE threadgroup dispatched. | Sentinel-fill + RMSE-vs-oracle smoke gate on a tiny shape, run BEFORE any production-shape timing | **Output sanity** |
| 2 | Sprint D Track C | `patch_seedvr2_vae` patcher's instance-level `__call__` override silently failed. Python looks up `__call__` on the TYPE, not the instance — the per-instance override was a no-op. All 4 correctness tests passed because both "patched" and "unpatched" paths invoked the same class-level `__call__`. | Patcher A/B perf bench measuring wall-clock with and without the patch. Pre-fix: 1.00× (no speedup → dead override caught). Post-fix (`__class__` swap): 2.29×. | **Path entered** |
| 3 | v2.33.1 patch | Initial design substituted a bool mask for the float bias to skip the `mx.where` step (~1.3 ms saved unconditionally). Bit-exact on normal cases. But broke the all-False-row → NaN semantic that downstream code relies on (MLX SDPA with all-False bool mask produces finite garbage, not NaN). | Existing `test_all_false_mask_row_gives_nan_or_zero` in `TestSparseAttentionKernel`. Failed on the first revision; revision used float-bias cache instead, preserving semantics. | **Edges preserved** |

Three bugs, three classes, three distinct gate classes required.

---

## Proposed `CLAUDE_V6_NAX.md` insertion

Inserted at the appropriate section level (e.g., after the existing §3
benchmarking discipline section, or as a new top-level methodology rule):

````markdown
## §X. Dispatch-path modification — three-axis validation rule

Any patch that modifies a dispatch decision, routing logic, or kernel
selection path must validate three distinct axes before shipping:

1. **Output sanity** — correctness oracle (PyTorch CPU FP32 cross-check,
   RMSE bar, sentinel-fill coverage gate). Catches: physically
   impossible outputs, addressing bugs that leave gaps, kernel
   miscompiles that produce garbage.
2. **Path entered** — perf or sanity A/B bench that detects whether the
   new path is actually taken. Catches: dispatch elision (silent no-op
   overrides, Python `__call__` type-vs-instance dunder lookup
   gotchas, fallback paths that engage when they shouldn't, env-var
   propagation gaps between Python and C++).
3. **Edges preserved** — semantic edge-case tests for NaN propagation,
   all-zero / all-masked inputs, denormal inputs, boundary conditions.
   Catches: optimizations that are bit-exact on mainline cases but
   break edge semantics other code depends on.

**Mainline correctness alone is insufficient.** The Sprint C → D →
v2.33.1 arc surfaced one silent bug per axis, each caught by the
corresponding gate class. All three axes are mandatory for
dispatch-path patches.

### Practical checklist

Before tagging a release that modifies dispatch:

- [ ] **Output sanity gate**: smoke test with sentinel-fill + oracle
      RMSE check on a small shape, BEFORE any production-shape timing.
      The smoke gate's pre-flight signature must include a non-trivial
      correctness verification — not just "did it run".
- [ ] **Path-entered gate**: A/B perf comparison between the old and new
      paths on at least one representative shape. If perf ratio is
      ~1.00× when the new path is supposed to be faster, the new path
      isn't actually engaged (dead override / fallback engagement).
- [ ] **Edges preserved gate**: run the full pre-existing test suite,
      with NaN/Inf checks active. Any test that was passing before the
      patch and now fails — even if "the new behavior is reasonable" —
      indicates an edge-case semantic shift that downstream code may
      rely on.

### Worked examples

#### Sprint C Phase 1.1 (Output sanity)

`bench/conv_nax_matmul2d_microbench.py` v1 reported 101 TFLOPS on M5 Max
(NAX FP16 peak is 38 TF). Microbench passed because there was no
correctness check on its output — just timing. The methodology bug was
caught by adding a sentinel-fill + `mx.matmul(A.f32, B.f32)` oracle check
on a tiny shape (M=128, K=64, N=64) as a pre-flight gate. After fix:
RMSE=0 on the smoke shape, production timings physically plausible
(43 TF on mid_resnet).

Reference: `docs/conv-nax/conv-nax-phase1_1-microbench-blocker.md`.

#### Sprint D Track C (Path entered)

`patch_seedvr2_vae(model)` initially used `object.__setattr__(mod,
"__call__", patched_fn)`. Python's `__call__` resolution looks up the
TYPE, not the instance — the override was dead. All 4 correctness tests
passed (because both paths called the same class-level `__call__`).
The A/B perf bench in `bench/conv_nax_patcher_ab.py` measured
speedup 1.00× and revealed the dead override. Fix: `mod.__class__ = …`
swap to a dynamically-created subclass with overridden `__call__`.
After fix: 2.29× speedup (matches Phase 1.5 mid_resnet 2.26× ratio).

Reference: `docs/conv-nax/conv-nax-prod-decisions.md` D34.

#### v2.33.1 patch (Edges preserved)

Initial fast-fallback design substituted bool mask for float bias to skip
`mx.where` (~1.3 ms saved unconditionally). Bit-exact on normal cases —
all correctness equivalence tests passed. But MLX SDPA with all-False
bool mask produces finite garbage (no attention), while the float-bias
all-`-inf` row produces NaN softmax (the semantic downstream code
depends on for "no information available" detection). Caught by the
existing `test_all_false_mask_row_gives_nan_or_zero` test failing on
the first revision. Revised patch caches the FLOAT BIAS (not bool mask),
preserving the NaN-for-fully-masked-rows contract.

Reference: `docs/sparse-fallback-audit.md` + commit `9e0ab6a`.

### When to apply

This rule applies to any patch that:
- Changes the kernel selection (`if config_X: use_kernel_A else: use_kernel_B`)
- Routes through a different fallback or fast path
- Swaps `__call__` / `forward` methods (patchers, decorators)
- Modifies the M1-vs-M3 vs M5+ hardware dispatch
- Adds or removes a cache layer (id-keyed, lru-keyed, content-keyed)
- Inlines or unrolls a previous indirect dispatch

It does NOT apply to:
- Pure refactors that preserve the exact dispatch graph
- New API surfaces that don't touch existing routing
- Documentation, tests, build-system changes

When in doubt, apply the three axes anyway — the cost is small (a
focused A/B bench + a `find . -name "test_*.py" -exec` of the existing
edge tests), the cost of a silent bug shipping to PyPI is large.
````

---

## Marco's options

1. **Commit as-is** to `CLAUDE_V6_NAX.md` (one edit, no review cycle).
2. **Refine wording** in editor before commit (e.g., adjust section
   placement, tighten the worked examples, add a fourth axis if Marco
   has a class to add).
3. **Defer**: keep this proposal doc in `docs/proposed-claude-v6-nax-updates.md`
   as a queued methodology change; pick up at next CLAUDE_V6_NAX.md
   revision pass.
4. **Reject** if Marco prefers a different framing (e.g., a single
   "smoke gate is mandatory" rule rather than the three-axis split).

Recommendation: commit option 1 or 2 now. The three concrete bug catches
within ~4 weeks of work (Sprint C → D → v2.33.1) are strong evidence
that the rule's payoff is immediate, not theoretical.
