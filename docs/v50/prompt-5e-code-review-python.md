# Prompt 5e — Python Code Review (v2.39.1 → HEAD `53c914c`)

Scope: `mlx_mfa/attention.py`, `mlx_mfa/dispatch_policy.py`, `mlx_mfa/lcsa_nax.py`,
`mlx_mfa/__init__.py`. Baseline: 1249 pass, 2 xfail.
Diff: 761 LOC added, 48 deleted across 4 files.

## Critical findings (must fix before release)

**None.** The four critical areas examined (Section C wrapper, hybrid orchestrator,
full-native orchestrator, top-K bisection routing) are functionally correct and
matched by tests. No data corruption, no crashing edge case, no
backward-incompatible API drift observed.

## High-priority findings

### H1. `_v34_sparse_hybrid_vjp` recomputes the forward inside backward
`mlx_mfa/attention.py:2339-2341` calls `sparse_attention_nax_with_lse(q,k,v,...)`
a *second* time inside `_backward`, discarding the `O,L` already computed in
`_impl` (line 2324). The inline comment at 2335 acknowledges this ("cheap; could
be cached via primal trace") but for the documented use-case (D=128 hybrid path),
the sparse forward is non-trivial (~17 ms at audit shape per the perf table in
the comment block 2615-2618). Net cost in any `value_and_grad` call is one
redundant sparse-forward dispatch. Same pattern at `attention.py:2426-2428` in
the full-native orchestrator.

Fix: pass `O,L` through via `outputs` parameter of `@_impl.vjp`. The pattern is
already used by `_make_mfa_sparse_custom._backward` at line 2731
(`O, L = outputs`). The hybrid/full-native versions ignore `outputs` and
recompute. Confidence HIGH — verified by reading both code paths.

### H2. Inconsistent `os.environ` import style (`attention.py:2567`)
`flash_attention_sparse` re-imports `os as _os` locally even though `os` is
already imported at module scope (line 26). Three lines later it falls back to
the module-level `os` (lines 2600, 2632). This is dead code at best; at worst
it suggests the author copied a snippet without checking imports. Fix: drop
line 2567-2568 and use the module-level `os` throughout. Confidence VERIFIED.

### H3. Top-K `else` branch can dispatch with unsupported D
`attention.py:2448-2454` in `_v34_backward_vjp_sparse_full_native` uses
`if head_dim == 64: ... else: # D=128 ...`. The outer eligibility check at
line 2604 enforces `D in (64, 128)`, so today the `else` only ever fires for
D=128. But the inner function has no defensive guard, so any future change to
the outer predicate (e.g., adding D=256) silently routes to a D=128-only
kernel. Fix: make the branch explicit (`elif head_dim == 128:` + `else: raise`)
or assert at function entry. Confidence VERIFIED — read both paths.

### H4. `flash_attention_topk` lacks input validation on `topk_ratio`
`attention.py:3043`: `k_count = max(1, math.ceil(topk_ratio * S))`. No check
that `topk_ratio > 0` or `topk_ratio <= 1`. Negative values yield `k_count=1`
silently. Values `> 1` fall through to the reference path via the `k_count < S`
eligibility check (line 3089) — eventually correct (returns full-attention
output) but wasteful. Per Rule 8 (loud failure), prefer an explicit
`raise ValueError` for out-of-range input. Confidence VERIFIED.

## Medium-priority findings

### M1. Stale env-var name in comments (`attention.py:3079, 3093`)
The bisection block comments reference `MFA_TOPK_BISECT=1` as the opt-in env
var, but the actual code reads `MFA_DISABLE_TOPK_BISECT` (line 3082). Line
3079 explicitly says "(deprecated) MFA_TOPK_BISECT=1: previously opt-in; now
redundant" but line 3093 in the conditional block still uses the old name in
the explanatory comment without a "deprecated" qualifier. Confusing for a user
grepping. Fix: replace line 3093 comment with current env-var semantics.

### M2. `_bisect_opt_in` is misleadingly named (`attention.py:3084`)
`_bisect_opt_in = not _disable_bisect` — but bisection is the production
default (per comment 3083-3084 "Bisection IS the default"). Rename to
`_use_bisect` for clarity. Same file, line 3109.

### M3. `_v34_sparse_hybrid_vjp` docstring mislabels it DEPRECATED
`attention.py:2390`: the docstring says "DEPRECATED in v2.50 Prompt 5d Section
A.4". But the production routing at `flash_attention_sparse:2637` *calls the
hybrid by default* (full-native is the opt-in via
`MFA_V34_BWD_SPARSE_NATIVE=1`, lines 2631-2639). Per the docstring + Prompt 5d
comment block (2610-2627), hybrid is production-optimal at typical shapes.
"DEPRECATED" is wrong — should read "Hybrid is the production default;
full-native opt-in for benchmarking only".

### M4. Magic-number `_wm = 4` (`attention.py:2442, 2445, 2450, 2452`)
The warp-multiplier `_wm = 4` is passed positionally to four kernels with no
explanation. Promote to module-level constant or named kwarg with a one-line
comment ("V34 sparse kernels: warp multiplier matches forward dispatch
config"). Same number appears in `_make_v34_sparse_hybrid_vjp:2352` as bare
literal `4`.

### M5. `mask_bytes >= 4096` threshold is undocumented in lcsa_nax
`attention.py:2587` enforces `mask_bytes >= 4096` with an inline comment
explaining the constraint. The same constraint is documented in
`lcsa_nax.py:18-19` for `sparse_attention_nax` but NOT in
`sparse_attention_nax_with_lse` docstring (`lcsa_nax.py:182-191`). Users
calling `sparse_attention_nax_with_lse` directly will get a C++ exception
rather than a documented constraint failure. Add the constraint to both
docstrings.

### M6. `dV_partials` shape assumption (`attention.py:2350-2353`)
`mx.sum(dV_partials, axis=2)` assumes the kernel returns a `[B,H,WM,N,D]`-like
shape with WM in axis 2. No assert. If the C++ binding shape ever drifts, the
sum collapses the wrong axis silently and the resulting gradient is
plausible-looking garbage. Per Rule 7 (regression awareness) and Rule 8 (loud
failure), add `assert dV_partials.ndim == 5 and dV_partials.shape[2] == _wm`.
Same pattern at lines 2444-2454.

## Low-priority / nit findings

### L1. Type annotations missing
`_make_sparse_nax_with_sdpa_vjp` (line 2212): signature has type hints; nested
`_impl(q, k, v, block_mask)` and `_backward` do not. Same for the hybrid +
full-native makers. Public-API surface (`flash_attention_topk`,
`flash_attention_sparse`) is well-annotated; internal closures could match.

### L2. `_v34_backward_carveout` predicate has redundant kv-len check absent
`dispatch_policy.py:384-390` checks `seq_len >= 2048` but ignores `kv_seq_len`.
For cross-attention (qL != kL), the predicate routes based on qL only. Today
all V34 sites are self-attention so this is harmless. If cross-attention V34
ever ships, the predicate needs revisiting. Flag as DEDUCED.

### L3. `_make_v34_sparse_full_native_vjp` ignores the saved forward output
`attention.py:2411-2416`: the forward computes `O, L` and returns only `O`.
`L` is discarded (not passed to backward via `outputs`). Backward re-runs
forward and gets a fresh `L_sparse` (line 2426). Same as H1 — fix in concert.

### L4. `_V2_DEFAULT_WORK_THRESHOLD` magic literal (`lcsa_nax.py:56`)
`2_147_483_648` is computed `= 4096 * 4096 * 128`. Currently expressed as a
comment beside the literal. Prefer `_V2_DEFAULT_WORK_THRESHOLD = 4096 * 4096 *
128` so the rationale lives in the value.

### L5. `_disable_auto` is set but only used in one branch
`attention.py:2568-2569`: `_disable_auto` gates the M5+ symmetric-bt path. The
asymmetric-mask M5+ path at line 2683 ignores `MFA_DISABLE_AUTO_HOOKS`. If a
user sets the env var, only some auto-routes disengage. Either honor the var
at line 2683 too, or document the asymmetry in the env-var help.

### L6. `_v34_sparse_hybrid_vjp` doesn't check `D in (64, 128)` defensively
The outer guard at `flash_attention_sparse:2604` enforces this, but a direct
caller into `_v34_sparse_hybrid_vjp(...)` (it's module-private but not
underscored at the module surface — accessible via attribute access) would hit
an unhelpful C++ exception. Low-priority because it's `_`-prefixed.

### L7. Test gap: `_make_sparse_nax_with_sdpa_vjp` LRU-cache pollution
The factory is `@functools.lru_cache(maxsize=64)` keyed on `(scale, causal,
bt)`. No test verifies cache eviction behavior or that the same closure is
returned for identical keys. Low risk because the closure is pure relative to
its key. Flag as test gap, not a bug.

## API surface consistency

The four V34-related entry points
(`_sparse_nax_with_sdpa_vjp`, `_v34_sparse_hybrid_vjp`,
`_v34_backward_vjp_sparse_full_native`, `flash_attention_sparse`) all accept
`block_mask` as a *positional* arg in the same slot (4th). Good.
`flash_attention_topk` takes `mask` as keyword-only via default-None — different
name and different position. Acceptable because the function is semantically
different (top-K is not block-sparse), but a one-line cross-reference docstring
note would help users navigate.

## Test coverage assessment

Modified functions vs. tests-added:

| Function | Tests file | Coverage |
|---|---|---|
| `_make_sparse_nax_with_sdpa_vjp` | `test_v50_sprint_5d_*.py:197` (Section C fallback test) | adequate |
| `_v34_sparse_hybrid_vjp` | `test_v50_sprint_5c_*.py` | full |
| `_v34_backward_vjp_sparse_full_native` | `test_v50_sprint_5d_*.py` (6 test cases) | full |
| `_topk_bisect_threshold_kernel` | `test_v50_sprint_5b_section_b_*.py` | direct + via topk |
| `_v34_backward_carveout` D=128 | `__init__.py:diagnostics` smoke covers it | adequate |
| `sparse_attention_nax_with_lse` | `test_v50_sprint_5c_*.py:44-77` | adequate |

No test gaps blocking release.

## Overall verdict: **GO** for v2.50 release

No critical issues, no correctness blockers, no test gaps for new functionality.
The high-priority findings (H1-H4) are quality/maintainability improvements
that should land in a v2.50.x follow-up; none risk a wrong-result or crash on
the documented happy paths. The medium-priority items are docstring/naming
cleanup. None are release blockers.

Recommended follow-up sprint (post-tag):
- H1+L3 in one commit: pass `O,L` through `outputs` for both hybrid +
  full-native, eliminate the duplicate sparse-forward dispatch.
- H2+H3+H4 in one commit: defensive cleanups in the V34 sparse path.
- M1-M6 as a coordinated docstring/comment audit.
