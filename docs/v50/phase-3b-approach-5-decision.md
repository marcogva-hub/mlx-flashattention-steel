# v2.50 Prompt 5c Section B — Approach 5 decision + Architecture B promotion

**Status**: Architecture B (bisection kernel) PROMOTED to AUTO production
default.  Approach 5 (single-pass running top-K state machine with
scatter-gather PASS-2) is Section B v3 follow-up due to
implementation-feasibility constraints documented below.

## Architecture B promotion (Section B Phase B.5 verdict)

Per the Section B decision tree in Prompt 5c:

**Scenario 1 outcome**: Architecture B delivers exact-comparable semantics
to Phase 3a (same FP16 boundary ambiguity — both produce 64-69 elements
per row due to FP16 ties at the threshold) AND speedup ≥ Architecture
B threshold (3.85× empirically, 42.91 ms → 11.15 ms at audit shape).

Decision: **Architecture B → AUTO production default**.

### Env semantics (post-promotion)

| Env | Behavior |
|---|---|
| (unset) | **Architecture B bisection** (3.85× over Phase 3a) — DEFAULT |
| `MFA_DISABLE_TOPK_BISECT=1` | Revert to Phase 3a mx.topk semantics (legacy) |
| `MFA_DISABLE_TOPK_NAX=1` | Opt out entirely (Python reference path) |
| `MFA_TOPK_BISECT=1` | Deprecated (now redundant with AUTO default; back-compat) |

## Approach 5 investigation summary

Per `docs/v50/phase-3b-architectures-comparison.md` §"Section B v2
follow-up roadmap", Approach 5 design:

1. `mx.fast.metal_kernel` with per-row min-heap state in threadgroup
   memory (~16 KB TGM for K_top=64)
2. K-tile streaming: each TG processes K_BLOCK columns at a time
3. Per-thread heap insert/replace operations
4. Output: top-K indices [B, H, N, K_top]
5. PASS-2: SDPA with scatter-gather K/V using indices

### Implementation feasibility analysis

**Step 1-4 (heap-based streaming top-K)**: feasible but complex.
Heap insert/replace operations are SIMD-divergent (each thread may
insert/replace at different heap positions).  Estimated 4-6h focused
implementation + iteration.

**Step 5 (PASS-2 scatter-gather SDPA)**: **NOT FEASIBLE with Apple
SDPA NAX**.  `mx.fast.scaled_dot_product_attention` accepts K/V as
contiguous tensors with optional mask; it does NOT natively support
indexed K/V (top-K indices).  Workarounds:
- (a) Custom Metal kernel for filtered SDPA (~XL effort, 8-12h)
- (b) Materialize filtered K/V via mx.take then call SDPA — but
      mx.take + SDPA is ~10ms at audit shape (eliminates the savings)
- (c) Use bias mask with -INFINITY at non-top-K positions — but this
      is EXACTLY what Architecture B does

### Verdict on Approach 5

The full Approach 5 (steps 1-5) requires either:
- A custom Metal attention kernel for filtered K/V (XL effort), OR
- Reverting to bias-mask approach at step 5 (which is Architecture B)

Given the implementation feasibility constraint at step 5, the
incremental value over Architecture B is bounded by the savings on
the score-materialization step (steps 1-4 vs Architecture B's
materialized scores).  Empirical projection:
- Architecture B: ~11 ms total (4 ms matmul + 5 ms bisection + 2 ms SDPA-bias)
- Approach 5 (steps 1-4 via heap + step 5 via Architecture B bias mask):
  ~9-10 ms (saves materialization but bisection becomes streaming)
- Approach 5 with custom kernel for step 5: ~6 ms theoretical floor

The middle option (steps 1-4 streaming + step 5 bias mask) saves ~1-2
ms over Architecture B — marginal incremental benefit at 4-6h
implementation cost.

**Decision (Section B Phase B.5 Scenario 3 framing)**: Approach 5
deferred to Section B v3 focused follow-up.  Implementation requires
either (a) full custom attention kernel for scatter-gather PASS-2 OR
(b) acceptance of marginal speedup over Architecture B.  Architecture
B is the implementation-feasible native top-K production path
shipped in this section.

## Validation post-promotion

- `tests/test_v50_sprint_5b_section_b_topk_bisect.py`: 8 tests, all
  green post-env-semantic flip (test names updated for promotion).
- `test_env_unset_uses_bisect_default` (was `test_env_unset_uses_phase_3a`):
  validates new AUTO default.
- `test_opt_out_via_disable_env`: validates `MFA_DISABLE_TOPK_BISECT=1`
  reverts to Phase 3a.

## Skill invocations (§AA.2)

| Skill | When | Result |
|---|---|---|
| `/mlx-mfa-apple-primitives-coverage` | Approach 5 feasibility check | Apple SDPA NAX doesn't support indexed K/V → step 5 requires custom kernel OR bias-mask fallback (which is Architecture B) |
| `/metal-kernel-dev` | Approach 5 design review | Heap-based steps 1-4 GREEN feasibility; step 5 custom kernel XL effort |
| `/mlx-code-review` | Pre-merge promotion | Env semantics inverted cleanly; back-compat preserved via `MFA_TOPK_BISECT=1` deprecation pattern |

## Cross-references

- `docs/v50/phase-3b-architectures-comparison.md` (5-architecture
  investigation from Prompt 5b Section B)
- `docs/HARDWARE_SUPPORT.md` (Top-K row update post-promotion)
- `mlx_mfa/attention.py::flash_attention_topk` (production path)
