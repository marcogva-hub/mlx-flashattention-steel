# Prompt 5e Phase 0 — Modifications inventory v2.39.1 → master 53c914c

**Scope**: 56 commits, 83 unique files modified across 5 main categories.

## Files modified by category

| Category | Count | Notable files |
|---|---|---|
| C++/Metal kernels (`csrc/`) | 8 | `NAAttentionKernel.cpp` (5935→7507 LOC), `mfa_v6_nax_primitive.cpp` (1438→2555), `v6_nax_compile.mm` (722→894), `mfa_sparse_attention.cpp` (1078→1217), `mfa_attention.cpp`, `bindings.cpp`, headers |
| Python core (`mlx_mfa/`) | 4 | `attention.py`, `dispatch_policy.py`, `lcsa_nax.py`, `__init__.py` |
| Tests (`tests/`) | 23 | 11 new v50 tests + 12 modified |
| Documentation (`docs/`) | 44 | 18+ v50 status/decisions docs, audits, methodology |
| Top-level (CHANGELOG, CLAUDE.md, etc.) | 8 | Including AGENTS.md, ENV_VARS.md, README |

## Sprint accumulation mapping

| Sprint / Prompt | Files touched | Notable code |
|---|---|---|
| Sprint 1 (density threshold) | `lcsa_nax.py`, tests | `DEFAULT_DENSITY_THRESHOLD = 1.01` |
| Sprint 2 (RoPE NAX) | `attention.py` | `flash_attention_rope_unified` routes to `mx.fast.rope` |
| Sprint 3 (Top-K Phase 3a) | `attention.py` | Phase 3a Apple SDPA NAX dispatch (1.25× speedup) |
| Sprint 4 (Phase 4a fwd causal) | `NAAttentionKernel.cpp` | `createV34Source()` causal extension |
| Sprint 4 (Phase 4b dQ causal) | `NAAttentionKernel.cpp`, `mfa_v6_nax_primitive.cpp`, `v6_nax_compile.mm` | dQ kernel causal mask |
| Sprint 4 (Phase 4b 4 K-parallel) | Same C++ files | Split + fused dKdV causal |
| Sprint B v2.40-internal (D=128 split) | C++ files | D=128 split kernels |
| Prompt 4 (multi-gate dV residual) | `mfa_v6_nax_primitive.cpp:625` | Causal-routing gate lifted |
| Prompt 5a Section A (Sprint 1 bwd fix) | `attention.py` | `_sparse_nax_with_sdpa_vjp` custom_function wrapper |
| Prompt 5a Section B (xfails) | tests + multi files | 8 xfails resolved |
| Prompt 5b Section A (PoC dV sparse) | C++ + `attention.py` + tests | First native sparse kernel |
| Prompt 5b Section B (Top-K bisection opt-in) | `attention.py` + test | Architecture B opt-in `MFA_TOPK_BISECT=1` |
| Prompt 5b Section C (D=128 attn_bias fix) | `mfa_attention.cpp:892` | V1 STEEL bias-drop bug |
| Prompt 5b Section D (D=128 backward broadening) | `dispatch_policy.py:374` | `head_dim in (64, 128)` |
| Prompt 5c Section A (sparse-LSE foundation) | `mfa_sparse_attention.cpp`, `lcsa_nax.py`, `attention.py` | `sparse_attention_forward_with_lse` + hybrid orchestrator |
| Prompt 5c Section B (Architecture B AUTO) | `attention.py` | Bisection promoted to AUTO default |
| Prompt 5d Section A (3 new sparse kernels) | C++ + `attention.py` + tests | dQ + dK split + fused dKdV sparse |
| Prompt 5d Section B v3 (Pattern #6) | docs + `attention.py` routing revert | Routing reverted to hybrid; native = opt-in |

## Critical paths classification

### Production-active (routed by default)

| Path | Env requirement | Code |
|---|---|---|
| Dense forward D ∈ {64, 128} | none | Apple SDPA NAX |
| Dense backward D=64 qL≥2048 | `MFA_ENABLE_V34_BACKWARD=1` | V34 NAX-direct fused |
| Dense backward D=128 qL≥2048 | `MFA_ENABLE_V34_BACKWARD=1` | V34 NAX-direct split |
| Causal backward D ∈ {64, 128} qL≥2048 | `MFA_ENABLE_V34_BACKWARD=1` | V34 NAX-direct (Prompt 4 multi-gate fix) |
| Sparse forward | none | LCSA NAX dispatcher (Sprint 1 density threshold fix) |
| Sparse backward V34-eligible | `MFA_ENABLE_V34_BACKWARD=1` | `_v34_sparse_hybrid_vjp` (NAX sparse fwd + native dV + SDPA-vjp dQ/dK) |
| Sparse backward V34-ineligible | none | Section C wrapper (SDPA-vjp throughout) |
| Top-K | none | Architecture B bisection (Apple SDPA NAX bias-mask PASS-2) |
| Causal D=128 + attn_bias mode 1/2 | none | V2 STEEL bias-aware (Prompt 5b Section C fix) |

### Opt-in (research / benchmarking)

| Path | Env | Code |
|---|---|---|
| V34 full native sparse backward | `MFA_V34_BWD_SPARSE_NATIVE=1` + above | `_v34_backward_vjp_sparse_full_native` (Prompt 5d, 4 native sparse kernels) |
| Phase 3a `mx.topk` Top-K | `MFA_DISABLE_TOPK_BISECT=1` | Legacy mx.topk path |
| Python reference Top-K | `MFA_DISABLE_TOPK_NAX=1` | Pure Python reference |

### Research-only / unused-by-default-paths

| Path | Why preserved | Code |
|---|---|---|
| 4 V34 backward sparse kernels (dQ, dV PoC, dK split, fused dKdV) | Reference impl + future hardware re-test | C++ source generators + Primitives + bindings |
| STEEL backward D=128 N≥2048 (2 xfails) | Legacy path; bug; V34 is production | csrc/mfa_steel_bwd.cpp |
| Various legacy v6_nax_backward_* helpers | Back-compat | csrc/bindings.cpp |

## CHANGELOG entries (Unreleased — for v2.50)

Per `grep "^### " CHANGELOG.md` between `[Unreleased]` and next `[2.39.X]`:

1. Decisions (v2.50 Prompt 5d — Pattern #6 empirical findings)
2. Changed (Section A v3 — V34 backward sparse routing REVERTED per Pattern #6)
3. Decided (Section B v3 — Approach 5 SKIPPED per Pattern #6 inference)
4. Added (Section A v3 — Prompt 5d — V34 backward sparse FULL NATIVE)
5. Changed (Section B Prompt 5c — Top-K bisection PROMOTED to AUTO default)
6. Docs (Section E Prompt 5b — HARDWARE_SUPPORT.md final narrative)
7. Fixed (Section C Prompt 5b — D=128 attn_bias mode 1/2 causal bug)
8. Added (Section A Prompt 5c — Sparse backward hybrid + sparse-LSE foundation)
9. Added (Section B Prompt 5b — Top-K bisection kernel as opt-in)
10. Added (Section A Prompt 5b — V34 backward block-sparse NAX PoC + scaffold)
11. Added (Section D Prompt 5b — D=128 V34 backward broadening)
12. Fixed (Section C Prompt 5a — Sprint 1 backward regression RESOLVED)
13. Various earlier sprints (Phase 4b dV residual, RoPE NAX, density threshold, ...)

## File size summary (top 5 by post-v2.50 line count)

| File | LOC | Major sections |
|---|---|---|
| `csrc/mfa/v6_nax/NAAttentionKernel.cpp` | 7507 | 4 forward kernel sources + 5 backward (dQ/dK/dV split + fused dKdV + legacy fused dKV) + 3 sparse variants + 1 dV sparse PoC |
| `mlx_mfa/attention.py` | 6641 | flash_attention + 30+ variants + dispatch policies + Top-K bisection kernel |
| `csrc/mfa_attention.cpp` | 3331 | MFAttention Primitive + Forward dispatch + V1/V2 STEEL routing |
| `csrc/mfa_v6_nax_primitive.cpp` | 2555 | 5 V34 backward Primitives + 3 sparse Primitives |
| `csrc/mfa_sparse_attention.cpp` | 1217 | sparse_attention_forward + V1/V2 kernels + sparse-LSE return |

## Pre-existing flake

`tests/test_v50_sprint_5b_section_b_topk_bisect.py::test_bisect_threshold_basic_correctness` passes in isolation but fails in full suite (state contamination from prior tests). Pre-existing since Prompt 5b. Not a Prompt 5d/5e regression.

## Tests baseline at master 53c914c

```
1249 passed, 2 xfailed, 32 xpassed (xpass = formerly xfail decorators that no longer apply but not unmarked)
```

The 2 xfails are STEEL backward D=128 legacy bugs (preserved per Section C Prompt 5a decision tree verdict β).
