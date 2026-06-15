# Sprint III-9 — Pre-Release Full Correctness Sweep

**Date:** 2026-06-16
**Executor:** Claude Opus 4.8 High + 9 parallel sub-agents (one per class) via Workflow `w8o0y0fru`
**HEAD:** 240b226 (V2 split-K / flash-decode scratch-lifetime fix)
**Purpose:** pre-release gate before the v2.53.0+ PyPI tag. Repeat-until-clean across the known
classes (G–J) + the four lesson-driven new classes (D, E, F, K) + the flagged single-pass NaN.

## Outcome: NOT yet clean — one real bug found (forced-path-only); release BLOCKED pending its disposition.

## Per-class results (each backed by exhaustive enumeration; all validated vs independent fp32)

| Class | Topic | Verdict | Evidence |
|---|---|---|---|
| **K** | Lazy-eval scratch lifetime (#15) | **CLEAN** | Every `allocator::malloc` in the compiled extension is output-array-backed or `add_temporary`-managed; ZERO `allocator::free` remain in csrc; only `mfa_smooth_quant.cpp:252 mean_buf` binds raw, and it's output-backed (verified byte-stable under concurrent-alloc differential across fresh processes). Both fixed sites (split-K, flash-decode) re-verified correct under the concurrent-alloc trigger. No third site. (`.varlen_append` has a raw malloc but is uncompiled dead code — hygiene cleanup, not a runtime risk.) |
| **D** | Which-binary / macOS-26 workarounds (#14) | **CLEAN** | `async_v2.metallib` (only shipped precompiled binary) gated off at macOS 26+; `MFA_DISABLE_ASYNC` differential bit-identical (inert). `try_precompiled_pipeline` (CP9) serves nothing on M5 (metallib misses M0/BK32 vs live M1/BK64); even a hit == JIT (compile_metallib bakes live source). Other OS/gen-gated paths probed vs fp32, clean. |
| **G** | Cache-key / knob / dispatch | **CLEAN** | The 3 fixes since v2.52.1 introduced no new key/field/knob; static cache-key invariant test green (20); `tie()` covers all 18 members; knobs (`MFA_DISABLE_V2`, `MFA_FORCE_SPLITK`) live + correct; `auto` routes dense→SDPA as documented. |
| **H** | Gradient / vjp completeness | **CLEAN** | All autograd paths derive backward via `mx.vjp` of fp32 SDPA (anchored to oracle); `mx.grad` probes within f16 tol (≤7.8e-3) for dense/GQA/sparse/forced-mfa; conv3d loud-fails on grad (correct). 2 coverage nits (not bugs). |
| **I** | Empty-row / numerical-edge | **CLEAN (0 new)** | Empty/masked rows → zeros no-NaN; NaN/Inf propagate loud+local (Rule 8); fp16 overflow protected by online-softmax max-sub; degenerate shapes clean (D=16/32 loud-reject). The one NaN (N=128 S=1 mfa non-causal) is the short-S corner of the single-pass finding below, not new. |
| **J** | Hook coverage (Pattern #8) | **CLEAN** | `install_hooks` patches exactly `mx.conv_general`/`mx.conv3d`; all other auto-routed optims are internal public fns; no silent-hook-fallback; recent fixes didn't change hook surface. |
| **E / K-adjacent / F / I-corner** | Unfaithful-oracle / single-pass NaN / partial-tile / short-S | **1 REAL BUG (below)** | Four independent agents converged on the same defect. |

## The one real bug — V2 single-pass non-causal last-head corruption (forced-path-only)

**Path:** `flash_attention(..., causal=False, backend="mfa")` → `SteelForwardV2` single-pass
(`MFA_FORCE_SPLITK=0` / occupied grid). **NOT** split-K (that's fixed), **NOT** the async metallib.

**Symptom:** the **last head** (h=H−1) of the output is corrupted (NaN / stale-pool garbage) at:
- D=64: N ∈ {224, 992–1020, 1023}; and short S ∈ {1,2,3,4,16}
- D=128: N = 383; and short S ∈ {1,2}

**Class:** uninitialized-memory / pool-history-dependent (NOT a deterministic coverage gap — the
sentinel pre-fill shows the kernel writes all cells when run *alone*; corruption appears only with a
concurrent allocation or a heterogeneous in-process allocation history). The corrupted cells form an
even-D-column MMA-fragment pattern in the last head; written cells match fp32 at ~6e-6. Likely an
**OOB / uninitialized read at the last head for the partial last K-tile** (the last head sits at the
end of the K/V buffer, so an over-read spills into freed pool memory → NaN), distinct from the
split-K scratch-free bug. Pinning the exact line is hard kernel work (resisted static reading, same
as split-K).

**Reachability / severity:** **Not default-reachable.** `backend="auto"` routes non-causal dense
D∈{64,128} → SDPA and is clean across the entire envelope (0/24 poisoned configs). `backend="mfa"`
**causal** clean; **V1** (`MFA_DISABLE_V2=1`) clean; **split-K** clean; bf16 sometimes affected.
This is the genuine unresolved tail of the III-8 single-pass non-causal investigation — there WAS a
single-pass bug (uninitialized read), separate from the split-K lifetime bug III-9 fixed.

**Disposition (Marco-gated, pre-authorized in `backend-mfa-noncausal-divergence.md`):**
1. **Fix the V2 single-pass kernel** — pin the last-head OOB/uninitialized read and mask/initialize
   it (deep kernel work; preserves a fast V2 non-causal path that auto never selects anyway).
2. **Decline V2 non-causal** — route forced `backend="mfa"` non-causal D∈{64,128} to V1 (verified
   clean, a real MFA kernel) with a Rule-8 note. A CORRECTNESS fix (removes the silent-garbage forced
   path), low-risk; the path is non-default and slower than SDPA on M5, and has now had TWO
   correctness bugs. Recommended.

## Loop status
Iteration 1 found material (the single-pass bug). Per the loop protocol, after its disposition is
applied + locked, a fresh full iteration must run to zero-finding before the pre-release gate is met.
Classes D, G, H, I, J, K are clean and unlikely to regress from a single-pass dispatch/kernel change,
but will be re-confirmed in the closing iteration.

## Pre-release gate: NOT MET (one open finding). Release held.
