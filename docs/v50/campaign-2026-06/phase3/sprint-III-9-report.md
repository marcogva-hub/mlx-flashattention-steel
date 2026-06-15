# Sprint III-9 — Pre-Release Full Correctness Sweep

**Date:** 2026-06-16
**Executor:** Claude Opus 4.8 High + 9 parallel sub-agents (one per class) via Workflow `w8o0y0fru`
**HEAD:** 240b226 (V2 split-K / flash-decode scratch-lifetime fix)
**Purpose:** pre-release gate before the v2.53.0+ PyPI tag. Repeat-until-clean across the known
classes (G–J) + the four lesson-driven new classes (D, E, F, K) + the flagged single-pass NaN.

## Outcome (final, after 3 iterations): PRE-RELEASE GATE MET.
Iteration 1 found one real bug (V2 single-pass non-causal last-head OOB-V, forced-path-only);
iteration 2 fixed it AND — via §AA.5.x multi-gate — found+fixed the same pattern in two sibling
kernels (GNA native [default-reachable], STEEL V5); iteration 3 is zero-finding. Full timeline in
**Iteration 2** and **Iteration 3** sections below; the iteration-1 narrative immediately following
is preserved as the historical record. Final fix commits: 240b226, eb68af5, eb5b890.

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

---

## Iteration 2 — disposition applied + §AA.5.x multi-gate (found MORE material)

Marco chose **fix the V2 single-pass kernel**. Root cause pinned: all failing N have a PARTIAL
final K-tile; on M5's MFA_DIRECT_READS path the V direct-read had no bounds check, so for masked
keys (P=0) past kL the OOB read returned NaN and **0 × NaN = NaN** corrupted the output. The last
head sits at the end of the K/V buffer, so its over-read spilled into freed pool memory (hence
pool-history/concurrent-alloc dependence + last-head-only + even-D-column MMA pattern). The
K-boundary mask handles K (its OOB NaN is overwritten by -inf), but V flowed through 0×NaN.

**Fix (eb68af5):** clamp the V direct-read key-row to the last valid key (kL_rem-1) on the partial
final tile — OOB keys read a finite V; P=0 → 0×finite = 0 (correct masked contribution). R.4: 792-config
sweep 0 failures; R.5: suite 1796 ×2; R.7 lock: test_iii9_v2_singlepass_lasthead.py (28).

**Iteration-2 sweep then found the §AA.5.x miss:** the fix was scoped to ONE generator. The IDENTICAL
unbounded direct-V-read pattern existed in TWO sibling kernels (the F agent caught it — vindicating
loop-until-clean):
- **GNA native** (`mfa_gna_fwd.cpp`) — **HIGH, DEFAULT-REACHABLE** via `flash_attention_gna` (D=128/3D/f16/N%32≠0).
- **STEEL V5** (`mfa_steel_fwd_v5.cpp`) — MEDIUM, opt-in (`MFA_ENABLE_V5`); + a falsified "→ safe" comment.

**Fix (eb5b890):** identical kL_rem-1 clamp applied to both; V5 comment corrected. **Multi-gate verified
complete** (lead grep): the ONLY 3 device-direct-V-read sites (`V_cur`) are V2 single-pass / GNA / V5,
all now clamped; every other kernel reads V from smem (`&Vs[...]`, load_safe zero-pad) → structurally
safe. V4 reads only K directly (safe). GNA validated vs an independent element-level fp32 GNA-mask+SDPA
oracle (0 NaN, MAE ≤5e-4); V5 vs fp32 SDPA (0 NaN, MAE ≤9e-4); both under the pool-history+concurrent-alloc trigger.

## Iteration 3 — zero-finding (pre-release gate)

Re-ran the classes whose verdict the V-read clamps could touch, on HEAD=eb5b890:
- **recheck-lasthead**: CLEAN (V2 fix holds across the full envelope + regression set).
- **E** (unfaithful-oracle): CLEAN — 3 fixed paths re-confirmed vs independent fp32; 3 LOW test-coverage
  gaps flagged (turboquant/sage/varlen test refs not fp32-cast — confirmed correct by probe, not defects).
- **F** (partial-tile): multi-gate verified COMPLETE (no 4th direct-V site).
- **I** (numerical-edge): CLEAN — empty-row/NaN-propagation(Rule 8)/overflow/degenerate/short-S all clean;
  all fixed paths + the separately-flagged D=64 N=1000 single-pass item re-confirmed clean (181 III-9 locks).
- **D, G, H, J, K**: structurally invariant to a V-read clamp (no scratch lifetime, cache key, dispatch
  routing, precompiled binary, gradient surface, or hook changes); CLEAN in iteration 1.

Full suite **1820 passed, 2 skipped, ×2 consecutive**.

## PRE-RELEASE GATE: MET.
Every bug class hunted exhaustively (incl. the four lesson-driven classes); the final iteration is
zero-finding. Release scope (Marco-gated, version 2.53.0+ per total scope):
- `da737e7` — async metallib macOS-26 gate (defense; inert for the V2 bugs)
- `240b226` — split-K / flash-decode scratch lifetime fix
- `eb68af5` — V2 single-pass non-causal last-head OOB-V clamp
- `eb5b890` — GNA + V5 multi-gate OOB-V clamp
- the two III-7 quantize_model fixes (already on master)
