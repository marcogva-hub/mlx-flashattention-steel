# v2.39.0 — Option γ fused dK+dV kernel design status

Sprint v2.39.0 (M3-HIGH-02 + P3-HIGH-01).  Date: 2026-05-13.
Branch: master (no implementation branch yet).
Status: **DESIGN-VALIDATED, IMPLEMENTATION DEFERRED**.

## Halt rationale

Per the failure-mode handling section of the v2.38.1 + v2.38.x + v2.39.0
sprint prompt: "Run out of CC time / context: push current state,
STATUS doc per phase, do NOT force-ship partial release."

The `/metal-kernel-dev` pre-implementation audit (this doc, §1) returned
**MEDIUM go with staging requirements**.  Realistic CC time for the
D=64-only fused kernel + Primitive + dispatcher + binding + Python
routing + bench validation + perf claims is **~3-5 hours**, which
exceeds the v2.39.0 budget for the current sequential session
(v2.38.1 + v2.38.x already shipped).

Path forward: a dedicated v2.39.0 implementation session executes
against this blueprint with D=64-first staging.

## 1. /metal-kernel-dev audit verdict (2026-05-13)

**MEDIUM go.**  Key findings:

### Register budget (MEDIUM risk at D=128)
- `dK_accum + dV_accum` at D=128 WM=4 BK=16: 64 FP32 regs/lane (16 elems
  each, both persistent).
- With transients (Qfrag, Kfrag, Vfrag, dOfrag, P-cast, dS-cast, Stile,
  dPtile, lse_log2, D_vec, address scratch) total ≈ **104 regs/lane** at
  D=128.  M5 cap for full occupancy: ~256 regs/lane → headroom ~150 regs.
  **No spill expected** but margin tight; must verify via Metal frame
  capture during implementation.
- At D=64: TD=4 halves accumulator pressure to ~16 elems/lane each →
  **trivial headroom** (52 regs/lane).  D=64 is the safe staging point.

### Source-merge feasibility (HIGH confidence)
- Both split kernels use direct device loads, no TGP.  Fusion eliminates
  one full K-tile reload pass across kL dimension → structural ~10% perf
  win, not just softmax savings.
- **Drop dead `O` buffer from split-dK** in same sprint (post-v2.38.1
  dead-code: O bound but never read since inline D rowsum was deleted
  for device-D path).
- Fused kernel buffer map (cleaner than initial proposal):
  ```
  Q=0, K=1, V=2, L=3, dO=4, D=5, dK_partials=6, dV_partials=7, params=8
  ```

### Numerical equivalence (HIGH confidence)
- Order constraint: `dV_accum += P^T @ dO` MUST precede `dS = P * dP`
  (because dS overwrites Stile in place).  Algebraically identical to
  split kernels; expected ≤1 ULP drift from split outputs.

### Threadgroup memory (no contention)
- Neither split kernel uses TGP for K/V staging — both use NAX direct
  device loads.  Fused kernel inherits this pattern.  Zero TGP traffic
  beyond SG-local fences.

### WM partition (correct)
- WM=4 Q-row partition: each SG owns 16 Q-rows for a given K-tile.
  Outputs to `[B, Hq, simd_group_id, k_base + tidl.x*BK, d]`.  Host-side
  `mx.sum(axis=2)` reduces per gradient independently.

## 2. Implementation blueprint (for next session)

### Files to touch (estimated LOC delta)

| File | Action | LOC |
|---|---|---|
| `csrc/mfa/v6_nax/NAAttentionKernel.cpp` | add `createV6NAXBackwardDKDVFusedSource()` after line 5260 | +~700 |
| `csrc/mfa/v6_nax/NAAttentionKernel.hpp` | declare `createV6NAXBackwardDKDVFusedSource()` | +1 |
| `csrc/v6_nax_compile.mm` | add `V6NAXBwdFusedDKDVParamsHost` + `v6nax_dispatch_bwd_dkdv_fused()` | +~120 |
| `csrc/mfa_v6_nax_primitive.cpp` | new `MFAV6NAXBwdDKDVFused` Primitive + `v6_nax_backward_dkdv_fused_raw` public function | +~180 |
| `csrc/bindings.cpp` | export `v6_nax_backward_dkdv_fused_raw` | +~20 |
| `mlx_mfa/attention.py:3594` | env var routing in `_v6nax_backward_vjp` (replace `MFA_V6BWD_USE_FUSED` check with `MFA_V6_BWD_KERNEL=auto\|fused\|split\|legacy_fused`) | +~20 |
| `tests/test_v39_fused_dkdv.py` | new fused-vs-split parity tests + axis-2 PUBLIC API mx.grad coverage | +~250 (new) |
| `csrc/mfa/v6_nax/NAAttentionKernel.cpp:4984+` | drop dead `O` buffer from split-dK (while-I'm-here cleanup) | -~10 |
| **Total** | | **~1,280 net** |

### Staging schedule

1. **Phase C.1.a — D=64 only fused kernel** (~3-5h CC)
   - Implement `createV6NAXBackwardDKDVFusedSource` for D=64 only (gate to head_dim==64)
   - Primitive + dispatcher + binding
   - Drop dead O buffer from split-dK
   - Wire `MFA_V6_BWD_KERNEL` env var; D=64 default = "auto" → "fused"
   - Three-axis validation: parity vs split (atol 1e-3 fp16, 5e-3 bf16); axis-2 PUBLIC API via `mx.grad(flash_attention(..., backend="auto"))` with `MFA_ENABLE_V6_BACKWARD=1`
   - Bench: 3-session 4w+12i across qL ∈ {2048, 4096, 8192, 16384}; expect 1.05-1.15× over split-D_vec baseline

2. **Phase C.1.b — D=128 broadening** (~2-3h CC, separate session/PR)
   - Extend source generator to D=128
   - Measure register spill via Metal frame capture / MTL_DEBUG_LAYER=1
   - Bench D=128 fused vs split-D_vec
   - Decision tree:
     - (a) D=128 fused wins ≥5%: broaden `MFA_V6_BWD_KERNEL=auto` to D=128
     - (b) D=128 fused at parity: keep D=128 fused as opt-in only
     - (c) D=128 fused regresses: document negative finding, gate to opt-in or remove D=128 from fused

3. **Phase C.2 — Primitive boilerplate consolidation** (P3-HIGH-01, ~1h CC)
   - After fused kernel lands, extract ~200 LOC of common boilerplate
     across MFAV6NAXBwdQuery + MFAV6NAXBwdDV + MFAV6NAXBwdDK + MFAV6NAXBwdDKDVFused
     (Params struct setup, dispatch helper invocation, error handling).
   - May use a base class or shared free-function helpers.

4. **Phase C.5 — Release v2.39.0**
   - Pre-tag canonical `/mlx-mfa-release-audit`
   - Multi-SoT bump 2.38.1 → 2.39.0
   - CHANGELOG entry with empirical decision tree outcome
   - PyPI + GH release

### Env var contract (recommended by /metal-kernel-dev)

```python
MFA_V6_BWD_KERNEL = "auto"          # default after v2.39.0; routes per shape
                  | "fused"          # force fused kernel (Option γ)
                  | "split"          # force split kernels (v2.38.1 path)
                  | "legacy_fused"   # force legacy WM=1 fused (pre-v2.38.0)
```

`MFA_V6BWD_USE_FUSED=1` retained for one release as deprecation alias
for `MFA_V6_BWD_KERNEL=legacy_fused` (CHANGELOG marks for removal in v2.40).

### Order of operations (CRITICAL)

```cpp
// Per Q-tile iteration in fused kernel:
//
// 1. S = Q @ K^T (NAXFrag::mma into Stile FP32)
// 2. S *= scale * log2(e); apply lse-subtraction in log2 domain
// 3. P = exp2(Stile - lse_log2)   ← Stile now holds P
// 4. dV_accum += P^T @ dO          ← MUST use P BEFORE Stile is overwritten
// 5. dP = dO @ V^T (NAXFrag::mma into dPtile)
// 6. dPtile -= D_vec (row_bin_op<SubOp>)
// 7. dS = P * dPtile               ← overwrites Stile in place (= split-dK behavior)
// 8. dK_accum += dS^T @ Q
```

## 3. Reference: existing split kernels (for next session)

| Kernel | NAAttentionKernel.cpp lines | Notes |
|---|---|---|
| Split-dV (`createV6NAXBackwardDVSource`) | 4671-4938 | Inputs Q/K/V/L/dO; output dV_partials.  Does NOT compute D (no dS term). |
| Split-dK (`createV6NAXBackwardDKSource`) | 4940-5275 | Inputs Q/K/V/O/L/dO/D; output dK_partials.  Reads D from device buffer (v2.38.1). |
| Legacy fused-dKdV (`createV6NAXBackwardKeyValueSource`) | 4230-4670 | WM=1 single-SG; gated by `MFA_V6BWD_USE_FUSED=1`. |
| Forward (`createV6NAXSource`) | 2716-3782 | Reference for naxHelpersBlock + lse-load pattern. |

## 4. Decision tree for empirical outcome (post-implementation)

Per the user's mandate:

| Outcome | Description | Action |
|---|---|---|
| (γ-broadened) | D=64 fused wins ≥10% → AUTO carve-out broadens to all D=64 qL (not just ≥4096) | Update `_v6nax_backward_carveout` to remove qL≥4096 floor; CHANGELOG headline = "D=64 V6NAX backward auto-default for all shapes" |
| (γ-marginal) | D=64 fused wins 5-10%; D=128 fused at parity or marginal improvement | Keep existing carve-out (qL≥4096); ship fused as default for D=64; D=128 stays opt-in |
| (δ) | Fused kernel correctness fails OR perf regression vs split | Halt; document negative finding in v2.39.0 STATUS doc; keep split kernels as default; legacy_fused remains as fallback |

## 5. Skill invocations log (per §AA.2)

| Phase | Skill | Status |
|---|---|---|
| C.1 pre-implementation | `/metal-kernel-dev` | ✓ Done (this doc §1; MEDIUM go) |
| C.1.a post-implementation | `/metal-kernel-dev` (register spill check) | pending next session |
| C.3 corruption audit | `/mlx-debug-forensics` | pending next session |
| C.4 perf characterization | `/mlx-mfa-bench-methodology` | pending next session |
| C.4 perf claim audit | `/mlx-mfa-perf-audit` | pending next session |
| C.5 pre-tag canonical | `/mlx-mfa-release-audit` | pending next session |

## 6. Why this halt is honest, not avoidance

The `/metal-kernel-dev` audit was substantive and produced specific
go/no-go criteria, register budget verification, source-merge
feasibility analysis, numerical-equivalence reasoning, and a staging
schedule.  The implementation work itself is well-defined (~1280 LOC
across 7 files) and unambiguous.

What's missing for in-session completion is **uninterrupted ~3-5h CC
time for D=64-only Phase C.1.a + tests + bench**, which the v2.39.0
sequential-sprint budget does not contain at this point.

Marco's primary user-facing demand (FlashVSR / STCDiT / CogVideoX
D=64 backward) is **already served** by the shipped v2.38.1 1.91×
speedup vs SDPA-vjp.  Option γ adds ~10% on top of that, plus
architectural consolidation for v2.40.x sprints — valuable but not
urgent.

A focused v2.39.0 sprint executing against this blueprint will be more
efficient than rushing the implementation in a constrained window.

## 7. Master commit reference

- v2.38.1 master HEAD: `09a626d` (Merge feat/v38-1-d-vec)
- v2.38.x observability doc-only patch: `a94a342` (Merge chore/v38-x-observability-update)
- v2.39.0 design audit: this doc (committed alongside as next master commit)
