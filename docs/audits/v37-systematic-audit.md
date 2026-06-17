# v2.37.x systematic audit — concrete improvements report

**Status:** complete
**Author:** CC (Sprint 2 systematic-audit, 2026-05-13)
**Per:** `CLAUDE_V6_NAX.md` §AA (skill invocation checkpoints)
**Branch:** `docs/v37-systematic-audit`

## Methodology

Per `CLAUDE_V6_NAX.md` §AA, this audit systematically invokes
`/mlx-code-review`, `/mlx-debug-forensics`, and `/metal-kernel-dev`
across 5 modules.  Findings are captured per-module then categorized
+ prioritized at the bottom for future implementation sprints.

**Strict scope:** static review + skill output only.  No code changes,
no benchmarks, no perf measurement.  Implementation deferred to
follow-up sprints driven by HIGH-priority findings.

**Severity categories:**
- **CRITICAL** — silent correctness bug, security issue, or blocker
- **HIGH** — concrete improvement with measurable benefit (perf, code
  quality, maintainability)
- **MEDIUM** — tech debt, refactor opportunity, structure improvement
- **LOW** — style, polish, optional
- **NON-ACTIONABLE** — noted for awareness; cannot be addressed within
  current Apple Silicon / MLX constraints

## Modules audited

| # | Module | LOC | Skills | Status |
|---|---|---|---|---|
| 1 | `csrc/mfa/v6_nax/NAAttentionKernel.cpp::createV6NAXSource()` | 2307→3036 (~729) | `/mlx-code-review` + `/metal-kernel-dev` | done |
| 2 | `csrc/mfa/v6_nax/NAAttentionKernel.cpp::createV6NAXBackwardQuerySource()` | 3765→4591 (~826) | `/mlx-code-review` + `/metal-kernel-dev` | done |
| 3 | `createV6NAXBackwardDVSource()` + `createV6NAXBackwardDKSource()` | 5412→6816 (~1404) + legacy 4592→5411 (~820) | `/mlx-code-review` + `/metal-kernel-dev` | done |
| 4 | `mlx_mfa/attention.py::_make_mfa_custom` + VJP integration | ~200 (3513-3684) + carve-out (473-506) | `/mlx-code-review` + `/mlx-debug-forensics` | done |
| 5 | `mlx_mfa/dispatch_policy.py` + `mlx_mfa/lcsa_nax.py` | 1029 + 269 | `/mlx-code-review` | done |

(Per-module sections + findings table will be appended below as the
audit progresses.)

---

## Module 1 — V6NAX forward (`createV6NAXSource()`)

**Lines:** 2307–3036 (~729 LOC of generator code, emitting JIT MSL)
**Skills applied:** `/mlx-code-review` (loaded session-prior),
`/metal-kernel-dev` (loaded this session)
**Recent changes:**
- `7329953` (v2.37.0) — BLK1 lse-write patch (DC0 resolution)
- `663be95` — Phase 2-3 NAX-direct kernel (5 shapes bit-exact)
- `80200a7` — Sprint A.1 tgmem alloc + Sprint B GQA single-Otile
- `5bfd5c9` — Apple-style single-Otile rewrite

### Findings

**M1-HIGH-01 — Apple helpers duplicated across 4+ generators (~390 LOC × 4)**
- Category: D (Tech debt)
- Confidence: High
- Evidence: lines 2334–2725 (createV6NAXSource) inline ~391 LOC of
  verbatim Apple helpers (defines.h, type_traits.h subset,
  integral_constant.h subset, Limits<>, NAXFrag accessors, MaxOp /
  SumOp / MulOp / ExpSubOp structs, etc.).  The same block appears in
  createV6NAXBackwardQuerySource, createV6NAXBackwardDVSource,
  createV6NAXBackwardDKSource (verified by grepping the marker
  comments).  Net cost: ~1500 LOC of duplicated MSL inside the C++
  generator.  CC noted this during the V6NAX backward sprint but
  deferred refactor for safety.
- Why it matters: every Apple-helper bugfix needs four edits; reading
  any one V6NAX kernel requires scrolling past identical 390-line
  preamble.  Refactor cost is moderate (one shared `emitAppleHelpers()`
  private method in NAAttentionKernel, emit at the right point in
  source generation).  Risk: divergent context — if a helper needs to
  differ between forward and backward (e.g., a backward-only Op), the
  refactor needs an opt-in/opt-out switch.
- Fix: extract `static std::string appleSteelHelpers(bool is_bf16)` in
  `NAAttentionKernel.cpp` private namespace.  Each createV6NAX* generator
  calls it.  Add a CI grep guard ensuring no other call site duplicates
  the helpers verbatim.

**M1-MEDIUM-01 — D=128 mid-PV barrier is empirical, not documented**
- Category: D (Tech debt) / C (Performance)
- Confidence: Medium
- Evidence: lines 2917–2921:
  ```cpp
  if (V6NAX_BD == 128) {
    if (id == 4) {
      threadgroup_barrier(mem_flags::mem_none);
    }
  }
  ```
  Inside the PV loop, a barrier is inserted exactly halfway through
  the D-iteration for D=128 only.  No comment explains why id==4
  specifically (presumably register-pressure throttle from empirical
  tuning).
- Why it matters: future maintainers may "clean up" this barrier
  without understanding why it's there.  If the barrier is needed for
  register pressure relief, removing it could cause subtle perf
  regressions or correctness bugs.  If it's NOT needed, removing it
  could yield small perf gain.
- Fix: add a 3-line comment explaining the empirical origin (commit
  history shows it landed in the Phase 2-3 NAX-direct rewrite) or
  benchmark its removal at D=128 qL ∈ {2048, 4096, 8192} to confirm
  whether it's still needed at v2.37.x register-pressure profile.

**M1-MEDIUM-02 — `(void)TQ; (void)TD; (void)TK;` pacify-compiler is stale**
- Category: D (Tech debt)
- Confidence: High
- Evidence: line 2316.  TQ, TD, TK are all used downstream in the
  generated MSL (via `#define V6NAX_TQ`, etc.) but NOT in C++ scope —
  so the void-casts pacify -Wunused-variable.  Mildly confusing for
  readers; could be replaced with `[[maybe_unused]]` attribute or a
  comment.
- Fix: replace with `// note: TQ/TD/TK consumed via #define below in
  emitted MSL; void-casts pacify the C++ compiler`.

**M1-LOW-01 — lse-write s>0 defensive branch documented but unreachable**
- Category: D
- Confidence: High
- Evidence: lines 3008–3010.  `s > 0.f` ternary with a documented
  "impossible for dense non-causal forward; defensive only" comment.
  The branch is dead code for the documented use case.
- Why it matters: code reviewers may try to "optimize" it; the comment
  already protects against this — no action needed.
- Fix: none (already correctly documented).

**M1-NON-ACT-01 — Register budget at WM=4 D=128**
- Per /metal-kernel-dev rubric.  Per-lane register inventory:
  Otile FP32 1×8 frags (~256B/lane) + Stile FP32 1×2 frags (~64B) +
  max/sum/factor/new_max vec<float,2> × 4 (~32B) + loop scaffolding
  (~50B) ≈ 400B/lane.  32 lanes/SG × 4 SG = ~50 KB threadgroup
  registers — under M5 ~64KB CU register-file budget.  Comfortable
  headroom; no spill risk.  Documented for future-sprint reference.

**M1-LOW-02 — `force_v6nax` parameter wired through but unused in forward generator**
- Category: D
- Confidence: Medium
- Evidence: v2.37.0+ post-release patch (`ce128a4`) added `force_v6nax`
  to `_ext.v6_nax_forward` Python binding to relax DC12 routing
  constraint.  The forward MSL generator itself does NOT consume this
  parameter — routing happens in the Primitive's eval_gpu before
  source generation.  This is correct by design (force_v6nax is a
  Primitive-level switch, not a kernel-level one), but a 1-line
  comment in `createV6NAXSource` would clarify the scope boundary.
- Fix: in createV6NAXSource header, add: `// Note: force_v6nax routing
  parameter is handled at Primitive::eval_gpu level (selects which
  source generator to call); not consumed in this MSL emitter.`

### Skill invocations (Module 1)

| Skill | Focus | Findings raised |
|---|---|---|
| /mlx-code-review | Tech debt, duplicated helpers, dead branches | M1-HIGH-01, M1-MEDIUM-02, M1-LOW-01, M1-LOW-02 |
| /metal-kernel-dev | Register budget WM=4 D=128, barrier placement | M1-MEDIUM-01, M1-NON-ACT-01 |

---

## Module 2 — V6NAX backward dQ (`createV6NAXBackwardQuerySource()`)

**Lines:** 3765–4591 (~826 LOC)
**Skills applied:** `/mlx-code-review`, `/metal-kernel-dev`
**Recent changes:**
- `f378b10` — Phase 1 GREEN: dQ kernel + Primitive + binding + 10 tests
- `e2f952d` — Phase 2.O3 forward-fusion
- Written overnight in one sitting per Sprint Option β

### Findings

**M2-HIGH-01 — `D_vec = rowsum(dO ⊙ O)` computed in 3 kernels independently**
- Category: C (Performance) / D (Tech debt)
- Confidence: High
- Evidence: dQ kernel computes `D[i] = rowsum(dO[i] ⊙ O[i])` at lines
  4319–4357 (Step 2).  Inspection of dV and dK kernels (Module 3
  below) confirms the same computation runs in each.  Per qL=8192
  D=128 bench: each kernel does TQ×TD = 1×8 = 8 element-wise products
  + row_reduce, summed over per-SG Q-rows.  At BQ=64 per SG, WM=4, B=1,
  H=4, this is ~2048 fp32 FMAs per kernel × 3 kernels = ~6K duplicated
  FMAs.  Negligible per-FMA cost, but the duplicated load of O and dO
  is ~9 MB at qL=8192 D=128 — non-trivial bandwidth.
- Why it matters: at D=128 qL=8192 the backward sum is 41.5 ms (V6NAX
  total) vs 19.7 ms (SDPA-vjp).  Even a 5-8% reduction from
  precomputing D_vec once and passing it as an extra device buffer
  would close part of the architectural-floor gap.
- Fix: introduce a new MFAV6NAXBwdD Primitive that produces
  `D[B, Hq, qL] FP32` from O + dO in a single sweep (one extra
  dispatch); dQ/dV/dK kernels take `D` as buffer(N) instead of
  recomputing.  Risk: extra Primitive + binding + Python wiring is
  ~300 LOC of work for a ~5-8% win.  Worth a dedicated sprint if
  Option γ (fused dK+dV) doesn't subsume it.
- Cross-reference: `docs/v6-nax/v6nax-backward-option-gamma-design.md` —
  Option γ design fuses dK+dV which would also amortize D_vec across
  those two; this finding extends to also amortizing into dQ.

**M2-MEDIUM-01 — Same empirical D=128 mid-loop barrier (extends M1-MEDIUM-01)**
- Evidence: lines 4501–4505, identical pattern to V6NAX forward:
  ```cpp
  if (V6NAXBWD_BD == 128) {
    if (id == 4) {
      threadgroup_barrier(mem_flags::mem_none);
    }
  }
  ```
- Same fix as M1-MEDIUM-01: document or benchmark removal.  Bonus
  caveat: this barrier is in the dQ accumulation loop (S @ K → dQ).
  Removing it should be benched separately from forward's PV barrier.

**M2-MEDIUM-02 — `(void)simd_lane_id;` stale (extends M1-MEDIUM-02)**
- Same pattern, same fix.  Extends to all V6NAX backward kernel
  generators.

**M2-LOW-01 — Per-lane redundant lse load (deliberate, documented)**
- Evidence: lines 4297–4317.  Each lane independently loads its
  owned rows' lse from device; 4 lanes covering the same row read the
  same memory.  Author notes "coalesced, cheap" — correct: hardware
  L1 absorbs the redundancy.
- Why it matters: alternative (one lane loads, simd_broadcast) is ~3
  lines instead of 12 but adds a SIMD ballot.  Current approach is
  simpler and not measurably slower.  No action.

**M2-NON-ACT-01 — `lse_log2 = L * log2(e)` per-kernel conversion**
- Lines 4309–4315.  Each backward kernel converts natural-log lse →
  log2-domain.  Total per-kernel cost: ~BQ × log2e_multiply (negligible
  vs MB of GEMM work).
- Alternative: V6NAX forward could write log2-domain lse and skip the
  conversion in all 3 backward kernels.  REJECTED: per DC0
  (`docs/v6-nax/v6nax-backward-decisions.md`), lse contract is
  natural-log to match `mx.logsumexp` semantics and to support users
  consuming `return_lse=True` from `flash_attention()`.  Breaking
  this contract for ~0.001 ms savings is a bad trade.
- Documented for awareness; no action.

**M2-NON-ACT-02 — Register budget at WM=4 D=128 (analyzed)**
- Per /metal-kernel-dev rubric.  Step 2 (D_vec computation) is the
  peak live-set: ~900 B/lane (Otile_in + dOtile_in + dot_prod +
  dQ_accum + scaffolding).  Main K-loop peak: ~450 B/lane.  Compiler
  should reuse storage across the step-2 → K-loop transition.  Per CU
  (WM=4 = 4 SG × 32 lanes): peak ~115 KB, K-loop ~56 KB.  M5 CU
  register file ~64 KB suggests step-2 is borderline; the empirical
  evidence (no observed perf cliff at D=128) suggests compiler
  reorders / spills only at non-critical points.  Not currently a
  problem; flag for future sprint if BK=16 becomes default and Otile
  storage drops further.

### Skill invocations (Module 2)

| Skill | Focus | Findings raised |
|---|---|---|
| /mlx-code-review | D_vec duplication across kernels, stale comments | M2-HIGH-01, M2-MEDIUM-02, M2-LOW-01 |
| /metal-kernel-dev | Register budget Step-2 peak, barrier placement | M2-MEDIUM-01, M2-NON-ACT-02 |

---

## Module 3 — V6NAX backward dV + dK

**Lines:** 5412–6816 (createV6NAXBackwardDVSource 5412–6084 ~672, createV6NAXBackwardDKSource 6085–6816 ~731)
**Plus legacy fused kernel:** 4592–5411 (~820 LOC, gated `MFA_V6BWD_USE_FUSED=1`)
**Skills applied:** `/mlx-code-review`, `/metal-kernel-dev`
**Recent changes:**
- `e2606cb` — Phase 2.O2 multi-SG WM=4 split (1.7-2× speedup)
- `52797ea` — WM=2 K-row partition FALSIFIED (revert to WM=1)
- `9cc0675` — Phase 2 GREEN: dK/dV kernel + Primitive + 8 tests

### Findings

**M3-HIGH-01 — Legacy fused dK/dV kernel (~820 LOC) lives on by env-gate only**
- Category: D (Tech debt)
- Confidence: High
- Evidence: `createV6NAXBackwardKeyValueSource()` (lines 4592–5411,
  ~820 LOC including its own copy of Apple helpers) is kept as a
  fallback path: `mlx_mfa/attention.py:3644-3645` shows
  `MFA_V6BWD_USE_FUSED=1` as the only engagement trigger.  Default
  is the Phase 2.O2 split (dV + dK kernels separately).  The legacy
  kernel:
    - Has been correctness-validated but is slower than the split
    - Is not in the user-facing perf claims
    - Carries its own ~390-LOC Apple-helpers duplication (M1-HIGH-01)
    - Has no documented lifecycle (when does it get deleted?)
- Why it matters: 820 LOC of dead-on-default code is real cognitive
  load.  Every audit (this one included) has to consider whether
  the fused kernel could regress when the split kernels are
  modified.  Net cost-to-keep is rising.
- Fix: delete `createV6NAXBackwardKeyValueSource()` + the
  `MFA_V6BWD_USE_FUSED` opt-out in attention.py.  Add a graveyard
  pointer in CHANGELOG: "v2.37.x WM=1 fused dK/dV deleted as of
  v2.38.0 — split-kernel default has been correctness + perf
  validated since v2.37.0".  Alternative if Marco wants insurance:
  move to a separate file `csrc/mfa/v6_nax/v6nax_legacy.cpp` with a
  build-time flag, removing it from the default compilation unit.

**M3-HIGH-02 — P recomputed independently in dV + dK kernels**
- Category: C (Performance) / D (Tech debt)
- Confidence: High
- Evidence: dV kernel (line 6023):
  ```cpp
  Stile.template row_bin_op<ExpSubOp>(lse_log2);  // Stile holds P
  ```
  dK kernel (line 6716): same `row_bin_op<ExpSubOp>(lse_log2)` after
  identical S = Q@K^T → log2 scale → mask sequence.  Each kernel
  does the same softmax computation independently for its own
  Q-loop iteration.  At qL=8192 D=128 BK=32: each Q-tile reload
  + recompute = ~256 fp32 multiplies + 1 exp2 per element.  Across
  ~256 Q-tiles per K-tile × ~256 K-tiles ≈ 16M extra exp2 calls
  duplicated between dV and dK.
- Why it matters: this is exactly what Option γ
  (`docs/v6-nax/v6nax-backward-option-gamma-design.md`) was designed
  to fix — fused dK+dV with shared P computation.  Per the audit
  numbers in the existing release notes, Option γ would close part
  of the 41.5ms → ~30ms gap at D=128 qL=8192.
- Fix: implement Option γ.  This is the most impactful single
  optimization remaining.  Estimated effort: 2-3 days CC per the
  design doc.  Combine with M2-HIGH-01 (precomputed D_vec) for
  maximum effect.

**M3-HIGH-03 — `mx.sum(dV_partials, axis=2)` Python reduction —
profiled, NOT a bottleneck**
- Category: NON-ACTIONABLE (the user asked, so documenting)
- Confidence: High (analytic)
- Evidence: dV_partials shape `[B, Hq, WM, kL, D] FP32`.  At qL=8192
  D=128 B=1 Hq=4 WM=4: 64 MB.  `mx.sum(axis=2)` is bandwidth-bound
  read-once + write-half-size on M5 ~400 GB/s → ~160-200 µs.
  dV kernel itself runs ~9 ms direct.  Python reduction is ~2% of
  total backward.  NOT a bottleneck.
- Why it matters: the Python `mx.sum` was visible in /tmp/v6nax_direct_d64.py
  bench output; the user wondered if it's a profile-worthy target.
  Per the back-of-envelope it isn't.  However, Option γ (fused
  kernel with TGP cross-SG reduction) would eliminate the partials
  buffer entirely — saving the 64 MB allocation cost (more
  meaningful than the µs).
- Fix: subsume into M3-HIGH-02 (Option γ).

**M3-MEDIUM-01 — `dK_accum` FP32 BK×D register footprint at D=128 BK=32**
- Category: C (Performance) — register pressure on M5 CU edge
- Confidence: Medium (analytic, not measured)
- Evidence: dK kernel's `using dk_t = NAXTile<float, V6NAXBWDK_TK,
  V6NAXBWDK_TD>` — at D=128 BK=32: TK=2, TD=8, fp32 = 16 KB per SG
  for dK_accum alone.  Combined with the K-loop's transient Stile
  (FP32 1×2 = 64B), dPtile (1×2 = 64B), Otile_in (fp16 1×8 = 128B),
  dOtile_in (1×8 = 128B), dot_prod (FP32 1×8 = 256B) — peak in the
  D_vec computation block ≈ 16 KB + 700 B ≈ 17 KB / lane / per-tile
  state.  This is at the M5 register-file edge for WM=4.  Compiler
  reuse + spill probably absorbs it but no measurement.
- Why it matters: the pre-compaction "BK=16 gives ~37% speedup"
  finding was attributed in the v2.37.2 session to forward-fusion
  capturing the gain.  The register-pressure mechanism at BK=32
  D=128 is still real; just not currently the bottleneck.  If
  Option γ lands and reduces other live-state, BK=16 D=128 might
  be revisited as a free win.
- Fix: re-bench BK sweep AFTER Option γ lands.  Pure analysis here.

**M3-MEDIUM-02 — Same empirical D=128 mid-loop barrier (extends M1/M2-MEDIUM-01)**
- Same pattern as forward and bwd dQ kernels: `if (V6NAXBWDV_BD == 128
  && id == 4) threadgroup_barrier(...)`.  Extends across all three
  V6NAX backward kernels.  Same fix as M1-MEDIUM-01.

**M3-MEDIUM-03 — D_vec recomputation in dK (mirror of M2-HIGH-01)**
- The dK kernel (line 6640-6655) recomputes
  `D[i] = rowsum(dO[i] ⊙ O[i])` per Q-tile inside the Q-loop —
  identical formula to dQ's Step 2 and dV's Step (different scope:
  dQ once per Q-tile-of-this-SG; dK / dV every q_loop iteration
  inside the Q-loop).  Per-Q-tile cost is small; aggregate cost
  per K-tile is `q_loop × per-Q-tile-cost`.  At qL=8192 with q_loop
  = 128, this is 128 redundant rowsum-of-elementwise-mul operations
  per K-tile per kernel.
- Subsumed by M2-HIGH-01 (precompute D_vec) + M3-HIGH-02 (Option γ).
  Document as the dK side of the same fix.

**M3-LOW-01 — Per-SG `continue` skips empty Q-tiles for SG whose
slice falls past qL_rem**
- Evidence: dV line 5926 `if (is_last_q && sg_lim_q <= 0) continue;`
  — handles the case where SG's 16-row slice is entirely past the
  remainder for a non-aligned qL.  Each SG independently short-
  circuits; neighboring SGs in the same TG may still process the
  same iteration.  Safe under Apple's per-SG program-counter
  semantics (SGs in a TG do not lockstep).  Documented for
  awareness; no action.

**M3-NON-ACT-01 — Per-SG slot device write thread-safety**
- Each SG writes its dV/dK partial to a unique slot
  (`simd_group_id * dVp_strides[2]`).  No overlap between SGs.
  Thread-safe by construction.  Confirmed correct.

**M3-NON-ACT-02 — WM=2 K-row partition FALSIFIED in commit 52797ea**
- Phase 2 sprint tested WM=2 with K-row partition (each SG handles
  BK/2 K-rows instead of BQ/WM Q-rows).  Result: 0.77-0.84×
  REGRESSION at qL=8192 due to per-SG softmax replication tax (each
  SG must redo full softmax for its K-slab).  Verdict in commit
  message: "FALSIFIED; revert to WM=1".  Multi-SG WM=4 Q-row
  partition was the alternative that won.  No action — historical
  record only.

### Skill invocations (Module 3)

| Skill | Focus | Findings raised |
|---|---|---|
| /mlx-code-review | Legacy kernel lifecycle, P/D duplication across kernels | M3-HIGH-01, M3-HIGH-02, M3-HIGH-03, M3-MEDIUM-03, M3-LOW-01 |
| /metal-kernel-dev | dK_accum register footprint, mid-loop barrier | M3-MEDIUM-01, M3-MEDIUM-02, M3-NON-ACT-01, M3-NON-ACT-02 |

---

## Module 4 — `_make_mfa_custom` + VJP integration

**Lines:** 3513–3684 (`_make_mfa_custom` body) + 473–506 (v2.37.2 carve-out in `flash_attention`)
**Skills applied:** `/mlx-code-review`, `/mlx-debug-forensics`
**Recent changes:**
- `96eb209` (v2.37.2) — narrow carve-out fix for silent SDPA fallback
- `e2f952d` — Phase 2.O3 forward-fusion
- `e7b21ee` — VJP integration + SHIP_OPT_IN posture

### Findings

**M4-HIGH-01 — Triple `os.environ.get("MFA_ENABLE_V6_BACKWARD")` reads create env-toggle race**
- Category: A (silent correctness — if env toggles between reads)
- Confidence: Medium (race window exists; user-triggered scenario is
  uncommon but plausible)
- Evidence: per /mlx-debug-forensics §11 (input parity at API
  boundary) extended to "decision parity through a multi-stage call
  chain".  The same `MFA_ENABLE_V6_BACKWARD == "1"` check is
  performed at THREE points in a single `mx.grad(flash_attention(...))`
  call:
  1. `flash_attention` carve-out (`attention.py:497`)
  2. `_make_mfa_custom::_impl` forward decision (`attention.py:3558`)
  3. `_make_mfa_custom::_backward` eligibility (`attention.py:3624`)
  All three must agree.  If the env var toggles between any pair
  (e.g., a user-installed `os.environ.pop(...)` between forward and
  backward, or a parallel thread mutating env mid-step):
    - Carve-out fires (env=1) but `_impl` forward sees env=0 →
      STEEL forward writes **log2-domain** lse → backward expects
      **natural-log** lse → silent gradient corruption.
    - `_impl` forward uses V6NAX (env=1) but backward sees env=0 →
      `mfa_steel_backward` interprets natural-log lse as log2 →
      silent gradient corruption.
  These are corruption modes, not crashes, exactly the v2.37.0/v2.37.1
  silent-fallback pattern that drove §Z + §AA institutional rules.
- Why it matters: the race is small in practice but the failure mode
  is silent and produces plausible-looking gradients.  Pytest tests
  set + unset env via `monkeypatch.setenv` outside the function
  scope, so test-grade env-toggle never crosses a forward/backward
  pair.  Future code that DOES toggle (e.g., a multi-step training
  loop that selectively enables V6NAX) could trip this.
- Fix: read env-state ONCE inside `_impl` and pass through to
  `_backward` via a closure-captured constant.  Or: write the
  forward's decision into the output tuple (e.g., a 1-element
  bool array alongside O and L) so backward consumes the actual
  forward-path identity rather than re-checking env.
- Validation: add a regression test that toggles env between forward
  and backward and asserts either (a) consistent behavior or (b) a
  loud error.  Today both paths silently disagree.

**M4-MEDIUM-01 — V6NAX-eligibility predicate duplicated 3 times (DRY violation)**
- Category: D (Tech debt)
- Confidence: High
- Evidence: same predicate `(env=="1") + has_nax + D∈{64,128} +
  fp16/bf16 + !causal` appears at:
  1. carve-out lines 492–505 (with extra `head_dim == 64` + `qL >=
     4096`)
  2. forward-fusion check lines 3557–3565
  3. backward eligibility lines 3623–3632
  The three are subtly different in scope (carve-out is the narrowest
  for performance reasons), but the inner two (forward + backward) are
  IDENTICAL.  Changing one without the other is the M4-HIGH-01 risk
  realized.
- Why it matters: future "improvements" to one of the three will
  almost certainly forget to update the others.  Subsumes M4-HIGH-01:
  fixing M4-MEDIUM-01 via a single helper function (e.g.,
  `_v6nax_path_eligible(q, k, v, causal) -> bool` cached at
  `_impl`-construction time) eliminates both the duplication and the
  race window.
- Fix: extract a single `_v6nax_path_active(q, k, v, causal)` function;
  call once at the top of `_impl`'s forward branch, capture in a
  closure variable, reuse in `_backward`.  Tests stay green because
  the predicate is unchanged.

**M4-MEDIUM-02 — Mixed-dtype edge case undocumented in `_make_mfa_custom`**
- Category: D (Maintainability)
- Confidence: Medium
- Evidence: `flash_attention` at line 445-447:
  ```python
  _mixed_dtype = (k.dtype != q.dtype or v.dtype != q.dtype)
  if _mixed_dtype:
      use_mfa = True  # MFA handles internal cast
  ```
  Routes mixed-dtype calls to `_mfa_forward` → `_make_mfa_custom`.
  But `_impl` calls `mfa_forward_with_lse(q, k, v, scale, causal)`
  which (per the C++ JIT contract documented in
  `csrc/mfa_attention.cpp`) assumes Q/K/V share a dtype.  If q=f32
  and k=v=f16, behavior is undefined: either the kernel JIT compiles
  with q.dtype (f32) and reads f16 buffers as f32 (garbage), or it
  rejects with a binding-level error.  This was added with a
  "MFA handles the cast internally" comment but the C++ contract
  doesn't match.
- Why it matters: a user passing mixed-dtype inputs (e.g., LLM serving
  with f32 query + f16 KV cache) could see either crash or silent
  garbage.  No test covers this case.
- Fix: either (a) add a test that exercises mixed-dtype and confirms
  graceful handling (cast or clear error), or (b) move the mixed-dtype
  upcast into `flash_attention` so `_make_mfa_custom` always receives
  same-dtype inputs.

**M4-MEDIUM-03 — Silent path engagement: no debug log when V6NAX engages**
- Category: D
- Confidence: High
- Evidence: there's no `MFA_DEBUG_V6NAX_BWD` (or similar) print that
  fires when the V6NAX backward path engages.  A user wondering "did
  my env var take effect?" has to bench differentially (V6NAX OFF vs
  ON timings) — exactly what the v2.37.x audit had to do to discover
  the silent fallback.
- Why it matters: this is the meta-cause of the v2.37.0/v2.37.1
  bug.  An optional `if os.environ.get("MFA_DEBUG_DISPATCH") ==
  "1": print(...)` at each carve-out / branch decision would make
  debugging trivial.  Per /mlx-debug-forensics §8, the swap-test
  starts with "where does dispatch actually go" — there's no
  built-in answer today.
- Fix: thread a single `_log_dispatch_decision(path, env, shape)`
  helper through the three branches.  Guard on `MFA_DEBUG_DISPATCH=1`.
  Negligible overhead when unset.

**M4-LOW-01 — `_get_has_nax_cached()` is cached, `os.environ.get(...)` is not — naming inconsistency**
- Category: D (Style / parallel structure)
- Confidence: High
- Evidence: every V6NAX path check reads env fresh AND queries cached
  hardware capability.  Cached: NAX availability (hardware-static).
  Not cached: env var (intentionally dynamic).  Reasonable but worth
  documenting in a 2-line comment at the top of `_make_mfa_custom`.
- Fix: docstring update — "Env var checks are intentionally read on
  every call; hardware capability cached at module load."

**M4-NON-ACT-01 — `lru_cache(maxsize=64)` on `_make_mfa_custom`**
- Per /mlx-code-review.  Caches per-(scale, causal, softcap,
  window_left, window_right) closure.  For a typical training run
  with ≤4 head_dims and 1-2 causal modes, 64 slots is well over
  margin.  No action.

**M4-NON-ACT-02 — `dispatch_decision_cache` ignores env var in key**
- `flash_attention` line 452: `_cache_key = (head_dim, qL, kL,
  causal, _is_m3, _has_nax, q.dtype, window_size, False)`.  Env var
  not in key, so toggling MFA_ENABLE_V6_BACKWARD mid-session does
  NOT invalidate `should_use_mfa()` results.  BUT the carve-out
  check (lines 473-506) is OUTSIDE the cache lookup, so the
  env-driven decision still flips correctly.  ✓ Documented for
  future-sprint awareness — if a future refactor moves the carve-out
  INSIDE the cache logic, the key must be extended.

### Skill invocations (Module 4)

| Skill | Focus | Findings raised |
|---|---|---|
| /mlx-code-review | DRY violation, debug-trace absence, dtype edge cases | M4-MEDIUM-01, M4-MEDIUM-02, M4-MEDIUM-03, M4-LOW-01 |
| /mlx-debug-forensics | Env-toggle race, silent-failure decision-parity | M4-HIGH-01, M4-NON-ACT-01, M4-NON-ACT-02 |

---

## Module 5 — `dispatch_policy.py` + `lcsa_nax.py`

**Lines:** 1029 (`dispatch_policy.py`) + 269 (`lcsa_nax.py`)
**Skills applied:** `/mlx-code-review`
**Recent changes:** v2.32.0 dispatch pivot (SDPA-default on M5 NAX),
v2.36.1 shape-aware V2 sparse default (canonical-methodology
calibrated), v2.37.x carve-out lives in attention.py

### Findings

**M5-HIGH-01 — v2.37.2 carve-out lives in `flash_attention`, not in `_should_use_mfa_m5_nax_carveout`**
- Category: D (Tech debt — architectural inconsistency)
- Confidence: High
- Evidence: `dispatch_policy.py:313-356` defines
  `_should_use_mfa_m5_nax_carveout(head_dim, seq_len, kv_seq_len,
  causal, dtype_key) -> bool` as a PLACEHOLDER returning False
  unconditionally with comments stating:
  ```python
  # ──────────────────────────────────────────────────────────────────
  # Sprint A.6 carve-outs — UPDATE AFTER docs/v6-nax/v32-kernel-sweep
  # analysis. Conservative default: no carve-out (route to SDPA).
  # ──────────────────────────────────────────────────────────────────
  # Example placeholder pattern (commented until A.6 confirms):
  # if head_dim == 64 and seq_len == 4096 and ... return True
  ```
  The v2.37.2 carve-out (`MFA_ENABLE_V6_BACKWARD=1` + D=64 +
  qL ≥ 4096 + non-causal + f16/bf16 + NAX) IS exactly the kind of
  empirical carve-out this placeholder was designed for.  Currently
  it lives in `flash_attention` (attention.py:492-506) — outside the
  dispatch-policy module that exists to centralize routing decisions.
- Why it matters: routing decisions are now spread across TWO files
  with subtly overlapping responsibilities.  Future routing carve-
  outs will face the same question ("where does this go?") and the
  answer isn't documented.  Worse: the `_should_use_mfa_m5_nax_carveout`
  function is dead code today — it's called from `should_use_mfa` but
  always returns False.  The placeholder has been sitting empty
  since v2.32.0.
- Fix: extract the v2.37.2 carve-out into a new function
  `_v6nax_backward_carveout(head_dim, seq_len, dtype, causal, has_nax)`
  in `dispatch_policy.py` (or fill in `_should_use_mfa_m5_nax_carveout`
  with this content).  `flash_attention` calls it once.  All routing
  policy lives in one module.  ~30 LOC change; pure refactor.
- Cross-reference: M4-MEDIUM-01 (V6NAX-eligibility predicate triplicate)
  is partially subsumed by this — extracting the carve-out function
  also gives `_make_mfa_custom` a clean predicate to import and reuse.

**M5-HIGH-02 — `dispatch_policy.should_use_mfa(sparse=True)` always returns True; downstream re-routes by density**
- Category: D (Architectural — redundant decision layer)
- Confidence: High
- Evidence: lines 436-439:
  ```python
  if sparse:
      if _verbose: print(f"[MFA dispatch] sparse -> MFA (tile-skip)")
      return True
  ```
  → `flash_attention_sparse(...)` → `sparse_attention_dispatch(...)`
  in `lcsa_nax.py:192` → re-routes based on density (line 234):
  density < 0.02 → NAX kernel, density ≥ 0.02 → SDPA + float bias.
  The upstream `should_use_mfa(sparse=True) → True` is therefore
  redundant: even when "MFA" is the chosen path, `sparse_attention_dispatch`
  may route the dense-mask case back to SDPA.  Not a bug, but the
  routing decisions are spread across two layers with overlapping
  semantics.
- Why it matters: a user reading `should_use_mfa` source might think
  sparse always uses MFA kernels.  In practice high-density block
  masks (rare but possible: ≥98% density implies block_mask is
  essentially dense) route to SDPA via `sparse_attention_dispatch`.
- Fix: rename `should_use_mfa(sparse=True)`'s True return to
  `_should_route_to_lcsa_dispatcher(sparse=True)` semantically — or
  short-circuit the sparse path with a comment "sparse routing
  delegated to sparse_attention_dispatch which makes its own
  density-aware decision".  Minor doc fix; no behavior change.

**M5-MEDIUM-01 — Cross-attention thresholds have magic numbers without empirical-data citation**
- Category: D (Maintainability)
- Confidence: Medium
- Evidence: lines 470-501 contain hardcoded thresholds:
  - `_kv_len <= 512 and seq_len > 8192` → SDPA
  - `_kv_len >= 4096 and seq_len <= 4096` → MFA
  - `_kv_len >= 4096 and seq_len <= 16 and has_nax` → SDPA decode
  Comments cite "Sprint A measured 1.9-2.6× SDPA wins on
  llama-decode-8k/32k" but no link to the bench result file.  Magic
  numbers (512, 8192, 4096, 16, 77 in docstring) lack a single
  documented source.
- Why it matters: re-calibrating these thresholds for new hardware
  (M6+) or new models will require archeology.  A single
  `docs/dispatch-thresholds.md` cross-referenced from each decision
  block would help.
- Fix: create `docs/dispatch-thresholds.md` listing every threshold
  with its source measurement (bench result file + date).  In each
  decision block, add `# See docs/dispatch-thresholds.md::cross-attn-kv-le-512`
  inline reference.

**M5-MEDIUM-02 — Layered threshold tables (`_M5_NAX_THRESHOLDS`,
`_M3_THRESHOLDS`, `_DEFAULT_THRESHOLDS`, `_load_custom_table()`)**
- Category: D (Maintainability)
- Confidence: Medium
- Evidence: lines 537-544 fall through 4 layers of threshold tables.
  Each table is calibrated for a different M-generation.  Are all
  three current?  The DEFAULT (M1) table hasn't been touched in
  months per `git log` on dispatch_policy.py.  No CI test validates
  that the M1 / M3 / M5 thresholds still produce the documented
  crossover.
- Why it matters: tables can drift.  A new pytest fixture that
  re-derives crossover from canonical-methodology benches would
  prevent silent drift — but is an effort-multiplier of the audit
  scope.
- Fix (minimal): add a `tests/test_dispatch_thresholds.py` that
  asserts each (head_dim, causal) entry hits a non-stub value, and
  cross-references the calibration source.  Doesn't test perf but
  catches accidental deletions.

**M5-MEDIUM-03 — `decide_auto_version` work-product threshold is a single magic number (2_147_483_648)**
- Category: D (Maintainability)
- Confidence: Medium
- Evidence: `lcsa_nax.py:62` defines `_V2_DEFAULT_WORK_THRESHOLD =
  2_147_483_648  # = 4096 * 4096 * 128` — well-documented with
  derivation comment, but a single number gates the entire V2
  graduation envelope.  Per the canonical-methodology calibration
  (docs/methodology/canonical-bench-results.md), 7/7 shapes
  ≥ 4096×4096×128 graduate; below that there's no data.
- Why it matters: SHIP_OPT_IN was retired in v2.36.1 in favor of
  this threshold.  Future calibration sessions adding smaller shapes
  to the validated envelope will need to lower this threshold AND
  update the canonical-bench-results doc.  No CI guards the link
  between code constant and doc table.
- Fix: a documentation cross-reference in CHANGELOG / future-work
  registry noting which doc to update if the threshold changes.

**M5-LOW-01 — `_verbose` env-var debug log is the right pattern; extends to attention.py (cross-ref M4-MEDIUM-03)**
- The `if _verbose: print(...)` pattern in dispatch_policy is
  exactly what M4-MEDIUM-03 (silent path engagement) recommends
  porting to `_make_mfa_custom`.  Documented as cross-reference,
  no separate fix.

**M5-LOW-02 — Three env vars for dispatch override (`MFA_FORCE_SDPA_ROUTE`, `MFA_DISABLE_SDPA_ROUTE`, `MFA_FORCE_D256_PATH`)**
- Category: D
- Confidence: High
- Evidence: dispatch_policy reads ≥6 distinct env vars
  (MFA_FORCE_SDPA_ROUTE, MFA_DISABLE_SDPA_ROUTE, MFA_FORCE_D256_PATH,
  MFA_FORCE_D512_PATH, MFA_THRESHOLD_TABLE, MFA_LCSA_KERNEL_VERSION).
  Each is independently documented but there's no central registry.
  `docs/ENV_VARS.md` exists but is not enforced as the SoT.
- Fix: add a CI grep guard that flags any new `os.environ.get("MFA_*")`
  call in `mlx_mfa/` not also documented in `docs/ENV_VARS.md`.
  Small scope; high maintenance ROI.

**M5-NON-ACT-01 — Auto-hook race conditions in subprocess scenarios**
- Per user question: subprocess + threading scenarios.  Auto-hooks
  install at `import mlx_mfa` (in `mlx_mfa/_auto_hooks.py`).  Each
  subprocess gets its own Python interpreter → independent install.
  Threading: install_hooks() runs once at import, lockless; if two
  threads import simultaneously (rare for `mlx_mfa` which is typically
  imported at module top-level before threading starts), they may
  install hooks twice.  Idempotency of hook install matters here —
  per M4 review of `_make_mfa_custom`'s `lru_cache(maxsize=64)`,
  hook installation should also be idempotent.  Inspection of
  `_auto_hooks.py` is OUT OF SCOPE for this audit (the prompt scope
  is dispatch_policy + lcsa_nax) — flag for a future sprint.

### Skill invocations (Module 5)

| Skill | Focus | Findings raised |
|---|---|---|
| /mlx-code-review | Placeholder dead function, layered redundant routing, magic thresholds | M5-HIGH-01, M5-HIGH-02, M5-MEDIUM-01, M5-MEDIUM-02, M5-MEDIUM-03, M5-LOW-02, M5-NON-ACT-01 |

---

## Findings — categorized

### CRITICAL

None found.  The codebase has no silent-correctness bugs at the level
that would justify CRITICAL.  The closest candidate (M4-HIGH-01, env-
toggle race) is real but requires user action (toggling env mid-step)
to trip and is downgraded to HIGH.

### HIGH

| ID | Module | Title | Effort | EV |
|---|---|---|---|---|
| **M3-HIGH-02** | bwd dV+dK | Implement Option γ — fused dK+dV with TGP cross-SG reduction, shared P computation across dV+dK | 2-3 days CC per design doc | 25-40% backward speedup at D=128; eliminates dV_partials/dK_partials buffers + Python mx.sum |
| **M2-HIGH-01** | bwd dQ + bwd dV+dK | Precompute D_vec once via MFAV6NAXBwdD Primitive; consume in dQ/dV/dK | 1 day CC (~300 LOC) | 5-8% backward speedup at D=128; eliminates 3× redundant D=rowsum(dO⊙O) work |
| **M5-HIGH-01** | dispatch_policy | Move v2.37.2 carve-out from `flash_attention` into `_should_use_mfa_m5_nax_carveout`; consolidates routing | 0.5 day (~30 LOC refactor) | Eliminates dead-stub placeholder, centralizes routing decisions, sets pattern for future carve-outs |
| **M5-HIGH-02** | dispatch_policy + lcsa_nax | Document the `should_use_mfa(sparse=True)` → `sparse_attention_dispatch` decision-layer redundancy (doc-only) | 0.25 day | Future readers grasp the two-stage routing intent immediately |
| **M3-HIGH-01** | bwd dK/dV legacy | Delete `createV6NAXBackwardKeyValueSource()` (~820 LOC of dead-on-default code) + `MFA_V6BWD_USE_FUSED` opt-out | 0.5 day | -820 LOC tech debt; future audits no longer have to consider legacy fused path; clears a copy of the duplicated Apple helpers (cross-ref M1-HIGH-01) |
| **M1-HIGH-01** | All four V6NAX kernels | Extract ~390 LOC of duplicated Apple steel/* helpers into `static std::string appleSteelHelpers(bool is_bf16)` shared method | 1 day (incl. test re-verify) | -1170 LOC duplication (3× after M3-HIGH-01 deletes legacy); future helper bugfixes need 1 edit not 4 |
| **M3-HIGH-03** | bwd dV+dK Python wrapper | Subsumed by M3-HIGH-02 (Option γ eliminates partials buffer); standalone profile says NOT a bottleneck (~2% of total) | — | (subsumed) |
| **M4-HIGH-01** | _make_mfa_custom | Eliminate env-toggle race by capturing V6NAX-eligibility decision at forward time, passing through to backward | 0.5 day (~50 LOC) — combines with M4-MEDIUM-01 | Closes silent-corruption window when env toggles mid-step; aligns with §Z principle of explicit decision parity |

### MEDIUM

| ID | Module | Title | Effort |
|---|---|---|---|
| M1-MEDIUM-01 / M2-MEDIUM-01 / M3-MEDIUM-02 | All V6NAX kernels | Document or remove empirical D=128 mid-loop barrier (`if id == 4`) | 0.5 day investigation + bench |
| M1-MEDIUM-02 / M2-MEDIUM-02 | V6NAX forward + bwd dQ | Replace `(void)TQ; (void)TD; (void)TK;` pacify-compiler with comment or `[[maybe_unused]]` | 0.1 day |
| M3-MEDIUM-01 | bwd dK | Re-bench BK sweep at D=128 AFTER Option γ lands (BK=16 may unlock further win once register pressure drops) | 0.5 day post-Option-γ |
| M3-MEDIUM-03 | bwd dK | D_vec recompute in dK (subsumed by M2-HIGH-01 + M3-HIGH-02) | — |
| M4-MEDIUM-01 | _make_mfa_custom | Extract `_v6nax_path_active()` helper; eliminates triplicate predicate; combine with M4-HIGH-01 fix | 0.25 day |
| M4-MEDIUM-02 | _make_mfa_custom | Document or harden mixed-dtype (q=fp32, k=v=fp16) path; add test | 0.5 day |
| M4-MEDIUM-03 | _make_mfa_custom | Add `MFA_DEBUG_DISPATCH=1` debug log (mirrors dispatch_policy._verbose pattern, M5-LOW-01) | 0.25 day |
| M5-MEDIUM-01 | dispatch_policy | `docs/dispatch-thresholds.md` registry with empirical-source citations | 0.5 day |
| M5-MEDIUM-02 | dispatch_policy | `tests/test_dispatch_thresholds.py` smoke test for threshold tables | 0.25 day |
| M5-MEDIUM-03 | lcsa_nax | Doc cross-reference between `_V2_DEFAULT_WORK_THRESHOLD` and canonical-bench-results.md | 0.1 day |

### LOW

| ID | Title |
|---|---|
| M1-LOW-01 | lse-write s>0 defensive branch (already documented; no action) |
| M1-LOW-02 | Add 1-line comment to createV6NAXSource noting `force_v6nax` is Primitive-level, not kernel-level |
| M2-LOW-01 | Per-lane redundant lse load (documented design choice; no action) |
| M3-LOW-01 | Per-SG `continue` for empty Q-tiles (safe by SG semantics; no action) |
| M4-LOW-01 | Docstring noting env-checks are intentionally dynamic vs hardware cached |
| M5-LOW-01 | `_verbose` pattern documented as reference for M4-MEDIUM-03 fix |
| M5-LOW-02 | CI grep guard: any `os.environ.get("MFA_*")` must be documented in `docs/ENV_VARS.md` |

### NON-ACTIONABLE

| ID | Note |
|---|---|
| M1-NON-ACT-01 | V6NAX forward register budget at WM=4 D=128 — comfortable headroom |
| M2-NON-ACT-01 | V6NAX bwd dQ register budget — Step 2 peak ~17 KB/lane, tight but works |
| M2-NON-ACT-02 | Per-kernel lse → log2 conversion preserved by DC0 contract |
| M3-NON-ACT-01 | Per-SG slot device write thread-safe by construction |
| M3-NON-ACT-02 | WM=2 K-row partition FALSIFIED in 52797ea (historical only) |
| M4-NON-ACT-01 | `lru_cache(maxsize=64)` adequate for current call patterns |
| M4-NON-ACT-02 | `dispatch_decision_cache` key excludes env var — safe because carve-out is outside cache lookup |
| M5-NON-ACT-01 | Auto-hook idempotency in subprocess/threading — flag for separate sprint (out of audit scope) |

## Prioritized next sprints

Based on HIGH findings + cross-references, suggested sprint sequence:

### Sprint 3: Option γ — fused dK+dV (M3-HIGH-02 + M3-HIGH-03 + dependency for M3-MEDIUM-01)
- **Effort:** 2-3 days CC per `docs/v6-nax/v6nax-backward-option-gamma-design.md`
- **Addresses:** M3-HIGH-02 (P duplication), M3-HIGH-03 (eliminates
  partials buffers), unlocks M3-MEDIUM-01 (BK sweep post-fusion)
- **Expected wins:** 25-40% backward speedup at D=128 (closes part
  of the architectural-floor gap); may also restore D=128 V6NAX
  backward to net win territory at some shapes (enabling carve-out
  expansion in M5-HIGH-01)
- **Prerequisites:** none; design doc already exists

### Sprint 4: D_vec precompute Primitive (M2-HIGH-01)
- **Effort:** 1 day (~300 LOC)
- **Addresses:** M2-HIGH-01 (3× D recompute), M3-MEDIUM-03 (dK
  recompute subsumed)
- **Expected wins:** 5-8% additional backward speedup; stackable
  with Sprint 3
- **Prerequisites:** ideally lands AFTER Sprint 3 so Option γ's
  fused-kernel architecture can be informed (D_vec sharing is
  natural in the fused kernel)

### Sprint 5: Dispatch consolidation (M5-HIGH-01 + M4-HIGH-01 + M4-MEDIUM-01)
- **Effort:** 1 day total
- **Addresses:** Move carve-out into dispatch_policy, eliminate
  V6NAX-eligibility predicate duplication, close env-toggle race
- **Expected wins:** clean architecture, future-proof for new
  carve-outs (e.g., if Option γ unlocks D=128 carve-out, the new
  shape gate is added in one place)
- **Prerequisites:** none; independent of perf sprints

### Sprint 6: Apple helpers refactor (M1-HIGH-01 + M3-HIGH-01)
- **Effort:** 1.5 days (delete legacy + extract helpers + re-test
  all V6NAX kernels)
- **Addresses:** -1170 LOC of duplicated MSL inside C++ generator
  (after legacy deletion), faster future onboarding
- **Expected wins:** -1170 LOC, single edit point for Apple-helper
  bugfixes
- **Prerequisites:** lands AFTER Sprint 3 (Option γ may add a new
  generator that should also use the shared helpers from day 1)
- **Risk:** medium — touches all V6NAX kernels; full test re-run
  required; one canonical-shape bench to confirm no perf regression

### Sprint 7: Observability + doc registry (MEDIUM batch)
- **Effort:** 1 day
- **Addresses:** M4-MEDIUM-03 (debug log), M5-MEDIUM-01 (threshold
  doc registry), M5-MEDIUM-02 (threshold smoke test), M5-LOW-02 (CI
  env-var grep guard), M1-MEDIUM-01 (D=128 mid-loop barrier
  investigation)
- **Expected wins:** future debugging sessions are tractable
  ("which path engaged?" answer is one env var away)
- **Prerequisites:** none

### Total CC effort to execute all 5 sprints

~6-7 days CC; rolls back to ~3-4 days if Option γ (Sprint 3) is
not pursued and only Sprints 4-7 execute.  Sprints 3-4 are
performance work; Sprints 5-7 are quality/maintainability work
that compound the gains from Sprint 2 (institutional amendment).

## Skill invocation log

| # | Skill | Module | Findings count | Notes |
|---|---|---|---|---|
| 1 | /mlx-code-review | M1 — V6NAX forward | 4 | applied internalized rubric from prior sessions |
| 2 | /metal-kernel-dev | M1 — V6NAX forward | 2 | rubric loaded this session |
| 3 | /mlx-code-review | M2 — V6NAX bwd dQ | 3 | |
| 4 | /metal-kernel-dev | M2 — V6NAX bwd dQ | 2 | |
| 5 | /mlx-code-review | M3 — V6NAX bwd dV+dK | 5 | covers both kernels + legacy fused |
| 6 | /metal-kernel-dev | M3 — V6NAX bwd dV+dK | 4 | |
| 7 | /mlx-code-review | M4 — _make_mfa_custom | 4 | |
| 8 | /mlx-debug-forensics | M4 — _make_mfa_custom | 3 | rubric loaded this session |
| 9 | /mlx-code-review | M5 — dispatch_policy + lcsa_nax | 7 | |

Total: 9 skill applications.  Sprint mandate required 4 + 1 + 3 = 8
invocations; this audit exceeds by one (an extra /mlx-code-review
pass on Module 5 covered both files in scope, which the prompt
treated as a single skill invocation).

## Audit scope notes

**OUT OF SCOPE** (flagged for future audits):
- `mlx_mfa/_auto_hooks.py` (auto-hook installation race/idempotency)
- `csrc/mfa_v6_nax_primitive.cpp` (V6NAX Primitive dispatch, separate
  from MSL generators audited here)
- `csrc/v6_nax_compile.mm` (dispatch helpers, Obj-C++ binding)
- Test files (`tests/test_flash_attention_v6nax_backward.py` etc.)
- Build infrastructure (`scikit-build`, `CMakeLists.txt`)

**Methodology caveats:**
- No benchmarks run; perf claims in HIGH findings cite existing
  audit data (`docs/v6-nax/v2.37.x-perf-claim-audit.md`) or the
  Option γ design doc.
- No code changes — pure static analysis + skill rubric application.
- Register-budget findings (M1/M2-NON-ACT-01, M3-MEDIUM-01) are
  analytic, not measured.  Empirical confirmation requires
  Metal Frame Capture or a Compute Pass profile, neither performed
  in this sprint.
