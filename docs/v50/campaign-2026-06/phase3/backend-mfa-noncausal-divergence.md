# Investigation — `backend="mfa"` non-causal D∈{64,128} divergence

**Date:** 2026-06-15 (III-7 follow-up; Marco chose "investigate the kernel")
**Status:** **RESOLVED (III-9).** ⚠ The III-8 "async metallib root cause"
below was a **WRONG TURN** — see **§ Resolution (III-9 — CORRECTED)** at the
bottom for the actual root cause. In short: `backend="mfa"` non-causal D∈{64,128}
dispatches to the V2 **split-K** path, which allocated `pO`/`pL` scratch and
**freed it at encode time** while MLX's lazy execution left Phase 1/2 pending —
a concurrent allocation reused the pool memory and corrupted the not-yet-run
reduce. Fix: tie scratch lifetime to command-buffer completion via
`enc.add_temporary` (`csrc/mfa_attention.cpp`). The III-8 async gate is
*independently defensible* (the metallib's `simdgroup_async_copy` is genuinely
broken on macOS 26) but did **not** fix this divergence — the async path was
never even on the failing dispatch. **Not default-reachable** (auto→SDPA).
**Method:** independent fp32 reference (Apple SDPA at fp32) throughout; the
auto-hooks do NOT patch `mx.fast.scaled_dot_product_attention`, so it is a
genuinely independent attention oracle (lesson #11).

## Symptom (independently confirmed)
`flash_attention(q, k, v, causal=False, backend="mfa")` at D∈{64,128} fp16
diverges from fp32 SDPA — **MAE ~0.12–0.13** at all N (including
block-aligned N=256/4096), dimension- and scale-independent. `backend="auto"`
(the default) is correct everywhere (MAE 0.0 — routes to SDPA). `backend="mfa"`
**causal** is correct (MAE 0.0). So only the **forced, non-causal** path is wrong.

## Divergence signature
On a B=1 H=1 N=8 D=64 probe vs fp32:
- Output rows are **duplicated**: rows 0 and 1 have identical norm (8.73);
  rows 2, 3, 6 identical (8.50). The correct (fp32) output has all-distinct
  rows. → a per-query-row **index-collapse**: adjacent query rows produce the
  same output.
- Output magnitude is **~2× too large** (row norms ~8.7 vs correct ~4.9).
  → a softmax-**normalization** error (the online-softmax denominator / final
  divide is wrong for the non-causal case).

Both together: the non-causal path is mis-accumulating / mis-normalizing
per query row. NOT a partial-tile/boundary bug (present at aligned N).

## Localization (bisection via dispatch knobs) — REFINED to V2 single-pass
| Config | MAE vs fp32 |
|---|---|
| default, B·H ∈ {1,2} (under-occupied → split-K) | **0.0** |
| default, B·H ≥ 4 (occupied → single-pass) | **0.1276** |
| `MFA_DISABLE_V2=1` (forces V1), any B·H | **0.0** |
| `MFA_FORCE_SPLITK=0` (force single-pass), H=1/2/4 | **0.127 (all)** |
| `backend="mfa"` **causal** single-pass H=4 | **0.0** |

→ The bug is in the **STEEL V2 SINGLE-PASS forward, non-causal branch**
(`csrc/mfa_steel_fwd_v2.cpp::generate_steel_v2_source`). Established:
- **V1 non-causal correct**, **V2 causal (single-pass) correct**, **V2
  split-K non-causal correct** — only V2 single-pass non-causal is wrong.
- The earlier "H=1 correct" was split-K (under-occupied grids route there);
  forcing single-pass (`MFA_FORCE_SPLITK=0`) reproduces the bug at **all H**
  including H=1.
- (v1.4.0 notes benched "V2 1.04–1.32× vs V1 non-causal" — likely a
  regression from a later single-pass change; git archaeology pending.)

## Mechanism — narrowed
At N=256 (no partial K-tile, no window) the non-causal single-pass path
executes the SAME math as causal-minus-diagonal-mask over more K-tiles.
Characterization (single-pass, H=1, N=256, vs fp32):
- output magnitude **~1.5× too large** (row norms ~2.6×) — softmax
  denominator effectively under-weighted / numerator over-accumulated;
- per-row error **decreases monotonically with query position** (Q-tile 0
  MAE 0.21 → Q-tile 7 MAE 0.085);
- deterministic (not a race in the naive sense — same MAE every run).

Ruled OUT as the cause (identical for causal, which is correct): the final
O normalization (`Otile.row_bin_op<MFADivOp>(sum_score)`, line 838); the
online-softmax max/sum update (line 721+, "same as V1"); the L write. The
only causal-conditional differences for this shape are `kb_lim` (non-causal
= full `NK`) and the diagonal mask block (causal-only). → the fault lies in
V2's **single-pass-specific KV_smem reuse / K-preload / barrier dance**
(the K-phase→V-phase shared-smem machinery, preload K[kb_start] + per-iter
preload K[kb+1]) when it runs over the FULL non-causal tile range — a regime
the causal path (fewer tiles) and split-K (different reduce) never stress
the same way. The monotonic-in-q error suggests a cross-tile accumulation
or a V-tile/barrier ordering issue, not a per-row normalization constant.

## Next step (repair — Marco approved "repair V2 non-causal")
Read the V2 single-pass KV_smem preload + barrier sequence (lines ~322-375
preload, ~775-831 per-iter V-load/K-preload/barriers) and the P@V
accumulation, diffing the causal vs non-causal execution over the full tile
range; git-blame the single-pass K/V-smem block against the v1.4.0 working
state. The magnitude-too-large + monotonic-in-q signature is the anchor.
This is delicate barrier/smem work — to be done with fresh focus, validated
vs fp32 across D/N/dtype/B·H, then locked with a forced-single-pass
non-causal regression test. **Not yet fixed.**

## Why it was never caught
- `dispatch_policy` routes **all non-causal dense to SDPA for performance**
  (documented: "non-causal dense routes remain conservative SDPA"; M3+/M5
  non-causal D≤128 "SDPA wins. Disabled."). So V2 non-causal is **never
  auto-selected** — only reachable by explicitly forcing `backend="mfa"`.
- No test forces non-causal `backend="mfa"` at D=64/128 (the §Z perf-claim
  tests use the default/auto path; correctness tests that force mfa are
  causal). A perf-motivated gate masking a latent correctness bug on a
  forced expert path — a Class-B.2 cousin of the conv3d "only reached when
  [traffic is routed] elsewhere" trait (here a *perf* gate, not a
  *correctness* gate).

## Next steps (continuation)
1. **Root-cause to line:** read the V2 non-causal softmax-normalization and
   output-write (`mfa_steel_fwd_v2.cpp` online-softmax block ~721+, output
   write ~665-720), comparing against the V1 path (which is correct) — the
   duplicate-row + 2× signature should pinpoint a query-row stride or a
   missing final `/ l_sum` normalization on the non-causal branch. Git-blame
   the V2 non-causal path against the v1.4.0 "working" state.
2. **Fix options** (Marco-gated disposition):
   - Fix the V2 non-causal normalization/indexing (restores a correct, fast
     non-causal V2; validate vs fp32 across D/N/dtype).
   - OR, if V2 non-causal has no production value (auto routes to SDPA), make
     the V2 dispatch decline non-causal (route forced-mfa non-causal to V1 or
     SDPA) + a Rule-8 note — smaller, removes the silent-garbage forced path.
3. **Lock:** add a forced-`backend="mfa"` non-causal correctness test vs fp32
   (the regime no test currently exercises) so it can't silently regress again.

## Disposition
Not default-reachable; no production path affected. The fix approach (repair
V2 non-causal vs route around it) is Marco's call. No release implication for
the default path.

---

## III-8 — repair attempt: further localization (NOT yet fixed)

Marco approved "repair." III-8 narrowed substantially further but did **not**
land the fix — the exact mis-computed line was not isolated by reading, and
per the sprint's own rule (and race risk) a kernel edit must NOT be guessed.
Honest status: **localized very tightly; repair pending instrumentation.**

New empirical facts (all vs independent fp32):
1. **Both code paths equally wrong.** `MFA_FORCE_GEN=14` (smem+barrier path)
   and default (M5+ `MFA_DIRECT_READS`) give the SAME non-causal MAE → the
   bug is in **path-independent shared code**, not the barrier/smem machinery
   the III-7 doc hypothesized.
2. **The non-causal code path is intrinsically broken — not a key-subset
   issue.** The LAST query row attends to all keys under BOTH causal and
   non-causal (identical math). Result: causal-MFA last row MAE **0.0**,
   non-causal-MFA last row MAE **0.115**, and the two differ by 0.115. So
   non-causal produces a wrong answer for a row whose computation is
   mathematically identical to the (correct) causal one.
3. **Params ruled out.** The V2 single-pass dispatch (`MFASteelParams sp2`,
   mfa_attention.cpp:950+) builds identical params for causal vs non-causal
   (only `qL_off` differs, = 0 when N==S). So the divergence is in the
   generated MSL non-causal path, not the params.
4. **Present at a single tile; dilutes with N.** Error at N=16 (1 tile) is
   ~0.28, falling to ~0.12 at N=256 — consistent with a fixed spurious
   contribution diluted as the (growing) softmax denominator normalizes it.
5. **Longstanding.** git-blame puts the kb_lim / softmax / preload lines at
   the original V2 commit (81d801f7, 2026-03-10) — not a recent regression;
   never caught because non-causal single-pass is never auto-dispatched and
   no test forced it.

**Logical corner:** the ONLY kernel-source difference between causal and
non-causal for this shape (N=128, no partial tile, no window) is the causal
diagonal-mask block (sets future-key fragments to -INF) and the kb_lim
expression (both = NK for the last Q-tile). Yet the last row — which the
causal mask does not touch — differs. This implies the causal mask block is
incidentally zeroing Stile fragments that, in the non-causal path, hold
values that are WRONG (stale/garbage/mis-mapped), i.e. the bug is in how the
Q@K^T scores populate the Stile MMA fragments (or how `row_reduce` consumes
them) for positions the causal path never uses. Resolving it requires
reading the MMAFrag lane→(row,col) mapping + `row_reduce` semantics, or
kernel instrumentation (dump raw Stile / sum_score for a 1-tile non-causal
case and diff vs numpy).

6. **Wrong DIRECTION, not just magnitude (corrects the III-7 framing).**
   Per-row cosine(non-causal-MFA, fp32) is **low — mean 0.36, max 0.66** (not
   ~1.0), with per-row norm ratio ranging 0.6–4.1. So the output is not a
   per-row rescale of the correct answer (which a pure `sum_score`/denominator
   bug would give) — the attention **weights P themselves are wrong**, so P@V
   combines values in the wrong proportion. The III-7 "denominator
   under-accumulated" read (inferred from mean magnitude) was incomplete: the
   numerator/weights are wrong too. The fault is in the Q@K^T → Stile score
   population (or `row_reduce`/`MFAExpSubOp` consuming the fragments) for the
   non-causal path.

**Next step (instrumentation, not a guessed edit):** dump the post-Q@K^T
Stile scores AND the post-softmax P for a 1-tile non-causal case
(B=1 H=4 N=32 D=64) to a scratch buffer, rebuild once, and diff against a
numpy `Q@K^T`/softmax — the low cosine says the scores or P are wrong, so this
pins whether it's the matmul fragment mapping, the scale, or the exp/reduce.
Then fix, validate vs fp32 across the domain, re-bench (per the keep-all-
paths + re-bench principle), and lock with a forced-single-pass non-causal
regression test.

**Status: STILL NOT FIXED.** The repair was not completed in III-8; the
localization is now tight enough that the instrumentation step above should
resolve it in a focused pass. No edit was made to the kernel (no guess).

---

## III-8 repair session — outcome: NOT fixed; sub-component NOT reliably isolated

Marco scoped III-8 to repair. I instrumented extensively but did **not** land
the fix and made **no functional kernel edit** (only temporary, reverted debug
dumps). Critically, I could **not reliably isolate the sub-component** (Q@K^T
vs softmax vs P@V), because every available probe was either unreliable or
confounded. Documenting the dead-ends so the next pass doesn't repeat them:

**Reliable facts (unchanged / strengthened):**
- Deterministic across repeated runs (maxdiff 0.0) → **not a race**.
- Bug is the V2 single-pass non-causal **compiled variant**; the causal
  variant (separately JIT-compiled from the same largely-shared source) is
  correct. Only `kb_lim` and the diagonal-mask block differ in source.
- Softmax **normalization** is not the (sole) fault: with a reliable full-
  pipeline probe the output distribution is well-formed.

**Probes attempted and why each could not pin the sub-component:**
1. **Mid-kernel register dump of Stile/P** (two methods: `store_contiguous`
   and faithful element-wise `frag_at` with the masking code's (row,col)
   map). **UNRELIABLE** — proven by a control: dumping the *causal* Stile
   (whose output is correct) also yields "wrong" values vs the true scores,
   and the two dtype/methods disagree. The cooperative simdgroup-MMA
   fragment register layout is not faithfully serialized by a naive
   per-lane write, and the early `return` perturbs state. **Do not trust
   mid-kernel register dumps for this kernel without a layout-correct
   gather (e.g. simd_shuffle-based) probe.**
2. **One-hot V (O=P) recovery.** Reliable for **causal** (causal P@V is
   proven correct by causal's correct real-V output) — and it shows causal
   lower-triangle scores are correct (MAE 1e-4). But for **non-causal** it
   is **confounded**: O=P assumes P@V is correct, which is exactly one of
   the unproven suspects for the non-causal path — so a non-causal O≠P could
   mean wrong scores OR wrong P@V, indistinguishably.

**Net:** the sub-component (Q@K^T vs exp/reduce vs P@V) is **not reliably
isolated**. The earlier "it's Q@K^T / the scores" read (III-7 + the first
III-8 dumps) is **retracted** — it rested on the unreliable register dump.

**Reliable next approaches (for a future focused pass):**
- A **layout-correct fragment gather** probe (simd_shuffle the MMA fragment
  into a known-order scratch buffer) so mid-kernel Stile/P can be trusted; OR
- a **reference-impl differential**: compare the non-causal single-pass
  output against the (correct) V2 split-K output stage-by-stage to localize
  the divergence (kernel-vs-kernel as a localization aid only, per lesson
  #11 — not as the correctness oracle); OR
- re-evaluate the **route-around** option (forced-mfa non-causal → V1/SDPA),
  which achieves correct output on the API surface at lower risk than a
  blind kernel edit, given the repair is proving expensive to pin.

**Effort note:** this non-default-reachable bug has now consumed III-7 + III-8.
It is genuinely resisting the diagnostic methods tried. No guessed edit was
made (a wrong barrier/MMA edit would risk a race — worse than the current
localized, non-default bug).

---

## III-8c (static-diff reorientation) — narrowed further; still not isolated

Full report: `sprint-III-8c-report.md`. Register dumps abandoned (lesson #12).
New reliable findings (O vs fp32 only):
- **Exactly two source diffs** (causal vs non-causal single-pass): `kb_lim`
  (435) and the diagonal-mask block (665). Both read correct; both original
  (commit 81d801f7), unchanged.
- **v1.4.0 contradiction RESOLVED**: no v1.4.0 tag; code is original. "Benched
  working" = coverage illusion (non-causal single-pass correctness never
  fp32-validated). Not a regression — longstanding original bug.
- **Isolated to mask-absence on a single FULL tile**: N=64 D=128 (1 tile,
  `kb_lim=1` for both causal+non-causal) — causal 0.0001, non-causal 0.19. The
  only operative difference is the diagonal mask. → the **`col>row`
  (future-key) positions' contributions are wrong**; causal masks them to zero
  (so causal never exercises them), non-causal uses them.
- **BK-independent** (`MFA_V2_FORCE_BK=32` doesn't fix it). Deterministic.
- **Cannot separate wrong-scores vs wrong-P@V** for the `col>row` positions:
  causal zeros them (no causal signal), and the dump (unreliable) / one-hot-V
  (confounded for non-causal) probes can't decide. → a subtle MMA-fragment
  issue at `col>row`, invisible to source reading.

**Next fork (scoped, not executed):** (1) layout-correct `simd_shuffle`
fragment gather to read `col>row` scores reliably; or (2) minimal standalone
MMA reproduction (32×64 Q@K^T tile vs numpy). Marco's call: next-fork vs
route-around (forced-mfa non-causal → V1/SDPA, low-risk correct API output)
vs pause. No guessed edit. Not default-reachable; no release impact.

---

## III-8d — standalone MMA prep + rigorous dump closure (still not isolated)

Full report: `sprint-III-8d-report.md`.
- Understood the MMA machinery: `MFAMMAFrag`=`simdgroup_matrix<T,8,8>`;
  `Stile` is a Q@K^T **accumulator** tile; `store<U,1,1>` is a per-lane
  write (`base=i*8*ld+j*8`, dst pre-offset by `simd_coord`).
- **Mandated `col≤row` self-check RIGOROUSLY closes the register-dump
  approach**: the causal `Stile.store` dump gives MAE **1.18** vs numpy on
  `col≤row` — a region causal output PROVES correct — so the store read is
  **unfaithful** (and causal vs non-causal dumps differ pre-mask, MAE 0.24,
  though they must be identical). The kernel consumes `Stile` correctly via
  `frag_at` (causal works) but **serializing it to memory via `store` is
  unfaithful**, even though the identical `store` is faithful for `Otile`.
- **Blocker:** reading the `col>row` scores requires a faithful
  cooperative-MMA-**accumulator** serialization; 3 read methods
  (`store_contiguous`, element-wise `frag_at`, `store`) all fail the
  self-check. The kernel never serializes `Stile` to memory, so there's no
  proven-faithful path to borrow. A standalone harness uses the same store
  → same blocker.
- **Next fork:** (1) derive the exact accumulator lane→(row,col) layout via
  a trivial numpy-parity single-MMA test (Q=I,K=I, known scores), validate
  on `col≤row`, then read `col>row`; or (2) escalate to the MLX
  MMA-primitive layer. Marco's call: dedicated fork session vs route-around
  interim. No guessed edit; not default-reachable.

---

## III-8e — MECHANISM CRACKED (behavioral, reliable); exact line still open

Full report: `sprint-III-8e-report.md`. The known-answer uniform-P probe
(Q=0 ⇒ uniform P; V=ramp ⇒ attended-key mean; V=indicator ⇒ attended set)
cracked it reliably (no dump, no confound, validated vs fp32):

**Non-causal single-pass attends exactly `(qb+1)·BQ` keys per Q-tile**
(qb=0→32, qb=1→64, qb=2→96, qb=3→128 at BQ=32), tile-uniform, sub-tile
granular. = the **causal `q_max=(qb+1)·BQ` key bound leaking into the
non-causal path**. Keys ≥ q_max are **truncated (not attended), not
miscomputed**. Resolves the four-sprint paradox: causal masks ≥q_max anyway.

**Exact line OPEN**: the `(qb+1)·BQ`-keys signature matches NO obvious source
— `kb_lim`=NK (q-independent, =64 keys not 32), causal kb_lim formula (=64
not 32), masks (gated/inactive at the repro), dispatch params (correct,
q-independent) — all ruled out. The q-dependence requires an in-kernel
`qb`-using key limit not located by static reading.

**Next**: the uniform-P probe is a reliable cheap ORACLE — bisect the kernel
source with it (disable/alter candidate qb-using regions, re-measure
attended-key count until it = N for all qb) to pin the line empirically, then
fix + O-vs-fp32 + generalize + re-bench + lock. No guessed edit.

---

## § Resolution (III-8)

**The bug is NOT in the JIT source.** It is the precompiled
`mlx_mfa/precompiled/async_v2.metallib`.

### Why the line was "open" — the tell we missed for 4 sprints
III-8e's own finding was the smoking gun, misread: the `(qb+1)·BQ`-keys
signature matched **no obvious line in `generate_steel_v2_source`** because
**that source was never executing.** `shader_cache.mm::get_or_compile()` calls
`try_async_pipeline()` *first* (before any JIT), and for `SteelForwardV2` keys
on macOS 26 it returned a pipeline built from `async_v2.metallib`. That metallib
uses `simdgroup_async_copy` (hardware DMA), which Apple **removed from the
runtime AIR compiler in macOS 26** (confirmed by liuliu; see CLAUDE.md
"simdgroup_async_copy — Definitive Status"). The broken DMA loads only a
*partial, progressively-growing* K range — exactly `~(qb+1)·BQ` keys per
Q-tile, which is what the uniform-P oracle measured. Causal coincidentally
survives because its mask zeroes the unloaded keys; the default dispatch routes
non-causal dense → SDPA, so only `backend="mfa"` (expert) reaches it.

The four sprints of register dumps / static differential reading / oracle
bisection of `generate_steel_v2_source` were **all inert** — they instrumented
a binary that does not run on this machine. (Meta-lesson #14, audit-framing.)

### How it was confirmed (not guessed)
- **`MFA_DISABLE_ASYNC=1` differential:** the env var that skips
  `try_async_pipeline` flips non-causal `backend="mfa"` from MAE ~0.12 → bit-
  exact vs fp32. Source-only theories cannot explain an env var that bypasses
  source compilation fixing it.
- **Sentinel-write test:** a sentinel-777 store added to
  `generate_steel_v2_source` never appeared in the output → the JIT source is
  not the running binary for that key. (Sentinel-write debugging, §AA.5.x /
  kernel-debugging.md.)

### The fix
`csrc/shader_cache.mm::try_async_pipeline()` — gate the async fast-path off on
macOS 26+ (after the existing `SteelForwardV2`-only guard and the
`MFA_DISABLE_ASYNC` check):

```objc
if ([[NSProcessInfo processInfo] operatingSystemVersion].majorVersion >= 26) {
  return nullptr;  // simdgroup_async_copy broken on macOS 26 → use JIT path
}
```

All `SteelForwardV2` dispatch then falls through to the correct JIT path
(`generate_steel_v2_source`, `preferAsyncCache=true` per-lane device reads),
which was correct on macOS 26 all along.

### Validation
- **R.4 (O vs fp32):** full `backend="mfa"` sweep — D∈{64,128} × {fp16,bf16} ×
  causal/non-causal × N∈{32,64,127,128,256,1000,4096} × B·H 4–16 — all correct
  vs independent fp32 SDPA (lesson #11). Deterministic (maxdiff 0.0 across runs).
- **R.5 (generalization):** the fix disables the broken metallib for **all** its
  keys on macOS 26 (it only ever served `SteelForwardV2`); inherent, not
  per-shape. Full suite **1728 passed, 2 skipped** with the fix + lock test.
- **R.6 (re-bench + conditional promotion):** the JIT V2 forward is ~3–4×
  *slower* than Apple SDPA on M5+, so the default auto-dispatch (≈ SDPA) is
  already optimal and the async metallib was only reachable via `backend="mfa"`.
  The fix is **correctness-only; no production perf impact; no promotion**
  (consistent with the dispatch policy that routes non-causal dense → SDPA).
- **R.7 (lock):** `tests/test_iii8_backend_mfa_noncausal.py` — fp32-parity sweep
  + a forced-single-pass non-causal known-answer test
  (`MFA_FORCE_SPLITK=0` autouse fixture) that asserts each Q-tile attends ALL
  keys (`O[qb*32,0] == (N-1)/2`), precisely exercising the truncated path the
  async metallib broke. 57 passed. This closes the v1.4.0 coverage illusion
  (every prior V2 test used `backend="mfa"` causal or auto, never forced
  non-causal single-pass).

### ⚠ Separate, pre-existing bug found (flagged, NOT fixed here)
While building R.7, the lock surfaced an **independent** bug: V2 **split-K**
non-causal at **partial N** (N∈{127,160,191,224}, D=128) gives MAE ~7–16 vs
fp32. This is NOT caused by — and not fixed by — the async gate: `try_async_pipeline`
only ever served the **single-pass** path, so split-K always used JIT. It is a
distinct defect in the split-K partial-N reduction/range handling, non-default-
reachable (`backend="mfa"` + the split-K occupancy heuristic + partial N). The
lock test scopes itself to single-pass (`MFA_FORCE_SPLITK=0`) to stay precise.
**Follow-up sprint required** — see spawned task / III-9 candidate.

---

## § Resolution (III-9 — CORRECTED ROOT CAUSE)

**The III-8 async-metallib theory was wrong.** Applying lesson #14 (confirm
which binary/path actually runs) with instrumentation showed `backend="mfa"`
non-causal D∈{64,128} dispatches to the V2 **split-K** two-phase path
(Phase 1 `SteelV2SplitKPartial` → Phase 2 `FlashDecodeReduce`), not the
`SteelForwardV2` single-pass that `try_async_pipeline` serves. The async gate
never touches the failing dispatch.

### Actual root cause: premature free of lazy-executed scratch
`csrc/mfa_attention.cpp` (split-K Phase 3, and the flash-decode Track-H path)
allocated `pO`/`pL` scratch with `allocator::malloc` and **freed them at encode
time** (`allocator::free`), with a comment assuming "Metal retains them until
the command buffer completes." That is false under MLX's lazy execution:
`dispatch_threadgroups` only *encodes*; the kernels run later at `mx.eval`. The
encode-time free returned the pool memory immediately, so any allocation between
the `flash_attention` call and `mx.eval` (the reference SDPA in a test, or any
downstream layer in real use) could reuse it and **corrupt the not-yet-executed
Phase-2 reduce read.**

### Isolation ladder that pinned it (each vs independent fp32)
| Probe | Result | Inference |
|---|---|---|
| per-split `pO` dump, 6× cold | reliably correct | partial kernel writes correct `pO` |
| per-split `pL` dump, 6× cold | reliably correct | partial kernel writes correct `pL` |
| `lse_max`/`sum_w`/`num_splits` dump | all correct | reduce Pass 1/2 correct |
| numpy-combine(dumped `pO`,`pL`) | == fp32 (MAE 7e-5) | inputs + reduce math are sufficient |
| `mx.eval(o)` **alone** | correct (4/4) | no bug without a concurrent alloc |
| `mx.eval(o, concurrent_sdpa)` | garbage (5/5) | concurrent alloc corrupts it |
| suppress the free (`MFA_NO_FREE`) | correct (4/4) | **the free is the cause** |

This explains every earlier symptom: nondeterministic (allocator timing),
"upper query rows garbage" (reduce reads after partial pool overwrite),
"Q=0 uniform-P masks it" (those probes had no concurrent allocation),
gen-independent (allocator, not hardware), output saturating to fp16 512.

### Fix
Wrap the scratch in `mlx::core::array`s and register them as command-encoder
temporaries (`enc.add_temporary`), so MLX frees them **only after the command
buffer completes**. Removed the encode-time `allocator::free`. Applied to BOTH
the split-K path and the flash-decode path (same latent pattern — §AA.5.x
multi-gate audit).

### Validation
- **R.4**: 576-config sweep (`force_splitk` 0/1/heuristic × D{64,128} × B·H ×
  N{127..1000} × causal × {f16,bf16}) **with concurrent-allocation stress** →
  0 split-K failures vs fp32.
- **R.5**: full suite **1768 passed, 2 skipped, ×2 consecutive**.
- **R.6**: lifetime-only change (no kernel compute change) → split-K perf
  identical; non-default path.
- **R.7 lock**: `tests/test_iii9_splitk_lifetime.py` (40) — forces split-K +
  concurrent alloc + fp32 reference.

### Separate items flagged (not fixed here)
- single-pass D=64 N=1000 non-causal fp16: NaN only deep in the 576-config
  exhaustive sweep; clean in isolation and a 30× warm loop. Pre-existing
  memory-interaction artifact, distinct from this lifetime bug.
