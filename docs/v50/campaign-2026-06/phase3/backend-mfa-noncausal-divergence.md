# Investigation — `backend="mfa"` non-causal D∈{64,128} divergence

**Date:** 2026-06-15 (III-7 follow-up; Marco chose "investigate the kernel")
**Status:** LOCALIZED to STEEL V2 forward non-causal path. Root-cause-to-line
+ fix is the continuation. **Not default-reachable** — see dispatch note.
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
