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
