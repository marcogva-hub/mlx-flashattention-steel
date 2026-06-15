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

## Localization (bisection via dispatch knobs)
| Config | MAE vs fp32 |
|---|---|
| default (V2 active) | **0.1276** |
| `MFA_DISABLE_V2=1` (forces V1) | **0.0** |
| `MFA_DISABLE_V3=1` (V2 still active) | 0.1276 |
| `MFA_DISABLE_V2=1 MFA_DISABLE_V3=1` | **0.0** |

→ The bug is in the **STEEL V2 forward kernel** (`csrc/mfa_steel_fwd_v2.cpp`),
**non-causal path only**. V1 non-causal is correct; V2 causal is correct.
V3 is not involved. (Note: V2 non-causal was benchmarked working in an earlier
version per the v1.4.0 notes — "V2 1.04–1.32× vs V1 non-causal" — so this may
be a regression from a later V2 change, e.g. vec2 loads / split-K / a refactor.
Confirming via git archaeology is the next step.)

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
