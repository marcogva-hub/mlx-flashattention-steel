# Sprint II-6 — Numerical/Precision Deep Audit (2026-06-12)

**Status**: COMPLETE.
**Headline**: TWO production bugs found and fixed. (1) **CRITICAL — the
II-0-promoted default-on V34 backward path silently corrupted dK/dV**:
the fused dKdV kernel's BK=16 default (v2.39.1) violated the paired
16x32x16 MMA's even-TK requirement, reading past the K-tile and writing
a fragment out of bounds. (2) The M5 sparse SDPA+bias fallback returned
**NaN instead of the contractual zeros** for all-False mask rows.
The promotion's headline perf survives intact on the clean split
kernel: **2.15x / 2.61x / 2.67x** vs SDPA-vjp at N=2048/4096/8192.

## Method

1. Code-site inventory (sub-agent): 6 categories — accumulation, LSE
   conventions, mixed precision, determinism, guards/overflow, quant
   numerics — with a top-10 risk list.
2. Empirical battery (`/tmp/ii6_numerics_audit.py`): run-to-run bit
   determinism on every dispatch surface, all-masked-row safety,
   adversarial-magnitude logits (fp16 tails), TQ zero-block edge,
   gradient finiteness.  The battery — not the inventory — found both
   bugs; static reading had passed the fused kernel's mask block.

## Finding 1 (CRITICAL) — V34 fused dKdV paired-MMA out-of-bounds

| Step | Evidence |
|---|---|
| Battery | dK/dV non-finite under mag-12 inputs on the DEFAULT path; SDPA-vjp clean; opt-out clean |
| Magnitude sweep | inf onset already at input std **2.0** (realistic); count grows with magnitude |
| Kernel-mode bisect | fused (auto default) corrupt; split + legacy_fused at fp16 noise floor (4e-3/8e-3) |
| Unit-scale per-row map | dV row errors up to **35.9 vs reference magnitudes ~8** at std 1.0 — silent corruption inside the validated envelope |
| Why the II-0 gate missed it | promotion fixtures used **0.1-scale** inputs; corruption ~exp(score scale) → suppressed below the rmse gate |
| Effective-P extraction (one-hot dO) | P up to 5.78 (max possible 1.0) for nearly every row |
| L-perturbation probe | LSE row mapping CORRECT — so the recomputed S is wrong |
| S extraction (L=0) | S values shuffled by exactly **+16 = BK**; no causal mask leaks |
| Source | `for (ik = 0; ik < TK; ik += 2)` paired MMA writes `frag_at(iq, ik+1)`; at BK=16 → TK=1: K-load reads 16 rows past the tile, second output fragment lands out of bounds |
| Confirmation | `MFA_V34BWDF_BK=32` → clean at noise floor |

**Root cause class**: Pattern #9 (generator/dispatch constant mismatch
— the KD-5 class).  v2.39.1 lowered BK in the *Primitive* to fix the
v2.39.0 register spill; the *generator's* even-TK assumption (17
emission sites, all backward kernels) was never re-audited.  The
v2.39.1 "fused 1.01–1.12x vs split" claim was measured on corrupt math
and is **WITHDRAWN**.

**Fix** (commit `d76cb6e`):
- `compile_v34_backward_pipeline()`: loud `BK % 32 == 0` guard — covers
  all 8 backward Primitives including every `MFA_V34BWD*_BK` env
  override (MPP has no 16x16x16 cooperative matmul; header
  static_assert requires one dim = 32).
- Fused default BK 16 → 32 (minimum valid).
- `_v34_backward_vjp` auto → **split** for all D (fused at BK=32 is the
  v2.39.0 spill config; split is clean and carries the full promotion
  win).  Fused stays reachable via `MFA_V34_BWD_KERNEL=fused`.
- Promotion fixtures raised to unit scale; new lock file
  `test_phase2_ii6_v34_bwd_paired_mma.py` (per-element max-err at
  std 1.0, finiteness at std 2/12, BK guard raises, auto==split
  bitwise).

**Re-validated promotion numbers (split, M5 Max, B=1 H=8 D=64 causal
fp16, median of 30)**: 2.15x (N=2048), 2.61x (N=4096), 2.67x (N=8192)
vs SDPA-vjp — the II-0 headline (2.14–2.71x) was never owed to the
fused kernel.

**Marco-gated follow-up**: a true TK=1 generator variant (zeroed second
K-fragment + scratch output) to honestly re-test BK=16's register
relief — 17 emission sites, M effort.

## Finding 2 — sparse all-False rows: NaN on the M5 fallback

Native sparse kernels write zeros for a query row with no active blocks
(Track-B contract); the SDPA+float-bias fallback produced NaN (softmax
over an all--inf row) — and the v2.50 Sprint-1 dispatch migration moved
most M5 sparse shapes onto the fallback, silently changing public
semantics.  The code even documented the NaN as "preserved" behavior.

**Fix** (commit on top of `d76cb6e`): cached host-side row-activity
check (`None` on the common path → zero overhead) + cached sanitized
bias (inactive rows 0, kept alive — see pool note below) + where-zeroed
output rows.  Causal x mask interaction handled at block granularity
(matches the native kernel's tile loop bound).  Locked by
`test_phase2_ii6_sparse_allfalse_rows.py` (all-False row, causally-
unreachable-only row, per-head 3D mask, fully-active no-op).

## Finding 3 (open) — Metal buffer-pool stale-value sensitivity

While stabilizing Finding 2's fix: releasing -inf/NaN-laden temporaries
into the Metal pool flaked three unrelated finite-value kernel tests
(STEEL mixed-dtype forward, topk-bisect thresholds, sparse-native
engagement) at ~3/5 suite runs.  Mitigated by (a) never churning
-inf-laden temporaries on the fixup path (cached alive instead) and
(b) subprocess-isolating the contract tests.  The underlying question —
which of our kernels (or MLX ops) reads memory it didn't initialize —
is a real robustness issue, **queued for II-7/II-8** with repro recipe:
run the in-process variant of the contract tests before the full suite
and watch the three victims.  One victim already carries a
"stale-buffer NaN" comment from v1.3.0 — this class is old.

## Audit verdicts on the remaining inventory top-10

| Site | Verdict |
|---|---|
| LSE convention crossing (sparse natural-log vs log2) | NO BUG — V34 forward emits natural-log (verified vs reference at N=64, max err 0.0000 incl. row 0); backward kernels convert via `* log2e_f` at load; L-row mapping probe-verified |
| Softcap/ALiBi log2 constants | correctly-rounded doubles (1.4426950408889634, 0.6931471805599453); conversions paired [verified] |
| Split-K determinism | run-to-run bit-identical on dense fwd 4k, decode N_q=1 S=8k, sparse 2k, V34 bwd 4k [verified].  TM-style *batch-invariance* (length-invariant reduction order) remains an unimplemented feature — ledger item (II-5 routed, Marco priority call) |
| All-masked-row exp2 NaN guards (kernels) | hold under battery; the NaN was the SDPA fallback (Finding 2) |
| bf16 type-punning in conv_nax | entry-gated loudly (A-8); kernel unreachable with bf16 [verified inventory] |
| TQ safe-scale 1e-10 | zero-block compress path exercised, no crash/NaN [verified] |
| fp16 QK overflow pre-softmax | FP32 accumulators throughout (no fp16 long-K accumulation found); forward exact vs SDPA at std 50 [verified] |
| ALiBi log2-domain cancellation | theoretical only; magnitudes bounded by slope*distance*log2e at fp32 [assessed, no action] |
| -1e30 sentinel (sparse V34) | consumers handle via explicit compare; sparse backward suites green [deduced from tests] |
| SVD rank truncation | quality-side, error tracked at quantize time [no action] |

## Skill invocations (§AA.2)

| Skill | When | Result |
|---|---|---|
| `/mlx-debug-forensics` (protocol) | Finding 1 root cause | §8 swap-test discipline: magnitude sweep → kernel-mode bisect → effective-P/L/S extraction probes → single-config confirmation |
| `/mlx-mfa-bench-methodology` (protocol) | promotion re-bench | warm-up + median-of-30, both directions same harness |
| `/mlx-mfa-perf-audit` | applied | v2.39.1 fused perf claim WITHDRAWN (measured on corrupt math); re-stated promotion numbers measured on the clean kernel via the public API path |

## Commits

- `d76cb6e` fix(v34-bwd): paired-MMA out-of-bounds + demote auto to split + BK guard
- (this commit) fix(sparse): all-False-row contract + report

Suite: **1391 passed x6 consecutive runs** (was 1380 at sprint start;
+11 new locks).
