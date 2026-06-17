# Sprint IV-0 — Tech-Debt + Decode-Overhead Analysis

**Date:** 2026-06-17
**Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `75f3510` (clean), macOS 26.6 (25G5028f), Apple M5 Max 128GB, mlx 0.31.2.
**Type:** profile-first diagnostic → execute only strict-bar-passing gains. **Executed in-sprint: 0**
(definitive finding: decode overhead is sync-floor-bound, not reducible Python). Levers laddered as
Phase IV backlog.

## Skill invocations (§AA.2)

| Checkpoint | Skill | Outcome |
|---|---|---|
| Decode perf profile | `/mlx-mfa-bench-methodology` | profile (call/eval attribution) + decomposition |
| Tech-debt pass | `engineering:tech-debt` | impact×risk prioritization |

---

## R.1 — Decode hot-path profile (the deliverable that decides everything)

**Headline: decode wall-clock is overwhelmingly irreducible kernel + MLX's fixed per-eval sync
floor, NOT mlx-mfa Python interpretation.** The "replace Python with something faster" lever is
small. Profilers: `benchmarks/methodology/decode_profile/{decode_profile,tq_decode_profile,tq_step_attrib}.py`.

### Incremental decode (`flash_attention`, N_q=1) — Python ~2%

| shape | fa total | Python dispatch (call) | GPU/eval | wrapper overhead vs raw SDPA |
|---|---|---|---|---|
| S=2048 D128 MHA | 269.0us | 5.1us (1.9%) | 98.0% | +7.2us (2.7%) |
| S=4096 D128 MHA | 264.8us | 5.7us (2.2%) | 97.9% | +7.5us (2.8%) |
| S=16384 D128 MHA | 347.7us | 5.7us (1.6%) | 98.4% | +0.3us (0.1%) |
| S=4096 D128 GQA32:8 | 256.5us | 5.6us (2.2%) | 97.8% | −1.5us (noise) |
| S=4096 D64 MHA | 256.8us | 5.7us (2.2%) | 97.5% | −0.9us (noise) |

On M5 dense decode routes to Apple SDPA; mlx-mfa's wrapper adds **≤7.5us (≤2.8%)**, →0/noise at
large S. **Python dispatch is already in the noise.** Not a lever.

### TQ paged decode (`step()`, N_q=1) — Python orchestration 50–60%, but it's a per-step EVAL, not interpretation

`step()` call-time = ~480us = 50–60% of the ~860us step. Attribution (S=4096):

| step() sub-component | call-time | note |
|---|---|---|
| **`append()`** | **437.5us** | **98% of the orchestration** |
| `tq_decode_attend` (kernel call) | 7.0us | |
| `cu_q = mx.array([0,1])` | 1.1us | already lean |
| `get_block_table` | 0.6us | cached (Sprint C #4) |
| `get_seq_lens` / `seq_length` | 1.5us | lean |

`append()` decomposed (the crux):

| measurement | time | meaning |
|---|---|---|
| pack_k **graph-build** (pure Python) | 66.6us | reducible Python (interpretation) |
| pack_k + **eval** (materialize) | 543.7us | +477us GPU pack/scatter + sync |
| **bare `x+1` + eval** (M5 sync floor) | **240.9us** | **irreducible MLX per-eval round-trip** |

`append()` ends with a **mandatory per-step `mx.eval(self._k_pool, ...)`** (`inference.py:994`): the
pools are bound **raw** (`set_input_array`) by the downstream gather, so they must be materialized
before the next dispatch and **cannot fold into the step's final `eval(o)`**. So the 437us is
**layer-3 (GPU materialization + per-token sync), not Python**: ~66us reducible Python graph-build,
~240us irreducible MLX eval round-trip floor, ~130us actual pack/scatter kernel work.

**Answer to "can we replace Python with something faster?":** the Python-interpretation lever is
**~2% (incremental) to ~8% (TQ)** of decode wall-clock. The dominant decode cost is the kernel +
MLX's fixed ~240us per-eval sync floor — both irreducible by rewriting mlx-mfa Python. The one
material *structural* lever (≈240us, ~28% of TQ step) is collapsing the TQ `append` per-step eval
into the step's final eval — a **dispatch/lifetime change** (the `add_temporary` risk class), NOT a
safe Python rewrite.

## R.2 — Tech-debt pass

Env-knob inventory clean (all live/documented; the `MFA_FORCE_NATIVE_BWD` dead knob was already
removed in v2.56.0). No live dead code surfaced (`_sever_lazy_graph` was already removed v2.20.0).
No stale arch-gen guard (all `>=15`/`>=17` correct). Findings:

| # | Item | Category | Impact×Risk | Disposition |
|---|---|---|---|---|
| D1 | TQ `append()` per-step `mx.eval` (~240us sync floor + ~130us pack/scatter) | Overhead (hot-path) | high impact / **high risk** (raw-buffer lifetime; `add_temporary` class) | **DIAGNOSTIC-ONLY** |
| D2 | `MFA_TOPK_BISECT` deprecated-but-functional opt-in | Dead-ish knob | low / med (keep-all-paths: working path) | clarity/diagnostic — KEEP |
| D3 | `backward='sdpa_sparse'` soft-deprecated public API | Dead-ish path | low / med (no committed removal target, unlike the flag) | clarity/diagnostic — KEEP |
| D4 | V2/V3/V4/V5 dispatch-block boilerplate duplication (`mfa_attention.cpp`) | Redundancy | med / **high risk** (touches dispatch path) | diagnostic — refactor backlog |
| D5 | `mfa_paged_gather.cpp:169` hardcoded `gqa_factor (unused)` | Cosmetic | trivial / cold | clarity-only |

## R.3 — Prioritized: in-hot-path × safe-to-fix

| | in decode hot-path | cold |
|---|---|---|
| **safe to fix** | *(none)* | D5 (cosmetic, not worth a commit) |
| **risky (dispatch/lifetime)** | **D1** (TQ append eval — the only material lever) | D2, D3, D4 |

**No item is BOTH in the hot-path AND safe-to-fix.** The hot-path lever (D1) is a dispatch/lifetime
change; the safe items are cold/cosmetic.

## R.4 — Executed gains: **NONE** (disciplined per the strict bar)

No item clears all three execute-bar conditions (measured-dominant + correctness-neutral + structural):
- D1 is measured-dominant but **not correctness-neutral** (raw-buffer materialization; removing/
  deferring the eval risks the gather reading stale pool — the inverse of the III-9 `add_temporary`
  bug) → diagnostic-only.
- D2/D3 are working paths (keep-all-paths) with soft deprecations and no committed removal cycle →
  not removal-due (unlike the flag, which had "target removal v2.51+").
- D4 is a dispatch-path refactor → not a "safe gain".
- D5 is cold + cosmetic → not worth a commit.

Per **Pattern #6 for optimization** (do NOT rewrite an overhead not measured as dominant) + the
strict bar, **executing zero is the correct outcome.** Manufacturing a noise-level Python tweak
would add risk for no measurable decode gain.

## R.5 — Phase IV backlog ledger (for Marco's direction)

**Diagnostic-only (high-value, risky — needs three-axis + Pattern #6 if pursued):**
- **IV-D1 — TQ `append` per-step eval collapse.** Potential ~240us/step (~28% of TQ step) by making
  the K/V pools lazy graph-inputs of the gather (so the step's final `eval(o)` materializes them) and
  dropping the separate append eval. **Risk: the `add_temporary`-class lifetime bug** (gather reading
  unmaterialized/stale pool) — requires the raw `set_input_array` → graph-input binding change in
  `tq_decode_attend` + the paged kernels, full three-axis, and a multi-step decode-equivalence soak.
  This is the single biggest decode structural lever; it is NOT a safe in-sprint gain.
- **IV-D4 — dispatch-block boilerplate consolidation** (V2/V3/V4/V5 in `mfa_attention.cpp`): a
  param-struct/key-build helper would cut duplication, but it touches every dispatch path →
  three-axis per kernel. Clarity + maintainability, not perf.

**Clarity-only (cold, low priority):**
- IV-D2 `MFA_TOPK_BISECT` + IV-D3 `backward='sdpa_sparse'`: working soft-deprecated paths; if Marco
  wants them gone, each needs a committed deprecation cycle (announce target version → remove), like
  the flag. Keep-all-paths until then.
- IV-D5 `gqa_factor (unused)` cosmetic literal.

## Release disposition

**Nothing executed → nothing new to release.** The held **v2.56.0** scope (MFA_FORCE_NATIVE_BWD
removal + V3 validation) stands unchanged. The decode profile is the deliverable; the levers are
Marco-gated backlog.

## Bottom line for Marco

Decode wall-clock is **~98% kernel + MLX sync floor** (incremental) / **~50% append-eval +
gather/dequant kernels** (TQ), with mlx-mfa **Python interpretation at ~2–8%** — small. The
Python-rewrite lever is not where the gains are. The real decode lever is **MLX's per-eval sync
floor** (~240us/step), recoverable only by collapsing the TQ append eval into the step graph — a
dispatch/lifetime change (IV-D1) that needs careful three-axis validation, not a safe sweep. **No
safe in-sprint gain existed; zero executed by design.** Phase IV's real perf work is IV-D1 (if the
lifetime change can be made safe) — everything else is clarity/maintainability.
