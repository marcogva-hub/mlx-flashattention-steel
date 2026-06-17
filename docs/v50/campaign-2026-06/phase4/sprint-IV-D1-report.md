# Sprint IV-D1 — TQ `append()` Per-Step Eval Collapse

**Date:** 2026-06-17
**Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `2686d9f` (clean, post-IV-0), macOS 26.6 (25G5028f), Apple M5 Max 128GB,
mlx 0.31.2.
**Type:** targeted decode optimization of the `add_temporary` inverse-risk class.
**Outcome: SHIPPED for `tq_v=False` (~1.63× TQ-decode step, ~250us/step) + structural finding for
the default `tq_v=True` (eval required pending the lazy-packed-V restructuring).**

## Skill invocations (§AA.2)

| Checkpoint | Skill | Outcome |
|---|---|---|
| Decode perf measurement | `/mlx-mfa-bench-methodology` | A/B eager-vs-deferred, 3 sessions, detached |

---

## R.1 — Mechanism recovered (the binding was already correct)

The IV-0 worry — that the decode path binds pools RAW and would need a risky binding change — was
**wrong**. `tq_decode_attend` (`mlx_mfa/tq_decode.py:190-196`) reads the pools as
`mx.fast.metal_kernel(inputs=[k_pool_tq, k_scales, ...])` / `inputs=[v_pool_fp16, ...]` — i.e. as
**MLX graph-inputs**. The gather outputs K, V depend on the pools *through the graph*, and SDPA's
`o` depends on K, V → **`eval(o)` already materializes the pool writes in dependency order.** So
`append()`'s per-step eager `mx.eval` is **redundant for the decode path** — this is a
redundant-eval removal, not a binding change.

The eager eval exists for the OTHER consumer: the fused `flash_attention_paged_varlen_turboquant`
primitive binds pools RAW (`set_input_array`, `csrc/mfa_attention.cpp`) and genuinely needs them
materialized before dispatch. So the eval is structurally required there.

**Constraint — `tq_v` defaults to `True`:** with `tq_v=True`, `append()` also writes the packed-V
pools (`_v_pool_tq`, `_v_scales`) that the decode path does NOT read (decode uses fp16 V). Deferring
the eval there would leave an unbounded lazy-scatter chain on the unread pools (graph growth +
raw-read leak for any later fused call). The MLX per-eval round-trip floor is per-*eval*, not
per-buffer, so evaling only the unread pools saves nothing. → The eval is safely droppable **only
when `tq_v=False`** (decode reads every pool `append` writes).

## R.2/R.3 — Design + implement (correctness-first)

`append()` gains `defer_pool_materialize: bool = False` (skips the per-step eager eval). `step()`
sets it True **only** on `_decode_branch and not self.tq_v` — the case where the gather's `eval(o)`
provably materializes every written pool via the graph-input dependency. Default False = eager
(the safe public contract; standalone `append` callers and the fused/raw + tq_v=True paths keep
eager materialization). **No binding change, no kernel change, no path rerouted** (keep-all-paths).
The dependency is REAL (graph-input via `mx.fast.metal_kernel`), not a raw-pointer bypass.

## R.4 — Validation (the critical gate)

- **Output sanity / bit-identity (inverse-`add_temporary` gate):** the deferral is a pure
  materialization-ORDERING change → must be bit-identical to eager. Soak
  (`benchmarks/methodology/iv_d1_soak.py`, 200 steps, tq_v=False) compared the post-change deferred
  run to the pre-change eager reference **under concurrent-alloc churn across 5 independent process
  launches**: `max_abs_diff = 0.00e+00` (bit-identical), finite, **every run**. The III-9/V2
  nondeterministic-under-churn trigger found nothing — the graph dependency is truly expressed.
- **Path entered:** the gain bench confirms the per-step eval floor (~250us) is gone on the
  tq_v=False decode branch (R.5).
- **Edges:** soak spans 712 tokens (11+ blocks) — first/last step, block-boundary wraparound, all
  bit-identical. Default `tq_v=True` retains eager eval (guarded by a test).
- **Permanent regression guard:** `tests/test_iv_d1_tq_append_defer.py` — deferred vs forced-eager
  through `step()`, bit-identical over 40 steps under churn (2 tests).
- **Full suite green ×consecutive** (1820 pass incl. the 2 new IV-D1 tests; the 311 TQ/decode tests
  pass — default tq_v=True path unchanged).

## R.5 — Measured gain (Pattern #6, M5/26.6, 3 sessions, detached)

`benchmarks/methodology/iv_d1_bench.py` — TQ decode step, eager-append vs deferred-append,
tq_v=False, same ctx (only the append eval differs):

| S | eager step | deferred step | saved | speedup |
|---|---|---|---|---|
| 2048 | 640.1us | 389.2us | **250.9us (39.2% of step)** | **1.64×** |
| 4096 | 663.4us | 407.4us | **256.0us (38.6% of step)** | **1.63×** |

The ~251–256us saved matches the predicted ~240us MLX per-eval round-trip floor (+ the K-pack/
scatter materialization that now folds into `eval(o)` with no second floor). Deferred runs are
tight (388–412us across sessions); the eager path carried the extra eval's variance (630–770us).
**Net perf monotonically non-worse:** the change is TQ-decode-local (tq_v=False branch); default
tq_v=True, non-TQ decode, and large-N are untouched (suite green ×2 confirms).

## R.6 — Disposition

**SHIPPED (tq_v=False TQ decode): ~1.63× step, ~250us/step recovered** — soak-validated bit-identical
under churn × processes, permanent regression guard added. This lowers the TQ-decode latency for the
`tq_v=False` config (the pure-decode / accuracy-preferring choice, where V is kept fp16).

**Structural finding (default tq_v=True):** the per-step eval is **required** — the packed-V pools
are written-but-unread by decode, so deferral would leak. Recovering the floor for the default needs
the **lazy-packed-V-pool restructuring** (already a Marco-gated queue item: PHASE-III-CLOSE "Lazy
packed-V pool when only the III-2 decode path is used"). That is the extension path; honestly, the
floor stands for tq_v=True until then.

**Release:** the gain is additive and join-ready for the held **v2.56.0** (flag removal + V3
validation + this TQ-decode gain) — Marco's call on version. No tag/publish here.

## Phase IV backlog update

- **IV-D1: CLOSED** — shipped for tq_v=False (~1.63×, soak-validated); tq_v=True eval structurally
  required pending lazy-packed-V (IV-D2 below).
- **IV-D2 (new/promoted)** — lazy-packed-V pool: defer/materialize-on-read the `_v_pool_tq`/
  `_v_scales` writes so the per-step eval can also drop for the **default tq_v=True** decode
  (~250us/step for the common config). Same inverse-`add_temporary` discipline + soak required.
  This is now the top Phase IV decode lever.
