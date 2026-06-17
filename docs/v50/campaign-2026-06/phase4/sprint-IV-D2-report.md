# Sprint IV-D2 — Extend the Eval Collapse to the Default `tq_v=True`

**Date:** 2026-06-17
**Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `c93a03f` (clean, post-IV-D1), macOS 26.6 (25G5028f), Apple M5 Max 128GB,
mlx 0.31.2.
**Type:** extend IV-D1's per-step eval collapse from `tq_v=False` to the DEFAULT `tq_v=True` via a
combined eval. Same inverse-`add_temporary` class + the NEW fused-read-after-decode edge.
**Outcome: SHIPPED — default `tq_v=True` TQ decode ~1.36–1.39× / ~205–217us/step recovered,
bit-identical + fused-read-safe under churn across processes.**

## Skill invocations (§AA.2)

| Checkpoint | Skill | Outcome |
|---|---|---|
| Default-config decode bench | `/mlx-mfa-bench-methodology` | A/B eager(2-floor) vs combined(1-floor), 3 sessions, detached |

---

## R.1 — Mechanism confirmed

At `tq_v=True`, `append()` writes 5 pools: `_k_pool`, `_k_scales`, `_v_pool_fp16` (READ by the
gather decode path as `mx.fast.metal_kernel` graph-inputs → materialized by `eval(o)`), plus
`_v_pool_tq`, `_v_scales` (the packed-V pools — decode-UNREAD; read only by the fused
`flash_attention_paged_varlen_turboquant` primitive, which binds them RAW via `set_input_array`).
The second per-step eval floor IV-D1 couldn't remove for the default comes specifically from
materializing those two decode-unread packed-V pools. (`inference.py` append L985-988, eval
L1014-1017; fused args L1105-1107.)

## R.2/R.3 — Combined-eval design + implement (dependency proven)

Since the MLX per-eval floor is **per-eval, not per-buffer**, fold the decode-unread packed-V pools
into ONE combined eval at step end: `mx.eval(o, self._v_pool_tq, self._v_scales)`. `eval(o)`
materializes the read pools via their graph-input dependency; the packed-V pools are materialized
explicitly in the SAME round-trip. Implementation: IV-D1's `defer_pool_materialize` now fires on the
decode branch for BOTH `tq_v` values; the `tq_v=True` decode branch adds the combined eval before
returning `o`.

**Dependency proven (not assumed):** (a) the combined eval materializes packed-V EVERY step → no
unbounded lazy-scatter chain; (b) packed-V is concrete at step end → any later FUSED read (raw
binding) sees materialized data; (c) cross-step ordering preserved (`__setitem__` rebinds the pool
to a scatter on the prior-step materialized base). The fused path itself is **code-unchanged**
(eager append, no combined eval — `_defer_mat` is False for N_q>1 / opt-out). Keep-all-paths.

## R.4 — Validation (three-axis + two soaks)

- **Bit-identical cross-process soak** (`benchmarks/methodology/iv_d2_soak.py`, tq_v=True, 160 decode
  steps): post-change vs the PRE-change eager reference (captured via `git stash`), under
  concurrent-alloc churn, **across 5 independent process launches** → **decode `max_abs_diff =
  0.00e+00` every run.**
- **NEW fused-read-after-decode soak** (the edge IV-D1 lacked): the soak interleaves a FUSED call
  (reads packed-V raw) every 40 decode steps → **fused-read `max_abs_diff = 0.00e+00` every run.**
  The combined eval defeats the stale-read risk — the fused read sees the packed-V the decode
  steps' combined evals materialized.
- **Output sanity:** deferred == eager bit-identically (above); eager is the validated path.
- **Path entered:** the bench (R.5) confirms one floor, not two.
- **Edges:** soak spans 672 tokens (10+ blocks) + 4 interleaved fused reads — first/last, block
  wraparound, decode↔fused interleaving — all bit-identical.
- **Permanent regression guards** (`tests/test_iv_d1_tq_append_defer.py`): `tq_v=True` combined-eval
  bit-identity + fused-read-after-decode bit-identity (under churn).
- **Full suite green ×consecutive** (1821 pass).

## R.5 — Measured DEFAULT-config gain (Pattern #6, M5/26.6, 3 sessions, detached)

`benchmarks/methodology/iv_d2_bench.py` — tq_v=True step, eager (2 floors) vs combined (1 floor):

| S (tq_v=True default) | eager step | combined step | saved | speedup |
|---|---|---|---|---|
| 2048 | 776.6us | 559.3us | **217.3us (28.0%)** | **1.39×** |
| 4096 | 781.6us | 576.7us | **204.9us (26.2%)** | **1.36×** |

Recovers ~205–217us/step (the second MLX per-eval round-trip floor) on the config real users get
by default. (Smaller fraction than IV-D1's tq_v=False ~1.63× because the tq_v=True step is heavier —
it also packs V — so the single recovered floor is a smaller share.) Runs tight across sessions
(combined 549–590us). **Net perf monotonically non-worse:** the fused path is code-unchanged (eager
append, no combined eval) → latency unchanged by construction (fused-read soak confirms correctness);
`tq_v=False` (IV-D1), non-TQ, and large-N untouched; suite green ×2.

## R.6 — Disposition

**SHIPPED (default `tq_v=True` TQ decode): ~1.36–1.39× step, ~205–217us/step recovered** —
bit-identical + fused-read-safe under churn across processes, permanent guards added. Combined with
IV-D1, the per-step eval collapse now covers BOTH TQ configs:

| config | gain | floors |
|---|---|---|
| `tq_v=False` (IV-D1) | ~1.63× (~250us/step) | 1 → caller eval(o) |
| **`tq_v=True` default (IV-D2)** | **~1.36–1.39× (~205–217us/step)** | 2 → 1 combined eval |

**Release:** this makes the held **v2.56.0** substantial — `MFA_FORCE_NATIVE_BWD` removal + V3
auto-routing validation + **TQ-decode gain on the default config**. A real user-facing decode
improvement. Marco's call on version + go.

## Phase IV backlog update

- **IV-D2: CLOSED — shipped for the default `tq_v=True`** (~1.36–1.39×; bit-identical +
  fused-read-safe). The per-step eval floor is now recovered for both TQ-decode configs.
- IV-D3 (the deeper lazy-packed-V restructuring contemplated in IV-D1) is **no longer needed** for
  this gain — the combined eval recovers the floor without deferring the packed-V writes. (A future
  sprint could still skip packing V entirely on pure-decode workloads, but that is a separate
  memory/compute optimization, not the eval floor.)
