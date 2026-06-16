# Sprint III-11 — M5 Max Benchmark Re-Run (macOS 26.6, HEAD 6a7d79c)

**Date:** 2026-06-16
**Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `6a7d79c` (clean), macOS **26.6** (build 25G5028f), Apple M5 Max 128 GB,
mlx 0.31.2, mlx_mfa 2.52.1. Methodology: `/mlx-mfa-bench-methodology` dispatcher, 3 sessions,
subprocess isolation, strict 4s-cooldown for ≥1.5 ms shapes.

## Headline outcome
1. **Production perf is non-worse vs v2.52.1** — the Phase III fixes are perf-neutral (correctness
   clamps + lifetime registration); the kernels are byte-identical to v2.52.1, so v2.52.1 on 26.6
   would measure the same. **No regression. The release is perf-safe.**
2. **BUT every inherited v2.50-era headline speedup is materially LOWER on macOS 26.6** — a
   systematic effect: Apple improved their primitives (SDPA-vjp, conv) between the OS the claims
   were measured on and 26.6, shrinking mlx-mfa's relative advantage. This is **not a regression**
   (mlx-mfa is non-worse), but the **published numbers are stale and over-optimistic on 26.6**.
3. **The numbers are high-variance + qL/regime-dependent at these kernel sizes** (clock-state
   bimodality, §4.3) — single-ratio claims swing ±30–40% run-to-run and are unreliable without a CI.
   (The bench infrastructure was FIXED to complete the full strict re-bench — see the
   **R.2/R.3 FULL CLEAN RE-BENCH** section below, which supersedes the "impasse" note. The
   "blocked" framing in that note is historical; the detached runner resolved it.)

## Measured data (26.6, B=2 H=8 unless noted)

### Forward attention vs SDPA (production / auto path)
| shape | ratio | verdict |
|---|---|---|
| D64 c N4096 fp16 | 0.98× | CONFIDENT |
| D128 c N4096 fp16 | 1.05× | HIGH_VAR |
| D128 nc N4096 fp16 | 0.98× | CONFIDENT |
| D128 c N4096 bf16 | 1.00× | HIGH_VAR |
| D256 c N4096 fp16 | 0.99× | HIGH_VAR |
→ Forward ≈ SDPA (0.98–1.05×). Production auto-dispatch routes forward to SDPA correctly;
net-non-worse. (Forward is documented as "bit-identical to SDPA", not a speedup claim.)

### V34 backward vs SDPA-vjp (strict 4s cooldown)
| shape | measured 26.6 | README claim |
|---|---|---|
| D64 causal N2048 | 1.29× (CONFIDENT) | 2.06–2.58× |
| D64 causal N4096 | 2.00× (CONFIDENT) | 2.06–2.58× |
| D64 causal N8192 | 1.94× (BOUNDARY) | 2.06–2.58× |
| D64 non-causal N4096 | 1.29× (CONFIDENT) | 1.72–2.01× |
| D64 non-causal N8192 | 1.88× (CONFIDENT) | 1.72–2.01× |
| D128 causal N4096 | 0.99× (CONFIDENT) | (broadened v2.50) |
| D128 causal N8192 | 1.00× (CONFIDENT) | (broadened v2.50) |
→ **D=64 backward ~1.3–2.0× (was 2.06–2.58×); D=128 backward now BREAK-EVEN (~1.0×).** Materially down.

### conv MPP vs legacy (T8/T16 64×64 C128)
| regime | T8 fp16 | T16 fp16 | README claim |
|---|---|---|---|
| cold-strict (4s cooldown) | 0.98× (CONFIDENT) | 1.30× (HIGH_VAR) | 2.3–2.5× |
| warm (2s, contaminated) | 1.71× (BOUNDARY) | 1.99× (HIGH_VAR) | 2.3–2.5× |
→ conv MPP is **0.98–1.3× cold, ~1.7–2.0× warm — far below 2.3–2.5× in every clean regime on 26.6**,
and high-variance. The bf16 "1.4–2.7× vs legacy" claim is **unmeasurable as documented** — the legacy
im2col path is fp16-only (KD-7), so bf16 has no "legacy" baseline (needs `mx.conv_general`).

### Not yet measured
TurboQuant paged decode (6.0–14.4×), D=256 causal inversion, LCSA mask-build 15.4×. (Decode is a
different memory-bound regime that may be more OS-stable, but its harness wasn't set up before the
impasse below.)

## The impasse (honest)
"Full clean re-bench of all claims to CONFIDENT" is **not achievable in this environment**:
- The background-job kill limit (~5–8 min) prevents the full strict protocol (8 timed iters × 4s
  cooldown × 3 sessions × target+baseline) at the ≥10 ms shapes; reduced protocols (5 iters) yield
  HIGH_VARIANCE at T16/D128/D256 (the clock-state bimodality §4.3 warns is "meaningless without a CI").
- conv is strongly regime-dependent (cold 1.0× vs warm 1.7×); production VAE convs run warm/back-to-
  back, so the warm regime is representative — but a clean warm-continuous CI at ≥10 ms also needs
  long runs.
- The 2s-cooldown numbers that looked like 1.7–2.0× were thermal-variance-contaminated (the exact
  v2.36.0 incident the methodology exists to prevent) — so the *original* 2.3–2.5× claim may itself
  have been a non-canonical point estimate (cf. the v2.37.x BK=16 "37%" artifact).

## What is solid regardless
- Release is **perf-safe** (non-worse vs v2.52.1; correctness fixes are perf-neutral).
- The documented v2.50-era headline speedups **do not hold on 26.6** (V34 backward + conv both
  confirmed materially lower; systematic Apple-baseline improvement).

## Recommendation (Marco-gated)
Re-characterizing every claim to CONFIDENT on 26.6 is a **dedicated perf sprint** needing bench
infrastructure that survives long strict runs (a detached/uninterrupted harness) — it is not a quick
pre-tag confirmation, and it should not hold the **default-reachable GNA correctness fix** hostage.
Options: (a) soften the headline perf language to OS-honest ranges + ship correctness now, re-bench
properly as a fast-follow; (b) invest in the dedicated perf sprint with fixed infrastructure first.
The correctness gate (III-9/III-10) is MET; the release is correctness-safe today.

---

## R.2/R.3 FULL CLEAN RE-BENCH (infra fixed) — authoritative 26.6 table

Bench infrastructure fixed (`benchmarks/methodology/robust_pool_runner.py`: detached `nohup` +
incremental JSONL + per-spec subprocess isolation + idempotent resume — survives the harness
kill limit that blocked the earlier attempts). Full 24-spec catalog ran ~1h detached, **0 errors**,
strict 4s-cooldown protocol, 3 sessions. Raw: `benchmarks/methodology/iii11_26.6_results.jsonl`.

### Forward attention vs SDPA (production / auto path)
All shapes 0.77–1.08× SDPA (the <0.9 and >1.05 outliers are HIGH_VARIANCE clock-state bimodality,
not real deviations). **Production forward ≈ SDPA; net-non-worse vs v2.52.1. Confirmed.**

### V34 backward vs SDPA-vjp (strict)
| shape | 26.6 measured | verdict | README claim |
|---|---|---|---|
| D64 causal N2048 | 0.86× | BOUNDARY | 2.06–2.58× |
| D64 causal N4096 | 1.49× | CONFIDENT | 2.06–2.58× |
| D64 causal N8192 | 2.52× | CONFIDENT | 2.06–2.58× |
| D64 non-causal N4096 | 1.39× | CONFIDENT | 1.72–2.01× |
| D64 non-causal N8192 | 2.09× | CONFIDENT | 1.72–2.01× |
| D128 causal N4096 | 0.97× | CONFIDENT | (broadened v2.50) |
| D128 causal N8192 | 0.99× | CONFIDENT | (broadened v2.50) |
→ **D=64 backward is strongly qL-dependent**: break-even at N=2048, ~1.4–1.5× at N=4096, ~2.1–2.5×
at N=8192. The 2.06–2.58× claim holds only at the largest qL. **D=128 backward is break-even** (~1.0×,
reliable across runs). Also: the same spec swings ±30–40% run-to-run (e.g. N4096 causal 1.49× here
vs 2.00× in a prior run) — single precise numbers are not reproducible at these sizes.

### conv MPP
| shape | 26.6 measured | verdict | README claim |
|---|---|---|---|
| T8 fp16 vs legacy | 1.35× | HIGH_VAR | 2.3–2.5× |
| T16 fp16 vs legacy | 1.22× | CONFIDENT | 2.3–2.5× |
| T8 bf16 vs conv_general | 1.06× | HIGH_VAR | 1.4–2.7× |
| T16 bf16 vs conv_general | 1.09× | HIGH_VAR | 1.4–2.7× |
→ conv fp16 **~1.2–1.35×** vs legacy (was 2.3–2.5×); conv bf16 **~1.0×** vs `mx.conv_general` (the
documented "vs legacy" baseline doesn't exist for bf16 — legacy im2col is fp16-only, KD-7).

### Not measured (separate blockers)
- **TQ paged decode (6–14×)**: `bench_turboquant_full.py` crashes in `_build_tq_pool` (a bench-harness
  indexing bug, not a kernel bug) before producing numbers. Decode is memory-bandwidth-bound (3-bit
  KV compression) — a different regime that may be more OS-stable — but it needs a harness fix to
  re-measure. **Flagged for a fast-follow.**
- **LCSA mask-build 15.4× (11.19→0.73 ms)**: a *historical* build-time improvement; the old impl is
  gone, so only the current absolute build time is measurable, not the ratio.

## Synthesis
On macOS 26.6, Apple's primitives (SDPA-vjp, conv) improved, **systematically shrinking mlx-mfa's
documented v2.50-era compute-bound speedups** — V34 backward and conv both materially lower, qL-
dependent, and high-variance/run-unstable at these sizes. This is **not a regression** (kernels
byte-identical to v2.52.1; production forward ≈ SDPA; release perf-safe), but the published precise
ranges are stale and not reproducible on 26.6. **Doc action:** replace precise ranges with measured,
qL-qualified, OS-caveated values (below); re-measure TQ decode after a harness fix.
