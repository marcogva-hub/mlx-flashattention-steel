# Audit Phase E — Complete M5 Re-Bench (every accumulated perf item resolved)

**Date:** 2026-06-18 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `1f088ea`, M5 Max, macOS 26.6, mlx 0.31.2. Pre-flight:
`benchmark-measurement-correctness` (PRIMARY). **MEASURE + DOCUMENT only** — no routing/kernel change
(Phase F). Discipline: lesson #15 (absolute ms + ratio + direction), Pattern #6 (which-binary
annotated, fingerprinted via the A/B/C locks), effective-FLOP plausibility-gated ≤51.8 TFLOPS,
3-replicate median (warm 8 / time 20). **Perf = Verified-at-2026-06-18, NOT executable-locked** (timing
is CI-flaky; the anti-drift for perf is re-measure, by design — unlike the locked correctness).

## The 6 items — measured answers

### 1. sparse V1 / V2 / SDPA crossover (B1 2³¹ open item) — V1 is NEVER fastest
Symmetric mask, d=0.25, B2 H8 (which-binary = env-forced `MFA_LCSA_KERNEL_VERSION`):
| D | N | work/2³¹ | V1 (scalar) | V2 (matmul2d) | SDPA | fastest |
|---|---|---|---|---|---|---|
| 64 | 512 | 0.01 | 0.34ms | 0.26ms | 0.26ms | SDPA≈V2 |
| 64 | 1024 | 0.03 | 0.31ms | 0.32ms | 0.31ms | tie |
| 64 | 2048 | 0.12 | 2.52ms | **0.52ms** | 0.56ms | V2 (4.9× vs V1) |
| 64 | 4096 | 0.50 | 7.03ms | **0.79ms** | 1.56ms | V2 (9.0× vs V1) |
| 128 | 2048 | 0.25 | 13.57ms | **0.69ms** | 1.05ms | V2 (19.5× vs V1) |
| 128 | 4096 | 1.00 | 50.29ms | **1.23ms** | 3.34ms | V2 (40.8× vs V1) |
| 128 | 8192 | 4.00 | 201.4ms | **3.43ms** | 13.31ms | V2 (58.8× vs V1) |
V2 eff 12.4–40.1 TFLOPS (≤51.8 ✓). **V1-scalar is never the fastest** (tied at tiny N, catastrophic
above). The `decide_auto_version` 2³¹ threshold routes D=64 (always <2³¹) + D=128 N<4096 to V1 — the
slow path (e.g. D=64 N4096: 7.03ms V1 vs 0.79ms V2 = 9× loss; D=128 N2048: 13.6 vs 0.69 = 19.5×).

### 2. symmetric-NAX-sparse (V2) vs SDPA at D=128 — the F justification CONFIRMED on current M5
N4096 B2 H8 (which-binary = v2, the default for this shape):
| density | V2-sparse | SDPA | ratio | eff TF |
|---|---|---|---|---|
| 0.0625 | 0.73ms | 3.03ms | **4.16× faster** | 11.8 |
| 0.125 | 0.87ms | 3.03ms | 3.49× | 19.8 |
| 0.25 | 1.20ms | 3.03ms | 2.53× | 28.7 |
| 0.50 | 1.88ms | 3.03ms | 1.61× | 36.5 |
| 0.75 | 2.58ms | 3.03ms | 1.17× | 39.9 |
| 1.00 | 3.25ms | 3.03ms | 0.93× SLOWER | 42.3 |
**Crossover ≈ d=0.78**: NAX-sparse beats SDPA for d ≲ 0.75. The F premise (route D=128 sparse →
symmetric NAX) **holds** — large win at low density.

### 3. dense STEEL vs SDPA on M5 — STEEL is M1–M4-legacy, SDPA wins
N4096 B2 H8, `backend="mfa"` (STEEL) vs `backend="auto"` (SDPA):
| shape (variant) | STEEL | SDPA | verdict |
|---|---|---|---|
| D=128 non-causal (V2) | 12.52ms | 3.06ms | SDPA **4.1×** faster |
| D=128 causal (V3) | 6.16ms | 1.80ms | SDPA **3.4×** faster |
| D=64 causal (V3) | 2.78ms | 0.81ms | SDPA **3.4×** faster |
**No STEEL variant beats SDPA on M5.** `backend="mfa"` is legacy-on-M5; the default `auto`→SDPA is
correct. (No routing change needed — default is already SDPA; document `backend="mfa"` as legacy.)

### 4. sage int8 worth — NOT worth on M5
D=128 N4096: sage int8 **14.51ms** vs SDPA **3.06ms** = **0.21× (4.7× slower)**, at cos ~0.997 quality.
The Python quantize overhead dominates; int8 is both slower AND lossier than fp16 SDPA here. Not worth
on M5 (consistent with the known "needs pre-quantized KV to win" caveat). Do not auto-route.

### 5. V5 reachability — DEAD
B2: ineligible at all tested shapes (env-gated `MFA_ENABLE_V5`, no eligible regime found). Document as
**compiled-but-unrouted**.

### 6. v3_min_N re-confirm — holds
D=64 causal N4096: V3 2.80ms vs V2 3.12ms = V3 **1.12× faster** → the `(D==64)?4096:2048` crossover
holds (within the `backend="mfa"` expert path — which itself loses to SDPA, item 3).

## Phase-F perf-fix targets (measured thresholds for F to route on)
1. **Route eligible sparse → V2 always; NEVER V1-scalar.** V2 wins at every measured work product
   (small N tie, large N 9–59×). Retire the 2³¹ V1/V2 threshold (set ~0 → V2 when eligible);
   sub-V2-eligible (asymmetric/causal/small-mask) → SDPA, not V1. Biggest single win: D=64 sparse
   (always V1 today → 9× faster as V2).
2. **D=128 sparse-API → symmetric NAX-sparse** (re-block the asymmetric default mask to symmetric):
   unlocks 4.16×@d=0.06 → 1.61×@d=0.5 vs SDPA. **Route to NAX-sparse for d ≲ 0.75; SDPA for d ≳ 0.78.**
3. **Document `backend="mfa"` as legacy-on-M5** (SDPA 3–4× faster); default auto→SDPA already correct.
4. **Do not auto-route sage int8 on M5** (4.7× slower); keep opt-in/expert.
5. V5 dead → document compiled-but-unrouted.

## F-premise revision? NONE
The symmetric-NAX-sparse win HOLDS on current M5 (item 2: 4.2× → 1.6× for d≤0.5). F's D=128-sparse
routing fix is justified. The V1-over-use (item 1) is a larger-than-expected win (D=64 9×). No premise
loudly revised.

## Disposition
Complete M5 perf truth measured on the now-known real dispatch — every number annotated with its
fingerprinted binary (the audit's payoff: no measuring SDPA-believing-it's-sparse). RESULTS.md
refreshed (verified-at-2026-06-18 banner). Perf is re-measure-not-lock (stated). No routing/kernel
change. Suite green. No orphans. Not tagged. **Phase F (the routing fix on these measured thresholds:
sparse→V2-not-V1, D=128-sparse→symmetric-NAX) is next, then Phase G (ship).**
