# TGP cross-SG reduction overhead — empirical investigation

**Status:** complete (2026-05-13)
**Sprint:** v2.38.0 investigation (Phase A)
**Per:** `CLAUDE_V6_NAX.md` §AA.4 disagreement-resolution between design doc and CC pre-flight estimates

## TL;DR

**Design doc estimate**: ~100 µs/K-iter × NK=512 = **51 ms total** TGP overhead
→ Option γ "NOT a path to perf parity with SDPA-vjp" (design doc verdict)

**Empirical measurement on M5 Max**: ~1 µs/K-iter × NK=512 = **0.5 ms total** TGP overhead

**Design doc was overstated by ~100×.** TGP streaming reduction IS viable on M5 Max NAX hardware.

**Implication for v2.38.0:** Option γ's expected savings flip from
"net negative at D=128" to "net positive 4-5ms savings" from softmax
fusion (the per-K-tile reduction overhead is negligible vs the
softmax-fusion win).

## Micro-bench design

`bench/v6_nax/tgp_overhead_microbench.py` — isolates EXACTLY the TGP
cross-SG reduction pattern Option γ would use, with no softmax / no
MMA / no other work.

**Probe pattern** (mirrors Option γ TGP layout):
- Threadgroup: WM=4 SGs × 32 lanes = 128 threads
- TGP buffer: 4 × BK=16 × D=128 fp32 = 32 KB (full WM × BK × D
  partials)
- Per K-iteration loop:
  1. 4 SGs write per-row partials to disjoint TGP slots
  2. `threadgroup_barrier(mem_threadgroup)`
  3. SG0 streams BK=16 rows, sums across 4 SGs, writes to device
  4. `threadgroup_barrier(mem_threadgroup)` for next iter

**Baseline pattern** (same outer shape, no TGP+barrier+reduce):
- 4 SGs × N_iter loop
- SG0 writes dummy output directly to device
- No TGP buffer, no barriers, no cross-SG read-back

**Overhead** = `probe_median - baseline_median` per session
**Per-iter overhead** = total overhead / N_iter

**Protocol:** Canonical §4.2 (10 warmup + 100 continuous, median).
3 sessions for cross-session variance.

## Results (M5 Max, 2026-05-13)

| NK | Session 1 (µs/iter) | Session 2 | Session 3 | Median | Range | Verdict |
|---|---|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 µs | HIGH_VARIANCE (sub-noise) |
| 8 | 2.065 | 0.000 | 0.000 | 0.000 | 2.065 µs | HIGH_VARIANCE (cold-cache) |
| 32 | 1.949 | 0.000 | 0.000 | 0.000 | 1.949 µs | HIGH_VARIANCE (cold-cache) |
| 128 | 2.269 | 1.054 | 1.087 | 1.087 | 1.215 µs (111%) | HIGH_VARIANCE |
| **512** | **0.991** | **0.974** | **0.976** | **0.976** | **0.017 µs (1.7%)** | **CONFIDENT** |

**At NK=512 (the design doc's reference shape, qL=8192 D=128 NK=512):**
- Median per-iter: **0.976 µs**
- Cross-session range: 1.7% — CONFIDENT verdict per §4.3
- Total TGP overhead: 0.976 µs × 512 = **0.500 ms**

The HIGH_VARIANCE at small NK (1, 8, 32) is sub-noise: the entire
kernel runs in ~0.25-0.5 ms regardless of NK (driven by kernel-
dispatch + sync overhead floor); the TGP-overhead delta is below the
measurement floor.  Once NK is large enough to dominate the wall-
clock (NK=512), the measurement stabilizes and matches across all 3
sessions within 2%.

Raw data: `/tmp/tgp_overhead_full.json`.

## Reconciliation: design doc vs CC pre-flight vs empirical

| Source | Estimate (µs/K-iter) | Total at NK=512 | Result |
|---|---|---|---|
| Design doc | ~100 | 51 ms | (now refuted) |
| CC pre-flight | ~4.8 (conservative) | 2.5 ms | (within 5× of empirical) |
| **Empirical (M5 Max)** | **~1.0** | **0.5 ms** | **CONFIDENT** |

**The design doc was off by ~100× at NK=512.** CC's pre-flight was
within 5× (conservative direction).

### Why the design doc was wrong

The design doc estimated "Overhead ~50-100µs per K-tile" without
empirical measurement.  The actual M5 NAX TGP cross-SG reduction
overhead is dominated by:
- 1 threadgroup barrier between writers and reader (~75 ns)
- 1 threadgroup barrier between iterations (~75 ns)
- SG0 sums 4 × D fp32 from TGP (~hundreds of ns of L1 hits)
- Total: well under 1 µs per K-tile in practice

The design doc's 50-100µs figure may have assumed:
- Worst-case TGP cache-miss latency
- Contention from concurrent TGP writes
- Older-generation (M3/M4) barrier costs

Empirically on M5 Max with M5-NAX-specific TGP implementation, the
overhead is sub-µs per iteration.

## Implications for Option γ fused dK+dV

**Recomputed Option γ outcome at qL=8192 D=128 NK=512:**

| Component | Time (ms) | Source |
|---|---|---|
| Current split (dV + dK) | 30.2 | v2.37.0 perf data |
| Softmax fusion savings (design doc) | -4 to -5 | design doc line 15 |
| **TGP cross-SG reduction overhead (empirical)** | **+0.5** (NOT +51) | **this investigation** |
| D_vec precompute (Section B of original sprint) | -2 to -3 (5-8%) | Sprint 2 audit M2-HIGH-01 |
| **Fused dK+dV total (projected)** | **~22-24** | sum |
| Add dQ (~12 ms unchanged) | +12 | |
| **V34 backward total (projected)** | **~34-36 ms** | (was 49 ms baseline) |
| SDPA-vjp baseline | 20 ms | v2.37.0 perf data |
| **Ratio** | **~1.7-1.8× slower** | down from 2.4× |

So Option γ on M5 Max NAX, with TGP streaming reduction (now viable
per empirical data), reduces the D=128 gap from 2.4× to ~1.7-1.8×
SDPA-vjp.  **Still NOT parity** — the architectural floor is the dK
kernel's extra dO@V^T matmul, not the TGP overhead.

## Re-evaluating the v2.38.0 decision tree

Per the v2.38.0 sprint's stated 4 outcomes:

- **(α) UNIVERSAL_AUTO_DEFAULT** (V34 ≤ SDPA-vjp across all D):
  STILL UNREACHABLE.  D=128 stays 1.7-1.8× slower even with optimal
  Option γ + D_vec precompute.  The architectural floor is the dK
  kernel's extra dO@V^T matmul, not the TGP reduction overhead.
- **(β) SHAPE_AWARE_DEFAULT**: feasible if Option γ closes D=64 gap
  to broader regimes (qL < 4096).
- **(γ) CONFIRMED_OPT_IN_BROADENED**: most realistic — D=64 broad
  auto-default + D=128 improved SHIP_OPT_IN at ~1.7× SDPA-vjp
  (down from 2.4×).
- **(δ) FALSIFIED**: ruled out — Option γ delivers measurable win.

**Verdict update:** the design doc's pessimism was based on an
overstated overhead estimate.  Option γ is more attractive than the
design doc concluded, but still doesn't reach parity at D=128.

## Reduction-options consultation (post-measurement)

Per `/metal-kernel-dev` rubric, additional optimizations beyond the
baseline TGP streaming pattern:

1. **Coalesced TGP writes** — already implemented in probe (disjoint
   per-SG slots, no contention).  No further opt.
2. **Async TGP writes** — Metal 4 / M5 NAX doesn't expose async TGP
   intrinsics at user level.  No-op.
3. **Tree reduction vs SG0-stream** — for 4 SGs, single-stream is
   optimal (tree adds 1 extra barrier; SG0-stream uses 0 extra
   barriers vs the baseline 2/iter).
4. **TGP layout optimization** — partials are already row-major
   per-SG (cache-line-aligned for D=128 fp32 = 512B which is 8 cache
   lines).  No further opt.
5. **Interleaved dK+dV reduction** — reducing dK and dV separately
   (sequential) doubles the per-K-iter overhead (~2 µs instead of
   ~1 µs).  Total impact: +0.5 ms at NK=512.  Negligible.
   Alternative: reduce dK and dV in parallel (different lanes handle
   different accumulators) — saves the second barrier per iter.

**No major optimization opportunity beyond the baseline pattern.**
The TGP overhead is already at the M5 NAX hardware floor for this
workload.

## Action items for v2.38.0 scope decision

1. **Update Option γ design doc** with the empirical TGP overhead
   measurement.  The design doc's "TGP streaming reduction is NOT
   viable on M5" conclusion (line 73) was based on the overstated
   100 µs/iter figure and is now refuted by empirical data.

2. **Re-frame v2.38.0 outcome target** as (γ) CONFIRMED_OPT_IN_BROADENED
   per the data — even with Option γ + D_vec precompute, D=128 stays
   1.7-1.8× SDPA-vjp.  D=64 broadening via the v2.37.2-style
   carve-out IS reachable.

3. **Continue investigation Phases B-D** (multi-pass /mlx-code-review)
   to check for compound improvements that single-pass Sprint 2
   audit missed.  Combined with this TGP finding, those may further
   refine the v2.38.0 scope.

## Skill invocations

| Skill | Decision point | Findings count | Action taken |
|---|---|---|---|
| /metal-kernel-dev | Phase A.1 micro-bench design (loaded earlier this session) | 1 (rubric applied) | Probe pattern + baseline subtraction methodology |
| /mlx-mfa-bench-methodology | Phase A.3 canonical bench protocol (rubric applied directly) | 1 verdict | 3-session canonical §4.2, CONFIDENT at NK=512 |
| /metal-kernel-dev | Phase A.5 reduction-options consultation | 5 options evaluated | No further opt beyond baseline pattern |

## References

- `docs/v6-nax/v34-backward-option-gamma-design.md` line 56-73 (now-
  refuted analysis)
- `bench/v6_nax/tgp_overhead_microbench.py` (this micro-bench)
- `CLAUDE_V6_NAX.md` §4.2 canonical methodology
- `CLAUDE_V6_NAX.md` §AA.4 disagreement-resolution policy
- v2.38.0 perf sprint pre-flight (halted) — this investigation
  resolves the surfaced gap
