# v2.31.0 cross-session drift diagnostic — final report

**Sprint:** v2.32.0 drift diagnostic
**Branch:** `experiment/v32-drift-diagnostic` @ `224d039` + Phase A work
**Date:** 2026-05-06 18:10–18:45
**Hardware:** M5 Max (`applegpu_g17s`), macOS 26.5 (25F5068a)
**Author:** [CLAUDE]

## Executive summary

Phase A's PSO cache hypothesis (the audit's primary suspect for the
v2.31.0 → Phase 0 cross-session drift) is **REJECTED**. The GPU
ramp-up / P-state hypothesis is also REJECTED. Today's measurements
converge across 4 different bench configurations (Phase 0, A.1 cold-cache,
A.1 warm-cache, A.3.1 post-aggressive-warmup) on the same legacy
timings — within ±10% on every D=128 production shape. **v2.31.0's
slower numbers cannot be reproduced in this session.**

The drift is **not transient or manipulable**: it's a steady-state offset
between v2.31.0 measurement time (2026-05-06 02:48 AM) and now. The
cause is beyond session-feasible discrimination and requires multi-day
investigation.

**Recommendation: hold v2.32.0 release.** Ship architectural improvements
(Sprints 1, 2, 3, 5 + Sprint 4's BK=32 tile fix) only after multi-session
investigation establishes which regime is representative of user
experience. v2.31.0's published perf table is internally consistent with
its measurement-time conditions but **does not represent steady-state
user experience** in this session's regime.

## Phase A — measurements

### A.0 — Conditions inspection

Documented separately at
[`v32-drift-diagnostic-conditions.md`](v32-drift-diagnostic-conditions.md).
Key findings:

- **Metal PSO cache moved on macOS 26**: `~/Library/Caches/com.apple.metal/`
  is empty/obsolete; the actual Python-process cache is at
  `/var/folders/c2/<user-hash>/C/org.python.python/com.apple.metal/`.
  155 MB of cached pipelines existed before the test.
- **OS reboot between v2.31.0 measurement and Phase 0**: v2.31.0 bench
  at 02:48 AM, system rebooted ~12:04, Phase 0 bench at 13:24. Different
  cache state, possibly different power-state baseline.
- **Different cache files have different mtimes**: `*.list` files
  touched during Phase 0 bench; main `*.data` files built post-boot;
  overflow `*1.data` files from the day before. Cache rotation is
  active.

### A.1 — PSO cache discriminant test (subprocess-isolated, 5 runs/round)

Procedure:
1. Snapshot 155 MB cache → clear → verify 0 B
2. 180s cooldown
3. Cold legacy bench: SeedVR2-small + CogVideoX + SeedVR2-large
4. 30s cooldown
5. Warm legacy bench: same 3 shapes (cache populated by step 3)

Results:

| Shape | cold ms | warm ms | v2.31.0 ms | Phase 0 ms | cold/v2.31.0 | cold/Phase 0 | warm/Phase 0 | cold/warm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SeedVR2-small | 182.18 | 183.25 | 275.6 | 167.75 | **−33.9%** | +8.6% | +9.2% | −0.6% |
| CogVideoX | 2370.46 | 2332.98 | 3669 | 2344.0 | **−35.4%** | +1.1% | −0.5% | +1.6% |
| SeedVR2-large | 3886.55 | 3908.17 | 6780 | 3982.0 | **−42.7%** | −2.4% | −1.9% | −0.6% |

**Cold ≈ Warm on all 3 shapes** (Δ < ±2%). **Both ≈ Phase 0** (Δ < ±10%).
**Neither ≈ v2.31.0** (Δ −34 to −43% — very large, very systematic).

Cache rebuild during cold round only accumulated 232 KB (the small subset
of pipelines actually exercised by 3 shapes × 1 mode), so the JIT cost
was minimal. PSO state has no measurable impact on these timings.

**Verdict: PSO cache REJECTED.**

### A.3.1 — GPU ramp-up / P-state test

Procedure:
1. 60s cooldown
2. 30s sustained matmul workload (1.2M iters of 4096×4096 fp16) to push
   GPU to highest P-state
3. Immediately bench SeedVR2-small legacy, 5 runs

Result: **185.25 ms**.

Comparison:

| Configuration | SeedVR2-small ms | Δ vs Phase 0 | Δ vs A.1 warm |
|---|---:|---:|---:|
| Phase 0 R1 (legacy) | 162.13 | — | −12% |
| Phase 0 R3 (legacy) | 173.37 | — | −5% |
| A.1 cold | 182.18 | +9% | −0.6% |
| A.1 warm | 183.25 | +9% | — |
| **A.3.1 post-warmup** | **185.25** | **+10%** | **+1.1%** |
| v2.31.0 average legacy | 275.6 | +64% | +50% |

A.3.1 is within ±2% of the A.1 warm round. The aggressive 30s warmup
did NOT bring timings any closer to v2.31.0's regime. **Ramp-up
hypothesis REJECTED.**

### A.2 — thermal regime (skipped)

Phase A.2 needs `sudo powermetrics`. After A.1 + A.3.1 already produced
clear, consistent results across 4 different bench configurations, the
discrimination value of A.2 is low: it would tell us the GPU's current
P-state and thermal headroom, but we already have empirical evidence
that P-state activation (A.3.1) doesn't move the timing. Deferred to
multi-session work.

### A.4 — complementary tests (skipped)

`sw_vers` already captured in A.0 (26.5 25F5068a, current). Memory
pressure: A.0's `vm_stat` showed no compressor activity, ample free
pages (~67 GB free) — no memory pressure factor. MLX-side caching:
A.1's cold-vs-warm discrimination on the OS PSO cache transitively
covers MLX-internal caching too (any MLX cache that mattered would have
shown up as cold/warm divergence; none did).

## Phase B — synthesis

### What we now know

| Hypothesis | Status | Evidence |
|---|---|---|
| PSO compilation cache | **REJECTED** | A.1 cold ≈ A.1 warm on all 3 shapes; both match Phase 0, neither matches v2.31.0 |
| GPU ramp-up / P-state | **REJECTED** | A.3.1 post-aggressive-warmup ≈ A.1 warm; +10% from Phase 0 (within noise) |
| Cooldown sensitivity | **REJECTED** | 180s, 60s, 30s, and post-bench cooldowns all produce ≈ same legacy timings (Phase 0 + A.1 + A.3.1) |
| MLX-side cache | **REJECTED transitively** | If MLX cache mattered, it would have shown in cold-vs-warm A.1 |
| macOS update mid-day | **UNLIKELY** | sw_vers identical (26.5 25F5068a) at A.0; v2.31.0 release was earlier same day |
| Hardware (different chip / firmware) | **REJECTED** | `applegpu_g17s` confirmed, no hardware swap |
| Memory pressure | **REJECTED** | vm_stat shows no compression, no swap |

### What we still don't know (requires multi-session)

- **Was the v2.31.0 measurement valid for steady-state, or an
  artifact of the system's state at 02:48 AM?** The reproducibility
  of today's faster regime (4 separate measurement configurations all
  converging) suggests v2.31.0 was the outlier. But we cannot prove
  this without measuring at a future 02:48 AM (or in similar deep-idle
  conditions).
- **Long-overnight-idle effects.** The audit's hypothesis about GPU
  in deep low-P state after long idle remains untested. A 30s warmup
  doesn't replicate it; an actual idle-then-bench on cold-boot
  morning might.
- **Multi-day natural variance.** What's the stable baseline? Today's
  numbers across 4 bench configurations are within ±10%. Are they
  always within ±10% across days? Or do they drift to v2.31.0's
  regime sometimes?
- **macOS softwareupdated background activity.** Could the v2.31.0
  measurement have coincided with a Spotlight indexing burst, a
  background `softwareupdated` check, or another sustained
  GPU consumer that we never observed?

### Why this matters for v2.32.0

v2.31.0's published perf claims:

| Shape | v2.31.0 V6NAX ms | v2.31.0 legacy ms | v2.31.0 V6NAX vs SDPA |
|---|---:|---:|---:|
| SeedVR2-small | 170.92 | 265.13 | 0.890× |
| CogVideoX | 2399.19 | 3610.79 | 1.033× |
| SeedVR2-large | 4042.73 | 6776.12 | 1.008× |

These claims are **published on PyPI right now** (v2.31.0). They were
measured under conditions we cannot reproduce. Phase 0 and Phase A
together establish that, under steady-state conditions today, V6NAX's
gain over legacy is much smaller than v2.31.0 claimed (or in
SeedVR2-small's case, inverted).

The release-decision implications were articulated by the prompt as
options a/b/c/d. With Phase A's findings:

- **(a) Yank v2.31.0** — nuclear option, only justified if v2.31.0 is
  *wrong*. Phase A doesn't establish wrongness, just non-reproducibility
  in current regime. Not recommended.
- **(b) Publish v2.31.1 perf-correction addendum** — viable, requires
  multi-session investigation first to know what to write.
- **(c) Recalibrate in v2.32.0** — viable, same prerequisite.
- **(d) README disclaimer** — minimum-effort path, but the prompt's
  spirit is "understand before publishing", which (d) doesn't honor.

### Recommendation to Marco (decision required)

1. **Hold v2.32.0** until multi-session investigation. Phase A has done
   what one session can do. The remaining hypotheses (deep-idle effects,
   multi-day variance, background-activity coincidence) need timing
   discipline beyond a single afternoon.
2. **Document Phase A in the v2.31.0 PyPI page or CHANGELOG**:
   "Performance numbers in v2.31.0 release notes were measured on
   2026-05-06 02:48 AM under conditions that cannot be reproduced in
   subsequent benches; investigation ongoing." This is honest and
   safe — readers who try to reproduce v2.31.0's numbers and get
   smaller V6NAX gains have a context for the discrepancy.
3. **Schedule multi-session bench protocol**: 3-5 sessions over 1-3
   days, varying:
   - Time of day (early AM after long idle vs afternoon mid-activity)
   - Pre-bench state (cold-boot morning vs mid-day after sustained activity)
   - Background load (clean idle vs simulated browser/Spotlight load)

   For each session, capture: time of day, uptime at bench start,
   cache state before clear (size and mtime range), GPU freq via
   `sudo powermetrics`, the standard A/B/A wrapper output.

4. **In parallel**: ship the architectural improvements (Sprint 1
   causal port, Sprint 2 LSE writeback, Sprint 3 align FCs) as
   **bug-fix-only v2.31.1** with no perf claims. These are independent
   of perf measurement and will land cleanly. They're the part of
   the V6NAX-FORWARD-MAX work that is unambiguously valuable regardless
   of how the perf-claim question resolves.

### Methodology additions for `CLAUDE_V6_NAX.md`

To prevent this category of issue in the future, add Artifact #5 with
two sub-points:

1. **Metal PSO cache path is per-app on macOS 26+** —
   `/var/folders/<user_dir_hash>/C/<bundle-id>/com.apple.metal/`, not
   `~/Library/Caches/com.apple.metal/` (which is empty/obsolete on
   macOS 26).
2. **Marketing-grade benchmarks need cross-session, multi-condition
   repro before publication**. A single bench session — even a
   well-controlled cross-session A/B/A — can lock in a regime-specific
   measurement. Before publishing perf claims:
   - Same-shape, same-mode bench across 3+ separate sessions (different
     times of day, different pre-bench states)
   - Document Metal cache state, hardware uptime, macOS version,
     background load (`GPU Active <5%`) for each session
   - Use the median of session medians, not within-session statistics

## Files

- `outputs/diagnostic/system-state-now.txt` — A.0 system snapshot
- `outputs/diagnostic/git-history.txt` — A.0 timeline reconstruction
- `outputs/diagnostic/phase-a1-pso-aba.log` — A.1 raw bench log
- `outputs/diagnostic/a1-cold-*.json`, `a1-warm-*.json` — A.1 raw data per shape
- `outputs/diagnostic/phase-a3-warmup-test.log` — A.3.1 raw bench log
- `outputs/diagnostic/a3-postwarmup-seedvr2small.json` — A.3.1 raw data
- `bench/v32_pso_cache_aba.sh` — A.1 wrapper script
- `bench/v32_pso_analyze.py` — A.1 discriminant analyzer
- `docs/v6-nax/v32-drift-diagnostic-conditions.md` — A.0 conditions doc
- `docs/v6-nax/v32-drift-diagnostic-report.md` — this report

## Decision needed from Marco

The diagnostic has reached its session-feasible limit. To proceed:

1. **Approve the multi-session protocol** (3-5 sessions, varied conditions),
   and Claude/Codex will execute it across the next 1-3 days.
2. **Approve a v2.31.0 PyPI addendum** explaining the measurement
   non-reproducibility (option above).
3. **Approve v2.31.1 bug-fix-only release** (Sprints 1, 2, 3 +
   Sprint 4 BK=32 fix, no perf claims) as a parallel deliverable.
4. **Approve `CLAUDE_V6_NAX.md` Artifact #5 addition** as documented above.

No code changes, no version bump, no PyPI publish in this session.
The output is the empirical evidence above + the proposed next steps.
