# v2.39.0 Option γ fused dK+dV — empirical results (outcome δ)

Sprint v2.39.0 Phase C.1.a (M3-HIGH-02).  Date: 2026-05-13.
Hardware: M5 Max 128GB, M5 NAX (gpu_family_gen=17), macOS 26.4.

## Verdict summary

**Outcome (δ) per blueprint decision tree**: fused kernel ships
**available but NOT auto-default**.  Correctness verified bit-identical
to split (RMSE=0 across all D=64 shapes); perf characterization showed
**25-33% regression vs split** at qL≥4096 on M5 Max.  Fused remains
opt-in via `MFA_V34_BWD_KERNEL=fused`.  Split path stays auto-default
(v2.38.1 D_vec precompute behavior preserved).

This is the honest scope outcome — `/metal-kernel-dev` audit predicted
~10% K-bandwidth-amortization win; empirical M5 Max bench shows
regression instead.  The investigation foundation now includes evidence
that fusion-for-bandwidth-amortization is NOT the optimization story
on Apple's M5 NAX hardware; the real perf characteristics differ from
the audit's mental model.  v2.39.1+ work can re-attempt fusion with a
different design (e.g., explicit tile-loading reorder, register-budget
control via `MFA_V6_MAX_THREADS`, smaller WM) informed by this finding.

## Empirical data (PUBLIC AUTO API, M5 Max, 3 sessions × 4w+12i)

| ID | qL | fused (ms) | split (ms) | SDPA (ms) | fused/split | fused/SDPA | split/SDPA |
|---|---|---|---|---|---|---|---|
| S1 | 2048 | 4.68 | 4.65 | 4.63 | 1.007× | 0.99× | 0.99× |
| S2 | 4096 | 13.79 | 9.22 | 17.50 | **0.668×** | 1.27× | 1.90× |
| S3 | 8192 | 54.85 | 37.18 | 68.84 | **0.678×** | 1.25× | 1.85× |
| S4 | 16384 | 231.24 | 169.89 | 284.45 | **0.735×** | 1.23× | 1.67× |

Fixed: B=2, H=8, D=64, fp16, non-causal.  Values are 3-session medians.
Variance ratios across sessions: max/min < 1.05 for all shapes (tight
reproducibility — this is not measurement noise).

### Per-session breakdown (fused/split speedup ratio)

| Shape | Session 1 | Session 2 | Session 3 | Median | Verdict |
|---|---|---|---|---|---|
| S1 qL=2048 | 0.992× | 1.001× | 0.990× | 0.992× | parity |
| S2 qL=4096 | 0.664× | 0.673× | 0.669× | 0.669× | -33% |
| S3 qL=8192 | 0.676× | 0.681× | 0.677× | 0.677× | -32% |
| S4 qL=16384 | 0.735× | 0.754× | 0.717× | 0.735× | -27% |

## Methodology

- **Public API entry**: `mx.grad(flash_attention(..., backend="auto"))`
- **Routing arms**:
  - `fused`: `MFA_ENABLE_V34_BACKWARD=1` + `MFA_V34_BWD_KERNEL=fused` (new kernel)
  - `split`: `MFA_ENABLE_V34_BACKWARD=1` + `MFA_V34_BWD_KERNEL=split` (v2.38.1 path)
  - `sdpa`:  `MFA_DISABLE_V34_BACKWARD=1` (SDPA-vjp baseline)
- 4 warmup + 12 timed iters, median ms reported.  `mx.eval` + `mx.synchronize()` after each iter.
- 3 sessions, separate Python processes, cooldown between runs.
- All variance ratios <1.05 → finding is reproducible, not noise.

## Why fused regressed at D=64 (hypotheses)

The `/metal-kernel-dev` audit predicted that K-bandwidth amortization
(loading K and V once per K-tile instead of twice across split-dV +
split-dK) would deliver ~10% speedup.  The empirical M5 Max regression
suggests one or more of:

1. **Register pressure causing partial spilling at D=64 too.**  The
   audit estimated 52 regs/lane at D=64 with WM=4 — well under the
   M5 cap of ~256 regs/lane for full occupancy.  But the actual
   Metal compiler may allocate registers more aggressively for the
   fused kernel (which has 2 persistent FP32 accumulators
   dV_accum + dK_accum, vs 1 in each split kernel).  Even partial
   spilling at the per-SG level could explain 25-33% regression.

2. **L1/L2 cache absorbs split's K-reload.**  M5 has substantial
   L1 (and L2 partitioned per-CU).  Split-dV and split-dK are
   *sequential* dispatches sharing the same K-tile in L1; the second
   kernel may hit L1 entirely.  The fused kernel's predicted memory-
   bandwidth saving may not materialize because there was no
   bandwidth penalty to begin with.

3. **Occupancy reduction at TG level.**  More registers per SG ×
   WM=4 SGs per TG = more total per-TG register pressure.  If this
   pushes the GPU from 2 TGs/CU co-resident to 1 TG/CU, the GPU's
   ability to hide memory latency drops dramatically — independent
   of L1 caching.  This is the classic "more arithmetic per kernel
   ≠ faster wall time" Apple Silicon trap.

4. **Q-tile is loaded twice within the fused kernel** (Qfrag for
   S = Q @ K^T, Qfrag2 for dK_accum += dS^T @ Q).  This matches the
   split-dK kernel exactly, so it's not a regression source — but
   it means the fused kernel does NOT save Q reloads, only K and V.
   The expected per-K-tile bandwidth saving is therefore smaller
   than the audit's 10% estimate suggested.

5. **Possible cause specific to MSL4 / M5 NAX scheduler**: the
   compiler may schedule the dV-mma block (which has the explicit
   `ORDER-CRITICAL` ordering constraint) suboptimally because of
   the extra register pressure from dV_accum being live across the
   subsequent dPtile compute.  This is the M5-NAX equivalent of
   register-pressure stalls.

Diagnosing the exact cause would require Metal frame capture +
register-pressure profiling, which is outside the Phase C.1.a budget.
Documented as a hypothesis set for v2.39.1+ if Marco wants to revisit.

## Conclusion + path forward

**Outcome δ ship strategy**: v2.39.0 architectural additions are
preserved (kernel + Primitive + binding + env var contract + helpers
infrastructure for future fusion work).  Auto-default routes to
**split** for D=64 (preserves v2.38.1 1.91× SDPA-vjp speedup).
Fused kernel ships as opt-in for:
- Users who want to characterize on their own workloads (different
  qL distributions, GQA factors, etc.).
- Future-sprint perf-tuning experiments without re-implementing the
  kernel from scratch.
- Researchers comparing the architectural design.

**No perf claim** in CHANGELOG for fused vs split.  The CHANGELOG
documents the negative finding honestly.  Investigation foundation
banked for v2.39.1+ work.

## Architectural value preserved

- Fused source generator (`createV34BackwardFusedDKDVSource`, ~440 LOC)
  is a reusable infrastructure component.  Future fused-kernel work
  (different blocking, different WM, D=128 with register-budget
  controls, TGP streaming reduction) can fork from this rather than
  re-implementing from split.
- `MFA_V34_BWD_KERNEL` env var contract introduced; documents the
  routing decision space.
- Test file `tests/test_v39_fused_dkdv.py` (17 tests) becomes the
  parity-verification harness for future fused-kernel iterations.

## Skill invocations log (per §AA.2)

| Phase | Skill | Status |
|---|---|---|
| C.1 pre-implementation | `/metal-kernel-dev` | ✓ Done (v39-0-option-gamma-design-status.md §1; MEDIUM go) |
| C post-implementation | `/mlx-debug-forensics` | ✓ Done — HIGH confidence SHIP (5-axis byte-equivalence; bit-identical RMSE=0) |
| D bench characterization | `/mlx-mfa-bench-methodology` | ✓ Done (blueprint adopted; 4 shapes, 3 sessions, public-AUTO API) |
| D perf claim audit | `/mlx-mfa-perf-audit` | N/A (outcome δ → no perf claim in CHANGELOG) |
| E pre-merge | `/mlx-code-review` | pending |
| E pre-tag canonical | `/mlx-mfa-release-audit` | pending |
| E release notes finalization | `/repo-release-prep` | pending |
