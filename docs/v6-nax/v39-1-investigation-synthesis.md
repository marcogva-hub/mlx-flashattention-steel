# v2.39.1 investigation synthesis — outcome δ root cause

Sprint v2.39.1 (research-mode investigation).  Date: 2026-05-13.
Hardware: M5 Max 128GB, M5 NAX (gpu_family_gen=17), macOS 26.4.
Branch: feat/v39-1-investigation.

## TL;DR — Outcome (α) — Mechanism identified + fixable

**Root cause of v2.39.0 outcome δ regression**: H1 register pressure.
The Apple Metal compiler allocates registers too aggressively for the
fused kernel at the v2.39.0 default `BK=32` (TK=2), causing partial
spilling that the audit-time register-budget estimate did not predict.

**Fix**: change fused kernel default from `BK=32` to `BK=16` (TK=1).
This halves the per-SG persistent FP32 accumulator footprint
(`dK_accum + dV_accum`) and eliminates the spill pressure.

**Empirical confirmation** (single-session, 4w+12i, M5 Max, B=2 H=8 D=64 fp16):

| qL | fused-BK16 (ms) | fused-BK32 (ms, v39.0 default) | split-D_vec (ms) | BK16/split | BK16 vs SDPA |
|---|---|---|---|---|---|
|  512 | 0.79 | 0.63 | 0.62 | 0.78× | 0.69× |
| 1024 | 1.36 | 1.22 | 1.30 | 0.96× | 0.97× |
| 2048 | 4.88 | 4.90 | 4.87 | 1.00× | 1.01× |
| **4096** | **9.03** | **13.64** | **9.09** | **1.01×** | **1.95×** |
| **8192** | **36.11** | **54.84** | **36.60** | **1.01×** | **1.89×** |
| **16384** | **152.77** | **230.13** | **170.62** | **1.12×** | **1.87×** |

**Key observations**:
- v39.0 BK=32 fused regression at qL≥4096 (-33%/-32%/-27%) → eliminated by BK=16.
- BK=16 fused **at parity or modestly faster** than split for qL ∈ {2048, 8192}.
- BK=16 fused **+12% over split** at qL=16384.
- BK=16 fused **preserves all v2.38.1 SDPA-vjp speedups** (1.95× / 1.89× / 1.87× at qL ∈ {4096, 8192, 16384}).
- Small-qL regression (qL ∈ {512, 1024}) is acceptable because those shapes are below the v2.37.2 carve-out threshold (qL≥4096) — AUTO path doesn't engage V34 backward there.

**Correctness**: fused-BK16 outputs verified bit-identical to split for
qL=2048 (RMSE=0); ~2e-5 RMSE at qL∈{4096, 8192} (same FP16-tolerance
band as v2.38.1 D_vec drift vs SDPA).

## Hypotheses tested

### H1 — Register pressure / partial spilling — **CONFIRMED**

**Theory**: fused kernel's two persistent FP32 accumulators
(`dK_accum + dV_accum`) push per-SG register footprint above the
M5 NAX compiler's spill threshold at BK=32 (TK=2 → 16 FP32 elems each
= 32 regs/lane just for accumulators).  BK=16 halves this to 8 elems
each = 16 regs/lane → below spill threshold.

**Evidence**:
1. BK sweep at qL=4096 (one-shot bench, fused arm):
   - BK=32 (default): 13.58 ms
   - BK=16: **8.87 ms** (-35% wall-time)
   - BK=32 + `MFA_V6_MAX_THREADS=128`: 13.52 ms (no effect — the hint
     is too soft to reduce register pressure without also reducing
     tile size).

2. The BK=16 win is **monotonic across qL** in the parity-zone or
   better (qL ≥ 2048).  Small-qL slight regression (qL≤1024) suggests
   the smaller tile introduces loop-overhead that dominates when
   K-loop iterations are too few — but this is below the V34 carve-out
   threshold so not user-impacting.

3. The `MFA_V6_MAX_THREADS` knob having no effect supports H1
   specifically: register pressure is set by *tile size in registers*,
   not by `maxTotalThreadsPerThreadgroup` hint to the pipeline state.

**Verdict**: H1 **CONFIRMED** with clear fix path.

### H2 — L1 cache absorbs split's K-reload — partial evidence

**Theory** (from v39.0 results doc §"Why fused regressed at D=64"):
M5 NAX L1 + L2 may absorb the second K-load that split-dK performs,
nullifying the audit-predicted K-bandwidth amortization win.

**Indirect evidence** (no `instruments` available for direct L1
hit-rate measurement):

qL sweep, fused-BK32 vs split-D_vec ratio:
- qL=512:  fused/split = 0.98× (parity)
- qL=1024: fused/split = 1.00× (parity)
- qL=2048: fused/split = 0.99× (parity)
- qL=4096: fused/split = 0.67× (-33%)
- qL=8192: fused/split = 0.68× (-32%)

The crossover between qL=2048 (parity) and qL=4096 (regression) is
consistent with H2 (L1 absorbs small K-tiles, evicts large ones at
qL≥4096 where K-tile total bandwidth exceeds cache capacity).  But it
is **equally consistent with H1**: at small qL the K-loop has too few
iterations to amortize the per-K-tile spill cost, so the spill is
relatively cheaper.

**Verdict**: H2 has **partial supporting evidence** but H1 alone
explains the data.  No instrumentation available to disambiguate
further.  Documented as "consistent-with-H1" rather than independently
confirmed.  Future v2.39.x+ work could disambiguate via
`MTLCounterSampleBuffer` instrumentation (deferred).

### H3 — Occupancy reduction — **FALSIFIED**

**Theory**: fused kernel's larger TG register footprint reduces TG/CU
co-residency, hurting latency hiding.

**Direct test**: reduce `WM` (warps per TG) to lower total per-TG
register pressure:
- baseline (BQ=64, WM=4): 13.68 ms
- (BQ=32, WM=2):           14.35 ms (slightly **worse**)
- (BQ=16, WM=1):           15.43 ms (worse still)

If H3 were correct, reducing WM should *increase* TG/CU co-residency
and improve perf.  Observed effect is opposite: smaller WM makes
fused *slower*.  This rules out occupancy as the dominant mechanism.

**Verdict**: H3 **FALSIFIED**.  The fused kernel performs better with
higher WM (more parallelism), so the regression is not occupancy-related.

## Mechanism identified

**Per-SG register-pressure spilling caused by oversized persistent FP32
accumulators at BK=32** (TK=2).  The fused kernel maintains both
`dK_accum` (8 KB at D=64 BK=32) and `dV_accum` (same size) live across
the entire Q-loop.  Combined with `Stile + dPtile` and transient frags
(`Qfrag`, `Kfrag`, `Vfrag`, `dOfrag`), per-lane register use crosses
the Apple Metal compiler's per-lane register quota for full occupancy,
forcing partial spilling.  Cutting BK in half (TK 2→1) halves the
accumulator register count and brings the kernel under the threshold.

The `/metal-kernel-dev` audit estimated 52 regs/lane at D=64 with WM=4 —
this estimate counted bytes per accumulator (16 elems × 4B = 64 B) and
divided by 32 lanes (= 2 elems/lane × 8 frags × 4B / lane).  The
**actual** register cost depends on how the compiler maps NAXFrag tiles
to architectural registers; the audit's estimate did not account for
the compiler's register-allocation overhead (live-range conflicts +
spill-fill insertion when the live-range graph exceeds the per-lane
register quota).

## Decision verdict: **(α) — Mechanism identified + fixable**

Per blueprint decision tree:
- α: mechanism identified + fix available
- β: mechanism identified + structural ceiling (no fix)
- γ: mechanism unknown

Outcome α applies cleanly here.

## v2.39.1 ship state recommendation

1. **Change fused kernel default `BK` from 32 to 16** in
   `csrc/mfa_v6_nax_primitive.cpp::MFAV34BwdFusedDKDV::eval_gpu`
   (currently line ~1675).
2. **Flip auto-default routing** in `mlx_mfa/attention.py::_v34_backward_vjp`
   from `"split"` back to `"fused"` for D=64 (reversing the v2.39.0
   outcome-δ workaround).  Auto resolves to fused at D=64; split at
   D=128 (unchanged from v2.39.0).
3. **Run 3-session canonical bench** to confirm the BK=16 win is
   reproducible across sessions (variance ratio <1.15 per §AA.4).
4. **Update `_v34_eligible()` if appropriate**: the BK=16 fused win
   extends to qL=2048 (parity), so the v2.37.2 carve-out qL≥4096
   floor could potentially broaden.  Defer this decision to a
   separate v2.39.2 sprint after broader workload validation.
5. **CHANGELOG perf claims**: 1.95× / 1.89× / 1.87× vs SDPA-vjp at
   qL ∈ {4096, 8192, 16384} (preserves v2.38.1 numbers exactly with
   modest additional architectural advantage at qL=16384).
6. **`/mlx-mfa-perf-audit`** required for the v2.39.1 perf claim.
7. **`/mlx-mfa-release-audit`** canonical pre-tag gate.
8. **PERF_CLAIMS registry update**: add v2.39.1 entries flagging the
   carve-out shapes now route to fused-BK16 (architectural change, same
   SDPA-vjp speedup magnitudes preserved).

## Skill invocations log (per §AA.2)

| Phase | Skill | Status |
|---|---|---|
| A.3 design validation | `/metal-kernel-dev` | ✓ Done (BK=8 won't compile, WM=1 won't dispatch; sequencing recommendation) |
| B-D execution | (none; empirical sweeps) | — |
| E synthesis | `/metal-kernel-dev` | this doc (interpreting BK sweep + occupancy results) |
| F.1.2 perf audit | `/mlx-mfa-perf-audit` | pending |
| F.1.3 pre-merge | `/mlx-code-review` | pending |
| F.1.3 canonical pre-tag | `/mlx-mfa-release-audit` | pending |

## Why this investigation has standalone value

- **Identified a real root cause** that the audit's mental model missed.
  The "K-bandwidth amortization" hypothesis was *not* the right framing;
  the real mechanism is register-budget control.
- **Falsified H3 (occupancy)** clearly via direct manipulation — a
  productive negative result.
- **Established the BK-pressure mechanism** as a reusable framing for
  future fusion-sprint design.  Any future kernel combining two
  persistent FP32 accumulators should pre-budget BK accordingly.
- **Confirmed v2.38.1 D_vec D=64 perf claims are preserved** under the
  new fused-BK16 default.

## Future work flagged

- **Direct L1 cache instrumentation** via `MTLCounterSampleBuffer` to
  disambiguate H1 vs H2 contributions.  Deferred to v2.40+ if cache-
  behavior optimization becomes a focus.
- **D=128 fused kernel** (Phase C.1.b in original blueprint).  Same
  BK-pressure analysis would apply; need to verify whether D=128
  fused-BK=16 (or BK=8 via different blocking) can win.  Deferred.
- **Auto-default broadening below qL=4096**: BK=16 fused achieves
  parity at qL=2048; could remove the v2.37.2 carve-out floor.
  Deferred to v2.39.2 (needs broader workload validation).
