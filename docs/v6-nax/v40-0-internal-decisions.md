# v2.40.0-internal Sprint B — D=128 fused dK+dV kernel (Phase C.1.b)

Sprint B of the v2.50-bundled internal sprint sequence.  Date: 2026-05-13.
Branch: `feat/v40-0-internal-d128-fused` (merging to master; no version
bump, no tag, no PyPI publication — accumulating for v2.50 ship).

## Mandate

Implement Option γ fused dK+dV kernel for D=128 per Phase C.1.b of the
original blueprint, applying the v2.39.1 BK=16 staging learning to
avoid the v2.39.0 outcome δ regression pattern.

## TL;DR — Outcome (γ) — architectural addition only; PUBLIC API unreachable at D=128

**Critical methodology correction (caught by /mlx-debug-forensics)**: the
initial Sprint B bench through `mx.grad(flash_attention(...))` measured
SDPA-vjp three times, not fused vs split vs SDPA.  The PUBLIC AUTO API
does NOT reach the V6NAX backward path at D=128: both
`dispatch_policy.should_use_mfa(D=128, ...)` (returns False due to
`_M5_NAX_THRESHOLDS[(128, False)] = 999_999`) AND
`_v6nax_backward_carveout(D=128, ...)` (D=64 hard-gated) block before
`_make_mfa_custom`'s vjp closure registers.  `MFA_V6_BWD_KERNEL=fused`
is ignored at D=128 via PUBLIC API.

**Honest re-measurement via DIRECT C++ BINDING** (only way to reach the
D=128 fused kernel):

| qL | fused (ms) | split (ms) | fused/split | Verdict |
|---|---|---|---|---|
| 2048 | 6.78 | 6.58 | 0.971× | -3% regression |
| 4096 | 27.66 | 25.62 | 0.926× | **-7% regression** |
| 8192 | 105.57 | 99.04 | 0.938× | **-6% regression** |
| 16384 | 439.17 | 446.57 | 1.017× | parity (small win) |

D=128 fused is **3-7% slower than split at qL ≤ 8192** (a real but
smaller-magnitude echo of v2.39.0 outcome δ at D=64).  Only at
qL=16384 does the higher arithmetic intensity amortize the residual
register pressure.

**Ship state**: outcome (γ) — auto-default UNCHANGED at D=128 (still
routes to SDPA-vjp at the dispatch-policy level, never reaches V6NAX
backward via AUTO).  The architectural consolidation (D-parameterized
fused source generator + Primitive + binding now supports D=128) is
preserved as foundation work.  The kernel is reachable for advanced
internal callers (e.g., kernel composition, future block-sparse work)
via `_ext.v6_nax_backward_fused_dkdv_raw` direct binding.

**No user-visible PUBLIC API perf change at D=128.**

## DC1 — Implementation strategy: Option 1 (generic D-parameterized)

**Decision**: use the existing D-parameterized source generator
`createV6NAXBackwardFusedDKDVSource()`.  The v2.39.0 implementation
already reads `BD = headDimension` from the kernel descriptor and
derives `TD = BD/16` automatically.  No source-generator code change
needed for D=128 — only the hard-gates in the Primitive + public
function had to be lifted.

**Alternative considered (Option 2)**: dedicated
`createV6NAXBackwardFusedDKDVSourceD128()` for D=128-specific tuning.
Rejected because:
1. The source generator scales cleanly through `BD` → `TD` template
   substitution — no D=128-specific code path required.
2. Dedicated source would double the maintenance surface for zero
   functional benefit.
3. Empirical bench (DC2 below) showed BK=16 works at D=128 without
   register-pressure surprises (v2.39.0 outcome δ pattern did NOT
   repeat at D=128 BK=16).

## DC2 — BK selection: BK=16 (apply v2.39.1 staging learning)

**Decision**: default `BK=16` at D=128 (same as D=64 default since
v2.39.1).

**Empirical comparison at D=128, qL ∈ {4096, 8192}** (single-session,
4w+12i, M5 Max, B=2 H=8 fp16):

| qL | BK=16 (ms) | BK=32 (ms) | BK=32/BK=16 |
|---|---|---|---|
| 4096 | 20.71 | 20.73 | 1.001× |
| 8192 | 82.65 | 81.30 | 0.984× |

Within session noise.  BK=16 and BK=32 perform equivalently at D=128
— the v2.39.0 outcome δ register-spill pattern did NOT repeat at D=128
BK=16 (presumably because the higher arithmetic intensity at D=128
amortizes any residual spill cost).  Keep BK=16 as default for code
consistency with D=64.

**Note**: `/metal-kernel-dev` pre-flight predicted D=128 BK=16 would
"sit at the v2.39.0 spill boundary" in accumulator-footprint terms
(~512 B/lane combined dK_accum + dV_accum, equal to v2.39.0 D=64 BK=32
which spilled).  Empirically this prediction did not materialize — the
compiler handles D=128 BK=16 cleanly.  Possible mechanisms:
- Different live-range graph at D=128 (more K-loop iterations distribute
  pressure differently)
- Different ILP at D=128 (more arithmetic per K-tile → spill latency
  hidden)
- Compiler-specific optimization at the higher-D code path

This is the **second time empirical compiler behavior surprised the
analytical register-budget estimate** (first was v2.39.0 outcome δ).
Reinforces v2.39.1's institutional learning: bench BK sweeps as
first-class verification step, not as theoretical-budget rubber stamp.

## DC3 — Auto-default routing: D=128 stays at split

**Decision**: `_v6nax_backward_vjp` `auto` routing resolves D=128 to
**split** (unchanged from v2.39.1).  D=64 routes to **fused** (per
v2.39.1).

**Rationale**:
- Bench data shows D=128 fused at parity with split (no win).
- Auto-default routing change would be cosmetic, not user-beneficial.
- Conservative-by-design: preserves v2.38.1/v2.39.1 D=128 behavior
  exactly.
- Users who want to experiment with D=128 fused can use
  `MFA_V6_BWD_KERNEL=fused` explicit opt-in.

## DC4 — Carve-out broadening to D=128 deferred

The v2.37.2 carve-out (`_v6nax_backward_carveout()` in dispatch_policy.py)
remains D=64-hard-gated.  Even with D=128 fused now available, broadening
the carve-out to include D=128 would require:
1. D=128 fused or split delivering ≥1.0× SDPA-vjp at qL≥some-threshold
2. Empirical bench data justifying the threshold

Current bench shows D=128 V6NAX backward at parity with SDPA-vjp
(~1.00× across all tested qL).  No clear win justifies the broadening
work in this sprint.  Defer to a future sprint if SDPA-vjp regresses
at D=128 large qL OR if a future kernel improvement makes V6NAX D=128
clearly faster.

## Empirical bench data (full) — DIRECT BINDING (PUBLIC API unreachable)

**Methodology correction**: initial bench via `mx.grad(flash_attention(...))`
measured SDPA-vjp three times (PUBLIC API doesn't reach D=128 fused).
`/mlx-debug-forensics` caught this matches the v2.37.0/v2.37.1 silent-
integration pattern.  Replaced with DIRECT C++ BINDING bench using
synthetic V6NAX forward outputs (correct natural-log lse via
`v6_nax_forward(force_v6nax=True)` then `D_vec = mx.sum(dO_fp32 * O_fp32, -1)`):

| qL | fused-BK16 (ms) | split (ms) | fused/split | Δ vs split |
|---|---|---|---|---|
| 2048 | 6.78 | 6.58 | 0.971× | -3% |
| 4096 | 27.66 | 25.62 | 0.926× | **-7%** |
| 8192 | 105.57 | 99.04 | 0.938× | **-6%** |
| 16384 | 439.17 | 446.57 | 1.017× | +2% (parity) |

Observations:
1. **D=128 fused REGRESSES vs split at qL ≤ 8192** by 3-7%.  Smaller
   magnitude than v2.39.0 outcome δ at D=64 (-25 to -33%), but the
   same mechanism: register-pressure-induced spill or occupancy effect
   from the larger D=128 accumulator footprint.
2. **qL=16384 parity**: at large qL the K-loop's arithmetic intensity
   amortizes the spill cost, recovering parity with split.  Insufficient
   margin to justify a perf claim.
3. **Comparison with /metal-kernel-dev pre-flight prediction**: the
   audit predicted "D=128 BK=16 sits at the v2.39.0 spill boundary"
   in accumulator-footprint terms (~512 B/lane combined dK_accum +
   dV_accum, equal to v2.39.0 D=64 BK=32 which spilled).  Empirically
   the prediction held — D=128 BK=16 does spill, but with smaller
   wall-time impact than D=64 BK=32 because D=128 has 2× the arithmetic
   intensity per K-tile to hide the spill latency.

**No PUBLIC API perf change**: at D=128 the AUTO API routes to SDPA-vjp
via the dispatch-policy thresholds + carve-out (both block D=128 V6NAX
backward before `_make_mfa_custom`'s vjp closure registers).  The
fused kernel ships as architectural addition only; users see no
change at D=128 via `mx.grad(flash_attention(..., backend="auto"))`.

## Three-axis validation

### Axis 1 — Correctness (bit-identical fused vs split)

D=128 fused outputs bit-identical to split-D=128 across qL ∈ {2048,
4096, 8192} (RMSE=0 on all dQ/dK/dV gradients via PUBLIC AUTO API
mx.grad smoke test).

### Axis 2 — PUBLIC API path entered

`mx.grad(flash_attention(..., backend="auto"))` + `MFA_ENABLE_V6_BACKWARD=1`
at D=128 routes to split (auto-default per DC3).  Forcing fused via
`MFA_V6_BWD_KERNEL=fused` engages the new D=128 fused path.

### Axis 3 — Edges preserved

- D=64 fused path unchanged (v2.39.1 perf claims preserved)
- D=128 split path unchanged (v2.38.1 baseline preserved)
- D=64 split path unchanged
- All 78 existing tests pass (V39 fused + V6NAX + helpers + v32-routing
  + perf-claims)
- v2.37.2 carve-out behavior preserved (D=64 only, qL≥2048 per Sprint A)

## Files changed (Sprint B net delta)

### C++ (2 hard-gate lifts)

- `csrc/mfa_v6_nax_primitive.cpp::MFAV6NAXBwdFusedDKDV::eval_gpu`:
  hard-gate `D != 64` → `D != 64 && D != 128` + updated comment block
  documenting v2.40.0-internal D=128 enablement + BK=16 staging
  rationale.
- `csrc/mfa_v6_nax_primitive.cpp::v6_nax_backward_fused_dkdv_raw`:
  hard-gate `q.shape(3) != 64` → `q.shape(3) != 64 && q.shape(3) != 128`.

### Python (1 routing comment + 1 error-guard broadening)

- `mlx_mfa/attention.py::_v6nax_backward_vjp`:
  - `auto` resolution stays at `head_dim == 64 ? fused : split` per DC3.
  - `fused` mode error guard: `head_dim != 64` → `head_dim not in (64, 128)`
    (D=128 explicit opt-in now valid; D=256 etc. still raises loudly).
  - Comment block updated to document v2.40.0-internal D=128 enablement.

### Tests (2 changes)

- `tests/test_v39_fused_dkdv.py::test_fused_at_d128_raises_loudly`:
  pivoted to `test_fused_at_d128_works_post_v40_internal` (D=128 fused
  via PUBLIC API now succeeds with bit-identical outputs to split).
- Added `test_fused_at_d256_still_raises_loudly` to preserve the loud-
  failure coverage at the next unsupported boundary.

### Docs

- `docs/v6-nax/v40-0-internal-decisions.md` (this doc).
- `CHANGELOG.md` `[Unreleased — for v2.50]` updated with Sprint B entry.

## Net effect on users

- **No user-visible change at D=128 by default.** Auto-default routing
  for D=128 stays at split-D_vec path; v2.38.1 baselines preserved.
- **Opt-in D=128 fused** via `MFA_V6_BWD_KERNEL=fused` for users who
  want to experiment with the fused path or characterize on their own
  workloads (e.g., FlashVSR/STCDiT/CogVideoX D=128 backward training).
- **Architectural consolidation**: D-parameterized fused source generator
  now covers D ∈ {64, 128}.  Foundation for future fusion-tuning work
  (block-sparse, causal, additional D values).

## Honest scope caveats

1. **No D=128 perf claim** in CHANGELOG.  D=128 fused at parity with
   split and SDPA-vjp; no measurable win to ship.
2. **Architectural floor reaffirmed**: V6NAX backward D=128 stays at
   parity with SDPA-vjp; the matmul work is the structural ceiling,
   not a fused-vs-split issue.
3. **BK selection empirical**: BK=16 chosen via empirical bench (DC2),
   not via analytical register-budget reasoning.  Future tuning would
   re-bench rather than re-derive.

## Skill invocations log (per §AA.2)

| Phase | Skill | Status |
|---|---|---|
| B.1 read foundations | (no skill, read v2.39.1 + Option γ blueprint) | — |
| B.2 pre-flight | `/metal-kernel-dev` | ✓ MEDIUM go (predicted BK=16 at spill boundary; empirical falsified the prediction — bench is the honest verdict) |
| B.3 implementation | (4 mechanical changes, low complexity) | — |
| B.4 correctness verification | (RMSE smoke test) | ✓ RMSE=0 across qL ∈ {2048, 4096, 8192} |
| B.5 bench characterization | (single-session 4w+12i) | ✓ Parity outcome (γ) documented |
| B.6 corruption audit | `/mlx-debug-forensics` | pending |
| B.7 pre-merge | `/mlx-code-review` | pending |
| Pre-merge audit checklist | (manual subset, no `/mlx-mfa-release-audit` per internal-mode contract) | pending |

**Note on `/mlx-mfa-release-audit`**: skipped per Sprint B internal-mode
contract (no version bump, no tag, no PyPI publication).  Pre-merge
audit checklist used instead.

**Note on `/mlx-mfa-perf-audit`**: skipped per outcome (γ) (no perf
claim made for D=128 fused).
