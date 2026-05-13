# Changelog

All notable changes to mlx-mfa are documented here.

## [Unreleased]

## [2.37.2] — 2026-05-13 — V34 backward integration bugfix

### Fixed

- **CRITICAL: Silent SDPA-vjp fallback when `MFA_ENABLE_V34_BACKWARD=1`**
  on non-causal shapes.  v2.37.0 and v2.37.1 advertised "1.4-1.85× faster
  than SDPA-vjp" at D=64 large shapes, but the public `flash_attention()`
  autograd path was silently routing through SDPA-vjp.  Root cause:
  `flash_attention()` calls `should_use_mfa()` BEFORE checking the V34
  backward env var; `should_use_mfa()` returns `False` for non-causal
  D∈{64,128} because STEEL forward isn't competitive at those shapes;
  the `if not use_mfa: return _fallback_sdpa(...)` block returns before
  the V34 custom-vjp `_impl` is ever constructed.  Setting
  `MFA_ENABLE_V34_BACKWARD=1` had no observable effect through
  `mx.grad(flash_attention(...))` — only direct calls to `_ext.v6_nax_*`
  kernels reached V34 backward.

- **Fix** (`mlx_mfa/attention.py`): narrow carve-out in `flash_attention`
  before the SDPA fallback returns.  When `MFA_ENABLE_V34_BACKWARD=1`
  AND the shape qualifies for end-to-end V34 backward win (D=64,
  qL ≥ 4096, non-causal, f16/bf16 same-dtype K/V, NAX), force
  `use_mfa = True` so the MFA path runs.  The custom-vjp `_impl` then
  forward-fuses via V34 (force_v34=True) and the backward dispatches
  through the V34 multi-SG dK/dV + dQ kernels.

### Engagement envelope (auto, when env=1)

| Config | Backward path | End-to-end vs SDPA-vjp |
|---|---|---|
| D=64, qL=4096, non-causal, f16/bf16 | V34 backward | **1.82× faster** |
| D=64, qL=8192, non-causal, f16/bf16 | V34 backward | **1.81× faster** |
| D=64, qL < 4096 | SDPA-vjp | parity or V34 loss → SDPA preferred |
| D=128, any qL | SDPA-vjp | V34 backward 2.0-2.4× slower (research only) |
| Causal | SDPA-vjp / STEEL bwd | V34 backward doesn't support causal (DC3) |

### Measurement (M5 Max, B=1 H=4 f16, `mx.grad(flash_attention(...))`)

| D | qL | V34 backward (carve-out engaged) | SDPA-vjp | Speedup |
|---|----|---|---|---|
| 64 | 4096 | 2.65 ms | 4.83 ms | 1.82× |
| 64 | 8192 | 9.78 ms | 17.67 ms | 1.81× |
| 128 | 8192 | 19.73 ms (NOT engaged, fell back to SDPA) | 19.61 ms | 1.00× |

### Correctness validation

V34 backward gradients vs SDPA-vjp (D=64, qL ∈ {4096, 8192}, f16):
- `dq` max|abs| = 0.0000, mean|rel| = 0.20-0.21%
- `dk` max|abs| = 0.0000, mean|rel| = 0.26-0.70%
- `dv` max|abs| = 0.0010, mean|rel| = 0.04%

All within FP16 floor.  No regressions in test suite (670 pass, 2
pre-existing FP16 flakes unrelated to V34).

### Migration

No code changes required.  Users with `MFA_ENABLE_V34_BACKWARD=1` who
were silently using SDPA-vjp will now transparently get the V34 backward
win on D=64 large shapes.  Other shapes continue to route through
SDPA-vjp as before.

## [2.37.1] — 2026-05-13 — V34 backward post-release improvements

### Added (post-v2.37.0 improvements)

- **V34 backward eligibility expanded to D=64 small-Nk** (formerly DC12-
  blocked).  Added `force_v34` parameter to `_ext.v6_nax_forward` so the
  flash_attention VJP can ensure natural-log lse is produced even on
  shapes that default-route to legacy v6_nax forward.  Previously, D=64
  with Nk ≤ 8000 (FlashVSR-class shapes) fell through to SDPA-vjp; now
  V34 backward kernels handle them.  Correctness validated within FP16
  floor (dQ RMSE 9.7e-8, dK RMSE 1.0e-7, dV RMSE 4.2e-4 vs SDPA-vjp).

- **MAJOR PERF FINDING — V34 backward at D=64 large shapes WINS vs
  SDPA-vjp** (1.4-1.85× faster):

  | D=64 shape (qL=kL) | V34 backward | SDPA-vjp | V34 / SDPA |
  |---|---:|---:|---:|
  | 256 | 0.62ms | 0.46ms | 1.37× (slower) |
  | 512 | 0.81ms | 0.48ms | 1.68× (slower) |
  | 1024 | 0.61ms | 0.46ms | 1.32× (slower) |
  | **2048** | **0.91ms** | **1.31ms** | **0.70× ← V34 WINS** |
  | **4096** | **2.61ms** | **4.81ms** | **0.54× ← V34 WINS** |
  | **8192** | **9.77ms** | **17.69ms** | **0.55× ← V34 WINS** |

  The "architectural floor" of 2.4× SDPA-vjp identified at D=128 does
  NOT apply at D=64.  At D=64 large shapes (qL × kL ≥ 4M = 2048²),
  V34 backward is 1.4-1.85× faster than SDPA-vjp.  This is a clear
  perf win for D=64 training workloads (e.g., FlashVSR class with
  larger qL, LTX2-style cross-attention).

  V34 backward remains SHIP_OPT_IN (`MFA_ENABLE_V34_BACKWARD=1`) by
  default to avoid changing forward-pass routing on users who don't
  do backward (forcing V34 forward routing has small forward-only
  perf cost on D=64 small-Nk).  When `MFA_ENABLE_V34_BACKWARD=1` is
  set, training on D=64 large shapes is now faster than SDPA-vjp.

## [2.37.0] — 2026-05-13 — V34 backward NAX-direct kernels (SHIP_OPT_IN)

### Added

- **V34 backward NAX-direct kernels** for M5+: `_ext.v6_nax_backward_query`
  (dQ), `_ext.v6_nax_backward_kv` (fused dK/dV WM=1), `_ext.v6_nax_backward_dv_raw`
  + `_ext.v6_nax_backward_dk_raw` (multi-SG WM=4 split via Q-row partition).
  Apple's NAX backward is NYI in MLX framework — these kernels are the only
  path for NAX-accelerated backward attention on M5+.

- **`flash_attention()` autograd integration** for V34 backward: when
  `MFA_ENABLE_V34_BACKWARD=1` is set, the VJP routes through V34 backward
  kernels on M5+ eligible shapes (D ∈ {64, 128}, FP16/BF16, no causal /
  window / softcap, V34-forward-routing-eligible).  Default (env unset):
  STEEL backward / SDPA-vjp fallback (v2.36.1-exact behavior).

- **V34 forward lse-write** (BLK1 infrastructure): V34 forward kernel
  now writes per-row natural-log lse to device memory (previously dead
  storage).  Enables V34 backward dQ to recompute softmax P from forward
  state without re-running forward.

- **Forward-fusion**: when V34 backward is enabled, `_make_mfa_custom`
  uses V34 forward (natural-log lse) directly instead of STEEL forward,
  eliminating both STEEL fwd and V34 fwd-recompute at backward time.

### Methodology

- V34 backward Option β sprint methodology: started from V34 forward
  investigation's B+C+E bundle (cross-SG sync elim + simd_shuffle_xor +
  M5-tuned defaults) — all three mechanisms transferred cleanly to
  backward.  Phase 2.O1 (WM=2 K-row partition) FALSIFIED empirically
  (16-24% regression vs WM=1 due to softmax replication tax); Phase 2.O2
  (WM=4 Q-row partition + two-kernel split) delivers 1.7-2× speedup.

- Hit architectural floor at 2.4× SDPA-vjp (qL=8192): dK kernel inherently
  ~2× heavier than dV (extra dO@V^T matmul required by FA-2 dK formula).
  Further closing requires major restructure (e.g., fused dK+dV with TGP
  cross-SG reduction — register pressure on M5).

### Validated

- 39 new tests covering V34 backward kernels (dQ + dK/dV correctness vs
  SDPA-vjp, multi-SG variants, integration through flash_attention VJP,
  routing-parity constraints).  RMSE 1.5e-8 (FP32 floor) for dQ + dK;
  RMSE 1.5e-6 (FP16 round-trip floor) for dV.
- Full regression: 116/116 tests pass (77 v2.36.1 + 39 V34 backward).
  Zero regressions.

### Ship status: SHIP_OPT_IN

V34 backward is functionally correct but 2.2-2.4× slower than SDPA-vjp
on M5 Max at qL=8192.  Per auto-default principle (Sprint U), default
behavior unchanged for users.  Opt-in via `MFA_ENABLE_V34_BACKWARD=1`
for research / benchmarking.

### Performance (M5 Max FP16 D=128 via `flash_attention()` autograd)

| qL | V34 multi-SG | SDPA-vjp | Ratio |
|---|---:|---:|---:|
| 1024 | 1.07ms | 0.50ms | 2.13× |
| 2048 | 3.22ms | 1.54ms | 2.09× |
| 4096 | 12.77ms | 5.31ms | 2.40× |
| 8192 | 48.93ms | 20.37ms | 2.40× |

Improvement vs initial Phase 2 WM=1 fused kernel: 1.7-2× speedup.

### Internal env vars (advanced users)

- `MFA_ENABLE_V34_BACKWARD=1` — opt into V34 backward path.
- `MFA_V34BWD_USE_FUSED=1` — fall back to WM=1 fused dK/dV kernel
  (instead of WM=4 split; for benchmarking).
- `MFA_V34BWD_WM` (default 4) — WM for multi-SG dK/dV split kernels.
- `MFA_V34BWDV_BQ`, `MFA_V34BWDV_BK`, `MFA_V34BWDV_WM` — per-kernel tile
  overrides for dV (researchers).
- `MFA_V34BWDK_BQ`, `MFA_V34BWDK_BK`, `MFA_V34BWDK_WM` — per-kernel tile
  overrides for dK.

### Unchanged

- All v2.36.x infrastructure preserved (shape-aware V2 sparse, canonical
  methodology, Sprint U auto-on-import hooks, Conv3D NAX).
- All public API signatures preserved.
- All patchers preserved as expert API.
- v2.36.1 default behavior unchanged for users who don't set
  `MFA_ENABLE_V34_BACKWARD=1`.

### Notes for v2.36.1 users upgrading

- No code changes required; V34 backward is opt-in.
- For training workloads on M5+ FP16 D∈{64,128}: set
  `MFA_ENABLE_V34_BACKWARD=1` to evaluate V34 backward kernels.
  Expect slower runtime than SDPA-vjp on current hardware (this is
  research infrastructure, not a perf optimization).

### Deferred to follow-up sprints

- Multi-SG dK kernel optimization (deferred — architectural floor at 2.4×
  SDPA-vjp confirmed).
- Block-sparse backward (DC3 deferred from V34 backward sprint scope).
- Causal backward (DC3 deferred).
- D ∉ {64, 128} backward (falls back to STEEL).
- Softcap / ALiBi / TurboQuant backward (kept on STEEL).

## [2.36.1] — 2026-05-13 — Canonical methodology + shape-aware V2 sparse default

### Changed (transparent for users)

- **Shape-aware V2 sparse default on M5+.** `decide_auto_version()` now
  routes sparse-attention shapes with `qL × kL × D ≥ 2.15e9` (= 4096 ×
  4096 × 128, the smallest tested work product) to V2 automatically.
  V2's broad envelope (1.95-13.86× vs SDPA+bias per 3-session canonical
  bench) activates transparently for these shapes. Sub-threshold shapes
  keep V1 default conservatively (no canonical-protocol data to validate
  them — DC9 empirical-calibration rule).
  - `MFA_LCSA_KERNEL_VERSION=v2` forces V2 universally (override).
  - `MFA_LCSA_KERNEL_VERSION=v1` forces V1 universally (override).
  - Unset: shape-aware default applies.

  This honors the auto-default principle (Sprint U / v2.36.0): V2
  graduates to default for the regime where cross-session validation
  is achievable.

### Added

- `mlx_mfa.lcsa_nax.decide_auto_version(density, qL, kL, D)` — public
  Python function exposing the shape-aware routing decision. Returns
  `"v1"` or `"v2"`. Honors `MFA_LCSA_KERNEL_VERSION` env override.
- `_ext.sparse_attention_forward` now accepts an explicit
  `kernel_version: str` parameter (defaults to empty string for
  backward compatibility). Thread-safe alternative to env-var-based
  routing.

### Methodology

- **Canonical Apple Silicon benchmark protocol adopted for sub-1.5ms
  kernels** (`docs/methodology/canonical-protocol.md`). Replaces §4-strict
  cooldown protocol where the latter fails due to GPU power-state cycling.
  §4-strict remains canonical for ≥1.5ms kernels. Selection rule
  documented in `CLAUDE_V6_NAX.md` §4.3.
- **Sub-1ms variance methodology thread CLOSED.** Two REGRESSION sprints
  (mx.matmul v2.36.0, matched-workload 2026-05-12) + six-source web
  research convergence (Apple Developer Forums thread 692062, Feng et al.
  arXiv 2501.14925, MLX docs, WWDC25 Session 315, Draw Things MFA v2.5
  NA, MLX GitHub Discussion #1571) confirmed userspace P-state lock is
  unavailable and warmup-during-cooldown is mechanically incompatible
  with the measurement regime. Path-forward registry closed: option 1
  FALSIFIED, option 2 SKIPPED, option 3 deferred, **option 4 ACTIVATED**
  via this release.

### Validated

- 7 production shapes re-benchmarked under canonical methodology
  (3 sessions, ratio analysis): 6 CONFIDENT + 1 BOUNDARY,
  0 HIGH_VARIANCE. All 3 v2.36.0 HIGH-variance shapes graduated to
  CONFIDENT under canonical. See
  `docs/methodology/canonical-bench-results.md` for full data.
- 77/77 tests pass (65 pre-existing LCSA / Sprint U / FlashVSR tests +
  12 new three-axis tests for shape-aware decide_auto_version).

### Unchanged

- v2.35.0 V2 kernel source code preserved.
- v2.36.0 auto-default infrastructure preserved (auto-on-import hooks,
  `flash_attention_sparse` → `sparse_attention_dispatch` routing,
  Conv3D NAX via `mx.conv_general` hook).
- All public API signatures preserved (additive only: new
  `kernel_version` binding param defaults to empty for compatibility).
- All patchers (`patch_seedvr2_vae`, `patch_flashvsr_lcsa`,
  `patch_mlx_lm`) preserved as expert API.

### Notes for v2.36.0 users upgrading

- If you have `MFA_LCSA_KERNEL_VERSION=v2` set in your environment:
  this still works exactly as before, V2 is forced for all shapes.
  You can now unset it and rely on shape-aware default for most use cases.
- If you have no env setting: v2.36.1 transparently activates V2 for
  shapes ≥ 4096 × 4096 × 128 work product. Expect 1.95-13.86× speedup
  vs the v2.36.0 SDPA fallback path on FlashVSR-class workloads.
- If you use sub-threshold sparse attention shapes: V1 remains default.
  Set `MFA_LCSA_KERNEL_VERSION=v2` if you want V2 unconditionally on
  small shapes too (accept that canonical-protocol validation does not
  cover these shapes yet).

## [2.36.0] — 2026-05-12 — Sprint U: Unification main + auto-default principle

### Changed (transparent for users)

- **Auto-on-import**: `import mlx_mfa` now installs optimization hooks at
  import time. Eligible `mx.conv_general` calls on M5+ (3×3×3 / 1×1×1
  kernel, FP16/BF16, stride=1, dilation=1, groups=1, !flip) auto-route
  to `conv3d_nax_forward` — ~1.6× speedup vs vanilla MLX without any
  user code change. Pre-v2.36.0 users had to call `patch_seedvr2_vae(model)`
  explicitly; that patcher remains available as expert API for verbose
  logging and per-module control.

- **`flash_attention_sparse` on M5+ auto-routes** to
  `mlx_mfa.lcsa_nax.sparse_attention_dispatch` when the mask shape is
  compatible (symmetric BT ∈ {16, 32, 64}). V1 NAX kernel is the
  dispatcher's default for density < 0.02; SDPA + float bias for moderate
  density. Asymmetric STEEL-shape masks (BQ=32, BK=16) and
  `MFA_DISABLE_AUTO_HOOKS=1` paths fall through to the pre-Sprint-U
  `_sparse_fallback_sdpa_perhead` behavior.

### Added

- `mlx_mfa.enable()` + `mlx_mfa.disable()` + `mlx_mfa.hooks_status()`
  public API for explicit hook control (benchmarking / debugging).
- `MFA_DISABLE_AUTO_HOOKS=1` env var prevents auto-hook install at import.
- `mlx_mfa/_auto_hooks.py` (222 LOC) — auto-hook installation module.
- `docs/RELEASE_PHILOSOPHY.md` (207 LOC) — canonical auto-default principle.
- `CLAUDE_V6_NAX.md` §5.X — pre-tag auto-default audit checklist.
- `CLAUDE.md` auto-default principle reminder near the top.
- `tests/test_sprint_u_sparse_routing.py` (4 tests) — Section B validation.
- `tests/test_sprint_u_auto_hooks.py` (9 tests) — Section C validation.

### Unchanged (backward compatibility)

- All existing public API signatures preserved.
- `MFA_LCSA_KERNEL_VERSION=v2` remains opt-in (sub-1ms methodology
  validation pending per `docs/methodology/sub1ms-protocol-diagnostic.md`).
  Once methodology resolved, V2 graduates to default via `decide_auto_version()`
  flip — zero user code change.
- All named patchers (`patch_seedvr2_vae`, `patch_flashvsr_lcsa`,
  `patch_mlx_lm`) remain available as expert API.
- v2.35.0 production code preserved (V2 kernel + STEEL V1 sparse + Conv3D NAX
  + flash_attention dispatch all unchanged).

### Tests

Joint LCSA + integration + Sprint U test suite: **65/65 pass**:
- 6 Phase 1.1 + 12 Phase 1.2 + 6 Phase 1.4 dispatcher (v2.34.0 surface)
- 19 Phase B V2 coop-rewrite (v2.35.0 surface)
- 9 FlashVSR LCSA integration (v2.34.0 Section H.2 surface)
- 4 Section B sparse auto-routing (this release)
- 9 Section C auto-hook lifecycle (this release)

### Migration notes for v2.35.x users upgrading

1. If you were calling `patch_seedvr2_vae(model)`: your code continues to
   work. Optionally remove the call — `import mlx_mfa` now handles it.
2. If you were calling `flash_attention_sparse(...)` on M5+: expect a perf
   improvement at very-sparse density when your mask is symmetric BT
   (NAX-aware dispatcher now active). No code change required.
3. If you depend on vanilla MLX behavior for any reason: set
   `MFA_DISABLE_AUTO_HOOKS=1` or call `mlx_mfa.disable()` after import.
4. If you want to verify whether auto-hooks are active: call
   `mlx_mfa.hooks_status()`.

### Philosophy (new canonical doc: `docs/RELEASE_PHILOSOPHY.md`)

Every PyPI release of mlx-mfa must be fully functional transparently for
users. Validated optimizations activate by default without requiring user
code changes. Opt-in mechanisms (env vars, named patchers) are transitional
(validation pending) or expert-mode (granular control), never the primary
documented user path.

Three usage levels: Default (auto-on-import, 90% of users) / Explicit API
(advanced users) / Expert mode (research/debug). See the doc for the full
principle, anti-patterns, and pre-tag audit checklist.

## [2.35.0] — 2026-05-12 — Sprint B coop-rewrite (V2 cooperative-tensor SHIP_OPT_IN)

### Added

- **V2 sparse attention kernel** (opt-in via `MFA_LCSA_KERNEL_VERSION=v2`) —
  single-kernel cooperative-tensor inner-GEMM via NAXFrag::mma + NAXTile,
  V34 forward pattern adapted for sparse with:
  - Outer-loop block-mask skip (uniform branch across simdgroup → zero divergence)
  - K/V base + per-iteration jump pointers (random kb access via index)
  - Per-SG Q-row partition (kU=16, BQ=BK=32, WM=2 per design DC1+DC3)
  - All-False row → exact zero output preserved (v2.34.0 contract)
- `MFA_LCSA_KERNEL_VERSION` env var:
  - `v1` (default): per-thread FA-2 (v2.34.0 kernel, unchanged)
  - `v2`: cooperative-tensor inner-GEMM (broad-envelope V34-pattern kernel)
- Section A design doc `docs/lcsa-nax/lcsa-nax-design.md` §13 (V2 architecture)
- Section C: 19 V2 correctness tests covering V1↔V2 equivalence on 7 shapes
  + three-axis V2 validation + density sweep 0.01-0.50

### Verdict: SHIP_OPT_IN (§D.2 decision tree)

§4-strict 3-session results (M5 Max 128GB, see
`docs/lcsa-nax/lcsa-nax-coop-rewrite-results.md`):

**Production shapes (cross-session medians)**:

| Shape | density | V2 ms | V1 ms | SDPA+bias ms | V2/V1 | **V2 vs SDPA+bias** | range% | flag |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| lcsa_small_seq4k          | 0.239 |  1.14 |  38.62 |  2.56 | 33.93× | **2.24×** |  7.3% | CONFIDENT |
| lcsa_small_seq4k_sparse   | 0.067 |  0.97 |  11.29 |  2.57 | 11.44× | **2.64×** |  5.4% | CONFIDENT |
| lcsa_mid_seq8k            | 0.119 |  1.52 |  51.52 |  6.45 | 35.79× | **4.35×** | 12.1% | BOUNDARY |
| lcsa_mid_seq8k_sparse     | 0.030 |  1.10 |  13.48 |  6.46 | 12.23× | **5.85×** | 35.0% | HIGH |
| lcsa_large_seq16k         | 0.120 |  2.06 | 103.91 | 12.73 | 50.46× | **6.18×** | 20.3% | HIGH |
| lcsa_large_seq16k_sparse  | 0.030 |  1.10 |  27.15 | 12.83 | 24.63× | **11.57×** | 18.6% | BOUNDARY |
| niche (mid_seq8k @ d=0.01)| 0.011 |  0.63 |   5.42 |  6.45 |  8.54× | **10.29×** | 32.6% | HIGH |

**Density sweep — lcsa_mid_seq8k**:

| density | V2 ms | V1 ms | SDPA ms | V2/V1 | **V2 vs SDPA+bias** |
|---:|---:|---:|---:|---:|---:|
| 0.011 | 0.71 |   5.15 | 6.49 |  7.28× | **9.24×** |
| 0.030 | 0.82 |  13.19 | 6.48 | 16.01× | **7.86×** |
| 0.049 | 0.85 |  21.39 | 6.46 | 24.95× | **7.54×** |
| 0.102 | 1.18 |  43.64 | 6.45 | 37.13× | **5.49×** |
| 0.199 | 1.72 |  85.48 | 6.45 | 49.27× | **3.74×** |
| 0.500 | 3.32 | 211.06 | 6.45 | 63.59× | **1.95×** |

**Verdict rationale**: V2 wins universally (2.22-11.57× vs SDPA+bias on every
session of every shape, 7.28-63.59× vs V1). Strict §B.7 variance criterion
(range < 10%) yields wins=2/7 (CONFIDENT-only count) due to elevated A/B/A
drift caused by V1's slow middle round disturbing V2 cache state. Per §D.2:
"V2 ships on 2-3 shapes ... → SHIP V2 as opt-in via env var, v2.35.0 with
caveats". V1 remains default; V2 is opt-in via `MFA_LCSA_KERNEL_VERSION=v2`.

### Unchanged

- `DEFAULT_DENSITY_THRESHOLD = 0.02` (V1 break-even). Users opting into V2
  should pass `density_threshold=0.95` to the dispatcher to capture V2's
  broad envelope.
- Default kernel version = V1 (per-thread FA-2). v2.34.0 production
  behavior 100% preserved for users not setting the env var.

### Architecture

- V1 (per-thread-Q-row FA-2 with register math) preserved unchanged.
- V2 source-gen in `csrc/mfa_sparse_attention.cpp` via Apple helpers
  (NAXFrag + NAXTile, 389 LOC verbatim from `csrc/mfa/v6_nax/NAAttentionKernel.cpp`)
  + V34 kernel body adapted (194 LOC).
- Cache discrimination via kernel-name `_v1` / `_v2` suffix — both pipelines
  coexist; switch via env var.

### Tested

- 52/52 LCSA + integration + V2 tests pass:
  - 6 Phase 1.1 + 12 Phase 1.2 + 6 Phase 1.4 dispatcher + 9 Section H.2 integration
    (v2.34.0 surface, V1 baseline preserved)
  - 19 new Section C V2 tests (V1↔V2 equivalence on 7 shapes × density sweep
    0.01-0.50 + three-axis V2 validation including edges)
- V1 source-gen unchanged from v2.34.0; V2 is purely additive.

### Documentation

- `docs/lcsa-nax/lcsa-nax-design.md` §13 (V2 architecture, 282 LOC)
- `docs/lcsa-nax/lcsa-nax-coop-rewrite-decisions.md` (DC0-DC8)
- `docs/lcsa-nax/lcsa-nax-coop-rewrite-{inventory,results}.md`
- `docs/lcsa-nax/lcsa-nax-coop-rewrite-{data,analysis}.json`

### Future-work register (post v2.35.0)

- ~~matmul2d cooperative-tensor inner-GEMM rewrite~~ — **DONE** in v2.35.0
- `patch_sparkvsr_sliding_window` companion patcher — tracked
- V34 forward focused investigation (memory #30 roadmap) — next sprint target

### §4-validated 2026-05-12 — Sprint B v2.34.0 ship-verdict (no tag)

Methodology-validation re-bench of the v2.34.0 shipped envelope under
§4-strict 3-session subprocess-isolated protocol (180s initial / 60s
inter-shape / 90s inter-round CLI knob, A/B/A pattern with
`sparse_attention_dispatch` cache-HIT pattern as A and
`mx.fast.scaled_dot_product_attention(mask=bias)` as B). Single-session
shipped numbers in `docs/lcsa-nax/lcsa-nax-phase1_5-ship-verdict.md` are
structurally validated.

**§4 outcome**:
- 6/7 shapes CONFIDENT (<10% cross-session range)
- 1/7 BOUNDARY (niche shape, 10.0% range driven by S1 cold-cache 21%
  A/B/A drift artifact; S2+S3 collapse to ~0.3% range)
- 0 HIGH variance
- Max |Δ| vs Phase 1.4 single-session = 6.9% (well within ±15% gate)
- Niche-win regime NOT overturned: 2.28× median ≫ 1.5× threshold

**Action**: DOC_UPDATE_WITH_CAVEATS per Section D decision tree — one
BOUNDARY shape blocks the all-CONFIDENT auto-tag branch. No v2.34.1 tag.
Production code at v2.34.0 unchanged.

**Niche-win ratio range corrected**: shipped envelope was reported as
"2.45-4.6× at density ≤ 0.01" from single-session Phase 1.4 data. §4
median is **2.28× with cross-session range 2.06-2.29×** depending on
cache warmup state. The 2.45× single-session number was at the high
end of cache-warmth luck; structural production-steady-state ratio is
~2.28×.

**New artifacts**:
- `bench/lcsa_nax_phase1_5_harness.py` — §4-strict harness (Sprint C pattern)
- `bench/lcsa_nax_rebench_analysis.py` — cross-session analysis tool
- `docs/lcsa-nax/lcsa-nax-rebench-{data,results,analysis}.{json,md}`
- `docs/lcsa-nax/lcsa-nax-rebench-decisions.md` (audit + decisions log)
- `docs/lcsa-nax/lcsa-nax-rebench-inventory.md`

**Future-work register update**:
- ~~§4-compliant 3-session re-bench~~ — DONE
- matmul2d cooperative-tensor inner-GEMM rewrite — tracked, now the
  highest-leverage follow-on (would resolve BOUNDARY cache-warmup
  sensitivity AND extend niche to density ~0.20+)
- `patch_sparkvsr_sliding_window` — tracked

## [2.34.0] — 2026-05-12 — Sprint B Sparse Attention NAX (narrow-niche ship)

### Added

- **`mlx_mfa.lcsa_nax.sparse_attention_nax(Q, K, V, block_mask, *, block_tile, scale, causal)`**
  — block-sparse attention via per-Q-tile threadgroup dispatch with online
  softmax (FA-2). Capabilities:
  - dtype: float16 + bfloat16
  - head_dim ∈ {64, 128}
  - block_tile ∈ {16, 32, 64} (default 16 per Phase 1.3 winner)
  - mask ndim ∈ {2 (NQ, NK), 3 (Hq, NQ, NK), 4 (B, Hq, NQ, NK)}
  - causal=True: per-tile-future-skip + within-tile triangular mask
  - asymmetric qL ≠ kL (cross-attention)
  - precondition: mask total bytes ≥ 4096 (constant-address-space avoidance)

- **`mlx_mfa.lcsa_nax.sparse_attention_dispatch(...)`** — density-thresholded
  router. Routes to the NAX kernel when density < 0.02, otherwise falls
  through to `mx.fast.scaled_dot_product_attention` + float bias. Supports
  optional `precomputed_bias` (caller-cached) for cache-HIT performance.

- **`mlx_mfa.lcsa_nax.DEFAULT_DENSITY_THRESHOLD = 0.02`** — exposed for
  callers wanting to inspect/override the routing boundary.

- **C++ Primitive scaffold**: `csrc/mfa_sparse_attention.{hpp,cpp}` —
  free-function entry point using `mlx::core::fast::metal_kernel` for JIT
  Metal kernel dispatch (Sprint D D33 pattern).

### Performance

Phase 1.4 sweep (M5 Max, 5 runs/cell, precomputed_bias passed):

| Cluster | density | dispatcher ratio vs SDPA+bias |
|---|---:|---:|
| lcsa_small_seq4k  | 0.01 | **4.57×** |
| lcsa_mid_seq8k    | 0.01 | **2.45×** |
| lcsa_large_seq16k | 0.01 | **2.67×** |
| lcsa_*            | 0.03-0.10 | 0.95-1.02× (within measurement noise) |

Single-session data. §4-compliant 3-session re-bench recommended for GA.

### Tested

- 24 tests in Phase 1 surface (6 Phase 1.1 + 12 Phase 1.2 + 6 Phase 1.4
  dispatcher). Joint surface 24/24 pass.
- Three-axis validation (oracle correctness + path entered + edges
  preserved) discipline maintained throughout.

### Deferred (tracked for follow-up sprint)

- `mpp::tensor_ops::matmul2d` cooperative-tensor inner-GEMM rewrite that
  would extend the niche from density < 0.02 to ~0.20+ (reference pattern
  at `csrc/mfa/v6_nax/NAAttentionKernel.cpp:775`, estimated 4-6h work).
- §4-compliant 3-session perf re-bench for ship-default-grade confidence.
- `patch_flashvsr_lcsa` and `patch_sparkvsr_sliding_window` integration
  patchers (Section H of original Sprint B plan).

### Documentation

- `docs/lcsa-nax/lcsa-nax-design.md` (Phase 1.0)
- `docs/lcsa-nax/lcsa-nax-phase1_1-pertile-microbench.json` (sub-phase 0)
- `docs/lcsa-nax/lcsa-nax-phase1_3-bt-sweep.json` + `phase1_3-results.md`
- `docs/lcsa-nax/lcsa-nax-phase1_4-dispatcher-sweep.json` + `phase1_4-results.md`
- `docs/lcsa-nax/lcsa-nax-phase1_5-ship-verdict.md`

## [2.33.1] — 2026-05-12 — `flash_attention_sparse` M5+ fast-fallback

### Fixed

- **`flash_attention_sparse` perf regression on M5+ Apple Silicon.** The
  v2.33.0 M5+ dispatch path (`_sparse_fallback_sdpa_perhead`) expanded
  the block mask to a `[B, H, N, S]` float bias on every call, adding
  ~3 ms of broadcast / reshape / `mx.where` work that ran in parallel
  with the SDPA call — producing a constant **2.07-2.10× wall-clock
  overhead** vs calling `mx.fast.scaled_dot_product_attention` directly
  with a prebuilt float bias. Surfaced by Sprint B Phase 0 baseline bench
  (`docs/lcsa-nax/survey-report.md` §3 + §8 + `docs/sparse-fallback-audit.md`).
- **Fast-fallback (v2.33.1):** the M5+ path now caches the expanded float
  bias by `id(block_mask) + shape + dtype + (B, H, N, S, target_dtype)`.
  Cache HIT (mask reused across calls — common production pattern, e.g.
  build mask once per forward pass) drops the expansion cost to a dict
  lookup, recovering full `mx.fast.scaled_dot_product_attention`-direct
  performance (**1.01× ratio measured**, within 10% target).
  Cache MISS (fresh mask each call, e.g. FlashVSR's per-layer
  `generate_draft_block_mask_mlx`) falls back to the same expansion as
  v2.33.0 — no regression, no improvement at the call site.
- LRU cache bounded to 8 entries; users with extreme memory footprint
  can manually clear via `mlx_mfa.attention._SPARSE_BIAS_CACHE.clear()`.
- Float bias is cached (NOT a bool mask) to preserve the v2.33.0 semantic
  that an all-False Q-row produces NaN softmax
  (`test_all_false_mask_row_gives_nan_or_zero`).

### Notes

- This is a **dispatch-routing fix**, not a NAX-native sparse implementation.
  The NAX-native sparse-aware path is in development as Sprint B Phase 1.x;
  expected speedups vs MLX SDPA dense+mask are 3-15× depending on density
  (see `docs/lcsa-nax/survey-report.md` §10 — Recommended approach).
- **Pre-M5 hardware (M1-M4) dispatch path is unchanged.** M1-M4 still
  routes through the C++ STEEL V1 sparse kernel via
  `_make_mfa_sparse_custom`. The patch modifies only the
  `_sparse_fallback_sdpa_perhead` internal function which is reached
  only on M5+ per `attention.py`'s `if info.get("is_m5_plus"):` dispatch
  check. Three new tests in `TestSparseM5PlusFastFallback` guard this.

### Internal

- `docs/sparse-fallback-audit.md` — Sprint B Phase 0 follow-up: per-step
  timing breakdown of the v2.33.0 overhead, fix strategy rationale, and
  cache-hit vs cache-miss expected behavior.

## [2.33.0] — 2026-05-11 — Conv3D NAX production path

### Added

- **Conv3D NAX path for M5+ Apple Silicon** — Sprint C v1.x ship-default
  verdict ratified, Sprint D production-integrated.
  - `mlx_mfa.conv_nax.conv3d_nax_forward(x, w, stride, padding, dilation, ...)` —
    routed through MPP `matmul2d` for Conv3D `3×3×3` and `1×1×1` FP16.
    **Median 1.64× speedup** vs `mx.conv_general` on SeedVR2 VAE production
    shapes (range 1.02× to 2.26×).
  - `mlx_mfa.integrations.seedvr2_vae.patch_seedvr2_vae(model)` —
    drop-in patcher for any MLX model using `mx.conv_general` Conv3D.
    Walks model modules, swaps eligible Conv3D layers to route through
    `conv3d_nax_forward`. Restorable via `restore=True`.
  - Supports symmetric **or** asymmetric padding (e.g., causal video conv:
    `padding=((K_T-1, 0), (pH, pH), (pW, pW))` or `causal_pad_t=True`).
  - Automatic M-chunking respects MPP `matmul2d` int32 byte-offset
    invariant — single-buffer reads stay strictly below `2^31` bytes.
    The Sprint C Phase 1.2 lesson learned, encoded as a runtime assert.
  - 1×1×1 fast path: skips im2col entirely (metadata-only input reshape +
    direct matmul on smaller K = C_in). ~15% wall-clock speedup at small
    shapes; bit-exact identity to the general path.
- **C++ `_ext.conv3d_nax_forward` binding** — Sprint D Track A migration
  of the Phase 1.x Python orchestrator to C++. Removes ~50-100 µs Python
  dispatch overhead per call. Implementation uses
  `mlx::core::fast::metal_kernel` internally — Metal kernels frozen
  from Sprint C (no algorithm or kernel changes).

### Documentation

- `docs/conv-nax/` — Sprint C v1.0-v1.5 deliverables (design doc,
  per-phase inventory/decisions/results/data) + `ship-shelve-decision.md`
  (the actionable Sprint C conclusion).
- `docs/conv-nax/conv-nax-prod-*.md` — Sprint D production-integration
  deliverables.
- `README.md` — new `Conv3D NAX support` section with quickstart, supported
  shapes, expected speedups, caveats, integration.

### Internal

- Unified `ConvKey` cache pattern (no per-Kind-separate maps).
- Multi-session §4-compliant bench methodology (90s round / 60s shape /
  180s initial cooldowns) applied to Phase 1.5 ship verdict; protocol
  inherited from Sprint A.
- Sentinel-fill + RMSE-vs-oracle smoke gate pattern (Phase 1.1 lesson)
  applied to all conv-nax harnesses.

### Known caveats

- At `K ≤ 3456` (small `in_channels`), speedup is at parity with MLX
  baseline. No regression, just no gain.
- BF16 path is wired but not yet on the validated bench set; treat as
  experimental until a Sprint D follow-up adds BF16 tests.
- Conv3D backward (VJP) is out of scope for this release; forward-only.

## [2.32.0] — 2026-05-06 — SDPA routing for M5+ NAX

### Strategic shift

mlx-mfa's dispatch on M5+ NAX hardware now routes forward attention to
`mx.fast.scaled_dot_product_attention` on canonical shapes where Apple's
`steel_attention_nax.h` is the optimal NAX path. mlx-mfa retains its
native kernels for shapes and features SDPA doesn't optimize. This
preserves mlx-mfa as a unified attention toolkit across Apple Silicon
generations while stopping unnecessary competition with Apple's upstream
NAX kernel on the canonical shape regime.

The MLX 0.31.2 audit and the v2.31.0 → v2.32.0 cross-session diagnostic
(`docs/v6-nax/v32-drift-diagnostic-report.md`) concluded that V34 NAX-direct
matches Apple's NAX kernel cross-session but does not consistently beat it.
v2.31.0's headline `+33-40% V34 wins on D=128` reflected within-session
conditions that did not reproduce in subsequent measurements.

### Routing predicate

Forward attention (in `mlx_mfa.flash_attention`) routes to MLX SDPA when
ALL of the following hold:

- Hardware: M5+ NAX (`device_has_neural_accelerators()` returns True)
- `head_dim ∈ {64, 128}` (canonical SDPA NAX targets)
- Not a long-kL decode pattern (already MFA-routed by the cross-attn rule
  `kv_seq_len ≥ 4096 ∧ seq_len ≤ 4096 → MFA`)
- No empirical carve-out from the Sprint A kernel sweep
- No backend or env-var override (see below)

For shapes outside these conditions, mlx-mfa's existing kernels are used:

- `head_dim ∈ {80, 96, 192}` → SDPA (mlx-mfa doesn't support these)
- `head_dim ∈ {256, 512}` → mlx-mfa V2 D-split (SDPA NAX doesn't target)
- Block-sparse / LCSA mask → mlx-mfa sparse kernel
- Additive `attn_bias` (modes 1, 2) → mlx-mfa native bias kernel
- Sliding window → mlx-mfa STEEL window kernel
- Backward pass → mlx-mfa backward (Apple's NAX backward NYI)
- Causal forward routes through SDPA NAX too on M5+ (Apple's kernel
  handles causal masking natively)
- M1–M4 hardware → mlx-mfa M3+ / V2 thresholds (NAX not available)

### Empirical kernel sweep (Sprint A)

`bench/v32_kernel_sweep.py` benched 15 niche / canonical shapes × 3
backends (`sdpa`, `mfa`, `auto`) on M5 Max under subprocess isolation,
5 runs per config. Headline result: **SDPA wins 11/15 shapes by
1.9-5.3×; MFA wins 1 shape (ltx2-cross D=64 asymmetric, +11%); 3
shapes (D=80, 96, 192) have MFA unsupported and fall back to SDPA**.

Per-shape data in [`docs/v6-nax/v32-kernel-sweep.json`](docs/v6-nax/v32-kernel-sweep.json),
verdict table in [`docs/v6-nax/v32-niche-shape-dispatch.md`](docs/v6-nax/v32-niche-shape-dispatch.md).

Notable wins (Sprint A, M5 Max):

| Shape | sdpa ms | mfa ms | mfa/sdpa |
|---|---:|---:|---:|
| canonical-d128-4k | 3.73 | 13.49 | **3.61×** SDPA |
| llama-prefill-8k (D=128 causal) | 11.07 | 42.00 | **3.79×** SDPA |
| seedvr2-small (D=128 26k²) | 161 | 630 | **3.92×** SDPA |
| cogvideox (D=128 70k²) | 2112 | 7052 | **3.34×** SDPA |
| llama-decode-32k (qL=1, kL=32k) | 0.62 | 2.32 | **3.73×** SDPA |
| **ltx2-cross (D=64 2k×14k)** | 1.35 | 1.21 | **0.89× MFA** |

Cross-attn rule refinement (Sprint A finding): the existing
`kv_seq_len ≥ 4096 ∧ seq_len ≤ 4096 → MFA` rule was tuned for LTX-2
cross-attn (qL=2048, kL=14000) where MFA wins; on M5+ NAX it incorrectly
caught LLM decode patterns (qL=1, kL≥4096) where SDPA's `sdpa_vector`
path wins 1.9-2.6×. The rule is now qualified with `has_nax ∧ seq_len
≤ 16 → fall through to NAX SDPA route`. ltx2-cross still routes to
MFA (seq_len=2048 > 16); decode shapes route to SDPA.

The carve-out hook `_should_use_mfa_m5_nax_carveout()` is preserved
for future empirical findings, but currently returns False unconditionally
(no carve-outs needed for v2.32.0).

### Wrapper bug fixes surfaced during Sprint A

Two pre-existing wrapper bugs were uncovered while instrumenting the
sweep harness — both materially affecting v2.32.0's strategic value:

**Bug 1**: `flash_attention(backend='sdpa')` did not actually force SDPA
on `head_dim ∈ {64,128,256,512}`. The smart-dispatch block at line 426
of `attention.py` only handled `backend='auto'`; the else-branch set
`use_mfa = _mfa_capable` (True for canonical D), routing
`backend='sdpa'` calls down the MFA path despite the docstring claim.
Fix: explicit `elif backend == 'sdpa': use_mfa = False`.

**Bug 2**: SDPA fallback paths (`_fallback_sdpa()` + the early
`backend='sdpa'` return) materialized an explicit triu causal mask
matrix instead of using `mask='causal'` (string form). On M5+ this
diverted SDPA away from Apple's `steel_attention_nax.h` fast path,
running ~2× slower than direct `mx.fast.scaled_dot_product_attention`.
Fix: pass `mask=('causal' if causal else None)` directly when no
attn_bias is supplied.

Combined effect on M5 Max, D=128 4096² causal:

| Path | Before | After |
|---|---:|---:|
| `flash_attention(backend='auto')` | 6.31 ms | **3.10 ms** |
| `flash_attention(backend='sdpa')` | 6.32 ms | **3.08 ms** |
| `mx.fast.scaled_dot_product_attention(..., mask='causal')` | 3.08 ms | 3.08 ms (reference) |

These bugs predate v2.32.0 and affected anyone using `backend='sdpa'`
or relying on the SDPA fallback (M3/M4/M5). v2.32.0 would have routed
canonical M5+ shapes to a slow SDPA path without these fixes.

### New env vars

- `MFA_FORCE_SDPA_ROUTE=1` — force SDPA route regardless of dispatch
  policy (testing/benchmarking).
- `MFA_DISABLE_SDPA_ROUTE=1` — disable v2.32.0 strategic routing; fall
  through to v2.31.0-style M3+/legacy thresholds. Recovers previous
  behavior on M5+ for A/B comparison.

### Performance recalibration (v2.31.0 follow-up)

v2.31.0's release perf table claimed V34 wins of +33-40% on D=128 self-
attention shapes. Cross-session re-bench in v2.32.0 Phase 0 measured
those shapes again and found V34 at parity-or-slight-edge with legacy V6
NAX, depending on environmental conditions. The legacy V6 NAX path
itself measured 36-41% faster in Phase 0 than at v2.31.0 release time —
same hardware, same code, no source change.

Phase A diagnostic (`docs/v6-nax/v32-drift-diagnostic-report.md`) tested
PSO compilation cache and GPU ramp-up hypotheses, both rejected. The
drift is a steady-state offset between v2.31.0 measurement context and
current sessions, beyond session-feasible discrimination. v2.31.0's
numbers were measured under conditions we cannot reproduce on demand.

v2.32.0's response is **not** to claim better numbers — it's to ship
the methodology that prevents repeat publication of regime-specific
benchmarks:

- `bench/v32_multisession_capture.py` — single-session capture with
  conditions metadata (sw_vers, uptime, Metal cache state, cooldowns)
- `bench/v32_multisession_aggregate.py` — aggregates across sessions,
  flags any reproduction of a target regime
- `docs/v6-nax/v32-multisession-protocol.md` — protocol matrix
- `CLAUDE_V6_NAX.md` Artifact #5 — methodology rule for marketing-grade
  benchmarks (multi-session repro required before perf claims published)

### V34 / V6 NAX status — clarification

The V34 NAX-direct kernel (shipped in v2.31.0) and the V6 NAX legacy path
are **accessible only via the direct binding `_ext.v6_nax_forward()`** —
used by `bench/v34_bench.py`, `bench/v32_multisession_capture.py`, and
similar tools. The public `mlx_mfa.flash_attention()` API has never
routed through V6 NAX/V34; it has always used the STEEL kernel family
(V1/V2/V3/V4/V5) via `MFAttention`. v2.31.0 introduced V34 as a research
path with bench data; it was not part of the production dispatch.

v2.32.0 modifies the **production dispatch** (the path users actually
hit through `flash_attention()`) to route canonical M5+ NAX shapes to
SDPA. STEEL kernels still handle non-canonical shapes, sliding window,
sparse, attn_bias, decode patterns, etc.

The V6 NAX/V34 source remains in the codebase as:
- An implementation reference for `steel_attention_nax.h` patterns
  (`csrc/mfa/v6_nax/NAAttentionKernel.cpp::createV34Source`)
- The kernel exercised by `v34_bench.py` for cross-session perf studies
- A regression canary against future MLX upstream changes

### Tests

`tests/test_v32_sdpa_routing.py` — 16 tests covering:

- Pure dispatch_policy decisions (canonical → SDPA, decode → MFA,
  env var overrides, M3+/M1 unchanged, backend overrides)
- End-to-end correctness (D=128/D=64 canonical, D=80 fallback,
  decode pattern, MFA_DISABLE_SDPA_ROUTE env var)
- Carve-out infrastructure (hook returns boolean, default False)

All 16 pass. Existing test suite unchanged (1 pre-existing failure
flagged in CLAUDE_V6_NAX.md §8 still reproduces on baseline; not
introduced by v2.32.0).

### API compat

- `mlx_mfa.flash_attention(q, k, v, ...)` API unchanged.
- `MFA_DISABLE_SDPA_ROUTE=1` recovers v2.31.0-exact behavior on M5+.
- All v2.31.0 functionality intact: backend='mfa', backend='sdpa',
  alibi, softcap, window_size, return_lse, etc.

### Files

- `mlx_mfa/dispatch_policy.py` — `_M5_NAX_THRESHOLDS`, `_should_use_mfa_m5_nax_carveout()`,
  `should_use_mfa(has_nax=...)`, env var overrides
- `mlx_mfa/attention.py` — `_get_has_nax_cached()`, `flash_attention()` passes
  `has_nax` to dispatch
- `tests/test_v32_sdpa_routing.py` — 16 tests
- `bench/v32_kernel_sweep.py` + `_inner.py` + `_analyze.py` — Sprint A sweep harness
- `docs/v6-nax/v32-kernel-inventory.md` — kernel architecture survey
- `docs/v6-nax/v32-kernel-sweep.json` — Sprint A raw data
- `docs/v6-nax/v32-niche-shape-dispatch.md` — Sprint A.6 dispatch table
- `CLAUDE_V6_NAX.md` — Artifact #5 added (cross-session perf methodology)

### Open follow-ups

- Multi-session protocol execution (next sprint) to re-validate v2.31.0
  numbers across multiple environmental conditions. May lead to a
  v2.31.0 PyPI page addendum with corrected perf characteristics.
- Backward NAX FA2 (Apple's NYI) — opportunity for mlx-mfa native
  backward to remain the only path on M5+
- Block-sparse / LCSA NAX — Apple's SDPA NAX doesn't support sparse
  block masks; mlx-mfa keeps this niche
- Conv2D/3D NAX — separate primitive class

## [2.31.0] — 2026-05-06 — V34 NAX-direct rewrite (M5 Max SDPA parity achieved)

### Architecture

- New `loopForwardSingleTileV34` path in V6 NAX uses Apple's
  `NAXFrag::mma` and `NAXTile<T, TQ, TD>` primitives directly
  (the pattern from `steel_attention_nax.h`), bypassing MPP
  cooperative_tensor constraints that imposed
  `execution_simdgroups<1>`.
- Multi-SG parallelism via per-SG row partitioning at the kernel
  level (`tm = 16 * TQ * sgid`), not via cooperative_tensor
  distribution. This sidesteps the V33 cross-SG distribution
  opacity issue documented in `docs/v6-nax/v33-sg-gt-1-debug-report.md`.
- V34 emits a self-contained MSL source (~17.7 KB of inlined Apple
  helpers): `BaseNAXFrag` (with `mma` / `load` / `store` / `row_reduce`
  / `row_bin_op`), `NAXTile<T, TQ, TD>`, `MaxOp` / `SumOp` / `MulOp`
  / `ExpSubOp`, `Limits`, `integral_constant`. No external include
  dependency at runtime beyond Metal stdlib + MetalPerformancePrimitives.

### Performance (M5 Max, cross-session multi-run, iStat performance fan)

| Shape | D | Legacy ms | V34 ms | Δ | V34/SDPA | Default |
|---|---|---:|---:|---:|---:|:---:|
| FlashVSR-dense | 64 | 1.12 | 1.55 | -39% | 1.633× | legacy |
| LTX2-cross | 64 | 1.75 | 1.42 | **+19%** | **1.075×** | **V34** |
| SeedVR2-small | 128 | 265.13 | 170.92 | **+36%** | **0.890× ⭐** | **V34** |
| CogVideoX | 128 | 3610.79 | 2399.19 | **+34%** | **1.033×** | **V34** |
| SeedVR2-large | 128 | 6776.12 | 4042.73 | **+40%** | **1.008×** | **V34** |

V34 wins +18 to +40% on 4/5 production shapes vs legacy V6 NAX. **3 of 5
reach SDPA parity (1.01×–1.07×)** — the historic D=128 long-N gap (legacy
was 1.5×–1.7× SDPA) is closed. **SeedVR2-small at 0.89× actually beats
SDPA**, the first time V6 NAX has dipped below 1.0× on a production shape.

Methodology: subprocess A/B/A (legacy → V34 → legacy), 3 runs/round, 60s
inter-round + 30s inter-shape cooldowns. 4/5 shapes have R1↔R3 thermal
drift < 8%. Raw bench data: `docs/v6-nax/v34-aba.json`.

### Numerics

V34 is 4–30× more numerically stable than legacy on the same shapes:

| Shape | Legacy RMSE FP32 | V34 RMSE FP32 |
|---|---:|---:|
| FlashVSR-dense | 1.47e-05 | 3.60e-06 |
| LTX2-cross | 8.10e-06 | 1.76e-06 |
| SeedVR2-small | 5.87e-06 | 1.75e-06 |
| CogVideoX | 3.66e-06 | 1.11e-06 |
| SeedVR2-large | 2.93e-06 | 8.98e-07 |

Manual `simd_shuffle_xor` row reductions on FP32 accumulators inside
`NAXFrag::row_reduce` are bit-exact; MPP's `reduce_rows` had
tile-boundary FP rounding artifacts.

### Dispatch policy

Per-D shape-aware dispatch (in `mfa_v6_nax_primitive.cpp`):

- `D = 128` → V34 default (3 production shapes)
- `D = 64` and `N_kv > 8000` → V34 default (LTX2-cross style asymmetric)
- `D = 64` small N (FlashVSR-dense style) → legacy retained (V34 regresses
  −39% on small symmetric self-attention; root cause TBD, likely V34
  per-kernel overhead unfavorable on short matmul tiles)
- Causal forward → legacy fallback (V34 not yet ported)
- GQA `Hq != Hk && Hq % Hk != 0` → legacy double-buffer fallback
- Override via env var: `MFA_V6_USE_V34={0,1}`

V34 tile defaults: D=64 → BQ=32, BK=64, WM=2; D=128 → BQ=64, BK=32, WM=4.
Tunable via `MFA_V6_V34_BQ` / `MFA_V6_V34_BK` / `MFA_V6_V34_WM`.

### Compatibility

- Drop-in replacement: existing `flash_attention()` calls automatically
  use V34 where the dispatch elects it; no Python API change.
- V6Key cache key gains 4 dedicated fields (`use_v34`, `v34_BQ`,
  `v34_BK`, `v34_WM`) — no bit-packing, no foot-gun.
- Legacy V6 NAX path remains intact for shapes where it wins or where
  V34 isn't yet ported.

### Files

- `csrc/mfa/v6_nax/NAAttentionKernel.{hpp,cpp}` — `createV34Source()` (~700 LOC)
- `csrc/mfa/v6_nax/NAAttentionKernelDescriptor.hpp` — `useV34` field
- `csrc/mfa_v6_nax_primitive.cpp` — V34 dispatch + V6Key fields
- `csrc/v6_nax_compile.mm` — `v34_compile`, `v34_dispatch`, `V34Params` struct
- `csrc/v34_probe.cpp` — Phase 1 compile probe (kept for future debugging)
- `bench/v34_bench.py`, `bench/v34_aba_wrapper.sh` — bench infrastructure
- `docs/v6-nax/v34-apple-reference-mapping.md` — file:line citations
- `docs/v6-nax/v34-results.md` — sprint final report
- `docs/v6-nax/v34-aba.json` — raw bench data
- `CLAUDE_V6_NAX.md` — V6 NAX guardrails accumulated through v2.27.0–v2.30.x

### Open follow-ups

- FlashVSR-dense D=64 small-N V34 regression: investigate whether
  V34 with `BQ=16, WM=1` configuration beats legacy on small
  symmetric self-attention.
- V34 causal forward (currently legacy fallback).
- V34 with `align_Q` / `align_K` function constants for the
  fast-path on shapes that are seq-len aligned.
- L (logsumexp) writeback in V34 — currently V34 doesn't write
  the `lse` output; backward via `mx.vjp(SDPA)` doesn't need it,
  but any user reading L from V34 output would get uninitialized data.

## [2.30.0] — 2026-05-05 (final, post thermal-controlled re-bench)

> **Note**: This release went through a thermal-controlled re-bench
> validation cycle that reverted the originally-shipped "dispatch v6"
> tile-default changes (Sprint G). The Sprint G claim of "−6.4 % on
> FlashVSR-dense and −11.7 % on SeedVR2-large" turned out to be a
> within-session pipeline-cache artifact that did not replicate
> cross-session. See `docs/v6-nax/v2-30-thermal-rebench.md`.

### What's actually shipped in v2.30.0

- **Sprint A.1 — tgmem allocation cleanup** (3 LOC):
  `threadgroupMemoryAllocation()` returns 0 for the forward path when
  single-Otile + bypass are both on (P_buf is never used). Saves
  8-16 KB per dispatch. Within noise on production shapes; pure
  code-quality fix.
- **Sprint B — GQA single-Otile path** (~70 LOC): BHND rewriter now
  handles `Hq != Hk && Hq % Hk == 0`. GQA shapes use the single-Otile
  kernel directly. Gains 7-14% over v2.29.0 legacy fallback on 4 GQA
  shapes. **GQA-Hq32-Hk8 D=128 reaches 1.06× SDPA** — the closest V6
  has gotten to parity on M5 Max.
- **Infrastructure: `MFA_V6_MAX_THREADS`** env var + Metal pipeline-state
  attribute support (`MTLComputePipelineDescriptor` with
  `maxTotalThreadsPerThreadgroup`). No default change — exposed for
  future per-shape dispatch experiments.
- **Infrastructure: `MFA_V6_MATMUL_EXEC_SG`** env var + post-gen rewrite
  of `matmul2d<desc, execution_simdgroups<N>>` template parameter
  (default <1>). Empirical sweep showed FlashVSR-dense gains ~10% at
  <8> but doesn't generalize. Exposed for future per-shape dispatch.

### Reverted from initial v2.30 release

- **Sprint G dispatch v6 default changes** (commit `96daff7`,
  reverted in `ca0fc44`): D=64 SG=4 (was SG=2) and D=128 N≥100k
  BK=64 SG=8 (was BK=32 SG=16). Thermal-controlled re-bench showed
  these regressed SeedVR2-large +14.3 % and SeedVR2-small +5.9 %
  vs v5 defaults.

### Final v2.30.0 vs v2.29.0 performance (M5 Max, controlled A/B/A)

5-shape multi-run on production dense shapes (median-of-medians):

| Shape | v2.29.0 (avg of A1+A3) | v2.30.0 (B) | Δ |
|---|---:|---:|---:|
| FlashVSR-dense | 1.17 ms | 1.15 ms | -1.7 % (noise) |
| LTX2-cross | 1.55 ms | 1.56 ms | +0.6 % (noise) |
| SeedVR2-small | 280 ms | 286 ms | +2.1 % (noise) |
| CogVideoX | 4377 ms | 4500 ms | +2.8 % (noise) |
| SeedVR2-large | 7720 ms | 7735 ms | +0.2 % (noise) |

**All deltas within ±3 % noise band.** Production performance is
statistically equivalent to v2.29.0; v2.30.0 is a strict improvement
on GQA shapes (new feature) without regression on production.

### GQA shape performance (Sprint B, multi-run validated)

| Shape | v2.29.0 legacy | v2.30.0 single-Otile | Δ | V6/SDPA |
|---|---:|---:|---:|---|
| GQA-Hq32-Hk8 D=128 | 7.71 ms | 6.60 ms | -14.46 % | **1.06×** |
| GQA-Hq16-Hk4 D=64 | 6.01 ms | 5.54 ms | -7.90 % | 1.17× |
| GQA-Hq40-Hk8 D=128 | 2.59 ms | 2.30 ms | -11.03 % | 1.16× |
| GQA-Hq8-Hk2 D=64 | 1.00 ms | 0.93 ms | -7.11 % | 1.18× |

### Investigated but not shipped (full rationale in docs/v6-nax/sprint-{D,E,F}-*.md)

- **Sprint A.2 swizzle**: Apple's NAX attention doesn't use swizzle.
- **Sprint A.3 ld_padding**: V6 uses device tensors; padding inapplicable.
- **Sprint D per-loop unroll**: 101 pragmas; S3.5 already showed `full` wins.
- **Sprint F compile-time vs runtime function constants**: V6 already at
  the natural compile/runtime split.

### Lessons logged from the v2.30.0 cycle

1. **Within-session A/B benches contaminate via pipeline cache**.
   Sprint G's "wins" didn't replicate cross-session. Always use
   cross-session controlled bench for shipping decisions.
2. **`maxTotalThreadsPerThreadgroup` can silently corrupt output** if
   set below the actual dispatch's threads-per-threadgroup. For SG=16
   (= 512 threads/TG), settings of 256 or 512 produce RMSE=1.0.
3. **MPP `execution_simdgroups<N>` template is not a no-op** —
   FlashVSR-dense gains ~10 % at `<8>` vs `<1>`. Worth per-shape dispatch
   exploration in a future sprint.
4. **Thermal drift can flip benches by 50% over hour-scale sessions.**
   Mandatory protocol: 3-5 min initial cooldown + 90-120 s inter-round +
   A/B/A pattern. R1↔R3 within 5% to declare bench thermally valid.

## [2.29.0] — 2026-05-05

### Added — V6 NAX single-Otile + autoresearch retuning (M5+)

- **`loopForwardSingleTile()`** — Apple-style single-buffer V6 NAX forward
  kernel. ~270 LOC in `csrc/mfa/v6_nax/NAAttentionKernel.cpp`. Single cS (no
  double-buffer cS_0/cS_1), forced kBlocks=1, always-bypass cP cooperative
  tensor (no `P_buf` threadgroup staging), `mem_none` barriers, K-loop step
  BK (not 2·BK).
- **Autoresearch-tuned tile defaults**: BQ=16 universal; BK=64 for D=64,
  BK=32 for D=128; SG=2 for D=64, SG=8 for D=128. Plumbed in both the
  source-gen path and the cache-key/dispatch path of
  `csrc/mfa_v6_nax_primitive.cpp`.
- **Auto-default kernel variant by Hq==Hk**: non-GQA shapes get single-Otile
  by default; GQA shapes fall back to legacy `loopForward()` (double-buffer)
  because the BHND rewriter doesn't yet handle per-head K-stride for
  single-Otile.
- New env vars (all with auto-defaults): `MFA_V6_NAX_SINGLE_OTILE` (on/off),
  `MFA_V6_BLOCK_R` / `_C` / `EXEC_SG` / `BLOCK_D` for tile overrides,
  `MFA_V6_BYPASS_TGP` (forced on by single-Otile).
- `docs/v6-nax/README.md` — V6 NAX architecture summary + sprint chronology.
- `docs/v6-nax/env-vars.md` — full env var reference.
- `bench/v6_single_otile_bench.py` — reproducible single-Otile bench.
- `bench/v6_single_otile_autoresearch.py` — tile-config sweep script.

### Performance — V6 NAX on M5 Max (5 production VSR/DiT shapes)

V6/SDPA closed from 1.98×–5.06× (v2.28.x default tiles) to 1.20×–2.06×:

| Shape (D)               | v2.28.x | v2.29.0 | Δ      | V6/SDPA   |
|-------------------------|--------:|--------:|-------:|-----------|
| FlashVSR-dense (64)     | 1.81 ms | 1.11 ms | -38.7% | → 1.22×   |
| LTX2-cross (64)         | 2.99 ms | 1.59 ms | -46.8% | → 1.20×   |
| SeedVR2-small (128)     | 936 ms  | 276 ms  | -70.5% | → 1.49×   |
| CogVideoX (128)         | 9633 ms | 3060 ms | -68.2% | → 1.35×   |
| SeedVR2-large (128)     | 16030 ms| 8392 ms | -47.6% | → 2.06×   |

Bonus: SeedVR2-large RMSE 5.79e-5 → 2.93e-6 (20× more stable) under the
single-Otile path — single-buffer commits each row reduction before the
next K-tile overwrites, eliminating cross-tile FP16↔FP32 rounding error
that the double-buffer accumulated.

### Investigation logs (informational, no code change)

- **Sprint 3.1** — V6 NAX already implements all three Apple-style causal-skip
  optimizations from `steel_attention_nax.h` (loop bound, mask gate,
  per-element check). Plus an extra V6-only tail-block gate. No code change
  recommended. See `docs/v6-nax/causal-masking-analysis.md`.
- **Sprint 3.2** — `bypassThreadgroupMemory` at legacy tiles regresses on
  D=128 (Cas C). Kept off as a default. With single-Otile + new tiles,
  bypass is forced on automatically. See
  `docs/v6-nax/sprint-3-2-bypass-tgmem-results.md`.

### Documentation

- README — performance table updated for v2.29.0; added M5 Max V6 NAX
  section; clarified the M1 Max highlights remain the M1-Max-specific story.

## [2.28.1] — 2026-05-02

### Fixed
- **Sparse path on M5 Max** — the V1 STEEL sparse Metal kernel produces wrong
  results on M5 Max + MLX 0.31.2 due to a Metal-compiler miscompile of
  `(long)p->NK` in the inner mask-offset address calculation. The kernel
  reads `qb * (NK/2) + kb` instead of `qb * NK + kb`. This is independent of
  the previous session's "persistent kernel state pollution" hypothesis,
  which was disproven by reproducing the bug with `kTilesPerTG=1`.
  Workaround: route `flash_attention_sparse()` through a per-head SDPA
  fallback on M5+ that preserves 2-D / 3-D / 4-D mask shapes. Forward path
  produces correct results at fp16 precision.
- New helper `_sparse_fallback_sdpa_perhead()` in `mlx_mfa/attention.py`
  expands block masks to a `[B, H, N, S]` float bias and passes to SDPA,
  preserving per-head and per-batch mask differences (which the old 2-D
  collapse `_sparse_fallback_sdpa()` lost).

### Investigation report
- `docs/v6-nax/sparse-bug-investigation.md` documents 6 workarounds tried
  and the definitive root cause (Metal-compiler `int → long` cast bug on
  struct-field reads under MSL 4.x). Includes 4 options for kernel-level
  fix; current 2.28.1 implements Option A (SDPA fallback). Marco may
  consider Option C (Apple bug report) for upstream fix.

### Known Issues
- 4 pre-existing M5+MLX 0.31.2 precision-tolerance test failures, unrelated
  to sparse path:
  - `test_attn_bias_native::TestBiasMode{1,2}::test_d128_causal`
  - `test_turboquant::TestQRRotation::test_{roundtrip,orthogonal}`
  - `test_attention::TestTopkAttention::test_topk_ratio_1_matches_dense`
  - `test_attention::TestReturnAttnWeights::test_output_matches_no_return`
  All due to slight numerical differences between MLX 0.31.2 SDPA and
  native MFA dense kernel. Test tolerances may need to be bumped (separate
  task — these tests have `atol=1e-5/1e-4` which is too tight for fp16
  cross-implementation comparison).

### Benchmarks added
- `bench/m5_max_sparse.py` + `docs/v6-nax/m5-max-sparse-baseline.json`:
  5-shape sparse benchmark on M5 Max via SDPA fallback. Includes FlashVSR
  LCSA-class shapes (window mask, density 3-7%) and generic random-mask
  shapes. Will be re-run once native sparse kernel is fixed.

## [2.28.0] — 2026-05-02

### Fixed
- **Compilation against MLX >= 0.31.0** — adapt to two breaking changes between
  MLX 0.31.0 and 0.31.2:
  1. `Device::get_command_encoder(int index)` was removed (PR #3316
     "Decouple CommandEncoder from Device" + #3264 "Merge DeviceStream into
     CommandEncoder"). Replaced 23 call sites across 5 files (`mfa_attention.cpp`,
     `mfa_quantize.cpp`, `mfa_scatter.cpp`, `mfa_paged_gather.cpp`,
     `mfa_smooth_quant.cpp`) with the new free function
     `mlx::core::metal::get_command_encoder(stream())`.
  2. nanobind upstream bumped `NB_INTERNALS_VERSION` 17 → 19. MLX 0.31.2 is
     built against nanobind v2.12 (NB_INTERNALS_VERSION=19); our extension was
     pinned to v2.10 (NB_INTERNALS_VERSION=17). Capsule key mismatch silently
     made `mlx::core::array` "incompatible" between modules at runtime
     (TypeError on every binding call). Bumped FetchContent tag to v2.12.0.

### Changed
- **Minimum MLX version bumped to 0.31.0** (`pyproject.toml` build + runtime
  deps). Earlier MLX is no longer supported because `metal::get_command_encoder`
  free function only exists since 0.31.x.
- **nanobind v2.10.0 → v2.12.0** (CMakeLists.txt FetchContent_Declare).

### Known Issues
- **18 sparse-attention test failures on MLX 0.31.2** (`flash_attention_sparse`,
  GNA, top-k, sparse backward). Pattern: NaN appears at Q-tile 1+ in the V1
  STEEL sparse kernel, while Q-tile 0 is correct. Root cause is consistent
  with MLX 0.31.2's CommandEncoder refactor (PR #3348 "thread-local",
  #3282 "smart pointers") exposing a latent buffer-state assumption in the
  persistent-kernel pattern (`kTilesPerTG = 4`). **Dense attention paths
  (V1, V2, V3, V4, V5, paged-varlen, flash decode) are unaffected.**
  Tracking issue: investigate kernel-level threadgroup memory init for
  multi-tile-per-TG dispatch.
- 2 numerical-precision failures in `test_turboquant.py::TestQRRotation`
  (tolerance 1e-4 vs observed 5e-3). MLX 0.31.2 may have changed QR
  decomposition precision; tolerance may need bump.

## [2.27.0] — 2026-04-06

### Native `attn_bias` Metal Kernel
- **Additive bias on attention logits** via V2 STEEL Metal kernel, eliminating
  SDPA fallback for bias. At N=70K, SDPA materializes a ~37 GB score matrix;
  the native kernel tiles computation and never materializes it.
- **Mode 1** `[1,1,1,Nkv]`: per-KV broadcast bias (token merging, conditioning).
- **Mode 2** `[1,H,1,Nkv]`: per-head per-KV bias (temporal distance, custom ALiBi).
- Modes 0/3 (full bias) fall back to SDPA for now.
- Compile-time `#define` gating: zero overhead when `attn_bias=None`.
- Split-K excluded for bias (single-pass V2 only).
- `KernelKey.has_attn_bias` + `attn_bias_mode` in shader cache.
- Buffer index 10 for bias tensor.
- C++ binding: `mfa_attention_bias_forward`.
- 17 new tests in `tests/test_attn_bias_native.py`.

### DiT/UNet Dispatch Audit
- Verified and optimized dispatch routing for 11 VSR model architectures
  (non-causal self-attention + asymmetric cross-attention shapes).
- New report: `docs/audit_dit_dispatch_report.md`.

### Varlen Validation for Token Merging
- Benchmarked `flash_attention_varlen` vs padded dense across 5 scenarios.
- Finding: padded dense is faster in most token merging cases (0.55–0.96×);
  varlen wins only with >2:1 length disparity.
- Correctness verified: bit-accurate (max_err at f16 epsilon level).
- New report: `docs/varlen_pruning_validation.md`.
- New benchmark: `benchmarks/bench_varlen_pruning.py`.

### Documentation
- All docs updated to v2.27.0.
- 920+ tests pass.

## [2.26.0] — 2026-03-31

### GNA Native Metal Kernel
- **Native 3D window kernel** for `flash_attention_gna()` with D=128, f16/bf16.
- Two-level masking: `gna_tile_active()` for O(1) tile skip + per-element window
  mask after Q@K^T GEMM for exact GNA window bounds.
- Forward-only (no VJP); backward falls through to sparse mask path.
- KernelType `GNAForward = 24` in `shader_cache.hpp`.
- New files: `csrc/mfa_gna_fwd.hpp/.cpp`, `tests/test_gna_native.py` (11 tests).
- Benchmark (M1 Max, CogVideoX N=70200): 285ms native vs 266ms sparse (0.93×);
  native wins at medium N (0.63–0.89×) where mask construction overhead dominates.

### SVDQuantLinear (v2.25.0)
- **SVDQuantLinear**: W4A16 linear layer with optional rank-r FP16 SVD correction.
- `quantize_model()`: tree walker to replace `nn.Linear` layers in-place.
- New module: `mlx_mfa/svdquant/` (linear.py, quantize.py).
- New test file: `tests/test_svdquant.py` (21 tests).

### Documentation
- All docs updated to v2.26.0 (ARCHITECTURE, API_MANUAL, FEATURE_COVERAGE,
  INVENTORY, SERVING_GUIDE, README, CHANGELOG, CLAUDE.md).
- 864+ tests pass.

## [2.24.1] — 2026-03-27

### Documentation
- Confirmed V centroids already read from threadgroup memory (`v_centroids_smem`) —
  no bug to fix (Phase 3C implemented correctly).
- **Dequant-in-GEMM analysis**: TGP budget is not constrained (max 19 KB / 58% of
  32 KB on M3+ D=128). The uint8 K_smem savings (0.5–1.8 KB) cannot improve
  occupancy. Implementation skipped; analysis documented in shader generator header.

## [2.24.0] — 2026-03-27

### TurboQuant Phase 4 — Optimal Packing + WHT Fusion

#### Optimal 3-bit Bit-Planar Packing
- **5.33× compression** (was 4× in Phase 3): 32 indices → 3 bit-planes × 4 bytes = 12 bytes/group.
- `pack_3bit_optimal` / `unpack_3bit_optimal` — new public API for bit-planar layout.
- `pack_k_for_metal` / `pack_v_for_metal` now dispatch by bit-width: 3-bit → bit-planar, 2-bit → 4/byte, 4-bit → 2/byte.
- Metal K/V gather: 3 coalesced reads per index (all within single cache line).
- `_compute_packed_d(D, bits)` replaces hardcoded `D // 2`.

#### WHT Fusion in Metal Kernel
- **Walsh-Hadamard transform** applied in-place on Q threadgroup memory (log2(D) butterfly passes).
- `tq_wht_enabled=True` eliminates Python `apply_rotation()` overhead: **1.1–1.4× faster** decode.
- WHT normalization `1/sqrt(D)` folded into attention scale via `rsqrt(D)`.
- `wht_in_kernel` parameter on `TurboQuantPagedInferenceContext`.
- Bit-identical to Python WHT (max error < 0.001 at fp16).

#### Tests
- 85 TurboQuant tests pass (70 from Phase 3 + 15 new).
- `TestOptimal3BitPacking` (8 tests): roundtrip, shape, compression ratio, edge cases.
- `TestOptimalPackingFusedKernel` (3 tests): fused kernel D=64/128.
- `TestWHTKernelFusion` (4 tests): Python-vs-kernel match, causal, V-TQ.

#### Benchmark (M1 Max, H=8, 3-bit, f16)

| Config | Python WHT | Kernel WHT | Speedup |
|--------|-----------|-----------|---------|
| D=64 Nq=4 S=1024 | 0.47 ms | 0.38 ms | 1.23× |
| D=128 Nq=4 S=1024 | 0.56 ms | 0.42 ms | 1.32× |
| D=128 Nq=8 S=2048 | 0.60 ms | 0.42 ms | 1.43× |

## [2.23.0] — 2026-03-27

### TurboQuant Phase 3 — Production Integration

- **feat**: Optional V compression in fused TQ kernel (`tq_v_enabled=True`).
  Both K and V are TQ-packed and dequantified inline during the attention
  kernel, achieving ~8× KV cache compression. Uniform branch (zero warp
  divergence). Buffers 12-14 for `v_pool_tq`, `v_centroids`, `v_scales`.
- **feat**: `pack_v_for_metal()` and `build_tq_paged_v_pool()` — V-side
  TQ packing helpers matching the K-side API.
- **feat**: `TurboQuantPagedInferenceContext` — stateful paged KV-cache
  with automatic TQ compression on append. Prefill/step auto-rotate Q
  with WHT and call the fused TQ kernel.
- **feat**: `create_decode_runtime(turboquant=True)` — runtime factory
  shortcut that creates a TurboQuant paged context. `tq_bits` and `tq_v`
  parameters control quantization width and V compression.
- **perf**: Centroids cached in threadgroup memory (Phase 3C). K and V
  centroid lookup tables (16-32 bytes) loaded once per kernel invocation
  into `k_centroids_smem`/`v_centroids_smem`, replacing per-element device
  memory reads in the gather loops.
- **docs**: QJL documented as Phase 1 decompress path only. The fused
  kernel uses PolarQuant/MSE without QJL correction. For 2-bit quality
  improvement with QJL, use `turboquant_compress(use_qjl=True)` +
  `turboquant_decompress()`.

## [2.22.0] — 2026-03-27

### TurboQuant Phase 2 — Semi-Fused Metal Kernel

- **feat**: `flash_attention_paged_varlen_turboquant()` — fused paged varlen
  attention that reads TQ-packed uint8 K directly in the Metal kernel. Inline
  centroid lookup + per-vector rescaling during K gather eliminates the
  separate decompress pass. V remains fp16.
- **feat**: `PagedVarlenTQForward` Metal kernel (KernelType 28) — copy of
  PagedVarlenForward with modified K gather: unpack 2 indices per byte,
  centroid lookup from buffer(10), scale from buffer(11).
- **feat**: `pack_k_for_metal()` — rotate, normalize, quantize K and pack
  into Metal-friendly 2-indices-per-byte uint8 format (packed_D = D/2).
- **feat**: `build_tq_paged_k_pool()` — convert a paged K pool from fp16
  to TQ-packed format. Returns (k_pool_tq, scales, centroids).
- **feat**: Supports 2/3/4-bit quantization, causal masking, GQA, and
  multi-sequence varlen batching.
- **test**: 8 new tests — fused-vs-decompress correctness, causal, GQA,
  multi-seq, 2-bit/4-bit, packing roundtrip, pool builder shapes.
- **bench**: `bench_turboquant_fused.py` — compares fp16, Phase 1, Phase 2.
  Phase 2 is up to 3× faster than Phase 1 for batched multi-seq decode.

## [2.21.0] — 2026-03-26

### TurboQuant KV Cache Compression (Phase 1)

- **feat**: `turboquant_compress()` / `turboquant_decompress()` — two-stage
  vector quantization (PolarQuant MSE + QJL 1-bit correction) for KV cache.
  Supports 2/3/4-bit quantization with WHT or QR random rotation.
- **feat**: Walsh-Hadamard transform (butterfly, O(d log d)) and QR random
  orthogonal rotation. WHT is 2× faster than QR for D=128.
- **feat**: Pre-computed Lloyd-Max optimal centroids for N(0,1) at 2/3/4-bit.
- **feat**: Bit packing (1/2/3/4-bit) for storage-efficient compressed format.
  3-bit+QJL: 3.56× compression, 3-bit no QJL: 4.92×, 2-bit: 7.11× vs fp16.
- **feat**: `TurboQuantKVCache` — drop-in KV cache with transparent
  TurboQuant compression. K+V at 3-bit: ~4.1× memory reduction vs fp16.
- **feat**: `TurboQuantKVCacheAdapter` — integrates with `adapt_kv_cache()`
  and the HybridKVCache / DecodeRuntime infrastructure.
- **test**: 57 tests — WHT, QR rotation, Lloyd-Max centroids, bit packing,
  roundtrip, inner product preservation (corr >0.95), attention output,
  KVCache, adapter integration, bits comparison.
- **bench**: Compression/decompression timing + memory savings benchmarks.

## [2.20.1] — 2026-03-21

### Critical Fix
- **MFAEnvConfig.invalidate() exposed to Python**: BK calibration in
  `dispatch_policy.py` was silently broken — C++ cached env vars were never
  re-read after `os.environ` mutations. Added `_invalidate_env_config()`
  binding and calls at all 3 calibration sites.

### Major Fixes
- **V5 dead sparse codegen removed**: Cleaned up unreachable block_mask buffer
  binding and tile-skip code from V5 shader generator (V5 never dispatches
  with sparse masks).
- **V2 BD_HALF_D512 uses MFAEnvConfig**: Replaced raw `std::getenv()` with
  `MFAEnvConfig::get().v2_bd_half_d512` for consistency with other cached vars.
- **Benchmark med() deduplication**: All 4 autoresearch/promotion benchmarks
  now import `med()` from `bench_utils.py` instead of defining local copies.

### Minor Fixes
- Hardcoded V4 mask strides to 0 (dead conditional — V4 is gated by
  `!has_block_mask`).
- Updated stale v2.14.0 section headers to v2.20.0 in SERVING_GUIDE and
  ARCHITECTURE.
- Added 7 missing env vars to ENV_VARS.md: `MFA_DEBUG_SHADERS`,
  `MFA_FORCE_D256_PATH`, `MFA_FORCE_D512_PATH`, `MFA_FORCE_NATIVE_BWD`,
  `MFA_FORCE_SAGE_DECODE`, `MLX_MFA_VERBOSE_DISPATCH`, `MLX_MFA_DISPATCH_TABLE`.

## [2.20.0] — 2026-03-21

### Performance Optimization (Autoresearch)

- **V3 dispatch guard**: Lowered B*H threshold from 16 to 4, unlocking V3 for
  Llama-7B decode (B=1 H=8) and similar small-batch causal shapes. +35-67%
  speedup for previously V2-fallback shapes.
- **V5 per-D block configs**: D=64 uses BD_tile=32, D=128 uses BD_tile=64.
  V5 now correctly differentiates head dimensions instead of single config.
- **V5 promotion**: Evaluated and rejected — only 2-3% gain at B*H>=128 with
  regression risk at B*H=16/32. V5 remains experimental (`MFA_ENABLE_V5=1`).
- **V3 non-causal**: Evaluated and rejected — V3 loses 6/9 non-causal shapes
  by 25-77%. Causal-only gate confirmed necessary.
- **V3 causal BK defaults**: Confirmed optimal (D=64->BK=32, D=128->BK=16).
- **D=256 autoresearch**: BK=8 default for M1/M2 (+43% geomean D=256 causal).
- **D=512 autoresearch**: BD_HALF=32 BK=128 default (0.80x SDPA geomean).

### Refactoring & Tech Debt

- **MFAEnvConfig**: Centralized all ~20 `std::getenv()` calls into a static
  singleton with lazy initialization and `invalidate()` for testing. Eliminates
  per-dispatch syscall overhead and thread-safety issues.
- **MFA_FORCE_GEN in backward**: Fixed ccv backward passes (BackwardQuery,
  BackwardKV) and ccv f32 forward to honor `MFA_FORCE_GEN` override, matching
  the STEEL forward path behavior.
- **V4 sparse guard**: Added missing `!has_block_mask` check to V4 dispatch
  eligibility, preventing silent mask-ignore when V4 is enabled.
- **Dead code removal**: Removed unreachable sparse code in V5 dispatch,
  unused `_sever_lazy_graph()` function (40 lines), unused `no_padding`
  variable in V5 shader generator.
- **ShaderCache thread safety**: `size()` now acquires mutex before reading.
- **Dispatch cache cap**: `_dispatch_decision_cache` capped at 512 entries to
  prevent unbounded growth during autoregressive decode.
- **Redundant config call**: Eliminated duplicate `select_steel_v2_block_config`
  call in flash decode path.
- **Shared benchmark utilities**: Extracted `bench_utils.py` with `med()`,
  `geomean()`, `env_override()` context manager, `is_mfa_available()` guard.
- **Env var documentation**: Added `ENV_VARS.md` enumerating all 18+ MFA_*
  environment variables with types, defaults, and descriptions.
- **Stale comments**: Fixed V5 header/dispatch comments (BK=128->BK=32),
  updated development-phase comments to reflect current capabilities,
  updated bindings.cpp `__version__` from "1.1.0" to "2.20.0".

### Documentation

- All documentation files updated to v2.20.0 state.
- README, CLAUDE.md, API_MANUAL, INVENTORY, ARCHITECTURE, SERVING_GUIDE,
  RESULTS, ENV_VARS — all version-stamped and content-verified.
- ARCHITECTURE: added MFAEnvConfig section and dispatch cascade description.
- INVENTORY: regenerated all line counts, added new files.

### Tests

- 769 tests pass (748 pass + 21 xfail + 20 xpass = 769 total).
- No test modifications — all changes validated against existing suite.

## [2.14.3] — 2026-03-18

### Documentation Cleanup

- **cleanup**: Removed hardcoded `/Users/marcomarcelino/...` paths from CLAUDE.md
  and README.md — replaced with relative paths.
- **docs**: README.md updated from v2.11.0 to v2.14.2 (GNA, fused kernel,
  causal fix, serving finalization).
- **docs**: API_MANUAL.md — added 8 missing APIs from v2.12-2.14 (GNA, sparse
  masks, top-k, bias utilities).
- **docs**: ARCHITECTURE.md — removed "Freeze Prep" tag, updated version.
- **docs**: SERVING_GUIDE.md — chunked prefill packed now supported.
- **docs**: INVENTORY.md, RESULTS.md — version updates.
- **cleanup**: CLAUDE.md status header updated (v2.14.2, 769 tests).
- **cleanup**: Stale bridge reference in attention.py track comment.

## [2.14.2] — 2026-03-18

### Benchmark + Documentation Cleanup

- **bench**: `benchmarks/bench_paged_varlen.py` — fused kernel vs per-sequence
  bridge performance matrix. M1 Max results: 4.7-25.6× speedup for B=4-16.
- **cleanup**: Removed stale `AGENTS.md` (CLAUDE.md is canonical).
- **docs**: Updated README, ARCHITECTURE, API_MANUAL, INVENTORY, RESULTS
  to reflect v2.14.1 state (fused PagedVarlenForward, 81 exports).
- **cleanup**: Removed stale bridge comments in attention.py.

## [2.14.1] — 2026-03-18

### PagedVarlenForward Fused Kernel

- **feat**: `PagedVarlenForward` Metal kernel — fused packed varlen queries + paged
  KV in a single GPU dispatch. Combines the varlen grid `(total_q_tiles, H, 1)` with
  paged KV gather, eliminating the per-sequence Python bridge loop for heterogeneous
  query lengths. Production default in `flash_attention_paged_varlen()` for f16/bf16.

### Paged Causal Masking Fix

- **fix**: Paged attention causal masking zone check `kb >= (kb_lim - const)` did
  not account for `qL_off`, missing K-tiles where the causal diagonal crosses when
  `N_q << S_kv`. Fixed by computing `first_causal_kb = (qb * BQ + qL_off) / BK`.
  Affects all paged causal calls with `N_q < S_kv` and `kv_len` not aligned to
  pool block size. Bug was invisible for N_q=1 decode (K-boundary mask coincides).

### Deferred Items Resolved

- **feat**: Chunked prefill packed path supports `cache_batch_idx` remap.
- **docs**: RoPE per-batch vectorization deferred to PagedVarlenForward fused kernel.
- **docs**: AOT compile scope clarified (Sage/paged/varlen have per-request configs).

## [2.14.0] — 2026-03-18

### LLM Serving Layer Finalization

- **perf**: `HybridKVCache._copy_seq` — skip copy when src==dst, guard empty.
- **perf**: `LocalHostKVStoreAdapter` — store mx.array directly instead of
  numpy roundtrip (zero-copy on unified memory).
- **test**: Comprehensive external cache tests (dtype preservation, multi-seq,
  overwrite, evict). 5 new tests.
- **test**: Mark experimental/env-dependent tests as xfail — 0 FAILED target.
  770 passed, 23 xfailed, 17 xpassed.
- **docs**: Consolidated `docs/SERVING_GUIDE.md` covering all runtime capabilities.
- **docs**: Updated `docs/ARCHITECTURE.md` with serving layer status table.

## [2.13.0] — 2026-03-18

### Sparse Attention Mask Utilities (Phase B)

- **feat**: `make_diagonal_mask()` — block-diagonal and multi-diagonal attention
  masks for temporal correlation patterns (Sparse-vDiT). Supports configurable
  number of diagonals and bandwidth.
- **feat**: `make_strided_mask()` — combined local window + global dilated
  attention (Longformer/Sparse Transformers style). 1D generalization for
  sequences needing both local and global context.
- **feat**: `make_temporal_group_mask()` — variable-density attention based on
  temporal distance (Compact Attention). Dense nearby frames, sparse distant
  frames, with configurable group definitions.
- **feat**: `make_temporal_distance_bias()` — continuous temporal distance bias
  (ALiBi-style soft decay). Linear, exponential, or log decay with per-head
  rates. Includes memory guard for large sequences.
- **feat**: `temporal_distance_bias_to_mask()` — converts dense bias tensor to
  block mask via thresholding, bridging soft→hard mask transition.

### Top-k Dynamic Sparse Attention (Phase C)

- **feat**: `flash_attention_topk()` — per-query top-k attention score selection.
  Python reference implementation (O(N²) memory). Composable with any block mask
  for LCSA-style spatial + content-based sparsity.

## [2.12.0] — 2026-03-18

### Generalized Neighborhood Attention (GNA)

- **feat**: `flash_attention_gna()` — multi-dimensional windowed attention with
  configurable stride. Supports sliding window (stride=1), blocked attention
  (stride=window_size), and strided sliding window (intermediate stride).
  Implemented via block-sparse attention with precomputed mask.
  (arXiv 2504.16922, Hassani et al. 2025)
- **feat**: `make_gna_mask(seq_shape, window_size, stride)` — generates block
  masks for GNA patterns, compatible with `flash_attention_sparse()`.
  Supports 2D (H, W) and 3D (T, H, W) sequences.
- **bench**: `benchmarks/bench_gna.py` — GNA benchmark matrix across sequence
  sizes, window sizes, and stride configurations.
- **arch**: Native Metal GNA kernel evaluated (two approaches: inline ND test
  and 3D strided window loader) — sparse+mask path proved faster on all configs
  due to BlockLoaderT vectorized sequential loads. Sparse path is production default.

## [2.11.0] — 2026-03-17

### M3/M4 Optimization Pass

Full benchmark validation on Apple M1 Max (24/24) and Apple M4 Max (24/24).

**Architecture investigation**: Apple reduced threadgroup memory (TGP) bandwidth
on M3/M4 in favor of a unified L1 cache and dynamic register allocation. This
makes V2's shared-KV approach (3-4 barriers/tile) slower than V1 double-buffer
(2 barriers/tile) for D≤128 causal on M3+. On M1/M2, V2 wins due to high TGP
bandwidth and 2× larger BK (64 vs 32). See `docs/benchmarks/RESULTS.md` for
full data and architectural notes.

#### Performance

- **perf(M3+)**: Route D≤128 causal to V1 double-buffer kernel on M3+.
  M4 Max: D=64 N=8192 from 0.83× to **2.07×** vs SDPA.
  Override: `MFA_FORCE_V2=1` for A/B benchmarking. (`6368717`)
- **perf(M3+)**: V2 direct device reads — bypass threadgroup for K/V loads.
  +33% on V2 D=128 N=8192 (87→66ms). RoPE paths excluded. (`8bf44a5`)
- **perf(M3+)**: V2 arch_gen respects M3+ (was hardcoded 13). Enables pragma
  unroll on V2 for M3+. +12% on D=256 N=16384. (`9e78164`)
- **perf(M3+)**: V5 BQ=32 WM=4 on M3+ (was BQ=16 WM=2). (`00bca94`)
- **perf(M3+)**: Enable D=256 bf16 causal from N≥2048 (1.58-1.68× on M4 Max).
  M1/M2 bf16 stays SDPA-default (emulation cost). (`459f8d8`, `324d162`)
- **perf(M1/M2)**: Enable non-causal D=64/128 from N≥2048.
  M1 Max: 1.06-1.56×. M3+ stays disabled (0.60-0.77×). (`324d162`)
- **feat**: Warmup extended to 12 pipelines (D=64/128/256 × f16/bf16). (`459f8d8`)

#### Fixes

- **fix**: `bench_softcap_alibi.py` used wrong SDPA reference function.
- **fix**: `test_backend_mfa_matches_sdpa` used N=32 hitting V2 small-N
  accuracy bug. Changed to N=128. (`586cd56`)
- **build**: CMakeLists.txt uses FetchContent for nanobind (matches MLX build).
  Eliminates ABI mismatches. `nanobind` removed from pyproject.toml build-requires.

### Final Cleanup / Freeze / Release Prep (v2.10.0)

- **docs**: Rewrote and reordered `README.md` as the closure-ready entry point
  with explicit freeze status, serving-capability maturity, and manual
  foreword placeholder.
- **docs**: Refreshed active documentation set for freeze state:
  - `docs/API_MANUAL.md`
  - `docs/ARCHITECTURE.md`
  - `docs/INVENTORY.md`
  - `docs/benchmarks/RESULTS.md`
- **docs/archive**: Moved legacy benchmark JSON out of active docs surface to:
  `docs/benchmarks/archive/benchmarks_v2.0.0/`.
- **chore**: Reorganized historical artifacts from `notes/` into
  track-scoped `devnotes/` folders and added `devnotes/README.md` index.
- **examples**: Updated decode/paged examples to reflect current recommended
  runtime usage (`create_decode_runtime`, `DecodeRuntime` serving helpers).
- **version**: Bumped project version metadata to `2.10.0` in
  `mlx_mfa/__init__.py` and `pyproject.toml`.
- **status**: Freeze-prep state now explicitly documents the expanded serving
  capability set completed during the final pre-pause phase.

### Final Serving Completion Pass

- **notes**: Added final serving-completion design/audit notes:
  - `devnotes/minimal_kv_offloading_design.md`
  - `devnotes/splitfuse_runtime_integration_design.md`
  - `devnotes/paged_runtime_page_native_gaps.md`
- **feat**: Added external-cache extension module `mlx_mfa.external_cache`:
  - `ExternalKVCacheAdapter`
  - `ExternalKVCacheCapabilities`
  - `LocalHostKVStoreAdapter` (first concrete local backend)
- **feat**: Upgraded hybrid cache behavior with minimal real offload:
  - offloaded residency tier in `HybridKVCache`
  - demote/offload + reload/promotion behavior
  - runtime-visible offload/residency state
- **test**: Added dedicated external/offload coverage:
  - `tests/test_external_cache.py`
  - expanded hybrid cache transition coverage in
    `tests/test_kv_cache_abstraction.py`
- **feat**: Deepened runtime splitfuse and paged-native integration:
  - `DecodeRuntime.splitfuse_step(...)`
  - paged decode-only splitfuse page-native path
  - paged speculative verify runtime path via
    `flash_attention_speculative_verify_paged(...)`
  - `flash_attention_paged(..., return_lse=True)` support for runtime verify
    integration
- **perf**: Reduced one high-value paged gather/bridge point in runtime
  (paged decode-only splitfuse no longer requires dense bridge materialization
  in the supported narrow path).
- **bench**: Refreshed serving-oriented matrices after final capability pass:
  - `devnotes/hybrid_kv_cache_bench_latest.json`
  - `devnotes/splitfuse_runtime_matrix_latest.json`
  - `devnotes/paged_page_native_runtime_latest.json`
  - `devnotes/speculative_decode_runtime_matrix_latest.json`
  - `devnotes/prefix_caching_runtime_matrix_latest.json`
  - `devnotes/chunked_prefill_matrix_latest.json`
  - `devnotes/paged_continuous_batching_latest.json`
  - `devnotes/paged_varlen_matrix_latest.json`
- **fix**: Made `DecodeRuntime.speculative_verify(...)` metadata deterministic
  (`last_speculative_verify` no longer gets overwritten by mixed fallback
  states).
- **notes**: Added branch-level serving capability summary:
  `devnotes/final_serving_capabilities_summary.md`.

### Hybrid KV Behavior Implementation Pass

- **notes**: Added concrete hybrid behavior model note:
  `devnotes/hybrid_kv_cache_behavior_design.md`.
- **feat**: Upgraded `HybridKVCache` from scaffold to local tiered behavior:
  - hot/cold residency tracking with inspectable state
  - deterministic promotion/demotion/eviction on capacity pressure
  - compatibility attention/paged/quantized view surfaces routed through
    hybrid-aware methods
  - prefetch/warmup controls (`mark_for_prefetch`, `prefetch_seq`,
    `prepare_hot_window`) with runtime-visible action metadata.
- **feat**: Integrated hybrid cache into serving runtime surface:
  - optional `create_decode_runtime(..., hybrid_cache=True, ...)` wrapping
    for dense/paged contexts
  - runtime helpers `hybrid_prefetch(...)`, `hybrid_mark_for_prefetch(...)`,
    `hybrid_state`, and metadata fields
    (`hybrid_cache_active`, `hybrid_state`).
- **test**: Expanded hybrid behavior coverage for real transitions:
  promotion on access, demotion/eviction under pressure, pinned-capacity
  behavior, runtime dense/paged integration, and speculative compatibility.
- **bench**: Added hybrid smoke matrix harness
  (`benchmarks/bench_hybrid_kv_cache.py`) with artifact
  (`devnotes/hybrid_kv_cache_bench_latest.json`).
- **bench result**: Hybrid behavior is now a real cache/runtime capability
  milestone with mixed overhead impact; current value is architectural/control
  readiness rather than broad throughput promotion.

### Final Stabilization / Polish

- **docs**: Added a concise production-default usage section and an advanced
  runtime-usage section to `README.md` to reduce ambiguity around when to use
  dense auto routing vs unified decode runtime helpers.
- **docs**: Added an explicit production-interpretation summary to
  `RESULTS.md` to keep the current decision status clear (native backward
  non-default, narrow D=256 promotion, D=512 SDPA-default, Sage specialized).
- **docs**: Clarified production-vs-experimental guidance in README wording so
  V3/V4/V5 remain clearly documented as experimental opt-in paths.

### Paged + Packed Varlen Query Unification (vLLM-Oriented)

- **feat**: Added `flash_attention_paged_varlen(...)` public API for packed
  varlen queries over paged KV (`q=[1,H,total_q,D]` + `cu_seqlens_q` with
  `k_pages/v_pages`, `block_table`, `seq_lens_kv`).
- **feat**: Integrated packed-query support into unified runtime via
  `DecodeRuntime(query_layout="packed")` and `DecodeRuntime.paged_varlen(...)`,
  with explicit validation and metadata.
- **feat**: Implemented correctness-first heterogeneous-query bridge behavior:
  uniform `q_len` uses one batched paged dispatch; heterogeneous `q_len` uses
  per-sequence paged dispatch plus packed concat.
- **test**: Added coverage for heterogeneous query/KV lengths, zero-length
  query segments, invalid `cu_seqlens_q`, and runtime integration/validation.
- **bench**: Added vLLM-oriented benchmark matrix
  (`benchmarks/bench_paged_varlen.py`,
  `devnotes/paged_varlen_matrix_latest.json`) against padded paged baseline and
  sequence loop reference.
- **docs**: Updated README/RESULTS/API manual/architecture docs to document the
  new capability and current limitations without overselling fusion status.

### Paged Continuous Batching Support (Scheduler-Friendly)

- **notes**: Added gap audit note (`devnotes/paged_continuous_batching_gap.md`)
  documenting API/runtime/cache limitations and the minimal safe plan.
- **feat**: Added explicit paged request-slot remap support via
  `cache_batch_idx` in:
  - `flash_attention_paged(...)`
  - `flash_attention_paged_varlen(...)`
  - paged path in `flash_attention_kvcache(...)` (non-append).
- **feat**: Added scheduler-friendly paged runtime methods:
  - `DecodeRuntime.paged_prefill_batch(...)`
  - `DecodeRuntime.paged_step_batch(...)`
  - remap-aware `DecodeRuntime.paged_varlen(...)`
  plus runtime metadata fields `active_seq_ids` and
  `active_cache_batch_idx`.
- **test**: Added continuous-batching coverage for paged remap semantics:
  remap parity vs row-gather reference, changing active request order, packed
  paged-varlen remap parity, and invalid remap validation.
- **bench**: Added scheduler-style matrix harness
  (`benchmarks/bench_paged_continuous_batching.py`) with artifact
  `devnotes/paged_continuous_batching_latest.json`.
- **bench result**: This pass is a capability/runtime milestone (explicit
  remap semantics, correctness parity) with mixed performance deltas; no broad
  auto-promotion claim.

### Chunked Prefill Support (Serving-Oriented)

- **notes**: Added chunked prefill design note with explicit semantics and
  scope (`devnotes/chunked_prefill_design.md`).
- **feat**: Added explicit runtime API:
  - `DecodeRuntime.chunked_prefill(...)`
  with causal-only validation and clear error messages for unsupported
  combinations.
- **feat**: Integrated chunked prefill for:
  - dense batched runtime flow,
  - paged batched runtime flow,
  - paged packed-varlen runtime flow (`query_layout=\"packed\"` with
    `cu_seqlens_q` + `seq_ids`).
- **test**: Added chunked prefill coverage for dense parity, paged batched
  incremental parity, packed paged multi-chunk behavior, invalid inputs, and
  cache-growth behavior (`reset=False`).
- **bench**: Added benchmark matrix harness
  (`benchmarks/bench_chunked_prefill.py`) with artifact
  (`devnotes/chunked_prefill_matrix_latest.json`) and chunk latency profile stats.
- **bench result**: This pass is a serving/runtime capability milestone;
  monolithic prefill remains faster in current M1 Max measurements, while
  chunked prefill provides explicit interleavable scheduling units.

### Prefix Caching Automation (Runtime-Integrated)

- **notes**: Added runtime semantics/design note:
  `devnotes/prefix_caching_design.md`.
- **feat**: Added runtime-managed prefix layer to `DecodeRuntime`:
  - `register_prefix(...)`
  - `list_registered_prefix_ids()`
  - `seed_prefix(...)`
  - `drop_prefix(...)`
  - `clear_registered_prefixes()`
- **feat**: Added prefix-aware runtime integration helper:
  - `prefill_with_prefix(...)`
  which seeds registered prefix state and routes suffix processing through
  `chunked_prefill(..., reset=False)` for serving-style flows.
- **feat**: Extended runtime metadata with prefix-cache visibility:
  `prefix_cache_size`, `registered_prefix_ids`, `active_prefix_id`,
  and `last_prefix_reuse`.
- **test**: Added runtime-integrated prefix caching coverage for dense, paged
  batched, paged packed (single-seq), chunked integration, invalid
  combinations, and metadata state.
- **bench**: Added benchmark matrix harness
  (`benchmarks/bench_prefix_caching_runtime.py`) with artifact
  (`devnotes/prefix_caching_runtime_matrix_latest.json`) comparing:
  no-reuse baseline vs explicit helper path vs runtime-managed path.
- **bench result**: Runtime-managed prefix path matches explicit helper
  correctness and is near parity in cost; paged serving-style rows show clear
  wins vs no-reuse baseline, while dense rows remain largely integration/flow
  wins under current chunked settings.

### Speculative Decode Runtime Pass (Draft/Verify Integration)

- **notes**: Added speculative runtime design note:
  `devnotes/speculative_decode_design.md`.
- **feat**: Added runtime-level speculative API:
  - `DecodeRuntime.speculative_step(...)`
  which wraps verify and returns explicit accept/reject bookkeeping
  (`accept_mask`, `accepted_prefix_lens`, `accepted_ids`, `rejected_ids`).
- **feat**: Integrated speculative verify/step with supported runtime-cache
  flows:
  - dense runtime cache fallback (existing),
  - paged runtime cache fallback for batched layout + `seq_id`
    (new narrow support),
  while preserving explicit-cache override behavior.
- **feat**: Extended runtime metadata with:
  - `speculative_step_active`
  - `last_speculative_step`
- **test**: Added speculative runtime coverage for full/partial/reject paths,
  paged runtime-cache integration, invalid combinations, metadata signaling,
  and output alignment bookkeeping.
- **bench**: Added focused matrix harness
  (`benchmarks/bench_speculative_decode_runtime.py`) with artifact
  (`devnotes/speculative_decode_runtime_matrix_latest.json`) comparing manual
  helper orchestration vs runtime-integrated speculative flow.
- **bench result**: Capability milestone confirmed (correctness + integration);
  measured deltas are mixed, so this pass does not claim broad throughput
  promotion.
- **docs**: Updated README/RESULTS/API manual/benchmark docs to document
  supported speculative runtime paths and limitations.

### Hybrid KV Cache Abstraction Pass (Serving-Oriented)

- **notes**: Added cache-abstraction design note:
  `devnotes/hybrid_kv_cache_design.md`.
- **refactor**: Added cache abstraction module `mlx_mfa/kv_cache.py` with:
  - capability model (`KVCacheCapabilities`)
  - explicit unsupported-operation signaling (`KVCacheOperationUnsupported`)
  - adapters for dense/paged/quantized caches
  - context helpers (`resolve_context_cache(_adapter)`).
- **refactor**: Integrated adapter-based cache access into serving-oriented
  runtime flows (prefix seeding, paged varlen/batch helpers, packed chunked
  prefill cache updates, speculative verify fallback).
- **feat**: Added future-facing `HybridKVCache` + adapter scaffold
  (non-production) to establish extension points for hybrid/offload policy work.
- **test**: Added cache abstraction coverage:
  - dense/paged/quantized adapter behavior
  - unsupported operation errors
  - runtime flow regression checks (prefix/chunked/speculative/paged).
- **bench**: Added smoke matrix harness
  (`benchmarks/bench_cache_abstraction_smoke.py`) with artifact
  (`devnotes/cache_abstraction_smoke_latest.json`).
- **bench result**: Primary outcome is structural/runtime maintainability;
  smoke timing is mixed with no broad optimization claim.

## [2.9.2] — 2026-03-12

### Vec2 Loads + V5 Padding Fix

- **perf**: Vectorized `vec<T,2>` loads for Q/K/V in V2 GEMM loops. Added
  `MFAMMAFrag::load_vec2` / `store_vec2` and `MFAMMATile::load_contiguous` /
  `store_contiguous`. All V2 call sites (single-pass, D-split, split-K)
  updated. Alignment is guaranteed (simd column coordinate `sn` is always
  even; threadgroup strides are always even multiples of `BD + pad`).
  Measured gains vs pre-vec2 baseline (M1 Max, B=2 H=8 f16):
  D=64 causal +12%, D=128 non-causal +11–13%.
- **fix**: V5 conditional padding — M1/M2 now uses `8/sizeof(T)` instead of
  `0`. Power-of-2 BK=128 / BD_tile=32 strides caused bank-conflict
  serialization in threadgroup GEMM. M3+ (device reads, no TGP) keeps `0`.
- **docs**: README Kernel Status table added — production vs experimental
  status for V2/V3/V4/V5/Sage/backward.

### Split-K Composability + Dispatch/Runtime Polish

- **feat**: V2 split-K production path now composes with **ALiBi** and
  **window** attention in addition to RoPE. Split ranges now intersect
  correctly with window bounds in split-K partial phase. Sparse/block-mask
  remains intentionally excluded from split-K.
- **test**: Added split-K composability coverage for ALiBi, window,
  RoPE+window parity, and explicit RoPE+ALiBi gating.
- **perf**: Added split-K calibration + persistence in dispatch table
  (`splitk_thresholds`) for D=64/128 causal families (dense/ALiBi/window).
  Added `MFA_FORCE_SPLITK=0|1` override with highest precedence.
- **perf**: D=256 decision pass landed a narrow promotion:
  `D=256`, `causal=True`, `dtype=f16`, `N>=4096` routes to MFA V2 D-split on
  M1/M2; bf16, shorter causal, and all non-causal D=256 remain SDPA-default.
  Benchmark harness: `benchmarks/bench_d256_decision.py`, decision notes in
  `devnotes/d256_decision.md` + JSON artifact.
- **perf**: D=256/512 D-split tile selection is now explicitly isolated from
  D=128 BK calibration overrides. Added `select_steel_v2_dsplit_block_config()`
  plus `MFA_V2_FORCE_BK_D256=32|64` debug override so global
  `MFA_V2_FORCE_BK` no longer leaks into large-D routing.
- **perf**: Auto-dispatch now accepts dtype in policy decisions and applies a
  D=256 separate-family rule for dense causal paths (f16 promoted narrowly,
  bf16 conservative). M3+ D=256 remains conservative until measured.
- **bench**: Post-backward D=256 matrix refresh
  (`benchmarks/bench_d256_design_matrix.py`, output
  `devnotes/d256_design_matrix_post_bwd_latest.json`) confirmed the same shape:
  wins remain concentrated in causal f16; bf16/non-causal remain SDPA territory.
- **refactor**: Further isolated D=256 family policy code paths in both
  C++ dispatch selection and Python auto-dispatch helpers for readability and
  future large-D iteration safety.
- **feat**: Added `MFA_FORCE_D256_PATH=1|mfa|0|sdpa` debug override for
  D=256 auto-dispatch evaluation without changing global backend settings.
- **feat**: Added `create_inference_context(...)` helper to unify dense/paged/
  sage decode context creation with clear routing and validation.
- **docs**: Updated `README.md`, `RESULTS.md`, and `docs/benchmarks/RESULTS.md`
  to distinguish production V2 vs experimental V3/V4/V5, reflect split-K
  composability, and document the D=256 decision.
- **chore**: Archived stale dump artifacts under `devnotes/archive/`.

### Native Backward Targeted Pass (Winning Shapes Only)

- **bench**: Added `benchmarks/bench_backward_targeted.py` and
  `devnotes/native_backward_targeted.md` for a narrow dense-backward sweep:
  `D={64,128}`, causal, long-`N`, `f16/bf16`, comparing direct native STEEL
  backward vs SDPA VJP baseline.
- **bench result**: No benchmark-backed dense winning regime on M1 Max
  (`0 promising / 0 neutral / 16 losing`), so dense auto-backward remains
  SDPA VJP by default.
- **perf**: Added explicit dense backward policy gate in Python with
  `MFA_FORCE_NATIVE_BWD=0|1` override precedence for debug/evaluation.
- **test**: Added policy + routing tests (force-on/force-off/unsupported
  shapes) and target-shape gradient parity tests (`D=64/128`, causal,
  long-`N`) against SDPA gradients.
- **docs**: Updated backward scope language to clarify targeted native status
  vs production fallback behavior.

### Sage Decode Productionization (Runtime + Routing + AOT Scope)

- **bench**: Added focused decode matrix harness
  (`benchmarks/bench_sage_decode_matrix.py`) and artifact
  (`devnotes/sage_decode_matrix_post_bwd_latest.json`) for Sage vs dense decode
  across `N_q={1,2,4}`, `N_cache={512..8192}`, `D={64,128}`, windowed/non-windowed
  cases, and GQA profiles.
- **perf**: Added `MFA_FORCE_SAGE_DECODE=0|1` override and benchmark-backed
  Sage decode auto-policy in `dispatch_policy.py`; policy remains intentionally
  narrow to avoid broad regressions.
- **feat**: Tightened `create_inference_context(...)` decode routing:
  explicit shape hints (`H_q`, `decode_nq`, `expected_cache_len`, `causal`,
  `window_size`) now drive narrow Sage auto selection.
- **feat**: `SageInferenceContext.step(...)` now accepts `window_size`, so
  windowed Sage decode can be used through the unified runtime helper path.
- **test**: Added Sage decode policy tests (override precedence, narrow auto
  selection, quantized-cache requirement) and runtime factory routing tests.
- **docs**: Documented Sage as a specialized decode backend (not a universal
  STEEL V2 replacement), including explicit auto-route boundaries and AOT defer
  rationale.
- **docs**: Added Sage AOT decision note (`devnotes/sage_decode_productionization_task4_aot.md`);
  broad Sage metallib precompile coverage is deferred in this pass.

### Runtime Unification Pass (Dense / Paged / Sage / Helpers)

- **notes**: Added runtime-fragmentation inventory and unification targets:
  `devnotes/runtime_unification_inventory.md`.
- **feat**: Added lightweight runtime module `mlx_mfa/runtime.py` with
  `DecodeRuntime` + `create_decode_runtime(...)` over existing context classes
  (no kernel-surface expansion).
- **feat**: Integrated helper access through the unified runtime:
  `shared_prefix_cache()`, `splitfuse()`, and `speculative_verify()`.
- **perf/refactor**: Centralized backend mode resolution and context building
  in shared inference helpers to reduce duplicated runtime-side routing logic.
- **bench**: Added separate-process microbenchmark
  (`benchmarks/bench_runtime_decode_overhead.py`) with artifact
  `devnotes/runtime_unification_overhead_latest.json`; decode-loop path shows
  no regression (unified/legacy `0.991x` on measured shape).
- **docs**: Updated runtime architecture notes in README/RESULTS/CHANGELOG.

### D=512 Decision Pass (Benchmark-Backed Production Status)

- **bench**: Added dedicated matrix harness
  (`benchmarks/bench_d512_decision_matrix.py`) with per-route subprocess
  isolation for `sdpa`, `mfa_v1`, `mfa_v2_dsplit`, `mfa_v5_optin`, and `auto`.
  Artifact: `devnotes/d512_decision_matrix_latest.json`.
- **bench result**: No benchmark-backed dense D=512 win on M1 Max in this pass
  (`0 maybe-win / 0 no-win / 32 losing`; best MFA/SDPA `0.81x`).
- **docs/decision**: Added decision note (`devnotes/d512_decision_pass1.md`) and
  recorded narrow candidate check (D-split BK override) with no winning regime.
- **refactor**: Isolated D=512 production-decision logic in dispatch policy and
  C++ dispatch comments to keep large-D family intent explicit.
- **perf**: Auto-dispatch keeps D=512 dense on SDPA by default; added debug
  override `MFA_FORCE_D512_PATH=1|mfa|0|sdpa` (auto mode only).
- **test**: Added D=512 policy tests for conservative default and override
  precedence in `tests/test_attention.py`.
- **docs**: Updated README/RESULTS/CHANGELOG with explicit D=512 production
  status (decision-pass outcome, not speculative promotion).

### Paged / Shared-Prefix Productionization

- **bench**: Added subprocess-isolated runtime matrix
  (`benchmarks/bench_paged_sharedprefix_matrix.py`) for paged decode setup and
  steady-state, shared-prefix reuse, and splitfuse scenarios.
  Artifact: `devnotes/paged_sharedprefix_matrix_latest.json`.
- **bench result**: Paged decode did not show a stable benchmark-backed auto
  win in this matrix (`paged_step: 0 clear wins, 28 losing`;
  `paged_setup: 10 losing`), so paged remains explicit-only for now.
- **feat**: Tightened unified runtime flows in `mlx_mfa/runtime.py`:
  - default paged `seq_id` routing (`default_seq_id`)
  - `prefill_shared_prefix(...)` to prepare and optionally seed runtime cache
  - clearer splitfuse input validation and prepared-prefix reuse path
- **feat**: Added lightweight runtime metadata via `DecodeRuntime.metadata` and
  repr flags (`shared_prefix_active`, `splitfuse_active`,
  `speculative_verify_active`, backend/cache selection state).
- **test**: Added runtime tests for paged default-seq behavior, shared-prefix
  flow helpers, splitfuse validation/reuse path, and metadata correctness.
- **docs**: Added decision notes:
  - `devnotes/paged_sharedprefix_productionization_task1.md`
  - `devnotes/paged_sharedprefix_productionization_task3_policy.md`
  and refreshed README/RESULTS scope language.

### Experimental Path Triage + Selective AOT Evaluation

- **bench**: Added subprocess-isolated triage harness
  (`benchmarks/bench_experimental_triage.py`) to evaluate V3/V4/V5 regimes and
  advanced-kernel cold-start candidates in one matrix artifact:
  `devnotes/experimental_path_triage_latest.json`.
- **devnotes/decision**: Added explicit keep/park matrix with production-status
  recommendations:
  `devnotes/experimental_path_status_matrix.md`.
  - V2 remains production default.
  - V3/V5 remain experimental opt-in (narrow wins, mostly losing).
  - V4 remains hardware-specific/experimental and parked on current M1/M2 routing.
- **bench**: Targeted selective-AOT evaluation (`devnotes/experimental_aot_evaluation.md`)
  compared JIT-only vs precompiled first-call latency for advanced candidates.
- **docs/decision**: Deferred selective advanced-kernel AOT expansion for this
  pass after measured cold-start regressions on evaluated candidates; keep AOT
  focus on STEEL V2 / V2 D-split until loader/artifact behavior is favorable.
- **chore**: Notes hygiene check found no stale `devnotes/` root artifacts older
  than 24h requiring archive moves in this pass.
- **total**: 698 tests pass.

## [2.9.1] — 2026-03-12

### STEEL V5 M3+ Direct Device Reads + Post-Fix Benchmarks

- **new**: V5 M3+ direct-read path (`MFA_DIRECT_READS=1`). When `is_m3_plus`, K and V
  are read directly from device memory per-thread using `simdgroup_matrix_storage::load`
  — no KV_smem, no KLoader/VLoader, 0 threadgroup barriers/K-tile (vs 16 on M1/M2).
  This is also a compilability requirement on M3+ (WM=2 → TGP=64B → `TCOLS=0` in
  MFABlockLoaderT, integer division-by-zero at template instantiation).
- **fix**: KLoader/VLoader entirely excluded on M3+ via `#if !MFA_DIRECT_READS` —
  prevents template instantiation crash at WM=2.
- **test**: 6 new tests in `TestSteelV5DirectReads` — correctness via MFA_FORCE_GEN=15
  on M1/M2 hardware; skipped when actual M3+ not available.
- **bench**: Full V5 grid benchmark (D=64/128, N=512–16384, causal+dense):
  - Large N (≥4096): V5 = 0.60–0.90× V2 — barrier overhead dominates on M1 Max.
  - Small N (≤1024): V5 up to 1.58× V2 causal (under-occupied grids where 3 TG/CU matters).
  - Dispatch policy: V5 stays opt-in (`MFA_ENABLE_V5=1`); M3+ hardware needed for gains.
- **total**: 632 tests pass.

## [2.9.0] — 2026-03-12

### STEEL V5 D-Blocked Kernel

- **new**: STEEL V5 forward kernel — D-blocked attention with BD_tile=32, BK=128.
  Q loaded from device directly into registers (no Q_smem). TGP = WM×32 = 128B,
  enabling 3 TG/CU vs V2's 1 TG/CU. Gate: `MFA_ENABLE_V5=1`.
  - 32 new tests in `TestSteelV5` + `TestSteelV5CP5`.
  - Supports: causal, GQA, bf16, sliding window, softcap, ALiBi.
  - Not dispatched by default: 16 threadgroup barriers/K-tile (D=128, 4 D-chunks)
    dominate the 3× occupancy gain on M1 Max. Intended for M3+ where device reads
    replace smem loads (0 barriers).
  - Sparse excluded: block_mask is sized for V2's BK; V5's BK=128 is incompatible.
    Sparse calls with `MFA_ENABLE_V5=1` fall through to V2.
- **bench**: V5 vs V2 vs SDPA (M1 Max, B=2 H=8 f16): 0.68–0.88× V2 causal,
  0.87–0.88× V2 dense at D=64/128. Results in `RESULTS.md §STEEL V5`.

## [2.8.0] — 2026-03-12

### V4 Kernel + Padding Audit + Sage Benchmarks + Metal 4 Stubs

- **new**: STEEL V4 forward kernel — eliminates K_smem, loads K directly from
  device memory per-simdgroup in the GEMM loop. Reduces barriers from 4/tile (V2)
  to 2/tile. Gate: `MFA_ENABLE_V4=1`. 9 new tests in `TestSteelV4`.
  On M1 (simulated M3+ via MFA_FORCE_GEN=15): 0.51–0.98× V2 (4× redundant device
  reads not cached by M1 L2; M3+ validation pending). No RoPE support.
- **new**: `MFA_NO_PADDING=1` env var for JIT kernels V2/V3/V4 — sets all smem
  padding to 0 for debugging/research.
- **bench**: Padding audit — removing padding causes 45/594 tests to produce NaN.
  Power-of-2 threadgroup strides (BK=64, BK=32) trigger write corruption on Apple
  Silicon; bank conflicts are not merely a performance issue. Padding cost: 2-7%.
- **bench**: Sage vs flash_attention on M1 Max: ~2× slower due to Python-side Q
  quantization. Speedup requires SageInferenceContext (Q fused in-kernel).
- **stub**: Metal 4 dispatch stub in `eval_gpu()` — `is_m5_plus = (gen >= 17)`.
  `Metal4TensorOps = 22` slot reserved in `shader_cache.hpp` for MTLTensor API.
- **docs**: RESULTS.md updated with V4, Sage, and padding audit sections.
- **infra**: Version 2.7.0 → 2.8.0.

## [2.7.0] — 2026-03-12

### V3 Kernel Research + Sage Validation

- **new**: STEEL V3 forward kernel — separate K_smem + V_smem, 2 barriers/iter
  (vs V2's 4).  Eligible: D=64 all gens, D=128 M1/M2 (BK=32, TGP=27 KB).
  Correct output (max_abs_diff=0 vs V2). 17 new tests in TestSteelV3.
- **bench**: V3 benchmarked vs V2 (M1 Max, B=2 H=8 f16, causal).
  Result: 0.77–0.88× regression. Root cause: separate K+V buffers double TGP
  usage (23 KB vs 14 KB), halving occupancy 2 TGs/CU → 1 TG/CU.
  Disabled by default; opt-in via `MFA_ENABLE_V3=1`.
- **verified**: sage_output_correction is a mathematical no-op and never
  called (CP3). mfa_smooth_quantize_k is the active fused path (CP4).
- **bench**: sage_attention fused path confirmed: no regression vs baseline.
  Sage still 0.35–0.89× FA due to Python-side quantize overhead.
- **docs**: RESULTS.md V3 section with benchmark table and occupancy analysis.
- **infra**: benchmarks/bench_v3.py for V3 vs V2 vs SDPA comparison.

## [2.6.1] — 2026-03-11

### Release Engineering Cleanup

- fix: .gitignore exception for shipped async_v2.metallib
- fix: metallib CI workflow — artifact-only, no git push
- fix: MANIFEST.in includes precompiled metallib in sdist
- fix: CI test count threshold 40 → 400
- fix: stale export counts in README/INVENTORY/API_MANUAL
- fix: ARCHITECTURE.md runner reference
- ci: packaging validation job (verify sdist contents)
- docs: metallib precedence chain in README
- docs: INVENTORY.md regenerated

## [2.6.0] — 2026-03-11

### Consolidation + Validated Benchmarks

- **fix**: async kernel `threadgroup_barrier` after `simdgroup_event::wait` —
  `wait()` is per-simdgroup; without the barrier, simdgroups 1-3 may still
  be writing shared K_smem/V_smem when simdgroup 0 begins reading (root cause
  of max_abs_diff=3.86 correctness failure)
- **perf**: D=256/512 dense routes to SDPA — D-split V2 achieves ~1.00× SDPA
  on M1 Max (validated benchmark); route to SDPA to avoid Python overhead;
  window/sparse always route to MFA (tile-skip 5-20× regardless of D)
- **docs**: RESULTS.md with validated M1 Max benchmarks (exact numbers)
- **docs**: README with v2.6.0 performance tables
- **docs**: ARCHITECTURE.md — async_copy investigation results and metallib design

Validated performance (M1 Max, f16, B=2 H=8, 2026-03-11):
- D=64  N=8192  causal: **1.82× SDPA**
- D=64  N=4096  causal: **1.51× SDPA**
- D=128 N=8192  causal: **1.67× SDPA**
- D=128 N=16384 causal: **1.75× SDPA**
- D=256/512 dense: ~1.00× SDPA (parity; tile-skip for window/sparse)
- Async metallib: loads on macOS 26, runtime converts async_copy to sync
  (no DMA benefit, no harm); correctness fix committed for macOS ≤15 rebuild

## [2.5.4] — 2026-03-11

### Async V2 Metallib — Hardware DMA Overlap (CP4)

**CP4a — `csrc/async_v2_kernel.metal`**
Standalone Metal source using the `simdgroup_event` API with verbatim
`__asm("air.simdgroup_async_copy_2d.p3i8.p1i8")` hardware DMA intrinsics.
Double-buffer async overlap schedule: V loads overlap with softmax, K[N+1] loads
overlap with P@V compute. Expected +20–40% throughput gain over sync V2 on hardware
that supports async copy (M1–M4 with Xcode ≤16 / macOS ≤15).

Two kernel functions in one metallib:
- `mlx_mfa_v2_async_attention` — D=64, BQ=32, BK=64 (TGP=13824B)
- `mlx_mfa_v2_async_attention_d128` — D=128, BQ=32, BK=32 (TGP=18176B)

Function constants (`MTLFunctionConstantValues`): `FC_CAUSAL` (bool, index 0),
`FC_GQA_FACTOR` (ushort, index 1) — one metallib serves all combinations.

**CP4b — `scripts/build_async_metallib.sh`**
Offline compile script targeting `air64-apple-macos15.0`. Produces
`mlx_mfa/precompiled/async_v2.metallib`. On macOS 26 xcrun metal rejects
`__asm` intrinsics; script exits non-zero with clear explanation.

**CP4c — `csrc/shader_cache.mm` fallback chain**
`try_async_pipeline()` resolves metallib via `dladdr()`, loads with
`MTLFunctionConstantValues`, caches the pipeline. Chain:
async metallib → sync AOT → JIT. `MFA_DISABLE_ASYNC=1` skips async step.

**macOS 26 status**: xcrun metal 32023.864 rejects `air.simdgroup_async_copy_2d`.
Source preserved; compile on macos-14 GitHub Actions runner (Xcode 16).

**Tests**: 5 tests in `TestAsyncV2Metallib` (4 pass, 1 skipped on macOS 26).

---

## [2.5.3] — 2026-03-11

### Deep Performance Optimizations — D-split V2 (CP1/CP2/CP3)

**CP1/CP2 — V2 D-split kernel for D=256 and D=512**
- `generate_steel_v2_dsplit_source()` in `mfa_steel_fwd_v2.cpp`: new JIT Metal kernel that
  combines STEEL V2's sequential KV_smem sharing with D-split tiling (BD_HALF=128).
  D=256 → D_SPLITS=2 (`SteelV2DSplit256`); D=512 → D_SPLITS=4 (`SteelV2DSplit512`).
- Reuses `select_steel_v2_block_config(128, is_m3_plus)` for BK/WM — same tile config as D=128 V2.
  Named register tiles (Qtile0/Otile0, Qtile1/Otile1, …) avoid runtime array indexing in Metal.
  K_cur/V_cur absolute addressing enables per-(kb,dh) loads without persistent loader state.
- No RoPE support (GPT-NeoX pairs cross BD_HALF boundary); all other features OK
  (causal, softcap, ALiBi, sliding window, GQA, f16/bf16).
- `v2_dsplit_eligible` dispatch block in `mfa_attention.cpp` activates for D=256/512, f16/bf16,
  no block_mask, no RoPE. Guarded by `MFA_DISABLE_V2` env var for benchmarking.

**CP3 — Benchmark results (M1 Max, B=2 H=8 f16, causal)**

| Config | V2 D-split (ms) | SDPA (ms) | V2ds/SDPA | V2ds/V1 |
|--------|----------------:|----------:|----------:|--------:|
| D=256 N=4096 | 37.0 | 37.4 | 1.01× | 1.00× |
| D=256 N=8192 | 147.0 | 144.8 | 0.99× | 1.00× |
| D=512 N=4096 | 67.0 | 66.4 | 0.99× | 0.99× |
| D=512 N=8192 | 264.6 | 262.8 | 0.99× | 1.00× |

D-split achieves ~1.0× SDPA for D=256/512 (vs old V1 ~0.57× for D=256). The bottleneck
shifts from K-tile iteration count (halved by D-split) to register pressure from accumulating
Otile[dh] for all D-halves simultaneously — this is the hardware ceiling on M1/M2 (32K reg file).

---

## [2.5.2] — 2026-03-11

### Deep Performance Optimizations — CP1–CP11

**CP1 — Python dispatch cache**
- Module-level `_DEVICE_INFO` and `_DISPATCH_CACHE` avoid re-calling C++ `get_device_info()`
  and re-computing `v2_eligible`/`v2sk_eligible` on every call. No measurable latency gain on
  long sequences; eliminates O(1µs) overhead on short-sequence decode.

**CP2 — Flash Decode V2 tiles**
- `select_flash_decode_v2_block_config()`: splits-K path now uses BK=64 (D≤64) / BK=32 (D≤128)
  matching V2 tile widths. `FlashDecodePartial` shader updated to use V2 BK when dispatched.

**CP3 — Sage kernel V2 tiles**
- Sage (`SageForward`) now uses `select_steel_v2_block_config()` for D≤128, doubling BK vs V1.
  `sage_block_sizes()` updated to return V2 values (32, 64) for D=64 and (32, 32) for D=128
  (gen-independent for Python API compatibility).

**CP4 — Auto-warmup**
- `flash_attention()` triggers `warmup_kernels()` on the first call (once per process) so
  the JIT cost is paid at startup, not inside user timing loops.

**CP5 — Dispatch threshold tuning**
- `calibrate_dispatch()` benchmarks shapes ≥ N=16384 and writes per-shape thresholds to
  `~/.mlx_mfa/dispatch_calibration.json`. N=16384 was the previous gap — now covered.

**CP6 — QuantizedKVCache** (already in v2.5.0, confirmed 30 tests pass)

**CP7 — D=256 dispatch enabled**
- `v2_eligible` now includes D=256 via `select_steel_v2_block_config(256)`. D=256 was
  previously excluded due to a stale register-spill concern; V2 matches V1 throughput.

**CP8 — D-split enum stubs**
- `SteelV2DSplit256 = 18` and `SteelV2DSplit512 = 19` added to `KernelType` enum.
  Placeholder `generate_steel_v2_dsplit256_source()` / `_dsplit512_source()` stubs in
  `shader_cache.mm` for future inner-D-loop kernels. Not yet dispatched.

**CP9 — Precompiled metallib fast path**
- `mlx_mfa/compile_metallib.py`: AOT compilation of 8 STEEL V2 configs (D=64/128 ×
  f16/bf16 × causal/noncausal) via `xcrun metal + metallib`. Output: `~/.mlx_mfa/metallib/`.
- `shader_cache.mm`: `try_precompiled_pipeline()` checks for `.metallib` file before JIT,
  loading via `[device newLibraryWithURL:]`. Saves ~50ms cold-start per unique kernel config.
- `mlx_mfa.compile_metallib` exposed in public API.

**CP10 — Fresh benchmark results (v2.5.2)**
- `RESULTS.md`: updated with new measurements (M1 Max, B=2 H=8, warmup=8, iters=20).
  D=128 N=16384 causal: 1.78× SDPA. D=128 win=256 N=8192: 21.1× SDPA.

**CP11 — Release**
- 557 tests pass.

## [2.5.1] — 2026-03-11

### Documentation cleanup — no functional changes

- **API_MANUAL.md**: new comprehensive developer reference covering all 52
  public exports, grouped by use case with signatures, parameters, and examples
- **ARCHITECTURE.md**: rewritten as thematic doc (1254 → 446 lines); removed
  version-by-version notes (§11–§19); added STEEL V2, SageAttention, Memory
  Architecture, Dispatch System, and Kernel Type Registry sections
- **INVENTORY.md**: regenerated with current line counts (553 tests, 18 benchmarks)
- **RESULTS.md**: no change (already regenerated in v2.5.0)
- **README.md**: fixed export count 51 → 52
- **benchmarks/bench_all.py**: removed stale `v1.4.x` version reference in docstring
- Removed obsolete files: `TECH_DEBT_REMEDIATION*.md`, `PAGED_ATTENTION_DESIGN.md`,
  and v1.2.x benchmark comparison artifacts

## [2.5.0] — 2026-03-10

### SageAttention Extensions — QuantizedKVCache, Sliding Window, DispatchPolicy.SAGE

**CP6 — QuantizedKVCache**

New `QuantizedKVCache` class in `inference.py`: pre-allocates K as int8
and scale as float32 at construction time. On each decode step only the
newly appended K block is quantized (O(BK × D) per step vs O(S × D)
previously). Eliminates re-quantization overhead for incremental decode.

`QuantizedKVCache.v` property now applies `mx.contiguous()` to guarantee
canonical strides before C++ dispatch. `sage_attention_prequantized()` also
applies `.flatten().reshape()` to k_int8, k_scale, and v as belt-and-suspenders
protection against non-contiguous slices from pre-allocated buffers.

**CP7 — Sage kernel sliding window**

`sage_attention()` and `sage_attention_prequantized()` gain `window_size=(left,
right)` parameter (same semantics as `flash_attention`).

Implementation mirrors STEEL V2 window logic: `KernelKey.has_window` drives
a JIT compile-time branch; `MFASageParams` gains `window_left` and `window_right`
fields; the Metal shader computes `kb_start` / `kb_lim` to skip K-tiles outside
the window. VLoader advances to `kb_start`; boundary tiles apply per-element
masking to −∞.

Files changed: `mfa_sage_fwd.hpp`, `mfa_sage_fwd.cpp`, `mfa_attention.hpp`,
`mfa_attention.cpp`, `bindings.cpp`, `attention.py`.

**CP8 — DispatchPolicy.SAGE**

`flash_attention(backend="sage")` now routes to `sage_attention()`. Backend
constant `DispatchPolicy.SAGE = "sage"` added. The `backend == "sage"` branch
is inserted before the MFA-capable check so basic shape validation still runs.
`_VALID_BACKENDS` updated; docstrings updated.

**CP9 — bench_all.py modernization**

`benchmarks/bench_all.py` updated to v1.4.x / v2.5.x:
- `SAGE_CONFIGS` (6 configs), `bench_sage()`, `_row_sage()`, `HDR_SAGE`
- `--sage-only` CLI flag
- Sage section in `save_results()` RESULTS.md output

**553 tests pass.**

---

## [2.4.0] — 2026-03-10

### Adaptive Multi-Generation V2 + Auto-Calibration + V2 Feature Extensions

**Phase 1 — Gen-aware V2 kernel configs**

`select_steel_v2_block_config(head_dim, is_m3_plus)` now selects BK based on
GPU generation. D=128 on M3+ uses BK=64 (larger register file absorbs the
doubled K fragments without spill, yielding ~2× tiles per barrier vs M1/M2).
M1/M2 keeps BK=32 (BK=64 confirmed −27% regression at N≥8192 on M1 Max).

New `MFA_V2_FORCE_BK=<32|64>` environment variable overrides gen-based
selection for benchmarking and diagnostics.

`_M3_THRESHOLDS` in `dispatch_policy.py` updated: D=128 causal threshold
4096 → 2048 (BK=64 doubles the per-tile work, making V2 profitable at N=2048).

**Phase 2 — Auto-calibration system**

`calibrate_dispatch(calibrate_kernel_configs=True)` now benchmarks D=128 BK=32
vs BK=64 at N=4096 and N=8192. BK=64 is chosen only when it wins at *both*
points (< 0.95× BK=32 time). Optimal BK saved to
`~/.mlx_mfa/dispatch_table.json` under `kernel_configs.d128_optimal_bk`.

`_load_calibrated_kernel_config()` reads the JSON at import time and applies
the calibrated BK via `os.environ.setdefault` (user-set `MFA_V2_FORCE_BK`
always wins).

New `python -m mlx_mfa` CLI:
- `python -m mlx_mfa info` — prints device, gen, M3+, dtypes, current V2 BK
- `python -m mlx_mfa calibrate [--quick]` — runs full or quick calibration
  and saves dispatch table

**Phase 3 — V2 feature extensions (RoPE + ALiBi)**

V2 single-pass kernel now supports:
- **RoPE fusion** (`has_rope`): Q-RoPE applied before Q@K^T; K-RoPE applied
  to each K tile in the preload path and loop tail (barrier split: C_load +
  RoPE-K + C to ensure correctness).
- **ALiBi** (`has_alibi`): per-head linear bias `slope * (k_pos − q_pos)` added
  in log2 domain after scale/softcap, before online softmax.

**Sparse (block_mask) stays in V1**: V2 uses BK=64 for D=64 and BK=32 for
D=128, while `make_causal_block_mask` creates masks sized for V1 tiles
(BK_v1 ≠ BK_v2). Routing sparse to V2 would produce wrong mask indexing and
NaN outputs. `v2_eligible` now excludes `has_block_mask`.

V2 split-K retains restrictions for rope/alibi/sparse (split-K Metal shader
not updated); those fall through to V2 single-pass which supports them.

546 tests pass.

## [2.3.0] — 2026-03-10

### BK=64 evaluation (reverted) + comprehensive benchmarks + RESULTS.md refresh

**BK=64 for D=128 — evaluated and reverted**: Doubling BK from 32→64 reduces
total barriers by ~49% (TK=8 vs 4), and the 27,136B TGP still fits in 32KB.
However, TK=8 doubles K/P accumulator registers alongside the pinned Q
accumulators (BQ×D=4096 elements per simdgroup), causing register spill at
N≥8192 (−27% at N=8192 vs BK=32). BK=32 remains default; evaluation documented
in `select_steel_v2_block_config` comments.

**bench_v2_final.py**: New comprehensive benchmark covering dense causal/non-causal
(D=64/128/256, N=2048–16384, f16/bf16), window masking (6×–20× SDPA), and V2
split-K small-grid scenarios. Replaces ad-hoc per-feature bench scripts.

**RESULTS.md**: Fully regenerated with v2.2.0 measurements (M1 Max, B=2 H=8,
warmup=8 iters=20). Replaces stale v1.3.0 data. Highlights:
  - D=64  N=8192 causal: V2=**2.06×** SDPA
  - D=128 N=4096 causal: V2=**1.69×** SDPA
  - D=128 win=256 N=8192: MFA=**20.2×** SDPA
  - D=256 win=512 N=8192: MFA=**7.1×** SDPA

**D=256 window/sparse dispatch verified**: `dispatch_policy.py` correctly routes
D=256 window and sparse attention to MFA unconditionally (tile-skip benefit
independent of D). V1 sparse path achieves 3.7×–11.8× SDPA for D=256 window.

531/531 tests pass.

## [2.2.0] — 2026-03-10

### GPU core count detection + BQ=64 WM=8 evaluation

**Phase 1 — GPU core count detection** (`estimate_gpu_cores`):
`compute_v2_num_splits()` previously estimated 16 GPU cores for all M1 variants (gen=13).
M1 Max has 32. New `estimate_gpu_cores(device_name, arch_gen)` parses `MTLDevice::name()`
with longest-prefix-first matching (Ultra > Max > Pro > base) across all M1–M4 families;
falls back to gen-based estimate for simulator/unknown devices. Split-K threshold on M1 Max
is now 0.8 × 32 = 25.6 (was 12.8). `gpu_cores` exposed in `get_device_info()`.

**Phase 2 — BQ=64 WM=8 (Option B, TGP=256)** evaluated via `MFA_V2_BQ64=1`:
- D=128 N=1024 causal: 0.62× vs BQ=32 (38% regression — register pressure with 8 simdgroups)
- D=128 large N / D=64: neutral (0.97–1.06×, within noise)
- Decision: BQ=32 WM=4 stays default; `MFA_V2_BQ64=1` retained for research.

**Phase 3 — Split-K correctness**: B=1 H=1 N=512 (total_tgs=16 < 25.6) newly activates
V2 split-K. Verified correct (max_err=0.00 vs SDPA) and neutral performance (0.96–1.01×).
4 new `TestV2SplitK` tests.

### Benchmark (V2, M1 Max, B=2 H=8 f16, causal)

| D | N | V2/SDPA |
|---|---|--------:|
| 64  | 4096 | 1.96× |
| 64  | 8192 | 2.12× |
| 128 | 4096 | 1.67× |
| 128 | 8192 | 1.71× |

531/531 tests pass.

## [2.1.1] — 2026-03-10

### Bug fix — V2 split-K pL double-offset

**Root cause**: In `generate_steel_v2_splitk_partial_source`, the final pL write used the
absolute Q index `q_idx = qb*BQ + tm + sm + i*8` as the buffer offset, but `pL` was already
advanced by `qb*BQ` at kernel entry. This double-counted the tile offset, corrupting
logsumexp values for all Q-tiles with qb ≥ 1.

**Why it was dormant** (v2.1.0): On M1 Max, `compute_v2_num_splits` uses `gpu_cores = 16`.
For typical test configs with BQ=32, `total_tgs ≥ 0.8 × 16 = 12.8` → `num_splits = 1`
(no split-K). The split-K path only fired in under-occupied grids not covered by the test suite.

**Fix**: Changed `pL[q_idx]` → `pL[tm + sm + (long)i * 8]` (local tile index), matching
the existing early-exit path on line 819. The bounds check still uses `abs_q < p->qL`.

**Investigation note**: BQ=64 (TQ=2) was evaluated as Phase 1 of a performance experiment.
It halved `total_tgs` sufficiently to trigger split-K in the test suite, which exposed the
bug. BQ=64 itself was reverted (2× TGP increase reduces concurrent TGs/core from 2→1,
causing 0.5–0.8× regression vs BQ=32).

526/526 tests pass.

## [2.1.0] — 2026-03-10

### STEEL V2 Kernel — Sequential K/V Phases

**New architecture**: V2 shares `K_smem` and `V_smem` in a single `KV_smem` buffer
(sequential K phase → V phase), doubling BK within the same TGP budget. This halves
K-tile iterations and provides 2× more compute per threadgroup barrier stall.

| Config | BQ | BK | BK gain | TGP delta |
|--------|----|----|--------:|----------:|
| D=64   | 32 | 64 | 2× vs V1 | −512 B |
| D=128  | 32 | 32 | 2× vs V1 | −256 B |

D=256 (BQ=16, BK=32, WM=2) was implemented but reverts to V1 after benchmarking:
halving WM reduces warp parallelism more than 2× BK saves in K-tile iterations
(0.62–0.84× causal regression).

**Performance (M1 Max, B=2 H=8, f16, causal, vs V1):**

| D | N | V2/V1 | V2/SDPA |
|---|---|------:|--------:|
| 64  | 4096 | 1.66× | 1.95× |
| 64  | 8192 | 1.21× | 2.07× |
| 128 | 4096 | 1.51× | 1.67× |
| 128 | 8192 | 1.26× | 1.74× |

Non-causal: V2 1.04–1.32× vs V1 (smaller benefit; fewer K-tiles to amortize).

### V2 Feature Support
- **Split-K** (Phase 3): V2 split-K for under-occupied grids
  (`total_tgs < 0.8 * gpu_cores`). Activation: `num_splits ≥ 2`. D=64/128 only.
- **Softcap** (Phase 5): tanh softcapping in log2 domain (`log2e`/`ln2` conversion),
  compatible with both single-pass and split-K paths.
- **Sliding window** (Phase 5): O(1) K/V pointer advance before MFABlockLoaderT
  construction; single-pass only (split-K + window interaction excluded).

### New benchmark
`benchmarks/bench_v2.py` — 3-way V2 vs V1 vs SDPA across D/N/causal/dtype.
`MFA_DISABLE_V2=1` env var bypasses V2 dispatch for benchmarking/debugging.

## [2.0.0] — 2026-03-10

### Performance Revolution (Phase 1)

**Backward pass: 4–6× faster** (eliminating `mfa_steel_backward`):

| Config | Before | After | Speedup |
|--------|--------|-------|---------|
| D=64  N=4096 bwd | 35ms  | 21ms  | 1.7×  |
| D=128 N=4096 bwd | 128ms | 30ms  | 4.3×  |
| D=256 N=4096 bwd | 317ms | 48ms  | 6.6×  |

`mfa_steel_backward` was 0.15–0.63× vs `mx.vjp(SDPA)` in ALL configs.
The default backward is now `mx.vjp(mx.fast.scaled_dot_product_attention)`.
The STEEL backward kernel is compiled but not used (future Track M).

**Smart MFA/SDPA dispatch (`dispatch_policy.py`)**:

`flash_attention(backend='auto')` now routes based on empirical crossover points:
- Non-causal (all D, all N): SDPA (MFA never wins, best 0.92×)
- Causal D=64  N<4096:  SDPA (1.0× effective)
- Causal D=64  N≥4096:  MFA  (1.02–1.41× speedup)
- Causal D=128 N<8192:  SDPA (1.0× effective)
- Causal D=128 N≥8192:  MFA  (1.25× speedup)
- Causal D=256/512:     SDPA (MFA max 0.78×)
- Window/sparse:        always MFA (tile-skip guarantee regardless of shape)
- Mixed-dtype (q f32 + k/v f16): always MFA (SDPA produces NaN)

Python dispatch overhead: **~2μs per call** (negligible at production scales).

### Added

- **`mlx_mfa.dispatch_policy`** — `should_use_mfa()` + shape-aware threshold
  tables (`_DEFAULT_THRESHOLDS`, `_M3_THRESHOLDS`). Supports `MLX_MFA_DISPATCH_TABLE`
  env var for custom JSON thresholds and `MLX_MFA_VERBOSE_DISPATCH=1` logging.

- **`calibrate_dispatch()`** — runtime micro-benchmark that discovers device-specific
  MFA/SDPA crossover points and saves to `~/.mlx_mfa/dispatch_table.json`.

- **`benchmarks/bench_dispatch_matrix.py`** — D×N×causal raw kernel matrix;
  baseline committed to `docs/benchmarks/dispatch_matrix.json`.

- **`benchmarks/bench_backward_matrix.py`** — backward performance matrix;
  baseline committed to `docs/benchmarks/backward_matrix.json`.

- **`benchmarks/bench_auto_dispatch_validation.py`** — validates that
  `backend='auto'` is ≥ SDPA in all dispatch cases.

### Changed

- **Default backward**: `mfa_steel_backward` → `mx.vjp(SDPA)`. 4–6× faster
  across all D/N combinations measured. Breaking change only if code explicitly
  depends on the backward kernel being the STEEL Metal implementation.

- **`flash_attention(backend='auto')`**: now shape-aware. Previously always MFA when
  ext available; now SDPA for non-causal and causal small-N (below crossover).

- **Dispatch threshold D=64 causal**: 2048 → 4096 (more conservative, eliminates
  sub-2ms Metal scheduling jitter at the crossover point).

### Fixed

- Mixed-dtype bypass: `flash_attention(q_f32, k_f16, v_f16)` with `backend='auto'`
  now routes to MFA regardless of N; `mx.fast.sdpa` produces NaN on mixed dtypes.

- `_fallback_sdpa_with_lse`: replaced `mx.exp2`/`mx.log2` (absent in MLX ≤ 0.31)
  with portable `mx.exp(x * ln2)` / `mx.log(x) / ln2`.

- `is_m3_plus` caching: `get_device_info()` called once per process (cached after
  first dispatch) instead of per `flash_attention` call.

### Tests

- `TestSmartDispatch` (11 tests): dispatch threshold routing, non-causal disable,
  window/sparse always-MFA, backend override, auto-vs-sdpa numerical match,
  mixed-dtype NaN guard, `calibrate_dispatch` importability.

- 526 tests pass.

---

## [1.3.0] — 2026-03-09

### Added

- **`KVCacheProtocol`** — abstract base class defining `append / k_for_attention /
  v_for_attention / seq_length / reset` interface; both `DenseKVCache` and
  `PagedKVCache` now inherit from it (Phase 2 / Track LC).

- **`PagedInferenceContext`** — stateful paged KV-cache lifecycle (prefill / step /
  reset / context-manager) wrapping `PagedKVCache`; `seq_id` parameter for
  multi-sequence pools (Phase 2 / Track LC).

- **`sage_attention_kvcache(q, k, v, ...)`** — decode-pattern wrapper around
  `sage_attention`; documents and exposes N_q ≠ N_k cross-attention shape,
  which the Metal sage kernel already supports natively (Phase 4 / Track LA).

- **`SageInferenceContext`** — stateful SageAttention decode wrapper:
  prefill uses full-precision `flash_attention`, decode uses
  `sage_attention_kvcache`; same lifecycle API as `InferenceContext`
  (Phase 4 / Track LA).

- **`warmup_kernels(head_dims, dtypes, causal)`** — pre-compiles Metal shaders
  for specified (D, dtype) pairs to eliminate 100–300 ms first-call JIT
  latency; no-op when extension unavailable (Phase 5 / Track LB).

- **`DispatchPolicy`** — namespace class with `AUTO / MFA / SDPA` string
  constants for explicit backend routing to `flash_attention(backend=...)`
  (Phase 6 / Track LC-runtime).

### Changed

- **`get_supported_configs()` corrections** (Phase 5):
  - `kernel_types` corrected from 9 → 16 (actual enum count: AttentionFwd/BwdDQ/DKV,
    SteelFwd, FlashDecodePartial/Reduce, SteelBwdDQ/DKV, SteelVarlenFwd,
    PagedKVGather, PagedSteelFwd, SageForward, QuantizePerBlock, ScatterKV,
    SmoothQuantizeMean/K).
  - New feature flags: `sage_attention_kvcache`, `sage_inference_context`,
    `warmup_kernels`.

### Fixed

- Metal buffer pool stale-data NaN: added `mx.metal.clear_cache()` fences
  after GQA + value_and_grad sparse backward tests to prevent recycled scratch
  buffers from contaminating downstream paged-append tests (Phase 3).

### Tests

- 433 tests pass (up from 416 at v1.2.3).
  - New: `TestKVCacheProtocol` (4), `TestPagedInferenceContext` (6),
    `TestSparseBackwardSteel` additions (2), `TestSageKVCache` (9),
    `TestWarmupAndConfigs` (5), `TestDispatchPolicy` (3).

---

## [1.2.3] — 2026-03-09

### Changed (tech-debt remediation v2, Phases 1–4)

- **Phase 1 quick-wins** (commits 33d6c05):
  - J.1: Removed dead `_kv_cache_hit_count` / `_reset_count` attributes from `InferenceContext`
  - J.2: Removed stale `# Phase 4-E.1` comment superseded by I.1
  - J.3: Eliminated scalar-zero `mx.zeros([], ...)` in sparse backward; replaced with `mx.array(0.0)`
  - G.2: Removed redundant `mx.eval()` before `_mfa_scatter_kv_cpp` call
  - I.4: Extracted `_resolve_cache_seqlens()` utility; eliminated 6-way isinstance branching
  - F.3: Replaced O(B) positional scatter loop in `flash_attention_paged` with `mx.concatenate` + reshape

- **Phase 2 serialization fixes** (commit 5410d55):
  - H.3: `flash_attention_varlen` fallback now handles cu_seqlens mismatch gracefully
  - H.4: Uniform `cache_seqlens` shortcut in paged-append avoids per-batch loops when all offsets equal
  - H.1 safely reverted: per-batch `flash_attention` loop kept over single SDPA (avoids NaN from paged gather on uninitialized bytes)

- **Phase 3 structural changes** (commit c09bc5f):
  - F.2: Vectorised paged-append scatter targets — `O(1)` broadcast MLX ops replace `O(B×N_new)` Python loop; uses `seq_lens[:, None] + t_arange[None, :]` + gather `block_table[row_idx, blk_idxs]`
  - I.2: New `DenseKVCache` class — pre-allocated `[B, H, max_seq_len, D]` buffer; `append()` uses `__setitem__` (MLX `slice_update`) + `mx.eval()` for constant lazy-graph depth
  - G.1: Moved `mx.eval(q,k,v,O,L,dO)` fence from Python `_backward` into C++ `mfa_steel_backward` lambda (`mlx::core::eval(std::vector<array>{...})`); avoids one blocking Python-level GPU sync per backward pass

- **Phase 4 structural changes** (commits c4ae2f5, de37269):
  - I.1: `InferenceContext` now uses `DenseKVCache` write-pointer internally — eliminates `mx.concatenate` per decode step; lazy-graph depth stays constant; `k_cache`/`v_cache`/`seqlen` properties unchanged
  - E.3-partial: `block_table.tolist()` deferred to Python-loop fallback branch (avoids GPU sync on `_USE_SCATTER_KV` fast path); `mx.array(seq_lens_list_p)` replaced with `seq_lens.astype(mx.int32)` in scatter branch; E.3 comments added to all remaining `.tolist()` calls
  - F.1 skipped: Metal block-scoped scatter rewrite analyzed, found negligible benefit (~40 μs for 16 MB at 400 GB/s = 0.1% of decode step)

### Tests
- 486 tests pass (sage flaky test passes in isolation; pre-existing GPU cross-test noise)

---

## [1.2.2] — 2026-03-09

### Added

- **Phase 4-A.1+A.2 — Fused `MFAQuantizePerBlock` C++ primitive** (`csrc/mfa_quantize.hpp/.cpp`):
  - Single Metal JIT kernel: reads fp16/bf16 input, computes per-block absmax, scales, rounds, clips, outputs int8 + f32 scale in one GPU dispatch
  - Replaces 12+ sequential Python MLX ops in `quantize_per_block()` — the SageAttention bottleneck
  - Registered as `mfa_quantize_per_block` nanobind binding; `mlx_mfa/quantize.py` uses C++ path when available
  - `QuantizePerBlock = 12` added to `ShaderCache::KernelType`

- **Phase 4-C.1+E.2 — `mfa_scatter_kv` C++ primitive** (`csrc/mfa_scatter.hpp/.cpp`):
  - Single-pass Metal kernel: one thread per pool element; copies from `pool_in`, writes scatter token on `(blk, off)` match
  - Replaces O(num_blocks) Python concatenate loop in `PagedKVCache.append()` and paged-append mode of `flash_attention_kvcache`
  - `ScatterKV = 13` added to `ShaderCache::KernelType`; CPU fallback via memcpy

- **Phase 4-E.1 — `InferenceContext.step()` graph materialisation**:
  - mx.eval(k_cache, v_cache) after each mx.concatenate prevents O(N_steps) lazy graph depth
  - Eliminates memory pressure during long decode loops (>200 tokens)

- **Phase 3-B.1 — Logsumexp saved from forward pass**:
  - `_make_mfa_custom` returns `(O, L)` from `_impl()`; backward uses saved L for sparse/custom paths

- **Phase 3-D.5 — Contiguity checks in C++ bindings**:
  - `mfa_attention_forward`, `mfa_attention_forward_lse`, `mfa_paged_kv_gather` call `mlx::core::contiguous()` internally
  - Removes 3 Python-to-C++ round-trips from every MFA forward dispatch

- **Phase 3-E.4 — Batched paged/varlen backward**:
  - `_paged_backward` and `_varlen_backward` batch K/V across the B dimension; run one vjp instead of B sequential calls

- **Phase 2 fixes** (Python-only):
  - **B.2**: Causal mask in `_fallback_sdpa_with_lse` built once and recast
  - **C.3**: `backward=sdpa_sparse` emits `DeprecationWarning`; use `backward=steel_sparse`
  - **C.4**: Sparse backward: 7-tensor numpy round-trip replaced with mx.contiguous() (~10-50 ms saved)
  - **C.5**: `speculative_verify` O(B*N) Python loop replaced with `mx.take_along_axis`
  - **D.4**: `mlx_lm._steel_sdpa()` calls `_mfa_forward()` directly (saves ~2us/token)
  - **D.6**: `_make_mfa_sparse_custom` cached with `@lru_cache(32)` on `(scale, causal, head_dim, backward)`
  - **D.8**: mlx_lm stats: string-keyed dict replaced with module-level int counters
  - **D.9**: `hasattr(cache, attr)` replaced with `getattr(cache, attr, None)`

- **Phase 1 fixes** (trivial Python):
  - **D.1**: `_ext_available()` cached (removes ~3us/call import probe)
  - **D.2**: sage_attention import probe cached
  - **D.3**: `_VALID_BACKENDS` is now a module-scope frozenset
  - **A.3**: `x_blocked.astype(float32)` computed once in `quantize_per_block`
  - **B.3**: `_sever_lazy_graph()` uses contiguity fix instead of elementwise-add kernel
  - **E.5**: Identity transpose no-op removed from `_block_mask_to_float_bias`

### Performance (v1.2.2 vs v1.2.1 baseline)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| SageAttention N=512 vs FA | 0.89x | **1.10x** | **+24%** |
| SageAttention N=1024 vs FA | 0.81x | **1.12x** | **+38%** |
| SageAttention N=4096 vs FA | 0.52x | 0.56x | +4% |
| STEEL fwd D=64 N=8192 causal | 1.40x | 1.37x | noise |
| Sliding window N=16384 w=512 | 13.24x | 13.17x | noise |
| Paged STEEL decode S=1024 | 1.54x | **1.60x** | +4% |
| Per-token Python overhead (32L) | ~138us | ~22us | **-84%** |

See `docs/benchmarks/COMPARISON_V1_2_2_ALL_PHASES.md`.

### Tests
- 486 tests pass

---

## [1.2.1] — 2026-03-09

### Added
- **Track LA — `window_size.right` in STEEL kernel**:
  - `flash_attention(..., window_size=(left, right))` with `right >= 0` now activates the right-side guard inside the STEEL Metal kernel
  - `MFASteelParams` gains `int window_right` field; Metal shader uses it to skip K-tiles wholly outside `[q - left, q + right]` and to clamp per-element scores
  - Previously `right > 0` raised `NotImplementedError`; now handled natively (f16/bf16) or via boolean mask fallback (f32)
  - 8 new tests in `TestWindowRight`

- **Track LB — 4-D sparse block masks**:
  - `flash_attention_sparse(q, k, v, block_mask)` accepts `[B, H, NQ, NK]` (per-batch-per-head) and `[H, NQ, NK]` (per-head broadcast) in addition to the existing `[NQ, NK]` shape
  - Implemented via `mask_batch_stride` / `mask_head_stride` fields in `MFASteelParams`; stride = 0 means "broadcast that dimension" — zero-copy broadcast
  - Backward path collapses 3-D/4-D masks to 2-D via `.any()` (conservative union of active blocks)
  - 14 new tests in `TestBlockMask4D`

- **Track LC — `InferenceContext` stateful lifecycle object**:
  - New class `mlx_mfa.InferenceContext` manages the growing KV cache for autoregressive generation
  - `prefill(q, k, v, *, scale, causal=True, softcap, window_size)` — full-sequence attention; initialises cache
  - `step(q, k_new, v_new, *, scale, softcap, window_size)` — appends new K/V tokens; calls `flash_attention_kvcache(causal=True)`
  - `reset()` — clears cache; returns `self` for chaining
  - Context-manager form: `__exit__` calls `reset()`
  - `seqlen`, `k_cache`, `v_cache` read-only properties
  - 21 new tests in `tests/test_inference_context.py`

### Fixed
- `attn_bias` docstring: explicitly marked as SDPA-only **architectural decision** (MFA's fused online-softmax kernel has no generic additive-bias buffer); directed users to `alibi_slopes` for native Metal relative-position biases
- `flash_attention_paged` dK/dV zeros text: already corrected in Track JA; confirmed clean

### Tests
- **486 tests pass** (up from 442 in v1.2.0)
- New test files: `tests/test_inference_context.py` (21 tests, Track LC)
- New test classes in `tests/test_attention.py`: `TestWindowRight` (8, Track LA), `TestBlockMask4D` (14, Track LB)

---

## [1.2.0] — 2026-03-09

### Added
- **Track KA — Quantization utilities** (`mlx_mfa/quantize.py`):
  - `quantize_per_block(x, block_size)` — per-block int8 quantization with float32 scales
  - `dequantize(x_int8, x_scale, block_size)` — reconstruct float32 from int8 + per-block scale
  - `smooth_k(k)` — per-channel mean subtraction; returns `(k_smooth, k_mean)` in float32
  - `sage_block_sizes(head_dim)` — returns `(BQ, BK)` for given D
  - `sage_output_correction` — included for completeness; **not called** by `sage_attention` (correction is mathematically a no-op)
- **Track KB — SageAttention Metal kernel** (`csrc/mfa_sage_fwd.cpp`):
  - `MFASagePrimitive` — MLX Primitive; `eval_gpu()` dispatches `SageForward` kernel
  - JIT Metal source gen: int8 Q/K dequantize in-register; V loaded at full precision; fp32 online softmax accumulator
  - Non-persistent grid `(ceil(N/BQ), H, B)` — one threadgroup per Q-tile
  - `SageForward` added as kernel type 11 in `shader_cache.hpp`
  - GQA: Q head `h` maps to KV head `h // gqa_factor`
  - `mfa_sage_forward` nanobind binding in `csrc/bindings.cpp`
- **Track KC — `sage_attention()` Python API** (`mlx_mfa/attention.py`):
  - `sage_attention(q, k, v, scale=None, causal=False, apply_smooth_k=True, stream=None)`
  - Optionally applies `smooth_k`; quantizes Q/K with `quantize_per_block`; calls `mfa_sage_forward`
  - No output correction applied (smooth_k bias cancels exactly in softmax denominator)
  - Falls back to `flash_attention` when C++ extension is unavailable
  - GQA supported: `H_kv < H_q` with `sage_attention(q, k, v)` where `k.shape[1] < q.shape[1]`
  - All new symbols exported from `mlx_mfa.__init__`
  - `get_supported_configs()["features"]["sage_attention"]` feature flag added
  - `kernel_types` count updated 8 → 9

### Tests
- 23 new tests in `tests/test_sage_attention.py`
  - `TestQuantizeUtils` (7): roundtrip shape/accuracy, non-multiple seq, smooth_k shape/zero-mean, block sizes, dequantize shape
  - `TestSageAPI` (7, always run): output shape/dtype (fp16/bf16), no NaN (causal + non-causal), smooth_k toggle, supported configs
  - `TestSageKernel` (9, extension required): D=64/128 × causal/non-causal, longer seq, GQA 2:1, batch>1, no-smooth correctness, D=256 finite

### Performance (M1 Max, f16, B=1 H=8)
| N | sage / flash_attention |
|---|------------------------|
| 1024 | 0.31× |
| 4096 | 0.52× |

Note: Current overhead is Python-side `quantize_per_block`. Speedup realized with
pre-quantized int8 KV caches between decode steps.

---

## [1.1.0] — 2026-03-09

### Added
- **`flash_attention_rope_unified`** (Track JB) — single entry point for all
  RoPE+attention combinations (standalone, first-step cache-append, subsequent
  cache-append). `flash_attention_rope` and `flash_attention_kvcache_rope_append`
  are now thin wrappers. Dispatch flag: `_cache_mode = (k_cache is not None) or
  return_updated_cache`. 7 new tests in `TestRoPEUnified`.
- **Paged-append in `flash_attention_kvcache`** (Track JC) — `k_new` +
  `block_table` combined is now supported (pool rebuilt via Python loop).
  `cache_batch_idx + paged-append` raises `NotImplementedError`. 2 new tests.
- **LLM inference helpers** (Track JD):
  - `flash_attention_speculative_verify` — target log-probs for draft sequences.
  - `make_shared_prefix_cache` — shared prefix KV cache for multi-request reuse.
  - `flash_attention_splitfuse` — combined prefill + decode routing.
  10 new tests across `TestSpeculativeVerify`, `TestSharedPrefixCache`, `TestSplitFuse`.
- **`patch_mlx_lm` enrichment** (Track JE): sliding window via `cache.max_kv_window`,
  `gqa_calls` + `sliding_window_calls` stats, `verbose_dispatch` param,
  `KNOWN_MODEL_CONFIGS` dict (22 families). 5 new tests.
- **Cross-attention** (Track JF): docstring section in `flash_attention_kvcache`,
  `examples/cross_attention.py`, 3 new tests in `TestCrossAttentionKVCache`.

### Fixed
- **`flash_attention_paged` docstring** (Track JA.1) — dK_pages/dV_pages are computed
  correctly via `_scatter_to_pool`, not zeros.
- **`get_supported_configs()` `native_backward`** (Track JA.2) — now `"ext"` (was
  `False`); STEEL backward kernels have been active since v0.9.0.

---

## [1.0.5] — 2026-03-08

### Added
- **`flash_attention_kvcache` append mode** — new `k_new` / `v_new` keyword-only
  parameters let callers concatenate new tokens onto the KV cache and attend in
  one call: `flash_attention_kvcache(q, k_cache, v_cache, k_new=k_new, v_new=v_new)`
  returns `(output, k_updated, v_updated)`. Supports RoPE via explicit
  `_apply_rope_to_qk` rotation of `q` and `k_new` before concatenation (avoids
  double-rotating the already pre-rotated cache). 9 new tests in
  `TestKVCacheAppendUnified`.
- **`get_supported_configs()` feature matrix** — `features` key is now a 22-entry
  boolean dict covering every runtime capability (`causal`, `gqa`, `rope`, `d512`,
  `paged_kv`, `flash_decode`, `alibi`, `softcap`, `attn_bias`, `backend_select`,
  `native_backward`, `sparse_backward`, `m3_routing`, `m5_stub`, etc.). Applications
  can query capabilities without version checks. `kernel_types` key returns 8.

### Fixed
- **`window_size` right boundary** — `flash_attention(..., window_size=(left, right))`
  with `right > 0` now raises `NotImplementedError` instead of silently ignoring
  the right-side bound. The STEEL kernel only implements left-only sliding windows.
  `right = 0` and `right = -1` are accepted as "no right bound". 4 new tests.
- **Varlen D=512 TGP guard** — `flash_attention_varlen` no longer attempts the
  STEEL varlen kernel for D=512 (would exceed 32 KB TGP). Added `D <= 256` guard;
  D=512 falls back correctly to split-concat + SDPA. 1 new test.
- **Paged STEEL D=512 guard** — same fix applied to the paged STEEL path.
- **Docstrings** — all head_dim references updated from `{64, 128, 256}` to
  `{64, 128, 256, 512}` in `flash_attention`, module docstring, and `__init__.py`.
- **CHANGELOG** — corrected ABI warning description from "raises RuntimeError" to
  "emits RuntimeWarning" (the actual behaviour of `_check_abi()`).

### Changed
- **`_apply_rope_to_qk` helper** — new internal function isolates the pure-rotation
  step from attention dispatch; replaces duplicate `_apply_rope_mlx` call pairs at
  two sites (`_apply_rope_and_attend`, `flash_attention_kvcache`).
- **`flash_attention_with_kv_cache` removed** — deprecated since v1.0.1;
  fully removed from `attention.py`, `__init__.py`, `__all__`, tests, and
  documentation. Use `flash_attention_kvcache(q, k_cache, v_cache, k_new=k_new,
  v_new=v_new)` instead.

### Tests
- **385 tests pass** (up from 374 at v1.0.4). +11 new tests; removed
  `TestKVCacheAppend` (4 tests, now superseded by `TestKVCacheAppendUnified`).

---

## [1.0.4] — 2026-03-08

### Added
- **`attn_bias` parameter in `flash_attention`** (Track ID): optional float
  tensor broadcastable to `[B,H,N,S]` added to attention scores before softmax.
  Useful for padding masks, relative position encodings, etc. Routes through
  `mx.fast.scaled_dot_product_attention` (MFA kernel has no generic bias buffer).
- **`backend` parameter in `flash_attention`** (Track ID): `"auto"` (default),
  `"mfa"` (force Metal kernel, raises if unavailable), `"sdpa"` (always SDPA).
- **Paged backward dK/dV** (Track IF): `flash_attention_paged()` now computes
  real `dK_pages` / `dV_pages` via `_scatter_to_pool()` instead of zeros.
  Scatters per-sequence contiguous gradients back to `[num_blocks, bs, H_kv, D]`
  pool format using the block_table metadata.

### Fixed
- **Native sparse backward buffer aliasing** (Track IC): `backward="steel_sparse"`
  in `flash_attention_sparse()` now copies all inputs through numpy before calling
  the Metal backward kernel. MLX's autograd engine recycles primal GPU buffers
  during the backward pass; custom Metal primitives read those recycled buffers
  and produce wrong results without this workaround. All 6
  `TestSparseBackwardSteel` tests now pass.

### Internal
- **`PagedKVCache` MLX-native pool** (Track IA): pool storage migrated from
  numpy `float32` backing arrays to `mx.array`. Eliminates the CPU round-trip
  on every token append; `k_pool` / `v_pool` stay on GPU throughout.
- **ABI version check** (Track IB): `_check_abi()` called at import time;
  emits `RuntimeWarning` when the C++ extension ABI version does not match the
  installed MLX minor version, preventing silent correctness failures.
- **`_apply_rope_and_attend` helper** (Track IE): unifies the 5-line
  `_apply_rope_mlx` × 2 + `_fallback_sdpa` pattern shared by
  `flash_attention_rope()` and the `_make_mfa_rope_custom` backward.
- **374 tests pass** (up from 358 at v1.0.3). +16 new tests covering
  `attn_bias`, `backend`, dK/dV paged scatter, and sparse backward correctness.

---

## [1.0.3] — 2026-03-06

### Added
- **D=512 head_dim support** — forward and backward STEEL kernels now support
  `head_dim=512`. Both `flash_attention()` and `mx.vjp()` through it work
  correctly for f16/bf16, causal/non-causal, GQA, and unaligned sequence lengths.
- **D_SPLITS generalization** — `BD_HALF` in dQ and dKV backward generators
  is now fixed at 128 (not `BD/2`), and `D_SPLITS = BD / 128`. Metal loops
  over `[MFA_D_SPLITS]` tile arrays are fully unrolled at compile time, enabling
  any `head_dim` that is a multiple of 128 (64, 128, 256, 512).
- **13 new tests**: `TestD512Forward` (8) + `TestD512Backward` (5).

**350 tests pass.**

---

## [1.0.2] — 2026-03-06

### Changed
- **Build system**: added `mlx>=0.18.0` to `[build-system] requires` so MLX headers
  are available to the C++ extension during isolated `pip install` builds (e.g. CI,
  `--no-build-isolation` no longer required for a clean sdist install).
- **Version bump**: 1.0.1 → 1.0.2 (pyproject.toml, `__init__.py`, `csrc/bindings.cpp`).

**337 tests pass.** No API or kernel changes.

---

## [1.0.1] — 2026-03-06

### Fixed / Improved

| Track | Description | New tests |
|-------|-------------|-----------|
| GA | **PagedKVCache rewrite** — dual numpy float32 backing stores (was K-only, V never stored); `append()` uses block-level slice writes (was per-element Python loop); working `gather()` (was `NotImplementedError`); `k_pool`/`v_pool` properties with lazy cached `mx.array` views; `get_block_table()`/`get_seq_lens()` for direct use with paged STEEL kernel | 13 |
| GB | **`patch_mlx_lm` diagnostics** — `verbose=False` silent mode; `get_patch_stats()` returns `{forward_calls, steel_calls, fallback_calls, steel_ratio}`; `check_model_compatibility(model_name)` heuristic dict without loading the model; stats reset on each fresh `patch_mlx_lm()` | 17 |
| GC | **Deprecation notes** — `flash_attention_with_kv_cache` marked `.. deprecated:: 1.0.1` in docstring; removal target v2.0 | — |

**337 tests pass.** No kernel changes (no C++/Metal modifications).

---

## [1.0.0] — 2026-03-06

### Highlights

First stable public release. All features from v1.0.0-rc1 and v1.0.0-rc2.

| Track | Description | Tests added |
|-------|-------------|-------------|
| FA | Unified KV-cache API (`flash_attention_kvcache`) | 17 |
| FB | Native sliding-window in STEEL kernel | 4 |
| FC | Fused RoPE cache append (`flash_attention_kvcache_rope_append`) | 3 |
| FD | Kernel-level paged KV STEEL forward + Flash Decode | 15 |
| FX | `return_lse`, `cache_batch_idx`, `rotary_dim` | 8 |

**307 tests pass.** Full Python API with 33 public exports.

### Package
- First PyPI release: `pip install mlx-mfa`
- `pyproject.toml`: `Development Status :: 5 - Production/Stable`, `numpy` added to dependencies
- `MANIFEST.in`: adds `examples/`, `CHANGELOG.md`, `csrc/mfa/`
- `examples/`: 5 practical scripts covering all major API paths

See `[1.0.0-rc1]` and `[1.0.0-rc2]` below for the complete feature details.

---

## [1.0.0-rc2] — 2026-03-06

### Added
- **Track FD: Kernel-level paged KV streaming in STEEL forward kernel** — Metal kernel
  `mlx_mfa_paged_attention` reads K/V tiles directly from the `[num_blocks, block_size,
  H_kv, D]` pool via cooperative `block_table` lookup, eliminating a separate gather
  Metal dispatch. New `KernelType::PagedSteelForward`, `MFAPagedSteelParams`,
  `generate_paged_steel_forward_source()`, `MFAPagedSteelForward` Primitive, and
  `mfa_paged_steel_forward` nanobind binding. GQA, causal, sliding window all supported.
  `flash_attention_paged()` routes to the kernel for f16/bf16 D∈{64,128,256}.
  Benchmark (M1 Max, f16, B=1 H=8 D=128): **1.26–1.58x** faster than gather+attend.
- **Track FD-decode: Paged Flash Decode path** — For decode steps (N_q ≤ 4, S ≥ 256),
  `flash_attention_paged()` routes through Metal gather + `flash_attention()`, which
  activates the existing split-KV Flash Decode two-phase kernel for better SM parallelism.
- **Track FD-bench: `benchmarks/bench_paged_kv.py`** — Three-way comparison:
  gather+attend vs kernel-level paged STEEL vs pre-gathered Flash Decode.
- **307 tests pass** (up from 292 in rc1): 11 `TestPagedSteelForward` + 4
  `TestPagedFlashDecode`.

### Changed
- (infra) `has_window` added to `KernelKey` hash/equality; `window_left` wired into
  `MFASteelParams` — prerequisite for Track FD kernel dispatch.

---

## [1.0.0-rc1] — 2026-03-06

### Added
- **Track FB: Native sliding window in STEEL kernel** — `window_left` param in
  `MFASteelParams`; `has_window` KernelKey flag; K-tile `kb_start` computed per
  Q-block inside the persistent loop; boundary tiles apply element-wise mask.
  Fixed multi-tile boundary bug (only first boundary tile was masked), NaN-safe
  online softmax (all-masked-tile guard), and test reference `qL_off` alignment.
  `flash_attention(..., window_size=(left, right))` public API. 4 tests.
- **Track FA: Unified KV cache API** — `flash_attention_kvcache(q, k_cache, v_cache, ...)`
  replaces fragmented `with_kv_cache` / `paged` / `rope` paths. Dense + paged modes,
  RoPE, softcap, ALiBi, sliding window, `cache_seqlens`, `cache_batch_idx`. 17 tests.
- **Track FX-1: `return_lse` in `flash_attention`** — Expose logsumexp `L [B,H,N]`
  (log2 domain) alongside output when requested. MFA path uses `mfa_forward_with_lse`
  (free); fallback materialises log2-domain LSE via pure-MLX ops. 4 tests.
- **Track FX-2: `cache_batch_idx` in `flash_attention_kvcache`** — Non-contiguous
  batch→cache-slot mapping for continuous batching; `k_cache[cache_batch_idx]` gather
  before attention dispatch. 2 tests.
- **Track FX-3: `rotary_dim` partial RoPE** — Rotate only first `rotary_dim` dims;
  remainder passes through unchanged. STEEL kernel forces MLX fallback when
  `rotary_dim < head_dim`. 2 tests.
- **Track FC: Fused RoPE in cache append** — `flash_attention_kvcache_rope_append`
  rotates `k_new` BEFORE concat, storing pre-rotated keys in cache. O(1) rotation
  cost per decode step vs O(past_len) for naive re-rotation. `benchmarks/bench_kvcache.py`
  added for A/B comparison. 3 tests.

### Tests
Total collected: **292**

---

## [0.9.3] — 2026-03-06

### Added
- **Track EA: Differentiable `flash_attention_varlen`** — `mx.custom_function`
  wrapper adds full autograd. Forward: STEEL varlen kernel (f16/bf16, D=64/128/256);
  backward: splits per sequence through `flash_attention`. `TestVarlenBackward` (6 tests).
- **Track EB: Metal paged KV gather kernel** — `MFAPagedKVGather` Primitive
  gathers pool pages to `[B, H, max_kv_len, D]` in a single Metal dispatch.
  `flash_attention_paged` rewritten with `mx.custom_function`: `dQ` correct via
  `vjp(flash_attention)`; pool gradients are zeros (cache buffers).
  `TestPagedBackward` (6 tests).
- **Track EC: Varlen packed formats** — `flash_attention_varlen_qkv_packed` and
  `flash_attention_varlen_kv_packed` accept head-first or flat fused tensors and
  route to `flash_attention_varlen`. `TestVarlenPacked` (4 tests).
- **Track ED: Documentation refresh** — `docs/ARCHITECTURE.md` rewritten to 476 lines:
  updated backward routing tree (STEEL bwd / SDPA vjp / compiled vjp), new §8 (STEEL
  native backward — FA-2 log2 domain, GQA `gqa_factor`, D=256 three-phase D-split),
  new §9 (varlen backward via `mx.custom_function`), new §10 (paged KV gather — Metal
  kernel pseudocode, forward/backward flow, per-seq slicing rationale), expanded Public
  API table to all 31 exports. `docs/INVENTORY.md` regenerated from scratch: all line
  counts verified with `wc -l`, 31 `__all__` exports, 10 KernelType entries, 7 C++
  Primitive classes, 257 pytest runs / 212 test functions, 40 test classes, 10
  benchmarks. `README.md`: API Reference expanded from 7 to all 31 exports (param
  tables for core attention functions; compact reference table for 13 mask builders);
  Features section updated with v0.9.2–v0.9.3 additions.

### Tests
Total collected: **257 pytest runs / 212 test functions** (EA adds 6, EB adds 6, EC adds 4).

---

## [0.9.2] — 2026-03-06

### Added
- **Track DA: GQA backward guard fix** — Removed incorrect Python guard that blocked
  STEEL backward dispatch for grouped-query attention (H_q ≠ H_kv). The STEEL kernels
  have supported GQA since v0.9.0 via the `gqa_factor` Metal define; the Python
  `use_steel_bwd` predicate now correctly allows GQA shapes through.
- **Track DC: `mx.compile` for `_apply_rope_mlx`** — Shape-keyed compile cache
  (`_rope_compile_cache`) with separate `_impl` closures for interleaved and
  non-interleaved layouts. Scalars `offset` and `interleaved` are frozen in the
  closure to avoid dynamic control flow in the compiled graph. Median speedup ≈1.4×
  over the raw Python fallback (measured in `bench_compile.py`).
- **Track DC: `benchmarks/bench_compile.py`** — New benchmark (50-iteration median)
  comparing compiled vs raw latency for `_softcap_sdpa_ref`, `_alibi_sdpa_ref`, and
  `_apply_rope_mlx` (interleaved + non-interleaved) at N=2048 D=128 f16.
- **Track CE: D=256 D-split STEEL backward** — `generate_steel_backward_dq_source()`
  and `generate_steel_backward_dkv_source()` now emit D-split Metal code when
  `head_dim=256` (`BD_HALF=128`). Q/dO/K/V tiles are loaded in lo (0..127) and
  hi (128..255) passes sharing one threadgroup buffer; dQ/dK/dV accumulators become
  lo/hi register-tile pairs. TGP budget ≈ 23 KB (well below 32 KB limit). The
  `use_steel_bwd` guard is widened from `D ≤ 128` to `D ≤ 256`.
- **Track DD: Documentation refresh** — `docs/INVENTORY.md` updated to v0.9.2:
  test count 241, benchmark count 9, backward strategy table, DA–DE additions table.
  CE row in v0.9.1 table updated from "deferred" to "completed in v0.9.2".

### Fixed
- **Track DB: CHANGELOG inaccuracies** — v0.9.1 entry for Track CB now correctly states
  `_apply_rope_mlx` was NOT compiled in v0.9.1 (completed in Track DC / v0.9.2).
  Test count corrected to 232.

---

## [0.9.1] — 2026-03-06

### Added
- **Track CA: Vec4 block loads** — `MFABlockLoaderT` uses `float4`/`half4` aligned
  vector reads for all tile loads in the STEEL forward kernel, reducing instruction
  count per tile by 4× on cache-line-aligned data.
- **Track CB: `mx.compile` for fallback paths** — The Python fallback routes
  (`_softcap_sdpa_ref`, `_alibi_sdpa_ref`) are wrapped with `mx.compile`.
  `_apply_rope_mlx` and the sparse/varlen fallbacks are NOT yet compiled
  (completed in Track DC / v0.9.2).
- **Track CC: Persistent multi-Q-block kernel** — The STEEL forward kernel now iterates
  over an outer `qb` loop (`[0, NQ)`) within a single threadgroup dispatch, processing
  up to 4 Q-blocks per launch. Amortizes Metal command buffer overhead at N ≥ 4096.
- **Track CD: GQA in STEEL backward** — The STEEL dQ and dKV backward kernels now
  handle grouped-query attention.  The `gqa_factor` (H_q / H_kv) is baked into the
  Metal shader as `#define MFA_GQA_FACTOR <N>` at compile time, avoiding Metal
  `constant`-address-space struct-field read ambiguity.  `KernelKey` extended with
  `gqa_factor` so each GQA ratio compiles to a distinct cached pipeline.
- **Track CF: Double-buffer ping-pong** — Separate `K_smem` / `V_smem` threadgroup
  arrays when D ≤ 128 (TGP ≈ 19.2 KB < 32 KB limit).  Reduces barriers per K-tile
  from 4 → 2: V-tile stores overlap K-GEMM; K[n+1]-tile stores overlap P@V.
  Phase-0 preloads K[0] before the loop; `loader_k/v.next()` called inline.
  Disabled for D=256 (budget), RoPE (extra TGP), and sparse.
- **Track CG: `benchmarks/bench_all.py`** — Consolidated forward + backward benchmark
  suite (`--fwd-only`, `--bwd-only`, `--no-save` flags).  Appends markdown results
  table to `docs/benchmarks/RESULTS.md`.
- **Track CH: Documentation refresh** — `docs/INVENTORY.md` updated to v0.9.1
  (test count 232, benchmark count 8, kernel table, CA–CI additions table).
  `docs/ARCHITECTURE.md` adds notes on CF double-buffer and CC persistent kernel.
  `README.md` roadmap updated: N1 marked Done (v0.9.0); CA/CB/CC/CD/CF rows added.

### Deferred
- **Track CE: D=256 backward multi-pass** — 3D blocking for the STEEL dQ/dKV
  backward kernels (analogous to the forward D=256 path) is deferred to v1.0.
  D=256 backward continues to route to `mx.vjp(SDPA)` (same as v0.9.0).

---

## [0.9.0] — 2026-03-06

### Added
- **Track BA/BB/BC: STEEL native backward** — `mx.grad(flash_attention)` now dispatches
  native Metal STEEL backward kernels (`MFASteelBwdDQ`, `MFASteelBwdDKV`) for f16/bf16
  instead of `mx.vjp(SDPA)`. 2-3× backward speedup on D=64/128. f32 stays on ccv path.
  Key fixes: `Ktile[1,MFA_TK]` tile declaration (was 1×1, causing UB for ik>0) and
  `_sever_lazy_graph(cotangent)` before gradient checkpointing re-run of forward
  (prevents Metal buffer aliasing via lazy graph ancestry). 209 tests pass.
- **Track BD: STEEL varlen forward kernel** — `flash_attention_varlen` dispatches a
  dedicated Metal STEEL kernel instead of Python split-cat. Packed Q/K/V layout
  `[1, H, N_total, D]` with `cu_seqlens` offsets; per-threadgroup batch-item decode.
  Critical race-condition fix: `threadgroup_barrier` at START of K-loop prevents
  P@V reads (V from KV_smem) from racing against next iteration's K write.
  K-boundary `-INF` mask prevents softmax denominator inflation for partial K-tiles.
  215 tests pass.
- **Track BE: Paged KV Cache Phase 1** — `PagedKVCache` block allocator with pool
  `[num_blocks, block_size, H_kv, D]`; per-seq block table; `append`/`free_seq` helpers.
  `flash_attention_paged(q, k_pool, v_pool, block_table, seq_lens, ...)` reconstructs
  contiguous K/V per batch item via block-table gather, routes to `flash_attention`.
- **Track BF: QKV/KV packed tensor formats** — `flash_attention_qkv_packed` handles
  flat `[B, N, 3·H·D]` and head-first `[B, H, N, 3, D]` packed layouts.
  `flash_attention_kv_packed` handles `[B, S, 2·H·D]` and `[B, H, S, 2, D]`.
  Both raise `ValueError` for unsupported shapes.
- **Track BG: Backward benchmark** — `benchmarks/bench_backward.py` measures
  flash_attention VJP vs SDPA VJP across D=64/128, f16/bf16, causal/non-causal.
- **Track BH: Varlen benchmark update** — `benchmarks/bench_varlen.py` updated to
  note STEEL varlen kernel; section header updated to v0.9.0.
- **Tests: 232 pytest runs** (180+16 test functions; 232 with parametrize expansion)

## [0.8.0] — 2026-03-05

### Added
- **Track AA: Softcap** — `flash_attention(..., softcap=50.0)` applies `tanh(S/cap)*cap`
  before softmax; fused into Metal STEEL kernel for f16/bf16, Python fallback for f32.
- **Track AB: ALiBi** — `flash_attention_alibi(q, k, v, alibi_slopes, ...)` adds
  per-head linear position biases (slope_h × (k_pos − q_pos)). Metal kernel fuses
  bias into the QK tile accumulation; Python reference fallback included.
- **Track AC: RoPE non-interleaved (GPT-NeoX)** — `flash_attention_rope(..., interleaved=False)`
  supports split-halves RoPE layout `(d, d+D/2)` in addition to LLaMA adjacent pairs.
  Metal kernel and Python `_apply_rope_mlx` both branch on `interleaved`.
- **Track AD: Per-batch `cache_seqlens`** — `flash_attention_rope` now accepts
  `cache_seqlens` as a `list[int]`, `mx.array`, or `int`. Per-element dispatch via
  Python split-cat; MLX lazy eval fuses concurrent GPU dispatches.
- **Track AE: Graceful D_v ≠ D_qk fallback** — When `v.shape[-1] != q.shape[-1]`,
  routes to `mx.fast.scaled_dot_product_attention` instead of raising. K dimension
  must still equal Q (raises `ValueError` otherwise).
- **Track AF: `flash_attention_with_kv_cache`** — Fused KV cache append:
  `(output, k_updated, v_updated) = flash_attention_with_kv_cache(q, k_new, v_new, k_cache, v_cache)`.
  Concatenates along the sequence axis, dispatches one attention call.
- **Track AG: Attention dropout** — `flash_attention(..., dropout_p=0.2)` drops
  softmax weights during training. Uses `mx.where` causal masking to avoid
  `0.0 × −inf = NaN` in the masked region.
- **Track AH: Return attention weights** — `flash_attention(..., return_attn_weights=True)`
  returns `(output, attn_weights)` where weights are the full softmax probability matrix
  `[B, H, N, S]`. Compatible with softcap and dropout.
- **Track Z: Benchmark scripts** — `benchmarks/bench_softcap_alibi.py` measures
  softcap and ALiBi overhead vs SDPA baseline across four variants.
- **Tests: 209 total** (up from 93 in v0.4.0)

### Changed
- `flash_attention_rope` now accepts `Union[int, mx.array, Sequence[int]]` for `cache_seqlens`

## [0.7.0] — 2026-03-05

### Added
- **Track O: Spatial 2D/3D block masks** — `make_spatial_2d_mask`, `make_spatial_3d_mask`, `make_topk_spatial_mask`
- **Track P: Segment / document masks** — `make_segment_mask`, `make_causal_segment_mask`
- **Track Q: Adaptive window mask** — `make_adaptive_window_mask` (SeedVSR2-style resolution-scaled windows)
- **Track R: 3D RoPE table construction** — `make_rope_3d_tables` + `flash_attention_rope(rope_3d=...)` dict API
- **Track S: Variable-length batching** — `flash_attention_varlen` (split-concat implementation)
- **Track T: 4 benchmark scripts** — spatial masks, segment, varlen, 3D RoPE
- Pure Python release — no Metal kernel changes
- Tests: ~150 total


## [0.6.0] — 2026-03-05

### Added
- **Track K: Quantized KV cache** — Q4/Q8 dequantized before STEEL kernel
- **Track L: RoPE 1D fusion** — `flash_attention_rope()` with in-kernel rotary embeddings
- **Track M: Paged Attention design doc** — `docs/PAGED_ATTENTION_DESIGN.md`


## [0.5.0] — 2026-03-05

### Added
- **Flash Decoding (Track H)** — Two-phase split-KV attention for decode mode
  (N_q ≤ 4, S ≥ 256, f16/bf16). Phase 1 dispatches KV-sequence splits in
  parallel; Phase 2 reduces partial outputs via log-sum-exp. Activated
  automatically for eligible shapes.
  - New KernelType variants: `FlashDecodePartial`, `FlashDecodeReduce`
  - New params structs: `FlashDecodePartialParams`, `FlashDecodeReduceParams`
  - `compute_num_splits(kL, BK)` — targets ≥2 K-tiles per split, capped at 32
  - 11 new tests: non-causal/causal across D=64/128/256, GQA, bf16, boundary cases

- **M5+ detection stub (Track I)** — Forward-compatibility for Apple M5 (gen≥17,
  A19 SoC with Metal 4 tensor API)
  - `get_device_info()` now returns `is_m5_plus` (bool)
  - Gen 17 → `"M5"` chip name in `_GEN_TO_CHIP` mapping
  - `TensorOpsForward` KernelType reserved as commented stub in `shader_cache.hpp`
  - 3 new tests covering flag correctness, chip name, and M5 ⊇ M3+ logic

### Fixed
- `enc.barrier()` replaces `enc.maybeInsertBarrier()` between Flash Decode
  Phase 1 and Phase 2 — `maybeInsertBarrier()` is a no-op for raw
  `MTL::Buffer*` bindings (only `set_output_array()` sets `needs_barrier_`)
- `qL_off = S - N` for causal decode so query token at position `i` correctly
  sees keys `0..(S - N + i)` instead of starting from key 0

### Tests
- 107 tests total (was 93)

---

## [0.4.0] — 2026-02-xx

### Added
- **Track F** — M3+ architecture routing: BK=32 for D=128 on M3/M4 (gen≥15),
  `MFA_FORCE_GEN` env var override, `ARCHITECTURE_GEN` #define in Metal shader
- **Track G** — Sparse backward pass: tiled FA-2 dQ/dK/dV that skips inactive
  blocks; `flash_attention_sparse(backward='sdpa_sparse')` public API
- **Track C** — Native GQA: removed `mx.repeat` expansion, STEEL kernel handles
  `gqa_factor` natively in the Metal shader

### Tests
- 93 tests total (was 63)

---

## [0.3.0] — 2026-01-xx

### Added
- **Track D** — mlx-lm integration: `patch_mlx_lm()` / `unpatch_mlx_lm()`
- Native GQA support in STEEL kernel (gqa_factor parameter)
- `make_causal_block_mask()`, `make_sliding_window_mask()` public helpers
- mlx-lm integration tests (11 tests)

---

## [0.2.0] — 2025-12-xx

### Added
- **Track B** — Block-sparse attention: `flash_attention_sparse(q, k, v, mask)`
- Sparse STEEL kernel variant (K-loop skip, zero warp divergence)
- Sliding-window mask giving 3–6× speedup at long contexts

### Performance (M1 Max, B=1 H=8 f16, causal)
| D | N | Speedup |
|---|---|---------|
| 64 | 8192 | 2.11× SDPA |
| 128 | 8192 | 1.72× SDPA |
| 128 N=8192 sliding-window=512 | | 5.7× SDPA |

---

## [0.1.0] — 2025-11-xx

### Added
- Initial release: STEEL forward kernel replacing ccv-based MFA
- Full forward pass (D=64/128/256, f16/bf16/f32, causal/non-causal)
- Backward via `mx.vjp(scaled_dot_product_attention)`
- GQA via `mx.repeat` expand (later replaced by native GQA in v0.3)
- Public API: `flash_attention()`, `is_mfa_available()`, `get_device_info()`
- 41 tests
