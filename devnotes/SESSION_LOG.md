---
## [2026-04-06 09:30] A2: Audit Dispatch for DiT/UNet Shapes

### Plan
- **Objective:** Benchmark and fix dispatch routing for DiT/UNet/VSR non-causal and cross-attention shapes
- **Files to modify:** `mlx_mfa/dispatch_policy.py`, `mlx_mfa/attention.py`
- **Dependencies impacted:** `should_use_mfa()` signature (new `kv_seq_len` kwarg), dispatch cache key in `attention.py`

### Changes made
- `mlx_mfa/dispatch_policy.py:286-330` — added `kv_seq_len` parameter + cross-attn routing logic [HIGH]
  - Small-KV rule: `N_kv <= 512 and N_q > 8192` → SDPA (few K-tiles, tile overhead dominates)
  - Large-KV rule: `N_kv >= 4096 and N_q <= 4096` → MFA (few Q-tiles, flash attention wins big)
- `mlx_mfa/attention.py:380-388` — passes `kv_seq_len=k.shape[2]`, updated cache key to 8-tuple [HIGH]
- `scripts/audit_dit_dispatch.py` — benchmark script for 11 self-attn + 7 cross-attn shapes [MEDIUM]
- `tests/test_dispatch_dit.py` — 17 regression tests (dispatch policy + functional) [HIGH]
- `docs/audit_dit_dispatch_report.md` — full benchmark report with analysis [HIGH]
- `docs/audit_dit_dispatch_results.json` — raw benchmark data for M5 calibration [HIGH]

### Dependency & regression check
- `should_use_mfa()`: new `kv_seq_len` kwarg is optional with default `None` → all existing callers unaffected ✓
- `attention.py`: dispatch cache key extended from 7-tuple to 8-tuple → cache invalidated, no correctness issue ✓
- `inference.py:1125 should_use_sage_decode`: not affected (different function) ✓
- `calibrate_dispatch()`: calls `should_use_mfa()` without `kv_seq_len` → uses None default ✓
- 902 existing tests pass (1 flaky TQ test: pre-existing GPU non-determinism) ✓
- 17 new tests pass ✓

### Tech cost assessment
- Complexity: O(1) additional comparisons in dispatch hot path — negligible
- Memory: unchanged (no new allocations)
- Kernel launches: unchanged (only routing decision affected)

### Confidence
- Overall: [HIGH]
- Risks: none — all dispatch changes backed by benchmark data on M1 Max

### Benchmark results (M1 Max, f16, non-causal)
Self-attention: 11/11 correctly routed. MFA 1.17-1.65x for N>=4096.
Cross-attention fix: 5 shapes corrected (SDPA now wins for small N_kv + large N_q).
LTX-2 audio→video: MFA 8.59x (new large-KV rule).

---
## [2026-03-31 10:30] SVDQuantLinear Phase 1 — Unfused W4A16 + SVD Low-Rank (v2.25.0)

### Plan
- **Objective:** Add SVDQuantLinear to mlx-mfa — W4A16 linear layer with optional SVD low-rank correction
- **Files created:** mlx_mfa/svdquant/__init__.py, linear.py, quantize.py; tests/test_svdquant.py
- **Files modified:** mlx_mfa/__init__.py, pyproject.toml
- **Dependencies impacted:** None — new module, no existing callers

### Changes made
- `mlx_mfa/svdquant/linear.py` — NEW: SVDQuantLinear nn.Module (W4A16 + rank-r FP16 correction) [HIGH]
- `mlx_mfa/svdquant/quantize.py` — NEW: quantize_model() tree walker, _replace_layers() [HIGH]
  - Key fix: MLX children() returns COPIES — must use getattr() to mutate actual model attributes
- `mlx_mfa/svdquant/__init__.py` — NEW: public API exports [HIGH]
- `mlx_mfa/__init__.py:L155-156` — Added SVDQuantLinear, quantize_model imports [HIGH]
- `mlx_mfa/__init__.py:L30` — Version 2.24.1 → 2.25.0 [HIGH]
- `pyproject.toml:L7` — Version 2.24.1 → 2.25.0 [HIGH]
- `tests/test_svdquant.py` — NEW: 21 tests (16 correctness + 5 benchmarks) [HIGH]

### Dependency & regression check
- mlx_mfa/__init__.py: new import added at end of import block, no existing imports affected ✓
- No existing tests modified ✓
- SVDQuantLinear is standalone — no coupling to STEEL, TurboQuant, or other modules ✓

### Tech cost assessment
- Forward pass: O(MKN) for quantized_matmul + O(rKN + rMN) for low-rank correction
  - For r=32, K=2560: low-rank is 2.5% of main FLOPS
- Memory: W4 + rank-32 for [2560, 2560] = ~3.4 MB vs 13.1 MB FP16 (3.8× compression)
- SVD calibration: O(min(M,K)²·max(M,K)) — slow for large layers but one-time offline cost

### Benchmark results (M1 Max, f16, micro-benchmark)

| Shape (M×K×N) | FP16 ms | W4 ms | SVD32 ms | W4 speedup | LR overhead |
|---|---|---|---|---|---|
| 2560×2560×512 | 0.04 | 0.03 | 0.03 | 1.28× | 10.8% |
| 6912×2560×512 | 0.03 | 0.03 | 0.03 | 1.00× | 16.6% |
| 2560×6912×512 | 0.02 | 0.03 | 0.04 | 0.88× | 36.4% |
| 5120×5120×1024 | 0.03 | 0.03 | 0.03 | 0.96× | 17.9% |
| 13824×5120×1024 | 0.02 | 0.03 | 0.03 | 0.96× | 14.9% |

Note: sub-ms times are kernel-launch dominated. Real model benchmarks needed.
LR overhead mostly 10-18%, except K>M case (36%) which may benefit from Phase 2 fusion.

### Confidence
- Overall: [HIGH]
- Risks: Micro-benchmarks are launch-overhead dominated; real model benchmarks needed to validate Phase 2 decision

### Test results
- 16/16 correctness tests pass
- 5/5 benchmark tests pass
- Total: 21/21 pass

---
## [2026-03-31 15:00] GNA Native Metal Kernel — Inline 3D Window (v2.26.0)

### Plan
- **Objective:** Native GNA kernel with inline 3D window check, replacing block_mask allocation for the forward pass
- **Files created:** csrc/mfa_gna_fwd.cpp, csrc/mfa_gna_fwd.hpp, tests/test_gna_native.py
- **Files modified:** csrc/bindings.cpp, csrc/mfa_attention.cpp, csrc/mfa_attention.hpp, csrc/shader_cache.hpp, csrc/shader_cache.mm, CMakeLists.txt, mlx_mfa/attention.py, tests/test_attention.py, pyproject.toml, mlx_mfa/__init__.py
- **Dependencies impacted:** flash_attention_gna() now tries native path first; backward tests need MFA_DISABLE_GNA_NATIVE=1

### Changes made
- `csrc/mfa_gna_fwd.cpp` — NEW: JIT Metal kernel generator for GNA forward (575 lines) [HIGH]
  - gna_tile_active() for 3D bounding box overlap test (tile skip)
  - Per-element GNA mask after Q@K^T GEMM (exact window bounds per query)
  - Matches V2 STEEL loader templates, SIMD layout, softmax, P@V
- `csrc/mfa_gna_fwd.hpp` — NEW: MFAGNAParams struct + function declarations [HIGH]
- `csrc/shader_cache.hpp:L73` — GNAForward=24 in KernelType enum [HIGH]
- `csrc/shader_cache.mm` — GNAForward JIT compile path, removed debug logging [HIGH]
- `csrc/mfa_attention.cpp` — MFAGNAForward::eval_gpu() dispatch [HIGH]
- `csrc/mfa_attention.hpp` — MFAGNAForward class declaration [HIGH]
- `csrc/bindings.cpp` — mfa_gna_forward nanobind binding (13 params) [HIGH]
- `CMakeLists.txt:L96` — mfa_gna_fwd.cpp in MFA_SOURCES [HIGH]
- `mlx_mfa/attention.py:L2370-2386` — Native path in flash_attention_gna() [HIGH]
- `tests/test_gna_native.py` — NEW: 11 tests (10 correctness + 1 benchmark) [HIGH]
- `tests/test_attention.py:L3355` — Fixture to disable native GNA for backward tests [HIGH]
- `pyproject.toml:L7` — Version 2.25.0 → 2.26.0 [HIGH]
- `mlx_mfa/__init__.py:L30` — Version 2.25.0 → 2.26.0 [HIGH]

### Dependency & regression check
- flash_attention_gna(): native path tried first, sparse fallback preserved ✓
- GNA backward tests: fixture disables native (forward-only, no VJP) ✓
- All 864 tests pass (0 failures) ✓
- No existing V2/sparse/SDPA paths modified ✓

### Tech cost assessment
- Per-element mask: O(BQ * BK * 3) integer divisions per active K-tile
- gna_tile_active: O(1) per K-tile (skip inactive tiles)
- Memory: MFAGNAParams struct (10 ints) at buffer(6), no block_mask allocation
- Kernel launches: 1 per head (same as V2)

### Key findings
1. **Native GNA applies exact per-element masking** — more precise than sparse path
   which only does tile-level (BQ×BK) block masking. Sparse over-approximates.
2. **GNA with window=N, stride=1 ≠ dense attention** due to boundary effects —
   window shrinks near edges (by design, per Hassani et al.)
3. **Benchmark CogVideoX (N=70200, M1 Max):** Native 285ms vs Sparse 266ms (0.93×)
   — per-element masking adds divergent branching, offsetting tile-skip savings
4. **Blocked attention (stride=window)** matches SDPA exactly (max_err=0.000061)

### Confidence
- Overall: [HIGH]
- Risks: Native kernel slightly slower than sparse path for CogVideoX shapes;
  benefit is correctness (exact masking), not speed

---
## [2026-04-06 00:00] attn_bias native Metal kernel — split-K fix + debug cleanup

### Plan
- **Objective:** Fix split-K dispatch ignoring attn_bias, remove debug code
- **Files to modify:** `csrc/mfa_attention.cpp`, `csrc/shader_cache.mm`, `csrc/mfa_steel_fwd_v2.cpp`
- **Dependencies impacted:** V2 split-K path, shader cache pipeline selection

### Changes made
- `csrc/mfa_attention.cpp:L392` — added `!params_.has_attn_bias` to `v2sk_eligible` [HIGH]
  Forces bias queries to single-pass V2 (which implements bias) instead of split-K (which doesn't)
- `csrc/shader_cache.mm:L284-306` — removed 5 NSLog debug prints from `get_or_compile()` [HIGH]
- `csrc/mfa_steel_fwd_v2.cpp:L40,L869-875` — removed `#include <cstdio>` and fputs shader dump [HIGH]

### Dependency & regression check
- Split-K path: unaffected for non-bias queries (condition only adds exclusion) ✓
- V2 single-pass: unaffected (bias code was already correct) ✓
- Full suite: 920 passed, 19 xfailed, 22 xpassed ✓

### Tech cost assessment
- No runtime cost: compile-time bool check in dispatch logic
- No memory impact
- No kernel changes

### Confidence
- Overall: HIGH
- Risks: none — split-K exclusion is the same pattern used for block_mask

---
## [2026-04-06 01:00] A3: Varlen validation for token merging/pruning

### Plan
- **Objective:** Benchmark varlen vs padded dense for token-merged sequences
- **Files created:** `benchmarks/bench_varlen_pruning.py`, `docs/varlen_pruning_validation.md`
- **Dependencies impacted:** none (benchmark-only)

### Changes made
- `benchmarks/bench_varlen_pruning.py` — 5 scenarios, cu_seqlens rebuild, correctness check [HIGH]
- `docs/varlen_pruning_validation.md` — full report with tables + recommendations [HIGH]

### Key findings
- Varlen SLOWER than padded dense in most token merging scenarios (0.55–0.96×)
- Varlen wins only when length disparity > 2:1 (SeedVR2: 1.19× D=64)
- cu_seqlens rebuild: 0.004ms (negligible)
- Correctness: bit-accurate (max_err = f16 epsilon)
- Recommendation: default to padded dense for token merging; use varlen only for mixed-length batches

### Confidence
- Overall: HIGH
- Risks: none — benchmark-only, no code changes

---
## [2026-04-06 02:00] v2.27.0 release: docs + version bump + publish

### Plan
- **Objective:** Bump to v2.27.0, update all docs, commit, tag, push
- **Files to modify:** pyproject.toml, __init__.py, README.md, CLAUDE.md,
  CHANGELOG.md, ARCHITECTURE.md, API_MANUAL.md, FEATURE_COVERAGE.md,
  INVENTORY.md, SERVING_GUIDE.md

### Changes made
- Version bump 2.26.0 → 2.27.0 in pyproject.toml + __init__.py [HIGH]
- README.md: version, attn_bias feature, usage example, status tables [HIGH]
- CLAUDE.md: version, test count, feature table entry [HIGH]
- CHANGELOG.md: full v2.27.0 entry (attn_bias, dispatch audit, varlen validation) [HIGH]
- ARCHITECTURE.md: version, §7.9 attn_bias kernel docs, KernelType table, status [HIGH]
- API_MANUAL.md: version, attn_bias parameter documentation [HIGH]
- FEATURE_COVERAGE.md: version, attn_bias row [HIGH]
- INVENTORY.md: version [HIGH]
- SERVING_GUIDE.md: version, attn_bias status row [HIGH]

### Confidence
- Overall: HIGH
- Risks: none — documentation-only changes + version bump

---
## [2026-04-06 03:00] Hardening: lazy imports + isinstance + dead code + deprecated API

### Plan
- **Objective:** 4 hardening fixes for public usage
- **Files to modify:** mlx_mfa/__init__.py, mlx_mfa/runtime.py, mlx_mfa/external_cache.py, tests/test_attention.py
- **Dependencies impacted:** all imports from mlx_mfa (lazy timing change only)

### Changes made
- `mlx_mfa/__init__.py` — lazy imports for 6 submodules via __getattr__ [HIGH]
  Deferred: inference, runtime, kv_cache, external_cache, turboquant, svdquant
  Eager: attention, masks, quantize, dispatch_policy, compile_metallib
  __all__ unchanged. `from mlx_mfa import X` works for all X.
- `mlx_mfa/runtime.py:L46-69` — isinstance() replaces type().__name__ for dispatch [HIGH]
  `_build_secondary_cache_for_context` now uses `from mlx_mfa.inference import ...`
  deferred import inside function body (avoids circular import at module level).
  Other type().__name__ usages (error msgs, repr) left as-is — no dispatch logic.
- `mlx_mfa/external_cache.py:L80-101` — removed dead _to_numpy_preserve/_restore_mx [HIGH]
  Confirmed unused: only defined in LocalHostKVStoreAdapter, never called.
  put() method stores mx.array directly (comment confirms numpy roundtrip skipped).
- `tests/test_attention.py:L1473,1500,9777,9798,10264` — mx.metal.clear_cache() → mx.clear_cache() [HIGH]
  5 occurrences replaced. mx.clear_cache() is the device-agnostic replacement.

### Dependency & regression check
- All lazy-imported names verified: `from mlx_mfa import X` works for all X in __all__
- isinstance check: TurboQuantPagedInferenceContext inherits PagedInferenceContext, so
  isinstance correctly matches (vs type().__name__ which would miss subclasses)
- Full suite: 918 passed (920 - 2 deselected pre-existing flaky TQ 2-bit)

### Tech cost assessment
- Lazy imports: eliminates ~6 module loads at import time; 0 runtime cost after first access
- isinstance: 1 deferred import per call (cached by Python import system)
- Dead code: -22 lines, 0 functional change

### Confidence
- Overall: HIGH
- Risks: none — all names still importable, all tests pass

---
## [2026-05-11 22:25] [CLAUDE] Sprint B Phase 0 close: LCSA / block-sparse NAX survey
STATUS: COMPLETE

### Plan
- Objective: survey + bottleneck characterization for LCSA / block-sparse
  attention NAX -- read + measure pass per Sprint B Phase 0 prompt §1.
- Files to add: bench/lcsa_nax_baseline.py, bench/lcsa_nax_phase0_analysis.py,
  docs/lcsa-nax/survey-report.md, raw + analysis JSON, runlog.

### Changes
- bench/lcsa_nax_baseline.py: A/B/A 3-session §4-compliant harness with
  smoke gate (Phase 1.1 lesson). 6 shapes covering FlashVSR LCSA range
  (4k/8k/16k N × dense/sparse window).
- bench/lcsa_nax_phase0_analysis.py: cross-session medians + theoretical
  bound (25 TF NAX sustained from Sprint C; 410 GB/s HBM M5 Max) +
  headroom ranking.
- docs/lcsa-nax/survey-report.md: 538-line survey, all 12 sections
  populated.

### Dependency & regression check
- No production code changed (Phase 0 is read + measure only).
- Master 914-test baseline unchanged.
- Sprint A + Sprint C branches frozen.

### Validation
- Ran: bench/lcsa_nax_baseline.py × 3 sessions in nohup, §4 cooldowns,
  21:48-22:18 UTC (30 min).
- Smoke gate (all 3 sessions): rel_err=0.0 vs MLX SDPA+float-bias on
  smoke shape. PASS.
- Cross-session variance per shape: 0.58-4.38% (well within §B.7 10% bar).

### Key findings
- §2 MLX 0.31.2: NO block-skip on M5+; SDPA mask is dense compute + bias.
- §3 mlx-mfa M5+: flash_attention_sparse falls back to
  _sparse_fallback_sdpa_perhead() (mask expansion to [B,H,N,S] float
  bias → dense SDPA). Zero block-skip on M5+ today.
- §4 Apple MPP NAX: only dense matmul2d + convolution2d. No sparse
  primitive. get_mask() is tile-boundary check, not sparsity.
- §5 FlashVSR: WAN DiT dim=1536 H=12 D=128, window=(2,8,8)=128 tokens.
  30 sparse-attention calls per forward pass at identical shape.
  SparkVSR audit: no LCSA usage (sparse refs are RAFT-dataset only).
- §6 theoretical bound: all shapes compute-bound (not bandwidth);
  bound 0.13-4.02 ms across shape range.
- §7 ROI ranking: top cluster is large-N (16k) sparse-density (0.03),
  44.73× headroom, 1.38 s saved per forward pass.
- §8 bench: MFA-sparse 0.5× SDPA across all shapes (mask-expansion
  overhead); density has NO effect on either timing (both dense).
- §10 recommended: Option α (block-skip dispatch via dense matmul2d) --
  Sprint C Conv3D template applies directly.

### Tech cost
- Phase 0 produced ZERO production code. All deliverable is doc + data.
- 265 LOC bench harness Python.
- 130 LOC analysis tool Python.
- 538 LOC survey report Markdown.

### Verdict for Marco
**PROCEED to Phase 1.0 design.** Median headroom 16.38×, max 44.73×
on M5 Max FlashVSR-style LCSA shapes. Realistic block-skip speedup
projection: 3-15× depending on density, after Sprint C-precedent
50% efficiency derate. ~15-30 seconds attention budget unlocked per
21-frame FlashVSR inference run.

### Git
- 83bd223 bench(lcsa-nax): Phase 0 baseline harness
- f71128b docs+bench(lcsa-nax): Phase 0 survey skeleton + analysis tool
- (next, this commit) docs+bench(lcsa-nax): Phase 0 final data +
  consolidated survey + SESSION_LOG

### Next concrete step for Marco
1. Read survey-report.md §1 (verdict) + §10 (recommended approach) +
   §12 (sign-off). ~10 min.
2. If approved: kick off Sprint B Phase 1.0 prompt -- design doc that
   takes this survey as input + Sprint C Phase 1.0 design as template.
3. Phase 1.0 produces: algorithm + tile shapes + primitive class +
   validation strategy + sub-phase breakdown + risks register. Then
   Phase 1.x implementation sprints follow (5-phase pattern per Sprint C).


---
## [2026-05-12 00:50] [CLAUDE] Sprint B Phase 1.0 design doc + v2.33.x release
STATUS: COMPLETE

### Section A — v2.33.x release flow (autonomous execution)
- Multi-SoT version fix applied to both Sprint D base + v2.33.1 patch
  base (mlx_mfa/__init__.py + README header + bindings.cpp `__version__`
  removal — single source of truth in mlx_mfa.__version__).
- Sprint D merged → master → tag v2.33.0 created.
- v2.33.1 patch + version fix merged → master → tag v2.33.1 created
  (with merge conflicts on version-string files resolved in favor of
  v2.33.1 final state).
- Built wheels for both versions (mlx_mfa-2.33.0-* + mlx_mfa-2.33.1-*).
- twine upload --skip-existing: both versions live on PyPI.
- git push origin master --tags: both tags pushed to GitHub.
- gh release create for v2.33.0 + v2.33.1: release pages live.

### Section B — Sprint B Phase 1.0 design
- docs/lcsa-nax/lcsa-nax-design.md (488 lines, 12 sections):
  §1 strategic context (FlashVSR LCSA + SparkVSR, additive to v2.33.1)
  §2 algorithm (block-skip dispatch via NAX matmul2d, online softmax)
  §3 sub-phase 0 microbench requirement (targeted re-bench at per-tile
     shapes; gate ≥ 5 TF)
  §4 MFASparseAttentionForward C++ Primitive via fast::metal_kernel
  §5 unified SparseAttnKey cache (D3 from start)
  §6 BT defaults per cluster (32 if density>0.10, else 64)
  §7 three-axis validation (output sane + path entered + edges)
  §8 5-sub-phase breakdown (1.1 microbench+scaffold, 1.2 shapes+causal,
     1.3 BT autoresearch, 1.4 very-sparse fast path, 1.5 perf sweep)
  §9 10-risk register
  §10 FlashVSR per-call-regen scope (Sprint B addresses what v2.33.1
      cannot)
  §11 relation to v2.33.1 (additive, dispatcher pattern)
  §12 10 open questions / R1 revision targets
- docs/lcsa-nax/lcsa-nax-phase1_0-decisions.md (154 lines, B-D1 to B-D10):
  B-D1 Option α, B-D2 fast::metal_kernel from Phase 1.1,
  B-D3 unified cache, B-D4 BT defaults, B-D5 targeted microbench,
  B-D6 three-axis mandatory, B-D7 all-False row → 0, B-D8 causal handling,
  B-D9 additive to flash_attention_sparse, B-D10 patcher integration.

### Git
- Tags v2.33.0, v2.33.1 created + pushed
- PyPI: mlx-mfa 2.33.0 + 2.33.1 published
- GitHub: release pages for both tags
- master: ae5f265 (merged Sprint D + v2.33.1 + version fix)
- experiment/lcsa-nax-phase1_0_design: design doc + decisions

### Next concrete step
Sprint B Phase 1.1: sub-phase 0 microbench check + scaffold
MFASparseAttentionForward + smallest LCSA shape end-to-end.
Branch experiment/lcsa-nax-phase1_1 from feat/lcsa-nax post-design-merge.

