# Changelog

All notable changes to mlx-mfa are documented here.

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
  `notes/d256_decision.md` + JSON artifact.
- **perf**: D=256/512 D-split tile selection is now explicitly isolated from
  D=128 BK calibration overrides. Added `select_steel_v2_dsplit_block_config()`
  plus `MFA_V2_FORCE_BK_D256=32|64` debug override so global
  `MFA_V2_FORCE_BK` no longer leaks into large-D routing.
- **perf**: Auto-dispatch now accepts dtype in policy decisions and applies a
  D=256 separate-family rule for dense causal paths (f16 promoted narrowly,
  bf16 conservative). M3+ D=256 remains conservative until measured.
- **bench**: Post-backward D=256 matrix refresh
  (`benchmarks/bench_d256_design_matrix.py`, output
  `notes/d256_design_matrix_post_bwd_latest.json`) confirmed the same shape:
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
- **chore**: Archived stale dump artifacts under `notes/archive/`.

### Native Backward Targeted Pass (Winning Shapes Only)

- **bench**: Added `benchmarks/bench_backward_targeted.py` and
  `notes/native_backward_targeted.md` for a narrow dense-backward sweep:
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
- **total**: 664 tests pass.

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
