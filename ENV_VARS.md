# MFA Environment Variables

All `MFA_*` env vars controlling dispatch and kernel configuration.
Source of truth: `csrc/mfa_env.hpp` (cached values) + live reads in `mfa_attention.cpp`.
M5/V6NAX tuning knobs are documented in `.doc-archive/docs/v6-nax/env-vars.md`.

## Dispatch Gates

| Variable | Type | Default | Kernel | Cached | Description |
|----------|------|---------|--------|:------:|-------------|
| `MFA_ENABLE_V3` | bool | (set) | V3 | No | Bypass V3 shape guard (backward compat) |
| `MFA_DISABLE_V3` | bool | unset | V3 | No | Force-disable V3 (fall through to V2) |
| `MFA_ENABLE_V4` | bool | unset | V4 | No | Opt-in V4 (M3+ only, experimental) |
| `MFA_ENABLE_V5` | bool | unset | V5 | No | Opt-in V5 (experimental, D-blocked) |
| `MFA_DISABLE_V2` | bool | unset | V2 | No | Disable all V2 paths (split-K + single-pass + D-split) |
| `MFA_FORCE_V2` | bool | unset | V2 | No | Bypass M3+ V1 preference (force V2 single-pass) |
| `MFA_FORCE_SPLITK` | tri | unset | V2-SK | No | -1=heuristic(unset), 0=disable, 1=force |

## Architecture Override

| Variable | Type | Default | Cached | Description |
|----------|------|---------|:------:|-------------|
| `MFA_FORCE_GEN` | int | 0 (auto) | Yes | Override GPU architecture gen (13=M1, 14=M2, 15=M3, 16=M4, 17=M5) |

## V2 Config Overrides

| Variable | Type | Default | Cached | Description |
|----------|------|---------|:------:|-------------|
| `MFA_V2_FORCE_BK` | int | 0 (auto) | Yes | Override BK for D=64/128 (valid: 32, 64) |
| `MFA_V2_BQ64` | bool | false | Yes | Use BQ=64 WM=8 (Option B config) |
| `MFA_V2_FORCE_BK_D256` | int | 0 (auto) | Yes | Override BK for D=256 D-split (valid: 8,16,32,64) |
| `MFA_V2_FORCE_BK_D512` | int | 0 (auto) | Yes | Override BK for D=512 D-split |
| `MFA_V2_FORCE_BQ_D512` | int | 0 (auto) | Yes | Override BQ for D=512 (valid: 16,32,64) |
| `MFA_V2_BD_HALF_D512` | int | 0 (auto) | Yes | Override BD_HALF for D=512 (valid: 32,64,128) |

## V3 Config Overrides

| Variable | Type | Default | Cached | Description |
|----------|------|---------|:------:|-------------|
| `MFA_V3_FORCE_BK_D64` | int | 0 (auto) | Yes | Override V3 BK for D=64 (valid: 8,16,32,64) |
| `MFA_V3_FORCE_BK_D128` | int | 0 (auto) | Yes | Override V3 BK for D=128 (valid: 8,16,32,64) |

## V5 Config Overrides

| Variable | Type | Default | Cached | Description |
|----------|------|---------|:------:|-------------|
| `MFA_V5_FORCE_BK` | int | 0 (auto) | Yes | Override V5 BK (D=64: 32, D=128: 32) |
| `MFA_V5_FORCE_BD_TILE` | int | 0 (auto) | Yes | Override V5 BD_tile (D=64: 32, D=128: 64) |
| `MFA_V5_FORCE_BQ` | int | 0 (auto) | Yes | Override V5 BQ (default: 32) |
| `MFA_V5_FORCE_WM` | int | 0 (auto) | Yes | Override V5 WM (default: 4) |

## Dispatch Policy (Python-side, live-read)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MFA_FORCE_SDPA_ROUTE` | bool | unset | Force SDPA routing on M5+ NAX regardless of shape/dtype (debug/benchmark override). *(documented repo review 2026-05)* |
| `MFA_DISABLE_SDPA_ROUTE` | bool | unset | Disable the M5+ SDPA route; dispatch falls through to M3+/M1-M2 thresholds. *(documented repo review 2026-05)* |
| `MFA_FORCE_D256_PATH` | str | unset | Force D=256 auto route: `1`/`mfa` → MFA, `0`/`sdpa` → SDPA |
| `MFA_FORCE_D512_PATH` | str | unset | Force D=512 auto route: `1`/`mfa` → MFA, `0`/`sdpa` → SDPA |
| ~~`MFA_FORCE_NATIVE_BWD`~~ | — | — | **REMOVED v2.56.0.** Deprecated v2.50.0 ("target removal v2.51+"); removed after the deprecation cycle completed (5 minor versions of `DeprecationWarning`) — forced STEEL backward was dominated at every cell (V6NAX at D=64, SDPA-vjp at D=128; sprint-C Track 2). The env var is now inert. The STEEL backward kernel itself is retained (keep-all-paths) and reachable via the direct `_ext.mfa_steel_backward` binding; routing follows the benchmark-backed policy table. |
| `MFA_FORCE_SAGE_DECODE` | str | unset | Force sage decode routing: `1` → sage, `0` → standard FA |
| `MFA_LCSA_KERNEL_VERSION` | str | unset (shape-aware) | Sparse attention kernel version override. **v2.36.1**: when unset, `decide_auto_version()` picks V2 for `qL × kL × D ≥ 2.15e9` (validated under canonical-protocol) and V1 below. `=v1` forces V1 universally; `=v2` forces V2 universally. Unrecognised values fall through to shape-aware default. |
| `MFA_ENABLE_V6_BACKWARD` | bool | unset (off) | **v2.37.0** (updated v2.51.0): opt-in for **D=128 only** — D=64 (causal + non-causal) is **default-on since v2.51.0** and needs no env var. Enables V6NAX NAX-direct backward kernels via `flash_attention()` autograd on M5+ eligible shapes (FP16/BF16, qL ≥ 2048). Requires default scale (1/sqrt(D)) — custom scale falls back per repo-review 2026-05 gate. **Path-dependent effect (audit B3/C2, verified by per-gradient byteΔ vs SDPA-vjp): DENSE D=128 backward → FULL-native dQ/dK/dV; SPARSE hybrid backward → NATIVE-dV ONLY (dQ/dK stay SDPA-vjp); full-native sparse needs `MFA_V6_BWD_SPARSE_NATIVE=1` + `bt≥64`.** |
| `MFA_DISABLE_V6_BACKWARD` | bool | unset | **Phase II-0 / v2.51.0**: opt-OUT of the default-on V6NAX backward D=64 (causal + non-causal) promotion (1.7-2.7x vs SDPA-vjp, qL>=2048, fp16/bf16, M5+; incl. GQA/MQA post shape-fix).  Set =1 to restore SDPA-vjp at that cell. |
| `MFA_V6BWD_USE_FUSED` | bool | unset (split) | **v2.37.0**: with V6NAX backward enabled, choose the fused WM=1 dK/dV kernel (single dispatch) instead of the WM=4 multi-SG split (two dispatches).  Default off (multi-SG split, 1.7-2× faster).  Set =1 for fallback / benchmarking. |
| `MFA_V6BWD_WM` | int | 4 | **v2.37.0**: WM for the multi-SG dK + dV split kernels.  Default 4 (Q-row partition with each SG owning 16 Q-rows).  Override for autoresearch sweeps. |
| `MFA_V6BWDV_BQ`, `MFA_V6BWDV_BK`, `MFA_V6BWDV_WM` | int | 64, 32, 4 | Per-kernel tile overrides for dV kernel (v2.37.0).  Researchers. |
| `MFA_V6BWDK_BQ`, `MFA_V6BWDK_BK`, `MFA_V6BWDK_WM` | int | 64, 32, 4 | Per-kernel tile overrides for dK kernel (v2.37.0).  Researchers. |
| `MFA_V6_BWD_KERNEL` | str | `auto` | **v2.39.0/v2.40.0-internal**: V6NAX backward kernel mode selection.  `auto` → D=64 fused, D=128 split (per Sprint B outcome γ).  `fused` → forced fused (D ∈ {64, 128}; D=128 may regress 3-7%).  `split` → forced split-dKdV (works for any D ∈ {64, 128}).  `legacy_fused` → WM=1 fused (escape hatch for one release).  Default `auto` is empirically optimal. |
| `MFA_V6_BWD_SPARSE_NATIVE` | bool | unset | **v2.50 Prompt 5d**: opt-in to full-native V6NAX backward sparse kernels (4 sparse kernels: dQ + dV + dK split + fused dKdV) instead of Prompt 5c hybrid orchestrator.  Default off (hybrid is production per Pattern #6 empirical bench — V6NAX NAX backward slower than Apple SDPA NAX on M5+).  Set `=1` for research/benchmark access.  See `.doc-archive/docs/v50/section-a-v3-empirical-verification.md`. |
| `MFA_TOPK_BISECT` | bool | unset | **GHOST (campaign 2026-06 Track 0)**: not read by ANY code path — setting it is a no-op.  Bisection is the AUTO default; the live opt-out is `MFA_DISABLE_TOPK_BISECT`.  Row retained for historical reference only. |
| `MFA_DISABLE_TOPK_BISECT` | bool | unset | **v2.50 Prompt 5c**: opt-out of Top-K bisection kernel AUTO default; falls back to Phase 3a legacy `mx.topk` path.  Use for exact-mx.topk-semantics or debugging. |
| `MFA_DISABLE_TOPK_NAX` | bool | unset | Disable Top-K NAX dispatch entirely; falls back to Python reference (very slow at scale, for correctness comparison). |
| `MFA_DISABLE_ROPE_NAX` | bool | unset | **Sprint 2**: opt-out of `mx.fast.rope` dispatch path in `flash_attention_rope_unified`; falls back to STEEL host-side RoPE. |
| `MFA_DISABLE_TQ_DECODE_SDPA` | bool | unset | **v2.51.0**: opt-out of the default TurboQuant paged-decode (N_q=1) gather/dequant + Apple SDPA path (6.0-14.4x); falls back to the fused TQ kernel path. |
| `MFA_DISABLE_AUTO_HOOKS` | bool | unset | Disable auto-hook installation at `import mlx_mfa` (`_auto_hooks.py::install_hooks()`); no `mx.*` surfaces are patched. |
| `MLX_MFA_VERBOSE_DISPATCH` | bool | false | Print dispatch decisions to stderr |
| `MLX_MFA_DISPATCH_TABLE` | path | unset | JSON file with custom per-config dispatch thresholds |
| `MLX_MFA_HOOK_TELEMETRY` | str | `summary` | **v2.50.1 Prompt 5g Phase C**: hook execution/fallback telemetry mode (Pattern #8 prevention).  Values: `off` (zero overhead, no counters), `summary` (default; per-hook executed/fallback counters readable via `mlx_mfa.get_hook_stats()`), `verbose` (summary + `UserWarning` per fallback for active debugging).  See `docs/reference/HOOK_TELEMETRY.md`. |

## Shader Generation (cold path, not cached)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MFA_DEBUG_SHADERS` | bool | false | Dump generated Metal shader source to stderr during compilation |
| `MFA_NO_PADDING` | bool | false | Disable threadgroup memory padding (causes NaN in 45/594 tests) |
| `MFA_IR_INVESTIGATE` | bool | false | Dump Metal IR during shader compilation |
| `MFA_DISABLE_ASYNC` | bool | false | Disable precompiled async metallib lookup |
| `MFA_DISABLE_GNA_NATIVE` | bool | false | Disable native GNA kernel; fall back to sparse path |

## Conv NAX (Python + C++ side)

*(documented repo review 2026-05)*

| Variable | Type | Default | Description |
|---|---|---|---|
| `MFA_CONV_NAX_NO_FAST_PATH` | bool | unset | Bypass the 1×1×1 fast path in Conv3D NAX (forces the general path; used by perf tests). |
| `MFA_CONV_NAX_USE_PYTHON_LEGACY` | bool | unset | Route Conv3D NAX through the Phase 1.x legacy Python implementation (debug). |
| `MFA_DISABLE_CONV3D_MPP` | bool | unset | **v2.50.2/v2.51.0**: opt-out of the default MPP `convolution2d` conv3d path (fp16 2.3-2.5x, bf16 1.4-2.7x); falls back to the legacy materialized-im2col path (fp16-only there, KD-7). |
| `MFA_REQUIRE_MSL4` | (not an env var) | — | **Corrected (campaign 2026-06 Track 0)**: this is a SOURCE-STRING SENTINEL (`// MFA_REQUIRE_MSL4` comment in generated Metal), detected by `shader_cache.mm` to select MTLLanguageVersion4_0.  It is never read from the environment. |

## Calibration (dynamic keys, not cached)

| Variable pattern | Type | Description |
|-----------------|------|-------------|
| `MFA_SPLITK_MAX_N_D{D}_C{0\|1}_A{0\|1}_W{0\|1}` | int | Per-config max N for split-K dispatch |

## V6NAX backward tile overrides + diagnostics (documented campaign 2026-06 Track 0)

All apply when the V6NAX backward path is active (default-on for D=64
since v2.51.0; D=128 via `MFA_ENABLE_V6_BACKWARD=1`); expert/bench knobs.  Values flow
into the pipeline cache keys (live; Sprint A verified key completeness).

| Variable | Type | Default | Description |
|---|---|---|---|
| `MFA_V6BWD_BQ` / `MFA_V6BWD_BK` | int | auto | dQ kernel tile override (pairs with documented `MFA_V6BWD_WM`). |
| `MFA_V6BWDKV_BQ` / `MFA_V6BWDKV_BK` / `MFA_V6BWDKV_WM` | int | auto | Legacy fused dK+dV kernel tile overrides. |
| `MFA_V6BWDF_BQ` / `MFA_V6BWDF_BK` / `MFA_V6BWDF_WM` | int | auto | Fused dKdV kernel tile overrides. |
| `MFA_V6_SENTINEL_FILL` | bool | unset | Debug: pre-fill V6 output/LSE buffers with sNaN before dispatch (dispatch-routing forensics). |
| `MFA_V6_DUMP_SOURCE` / `MFA_V6BWD_DUMP_SOURCE` | bool | unset | Debug: dump generated V6NAX fwd/bwd Metal source to stderr on pipeline-cache miss. |
| `MFA_V6BWDF_DUMP_SOURCE` / `MFA_V6BWDF_DUMP_PATH` | bool / path | unset | Debug: dump the generated fused-dKdV MSL source (to stderr, or to the file given by `MFA_V6BWDF_DUMP_PATH`). |

Interaction notes (campaign 2026-06 Track 0):
- **`MFA_V6_BLOCK_R`, `MFA_V6_BLOCK_C`, `MFA_V6_EXEC_SG` are VESTIGIAL on the V6 NAX path**
  (no effect). They fed the simdgroup descriptor path that audit **F-3 removed**; the
  dispatched NAX kernel's tile is governed entirely by **`MFA_V6_NAX_BQ` / `MFA_V6_NAX_BK` /
  `MFA_V6_NAX_WM`** — use those to tune it. Confirmed by runtime fingerprint
  (`research/nax-autotune-m5` Phase 0); setting any of the three emits a one-shot stderr notice.
- `MFA_V6_BYPASS_TGP=0` is a no-op when single-Otile mode auto-fires (forced true).
- `MFA_V6_UNROLL_MODE`, `MFA_V6_RELAXED_PRECISION`, `MFA_V6_BLOCK_D`, `MFA_V6_FORCE_DYNAMIC_K`,
  `MFA_V6_MAX_THREADS`: measured NO-OP for the NAX forward (Phase-0 knob-map); they affect only
  the vestigial/legacy source paths.
