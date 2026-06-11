# MFA Environment Variables

All `MFA_*` env vars controlling dispatch and kernel configuration.
Source of truth: `csrc/mfa_env.hpp` (cached values) + live reads in `mfa_attention.cpp`.

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
| `MFA_FORCE_NATIVE_BWD` | str | unset | Force native backward: `1` → native kernel, `0` → SDPA VJP |
| `MFA_FORCE_SAGE_DECODE` | str | unset | Force sage decode routing: `1` → sage, `0` → standard FA |
| `MFA_LCSA_KERNEL_VERSION` | str | unset (shape-aware) | Sparse attention kernel version override. **v2.36.1**: when unset, `decide_auto_version()` picks V2 for `qL × kL × D ≥ 2.15e9` (validated under canonical-protocol) and V1 below. `=v1` forces V1 universally; `=v2` forces V2 universally. Unrecognised values fall through to shape-aware default. |
| `MFA_ENABLE_V34_BACKWARD` | bool | unset (off) | **v2.37.0** (updated v2.50): opt in to V34 NAX-direct backward kernels via `flash_attention()` autograd on M5+ eligible shapes (D ∈ {64, 128}, FP16/BF16, qL ≥ 2048; **causal AND non-causal eligible** as of v2.50 Prompt 4 Section B — the "no causal" restriction is obsolete). Default off (SDPA-vjp fallback). Requires default scale (1/sqrt(D)) — custom scale falls back per repo-review 2026-05 gate. |
| `MFA_V34BWD_USE_FUSED` | bool | unset (split) | **v2.37.0**: with V34 backward enabled, choose the fused WM=1 dK/dV kernel (single dispatch) instead of the WM=4 multi-SG split (two dispatches).  Default off (multi-SG split, 1.7-2× faster).  Set =1 for fallback / benchmarking. |
| `MFA_V34BWD_WM` | int | 4 | **v2.37.0**: WM for the multi-SG dK + dV split kernels.  Default 4 (Q-row partition with each SG owning 16 Q-rows).  Override for autoresearch sweeps. |
| `MFA_V34BWDV_BQ`, `MFA_V34BWDV_BK`, `MFA_V34BWDV_WM` | int | 64, 32, 4 | Per-kernel tile overrides for dV kernel (v2.37.0).  Researchers. |
| `MFA_V34BWDK_BQ`, `MFA_V34BWDK_BK`, `MFA_V34BWDK_WM` | int | 64, 32, 4 | Per-kernel tile overrides for dK kernel (v2.37.0).  Researchers. |
| `MFA_V34_BWD_KERNEL` | str | `auto` | **v2.39.0/v2.40.0-internal**: V34 backward kernel mode selection.  `auto` → D=64 fused, D=128 split (per Sprint B outcome γ).  `fused` → forced fused (D ∈ {64, 128}; D=128 may regress 3-7%).  `split` → forced split-dKdV (works for any D ∈ {64, 128}).  `legacy_fused` → WM=1 fused (escape hatch for one release).  Default `auto` is empirically optimal. |
| `MFA_V34_BWD_SPARSE_NATIVE` | bool | unset | **v2.50 Prompt 5d**: opt-in to full-native V34 backward sparse kernels (4 sparse kernels: dQ + dV + dK split + fused dKdV) instead of Prompt 5c hybrid orchestrator.  Default off (hybrid is production per Pattern #6 empirical bench — V34 NAX backward slower than Apple SDPA NAX on M5+).  Set `=1` for research/benchmark access.  See `docs/v50/section-a-v3-empirical-verification.md`. |
| `MFA_TOPK_BISECT` | bool | unset (deprecated) | **v2.50 Prompt 5b** opt-in for Top-K bisection kernel.  **Deprecated v2.50 Prompt 5c**: bisection promoted to AUTO default; this env var is now redundant and ignored.  Preserved for back-compat. |
| `MFA_DISABLE_TOPK_BISECT` | bool | unset | **v2.50 Prompt 5c**: opt-out of Top-K bisection kernel AUTO default; falls back to Phase 3a legacy `mx.topk` path.  Use for exact-mx.topk-semantics or debugging. |
| `MFA_DISABLE_TOPK_NAX` | bool | unset | Disable Top-K NAX dispatch entirely; falls back to Python reference (very slow at scale, for correctness comparison). |
| `MFA_DISABLE_ROPE_NAX` | bool | unset | **Sprint 2**: opt-out of `mx.fast.rope` dispatch path in `flash_attention_rope_unified`; falls back to STEEL host-side RoPE. |
| `MLX_MFA_VERBOSE_DISPATCH` | bool | false | Print dispatch decisions to stderr |
| `MLX_MFA_DISPATCH_TABLE` | path | unset | JSON file with custom per-config dispatch thresholds |
| `MLX_MFA_HOOK_TELEMETRY` | str | `summary` | **v2.50.1 Prompt 5g Phase C**: hook execution/fallback telemetry mode (Pattern #8 prevention).  Values: `off` (zero overhead, no counters), `summary` (default; per-hook executed/fallback counters readable via `mlx_mfa.get_hook_stats()`), `verbose` (summary + `UserWarning` per fallback for active debugging).  See `docs/HOOK_TELEMETRY.md`. |

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
| `MFA_REQUIRE_MSL4` | bool | unset | C++-side gate requiring Metal Shading Language 4 support for V34 probe / NAX compile paths. |

## Calibration (dynamic keys, not cached)

| Variable pattern | Type | Description |
|-----------------|------|-------------|
| `MFA_SPLITK_MAX_N_D{D}_C{0\|1}_A{0\|1}_W{0\|1}` | int | Per-config max N for split-K dispatch |
