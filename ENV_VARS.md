# Environment variable reference

This file describes variables read by the current 2.61.0 source. The registry
of accepted names is `mlx_mfa._knobs.KNOWN_KNOBS`; unknown `MFA_*` names can be
reported by `mlx_mfa._knobs.validate_env(strict=True)`.

## Value rules

Boolean variables accept exactly `0` or `1`. An explicit empty string, `2`,
`true`, `yes` or any other value raises `ValueError` when the variable is read.
Absence means the documented default. This rule is shared by the Python and
C++ readers.

`MFA_KNOB_STRICT` enables import-time validation of known names and values when
set to `1`.
Removed names produce a removal warning rather than being interpreted as a
live control.

Some generator variables are cached. Set them before the first affected
kernel dispatch. Live dispatch controls are read at each decision. When a
cached C++ variable is changed in a benchmark, call the expert
`mlx_mfa._invalidate_env_config()` helper unless the variable is documented as
load-time-only.

## User-facing controls

| Variable | Type/default | Effect |
|---|---|---|
| `MFA_DISABLE_AUTO_HOOKS` | bool, `0` | Do not install transparent MLX hooks during import. |
| `MFA_DISABLE_GNA_NATIVE` | bool, `0` | Send public GNA calls to the non-native fallback path. |
| `MFA_DISABLE_ROPE_NAX` | bool, `0` | Disable the fused RoPE NAX experiment. |
| `MFA_DISABLE_V6_DENSE` | bool, `0` | Disable automatic dense V6 NAX forward selection. |
| `MFA_DISABLE_V6_BACKWARD` | bool, `0` | Disable default D64 V6 backward selection. |
| `MFA_ENABLE_V6_BACKWARD` | bool, `0` | Allow the D128 V6 backward research envelope. |
| `MFA_V6_BWD_SPARSE_NATIVE` | bool, `0` | Request the full-native sparse backward orchestration. |
| `MFA_ENABLE_VARLEN_NAX` | bool, `0` | Enable the narrow beta-3 packed-varlen V6 NAX route. |
| `MFA_ENABLE_CONV3D_PAD_SLICE` | bool, `0` | Enable the measured channel pad/slice Conv3D experiment. |
| `MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE` | bool, `0` | Enable the measured SeedVR2 108x132 spatial pad/slice route. |
| `MFA_GNA_NAX_PRECOMPUTE_RANGE` | bool, `0` | Select the default-off `_pr1` GNA range-precompute variant. |
| `MFA_NAX_SPARSE_DENSITY_CEILING` | float, `0.30` | Further restrict the measured sparse route. It cannot add an unmeasured cell. |
| `MFA_REQUIRE_NAX` | bool, `0` | Raise when NAX was expected but the extension is unavailable. |
| `MFA_SILENCE_NAX_WARNING` | bool, `0` | Suppress the one-time acceleration-unavailable warning. |
| `MFA_PAGED_TRUST_INDICES` | bool, `0` | Skip host value validation for paged metadata; kernel bounds checks remain. |
| `MFA_VARLEN_TRUST_METADATA` | bool, `0` | Skip host value validation for varlen metadata. |
| `MLX_MFA_VERBOSE_DISPATCH` | bool, `0` | Print dispatch decisions. |
| `MLX_MFA_HOOK_TELEMETRY` | enum, `summary` | `off`, `summary` or `verbose` counters for transparent hooks. |
| `MLX_MFA_DISPATCH_TABLE` | path, unset | Load a JSON threshold table; path and file mtime key the loader cache. |

The varlen opt-in accepts only B=1, D128, f16/bf16, equal Q/K segment
boundaries, GQA factor 2/4/8, 20 or 24 segments and total length 35018-35250.
Its BQ32/BK32/WM2 configuration travels through source generation, cache
identity and host dispatch as one value set.

The spatial Conv3D opt-in accepts only fp16 B=1, T=4 or 5, HxW=108x132,
C-in=C-out=512, 3x3x3, stride/dilation 1, groups 1, no flip, temporal padding
0 and spatial padding 1. Other shapes keep the original MLX function.

## Dispatch overrides

These variables exist for diagnostics or compatibility. They can make an
automatic choice slower and should not be used as performance claims.

| Variable | Type | Purpose |
|---|---|---|
| `MFA_FORCE_SDPA_ROUTE`, `MFA_DISABLE_SDPA_ROUTE` | bool | Force or bypass the strategic MLX-SDPA decision. |
| `MFA_FORCE_V2`, `MFA_DISABLE_V2` | bool | Select or exclude the V2 STEEL family where supported. |
| `MFA_ENABLE_V3`, `MFA_DISABLE_V3` | bool | Enable or disable the compiled V3 route. |
| `MFA_FORCE_SPLITK` | bool | Force split-K selection. |
| `MFA_FORCE_D256_PATH`, `MFA_FORCE_D512_PATH` | bool | Exercise dimension-specific expert paths; D512 public behavior remains delegation. |
| `MFA_FORCE_SAGE_DECODE` | bool | Force the Sage decode experiment. |
| `MFA_DISABLE_TOPK_NAX`, `MFA_DISABLE_TOPK_BISECT` | bool | Disable Top-K NAX or its bisection path. |
| `MFA_DISABLE_TQ_DECODE_SDPA` | bool | Disable the TurboQuant decode SDPA bridge. |
| `MFA_ENABLE_MACOS27_ROUTING` | bool | Enable additional beta OS routing experiments. |
| `MFA_LCSA_KERNEL_VERSION` | string | Sparse expert selector; `v1` aliases `scalar_fallback`, `v2` aliases `v6nax_sparse`. |
| `MFA_UNSAFE_D128_SPARSE` | bool | Expert-only unsafe legacy sparse override. |

## Hook and Conv3D controls

| Variable | Type/default | Purpose |
|---|---|---|
| `MFA_DISABLE_CONV3D_MPP` | bool, `0` | Disable the MPP Conv3D route. |
| `MFA_CONV_NAX_NO_FAST_PATH` | bool, `0` | Disable the NAX fast path. |
| `MFA_CONV_NAX_USE_PYTHON_LEGACY` | bool, `0` | Use the legacy Python implementation for comparison. |
| `MFA_CONV3D_PAD_RATIO_MAX` | float | Maximum expansion ratio for channel pad/slice. |
| `MFA_HOOK_VERBOSE` | bool, `0` | Emit additional hook diagnostics. |

## Expert tile and source controls

These names are accepted by the current registry. They are intended for
kernel development, source dumps and controlled benchmarks, not application
configuration.

### Dense and legacy STEEL

`MFA_DEBUG_SHADERS`, `MFA_DISABLE_ASYNC`, `MFA_NO_PADDING`,
`MFA_SPLITK_MAX_N_D`, `MFA_V2_BD_HALF_D512`, `MFA_V2_BQ64`,
`MFA_V2_FORCE_BK`, `MFA_V2_FORCE_BK_D256`, `MFA_V2_FORCE_BK_D512`,
`MFA_V2_FORCE_BQ_D512`, `MFA_V3_FORCE_BK_`, `MFA_V3_FORCE_BK_D64`,
`MFA_V3_FORCE_BK_D128`, `MFA_STEEL_MSL`, `MFA_FORCE_GEN`.

`MFA_NO_PADDING` is frozen at first read because padding mode is not a
mid-process dispatch switch. `MFA_STEEL_MSL` participates in the shader cache
key, so replacing source in one process compiles a distinct pipeline.

### V6 NAX forward and varlen

`MFA_V6_BLOCK_C`, `MFA_V6_BLOCK_D`, `MFA_V6_BLOCK_R`,
`MFA_V6_BNHD_LEGACY`, `MFA_V6_BYPASS_TGP`, `MFA_V6_DENSE_MIN_N`,
`MFA_V6_DUMP_SOURCE`, `MFA_V6_EXEC_SG`, `MFA_V6_FORCE_DYNAMIC_K`,
`MFA_V6_MAX_THREADS`, `MFA_V6_NAX_BK`, `MFA_V6_NAX_BQ`,
`MFA_V6_NAX_D_SUBTILE`, `MFA_V6_NAX_SINGLE_OTILE`, `MFA_V6_NAX_WM`,
`MFA_V6_RELAXED_PRECISION`, `MFA_V6_SENTINEL_FILL`, `MFA_V6_UNROLL_MODE`,
`MFA_V6_USE_NAX`, `MFA_V6_USE_V34`, `MFA_V6_V34_BK`, `MFA_V6_V34_BQ`,
`MFA_V6_V34_WM`, `MFA_V6_VARLEN_DUMP_PATH`.

### V6 backward

`MFA_V6_BWD_KERNEL`, `MFA_V6BWD_BK`, `MFA_V6BWD_BQ`, `MFA_V6BWD_WM`,
`MFA_V6BWD_DUMP_SOURCE`, `MFA_V6BWD_USE_FUSED`,
`MFA_V6BWDF_BK`, `MFA_V6BWDF_BQ`, `MFA_V6BWDF_WM`,
`MFA_V6BWDF_DUMP_PATH`, `MFA_V6BWDF_DUMP_SOURCE`,
`MFA_V6BWDK_BK`, `MFA_V6BWDK_BQ`, `MFA_V6BWDK_WM`,
`MFA_V6BWDV_BK`, `MFA_V6BWDV_BQ`, `MFA_V6BWDV_WM`,
`MFA_V6BWDKV_BK`, `MFA_V6BWDKV_BQ`, `MFA_V6BWDKV_WM`.

### GNA, linear and quantized matmul

`MFA_GNA_NAX_BQ`, `MFA_GNA_NAX_BK`, `MFA_GNA_NAX_WM`,
`MFA_GNA_NAX_SWIZZLE_LOG`, `MFA_GNA_NAX_DUMP_PATH`,
`MFA_FFN_NAX_BM`, `MFA_FFN_NAX_BN`, `MFA_FFN_NAX_BK`,
`MFA_FFN_NAX_WM`, `MFA_FFN_NAX_WN`, `MFA_QMM_NAX_BM`,
`MFA_QMM_NAX_BN`, `MFA_QMM_NAX_BK`, `MFA_QMM_NAX_WM`,
`MFA_QMM_NAX_WN`, `MFA_IR_INVESTIGATE`.

## Deprecated V34 aliases

Thirty V34-era names remain one-shot-warning aliases for canonical V6 names.
The Python and C++ alias maps are executable-locked together. Examples:

- `MFA_ENABLE_V34_BACKWARD` -> `MFA_ENABLE_V6_BACKWARD`
- `MFA_DISABLE_V34_BACKWARD` -> `MFA_DISABLE_V6_BACKWARD`
- `MFA_V34_BWD_KERNEL` -> `MFA_V6_BWD_KERNEL`
- `MFA_V34_BWD_SPARSE_NATIVE` -> `MFA_V6_BWD_SPARSE_NATIVE`
- `MFA_V34_DUMP_SOURCE` -> `MFA_V6_DUMP_SOURCE`
- `MFA_V34BWDF_*`, `MFA_V34BWDK_*`, `MFA_V34BWDV_*`,
  `MFA_V34BWDKV_*` -> matching `MFA_V6...` names

When both names are set, the canonical V6 name wins.

## Removed names

The following are not live controls: `MFA_BD_FRAGS`, `MFA_BD_TILE`,
`MFA_D_CHUNKS`, `MFA_ENABLE_V34_D128`, `MFA_ENABLE_V4`, `MFA_ENABLE_V5`,
`MFA_ENABLE_V6_D128`, `MFA_FORCE_NATIVE_BWD`, `MFA_GQA_DECODE_CIDER`,
`MFA_TOPK_BISECT`, `MFA_TOPK_STREAM_V5`, `MFA_V34BWD`,
`MFA_V5_FORCE_BD_TILE`, `MFA_V5_FORCE_BK`, `MFA_V5_FORCE_BQ`,
`MFA_V5_FORCE_WM`, `MFA_V6`, and `MFA_V6BWD`.

V4 and V5 forward sources are absent from the build. Their old enable and tile
variables are retained only in the removed-name validator.

## Build-time probe switch

`MFA_BUILD_PROBES` is a CMake option, not an env var read by the
runtime. It defaults to `OFF`. Enabling it adds V6 bring-up and int8/fp8
microbenchmark symbols to `_ext`; production builds do not expose them.
