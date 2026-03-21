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

## Shader Generation (cold path, not cached)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MFA_NO_PADDING` | bool | false | Disable threadgroup memory padding (causes NaN in 45/594 tests) |
| `MFA_IR_INVESTIGATE` | bool | false | Dump Metal IR during shader compilation |
| `MFA_DISABLE_ASYNC` | bool | false | Disable precompiled async metallib lookup |

## Calibration (dynamic keys, not cached)

| Variable pattern | Type | Description |
|-----------------|------|-------------|
| `MFA_SPLITK_MAX_N_D{D}_C{0\|1}_A{0\|1}_W{0\|1}` | int | Per-config max N for split-K dispatch |
