# MFA Environment Variables

All `MFA_*` env vars controlling dispatch and kernel configuration.
Source of truth: `csrc/mfa_env.hpp` (cached values) + live reads in `mfa_attention.cpp`.
M5/V6NAX tuning knobs are documented in `.doc-archive/docs/v6-nax/env-vars.md`.

Boolean knobs accept exactly `0` or `1`; absence selects the documented
default. Any other explicit value, including an empty string, raises at first
access. `validate_env(strict=True)` also reports invalid values of known knobs.

## Dispatch Gates

| Variable | Type | Default | Kernel | Cached | Description |
|----------|------|---------|--------|:------:|-------------|
| `MFA_ENABLE_V3` | bool | (set) | V3 | No | Bypass V3 shape guard (backward compat) |
| `MFA_DISABLE_V3` | bool | unset | V3 | No | Force-disable V3 (fall through to V2) |
| `MFA_DISABLE_V2` | bool | unset | V2 | No | Disable all V2 paths (split-K + single-pass + D-split) |
| `MFA_FORCE_V2` | bool | unset | V2 | No | Bypass M3+ V1 preference (force V2 single-pass) |
| `MFA_FORCE_SPLITK` | tri | unset | V2-SK | No | -1=heuristic(unset), 0=disable, 1=force |
| `MFA_DISABLE_V6_DENSE` | bool | unset | V6-NAX | No | Opt out of the M5 dense D=128 NAX matmul2d forward route (stay SDPA) |
| `MFA_V6_DENSE_MIN_N` | int | 2048 | V6-NAX | No | Min N for the dense D=128 NAX forward route (0 = force all-N NAX) |
| `MFA_NAX_SPARSE_DENSITY_CEILING` | float | 0.30 | Sparse | No | Further-restrict-only density cap inside the measured β3 V6NAX-sparse gate; it cannot widen the canonical shape/dtype envelope |
| `MFA_PAGED_TRUST_INDICES` | bool | unset | Paged | No | Skip the host-side `block_table`/`seq_lens` value-range check on the paged decode hot path (eliminates a per-call device sync; the kernel still bounds-guards every physical block, so OOB stays memory-safe but no longer raises early). For high-frequency loops whose metadata is already valid. |
| `MFA_VARLEN_TRUST_METADATA` | bool | unset | Varlen | No | Skip the host-side `cu_seqlens`/`tile_offsets` **value** validation (monotonic, `[0]==0`, final-sum) on the varlen forwards (eliminates a per-call device sync that reads the small metadata arrays). Mirrors `MFA_PAGED_TRUST_INDICES`; for callers whose varlen metadata is already valid. Default (unset) validates and raises on malformed metadata to prevent silent finite-wrong output. |
| `MFA_HOOK_VERBOSE` | bool | unset | Auto-hooks | No | Verbose logging of the `mx.*` auto-hook install/dispatch (debug) |

## Architecture Override

| Variable | Type | Default | Cached | Description |
|----------|------|---------|:------:|-------------|
| `MFA_FORCE_GEN` | int | 0 (auto) | Yes | Override GPU architecture gen (13=M1, 14=M2, 15=M3, 16=M4, 17=M5) |

## OS-Aware Routing (macOS 26 vs 27)

The macOS-27 / Metal-4.1 path **auto-detects and self-enables** via a **functional capability
probe** — it compiles AND verifies a Metal-4.1 kernel on your toolchain (a version string is not
enough: "macOS 27" does not guarantee a working 4.1 compiler). **No configuration required.** If the
toolchain does not functionally support 4.1 (e.g. a beta whose compiler hasn't caught up, or that
miscompiles), the library transparently uses the validated macOS-26 path (fail-safe). The probe is
lazy (first-use, cached), never runs on macOS ≤26, and never on import. `get_device_info()` exposes
`macos_major` / `macos_minor`. The activation path is **byte-identical today** (no 26↔27 behavioral
divergence yet — the seam is forward-looking); the **sparse fallback stays engaged regardless** (the
STEEL `(long)p->NK` bug is not fixed by 4.1). See `devnotes/macos27_functional_gate.md`.

| Variable | Type | Default | Cached | Description |
|----------|------|---------|:------:|-------------|
| `MFA_ENABLE_MACOS27_ROUTING` | bool | unset | Yes | **Optional override** (default activation is the functional probe, no config). `=1` force-ON (test the macOS-27 path on a toolchain the probe would reject); `=0` force-OFF (pin the validated macOS-26 path). Unset ⇒ the functional Metal-4.1 probe decides. |
| `MFA_UNSAFE_D128_SPARSE` | bool | unset | No | **DIAGNOSTIC-ONLY — DANGER.** Opens the C++ D=128 sparse OOB guard so the known-incorrect raw STEEL sparse kernel can be run for OS re-characterization (e.g. re-verifying the `(long)p->NK` mis-read under a new Metal compiler). Default off ⇒ the guard raises (shipping behavior byte-identical). **NEVER enable in production** — the D=128 sparse kernel is out-of-bounds + non-deterministic on M3+. |

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

## V4 / V5 STEEL forward variants — REMOVED (v2.61.0)

The V4 and V5 experimental STEEL forward kernels were **removed from the build**
(Lot-2): the `mfa_steel_fwd_v{4,5}.cpp` sources are dropped from CMake and the
dispatch + env gates are gone.  These knobs **no longer exist** and have no
effect: `MFA_ENABLE_V4`, `MFA_ENABLE_V5`, `MFA_V5_FORCE_BK`,
`MFA_V5_FORCE_BD_TILE`, `MFA_V5_FORCE_BQ`, `MFA_V5_FORCE_WM`.  The compiled +
routed STEEL forwards are **V1 / V2 / V3 / V6_NAX** (source recoverable via the
`archive/v4-v5-prototypes` tag).  V5 was M5-validated before removal and showed
no advantage (3.1–4.4× slower than the routed NAX/SDPA default).

## Dispatch Policy (Python-side, live-read)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MFA_FORCE_SDPA_ROUTE` | bool | unset | Force SDPA routing on M5+ NAX regardless of shape/dtype (debug/benchmark override). *(documented repo review 2026-05)* |
| `MFA_DISABLE_SDPA_ROUTE` | bool | unset | Disable the M5+ SDPA route; dispatch falls through to M3+/M1-M2 thresholds. *(documented repo review 2026-05)* |
| `MFA_FORCE_D256_PATH` | bool | unset | Force D=256 auto route: `1` → MFA, `0` → SDPA. Boolean values are strictly `0` or `1`. |
| `MFA_FORCE_D512_PATH` | bool | unset | Force D=512 auto route: `1` → MFA, `0` → SDPA. Boolean values are strictly `0` or `1`. |
| ~~`MFA_FORCE_NATIVE_BWD`~~ | — | — | **REMOVED v2.56.0.** Deprecated v2.50.0 ("target removal v2.51+"); removed after the deprecation cycle completed (5 minor versions of `DeprecationWarning`) — forced STEEL backward was dominated at every cell (V6NAX at D=64, SDPA-vjp at D=128; sprint-C Track 2). The env var is now inert. The STEEL backward kernel itself is retained (keep-all-paths) and reachable via the direct `_ext.mfa_steel_backward` binding; routing follows the benchmark-backed policy table. |
| `MFA_FORCE_SAGE_DECODE` | str | unset | Force sage decode routing: `1` → sage, `0` → standard FA |
| `MFA_LCSA_KERNEL_VERSION` | str | unset (shape-aware) | Sparse attention kernel version override. **Phase F (M-07)**: when unset, `decide_auto_version()` routes **D∈{64,128} → V2 (matmul2d) always** — the old `qL × kL × D ≥ 2.15e9` work-product gate is **RETIRED** (V1-scalar was never fastest; V1 kept only as the genuine fallback for D∉{64,128}). `=v1` forces V1 universally; `=v2` forces V2 universally. Unrecognised values fall through to shape-aware default. |
| `MFA_ENABLE_V6_BACKWARD` | bool | unset (off) | **v2.37.0** (updated v2.51.0): opt-in for **D=128 only** — D=64 (causal + non-causal) is **default-on since v2.51.0** and needs no env var. Enables V6NAX NAX-direct backward kernels via `flash_attention()` autograd on M5+ eligible shapes (FP16/BF16, qL ≥ 2048). Requires default scale (1/sqrt(D)) — custom scale falls back per repo-review 2026-05 gate. **Path-dependent effect (audit B3/C2, verified by per-gradient byteΔ vs SDPA-vjp): DENSE D=128 backward → FULL-native dQ/dK/dV; SPARSE hybrid backward → NATIVE-dV ONLY (dQ/dK stay SDPA-vjp); full-native sparse needs `MFA_V6_BWD_SPARSE_NATIVE=1` + `bt≥64`.** |
| `MFA_DISABLE_V6_BACKWARD` | bool | unset | Opt out of the default-on V6NAX backward D=64 route. Fresh public-path engagement measurement (2026-07-13, M5 Max / macOS 27 beta / MLX 0.31.2, qL=4096): **2.58–2.84× causal** and **2.05–2.14× non-causal** vs SDPA-vjp. Set `=1` to restore SDPA-vjp. |
| `MFA_V6BWD_USE_FUSED` | bool | unset (split) | **v2.37.0**: with V6NAX backward enabled, choose the fused WM=1 dK/dV kernel (single dispatch) instead of the WM=4 multi-SG split (two dispatches).  Default off (multi-SG split, 1.7-2× faster).  Set =1 for fallback / benchmarking. |
| `MFA_V6BWD_WM` | int | 4 | **v2.37.0**: WM for the multi-SG dK + dV split kernels.  Default 4 (Q-row partition with each SG owning 16 Q-rows).  Override for autoresearch sweeps. |
| `MFA_V6BWDV_BQ`, `MFA_V6BWDV_BK`, `MFA_V6BWDV_WM` | int | 64, 32, 4 | Per-kernel tile overrides for dV kernel (v2.37.0).  Researchers. |
| `MFA_V6BWDK_BQ`, `MFA_V6BWDK_BK`, `MFA_V6BWDK_WM` | int | 64, 32, 4 | Per-kernel tile overrides for dK kernel (v2.37.0).  Researchers. |
| `MFA_V6_BWD_KERNEL` | str | `auto` | V6NAX backward kernel mode selection. `auto` → **split for every D** (D∈{64,128}); `fused` → forced fused; `split` → forced split-dKdV; `legacy_fused` → WM=1 escape hatch. Fresh D64 split measurement: **2.58–2.84× causal**, **2.05–2.14× non-causal** vs SDPA-vjp (2026-07-13, M5 Max / macOS 27 beta / MLX 0.31.2, qL=4096). |
| `MFA_V6_BWD_SPARSE_NATIVE` | bool | unset | **v2.50 Prompt 5d**: opt-in to full-native V6NAX backward sparse kernels (4 sparse kernels: dQ + dV + dK split + fused dKdV) instead of Prompt 5c hybrid orchestrator.  Default off (hybrid is production per Pattern #6 empirical bench — V6NAX NAX backward slower than Apple SDPA NAX on M5+).  Set `=1` for research/benchmark access.  See `.doc-archive/docs/v50/section-a-v3-empirical-verification.md`. |
| `MFA_TOPK_BISECT` | bool | unset | **GHOST (campaign 2026-06 Track 0)**: not read by ANY code path — setting it is a no-op.  Bisection is the AUTO default; the live opt-out is `MFA_DISABLE_TOPK_BISECT`.  Row retained for historical reference only. |
| `MFA_DISABLE_TOPK_BISECT` | bool | unset | **v2.50 Prompt 5c**: opt-out of Top-K bisection kernel AUTO default; falls back to Phase 3a legacy `mx.topk` path.  Use for exact-mx.topk-semantics or debugging. |
| `MFA_DISABLE_TOPK_NAX` | bool | unset | Disable Top-K NAX dispatch entirely; falls back to Python reference (very slow at scale, for correctness comparison). |
| `MFA_DISABLE_ROPE_NAX` | bool | unset | **Sprint 2**: opt-out of `mx.fast.rope` dispatch path in `flash_attention_rope_unified`; falls back to STEEL host-side RoPE. |
| `MFA_DISABLE_TQ_DECODE_SDPA` | bool | unset | **v2.51.0**: opt-out of the default TurboQuant paged-decode (N_q=1) gather/dequant + Apple SDPA path (6.0-14.4x); falls back to the fused TQ kernel path. |
| `MFA_DISABLE_AUTO_HOOKS` | bool | unset | Disable auto-hook installation at `import mlx_mfa` (`_auto_hooks.py::install_hooks()`); no `mx.*` surfaces are patched. |
| `MLX_MFA_VERBOSE_DISPATCH` | bool | false | Print dispatch decisions to stderr. **Read once at import (CC-07):** captured into a module global when `mlx_mfa` is first imported — set it **before `import mlx_mfa`**; changing it later in the same process has no effect. |
| `MLX_MFA_DISPATCH_TABLE` | path | unset | JSON file with custom per-config dispatch thresholds |
| `MLX_MFA_HOOK_TELEMETRY` | str | `summary` | **v2.50.1 Prompt 5g Phase C**: hook execution/fallback telemetry mode (Pattern #8 prevention).  Values: `off` (zero overhead, no counters), `summary` (default; per-hook executed/fallback counters readable via `mlx_mfa.get_hook_stats()`), `verbose` (summary + `UserWarning` per fallback for active debugging).  Invalid values now emit a `RuntimeWarning` and fall back to `summary` (audit L-03) rather than silently defaulting.  **Read once at import (CC-16):** captured into a module global when `mlx_mfa` is first imported — set it **before `import mlx_mfa`**; changing it later in the same process has no effect.  See `docs/reference/HOOK_TELEMETRY.md`. |

## Availability & validation (Python-side, live-read)

These knobs were live-read in `mlx_mfa` but missing from the registry/doc until the
2026-06-21 audit (H4) — added here and to `_knobs.KNOWN_KNOBS`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MFA_REQUIRE_NAX` | bool | unset | When `=1`, `flash_attention` RAISES `NaxUnavailable` instead of silently falling back to SDPA when the Neural-Accelerator (`_ext`) is unavailable on Apple Silicon — opt-in loud failure for "acceleration must be on" deployments (read at `attention.py`). |
| `MFA_SILENCE_NAX_WARNING` | bool | unset | When `=1`, suppress the one-time loud `RuntimeWarning` emitted on an UNEXPECTED silent NAX fallback (Apple-Silicon + `_ext` failed). Expected fallbacks (non-target platform / pre-M5) are silent regardless. |
| `MFA_KNOB_STRICT` | bool | unset | When `=1`, `mlx_mfa._knobs.validate_env()` warns on unrecognized (`possible typo`) and removed (`removed — no effect`) `MFA_*`/`MLX_MFA_*` env vars. Off by default so a missing-registry knob never disrupts a setup. |

## Shader Generation (cold path, not cached)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MFA_DEBUG_SHADERS` | bool | false | Dump generated Metal shader source to stderr during compilation |
| `MFA_NO_PADDING` | bool | false | Disable threadgroup memory padding (causes NaN in 45/594 tests). **Load-time-only (CX-07):** frozen at first read (it is absent from the shader-cache key, so a mid-process toggle would return a stale-padding kernel) — set it **before the first attention call**; `_invalidate_env_config()` does **not** reset it. |
| `MFA_IR_INVESTIGATE` | bool | false | Dump Metal IR during shader compilation |
| `MFA_DISABLE_ASYNC` | bool | false | Disable precompiled async metallib lookup |
| `MFA_DISABLE_GNA_NATIVE` | bool | false | Disable native GNA kernel; fall back to sparse path |

## Conv NAX (Python + C++ side)

*(documented repo review 2026-05)*

| Variable | Type | Default | Description |
|---|---|---|---|
| `MFA_CONV_NAX_NO_FAST_PATH` | bool | unset | Bypass the 1×1×1 fast path in Conv3D NAX (forces the general path; used by perf tests). |
| `MFA_CONV_NAX_USE_PYTHON_LEGACY` | bool | unset | Route Conv3D NAX through the Phase 1.x legacy Python implementation (debug). |
| `MFA_DISABLE_CONV3D_MPP` | bool | unset | **v2.50.2/v2.51.0 (corrected H-07)**: opt-out of the default MPP `convolution2d` conv3d path; the PUBLIC fallback routes to **`mx.conv_general`** (MLX's own conv — the legacy NAX/materialized-im2col path is bypassed entirely). The fp16 2.3-2.5x / bf16 1.4-2.7x speedups were measured vs an **internal** direct-binding materialized-im2col baseline (a methodology denominator, NOT the public-knob fallback); see PERF_CLAIMS H-07 note. |
| `MFA_ENABLE_CONV3D_PAD_SLICE` | bool | unset (off) | **Opt-in (β3-indicative, default OFF).** MLX-style pad-and-slice around the NAX conv datapath: Conv3D shapes whose ONLY MPP-ineligibility reason is channel alignment (`C_in/C_out` not `%16` or `<32`) are padded into the MPP envelope (C_in→mult-of-32 so K=C_in·27 is a clean 32-tile ⇒ no partial-K NaN; C_out→mult-of-16; zero-filled inert), run on NAX, sliced back — instead of forfeiting NAX to `mx.conv_general`. Measured M5/macOS-27-β3: 1.3–3.9× faster than `mx.conv_general` on RGB/latent VAE convs, cos=1.0 (fp16+bf16). Path to default-on: stable-macOS re-validation + coordinated dispatch-map/lock update. |
| `MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE` | bool | unset (off) | **Expert opt-in (β3-indicative, default OFF).** Routes only the measured fp16 SeedVR2 VAE spatial-tail family (`B=1`, `T∈{4,5}`, `H×W=108×132`, `C_in=C_out=512`, `k=3³`, stride 1, causal temporal pad 0, spatial pad 1) through zero-pad to `112×136` → NAX MPP → slice. The `54×66` family remains on MLX: its 1.30–1.61× micro win did not survive VAE-unit measurement. Unmeasured bf16, stride-2, and channel-tail families also remain on MLX. Revalidate on stable macOS before considering default-on. |
| `MFA_ENABLE_VARLEN_NAX` | bool | unset (off) | **Opt-in (β3-indicative, default OFF).** Routes only the measured packed-varlen V6 NAX envelope: `B=1`, D128, fp16/bf16, causal or non-causal, GQA factor 2/4/8, exactly 20 or 24 equal-Q/K segments, and `35018≤total_q=total_k≤35250`. The route passes fixed `BQ32/BK32/WM2` explicitly through MSL generation, cache-keying, metadata validation, and host dispatch; it never relies on a transient tile environment. All other varlen inputs keep their STEEL or split-concat behavior. Revalidate on stable macOS before considering default-on. |
| `MFA_CONV3D_PAD_RATIO_MAX` | float | `12.0` | Cost gate for `MFA_ENABLE_CONV3D_PAD_SLICE`: skip pad-and-slice (→ `mx.conv_general`) when channel padding inflates work by more than this multiplier `(C_pad/C_in)·(O_pad/C_out)`. Default admits the validated win envelope (measured wins up to 10.67, e.g. RGB 3↔128). |
| `MFA_REQUIRE_MSL4` | (not an env var) | — | **Corrected (campaign 2026-06 Track 0)**: this is a SOURCE-STRING SENTINEL (`// MFA_REQUIRE_MSL4` comment in generated Metal), detected by `shader_cache.mm` to select MTLLanguageVersion4_0.  It is never read from the environment. |

## Calibration (dynamic keys, not cached)

| Variable pattern | Type | Description |
|-----------------|------|-------------|
| `MFA_SPLITK_MAX_N_D{D}_C{0\|1}_A{0\|1}_W{0\|1}` | int | Per-config max N for split-K dispatch |

## Expert NAX tile and source probes

These knobs are research-only and do not change public routing. Tile values are
positive integers; boolean values follow the global strict `0|1` contract.

| Variable | Type | Default | Description |
|---|---|---|---|
| `MFA_GNA_NAX_BQ` / `MFA_GNA_NAX_BK` / `MFA_GNA_NAX_WM` | int | shape default | Expert GNA NAX tile override. |
| `MFA_GNA_NAX_PRECOMPUTE_RANGE` | bool | `0` | Select the default-off `_pr1` range-precompute GNA variant. |
| `MFA_GNA_NAX_SWIZZLE_LOG` | int | `0` | Expert GNA dispatch swizzle probe. |
| `MFA_GNA_NAX_DUMP_PATH` | path | unset | Dump the generated GNA MSL and selected kernel name; debug only. |
| `MFA_FFN_NAX_BM` / `MFA_FFN_NAX_BN` / `MFA_FFN_NAX_BK` / `MFA_FFN_NAX_WM` / `MFA_FFN_NAX_WN` | int | shape default | Expert FFN NAX tile overrides. |
| `MFA_QMM_NAX_BM` / `MFA_QMM_NAX_BN` / `MFA_QMM_NAX_BK` / `MFA_QMM_NAX_WM` / `MFA_QMM_NAX_WN` | int | shape default | Expert quantized-matmul NAX tile overrides. |
| `MFA_V6_NAX_D_SUBTILE` | int | head dimension | Expert head-dimension sub-tile probe; not public routing. |
| `MFA_STEEL_MSL` | enum | unset | Developer-only STEEL MSL source selector. |

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
| `MFA_V6_VARLEN_DUMP_PATH` | path | unset | **Debug only.** Dump packed-varlen V6 NAX generated MSL to a file for forensic source comparison. Never enable for routing or performance measurement. |
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
