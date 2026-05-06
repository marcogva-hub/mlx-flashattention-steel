# v2.32.0 — kernel inventory

**Sprint:** Sprint A.0 (kernel survey)
**Branch:** `experiment/v32-sdpa-routing`

## Architecture

mlx-mfa exposes two C++ MLX primitives for forward attention:

1. **`MFAttention`** (`csrc/mfa_attention.cpp`, ~3320 LOC) — STEEL family
2. **`MFAV6NAXForward`** (`csrc/mfa_v6_nax_primitive.cpp`) — V6 NAX legacy + V34

Plus specialized primitives: `MFAVarlenAttention`, `MFAPagedSteelForward`,
`MFASageForward`, `MFAGNAForward`, `MFAPagedVarlenForward`,
`MFAPagedVarlenTQForward`.

The Python wrapper `mlx_mfa.flash_attention()` (in `mlx_mfa/attention.py`)
decides MFA vs SDPA via `mlx_mfa.dispatch_policy.should_use_mfa()`. When
MFA is selected, it calls `_mfa_forward()` → `_make_mfa_custom()` → C++
primitive. The exact primitive (MFAttention vs MFAV6NAXForward) is
selected inside the C++ layer based on dtype, shape, hardware, and env vars.

## Forward kernels by KernelType (`csrc/shader_cache.hpp`)

| ID | KernelType | Path | Dispatched when |
|---:|---|---|---|
| 0 | `AttentionForward` | ccv legacy V1 | dtype=fp32 (line 120) |
| 3 | `SteelForward` | STEEL V1 | fallback for D=256/512 + dtypes/shapes others reject |
| 4 | `FlashDecodePartial` | Flash decoding split-KV | N≤4, S≥256, f16/bf16, no block_mask (line 211) |
| 5 | `FlashDecodeReduce` | Phase 2 of flash decode | (always paired with 4) |
| 16 | `SteelForwardV2` | STEEL V2 (sequential KV_smem, BQ=32 BK=64/32) | M1/M2 default; M3+ guarded by MFA_DISABLE_V2 (line 800-915) |
| 17 | `SteelV2SplitKPartial` | V2 split-K | under-occupied grid (total_tgs < 0.8 × cores), N>4 (line 359) |
| 18 | `SteelV2DSplit256` | V2 D-split D=256 | f16/bf16, D=256 |
| 19 | `SteelV2DSplit512` | V2 D-split D=512 | f16/bf16, D=512 |
| 20 | `SteelForwardV3` | V3 (separate K_smem+V_smem) | M3+, causal, N≥1024(D=64)/2048(D=128), B*H≥4 (line 691) |
| 21 | `SteelForwardV4` | V4 (direct device K reads) | M3+, opt-in via MFA_ENABLE_V4 (line 470) |
| 22 | `SteelForwardV6NAX` | V6 NAX legacy (MPP cooperative_tensor) | M5+, dispatched in MFAV6NAXForward (separate primitive) |
| 23 | `SteelForwardV5` | V5 (D-blocked, BD_tile=32 BK=128) | opt-in via MFA_ENABLE_V5 (line 578) |
| 24 | `GNAForward` | GNA 3D window inline | flash_attention_gna() entry |
| 25 | (V34) | V34 NAX-direct via NAXFrag::mma | Inside MFAV6NAXForward when use_v34 = true |
| 27 | `PagedVarlenForward` | Fused packed Q + paged KV | flash_attention_paged_varlen() entry |
| 28 | `PagedVarlenTQForward` | TurboQuant paged | TQ KV cache flow |

## Activation matrix on M5 Max (gen=17)

| Kernel | Activated by | Default on M5+ (NAX)? |
|---|---|---|
| ccv V1 (KT=0) | dtype=fp32 | yes for fp32 only |
| STEEL V1 (KT=3) | fallback after V2-V5 reject | rare |
| Flash decode (KT=4-5) | N≤4 small qL | yes for decode shapes |
| V2 (KT=16) | dtype=f16/bf16, D≤256, NOT M3+ default-disabled | NO — M3+ guard skips V2 |
| V2 split-K (KT=17) | V2-eligible AND under-occupied grid | NO — same M3+ guard |
| V2 D-split 256/512 (KT=18-19) | D=256/512, f16/bf16 | yes |
| V3 (KT=20) | M3+ causal AND B*H≥4 AND N≥(1024/2048) | yes — M3+ default for D≤128 causal |
| V4 (KT=21) | M3+, MFA_ENABLE_V4=1 | OFF by default (opt-in) |
| V5 (KT=23) | MFA_ENABLE_V5=1 | OFF by default (opt-in) |
| V6 NAX legacy (KT=22) | M5+, MFAV6NAXForward primitive, use_v34=false | yes for D=64 small-N (FlashVSR) |
| V34 NAX-direct | M5+, MFAV6NAXForward, use_v34=true | yes for D=128 + D=64 N_kv>8000 |
| GNA (KT=24) | flash_attention_gna() entry | per call site |
| Sage (KT=11) | backend="sage" or sage_attention() | per call site |

## Existing env vars (relevant for sweep)

| Env | Effect |
|---|---|
| `MFA_DISABLE_V2=1` | Forces V1 path |
| `MFA_FORCE_V2=1` | Forces V2 even on M3+ (bypasses M3+ guard) |
| `MFA_DISABLE_V3=1` | Skips V3, falls to V2 |
| `MFA_ENABLE_V4=1` | Opts into V4 (M3+ direct device reads) |
| `MFA_ENABLE_V5=1` | Opts into V5 (D-blocked) |
| `MFA_FORCE_SPLITK=0\|1` | Disable/force V2 split-K |
| `MFA_V6_USE_V34=0\|1` | Force V34 OFF/ON within V6 NAX primitive |
| `MFA_V6_NAX_SINGLE_OTILE=0\|1` | V6 NAX single-Otile vs double-buffered |
| `MFA_FORCE_GEN=N` | Override architecture gen detection |

## Python-level controls

| Env / param | Effect |
|---|---|
| `flash_attention(backend="mfa")` | Force MFA — internal dispatch picks sub-kernel |
| `flash_attention(backend="sdpa")` | Force MLX SDPA |
| `flash_attention(backend="auto")` | Default — `should_use_mfa()` decides |
| `MLX_MFA_VERBOSE_DISPATCH=1` | Log every dispatch decision |
| `MLX_MFA_DISPATCH_TABLE=path` | Custom thresholds JSON |

## Implication for Sprint A sweep design

To answer "does mlx-mfa beat SDPA on this niche shape on M5+ NAX?", we
only need:

```python
mfa_ms  = bench(flash_attention(q, k, v, backend="mfa", ...))
sdpa_ms = bench(flash_attention(q, k, v, backend="sdpa", ...))
```

The internal MFA dispatch picks the best sub-kernel naturally. We
don't need a new `MFA_FORCE_KERNEL` env var.

For documentation purposes, `MLX_MFA_VERBOSE_DISPATCH=1` reveals which
sub-kernel was picked. Sprint A.6's dispatch table can record this.
