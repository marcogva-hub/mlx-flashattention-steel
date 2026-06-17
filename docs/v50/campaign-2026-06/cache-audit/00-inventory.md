# Sprint A Phase A.0 — cache/key/dedup site inventory

Exhaustive enumeration, 2026-06-12.  Cross-checked by two grep passes
(dict/lru/memo patterns + unordered_map/Key-struct patterns + a third
pass over factory/closure/is_equivalent constructs).

## C++ pipeline caches (11)

| # | Cache | File:line | Maps | Value |
|---|---|---|---|---|
| 1 | `ShaderCache::cache_` | shader_cache.hpp:130 / .mm | KernelKey (18 fields) → PSO | pipeline (feeds 14+ generators: STEEL V1/V2/splitK/dsplit/V3/V4/V5, bwd dQ/dKV, sparse, GNA, sage, paged-varlen-TQ, flash-decode, ccv) |
| 2 | `v6_pipelines` | mfa_v6_nax_primitive.cpp:124 | V6Key → PSO | V6 NAX fwd (legacy + V6NAX) |
| 3-11 | `v6nax_bwd{q,kv,v,v_sparse,k,fused,q_sparse,k_sparse,f_sparse}_pipelines` | same file | V6NAXBwd*Key → PSO | 9 V6NAX backward kernels |

## C++ graph-dedup predicates (25 is_equivalent sites)

mfa_attention.hpp ×11 (MFAttention + 10 aux primitives), mfa_attention.cpp
(MFAttention impl), mfa_v6_nax_primitive.cpp ×13, mfa_smooth_quant /
mfa_paged_gather / mfa_scatter / mfa_quantize hpp ×4.

## Python caches / factories (~20)

| Site | Key | Value |
|---|---|---|
| `_dispatch_decision_cache` (attention.py:45) | shape/flags + 6 env vars (post A-5) | use_mfa decision |
| 7× `lru_cache` factories (attention.py 2241/2496/2604/2912/4285/4511/4914) | factory args | custom_function closures w/ registered vjp |
| `_SPARSE_BIAS_CACHE` (attention.py:3772) | id+shape+dims, holds (mask_ref, bias) | expanded float bias |
| `conv_nax._KERNEL_CACHE` (2 key sites) | `_conv_key(...)` / Pointwise tuple | mx.fast.metal_kernel handles |
| `turboquant._centroid_cache` | bits | static Lloyd-Max tables |
| `_auto_hooks._M5_PLUS_CACHE` | — (process constant) | hardware probe |
| `dispatch_policy._custom_thresholds` | env path (post A-5) | dispatch table |
| mlx_lm `_SUPPORTED_*` | refreshed per patch call | integration config |
| lcsa_nax / compile_metallib / masks / integrations | — | verified no incomplete caches |

Full per-site affecting-input derivations: `01-affecting-inputs.md`.
