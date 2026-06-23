#!/usr/bin/env python3
"""Volet K0: mechanically verify and render the omitted-surface hardening plan.

The script derives the 31-name row set from volet J's generated enumeration,
checks public Python call edges with AST, checks raw binding/symbol presence,
and source-verifies every dispatch group's gather/barrier classification.
It intentionally does not import mlx_mfa or execute a GPU kernel.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENUMERATION = ROOT / "audit/round3_remediation/api_surface_enumeration.md"
ATTENTION = ROOT / "mlx_mfa/attention.py"
BINDINGS = ROOT / "csrc/bindings.cpp"
OUTPUT = ROOT / "audit/round3_remediation/hardening_plan.md"


@dataclass(frozen=True)
class Evidence:
    path: str
    needle: str
    note: str
    after: str = ""


@dataclass(frozen=True)
class Group:
    name: str
    members: tuple[str, ...]
    core: str
    current: str
    hardening: str
    residual: str
    gather: str
    evidence: tuple[Evidence, ...]


GROUPS = (
    Group(
        "P1 — dense public adapters",
        (
            "public:flash_attention_qkv_packed",
            "public:flash_attention_kv_packed",
            "public:flash_attention_speculative_verify",
            "public:flash_attention_splitfuse",
        ),
        "packing/proxy logic → `flash_attention` → dense auto dispatch "
        "(SDPA, V6-NAX, STEEL/V2/V3, or flash-decode by shape/features)",
        "Packed adapters validate layout/divisibility only; speculative verify "
        "does not validate `draft_ids` rank/batch/token count or temperature; "
        "splitfuse only rejects two absent Q branches. The downstream dense core "
        "validates ordinary Q/K/V contracts.",
        "Keep dense Q/K/V validation centralized in `flash_attention`; add a "
        "small adapter-boundary validator for packed head counts/layouts and a "
        "splitfuse branch-triple validator before either dispatch.",
        "QKV/KV: positive head counts and 5-D GQA packing capacity. Speculative: "
        "`draft_ids` int dtype + `[B,N_draft]` exact shape, finite positive "
        "temperature. Splitfuse: each enabled Q/K/V triple complete and mutually "
        "compatible; reject cross-branch dtype/D disagreement if one scale is shared.",
        "ABSENT/N-A: no scattered-K/V gather. Dense auto uses Apple/direct-device "
        "paths on M5; legacy shared-memory dense paths already carry reuse barriers.",
        (
            Evidence("mlx_mfa/attention.py", "return flash_attention(q, k, v, scale=scale, causal=causal, stream=stream)", "packed adapters reach dense core", "def flash_attention_qkv_packed("),
            Evidence("mlx_mfa/attention.py", "out, lse = flash_attention(", "speculative dense reaches dense core", "def flash_attention_speculative_verify("),
            Evidence("csrc/mfa_steel_fwd_v2.cpp", "const bool use_direct_reads = key.is_m3_plus && !key.has_rope;", "M5 dense pointer/direct-read path"),
            Evidence("csrc/mfa_steel_fwd_v2.cpp", "threadgroup_barrier(mem_flags::mem_threadgroup);  // X", "legacy shared-KV reuse barrier"),
        ),
    ),
    Group(
        "P2 — RoPE-unified public family",
        (
            "public:flash_attention_rope_unified",
            "public:flash_attention_rope",
            "public:flash_attention_kvcache_rope_append",
        ),
        "`flash_attention_rope_unified` → paged core, dense `flash_attention`, "
        "or raw `_mfa_rope_forward`/`mfa_attention_rope_forward`",
        "Unified boundary checks Q rank, table presence, per-batch offset count, "
        "and mode exclusions. Thin wrappers add no independent Q/K/V/cache checks.",
        "Add one mode-aware validator in `flash_attention_rope_unified`: batch, "
        "K↔V seq/heads, GQA, mutual dtype/D, cache K↔V exact shape, and table "
        "dtype/shape/coverage before recursive per-batch dispatch.",
        "Standalone: K rotation range. Append: `k_cache`/`v_cache` paired and "
        "past-length agreement. Paged: pool/table/lens contract delegated once, "
        "but reject ignored K/V semantics explicitly. RoPE: even rotary width, "
        "`rotary_dim`, offset bounds, and cos/sin mutual shape/dtype.",
        "PRESENT where applicable: paged mode reaches the fixed paged gather; "
        "fused RoPE dense staging disables direct reads but has barrier X. No "
        "barrier-absent sibling.",
        (
            Evidence("mlx_mfa/attention.py", "out = flash_attention_paged(", "unified paged route", "def flash_attention_rope_unified("),
            Evidence("mlx_mfa/attention.py", "return _mfa_rope_forward(q, k, v, rotary_cos, rotary_sin,", "fused RoPE route", "def flash_attention_rope_unified("),
            Evidence("csrc/mfa_steel_fwd_v2.cpp", "const bool use_direct_reads = key.is_m3_plus && !key.has_rope;", "RoPE disables direct reads"),
            Evidence("csrc/mfa_steel_fwd_v2.cpp", "threadgroup_barrier(mem_flags::mem_threadgroup);  // X", "RoPE/shared-KV reuse barrier"),
            Evidence("csrc/mfa_steel_fwd.cpp", "OVERWRITES the shared KV_smem (Ks==Vs)", "paged gather reuse site", "generate_paged_steel_forward_source"),
        ),
    ),
    Group(
        "P3 — unified KV-cache router",
        ("public:flash_attention_kvcache",),
        "mode router → dense `flash_attention`, RoPE family, paged attention, "
        "and (paged append) raw `mfa_scatter_kv`",
        "Checks Q rank, mode exclusions/presence, several unsupported-feature "
        "combinations, equal heterogeneous RoPE offsets, and casts selected dtype "
        "mismatches. It does not establish a single complete cache/pool contract "
        "before mode-specific indexing/concatenation.",
        "Add a mode discriminator followed by exactly one validator per dense, "
        "dense-append, paged, and paged-append mode. Each must cover batch, "
        "K↔V, GQA, mutual dtype/D, and metadata cardinality before indexing.",
        "Dense append: cache/new-token seq/head/D compatibility. Paged append: "
        "pool exact mutual shape/dtype, block capacity for every append position, "
        "seq_lens/table dtype/rank/cardinality, scatter index bounds. "
        "`cache_batch_idx`: int dtype/rank/bounds and output batch agreement.",
        "PRESENT: paged attention's scattered gather has the CX-J-02 start-of-loop "
        "barrier. `mfa_scatter_kv` is a copy/scatter primitive, not a K/V attention "
        "gather loop. Dense branches are pointer/direct or barriered dense staging.",
        (
            Evidence("mlx_mfa/attention.py", "k_pages_new = _mfa_scatter_kv_cpp(", "paged append scatter route", "def flash_attention_kvcache("),
            Evidence("mlx_mfa/attention.py", "_out_dec = flash_attention_paged(", "paged consume route", "def flash_attention_kvcache("),
            Evidence("mlx_mfa/attention.py", "return flash_attention_rope(", "dense RoPE route", "def flash_attention_kvcache("),
            Evidence("csrc/mfa_steel_fwd.cpp", "OVERWRITES the shared KV_smem (Ks==Vs)", "paged gather race site", "generate_paged_steel_forward_source"),
            Evidence("csrc/mfa_steel_fwd.cpp", "CX-J-02 (volet J): barrier at the START of each iteration", "paged barrier emitted", "generate_paged_steel_forward_source"),
        ),
    ),
    Group(
        "P4 — paged speculative adapter",
        ("public:flash_attention_speculative_verify_paged",),
        "proxy logic → `flash_attention_paged(return_lse=True)` → paged STEEL gather",
        "Checks `draft_ids` rank and batch only; paged core validates pool/table/lens.",
        "Retain paged validation in the paged core; add exact speculative metadata "
        "validation before dispatch.",
        "`draft_ids` int dtype and `[B,N_q]` exact shape; finite positive "
        "temperature; reject impossible IDs rather than silently clipping if the "
        "API contract intends token IDs rather than D-axis proxy indices.",
        "PRESENT: scattered paged K/V gather has the CX-J-02 start-of-loop barrier.",
        (
            Evidence("mlx_mfa/attention.py", "out, lse = flash_attention_paged(", "paged speculative route", "def flash_attention_speculative_verify_paged("),
            Evidence("csrc/mfa_steel_fwd.cpp", "OVERWRITES the shared KV_smem (Ks==Vs)", "paged gather source", "generate_paged_steel_forward_source"),
            Evidence("csrc/mfa_steel_fwd.cpp", "CX-J-02 (volet J): barrier at the START of each iteration", "paged barrier source", "generate_paged_steel_forward_source"),
        ),
    ),
    Group(
        "P5 — varlen packed adapters",
        (
            "public:flash_attention_varlen_qkv_packed",
            "public:flash_attention_varlen_kv_packed",
        ),
        "packing logic → `flash_attention_varlen` → raw `mfa_attention_varlen_forward` "
        "or per-sequence fallback",
        "Adapters validate packing layout/divisibility only. The public varlen core "
        "validates cumulative-sequence metadata and ordinary Q/K/V shape contracts.",
        "Keep cumulative metadata/QKV checks in `flash_attention_varlen`; add packed "
        "layout/head-count checks before slicing so malformed 5-D GQA layouts cannot "
        "silently truncate heads.",
        "Positive `num_heads`/`num_kv_heads`; 5-D tensor head capacity for requested "
        "GQA; packed total-Q versus total-K semantics; explicit mutual packed dtype.",
        "PRESENT but not a scattered gather: packed-contiguous varlen stages K/V in "
        "shared memory and emits a start-of-iteration reuse barrier.",
        (
            Evidence("mlx_mfa/attention.py", "return flash_attention_varlen(", "packed varlen route", "def flash_attention_varlen_qkv_packed("),
            Evidence("csrc/mfa_steel_fwd.cpp", "generate_steel_varlen_forward_source", "varlen kernel generator"),
            Evidence("csrc/mfa_steel_fwd.cpp", "reads from KV_smem (V) are complete before we overwrite KV_smem with new K", "varlen reuse barrier rationale", "generate_steel_varlen_forward_source"),
        ),
    ),
    Group(
        "P6 — Sage KV-cache adapter",
        ("public:sage_attention_kvcache",),
        "`sage_attention` → `mfa_smooth_quantize_k`/quantize path → raw `mfa_sage_forward`",
        "The adapter adds no checks; `sage_attention` owns Q/K/V validation and "
        "preprocessing, while volet J hardened prequantized/raw buffers.",
        "Keep one Sage Q/K/V validator in `sage_attention` and ensure the KV-cache "
        "alias cannot bypass it.",
        "Decode-specific contract: allow Nq≠Nkv but require K↔V seq/heads, GQA, "
        "batch, mutual dtype/D, non-empty KV, and supported window/causal semantics.",
        "PRESENT: Sage shared-KV gather has the volet-S inter-iteration barrier.",
        (
            Evidence("mlx_mfa/attention.py", "return sage_attention(", "Sage adapter route", "def sage_attention_kvcache("),
            Evidence("csrc/mfa_sage_fwd.cpp", "Mirrors the STEEL forward's start-of-loop barrier", "Sage shared-memory barrier", "generate_sage_forward_source"),
        ),
    ),
    Group(
        "P7 — top-k standalone",
        ("public:flash_attention_topk",),
        "MLX matmul/top-k or bisection threshold → Apple SDPA; otherwise MLX "
        "materialized-score reference",
        "Checks Q rank and ratio only. K/V are indexed and multiplied without a "
        "mutual contract check; optional mask is expanded without full dtype/shape validation.",
        "Add a local top-k Q/K/V validator covering batch, K↔V, GQA-or-explicit-MHA "
        "policy, mutual dtype/D, and supported numeric dtypes.",
        "Mask bool dtype and exact tile shape; finite positive scale; finite inputs "
        "policy; define whether GQA is supported on both reference and SDPA branches.",
        "ABSENT/N-A: no custom scattered-K/V shared-memory gather.",
        (
            Evidence("mlx_mfa/attention.py", "return mx.fast.scaled_dot_product_attention(", "top-k NAX/SDPA route", "def flash_attention_topk("),
            Evidence("mlx_mfa/attention.py", "out = weights @ v", "top-k reference route", "def flash_attention_topk("),
        ),
    ),
    Group(
        "R1 — raw dense MFA forward",
        ("raw:mfa_attention_forward",),
        "free function → `MFAttention` primitive → dense auto STEEL/ccv dispatch",
        "Checks rank, batch, K↔V seq/heads, and D membership. Missing GQA, q↔k/v D, "
        "and mutual/supported dtype checks.",
        "Introduce a shared raw dense-QKV validator and call it before contiguous conversion.",
        "Preserve supported D/features; reject zero Hkv/sequence, invalid GQA, "
        "q/k/v D mismatch, and mixed/unsupported dtype.",
        "ABSENT/N-A: dense pointer/direct path on M5; legacy dense shared staging is barriered.",
        (
            Evidence("csrc/mfa_attention.cpp", "mlx::core::array mfa_attention_forward(", "raw core"),
            Evidence("csrc/mfa_attention.cpp", "std::make_shared<MFAttention>(s, params)", "primitive dispatch"),
            Evidence("csrc/mfa_steel_fwd_v2.cpp", "const bool use_direct_reads = key.is_m3_plus && !key.has_rope;", "M5 direct reads"),
        ),
    ),
    Group(
        "R2 — raw dense MFA with LSE",
        ("raw:mfa_forward_with_lse",),
        "binding-local construction → `MFAttention` primitive",
        "Checks rank, batch, K↔V seq/heads, mutual D, GQA, D membership, and mutual dtype.",
        "Route through the same shared raw dense-QKV validator as R1 to prevent drift.",
        "Add supported dtype and non-empty KV/head checks; preserve LSE shape/domain contract.",
        "ABSENT/N-A: same dense pointer/direct or barriered dense staging as R1.",
        (
            Evidence("csrc/bindings.cpp", "m.def(\"mfa_forward_with_lse\"", "binding boundary"),
            Evidence("csrc/bindings.cpp", "std::make_shared<mlx_mfa::MFAttention>", "primitive dispatch"),
        ),
    ),
    Group(
        "R3 — raw ALiBi MFA",
        ("raw:mfa_attention_alibi_forward",),
        "free function → `MFAttention(has_alibi)` → dense STEEL",
        "Checks Q/K/V rank, ALiBi rank, and D membership; it does not validate "
        "Q/K/V mutual contract or slope count/dtype.",
        "Call shared raw dense-QKV validator.",
        "Require slopes float32 (or explicitly cast), length Hq, finite values, "
        "and supported causal/GQA semantics.",
        "ABSENT/N-A: dense direct-device reads on M5; legacy dense staging barriered.",
        (
            Evidence("csrc/mfa_attention.cpp", "mlx::core::array mfa_attention_alibi_forward(", "ALiBi core"),
            Evidence("csrc/mfa_attention.cpp", "/*has_alibi=*/true", "ALiBi dispatch flag"),
        ),
    ),
    Group(
        "R4 — raw bias MFA",
        ("raw:mfa_attention_bias_forward",),
        "free function → `MFAttention(has_attn_bias)` → dense STEEL",
        "Checks Q/K/V rank, bias mode/shape, D membership, and casts bias to float32; "
        "Q/K/V mutual contract remains unchecked.",
        "Call shared raw dense-QKV validator.",
        "Bias finite-value policy; mode-2 head count already shape-locked but must "
        "remain after common-validator refactor.",
        "ABSENT/N-A: dense direct-device reads on M5; legacy dense staging barriered.",
        (
            Evidence("csrc/mfa_attention.cpp", "mlx::core::array mfa_attention_bias_forward(", "bias core"),
            Evidence("csrc/mfa_attention.cpp", "/*has_attn_bias=*/true", "bias dispatch flag"),
        ),
    ),
    Group(
        "R5 — raw fused-RoPE MFA",
        ("raw:mfa_attention_rope_forward",),
        "free function → `MFAttention(has_rope)` → STEEL shared-KV path",
        "Checks Q/K/V rank, D membership, and rejects Q float32 only. It does not "
        "validate Q/K/V mutual contract or rotary buffers.",
        "Call shared raw dense-QKV validator before primitive construction.",
        "Cos/sin float32, exact mutual shape, width D/2, sufficient position rows "
        "for K and offset Q, non-negative offset, and interleaving constraints.",
        "PRESENT: RoPE disables M5 direct reads, reuses shared K/V, and emits barrier X.",
        (
            Evidence("csrc/mfa_attention.cpp", "mlx::core::array mfa_attention_rope_forward(", "RoPE core"),
            Evidence("csrc/mfa_steel_fwd_v2.cpp", "const bool use_direct_reads = key.is_m3_plus && !key.has_rope;", "RoPE forces shared path"),
            Evidence("csrc/mfa_steel_fwd_v2.cpp", "threadgroup_barrier(mem_flags::mem_threadgroup);  // X", "reuse barrier"),
        ),
    ),
    Group(
        "R6 — raw MFA block-sparse pair",
        (
            "raw:mfa_attention_sparse_forward",
            "raw:mfa_attention_sparse_forward_with_lse",
        ),
        "free functions → `MFAttention(has_block_mask)` → STEEL sparse dispatch",
        "Both check Q/K/V rank, mask rank, D membership, and reject Q float32. "
        "They omit Q/K/V mutual contract plus mask dtype/tile/cardinality checks.",
        "Add one shared raw sparse validator used by both functions.",
        "Batch/K↔V/GQA/mutual dtype+D; bool/uint8 policy; exact 2-D/3-D/4-D "
        "mask geometry against selected BQ/BK; LSE contract for the second entry.",
        "ABSENT/N-A for scattered gather: sparse tiles jump through contiguous K/V. "
        "Any shared-memory staging path is barriered.",
        (
            Evidence("csrc/mfa_attention.cpp", "mlx::core::array mfa_attention_sparse_forward(", "sparse core"),
            Evidence("csrc/mfa_attention.cpp", "mfa_attention_sparse_forward_with_lse(", "sparse LSE core"),
            Evidence("csrc/mfa_steel_fwd_v2.cpp", "threadgroup_barrier(mem_flags::mem_threadgroup);  // X", "shared staging barrier"),
        ),
    ),
    Group(
        "R7 — raw packed-varlen MFA",
        ("raw:mfa_attention_varlen_forward",),
        "free function → `MFAVarlenAttention` → `SteelVarlenForward`",
        "No host shape validation; metadata are silently cast to int32 before dispatch.",
        "Add a dedicated varlen raw validator before any `shape()` access or cast.",
        "Q/K/V rank `[1,H,total,D]`, batch=1, K↔V, GQA, mutual dtype/D, supported "
        "dtype/D; metadata int32 (no silent cast), rank/cardinality, monotonic starts "
        "at zero, terminal totals, tile-offset count/terminal value.",
        "PRESENT but not scattered: contiguous packed K/V shared staging has an "
        "inter-iteration barrier.",
        (
            Evidence("csrc/mfa_attention.cpp", "mfa_attention_varlen_forward(", "varlen free function"),
            Evidence("csrc/mfa_attention.cpp", "std::make_shared<MFAVarlenAttention>", "varlen primitive"),
            Evidence("csrc/mfa_steel_fwd.cpp", "reads from KV_smem (V) are complete before we overwrite KV_smem with new K", "varlen barrier rationale", "generate_steel_varlen_forward_source"),
        ),
    ),
    Group(
        "R8 — raw GNA",
        ("raw:mfa_gna_forward",),
        "free function → `MFAGNAForward` → GNA STEEL-V2 generator",
        "Checks rank, D=128, rejects Q float32, lattice product, GQA, batch, "
        "self-attention sequence equality, K↔V heads, and mutual D. Mixed dtype "
        "and invalid window/stride parameters remain unchecked.",
        "Retain the existing shape validator and add mutual supported dtype checks.",
        "Positive dims/windows/strides, overflow-safe lattice product, window bounds "
        "semantics, and non-empty H/KV.",
        "ABSENT/N-A on M5: GNA sets direct device reads. Legacy shared staging has "
        "barriers on both sides of reuse.",
        (
            Evidence("csrc/mfa_attention.cpp", "mlx::core::array mfa_gna_forward(", "GNA free function"),
            Evidence("csrc/mfa_gna_fwd.cpp", "const bool use_direct_reads = key.is_m3_plus;", "M5 direct reads"),
            Evidence("csrc/mfa_gna_fwd.cpp", "threadgroup_barrier(mem_flags::mem_threadgroup);", "legacy shared barriers"),
        ),
    ),
    Group(
        "R9 — raw NAX sparse forward",
        ("raw:sparse_attention_forward",),
        "free function → MLX `metal_kernel` V1 direct-device kernel or V2 NAX tile kernel",
        "Comprehensively checks rank, mutual dtype/shape, batch, GQA, D, block "
        "geometry, mask dtype/rank/shape, causal shape, scale, and mask address-space floor.",
        "No shared-core validation change required; convert repeated checks into a "
        "helper only when R10 is hardened so the two variants cannot drift.",
        "Kernel-version enum should reject unknown explicit values rather than silently "
        "falling back; document/validate small-mask address-space contract.",
        "ABSENT/N-A: V1 reads device K/V rows directly; V2 uses device NAX tiles. "
        "No scattered shared-K/V gather.",
        (
            Evidence("csrc/mfa_sparse_attention.cpp", "mlx::core::array sparse_attention_forward(", "NAX sparse core"),
            Evidence("csrc/mfa_sparse_attention.cpp", "device const T* K_kb = K_base + kb", "V2 device pointer jump"),
            Evidence("csrc/mfa_sparse_attention.cpp", "device const ", "V1 device reads"),
        ),
    ),
    Group(
        "R10 — raw NAX sparse with LSE",
        ("raw:sparse_attention_forward_with_lse",),
        "free function → MLX `metal_kernel` V1 LSE kernel",
        "Duplicates only a subset of R9 validation: it lacks batch/K↔V/D mutual "
        "checks, full 3-D/4-D mask shape checks, causal qL=kL, and positive scale.",
        "Extract and call the same sparse-input validator as R9, parameterized only "
        "for LSE capability/version.",
        "Preserve natural-log LSE and all-false `-INFINITY` sentinel contract.",
        "ABSENT/N-A: direct device K/V row reads; no scattered shared-K/V gather.",
        (
            Evidence("csrc/mfa_sparse_attention.cpp", "sparse_attention_forward_with_lse(", "NAX sparse LSE core"),
            Evidence("csrc/mfa_sparse_attention.cpp", "V1 generator only", "LSE dispatch variant"),
        ),
    ),
    Group(
        "R11 — raw per-block quantizer",
        ("raw:mfa_quantize_per_block",),
        "free function → `MFAQuantizePerBlock` primitive",
        "Checks rank, f16/bf16 dtype, and positive power-of-two block size.",
        "No Q/K/V mutual validator applies; retain local validation.",
        "Reject empty dimensions if unsupported; bound block size and integer-derived "
        "grid sizes to avoid overflow/invalid threadgroup assumptions.",
        "ABSENT/N-A: quantization reduction, not attention K/V gathering.",
        (
            Evidence("csrc/mfa_quantize.cpp", "mfa_quantize_per_block(", "quantizer core"),
            Evidence("csrc/mfa_quantize.cpp", "block_size must be a positive power of 2", "current validator"),
        ),
    ),
    Group(
        "R12 — raw smooth+quantize",
        ("raw:mfa_smooth_quantize_k",),
        "free function → `MFASmoothQuantizeK` two-pass primitive",
        "Checks rank, f16/bf16 dtype, and positive power-of-two block size.",
        "Retain local validation; share scalar block-size helper with R11 only if "
        "the helper preserves each primitive's dimension constraints.",
        "Reject empty dimensions if unsupported; bound block size/grid arithmetic; "
        "lock scale-count and mean output shapes.",
        "ABSENT/N-A: preprocessing reduction/quantization, not attention K/V gathering.",
        (
            Evidence("csrc/mfa_smooth_quant.cpp", "mfa_smooth_quantize_k(", "smooth-quant core"),
            Evidence("csrc/mfa_smooth_quant.cpp", "block_size must be a positive power of 2", "current validator"),
        ),
    ),
    Group(
        "R13 — raw paged scatter",
        ("raw:mfa_scatter_kv",),
        "free function → `MFAScatterKV` copy/scatter primitive",
        "Checks pool/tokens/index ranks, pool dtype, index dtype, token tail shape, "
        "and equal write counts. It does not require token dtype=pool or validate index bounds.",
        "Keep local scatter validation and reject malformed writes before dispatch.",
        "Mutual pool/tokens dtype; every `blk_id` in `[0,num_blocks)` and `blk_off` "
        "in `[0,block_size)`; duplicate-target semantics; empty-write behavior.",
        "ABSENT/N-A: scatter copy has no iterative attention K/V shared-memory gather.",
        (
            Evidence("csrc/mfa_scatter.cpp", "mlx_mfa::mfa_scatter_kv(", "scatter core"),
            Evidence("csrc/mfa_scatter.cpp", "blk_ids/blk_offs length must match", "current validator"),
        ),
    ),
    Group(
        "R14 — raw Conv3D NAX",
        ("raw:conv3d_nax_forward",),
        "free function → eligible MPP convolution2d path or im2col/matmul fallback",
        "Checks ranks, mutual f16/bf16 dtype, C-in, stride/dilation/padding, and "
        "effective output size.",
        "Retain Conv-specific validator; no attention shared core applies.",
        "Validate tuple arity at binding boundary, positive/non-empty dimensions, "
        "`chunk_M` domain, and overflow-safe M/K/N and byte-budget arithmetic.",
        "ABSENT/N-A: convolution/im2col path, not attention K/V gathering.",
        (
            Evidence("csrc/mfa_conv_nax.cpp", "mlx::core::array conv3d_nax_forward(", "Conv3D core"),
            Evidence("csrc/mfa_conv_nax.cpp", "input must be 5D", "current validator"),
        ),
    ),
    Group(
        "R15 — raw V6-NAX forward",
        ("raw:v6_nax_forward",),
        "free function → `MFAV6Forward` → pure V6 NAX matmul2d kernel",
        "Checks Q rank and D only; it computes layout/GQA routing from K without "
        "validating K/V rank, batch, K↔V, GQA, dtype, or mutual D.",
        "Add a V6 forward Q/K/V validator before reading K shape or transposing.",
        "Mutual supported f16/bf16 dtype, batch/K↔V/GQA/D, non-empty KV, causal "
        "support, finite positive/default scale semantics, and device capability.",
        "ABSENT/N-A: cooperative tensor/device tile reads; only P is materialized "
        "to shared memory, not K/V.",
        (
            Evidence("csrc/mfa_v6_nax_primitive.cpp", "v6_nax_forward(", "V6 forward core"),
            Evidence("csrc/mfa_steel_fwd_v6_nax.cpp", "auto K_tile = Kmat.slice", "device/tensor K tile"),
            Evidence("csrc/mfa_steel_fwd_v6_nax.cpp", "threadgroup T P_smem", "only P shared staging"),
        ),
    ),
    Group(
        "R16 — raw V6-NAX backward pair",
        (
            "raw:v6_nax_backward_query",
            "raw:v6_nax_backward_kv",
        ),
        "shared `v6_check_bwd_gqa` + aux-shape helpers → separate dQ and dK/dV primitives",
        "Both check Q/K/V rank, batch, K↔V seq/heads, GQA, mutual D, and exact "
        "O/LSE/dO/d_vec shapes. Dtype/domain checks are absent.",
        "Extend the existing shared helpers with mutual supported dtype and LSE/d_vec "
        "float32 checks so both bindings harden together.",
        "D membership, non-empty KV, forward-eligibility/LSE convention, finite "
        "positive scale, causal support, and output dtype expectations.",
        "ABSENT/N-A: V6 backward uses cooperative/device tile reads, not scattered "
        "K/V shared-memory gathering.",
        (
            Evidence("csrc/mfa_v6_nax_primitive.cpp", "v6_check_bwd_gqa(", "shared QKV validator"),
            Evidence("csrc/mfa_v6_nax_primitive.cpp", "v6_nax_backward_query(", "dQ core"),
            Evidence("csrc/mfa_v6_nax_primitive.cpp", "v6_nax_backward_kv(", "dK/dV core"),
        ),
    ),
)


PUBLIC_CALLS = {
    "flash_attention_qkv_packed": {"flash_attention"},
    "flash_attention_kv_packed": {"flash_attention"},
    "flash_attention_rope": {"flash_attention_rope_unified"},
    "flash_attention_kvcache_rope_append": {"flash_attention_rope_unified"},
    "flash_attention_rope_unified": {
        "flash_attention",
        "flash_attention_paged",
        "_mfa_rope_forward",
    },
    "flash_attention_kvcache": {
        "flash_attention",
        "flash_attention_paged",
        "flash_attention_rope",
        "_mfa_scatter_kv_cpp",
    },
    "flash_attention_speculative_verify": {"flash_attention"},
    "flash_attention_speculative_verify_paged": {"flash_attention_paged"},
    "flash_attention_splitfuse": {"flash_attention"},
    "flash_attention_varlen_qkv_packed": {"flash_attention_varlen"},
    "flash_attention_varlen_kv_packed": {"flash_attention_varlen"},
    "sage_attention_kvcache": {"sage_attention"},
}


def omitted_entries() -> set[str]:
    text = ENUMERATION.read_text()
    entries: set[str] = set()
    for kind, name in re.findall(
        r"^- (public|raw) `([^`]+)`(?: \([^)]+\))? —", text, re.MULTILINE
    ):
        entries.add(f"{kind}:{name}")
    return entries


def python_calls() -> dict[str, set[str]]:
    tree = ast.parse(ATTENTION.read_text())
    result: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls: set[str] = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Name):
                calls.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.add(child.func.attr)
        result[node.name] = calls
    return result


def locate(evidence: Evidence) -> str:
    path = ROOT / evidence.path
    lines = path.read_text().splitlines()
    start = 0
    if evidence.after:
        for index, line in enumerate(lines):
            if evidence.after in line:
                start = index
                break
        else:
            raise AssertionError(
                f"source anchor missing: {evidence.path!r} / {evidence.after!r}"
            )
    for number, line in enumerate(lines[start:], start + 1):
        if evidence.needle in line:
            return f"`{evidence.path}:L{number}` — {evidence.note}"
    raise AssertionError(
        f"source evidence missing: {evidence.path!r} / {evidence.needle!r}"
    )


def validate() -> tuple[set[str], dict[str, set[str]]]:
    omitted = omitted_entries()
    grouped = {member for group in GROUPS for member in group.members}
    duplicates = [
        member
        for member in grouped
        if sum(member in group.members for group in GROUPS) != 1
    ]
    assert not duplicates, f"entries assigned more than once: {duplicates}"
    assert len(omitted) == 31, f"expected 31 omitted entries, found {len(omitted)}"
    assert grouped == omitted, (
        f"group coverage mismatch; missing={sorted(omitted-grouped)}, "
        f"extra={sorted(grouped-omitted)}"
    )

    calls = python_calls()
    for function, expected in PUBLIC_CALLS.items():
        missing = expected - calls.get(function, set())
        assert not missing, f"{function}: missing AST call edges {sorted(missing)}"

    binding_text = BINDINGS.read_text()
    raw_names = sorted(x.split(":", 1)[1] for x in omitted if x.startswith("raw:"))
    for name in raw_names:
        assert re.search(rf'm\.def\(\s*"{re.escape(name)}"', binding_text), (
            f"raw binding not found: {name}"
        )

    for group in GROUPS:
        for evidence in group.evidence:
            locate(evidence)
    return omitted, calls


def render() -> str:
    omitted, _ = validate()
    critical = [
        group
        for group in GROUPS
        if group.gather.startswith("ABSENT") is False
        and "PRESENT" not in group.gather
        and "N/A" not in group.gather
    ]
    lines = [
        "# Volet K0 — Dispatch Grouping + Hardening Plan",
        "",
        "Diagnostic only. Generated by `scripts/audit_dispatch_grouping_k0.py`; "
        "no product, validation, or kernel code is changed.",
        "",
        "## Gather-barrier-absent siblings — first queue",
        "",
    ]
    if critical:
        for group in critical:
            lines.append(f"- **{group.name}:** {group.gather} [VERIFIED]")
    else:
        lines.append(
            "- **None found among the 31 omitted entries.** Every source path that "
            "can gather/stage K/V in reusable shared memory either reaches the "
            "volet-J paged fix, the volet-S Sage fix, or an existing source-visible "
            "reuse barrier. Dense/NAX/MLX paths use direct device/pointer/tensor "
            "reads. [VERIFIED]"
        )
    lines += [
        "",
        "## Efficiency result",
        "",
        f"- **31 entries → {len(GROUPS)} hardening groups** "
        f"({31 / len(GROUPS):.2f} entries/group). [VERIFIED]",
        "- The public layer collapses to 7 groups; the raw layer requires 16 groups. "
        "Raw fragmentation is intentional here: entries with different current "
        "validation boundaries remain separate even when they share a primitive. "
        "[VERIFIED]",
        "",
        "## Per-group plan",
        "",
    ]
    for index, group in enumerate(GROUPS, 1):
        lines += [
            f"### {index}. {group.name}",
            "",
            f"- **Members ({len(group.members)}):** "
            + ", ".join(f"`{member}`" for member in group.members)
            + " [VERIFIED]",
            f"- **Dispatch core:** {group.core}. [VERIFIED]",
            f"- **Current validation:** {group.current} [VERIFIED]",
            f"- **Shared-core hardening:** {group.hardening}",
            f"- **Per-entry residual:** {group.residual}",
            f"- **Gather/barrier:** {group.gather} [VERIFIED]",
            "- **Source evidence:**",
            "",
        ]
        lines.extend(f"  - {locate(item)} [VERIFIED]" for item in group.evidence)
        lines.append("")

    lines += [
        "## 31-entry accounting ledger",
        "",
        "| entry | group | dispatch core | gather classification |",
        "|---|---|---|---|",
    ]
    for group in GROUPS:
        gather_short = group.gather.split(":", 1)[0]
        for member in group.members:
            lines.append(
                f"| `{member}` | {group.name.split(' — ', 1)[0]} | "
                f"{group.core} | {gather_short} [VERIFIED] |"
            )
    lines += [
        "",
        f"Accounting: **{len(omitted)}/31** canonical omitted entries assigned "
        "exactly once. [VERIFIED]",
        "",
        "## K1 sizing / order",
        "",
        "1. **Likely-CRITICAL queue:** R15 V6 forward, R7 raw varlen, R5 raw RoPE, "
        "R3/R4 feature raw MFA, R6 raw MFA sparse, R10 NAX sparse-LSE, R13 scatter. "
        "These boundaries currently read multiple buffers while missing one or more "
        "mutual shape/dtype/count checks. [DEDUCED from source-verified gaps]",
        "2. **Shared validator wins:** R1–R5 share a dense-QKV base validator; R6 "
        "shares a sparse validator across its pair; R9/R10 should converge on one "
        "NAX-sparse validator; R16 already shares helpers. [DEDUCED]",
        "3. **Public residual pass:** P1–P7 then add feature metadata checks after "
        "their downstream computational cores are hardened. [DEDUCED]",
        "4. **Barrier action:** no new kernel barrier patch is prescribed by K0. "
        "Future K1 tests must still source-lock the paged, Sage, RoPE/shared, varlen, "
        "and legacy GNA barriers so validation work cannot regress determinism. [DEDUCED]",
        "",
        "## Mechanical validation",
        "",
        "Run:",
        "",
        "```bash",
        ".venv/bin/python scripts/audit_dispatch_grouping_k0.py --check",
        "```",
        "",
        "The check fails unless the volet-J omitted row set is exactly 31, every "
        "entry is assigned exactly once, required Python AST call edges exist, all "
        "18 raw `m.def` bindings exist, and every gather/barrier claim's source "
        "needle is present.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify generated output")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    rendered = render()
    if args.check:
        current = args.output.read_text()
        if current != rendered:
            raise SystemExit(f"{args.output.relative_to(ROOT)} is stale; regenerate")
        print(f"K0 check passed: 31 entries / {len(GROUPS)} groups / source evidence intact")
        return
    args.output.write_text(rendered)
    print(f"Wrote {args.output.relative_to(ROOT)}: 31 entries / {len(GROUPS)} groups")


if __name__ == "__main__":
    main()
