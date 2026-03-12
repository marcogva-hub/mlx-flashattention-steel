/// mfa_sage_fwd.hpp — SageAttention forward kernel header.
///
/// SageAttention-style attention for Apple Silicon:
///   - Q and K are stored as int8 (quantized per STEEL-tile block)
///   - Metal kernel loads int8 and dequantizes to fp16 in threadgroup memory
///   - The simdgroup_matrix GEMM operates on fp16 tiles (no int8 GEMM hardware)
///   - V stays fp16; P@V is unchanged from standard STEEL forward
///   - Speedup comes from 2× reduced device→threadgroup bandwidth for Q@K^T
///
/// This is inference-only (no backward pass in v1.2.0).

#pragma once

#include "shader_cache.hpp"
#include "mfa_steel_fwd.hpp"   // SteelBlockConfig, select_steel_block_config
#include <cstdint>
#include <string>

namespace mlx_mfa {

/// Parameters passed from C++ to the Metal Sage kernel.
/// Layout MUST exactly match MFaSageParams in the Metal source string.
///
/// Identical prefix to MFASteelParams (B through window_left).
/// Sage-specific fields appended at end.
///
/// CP2: Q is now fp16 (not int8). Q_scale eliminated — no Q quantize dispatch.
struct MFASageParams {
    // ── MFASteelParams-compatible prefix (same layout) ─────────────────────
    int B, H, D;
    int qL, kL;
    int gqa_factor;
    float scale;
    int NQ, NK;
    int NQ_aligned;
    int NK_aligned;
    int qL_rem;
    int kL_rem;
    int qL_off;
    // RoPE fields (kept for struct alignment; unused in Sage kernel)
    int rope_q_base;
    int rope_cos_stride;
    int64_t Q_strides[3];     // [B,H,N] fp16 Q strides (element units)
    int64_t K_strides[3];     // [B,H,S] int8 K strides (element units)
    int64_t V_strides[3];     // [B,H_kv,S] fp16 V strides
    int64_t O_strides[3];     // [B,H,N] fp16 O strides
    int64_t L_strides[2];     // [B,H] f32 L strides
    float softcap;            // 0.0 = disabled
    int   has_alibi;          // 0 = disabled (always 0 in Sage)
    int   window_left;        // -1 = disabled; >=0 = left radius (tokens)
    int   window_right;       // -1 = disabled; >=0 = right radius (tokens)
    // ── Sage-specific scale index strides ─────────────────────────────────
    // K_scale: [B, H_kv, NK_blocks, 1]  (one float per K-tile)
    int NQ_blocks;            // ceil(qL / BQ)  (kept for symmetry)
    int NK_blocks;            // ceil(kL / BK)
    int k_scale_stride_b;     // H_kv * NK_blocks
    int k_scale_stride_h;     // NK_blocks  (stride for head dim in K_scale)
};

/// Generate the complete Metal shader source for the SageAttention forward kernel.
/// The source defines the kernel function "mlx_mfa_sage_attention".
///
/// Supports: D ∈ {64, 128, 256}, dtype ∈ {f16, bf16}, causal or full,
///           GQA (gqa_factor >= 1). No RoPE/ALiBi/window/sparse in v1.2.0.
std::string generate_sage_forward_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
