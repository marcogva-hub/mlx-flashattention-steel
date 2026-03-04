/// mfa_steel_fwd.hpp  –  STEEL-style cooperative forward attention kernel.
///
/// Replaces the ccv-derived forward kernel for better performance:
///  • Cooperative K/V tile loading into threadgroup SRAM (like MLX SDPA)
///  • All threads collaboratively load each K/V tile (no per-warp repetition)
///  • 128-bit coalesced vectorized reads vs. per-lane simdgroup_matrix::load
///
/// The kernel function name in the generated Metal source is "mlx_mfa_attention".

#pragma once

#include "shader_cache.hpp"
#include <cstdint>
#include <string>

namespace mlx_mfa {

/// Parameters passed from C++ to the Metal kernel.
/// Layout MUST exactly match MFASteelParams in the Metal source string.
///
/// C++ type → Metal type mapping:
///   int      → int   (4 bytes)
///   float    → float (4 bytes)
///   int64_t  → long  (8 bytes on both sides)
struct MFASteelParams {
    // Tensor dimensions
    int B, H, D;
    int qL, kL;
    int gqa_factor;    // kv_heads: q_heads / kv_heads (1 for standard MHA)
    float scale;       // attention scale factor
    // Block counts
    int NQ, NK;        // total threadgroup blocks along Q / K
    int NQ_aligned;    // last aligned (full) Q block index = NQ - (qL%BQ ? 1 : 0)
    int NK_aligned;    // last aligned (full) K block index = NK - (kL%BK ? 1 : 0)
    // Boundary remainders
    int qL_rem;        // elements in last Q block = qL % BQ  (0 if aligned)
    int kL_rem;        // elements in last K block = kL % BK
    int qL_off;        // query sequence offset for cross-attention (0 for self-attn)
    // Strides: [B, H, S] for Q/K/V/O  (D=1 stride is implicit)
    int64_t Q_strides[3];
    int64_t K_strides[3];
    int64_t V_strides[3];
    int64_t O_strides[3];
    // L (logsumexp) strides: [B, H]  (qL=1 stride is implicit)
    int64_t L_strides[2];
};

/// Select BQ/BK/BD/WM/WN based on head_dim and precision.
/// Returns a struct used both for dispatch sizing and kernel source generation.
struct SteelBlockConfig {
    int BQ;     // query tile rows
    int BK;     // key/value tile rows
    int BD;     // head dimension (== full head_dim, not a sub-tile)
    int WM;     // SIMD groups along query (= total warps since WN=1)
    int WN;     // SIMD groups along key (always 1 for attention)
    int PAD;    // threadgroup bank-conflict padding per row
};

SteelBlockConfig select_steel_block_config(int head_dim, bool is_low_prec);

/// Generate the complete Metal shader source for the STEEL-style forward kernel.
/// The source defines the kernel function "mlx_mfa_attention".
std::string generate_steel_forward_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
