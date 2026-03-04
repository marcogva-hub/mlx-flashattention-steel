/// mfa_shader_gen.hpp  –  Maps ShaderCache::KernelKey → Metal shader source string.
///
/// This is the bridge between the MLX-side KernelKey and the ccv-derived
/// AttentionKernel source generator that lives in csrc/mfa/.

#pragma once

#include "shader_cache.hpp"
#include <cstdint>
#include <string>

namespace mlx_mfa {

/// Resolved blocking configuration (tile sizes) for a given key.
/// Matches exactly what generate_attention_source() uses internally.
struct MFABlockConfig {
  unsigned short block_q;  // parallelization dim (rows of Q per tile)
  unsigned short block_k;  // traversal dim (KV columns per step)
  unsigned short block_d;  // head sub-tile (for 3D blocking when D=256)
};

/// Resolve the MFA blocking configuration for the given parameters.
/// low_prec_inputs: Q/K/V are FP16 or BF16.
/// low_prec_inter:  S/P accumulated in FP16/BF16 (mixed precision; false for Phase 1).
MFABlockConfig resolve_block_config(
    int head_dim, bool is_m3_plus, bool low_prec_inter, bool low_prec_inputs);

/// Generate the Metal kernel source string for the given key.
/// The returned string can be compiled by shader_cache.mm.
std::string generate_attention_source(const ShaderCache::KernelKey& key);

/// CPU-side mirror of the unified MFAParams Metal struct (emitted by
/// AttentionKernel::createConstants()).  Must stay bit-for-bit identical
/// to the Metal struct layout for all kernel types (forward + backward).
///
/// Forward kernels leave dO/dQ/dK/dV strides as 0.
/// Backward kernels populate all relevant strides.
/// L and D do not appear here: their pointer offsets are computed inline
/// in the kernel via "(gid.z * Hq + gid.y) * R" without a stride field.
struct MFAParams {
    uint32_t R;                 ///< output sequence length (rows = N)
    uint32_t C;                 ///< input sequence length (columns = S)
    uint32_t Hq;                ///< number of query heads
    uint32_t H_Hk_ratio;        ///< Hq / Hk  (1 for MHA / GQA ratio)
    float    dot_product_scale; ///< scale * log2(e)  (kernel uses exp2)
    uint32_t causal;            ///< 1 = apply causal mask, 0 = none
    // Forward batch strides (elements between consecutive batches):
    uint32_t Q_batch_stride;
    uint32_t K_batch_stride;
    uint32_t V_batch_stride;
    uint32_t O_batch_stride;
    // Backward batch strides (zero for forward kernels):
    uint32_t dO_batch_stride;
    uint32_t dQ_batch_stride;
    uint32_t dK_batch_stride;
    uint32_t dV_batch_stride;
};

} // namespace mlx_mfa
