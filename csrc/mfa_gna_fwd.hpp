/// mfa_gna_fwd.hpp — GNA (Generalized Neighborhood Attention) forward kernel.
///
/// Replaces the mask-based GNA path (make_gna_mask → flash_attention_sparse)
/// with an inline window check that avoids materializing the block mask.
/// Uses the same STEEL V2 tiling, loading, and accumulation pattern.
///
/// Window semantics per dimension d (matches make_gna_mask in masks.py):
///   group_base = (pos // stride[d]) * stride[d]
///   win_lo = group_base - (window[d] - stride[d]) // 2
///   win_hi = group_base + stride[d] + (window[d] - stride[d] + 1) // 2
///   Both clamped to [0, seq_shape[d]).
///
/// D=128 only (constraint from prompt — all video DiT models use D=128).

#pragma once

#include "shader_cache.hpp"
#include <cstdint>
#include <string>

namespace mlx_mfa {

/// GNA-specific params, passed to the Metal kernel at buffer(6).
/// Separate from MFASteelParams to keep the base struct unchanged.
struct MFAGNAParams {
    // Sequence 3D shape: N = dim0 * dim1 * dim2
    int dim0, dim1, dim2;
    // Window size per dimension
    int win0, win1, win2;
    // Stride per dimension
    int str0, str1, str2;
    // Precomputed for division avoidance in shader:
    // inv_dim2 = dim2, inv_dim12 = dim1 * dim2 (for linear → 3D conversion)
    int dim12;  // dim1 * dim2
};

/// Generate JIT Metal source for the GNA forward kernel.
/// Reuses STEEL V2 tiling/accumulation with inline GNA window check.
std::string generate_gna_forward_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
