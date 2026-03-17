/// mfa_steel_gna_fwd.hpp  –  Generalized Neighborhood Attention forward kernel.
///
/// GNA unifies sliding window, blocked (Swin), and strided attention via a
/// multi-dimensional window + stride parameter. Tokens are laid out in a
/// spatial/temporal grid (T, H, W) and each query attends only to keys
/// within its stride-group's window.
///
/// Reference: "Generalized Neighborhood Attention" (Hassani et al., 2025).

#pragma once

#include "shader_cache.hpp"
#include "mfa_steel_fwd.hpp"  // SteelBlockConfig, append_* helpers
#include <cstdint>
#include <sstream>
#include <string>

namespace mlx_mfa {

/// Parameters passed from C++ to the Metal GNA kernel.
/// Layout MUST exactly match MFAGNAParams in the Metal source string.
struct MFAGNAParams {
    int B, H, D;
    int seq_len;         // prod(seq_shape)
    float scale;
    int gqa_factor;
    // Multi-dimensional sequence layout
    int ndim;            // 2 or 3
    int seq_shape[3];    // (T, H, W) — padded with 1 if ndim < 3
    int seq_strides[3];  // linear strides: e.g. (H*W, W, 1)
    // GNA window parameters
    int window_size[3];
    int stride[3];
    int window_volume;   // prod(window_size)
    // Tiling
    int NQ, NK;
    int NQ_aligned, NK_aligned;
    int qL_rem, kL_rem;
    // Tensor strides: [B, H, S] for Q/K/V/O (D stride is implicit=1)
    int64_t Q_strides[3];
    int64_t K_strides[3];
    int64_t V_strides[3];
    int64_t O_strides[3];
    // L (logsumexp) strides: [B, H] (seq stride is implicit=1)
    int64_t L_strides[2];
};

/// Select BQ/BK block config for the GNA kernel.
/// Reuses STEEL V1 defaults: BQ=32, BK=16 for D=128; BQ=32, BK=32 for D=64.
SteelBlockConfig select_gna_block_config(int head_dim, bool is_low_prec);

/// Generate the complete Metal shader source for the GNA forward kernel.
/// The source defines the kernel function "mlx_mfa_gna_attention".
std::string generate_gna_forward_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
