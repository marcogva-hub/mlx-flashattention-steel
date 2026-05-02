/// V6 NAX forward kernel — MPP cooperative tensor attention for M5+.
///
/// Generates an MSL 4.0 + MetalPerformancePrimitives kernel that performs
/// causal-or-not flash-attention forward using:
///   - mpp::tensor_ops::matmul2d for Q@K^T and P@V GEMMs
///   - cooperative_tensor for in-register accumulator
///   - online softmax in log2 domain (matches MLX's steel_attention_nax.h)
///
/// Strict support matrix (Phase 1):
///   D ∈ {64, 128}, FP16/BF16, dense forward only (no mask/bias/softcap/window).
///   M5+ (Apple GPU family 10) only.

#pragma once

#include "shader_cache.hpp"
#include <string>

namespace mlx_mfa {

/// V6 NAX kernel parameters (passed to GPU).
struct MFAV6NaxParams {
  int B;          // batch
  int H;          // query heads
  int gqa_factor; // H_q / H_kv (1 for standard MHA)
  int N;          // qL / kL  (self-attention only in v1: N_q == N_kv)
  int D;          // head_dim (64 or 128)
  float scale;    // 1/sqrt(D), pre-multiplied by log2e for log2-domain softmax
  // Strides in elements ([B, H, S]) — [D] stride is 1 (contiguous).
  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t O_strides[3];
};

/// Build the MSL 4 + MPP source string for a V6 NAX forward kernel.
/// Parametrized by head_dim (BD) and dtype.
std::string generate_steel_v6_nax_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
