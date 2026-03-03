#pragma once

#include <mlx/mlx.h>
#include <optional>

namespace mlx_mfa {

/// MLX Primitive implementing Metal Flash Attention.
///
/// Forward: Tiled attention with online softmax (MFA algorithm).
/// Backward: Two separate kernels for dQ and dK/dV (7 GEMMs variant
///           that avoids FP32 atomics, per MFA design).
///
/// Supported head_dim: 64, 128, 256.
/// Supported dtypes: float16, bfloat16, float32.
class MFAttention : public mlx::core::Primitive {
 public:
  struct Params {
    int head_dim;       // D: 64, 128, or 256
    float scale;        // Usually 1/sqrt(D)
    bool causal;        // Causal (autoregressive) masking
    bool use_bf16_emu;  // BF16 emulation for M1/M2 (no native BF16)
  };

  explicit MFAttention(mlx::core::Stream stream, Params params);

  /// Forward pass.
  /// inputs:  Q [B,H,N,D], K [B,H,S,D], V [B,H,S,D]
  /// outputs: O [B,H,N,D], L [B,H,N] (logsumexp for backward)
  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override;

  void print(std::ostream& os) override {
    os << "MFAttention(D=" << params_.head_dim
       << ", scale=" << params_.scale
       << ", causal=" << params_.causal << ")";
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override;

  /// Backward pass (Phase 3).
  /// MFA 7-GEMM algorithm: Kernel1=dQ, Kernel2=dK/dV.
  std::vector<mlx::core::array> vjp(
      const std::vector<mlx::core::array>& primals,
      const std::vector<mlx::core::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mlx::core::array>& outputs) override;

 private:
  Params params_;

  struct BlockParams {
    int block_q;   // Tile size along Q sequence dim
    int block_k;   // Tile size along K sequence dim
    int block_d;   // Tile size along head_dim (3D blocking for D=256)
    int n_warps;   // SIMD groups per threadgroup
  };

  BlockParams resolve_block_params() const;
};

// ---- Free function exposed via nanobind ---- //

mlx::core::array mfa_attention_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    float scale,
    bool causal,
    std::optional<mlx::core::StreamOrDevice> stream = std::nullopt);

}  // namespace mlx_mfa
