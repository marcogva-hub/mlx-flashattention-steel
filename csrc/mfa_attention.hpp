/// mfa_attention.hpp — MFAttention MLX Primitive declaration.
///
/// MFAttention wraps the Metal Flash Attention GPU kernel as an MLX
/// Primitive, enabling lazy evaluation and autograd integration.
///
/// eval_gpu() dispatches either the STEEL cooperative kernel (f16/bf16,
/// D=64/128/256) or the ccv-derived kernel (f32 fallback).  Backward
/// gradients are computed via Python mx.vjp on the SDPA fallback — see
/// attention.py:_make_mfa_custom for the reason the C++ vjp is bypassed.

#pragma once

#include <mlx/mlx.h>
#include <mlx/primitives.h>
#include <optional>
#include <stdexcept>

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
    bool has_block_mask; // Block-sparse: 4th input is uchar mask [NQ_tiles, NK_tiles]
  };

  explicit MFAttention(mlx::core::Stream stream, Params params);

  const char* name() const override { return "MFAttention"; }

  /// CPU evaluation is not supported — MFA is GPU-only.
  void eval_cpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override {
    throw std::runtime_error("MFAttention: CPU evaluation not supported");
  }

  /// Forward pass (GPU).
  /// inputs:  Q [B,H,N,D], K [B,H,S,D], V [B,H,S,D]  (+ optional block_mask)
  /// outputs: O [B,H,N,D], L [B,H,N] (logsumexp for backward)
  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override;

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
};

// ---- Free functions exposed via nanobind ---- //

mlx::core::array mfa_attention_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    float scale,
    bool causal,
    std::optional<mlx::core::StreamOrDevice> stream = std::nullopt);

/// Block-sparse forward pass.
/// block_mask: uint8 array [NQ_tiles, NK_tiles].  1 = compute, 0 = skip.
mlx::core::array mfa_attention_sparse_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& block_mask,
    float scale,
    bool causal,
    std::optional<mlx::core::StreamOrDevice> stream = std::nullopt);

/// Block-sparse forward returning (O, L) where L is logsumexp [B, H, N].
/// Used by the native sparse backward to avoid gradient checkpointing.
std::vector<mlx::core::array> mfa_attention_sparse_forward_with_lse(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& block_mask,
    float scale,
    bool causal,
    std::optional<mlx::core::StreamOrDevice> stream = std::nullopt);

// =========================================================================
// MFABackwardQuery Primitive (Phase 3)
// =========================================================================

/// First backward kernel: computes dQ and the intermediate D buffer.
///
/// D (delta) = scale * rowsum(O ⊙ dO) is computed inside the Metal kernel
/// from O and dO (via COMPUTE_D / computeD()), then written to the D output
/// buffer.  backwardKeyValue reads this buffer to compute softmax derivatives.
///
/// Inputs  (6):  Q [B,H,N,D], K [B,H,S,D], V [B,H,S,D],
///               O [B,H,N,D], L [B,H,N], dO [B,H,N,D]
/// Outputs (2):  dQ [B,H,N,D]  (same dtype as Q),
///               D_computed [B,H,N]  (float32, = scale * rowsum(O⊙dO))
class MFABackwardQuery : public mlx::core::Primitive {
 public:
  using Params = MFAttention::Params;

  explicit MFABackwardQuery(mlx::core::Stream stream, Params params)
      : mlx::core::Primitive(stream), params_(params) {}

  const char* name() const override { return "MFABackwardQuery"; }

  void eval_cpu(
      const std::vector<mlx::core::array>&,
      std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFABackwardQuery: CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override;

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto* o = dynamic_cast<const MFABackwardQuery*>(&other);
    if (!o) return false;
    return params_.head_dim == o->params_.head_dim &&
           params_.scale    == o->params_.scale    &&
           params_.causal   == o->params_.causal;
  }

 private:
  Params params_;
};

// =========================================================================
// MFABackwardKeyValue Primitive (Phase 3)
// =========================================================================

/// Second backward kernel: computes dK and dV using the D buffer from
/// MFABackwardQuery.  MLX's dependency graph ensures D_computed is ready
/// before this primitive executes — no manual Metal barrier needed.
///
/// Inputs  (7):  Q [B,H,N,D], K [B,H,S,D], V [B,H,S,D],
///               O [B,H,N,D], L [B,H,N],
///               D_computed [B,H,N]  (output of MFABackwardQuery),
///               dO [B,H,N,D]
/// Outputs (2):  dK [B,H,S,D]  (same dtype as K),
///               dV [B,H,S,D]  (same dtype as V)
class MFABackwardKeyValue : public mlx::core::Primitive {
 public:
  using Params = MFAttention::Params;

  explicit MFABackwardKeyValue(mlx::core::Stream stream, Params params)
      : mlx::core::Primitive(stream), params_(params) {}

  const char* name() const override { return "MFABackwardKeyValue"; }

  void eval_cpu(
      const std::vector<mlx::core::array>&,
      std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFABackwardKeyValue: CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override;

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto* o = dynamic_cast<const MFABackwardKeyValue*>(&other);
    if (!o) return false;
    return params_.head_dim == o->params_.head_dim &&
           params_.scale    == o->params_.scale    &&
           params_.causal   == o->params_.causal;
  }

 private:
  Params params_;
};

}  // namespace mlx_mfa
