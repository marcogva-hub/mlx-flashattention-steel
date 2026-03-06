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
    bool has_rope;           // RoPE fusion: rotary_cos/sin at last two inputs
    bool rope_interleaved;   // true = LLaMA (d*2,d*2+1); false = GPT-NeoX (d,d+D/2)
    int  cache_seqlens;      // Q sequence offset for RoPE (= KV cache length, 0 otherwise)
    float softcap;       // 0.0 = disabled; >0 → tanh(S/cap)*cap before softmax
    bool has_alibi;      // false = disabled; alibi_slopes at last input
    int  window_left;    // -1 = disabled; >=0 = sliding window left radius (tokens)
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
    float softcap = 0.0f,
    int  window_left = -1,
    std::optional<mlx::core::StreamOrDevice> stream = std::nullopt);

/// Forward pass with ALiBi per-head position biases.
/// alibi_slopes: float32 [H] (one scalar slope per query head).
mlx::core::array mfa_attention_alibi_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& alibi_slopes,
    float scale,
    bool causal,
    std::optional<mlx::core::StreamOrDevice> stream = std::nullopt);

/// Forward pass with in-kernel RoPE fusion.
/// rotary_cos / rotary_sin: float32 [max_seq_len, D/2].
/// cache_seqlens: position of Q token 0 in the full sequence (KV cache length).
/// interleaved: true = LLaMA pairs (d*2, d*2+1); false = GPT-NeoX (d, d+D/2).
mlx::core::array mfa_attention_rope_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& rotary_cos,
    const mlx::core::array& rotary_sin,
    float scale,
    bool causal,
    int cache_seqlens,
    bool interleaved = true,
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

// =========================================================================
// MFASteelBwdDQ Primitive (STEEL native backward dQ)
// =========================================================================

/// STEEL native backward dQ kernel.
///
/// delta (= scale * rowsum(O⊙dO)) is precomputed by the caller as an MLX
/// lazy array and passed as inputs[6].  The kernel reads it directly.
///
/// Inputs  (7):  Q [B,H,N,D], K [B,H,S,D], V [B,H,S,D],
///               O [B,H,N,D]  (unused slot kept for API symmetry),
///               L [B,H,N],   dO [B,H,N,D],  delta [B,H,N] (float32)
/// Outputs (1):  dQ [B,H,N,D]  (same dtype as Q)
class MFASteelBwdDQ : public mlx::core::Primitive {
 public:
  using Params = MFAttention::Params;

  explicit MFASteelBwdDQ(mlx::core::Stream stream, Params params)
      : mlx::core::Primitive(stream), params_(params) {}

  const char* name() const override { return "MFASteelBwdDQ"; }

  void eval_cpu(
      const std::vector<mlx::core::array>&,
      std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFASteelBwdDQ: CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override;

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto* o = dynamic_cast<const MFASteelBwdDQ*>(&other);
    if (!o) return false;
    return params_.head_dim == o->params_.head_dim &&
           params_.scale    == o->params_.scale    &&
           params_.causal   == o->params_.causal;
  }

 private:
  Params params_;
};

// =========================================================================
// MFASteelBwdDKV Primitive (STEEL native backward dK/dV)
// =========================================================================

/// STEEL native backward dK/dV kernel.
///
/// delta (= scale * rowsum(O⊙dO)) is precomputed by the caller and passed
/// as inputs[5].  The kernel reads it; dO is at inputs[6].
///
/// Inputs  (7):  Q [B,H,N,D], K [B,H,S,D], V [B,H,S,D],
///               O [B,H,N,D]  (unused slot kept for API symmetry),
///               L [B,H,N],   delta [B,H,N] (float32),  dO [B,H,N,D]
/// Outputs (2):  dK [B,H,S,D],  dV [B,H,S,D]  (same dtype as K/V)
class MFASteelBwdDKV : public mlx::core::Primitive {
 public:
  using Params = MFAttention::Params;

  explicit MFASteelBwdDKV(mlx::core::Stream stream, Params params)
      : mlx::core::Primitive(stream), params_(params) {}

  const char* name() const override { return "MFASteelBwdDKV"; }

  void eval_cpu(
      const std::vector<mlx::core::array>&,
      std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFASteelBwdDKV: CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override;

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto* o = dynamic_cast<const MFASteelBwdDKV*>(&other);
    if (!o) return false;
    return params_.head_dim == o->params_.head_dim &&
           params_.scale    == o->params_.scale    &&
           params_.causal   == o->params_.causal;
  }

 private:
  Params params_;
};

// =========================================================================
// MFAVarlenAttention — STEEL varlen forward primitive
// =========================================================================
// Inputs: Q(0), K(1), V(2), cu_seqlens_q(3), cu_seqlens_k(4), tile_offsets(5)
// Outputs: O(0), L(1)
class MFAVarlenAttention : public mlx::core::Primitive {
 public:
  struct Params {
    float scale;
    bool  causal;
    int   head_dim;
  };

  MFAVarlenAttention(mlx::core::Stream stream, Params p)
      : mlx::core::Primitive(stream), params_(p) {}

  const char* name() const override { return "MFAVarlenAttention"; }

  void eval_cpu(
      const std::vector<mlx::core::array>&,
      std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAVarlenAttention: CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override;

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto* o = dynamic_cast<const MFAVarlenAttention*>(&other);
    if (!o) return false;
    return params_.head_dim == o->params_.head_dim &&
           params_.scale    == o->params_.scale    &&
           params_.causal   == o->params_.causal;
  }

 private:
  Params params_;
};

// Free function: dispatches the varlen kernel.
// tile_offsets[s] = cumulative Q-tile count for seqs 0..s-1 (computed in Python).
// Returns (O, L) pair.
std::pair<mlx::core::array, mlx::core::array> mfa_attention_varlen_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& cu_seqlens_q,
    const mlx::core::array& cu_seqlens_k,
    const mlx::core::array& tile_offsets,
    float scale,
    bool  causal,
    mlx::core::Stream stream);

// =========================================================================
// MFAPagedSteelForward — kernel-level paged KV (Track FD)
// =========================================================================

/// Primitive that runs the paged STEEL forward attention kernel.
///
/// Inputs (eval_gpu order):
///   0: Q          [B, H, N, D]
///   1: k_pool     [num_blocks, block_size, H_kv, D]
///   2: v_pool     [num_blocks, block_size, H_kv, D]
///   3: block_table [B, max_blocks] int32
///   4: seq_lens   [B] int32
///
/// Outputs:
///   0: O   [B, H, N, D]
///   1: L   [B, H, N] float32 (log2-domain logsumexp)
class MFAPagedSteelForward : public mlx::core::Primitive {
 public:
  struct Params {
    int   head_dim;
    float scale;
    bool  causal;
    int   window_left;   // -1 = disabled; >=0 = sliding window left radius
    int   block_size;    // tokens per page block
  };

  MFAPagedSteelForward(mlx::core::Stream stream, Params p)
      : mlx::core::Primitive(stream), params_(p) {}

  const char* name() const override { return "MFAPagedSteelForward"; }

  void eval_cpu(
      const std::vector<mlx::core::array>&,
      std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAPagedSteelForward: CPU evaluation not supported");
  }

  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override;

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto* o = dynamic_cast<const MFAPagedSteelForward*>(&other);
    if (!o) return false;
    return params_.head_dim    == o->params_.head_dim    &&
           params_.scale       == o->params_.scale       &&
           params_.causal      == o->params_.causal      &&
           params_.window_left == o->params_.window_left &&
           params_.block_size  == o->params_.block_size;
  }

 private:
  Params params_;
};

/// Free function: validate inputs and create paged forward MLX arrays.
/// Returns (O, L) pair.
std::pair<mlx::core::array, mlx::core::array> mfa_paged_steel_forward(
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    float scale,
    bool  causal,
    int   window_left,
    int   block_size,
    mlx::core::Stream stream);

}  // namespace mlx_mfa
