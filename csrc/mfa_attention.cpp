#include "mfa_attention.hpp"
#include "shader_cache.hpp"

#include <mlx/utils.h>

#include <cassert>
#include <stdexcept>

namespace mlx_mfa {

// =========================================================================
// Constructor
// =========================================================================

MFAttention::MFAttention(mlx::core::Stream stream, Params params)
    : mlx::core::Primitive(stream), params_(params) {}

// =========================================================================
// Block parameter resolution (MFA v2 blocking tables)
// =========================================================================

MFAttention::BlockParams MFAttention::resolve_block_params() const {
  // Values from MFA v2 blocking tables.
  // Aspect ratio intentionally deformed:
  //   parallelization dim: 16-32 (small, many tiles)
  //   traversal dim: 80-128 (large, amortize spilling)
  //
  // TODO(Phase 1.2): Import actual tables from ccv source.

  BlockParams bp;
  switch (params_.head_dim) {
    case 64:
      bp = {32, 64, 64, 4};
      break;
    case 128:
      bp = {16, 128, 128, 4};
      break;
    case 256:
      bp = {16, 80, 128, 8};  // 3D blocking: D split into 2 sub-tiles
      break;
    default:
      throw std::runtime_error(
          "MFAttention: unsupported head_dim=" +
          std::to_string(params_.head_dim));
  }
  return bp;
}

// =========================================================================
// Forward pass (eval_gpu)
// =========================================================================

void MFAttention::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {
  assert(inputs.size() == 3);

  auto& q = inputs[0];  // [B, H, N, D]
  auto& k = inputs[1];  // [B, H, S, D]
  auto& v = inputs[2];  // [B, H, S, D]

  int B = q.shape(0);
  int H = q.shape(1);
  int N = q.shape(2);
  int D = q.shape(3);
  int S = k.shape(2);

  // Allocate outputs
  auto& out = outputs[0];       // [B, H, N, D]
  auto& logsumexp = outputs[1]; // [B, H, N]

  out.set_data(mlx::core::allocator::malloc_or_wait(out.nbytes()));
  logsumexp.set_data(
      mlx::core::allocator::malloc_or_wait(logsumexp.nbytes()));

  auto bp = resolve_block_params();

  // -------------------------------------------------------------------
  // TODO(Phase 1.2-1.3): Metal kernel dispatch
  //
  // 1. Get/compile pipeline from ShaderCache (JIT shader)
  // 2. Get command encoder from MLX metal device
  // 3. Set buffer args (Q, K, V, O, L, params)
  // 4. Dispatch threadgroups:
  //      grid: (ceil(N/block_q), B*H, 1)
  //      threadgroup: (n_warps * 32, 1, 1)
  // -------------------------------------------------------------------

  (void)B; (void)H; (void)N; (void)D; (void)S; (void)bp;

  throw std::runtime_error(
      "MFAttention::eval_gpu: kernel not yet implemented (Phase 1.2-1.3)");
}

// =========================================================================
// Backward pass (Phase 3)
// =========================================================================

std::vector<mlx::core::array> MFAttention::vjp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mlx::core::array>& outputs) {
  (void)primals; (void)cotangents; (void)argnums; (void)outputs;
  throw std::runtime_error(
      "MFAttention::vjp: backward pass not yet implemented (Phase 3)");
}

// =========================================================================
// Equivalence
// =========================================================================

bool MFAttention::is_equivalent(const mlx::core::Primitive& other) const {
  auto* o = dynamic_cast<const MFAttention*>(&other);
  if (!o) return false;
  return params_.head_dim == o->params_.head_dim &&
         params_.scale == o->params_.scale &&
         params_.causal == o->params_.causal;
}

// =========================================================================
// Free function: mfa_attention_forward
// =========================================================================

mlx::core::array mfa_attention_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    float scale,
    bool causal,
    std::optional<mlx::core::StreamOrDevice> stream) {
  auto s = stream.has_value()
      ? mlx::core::to_stream(stream.value())
      : mlx::core::default_stream(q.device());

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument("MFA: expected 4D inputs [B, H, N, D]");
  }

  int D = q.shape(3);
  if (D != 64 && D != 128 && D != 256) {
    throw std::invalid_argument(
        "MFA: head_dim must be 64, 128, or 256, got " + std::to_string(D));
  }

  bool use_bf16_emu = (q.dtype() == mlx::core::bfloat16);

  MFAttention::Params params{D, scale, causal, use_bf16_emu};

  auto out_shape = q.shape();                           // [B, H, N, D]
  std::vector<int> lse_shape = {
      q.shape(0), q.shape(1), q.shape(2)};             // [B, H, N]

  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAttention>(s, params),
      {q, k, v});

  return outputs[0];
}

}  // namespace mlx_mfa
