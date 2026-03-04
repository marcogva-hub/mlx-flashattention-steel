#include "mfa_attention.hpp"
#include "mfa_shader_gen.hpp"
#include "mfa_steel_fwd.hpp"
#include "shader_cache.hpp"

#include <mlx/utils.h>
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <Metal/Metal.hpp>

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace mlx_mfa {

// =========================================================================
// Constructor
// =========================================================================

MFAttention::MFAttention(mlx::core::Stream stream, Params params)
    : mlx::core::Primitive(stream), params_(params) {}

// =========================================================================
// Forward pass (eval_gpu)
// =========================================================================

void MFAttention::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {
  assert(inputs.size() == 3);

  const auto& q = inputs[0]; // [B, H, N, D]
  const auto& k = inputs[1]; // [B, H, S, D]
  const auto& v = inputs[2]; // [B, H, S, D]

  int B = q.shape(0);
  int H = q.shape(1);
  int N = q.shape(2);
  int D = q.shape(3);
  int S = k.shape(2);
  int Hk = k.shape(1); // KV heads (for GQA)

  auto& out       = outputs[0]; // [B, H, N, D]
  auto& logsumexp = outputs[1]; // [B, H, N], float32

  out.set_data(mlx::core::allocator::malloc(out.nbytes()));
  logsumexp.set_data(mlx::core::allocator::malloc(logsumexp.nbytes()));

  // ── Device & dtype ──────────────────────────────────────────────────────
  auto& d = mlx::core::metal::device(stream().device);

  uint8_t dtype_code;
  if (q.dtype() == mlx::core::float16)       dtype_code = 0;
  else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
  else                                        dtype_code = 2;

  // ── Route to ccv kernel when Steel register pressure is too high.
  //  • f32 (dtype==2): simdgroup_matrix spills exceed 32 KB threadgroup limit.
  //  Note: D=256 (f16/bf16) stays on STEEL despite register pressure because ccv
  //  3D-blocking + async_copy fallback is slower than STEEL register spill on macOS 26.
  if (dtype_code == 2) {
    bool is_m3_plus = (d.get_architecture_gen() >= 15);
    const bool low_prec_inter  = false;
    const bool low_prec_inputs = false;
    auto ccv_cfg = resolve_block_config(D, is_m3_plus, low_prec_inter, low_prec_inputs);
    unsigned short bq = ccv_cfg.block_q, bk = ccv_cfg.block_k, bd = ccv_cfg.block_d;
    unsigned short nw = bq / 8;

    MFAParams fwd_p{};
    fwd_p.R                = static_cast<uint32_t>(N);
    fwd_p.C                = static_cast<uint32_t>(S);
    fwd_p.Hq               = static_cast<uint32_t>(H);
    fwd_p.H_Hk_ratio       = static_cast<uint32_t>(H / Hk);
    fwd_p.dot_product_scale = params_.scale * static_cast<float>(M_LOG2E);
    fwd_p.causal           = params_.causal ? 1u : 0u;
    fwd_p.Q_batch_stride   = static_cast<uint32_t>(H  * N * D);
    fwd_p.K_batch_stride   = static_cast<uint32_t>(Hk * S * D);
    fwd_p.V_batch_stride   = static_cast<uint32_t>(Hk * S * D);
    fwd_p.O_batch_stride   = static_cast<uint32_t>(H  * N * D);

    using KK = ShaderCache::KernelKey;
    KK ccv_key{ KK::KernelType::AttentionForward,
                D, (int)bq, (int)bk, (int)bd, (int)nw,
                params_.causal, is_m3_plus, dtype_code };
    void* raw = ShaderCache::get().get_or_compile(ccv_key, d.mtl_device());
    auto* pl  = reinterpret_cast<MTL::ComputePipelineState*>(raw);

    auto& enc = d.get_command_encoder(stream().index);
    enc.set_compute_pipeline_state(pl);
    enc.set_input_array(q,          0);
    enc.set_input_array(k,          1);
    enc.set_input_array(v,          2);
    enc.set_output_array(out,       3);
    enc.set_output_array(logsumexp, 4);
    enc.set_bytes(fwd_p,           10);

    uint32_t tiles = ((uint32_t)N + bq - 1u) / bq * (uint32_t)H * (uint32_t)B;
    enc.dispatch_threadgroups(
        MTL::Size::Make(tiles, 1, 1),
        MTL::Size::Make(nw * 32u, 1, 1));
    return;
  }

  // ── STEEL tile config (f16 / bf16) ───────────────────────────────────────
  auto cfg = select_steel_block_config(D, /*is_low_prec=*/true);
  int BQ = cfg.BQ;
  int BK = cfg.BK;
  int WM = cfg.WM;  // n_warps
  int WN = cfg.WN;
  int TGP_SIZE = WM * WN * 32;

  // ── Kernel cache key ─────────────────────────────────────────────────────
  using KK = ShaderCache::KernelKey;
  KK key{
    KK::KernelType::SteelForward,
    D,
    BQ, BK, D,   // block_d = full D (no sub-tiling in Steel)
    WM,
    params_.causal,
    false,       // is_m3_plus unused for Steel kernel
    dtype_code
  };

  void* raw_pipeline = ShaderCache::get().get_or_compile(key, d.mtl_device());
  auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw_pipeline);

  // ── Build MFASteelParams ─────────────────────────────────────────────────
  int NQ = (N + BQ - 1) / BQ;
  int NK = (S + BK - 1) / BK;
  int NQ_aligned = (N % BQ == 0) ? NQ : NQ - 1;
  int NK_aligned = (S % BK == 0) ? NK : NK - 1;

  MFASteelParams sp{};
  sp.B          = B;
  sp.H          = H;
  sp.D          = D;
  sp.qL         = N;
  sp.kL         = S;
  sp.gqa_factor = H / Hk;
  sp.scale      = params_.scale;
  sp.NQ         = NQ;
  sp.NK         = NK;
  sp.NQ_aligned = NQ_aligned;
  sp.NK_aligned = NK_aligned;
  sp.qL_rem     = (N % BQ == 0) ? BQ : (N % BQ);
  sp.kL_rem     = (S % BK == 0) ? BK : (S % BK);
  sp.qL_off     = 0; // self-attention; cross-attn would offset query start

  // Strides: [B, H, S] in elements (D=1 implicit)
  sp.Q_strides[0] = (int64_t)H * N * D;
  sp.Q_strides[1] = (int64_t)N * D;
  sp.Q_strides[2] = (int64_t)D;
  sp.K_strides[0] = (int64_t)Hk * S * D;
  sp.K_strides[1] = (int64_t)S * D;
  sp.K_strides[2] = (int64_t)D;
  sp.V_strides[0] = (int64_t)Hk * S * D;
  sp.V_strides[1] = (int64_t)S * D;
  sp.V_strides[2] = (int64_t)D;
  sp.O_strides[0] = (int64_t)H * N * D;
  sp.O_strides[1] = (int64_t)N * D;
  sp.O_strides[2] = (int64_t)D;
  // L strides: [B, H] with per-head stride = N
  sp.L_strides[0] = (int64_t)H * N;
  sp.L_strides[1] = (int64_t)N;

  // ── Dispatch ─────────────────────────────────────────────────────────────
  auto& enc = d.get_command_encoder(stream().index);
  enc.set_compute_pipeline_state(pipeline);

  // Buffers: Q=0, K=1, V=2, O=3, L=4, params=5
  enc.set_input_array(q,          0);
  enc.set_input_array(k,          1);
  enc.set_input_array(v,          2);
  enc.set_output_array(out,       3);
  enc.set_output_array(logsumexp, 4);
  enc.set_bytes(sp,               5);

  // 3D grid: (NQ, H, B) — each threadgroup handles one Q tile, one head, one batch
  enc.dispatch_threadgroups(
      MTL::Size::Make(NQ, H, B),
      MTL::Size::Make(TGP_SIZE, 1, 1));
}

// =========================================================================
// Backward pass (Phase 3) — MFAttention::vjp
// =========================================================================
//
// MFA 7-GEMM backward pass split into two primitives:
//
//   Step 1: MFABackwardQuery [Q,K,V,O,L,dO] → [dQ, D_computed]
//     • Metal kernel computes D = scale * rowsum(O⊙dO) from O and dO,
//       writes D to the D output buffer, and accumulates dQ.
//
//   Step 2: MFABackwardKeyValue [Q,K,V,O,L,D_computed,dO] → [dK, dV]
//     • Metal kernel reads D_computed (now correctly scaled), computes
//       softmax derivatives, and accumulates dK and dV.
//
// Using two primitives lets MLX's graph execution guarantee that
// D_computed is fully written before backwardKeyValue reads it —
// no manual Metal memory barrier required.

std::vector<mlx::core::array> MFAttention::vjp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mlx::core::array>& outputs) {

  const auto& q = primals[0];  // [B, H, N, D]
  const auto& k = primals[1];  // [B, H, S, D]
  const auto& v = primals[2];  // [B, H, S, D]

  // outputs[0] = O  (attention output, [B,H,N,D])
  // outputs[1] = L  (logsumexp,        [B,H,N], f32)
  const auto& O = outputs[0];
  const auto& L = outputs[1];

  // cotangents[0] = dO (gradient of O, same shape and dtype as O)
  // cotangents[1] = dL (gradient of L — always zero, safely ignored)
  const auto& dO = cotangents[0];

  // D shape: [B, H, N], always float32.
  mlx::core::Shape d_shape = {q.shape(0), q.shape(1), q.shape(2)};

  // Step 1: MFABackwardQuery → [dQ, D_computed]
  // The Metal kernel computes D = scale * rowsum(O⊙dO) internally and
  // writes it to outputs[1] (D_computed).  It does NOT read the D buffer.
  auto bwd_q = mlx::core::array::make_arrays(
      {q.shape(),         d_shape},
      {q.dtype(),  mlx::core::float32},
      std::make_shared<MFABackwardQuery>(stream(), params_),
      {q, k, v, O, L, dO});

  const auto& dQ_arr   = bwd_q[0];  // [B,H,N,D], input dtype
  const auto& D_kernel = bwd_q[1];  // [B,H,N], float32 = scale*rowsum(O⊙dO)

  // Step 2: MFABackwardKeyValue → [dK, dV]
  // D_kernel is an output of Step 1; MLX will not run this primitive
  // until D_kernel is fully materialized.
  auto bwd_kv = mlx::core::array::make_arrays(
      {k.shape(),  v.shape()},
      {k.dtype(), v.dtype()},
      std::make_shared<MFABackwardKeyValue>(stream(), params_),
      {q, k, v, O, L, D_kernel, dO});

  const auto& dK_arr = bwd_kv[0];
  const auto& dV_arr = bwd_kv[1];

  // Return only the gradients for the requested inputs (argnums order).
  // argnums maps {0→dQ, 1→dK, 2→dV}.
  const std::vector<mlx::core::array> all_grads = {dQ_arr, dK_arr, dV_arr};
  std::vector<mlx::core::array> result;
  result.reserve(argnums.size());
  for (int i : argnums) {
    result.push_back(all_grads[i]);
  }
  return result;
}

// =========================================================================
// MFABackwardQuery::eval_gpu
// =========================================================================
//
// Dispatches the backwardQuery Metal kernel.
// The kernel computes D = scale*rowsum(O⊙dO) from O and dO and writes it
// to outputs[1] (D_computed); it also writes dQ to outputs[0].
//
// Buffer assignments (AttentionOperand::bufferIndex()):
//   Q=0, K=1, V=2, O=3, L=4, D=5(output), dO=6, dQ=9(output), params=10

void MFABackwardQuery::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  assert(inputs.size()  == 6);  // Q, K, V, O, L, dO
  assert(outputs.size() == 2);  // dQ, D_computed

  const auto& q  = inputs[0];  // [B, H, N, D]
  const auto& k  = inputs[1];  // [B, H, S, D]
  const auto& v  = inputs[2];  // [B, H, S, D]
  const auto& o  = inputs[3];  // [B, H, N, D]
  const auto& l  = inputs[4];  // [B, H, N], float32
  const auto& dO = inputs[5];  // [B, H, N, D]

  int B = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  int S = k.shape(2);

  auto& dQ         = outputs[0];  // [B, H, N, D], input dtype
  auto& D_computed = outputs[1];  // [B, H, N],    float32

  dQ.set_data(mlx::core::allocator::malloc(dQ.nbytes()));
  D_computed.set_data(mlx::core::allocator::malloc(D_computed.nbytes()));

  // ── Device & dtype ─────────────────────────────────────────────────────
  auto& dev = mlx::core::metal::device(stream().device);
  bool is_m3_plus = (dev.get_architecture_gen() >= 15); // 13=M1 14=M2 15=M3 16=M4

  uint8_t dtype_code;
  if (q.dtype() == mlx::core::float16)       dtype_code = 0;
  else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
  else                                        dtype_code = 2;
  bool low_prec_inputs = (dtype_code != 2);
  const bool low_prec_inter = false; // forward blocks; see eval_gpu comment

  auto cfg = resolve_block_config(D, is_m3_plus, low_prec_inter,
                                  low_prec_inputs);
  unsigned short block_q = cfg.block_q;
  unsigned short block_k = cfg.block_k;
  unsigned short block_d = cfg.block_d;
  unsigned short n_warps = block_q / 8;

  MFAParams bw_params{};
  bw_params.R               = static_cast<uint32_t>(N);
  bw_params.C               = static_cast<uint32_t>(S);
  bw_params.Hq              = static_cast<uint32_t>(H);
  bw_params.H_Hk_ratio      = 1u;
  bw_params.dot_product_scale = params_.scale * static_cast<float>(M_LOG2E);
  bw_params.causal          = params_.causal ? 1u : 0u;
  bw_params.Q_batch_stride  = static_cast<uint32_t>(H * N * D);
  bw_params.K_batch_stride  = static_cast<uint32_t>(H * S * D);
  bw_params.V_batch_stride  = static_cast<uint32_t>(H * S * D);
  bw_params.O_batch_stride  = static_cast<uint32_t>(H * N * D);
  bw_params.dO_batch_stride = static_cast<uint32_t>(H * N * D);
  bw_params.dQ_batch_stride = static_cast<uint32_t>(H * N * D);
  bw_params.dK_batch_stride = static_cast<uint32_t>(H * S * D);
  bw_params.dV_batch_stride = static_cast<uint32_t>(H * S * D);

  using KK = ShaderCache::KernelKey;
  KK key{
    KK::KernelType::AttentionBackwardDQ,
    D, (int)block_q, (int)block_k, (int)block_d, (int)n_warps,
    params_.causal, is_m3_plus, dtype_code
  };
  void* raw = ShaderCache::get().get_or_compile(key, dev.mtl_device());
  auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw);

  auto& enc = dev.get_command_encoder(stream().index);
  enc.set_compute_pipeline_state(pipeline);
  enc.set_input_array(q,            0);
  enc.set_input_array(k,            1);
  enc.set_input_array(v,            2);
  enc.set_input_array(o,            3);
  enc.set_input_array(l,            4);
  enc.set_output_array(D_computed,  5);  // kernel WRITES D here (buffer 5)
  enc.set_input_array(dO,           6);
  enc.set_output_array(dQ,          9);
  enc.set_bytes(bw_params,         10);

  uint32_t num_q_tiles = (static_cast<uint32_t>(N) + block_q - 1u) / block_q;
  uint32_t grid_dq     = num_q_tiles
                         * static_cast<uint32_t>(H)
                         * static_cast<uint32_t>(B);
  enc.dispatch_threadgroups(
      MTL::Size::Make(grid_dq,       1, 1),
      MTL::Size::Make(n_warps * 32u, 1, 1));
}

// =========================================================================
// MFABackwardKeyValue::eval_gpu
// =========================================================================
//
// Dispatches the backwardKeyValue Metal kernel.
// Reads D_computed (inputs[5]) written by MFABackwardQuery and accumulates
// dK (outputs[0]) and dV (outputs[1]).
//
// Buffer assignments:
//   Q=0, K=1, V=2, O=3, L=4, D=5(input), dO=6, dV=7(output), dK=8(output),
//   params=10

void MFABackwardKeyValue::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  assert(inputs.size()  == 7);  // Q, K, V, O, L, D_computed, dO
  assert(outputs.size() == 2);  // dK, dV

  const auto& q          = inputs[0];
  const auto& k          = inputs[1];
  const auto& v          = inputs[2];
  const auto& o          = inputs[3];
  const auto& l          = inputs[4];
  const auto& D_computed = inputs[5];  // float32 [B,H,N], from MFABackwardQuery
  const auto& dO         = inputs[6];

  int B = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  int S = k.shape(2);

  auto& dK = outputs[0];  // [B, H, S, D]
  auto& dV = outputs[1];  // [B, H, S, D]

  dK.set_data(mlx::core::allocator::malloc(dK.nbytes()));
  dV.set_data(mlx::core::allocator::malloc(dV.nbytes()));

  auto& dev = mlx::core::metal::device(stream().device);
  bool is_m3_plus = (dev.get_architecture_gen() >= 15); // 13=M1 14=M2 15=M3 16=M4

  uint8_t dtype_code;
  if (q.dtype() == mlx::core::float16)       dtype_code = 0;
  else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
  else                                        dtype_code = 2;
  bool low_prec_inputs = (dtype_code != 2);
  const bool low_prec_inter = false; // forward blocks; see eval_gpu comment

  auto cfg = resolve_block_config(D, is_m3_plus, low_prec_inter,
                                  low_prec_inputs);
  unsigned short block_q = cfg.block_q;
  unsigned short block_k = cfg.block_k;
  unsigned short block_d = cfg.block_d;
  unsigned short n_warps = block_q / 8;

  MFAParams bw_params{};
  bw_params.R               = static_cast<uint32_t>(N);
  bw_params.C               = static_cast<uint32_t>(S);
  bw_params.Hq              = static_cast<uint32_t>(H);
  bw_params.H_Hk_ratio      = 1u;
  bw_params.dot_product_scale = params_.scale * static_cast<float>(M_LOG2E);
  bw_params.causal          = params_.causal ? 1u : 0u;
  bw_params.Q_batch_stride  = static_cast<uint32_t>(H * N * D);
  bw_params.K_batch_stride  = static_cast<uint32_t>(H * S * D);
  bw_params.V_batch_stride  = static_cast<uint32_t>(H * S * D);
  bw_params.O_batch_stride  = static_cast<uint32_t>(H * N * D);
  bw_params.dO_batch_stride = static_cast<uint32_t>(H * N * D);
  bw_params.dQ_batch_stride = static_cast<uint32_t>(H * N * D);
  bw_params.dK_batch_stride = static_cast<uint32_t>(H * S * D);
  bw_params.dV_batch_stride = static_cast<uint32_t>(H * S * D);

  using KK = ShaderCache::KernelKey;
  KK key{
    KK::KernelType::AttentionBackwardDKV,
    D, (int)block_q, (int)block_k, (int)block_d, (int)n_warps,
    params_.causal, is_m3_plus, dtype_code
  };
  void* raw = ShaderCache::get().get_or_compile(key, dev.mtl_device());
  auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw);

  auto& enc = dev.get_command_encoder(stream().index);
  enc.set_compute_pipeline_state(pipeline);
  enc.set_input_array(q,          0);
  enc.set_input_array(k,          1);
  enc.set_input_array(v,          2);
  enc.set_input_array(o,          3);
  enc.set_input_array(l,          4);
  enc.set_input_array(D_computed, 5);  // kernel READS D from here (buffer 5)
  enc.set_input_array(dO,         6);
  enc.set_output_array(dV,        7);
  enc.set_output_array(dK,        8);
  enc.set_bytes(bw_params,       10);

  uint32_t num_k_tiles = (static_cast<uint32_t>(S) + block_q - 1u) / block_q;
  uint32_t grid_dkv    = num_k_tiles
                         * static_cast<uint32_t>(H)
                         * static_cast<uint32_t>(B);
  enc.dispatch_threadgroups(
      MTL::Size::Make(grid_dkv,      1, 1),
      MTL::Size::Make(n_warps * 32u, 1, 1));
}

// =========================================================================
// Equivalence
// =========================================================================

bool MFAttention::is_equivalent(const mlx::core::Primitive& other) const {
  auto* o = dynamic_cast<const MFAttention*>(&other);
  if (!o) return false;
  return params_.head_dim == o->params_.head_dim &&
         params_.scale    == o->params_.scale    &&
         params_.causal   == o->params_.causal;
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
      : mlx::core::default_stream(mlx::core::Device::gpu);

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument("MFA: expected 4D inputs [B, H, N, D]");
  }

  int D = q.shape(3);
  if (D != 64 && D != 128 && D != 256) {
    throw std::invalid_argument(
        "MFA: head_dim must be 64, 128, or 256, got " + std::to_string(D));
  }

  MFAttention::Params params{D, scale, causal};

  auto out_shape  = q.shape();                      // Shape [B, H, N, D]
  mlx::core::Shape lse_shape = {
      q.shape(0), q.shape(1), q.shape(2)};          // Shape [B, H, N]

  // O dtype matches input dtype (kernel accumulates FP32 then writes input prec).
  // L (logsumexp for backward) is always FP32.
  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAttention>(s, params),
      {q, k, v});

  return outputs[0];
}

}  // namespace mlx_mfa
