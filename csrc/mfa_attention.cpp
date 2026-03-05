/// mfa_attention.cpp — MFAttention Primitive implementation.
///
/// eval_gpu() routing:
///   STEEL path (f16/bf16):  generate_steel_forward_source() → ShaderCache
///   ccv path (f32 / legacy): generate_attention_source() → ShaderCache
///
/// The device architecture generation is read via mlx::core::metal and
/// compared against gen >= 15 (M3+) to select block parameters.
///
/// Buffer layout (all kernels):
///   buffer(0) = Q  [B × H × N × D], row-major, contiguous
///   buffer(1) = K  [B × H × S × D], row-major, contiguous
///   buffer(2) = V  [B × H × S × D], row-major, contiguous
///   buffer(3) = O  [B × H × N × D], row-major, output
///   buffer(4) = L  [B × H × N],     logsumexp (STEEL only, used for bwd)
///   buffer(5) = params  (struct MFAttention::Params packed into bytes)

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
#include <cstdlib>
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
  // 3 = dense (Q, K, V)
  // 4 = sparse (Q, K, V, block_mask) or rope (Q, K, V, cos, sin) — 5 for rope
  // 5 = rope (Q, K, V, rotary_cos, rotary_sin)
  assert(inputs.size() >= 3 && inputs.size() <= 5);

  const auto& q = inputs[0]; // [B, H, N, D]
  const auto& k = inputs[1]; // [B, H, S, D]
  const auto& v = inputs[2]; // [B, H, S, D]
  // inputs[3] = block_mask [NQ_tiles, NK_tiles] uint8, only when has_block_mask

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
                params_.causal, /*sparse=*/false, is_m3_plus,
                /*has_rope=*/false,
                /*has_softcap=*/false, /*has_alibi=*/false,
                dtype_code };
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

  // ── Architecture gen (STEEL f16/bf16 path) ──────────────────────────────
  // MFA_FORCE_GEN overrides hardware detection for correctness testing:
  //   MFA_FORCE_GEN=15  → treat as M3 (gen=15) even on M1 (gen=13)
  //   MFA_FORCE_GEN=13  → treat as M1 even on M3 hardware
  int arch_gen_steel = static_cast<int>(d.get_architecture_gen());
  const char* force_gen_env = std::getenv("MFA_FORCE_GEN");
  if (force_gen_env) arch_gen_steel = std::atoi(force_gen_env);
  bool is_m3_plus_steel = (arch_gen_steel >= 15);

  // ── Flash Decoding (Split-KV) path ──────────────────────────────────────
  //
  // At decode time (N_q ≤ 4) the standard grid (NQ, H, B) = (1, H, B) leaves
  // most SMs idle.  Flash Decoding splits the KV sequence into num_splits
  // chunks dispatched in parallel, then a tiny reduce kernel combines them.
  //
  // Activation: N ≤ 4 AND S ≥ 256 AND f16/bf16 (dtype_code != 2)
  //             AND no block mask (sparse path keeps its own dispatch)
  auto cfg_fd = select_steel_block_config(D, /*is_low_prec=*/true, is_m3_plus_steel);
  const int BK_fd = cfg_fd.BK;
  const bool use_flash_decode = (N <= 4 && S >= 256 && dtype_code != 2
                                 && !params_.has_block_mask);
  if (use_flash_decode) {
    int num_splits = compute_num_splits(S, BK_fd);
    auto cfg_s = select_steel_block_config(D, /*is_low_prec=*/true, is_m3_plus_steel);
    int BQ_s = cfg_s.BQ;
    int BK_s = cfg_s.BK;
    int WM_s = cfg_s.WM;
    int TGP_s = WM_s * cfg_s.WN * 32;

    // ── Allocate scratch buffers pO and pL ─────────────────────────────────
    size_t pO_size = (size_t)num_splits * B * H * N * D * (dtype_code == 2 ? 4 : 2);
    size_t pL_size = (size_t)num_splits * B * H * N * sizeof(float);
    auto pO_buf = mlx::core::allocator::malloc(pO_size);
    auto pL_buf = mlx::core::allocator::malloc(pL_size);

    // ── Build FlashDecodePartialParams ─────────────────────────────────────
    int NQ_s = (N + BQ_s - 1) / BQ_s;
    int NK_total = (S + BK_s - 1) / BK_s;
    int NK_per_split = (NK_total + num_splits - 1) / num_splits;

    FlashDecodePartialParams pp{};
    pp.B = B; pp.H = H; pp.D = D;
    pp.qL = N; pp.kL = S;
    pp.gqa_factor = H / Hk;
    pp.scale = params_.scale;
    pp.NQ = NQ_s;
    pp.NQ_aligned = (N % BQ_s == 0) ? NQ_s : NQ_s - 1;
    pp.qL_rem     = (N % BQ_s == 0) ? BQ_s : (N % BQ_s);
    pp.qL_off     = (params_.causal && N < S) ? (S - N) : 0;
    pp.NK_total    = NK_total;
    pp.NK_aligned  = (S % BK_s == 0) ? NK_total : NK_total - 1;
    pp.kL_rem      = (S % BK_s == 0) ? BK_s : (S % BK_s);
    pp.num_splits  = num_splits;
    pp.NK_per_split = NK_per_split;
    // Input strides
    pp.Q_strides[0] = (int64_t)H  * N * D;
    pp.Q_strides[1] = (int64_t)N  * D;
    pp.Q_strides[2] = (int64_t)D;
    pp.K_strides[0] = (int64_t)Hk * S * D;
    pp.K_strides[1] = (int64_t)S  * D;
    pp.K_strides[2] = (int64_t)D;
    pp.V_strides[0] = (int64_t)Hk * S * D;
    pp.V_strides[1] = (int64_t)S  * D;
    pp.V_strides[2] = (int64_t)D;
    // pO strides (split outermost): [num_splits, B, H, qL, D]
    int64_t pO_head_stride  = (int64_t)N * D;
    int64_t pO_batch_stride = (int64_t)H * N * D;
    pp.pO_split_stride = (int64_t)B * H * N * D;
    pp.pO_batch_stride = pO_batch_stride;
    pp.pO_head_stride  = pO_head_stride;
    // pL strides: [num_splits, B, H, qL]
    pp.pL_split_stride = (int64_t)B * H * N;
    pp.pL_batch_stride = (int64_t)H * N;
    pp.pL_head_stride  = (int64_t)N;

    // Optional features
    pp.softcap = params_.softcap;   // 0.0 when disabled

    // ── Build FlashDecodeReduceParams ──────────────────────────────────────
    int reduce_tgp = std::min(D, 128);
    FlashDecodeReduceParams rp{};
    rp.B = B; rp.H = H; rp.D = D;
    rp.qL = N;
    rp.num_splits = num_splits;
    rp.pO_split_stride = pp.pO_split_stride;
    rp.pO_batch_stride = pp.pO_batch_stride;
    rp.pO_head_stride  = pp.pO_head_stride;
    rp.pL_split_stride = pp.pL_split_stride;
    rp.pL_batch_stride = pp.pL_batch_stride;
    rp.pL_head_stride  = pp.pL_head_stride;
    rp.O_batch_stride  = (int64_t)H * N * D;
    rp.O_head_stride   = (int64_t)N * D;
    rp.L_batch_stride  = (int64_t)H * N;
    rp.L_head_stride   = (int64_t)N;
    rp.reduce_tgp_size = reduce_tgp;

    // ── Compile Phase 1 and Phase 2 pipelines ─────────────────────────────
    using KK = ShaderCache::KernelKey;
    KK key_p1{
      KK::KernelType::FlashDecodePartial,
      D, BQ_s, BK_s, D, WM_s,
      params_.causal, /*sparse=*/false, is_m3_plus_steel,
      /*has_rope=*/false,
      params_.softcap > 0.0f,   // softcap variant
      params_.has_alibi,        // ALiBi position biases
      dtype_code
    };
    KK key_p2{
      KK::KernelType::FlashDecodeReduce,
      D, 0, 0, 0, 0,
      false, false, false, /*has_rope=*/false,
      /*has_softcap=*/false, /*has_alibi=*/false,
      dtype_code
    };
    auto* pl_p1 = reinterpret_cast<MTL::ComputePipelineState*>(
        ShaderCache::get().get_or_compile(key_p1, d.mtl_device()));
    auto* pl_p2 = reinterpret_cast<MTL::ComputePipelineState*>(
        ShaderCache::get().get_or_compile(key_p2, d.mtl_device()));

    auto& enc = d.get_command_encoder(stream().index);

    // ── Phase 1 dispatch ──────────────────────────────────────────────────
    enc.set_compute_pipeline_state(pl_p1);
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_buffer(reinterpret_cast<MTL::Buffer*>(pO_buf.ptr()), 3, 0);
    enc.set_buffer(reinterpret_cast<MTL::Buffer*>(pL_buf.ptr()), 4, 0);
    enc.set_bytes(pp, 5);
    if (params_.has_alibi) {
      // Flash Decode partial has no rope/block_mask: alibi_slopes is inputs[3]
      enc.set_input_array(inputs[3], 6);
    }
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)(NQ_s * num_splits), (size_t)H, (size_t)B),
        MTL::Size::Make((size_t)TGP_s, 1, 1));

    // ── Phase 2 dispatch ──────────────────────────────────────────────────
    // Use barrier() (unconditional) — maybeInsertBarrier() is a no-op for
    // raw MTL::Buffer* bindings since needs_barrier_ is only set by
    // set_output_array(); Phase 1 writes pO/pL as set_buffer() buffers.
    enc.barrier();
    enc.set_compute_pipeline_state(pl_p2);
    enc.set_buffer(reinterpret_cast<MTL::Buffer*>(pO_buf.ptr()), 0, 0);
    enc.set_buffer(reinterpret_cast<MTL::Buffer*>(pL_buf.ptr()), 1, 0);
    enc.set_output_array(out,       2);
    enc.set_output_array(logsumexp, 3);
    enc.set_bytes(rp, 4);
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)N, (size_t)H, (size_t)B),
        MTL::Size::Make((size_t)reduce_tgp, 1, 1));

    // Release scratch buffers (Metal retains them until command buffer completes)
    mlx::core::allocator::free(pO_buf);
    mlx::core::allocator::free(pL_buf);
    return;
  }

  // ── STEEL tile config (f16 / bf16) ───────────────────────────────────────
  auto cfg = select_steel_block_config(D, /*is_low_prec=*/true, is_m3_plus_steel);
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
    params_.has_block_mask,  // sparse variant when block_mask present
    is_m3_plus_steel,        // separate compiled pipeline for M3+ configs
    params_.has_rope,        // in-kernel RoPE fusion variant
    params_.softcap > 0.0f,   // tanh softcapping variant
    params_.has_alibi,        // ALiBi per-head position biases
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
  // For decode (N < S, causal), the first query row is at position S-N in the KV
  // sequence.  qL_off shifts the causal window so key position k is visible
  // to query row q when k <= q + qL_off.  For self-attention N==S → qL_off=0.
  sp.qL_off     = (N < S && params_.causal) ? (S - N) : 0;

  // RoPE fusion: absolute position of Q token 0 and stride of cos/sin table.
  // Both are zero when has_rope=false; the Metal kernel ignores them.
  sp.rope_q_base     = params_.cache_seqlens;
  sp.rope_cos_stride = D / 2;

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

  // Optional features — must be set even when disabled (struct is zero-init'd
  // above but explicit assignment is clearer and guards against future refactors).
  sp.softcap   = params_.softcap;           // 0.0 when disabled
  sp.has_alibi = params_.has_alibi ? 1 : 0;

  // ── Dispatch ─────────────────────────────────────────────────────────────
  auto& enc = d.get_command_encoder(stream().index);
  enc.set_compute_pipeline_state(pipeline);

  // Buffers: Q=0, K=1, V=2, O=3, L=4, params=5, (block_mask=6 if sparse)
  enc.set_input_array(q,          0);
  enc.set_input_array(k,          1);
  enc.set_input_array(v,          2);
  enc.set_output_array(out,       3);
  enc.set_output_array(logsumexp, 4);
  enc.set_bytes(sp,               5);
  if (params_.has_block_mask) {
    enc.set_input_array(inputs[3], 6);
  }
  if (params_.has_rope) {
    // rotary_cos and rotary_sin follow block_mask (if present) in the input list.
    // Dense + RoPE: inputs[3]=cos, inputs[4]=sin
    // Sparse + RoPE (not currently exposed): inputs[4]=cos, inputs[5]=sin
    int cos_idx = params_.has_block_mask ? 4 : 3;
    enc.set_input_array(inputs[cos_idx],     7);
    enc.set_input_array(inputs[cos_idx + 1], 8);
  }
  if (params_.has_alibi) {
    // alibi_slopes [H] follows block_mask/rope in the input list.
    // Dense + ALiBi (no block_mask, no rope): inputs[3]=alibi_slopes
    int alibi_idx = 3
        + (params_.has_block_mask ? 1 : 0)
        + (params_.has_rope       ? 2 : 0);
    enc.set_input_array(inputs[alibi_idx], 9);
  }

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
    params_.causal, /*sparse=*/false, is_m3_plus, /*has_rope=*/false,
    /*has_softcap=*/false, /*has_alibi=*/false, dtype_code
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
    params_.causal, /*sparse=*/false, is_m3_plus, /*has_rope=*/false,
    /*has_softcap=*/false, /*has_alibi=*/false, dtype_code
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
  return params_.head_dim       == o->params_.head_dim       &&
         params_.scale          == o->params_.scale          &&
         params_.causal         == o->params_.causal         &&
         params_.has_block_mask == o->params_.has_block_mask &&
         params_.has_rope       == o->params_.has_rope       &&
         params_.cache_seqlens  == o->params_.cache_seqlens  &&
         params_.softcap        == o->params_.softcap        &&
         params_.has_alibi      == o->params_.has_alibi;
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
    float softcap,
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

  MFAttention::Params params{D, scale, causal,
      /*has_block_mask=*/false, /*has_rope=*/false,
      /*cache_seqlens=*/0, softcap};

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

// =========================================================================
// Free function: mfa_attention_sparse_forward
// =========================================================================

mlx::core::array mfa_attention_sparse_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& block_mask,
    float scale,
    bool causal,
    std::optional<mlx::core::StreamOrDevice> stream) {
  auto s = stream.has_value()
      ? mlx::core::to_stream(stream.value())
      : mlx::core::default_stream(mlx::core::Device::gpu);

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument("MFA sparse: expected 4D inputs [B, H, N, D]");
  }
  if (block_mask.ndim() != 2) {
    throw std::invalid_argument(
        "MFA sparse: block_mask must be 2D [NQ_tiles, NK_tiles]");
  }

  int D = q.shape(3);
  if (D != 64 && D != 128 && D != 256) {
    throw std::invalid_argument(
        "MFA sparse: head_dim must be 64, 128, or 256, got " +
        std::to_string(D));
  }

  // Require f16/bf16 (sparse path is STEEL-only; f32 would need ccv update)
  if (q.dtype() == mlx::core::float32) {
    throw std::invalid_argument(
        "MFA sparse: float32 is not supported; use float16 or bfloat16");
  }

  MFAttention::Params params{D, scale, causal, /*has_block_mask=*/true};

  auto out_shape  = q.shape();
  mlx::core::Shape lse_shape = {q.shape(0), q.shape(1), q.shape(2)};

  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAttention>(s, params),
      {q, k, v, block_mask});

  return outputs[0];
}

std::vector<mlx::core::array> mfa_attention_sparse_forward_with_lse(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& block_mask,
    float scale,
    bool causal,
    std::optional<mlx::core::StreamOrDevice> stream) {
  auto s = stream.has_value()
      ? mlx::core::to_stream(stream.value())
      : mlx::core::default_stream(mlx::core::Device::gpu);

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4)
    throw std::invalid_argument("MFA sparse: expected 4D inputs [B, H, N, D]");
  if (block_mask.ndim() != 2)
    throw std::invalid_argument(
        "MFA sparse: block_mask must be 2D [NQ_tiles, NK_tiles]");

  int D = q.shape(3);
  if (D != 64 && D != 128 && D != 256)
    throw std::invalid_argument(
        "MFA sparse: head_dim must be 64, 128, or 256, got " +
        std::to_string(D));

  if (q.dtype() == mlx::core::float32)
    throw std::invalid_argument(
        "MFA sparse: float32 is not supported; use float16 or bfloat16");

  MFAttention::Params params{D, scale, causal, /*has_block_mask=*/true};
  auto out_shape = q.shape();
  mlx::core::Shape lse_shape = {q.shape(0), q.shape(1), q.shape(2)};

  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAttention>(s, params),
      {q, k, v, block_mask});

  return {outputs[0], outputs[1]};  // O, L
}

// =========================================================================
// Free function: mfa_attention_rope_forward
// =========================================================================
//
// Forward pass with in-kernel RoPE fusion.
//   rotary_cos / rotary_sin: float32 [max_seq_len, D/2].
//   cache_seqlens: position of Q token 0 in the full sequence (KV cache length
//                  for autoregressive decode; 0 for prefill).
//
// The RoPE rotation is applied in threadgroup SRAM immediately after loading
// Q and K tiles — before the GEMM accumulation.  This fuses the rotary step
// into the attention kernel and eliminates a separate elementwise pass.
//
// Only f16/bf16 is supported (STEEL path).  float32 raises an error.

mlx::core::array mfa_attention_rope_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& rotary_cos,
    const mlx::core::array& rotary_sin,
    float scale,
    bool causal,
    int cache_seqlens,
    std::optional<mlx::core::StreamOrDevice> stream) {
  auto s = stream.has_value()
      ? mlx::core::to_stream(stream.value())
      : mlx::core::default_stream(mlx::core::Device::gpu);

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument(
        "MFA rope: expected 4D inputs [B, H, N, D]");
  }

  int D = q.shape(3);
  if (D != 64 && D != 128 && D != 256) {
    throw std::invalid_argument(
        "MFA rope: head_dim must be 64, 128, or 256, got " +
        std::to_string(D));
  }

  // RoPE fusion only on the STEEL path (f16/bf16).
  if (q.dtype() == mlx::core::float32) {
    throw std::invalid_argument(
        "MFA rope: float32 is not supported; use float16 or bfloat16");
  }

  MFAttention::Params params{
    D, scale, causal,
    /*has_block_mask=*/false,
    /*has_rope=*/true,
    /*cache_seqlens=*/cache_seqlens
  };

  auto out_shape  = q.shape();
  mlx::core::Shape lse_shape = {q.shape(0), q.shape(1), q.shape(2)};

  // inputs: [Q, K, V, rotary_cos, rotary_sin]
  // buffers in Metal: Q=0, K=1, V=2, O=3, L=4, params=5, cos=7, sin=8
  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAttention>(s, params),
      {q, k, v, rotary_cos, rotary_sin});

  return outputs[0];
}

// =========================================================================
// Free function: mfa_attention_alibi_forward
// =========================================================================

mlx::core::array mfa_attention_alibi_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& alibi_slopes,
    float scale,
    bool causal,
    std::optional<mlx::core::StreamOrDevice> stream) {
  auto s = stream.has_value()
      ? mlx::core::to_stream(stream.value())
      : mlx::core::default_stream(mlx::core::Device::gpu);

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4)
    throw std::invalid_argument("MFA alibi: expected 4D inputs [B, H, N, D]");

  if (alibi_slopes.ndim() != 1)
    throw std::invalid_argument(
        "MFA alibi: alibi_slopes must be 1D [H]");

  int D = q.shape(3);
  if (D != 64 && D != 128 && D != 256)
    throw std::invalid_argument(
        "MFA alibi: head_dim must be 64, 128, or 256, got " +
        std::to_string(D));

  MFAttention::Params params{
    D, scale, causal,
    /*has_block_mask=*/false,
    /*has_rope=*/false,
    /*cache_seqlens=*/0,
    /*softcap=*/0.0f,
    /*has_alibi=*/true
  };

  auto out_shape  = q.shape();
  mlx::core::Shape lse_shape = {q.shape(0), q.shape(1), q.shape(2)};

  // inputs: [Q, K, V, alibi_slopes]
  // Metal buffers: Q=0, K=1, V=2, O=3, L=4, params=5, alibi_slopes=9
  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAttention>(s, params),
      {q, k, v, alibi_slopes});

  return outputs[0];
}

}  // namespace mlx_mfa
