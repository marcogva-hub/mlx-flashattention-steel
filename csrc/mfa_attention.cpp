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
#include "mfa_steel_fwd_v2.hpp"
#include "mfa_steel_fwd_v3.hpp"
#include "mfa_steel_bwd.hpp"
#include "mfa_gna_fwd.hpp"
// GNA native kernel removed (sparse path is faster)
#include "mfa_sage_fwd.hpp"
#include "mfa_steel_paged_varlen_fwd.hpp"
#include "mfa_steel_paged_varlen_tq_fwd.hpp"
#include "shader_cache.hpp"
#include "mfa_env.hpp"

#include <mlx/utils.h>
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <Metal/Metal.hpp>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace mlx_mfa {

namespace {

inline bool is_v2_small_d_family(int head_dim) {
  return head_dim == 64 || head_dim == 128;
}

inline bool is_v2_d256_family(int head_dim) {
  return head_dim == 256;
}

inline bool is_v2_d512_family(int head_dim) {
  return head_dim == 512;
}

inline bool is_v2_dsplit_family(int head_dim) {
  // Large-D families are explicit:
  //   - D=256: narrow benchmark-backed MFA promotion in Python policy.
  //   - D=512: conservative SDPA-default in Python policy.
  // Both share the same C++ D-split kernel family; auto-route policy lives in
  // mlx_mfa/dispatch_policy.py.
  return is_v2_d256_family(head_dim) || is_v2_d512_family(head_dim);
}

}  // namespace

// =========================================================================
// Split-K calibration env-key builder — SINGLE SOURCE OF TRUTH (audit B1)
// =========================================================================
// Both the dispatch lookup (MFAttention::eval_gpu) and the test-only binding
// `mlx_mfa._ext._splitk_env_key_cpp` call THIS function — never a parallel copy
// (a copy would make the cross-language identity test vacuous).  MUST stay
// byte-identical to dispatch_policy._splitk_env_key / _splitk_window_suffix.
// Key format: MFA_SPLITK_MAX_N_D{D}_C{0|1}_A{0|1}_{W0 | W{left}_{right}}.
std::string build_splitk_env_key(int D, bool causal, bool has_alibi,
                                 int window_left, int window_right) {
  std::string wsuf;
  if (window_left >= 0 || window_right >= 0) {
    const int wl = window_left >= 0 ? window_left : 0;
    const int wr = window_right >= 0 ? window_right : 0;
    wsuf = "W" + std::to_string(wl) + "_" + std::to_string(wr);
  } else {
    wsuf = "W0";
  }
  return "MFA_SPLITK_MAX_N_D" + std::to_string(D) +
         "_C" + std::to_string(causal ? 1 : 0) +
         "_A" + std::to_string(has_alibi ? 1 : 0) +
         "_" + wsuf;
}

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
    int arch_gen_ccv = static_cast<int>(d.get_architecture_gen());
    { const auto& e = MFAEnvConfig::get(); if (e.force_gen > 0) arch_gen_ccv = e.force_gen; }
    bool is_m3_plus = (arch_gen_ccv >= 15);
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
    // Repo review 2026-05: compute in int64 before narrowing — `H * N * D`
    // as int32 overflows (UB) for H*N*D > 2^31 before the uint32_t cast.
    fwd_p.Q_batch_stride   = static_cast<uint32_t>((int64_t)H  * N * D);
    fwd_p.K_batch_stride   = static_cast<uint32_t>((int64_t)Hk * S * D);
    fwd_p.V_batch_stride   = static_cast<uint32_t>((int64_t)Hk * S * D);
    fwd_p.O_batch_stride   = static_cast<uint32_t>((int64_t)H  * N * D);

    using KK = ShaderCache::KernelKey;
    KK ccv_key{ KK::KernelType::AttentionForward,
                D, (int)bq, (int)bk, (int)bd, (int)nw,
                params_.causal, /*sparse=*/false, is_m3_plus,
                /*has_rope=*/false, /*rope_interleaved=*/false,
                /*has_softcap=*/false, /*has_alibi=*/false,
                /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
                /*has_window=*/false, dtype_code };
    void* raw = ShaderCache::get().get_or_compile(ccv_key, d.mtl_device());
    auto* pl  = reinterpret_cast<MTL::ComputePipelineState*>(raw);

    auto& enc = mlx::core::metal::get_command_encoder(stream());
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
  const auto& env = MFAEnvConfig::get();
  if (env.force_gen > 0) arch_gen_steel = env.force_gen;
  bool is_m3_plus_steel = (arch_gen_steel >= 15);
  // M5+ (gen >= 17, A19/Apple Silicon 5th gen) exposes Metal 4 tensor API
  // (MTLTensor, cooperative tensor ops). Stub only — no kernel implemented yet.
  // When M5+ hardware is available, dispatch Metal4TensorOps kernel here.
  bool is_m5_plus = (arch_gen_steel >= 17);
  (void)is_m5_plus;  // unused until Metal4TensorOps kernel is implemented

  // ── GPU core count (for split-K occupancy heuristics) ────────────────────
  // estimate_gpu_cores() parses MTLDevice::name() for the actual chip variant
  // (M1 Max=32, M1 base=8, etc.) rather than using arch_gen alone (which maps
  // all M1 variants to gen=13). Falls back to gen-based estimate if unavailable.
  auto* mtl_dev_ptr = d.mtl_device();
  const std::string mtl_dev_name = mtl_dev_ptr
      ? std::string(mtl_dev_ptr->name()->utf8String()) : "";
  const int gpu_cores = estimate_gpu_cores(mtl_dev_name, arch_gen_steel);

  // ── Flash Decoding (Split-KV) path ──────────────────────────────────────
  //
  // At decode time (N_q ≤ 4) the standard grid (NQ, H, B) = (1, H, B) leaves
  // most SMs idle.  Flash Decoding splits the KV sequence into num_splits
  // chunks dispatched in parallel, then a tiny reduce kernel combines them.
  //
  // Activation: N ≤ 4 AND S ≥ 256 AND f16/bf16 (dtype_code != 2)
  //             AND no block mask (sparse path keeps its own dispatch)
  // CP2: D=64/128 use V2 tile sizes (larger BK) to reduce K-tile iterations per split.
  //      D=256/512 keep V1 tiles (V2 BQ=16/WM=2 for D=256 halves occupancy in V1 kernel).
  // Pre-compute block config for flash decode (reused in BK_fd and dispatch).
  int BQ_fd, BK_fd, WM_fd, WN_fd;
  if (D <= 128) {
    auto cfgv2 = select_steel_v2_block_config(D, is_m3_plus_steel);
    BQ_fd = cfgv2.BQ; BK_fd = cfgv2.BK; WM_fd = cfgv2.WM; WN_fd = cfgv2.WN;
  } else {
    auto cfgv1 = select_steel_block_config(D, /*is_low_prec=*/true, is_m3_plus_steel);
    BQ_fd = cfgv1.BQ; BK_fd = cfgv1.BK; WM_fd = cfgv1.WM; WN_fd = cfgv1.WN;
  }
  // Repo review 2026-05: `!params_.has_rope` guard added.  The flash-decode
  // partial kernel does not generate RoPE code (Phase-1 key hardcodes
  // has_rope=false and the Metal params struct lacks rope fields).  Latent
  // gap — the public API currently routes RoPE decode via
  // mfa_attention_rope_forward, never reaching here with has_rope=true —
  // but any future routing change would silently drop rotary embeddings.
  // RoPE decode falls through to split-K / V2 / V1 which implement it.
  const bool use_flash_decode = (N <= 4 && S >= 256 && dtype_code != 2
                                 && !params_.has_block_mask
                                 && !params_.has_rope);
  if (use_flash_decode) {
    int num_splits = compute_num_splits(S, BK_fd);
    int BQ_s = BQ_fd, BK_s = BK_fd, WM_s = WM_fd, WN_s = WN_fd;
    int TGP_s = WM_s * WN_s * 32;

    // RC-B (audit): eliminate empty trailing splits.  compute_num_splits returns
    // min(NK/2, 32); with NK_per_split = ceil(NK/num_splits) the product
    // num_splits*NK_per_split can exceed NK_total, leaving trailing splits whose
    // K-range is empty (kb_start >= NK_total).  An empty split's online softmax
    // never runs → its pO is normalized 0/0 = NaN and the reduce's 0*NaN poisons
    // the final output (decode-tail all-NaN).  Shrink num_splits to exactly cover
    // NK_total so every split owns >=1 K-tile (and (num_splits-1)*per < NK_total).
    {
      const int NK_total_fd = (S + BK_s - 1) / BK_s;
      const int per_fd = (NK_total_fd + num_splits - 1) / num_splits;
      num_splits = (NK_total_fd + per_fd - 1) / per_fd;   // <= old num_splits
    }

    // ── Allocate scratch buffers pO and pL ─────────────────────────────────
    // Scratch wrapped in arrays + registered as command-encoder temporaries
    // (freed only after the command buffer completes — see III-9 root cause:
    // encode-time free returned the pool memory while the lazy kernels were
    // still pending, letting a concurrent allocation corrupt the reduce read).
    size_t pO_size = (size_t)num_splits * B * H * N * D * (dtype_code == 2 ? 4 : 2);
    size_t pL_size = (size_t)num_splits * B * H * N * sizeof(float);
    mlx::core::array pO_arr(
        mlx::core::allocator::malloc(pO_size),
        {(int)num_splits, B, H, N, D}, q.dtype());
    mlx::core::array pL_arr(
        mlx::core::allocator::malloc(pL_size),
        {(int)num_splits, B, H, N}, mlx::core::float32);
    auto pO_buf = pO_arr.buffer();
    auto pL_buf = pL_arr.buffer();

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
    pp.qL_off     = (N < S && params_.causal) ? (S - N) : 0;
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
    pp.softcap       = params_.softcap;    // 0.0 when disabled
    pp.window_left   = params_.window_left;  // -1 when disabled
    pp.window_right  = params_.window_right; // -1 when disabled

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
      /*has_rope=*/false, /*rope_interleaved=*/true,
      params_.softcap > 0.0f,   // softcap variant
      params_.has_alibi,        // ALiBi position biases
      /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
      params_.window_left >= 0 || params_.window_right >= 0, // sliding window variant
      dtype_code
    };
    KK key_p2{
      KK::KernelType::FlashDecodeReduce,
      D, 0, 0, 0, 0,
      false, false, false, /*has_rope=*/false, /*rope_interleaved=*/true,
      /*has_softcap=*/false, /*has_alibi=*/false,
      /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
      /*has_window=*/false,
      dtype_code
    };
    auto* pl_p1 = reinterpret_cast<MTL::ComputePipelineState*>(
        ShaderCache::get().get_or_compile(key_p1, d.mtl_device()));
    auto* pl_p2 = reinterpret_cast<MTL::ComputePipelineState*>(
        ShaderCache::get().get_or_compile(key_p2, d.mtl_device()));

    auto& enc = mlx::core::metal::get_command_encoder(stream());

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

    // Lifetime: tie scratch to command-buffer completion (NOT encode time).
    enc.add_temporary(pO_arr);
    enc.add_temporary(pL_arr);
    return;
  }

  // ── STEEL V2 split-K dispatch (Phase 3) ─────────────────────────────────
  // For under-occupied grids (total_tgs < 0.8 * gpu_cores) that are V2-eligible
  // but NOT handled by flash decode (N > 4), use V2 split-K to fill the GPU.
  // Phase 1: SteelV2SplitKPartial  (grid: NQ * num_splits, H, B)
  // Phase 2: FlashDecodeReduce     (reused, grid: N, H, B)
  // Set MFA_DISABLE_V2=1 to force V1 path (for benchmarking/debugging only).
  if (!MFAEnvConfig::disable_v2()) {
    // MFA_FORCE_SPLITK override:
    //   1 -> force split-K attempt (bypass occupancy short-circuit)
    //   0 -> disable split-K entirely
    // unset/other -> normal heuristic + optional calibrated thresholds
    int force_splitk = MFAEnvConfig::force_splitk();  // -1=heuristic, 0=disable, 1=force
    const bool has_window = (params_.window_left >= 0 || params_.window_right >= 0);
    const auto splitk_calibrated_max_n = [&]() -> int {
      // B1: the REAL dispatch lookup routes through the single builder (also
      // exposed to Python via `_splitk_env_key_cpp` and locked byte-identical to
      // dispatch_policy._splitk_env_key by tests/test_audit_m02_*).
      const std::string env_key = build_splitk_env_key(
          D, params_.causal, params_.has_alibi,
          params_.window_left, params_.window_right);
      const char* v = std::getenv(env_key.c_str());
      if (!v || v[0] == '\0') return -1;
      return std::atoi(v);
    };
    const int calibrated_max_n = splitk_calibrated_max_n();

    // M3+ (gen>=15): V1 double-buffer is 1.5-3.7x faster than V2 at D<=128 causal.
    // V2's shared KV_smem requires 3-4 barriers/tile vs V1's 2 barriers/tile.
    // On M3+ hardware, reduced TGP bandwidth makes barriers more expensive.
    // Skip V2 for this regime, falling through to V1.
    // Override: set MFA_FORCE_V2=1 to bypass this guard (benchmarking).
    const bool m3_prefers_v1_sk = is_m3_plus_steel && D <= 128 && params_.causal
                                  && !MFAEnvConfig::force_v2();
    const bool v2sk_eligible =
        (dtype_code != 2) &&
        is_v2_small_d_family(D) &&
        // Sparse remains excluded from V2 split-K (block-mask uses V1 BK indexing).
        // RoPE/ALiBi/window are supported in V2 split-K (Phase 3 composability).
        // attn_bias excluded: split-K partial kernel doesn't implement bias addition.
        !params_.has_block_mask &&
        !params_.has_attn_bias &&
        !m3_prefers_v1_sk;

    const bool splitk_disabled_by_override = (force_splitk == 0);
    const bool splitk_disabled_by_calibration =
        (force_splitk < 0 && calibrated_max_n >= 0 && N > calibrated_max_n);

    if (v2sk_eligible && !splitk_disabled_by_override && !splitk_disabled_by_calibration) {
      auto cfg2 = select_steel_v2_block_config(D, is_m3_plus_steel);
      const int BQ2  = cfg2.BQ;
      const int BK2  = cfg2.BK;
      const int WM2  = cfg2.WM;
      const int TGP2 = WM2 * cfg2.WN * 32;
      const int NQ2  = (N + BQ2 - 1) / BQ2;
      const int total_tgs = NQ2 * H * B;
      const int num_splits = compute_v2_num_splits(
          total_tgs, S, BK2, gpu_cores, force_splitk == 1);

      if (num_splits >= 2) {
        const int NK2_total = (S + BK2 - 1) / BK2;
        const int NK2_per_split = (NK2_total + num_splits - 1) / num_splits;
        const int NQ2_aln = (N % BQ2 == 0) ? NQ2 : NQ2 - 1;
        const int NK2_aln = (S % BK2 == 0) ? NK2_total : NK2_total - 1;

        // ── Scratch buffers pO[num_splits, B, H, N, D] and pL[num_splits, B, H, N] ─
        // Wrapped in arrays + registered as command-encoder temporaries so MLX
        // frees them only AFTER the command buffer completes.  Freeing the raw
        // allocator buffers at encode time (the previous approach) returned them
        // to the pool while Phase 1/2 were still pending under lazy eval — a
        // concurrent allocation could then reuse the memory and corrupt the
        // not-yet-executed reduce (III-9 root cause).
        size_t pO_size = (size_t)num_splits * B * H * N * D * 2;  // f16/bf16 = 2 bytes
        size_t pL_size = (size_t)num_splits * B * H * N * sizeof(float);
        mlx::core::array pO_arr(
            mlx::core::allocator::malloc(pO_size),
            {(int)num_splits, B, H, N, D}, q.dtype());
        mlx::core::array pL_arr(
            mlx::core::allocator::malloc(pL_size),
            {(int)num_splits, B, H, N}, mlx::core::float32);
        auto pO_buf = pO_arr.buffer();
        auto pL_buf = pL_arr.buffer();

        // ── Build FlashDecodePartialParams for Phase 1 ─────────────────────
        FlashDecodePartialParams sk_pp{};
        sk_pp.B = B; sk_pp.H = H; sk_pp.D = D;
        sk_pp.qL = N; sk_pp.kL = S;
        sk_pp.gqa_factor  = H / Hk;
        sk_pp.scale       = params_.scale;
        sk_pp.NQ          = NQ2;
        sk_pp.NQ_aligned  = NQ2_aln;
        sk_pp.qL_rem      = (N % BQ2 == 0) ? BQ2 : (N % BQ2);
        sk_pp.qL_off      = (N < S && params_.causal) ? (S - N) : 0;
        sk_pp.NK_total    = NK2_total;
        sk_pp.NK_aligned  = NK2_aln;
        sk_pp.kL_rem      = (S % BK2 == 0) ? BK2 : (S % BK2);
        sk_pp.num_splits  = num_splits;
        sk_pp.NK_per_split = NK2_per_split;
        sk_pp.Q_strides[0] = (int64_t)H  * N * D;
        sk_pp.Q_strides[1] = (int64_t)N  * D;
        sk_pp.Q_strides[2] = (int64_t)D;
        sk_pp.K_strides[0] = (int64_t)Hk * S * D;
        sk_pp.K_strides[1] = (int64_t)S  * D;
        sk_pp.K_strides[2] = (int64_t)D;
        sk_pp.V_strides[0] = (int64_t)Hk * S * D;
        sk_pp.V_strides[1] = (int64_t)S  * D;
        sk_pp.V_strides[2] = (int64_t)D;
        sk_pp.pO_split_stride = (int64_t)B * H * N * D;
        sk_pp.pO_batch_stride = (int64_t)H * N * D;
        sk_pp.pO_head_stride  = (int64_t)N * D;
        sk_pp.pL_split_stride = (int64_t)B * H * N;
        sk_pp.pL_batch_stride = (int64_t)H * N;
        sk_pp.pL_head_stride  = (int64_t)N;
        sk_pp.softcap     = params_.softcap;
        sk_pp.window_left = params_.window_left;
        sk_pp.window_right = params_.window_right;
        sk_pp.rope_q_base    = params_.cache_seqlens;
        sk_pp.rope_cos_stride = D / 2;

        // ── Build FlashDecodeReduceParams for Phase 2 ─────────────────────
        int reduce_tgp = std::min(D, 128);
        FlashDecodeReduceParams sk_rp{};
        sk_rp.B = B; sk_rp.H = H; sk_rp.D = D;
        sk_rp.qL = N;
        sk_rp.num_splits      = num_splits;
        sk_rp.pO_split_stride = sk_pp.pO_split_stride;
        sk_rp.pO_batch_stride = sk_pp.pO_batch_stride;
        sk_rp.pO_head_stride  = sk_pp.pO_head_stride;
        sk_rp.pL_split_stride = sk_pp.pL_split_stride;
        sk_rp.pL_batch_stride = sk_pp.pL_batch_stride;
        sk_rp.pL_head_stride  = sk_pp.pL_head_stride;
        sk_rp.O_batch_stride  = (int64_t)H * N * D;
        sk_rp.O_head_stride   = (int64_t)N * D;
        sk_rp.L_batch_stride  = (int64_t)H * N;
        sk_rp.L_head_stride   = (int64_t)N;
        sk_rp.reduce_tgp_size = reduce_tgp;

        // ── Compile Phase 1 (V2 split-K partial) ──────────────────────────
        using KK2 = ShaderCache::KernelKey;
        KK2 key_sk{
          KK2::KernelType::SteelV2SplitKPartial,
          D, BQ2, BK2, D, WM2,
          params_.causal, /*sparse=*/false, is_m3_plus_steel,
          params_.has_rope, params_.rope_interleaved,
          params_.softcap > 0.0f,   // has_softcap
          params_.has_alibi,
          /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
          params_.window_left >= 0 || params_.window_right >= 0,  // has_window
          dtype_code, H / Hk
        };
        // Phase 2 reuses FlashDecodeReduce (key identical to flash_decode reduce).
        KK2 key_sk_reduce{
          KK2::KernelType::FlashDecodeReduce,
          D, 0, 0, 0, 0,
          false, false, false, false, true,
          false, false,
          /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
          false,
          dtype_code
        };

        auto* pl_sk1 = reinterpret_cast<MTL::ComputePipelineState*>(
            ShaderCache::get().get_or_compile(key_sk, d.mtl_device()));
        auto* pl_sk2 = reinterpret_cast<MTL::ComputePipelineState*>(
            ShaderCache::get().get_or_compile(key_sk_reduce, d.mtl_device()));

        auto& enc_sk = mlx::core::metal::get_command_encoder(stream());

        // ── Phase 1: V2 split-K partial ────────────────────────────────────
        enc_sk.set_compute_pipeline_state(pl_sk1);
        enc_sk.set_input_array(q, 0);
        enc_sk.set_input_array(k, 1);
        enc_sk.set_input_array(v, 2);
        enc_sk.set_buffer(reinterpret_cast<MTL::Buffer*>(pO_buf.ptr()), 3, 0);
        enc_sk.set_buffer(reinterpret_cast<MTL::Buffer*>(pL_buf.ptr()), 4, 0);
        enc_sk.set_bytes(sk_pp, 5);
        if (params_.has_rope) {
          // inputs[3]=rotary_cos, inputs[4]=rotary_sin (no block_mask in split-K path)
          enc_sk.set_input_array(inputs[3], 6);
          enc_sk.set_input_array(inputs[4], 7);
        }
        if (params_.has_alibi) {
          // Dense split-K input order: [Q, K, V, alibi] or [Q, K, V, cos, sin, alibi]
          const int alibi_idx = 3 + (params_.has_rope ? 2 : 0);
          enc_sk.set_input_array(inputs[alibi_idx], 9);
        }
        enc_sk.dispatch_threadgroups(
            MTL::Size::Make((size_t)(NQ2 * num_splits), (size_t)H, (size_t)B),
            MTL::Size::Make((size_t)TGP2, 1, 1));

        // ── Phase 2: reduce ────────────────────────────────────────────────
        enc_sk.barrier();
        enc_sk.set_compute_pipeline_state(pl_sk2);
        enc_sk.set_buffer(reinterpret_cast<MTL::Buffer*>(pO_buf.ptr()), 0, 0);
        enc_sk.set_buffer(reinterpret_cast<MTL::Buffer*>(pL_buf.ptr()), 1, 0);
        enc_sk.set_output_array(out,       2);
        enc_sk.set_output_array(logsumexp, 3);
        enc_sk.set_bytes(sk_rp, 4);
        enc_sk.dispatch_threadgroups(
            MTL::Size::Make((size_t)N, (size_t)H, (size_t)B),
            MTL::Size::Make((size_t)reduce_tgp, 1, 1));

        // Lifetime: tie scratch to command-buffer completion (NOT encode time).
        enc_sk.add_temporary(pO_arr);
        enc_sk.add_temporary(pL_arr);
        return;
      }
    }
  }  // end if (!MFA_DISABLE_V2) — split-K block


  // ── STEEL V3 dispatch (f16/bf16, D=64 all gens, D=128 M1/M2 only) ───────
  // Separate K_smem + V_smem → 2 barriers/iter instead of V2's 4.
  //
  // autoresearch (24 iters, M1 Max, 2026-03-20): BK=32 D=64 / BK=16 D=128
  //   Geomean V3/SDPA: 1.47x (causal, large N)
  //   Geomean V3/V2:   1.015x (causal only)
  //   Wins:  D=64 N≥4096 causal, D=128 N≥2048 causal, all B*H≥4
  //   Loses: small N, non-causal
  //
  // Guard sweep (Axis 1, 2026-03-21): V3 wins at all B*H≥4 (worst=0.665x V2).
  // At B=1 H=4 N=2048, grid=256 tiles >> 32 CUs — occupancy is not the limit.
  //
  // RE-VALIDATED on M5 Max / macOS 26.6 (Queue Closure Sprint, 2026-06-17,
  // 3-session §4-strict, V3 vs V2 the fallback): the M1 verdict HOLDS — V3 is
  // faster-or-parity at every measured auto-fire cell. Windowed (the
  // production-reachable path on M5): D=64 N4096 0.68x (V3 ~32% faster),
  // N8192 0.92x; D=128 N4096 0.97x, N8192 ~parity. backend="mfa": D=64 N4096
  // 0.86x, D=128 N4096 ~parity (1.02x). No cell where V3 loses. (D=128 N2048
  // was HIGH_VARIANCE r=0.43 — V3-faster-or-parity in all 3 sessions.)
  // Production routing: V3 dispatched when shape is in the winning regime.
  // Shape guard: causal only, N above threshold per D, B*H≥4.
  // Set MFA_DISABLE_V3=1 to force V2 for benchmarking/debugging.
  //
  // Note: V3 stays BEFORE V2 in dispatch order (V2 is the fallback).
  {
    const int v3_min_N = (D == 64) ? 4096 : 2048;  // N threshold per D
    const bool v3_shape_ok =
        params_.causal &&          // causal only
        (N >= v3_min_N) &&         // large enough sequence
        (B * H >= 4);              // sufficient parallelism (sweep: V3 wins all B*H≥4)
    const bool v3_force = MFAEnvConfig::enable_v3();  // backward compat: bypass shape guard
    const bool v3_eligible =
        !MFAEnvConfig::disable_v3() &&
        (v3_shape_ok || v3_force) &&
        (dtype_code != 2) &&
        v3_tgp_eligible(D, is_m3_plus_steel) &&
        !params_.has_block_mask;

    if (v3_eligible) {
      auto cfg3 = select_steel_v3_block_config(D, is_m3_plus_steel);
      const int BQ3      = cfg3.BQ;   // 32
      const int BK3      = cfg3.BK;   // 32 (D=64) | 16 (D=128)
      const int WM3      = cfg3.WM;   // 4
      const int TGP3     = WM3 * cfg3.WN * 32;  // 128
      const int NQ3      = (N + BQ3 - 1) / BQ3;
      const int NK3      = (S + BK3 - 1) / BK3;
      const int NQ3_aln  = (N % BQ3 == 0) ? NQ3 : NQ3 - 1;
      const int NK3_aln  = (S % BK3 == 0) ? NK3 : NK3 - 1;

      using KK3 = ShaderCache::KernelKey;
      KK3 key3{
        KK3::KernelType::SteelForwardV3,
        D, BQ3, BK3, D, WM3,
        params_.causal,
        /*sparse=*/false,
        is_m3_plus_steel,
        params_.has_rope, params_.rope_interleaved,
        params_.softcap > 0.0f,
        params_.has_alibi,
        /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
        params_.window_left >= 0 || params_.window_right >= 0,
        dtype_code,
        H / Hk
      };

      void* raw3     = ShaderCache::get().get_or_compile(key3, d.mtl_device());
      auto* pipeline3 = reinterpret_cast<MTL::ComputePipelineState*>(raw3);

      MFASteelParams sp3{};
      sp3.B          = B;
      sp3.H          = H;
      sp3.D          = D;
      sp3.qL         = N;
      sp3.kL         = S;
      sp3.gqa_factor = H / Hk;
      sp3.scale      = params_.scale;
      sp3.NQ         = NQ3;
      sp3.NK         = NK3;
      sp3.NQ_aligned = NQ3_aln;
      sp3.NK_aligned = NK3_aln;
      sp3.qL_rem     = (N % BQ3 == 0) ? BQ3 : (N % BQ3);
      sp3.kL_rem     = (S % BK3 == 0) ? BK3 : (S % BK3);
      sp3.qL_off     = (N < S && params_.causal) ? (S - N) : 0;
      sp3.rope_q_base     = params_.cache_seqlens;
      sp3.rope_cos_stride = D / 2;
      sp3.Q_strides[0] = (int64_t)H  * N * D;
      sp3.Q_strides[1] = (int64_t)N  * D;
      sp3.Q_strides[2] = (int64_t)D;
      sp3.K_strides[0] = (int64_t)Hk * S * D;
      sp3.K_strides[1] = (int64_t)S  * D;
      sp3.K_strides[2] = (int64_t)D;
      sp3.V_strides[0] = (int64_t)Hk * S * D;
      sp3.V_strides[1] = (int64_t)S  * D;
      sp3.V_strides[2] = (int64_t)D;
      sp3.O_strides[0] = (int64_t)H  * N * D;
      sp3.O_strides[1] = (int64_t)N  * D;
      sp3.O_strides[2] = (int64_t)D;
      sp3.L_strides[0] = (int64_t)H  * N;
      sp3.L_strides[1] = (int64_t)N;
      sp3.softcap      = params_.softcap;
      sp3.has_alibi    = params_.has_alibi ? 1 : 0;
      sp3.window_left  = params_.window_left;
      sp3.window_right = params_.window_right;
      sp3.mask_batch_stride = 0;
      sp3.mask_head_stride  = 0;

      auto& enc3 = mlx::core::metal::get_command_encoder(stream());
      enc3.set_compute_pipeline_state(pipeline3);
      enc3.set_input_array(q,          0);
      enc3.set_input_array(k,          1);
      enc3.set_input_array(v,          2);
      enc3.set_output_array(out,       3);
      enc3.set_output_array(logsumexp, 4);
      enc3.set_bytes(sp3,              5);
      if (params_.has_rope) {
        enc3.set_input_array(inputs[3], 7);
        enc3.set_input_array(inputs[4], 8);
      }
      if (params_.has_alibi) {
        int alibi_idx = 3 + (params_.has_rope ? 2 : 0);
        enc3.set_input_array(inputs[alibi_idx], 9);
      }

      enc3.dispatch_threadgroups(
          MTL::Size::Make((size_t)NQ3, (size_t)H, (size_t)B),
          MTL::Size::Make((size_t)TGP3, 1, 1));
      return;
    }
  }  // end V3 dispatch block

  // ── STEEL V2 dispatch (f16/bf16, D=64/128 only) ──────────────────────────
  // BQ=32 (TQ=1), BK=64 (D=64) / BK=32 (D=128): sequential KV_smem, 2× BK vs V1.
  // D=256 excluded: routes to V1 (BQ=32, BK=16, WM=4, TGP=128).
  // Sparse (block_mask) excluded: mask is sized for V1 BK (BK_v1 != BK_v2).
  // Set MFA_DISABLE_V2=1 to bypass (forces V1 path, useful for benchmarking).
  if (!MFAEnvConfig::disable_v2()) {
    // M3+ (gen>=15): V1 double-buffer is 1.5-3.7x faster than V2 at D<=128 causal.
    // V2's shared KV_smem requires 3-4 barriers/tile vs V1's 2 barriers/tile.
    // On M3+ hardware, reduced TGP bandwidth makes barriers more expensive.
    // Skip V2 for this regime, falling through to V1.
    // Override: set MFA_FORCE_V2=1 to bypass this guard (benchmarking).
    //
    // v2.50 Prompt 5b Section C: EXCEPTION for has_attn_bias.  V1 STEEL kernel
    // does NOT implement bias addition (params struct has the fields but no
    // code emits the bias-add).  Sending D<=128 + causal + bias to V1 silently
    // drops the bias → wrong output (max_err ~0.30 vs SDPA reference for
    // mode 1/2).  Force-route to V2 when bias is present so the bias addition
    // logic at mfa_steel_fwd_v2.cpp:609-645 is exercised.  V2 perf at
    // D<=128 causal is slightly slower than V1, but correctness > perf for
    // this narrow combo.  Test coverage: tests/test_attn_bias_native.py
    // ::TestBiasMode{1,2}::test_d128_causal.
    const bool m3_prefers_v1 = is_m3_plus_steel && D <= 128 && params_.causal
                               && !params_.has_attn_bias
                               && !MFAEnvConfig::force_v2();
    const bool v2_eligible =
        (dtype_code != 2) &&
        is_v2_small_d_family(D) &&
        // block_mask is sized for V1 tile BK (BK_v1 ≠ BK_v2) — route to V1.
        !params_.has_block_mask &&
        !m3_prefers_v1;

    if (v2_eligible) {
      auto cfg2 = select_steel_v2_block_config(D, is_m3_plus_steel);
      const int BQ2      = cfg2.BQ;   // 32
      const int BK2      = cfg2.BK;   // 64 (D=64) | M1/M2 D=128:32, M3+ D=128:64
      const int WM2      = cfg2.WM;   // 4
      const int TGP2     = WM2 * cfg2.WN * 32;  // 128
      const int NQ2      = (N + BQ2 - 1) / BQ2;
      const int NK2      = (S + BK2 - 1) / BK2;
      const int NQ2_aln  = (N % BQ2 == 0) ? NQ2 : NQ2 - 1;
      const int NK2_aln  = (S % BK2 == 0) ? NK2 : NK2 - 1;

      using KK2 = ShaderCache::KernelKey;
      KK2 key2{
        KK2::KernelType::SteelForwardV2,
        D, BQ2, BK2, D, WM2,
        params_.causal,
        /*sparse=*/false,        // sparse routes to V1 (mask sized for V1 BK)
        is_m3_plus_steel,
        params_.has_rope, params_.rope_interleaved,
        params_.softcap > 0.0f,
        params_.has_alibi,
        params_.has_attn_bias, params_.attn_bias_mode,
        params_.window_left >= 0 || params_.window_right >= 0,
        dtype_code,
        H / Hk
      };

      void* raw2     = ShaderCache::get().get_or_compile(key2, d.mtl_device());
      auto* pipeline2 = reinterpret_cast<MTL::ComputePipelineState*>(raw2);

      MFASteelParams sp2{};
      sp2.B          = B;
      sp2.H          = H;
      sp2.D          = D;
      sp2.qL         = N;
      sp2.kL         = S;
      sp2.gqa_factor = H / Hk;
      sp2.scale      = params_.scale;
      sp2.NQ         = NQ2;
      sp2.NK         = NK2;
      sp2.NQ_aligned = NQ2_aln;
      sp2.NK_aligned = NK2_aln;
      sp2.qL_rem     = (N % BQ2 == 0) ? BQ2 : (N % BQ2);
      sp2.kL_rem     = (S % BK2 == 0) ? BK2 : (S % BK2);
      sp2.qL_off     = (N < S && params_.causal) ? (S - N) : 0;
      sp2.rope_q_base     = params_.cache_seqlens;
      sp2.rope_cos_stride = D / 2;
      sp2.Q_strides[0] = (int64_t)H  * N * D;
      sp2.Q_strides[1] = (int64_t)N  * D;
      sp2.Q_strides[2] = (int64_t)D;
      sp2.K_strides[0] = (int64_t)Hk * S * D;
      sp2.K_strides[1] = (int64_t)S  * D;
      sp2.K_strides[2] = (int64_t)D;
      sp2.V_strides[0] = (int64_t)Hk * S * D;
      sp2.V_strides[1] = (int64_t)S  * D;
      sp2.V_strides[2] = (int64_t)D;
      sp2.O_strides[0] = (int64_t)H  * N * D;
      sp2.O_strides[1] = (int64_t)N  * D;
      sp2.O_strides[2] = (int64_t)D;
      sp2.L_strides[0] = (int64_t)H  * N;
      sp2.L_strides[1] = (int64_t)N;
      sp2.softcap      = params_.softcap;
      sp2.has_alibi    = params_.has_alibi ? 1 : 0;
      sp2.window_left  = params_.window_left;
      sp2.window_right = params_.window_right;
      // V2 is never sparse (block_mask sized for V1 BK ≠ V2 BK).
      sp2.mask_batch_stride = 0;
      sp2.mask_head_stride  = 0;
      sp2.has_attn_bias     = params_.has_attn_bias ? 1 : 0;
      sp2.attn_bias_mode    = params_.attn_bias_mode;
      sp2.attn_bias_nkv     = S;  // N_kv for bias indexing

      auto& enc2 = mlx::core::metal::get_command_encoder(stream());
      enc2.set_compute_pipeline_state(pipeline2);
      enc2.set_input_array(q,          0);
      enc2.set_input_array(k,          1);
      enc2.set_input_array(v,          2);
      enc2.set_output_array(out,       3);
      enc2.set_output_array(logsumexp, 4);
      enc2.set_bytes(sp2,              5);
      // buffer(6) = block_mask: unused in V2 (sparse routes to V1)
      if (params_.has_rope) {
        // Dense + RoPE: inputs[3]=cos, inputs[4]=sin (no block_mask in V2 path)
        enc2.set_input_array(inputs[3], 7);
        enc2.set_input_array(inputs[4], 8);
      }
      if (params_.has_alibi) {
        // Dense + ALiBi: inputs[3]=alibi_slopes (no block_mask, rope may or may not be set)
        int alibi_idx = 3 + (params_.has_rope ? 2 : 0);
        enc2.set_input_array(inputs[alibi_idx], 9);
      }
      if (params_.has_attn_bias) {
        // attn_bias tensor: after alibi_slopes (if present), rope (if present)
        int bias_idx = 3 + (params_.has_rope ? 2 : 0) + (params_.has_alibi ? 1 : 0);
        enc2.set_input_array(inputs[bias_idx], 10);
      }

      // One threadgroup per Q-block: grid x = NQ2.
      enc2.dispatch_threadgroups(
          MTL::Size::Make((size_t)NQ2, (size_t)H, (size_t)B),
          MTL::Size::Make((size_t)TGP2, 1, 1));
      return;
    }
  }  // end if (!MFA_DISABLE_V2) — single-pass block

  // ── STEEL V2 D-split dispatch (f16/bf16, D=256/512) ─────────────────────
  // BD_HALF=128; D_SPLITS=D/128 (2 for D=256, 4 for D=512).
  // Uses dedicated large-D config selector (BK=32/64, WM=4, TGP=128).
  // D=128 BK calibration override (MFA_V2_FORCE_BK) does not affect D-split.
  // No RoPE (GPT-NeoX pairs cross BD_HALF boundary).
  // Sparse excluded (block_mask sized for V1 BK).
  // Set MFA_DISABLE_V2=1 to bypass.
  if (!MFAEnvConfig::disable_v2()) {
    const bool v2_dsplit_eligible =
        (dtype_code != 2) &&
        is_v2_dsplit_family(D) &&
        !params_.has_block_mask &&
        !params_.has_rope;

    if (v2_dsplit_eligible) {
      const bool is_d256_path = is_v2_d256_family(D);
      const bool is_d512_path = is_v2_d512_family(D);
      int BD_HALF = (D == 512) ? 32 : 128;
      if (D == 512 && env.v2_bd_half_d512 > 0) {
        const int v = env.v2_bd_half_d512;
        if (v == 32 || v == 64 || v == 128) BD_HALF = v;
      }
      auto cfg_ds       = (D == 512)
          ? select_steel_v2_d512_block_config(is_m3_plus_steel)
          : select_steel_v2_dsplit_block_config(is_m3_plus_steel);
      const int BQ_ds   = cfg_ds.BQ;   // 32
      const int BK_ds   = cfg_ds.BK;   // 32 (M1/M2) or 64 (M3+)
      const int WM_ds   = cfg_ds.WM;   // 4
      const int TGP_ds  = WM_ds * cfg_ds.WN * 32;  // 128
      const int NQ_ds  = (N + BQ_ds - 1) / BQ_ds;
      const int NK_ds  = (S + BK_ds - 1) / BK_ds;
      const int NQ_ds_aln = (N % BQ_ds == 0) ? NQ_ds : NQ_ds - 1;
      const int NK_ds_aln = (S % BK_ds == 0) ? NK_ds : NK_ds - 1;

      using KKds = ShaderCache::KernelKey;
      // Keep the kernel-type branch explicit so D=512 decisions remain visible.
      auto kt_ds = KKds::KernelType::SteelV2DSplit512;
      if (is_d256_path) {
        kt_ds = KKds::KernelType::SteelV2DSplit256;
      } else if (is_d512_path) {
        kt_ds = KKds::KernelType::SteelV2DSplit512;
      }
      KKds key_ds{
        kt_ds,
        D, BQ_ds, BK_ds, BD_HALF, WM_ds,
        params_.causal,
        /*sparse=*/false,
        is_m3_plus_steel,
        /*has_rope=*/false, /*rope_interleaved=*/false,
        params_.softcap > 0.0f,
        params_.has_alibi,
        params_.has_attn_bias, params_.attn_bias_mode,
        params_.window_left >= 0 || params_.window_right >= 0,
        dtype_code,
        H / Hk
      };

      void* raw_ds      = ShaderCache::get().get_or_compile(key_ds, d.mtl_device());
      auto* pipeline_ds = reinterpret_cast<MTL::ComputePipelineState*>(raw_ds);

      MFASteelParams sp_ds{};
      sp_ds.B          = B;
      sp_ds.H          = H;
      sp_ds.D          = D;
      sp_ds.qL         = N;
      sp_ds.kL         = S;
      sp_ds.gqa_factor = H / Hk;
      sp_ds.scale      = params_.scale;
      sp_ds.NQ         = NQ_ds;
      sp_ds.NK         = NK_ds;
      sp_ds.NQ_aligned = NQ_ds_aln;
      sp_ds.NK_aligned = NK_ds_aln;
      sp_ds.qL_rem     = (N % BQ_ds == 0) ? BQ_ds : (N % BQ_ds);
      sp_ds.kL_rem     = (S % BK_ds == 0) ? BK_ds : (S % BK_ds);
      sp_ds.qL_off     = (N < S && params_.causal) ? (S - N) : 0;
      sp_ds.rope_q_base     = params_.cache_seqlens;
      sp_ds.rope_cos_stride = D / 2;
      sp_ds.Q_strides[0] = (int64_t)H  * N * D;
      sp_ds.Q_strides[1] = (int64_t)N  * D;
      sp_ds.Q_strides[2] = (int64_t)D;
      sp_ds.K_strides[0] = (int64_t)Hk * S * D;
      sp_ds.K_strides[1] = (int64_t)S  * D;
      sp_ds.K_strides[2] = (int64_t)D;
      sp_ds.V_strides[0] = (int64_t)Hk * S * D;
      sp_ds.V_strides[1] = (int64_t)S  * D;
      sp_ds.V_strides[2] = (int64_t)D;
      sp_ds.O_strides[0] = (int64_t)H  * N * D;
      sp_ds.O_strides[1] = (int64_t)N  * D;
      sp_ds.O_strides[2] = (int64_t)D;
      sp_ds.L_strides[0] = (int64_t)H  * N;
      sp_ds.L_strides[1] = (int64_t)N;
      sp_ds.softcap      = params_.softcap;
      sp_ds.has_alibi    = params_.has_alibi ? 1 : 0;
      sp_ds.window_left  = params_.window_left;
      sp_ds.window_right = params_.window_right;
      sp_ds.mask_batch_stride = 0;
      sp_ds.mask_head_stride  = 0;
      sp_ds.has_attn_bias     = params_.has_attn_bias ? 1 : 0;
      sp_ds.attn_bias_mode    = params_.attn_bias_mode;
      sp_ds.attn_bias_nkv     = S;

      auto& enc_ds = mlx::core::metal::get_command_encoder(stream());
      enc_ds.set_compute_pipeline_state(pipeline_ds);
      enc_ds.set_input_array(q,          0);
      enc_ds.set_input_array(k,          1);
      enc_ds.set_input_array(v,          2);
      enc_ds.set_output_array(out,       3);
      enc_ds.set_output_array(logsumexp, 4);
      enc_ds.set_bytes(sp_ds,            5);
      if (params_.has_alibi) {
        // No rope, no block_mask in D-split path: ALiBi slopes at inputs[3].
        enc_ds.set_input_array(inputs[3], 9);
      }
      if (params_.has_attn_bias) {
        int bias_idx = 3 + (params_.has_alibi ? 1 : 0);
        enc_ds.set_input_array(inputs[bias_idx], 10);
      }

      enc_ds.dispatch_threadgroups(
          MTL::Size::Make((size_t)NQ_ds, (size_t)H, (size_t)B),
          MTL::Size::Make((size_t)TGP_ds, 1, 1));
      return;
    }
  }  // end if (!MFA_DISABLE_V2) — D-split block

  // ── STEEL V1 tile config (f16 / bf16, or V2-disabled fallback) ───────────
  // III-4 D4 FIX: block masks are built and validated Python-side with
  // the BASE (non-M3+) geometry — `_steel_block_config(128)` = (BQ=32,
  // BK=16) unconditionally.  Passing is_m3_plus here made the D=128
  // sparse kernel index `block_mask[qb * NK + kb]` with NK = ceil(S/32)
  // against a buffer whose row stride is ceil(S/16) — silently wrong
  // sparse output on M3/M4 at D=128.  Force the base config whenever a
  // block mask is bound so kernel NK matches the mask geometry
  // (mirrors the V2/V5 sparse exclusions, which exist for this reason).
  auto cfg = select_steel_block_config(
      D, /*is_low_prec=*/true,
      params_.has_block_mask ? false : is_m3_plus_steel);
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
    params_.has_rope,           // in-kernel RoPE fusion variant
    params_.rope_interleaved,   // true=LLaMA, false=GPT-NeoX
    params_.softcap > 0.0f,    // tanh softcapping variant
    params_.has_alibi,          // ALiBi per-head position biases
    /*has_attn_bias=*/false, /*attn_bias_mode=*/0,  // V1 doesn't support attn_bias
    params_.window_left >= 0 || params_.window_right >= 0, // sliding window variant
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
  sp.softcap     = params_.softcap;           // 0.0 when disabled
  sp.has_alibi   = params_.has_alibi ? 1 : 0;
  sp.window_left  = params_.window_left;     // -1 when disabled
  sp.window_right = params_.window_right;    // -1 when disabled

  // Block-sparse mask strides — 0 = broadcast this dimension.
  // 2D [NQ, NK]:       both = 0  (one mask shared by all B, H)
  // 3D [H, NQ, NK]:    batch=0, head=NQ*NK
  // 4D [B, H, NQ, NK]: batch=H*NQ*NK, head=NQ*NK
  sp.mask_batch_stride = 0;
  sp.mask_head_stride  = 0;
  if (params_.has_block_mask) {
    const auto& bm = inputs[3];
    if (bm.ndim() == 3) {
      sp.mask_head_stride  = (int64_t)bm.shape(1) * bm.shape(2);
    } else if (bm.ndim() == 4) {
      sp.mask_head_stride  = (int64_t)bm.shape(2) * bm.shape(3);
      sp.mask_batch_stride = (int64_t)bm.shape(1) * sp.mask_head_stride;
    }
    // 2D: both remain 0 (initialized above)
  }

  // ── Dispatch ─────────────────────────────────────────────────────────────
  auto& enc = mlx::core::metal::get_command_encoder(stream());
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

  // 3D grid: persistent kernel — each TG handles 4 Q-tiles, so grid shrinks to ceil(NQ/4)
  static constexpr int kTilesPerTG = 4;
  const int NQ_tgs = (NQ + kTilesPerTG - 1) / kTilesPerTG;
  enc.dispatch_threadgroups(
      MTL::Size::Make(NQ_tgs, H, B),
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
  const auto& dO = cotangents[0];

  mlx::core::Shape d_shape = {q.shape(0), q.shape(1), q.shape(2)};
  const int D_val = static_cast<int>(q.shape(3));

  // Route f16/bf16 D≤128 to the STEEL backward kernels.
  // GQA (H_q != H_kv) is supported: gqa_factor is set in BwdDQ/BwdDKV.
  // Repo review 2026-05: `!has_block_mask` guard added.  The STEEL backward
  // path passes 7 inputs to BwdDQ, but BwdDQ::eval_gpu asserts
  // `7 + (has_block_mask ? 1 : 0)` — a sparse primitive reaching this vjp
  // directly (bypassing the Python sparse custom_function) would trip the
  // assertion / UB.  Sparse backward must use the dedicated sparse path.
  const bool use_steel_bwd =
      (q.dtype() != mlx::core::float32) &&
      (D_val <= 128) &&
      !params_.has_block_mask;

  std::vector<mlx::core::array> all_grads;

  if (use_steel_bwd) {
    // ---- STEEL backward path -----------------------------------------------
    //
    // delta = rowsum(dO * O)  [B, H, N], float32.
    // NOTE: do NOT pre-multiply by scale here.  The Metal kernel computes
    // dS = scale * P * (dP - delta), so delta must be the raw row-dot-product;
    // adding scale here would double-scale the subtracted term.
    // Computing this as a lazy MLX op lets the graph scheduler guarantee it
    // is ready before both bwd kernels execute; no explicit buffer chain needed.
    auto dO_f32    = mlx::core::astype(dO, mlx::core::float32, stream());
    auto O_f32     = mlx::core::astype(O,  mlx::core::float32, stream());
    auto dot_prod  = mlx::core::multiply(dO_f32, O_f32, stream());
    auto delta     = mlx::core::sum(dot_prod, std::vector<int>{3}, false, stream());

    // Step 1: STEEL dQ — grid (NQ, H, B), TGP = WM*32.
    // Buffer layout: Q(0),K(1),V(2),O(3-unused),L(4),dO(5),delta(6),dQ(7),p(8)
    auto bwd_q = mlx::core::array::make_arrays(
        {q.shape()},
        {q.dtype()},
        std::make_shared<MFASteelBwdDQ>(stream(), params_),
        {q, k, v, O, L, dO, delta});

    // Step 2: STEEL dKV — grid (NK, H, B), TGP = 32.
    // Buffer layout: Q(0),K(1),V(2),O(3-unused),L(4),delta(5),dO(6),dK(7),dV(8),p(9)
    auto bwd_kv = mlx::core::array::make_arrays(
        {k.shape(), v.shape()},
        {k.dtype(), v.dtype()},
        std::make_shared<MFASteelBwdDKV>(stream(), params_),
        {q, k, v, O, L, delta, dO});

    all_grads = {bwd_q[0], bwd_kv[0], bwd_kv[1]};

  } else {
    // ---- Legacy ccv backward path ------------------------------------------
    // Step 1: MFABackwardQuery → [dQ, D_computed]
    auto bwd_q = mlx::core::array::make_arrays(
        {q.shape(),         d_shape},
        {q.dtype(),  mlx::core::float32},
        std::make_shared<MFABackwardQuery>(stream(), params_),
        {q, k, v, O, L, dO});

    const auto& D_kernel = bwd_q[1];  // [B,H,N] f32, written by kernel

    // Step 2: MFABackwardKeyValue → [dK, dV]
    auto bwd_kv = mlx::core::array::make_arrays(
        {k.shape(),  v.shape()},
        {k.dtype(), v.dtype()},
        std::make_shared<MFABackwardKeyValue>(stream(), params_),
        {q, k, v, O, L, D_kernel, dO});

    all_grads = {bwd_q[0], bwd_kv[0], bwd_kv[1]};
  }

  // Return only the gradients for the requested argnums (0→dQ, 1→dK, 2→dV).
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
  int arch_gen_bwdq = static_cast<int>(dev.get_architecture_gen());
  { const auto& e = MFAEnvConfig::get(); if (e.force_gen > 0) arch_gen_bwdq = e.force_gen; }
  bool is_m3_plus = (arch_gen_bwdq >= 15); // 13=M1 14=M2 15=M3 16=M4

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
    /*rope_interleaved=*/false,
    /*has_softcap=*/false, /*has_alibi=*/false,
    /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
    /*has_window=*/false, dtype_code
  };
  void* raw = ShaderCache::get().get_or_compile(key, dev.mtl_device());
  auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw);

  auto& enc = mlx::core::metal::get_command_encoder(stream());
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
  int arch_gen_bwdkv = static_cast<int>(dev.get_architecture_gen());
  { const auto& e = MFAEnvConfig::get(); if (e.force_gen > 0) arch_gen_bwdkv = e.force_gen; }
  bool is_m3_plus = (arch_gen_bwdkv >= 15); // 13=M1 14=M2 15=M3 16=M4

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
    /*rope_interleaved=*/false,
    /*has_softcap=*/false, /*has_alibi=*/false,
    /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
    /*has_window=*/false, dtype_code
  };
  void* raw = ShaderCache::get().get_or_compile(key, dev.mtl_device());
  auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw);

  auto& enc = mlx::core::metal::get_command_encoder(stream());
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
// MFASteelBwdDQ::eval_gpu
// =========================================================================
//
// Dispatches the STEEL dQ backward kernel.
// Grid: (NQ, H, B) — one threadgroup per Q-tile.
//
// Buffer assignments (match generated Metal source):
//   Q(0), K(1), V(2), O(3-unused), L(4), dO(5), delta(6), dQ(7), params(8)

void MFASteelBwdDQ::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  // inputs: Q, K, V, O, L, dO, delta, [block_mask if sparse]
  assert(inputs.size()  == 7u + (params_.has_block_mask ? 1u : 0u));
  assert(outputs.size() == 1);  // dQ

  const auto& q     = inputs[0];  // [B, H, N, D]
  const auto& k     = inputs[1];  // [B, H, S, D]
  const auto& v     = inputs[2];  // [B, H, S, D]
  const auto& o     = inputs[3];  // [B, H, N, D]  (bound but not read)
  const auto& l     = inputs[4];  // [B, H, N], float32
  const auto& dO    = inputs[5];  // [B, H, N, D]
  const auto& delta = inputs[6];  // [B, H, N], float32

  const int B = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  const int S = k.shape(2), Hk = k.shape(1);  // Hk = H_kv for GQA

  auto& dQ = outputs[0];
  dQ.set_data(mlx::core::allocator::malloc(dQ.nbytes()));

  auto& dev = mlx::core::metal::device(stream().device);
  int arch_gen = static_cast<int>(dev.get_architecture_gen());
  if (MFAEnvConfig::get().force_gen > 0) arch_gen = MFAEnvConfig::get().force_gen;
  const bool is_m3_plus = (arch_gen >= 15);

  uint8_t dtype_code;
  if (q.dtype() == mlx::core::float16)       dtype_code = 0;
  else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
  else                                        dtype_code = 2;

  // III-4 D4 FIX (sparse geometry, same class as the forward site):
  // block masks are built Python-side with the BASE config; force it
  // when a mask is bound so kernel NK matches the mask row stride.
  const auto cfg = select_steel_block_config(
      D, /*is_low_prec=*/dtype_code != 2,
      params_.has_block_mask ? false : is_m3_plus);
  const int BQ = cfg.BQ, BK = cfg.BK, BD = cfg.BD, WM = cfg.WM;
  const int NQ = (N + BQ - 1) / BQ;
  const int NK = (S + BK - 1) / BK;

  MFASteelBackwardParams sp{};
  sp.B            = B;   sp.H = H;   sp.D = D;
  sp.qL           = N;   sp.kL = S;
  sp.gqa_factor   = H / Hk;  // 1 for standard MHA; >1 for GQA
  sp.scale        = params_.scale;
  sp.scale_log2   = params_.scale * static_cast<float>(M_LOG2E);
  sp.NQ           = NQ;  sp.NK = NK;
  sp.NQ_aligned   = N / BQ;
  sp.NK_aligned   = S / BK;
  sp.qL_rem       = N % BQ;
  sp.kL_rem       = S % BK;
  sp.qL_off       = 0;
  // Strides: [B_stride, H_stride, seq_stride] for each operand.
  // K/V batch stride uses Hk (= H_kv), not H_q, since K/V are [B, H_kv, S, D].
  sp.Q_strides[0]  = (int64_t)H  * N * D;  sp.Q_strides[1]  = (int64_t)N * D;  sp.Q_strides[2]  = D;
  sp.K_strides[0]  = (int64_t)Hk * S * D;  sp.K_strides[1]  = (int64_t)S * D;  sp.K_strides[2]  = D;
  sp.V_strides[0]  = (int64_t)Hk * S * D;  sp.V_strides[1]  = (int64_t)S * D;  sp.V_strides[2]  = D;
  sp.O_strides[0]  = (int64_t)H  * N * D;  sp.O_strides[1]  = (int64_t)N * D;  sp.O_strides[2]  = D;
  sp.dO_strides[0] = (int64_t)H  * N * D;  sp.dO_strides[1] = (int64_t)N * D;  sp.dO_strides[2] = D;
  sp.dQ_strides[0] = (int64_t)H  * N * D;  sp.dQ_strides[1] = (int64_t)N * D;  sp.dQ_strides[2] = D;
  sp.dK_strides[0] = (int64_t)Hk * S * D;  sp.dK_strides[1] = (int64_t)S * D;  sp.dK_strides[2] = D;
  sp.dV_strides[0] = (int64_t)Hk * S * D;  sp.dV_strides[1] = (int64_t)S * D;  sp.dV_strides[2] = D;
  sp.L_strides[0]  = (int64_t)H  * N;      sp.L_strides[1]  = N;

  using KK = ShaderCache::KernelKey;
  KK key{KK::KernelType::SteelBackwardDQ,
         D, BQ, BK, BD, WM,
         params_.causal, /*sparse=*/params_.has_block_mask, is_m3_plus,
         /*has_rope=*/false, /*rope_interleaved=*/false,
         /*has_softcap=*/false, /*has_alibi=*/false,
         /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
         /*has_window=*/false, dtype_code,
         /*gqa_factor=*/H / Hk};
  void* raw = ShaderCache::get().get_or_compile(key, dev.mtl_device());
  auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw);

  auto& enc = mlx::core::metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(pipeline);
  enc.set_input_array(q,     0);
  enc.set_input_array(k,     1);
  enc.set_input_array(v,     2);
  enc.set_input_array(o,     3);  // declared in kernel; not read
  enc.set_input_array(l,     4);
  enc.set_input_array(dO,    5);
  enc.set_input_array(delta, 6);
  enc.set_output_array(dQ,   7);
  enc.set_bytes(sp,          8);
  if (params_.has_block_mask) {
    enc.set_input_array(inputs[7], 9);  // block_mask [NQ_tiles, NK_tiles] uchar
  }

  enc.dispatch_threadgroups(
      MTL::Size::Make(NQ, H, B),
      MTL::Size::Make(WM * 32, 1, 1));
}

// =========================================================================
// MFASteelBwdDKV::eval_gpu
// =========================================================================
//
// Dispatches the STEEL dK/dV backward kernel.
// Grid: (NK, H, B) — one threadgroup per K/V-tile (WM=1, TGP=32).
//
// Buffer assignments (match generated Metal source):
//   Q(0), K(1), V(2), O(3-unused), L(4), delta(5), dO(6),
//   dK(7), dV(8), params(9)

void MFASteelBwdDKV::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  // inputs: Q, K, V, O, L, delta, dO, [block_mask if sparse]
  assert(inputs.size()  == 7u + (params_.has_block_mask ? 1u : 0u));
  assert(outputs.size() == 2);  // dK, dV

  const auto& q     = inputs[0];
  const auto& k     = inputs[1];
  const auto& v     = inputs[2];
  const auto& o     = inputs[3];  // declared in kernel; not read
  const auto& l     = inputs[4];
  const auto& delta = inputs[5];  // [B, H, N], float32
  const auto& dO    = inputs[6];

  const int B = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  const int S = k.shape(2), Hk = k.shape(1);  // Hk = H_kv for GQA

  auto& dK = outputs[0];
  auto& dV = outputs[1];
  dK.set_data(mlx::core::allocator::malloc(dK.nbytes()));
  dV.set_data(mlx::core::allocator::malloc(dV.nbytes()));

  auto& dev = mlx::core::metal::device(stream().device);
  int arch_gen = static_cast<int>(dev.get_architecture_gen());
  if (MFAEnvConfig::get().force_gen > 0) arch_gen = MFAEnvConfig::get().force_gen;
  const bool is_m3_plus = (arch_gen >= 15);

  uint8_t dtype_code;
  if (q.dtype() == mlx::core::float16)       dtype_code = 0;
  else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
  else                                        dtype_code = 2;

  // III-4 D4 FIX: base config under block masks (see MFASteelBwdDQ note).
  const auto cfg = select_steel_block_config(
      D, /*is_low_prec=*/dtype_code != 2,
      params_.has_block_mask ? false : is_m3_plus);
  const int BQ = cfg.BQ, BD = cfg.BD;
  // KD-5 root-cause fix (repo review 2026-05): generate_steel_backward_dkv_source
  // overrides BK to 16 for D > 64 (mfa_steel_bwd.cpp: `BK = (BD <= 64) ? cfg.BK : 16`).
  // Pre-fix, this dispatch used cfg.BK (= 32 on M3+ for D=128), launching
  // NK = ceil(S/32) threadgroups while the compiled kernel processes 16 K-rows
  // each at 16-row strides — leaving K-rows beyond NK*16 unwritten (dK/dV
  // zeroed for the upper half of rows at D=128 N>=2048 on M3+).  The grid and
  // params BK MUST mirror the generator's override.
  const int BK = (D <= 64) ? cfg.BK : 16;
  const int NK = (S + BK - 1) / BK;
  const int NQ = (N + BQ - 1) / BQ;
  // dKV kernel hardcodes WM=1 (single simdgroup, no inter-warp race).
  constexpr int WM_DKV = 1;

  MFASteelBackwardParams sp{};
  sp.B            = B;   sp.H = H;   sp.D = D;
  sp.qL           = N;   sp.kL = S;
  sp.gqa_factor   = H / Hk;  // 1 for standard MHA; >1 for GQA
  sp.scale        = params_.scale;
  sp.scale_log2   = params_.scale * static_cast<float>(M_LOG2E);
  sp.NQ           = NQ;  sp.NK = NK;
  sp.NQ_aligned   = N / BQ;
  sp.NK_aligned   = S / BK;
  sp.qL_rem       = N % BQ;
  sp.kL_rem       = S % BK;
  sp.qL_off       = 0;
  // K/V/dK/dV batch strides use Hk (H_kv) — those tensors are [B, H_kv, S, D].
  sp.Q_strides[0]  = (int64_t)H  * N * D;  sp.Q_strides[1]  = (int64_t)N * D;  sp.Q_strides[2]  = D;
  sp.K_strides[0]  = (int64_t)Hk * S * D;  sp.K_strides[1]  = (int64_t)S * D;  sp.K_strides[2]  = D;
  sp.V_strides[0]  = (int64_t)Hk * S * D;  sp.V_strides[1]  = (int64_t)S * D;  sp.V_strides[2]  = D;
  sp.O_strides[0]  = (int64_t)H  * N * D;  sp.O_strides[1]  = (int64_t)N * D;  sp.O_strides[2]  = D;
  sp.dO_strides[0] = (int64_t)H  * N * D;  sp.dO_strides[1] = (int64_t)N * D;  sp.dO_strides[2] = D;
  sp.dQ_strides[0] = (int64_t)H  * N * D;  sp.dQ_strides[1] = (int64_t)N * D;  sp.dQ_strides[2] = D;
  sp.dK_strides[0] = (int64_t)Hk * S * D;  sp.dK_strides[1] = (int64_t)S * D;  sp.dK_strides[2] = D;
  sp.dV_strides[0] = (int64_t)Hk * S * D;  sp.dV_strides[1] = (int64_t)S * D;  sp.dV_strides[2] = D;
  sp.L_strides[0]  = (int64_t)H  * N;      sp.L_strides[1]  = N;

  using KK = ShaderCache::KernelKey;
  KK key{KK::KernelType::SteelBackwardDKV,
         D, BQ, BK, BD, WM_DKV,
         params_.causal, /*sparse=*/params_.has_block_mask, is_m3_plus,
         /*has_rope=*/false, /*rope_interleaved=*/false,
         /*has_softcap=*/false, /*has_alibi=*/false,
         /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
         /*has_window=*/false, dtype_code,
         /*gqa_factor=*/H / Hk};
  void* raw = ShaderCache::get().get_or_compile(key, dev.mtl_device());
  auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw);

  auto& enc = mlx::core::metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(pipeline);
  enc.set_input_array(q,     0);
  enc.set_input_array(k,     1);
  enc.set_input_array(v,     2);
  enc.set_input_array(o,     3);  // declared in kernel; not read
  enc.set_input_array(l,     4);
  enc.set_input_array(delta, 5);
  enc.set_input_array(dO,    6);
  enc.set_output_array(dK,   7);
  enc.set_output_array(dV,   8);
  enc.set_bytes(sp,          9);
  if (params_.has_block_mask) {
    enc.set_input_array(inputs[7], 10);  // block_mask [NQ_tiles, NK_tiles] uchar
  }

  // dKV grid Y = Hk (H_kv) — each TG handles one KV-head, iterating Q-heads
  enc.dispatch_threadgroups(
      MTL::Size::Make(NK, Hk, B),
      MTL::Size::Make(WM_DKV * 32, 1, 1));
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
         params_.has_rope          == o->params_.has_rope          &&
         params_.rope_interleaved  == o->params_.rope_interleaved  &&
         params_.cache_seqlens     == o->params_.cache_seqlens     &&
         params_.softcap        == o->params_.softcap        &&
         params_.has_alibi      == o->params_.has_alibi      &&
         params_.has_attn_bias  == o->params_.has_attn_bias  &&
         params_.attn_bias_mode == o->params_.attn_bias_mode &&
         params_.window_left    == o->params_.window_left    &&
         params_.window_right   == o->params_.window_right;
}

// =========================================================================
// Shared raw dense-QKV validator (volet K1).
// The dense raw bindings (mfa_attention_forward + the feature variants alibi/
// bias/rope/sparse) derive B from Q and ALL K/V strides from K, then read V at
// K's offsets — so a Q/K/V that disagrees on batch, kv-seq, kv-heads, the Q@K^T
// head_dim, GQA divisibility, or dtype reads OOB / silent-wrong (observed:
// batch→NaN, k_seq/k_heads/q_D/dtype all no-raise on alibi/rope/sparse). One
// validator so the feature entries cannot drift. CC Batch Class C: it now ALSO
// constrains v.shape(3)==q.shape(3) — these raw kernels allocate output from
// q.shape, so asymmetric D_v is a wrong-shape silent-wrong here (public wrappers
// keep asym D_v via SDPA fallback). Rule 8: raise before dispatch.
// =========================================================================
static void validate_dense_qkv(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const char* fn) {
  auto bad = [&](const std::string& m) {
    throw std::invalid_argument(std::string(fn) + ": " + m);
  };
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4)
    bad("expected 4D inputs [B, H, N, D]");
  if (q.shape(0) != k.shape(0) || q.shape(0) != v.shape(0))
    bad("q, k, v must share the batch dim (Bq=" + std::to_string(q.shape(0)) +
        ", Bk=" + std::to_string(k.shape(0)) + ", Bv=" + std::to_string(v.shape(0)) + ")");
  if (k.shape(2) != v.shape(2))
    bad("k and v must share the kv sequence length (Sk=" +
        std::to_string(k.shape(2)) + ", Sv=" + std::to_string(v.shape(2)) + ")");
  // ZERO-KV reject (raw-host parity sweep — the zero-KV class, swept to the SOURCE):
  // the public flash_attention raises on empty KV; attention over zero keys is a
  // 0/0 online-softmax -> silent NaN. One edit here covers every dense+sparse raw
  // entry that calls validate_dense_qkv (forward, forward_with_lse, alibi, bias,
  // rope, sparse[_with_lse]).
  if (k.shape(2) == 0)
    bad("empty KV (k/v sequence length 0) is undefined — attention over zero keys "
        "yields 0/0 -> NaN; provide at least one key.");
  if (k.shape(1) != v.shape(1))
    bad("k and v must have the same number of heads (Hk=" +
        std::to_string(k.shape(1)) + ", Hv=" + std::to_string(v.shape(1)) + ")");
  if (q.shape(3) != k.shape(3))
    bad("q and k must share head_dim for Q@K^T (Dq=" +
        std::to_string(q.shape(3)) + ", Dk=" + std::to_string(k.shape(3)) + ")");
  // CC Batch Class C: these raw kernels allocate the output from q.shape, so an
  // asymmetric value head_dim (D_v != D_qk) silently returns a q-D-shaped output
  // (or non-finite on sparse) instead of SDPA's V-D shape — a WRONG-SHAPE
  // silent-wrong. Reject D_v != D_qk at the raw boundary (the kernel's real
  // contract). The PUBLIC functions keep asymmetric D_v via their SDPA fallback,
  // so the capability is retained at the public layer; raw enforces the kernel.
  if (v.shape(3) != q.shape(3))
    bad("v head_dim (D_v=" + std::to_string(v.shape(3)) + ") must equal q/k "
        "head_dim (D_qk=" + std::to_string(q.shape(3)) + ") — this raw kernel "
        "allocates output from q.shape and cannot produce a D_v-shaped output. "
        "Use the public wrapper for asymmetric D_v (SDPA fallback).");
  const int Hq = q.shape(1), Hk = k.shape(1);
  if (Hk <= 0 || Hq % Hk != 0)
    bad("q_heads (" + std::to_string(Hq) + ") must be a positive multiple of "
        "kv_heads (" + std::to_string(Hk) + ") for GQA");
  if (q.dtype() != k.dtype() || q.dtype() != v.dtype())
    bad("q, k, v must share dtype");
  if (q.dtype() != mlx::core::float16 && q.dtype() != mlx::core::bfloat16)
    bad("only float16/bfloat16 are supported");
}

// =========================================================================
// CC Batch — shared raw-entry validators (paged family + TQ buffers)
// =========================================================================
// The dense feature family routes through validate_dense_qkv (above). The PAGED
// family has its own input structure (q + pooled K/V, TQ adds packed uint8 K +
// fp16 centroids + fp32 scales), so it gets these companion validators — one
// enforcement point per contract, reused across the paged siblings (no per-site
// ad-hoc drift), mirroring the runtime-matrix gate + the Python _persist_validate.

// Scalar value-semantics (raw-host parity sweep — the scale-finite class swept to
// the SOURCE): the public flash_attention rejects a non-finite scale; every raw
// forward host bakes `scale` into the kernel, so nan/inf -> silent all-NaN output.
// One shared helper, called from every raw forward host that takes a float scale.
static inline void assert_scale_finite(float scale, const char* entry) {
  if (!std::isfinite(scale))
    throw std::invalid_argument(std::string(entry) + ": scale must be finite.");
}

// Per-segment zero-KV (raw-host parity sweep — the varlen analogue of the dense
// zero-KV class): a QUERIED segment (q_len>0) with k_len==0 is a 0/0 online-softmax
// -> silent NaN for that segment's rows. The public varlen/paged wrappers raise on
// this (_validate_cu_seqlens); the raw hosts did not. Cheap B-element scalar loop
// (same cost class as the cu-bounds check) -> UNGATED. Empty segments with q_len==0
// stay legal (padding). Variant for non-paged varlen (both prefix sums):
static inline void assert_varlen_segment_kv(const mlx::core::array& cu_q,
                                            const mlx::core::array& cu_k,
                                            const char* entry) {
  if (cu_q.size() < 2 || cu_k.size() < 2) return;
  mlx::core::array cq = mlx::core::contiguous(cu_q);
  mlx::core::array ck = mlx::core::contiguous(cu_k);
  mlx::core::eval(cq); mlx::core::eval(ck);
  const int32_t* pq = cq.data<int32_t>();
  const int32_t* pk = ck.data<int32_t>();
  const int nseg = static_cast<int>(cq.shape(0)) - 1;
  for (int j = 0; j < nseg; ++j) {
    if (pq[j + 1] - pq[j] > 0 && pk[j + 1] - pk[j] == 0)
      throw std::invalid_argument(
          std::string(entry) + ": segment " + std::to_string(j) +
          " is queried (q_len>0) but has k_len==0 (empty KV) -> 0/0 softmax = NaN; "
          "drop the empty segment or give it at least one key.");
  }
}

// Variant for paged varlen/tq (cu_q prefix sum + seq_lens_kv per-segment counts):
static inline void assert_paged_varlen_segment_kv(const mlx::core::array& cu_q,
                                                  const mlx::core::array& seq_lens_kv,
                                                  const char* entry) {
  if (cu_q.size() < 2 || seq_lens_kv.size() == 0) return;
  mlx::core::array cq = mlx::core::contiguous(cu_q);
  mlx::core::array sk = mlx::core::contiguous(seq_lens_kv);
  mlx::core::eval(cq); mlx::core::eval(sk);
  const int32_t* pq = cq.data<int32_t>();
  const int32_t* pk = sk.data<int32_t>();
  const int nseg = static_cast<int>(cq.shape(0)) - 1;
  const int nkv  = static_cast<int>(sk.shape(0));
  for (int j = 0; j < nseg && j < nkv; ++j) {
    if (pq[j + 1] - pq[j] > 0 && pk[j] == 0)
      throw std::invalid_argument(
          std::string(entry) + ": segment " + std::to_string(j) +
          " is queried (q_len>0) but seq_lens_kv==0 (empty KV) -> 0/0 softmax = NaN; "
          "drop the empty segment or give it at least one key.");
  }
}

// Class A — paged forwards that dispatch a float16/bfloat16 kernel via a 2-way
// `dtype_code=0; if(bf16)=1` map (no fp32 branch): fp32 q falls to the half
// kernel = FINITE GARBAGE (cos≈0, max_abs≈1e37), masked by the requested-dtype
// output alloc. Reject fp32 at the boundary. Public wrappers keep fp32 via their
// per-sequence SDPA fallback bridge, so this only closes the raw `_ext` hole.
static inline void assert_raw_fp16_bf16_only(
    const mlx::core::array& q, const char* entry) {
  const auto dt = q.dtype();
  if (dt == mlx::core::float16 || dt == mlx::core::bfloat16) return;
  throw std::invalid_argument(
      std::string(entry) +
      ": only float16/bfloat16 are supported "
      "— this raw forward dispatches a float16/bfloat16 kernel "
      "and a float32 input would route to a mismatched (half) kernel = silent-wrong "
      "output. Use float16/bfloat16; for float32 use the public wrapper "
      "(flash_attention_paged_varlen / *_turboquant), which provides an fp32 "
      "fallback bridge.");
}

// Class D — paged pools are read at q's dtype (dtype_code derived from q only),
// so a k_pool/v_pool whose dtype != q is reinterpreted byte-wise = finite
// silent-wrong (probe cos as low as 0.325). Require the value pools to match q.
static inline void assert_raw_pool_dtype_eq(
    const mlx::core::array& q, const mlx::core::array& k_pool,
    const mlx::core::array& v_pool, const char* entry) {
  if (k_pool.dtype() != q.dtype() || v_pool.dtype() != q.dtype())
    throw std::invalid_argument(
        std::string(entry) + ": k_pool/v_pool dtype must equal q dtype "
        "(the kernel derives its dtype from q and reads the pools at that dtype; "
        "a mismatched pool dtype is read byte-wise = silent-wrong output).");
}

// CC matrix batch — paged Q-vs-pool GQA. q_heads must be a positive multiple of
// the pool kv-heads; a bad ratio reads the pool at wrong head offsets =
// finite-wrong. Shared by every paged raw entry (steel / varlen / TQ) so they
// can't diverge. (Pool kv-heads = k_pool.shape(2) for [num_blocks,bs,H_kv,*].)
static inline void assert_raw_paged_gqa(
    const mlx::core::array& q, int pool_hkv, const char* entry) {
  const int Hq = q.shape(1);
  if (pool_hkv <= 0 || Hq % pool_hkv != 0)
    throw std::invalid_argument(
        std::string(entry) + ": q_heads (" + std::to_string(Hq) + ") must be a "
        "positive multiple of the pool kv-heads (" + std::to_string(pool_hkv) +
        ") for GQA.");
}

// Class D — TurboQuant backing-buffer required dtypes. The fused TQ kernel reads
// packed K as uint8, the centroid codebook as fp16, and the scales as fp32; a
// mismatched buffer dtype is reinterpreted = finite silent-wrong.
//
// Optional-input contract dimension (CC optional-input batch): the K-side buffers
// are mandatory; the V-side buffers (v_pool_tq / v_centroids / v_scales) are
// CONDITIONAL on tq_v_enabled. The earlier validator enforced only the mandatory
// (K) side — so a malformed V buffer dtype was silently accepted (finite-wrong).
// Enforce the V-side with the SAME contract, but ONLY when tq_v_enabled AND the
// optional buffer is present (no-op on absence — the conditional-presence crux).
static inline void assert_raw_tq_buffer_dtypes(
    const mlx::core::array& q, const mlx::core::array& k_pool_tq,
    const mlx::core::array& v_pool, const mlx::core::array& centroids,
    const mlx::core::array& k_scales, const char* entry,
    bool tq_v_enabled = false,
    const std::optional<mlx::core::array>& v_pool_tq = std::nullopt,
    const std::optional<mlx::core::array>& v_centroids = std::nullopt,
    const std::optional<mlx::core::array>& v_scales = std::nullopt) {
  if (k_pool_tq.dtype() != mlx::core::uint8)
    throw std::invalid_argument(std::string(entry) +
        ": k_pool_tq must be uint8 (packed TurboQuant K).");
  if (v_pool.dtype() != q.dtype())
    throw std::invalid_argument(std::string(entry) +
        ": v_pool dtype must equal q dtype (read at q's dtype).");
  if (centroids.dtype() != mlx::core::float16)
    throw std::invalid_argument(std::string(entry) +
        ": centroids must be float16 (the codebook is read as fp16).");
  if (k_scales.dtype() != mlx::core::float32)
    throw std::invalid_argument(std::string(entry) +
        ": k_scales must be float32 (read as fp32).");
  // V-side: same contract, only-when-present (tq_v_enabled + supplied buffer).
  if (tq_v_enabled) {
    if (v_pool_tq.has_value() && v_pool_tq->dtype() != mlx::core::uint8)
      throw std::invalid_argument(std::string(entry) +
          ": v_pool_tq must be uint8 (packed TurboQuant V) when tq_v_enabled.");
    if (v_centroids.has_value() && v_centroids->dtype() != mlx::core::float16)
      throw std::invalid_argument(std::string(entry) +
          ": v_centroids must be float16 (the V codebook is read as fp16) when tq_v_enabled.");
    if (v_scales.has_value() && v_scales->dtype() != mlx::core::float32)
      throw std::invalid_argument(std::string(entry) +
          ": v_scales must be float32 (read as fp32) when tq_v_enabled.");
  }
}

// CC feature-tensor dtype-misread class — several kernels read a host-passed
// SECONDARY/feature tensor (alibi_slopes, rope rotary_cos/rotary_sin) at a
// hardcoded fp32 in the shader. A non-fp32 input was byte-reinterpreted = finite
// silent-wrong (alibi f16 cos 0.87, rope f16 cos 0.72 vs the f32 reference).
// Remedy = upcast to the kernel's contract dtype (fp32) in the host: f16/bf16
// callers become CORRECT, f32 callers are a no-op (byteΔ=0), nothing is rejected.
static inline mlx::core::array upcast_to_f32(
    const mlx::core::array& a, mlx::core::Stream s) {
  return (a.dtype() == mlx::core::float32)
             ? a
             : mlx::core::astype(a, mlx::core::float32, s);
}

// Same class, contract dtype = uint8: the sparse block_mask is read byte-wise as
// uint8 tile-activity flags. A wider dtype (int32/float32) was misread = NaN/
// garbage tiles (cos=nan); bool happens to work (1-byte). Cast 0/1 masks to the
// uint8 contract in the host (lossless for 0/1; uint8 is a no-op → byteΔ=0).
static inline mlx::core::array cast_mask_to_u8(
    const mlx::core::array& a, mlx::core::Stream s) {
  return (a.dtype() == mlx::core::uint8)
             ? a
             : mlx::core::astype(a, mlx::core::uint8, s);
}

// CC feature-metadata SHAPE class — the sparse kernel reads block_mask at tile
// granularity [.., NQ, NK] with NQ=ceil(N/BQ), NK=ceil(S/BK) for the V1 sparse
// block config; an undersized mask drives an OOB tile read (non-finite) and a
// wrong head/batch dim is a finite-wrong read. The raw entry validated only the
// rank (2..4); the PUBLIC flash_attention_sparse rejects these correctly, so the
// raw entry adopts the kernel's real extent contract. (Validate-and-raise: a
// malformed shape can't be cast into a valid one.)
static void validate_sparse_block_mask_shape(
    const mlx::core::array& block_mask, const mlx::core::array& q,
    const mlx::core::array& k, mlx::core::Stream s, const char* fn) {
  const int B = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  const int S = k.shape(2);
  auto& dev = mlx::core::metal::device(s.device);
  int arch = static_cast<int>(dev.get_architecture_gen());
  if (MFAEnvConfig::get().force_gen > 0) arch = MFAEnvConfig::get().force_gen;
  const bool is_m3p = (arch >= 15);
  const auto cfg = select_steel_block_config(D, /*is_low_prec=*/true, is_m3p);
  const int BQ = cfg.BQ, BK = cfg.BK;
  const int NQ = (N + BQ - 1) / BQ, NK = (S + BK - 1) / BK;
  const int nd = block_mask.ndim();
  auto bad = [&](const std::string& m) {
    throw std::invalid_argument(std::string(fn) + ": " + m);
  };
  if (block_mask.shape(nd - 2) != NQ || block_mask.shape(nd - 1) != NK)
    bad("block_mask tile dims (" + std::to_string(block_mask.shape(nd - 2)) + "," +
        std::to_string(block_mask.shape(nd - 1)) + ") must be (NQ,NK)=(" +
        std::to_string(NQ) + "," + std::to_string(NK) + ") = (ceil(N/" +
        std::to_string(BQ) + "), ceil(S/" + std::to_string(BK) +
        ")); an undersized mask drives an out-of-bounds tile read.");
  if (nd >= 3 && block_mask.shape(nd - 3) != H)
    bad("block_mask head dim (" + std::to_string(block_mask.shape(nd - 3)) +
        ") must equal q heads (" + std::to_string(H) + ").");
  if (nd == 4 && block_mask.shape(0) != B)
    bad("block_mask batch dim (" + std::to_string(block_mask.shape(0)) +
        ") must equal q batch (" + std::to_string(B) + ").");
}

// CC metadata-VALUE class — varlen cu_seqlens/tile_offsets drive per-sequence
// memory access; dtype/rank/cardinality are checked, but bad VALUES (non-monotone,
// negative, [0]!=0, [-1]!=total) silently produce finite-wrong output (byteΔ up to
// ~2.4 vs valid). Reading values forces a host sync, so — mirroring the paged
// MFA_PAGED_TRUST_INDICES precedent — validate by DEFAULT (protect naive callers)
// with an opt-out `MFA_VARLEN_TRUST_METADATA=1` for callers who guarantee valid
// metadata and want the sync gone.
static bool varlen_trust_metadata() {
  const char* v = std::getenv("MFA_VARLEN_TRUST_METADATA");
  return v != nullptr && std::string(v) == "1";
}
static void check_prefix_sum_values(
    const mlx::core::array& a, int expected_last, const char* nm,
    const char* entry, int upper_bound = -1) {
  // `a` is already validated int32 + 1-D. Reads values (host sync).
  // expected_last>=0 → final value must EQUAL it (cu_q: the packed-q count must
  // be fully covered, else OOB). upper_bound>=0 → final value must be <= it
  // (cu_k: an over-allocated K tensor / empty trailing KV segment is legal —
  // cu_k[-1] < total_k is fine — but cu_k[-1] > total_k would OOB-read).
  mlx::core::eval(a);
  const int32_t* p = a.data<int32_t>();
  const int n = static_cast<int>(a.shape(0));
  auto bad = [&](const std::string& m) {
    throw std::invalid_argument(std::string(entry) + ": " + nm + " " + m);
  };
  if (p[0] != 0) bad("must start at 0 (got " + std::to_string(p[0]) + ").");
  for (int i = 1; i < n; ++i)
    if (p[i] < p[i - 1])
      bad("must be non-decreasing (offset " + std::to_string(i) + ": " +
          std::to_string(p[i]) + " < " + std::to_string(p[i - 1]) + ").");
  if (expected_last >= 0 && p[n - 1] != expected_last)
    bad("final value (" + std::to_string(p[n - 1]) + ") must equal the total "
        "packed token count (" + std::to_string(expected_last) + ").");
  if (upper_bound >= 0 && p[n - 1] > upper_bound)
    bad("final value (" + std::to_string(p[n - 1]) + ") must not exceed the "
        "packed K-token count (" + std::to_string(upper_bound) + ").");
}

// TRUST-FLAG COST-CLASS split (Phase 2 grid): MFA_VARLEN_TRUST_METADATA gated the
// ENTIRE check_prefix_sum_values — including the CHEAP structural scalar contracts
// (cu[0]==0 + the final-value bound). For non-paged varlen, cu_k[-1] <= total_k is the
// SOLE K-OOB guard (no in-kernel backstop), so gating it = a trust-mode OOB (non-det,
// verified). Same mis-scope as the paged capacity seam. This helper does ONLY the cheap
// scalar bounds (eval + [0]==0 + final-value), to run UNGATED; the per-element monotone
// scan + the derived tile_offsets recompute stay in the flag-scoped check_prefix_sum_values
// / validate_tile_offsets_derivation (the "trust the index structure" opt-out).
static void check_prefix_sum_bounds(
    const mlx::core::array& a, int expected_last, const char* nm,
    const char* entry, int upper_bound = -1) {
  if (a.size() == 0) return;
  mlx::core::array av = a; mlx::core::eval(av);
  const int32_t* p = av.data<int32_t>();
  const int n = static_cast<int>(av.shape(0));
  auto bad = [&](const std::string& m) {
    throw std::invalid_argument(std::string(entry) + ": " + nm + " " + m);
  };
  if (p[0] != 0) bad("must start at 0 (got " + std::to_string(p[0]) + ").");
  if (expected_last >= 0 && p[n - 1] != expected_last)
    bad("final value (" + std::to_string(p[n - 1]) + ") must equal the total packed "
        "token count (" + std::to_string(expected_last) + ").");
  if (upper_bound >= 0 && p[n - 1] > upper_bound)
    bad("final value (" + std::to_string(p[n - 1]) + ") must not exceed the packed "
        "K-token count (" + std::to_string(upper_bound) + ") — out-of-bounds K read.");
}

// CAPACITY-SEAM class-closure (Codex NO-GO): the logical capacity contract
// (seq_lens <= max_blocks*block_size, seq_lens >= 0) was present only on the public
// wrappers (and there gated under MFA_PAGED_TRUST_INDICES) and ABSENT on every raw
// paged forward — so the raw entries silently DILUTED an over-capacity seq_lens over
// -1-padding/zero blocks (finite-deterministic-WRONG; the physical-block guard keeps
// it memory-safe, not correct). This is the CHEAP structural half of the cost-class
// split (a B-element host reduction, NOT the expensive per-element block_table index
// scan the trust flag legitimately skips). Shared helper so the raw + public capacity
// contract can't diverge again. seq_lens must already be int32 (callers enforce).
static void assert_paged_capacity(const mlx::core::array& seq_lens, int max_blocks,
                                  int block_size, const char* entry,
                                  bool reject_zero = false) {
  if (seq_lens.size() == 0 || max_blocks <= 0 || block_size <= 0) return;
  // stride-safe (raw-host parity sweep): a non-contiguous seq_lens read via
  // data<int32_t>() would scan the wrong elements -> contiguize first (no-op when
  // already contiguous).
  mlx::core::array sl = mlx::core::contiguous(seq_lens);
  mlx::core::eval(sl);
  const int32_t* p = sl.data<int32_t>();
  const int n = static_cast<int>(sl.shape(0));
  int32_t smin = p[0], smax = p[0];
  for (int i = 1; i < n; ++i) { if (p[i] < smin) smin = p[i]; if (p[i] > smax) smax = p[i]; }
  if (smin < 0)
    throw std::invalid_argument(std::string(entry) + ": seq_lens must be >= 0; got min="
                                + std::to_string(smin) + ".");
  // zero-KV reject (forwards only — every batch row is queried in the dense-paged
  // steel path, so a 0-length row is a queried zero-KV -> non-deterministic NaN).
  // The gather passes reject_zero=false (gathering 0 tokens for a row is legal).
  if (reject_zero && smin == 0)
    throw std::invalid_argument(
        std::string(entry) + ": seq_lens has a zero-length (kv_len==0) queried entry "
        "-> the empty-softmax path produces non-deterministic NaN; pass kv_len>=1.");
  const long long cap = static_cast<long long>(max_blocks) * static_cast<long long>(block_size);
  if (static_cast<long long>(smax) > cap)
    throw std::invalid_argument(
        std::string(entry) + ": seq_lens max (" + std::to_string(smax) +
        ") exceeds max_blocks*block_size = " + std::to_string(max_blocks) + "*" +
        std::to_string(block_size) + " = " + std::to_string(cap) +
        " (a logical block index would run past the block_table columns).");
}

// CC derived-metadata class — `tile_offsets` is NOT primary: it is fully DERIVED
// from cu_seqlens_q + the tile width BQ (tile_offsets[j+1] = tile_offsets[j] +
// ceil(q_len[j]/BQ)). The generic prefix-sum check (monotone, [0]==0) accepted
// monotone-but-wrong values → finite-wrong (cos 0.64–0.88). Validate against the
// canonical RECOMPUTATION instead — a TOTAL contract: only the one correct value
// passes, every deviation (wrong-final, wrong-segment, off-by-one) raises by
// construction. BQ is derived the SAME way the kernel does
// (select_steel_block_config(D, low_prec, is_m3_plus).BQ — device-aware, not
// hardcoded), so the validator can never disagree with the dispatch.
static void validate_tile_offsets_derivation(
    const mlx::core::array& cu_q, const mlx::core::array& tile_offsets,
    const mlx::core::array& q, mlx::core::Stream s, const char* entry) {
  const int D = q.shape(3);
  auto& dev = mlx::core::metal::device(s.device);
  int arch = static_cast<int>(dev.get_architecture_gen());
  if (MFAEnvConfig::get().force_gen > 0) arch = MFAEnvConfig::get().force_gen;
  const bool is_m3p = (arch >= 15);
  const int BQ = select_steel_block_config(
      D, /*is_low_prec=*/q.dtype() != mlx::core::float32, is_m3p).BQ;
  mlx::core::eval(cu_q);
  mlx::core::eval(tile_offsets);
  const int32_t* cq = cu_q.data<int32_t>();
  const int32_t* to = tile_offsets.data<int32_t>();
  const int nseq1 = static_cast<int>(cu_q.shape(0));  // num_seqs + 1
  if (to[0] != 0)
    throw std::invalid_argument(std::string(entry) +
        ": tile_offsets[0] must be 0 (got " + std::to_string(to[0]) + ").");
  int acc = 0;
  for (int j = 0; j < nseq1 - 1; ++j) {
    const int qlen = cq[j + 1] - cq[j];
    acc += (qlen + BQ - 1) / BQ;
    if (to[j + 1] != acc)
      throw std::invalid_argument(std::string(entry) +
          ": tile_offsets[" + std::to_string(j + 1) + "] (" +
          std::to_string(to[j + 1]) + ") != the canonical cumulative "
          "ceil(q_len/BQ) (" + std::to_string(acc) + "); tile_offsets is fully "
          "derived from cu_seqlens_q and BQ=" + std::to_string(BQ) +
          " — pass the canonical value (or set MFA_VARLEN_TRUST_METADATA=1).");
  }
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
    int  window_left,
    int  window_right,
    std::optional<mlx::core::StreamOrDevice> stream) {
  auto s = stream.has_value()
      ? mlx::core::to_stream(stream.value())
      : mlx::core::default_stream(mlx::core::Device::gpu);

  // volet K1: complete dense-QKV contract (was missing GQA / q↔k head_dim /
  // dtype; C-01 added batch + K↔V only). Note R1 supports asymmetric D_v, which
  // the shared validator preserves (it checks q.D==k.D, not v.D).
  validate_dense_qkv(q, k, v, "MFA");
  assert_scale_finite(scale, "MFA");

  // D.5: Enforce row-major BHND layout inside the C++ binding entry point.
  // mlx::core::contiguous() is a no-op (zero allocation) when the array is
  // already contiguous — it returns the same buffer with no copy.  Moving
  // this here eliminates 3 Python->C++ round-trips per forward dispatch.
  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);

  int D = qc.shape(3);
  if (D != 64 && D != 128 && D != 256) {
    throw std::invalid_argument(
        "MFA: head_dim must be 64, 128, or 256, got " + std::to_string(D));
  }

  MFAttention::Params params{D, scale, causal,
      /*has_block_mask=*/false, /*has_rope=*/false,
      /*rope_interleaved=*/false, /*cache_seqlens=*/0, /*softcap=*/softcap,
      /*has_alibi=*/false, /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
      /*window_left=*/window_left, /*window_right=*/window_right};

  auto out_shape  = qc.shape();                     // Shape [B, H, N, D]
  mlx::core::Shape lse_shape = {
      qc.shape(0), qc.shape(1), qc.shape(2)};       // Shape [B, H, N]

  // O dtype matches input dtype (kernel accumulates FP32 then writes input prec).
  // L (logsumexp for backward) is always FP32.
  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {qc.dtype(), mlx::core::float32},
      std::make_shared<MFAttention>(s, params),
      {qc, kc, vc});

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

  validate_dense_qkv(q, k, v, "MFA sparse");  // volet K1: full Q/K/V contract
  assert_scale_finite(scale, "MFA sparse");
  if (block_mask.ndim() < 2 || block_mask.ndim() > 4) {
    throw std::invalid_argument(
        "MFA sparse: block_mask must be 2D [NQ,NK], 3D [H,NQ,NK], "
        "or 4D [B,H,NQ,NK]");
  }

  int D = q.shape(3);
  if (D != 64 && D != 128 && D != 256) {
    throw std::invalid_argument(
        "MFA sparse: head_dim must be 64, 128, or 256, got " +
        std::to_string(D));
  }
  // SPARSE-D128-OOB (sweep iter-1): the STEEL V1 block-sparse forward kernel is
  // NOT correct at head_dim=128 on M5+ — it leaves the last head's upper-half query
  // tiles unwritten, returning OOB/NaN (non-deterministic, scales with grid
  // occupancy; VERIFIED on M5). On M5+ the public flash_attention_sparse routes
  // D=128 sparse to SDPA (`_sparse_fallback_sdpa_perhead`, the documented dispatch-
  // map gotcha), so this raw kernel is never reached publicly there — gating the
  // refusal to gen>=17 closes the verified silent-wrong on the expert surface with
  // ZERO risk to the M1–M4 public path (which DOES call this kernel and which the
  // III-4 D4 fix targeted — its D=128 correctness is FLAGGED for M1–M4 hardware,
  // not assumed). Refuse loudly (RULE 8) rather than silently corrupt.
  {
    int gen_g = mlx::core::metal::device(s.device).get_architecture_gen();
    // Respect MFA_FORCE_GEN (mirrors arch_gen_ccv/steel at L144/199): so the
    // force-arch cross-test can swap the arch config and run it on M5.
    const auto& eg = MFAEnvConfig::get();
    if (eg.force_gen > 0) gen_g = eg.force_gen;
    // SPARSE-D128-OOB: the STEEL V1 block-sparse forward is OOB at head_dim=128 on
    // the is_m3_plus path (gen>=15: M3/M4/M5 — last head, upper-half query tiles).
    // FORCE-ARCH VERIFIED on M5: gen=15/16/17 all non-deterministic nan_frac~0.0625;
    // gen<15 (M1/M2, BK=16 base config) is CLEAN. The M5-only gate was too narrow.
    // Public flash_attention_sparse routes D=128 to SDPA on gen>=15; raw refuses.
    if (D == 128 && gen_g >= 15) {
      throw std::invalid_argument(
          "MFA sparse: the raw STEEL V1 block-sparse forward kernel is not correct "
          "at head_dim=128 on M3+ (gen>=15: out-of-bounds on the last head). Use "
          "flash_attention_sparse, which routes D=128 sparse to SDPA on M3+. "
          "D=64 sparse is supported; D=128 sparse is correct only on M1/M2.");
    }
  }

  // Require f16/bf16 (sparse path is STEEL-only; f32 would need ccv update)
  if (q.dtype() == mlx::core::float32) {
    throw std::invalid_argument(
        "MFA sparse: float32 is not supported; use float16 or bfloat16");
  }

  // CC shape class: block_mask tile-extent/head/batch contract (raw lacked it;
  // public flash_attention_sparse already rejects these).
  validate_sparse_block_mask_shape(block_mask, q, k, s, "MFA sparse");

  MFAttention::Params params{D, scale, causal, /*has_block_mask=*/true,
      /*has_rope=*/false, /*rope_interleaved=*/false, /*cache_seqlens=*/0,
      /*softcap=*/0.0f, /*has_alibi=*/false, /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
      /*window_left=*/-1, /*window_right=*/-1};

  auto out_shape  = q.shape();
  mlx::core::Shape lse_shape = {q.shape(0), q.shape(1), q.shape(2)};

  // CONTIG (sweep iter-2): enforce contiguity (D.5 pattern; no-op/zero-copy when
  // already contiguous) so a strided q/k/v VIEW isn't read with contiguous-assumed
  // strides → finite-WRONG. Covers the raw _ext entry (the public path also
  // contiguous-ifies in Python). VERIFIED finite-wrong pre-fix at D=64 (maxabs 0.95).
  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);
  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAttention>(s, params),
      {qc, kc, vc, cast_mask_to_u8(block_mask, s)});  // CC: kernel reads mask as uint8

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

  validate_dense_qkv(q, k, v, "MFA sparse");  // volet K1: full Q/K/V contract
  assert_scale_finite(scale, "MFA sparse");
  if (block_mask.ndim() < 2 || block_mask.ndim() > 4)
    throw std::invalid_argument(
        "MFA sparse: block_mask must be 2D [NQ,NK], 3D [H,NQ,NK], "
        "or 4D [B,H,NQ,NK]");

  int D = q.shape(3);
  if (D != 64 && D != 128 && D != 256)
    throw std::invalid_argument(
        "MFA sparse: head_dim must be 64, 128, or 256, got " +
        std::to_string(D));
  // SPARSE-D128-OOB (sweep iter-1): mirror the non-LSE sibling — the STEEL V1
  // block-sparse forward is OOB at head_dim=128 on M5+; refuse loudly (M5+ public
  // path uses SDPA at D=128). Sibling-bypass discipline: both variants enforce it.
  {
    int gen_g = mlx::core::metal::device(s.device).get_architecture_gen();
    const auto& eg = MFAEnvConfig::get();
    if (eg.force_gen > 0) gen_g = eg.force_gen;  // force-arch cross-test hook (L144/199 pattern)
    // SPARSE-D128-OOB (gen>=15, force-arch verified M3/M4/M5 broken; sibling of the
    // non-LSE site). Mirror it: refuse on M3+; M1/M2 (BK=16 base config) is clean.
    if (D == 128 && gen_g >= 15)
      throw std::invalid_argument(
          "MFA sparse: the raw STEEL V1 block-sparse forward kernel is not correct "
          "at head_dim=128 on M3+ (gen>=15: out-of-bounds on the last head). Use "
          "flash_attention_sparse, which routes D=128 sparse to SDPA on M3+. "
          "D=64 sparse is supported; D=128 sparse is correct only on M1/M2.");
  }

  if (q.dtype() == mlx::core::float32)
    throw std::invalid_argument(
        "MFA sparse: float32 is not supported; use float16 or bfloat16");

  // CC sibling-bypass fix: the LSE variant must run the SAME block_mask shape
  // validator as its non-LSE sibling (it had qkv + mask-dtype but skipped the
  // shape check → (1,1) non-finite, (H+1,..) finite-wrong).
  validate_sparse_block_mask_shape(block_mask, q, k, s, "MFA sparse");

  MFAttention::Params params{D, scale, causal, /*has_block_mask=*/true,
      /*has_rope=*/false, /*rope_interleaved=*/false, /*cache_seqlens=*/0,
      /*softcap=*/0.0f, /*has_alibi=*/false, /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
      /*window_left=*/-1, /*window_right=*/-1};
  auto out_shape = q.shape();
  mlx::core::Shape lse_shape = {q.shape(0), q.shape(1), q.shape(2)};

  // CONTIG (sweep iter-2): mirror the non-LSE sibling — enforce contiguity (D.5
  // pattern, no-op when contiguous) before dispatch so strided q/k/v isn't misread.
  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);
  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAttention>(s, params),
      {qc, kc, vc, cast_mask_to_u8(block_mask, s)});  // CC: kernel reads mask as uint8

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
    bool interleaved,
    std::optional<mlx::core::StreamOrDevice> stream) {
  auto s = stream.has_value()
      ? mlx::core::to_stream(stream.value())
      : mlx::core::default_stream(mlx::core::Device::gpu);

  validate_dense_qkv(q, k, v, "MFA rope");  // volet K1: full Q/K/V contract
  // SCALE value-semantics (Phase 2 pass-3 — wrapper-vs-raw: the Python
  // flash_attention_rope wrapper got the finite-guard in pass-1, the raw _ext host
  // did not). A nan/inf scale was silently clamped to NaN output. Push to source
  // (mirrors mfa_v6_nax_primitive's std::isfinite(scale) host guard).
  if (!std::isfinite(scale))
    throw std::invalid_argument("mfa_attention_rope_forward: scale must be finite.");

  int D = q.shape(3);
  if (D != 64 && D != 128 && D != 256) {
    throw std::invalid_argument(
        "MFA rope: head_dim must be 64, 128, or 256, got " +
        std::to_string(D));
  }
  // RoPE residual: cos/sin mutual shape + rotary width D/2 (volet K1).
  // CC CORRECTION: the K1 note "cos/sin dtype is flexible, kernel accepts both"
  // was empirically FALSE — the shader reads cos/sin at a hardcoded fp32, so a
  // float16/bfloat16 table was byte-misread = silent-wrong (f16 cos 0.72 vs f32).
  // CONTRACT: any dtype is accepted, but it is UPCAST to fp32 in the host (below,
  // via upcast_to_f32) — f16/bf16 tables now produce correct output, f32 is a
  // no-op (byteΔ=0). cos/sin dtype is therefore unconstrained AND correct.
  if (rotary_cos.ndim() != rotary_sin.ndim())
    throw std::invalid_argument("MFA rope: cos/sin rank mismatch");
  for (int d = 0; d < rotary_cos.ndim(); ++d)
    if (rotary_cos.shape(d) != rotary_sin.shape(d))
      throw std::invalid_argument("MFA rope: cos/sin shape mismatch");
  if (rotary_cos.shape(rotary_cos.ndim() - 1) != D / 2)
    throw std::invalid_argument(
        "MFA rope: rotary width must be D/2 (" + std::to_string(D / 2) + ")");
  // CC feature-metadata SHAPE class: the kernel indexes cos/sin by position
  // (row = cache_seqlens + query_index for Q, 0..S-1 for K), so it needs a rank-2
  // [rows, D/2] table with enough rows. rank-1 was read as a single row
  // (silent-wrong, cos 0.68); a too-short table indexes OOB rows (cos 0.87).
  if (rotary_cos.ndim() < 2)
    throw std::invalid_argument(
        "MFA rope: cos/sin must be rank-2 [rows, D/2] (a rank-1 table is read as "
        "a single position = silent-wrong).");
  {
    const int rope_rows = rotary_cos.shape(rotary_cos.ndim() - 2);
    const int required = std::max(k.shape(2), cache_seqlens + q.shape(2));
    if (rope_rows < required)
      throw std::invalid_argument(
          "MFA rope: cos/sin rows (" + std::to_string(rope_rows) + ") must be >= "
          "max(S, cache_seqlens + N) = " + std::to_string(required) +
          " (the kernel indexes row = cache_seqlens + query_index); a shorter "
          "table drives an out-of-bounds row read = silent-wrong.");
  }

  MFAttention::Params params{
    D, scale, causal,
    /*has_block_mask=*/false,
    /*has_rope=*/true,
    /*rope_interleaved=*/interleaved,
    /*cache_seqlens=*/cache_seqlens,
    /*softcap=*/0.0f,
    /*has_alibi=*/false,
    /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
    /*window_left=*/-1,
    /*window_right=*/-1
  };

  auto out_shape  = q.shape();
  mlx::core::Shape lse_shape = {q.shape(0), q.shape(1), q.shape(2)};

  // inputs: [Q, K, V, rotary_cos, rotary_sin]
  // buffers in Metal: Q=0, K=1, V=2, O=3, L=4, params=5, cos=7, sin=8
  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAttention>(s, params),
      // CC: kernel reads cos/sin as fp32 — upcast so f16/bf16 tables are correct.
      {q, k, v, upcast_to_f32(rotary_cos, s), upcast_to_f32(rotary_sin, s)});

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

  validate_dense_qkv(q, k, v, "MFA alibi");  // volet K1: full Q/K/V contract
  assert_scale_finite(scale, "MFA alibi");

  if (alibi_slopes.ndim() != 1 || alibi_slopes.shape(0) != q.shape(1))
    throw std::invalid_argument(
        "MFA alibi: alibi_slopes must be 1D [Hq=" +
        std::to_string(q.shape(1)) + "]");  // volet K1: length Hq (was rank-only)

  // D.5: enforce row-major layout at the C++ binding entry point.
  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);

  int D = qc.shape(3);
  if (D != 64 && D != 128 && D != 256)
    throw std::invalid_argument(
        "MFA alibi: head_dim must be 64, 128, or 256, got " +
        std::to_string(D));

  MFAttention::Params params{
    D, scale, causal,
    /*has_block_mask=*/false,
    /*has_rope=*/false,
    /*rope_interleaved=*/false,
    /*cache_seqlens=*/0,
    /*softcap=*/0.0f,
    /*has_alibi=*/true,
    /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
    /*window_left=*/-1,
    /*window_right=*/-1
  };

  auto out_shape  = qc.shape();
  mlx::core::Shape lse_shape = {qc.shape(0), qc.shape(1), qc.shape(2)};

  // inputs: [Q, K, V, alibi_slopes]
  // Metal buffers: Q=0, K=1, V=2, O=3, L=4, params=5, alibi_slopes=9
  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {qc.dtype(), mlx::core::float32},
      std::make_shared<MFAttention>(s, params),
      {qc, kc, vc, upcast_to_f32(alibi_slopes, s)});  // CC: kernel reads slopes as fp32

  return outputs[0];
}

// =========================================================================
// Free function: mfa_attention_bias_forward
// =========================================================================

mlx::core::array mfa_attention_bias_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& attn_bias,
    uint8_t attn_bias_mode,
    float scale,
    bool causal,
    std::optional<mlx::core::StreamOrDevice> stream) {
  auto s = stream.has_value()
      ? mlx::core::to_stream(stream.value())
      : mlx::core::default_stream(mlx::core::Device::gpu);

  validate_dense_qkv(q, k, v, "MFA bias");  // volet K1: full Q/K/V contract
  assert_scale_finite(scale, "MFA bias");

  if (attn_bias_mode < 1 || attn_bias_mode > 2)
    throw std::invalid_argument(
        "MFA bias: only modes 1 ([1,1,1,Nkv]) and 2 ([1,H,1,Nkv]) supported, got " +
        std::to_string(attn_bias_mode));

  // Validate bias shape against mode
  int S = k.shape(2);  // N_kv
  int H = q.shape(1);  // H_q
  if (attn_bias_mode == 1) {
    if (attn_bias.ndim() != 4 ||
        attn_bias.shape(0) != 1 || attn_bias.shape(1) != 1 ||
        attn_bias.shape(2) != 1 || attn_bias.shape(3) != S)
      throw std::invalid_argument(
          "MFA bias mode 1: expected shape [1,1,1," + std::to_string(S) + "]");
  } else if (attn_bias_mode == 2) {
    if (attn_bias.ndim() != 4 ||
        attn_bias.shape(0) != 1 || attn_bias.shape(1) != H ||
        attn_bias.shape(2) != 1 || attn_bias.shape(3) != S)
      throw std::invalid_argument(
          "MFA bias mode 2: expected shape [1," + std::to_string(H) +
          ",1," + std::to_string(S) + "]");
  }

  // Enforce contiguous layout
  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);

  // Bias must be float32 contiguous (read as device float* in kernel)
  auto bc = mlx::core::astype(attn_bias, mlx::core::float32, s);
  bc = mlx::core::contiguous(bc, false, s);

  int D = qc.shape(3);
  if (D != 64 && D != 128 && D != 256)
    throw std::invalid_argument(
        "MFA bias: head_dim must be 64, 128, or 256, got " +
        std::to_string(D));

  MFAttention::Params params{
    D, scale, causal,
    /*has_block_mask=*/false,
    /*has_rope=*/false,
    /*rope_interleaved=*/false,
    /*cache_seqlens=*/0,
    /*softcap=*/0.0f,
    /*has_alibi=*/false,
    /*has_attn_bias=*/true,
    /*attn_bias_mode=*/attn_bias_mode,
    /*window_left=*/-1,
    /*window_right=*/-1
  };

  auto out_shape  = qc.shape();
  mlx::core::Shape lse_shape = {qc.shape(0), qc.shape(1), qc.shape(2)};

  // inputs: [Q, K, V, attn_bias]
  // Metal buffers: Q=0, K=1, V=2, O=3, L=4, params=5, attn_bias=10
  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {qc.dtype(), mlx::core::float32},
      std::make_shared<MFAttention>(s, params),
      {qc, kc, vc, bc});

  return outputs[0];
}

// =========================================================================
// MFAVarlenAttention::eval_gpu
// =========================================================================
//
// Inputs:  Q(0), K(1), V(2), cu_seqlens_q(3), cu_seqlens_k(4), tile_offsets(5)
// Outputs: O(0), L(1)
// Metal:   Q=buf0, K=buf1, V=buf2, O=buf3, L=buf4, params=buf5, cu_q=buf6, cu_k=buf7, tiles=buf8

void MFAVarlenAttention::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  assert(inputs.size()  == 6);
  assert(outputs.size() == 2);

  const auto& q              = inputs[0];
  const auto& k              = inputs[1];
  const auto& v              = inputs[2];
  const auto& cu_seqlens_q   = inputs[3];
  const auto& cu_seqlens_k   = inputs[4];
  const auto& tile_offsets   = inputs[5];

  const int H  = q.shape(1);
  const int Hk = k.shape(1);
  const int total_q  = q.shape(2);
  const int total_kv = k.shape(2);
  const int D  = q.shape(3);
  const int num_seqs = (int)cu_seqlens_q.size() - 1;
  // tile_offsets is evaluated when eval_gpu() is called — last element = total tiles.
  const int total_q_tiles = tile_offsets.data<int>()[num_seqs];

  auto& O = outputs[0];
  auto& L = outputs[1];
  O.set_data(mlx::core::allocator::malloc(O.nbytes()));
  L.set_data(mlx::core::allocator::malloc(L.nbytes()));

  auto& dev = mlx::core::metal::device(stream().device);
  int arch_gen = static_cast<int>(dev.get_architecture_gen());
  if (MFAEnvConfig::get().force_gen > 0) arch_gen = MFAEnvConfig::get().force_gen;
  const bool is_m3_plus = (arch_gen >= 15);

  uint8_t dtype_code;
  if (q.dtype() == mlx::core::float16)       dtype_code = 0;
  else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
  else                                        dtype_code = 2;

  const auto cfg = select_steel_block_config(D, dtype_code != 2, is_m3_plus);
  const int BQ = cfg.BQ, BK = cfg.BK, BD = cfg.BD, WM = cfg.WM;

  MFASteelVarlenParams vp{};
  vp.H             = H;
  vp.D             = D;
  vp.gqa_factor    = H / Hk;
  vp.num_seqs      = num_seqs;
  vp.total_q       = total_q;
  vp.total_kv      = total_kv;
  vp.total_q_tiles = total_q_tiles;
  vp.scale         = params_.scale;
  vp.softcap       = 0.0f;
  vp.Q_head_stride = (long)total_q  * D;
  vp.K_head_stride = (long)total_kv * D;

  using KK = ShaderCache::KernelKey;
  KK key{KK::KernelType::SteelVarlenForward,
         D, BQ, BK, BD, WM,
         params_.causal, /*sparse=*/false, is_m3_plus,
         false, false, false, false,
         /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
         /*has_window=*/false, dtype_code};
  void* raw = ShaderCache::get().get_or_compile(key, dev.mtl_device());
  auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw);

  auto& enc = mlx::core::metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(pipeline);
  enc.set_input_array(q,            0);
  enc.set_input_array(k,            1);
  enc.set_input_array(v,            2);
  enc.set_output_array(O,           3);
  enc.set_output_array(L,           4);
  enc.set_bytes(vp,                 5);
  enc.set_input_array(cu_seqlens_q, 6);
  enc.set_input_array(cu_seqlens_k, 7);
  enc.set_input_array(tile_offsets, 8);

  enc.dispatch_threadgroups(
      MTL::Size::Make(total_q_tiles, H, 1),
      MTL::Size::Make(WM * 32, 1, 1));
}

// =========================================================================
// mfa_attention_varlen_forward (free function)
// =========================================================================
// tile_offsets pre-computed in Python from cu_seqlens_q to avoid C++ eval().
std::pair<mlx::core::array, mlx::core::array> mfa_attention_varlen_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& cu_q,
    const mlx::core::array& cu_k,
    const mlx::core::array& tile_offsets,
    float scale,
    bool  causal,
    mlx::core::Stream s) {

  // volet K1 (R7): the varlen raw binding had NO host shape validation and
  // SILENTLY cast metadata to int32 (the CX-R6-03 class — a float→int32 cast is
  // wrong, an int64 silently truncates). Validate Q/K/V + reject non-int32
  // metadata (no silent cast). Packed varlen layout is [1, H, total, D].
  validate_dense_qkv(q, k, v, "MFA varlen");
  assert_scale_finite(scale, "MFA varlen");
  if (q.shape(0) != 1)
    throw std::invalid_argument(
        "MFA varlen: packed layout requires batch dim 1 [1, H, total, D], got " +
        std::to_string(q.shape(0)));
  auto need_i32 = [&](const mlx::core::array& a, const char* nm) {
    if (a.dtype() != mlx::core::int32)
      throw std::invalid_argument(
          std::string("MFA varlen: ") + nm + " must be int32 (no silent cast)");
  };
  need_i32(cu_q, "cu_seqlens_q");
  need_i32(cu_k, "cu_seqlens_k");
  need_i32(tile_offsets, "tile_offsets");

  // CC feature-metadata SHAPE class: the kernel reads cu_seqlens_q/k and
  // tile_offsets per sequence (num_seqs = cu_seqlens_q.shape[0]-1); a wrong rank
  // or mismatched cardinality drives an out-of-bounds device read (non-finite).
  // Validate rank-1 + mutual cardinality before dispatch (Rule 8).
  auto need_1d = [&](const mlx::core::array& a, const char* nm) {
    if (a.ndim() != 1)
      throw std::invalid_argument(
          std::string("MFA varlen: ") + nm + " must be 1-D [num_seqs+1]");
  };
  need_1d(cu_q, "cu_seqlens_q");
  need_1d(cu_k, "cu_seqlens_k");
  need_1d(tile_offsets, "tile_offsets");
  if (cu_q.shape(0) < 2)
    throw std::invalid_argument(
        "MFA varlen: cu_seqlens_q must be [num_seqs+1] with num_seqs >= 1");
  const int nseq1 = (int)cu_q.shape(0);  // num_seqs + 1
  if ((int)cu_k.shape(0) != nseq1)
    throw std::invalid_argument(
        "MFA varlen: cu_seqlens_k length (" + std::to_string(cu_k.shape(0)) +
        ") must equal cu_seqlens_q length (" + std::to_string(nseq1) + ").");
  if ((int)tile_offsets.shape(0) != nseq1)
    throw std::invalid_argument(
        "MFA varlen: tile_offsets length (" + std::to_string(tile_offsets.shape(0)) +
        ") must equal num_seqs+1 (" + std::to_string(nseq1) + ").");

  // CC metadata-VALUE class: validate cu_seqlens/tile_offsets VALUES (monotone,
  // [0]==0, [-1]==total) — default-on, opt-out MFA_VARLEN_TRUST_METADATA=1.
  // TRUST-FLAG COST-CLASS (Phase 2 grid): the CHEAP scalar contracts run ALWAYS
  // (ungated) — cu_k[-1] <= total_k is the SOLE K-OOB guard for non-paged varlen, so
  // gating it under the trust flag was a silent OOB; cu_q coverage likewise.
  check_prefix_sum_bounds(cu_q, (int)q.shape(2), "cu_seqlens_q", "MFA varlen");
  check_prefix_sum_bounds(cu_k, -1, "cu_seqlens_k", "MFA varlen", (int)k.shape(2));
  assert_varlen_segment_kv(cu_q, cu_k, "MFA varlen");  // per-segment zero-KV (ungated)
  if (!varlen_trust_metadata()) {
    check_prefix_sum_values(cu_q, (int)q.shape(2), "cu_seqlens_q", "MFA varlen");
    // cu_k: empty trailing KV segment / over-allocated K is legal (cu_k[-1] <=
    // total_k); only OOB (cu_k[-1] > total_k) and non-monotone are rejected.
    check_prefix_sum_values(cu_k, -1, "cu_seqlens_k", "MFA varlen", (int)k.shape(2));
    // tile_offsets is DERIVED from cu_q + BQ → validate against the recomputation
    // (total contract), not the generic monotone check (which passed wrong values).
    validate_tile_offsets_derivation(cu_q, tile_offsets, q, s, "MFA varlen");
  }

  const int H  = q.shape(1);

  mlx::core::Shape out_shape = q.shape();
  mlx::core::Shape lse_shape = {1, H, q.shape(2)};

  MFAVarlenAttention::Params params{scale, causal, q.shape(3)};

  // Metadata is now guaranteed int32 (validated above); astype is a no-op.
  auto cu_q_i32 = mlx::core::astype(cu_q, mlx::core::int32, s);
  auto cu_k_i32 = mlx::core::astype(cu_k, mlx::core::int32, s);
  auto tile_i32 = mlx::core::astype(tile_offsets, mlx::core::int32, s);

  // CONTIG (sweep iter-2): enforce contiguity (D.5 pattern, no-op when contiguous);
  // VERIFIED finite-wrong pre-fix on a strided q view (maxabs 1.63).
  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);
  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAVarlenAttention>(s, params),
      {qc, kc, vc, cu_q_i32, cu_k_i32, tile_i32});

  return {outputs[0], outputs[1]};
}

// =========================================================================
// MFAPagedSteelForward::eval_gpu  (Track FD)
// =========================================================================
//
// Inputs:  Q(0)[B,H,N,D], k_pool(1)[num_blocks,BS,H_kv,D],
//          v_pool(2)[num_blocks,BS,H_kv,D], block_table(3)[B,max_blocks] int32,
//          seq_lens(4)[B] int32
// Outputs: O(0)[B,H,N,D], L(1)[B,H,N] float32
//
// Metal buffer layout:
//   Q=0, k_pool=1, v_pool=2, block_table=3, seq_lens=4, O=5, L=6, params=7

void MFAPagedSteelForward::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  assert(inputs.size()  == 5);
  assert(outputs.size() == 2);

  const auto& q           = inputs[0];  // [B, H, N, D]
  const auto& k_pool      = inputs[1];  // [num_blocks, block_size, H_kv, D]
  const auto& v_pool      = inputs[2];  // [num_blocks, block_size, H_kv, D]
  const auto& block_table = inputs[3];  // [B, max_blocks] int32
  const auto& seq_lens    = inputs[4];  // [B] int32

  const int B          = q.shape(0);
  const int H          = q.shape(1);
  const int N          = q.shape(2);   // query length
  const int D          = q.shape(3);
  const int block_size = k_pool.shape(1);
  const int H_kv       = k_pool.shape(2);
  const int num_blocks = k_pool.shape(0);
  const int max_blocks = block_table.shape(1);

  // kL = max(seq_lens) — used only for grid sizing, not per-batch masking.
  // seq_lens must already be evaluated by the free function (mx.eval'd).
  const int kL = [&]() -> int {
    int mx_len = 0;
    const int* sl = seq_lens.data<int>();
    for (int b = 0; b < B; ++b) mx_len = std::max(mx_len, sl[b]);
    return std::max(mx_len, 1);
  }();

  auto& O = outputs[0];
  auto& L = outputs[1];
  O.set_data(mlx::core::allocator::malloc(O.nbytes()));
  L.set_data(mlx::core::allocator::malloc(L.nbytes()));

  auto& dev = mlx::core::metal::device(stream().device);
  int arch_gen = static_cast<int>(dev.get_architecture_gen());
  if (MFAEnvConfig::get().force_gen > 0) arch_gen = MFAEnvConfig::get().force_gen;
  const bool is_m3_plus = (arch_gen >= 15);

  uint8_t dtype_code;
  if (q.dtype() == mlx::core::float16)       dtype_code = 0;
  else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
  else dtype_code = 2;  // f32 not currently routed here but keep safe

  const auto cfg   = select_steel_block_config(D, dtype_code != 2, is_m3_plus);
  const int BQ     = cfg.BQ;
  const int BK     = cfg.BK;
  const int WM     = cfg.WM;
  const int TGP_SIZE = WM * 32;

  // ── Kernel cache key ────────────────────────────────────────────────────
  using KK = ShaderCache::KernelKey;
  KK key{
    KK::KernelType::PagedSteelForward,
    D,
    BQ, BK, D,          // block_d = full D (no sub-tiling)
    WM,
    params_.causal,
    /*sparse=*/false,
    is_m3_plus,
    /*has_rope=*/false,
    /*rope_interleaved=*/false,
    /*has_softcap=*/false,
    /*has_alibi=*/false,
    /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
    params_.window_left >= 0 || params_.window_right >= 0,  // has_window
    dtype_code,
    /*gqa_factor=*/H / H_kv
  };

  void* raw = ShaderCache::get().get_or_compile(key, dev.mtl_device());
  auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw);

  // ── Build MFAPagedSteelParams ────────────────────────────────────────────
  const int NQ         = (N   + BQ - 1) / BQ;
  const int NK         = (kL  + BK - 1) / BK;
  const int NQ_aligned = (N   % BQ == 0) ? NQ : NQ - 1;
  const int NK_aligned = (kL  % BK == 0) ? NK : NK - 1;

  MFAPagedSteelParams pp{};
  pp.B          = B;
  pp.H          = H;
  pp.D          = D;
  pp.qL         = N;
  pp.kL         = kL;
  pp.gqa_factor = H / H_kv;
  pp.scale      = params_.scale;
  pp.NQ         = NQ;
  pp.NK         = NK;
  pp.NQ_aligned = NQ_aligned;
  pp.NK_aligned = NK_aligned;
  pp.qL_rem     = (N   % BQ == 0) ? BQ : (N   % BQ);
  pp.kL_rem     = (kL  % BK == 0) ? BK : (kL  % BK);
  // Decode: causal + N < kL → queries start at position kL-N in the KV seq.
  pp.qL_off     = (params_.causal && N < kL) ? (kL - N) : 0;
  pp.rope_q_base     = 0;   // RoPE not fused in paged path
  pp.rope_cos_stride = D / 2;
  // Q strides [B, H, N] (D=1 implicit)
  pp.Q_strides[0] = (int64_t)H  * N * D;
  pp.Q_strides[1] = (int64_t)N  * D;
  pp.Q_strides[2] = (int64_t)D;
  // O strides [B, H, N]
  pp.O_strides[0] = (int64_t)H  * N * D;
  pp.O_strides[1] = (int64_t)N  * D;
  pp.O_strides[2] = (int64_t)D;
  // L strides [B, H]
  pp.L_strides[0] = (int64_t)H  * N;
  pp.L_strides[1] = (int64_t)N;
  // Optional features
  pp.softcap      = 0.0f;
  pp.has_alibi    = 0;
  pp.window_left  = params_.window_left;
  pp.window_right = params_.window_right;
  // Paged-specific
  pp.block_size        = block_size;
  pp.max_blocks        = max_blocks;
  pp.pool_block_stride = block_size * H_kv * D;   // tokens/block * heads * D
  pp.pool_tok_stride   = H_kv * D;                // per-token stride (heads * D)
  pp.H_kv              = H_kv;
  pp.num_blocks        = num_blocks;              // OOB guard upper bound (CC-02)

  // ── Dispatch ─────────────────────────────────────────────────────────────
  auto& enc = mlx::core::metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(pipeline);

  // Buffer layout: Q=0, k_pool=1, v_pool=2, block_table=3, seq_lens=4,
  //                O=5, L=6, params=7
  enc.set_input_array (q,           0);
  enc.set_input_array (k_pool,      1);
  enc.set_input_array (v_pool,      2);
  enc.set_input_array (block_table, 3);
  enc.set_input_array (seq_lens,    4);
  enc.set_output_array(O,           5);
  enc.set_output_array(L,           6);
  enc.set_bytes       (pp,          7);

  // Persistent kernel: each TG handles 4 consecutive Q-tiles.
  static constexpr int kTilesPerTG = 4;
  const int NQ_tgs = (NQ + kTilesPerTG - 1) / kTilesPerTG;
  enc.dispatch_threadgroups(
      MTL::Size::Make(NQ_tgs, H, B),
      MTL::Size::Make(TGP_SIZE, 1, 1));
}

// =========================================================================
// mfa_paged_steel_forward (free function)
// =========================================================================

std::pair<mlx::core::array, mlx::core::array> mfa_paged_steel_forward(
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    float scale,
    bool  causal,
    int   window_left,
    int   window_right,
    int   block_size,
    mlx::core::Stream s) {

  // Validate shapes
  if (q.ndim() != 4)
    throw std::invalid_argument("mfa_paged_steel_forward: q must be 4-D [B,H,N,D]");
  if (k_pool.ndim() != 4)
    throw std::invalid_argument("mfa_paged_steel_forward: k_pool must be 4-D [num_blocks,BS,H_kv,D]");
  if (v_pool.ndim() != 4)
    throw std::invalid_argument("mfa_paged_steel_forward: v_pool must be 4-D [num_blocks,BS,H_kv,D]");
  // CC Batch Class D: pools read at q's dtype → reject mixed q/pool dtype.
  assert_raw_pool_dtype_eq(q, k_pool, v_pool, "mfa_paged_steel_forward");
  assert_raw_paged_gqa(q, (int)k_pool.shape(2), "mfa_paged_steel_forward");
  // CX-03 (volet H): the kernel derives num_blocks/block_size/H_kv/D and all
  // strides from k_pool and binds V at those SAME offsets — a V pool disagreeing
  // on any of the four dims drives an out-of-bounds device read (silent finite-
  // wrong). Require v_pool to match k_pool exactly.
  for (int d = 0; d < 4; ++d)
    if (v_pool.shape(d) != k_pool.shape(d))
      throw std::invalid_argument(
          "mfa_paged_steel_forward: v_pool shape must equal k_pool shape "
          "[num_blocks,block_size,H_kv,D] (mismatch at dim " + std::to_string(d) +
          ": k=" + std::to_string(k_pool.shape(d)) + " v=" +
          std::to_string(v_pool.shape(d)) + ").");
  // CX-03b (sweep iter-1, subset-derive OOB): the kernel bakes D = q.shape(3) and
  // strides the K/V pool at q's D. CX-03 above only proves v_pool==k_pool — it
  // never checks the pool head_dim against q. A pool with D != q.D (e.g. q.D=128,
  // pool.D=64) strides the pool at the wrong (2x) width → OOB / finite-WRONG with
  // NO raise. Cross-check pool head_dim vs q head_dim (mirrors the sage/gna/dense
  // raw kernels, which all enforce k.shape(3)==q.shape(3)).
  if (k_pool.shape(3) != q.shape(3))
    throw std::invalid_argument(
        "mfa_paged_steel_forward: k/v pool head_dim (" +
        std::to_string(k_pool.shape(3)) + ") must equal q head_dim (" +
        std::to_string(q.shape(3)) +
        "); the kernel strides the pool at q's D and would read out of bounds.");
  if (block_table.ndim() != 2)
    throw std::invalid_argument("mfa_paged_steel_forward: block_table must be 2-D [B,max_blocks]");
  if (seq_lens.ndim() != 1)
    throw std::invalid_argument("mfa_paged_steel_forward: seq_lens must be 1-D [B]");

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int N = q.shape(2);
  const int D = q.shape(3);

  // CX-02 (volet C2, raw host guard mirroring mfa_paged_kv_gather): the kernel
  // reads block_table[b]/seq_lens[b] for b in [0,B) where B is the query batch.
  // A shorter table/seq_lens drives an out-of-bounds device read (silent NaN /
  // finite-wrong).  Reject before dispatch (Rule 8).
  if (block_table.shape(0) != B)
    throw std::invalid_argument(
        "mfa_paged_steel_forward: block_table batch size (" +
        std::to_string(block_table.shape(0)) + ") must equal q batch B (" +
        std::to_string(B) + ").");
  if (seq_lens.shape(0) != B)
    throw std::invalid_argument(
        "mfa_paged_steel_forward: seq_lens length (" +
        std::to_string(seq_lens.shape(0)) + ") must equal q batch B (" +
        std::to_string(B) + ").");

  // seq_lens must be evaluated before eval_gpu() so we can compute kL for grid.
  // The free function forces evaluation here.
  mlx::core::eval(seq_lens);

  // CX-R6-03 (volet I): require int32 metadata — the prior silent astype masked
  // int64/float bugs (float seq_lens HANGS; int64 reads wrong indices). The public
  // wrappers already reject non-int32; the raw entry must too (Rule 8).
  if (block_table.dtype() != mlx::core::int32)
    throw std::invalid_argument(
        "mfa_paged_steel_forward: block_table must be int32 (got a different dtype; "
        "the kernel reads it as int32).");
  if (seq_lens.dtype() != mlx::core::int32)
    throw std::invalid_argument(
        "mfa_paged_steel_forward: seq_lens must be int32 (got a different dtype; "
        "a float/int64 seq_lens drives a wrong/garbage kv length).");
  auto bt_i32 = block_table;
  auto sl_i32 = seq_lens;
  mlx::core::eval(sl_i32);

  // CAPACITY-SEAM (raw, Codex NO-GO): the raw paged forward had NO logical capacity
  // check — an over-capacity seq_lens silently diluted over -1/zero blocks (finite-
  // wrong). Add the shared structural contract (block_table columns = max_blocks).
  // Pass-2 over-rejection fix: the steel kernel derives block_size from k_pool.shape(1)
  // and IGNORES the param (byteΔ=0 proven), so the capacity guard MUST use the pool's
  // real block dim — using the unvalidated param over-rejected a valid call (param<pool)
  // and under-protected (param>pool). varlen/tq kernels DO use params_.block_size, so
  // their guard sites (which pass the param) stay correct — do not change them.
  // reject_zero=true: every batch row is queried in the dense-paged steel path, so a
  // 0-length seq_lens row is a queried zero-KV -> non-deterministic NaN (raw-parity sweep).
  assert_paged_capacity(sl_i32, (int)block_table.shape(1), (int)k_pool.shape(1),
                        "mfa_paged_steel_forward", /*reject_zero=*/true);
  assert_scale_finite(scale, "mfa_paged_steel_forward");

  mlx::core::Shape out_shape = q.shape();          // [B, H, N, D]
  mlx::core::Shape lse_shape = {B, H, N};          // logsumexp

  MFAPagedSteelForward::Params params{D, scale, causal, window_left, window_right, block_size};

  // CONTIG (sweep iter-2): the kernel reads q/pools with contiguous-assumed BHND
  // strides. A NON-contiguous q (e.g. an ordinary sliced query VIEW q[:,:,0:2,:],
  // the common LLM-serving pattern) was read with wrong strides → DETERMINISTIC
  // finite-WRONG output (maxabs ~1.6), NO raise. Enforce contiguity here (the D.5
  // pattern used by the dense/sparse/bias hosts; mlx::core::contiguous is a no-op /
  // zero-copy when already contiguous) so BOTH the public wrapper and this raw _ext
  // entry are correct on strided inputs.
  auto qc  = mlx::core::contiguous(q, false, s);
  auto kpc = mlx::core::contiguous(k_pool, false, s);
  auto vpc = mlx::core::contiguous(v_pool, false, s);
  // INDEX-METADATA contiguity (raw-parity sweep): the kernel reads block_table /
  // seq_lens with contiguous-assumed strides; a strided index array (e.g. a sliced
  // block_table) read wrong physical-block ids -> finite-WRONG. Contiguize them too.
  auto btc = mlx::core::contiguous(bt_i32, false, s);
  auto slc = mlx::core::contiguous(sl_i32, false, s);

  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAPagedSteelForward>(s, params),
      {qc, kpc, vpc, btc, slc});

  return {outputs[0], outputs[1]};
}

// =========================================================================
// MFASageForward::eval_gpu  (Track KB, CP2)
// =========================================================================
//
// Inputs:  q(0)[B,H,N,D] fp16/bf16, k_int8(1)[B,H_kv,S,D] int8,
//          v(2)[B,H_kv,S,D], k_scale(3)[B,H_kv,NK_blocks]
// Outputs: O(0)[B,H,N,D], L(1)[B,H,N] float32
//
// Metal buffer layout (CP2): Q=0, K_int8=1, V=2, O=3, L=4, params=5, K_scale=6

void MFASageForward::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  assert(inputs.size()  == 4);
  assert(outputs.size() == 2);

  const auto& q       = inputs[0];  // [B, H,    N, D] fp16/bf16
  const auto& k_int8  = inputs[1];  // [B, H_kv, S, D] int8
  const auto& v       = inputs[2];  // [B, H_kv, S, D] fp16/bf16
  const auto& k_scale = inputs[3];  // [B, H_kv, NK_blocks]  float32

  const int B    = q.shape(0);
  const int H    = q.shape(1);
  const int N    = q.shape(2);   // query length
  const int D    = q.shape(3);
  const int H_kv = k_int8.shape(1);
  const int S    = k_int8.shape(2);   // KV length

  auto& O = outputs[0];
  auto& L = outputs[1];
  O.set_data(mlx::core::allocator::malloc(O.nbytes()));
  L.set_data(mlx::core::allocator::malloc(L.nbytes()));

  auto& dev = mlx::core::metal::device(stream().device);
  int arch_gen = static_cast<int>(dev.get_architecture_gen());
  if (MFAEnvConfig::get().force_gen > 0) arch_gen = MFAEnvConfig::get().force_gen;
  const bool is_m3_plus = (arch_gen >= 15);

  // V dtype code (O has same dtype as V)
  uint8_t dtype_code;
  if (v.dtype() == mlx::core::float16)       dtype_code = 0;
  else if (v.dtype() == mlx::core::bfloat16) dtype_code = 1;
  else dtype_code = 2;

  // CP3: Sage uses V2 tile sizes for D<=128. The Sage kernel already uses
  // sequential K/V sharing (Ks = KV_smem; Vs = KV_smem) — same as V2 STEEL.
  // V2 BK doubles K-tile coverage, halving K-tile iterations and barriers.
  // TGP: D=64 BK=64→13,824B; D=128 BK=32→18,944B — all <32KB.
  // Note: D=128 uses BK=32 on ALL gens (not M3+-adaptive BK=64) so that
  // sage_block_sizes() can return a gen-independent value for Python-side
  // quantization (K_scale.shape[-1] = S/BK must match kernel BK exactly).
  const bool is_low_prec = (dtype_code != 2);
  int BQ, BK, WM;
  if (D <= 128) {
    // Use V2 config but cap D=128 at BK=32 for Sage (Python API compatibility).
    const auto cfgv2_64 = select_steel_v2_block_config(64,  /*is_m3_plus=*/false);
    const auto cfgv2_128 = select_steel_v2_block_config(128, /*is_m3_plus=*/false);
    const auto& cfgv2 = (D <= 64) ? cfgv2_64 : cfgv2_128;
    BQ = cfgv2.BQ; BK = cfgv2.BK; WM = cfgv2.WM;
  } else {
    const auto cfgv1 = select_steel_block_config(D, is_low_prec, is_m3_plus);
    BQ = cfgv1.BQ; BK = cfgv1.BK; WM = cfgv1.WM;
  }
  const int TGP_SIZE = WM * 32;

  // ── Kernel cache key ────────────────────────────────────────────────────
  using KK = ShaderCache::KernelKey;
  KK key{
    KK::KernelType::SageForward,
    D,
    BQ, BK, D,
    WM,
    params_.causal,
    /*sparse=*/false,
    is_m3_plus,
    /*has_rope=*/false,
    /*rope_interleaved=*/false,
    /*has_softcap=*/false,
    /*has_alibi=*/false,
    /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
    /*has_window=*/params_.window_left >= 0 || params_.window_right >= 0,
    dtype_code,
    /*gqa_factor=*/params_.gqa_factor
  };

  void* raw = ShaderCache::get().get_or_compile(key, dev.mtl_device());
  auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw);

  // ── Build MFASageParams ──────────────────────────────────────────────────
  const int NQ         = (N + BQ - 1) / BQ;
  const int NK         = (S + BK - 1) / BK;
  const int NQ_aligned = (N % BQ == 0) ? NQ : NQ - 1;
  const int NK_aligned = (S % BK == 0) ? NK : NK - 1;

  // NQ_blocks and NK_blocks are the same as NQ and NK (one scale per tile)
  // since quantization block_size was set to BQ and BK respectively.
  const int NQ_blocks = NQ;
  const int NK_blocks = NK;

  MFASageParams sp{};
  sp.B          = B;
  sp.H          = H;
  sp.D          = D;
  sp.qL         = N;
  sp.kL         = S;
  sp.gqa_factor = params_.gqa_factor;
  sp.scale      = params_.scale;
  sp.NQ         = NQ;
  sp.NK         = NK;
  sp.NQ_aligned = NQ_aligned;
  sp.NK_aligned = NK_aligned;
  sp.qL_rem     = (N % BQ == 0) ? BQ : (N % BQ);
  sp.kL_rem     = (S % BK == 0) ? BK : (S % BK);
  // For self-attention qL_off = 0; for decode with causal + N < S, offset the
  // causal mask so query at position i sees keys 0..(S-N+i).
  sp.qL_off     = (N < S && params_.causal) ? (S - N) : 0;
  // RoPE fields unused (kept for struct layout compatibility)
  sp.rope_q_base     = 0;
  sp.rope_cos_stride = D / 2;
  // Q strides [B, H, N] for fp16 (element units; CP2: Q is no longer int8)
  sp.Q_strides[0] = (int64_t)H  * N * D;
  sp.Q_strides[1] = (int64_t)N  * D;
  sp.Q_strides[2] = (int64_t)D;
  // K strides [B, H_kv, S] for int8
  sp.K_strides[0] = (int64_t)H_kv * S * D;
  sp.K_strides[1] = (int64_t)S  * D;
  sp.K_strides[2] = (int64_t)D;
  // V strides [B, H_kv, S] for fp16
  sp.V_strides[0] = (int64_t)H_kv * S * D;
  sp.V_strides[1] = (int64_t)S  * D;
  sp.V_strides[2] = (int64_t)D;
  // O strides [B, H, N] for fp16
  sp.O_strides[0] = (int64_t)H  * N * D;
  sp.O_strides[1] = (int64_t)N  * D;
  sp.O_strides[2] = (int64_t)D;
  // L strides [B, H] for float32
  sp.L_strides[0] = (int64_t)H  * N;
  sp.L_strides[1] = (int64_t)N;
  sp.softcap      = 0.0f;
  sp.has_alibi    = 0;
  sp.window_left  = params_.window_left;
  sp.window_right = params_.window_right;
  // Scale strides: K_scale [B, H_kv, NK_blocks]. Q_scale eliminated (CP2).
  sp.NQ_blocks         = NQ_blocks;
  sp.NK_blocks         = NK_blocks;
  sp.k_scale_stride_b  = H_kv * NK_blocks;
  sp.k_scale_stride_h  = NK_blocks;

  // ── Dispatch ─────────────────────────────────────────────────────────────
  auto& enc = mlx::core::metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(pipeline);

  // Buffer layout (CP2): Q=0, K_int8=1, V=2, O=3, L=4, params=5, K_scale=6
  enc.set_input_array (q,       0);
  enc.set_input_array (k_int8,  1);
  enc.set_input_array (v,       2);
  enc.set_output_array(O,       3);
  enc.set_output_array(L,       4);
  enc.set_bytes       (sp,      5);
  enc.set_input_array (k_scale, 6);

  // Non-persistent grid: one TG per Q-tile
  enc.dispatch_threadgroups(
      MTL::Size::Make(NQ, H, B),
      MTL::Size::Make(TGP_SIZE, 1, 1));
}

// =========================================================================
// mfa_sage_forward (free function)
// =========================================================================

std::pair<mlx::core::array, mlx::core::array> mfa_sage_forward(
    const mlx::core::array& q,
    const mlx::core::array& k_int8,
    const mlx::core::array& v,
    const mlx::core::array& k_scale,
    float scale,
    bool  causal,
    int   window_left,
    int   window_right,
    mlx::core::Stream s) {

  // Shape validation
  if (q.ndim() != 4)
    throw std::invalid_argument("mfa_sage_forward: q must be 4-D [B,H,N,D]");
  if (k_int8.ndim() != 4)
    throw std::invalid_argument("mfa_sage_forward: k_int8 must be 4-D [B,H_kv,S,D]");
  if (v.ndim() != 4)
    throw std::invalid_argument("mfa_sage_forward: v must be 4-D [B,H_kv,S,D]");
  if (k_scale.ndim() != 3)
    throw std::invalid_argument("mfa_sage_forward: k_scale must be 3-D [B,H_kv,NK_blocks]");

  const int B    = q.shape(0);
  const int H    = q.shape(1);
  const int N    = q.shape(2);
  const int D    = q.shape(3);
  const int H_kv = k_int8.shape(1);
  const int S    = k_int8.shape(2);

  // ZERO-KV reject (Phase 2 pass-3 — the wrapper-vs-raw meta-class: the Python
  // sage_attention/_prequantized wrappers got this guard in pass-1, the raw _ext
  // host did not). Attention over zero keys is undefined (softmax over -inf → NaN).
  // Push the guard to the SOURCE so the wrapper AND the raw entry are both covered.
  if (S == 0)
    throw std::invalid_argument(
        "mfa_sage_forward: empty KV (k sequence length 0) is undefined — attention "
        "cannot attend to zero keys.");

  if (H % H_kv != 0)
    throw std::invalid_argument(
        "mfa_sage_forward: H must be divisible by H_kv (GQA).");

  // SCALE value-semantics (Phase 2 raw-host parity sweep — mirror the established
  // scale-finite class: flash_attention attention.py:607, NAX route 1460, and
  // mfa_v6_nax_primitive's std::isfinite(scale) host guard at line ~2339). A
  // nan/inf scale silently produces a NON-FINITE output (verified: scale=nan/inf
  // → output all-NaN, no raise) on BOTH the public sage wrappers AND this raw
  // host (neither validated it, unlike flash_attention). scale is baked into the
  // kernel as a #define; garbage scale → silent NaN. Push the guard to the SOURCE
  // so the wrapper AND the raw entry are both covered (Rule 8).
  if (!std::isfinite(scale))
    throw std::invalid_argument(
        "mfa_sage_forward: scale must be finite.");

  // CX-R7-01 (volet J): buffer-shape/dtype lock. The kernel derives extents from
  // q/k_int8 and reads v / k_scale at K's offsets without re-checking — a
  // half-length V → OOB, batch mismatch → NaN, wrong k_int8/k_scale dtype →
  // garbage (all observed). Validate before dispatch (Rule 8).
  if (k_int8.shape(0) != B || v.shape(0) != B)
    throw std::invalid_argument(
        "mfa_sage_forward: q, k_int8, v must share the batch dim.");
  if (v.shape(2) != S)
    throw std::invalid_argument(
        "mfa_sage_forward: k_int8 and v must share kv sequence length.");
  if (v.shape(1) != H_kv)
    throw std::invalid_argument(
        "mfa_sage_forward: k_int8 and v must have the same number of heads.");
  if (k_int8.shape(3) != D)
    throw std::invalid_argument(
        "mfa_sage_forward: q and k_int8 must share head_dim.");
  if (v.shape(3) != D)
    throw std::invalid_argument(
        "mfa_sage_forward: v head_dim must equal q/k head_dim "
        "(no D_v != D_qk).");
  if (k_int8.dtype() != mlx::core::int8)
    throw std::invalid_argument(
        "mfa_sage_forward: k_int8 must be int8.");
  if (k_scale.dtype() != mlx::core::float32)
    throw std::invalid_argument(
        "mfa_sage_forward: k_scale must be float32.");
  if (q.dtype() != v.dtype() ||
      (q.dtype() != mlx::core::float16 && q.dtype() != mlx::core::bfloat16))
    throw std::invalid_argument(
        "mfa_sage_forward: q and v must share an fp16/bf16 dtype.");
  // k_scale must cover ceil(S / BK) blocks; the kernel uses BK = (D<=64?64:32)
  // (cfgv2_64 / cfgv2_128), the same value sage_block_sizes() reports.
  {
    const int sage_BK = (D <= 64) ? 64 : 32;
    const int nblk = (S + sage_BK - 1) / sage_BK;
    if (k_scale.ndim() != 3 || k_scale.shape(0) != B ||
        k_scale.shape(1) != H_kv || k_scale.shape(2) < nblk)
      throw std::invalid_argument(
          "mfa_sage_forward: k_scale must be [B, H_kv, >=ceil(S/BK)] "
          "(BK=64 for D<=64 else 32); a short scale array reads out of bounds.");
  }

  const int gqa_factor = H / H_kv;

  mlx::core::Shape out_shape = {B, H, N, D};
  mlx::core::Shape lse_shape = {B, H, N};

  MFASageForward::Params params{D, scale, causal, gqa_factor, window_left, window_right};

  // CONTIG (sweep iter-2): enforce contiguity (D.5 pattern, no-op when contiguous);
  // VERIFIED finite-wrong pre-fix on a strided q view (maxabs 0.72).
  // k_scale ALSO contiguized (Phase 2 raw-host parity sweep — the recurring
  // wrapper-vs-raw class): the public sage_attention_prequantized wrapper
  // flatten+reshapes k_scale (attention.py:2222) precisely because KVCache-slice
  // head strides "confuse C++ kernel offset arithmetic", but the raw _ext host
  // passed k_scale through un-contiguized — a transposed/strided k_scale read
  // finite-WRONG (byteΔ 0.204 vs contiguous, deterministic). Push the guard to
  // the SOURCE so the wrapper AND the raw entry are both covered (Rule 8).
  auto qc  = mlx::core::contiguous(q, false, s);
  auto kc8 = mlx::core::contiguous(k_int8, false, s);
  auto vc  = mlx::core::contiguous(v, false, s);
  auto ksc = mlx::core::contiguous(k_scale, false, s);
  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {v.dtype(), mlx::core::float32},
      std::make_shared<MFASageForward>(s, params),
      {qc, kc8, vc, ksc});

  return {outputs[0], outputs[1]};
}

// =========================================================================
// MFAGNAForward::eval_gpu  (Phase A: GNA native kernel)
// =========================================================================
//
// Inputs:  q(0)[B,H,N,D] fp16/bf16, k(1)[B,H_kv,S,D], v(2)[B,H_kv,S,D]
// Outputs: O(0)[B,H,N,D], L(1)[B,H,N] float32
//
// Metal buffer layout: Q=0, K=1, V=2, O=3, L=4, MFASteelParams=5, MFAGNAParams=6

void MFAGNAForward::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  {
    FILE* dbg = fopen("/tmp/mfa_gna_debug.log", "a");
    if (dbg) { fprintf(dbg, "MFAGNAForward::eval_gpu called!\n"); fclose(dbg); }
  }

  assert(inputs.size()  == 3);
  assert(outputs.size() == 2);

  const auto& q = inputs[0];  // [B, H,    N, D]
  const auto& k = inputs[1];  // [B, H_kv, S, D]
  const auto& v = inputs[2];  // [B, H_kv, S, D]

  const int B    = q.shape(0);
  const int H    = q.shape(1);
  const int N    = q.shape(2);
  const int D    = q.shape(3);
  const int H_kv = k.shape(1);
  const int S    = k.shape(2);

  auto& O = outputs[0];
  auto& L = outputs[1];
  O.set_data(mlx::core::allocator::malloc(O.nbytes()));
  L.set_data(mlx::core::allocator::malloc(L.nbytes()));

  auto& dev = mlx::core::metal::device(stream().device);
  int arch_gen = static_cast<int>(dev.get_architecture_gen());
  if (MFAEnvConfig::get().force_gen > 0) arch_gen = MFAEnvConfig::get().force_gen;
  const bool is_m3_plus = (arch_gen >= 15);

  uint8_t dtype_code;
  if (q.dtype() == mlx::core::float16)       dtype_code = 0;
  else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
  else dtype_code = 2;

  // GNA uses V2 block config (same tiling/accumulation pattern)
  auto cfg = select_steel_v2_block_config(D, is_m3_plus);
  int BQ = cfg.BQ;
  int BK = cfg.BK;
  int WM = cfg.WM;
  int TGP_SIZE = WM * 32;

  // ── Kernel cache key ────────────────────────────────────────────────────
  using KK = ShaderCache::KernelKey;
  KK key{
    KK::KernelType::GNAForward,
    D,
    BQ, BK, D,
    WM,
    /*causal=*/false,
    /*sparse=*/false,
    is_m3_plus,
    /*has_rope=*/false,
    /*rope_interleaved=*/false,
    /*has_softcap=*/false,
    /*has_alibi=*/false,
    /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
    /*has_window=*/false,
    dtype_code,
    /*gqa_factor=*/params_.gqa_factor
  };

  void* raw = ShaderCache::get().get_or_compile(key, dev.mtl_device());
  auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw);

  // ── Build MFASteelParams ────────────────────────────────────────────────
  const int NQ         = (N + BQ - 1) / BQ;
  const int NK         = (S + BK - 1) / BK;
  const int NQ_aligned = (N % BQ == 0) ? NQ : NQ - 1;
  const int NK_aligned = (S % BK == 0) ? NK : NK - 1;

  MFASteelParams sp{};
  sp.B          = B;
  sp.H          = H;
  sp.D          = D;
  sp.qL         = N;
  sp.kL         = S;
  sp.gqa_factor = params_.gqa_factor;
  sp.scale      = params_.scale;
  sp.NQ         = NQ;
  sp.NK         = NK;
  sp.NQ_aligned = NQ_aligned;
  sp.NK_aligned = NK_aligned;
  sp.qL_rem     = (N % BQ == 0) ? BQ : (N % BQ);
  sp.kL_rem     = (S % BK == 0) ? BK : (S % BK);
  sp.qL_off     = 0;  // GNA is non-causal

  sp.rope_q_base     = 0;
  sp.rope_cos_stride = D / 2;

  // Strides
  sp.Q_strides[0] = (int64_t)H    * N * D;
  sp.Q_strides[1] = (int64_t)N    * D;
  sp.Q_strides[2] = (int64_t)D;
  sp.K_strides[0] = (int64_t)H_kv * S * D;
  sp.K_strides[1] = (int64_t)S    * D;
  sp.K_strides[2] = (int64_t)D;
  sp.V_strides[0] = (int64_t)H_kv * S * D;
  sp.V_strides[1] = (int64_t)S    * D;
  sp.V_strides[2] = (int64_t)D;
  sp.O_strides[0] = (int64_t)H    * N * D;
  sp.O_strides[1] = (int64_t)N    * D;
  sp.O_strides[2] = (int64_t)D;
  sp.L_strides[0] = (int64_t)H    * N;
  sp.L_strides[1] = (int64_t)N;

  sp.softcap     = 0.0f;
  sp.has_alibi   = 0;
  sp.window_left  = -1;
  sp.window_right = -1;
  sp.mask_batch_stride = 0;
  sp.mask_head_stride  = 0;

  // ── Build MFAGNAParams ──────────────────────────────────────────────────
  MFAGNAParams gna{};
  gna.dim0 = params_.dim0;
  gna.dim1 = params_.dim1;
  gna.dim2 = params_.dim2;
  gna.win0 = params_.win0;
  gna.win1 = params_.win1;
  gna.win2 = params_.win2;
  gna.str0 = params_.str0;
  gna.str1 = params_.str1;
  gna.str2 = params_.str2;
  gna.dim12 = params_.dim1 * params_.dim2;

  // ── Dispatch ────────────────────────────────────────────────────────────
  auto& enc = mlx::core::metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(pipeline);

  enc.set_input_array (q, 0);
  enc.set_input_array (k, 1);
  enc.set_input_array (v, 2);
  enc.set_output_array(O, 3);
  enc.set_output_array(L, 4);
  enc.set_bytes       (sp,  5);
  enc.set_bytes       (gna, 6);

  enc.dispatch_threadgroups(
      MTL::Size::Make(NQ, H, B),
      MTL::Size::Make(TGP_SIZE, 1, 1));
}

// =========================================================================
// mfa_gna_forward (free function)
// =========================================================================

mlx::core::array mfa_gna_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    float scale,
    int dim0, int dim1, int dim2,
    int win0, int win1, int win2,
    int str0, int str1, int str2,
    std::optional<mlx::core::StreamOrDevice> stream) {

  auto s = stream.has_value()
      ? mlx::core::to_stream(stream.value())
      : mlx::core::default_stream(mlx::core::Device::gpu);

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4)
    throw std::invalid_argument("MFA GNA: expected 4D inputs [B, H, N, D]");

  int D = q.shape(3);
  if (D != 128)
    throw std::invalid_argument(
        "MFA GNA: only D=128 supported, got " + std::to_string(D));

  if (q.dtype() == mlx::core::float32)
    throw std::invalid_argument(
        "MFA GNA: float32 not supported; use float16 or bfloat16");
  // volet K2 (R8): mutual dtype (k/v could differ from q → silent-wrong) +
  // positive lattice/window/stride (a 0/neg window or stride → NaN / invalid
  // grid). Production passes equal f16/bf16 and all params >= 1.
  if (k.dtype() != q.dtype() || v.dtype() != q.dtype())
    throw std::invalid_argument("MFA GNA: q, k, v must share dtype");
  assert_scale_finite(scale, "MFA GNA");
  if (dim0 <= 0 || dim1 <= 0 || dim2 <= 0)
    throw std::invalid_argument("MFA GNA: lattice dims (dim0,dim1,dim2) must be positive");
  if (win0 <= 0 || win1 <= 0 || win2 <= 0)
    throw std::invalid_argument("MFA GNA: window sizes (win0,win1,win2) must be positive");
  if (str0 <= 0 || str1 <= 0 || str2 <= 0)
    throw std::invalid_argument("MFA GNA: strides (str0,str1,str2) must be positive");

  int N = q.shape(2);
  int expected_N = dim0 * dim1 * dim2;
  if (N != expected_N)
    throw std::invalid_argument(
        "MFA GNA: N (" + std::to_string(N) + ") != dim0*dim1*dim2 (" +
        std::to_string(expected_N) + ")");

  int H    = q.shape(1);
  int H_kv = k.shape(1);
  if (H_kv <= 0 || H % H_kv != 0)
    throw std::invalid_argument("MFA GNA: H must be divisible by H_kv (GQA)");

  // CX-03 (volet C2b, raw host guard mirroring the public flash_attention_gna
  // boundary): the GNA kernel derives B/N/H/D from Q and reads K/V forward from
  // their own base pointers (no broadcast).  Bq>Bk, k_seq/v_seq != N, Hv != Hk,
  // or D_k/D_v != D → out-of-bounds device reads → silent finite-wrong.  Reject
  // the malformed contract before dispatch (Rule 8).
  if (!(q.shape(0) == k.shape(0) && k.shape(0) == v.shape(0)))
    throw std::invalid_argument(
        "mfa_gna_forward: q, k, v must share the batch dim. Got q_batch=" +
        std::to_string(q.shape(0)) + ", k_batch=" + std::to_string(k.shape(0)) +
        ", v_batch=" + std::to_string(v.shape(0)) + ".");
  if (!(k.shape(2) == N && v.shape(2) == N))
    throw std::invalid_argument(
        "mfa_gna_forward: k, v sequence length must equal q's N=" +
        std::to_string(N) + " (neighborhood self-attention). Got k_seq=" +
        std::to_string(k.shape(2)) + ", v_seq=" + std::to_string(v.shape(2)) + ".");
  if (k.shape(1) != v.shape(1))
    throw std::invalid_argument(
        "mfa_gna_forward: k, v must have the same number of heads. Got k_heads=" +
        std::to_string(k.shape(1)) + ", v_heads=" + std::to_string(v.shape(1)) + ".");
  if (!(k.shape(3) == D && v.shape(3) == D))
    throw std::invalid_argument(
        "mfa_gna_forward: k, v head_dim must equal q's D=" + std::to_string(D) +
        ". Got k_dim=" + std::to_string(k.shape(3)) + ", v_dim=" +
        std::to_string(v.shape(3)) + ".");

  int gqa_factor = H / H_kv;

  MFAGNAForward::Params params{
    D, scale, gqa_factor,
    dim0, dim1, dim2,
    win0, win1, win2,
    str0, str1, str2
  };

  auto out_shape  = q.shape();
  mlx::core::Shape lse_shape = {q.shape(0), q.shape(1), q.shape(2)};

  // CONTIG (sweep iter-2): enforce contiguity (D.5 pattern, no-op when contiguous);
  // VERIFIED finite-wrong pre-fix on a strided q view (maxabs 4.44).
  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);
  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAGNAForward>(s, params),
      {qc, kc, vc});

  return outputs[0];
}

// =========================================================================
// MFAPagedVarlenForward::eval_gpu
// =========================================================================

void MFAPagedVarlenForward::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {
  // inputs: Q(0), k_pool(1), v_pool(2), cu_seqlens_q(3), tile_offsets(4),
  //         block_table(5), seq_lens_kv(6)
  auto& q            = inputs[0];  // [1, H_q, total_q, D]
  auto& k_pool       = inputs[1];
  auto& v_pool       = inputs[2];
  auto& cu_seqlens_q = inputs[3];
  auto& tile_offsets = inputs[4];
  auto& block_table  = inputs[5];
  auto& seq_lens_kv  = inputs[6];

  auto& out = outputs[0];
  auto& lse = outputs[1];
  out.set_data(mlx::core::allocator::malloc(out.nbytes()));
  lse.set_data(mlx::core::allocator::malloc(lse.nbytes()));

  const int H_q     = q.shape(1);
  const int total_q = q.shape(2);
  const int D       = q.shape(3);
  const int H_kv    = k_pool.shape(2);  // [num_pages, block_size, H_kv, D]
  const int num_blocks = (int)k_pool.shape(0);  // OOB guard upper bound (CC-02)
  const int num_seqs  = (int)cu_seqlens_q.shape(0) - 1;
  const int max_blocks = (int)block_table.shape(1);

  // tile_offsets is evaluated by MLX before eval_gpu is called
  auto tile_offsets_data = tile_offsets.data<int32_t>();
  const int total_q_tiles = tile_offsets_data[num_seqs];

  auto cfg = select_steel_block_config(D, q.dtype() != mlx::core::float32);
  const int BQ  = cfg.BQ;
  const int BK  = cfg.BK;
  const int TGP = cfg.WM * cfg.WN * 32;

  // Build Metal params
  MFAPagedVarlenParams metal_params{};
  metal_params.H               = H_q;
  metal_params.D               = D;
  metal_params.gqa_factor      = H_q / H_kv;
  metal_params.num_seqs        = num_seqs;
  metal_params.total_q         = total_q;
  metal_params.total_q_tiles   = total_q_tiles;
  metal_params.scale           = params_.scale;
  metal_params.softcap         = 0.0f;
  metal_params.Q_head_stride   = (int64_t)total_q * D;
  metal_params.block_size      = params_.block_size;
  metal_params.max_blocks      = max_blocks;
  metal_params.pool_block_stride = params_.block_size * H_kv * D;
  metal_params.pool_tok_stride   = H_kv * D;
  metal_params.H_kv            = H_kv;
  metal_params.window_left     = -1;
  metal_params.window_right    = -1;
  metal_params.num_blocks      = num_blocks;   // OOB guard upper bound (CC-02)

  // Compile kernel
  auto& d = mlx::core::metal::device(stream().device);
  int arch_gen = static_cast<int>(d.get_architecture_gen());
  if (MFAEnvConfig::get().force_gen > 0) arch_gen = MFAEnvConfig::get().force_gen;
  bool is_m3_plus = (arch_gen >= 15);

  uint8_t dtype_code = 0;
  if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;

  using KK = ShaderCache::KernelKey;
  KK kkey{};
  kkey.type       = KK::KernelType::PagedVarlenForward;
  kkey.head_dim   = D;
  kkey.block_q    = BQ;
  kkey.block_k    = BK;
  kkey.n_warps    = cfg.WM;
  kkey.dtype      = dtype_code;
  kkey.causal     = params_.causal;
  kkey.sparse     = false;
  kkey.is_m3_plus = is_m3_plus;

  void* raw = ShaderCache::get().get_or_compile(kkey, d.mtl_device());
  auto* pl  = reinterpret_cast<MTL::ComputePipelineState*>(raw);

  // Dispatch
  auto& enc = mlx::core::metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(pl);
  enc.set_input_array(q,            0);
  enc.set_input_array(k_pool,       1);
  enc.set_input_array(v_pool,       2);
  enc.set_output_array(out,         3);
  enc.set_output_array(lse,         4);
  enc.set_bytes(metal_params,       5);
  enc.set_input_array(cu_seqlens_q, 6);
  enc.set_input_array(tile_offsets, 7);
  enc.set_input_array(block_table,  8);
  enc.set_input_array(seq_lens_kv,  9);

  enc.dispatch_threadgroups(
      MTL::Size::Make((size_t)total_q_tiles, (size_t)H_q, 1),
      MTL::Size::Make((size_t)TGP, 1, 1));
}

// =========================================================================
// mfa_paged_varlen_forward — Free function
// =========================================================================

std::pair<mlx::core::array, mlx::core::array> mfa_paged_varlen_forward(
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& cu_seqlens_q,
    const mlx::core::array& tile_offsets,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens_kv,
    float scale,
    bool causal,
    int block_size,
    mlx::core::Stream stream) {

  if (q.ndim() != 4 || q.shape(0) != 1)
    throw std::runtime_error("mfa_paged_varlen_forward: Q must be [1, H_q, total_q, D]");

  // CC Batch-1 Class A: reject fp32 (silent-wrong — fp32 falls to the half
  // kernel: cos≈0, max_abs≈4.3e37). Public wrapper has an fp32 fallback bridge.
  assert_raw_fp16_bf16_only(q, "mfa_paged_varlen_forward");
  // CC Batch Class D: pools read at q's dtype → reject mixed q/pool dtype.
  assert_raw_pool_dtype_eq(q, k_pool, v_pool, "mfa_paged_varlen_forward");
  assert_raw_paged_gqa(q, (int)k_pool.shape(2), "mfa_paged_varlen_forward");

  // CX-03 (volet H): V pool bound at K's strides — require exact shape match.
  if (k_pool.ndim() != 4 || v_pool.ndim() != 4)
    throw std::invalid_argument(
        "mfa_paged_varlen_forward: k_pool/v_pool must be 4-D [num_blocks,block_size,H_kv,D].");
  for (int d = 0; d < 4; ++d)
    if (v_pool.shape(d) != k_pool.shape(d))
      throw std::invalid_argument(
          "mfa_paged_varlen_forward: v_pool shape must equal k_pool shape "
          "(mismatch at dim " + std::to_string(d) + ").");
  // CX-03b (sweep iter-1, subset-derive OOB): the kernel bakes D = q.shape(3) and
  // strides the K/V pool at q's D; the v_pool==k_pool loop above never checks the
  // pool head_dim against q. A pool with D != q.D strides at the wrong width → OOB
  // finite-WRONG with NO raise. Cross-check pool head_dim vs q head_dim.
  if (k_pool.shape(3) != q.shape(3))
    throw std::invalid_argument(
        "mfa_paged_varlen_forward: k/v pool head_dim (" +
        std::to_string(k_pool.shape(3)) + ") must equal q head_dim (" +
        std::to_string(q.shape(3)) +
        "); the kernel strides the pool at q's D and would read out of bounds.");
  // CX-02 (volet H) + CX-R6-03 (volet I): the kernel reads cu_seqlens_q,
  // block_table AND seq_lens_kv as int32 — a float seq_lens_kv HANGS (garbage
  // length), int64 reads wrong values. Require int32 on all three.
  if (cu_seqlens_q.dtype() != mlx::core::int32)
    throw std::invalid_argument(
        "mfa_paged_varlen_forward: cu_seqlens_q must be int32 (read as int32).");
  if (block_table.dtype() != mlx::core::int32)
    throw std::invalid_argument(
        "mfa_paged_varlen_forward: block_table must be int32 (read as int32).");
  if (seq_lens_kv.dtype() != mlx::core::int32)
    throw std::invalid_argument(
        "mfa_paged_varlen_forward: seq_lens_kv must be int32 (a float/int64 array "
        "drives a garbage kv length — float HANGS).");

  // CX-04 (volet C2, raw host guard): the kernel reads block_table[seq] /
  // seq_lens_kv[seq] for seq in [0, num_seqs) where num_seqs =
  // cu_seqlens_q.shape[0]-1.  Validate metadata rank + batch cardinality before
  // dispatch — a short array drives an out-of-bounds device read (silent NaN /
  // finite-wrong) (Rule 8).
  if (cu_seqlens_q.ndim() != 1 || cu_seqlens_q.shape(0) < 1)
    throw std::invalid_argument(
        "mfa_paged_varlen_forward: cu_seqlens_q must be 1-D [num_seqs+1]");
  if (block_table.ndim() != 2)
    throw std::invalid_argument(
        "mfa_paged_varlen_forward: block_table must be 2-D [num_seqs, max_blocks]");
  if (seq_lens_kv.ndim() != 1)
    throw std::invalid_argument(
        "mfa_paged_varlen_forward: seq_lens_kv must be 1-D [num_seqs]");
  if (tile_offsets.ndim() != 1)
    throw std::invalid_argument(
        "mfa_paged_varlen_forward: tile_offsets must be 1-D [num_seqs+1]");
  // CAPACITY-SEAM (raw, Codex NO-GO): mirror the steel host — the raw paged-varlen
  // forward also lacked the logical capacity check. Shared structural contract.
  assert_paged_capacity(seq_lens_kv, (int)block_table.shape(1), block_size,
                        "mfa_paged_varlen_forward");
  assert_scale_finite(scale, "mfa_paged_varlen_forward");
  {
    const int num_seqs = (int)cu_seqlens_q.shape(0) - 1;
    if (block_table.shape(0) != num_seqs)
      throw std::invalid_argument(
          "mfa_paged_varlen_forward: block_table batch size (" +
          std::to_string(block_table.shape(0)) + ") must equal num_seqs (" +
          std::to_string(num_seqs) + " = cu_seqlens_q.shape[0]-1).");
    if (seq_lens_kv.shape(0) != num_seqs)
      throw std::invalid_argument(
          "mfa_paged_varlen_forward: seq_lens_kv length (" +
          std::to_string(seq_lens_kv.shape(0)) + ") must equal num_seqs (" +
          std::to_string(num_seqs) + ").");
    // CC shape class: tile_offsets is the per-sequence tile prefix [num_seqs+1].
    if (tile_offsets.shape(0) != num_seqs + 1)
      throw std::invalid_argument(
          "mfa_paged_varlen_forward: tile_offsets length (" +
          std::to_string(tile_offsets.shape(0)) + ") must equal num_seqs+1 (" +
          std::to_string(num_seqs + 1) + ").");
  }

  // CC metadata-VALUE class: cu_seqlens_q/tile_offsets VALUE contract (monotone,
  // [0]==0, cu[-1]==total_q) — default-on, opt-out MFA_VARLEN_TRUST_METADATA=1.
  // TRUST-FLAG COST-CLASS (Phase 2 grid): cheap Q-coverage scalar contract ALWAYS.
  check_prefix_sum_bounds(cu_seqlens_q, (int)q.shape(2), "cu_seqlens_q",
                          "mfa_paged_varlen_forward");
  assert_paged_varlen_segment_kv(cu_seqlens_q, seq_lens_kv, "mfa_paged_varlen_forward");
  if (!varlen_trust_metadata()) {
    check_prefix_sum_values(cu_seqlens_q, (int)q.shape(2), "cu_seqlens_q",
                            "mfa_paged_varlen_forward");
    validate_tile_offsets_derivation(cu_seqlens_q, tile_offsets, q, stream,
                                     "mfa_paged_varlen_forward");
  }

  int H_q     = q.shape(1);
  int total_q = q.shape(2);
  int D       = q.shape(3);

  MFAPagedVarlenForward::Params params{};
  params.scale      = scale;
  params.causal     = causal;
  params.D          = D;
  params.block_size = block_size;

  mlx::core::Shape out_shape = q.shape();       // [1, H_q, total_q, D]
  mlx::core::Shape lse_shape = {H_q, total_q};  // [H_q, total_q]

  // CONTIG (sweep iter-2): non-contiguous q (sliced query view) → contiguous-assumed
  // strides → finite-WRONG. Enforce contiguity (D.5 pattern; no-op when already
  // contiguous) before dispatch — covers the public wrapper + the raw _ext entry.
  auto qc  = mlx::core::contiguous(q, false, stream);
  auto kpc = mlx::core::contiguous(k_pool, false, stream);
  auto vpc = mlx::core::contiguous(v_pool, false, stream);
  // INDEX-METADATA contiguity (raw-parity sweep): strided cu_seqlens_q/tile_offsets/
  // block_table/seq_lens_kv read with assumed strides -> finite-WRONG. Contiguize.
  auto cuc = mlx::core::contiguous(cu_seqlens_q, false, stream);
  auto toc = mlx::core::contiguous(tile_offsets, false, stream);
  auto btc = mlx::core::contiguous(block_table, false, stream);
  auto skc = mlx::core::contiguous(seq_lens_kv, false, stream);

  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAPagedVarlenForward>(stream, params),
      {qc, kpc, vpc, cuc, toc, btc, skc});

  return {outputs[0], outputs[1]};
}

// =========================================================================
// MFAPagedVarlenTQForward::eval_gpu
// =========================================================================

void MFAPagedVarlenTQForward::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {
  // inputs: Q(0), k_pool_tq(1), v_pool(2), cu_seqlens_q(3), tile_offsets(4),
  //         block_table(5), seq_lens_kv(6), centroids(7), k_scales(8)
  //         [optional V-TQ: v_pool_tq(9), v_centroids(10), v_scales(11)]
  auto& q            = inputs[0];  // [1, H_q, total_q, D]
  auto& k_pool_tq    = inputs[1];  // [num_pages, block_size, H_kv, packed_D] uint8
  auto& v_pool       = inputs[2];  // [num_pages, block_size, H_kv, D]
  auto& cu_seqlens_q = inputs[3];
  auto& tile_offsets = inputs[4];
  auto& block_table  = inputs[5];
  auto& seq_lens_kv  = inputs[6];
  auto& centroids    = inputs[7];  // [n_centroids] fp16
  auto& k_scales     = inputs[8];  // [num_pages, block_size, H_kv] f32
  const bool has_v_tq = params_.tq_v_enabled && inputs.size() > 9;

  auto& out = outputs[0];
  auto& lse = outputs[1];
  out.set_data(mlx::core::allocator::malloc(out.nbytes()));
  lse.set_data(mlx::core::allocator::malloc(lse.nbytes()));

  const int H_q     = q.shape(1);
  const int total_q = q.shape(2);
  const int D       = q.shape(3);
  const int H_kv    = v_pool.shape(2);  // [num_pages, block_size, H_kv, D]
  const int packed_D = k_pool_tq.shape(3);
  const int num_seqs  = (int)cu_seqlens_q.shape(0) - 1;
  const int max_blocks = (int)block_table.shape(1);

  auto tile_offsets_data = tile_offsets.data<int32_t>();
  const int total_q_tiles = tile_offsets_data[num_seqs];

  auto cfg = select_steel_block_config(D, q.dtype() != mlx::core::float32);
  const int BQ  = cfg.BQ;
  const int BK  = cfg.BK;
  const int TGP = cfg.WM * cfg.WN * 32;

  // Build Metal params
  MFAPagedVarlenTQParams metal_params{};
  metal_params.H               = H_q;
  metal_params.D               = D;
  metal_params.gqa_factor      = H_q / H_kv;
  metal_params.num_seqs        = num_seqs;
  metal_params.total_q         = total_q;
  metal_params.total_q_tiles   = total_q_tiles;
  metal_params.scale           = params_.scale;
  metal_params.softcap         = 0.0f;
  metal_params.Q_head_stride   = (int64_t)total_q * D;
  metal_params.block_size      = params_.block_size;
  metal_params.max_blocks      = max_blocks;
  metal_params.pool_block_stride_v = params_.block_size * H_kv * D;
  metal_params.pool_tok_stride_v   = H_kv * D;
  metal_params.pool_block_stride_k = params_.block_size * H_kv * packed_D;
  metal_params.pool_tok_stride_k   = H_kv * packed_D;
  metal_params.H_kv            = H_kv;
  metal_params.packed_D        = packed_D;
  metal_params.tq_bits         = params_.tq_bits;
  metal_params.n_centroids     = 1 << params_.tq_bits;
  metal_params.window_left     = -1;
  metal_params.window_right    = -1;

  // V-TQ fields (Phase 3A)
  metal_params.tq_v_enabled    = has_v_tq ? 1 : 0;
  if (has_v_tq) {
    metal_params.tq_v_pool_block_stride = params_.block_size * H_kv * packed_D;
    metal_params.tq_v_pool_tok_stride   = H_kv * packed_D;
  } else {
    metal_params.tq_v_pool_block_stride = 0;
    metal_params.tq_v_pool_tok_stride   = 0;
  }

  // WHT fusion (Phase 4)
  metal_params.tq_wht_enabled = params_.tq_wht_enabled ? 1 : 0;
  metal_params.num_blocks     = (int)k_pool_tq.shape(0);  // OOB guard upper bound (CC-02)

  // Compile kernel
  auto& d = mlx::core::metal::device(stream().device);
  int arch_gen = static_cast<int>(d.get_architecture_gen());
  if (MFAEnvConfig::get().force_gen > 0) arch_gen = MFAEnvConfig::get().force_gen;
  bool is_m3_plus = (arch_gen >= 15);

  uint8_t dtype_code = 0;
  if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;

  using KK = ShaderCache::KernelKey;
  KK kkey{};
  kkey.type       = KK::KernelType::PagedVarlenTQForward;
  kkey.head_dim   = D;
  kkey.block_q    = BQ;
  kkey.block_k    = BK;
  kkey.n_warps    = cfg.WM;
  kkey.dtype      = dtype_code;
  kkey.causal     = params_.causal;
  kkey.sparse     = false;
  kkey.is_m3_plus = is_m3_plus;

  void* raw = ShaderCache::get().get_or_compile(kkey, d.mtl_device());
  auto* pl  = reinterpret_cast<MTL::ComputePipelineState*>(raw);

  // Dispatch
  auto& enc = mlx::core::metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(pl);
  enc.set_input_array(q,            0);
  enc.set_input_array(k_pool_tq,    1);
  enc.set_input_array(v_pool,       2);
  enc.set_output_array(out,         3);
  enc.set_output_array(lse,         4);
  enc.set_bytes(metal_params,       5);
  enc.set_input_array(cu_seqlens_q, 6);
  enc.set_input_array(tile_offsets, 7);
  enc.set_input_array(block_table,  8);
  enc.set_input_array(seq_lens_kv,  9);
  enc.set_input_array(centroids,    10);
  enc.set_input_array(k_scales,     11);

  // V-TQ buffers (Phase 3A) — only bound when V is TQ-packed
  if (has_v_tq) {
    enc.set_input_array(inputs[9],  12);  // v_pool_tq
    enc.set_input_array(inputs[10], 13);  // v_centroids
    enc.set_input_array(inputs[11], 14);  // v_scales
  }

  enc.dispatch_threadgroups(
      MTL::Size::Make((size_t)total_q_tiles, (size_t)H_q, 1),
      MTL::Size::Make((size_t)TGP, 1, 1));
}

// =========================================================================
// mfa_paged_varlen_tq_forward — Free function
// =========================================================================

std::pair<mlx::core::array, mlx::core::array> mfa_paged_varlen_tq_forward(
    const mlx::core::array& q,
    const mlx::core::array& k_pool_tq,
    const mlx::core::array& v_pool,
    const mlx::core::array& cu_seqlens_q,
    const mlx::core::array& tile_offsets,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens_kv,
    const mlx::core::array& centroids,
    const mlx::core::array& k_scales,
    float scale,
    bool causal,
    int block_size,
    int tq_bits,
    bool tq_v_enabled,
    bool tq_wht_enabled,
    const std::optional<mlx::core::array>& v_pool_tq,
    const std::optional<mlx::core::array>& v_centroids,
    const std::optional<mlx::core::array>& v_scales,
    mlx::core::Stream stream) {

  if (q.ndim() != 4 || q.shape(0) != 1)
    throw std::runtime_error("mfa_paged_varlen_tq_forward: Q must be [1, H_q, total_q, D]");
  // CC Batch-1 Class A: reject fp32 (silent-wrong — fp32 q falls to the half
  // kernel via the same 2-way dtype_code map; prior probe cos≈0.27).
  assert_raw_fp16_bf16_only(q, "mfa_paged_varlen_tq_forward");
  // CC Batch Class D: enforce the TQ backing-buffer dtypes (packed K uint8,
  // V pool == q, centroids fp16, scales fp32) — a mismatch is read byte-wise.
  // Optional-input batch: also enforce the V-side buffers when tq_v_enabled.
  assert_raw_tq_buffer_dtypes(q, k_pool_tq, v_pool, centroids, k_scales,
                              "mfa_paged_varlen_tq_forward",
                              tq_v_enabled, v_pool_tq, v_centroids, v_scales);
  if (tq_v_enabled && (!v_pool_tq || !v_centroids || !v_scales))
    throw std::runtime_error("mfa_paged_varlen_tq_forward: tq_v_enabled requires v_pool_tq, v_centroids, v_scales");

  // CX-R6-01 (volet I): TQ backing-buffer shape lock. The kernel derives
  // num_blocks/block_size/H_kv from the packed K pool and reads v_pool / k_scales
  // (+ optional v_pool_tq / v_scales) at K's block/head offsets without re-checking
  // them — an undersized/mis-shaped buffer drives an OOB device read (v_pool OOB,
  // k_scales OOB, wrong packed_D → garbage unpack). Validate before dispatch (Rule 8).
  {
    auto packed_for = [](int D, int bits) -> int {
      if (bits == 3) return (D / 32) * 12;
      if (bits == 2) return D / 4;
      if (bits == 4) return D / 2;
      throw std::invalid_argument("mfa_paged_varlen_tq_forward: tq_bits must be 2, 3, or 4");
    };
    if (k_pool_tq.ndim() != 4)
      throw std::invalid_argument("mfa_paged_varlen_tq_forward: k_pool_tq must be 4-D [num_blocks,block_size,H_kv,packed_D]");
    if (v_pool.ndim() != 4)
      throw std::invalid_argument("mfa_paged_varlen_tq_forward: v_pool must be 4-D [num_blocks,block_size,H_kv,D]");
    const int nb = k_pool_tq.shape(0), bsz = k_pool_tq.shape(1), hkv = k_pool_tq.shape(2);
    const int packed_d = k_pool_tq.shape(3), Dv = v_pool.shape(3);
    if (v_pool.shape(0) != nb || v_pool.shape(1) != bsz || v_pool.shape(2) != hkv)
      throw std::invalid_argument(
          "mfa_paged_varlen_tq_forward: v_pool [num_blocks,block_size,H_kv] must match k_pool_tq.");
    const int exp_packed = packed_for(Dv, tq_bits);
    if (packed_d != exp_packed)
      throw std::invalid_argument(
          "mfa_paged_varlen_tq_forward: k_pool_tq packed_D (" + std::to_string(packed_d) +
          ") incompatible with D=" + std::to_string(Dv) + " tq_bits=" + std::to_string(tq_bits) +
          " (expected " + std::to_string(exp_packed) + ").");
    if (k_scales.ndim() != 3 || k_scales.shape(0) != nb || k_scales.shape(1) != bsz || k_scales.shape(2) != hkv)
      throw std::invalid_argument(
          "mfa_paged_varlen_tq_forward: k_scales must be [num_blocks,block_size,H_kv] matching k_pool_tq.");
    if (tq_v_enabled && v_pool_tq) {
      const auto& vt = *v_pool_tq;
      if (vt.ndim() != 4 || vt.shape(0) != nb || vt.shape(1) != bsz || vt.shape(2) != hkv || vt.shape(3) != exp_packed)
        throw std::invalid_argument(
            "mfa_paged_varlen_tq_forward: v_pool_tq must be [num_blocks,block_size,H_kv,packed_D] matching k_pool_tq.");
      if (v_scales && ((*v_scales).ndim() != 3 || (*v_scales).shape(0) != nb || (*v_scales).shape(1) != bsz || (*v_scales).shape(2) != hkv))
        throw std::invalid_argument(
            "mfa_paged_varlen_tq_forward: v_scales must be [num_blocks,block_size,H_kv] matching k_pool_tq.");
    }
    // CC class-method batch (Class C, raw backstop): Q-vs-pool geometry. q_heads
    // must be a valid GQA multiple of the pool kv-heads, and q head_dim must equal
    // the pool D — else a bad GQA ratio / D-mismatch is finite-wrong, not raised.
    const int Dq = q.shape(3);
    assert_raw_paged_gqa(q, hkv, "mfa_paged_varlen_tq_forward");  // shared GQA
    if (Dq != Dv)
      throw std::invalid_argument(
          "mfa_paged_varlen_tq_forward: q head_dim (" + std::to_string(Dq) +
          ") must equal the pool D (" + std::to_string(Dv) + ").");
  }
  // CX-R6-03 (volet I): int32 metadata (float seq_lens HANGS, int64 wrong values).
  if (cu_seqlens_q.dtype() != mlx::core::int32 || block_table.dtype() != mlx::core::int32 ||
      seq_lens_kv.dtype() != mlx::core::int32)
    throw std::invalid_argument(
        "mfa_paged_varlen_tq_forward: cu_seqlens_q / block_table / seq_lens_kv must be int32.");

  // CX-04 (volet C2, raw host guard): the kernel reads block_table[seq] /
  // seq_lens_kv[seq] for seq in [0, num_seqs) where num_seqs =
  // cu_seqlens_q.shape[0]-1.  Validate metadata rank + batch cardinality before
  // dispatch — a short array drives an out-of-bounds device read (silent NaN)
  // (Rule 8).
  if (cu_seqlens_q.ndim() != 1 || cu_seqlens_q.shape(0) < 1)
    throw std::invalid_argument(
        "mfa_paged_varlen_tq_forward: cu_seqlens_q must be 1-D [num_seqs+1]");
  // CX-02 (volet H): the kernel reads cu_seqlens_q as int32.
  if (cu_seqlens_q.dtype() != mlx::core::int32)
    throw std::invalid_argument(
        "mfa_paged_varlen_tq_forward: cu_seqlens_q must be int32 (read as int32).");
  if (block_table.ndim() != 2)
    throw std::invalid_argument(
        "mfa_paged_varlen_tq_forward: block_table must be 2-D [num_seqs, max_blocks]");
  if (seq_lens_kv.ndim() != 1)
    throw std::invalid_argument(
        "mfa_paged_varlen_tq_forward: seq_lens_kv must be 1-D [num_seqs]");
  // CAPACITY-SEAM (raw, Codex NO-GO): TQ paged forward also lacked the logical
  // capacity check. Shared structural contract (block_table columns = max_blocks).
  assert_paged_capacity(seq_lens_kv, (int)block_table.shape(1), block_size,
                        "mfa_paged_varlen_tq_forward");
  assert_scale_finite(scale, "mfa_paged_varlen_tq_forward");
  // CODEBOOK-EXTENT (Phase 2 grid, wrapper-vs-raw parity): the public TQ wrapper
  // checks centroids.shape[0] >= 2**tq_bits, the raw entry did not — codes in
  // [0,2**tq_bits) indexed a too-small codebook (finite-wrong, no raise).
  {
    const long long need = 1LL << tq_bits;
    if ((long long)centroids.shape(0) < need)
      throw std::invalid_argument(
          "mfa_paged_varlen_tq_forward: centroids has " +
          std::to_string(centroids.shape(0)) + " entries but tq_bits=" +
          std::to_string(tq_bits) + " indexes " + std::to_string(need) +
          " codes — the K-dequant would read past the codebook.");
    if (tq_v_enabled && v_centroids &&
        (long long)v_centroids->shape(0) < need)
      throw std::invalid_argument(
          "mfa_paged_varlen_tq_forward: v_centroids has " +
          std::to_string(v_centroids->shape(0)) + " entries but tq_bits=" +
          std::to_string(tq_bits) + " indexes " + std::to_string(need) + " codes.");
  }
  {
    const int num_seqs = (int)cu_seqlens_q.shape(0) - 1;
    if (block_table.shape(0) != num_seqs)
      throw std::invalid_argument(
          "mfa_paged_varlen_tq_forward: block_table batch size (" +
          std::to_string(block_table.shape(0)) + ") must equal num_seqs (" +
          std::to_string(num_seqs) + " = cu_seqlens_q.shape[0]-1).");
    if (seq_lens_kv.shape(0) != num_seqs)
      throw std::invalid_argument(
          "mfa_paged_varlen_tq_forward: seq_lens_kv length (" +
          std::to_string(seq_lens_kv.shape(0)) + ") must equal num_seqs (" +
          std::to_string(num_seqs) + ").");
    if (tile_offsets.ndim() != 1 || (int)tile_offsets.shape(0) != num_seqs + 1)
      throw std::invalid_argument(
          "mfa_paged_varlen_tq_forward: tile_offsets must be 1-D [num_seqs+1] (" +
          std::to_string(num_seqs + 1) + ").");
  }

  // CC metadata-VALUE class: cu_seqlens_q/tile_offsets VALUE contract (monotone,
  // [0]==0, cu[-1]==total_q) — default-on, opt-out MFA_VARLEN_TRUST_METADATA=1.
  // TRUST-FLAG COST-CLASS (Phase 2 grid): cheap Q-coverage scalar contract ALWAYS.
  check_prefix_sum_bounds(cu_seqlens_q, (int)q.shape(2), "cu_seqlens_q",
                          "mfa_paged_varlen_tq_forward");
  assert_paged_varlen_segment_kv(cu_seqlens_q, seq_lens_kv, "mfa_paged_varlen_tq_forward");
  if (!varlen_trust_metadata()) {
    check_prefix_sum_values(cu_seqlens_q, (int)q.shape(2), "cu_seqlens_q",
                            "mfa_paged_varlen_tq_forward");
    validate_tile_offsets_derivation(cu_seqlens_q, tile_offsets, q, stream,
                                     "mfa_paged_varlen_tq_forward");
  }

  int H_q     = q.shape(1);
  int total_q = q.shape(2);
  int D       = q.shape(3);
  int packed_D = k_pool_tq.shape(3);  // inferred from pool shape (bit-planar: D*12/32 for 3-bit)

  MFAPagedVarlenTQForward::Params params{};
  params.scale         = scale;
  params.causal        = causal;
  params.D             = D;
  params.block_size    = block_size;
  params.tq_bits       = tq_bits;
  params.packed_D      = packed_D;
  params.tq_v_enabled  = tq_v_enabled;
  params.tq_wht_enabled = tq_wht_enabled;

  mlx::core::Shape out_shape = q.shape();       // [1, H_q, total_q, D]
  mlx::core::Shape lse_shape = {H_q, total_q};  // [H_q, total_q]

  // CONTIG (sweep iter-2): enforce contiguity on q (sliced query views) + the
  // backing pools (D.5 pattern; no-op when already contiguous) so the fused TQ
  // kernel never misreads a strided input as contiguous (finite-wrong).
  auto qc  = mlx::core::contiguous(q, false, stream);
  auto kpc = mlx::core::contiguous(k_pool_tq, false, stream);
  auto vpc = mlx::core::contiguous(v_pool, false, stream);
  // INDEX-METADATA + codebook contiguity (raw-parity sweep): strided cu_seqlens_q/
  // tile_offsets/block_table/seq_lens_kv/centroids/k_scales read with assumed strides
  // -> finite-WRONG. Contiguize (no-op when already contiguous).
  auto cuc = mlx::core::contiguous(cu_seqlens_q, false, stream);
  auto toc = mlx::core::contiguous(tile_offsets, false, stream);
  auto btc = mlx::core::contiguous(block_table, false, stream);
  auto skc = mlx::core::contiguous(seq_lens_kv, false, stream);
  auto cec = mlx::core::contiguous(centroids, false, stream);
  auto ksc = mlx::core::contiguous(k_scales, false, stream);

  // Build inputs vector — conditionally include V-TQ arrays
  std::vector<mlx::core::array> prim_inputs = {
    qc, kpc, vpc, cuc, toc, btc, skc,
    cec, ksc
  };
  if (tq_v_enabled) {
    prim_inputs.push_back(mlx::core::contiguous(*v_pool_tq, false, stream));
    prim_inputs.push_back(mlx::core::contiguous(*v_centroids, false, stream));
    prim_inputs.push_back(mlx::core::contiguous(*v_scales, false, stream));
  }

  auto outputs = mlx::core::array::make_arrays(
      {out_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAPagedVarlenTQForward>(stream, params),
      prim_inputs);

  return {outputs[0], outputs[1]};
}

}  // namespace mlx_mfa
