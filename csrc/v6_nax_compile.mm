// Helper: compile with function constant values pre-set.
// This is needed because the Draw Things kernel uses function constants for
// runtime params (R, C, batch strides) — pipeline state creation requires
// specializing these BEFORE newComputePipelineStateWithFunction.

#include "shader_cache.hpp"
#include <Metal/Metal.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx_mfa {

// Compile and return a pipeline state with function constants R, C, and
// batch strides set to provided values.
//
// v2.30 Sprint E: optional max_threads_hint (>0) uses MTLComputePipelineDescriptor
// to set maxTotalThreadsPerThreadgroup, giving the compiler an explicit upper
// bound so it can use more registers per thread (lower TG co-residency, higher
// per-thread register count). Default 0 = use Metal's default (typically 1024).
//
// Read from MFA_V6_MAX_THREADS env var per-call (no plumbing through C++ API).
static uint32_t v6_max_threads_hint_from_env() {
  if (const char* env = std::getenv("MFA_V6_MAX_THREADS")) {
    int v = std::atoi(env);
    if (v > 0 && v <= 1024) return (uint32_t)v;
  }
  return 0;  // 0 = default
}

void* v6_nax_compile_with_constants(
    const std::string& source,
    const std::string& function_name,
    void* raw_device,
    uint32_t R, uint32_t C,
    uint32_t Q_bs, uint32_t K_bs, uint32_t V_bs, uint32_t O_bs) {
  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)raw_device;
    NSError* error = nil;

    NSString* src = [NSString stringWithUTF8String:source.c_str()];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = (MTLLanguageVersion)((4 << 16) + 0);

    id<MTLLibrary> library = [device newLibraryWithSource:src
                                                  options:opts
                                                    error:&error];
    if (!library) {
      std::string msg = "V6 library compile failed";
      if (error) msg += std::string(": ") + [[error localizedDescription] UTF8String];
      throw std::runtime_error(msg);
    }

    MTLFunctionConstantValues* fcs = [[MTLFunctionConstantValues alloc] init];
    [fcs setConstantValue:&R type:MTLDataTypeUInt atIndex:0];
    [fcs setConstantValue:&C type:MTLDataTypeUInt atIndex:1];
    [fcs setConstantValue:&Q_bs type:MTLDataTypeUInt atIndex:2];
    [fcs setConstantValue:&K_bs type:MTLDataTypeUInt atIndex:3];
    [fcs setConstantValue:&V_bs type:MTLDataTypeUInt atIndex:4];
    [fcs setConstantValue:&O_bs type:MTLDataTypeUInt atIndex:5];

    NSString* fnName = [NSString stringWithUTF8String:function_name.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:fnName
                                            constantValues:fcs
                                                     error:&error];
    if (!function) {
      std::string msg = "V6 function specialization failed";
      if (error) msg += std::string(": ") + [[error localizedDescription] UTF8String];
      throw std::runtime_error(msg);
    }

    id<MTLComputePipelineState> pipeline = nil;
    uint32_t max_hint = v6_max_threads_hint_from_env();
    if (max_hint > 0) {
      // Sprint E: explicit max_total_threads_per_threadgroup gives the
      // compiler an upper bound for register-pressure decisions.
      MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
      desc.computeFunction = function;
      desc.maxTotalThreadsPerThreadgroup = max_hint;
      pipeline = [device newComputePipelineStateWithDescriptor:desc
                                                       options:0
                                                    reflection:nil
                                                         error:&error];
    } else {
      pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    }
    if (!pipeline) {
      std::string msg = "V6 pipeline creation failed";
      if (error) msg += std::string(": ") + [[error localizedDescription] UTF8String];
      throw std::runtime_error(msg);
    }
    return (void*)CFBridgingRetain(pipeline);
  }
}

}  // namespace mlx_mfa

// === Dispatch helper ===
// Uses mlx's CommandEncoder wrapper to bind the pipeline + tg memory
// + dispatch threadgroups. Buffers are bound via set_input_array/set_output_array
// at the call site (so MLX handles fence/barrier insertion).

#include <mlx/backend/metal/device.h>
#include <stdint.h>

namespace mlx_mfa {

// ceil_log2_u32 host helper (matches kernel definition)
static uint32_t v6_ceil_log2_u32_host(uint32_t x) {
  if (x <= 1) return 0;
  x -= 1;
  uint32_t b = 0;
  while (x > 0) { x >>= 1; ++b; }
  return b;
}

void v6_nax_dispatch(
    void* pipeline_raw,
    void* enc_raw,
    void* /*q_buf*/, uint64_t /*q_offset*/,
    void* /*k_buf*/, uint64_t /*k_offset*/,
    void* /*v_buf*/, uint64_t /*v_offset*/,
    void* /*o_buf*/, uint64_t /*o_offset*/,
    void* /*l_buf*/, uint64_t /*l_offset*/,
    uint32_t R, uint32_t Hq, uint32_t batchDimension,
    unsigned short BQ, uint16_t executionSIMDGroups,
    unsigned short tgmem_bytes) {
  @autoreleasepool {
    auto& enc = *reinterpret_cast<mlx::core::metal::CommandEncoder*>(enc_raw);
    id<MTLComputePipelineState> pl =
        (__bridge id<MTLComputePipelineState>)pipeline_raw;

    enc.set_compute_pipeline_state(reinterpret_cast<MTL::ComputePipelineState*>(pipeline_raw));

    if (tgmem_bytes > 0) {
      enc.set_threadgroup_memory_length((size_t)tgmem_bytes, 0);
    }

    // Forward grid: morton tiling per Draw Things.
    //   row_groups = ceil(R / (BQ * executionSIMDGroups))
    //   morton_bits = ceil_log2(row_groups) + ceil_log2(Hq)
    //   grid = (1 << morton_bits, 1, batchDimension)
    uint32_t denom = (uint32_t)BQ * (uint32_t)executionSIMDGroups;
    uint32_t row_groups = (R + denom - 1) / denom;
    uint32_t morton_bits = v6_ceil_log2_u32_host(row_groups) +
                           v6_ceil_log2_u32_host(Hq);
    uint64_t grid_x = uint64_t(1) << morton_bits;

    enc.dispatch_threadgroups(
        MTL::Size::Make(grid_x, 1, (size_t)batchDimension),
        MTL::Size::Make((size_t)32 * (size_t)executionSIMDGroups, 1, 1));
  }
}

// V6NAX — pipeline compile (no function constants; V6NAX uses a struct buffer
// instead of FCs for params).
void* v6nax_compile(
    const std::string& source,
    const std::string& function_name,
    void* raw_device) {
  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)raw_device;
    NSError* error = nil;

    NSString* src = [NSString stringWithUTF8String:source.c_str()];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = (MTLLanguageVersion)((4 << 16) + 0);

    id<MTLLibrary> library = [device newLibraryWithSource:src
                                                  options:opts
                                                    error:&error];
    if (!library) {
      std::string msg = "V6NAX library compile failed";
      if (error) msg += std::string(": ") + [[error localizedDescription] UTF8String];
      throw std::runtime_error(msg);
    }

    NSString* fnName = [NSString stringWithUTF8String:function_name.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:fnName];
    if (!function) {
      throw std::runtime_error("V6NAX function not found: " + function_name);
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
      std::string msg = "V6NAX pipeline creation failed";
      if (error) msg += std::string(": ") + [[error localizedDescription] UTF8String];
      throw std::runtime_error(msg);
    }
    return (void*)CFBridgingRetain(pipeline);
  }
}

// V6NAX dispatcher — Apple-style grid (NQ, H, B) with TG size 32*WM.
// Params struct is set via setBytes:length:atIndex: at buffer 4 (Apple convention).
struct V6NAXParamsHost {
  int qL, kL;
  int gqa_factor;
  int NQ, NK;
  int qL_rem, kL_rem;
  // v2.50 Sprint 4 — causal offset (0 for standalone forward; nonzero
  // for decode/prefill-with-history).  Field order MUST match device-
  // side V6NAXParams struct in NAAttentionKernel.cpp::createV6NAXSource().
  int qL_off;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
  // v2.36.x lse-write patch (must mirror V6NAXParams layout exactly).
  int64_t L_strides[3];
};

void v6nax_dispatch(
    void* pipeline_raw,
    void* enc_raw,
    int qL, int kL,
    int Hq, int Hk,
    int batchDimension,
    int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM) {
  @autoreleasepool {
    auto& enc = *reinterpret_cast<mlx::core::metal::CommandEncoder*>(enc_raw);

    enc.set_compute_pipeline_state(reinterpret_cast<MTL::ComputePipelineState*>(pipeline_raw));

    // Build params struct (Apple AttnParams subset).
    V6NAXParamsHost params{};
    params.qL = qL;
    params.kL = kL;
    params.gqa_factor = Hq / Hk;
    params.NQ = (qL + BQ - 1) / BQ;
    params.NK = (kL + BK - 1) / BK;
    params.qL_rem = qL % BQ;          // 0 if aligned
    params.kL_rem = kL % BK;          // 0 if aligned
    // v2.50 Sprint 4 — qL_off=0 for standalone forward.  Decode/prefill-
    // with-history would pass nonzero offset, but v6nax_dispatch is currently
    // only called from MFAV6Forward which always assumes qL_off=0.
    params.qL_off = 0;
    // BHND strides: Q is [B, Hq, qL, D] contiguous → strides (Hq*qL*D, qL*D, D, 1).
    params.Q_strides[0] = (int64_t)Hq * qL * head_dim;
    params.Q_strides[1] = (int64_t)qL * head_dim;
    params.Q_strides[2] = (int64_t)head_dim;
    params.K_strides[0] = (int64_t)Hk * kL * head_dim;
    params.K_strides[1] = (int64_t)kL * head_dim;
    params.K_strides[2] = (int64_t)head_dim;
    params.V_strides[0] = (int64_t)Hk * kL * head_dim;
    params.V_strides[1] = (int64_t)kL * head_dim;
    params.V_strides[2] = (int64_t)head_dim;
    params.O_strides[0] = (int64_t)Hq * qL * head_dim;
    params.O_strides[1] = (int64_t)qL * head_dim;
    params.O_strides[2] = (int64_t)head_dim;
    // lse is FP32 contiguous [B, Hq, qL] -> strides (Hq*qL, qL, 1).
    // No D dimension, so L_strides[2] = 1 (per-row element stride).
    params.L_strides[0] = (int64_t)Hq * qL;
    params.L_strides[1] = (int64_t)qL;
    params.L_strides[2] = (int64_t)1;

    enc.set_bytes(params, 4);

    // Dispatch grid (NQ, H, B), TG size (32*WM, 1, 1).
    uint32_t NQ = (qL + BQ - 1) / BQ;
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)NQ, (size_t)Hq, (size_t)batchDimension),
        MTL::Size::Make((size_t)32 * (size_t)WM, 1, 1));
  }
}

// Packed-varlen dispatcher. The device kernel maps each global Q tile to a
// segment using tile_offsets, then reads segment-local lengths from cu_q/cu_k.
// Buffer indices 6/7/8 are bound by MFAV6VarlenForward.
struct V6NAXVarlenParamsHost {
  int qL, kL;
  int gqa_factor;
  int NQ, NK;
  int qL_rem, kL_rem;
  int qL_off;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
  int64_t L_strides[3];
  int num_seqs;
};

void v6nax_dispatch_varlen(
    void* pipeline_raw,
    void* enc_raw,
    int total_q, int total_k, int num_seqs, int total_q_tiles,
    int Hq, int Hk, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM) {
  @autoreleasepool {
    auto& enc = *reinterpret_cast<mlx::core::metal::CommandEncoder*>(enc_raw);
    enc.set_compute_pipeline_state(
        reinterpret_cast<MTL::ComputePipelineState*>(pipeline_raw));

    V6NAXVarlenParamsHost params{};
    params.qL = total_q;
    params.kL = total_k;
    params.gqa_factor = Hq / Hk;
    params.NQ = (total_q + BQ - 1) / BQ;
    params.NK = (total_k + BK - 1) / BK;
    params.qL_rem = total_q % BQ;
    params.kL_rem = total_k % BK;
    params.qL_off = 0;
    params.Q_strides[0] = (int64_t)Hq * total_q * head_dim;
    params.Q_strides[1] = (int64_t)total_q * head_dim;
    params.Q_strides[2] = head_dim;
    params.K_strides[0] = (int64_t)Hk * total_k * head_dim;
    params.K_strides[1] = (int64_t)total_k * head_dim;
    params.K_strides[2] = head_dim;
    params.V_strides[0] = (int64_t)Hk * total_k * head_dim;
    params.V_strides[1] = (int64_t)total_k * head_dim;
    params.V_strides[2] = head_dim;
    params.O_strides[0] = (int64_t)Hq * total_q * head_dim;
    params.O_strides[1] = (int64_t)total_q * head_dim;
    params.O_strides[2] = head_dim;
    params.L_strides[0] = (int64_t)Hq * total_q;
    params.L_strides[1] = total_q;
    params.L_strides[2] = 1;
    params.num_seqs = num_seqs;
    enc.set_bytes(params, 4);

    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)total_q_tiles, (size_t)Hq, 1),
        MTL::Size::Make((size_t)32 * (size_t)WM, 1, 1));
  }
}

// =============================================================================
// V6NAX backward dQ dispatcher (Phase 1 Section B of V6NAX backward Option β).
// Mirrors v6nax_dispatch but with backward-specific param struct.
// Buffers: Q=0, K=1, V=2, O=3, L=4, dO=5, dQ=6, params=7.
// =============================================================================

struct V6NAXBwdQParamsHost {
  int qL, kL;
  int gqa_factor;
  int NQ, NK;
  int qL_rem, kL_rem;
  // v2.50 Sprint 4 Phase 4b — causal offset (mirrors device-side V6NAX*Params struct).
  int qL_off;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
  int64_t L_strides[3], dO_strides[3], dQ_strides[3];
  int64_t D_strides[3];  // v2.38.1: D=rowsum(dO⊙O) strides (FP32, [B,Hq,qL])
};

void v6nax_dispatch_bwd_query(
    void* pipeline_raw,
    void* enc_raw,
    int qL, int kL,
    int Hq, int Hk,
    int batchDimension,
    int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM) {
  @autoreleasepool {
    auto& enc = *reinterpret_cast<mlx::core::metal::CommandEncoder*>(enc_raw);

    enc.set_compute_pipeline_state(reinterpret_cast<MTL::ComputePipelineState*>(pipeline_raw));

    V6NAXBwdQParamsHost params{};
    params.qL = qL;
    params.kL = kL;
    params.gqa_factor = Hq / Hk;
    params.NQ = (qL + BQ - 1) / BQ;
    params.NK = (kL + BK - 1) / BK;
    params.qL_rem = qL % BQ;
    params.kL_rem = kL % BK;
    params.qL_off = 0;  // v2.50 Sprint 4 — standalone forward/backward (decode would set this nonzero)
    // BHND strides: Q is [B, Hq, qL, D] -> (Hq*qL*D, qL*D, D).  Same for K/V/O/dO/dQ.
    params.Q_strides[0] = (int64_t)Hq * qL * head_dim;
    params.Q_strides[1] = (int64_t)qL * head_dim;
    params.Q_strides[2] = (int64_t)head_dim;
    params.K_strides[0] = (int64_t)Hk * kL * head_dim;
    params.K_strides[1] = (int64_t)kL * head_dim;
    params.K_strides[2] = (int64_t)head_dim;
    params.V_strides[0] = (int64_t)Hk * kL * head_dim;
    params.V_strides[1] = (int64_t)kL * head_dim;
    params.V_strides[2] = (int64_t)head_dim;
    params.O_strides[0] = (int64_t)Hq * qL * head_dim;
    params.O_strides[1] = (int64_t)qL * head_dim;
    params.O_strides[2] = (int64_t)head_dim;
    // lse [B, Hq, qL] FP32: (Hq*qL, qL, 1)
    params.L_strides[0] = (int64_t)Hq * qL;
    params.L_strides[1] = (int64_t)qL;
    params.L_strides[2] = (int64_t)1;
    // dO and dQ are same shape as Q
    params.dO_strides[0] = (int64_t)Hq * qL * head_dim;
    params.dO_strides[1] = (int64_t)qL * head_dim;
    params.dO_strides[2] = (int64_t)head_dim;
    params.dQ_strides[0] = (int64_t)Hq * qL * head_dim;
    params.dQ_strides[1] = (int64_t)qL * head_dim;
    params.dQ_strides[2] = (int64_t)head_dim;
    // v2.38.1: D is contiguous [B, Hq, qL] FP32 (precomputed via MLX).
    params.D_strides[0] = (int64_t)Hq * qL;
    params.D_strides[1] = (int64_t)qL;
    params.D_strides[2] = (int64_t)1;

    enc.set_bytes(params, 7);

    uint32_t NQ = (qL + BQ - 1) / BQ;
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)NQ, (size_t)Hq, (size_t)batchDimension),
        MTL::Size::Make((size_t)32 * (size_t)WM, 1, 1));
  }
}

}  // namespace mlx_mfa

// =============================================================================
// V6NAX backward dK/dV dispatcher (Phase 2 of V6NAX backward Option β sprint).
// Buffers: Q=0, K=1, V=2, O=3, L=4, dO=5, dK=6, dV=7, params=8.
// Grid (NK, H, B), TG size 32 (single SG per TG).
// =============================================================================

namespace mlx_mfa {

struct V6NAXBwdKVParamsHost {
  int qL, kL;
  int gqa_factor;
  int NQ, NK;
  int qL_rem, kL_rem;
  // v2.50 Sprint 4 Phase 4b — causal offset (mirrors device-side V6NAX*Params struct).
  int qL_off;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
  int64_t L_strides[3], dO_strides[3], dK_strides[3], dV_strides[3];
  int64_t D_strides[3];  // v2.38.1: D=rowsum(dO⊙O) strides (FP32, [B,Hq,qL])
};

void v6nax_dispatch_bwd_kv(
    void* pipeline_raw,
    void* enc_raw,
    int qL, int kL,
    int Hq, int Hk,
    int batchDimension,
    int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM) {
  @autoreleasepool {
    auto& enc = *reinterpret_cast<mlx::core::metal::CommandEncoder*>(enc_raw);
    enc.set_compute_pipeline_state(reinterpret_cast<MTL::ComputePipelineState*>(pipeline_raw));

    V6NAXBwdKVParamsHost params{};
    params.qL = qL;
    params.kL = kL;
    params.gqa_factor = Hq / Hk;
    params.NQ = (qL + BQ - 1) / BQ;
    params.NK = (kL + BK - 1) / BK;
    params.qL_rem = qL % BQ;
    params.kL_rem = kL % BK;
    params.qL_off = 0;  // v2.50 Sprint 4 — standalone forward/backward (decode would set this nonzero)
    params.Q_strides[0] = (int64_t)Hq * qL * head_dim;
    params.Q_strides[1] = (int64_t)qL * head_dim;
    params.Q_strides[2] = (int64_t)head_dim;
    params.K_strides[0] = (int64_t)Hk * kL * head_dim;
    params.K_strides[1] = (int64_t)kL * head_dim;
    params.K_strides[2] = (int64_t)head_dim;
    params.V_strides[0] = (int64_t)Hk * kL * head_dim;
    params.V_strides[1] = (int64_t)kL * head_dim;
    params.V_strides[2] = (int64_t)head_dim;
    params.O_strides[0] = (int64_t)Hq * qL * head_dim;
    params.O_strides[1] = (int64_t)qL * head_dim;
    params.O_strides[2] = (int64_t)head_dim;
    params.L_strides[0] = (int64_t)Hq * qL;
    params.L_strides[1] = (int64_t)qL;
    params.L_strides[2] = (int64_t)1;
    params.dO_strides[0] = (int64_t)Hq * qL * head_dim;
    params.dO_strides[1] = (int64_t)qL * head_dim;
    params.dO_strides[2] = (int64_t)head_dim;
    // dK/dV: indexed by Q-head (matches Hq slot count) per DC12 design choice.
    // For non-GQA (Hq==Hk) this equals SDPA-vjp's dK/dV layout.  For GQA
    // (Hq>Hk), caller would sum across Q-heads sharing each KV-head.
    params.dK_strides[0] = (int64_t)Hq * kL * head_dim;
    params.dK_strides[1] = (int64_t)kL * head_dim;
    params.dK_strides[2] = (int64_t)head_dim;
    params.dV_strides[0] = (int64_t)Hq * kL * head_dim;
    params.dV_strides[1] = (int64_t)kL * head_dim;
    params.dV_strides[2] = (int64_t)head_dim;
    // v2.38.1: D is contiguous [B, Hq, qL] FP32 (precomputed via MLX).
    params.D_strides[0] = (int64_t)Hq * qL;
    params.D_strides[1] = (int64_t)qL;
    params.D_strides[2] = (int64_t)1;

    enc.set_bytes(params, 8);

    uint32_t NK = (kL + BK - 1) / BK;
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)NK, (size_t)Hq, (size_t)batchDimension),
        MTL::Size::Make((size_t)32 * (size_t)WM, 1, 1));
  }
}

}  // namespace mlx_mfa

// =============================================================================
// V6NAX backward dV-only dispatcher (Phase 2.O2 — WM=4 Q-row partition).
// Buffers: Q=0, K=1, V=2, L=3, dO=4, dV_partials=5, params=6.
// Grid (NK, Hq, B), TG size 32*WM.
// =============================================================================

namespace mlx_mfa {

struct V6NAXBwdVParamsHost {
  int qL, kL;
  int gqa_factor;
  int NQ, NK;
  int qL_rem, kL_rem;
  // v2.50 Sprint 4 Phase 4b — causal offset (mirrors device-side V6NAX*Params struct).
  int qL_off;
  int64_t Q_strides[3], K_strides[3], V_strides[3];
  int64_t L_strides[3], dO_strides[3];
  int64_t dVp_strides[4];  // [B, Hq, WM, kL] strides; D stride=1 implicit
};

void v6nax_dispatch_bwd_dv(
    void* pipeline_raw,
    void* enc_raw,
    int qL, int kL,
    int Hq, int Hk,
    int batchDimension,
    int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM) {
  @autoreleasepool {
    auto& enc = *reinterpret_cast<mlx::core::metal::CommandEncoder*>(enc_raw);
    enc.set_compute_pipeline_state(reinterpret_cast<MTL::ComputePipelineState*>(pipeline_raw));

    V6NAXBwdVParamsHost params{};
    params.qL = qL;
    params.kL = kL;
    params.gqa_factor = Hq / Hk;
    params.NQ = (qL + BQ - 1) / BQ;
    params.NK = (kL + BK - 1) / BK;
    params.qL_rem = qL % BQ;
    params.kL_rem = kL % BK;
    params.qL_off = 0;  // v2.50 Sprint 4 — standalone forward/backward (decode would set this nonzero)
    params.Q_strides[0] = (int64_t)Hq * qL * head_dim;
    params.Q_strides[1] = (int64_t)qL * head_dim;
    params.Q_strides[2] = (int64_t)head_dim;
    params.K_strides[0] = (int64_t)Hk * kL * head_dim;
    params.K_strides[1] = (int64_t)kL * head_dim;
    params.K_strides[2] = (int64_t)head_dim;
    params.V_strides[0] = (int64_t)Hk * kL * head_dim;
    params.V_strides[1] = (int64_t)kL * head_dim;
    params.V_strides[2] = (int64_t)head_dim;
    params.L_strides[0] = (int64_t)Hq * qL;
    params.L_strides[1] = (int64_t)qL;
    params.L_strides[2] = (int64_t)1;
    params.dO_strides[0] = (int64_t)Hq * qL * head_dim;
    params.dO_strides[1] = (int64_t)qL * head_dim;
    params.dO_strides[2] = (int64_t)head_dim;
    // dV_partials shape: [B, Hq, WM, kL, D] FP32 contiguous.
    // Strides: B*Hq*WM*kL*D, Hq*WM*kL*D, WM*kL*D, kL*D (then D=1 implicit).
    params.dVp_strides[0] = (int64_t)Hq * WM * kL * head_dim;
    params.dVp_strides[1] = (int64_t)WM * kL * head_dim;
    params.dVp_strides[2] = (int64_t)kL * head_dim;
    params.dVp_strides[3] = (int64_t)head_dim;

    enc.set_bytes(params, 6);

    uint32_t NK = (kL + BK - 1) / BK;
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)NK, (size_t)Hq, (size_t)batchDimension),
        MTL::Size::Make((size_t)32 * (size_t)WM, 1, 1));
  }
}

// =============================================================================
// V6NAX backward dV SPARSE dispatcher — Prompt 5b Section A PoC.
// Buffers: Q=0, K=1, V=2, L=3, dO=4, dV_partials=5, params=6, block_mask=7.
// Identical layout to dense dV except adds block_mask at buffer(7).
// =============================================================================

void v6nax_dispatch_bwd_dv_sparse(
    void* pipeline_raw,
    void* enc_raw,
    int qL, int kL,
    int Hq, int Hk,
    int batchDimension,
    int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM) {
  @autoreleasepool {
    auto& enc = *reinterpret_cast<mlx::core::metal::CommandEncoder*>(enc_raw);
    enc.set_compute_pipeline_state(reinterpret_cast<MTL::ComputePipelineState*>(pipeline_raw));

    V6NAXBwdVParamsHost params{};
    params.qL = qL;
    params.kL = kL;
    params.gqa_factor = Hq / Hk;
    params.NQ = (qL + BQ - 1) / BQ;
    params.NK = (kL + BK - 1) / BK;
    params.qL_rem = qL % BQ;
    params.kL_rem = kL % BK;
    params.qL_off = 0;
    params.Q_strides[0] = (int64_t)Hq * qL * head_dim;
    params.Q_strides[1] = (int64_t)qL * head_dim;
    params.Q_strides[2] = (int64_t)head_dim;
    params.K_strides[0] = (int64_t)Hk * kL * head_dim;
    params.K_strides[1] = (int64_t)kL * head_dim;
    params.K_strides[2] = (int64_t)head_dim;
    params.V_strides[0] = (int64_t)Hk * kL * head_dim;
    params.V_strides[1] = (int64_t)kL * head_dim;
    params.V_strides[2] = (int64_t)head_dim;
    params.L_strides[0] = (int64_t)Hq * qL;
    params.L_strides[1] = (int64_t)qL;
    params.L_strides[2] = (int64_t)1;
    params.dO_strides[0] = (int64_t)Hq * qL * head_dim;
    params.dO_strides[1] = (int64_t)qL * head_dim;
    params.dO_strides[2] = (int64_t)head_dim;
    params.dVp_strides[0] = (int64_t)Hq * WM * kL * head_dim;
    params.dVp_strides[1] = (int64_t)WM * kL * head_dim;
    params.dVp_strides[2] = (int64_t)kL * head_dim;
    params.dVp_strides[3] = (int64_t)head_dim;

    enc.set_bytes(params, 6);
    // block_mask buffer is set by the Primitive caller via enc.set_input_array
    // before invoking this dispatcher (binding(7) = block_mask).

    uint32_t NK = (kL + BK - 1) / BK;
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)NK, (size_t)Hq, (size_t)batchDimension),
        MTL::Size::Make((size_t)32 * (size_t)WM, 1, 1));
  }
}

}  // namespace mlx_mfa

// =============================================================================
// V6NAX backward dK-only dispatcher (Phase 2.O2 sister to dV dispatcher).
// Buffers: Q=0, K=1, V=2, O=3, L=4, dO=5, dK_partials=6, params=7.
// =============================================================================

namespace mlx_mfa {

struct V6NAXBwdKParamsHost {
  int qL, kL;
  int gqa_factor;
  int NQ, NK;
  int qL_rem, kL_rem;
  // v2.50 Sprint 4 Phase 4b — causal offset (mirrors device-side V6NAX*Params struct).
  int qL_off;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
  int64_t L_strides[3], dO_strides[3];
  int64_t dKp_strides[4];
  int64_t D_strides[3];  // v2.38.1: D=rowsum(dO⊙O) strides (FP32, [B,Hq,qL])
};

void v6nax_dispatch_bwd_dk(
    void* pipeline_raw,
    void* enc_raw,
    int qL, int kL,
    int Hq, int Hk,
    int batchDimension,
    int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM) {
  @autoreleasepool {
    auto& enc = *reinterpret_cast<mlx::core::metal::CommandEncoder*>(enc_raw);
    enc.set_compute_pipeline_state(reinterpret_cast<MTL::ComputePipelineState*>(pipeline_raw));

    V6NAXBwdKParamsHost params{};
    params.qL = qL;
    params.kL = kL;
    params.gqa_factor = Hq / Hk;
    params.NQ = (qL + BQ - 1) / BQ;
    params.NK = (kL + BK - 1) / BK;
    params.qL_rem = qL % BQ;
    params.kL_rem = kL % BK;
    params.qL_off = 0;  // v2.50 Sprint 4 — standalone forward/backward (decode would set this nonzero)
    params.Q_strides[0] = (int64_t)Hq * qL * head_dim;
    params.Q_strides[1] = (int64_t)qL * head_dim;
    params.Q_strides[2] = (int64_t)head_dim;
    params.K_strides[0] = (int64_t)Hk * kL * head_dim;
    params.K_strides[1] = (int64_t)kL * head_dim;
    params.K_strides[2] = (int64_t)head_dim;
    params.V_strides[0] = (int64_t)Hk * kL * head_dim;
    params.V_strides[1] = (int64_t)kL * head_dim;
    params.V_strides[2] = (int64_t)head_dim;
    params.O_strides[0] = (int64_t)Hq * qL * head_dim;
    params.O_strides[1] = (int64_t)qL * head_dim;
    params.O_strides[2] = (int64_t)head_dim;
    params.L_strides[0] = (int64_t)Hq * qL;
    params.L_strides[1] = (int64_t)qL;
    params.L_strides[2] = (int64_t)1;
    params.dO_strides[0] = (int64_t)Hq * qL * head_dim;
    params.dO_strides[1] = (int64_t)qL * head_dim;
    params.dO_strides[2] = (int64_t)head_dim;
    params.dKp_strides[0] = (int64_t)Hq * WM * kL * head_dim;
    params.dKp_strides[1] = (int64_t)WM * kL * head_dim;
    params.dKp_strides[2] = (int64_t)kL * head_dim;
    params.dKp_strides[3] = (int64_t)head_dim;
    // v2.38.1: D is contiguous [B, Hq, qL] FP32 (precomputed via MLX).
    params.D_strides[0] = (int64_t)Hq * qL;
    params.D_strides[1] = (int64_t)qL;
    params.D_strides[2] = (int64_t)1;

    enc.set_bytes(params, 7);

    uint32_t NK = (kL + BK - 1) / BK;
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)NK, (size_t)Hq, (size_t)batchDimension),
        MTL::Size::Make((size_t)32 * (size_t)WM, 1, 1));
  }
}

}  // namespace mlx_mfa


// =============================================================================
// V6NAX backward FUSED dK+dV dispatcher (Sprint v2.39.0 Phase C.1.a, Option γ).
// Buffers: Q=0, K=1, V=2, L=3, dO=4, dK_partials=5, dV_partials=6, params=7, D=8.
// Grid (NK, Hq, B), TG size 32*WM.  Per-SG-slot outputs reduced via mx.sum(axis=2).
// =============================================================================

namespace mlx_mfa {

struct V6NAXBwdFusedParamsHost {
  int qL, kL;
  int gqa_factor;
  int NQ, NK;
  int qL_rem, kL_rem;
  // v2.50 Sprint 4 Phase 4b — causal offset (mirrors device-side V6NAX*Params struct).
  int qL_off;
  int64_t Q_strides[3], K_strides[3], V_strides[3];
  int64_t L_strides[3], dO_strides[3];
  int64_t dKp_strides[4];
  int64_t dVp_strides[4];
  int64_t D_strides[3];  // v2.38.1: D=rowsum(dO⊙O), [B,Hq,qL]
};

void v6nax_dispatch_bwd_fused_dkdv(
    void* pipeline_raw,
    void* enc_raw,
    int qL, int kL,
    int Hq, int Hk,
    int batchDimension,
    int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM) {
  @autoreleasepool {
    auto& enc = *reinterpret_cast<mlx::core::metal::CommandEncoder*>(enc_raw);
    enc.set_compute_pipeline_state(reinterpret_cast<MTL::ComputePipelineState*>(pipeline_raw));

    V6NAXBwdFusedParamsHost params{};
    params.qL = qL;
    params.kL = kL;
    params.gqa_factor = Hq / Hk;
    params.NQ = (qL + BQ - 1) / BQ;
    params.NK = (kL + BK - 1) / BK;
    params.qL_rem = qL % BQ;
    params.kL_rem = kL % BK;
    params.qL_off = 0;  // v2.50 Sprint 4 — standalone forward/backward (decode would set this nonzero)
    params.Q_strides[0] = (int64_t)Hq * qL * head_dim;
    params.Q_strides[1] = (int64_t)qL * head_dim;
    params.Q_strides[2] = (int64_t)head_dim;
    params.K_strides[0] = (int64_t)Hk * kL * head_dim;
    params.K_strides[1] = (int64_t)kL * head_dim;
    params.K_strides[2] = (int64_t)head_dim;
    params.V_strides[0] = (int64_t)Hk * kL * head_dim;
    params.V_strides[1] = (int64_t)kL * head_dim;
    params.V_strides[2] = (int64_t)head_dim;
    params.L_strides[0] = (int64_t)Hq * qL;
    params.L_strides[1] = (int64_t)qL;
    params.L_strides[2] = (int64_t)1;
    params.dO_strides[0] = (int64_t)Hq * qL * head_dim;
    params.dO_strides[1] = (int64_t)qL * head_dim;
    params.dO_strides[2] = (int64_t)head_dim;
    // dK_partials + dV_partials shape: [B, Hq, WM, kL, D] FP32.
    params.dKp_strides[0] = (int64_t)Hq * WM * kL * head_dim;
    params.dKp_strides[1] = (int64_t)WM * kL * head_dim;
    params.dKp_strides[2] = (int64_t)kL * head_dim;
    params.dKp_strides[3] = (int64_t)head_dim;
    params.dVp_strides[0] = (int64_t)Hq * WM * kL * head_dim;
    params.dVp_strides[1] = (int64_t)WM * kL * head_dim;
    params.dVp_strides[2] = (int64_t)kL * head_dim;
    params.dVp_strides[3] = (int64_t)head_dim;
    // v2.38.1: D contiguous [B, Hq, qL] FP32.
    params.D_strides[0] = (int64_t)Hq * qL;
    params.D_strides[1] = (int64_t)qL;
    params.D_strides[2] = (int64_t)1;

    enc.set_bytes(params, 7);

    uint32_t NK = (kL + BK - 1) / BK;
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)NK, (size_t)Hq, (size_t)batchDimension),
        MTL::Size::Make((size_t)32 * (size_t)WM, 1, 1));
  }
}

}  // namespace mlx_mfa


// ============================================================================
// v2.50 Prompt 5d Section A — Sparse dispatch functions.
// Each mirrors its dense counterpart; block_mask buffer(9) is set by the
// Primitive's eval_gpu before calling these.  Function names suffixed
// with _sparse.
// ============================================================================

namespace mlx_mfa {

void v6nax_dispatch_bwd_query_sparse(
    void* pipeline_raw,
    void* enc_raw,
    int qL, int kL,
    int Hq, int Hk,
    int batchDimension,
    int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM) {
  @autoreleasepool {
    auto& enc = *reinterpret_cast<mlx::core::metal::CommandEncoder*>(enc_raw);
    enc.set_compute_pipeline_state(reinterpret_cast<MTL::ComputePipelineState*>(pipeline_raw));

    V6NAXBwdQParamsHost params{};
    params.qL = qL;
    params.kL = kL;
    params.gqa_factor = Hq / Hk;
    params.NQ = (qL + BQ - 1) / BQ;
    params.NK = (kL + BK - 1) / BK;
    params.qL_rem = qL % BQ;
    params.kL_rem = kL % BK;
    params.qL_off = 0;
    params.Q_strides[0] = (int64_t)Hq * qL * head_dim;
    params.Q_strides[1] = (int64_t)qL * head_dim;
    params.Q_strides[2] = (int64_t)head_dim;
    params.K_strides[0] = (int64_t)Hk * kL * head_dim;
    params.K_strides[1] = (int64_t)kL * head_dim;
    params.K_strides[2] = (int64_t)head_dim;
    params.V_strides[0] = (int64_t)Hk * kL * head_dim;
    params.V_strides[1] = (int64_t)kL * head_dim;
    params.V_strides[2] = (int64_t)head_dim;
    params.O_strides[0] = (int64_t)Hq * qL * head_dim;
    params.O_strides[1] = (int64_t)qL * head_dim;
    params.O_strides[2] = (int64_t)head_dim;
    params.L_strides[0] = (int64_t)Hq * qL;
    params.L_strides[1] = (int64_t)qL;
    params.L_strides[2] = (int64_t)1;
    params.dO_strides[0] = (int64_t)Hq * qL * head_dim;
    params.dO_strides[1] = (int64_t)qL * head_dim;
    params.dO_strides[2] = (int64_t)head_dim;
    params.dQ_strides[0] = (int64_t)Hq * qL * head_dim;
    params.dQ_strides[1] = (int64_t)qL * head_dim;
    params.dQ_strides[2] = (int64_t)head_dim;
    params.D_strides[0] = (int64_t)Hq * qL;
    params.D_strides[1] = (int64_t)qL;
    params.D_strides[2] = (int64_t)1;

    enc.set_bytes(params, 7);

    uint32_t NQ = (qL + BQ - 1) / BQ;
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)NQ, (size_t)Hq, (size_t)batchDimension),
        MTL::Size::Make((size_t)32 * (size_t)WM, 1, 1));
  }
}

void v6nax_dispatch_bwd_dk_sparse(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM) {
  @autoreleasepool {
    auto& enc = *reinterpret_cast<mlx::core::metal::CommandEncoder*>(enc_raw);
    enc.set_compute_pipeline_state(reinterpret_cast<MTL::ComputePipelineState*>(pipeline_raw));

    V6NAXBwdKParamsHost params{};
    params.qL = qL;
    params.kL = kL;
    params.gqa_factor = Hq / Hk;
    params.NQ = (qL + BQ - 1) / BQ;
    params.NK = (kL + BK - 1) / BK;
    params.qL_rem = qL % BQ;
    params.kL_rem = kL % BK;
    params.qL_off = 0;
    params.Q_strides[0] = (int64_t)Hq * qL * head_dim;
    params.Q_strides[1] = (int64_t)qL * head_dim;
    params.Q_strides[2] = (int64_t)head_dim;
    params.K_strides[0] = (int64_t)Hk * kL * head_dim;
    params.K_strides[1] = (int64_t)kL * head_dim;
    params.K_strides[2] = (int64_t)head_dim;
    params.V_strides[0] = (int64_t)Hk * kL * head_dim;
    params.V_strides[1] = (int64_t)kL * head_dim;
    params.V_strides[2] = (int64_t)head_dim;
    params.O_strides[0] = (int64_t)Hq * qL * head_dim;
    params.O_strides[1] = (int64_t)qL * head_dim;
    params.O_strides[2] = (int64_t)head_dim;
    params.L_strides[0] = (int64_t)Hq * qL;
    params.L_strides[1] = (int64_t)qL;
    params.L_strides[2] = (int64_t)1;
    params.dO_strides[0] = (int64_t)Hq * qL * head_dim;
    params.dO_strides[1] = (int64_t)qL * head_dim;
    params.dO_strides[2] = (int64_t)head_dim;
    params.dKp_strides[0] = (int64_t)Hq * WM * kL * head_dim;
    params.dKp_strides[1] = (int64_t)WM * kL * head_dim;
    params.dKp_strides[2] = (int64_t)kL * head_dim;
    params.dKp_strides[3] = (int64_t)head_dim;
    params.D_strides[0] = (int64_t)Hq * qL;
    params.D_strides[1] = (int64_t)qL;
    params.D_strides[2] = (int64_t)1;

    enc.set_bytes(params, 7);

    uint32_t NK = (kL + BK - 1) / BK;
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)NK, (size_t)Hq, (size_t)batchDimension),
        MTL::Size::Make((size_t)32 * (size_t)WM, 1, 1));
  }
}

void v6nax_dispatch_bwd_fused_dkdv_sparse(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM) {
  @autoreleasepool {
    auto& enc = *reinterpret_cast<mlx::core::metal::CommandEncoder*>(enc_raw);
    enc.set_compute_pipeline_state(reinterpret_cast<MTL::ComputePipelineState*>(pipeline_raw));

    V6NAXBwdFusedParamsHost params{};
    params.qL = qL;
    params.kL = kL;
    params.gqa_factor = Hq / Hk;
    params.NQ = (qL + BQ - 1) / BQ;
    params.NK = (kL + BK - 1) / BK;
    params.qL_rem = qL % BQ;
    params.kL_rem = kL % BK;
    params.qL_off = 0;
    params.Q_strides[0] = (int64_t)Hq * qL * head_dim;
    params.Q_strides[1] = (int64_t)qL * head_dim;
    params.Q_strides[2] = (int64_t)head_dim;
    params.K_strides[0] = (int64_t)Hk * kL * head_dim;
    params.K_strides[1] = (int64_t)kL * head_dim;
    params.K_strides[2] = (int64_t)head_dim;
    params.V_strides[0] = (int64_t)Hk * kL * head_dim;
    params.V_strides[1] = (int64_t)kL * head_dim;
    params.V_strides[2] = (int64_t)head_dim;
    params.L_strides[0] = (int64_t)Hq * qL;
    params.L_strides[1] = (int64_t)qL;
    params.L_strides[2] = (int64_t)1;
    params.dO_strides[0] = (int64_t)Hq * qL * head_dim;
    params.dO_strides[1] = (int64_t)qL * head_dim;
    params.dO_strides[2] = (int64_t)head_dim;
    params.dKp_strides[0] = (int64_t)Hq * WM * kL * head_dim;
    params.dKp_strides[1] = (int64_t)WM * kL * head_dim;
    params.dKp_strides[2] = (int64_t)kL * head_dim;
    params.dKp_strides[3] = (int64_t)head_dim;
    params.dVp_strides[0] = (int64_t)Hq * WM * kL * head_dim;
    params.dVp_strides[1] = (int64_t)WM * kL * head_dim;
    params.dVp_strides[2] = (int64_t)kL * head_dim;
    params.dVp_strides[3] = (int64_t)head_dim;
    params.D_strides[0] = (int64_t)Hq * qL;
    params.D_strides[1] = (int64_t)qL;
    params.D_strides[2] = (int64_t)1;

    enc.set_bytes(params, 7);

    uint32_t NK = (kL + BK - 1) / BK;
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)NK, (size_t)Hq, (size_t)batchDimension),
        MTL::Size::Make((size_t)32 * (size_t)WM, 1, 1));
  }
}

}  // namespace mlx_mfa
