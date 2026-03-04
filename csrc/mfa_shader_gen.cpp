/// mfa_shader_gen.cpp  –  Maps ShaderCache::KernelKey → Metal shader source.
///
/// Bridges the MLX-side KernelKey to the ccv-derived AttentionKernel shader
/// generator.  Replicates ccv's AttentionDescriptor::kernelDescriptor() logic
/// but without a live MTL::Device* (we encode device generation in KernelKey).

#include "mfa_shader_gen.hpp"
#include "shader_cache.hpp"

#include "mfa/AttentionKernel.hpp"
#include "mfa/AttentionKernelDescriptor.hpp"
#include "mfa/AttentionKernelType.hpp"
#include "mfa/AttentionOperand.hpp"
#include "mfa/GEMMOperandPrecision.hpp"

#include <simd/simd.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx_mfa {

// ---------------------------------------------------------------------------
// Internal: blocking-table lookup
// ---------------------------------------------------------------------------

/// A single row from ccv's AttentionParameterRow.
struct BlockRow {
  unsigned short max_head_dim;     // inclusive upper bound on head_dim
  unsigned short parallelization;  // block_q
  unsigned short traversal;        // block_k
  unsigned short head;             // block_d (head sub-tile)
  std::vector<int> cached_operands; // AttentionOperand::value members to cache
};

/// Pick the correct forward blocking row for the given head dimension.
/// Mirrors ccv's AttentionDescriptor::forward() / forwardMixed() + row().
static BlockRow forward_block_row(int head_dim, bool is_m3_plus, bool mixed) {
  // Tables from ccv AttentionDescriptor.cpp (AttentionDescriptor::forward /
  // forwardMixed).  Last entry covers all remaining head dims (<= 384).
  if (mixed) {
    if (is_m3_plus) {
      // forwardMixed / M3+
      const BlockRow table[] = {
        { 32,  16, 128, 16, {AttentionOperand::Q, AttentionOperand::O} },
        { 96,  16, 128, 32, {AttentionOperand::Q, AttentionOperand::O} },
        { 160, 16, 128, 32, {AttentionOperand::O} },
        { 224, 16, 128, 32, {AttentionOperand::Q} },
        { 384, 16, 128, 32, {} },
      };
      for (auto& r : table) { if (head_dim <= r.max_head_dim) return r; }
      return table[4];
    } else {
      // forwardMixed / M1+M2
      const BlockRow table[] = {
        { 96,  32, 128, 32, {AttentionOperand::Q, AttentionOperand::O} },
        { 128, 32, 128, 32, {AttentionOperand::Q} },
        { 384, 32, 128, 32, {} },
      };
      for (auto& r : table) { if (head_dim <= r.max_head_dim) return r; }
      return table[2];
    }
  } else {
    if (is_m3_plus) {
      // forward / M3+
      const BlockRow table[] = {
        {   8, 16, 128, 16, {AttentionOperand::Q, AttentionOperand::O} },
        {  16, 16,  64, 16, {AttentionOperand::Q, AttentionOperand::O} },
        {  48, 16,  32,  8, {AttentionOperand::Q, AttentionOperand::O} },
        { 192, 16,  64, 16, {AttentionOperand::O} },
        { 384, 16, 128, 16, {} },
      };
      for (auto& r : table) { if (head_dim <= r.max_head_dim) return r; }
      return table[4];
    } else {
      // forward / M1+M2
      const BlockRow table[] = {
        {  24, 32,  64, 24, {AttentionOperand::Q, AttentionOperand::O} },
        {  32, 32,  64, 32, {AttentionOperand::O} },
        {  56, 32,  32, 56, {AttentionOperand::Q} },
        { 384, 32,  80, 16, {} },
      };
      for (auto& r : table) { if (head_dim <= r.max_head_dim) return r; }
      return table[3];
    }
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

MFABlockConfig resolve_block_config(
    int head_dim, bool is_m3_plus, bool low_prec_inter, bool low_prec_inputs) {
  bool mixed = low_prec_inputs && low_prec_inter;
  BlockRow row = forward_block_row(head_dim, is_m3_plus, mixed);
  // Clamp head block to the padded head dimension.
  unsigned short hd   = (unsigned short)head_dim;
  unsigned short padH = (hd + 7) / 8 * 8;
  unsigned short head = row.head;
  if (head > padH) head = padH;
  return MFABlockConfig{row.parallelization, row.traversal, head};
}

std::string generate_attention_source(const ShaderCache::KernelKey& key) {
  using KT = ShaderCache::KernelKey::KernelType;

  const bool low_prec_inputs = (key.dtype != 2); // FP16 or BF16 input
  const bool is_bf16         = (key.dtype == 1);
  const bool is_m3_plus      = key.is_m3_plus;

  // lowPrecisionIntermediates: true when inputs are f16/bf16.
  // Selects the 'forwardMixed' blocking table (block_d 32 vs 16 for M1/M2),
  // halving inner loop iterations and enabling native f16 simdgroup_matrix GEMMs.
  // For f32 inputs (low_prec_inputs=false) this stays false → 'forward' table.
  const bool low_prec_inter = low_prec_inputs;

  // ---- Block dimensions ----
  BlockRow row = forward_block_row(key.head_dim, is_m3_plus,
                                   low_prec_inputs && low_prec_inter);
  // Override with key values if they were explicitly set (non-zero).
  // For phase 1 we let eval_gpu populate block_q/k/d from the table above.
  unsigned short par  = (key.block_q > 0) ? (unsigned short)key.block_q
                                           : row.parallelization;
  unsigned short trav = (key.block_k > 0) ? (unsigned short)key.block_k
                                           : row.traversal;
  unsigned short head = (key.block_d > 0) ? (unsigned short)key.block_d
                                           : row.head;
  // Clamp head block to head dimension
  unsigned short hd   = (unsigned short)key.head_dim;
  unsigned short padH = (hd + 7) / 8 * 8;
  if (head > padH) head = padH;

  simd::ushort3 blockDims = { par, trav, head };

  // ---- Memory precisions ----
  AttentionOperands<GEMMOperandPrecision> memP;
  if (low_prec_inputs) {
    auto prec = is_bf16 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16;
    memP[AttentionOperand::Q] = prec;
    memP[AttentionOperand::K] = prec;
    memP[AttentionOperand::V] = prec;
  } else {
    memP[AttentionOperand::Q] = GEMMOperandPrecision::FP32;
    memP[AttentionOperand::K] = GEMMOperandPrecision::FP32;
    memP[AttentionOperand::V] = GEMMOperandPrecision::FP32;
  }
  // O is written in the same precision as inputs; accumulation (regP) stays FP32.
  // Metal's simdgroup_matrix store() handles FP32-reg → FP16/BF16-mem conversion.
  if (low_prec_inputs) {
    auto prec = is_bf16 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16;
    memP[AttentionOperand::O] = prec;
  } else {
    memP[AttentionOperand::O] = GEMMOperandPrecision::FP32;
  }
  // L (logsumexp for backward) is always FP32.
  memP[AttentionOperand::L] = GEMMOperandPrecision::FP32;
  // D (delta = rowsum(O⊙dO), passed precomputed) is always FP32.
  memP[AttentionOperand::D] = GEMMOperandPrecision::FP32;
  // S and P memory precisions are not written to memory (register-only).
  memP[AttentionOperand::S] = GEMMOperandPrecision::FP32;
  memP[AttentionOperand::P] = GEMMOperandPrecision::FP32;
  // Backward gradient buffers share dtype with the forward operands they mirror.
  // dO has the same memory layout as O; dQ/dK/dV match Q/K/V respectively.
  {
    auto prec = (low_prec_inputs)
        ? (is_bf16 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16)
        : GEMMOperandPrecision::FP32;
    memP[AttentionOperand::dO] = prec;
    memP[AttentionOperand::dQ] = prec;
    memP[AttentionOperand::dK] = prec;
    memP[AttentionOperand::dV] = prec;
  }

  // ---- Register precisions ----
  AttentionOperands<GEMMOperandPrecision> regP;
  if (low_prec_inputs) {
    auto prec = is_bf16 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16;
    regP[AttentionOperand::Q] = prec;
    regP[AttentionOperand::K] = prec;
    regP[AttentionOperand::V] = prec;
  } else {
    regP[AttentionOperand::Q] = GEMMOperandPrecision::FP32;
    regP[AttentionOperand::K] = GEMMOperandPrecision::FP32;
    regP[AttentionOperand::V] = GEMMOperandPrecision::FP32;
  }
  // S/P: accumulate in FP32 always — even when low_prec_inter=true (forwardMixed path).
  // With low_prec_inputs=true, regP[Q/K/V]=FP16, and Metal dispatches the fast
  // f16×f16→f32 MAD instruction (hardware mixed-precision GEMM on Apple Silicon).
  // Setting regP[S]=FP32 is the CORRECT forwardMixed semantic: load/store in f16,
  // accumulate in f32. The forwardMixed blocking table (block_d=32 for M1/M2) is
  // calibrated for this case, halving inner loop iterations vs the forward table.
  regP[AttentionOperand::S] = GEMMOperandPrecision::FP32;
  regP[AttentionOperand::P] = GEMMOperandPrecision::FP32;
  // O register is FP32 for forward pass.
  regP[AttentionOperand::O] = GEMMOperandPrecision::FP32;
  regP[AttentionOperand::L] = GEMMOperandPrecision::FP32;
  regP[AttentionOperand::D] = GEMMOperandPrecision::FP32;
  // dO is loaded in input dtype; dQ/dK/dV are accumulated in FP32 then
  // downcast to input dtype on write (same as O in the forward pass).
  {
    auto prec = (low_prec_inputs)
        ? (is_bf16 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16)
        : GEMMOperandPrecision::FP32;
    regP[AttentionOperand::dO] = prec;
  }
  regP[AttentionOperand::dQ] = GEMMOperandPrecision::FP32;
  regP[AttentionOperand::dK] = GEMMOperandPrecision::FP32;
  regP[AttentionOperand::dV] = GEMMOperandPrecision::FP32;
  // dP and dS are register-only intermediates in backward (analogous to S/P in forward).
  // registerName(dP) is called by outerProduct(dO,V,dP) and outerProduct(V,dO,dP).
  regP[AttentionOperand::dP] = GEMMOperandPrecision::FP32;
  regP[AttentionOperand::dS] = GEMMOperandPrecision::FP32;

  // ---- Cache state ----
  AttentionOperands<bool> cacheState;
  // Forward operands: not cached by default (row.cached_operands overrides below).
  cacheState[AttentionOperand::Q] = false;
  cacheState[AttentionOperand::K] = false;  // needed: createSetup(backwardKV), outerProduct(K,Q,S)
  cacheState[AttentionOperand::V] = false;  // needed: createSetup(backwardKV), outerProduct(V,dO,dP)
  cacheState[AttentionOperand::O] = false;
  // Backward grad operands: not cached (row has cached_operands={} for all D on M1/M2).
  cacheState[AttentionOperand::dO] = false;
  cacheState[AttentionOperand::dQ] = false;
  cacheState[AttentionOperand::dK] = false;
  cacheState[AttentionOperand::dV] = false;
  // Apply cached_operands from the blocking table row.
  for (int op_val : row.cached_operands) {
    AttentionOperand op;
    op.value = static_cast<AttentionOperand::Value>(op_val);
    cacheState[op] = true;
  }

  // ---- Transpose state ----
  // false for all: MLX BHND arrays are row-major [seq, headDim] within each head.
  // Head offset (headDim * seqLen) is always added unconditionally in
  // operandLocationWithHeadOffsetValue, so transposeState only affects the inner
  // GEMM leading dimension (headDim) and apply_offset access pattern (row-major).
  AttentionOperands<bool> transposeState;
  transposeState[AttentionOperand::Q] = false;
  transposeState[AttentionOperand::K] = false;
  transposeState[AttentionOperand::V] = false;
  transposeState[AttentionOperand::O] = false;
  // Backward operands (not used for forward, but consistent).
  transposeState[AttentionOperand::dO] = false;
  transposeState[AttentionOperand::dQ] = false;
  transposeState[AttentionOperand::dK] = false;
  transposeState[AttentionOperand::dV] = false;

  // ---- Leading dimensions ----
  // All false: MLX arrays are contiguous, standard stride = headDimension.
  AttentionOperands<bool> leadingDims;

  // ---- Async preferences ----
  // M3+ uses preferAsyncCache=true (one-to-one thread-to-element reads).
  // M1/M2 uses preferAsyncLoad=true (shared reads, async DMA).
  bool preferAsyncCache = is_m3_plus;
  bool preferAsyncLoad  = !is_m3_plus;

  // ---- Kernel type ----
  AttentionKernelType ktype;
  switch (key.type) {
    case KT::AttentionForward:
      ktype = AttentionKernelType::forward;
      break;
    case KT::AttentionBackwardDQ:
      ktype = AttentionKernelType::backwardQuery;
      break;
    case KT::AttentionBackwardDKV:
      ktype = AttentionKernelType::backwardKeyValue;
      break;
  }

  // ---- Build descriptor and generate source ----
  AttentionKernelDescriptor desc(
      blockDims,
      cacheState,
      hd,
      memP,
      preferAsyncCache,
      preferAsyncLoad,
      regP,
      transposeState,
      leadingDims,
      ktype);

  AttentionKernel kernel(desc);
  return kernel.source;
}

} // namespace mlx_mfa
