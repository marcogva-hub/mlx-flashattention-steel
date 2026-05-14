// Derived from Draw Things's NAAttentionKernel.cpp
// Licensed under BSD 3-Clause. See LICENSE-DRAWTHINGS at repo root.
//
// mlx-mfa adaptations:
//   - #include adjusted: ccv_nnc_mfa.hpp/error.hpp -> mfa_compat.h
//   - Constructor no longer creates an MTL::Library (source-only)
//   - threadgroupMemoryAllocation/threadgroupSize do not need a pipelineState
//
#include "NAAttentionKernel.hpp"
#include "../GEMMHeaders.hpp"
#include "../CodeWriter.hpp"
#include "../mfa_compat.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace {

uint32_t ceil_log2_u32_host(uint32_t x) {
  if (x <= 1)
    return 0;
  x -= 1;
  uint32_t bits = 0;
  while (x > 0) {
    x >>= 1;
    ++bits;
  }
  return bits;
}

}

NAAttentionKernel::NAAttentionKernel(NAAttentionKernelDescriptor descriptor) {
  type = descriptor.type;
  memoryPrecisions = descriptor.memoryPrecisions;
  blockDimensions = descriptor.blockDimensions;
  headDimension = descriptor.headDimension;
  Hq = descriptor.Hq;
  Hk = descriptor.Hk;
  executionSIMDGroups = descriptor.executionSIMDGroups;
  checkCEdge1 = descriptor.checkCEdge1;
  scale = descriptor.scale;
  bypassThreadgroupMemory = descriptor.bypassThreadgroupMemory;
  isCausal = descriptor.isCausal;
  masked = descriptor.masked;
  isVarlen = descriptor.isVarlen;
  singleOtileMode = descriptor.singleOtileMode;
  useV34 = descriptor.useV34;

  // mlx-mfa: produce MSL 4 source string only. mlx-mfa's shader cache
  // performs the actual MTL::Library / pipeline state creation.
  source = createSource();
}

// MARK: - NAAttentionKernel

unsigned short NAAttentionKernel::threadgroupMemoryAllocation() const noexcept {
  if (type.value == AttentionKernelType::forward) {
    // Sprint A.1 (v2.30) — single-Otile + bypass forward kernel never uses
    // P_buf threadgroup memory (cP cooperative_tensor takes its place).
    // Allocating ~BQ*BK*SG*sizeof(O) anyway burns 8-16KB of the 32KB
    // threadgroup budget for nothing, halving threadgroup co-residency on
    // M5+ (tgmem-limited occupancy floor). Skip when both flags imply zero
    // tgmem need.
    if (singleOtileMode && bypassThreadgroupMemory) {
      return 0;
    }
    unsigned short threadgroupMemoryAllocation = blockDimensions[0] * blockDimensions[1] * executionSIMDGroups * memoryPrecisions[AttentionOperand::O].value().size();
    return threadgroupMemoryAllocation;
  }
  if (type.value == AttentionKernelType::backwardQuery &&
      memoryPrecisions[AttentionOperand::Q].value() != GEMMOperandPrecision::FP32 &&
      !bypassThreadgroupMemory) {
    return headDimension * blockDimensions[0] * executionSIMDGroups *
        memoryPrecisions[AttentionOperand::Q].value().size() * 2;
  }
  if (type.value == AttentionKernelType::backwardKeyValue &&
      memoryPrecisions[AttentionOperand::Q].value() != GEMMOperandPrecision::FP32 &&
      !bypassThreadgroupMemory) {
    return headDimension * blockDimensions[0] * executionSIMDGroups *
        memoryPrecisions[AttentionOperand::K].value().size() * 2;
  }
  return 0;
}

/// The number of threads per group. On Apple Silicon
/// pipelineState->threadExecutionWidth() is always 32, so we hardcode it.
uint16_t NAAttentionKernel::threadgroupSize() const noexcept {
  return 32 * executionSIMDGroups;
}

// threadgroupsPerGrid is computed by the dispatch site (mlx-mfa) — it has
// access to the runtime matrix dimensions and batch.
// (Removed from this file; the caller in mfa_attention.cpp calculates it.)

std::string NAAttentionKernel::memoryName(AttentionOperand operand) const noexcept {
  auto value = memoryPrecisions[operand];
  return value.value().name();
}

std::string NAAttentionKernel::sequenceLength(AttentionOperand operand) const noexcept {
  switch (operand.value) {
  case AttentionOperand::Q:
  case AttentionOperand::dQ:
    return "R";
  case AttentionOperand::K:
  case AttentionOperand::dK:
    return "C";
  case AttentionOperand::V:
  case AttentionOperand::dV:
    return "C";
  case AttentionOperand::O:
  case AttentionOperand::dO:
    return "R";
  default:
    CCV_NNC_MFA_PRECONDITION(false);
  }
  return "";
}

unsigned short NAAttentionKernel::blockSequenceLength(AttentionOperand operand) const noexcept {
  switch (type.value) {
  case AttentionKernelType::forward:
  case AttentionKernelType::backwardQuery:
    switch (operand.value) {
    case AttentionOperand::Q:
    case AttentionOperand::dQ:
      return blockDimensions[0];
    case AttentionOperand::K:
    case AttentionOperand::dK:
      return blockDimensions[1];
    case AttentionOperand::V:
    case AttentionOperand::dV:
      return blockDimensions[1];
    case AttentionOperand::O:
    case AttentionOperand::dO:
      return blockDimensions[0];
    default:
      CCV_NNC_MFA_PRECONDITION(false);
    }

  case AttentionKernelType::backwardKeyValue:
    switch (operand.value) {
    case AttentionOperand::Q:
    case AttentionOperand::dQ:
      return blockDimensions[1];
    case AttentionOperand::K:
    case AttentionOperand::dK:
      return blockDimensions[0];
    case AttentionOperand::V:
    case AttentionOperand::dV:
      return blockDimensions[0];
    case AttentionOperand::O:
    case AttentionOperand::dO:
      return blockDimensions[1];
    default:
      CCV_NNC_MFA_PRECONDITION(false);
    }
  }
  CCV_NNC_MFA_PRECONDITION(false);
  return 0;
}

// MARK: - NAAttentionKernel+Source

std::string NAAttentionKernel::createSource() const noexcept {
  // V34 path: emit a self-contained Apple-style attention_nax kernel. Skips
  // legacy createConstants/createBufferBindings/loopForward — the V34 source
  // has its own kernel signature, params struct, and BHND-correct addressing.
  // v2.50 Sprint 4 Phase 4a: V34 forward now supports causal masking
  // (Apple steel_attention_nax.h:176-187,279-301 pattern, kb_lim shrink +
  // per-element causal mask).  Block mask + varlen still excluded — those
  // route to legacy STEEL.
  if (useV34 && type.value == AttentionKernelType::forward && !masked && !isVarlen) {
    return createV34Source();
  }

  CodeWriter source;
  const bool lowPrecisionInputs =
      memoryPrecisions[AttentionOperand::Q].value() != GEMMOperandPrecision::FP32;
  const bool usesThreadgroupBlock =
      type.value == AttentionKernelType::forward ||
      (type.value == AttentionKernelType::backwardQuery &&
       lowPrecisionInputs && !bypassThreadgroupMemory) ||
      (type.value == AttentionKernelType::backwardKeyValue &&
       lowPrecisionInputs && !bypassThreadgroupMemory);

  // Inject the contents of the headers.
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

)";

  createConstants(source);

  if (type.value == AttentionKernelType::forward && masked) {
    source.SetValue("MEMORY_NAME_Q", memoryName(AttentionOperand::Q));
    source += R"(
kernel void generate_attention_block_mask(
    device const {{MEMORY_NAME_Q}} *Mask_buf [[buffer(15)]],
    device uchar *Block_mask_buf [[buffer(16)]],
    threadgroup uint *block_mask_scratch [[threadgroup(0)]],
    ushort tid [[thread_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
  const uint q_start = tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  const uint c_start = tgid.y * {{BLOCK_DIMENSIONS_TRAVERSAL}};
  const uint q_extent = min((uint){{BLOCK_DIMENSIONS_PARALLELIZATION}}, R - q_start);
  const uint c_extent = min((uint){{BLOCK_DIMENSIONS_TRAVERSAL}}, C - c_start);
  const uint element_count = q_extent * c_extent;
  uint all_zero = 1;
  uint all_masked = 1;
  for (uint i = tid; i < element_count; i += {{BLOCK_MASK_THREADS}}) {
    const uint row = q_start + i / c_extent;
    const uint column = c_start + i - (i / c_extent) * c_extent;
)";
    if (isCausal) {
      source += R"(
    if (int(column) > int(row) + int(C) - int(R)) {
      continue;
    }
)";
    }
    source += R"(
    const float value = (float)Mask_buf[tgid.z * Mask_batch_stride + row * C + column];
    if (value != 0.0f) {
      all_zero = 0;
    }
    if (!(value <= {{MASKED_THRESHOLD}})) {
      all_masked = 0;
    }
  }
  threadgroup uint *zero_scratch = block_mask_scratch;
  threadgroup uint *masked_scratch = block_mask_scratch + {{BLOCK_MASK_THREADS}};
  zero_scratch[tid] = all_zero;
  masked_scratch[tid] = all_masked;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint tile_all_zero = 1;
    uint tile_all_masked = 1;
    for (uint i = 0; i < {{BLOCK_MASK_THREADS}}; ++i) {
      tile_all_zero &= zero_scratch[i];
      tile_all_masked &= masked_scratch[i];
    }
    const uchar flag = tile_all_masked ? 0 : (tile_all_zero ? 1 : 2);
    Block_mask_buf[tgid.z * Block_mask_batch_stride + tgid.x * K_block_tiles + tgid.y] = flag;
  }
}

)";
  }

  if (type.value == AttentionKernelType::backwardQuery) {
    source += createComputeD();
  }

  source += R"(
    
    // Declare the function.
    kernel void attention(
)";
  source += createBufferBindings() + "\n";
  switch (type.value) {
  case AttentionKernelType::forward:
    source.SetValue("DISPATCH_DIMENSION", "R");
    source.SetValue("DISPATCH_HEADS", "Hq");
    break;
  case AttentionKernelType::backwardQuery:
    source.SetValue("DISPATCH_DIMENSION", "R");
    source.SetValue("DISPATCH_HEADS", "Hq");
    break;
  case AttentionKernelType::backwardKeyValue:
    source.SetValue("DISPATCH_DIMENSION", "C");
    source.SetValue("DISPATCH_HEADS", "Hk");
    break;
  }
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  source.SetValue("EXECUTION_SIMD_GROUPS", std::to_string(executionSIMDGroups));
  if (usesThreadgroupBlock) {
    source += R"(
      threadgroup uchar *threadgroup_block [[threadgroup(0)]],
)";
  }
  if (type.value == AttentionKernelType::forward) {
    source += R"(
      ushort sgid [[simdgroup_index_in_threadgroup]],
      uint3 tgid [[threadgroup_position_in_grid]]
    ) {
)";
  } else {
    source += R"(
      ushort tid [[thread_index_in_threadgroup]],
      ushort sgid [[simdgroup_index_in_threadgroup]],
      uint3 tgid [[threadgroup_position_in_grid]]
    ) {
)";
  }
  if (type.value == AttentionKernelType::forward) {
    source += R"(
  const uint row_group_count = ({{DISPATCH_DIMENSION}} + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}} - 1) / ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}});
  const uint row_group_bits = ceil_log2_u32(row_group_count);
  const uint head_bits = ceil_log2_u32({{DISPATCH_HEADS}});
  const uint2 morton_tile = morton_decode_rectangular_2d(tgid.x, row_group_bits, head_bits);
  if (morton_tile.y >= {{DISPATCH_HEADS}} || morton_tile.x >= row_group_count) {
    return;
  }
  tgid = uint3(morton_tile.x, morton_tile.y, tgid.z);
)";
    source += R"(
  tgid.x = tgid.x * {{EXECUTION_SIMD_GROUPS}} + sgid;
  if (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= {{DISPATCH_DIMENSION}}) {
    return;
  }
)";
  } else {
    source += R"(
  const uint linear_group = tgid.x;
  const uint row_group_count = ({{DISPATCH_DIMENSION}} + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}} - 1) / ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{EXECUTION_SIMD_GROUPS}});
  const uint row_group = )";
  if (type.value == AttentionKernelType::backwardKeyValue) {
    source += "linear_group % row_group_count";
  } else {
    source += "(linear_group / {{DISPATCH_HEADS}}) % row_group_count";
  }
  source += R"(;
  const uint head = )";
  if (type.value == AttentionKernelType::backwardKeyValue) {
    source += "(linear_group / row_group_count) % {{DISPATCH_HEADS}}";
  } else {
    source += "linear_group % {{DISPATCH_HEADS}}";
  }
  source += R"(;
  const uint batch = linear_group / ({{DISPATCH_HEADS}} * row_group_count);
  tgid = uint3(row_group, head, batch);
  tgid.x = row_group * {{EXECUTION_SIMD_GROUPS}} + sgid;
  if (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= {{DISPATCH_DIMENSION}}) {
    return;
  }
)";
  }
  source += createAdjustOffsets() + "\n";
  switch (type.value) {
  case AttentionKernelType::forward:
    loopForward(source);
    break;
  case AttentionKernelType::backwardQuery:
    loopBackwardQuery(source);
    break;
  case AttentionKernelType::backwardKeyValue:
    loopBackwardKeyValue(source);
    break;
  }
  source += "}\n";

  return source.ToString();
}

void NAAttentionKernel::createConstants(CodeWriter &source) const noexcept {
  source += R"(

// R = row dimension (output sequence)
// C = column dimension (input sequence)
// Hq = number of query heads.
constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];

)";
  std::vector<AttentionOperand> operands;
  switch (type.value) {
  case AttentionKernelType::forward:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O};
    break;
  case AttentionKernelType::backwardQuery:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::dO, AttentionOperand::dQ};
    break;
  case AttentionKernelType::backwardKeyValue:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::dO, AttentionOperand::dV, AttentionOperand::dK};
    break;
  }
  source.SetValue("HQ", std::to_string(Hq));
  source.SetValue("HK", std::to_string(Hk));
  source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_2", std::to_string(blockDimensions[1] * 2));
  source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
  source.SetValue("BLOCK_MASK_THREADS", std::to_string(blockMaskThreads));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  std::ostringstream maskedThreshold;
  maskedThreshold << std::setprecision(std::numeric_limits<float>::max_digits10)
      << ((memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::FP16) ?
          (-65504.0f * 0.5f) :
          (-std::numeric_limits<float>::max() * 0.5f));
  source.SetValue("MASKED_THRESHOLD", maskedThreshold.str());
  source += R"(
constant uint Hq = {{HQ}};
constant uint Hk = {{HK}};
)";
  if (type.value == AttentionKernelType::forward) {
    if (!isVarlen) {
      if (isCausal || masked) {
        source += R"(
constant uint C_single_remainder = C % {{BLOCK_DIMENSIONS_TRAVERSAL}};
constant uint C_single_edge = C >= {{BLOCK_DIMENSIONS_TRAVERSAL}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL}} : 0;
)";
      } else {
        source += R"(
// In this special case, leaving the rest to the trailing block to process.
constant uint C_remainder = (C % {{BLOCK_DIMENSIONS_TRAVERSAL_2}}) == {{BLOCK_DIMENSIONS_TRAVERSAL}} ? {{BLOCK_DIMENSIONS_TRAVERSAL}} : (C % {{BLOCK_DIMENSIONS_TRAVERSAL}});
)";
        if (checkCEdge1) {
          source += R"(
constant uint C_edge = C >= {{BLOCK_DIMENSIONS_TRAVERSAL}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL}} : 0;
constant uint C_edge_1 = C >= {{BLOCK_DIMENSIONS_TRAVERSAL_2}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL_2}} : 0;
)";
        } else {
          // When we are not checking C_edge, C_edge makes sure we process entire blockDimensions.C * 2 block, rather than one of.
          // And leaving the rest to the C_remainder path.
          source += R"(
constant uint C_edge = C >= {{BLOCK_DIMENSIONS_TRAVERSAL_2}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL_2}} : 0;
)";
        }
      }
      source += R"(
constant uint R_edge = R >= {{BLOCK_DIMENSIONS_PARALLELIZATION}} ? R + 1 - {{BLOCK_DIMENSIONS_PARALLELIZATION}} : 0;
constant uint R_remainder = R % {{BLOCK_DIMENSIONS_PARALLELIZATION}};
constant uint K_edge = {{HEAD_DIMENSION}} + 1 - {{BLOCK_DIMENSIONS_HEAD}};
)";
    }
    if (isVarlen) {
      source += R"(
constant uint K_edge = {{HEAD_DIMENSION}} + 1 - {{BLOCK_DIMENSIONS_HEAD}};
)";
    }
  } else if (type.value == AttentionKernelType::backwardQuery) {
    source += R"(
constant uint C_remainder = C % {{BLOCK_DIMENSIONS_TRAVERSAL}};
constant uint C_edge = C >= {{BLOCK_DIMENSIONS_TRAVERSAL}} ? C + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL}} : 0;
constant uint R_edge = R >= {{BLOCK_DIMENSIONS_PARALLELIZATION}} ? R + 1 - {{BLOCK_DIMENSIONS_PARALLELIZATION}} : 0;
constant uint R_remainder = R % {{BLOCK_DIMENSIONS_PARALLELIZATION}};
)";
  } else if (type.value == AttentionKernelType::backwardKeyValue) {
    source += R"(
constant uint KV_R_edge = R >= {{BLOCK_DIMENSIONS_TRAVERSAL}} ? R + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL}} : 0;
constant uint KV_R_remainder = R % {{BLOCK_DIMENSIONS_TRAVERSAL}};
constant uint KV_C_edge = C >= {{BLOCK_DIMENSIONS_PARALLELIZATION}} ? C + 1 - {{BLOCK_DIMENSIONS_PARALLELIZATION}} : 0;
constant uint KV_C_remainder = C % {{BLOCK_DIMENSIONS_PARALLELIZATION}};
)";
  }
  source += R"(
constant uint K_Hq = {{HEAD_DIMENSION}} * Hq;
constant uint K_Hk = {{HEAD_DIMENSION}} * Hk;
)";
  for (const auto& operand : operands) {
    source.SetValue("OPERAND_NAME", operand.name());
    source.SetValue("OPERAND_BUFFER_INDEX", std::to_string(operand.bufferIndex() + 2));
    source += R"(
constant uint {{OPERAND_NAME}}_batch_stride [[function_constant({{OPERAND_BUFFER_INDEX}})]];
)";
  }
  if (type.value == AttentionKernelType::forward && masked) {
    source += R"(
constant uint Mask_batch_stride [[function_constant(15)]];
constant uint Block_mask_batch_stride [[function_constant(16)]];
constant uint K_block_tiles = (C + {{BLOCK_DIMENSIONS_TRAVERSAL}} - 1) / {{BLOCK_DIMENSIONS_TRAVERSAL}};
)";
  }
  if (type.value == AttentionKernelType::forward) {
    source += R"(

inline uint compact_morton_even_bits(uint x) {
  x &= 0x55555555u;
  x = (x | (x >> 1)) & 0x33333333u;
  x = (x | (x >> 2)) & 0x0f0f0f0fu;
  x = (x | (x >> 4)) & 0x00ff00ffu;
  x = (x | (x >> 8)) & 0x0000ffffu;
  return x;
}

inline uint2 morton_decode_2d(uint code) {
  return uint2(compact_morton_even_bits(code),
               compact_morton_even_bits(code >> 1));
}

inline uint lower_bits_mask(uint bit_count) {
  if (bit_count == 0)
    return 0;
  return (1u << bit_count) - 1;
}

inline uint2 morton_decode_rectangular_2d(uint code,
                                          uint x_bits,
                                          uint y_bits) {
  const uint paired_bits = min(x_bits, y_bits);
  const uint paired_code = code & lower_bits_mask(paired_bits * 2);
  uint2 tile = morton_decode_2d(paired_code);
  uint tail = code >> (paired_bits * 2);
  if (x_bits > paired_bits) {
    const uint x_extra_bits = x_bits - paired_bits;
    tile.x |= (tail & lower_bits_mask(x_extra_bits)) << paired_bits;
    tail >>= x_extra_bits;
  }
  if (y_bits > paired_bits) {
    tile.y |= tail << paired_bits;
  }
  return tile;
}

inline uint ceil_log2_u32(uint x) {
  if (x <= 1)
    return 0;
  x -= 1;
  uint bits = 0;
  while (x > 0) {
    x >>= 1;
    ++bits;
  }
  return bits;
}
)";
  }
}

std::string NAAttentionKernel::createBufferBindings() const noexcept {
  std::vector<AttentionOperand> operands;
  switch (type.value) {
  case AttentionKernelType::forward:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::L};
    break;
  case AttentionKernelType::backwardQuery:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::L, AttentionOperand::D, AttentionOperand::dO, AttentionOperand::dQ};
    break;
  case AttentionKernelType::backwardKeyValue:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::L, AttentionOperand::D, AttentionOperand::dO, AttentionOperand::dV, AttentionOperand::dK};
    break;
  }
  std::string output = "";
  for (const auto& operand : operands) {
    output += "  device ";
    output += memoryName(operand);
    output += "* " + operand.name() + "_buf [[buffer(";
    output += std::to_string(operand.bufferIndex()) + ")]],\n";
  }
  if (type.value == AttentionKernelType::forward && masked) {
    output += "  device const ";
    output += memoryName(AttentionOperand::Q);
    output += "* Mask_buf [[buffer(15)]],\n";
    output += "  device const uchar* Block_mask_buf [[buffer(16)]],\n";
  }
  if (type.value == AttentionKernelType::forward && isVarlen) {
    output += "  device const int* QSeqOffsets_buf [[buffer(17)]],\n";
    output += "  device const int* KVSeqOffsets_buf [[buffer(18)]],\n";
  }
  return output;
}

std::string NAAttentionKernel::operandLocationWithHeadOffsetValue(AttentionOperand operand) const noexcept {
  CodeWriter source;
  source.SetValue("OPERAND", operand.name());
  if (operand.value == AttentionOperand::L || operand.value == AttentionOperand::D) {
    source += "{{OPERAND}}_buf + (tgid.z * Hq + tgid.y) * R\\";
  } else {
    source += "{{OPERAND}}_buf + tgid.z * {{OPERAND}}_batch_stride\\";
  }
  return source.ToString();
}

std::string NAAttentionKernel::createAdjustOffsets() const noexcept {
  std::vector<AttentionOperand> operands;
  switch (type.value) {
  case AttentionKernelType::forward:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::O, AttentionOperand::L};
    break;
  case AttentionKernelType::backwardQuery:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::L, AttentionOperand::D, AttentionOperand::dO, AttentionOperand::dQ};
    break;
  case AttentionKernelType::backwardKeyValue:
    operands = {AttentionOperand::Q, AttentionOperand::K, AttentionOperand::V, AttentionOperand::L, AttentionOperand::D, AttentionOperand::dO, AttentionOperand::dV, AttentionOperand::dK};
    break;
  }
  CodeWriter source;
  if (type.value == AttentionKernelType::forward && isVarlen) {
    source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
    source += R"(
  const uint q_start = uint(QSeqOffsets_buf[tgid.z]);
  const uint q_end = uint(QSeqOffsets_buf[tgid.z + 1]);
  const uint kv_start = uint(KVSeqOffsets_buf[tgid.z]);
  const uint kv_end = uint(KVSeqOffsets_buf[tgid.z + 1]);
  const uint R_seq = q_end - q_start;
  const uint C_seq = kv_end - kv_start;
  if (tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_seq) {
    return;
  }
  const uint C_single_remainder_seq = C_seq % {{BLOCK_DIMENSIONS_TRAVERSAL}};
  const uint C_single_edge_seq = C_seq >= {{BLOCK_DIMENSIONS_TRAVERSAL}} ? C_seq + 1 - {{BLOCK_DIMENSIONS_TRAVERSAL}} : 0;
  const uint R_edge_seq = R_seq >= {{BLOCK_DIMENSIONS_PARALLELIZATION}} ? R_seq + 1 - {{BLOCK_DIMENSIONS_PARALLELIZATION}} : 0;
  const uint R_remainder_seq = R_seq % {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  Q_buf = Q_buf + q_start * K_Hq;
  K_buf = K_buf + kv_start * K_Hk;
  V_buf = V_buf + kv_start * K_Hk;
  O_buf = O_buf + q_start * K_Hq;
  L_buf = L_buf + (tgid.z * Hq + tgid.y) * R;
)";
    return source.ToString();
  }
  for (const auto& operand : operands) {
    source.SetValue("OPERAND", operand.name());
    source.SetValue("OPERAND_LOCATION", operandLocationWithHeadOffsetValue(operand));
      source += R"(
  {{OPERAND}}_buf = {{OPERAND_LOCATION}};
)";
  }
  if (type.value == AttentionKernelType::forward && masked) {
    source += R"(
  Mask_buf += tgid.z * Mask_batch_stride;
  Block_mask_buf += tgid.z * Block_mask_batch_stride;
)";
  }
  return source.ToString();
}

// MARK: - Outer Loop

// Forward
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     (m, l, P) = softmax(m, l, S * scaleFactor)
//
//     O *= correction
//     load V[c]
//     O += P * V
//   }
//   O /= l
//
//   L = m + logBaseE(l)
//
// Backward Query
//   D = dO * O
//
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     P = exp(S - L)
//
//     load V[c]
//     dP = dO * V^T
//     dS = P * (dP - D) * scaleFactor
//
//     load K[c]
//     dQ += dS * K
//   }
//
// Backward Key-Value
//   for r in 0..<R {
//     load Q[r]
//     load L[r]
//     S^T = K * Q^T
//     P^T = exp(S^T - L)
//
//     load dO[r]
//     dV += P^T * dO
//
//     load dO[r]
//     load D[r]
//     dP^T = V * dO^T
//     dS^T = P^T * (dP^T - D) * scaleFactor
//
//     load Q[r]
//     dK += dS^T * Q
//   }

static std::string high_precision_to_string(float value) {
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  return oss.str();
}

static std::string dotProductScale(float rsqrtD, bool derivative) {
  float logBase2E = 1.442695041;

  if (!derivative) {
    return high_precision_to_string(logBase2E * rsqrtD);
  } else {
    return high_precision_to_string(rsqrtD);
  }
}

static std::string naxHelpersBlock() {
  // Apple NAX cooperative-tensor primitives — verbatim from
  // ~/code/mlx-source/mlx/backend/metal/kernels/steel/{defines.h,
  // utils/type_traits.h, utils/integral_constant.h} + the
  // steel_attention_nax.h Op structs and NAXFrag wrappers.
  //
  // Shared across all V34 source generators (forward, bwd dQ, bwd
  // dV, bwd dK, legacy fused dK/dV).  Sprint v2.38.x consolidation
  // per Sprint 2 audit M1-HIGH-01 + M3-HIGH-01.  Pre-consolidation:
  // 5x verbatim ~390-line duplication = ~1950 LOC of identical MSL
  // emitted as 5 separate raw-string literals.  Post-consolidation:
  // 1 shared definition + 5 one-line invocations.
  //
  // The content is byte-identical to the prior inline blocks (verified
  // pre-refactor via sed-extract + pairwise diff; all empty).
  return R"NAX_HELPERS(
// === defines.h ===
#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// === utils/type_traits.h (subset) ===
#pragma METAL internals : enable
namespace metal {
template <typename T> struct is_empty : metal::bool_constant<__is_empty(T)> {};
template <typename T> struct pointer_element {};
template <typename T> struct pointer_element<thread T*> { using type = remove_cv_t<T>; };
template <typename T> struct pointer_element<device T*> { using type = remove_cv_t<T>; };
template <typename T> struct pointer_element<constant T*> { using type = remove_cv_t<T>; };
template <typename T> struct pointer_element<threadgroup T*> { using type = remove_cv_t<T>; };
template <typename T> using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;
}
#pragma METAL internals : disable

// === utils/integral_constant.h (subset) ===
#pragma METAL internals : enable
namespace mlx { namespace steel {
template <typename T, T v> struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;
  METAL_FUNC constexpr operator value_type() const noexcept { return value; }
};
template <bool B> using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;
template <int val> using Int = integral_constant<int, val>;
#define integral_const_binop(__op__, __operator__)          \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto __operator__(                   \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv __op__ uv;                      \
    return integral_constant<decltype(res), res>{};         \
  }
integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);
template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    constexpr auto idx = Int<start>{};
    f(idx);
    const_for_loop<start + step, stop, step, F>(f);
  }
}
#undef integral_const_binop
}}
#pragma METAL internals : disable

// === Limits<float/half/bfloat> (Apple kernels/utils.h:55-70) ===
template <typename U> struct Limits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
  static const constant U finite_max = metal::numeric_limits<U>::max();
  static const constant U finite_min = metal::numeric_limits<U>::min();
};
template <> struct Limits<float> {
  static constexpr constant float max = metal::numeric_limits<float>::infinity();
  static constexpr constant float min = -metal::numeric_limits<float>::infinity();
  static constexpr constant float finite_max = metal::numeric_limits<float>::max();
  static constexpr constant float finite_min = -metal::numeric_limits<float>::max();
};

// === Apple steel/attn/nax.h — BaseNAXFrag + NAXTile (verbatim, nax.h:27-817) ===
namespace mlx { namespace steel {

struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;
  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;
  STEEL_CONST short kElemRows = 2;
  STEEL_CONST short kElemCols = 4;
  STEEL_CONST short kElemRowsJump = 8;

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
    return short2{fn, fm};
  }

  template <typename T, typename SrcPtrType, typename StrX, typename StrY,
            typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst, SrcPtrType src,
      StrX str_x, StrY str_y, OffX off_x = {}, OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j) * str_y]);
        }
      }
    }
  }

  template <typename T, typename SrcPtrType, typename StrX, typename StrY,
            typename LimX, typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst, SrcPtrType src,
      StrX str_x, StrY str_y, LimX lim_x, OffX off_x = {}, OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j)]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j) * str_y]);
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <typename T, typename DstPtrType, typename StrX, typename StrY,
            typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src, DstPtrType dst,
      StrX str_x, StrY str_y, OffX off_x = {}, OffY off_y = {}) {
    using U = metal::pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + (c + j) * str_y] = static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <typename T, typename DstPtrType, typename StrX, typename StrY,
            typename LimX, typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_rows(
      const thread dtype_frag_t<T>& src, DstPtrType dst,
      StrX str_x, StrY str_y, LimX lim_x, OffX off_x = {}, OffY off_y = {}) {
    using U = metal::pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + (c + j) * str_y] = static_cast<U>(src[i * kElemCols + j]);
          }
        }
      }
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals, thread T* reduced_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      T thr_reduce = Op::apply(
          Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
          Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));
      T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = Op::apply(thr_reduce, qgr_reduce);
      T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);
      reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_bin_op(
      thread dtype_frag_t<T>& inp_vals, thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] = Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }

  template <typename CType, typename AType, typename BType,
            bool transpose_a = false, bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cn0, thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A, metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0, const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, transpose_a, transpose_b, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), CType>();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) ct_a[i] = A[i];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = Bn0[i];
      ct_b[kElemsPerFrag + i] = Bn1[i];
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cn0[i];
      ct_c[kElemsPerFrag + i] = Cn1[i];
    }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cn0[i] = ct_c[i];
      Cn1[i] = ct_c[kElemsPerFrag + i];
    }
  }
};

template <typename T, short kTileRows_, short kTileCols_, class NAXFrag_ = BaseNAXFrag>
struct NAXTile {
  using NAXFrag_t = NAXFrag_;
  using elem_type = T;
  STEEL_CONST short kFragRows = NAXFrag_t::kFragRows;
  STEEL_CONST short kFragCols = NAXFrag_t::kFragCols;
  STEEL_CONST short kElemsPerFrag = NAXFrag_t::kElemsPerFrag;
  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;
  STEEL_CONST short kRows = kTileRows * kFragRows;
  STEEL_CONST short kCols = kTileCols * kFragCols;
  STEEL_CONST short kNumFrags = kTileRows * kTileCols;
  STEEL_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;
  STEEL_CONST short kFragThrRows = NAXFrag_t::kElemRows;
  STEEL_CONST short kFragThrCols = NAXFrag_t::kElemCols;
  STEEL_CONST short kFragRowsJump = NAXFrag_t::kElemRowsJump;
  STEEL_CONST short kRowsPerThread = kTileRows * NAXFrag_t::kElemRows;
  STEEL_CONST short kColsPerThread = kTileCols * NAXFrag_t::kElemCols;

  typedef typename NAXFrag_t::template dtype_frag_t<T> frag_type;
  frag_type val_frags[kNumFrags];

  METAL_FUNC NAXTile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) val_frags[i] = frag_type(0);
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }
  METAL_FUNC constexpr const thread frag_type& frag_at(const short i, const short j) const {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread metal::vec<T, kRowsPerThread>& vals) const {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_reduce<Op>(frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread metal::vec<T, kRowsPerThread>& vals) {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_bin_op<Op>(frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(frag_at(idx_row.value, idx_col.value), src, ld, Int<1>{},
                        idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(frag_at(idx_row.value, idx_col.value), dst, ld, Int<1>{},
                         idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load_rows(const device U* src, const int ld, const short n_rows) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_rows(frag_at(idx_row.value, idx_col.value), src, ld, Int<1>{},
                             n_rows, idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_rows(device U* dst, const int ld, const short n_rows) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_rows(frag_at(idx_row.value, idx_col.value), dst, ld, Int<1>{},
                              n_rows, idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }
};

}}  // namespace mlx::steel

// === Operator structs (steel_attention_nax.h:31-71) ===
struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};
struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};
struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x * y; }
};
struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); }
};
)NAX_HELPERS";
}


std::string NAAttentionKernel::createComputeD() const noexcept {
  CodeWriter source;
  source.SetValue("MEMORY_NAME_O", memoryName(AttentionOperand::O));
  source.SetValue("MEMORY_NAME_DO", memoryName(AttentionOperand::dO));
  source.SetValue("MEMORY_NAME_D", memoryName(AttentionOperand::D));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("COMPUTE_D_THREADS", std::to_string(computeDThreads));
  source.SetValue("DOT_SCALE_DERIVATIVE", dotProductScale(scale, true));
  source += R"(

kernel void compute_d(
    device const {{MEMORY_NAME_O}}* O_buf [[buffer(3)]],
    device const {{MEMORY_NAME_DO}}* dO_buf [[buffer(6)]],
    device {{MEMORY_NAME_D}}* D_buf [[buffer(5)]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const uint row = tgid.x % R;
  const uint head = tgid.x / R;
  O_buf += tgid.z * O_batch_stride;
  dO_buf += tgid.z * dO_batch_stride;
  D_buf += (tgid.z * Hq + head) * R;

  const uint offset = row * K_Hq + head * {{HEAD_DIMENSION}};
  float D_accumulator = 0;
  for (uint d = lane_id; d < {{HEAD_DIMENSION}}; d += {{COMPUTE_D_THREADS}}) {
    D_accumulator += (float)O_buf[offset + d] * (float)dO_buf[offset + d];
  }
  D_accumulator += simd_shuffle_xor(D_accumulator, 16);
  D_accumulator += simd_shuffle_xor(D_accumulator, 8);
  D_accumulator += simd_shuffle_xor(D_accumulator, 4);
  D_accumulator += simd_shuffle_xor(D_accumulator, 2);
  D_accumulator += simd_shuffle_xor(D_accumulator, 1);
  if (lane_id == 0) {
    D_buf[row] = ({{MEMORY_NAME_D}})(D_accumulator * {{DOT_SCALE_DERIVATIVE}});
  }
}

)";
  return source.ToString();
}

void NAAttentionKernel::loopForwardSingleCausal(CodeWriter &source) const noexcept {
  source.SetValue("MEMORY_NAME_Q", memoryName(AttentionOperand::Q));
  source.SetValue("MEMORY_NAME_K", memoryName(AttentionOperand::K));
  source.SetValue("MEMORY_NAME_V", memoryName(AttentionOperand::V));
  source.SetValue("MEMORY_NAME_O", memoryName(AttentionOperand::O));
  source.SetValue("MEMORY_NAME_L", memoryName(AttentionOperand::L));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("HEAD_DIMENSION_REMAINDER", std::to_string(headDimension % blockDimensions[2]));
  if (blockDimensions[1] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[1]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if (blockDimensions[2] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[2]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if ((headDimension % blockDimensions[2]) % 32 == 0) {
    source.SetValue("HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V", std::to_string(headDimension % blockDimensions[2]));
  } else {
    source.SetValue("HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if (Hq != Hk) {
    source.SetValue("H_HK_RATIO", "/ " + std::to_string(Hq / Hk));
  } else {
    source.SetValue("H_HK_RATIO", "");
  }
  source.SetValue("R_LENGTH", isVarlen ? "R_seq" : "R");
  source.SetValue("C_LENGTH", isVarlen ? "C_seq" : "C");
  source.SetValue("R_EDGE", isVarlen ? "R_edge_seq" : "R_edge");
  source.SetValue("R_REMAINDER", isVarlen ? "R_remainder_seq" : "R_remainder");
  source.SetValue("C_SINGLE_EDGE", isVarlen ? "C_single_edge_seq" : "C_single_edge");
  source.SetValue("C_SINGLE_REMAINDER", isVarlen ? "C_single_remainder_seq" : "C_single_remainder");
  source.SetValue("DOT_SCALE", dotProductScale(scale, false));
  source.SetValue("MASK_SCALE", dotProductScale(1.0f, false));
  source += R"(
  auto Q = tensor<device {{MEMORY_NAME_Q}},  dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, {{R_LENGTH}}));
  auto K = tensor<device {{MEMORY_NAME_K}},  dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, {{C_LENGTH}}));
  auto V = tensor<device {{MEMORY_NAME_V}},  dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, {{C_LENGTH}}));
  threadgroup {{MEMORY_NAME_O}} *P_buf = (threadgroup {{MEMORY_NAME_O}}*)threadgroup_block + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{BLOCK_DIMENSIONS_TRAVERSAL}} * sgid;
  auto P = tensor<threadgroup {{MEMORY_NAME_O}}, dextents<int32_t, 2>, tensor_inline>(P_buf, extents<int32_t, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>());
  constexpr auto qk_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> matmul_qk_op;
)";
  if (headDimension % blockDimensions[2] > 0) {
    source += R"(
  constexpr auto qk_desc_remainder = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc_remainder, execution_simdgroups<1>> matmul_qk_op_remainder;
)";
  }
  source += R"(
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
  auto cS_0 = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cM = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cL = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto correction = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
    if (cM.is_valid_element(k)) {
      cM[k] = -numeric_limits<float>::infinity();
      cL[k] = numeric_limits<float>::denorm_min();
    }
  }
  auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(0, 0);
  constexpr auto pv_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
)";
  const unsigned short kBlocks = (std::max(headDimension, blockDimensions[2]) + blockDimensions[2] - 1) / blockDimensions[2];
  if (bypassThreadgroupMemory) {
    source += "  auto cP = matmul_pv_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_O}}, {{MEMORY_NAME_V}}, float>();\n";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<decltype(cP), decltype(mV), float>();\n";
    }
  } else {
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();\n";
    }
  }
  if (isCausal || !isVarlen) {
    source += R"(
  const int causal_row_start = int(tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
)";
  }
  if (isCausal) {
    source += R"(
  const int causal_column_offset = int({{C_LENGTH}}) - int({{R_LENGTH}});
  const int causal_last_column = causal_row_start + int({{BLOCK_DIMENSIONS_PARALLELIZATION}}) - 1 + causal_column_offset;
  const int causal_first_column_limit = causal_row_start + causal_column_offset;
  const uint single_c_edge = causal_last_column < 0 ? 0 : min({{C_SINGLE_EDGE}}, uint(causal_last_column) + 1);
)";
  } else {
    source += R"(
  const uint single_c_edge = {{C_SINGLE_EDGE}};
)";
  }
  source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
    if (cO_0.is_valid_element(k)) {
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "      cO_{{LOOP_INDEX}}[k] = 0;\n";
  }
  source += R"(
    }
  }

  for (uint c = 0; c < single_c_edge; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
)";
  if (masked) {
    source += R"(
    const uchar mask_flags = Block_mask_buf[tgid.x * K_block_tiles + c / {{BLOCK_DIMENSIONS_TRAVERSAL}}];
    if (mask_flags == 0) {
      continue;
    }
)";
  }
  source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        cS_0[k] = 0;
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, c);
      matmul_qk_op.run(mQ, mK_0, cS_0);
    }
)";
  if (headDimension % blockDimensions[2] > 0) {
    source.SetValue("HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER", std::to_string(headDimension - (headDimension % blockDimensions[2])));
    source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, c);
      matmul_qk_op_remainder.run(mQ, mK_0, cS_0);
    }
)";
  }
  if (masked) {
    if (isCausal) {
      source += R"(
    const bool causal_mask_0 = int(c + {{BLOCK_DIMENSIONS_TRAVERSAL}} - 1) > causal_first_column_limit;
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        const int row = causal_row_start + idx[1];
        const int column = int(c) + idx[0];
        float score = cS_0[k] * {{DOT_SCALE}};
        if (mask_flags == 2 && row < int({{R_LENGTH}})) {
          score += (float)Mask_buf[row * C + column] * {{MASK_SCALE}};
        }
)";
    if (isCausal) {
      source += R"(
        const int causal_column_limit = row + causal_column_offset;
        if (causal_mask_0 && column > causal_column_limit) {
          score = -numeric_limits<float>::infinity();
        }
)";
    }
    source += R"(
        cS_0[k] = score;
      }
    }
)";
  } else if (isCausal) {
    source += R"(
    const bool causal_mask_0 = int(c + {{BLOCK_DIMENSIONS_TRAVERSAL}} - 1) > causal_first_column_limit;
    if (causal_mask_0) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
        if (cS_0.is_valid_element(k)) {
          auto idx = cS_0.get_multidimensional_index(k);
          const int causal_row = causal_row_start + idx[1];
          const int causal_column_limit = causal_row + causal_column_offset;
          if (int(c) + idx[0] > causal_column_limit) {
            cS_0[k] = -numeric_limits<float>::infinity();
          }
        }
      }
    }
)";
  }
  source += R"(
    // Online reduce maximum.
    auto cM_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cM_0_new, reduction_operation::max, -numeric_limits<float>::infinity());
    // Online correct O.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
)";
  if (masked) {
    source += R"(
        const float M_new = cM_0_new[k];
)";
  } else {
    source += R"(
        const float M_new = cM_0_new[k] * {{DOT_SCALE}};
)";
  }
  source += R"(
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
      }
    }
)";
  if (masked) {
    source += R"(
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        cS_0[k] = fast::exp2(cS_0[k] - *dst_it);
      }
    }
)";
  } else if (isCausal) {
    source += R"(
    // Softmax. cS becomes cP.
    if (causal_mask_0) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
        if (cS_0.is_valid_element(k)) {
          auto it = cS_0.get_iterator(k);
          auto dst_it = cM.map_iterator(it);
          auto idx = cS_0.get_multidimensional_index(k);
          const int causal_row = causal_row_start + idx[1];
          const int causal_column_limit = causal_row + causal_column_offset;
          cS_0[k] = int(c) + idx[0] <= causal_column_limit ?
              fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it) :
              0;
        }
      }
    } else {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
        if (cS_0.is_valid_element(k)) {
          auto it = cS_0.get_iterator(k);
          auto dst_it = cM.map_iterator(it);
          cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
        }
      }
    }
)";
  } else {
    source += R"(
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
      }
    }
)";
  }
  source += R"(
    // Online reduce sum.
    auto cL_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cL_0_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if(cL.is_valid_element(k)) {
        cL[k] = cL[k] * correction[k] + cL_0_new[k];
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = correction.map_iterator(it);
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "        cO_{{LOOP_INDEX}}[k] *= *dst_it;\n";
  }
  source += R"(
      }
    }
)";
  if (bypassThreadgroupMemory) {
    source += R"(
    simdgroup_barrier(mem_flags::mem_none);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if(cS_0.is_valid_element(k)) {
        cP[k] = ({{MEMORY_NAME_O}})cS_0[k];
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    auto mV_0_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    matmul_pv_op.run(cP, mV_0_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
    }
  } else {
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if(cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        P_buf[idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_0[k];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
)";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    auto mV_0_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    matmul_pv_op.run(P, mV_0_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
    }
  }
  source += R"(
  }
)";
  if (isCausal) {
    source += R"(
  if ({{C_SINGLE_REMAINDER}} > 0 && int({{C_LENGTH}} - {{C_SINGLE_REMAINDER}}) <= causal_last_column) {
)";
  } else {
    source += R"(
  if ({{C_SINGLE_REMAINDER}} > 0) {
)";
  }
  if (masked) {
    source += R"(
    const uint c = {{C_LENGTH}} - {{C_SINGLE_REMAINDER}};
    const uchar mask_flags = Block_mask_buf[tgid.x * K_block_tiles + c / {{BLOCK_DIMENSIONS_TRAVERSAL}}];
    if (mask_flags != 0) {
)";
  }
  source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}}) {
          cS_0[k] = -numeric_limits<float>::infinity();
        } else {
          cS_0[k] = 0;
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, {{C_LENGTH}} - {{C_SINGLE_REMAINDER}});
      matmul_qk_op.run(mQ, mK_0, cS_0);
    }
)";
  if (headDimension % blockDimensions[2] > 0) {
    source.SetValue("HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER", std::to_string(headDimension - (headDimension % blockDimensions[2])));
    source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, {{C_LENGTH}} - {{C_SINGLE_REMAINDER}});
      matmul_qk_op_remainder.run(mQ, mK_0, cS_0);
    }
)";
  }
  if (masked) {
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        const int row = causal_row_start + idx[1];
        const int column = int({{C_LENGTH}} - {{C_SINGLE_REMAINDER}}) + idx[0];
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}}) {
          cS_0[k] = -numeric_limits<float>::infinity();
        } else {
          float score = cS_0[k] * {{DOT_SCALE}};
          if (mask_flags == 2 && row < int({{R_LENGTH}})) {
            score += (float)Mask_buf[row * C + column] * {{MASK_SCALE}};
          }
)";
    if (isCausal) {
      source += R"(
          const int causal_column_limit = row + causal_column_offset;
          if (column > causal_column_limit) {
            score = -numeric_limits<float>::infinity();
          }
)";
    }
    source += R"(
          cS_0[k] = score;
        }
      }
    }
)";
  } else if (isCausal) {
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        const int causal_row = causal_row_start + idx[1];
        const int causal_column_limit = causal_row + causal_column_offset;
        if (int({{C_LENGTH}} - {{C_SINGLE_REMAINDER}}) + idx[0] > causal_column_limit) {
          cS_0[k] = -numeric_limits<float>::infinity();
        }
      }
    }
)";
  } else {
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}}) {
          cS_0[k] = -numeric_limits<float>::infinity();
        }
      }
    }
)";
  }
  source += R"(
    // Online reduce maximum.
    auto cM_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cM_0_new, reduction_operation::max, -numeric_limits<float>::infinity());
    // Online correct O.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
)";
  if (masked) {
    source += R"(
        const float M_new = cM_0_new[k];
)";
  } else {
    source += R"(
        const float M_new = cM_0_new[k] * {{DOT_SCALE}};
)";
  }
  source += R"(
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
      }
    }
)";
  if (masked) {
    source += R"(
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}}) {
          cS_0[k] = 0;
        } else {
          cS_0[k] = fast::exp2(cS_0[k] - *dst_it);
        }
      }
    }
)";
  } else if (isCausal) {
    source += R"(
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        auto idx = cS_0.get_multidimensional_index(k);
        const int causal_row = causal_row_start + idx[1];
        const int causal_column_limit = causal_row + causal_column_offset;
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}} || int({{C_LENGTH}} - {{C_SINGLE_REMAINDER}}) + idx[0] > causal_column_limit) {
          cS_0[k] = 0;
        } else {
          cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
        }
      }
    }
)";
  } else {
    source += R"(
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}}) {
          cS_0[k] = 0;
        } else {
          cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
        }
      }
    }
)";
  }
  source += R"(
    // Online reduce sum.
    auto cL_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cL_0_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if(cL.is_valid_element(k)) {
        cL[k] = cL[k] * correction[k] + cL_0_new[k];
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = correction.map_iterator(it);
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "        cO_{{LOOP_INDEX}}[k] *= *dst_it;\n";
  }
  source += R"(
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if(cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int){{C_SINGLE_REMAINDER}}) {
          P_buf[idx[0] - {{C_SINGLE_REMAINDER}} + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = 0;
        } else {
          P_buf[{{BLOCK_DIMENSIONS_TRAVERSAL}} - {{C_SINGLE_REMAINDER}} + idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_0[k];
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto mP = P.slice<dynamic_extent, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{BLOCK_DIMENSIONS_TRAVERSAL}} - {{C_SINGLE_REMAINDER}}, 0);
    constexpr auto pv_remainder_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, dynamic_length_v<int>, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_remainder_desc, execution_simdgroups<1>> matmul_pv_remainder_op;
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    source += R"(
    auto mV_0_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, dynamic_extent>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, {{C_LENGTH}} - {{C_SINGLE_REMAINDER}});
    matmul_pv_remainder_op.run(mP, mV_0_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
  }
  source += R"(
  }
)";
  if (masked) {
    source += R"(
  }
)";
  }
  source += R"(
  auto O = O_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hq) + tgid.y * {{HEAD_DIMENSION}};
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  if ({{R_REMAINDER}} > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= {{R_EDGE}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto idx = cO_0.get_multidimensional_index(k);
        if (idx[1] < (int){{R_REMAINDER}}) {
          auto it = cO_0.get_iterator(k);
          auto dst_it = cL.map_iterator(it);
          auto L_reciprocal = fast::divide(1, *dst_it);
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    if ((i < kBlocks - 1) || (headDimension % blockDimensions[2] == 0)) {
      source += R"(
          O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
)";
    } else {
      source += R"(
          if (idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} < {{HEAD_DIMENSION}}) {
            O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
          }
)";
    }
  }
  source += R"(
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        auto idx = cM.get_multidimensional_index(k);
        if (idx[0] < (int){{R_REMAINDER}}) {
          float L_sram = cM[k] + fast::log2(cL[k]);
          L[idx[0]] = ({{MEMORY_NAME_L}})L_sram;
        }
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = cL.map_iterator(it);
        auto L_reciprocal = fast::divide(1, *dst_it);
        auto idx = cO_0.get_multidimensional_index(k);
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    if ((i < kBlocks - 1) || (headDimension % blockDimensions[2] == 0)) {
      source += R"(
        O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
)";
    } else {
      source += R"(
        if (idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} < {{HEAD_DIMENSION}}) {
          O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
        }
)";
    }
  }
  source += R"(
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        auto idx = cM.get_multidimensional_index(k);
        float L_sram = cM[k] + fast::log2(cL[k]);
        L[idx[0]] = ({{MEMORY_NAME_L}})L_sram;
      }
    }
  }
)";
}

void NAAttentionKernel::loopForward(CodeWriter &source) const noexcept {
  if (isCausal || masked || isVarlen) {
    loopForwardSingleCausal(source);
    return;
  }
  // Sprint 3.3 — Apple-style single-Otile variant for non-causal, non-masked,
  // non-varlen forward path. Emits a kernel with single cS (no double-buffer),
  // forced kBlocks=1, always-bypass cP, mem_none barriers, K-loop step BK.
  if (singleOtileMode) {
    loopForwardSingleTile(source);
    return;
  }
  source.SetValue("MEMORY_NAME_Q", memoryName(AttentionOperand::Q));
  source.SetValue("MEMORY_NAME_K", memoryName(AttentionOperand::K));
  source.SetValue("MEMORY_NAME_V", memoryName(AttentionOperand::V));
  source.SetValue("MEMORY_NAME_O", memoryName(AttentionOperand::O));
  source.SetValue("MEMORY_NAME_L", memoryName(AttentionOperand::L));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("HEAD_DIMENSION_REMAINDER", std::to_string(headDimension % blockDimensions[2]));
  // In OS 26.1, K no longer can be arbitrary number, it has to be multiple of 32. This might / might not be
  // a bug. A workaround is to use dynamic_length_v<int> which will result correct value.
  if (blockDimensions[1] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[1]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if (blockDimensions[2] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[2]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if ((headDimension % blockDimensions[2]) % 32 == 0) {
    source.SetValue("HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V", std::to_string(headDimension % blockDimensions[2]));
  } else {
    source.SetValue("HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if (Hq != Hk) {
  source.SetValue("H_HK_RATIO", "/ " + std::to_string(Hq / Hk));
  } else {
    source.SetValue("H_HK_RATIO", "");
  }
  source += R"(
  auto Q = tensor<device {{MEMORY_NAME_Q}},  dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device {{MEMORY_NAME_K}},  dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
  auto V = tensor<device {{MEMORY_NAME_V}},  dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, C));
  threadgroup {{MEMORY_NAME_O}} *P_buf = (threadgroup {{MEMORY_NAME_O}}*)threadgroup_block + {{BLOCK_DIMENSIONS_PARALLELIZATION}} * {{BLOCK_DIMENSIONS_TRAVERSAL}} * sgid;
  auto P = tensor<threadgroup {{MEMORY_NAME_O}}, dextents<int32_t, 2>, tensor_inline>(P_buf, extents<int32_t, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>());
  constexpr auto qk_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> matmul_qk_op;
)";
  if (headDimension % blockDimensions[2] > 0) {
    source += R"(
  constexpr auto qk_desc_remainder = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc_remainder, execution_simdgroups<1>> matmul_qk_op_remainder;
)";
  }
  source += R"(
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, 0);
  auto cS_0 = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cS_1 = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cM = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cL = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto correction = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
    if (cM.is_valid_element(k)) {
      cM[k] = -numeric_limits<float>::infinity();
      cL[k] = numeric_limits<float>::denorm_min();
    }
  }
  auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(0, 0);
  constexpr auto pv_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
)";
  const unsigned short kBlocks = (std::max(headDimension, blockDimensions[2]) + blockDimensions[2] - 1) / blockDimensions[2];
  if (bypassThreadgroupMemory) {
    source += "  auto cP = matmul_pv_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_O}}, {{MEMORY_NAME_V}}, float>();\n";
    // Allocate O
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<decltype(cP), decltype(mV), float>();\n";
    }
  } else {
    // Allocate O
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cO_{{LOOP_INDEX}} = matmul_pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();\n";
    }
  }
  source += R"(
  for (uint c = 0; c < C_edge; c += {{BLOCK_DIMENSIONS_TRAVERSAL_2}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        cS_0[k] = 0;
)";
  if (checkCEdge1) {
    source += R"(
        if (c < C_edge_1) {
          cS_1[k] = 0;
        } else {
          auto idx = cS_1.get_multidimensional_index(k);
          if (idx[0] >= (int)C_remainder) {
            cS_1[k] = -numeric_limits<float>::infinity();
          } else {
            cS_1[k] = 0;
          }
        }
)";
  } else {
    source += R"(
        cS_1[k] = 0;
)";
  }
  source += R"(
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, c);
      auto mK_1 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, c + {{BLOCK_DIMENSIONS_TRAVERSAL}});
      matmul_qk_op.run(mQ, mK_0, cS_0);
      matmul_qk_op.run(mQ, mK_1, cS_1);
    }
)";
  if (headDimension % blockDimensions[2] > 0) {
    source.SetValue("HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER", std::to_string(headDimension - (headDimension % blockDimensions[2])));
    source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, c);
      auto mK_1 = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, c + {{BLOCK_DIMENSIONS_TRAVERSAL}});
      matmul_qk_op_remainder.run(mQ, mK_0, cS_0);
      matmul_qk_op_remainder.run(mQ, mK_1, cS_1);
    }
)";
  }
  source.SetValue("DOT_SCALE", dotProductScale(scale, false));
  source += R"(
    // Online reduce maximum.
    auto cM_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cM_0_new, reduction_operation::max, -numeric_limits<float>::infinity());
    auto cM_1_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_1, cM_1_new, reduction_operation::max, -numeric_limits<float>::infinity());
    // Online correct O
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
        const float M_new = max(cM_0_new[k], cM_1_new[k]) * {{DOT_SCALE}};
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
      }
    }
)";
  source += R"(
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
)";
  if (checkCEdge1) {
    source += R"(
        if (c < C_edge_1) {
          cS_1[k] = fast::exp2(cS_1[k] * {{DOT_SCALE}} - *dst_it);
        } else {
          auto idx = cS_1.get_multidimensional_index(k);
          if (idx[0] >= (int)C_remainder) {
            cS_1[k] = 0;
          } else {
            cS_1[k] = fast::exp2(cS_1[k] * {{DOT_SCALE}} - *dst_it);
          }
        }
)";
  } else {
    source += R"(
        cS_1[k] = fast::exp2(cS_1[k] * {{DOT_SCALE}} - *dst_it);
)";
  }
  source += R"(
      }
    }
)";
  source += R"(
    // Online reduce sum.
    auto cL_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cL_0_new, reduction_operation::sum, (float)0);
    auto cL_1_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_1, cL_1_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if(cL.is_valid_element(k)) {
        cL[k] = cL[k] * correction[k] + cL_0_new[k] + cL_1_new[k];
      }
    }
    if (c == 0) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
        if (cO_0.is_valid_element(k)) {
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "          cO_{{LOOP_INDEX}}[k] = 0;\n";
  }
  source += R"(
        }
      }
    } else {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
        if (cO_0.is_valid_element(k)) {
          auto it = cO_0.get_iterator(k);
          auto dst_it = correction.map_iterator(it);
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source += "          cO_{{LOOP_INDEX}}[k] *= *dst_it;\n";
  }
  source += R"(
        }
      }
    }
)";
  if (bypassThreadgroupMemory) {
    source += R"(
    simdgroup_barrier(mem_flags::mem_none);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if(cS_0.is_valid_element(k)) {
        cP[k] = ({{MEMORY_NAME_O}})cS_0[k];
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    auto mV_0_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    matmul_pv_op.run(cP, mV_0_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
    }
  } else {
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if(cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        P_buf[idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_0[k];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
)";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    auto mV_0_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
    matmul_pv_op.run(P, mV_0_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
    }
  }
  if (checkCEdge1) {
    if (bypassThreadgroupMemory) {
      source += R"(
    if (c < C_edge_1) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_1.get_capacity(); ++k) {
        if(cS_1.is_valid_element(k)) {
          cP[k] = ({{MEMORY_NAME_O}})cS_1[k];
        }
      }
)";
      for (unsigned short i = 0; i < kBlocks; i++) {
        source.SetValue("LOOP_INDEX", std::to_string(i));
        source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
        source += R"(
      auto mV_1_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c + {{BLOCK_DIMENSIONS_TRAVERSAL}});
      matmul_pv_op.run(cP, mV_1_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
      }
    } else {
      source += R"(
    if (c < C_edge_1) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_1.get_capacity(); ++k) {
        if(cS_1.is_valid_element(k)) {
          auto idx = cS_1.get_multidimensional_index(k);
          P_buf[idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_1[k];
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
)";
      for (unsigned short i = 0; i < kBlocks; i++) {
        source.SetValue("LOOP_INDEX", std::to_string(i));
        source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
        source += R"(
      auto mV_1_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c + {{BLOCK_DIMENSIONS_TRAVERSAL}});
      matmul_pv_op.run(P, mV_1_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
      }
    }
    source += R"(
    } else {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_1.get_capacity(); ++k) {
        if(cS_1.is_valid_element(k)) {
          auto idx = cS_0.get_multidimensional_index(k);
          if (idx[0] >= (int)C_remainder) {
            P_buf[idx[0] - C_remainder + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = 0;
          } else {
            P_buf[{{BLOCK_DIMENSIONS_TRAVERSAL}} - C_remainder + idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_1[k];
          }
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      // The reason to do this is because when K (in GEMM sense) is smaller (in this case, C_remainder is smaller than blockDimensions.C),
      // we need to start a new matmul descriptor with dynamic_extent for that, hence we copied the P_buf in this way and then sliced it.
      auto mP = P.slice<dynamic_extent, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{BLOCK_DIMENSIONS_TRAVERSAL}} - C_remainder, 0);
      constexpr auto pv_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, dynamic_length_v<int>, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
      matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
)";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
      auto mV_1_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, dynamic_extent>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, C - C_remainder);
      matmul_pv_op.run(mP, mV_1_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
    }
    source += R"(
    }
)";
  } else {
    if (bypassThreadgroupMemory) {
      source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_1.get_capacity(); ++k) {
      if(cS_1.is_valid_element(k)) {
        cP[k] = ({{MEMORY_NAME_O}})cS_1[k];
      }
    }
)";
      for (unsigned short i = 0; i < kBlocks; i++) {
        source.SetValue("LOOP_INDEX", std::to_string(i));
        source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
        source += R"(
    auto mV_1_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c + {{BLOCK_DIMENSIONS_TRAVERSAL}});
    matmul_pv_op.run(cP, mV_1_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
      }
    } else {
      source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_1.get_capacity(); ++k) {
      if(cS_1.is_valid_element(k)) {
        auto idx = cS_1.get_multidimensional_index(k);
        P_buf[idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_1[k];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
)";
      for (unsigned short i = 0; i < kBlocks; i++) {
        source.SetValue("LOOP_INDEX", std::to_string(i));
        source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
        source += R"(
    auto mV_1_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c + {{BLOCK_DIMENSIONS_TRAVERSAL}});
    matmul_pv_op.run(P, mV_1_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
      }
    }
  }
  source += R"(
  }
)";
  if (!checkCEdge1) { // Process the remainder path.
    source += R"(
  if (C_remainder > 0) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder) {
          cS_0[k] = -numeric_limits<float>::infinity();
        } else {
          cS_0[k] = 0;
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, C - C_remainder);
      matmul_qk_op.run(mQ, mK_0, cS_0);
    }
)";
    if (headDimension % blockDimensions[2] > 0) {
      source.SetValue("HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER", std::to_string(headDimension - (headDimension % blockDimensions[2])));
      source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_0 = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, C - C_remainder);
      matmul_qk_op_remainder.run(mQ, mK_0, cS_0);
    }
)";
    }
    source += R"(
    // Online reduce maximum.
    auto cM_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cM_0_new, reduction_operation::max, -numeric_limits<float>::infinity());
    // Online correct O
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
        const float M_new = cM_0_new[k] * {{DOT_SCALE}};
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
      }
    }
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        auto idx = cS_0.get_multidimensional_index(k);
)";
    source += R"(
        if (idx[0] >= (int)C_remainder) {
          cS_0[k] = 0;
        } else {
          cS_0[k] = fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it);
        }
      }
    }
    // Online reduce sum.
    auto cL_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cL_0_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if(cL.is_valid_element(k)) {
        cL[k] = cL[k] * correction[k] + cL_0_new[k];
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = correction.map_iterator(it);
)";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "        cO_{{LOOP_INDEX}}[k] *= *dst_it;\n";
    }
    source += R"(
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if(cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder) {
          P_buf[idx[0] - C_remainder + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = 0;
        } else {
          P_buf[{{BLOCK_DIMENSIONS_TRAVERSAL}} - C_remainder + idx[0] + idx[1] * {{BLOCK_DIMENSIONS_TRAVERSAL}}] = ({{MEMORY_NAME_O}})cS_0[k];
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // The reason to do this is because when K (in GEMM sense) is smaller (in this case, C_remainder is smaller than blockDimensions.C),
    // we need to start a new matmul descriptor with dynamic_extent for that, hence we copied the P_buf in this way and then sliced it.
    auto mP = P.slice<dynamic_extent, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{BLOCK_DIMENSIONS_TRAVERSAL}} - C_remainder, 0);
    constexpr auto pv_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, dynamic_length_v<int>, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
)";
    for (unsigned short i = 0; i < kBlocks; i++) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    auto mV_0_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, dynamic_extent>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, C - C_remainder);
    matmul_pv_op.run(mP, mV_0_{{LOOP_INDEX}}, cO_{{LOOP_INDEX}});
)";
    }
    source += R"(
  }
)";
  }
  source += R"(
  auto O = O_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hq) + tgid.y * {{HEAD_DIMENSION}};
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  if (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto idx = cO_0.get_multidimensional_index(k);
        if (idx[1] < (int)R_remainder) {
          auto it = cO_0.get_iterator(k);
          auto dst_it = cL.map_iterator(it);
          auto L_reciprocal = fast::divide(1, *dst_it);
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    if ((i < kBlocks - 1) || (headDimension % blockDimensions[2] == 0)) {
      source += R"(
          O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
)";
    } else {
      source += R"(
          if (idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} < {{HEAD_DIMENSION}}) {
            O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
          }
)";
    }
  }
source += R"(
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        auto idx = cM.get_multidimensional_index(k);
        if (idx[0] < (int)R_remainder) {
          float L_sram = cM[k] + fast::log2(cL[k]);
          L[idx[0]] = ({{MEMORY_NAME_L}})L_sram;
        }
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = cL.map_iterator(it);
        auto L_reciprocal = fast::divide(1, *dst_it);
        auto idx = cO_0.get_multidimensional_index(k);
)";
  for (unsigned short i = 0; i < kBlocks; i++) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    if ((i < kBlocks - 1) || (headDimension % blockDimensions[2] == 0)) {
      source += R"(
        O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
)";
    } else {
      source += R"(
        if (idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} < {{HEAD_DIMENSION}}) {
          O[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_{{LOOP_INDEX}}[k] * L_reciprocal);
        }
)";
    }
  }
source += R"(
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        auto idx = cM.get_multidimensional_index(k);
        float L_sram = cM[k] + fast::log2(cL[k]);
        L[idx[0]] = ({{MEMORY_NAME_L}})L_sram;
      }
    }
  }
)";
}

// Sprint 3.3 — Apple-style single-Otile forward kernel.
//
// Differences from loopForward():
//   - Single cS (no cS_0 / cS_1 double-buffering): K-loop step = BK, not 2·BK.
//   - Forced kBlocks=1: a single cO_0 covering full BD = head_dim.
//   - Always bypass tgmem: cP is a left-input cooperative_tensor (no P_buf).
//   - mem_none barriers (mem_threadgroup is unused since P_buf is gone).
//   - C_remainder is always handled in the trailing block (checkCEdge1 path
//     is irrelevant — we always pre-clamp the inner loop and re-run with
//     dynamic K for the partial tail).
//
// Limitations:
//   - Non-causal, non-masked, non-varlen only (loopForward dispatches accordingly).
//   - The softmax state (cM, cL, correction) remains a cooperative_tensor (the
//     brief calls for metal::vec, but reduce_rows() returns a coop_tensor and
//     swapping it out requires bypassing MPP's row-reduction primitive — out
//     of scope for this sprint).
void NAAttentionKernel::loopForwardSingleTile(CodeWriter &source) const noexcept {
  source.SetValue("MEMORY_NAME_Q", memoryName(AttentionOperand::Q));
  source.SetValue("MEMORY_NAME_K", memoryName(AttentionOperand::K));
  source.SetValue("MEMORY_NAME_V", memoryName(AttentionOperand::V));
  source.SetValue("MEMORY_NAME_O", memoryName(AttentionOperand::O));
  source.SetValue("MEMORY_NAME_L", memoryName(AttentionOperand::L));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("HEAD_DIMENSION_REMAINDER", std::to_string(headDimension % blockDimensions[2]));
  if (blockDimensions[1] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[1]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if (blockDimensions[2] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[2]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if ((headDimension % blockDimensions[2]) % 32 == 0) {
    source.SetValue("HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V", std::to_string(headDimension % blockDimensions[2]));
  } else {
    source.SetValue("HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  if (Hq != Hk) {
    source.SetValue("H_HK_RATIO", "/ " + std::to_string(Hq / Hk));
  } else {
    source.SetValue("H_HK_RATIO", "");
  }
  source.SetValue("DOT_SCALE", dotProductScale(scale, false));

  // Setup: tensors, descriptors, cooperative_tensors. Note that we do NOT
  // declare the `P` threadgroup tensor — bypass is forced on, so cP replaces it.
  source += R"(
  auto Q = tensor<device {{MEMORY_NAME_Q}},  dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device {{MEMORY_NAME_K}},  dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
  auto V = tensor<device {{MEMORY_NAME_V}},  dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, C));
  constexpr auto qk_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> matmul_qk_op;
)";
  if (headDimension % blockDimensions[2] > 0) {
    source += R"(
  constexpr auto qk_desc_remainder = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{HEAD_DIMENSION_REMAINDER_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc_remainder, execution_simdgroups<1>> matmul_qk_op_remainder;
)";
  }
  source += R"(
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, 0);
  // Single S accumulator (no double-buffer) — the structural change vs loopForward().
  auto cS = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cM = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cL = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto correction = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
    if (cM.is_valid_element(k)) {
      cM[k] = -numeric_limits<float>::infinity();
      cL[k] = numeric_limits<float>::denorm_min();
    }
  }
  auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(0, 0);
  constexpr auto pv_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
  // Always-bypass: cP is a cooperative_tensor (no P_buf threadgroup staging).
  auto cP = matmul_pv_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_O}}, {{MEMORY_NAME_V}}, float>();
  // Forced kBlocks=1: a single cO_0 covers the full BD == head_dim.
  auto cO_0 = matmul_pv_op.get_destination_cooperative_tensor<decltype(cP), decltype(mV), float>();

  // Main K-loop — single buffer, step = BLOCK_DIMENSIONS_TRAVERSAL (not _2).
  // Iterates over [0, C_aligned) where C_aligned = C - C_remainder. The
  // tail (C_remainder columns) is processed in a separate dynamic-K block
  // after the loop.
  for (uint c = 0; c < (C - C_remainder); c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cS[k] = 0;
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, c);
      matmul_qk_op.run(mQ, mK, cS);
    }
)";
  if (headDimension % blockDimensions[2] > 0) {
    source.SetValue("HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER", std::to_string(headDimension - (headDimension % blockDimensions[2])));
    source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, c);
      matmul_qk_op_remainder.run(mQ, mK, cS);
    }
)";
  }
  source += R"(
    // Online max reduce.
    auto cM_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS, cM_new, reduction_operation::max, -numeric_limits<float>::infinity());
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
        const float M_new = cM_new[k] * {{DOT_SCALE}};
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
      }
    }
    // Softmax: cS becomes cP (in cooperative-tensor, no tgmem staging).
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        auto it = cS.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        cS[k] = fast::exp2(cS[k] * {{DOT_SCALE}} - *dst_it);
      }
    }
    // Online sum reduce.
    auto cL_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS, cL_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if (cL.is_valid_element(k)) {
        cL[k] = cL[k] * correction[k] + cL_new[k];
      }
    }
    // First-iter init OR online correction of the running output accumulator.
    if (c == 0) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
        if (cO_0.is_valid_element(k)) {
          cO_0[k] = 0;
        }
      }
    } else {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
        if (cO_0.is_valid_element(k)) {
          auto it = cO_0.get_iterator(k);
          auto dst_it = correction.map_iterator(it);
          cO_0[k] *= *dst_it;
        }
      }
    }
    // Stage softmax output into cP (cooperative_tensor copy in registers).
    simdgroup_barrier(mem_flags::mem_none);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cP[k] = ({{MEMORY_NAME_O}})cS[k];
      }
    }
    // PV matmul — single cO_0 covers full BD = head_dim (kBlocks=1).
    auto mV_0 = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + 0, c);
    matmul_pv_op.run(cP, mV_0, cO_0);
  }
)";
  // Tail block: process the C_remainder columns (always — we don't use the
  // checkCEdge1 pre-padding trick here).
  source += R"(
  if (C_remainder > 0) {
    // Init cS: -inf for invalid columns (>= C_remainder), 0 elsewhere.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        auto idx = cS.get_multidimensional_index(k);
        cS[k] = idx[0] >= (int)C_remainder ? -numeric_limits<float>::infinity() : 0;
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += {{BLOCK_DIMENSIONS_HEAD}}) {
      auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + k, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + k, C - C_remainder);
      matmul_qk_op.run(mQ, mK, cS);
    }
)";
  if (headDimension % blockDimensions[2] > 0) {
    source.SetValue("HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER", std::to_string(headDimension - (headDimension % blockDimensions[2])));
    source += R"(
    {
      auto mQ = Q.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK = K.slice<{{HEAD_DIMENSION_REMAINDER}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{HEAD_DIMENSION_HEAD_DIMENSION_REMAINDER}}, C - C_remainder);
      matmul_qk_op_remainder.run(mQ, mK, cS);
    }
)";
  }
  source += R"(
    // Online max reduce.
    auto cM_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS, cM_new, reduction_operation::max, -numeric_limits<float>::infinity());
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
        const float M_new = cM_new[k] * {{DOT_SCALE}};
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
      }
    }
    // Softmax (zero-out invalid columns explicitly via idx check).
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        auto it = cS.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        auto idx = cS.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder) {
          cS[k] = 0;
        } else {
          cS[k] = fast::exp2(cS[k] * {{DOT_SCALE}} - *dst_it);
        }
      }
    }
    // Online sum reduce.
    auto cL_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS, cL_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if (cL.is_valid_element(k)) {
        cL[k] = cL[k] * correction[k] + cL_new[k];
      }
    }
    // Correct the running cO_0. (No first-iter init guard here: if the main
    // loop processed at least one tile, cO_0 is already initialized; if C
    // happened to be smaller than BLOCK_DIMENSIONS_TRAVERSAL so the main loop
    // didn't run, cO_0 still holds zeros from get_destination_cooperative_tensor.)
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = correction.map_iterator(it);
        cO_0[k] *= *dst_it;
      }
    }
    // Stage cS into cP and run dynamic-K PV matmul over the remainder.
    simdgroup_barrier(mem_flags::mem_none);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cP[k] = ({{MEMORY_NAME_O}})cS[k];
      }
    }
    auto mV_tail = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + 0, C - C_remainder);
    matmul_pv_op.run(cP, mV_tail, cO_0);
  }
)";
  // Output writeback (identical structure to loopForward(), kBlocks=1).
  source += R"(
  auto O = O_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hq) + tgid.y * {{HEAD_DIMENSION}};
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  if (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto idx = cO_0.get_multidimensional_index(k);
        if (idx[1] < (int)R_remainder) {
          auto it = cO_0.get_iterator(k);
          auto dst_it = cL.map_iterator(it);
          auto L_reciprocal = fast::divide(1, *dst_it);
          O[idx[0] + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_0[k] * L_reciprocal);
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        auto idx = cM.get_multidimensional_index(k);
        if (idx[0] < (int)R_remainder) {
          float L_sram = cM[k] + fast::log2(cL[k]);
          L[idx[0]] = ({{MEMORY_NAME_L}})L_sram;
        }
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = cL.map_iterator(it);
        auto L_reciprocal = fast::divide(1, *dst_it);
        auto idx = cO_0.get_multidimensional_index(k);
        O[idx[0] + idx[1] * K_Hq] = ({{MEMORY_NAME_O}})(cO_0[k] * L_reciprocal);
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        auto idx = cM.get_multidimensional_index(k);
        float L_sram = cM[k] + fast::log2(cL[k]);
        L[idx[0]] = ({{MEMORY_NAME_L}})L_sram;
      }
    }
  }
)";
}

// V34 — self-contained Apple-style attention_nax kernel using NAX-direct
// primitives (NAXTile / NAXFrag::mma). Replaces the legacy MPP cooperative_tensor
// path that was capped at execution_simdgroups<1> by Apple's static_asserts.
//
// Architecture (mirrors steel_attention_nax.h:73-482):
//   - Each TG: one Q-block × one head × one batch. Grid: (NQ, H, B).
//   - WM simdgroups per TG cooperate by row-partitioning BQ:
//     each SG handles `kU * TQ = BQ / WM` rows.
//   - QK matmul: NAXFrag::mma with two C frags + two B frags, transpose_b=true_type.
//   - Softmax: row_reduce<MaxOp>, row_bin_op<ExpSubOp>, row_reduce<SumOp> on Stile.
//   - Apply factor to Otile via row_bin_op<MulOp>(factor) (online correction).
//   - PV matmul: NAXFrag::mma with two C frags + two B frags, transpose_b=false_type.
//   - Final normalize: Otile *= rcp(sum_score) via row_bin_op<MulOp>.
//
// Scope: forward, non-causal, non-masked, non-varlen, single-Otile only.
// BHND layout (computes all addressing from AttnParams strides[3]).
// align_Q / align_K hardcoded to false for safety (uses load_rows / store_rows
// per Apple's safe path; can be specialized later via FCs).
//
// Apple file:line citations are inline at each substitution site.
std::string NAAttentionKernel::createV34Source() const noexcept {
  const int BQ = blockDimensions[0];
  const int BK = blockDimensions[1];
  const int BD = headDimension;
  const int WM = executionSIMDGroups;
  const int kU = 16;
  const int TQ = BQ / (WM * kU);   // expected = 1 per Apple's static_assert
  const int TD = BD / kU;          // 4 for D=64, 8 for D=128
  const int TK = BK / kU;          // BK / 16
  (void)TQ; (void)TD; (void)TK;

  const bool is_bf16 =
      memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::BF16;
  const char* dtype_str = is_bf16 ? "bfloat" : "half";
  const float dot_scale_log2e = scale * 1.4426950408889634f;

  std::ostringstream ss;
  ss << "// MFA_REQUIRE_MSL4\n";
  ss << "#include <metal_stdlib>\n";
  ss << "#include <metal_simdgroup>\n";
  ss << "#include <metal_simdgroup_matrix>\n";
  ss << "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  ss << "using namespace metal;\n";
  ss << "using namespace mpp::tensor_ops;\n";
  ss << "\n";
  ss << "// Apple steel/* helpers (verbatim from ~/code/mlx-source).\n";

  // === Apple NAX helpers (shared via naxHelpersBlock(), extracted Sprint v2.38.x Phase B) ===
  ss << naxHelpersBlock();

  // === V34 kernel ===
  ss << "\n// V34 kernel — Apple steel_attention_nax.h:73-482 pattern\n";
  ss << "using T = " << dtype_str << ";\n";
  ss << "using namespace mlx::steel;\n";
  ss << "\n";
  ss << "#define V34_BQ " << BQ << "\n";
  ss << "#define V34_BK " << BK << "\n";
  ss << "#define V34_BD " << BD << "\n";
  ss << "#define V34_WM " << WM << "\n";
  ss << "#define V34_TQ " << TQ << "\n";
  ss << "#define V34_TD " << TD << "\n";
  ss << "#define V34_TK " << TK << "\n";
  ss << "#define V34_DOT_SCALE " << dot_scale_log2e << "f\n";
  // v2.50 Sprint 4 Phase 4a: causal masking baked in as a compile-time
  // constant so the non-causal source remains bit-identical to pre-Sprint-4.
  ss << "#define V34_CAUSAL " << (isCausal ? 1 : 0) << "\n";
  ss << "\n";
  ss << R"MSL(
struct V34Params {
  int qL;        // query seq len
  int kL;        // key seq len
  int gqa_factor;
  int NQ;        // ceil(qL/BQ)
  int NK;        // ceil(kL/BK)
  int qL_rem;    // last-block remainder rows (0 means aligned)
  int kL_rem;    // last-block remainder cols (0 means aligned)
  // v2.50 Sprint 4 Phase 4a — causal offset for prefill-with-history /
  // decode cases.  Standalone forward passes qL_off=0; decode with KV
  // cache of length P passes qL_off=P so the causal diagonal is
  // (row + qL_off) < col → mask.  Apple convention
  // (steel_attention_nax.h:179-187).  Field exists unconditionally so
  // host-side V34ParamsHost layout is stable across causal/non-causal
  // pipelines; field is simply unused when V34_CAUSAL==0.
  int qL_off;
  // BHND strides (sequence stride = D, encoded in stride[2]).
  // Apple convention (steel_attention_nax.h:104-117).
  long Q_strides[3];  // [batch, head, seq]; D-stride implicit = 1
  long K_strides[3];
  long V_strides[3];
  long O_strides[3];
  // v2.36.x lse-write patch (BLK1 resolution per
  // docs/v6-nax/v34-backward-decisions.md DC0).  lse is [B, Hq, qL] FP32
  // contiguous; per-element stride is always 1 (no D dimension), so
  // L_strides[2] = 1 typically.
  long L_strides[3];
};

[[kernel, max_total_threads_per_threadgroup(V34_WM * 32)]]
void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    constant V34Params& params [[buffer(4)]],
    device float* L [[buffer(5)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {

  (void)simd_lane_id;  // pacify compiler

  // === Per-batch + per-head + per-Q-block ptr offsets (Apple lines 102-117) ===
  ulong3 tidl{tid.x, tid.y, tid.z};
  Q += tidl.z * params.Q_strides[0]
     + tidl.y * params.Q_strides[1]
     + tidl.x * V34_BQ * params.Q_strides[2];
  ulong kv_head_idx = ulong(tid.y) / ulong(params.gqa_factor);
  K += tidl.z * params.K_strides[0] + kv_head_idx * params.K_strides[1];
  V += tidl.z * params.V_strides[0] + kv_head_idx * params.V_strides[1];
  O += tidl.z * params.O_strides[0]
     + tidl.y * params.O_strides[1]
     + tidl.x * V34_BQ * params.O_strides[2];

  const float scale2 = V34_DOT_SCALE;  // scale * log2e (precomputed)

  // === MMA tiles + softmax state (Apple lines 127-166) ===
  using otile_t = NAXTile<float, V34_TQ, V34_TD>;
  otile_t Otile;
  Otile.clear();

  const short tm = 16 * V34_TQ * simd_group_id;
  Q += tm * int(params.Q_strides[2]);

  constexpr short kRowsPT = otile_t::kRowsPerThread;
  metal::vec<float, kRowsPT> max_score;
  metal::vec<float, kRowsPT> sum_score{0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<float>::finite_min;
  }

  // Last-block flags (Apple lines 189-194)
  const int NQ_aligned = params.qL / V34_BQ;
  const int NK_aligned = params.kL / V34_BK;
  const bool is_last_q = (int(tid.x) == NQ_aligned);
  const short lim_rows_q = (params.qL_rem > 0 ? params.qL_rem : V34_BQ) - tm;
  const short lim_rows_k = (params.kL_rem > 0 ? params.kL_rem : V34_BK);

  // v2.50 Sprint 4 Phase 4a — causal K-loop bound + mask-start tile
  // (Apple steel_attention_nax.h:176-187).  For causal:
  //   q_max = (tid.x+1)*BQ + qL_off   — last query row's absolute pos + 1
  //   kb_lim = ceil(q_max / BK)        — K-tiles past q_max are guaranteed masked
  //   kb_min_causal = q_min / BK       — first K-tile that overlaps the diagonal
  //                                       q_min = tid.x*BQ + qL_off
  // For non-causal: defaults preserved (kb_lim=NK, kb_min_causal=NK so the
  // per-element causal mask branch is never taken).
#if V34_CAUSAL
  int kb_lim;
  int kb_min_causal;
  {
    int q_max = (int(tid.x) + 1) * V34_BQ + params.qL_off;
    kb_lim = (q_max + V34_BK - 1) / V34_BK;
    kb_lim = metal::min(params.NK, kb_lim);
    int q_min = int(tid.x) * V34_BQ + params.qL_off;
    q_min = metal::max(0, q_min);
    kb_min_causal = q_min / V34_BK;
  }
#else
  const int kb_lim = params.NK;
  // kb_min_causal not declared in non-causal path — the causal mask
  // branch below is guarded by #if V34_CAUSAL so the symbol is absent.
#endif

  // === K-loop (Apple lines 197-457) ===
  for (int kb = 0; kb < kb_lim; kb++) {
    const bool is_last_k = (kb == NK_aligned);

    using stile_t = NAXTile<float, V34_TQ, V34_TK>;
    stile_t Stile;
    Stile.clear();

    // QK matmul (Apple lines 206-246)
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34_TD; id++) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;

          const int Q_load_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          const int K_load_off = ik * 16 * int(params.K_strides[2]) + id * 16;

          if (is_last_q) {
            Qtile.load_rows(Q + Q_load_off, int(params.Q_strides[2]), lim_rows_q - iq * 16);
          } else {
            Qtile.load(Q + Q_load_off, int(params.Q_strides[2]));
          }

          if (is_last_k) {
            Ktile.load_rows(K + K_load_off, int(params.K_strides[2]), lim_rows_k - ik * 16);
          } else {
            Ktile.load(K + K_load_off, int(params.K_strides[2]));
          }

          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale (Apple lines 248-252)
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }

    // Mask out length sequence on last K block (Apple lines 254-276)
    if (is_last_k) {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc = stile_t::NAXFrag_t::get_coord();
      const short sn = sc.x;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34_TK; ik++) {
          const short col_pos = ik * 16 + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < lim_rows_k) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // v2.50 Sprint 4 Phase 4a — causal mask (Apple lines 279-301).
    // Only run for K-tiles that overlap or cross the causal diagonal
    // (kb >= kb_min_causal); earlier tiles are entirely below the diagonal
    // and never need masking, later tiles past kb_lim are already skipped.
    // Per-element predicate: mask if absolute query row < absolute key col.
#if V34_CAUSAL
    if (kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc_c = stile_t::NAXFrag_t::get_coord();
      const short sn_c = sc_c.x;   // col base within fragment
      const short sm_c = sc_c.y;   // row base within fragment
      const int base_row = int(tid.x) * V34_BQ + params.qL_off + tm;
      const int base_col = kb * V34_BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34_TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const int r = base_row + iq * 16
                          + ii * stile_t::kFragRowsJump + sm_c;
              const int c = base_col + ik * 16 + jj + sn_c;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }
#endif

    // Online softmax (Apple lines 380-409)
    metal::vec<float, kRowsPT> new_max;
    metal::vec<float, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];

    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }
    Stile.template row_reduce<SumOp>(sum_score);

    // Apply factor to Otile (Apple line 412)
    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // PV matmul (Apple lines 417-452)
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V34_TD; id += 2) {
        if (V34_BD == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34_TK; ik++) {
          NAXTile<T, 1, 2> Vtile;
          const int V_load_off = ik * 16 * int(params.V_strides[2]) + id * 16;
          if (is_last_k) {
            Vtile.load_rows(V + V_load_off, int(params.V_strides[2]), lim_rows_k - ik * 16);
          } else {
            Vtile.load(V + V_load_off, int(params.V_strides[2]));
          }
          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    K += V34_BK * int(params.K_strides[2]);
    V += V34_BK * int(params.V_strides[2]);
  }

  // Normalize output (Apple lines 461-469)
  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<float, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }
  Otile.template row_bin_op<MulOp>(rcp);

  // === lse write (v2.36.x BLK1 resolution per
  // docs/v6-nax/v34-backward-decisions.md DC0) ===
  //
  // V34 forward keeps softmax state in LOG2 domain (S is scaled by
  // log2(e), max_score holds max(S_log2), sum_score holds
  // sum(exp2(S_log2 - max_score)) = sum(exp(S_natural - max_natural))).
  //
  // To produce lse in natural-log convention (matching mx.logsumexp and
  // STEEL forward write convention used by backward kernels):
  //   lse_natural = max_score * ln(2) + log(sum_score)
  // because max_score = log2(e) * max_natural, so dividing by log2(e)
  // (equivalently multiplying by ln(2)) recovers natural-log max.
  // sum_score is already invariant under the domain change.
  //
  // Layout: each lane in the simdgroup holds 2*V34_TQ row-state entries
  // (kRowsPT = otile_t::kRowsPerThread = V34_TQ * kElemRows where
  // kElemRows=2).  The 4 lanes covering the same row-group share the
  // same row_reduce result (sum/max), so only ONE lane per row must
  // write.  By convention we elect the lane with fn==0 (= get_coord().x).
  //
  // Row addressing: tile-row offset = iq*16 + fm + i*kElemRowsJump where
  //   fm = get_coord().y (lane's row-base within the 16-row fragment)
  //   kElemRowsJump = 8 (2-row stride within a frag)
  //   iq ∈ [0, V34_TQ) — fragment row index
  //   i  ∈ [0, kElemRows) — intra-frag row index
  // The base q-row pointer was advanced by tm rows above (see Q load);
  // we now advance L the same way.
  {
    const short2 sc_lse = otile_t::NAXFrag_t::get_coord();
    if (sc_lse.x == 0) {
      device float* L_row = L
          + tidl.z * params.L_strides[0]
          + tidl.y * params.L_strides[1]
          + tidl.x * V34_BQ * params.L_strides[2]
          + tm * params.L_strides[2];
      constexpr short kElemRows_lse = otile_t::NAXFrag_t::kElemRows;
      constexpr short kElemRowsJump_lse = otile_t::NAXFrag_t::kElemRowsJump;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kElemRows_lse; i++) {
          const short local_row = iq * 16 + sc_lse.y + i * kElemRowsJump_lse;
          const short row_idx = iq * kElemRows_lse + i;
          const bool in_range = (!is_last_q) || (local_row < lim_rows_q);
          if (in_range) {
            const float s = sum_score[row_idx];
            constexpr float ln2 = 0.6931471805599453f;  // log(2)
            // Edge: s == 0 only if no K-tile contributed (impossible for
            // dense non-causal forward; defensive only).
            // Formula: lse_natural = max_log2 * ln2 + log(sum) = max_natural + log(sum).
            const float lse_val = (s > 0.f)
                ? (max_score[row_idx] * ln2 + metal::log(s))
                : -Limits<float>::finite_max;
            L_row[local_row * params.L_strides[2]] = lse_val;
          }
        }
      }
    }
  }

  // Store O (Apple lines 471-481)
  O += tm * int(params.O_strides[2]);
  if (is_last_q) {
    if (lim_rows_q <= 0) return;
    Otile.store_rows(O, int(params.O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(params.O_strides[2]));
  }
}
)MSL";

  return ss.str();
}

void NAAttentionKernel::loopBackwardQuery(CodeWriter &source) const noexcept {
  const bool lowPrecisionInputs = memoryPrecisions[AttentionOperand::Q].value() != GEMMOperandPrecision::FP32;
  const bool useThreadgroupSharing = lowPrecisionInputs && !bypassThreadgroupMemory;
  const unsigned short kBlocks = (headDimension + blockDimensions[2] - 1) / blockDimensions[2];
  source.SetValue("MEMORY_NAME_Q", memoryName(AttentionOperand::Q));
  source.SetValue("MEMORY_NAME_K", memoryName(AttentionOperand::K));
  source.SetValue("MEMORY_NAME_V", memoryName(AttentionOperand::V));
  source.SetValue("MEMORY_NAME_DO", memoryName(AttentionOperand::dO));
  source.SetValue("MEMORY_NAME_DQ", memoryName(AttentionOperand::dQ));
  source.SetValue("MEMORY_NAME_DS", memoryName(AttentionOperand::D));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("KBLOCKS", std::to_string(kBlocks));
  if (blockDimensions[2] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[2]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  source.SetValue("DOT_SCALE", dotProductScale(scale, false));
  source.SetValue("DOT_SCALE_DERIVATIVE", dotProductScale(scale, true));
  if (Hq != Hk) {
    source.SetValue("H_HK_RATIO", "/ " + std::to_string(Hq / Hk));
  } else {
    source.SetValue("H_HK_RATIO", "");
  }
  source += R"(
  auto Q = tensor<device {{MEMORY_NAME_Q}}, dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device {{MEMORY_NAME_K}}, dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
  auto V = tensor<device {{MEMORY_NAME_V}}, dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, C));
  auto dO = tensor<device {{MEMORY_NAME_DO}}, dextents<int32_t, 2>, tensor_inline>(dO_buf, dextents<int32_t, 2>(K_Hq, R));
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mdO = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
  auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}}, 0);
  constexpr auto qk_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> matmul_qk_op;
  auto cS = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  constexpr auto dsk_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dsk_desc, execution_simdgroups<1>> matmul_dsk_op;
)";
  if (useThreadgroupSharing) {
    source += R"(
  threadgroup {{MEMORY_NAME_Q}} *Q_shared_buf = (threadgroup {{MEMORY_NAME_Q}}*)threadgroup_block +
      sgid * ({{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * 2);
  threadgroup {{MEMORY_NAME_DO}} *dO_shared_buf = Q_shared_buf + {{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  auto Q_shared = tensor<threadgroup {{MEMORY_NAME_Q}}, dextents<int32_t, 2>, tensor_inline>(
      Q_shared_buf, dextents<int32_t, 2>({{HEAD_DIMENSION}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}));
  auto dO_shared = tensor<threadgroup {{MEMORY_NAME_DO}}, dextents<int32_t, 2>, tensor_inline>(
      dO_shared_buf, dextents<int32_t, 2>({{HEAD_DIMENSION}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}));
  const uint lane = tid % 32;
  for (uint load_index = lane; load_index < {{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}}; load_index += 32) {
    const uint head_idx = load_index % {{HEAD_DIMENSION}};
    const uint row_idx = load_index / {{HEAD_DIMENSION}};
    const uint row = tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} + row_idx;
    if (row < R) {
      Q_shared_buf[load_index] = Q_buf[tgid.y * {{HEAD_DIMENSION}} + head_idx + row * K_Hq];
      dO_shared_buf[load_index] = dO_buf[tgid.y * {{HEAD_DIMENSION}} + head_idx + row * K_Hq];
    } else {
      Q_shared_buf[load_index] = 0;
      dO_shared_buf[load_index] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto cDS = matmul_dsk_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_DS}}, {{MEMORY_NAME_K}}, float>();
  auto cDP = matmul_qk_op.get_destination_cooperative_tensor<decltype(mdO), decltype(mV), float>();
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cDQ_{{LOOP_INDEX}} = matmul_dsk_op.get_destination_cooperative_tensor<decltype(cDS), decltype(mK), float>();\n";
    }
    source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDQ_0.get_capacity(); ++k) {
    if (cDQ_0.is_valid_element(k)) {
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "      cDQ_{{LOOP_INDEX}}[k] = 0;\n";
    }
    source += R"(
    }
  }
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  auto D = D_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  const bool query_row_tail = (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge);
  for (uint c = 0; c < C_edge; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cS[k] = 0;
        cDP[k] = 0;
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mdO_{{LOOP_INDEX}} = dO_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      matmul_qk_op.run(mQ_{{LOOP_INDEX}}, mK_{{LOOP_INDEX}}, cS);
      matmul_qk_op.run(mdO_{{LOOP_INDEX}}, mV_{{LOOP_INDEX}}, cDP);
    }
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDS.get_capacity(); ++k) {
      if (cDS.is_valid_element(k)) {
        auto idx = cDS.get_multidimensional_index(k);
        if (query_row_tail && idx[1] >= (int)R_remainder) {
          cDS[k] = 0;
        } else {
          const float P = fast::exp2(cS[k] * {{DOT_SCALE}} - (float)L[idx[1]]);
          cDS[k] = ({{MEMORY_NAME_DS}})(P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - (float)D[idx[1]]));
        }
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      matmul_dsk_op.run(cDS, mK_{{LOOP_INDEX}}, cDQ_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
  if (C_remainder > 0) {
    const uint c = C - C_remainder;
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cS[k] = 0;
        cDP[k] = 0;
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mdO_{{LOOP_INDEX}} = dO_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      matmul_qk_op.run(mQ_{{LOOP_INDEX}}, mK_{{LOOP_INDEX}}, cS);
      matmul_qk_op.run(mdO_{{LOOP_INDEX}}, mV_{{LOOP_INDEX}}, cDP);
    }
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDS.get_capacity(); ++k) {
      if (cDS.is_valid_element(k)) {
        auto idx = cDS.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder || (query_row_tail && idx[1] >= (int)R_remainder)) {
          cDS[k] = 0;
        } else {
          const float P = fast::exp2(cS[k] * {{DOT_SCALE}} - (float)L[idx[1]]);
          cDS[k] = ({{MEMORY_NAME_DS}})(P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - (float)D[idx[1]]));
        }
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      matmul_dsk_op.run(cDS, mK_{{LOOP_INDEX}}, cDQ_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
)";
  } else {
    source += R"(
  auto cDP = matmul_qk_op.get_destination_cooperative_tensor<decltype(mdO), decltype(mV), float>();
  auto cDS = matmul_dsk_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_DS}}, {{MEMORY_NAME_K}}, float>();
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cDQ_{{LOOP_INDEX}} = matmul_dsk_op.get_destination_cooperative_tensor<decltype(cDS), decltype(mK), float>();\n";
    }
    source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDQ_0.get_capacity(); ++k) {
    if (cDQ_0.is_valid_element(k)) {
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "      cDQ_{{LOOP_INDEX}}[k] = 0;\n";
    }
    source += R"(
    }
  }
  auto L = L_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  auto D = D_buf + tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  const bool query_row_tail = (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge);
  for (uint c = 0; c < C_edge; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cS[k] = 0;
        cDP[k] = 0;
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      matmul_qk_op.run(mQ_{{LOOP_INDEX}}, mK_{{LOOP_INDEX}}, cS);
      matmul_qk_op.run(mdO_{{LOOP_INDEX}}, mV_{{LOOP_INDEX}}, cDP);
    }
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDS.get_capacity(); ++k) {
      if (cDS.is_valid_element(k)) {
        auto idx = cDS.get_multidimensional_index(k);
        if (query_row_tail && idx[1] >= (int)R_remainder) {
          cDS[k] = 0;
        } else {
          const float P = fast::exp2(cS[k] * {{DOT_SCALE}} - (float)L[idx[1]]);
          cDS[k] = ({{MEMORY_NAME_DS}})(P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - (float)D[idx[1]]));
        }
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      matmul_dsk_op.run(cDS, mK_{{LOOP_INDEX}}, cDQ_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
  if (C_remainder > 0) {
    const uint c = C - C_remainder;
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cS[k] = 0;
        cDP[k] = 0;
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      matmul_qk_op.run(mQ_{{LOOP_INDEX}}, mK_{{LOOP_INDEX}}, cS);
      matmul_qk_op.run(mdO_{{LOOP_INDEX}}, mV_{{LOOP_INDEX}}, cDP);
    }
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDS.get_capacity(); ++k) {
      if (cDS.is_valid_element(k)) {
        auto idx = cDS.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder || (query_row_tail && idx[1] >= (int)R_remainder)) {
          cDS[k] = 0;
        } else {
          const float P = fast::exp2(cS[k] * {{DOT_SCALE}} - (float)L[idx[1]]);
          cDS[k] = ({{MEMORY_NAME_DS}})(P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - (float)D[idx[1]]));
        }
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y {{H_HK_RATIO}}* {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, c);
      matmul_dsk_op.run(cDS, mK_{{LOOP_INDEX}}, cDQ_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
)";
  }
  source += R"(
  auto dQ = dQ_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hq) + tgid.y * {{HEAD_DIMENSION}};
  if (R_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= R_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDQ_0.get_capacity(); ++k) {
      if (cDQ_0.is_valid_element(k)) {
        auto idx = cDQ_0.get_multidimensional_index(k);
        if (idx[1] >= (int)R_remainder) {
          continue;
        }
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    source += "      dQ[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_DQ}})cDQ_{{LOOP_INDEX}}[k];\n";
  }
  source += R"(
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDQ_0.get_capacity(); ++k) {
      if (cDQ_0.is_valid_element(k)) {
        auto idx = cDQ_0.get_multidimensional_index(k);
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    source += "      dQ[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hq] = ({{MEMORY_NAME_DQ}})cDQ_{{LOOP_INDEX}}[k];\n";
  }
  source += R"(
      }
    }
  }
)";
}

void NAAttentionKernel::loopBackwardKeyValue(CodeWriter &source) const noexcept {
  const bool lowPrecisionInputs = memoryPrecisions[AttentionOperand::Q].value() != GEMMOperandPrecision::FP32;
  const bool useThreadgroupSharing = lowPrecisionInputs && !bypassThreadgroupMemory;
  const unsigned short kBlocks = (headDimension + blockDimensions[2] - 1) / blockDimensions[2];
  source.SetValue("MEMORY_NAME_Q", memoryName(AttentionOperand::Q));
  source.SetValue("MEMORY_NAME_K", memoryName(AttentionOperand::K));
  source.SetValue("MEMORY_NAME_V", memoryName(AttentionOperand::V));
  source.SetValue("MEMORY_NAME_DO", memoryName(AttentionOperand::dO));
  source.SetValue("MEMORY_NAME_DK", memoryName(AttentionOperand::dK));
  source.SetValue("MEMORY_NAME_DV", memoryName(AttentionOperand::dV));
  source.SetValue("MEMORY_NAME_P", memoryName(AttentionOperand::O));
  source.SetValue("MEMORY_NAME_DS", memoryName(AttentionOperand::D));
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("KBLOCKS", std::to_string(kBlocks));
  if (blockDimensions[2] % 32 == 0) {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", std::to_string(blockDimensions[2]));
  } else {
    source.SetValue("BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V", "dynamic_length_v<int>");
  }
  source.SetValue("DOT_SCALE", dotProductScale(scale, false));
  source.SetValue("DOT_SCALE_DERIVATIVE", dotProductScale(scale, true));
  source += R"(
  auto Q = tensor<device {{MEMORY_NAME_Q}}, dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device {{MEMORY_NAME_K}}, dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
  auto V = tensor<device {{MEMORY_NAME_V}}, dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, C));
  auto dO = tensor<device {{MEMORY_NAME_DO}}, dextents<int32_t, 2>, tensor_inline>(dO_buf, dextents<int32_t, 2>(K_Hq, R));
  auto mK = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mV = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
  auto mQ = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, 0);
  auto mdO = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}}, 0);
  constexpr auto kqt_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<kqt_desc, execution_simdgroups<1>> matmul_kqt_op;
  auto cST = matmul_kqt_op.get_destination_cooperative_tensor<decltype(mK), decltype(mQ), float>();
  constexpr auto pdo_desc = matmul2d_descriptor({{BLOCK_DIMENSIONS_PARALLELIZATION}}, {{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pdo_desc, execution_simdgroups<1>> matmul_pdo_op;
)";
  if (useThreadgroupSharing) {
    source += R"(
  threadgroup {{MEMORY_NAME_K}} *K_shared_buf = (threadgroup {{MEMORY_NAME_K}}*)threadgroup_block +
      sgid * ({{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}} * 2);
  threadgroup {{MEMORY_NAME_V}} *V_shared_buf = K_shared_buf + {{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}};
  auto K_shared = tensor<threadgroup {{MEMORY_NAME_K}}, dextents<int32_t, 2>, tensor_inline>(
      K_shared_buf, dextents<int32_t, 2>({{HEAD_DIMENSION}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}));
  auto V_shared = tensor<threadgroup {{MEMORY_NAME_V}}, dextents<int32_t, 2>, tensor_inline>(
      V_shared_buf, dextents<int32_t, 2>({{HEAD_DIMENSION}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}));
  const uint lane = tid % 32;
  for (uint load_index = lane; load_index < {{HEAD_DIMENSION}} * {{BLOCK_DIMENSIONS_PARALLELIZATION}}; load_index += 32) {
    const uint head_idx = load_index % {{HEAD_DIMENSION}};
    const uint row_idx = load_index / {{HEAD_DIMENSION}};
    const uint row = tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} + row_idx;
    if (row < C) {
      K_shared_buf[load_index] = K_buf[tgid.y * {{HEAD_DIMENSION}} + head_idx + row * K_Hk];
      V_shared_buf[load_index] = V_buf[tgid.y * {{HEAD_DIMENSION}} + head_idx + row * K_Hk];
    } else {
      K_shared_buf[load_index] = 0;
      V_shared_buf[load_index] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto cP = matmul_pdo_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_P}}, {{MEMORY_NAME_DO}}, float>();
  auto cDS = matmul_pdo_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_DS}}, {{MEMORY_NAME_Q}}, float>();
  auto cDP = matmul_kqt_op.get_destination_cooperative_tensor<decltype(mV), decltype(mdO), float>();
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cDV_{{LOOP_INDEX}} = matmul_pdo_op.get_destination_cooperative_tensor<decltype(cP), decltype(mdO), float>();\n";
      source += "  auto cDK_{{LOOP_INDEX}} = matmul_pdo_op.get_destination_cooperative_tensor<decltype(cDS), decltype(mQ), float>();\n";
    }
    source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDV_0.get_capacity(); ++k) {
    if (cDV_0.is_valid_element(k)) {
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "      cDV_{{LOOP_INDEX}}[k] = 0;\n";
      source += "      cDK_{{LOOP_INDEX}}[k] = 0;\n";
    }
    source += R"(
    }
  }
  for (uint r = 0; r < KV_R_edge; r += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cST.get_capacity(); ++k) {
      if (cST.is_valid_element(k)) {
        cST[k] = 0;
        cDP[k] = 0;
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mK_{{LOOP_INDEX}} = K_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mV_{{LOOP_INDEX}} = V_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_kqt_op.run(mK_{{LOOP_INDEX}}, mQ_{{LOOP_INDEX}}, cST);
      matmul_kqt_op.run(mV_{{LOOP_INDEX}}, mdO_{{LOOP_INDEX}}, cDP);
    }
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP.get_capacity(); ++k) {
      if (cP.is_valid_element(k)) {
        auto idx = cP.get_multidimensional_index(k);
        const float L_value = (float)L_buf[r + idx[0]];
        const float D_value = (float)D_buf[r + idx[0]];
        const float P_value = fast::exp2(cST[k] * {{DOT_SCALE}} - L_value);
        cP[k] = ({{MEMORY_NAME_P}})P_value;
        cDS[k] = ({{MEMORY_NAME_DS}})(P_value * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - D_value));
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_pdo_op.run(cP, mdO_{{LOOP_INDEX}}, cDV_{{LOOP_INDEX}});
      matmul_pdo_op.run(cDS, mQ_{{LOOP_INDEX}}, cDK_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
  if (KV_R_remainder > 0) {
    const uint r = R - KV_R_remainder;
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cST.get_capacity(); ++k) {
      if (cST.is_valid_element(k)) {
        cST[k] = 0;
        cDP[k] = 0;
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mK_{{LOOP_INDEX}} = K_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mV_{{LOOP_INDEX}} = V_shared.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>({{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, 0);
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_kqt_op.run(mK_{{LOOP_INDEX}}, mQ_{{LOOP_INDEX}}, cST);
      matmul_kqt_op.run(mV_{{LOOP_INDEX}}, mdO_{{LOOP_INDEX}}, cDP);
    }
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP.get_capacity(); ++k) {
      if (cP.is_valid_element(k)) {
        auto idx = cP.get_multidimensional_index(k);
        if (idx[0] >= (int)KV_R_remainder) {
          cP[k] = 0;
          cDS[k] = 0;
        } else {
          const float L_value = (float)L_buf[r + idx[0]];
          const float D_value = (float)D_buf[r + idx[0]];
          const float P_value = fast::exp2(cST[k] * {{DOT_SCALE}} - L_value);
          cP[k] = ({{MEMORY_NAME_P}})P_value;
          cDS[k] = ({{MEMORY_NAME_DS}})(P_value * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - D_value));
        }
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_pdo_op.run(cP, mdO_{{LOOP_INDEX}}, cDV_{{LOOP_INDEX}});
      matmul_pdo_op.run(cDS, mQ_{{LOOP_INDEX}}, cDK_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
)";
  } else {
    source += R"(
  auto cDP = matmul_kqt_op.get_destination_cooperative_tensor<decltype(mV), decltype(mdO), float>();
  auto cP = matmul_pdo_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_P}}, {{MEMORY_NAME_DO}}, float>();
  auto cDS = matmul_pdo_op.get_left_input_cooperative_tensor<{{MEMORY_NAME_DS}}, {{MEMORY_NAME_Q}}, float>();
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "  auto cDV_{{LOOP_INDEX}} = matmul_pdo_op.get_destination_cooperative_tensor<decltype(cP), decltype(mdO), float>();\n";
      source += "  auto cDK_{{LOOP_INDEX}} = matmul_pdo_op.get_destination_cooperative_tensor<decltype(cDS), decltype(mQ), float>();\n";
    }
    source += R"(
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cDV_0.get_capacity(); ++k) {
    if (cDV_0.is_valid_element(k)) {
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source += "      cDV_{{LOOP_INDEX}}[k] = 0;\n";
      source += "      cDK_{{LOOP_INDEX}}[k] = 0;\n";
    }
    source += R"(
    }
  }
  for (uint r = 0; r < KV_R_edge; r += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cST.get_capacity(); ++k) {
      if (cST.is_valid_element(k)) {
        cST[k] = 0;
        cDP[k] = 0;
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_kqt_op.run(mK_{{LOOP_INDEX}}, mQ_{{LOOP_INDEX}}, cST);
      matmul_kqt_op.run(mV_{{LOOP_INDEX}}, mdO_{{LOOP_INDEX}}, cDP);
    }
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP.get_capacity(); ++k) {
      if (cP.is_valid_element(k)) {
        auto idx = cP.get_multidimensional_index(k);
        const float L_value = (float)L_buf[r + idx[0]];
        const float D_value = (float)D_buf[r + idx[0]];
        const float P = fast::exp2(cST[k] * {{DOT_SCALE}} - L_value);
        cP[k] = ({{MEMORY_NAME_P}})P;
        cDS[k] = ({{MEMORY_NAME_DS}})(P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - D_value));
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_pdo_op.run(cP, mdO_{{LOOP_INDEX}}, cDV_{{LOOP_INDEX}});
      matmul_pdo_op.run(cDS, mQ_{{LOOP_INDEX}}, cDK_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
  if (KV_R_remainder > 0) {
    const uint r = R - KV_R_remainder;
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cST.get_capacity(); ++k) {
      if (cST.is_valid_element(k)) {
        cST[k] = 0;
        cDP[k] = 0;
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mK_{{LOOP_INDEX}} = K.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mV_{{LOOP_INDEX}} = V.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_PARALLELIZATION}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_kqt_op.run(mK_{{LOOP_INDEX}}, mQ_{{LOOP_INDEX}}, cST);
      matmul_kqt_op.run(mV_{{LOOP_INDEX}}, mdO_{{LOOP_INDEX}}, cDP);
    }
)";
    }
    source += R"(
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP.get_capacity(); ++k) {
      if (cP.is_valid_element(k)) {
        auto idx = cP.get_multidimensional_index(k);
        if (idx[0] >= (int)KV_R_remainder) {
          cP[k] = 0;
          cDS[k] = 0;
        } else {
          const float L_value = (float)L_buf[r + idx[0]];
          const float D_value = (float)D_buf[r + idx[0]];
          const float P = fast::exp2(cST[k] * {{DOT_SCALE}} - L_value);
          cP[k] = ({{MEMORY_NAME_P}})P;
          cDS[k] = ({{MEMORY_NAME_DS}})(P * (cDP[k] * {{DOT_SCALE_DERIVATIVE}} - D_value));
        }
      }
    }
)";
    for (unsigned short i = 0; i < kBlocks; ++i) {
      source.SetValue("LOOP_INDEX", std::to_string(i));
      source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
      source += R"(
    {
      auto mQ_{{LOOP_INDEX}} = Q.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      auto mdO_{{LOOP_INDEX}} = dO.slice<{{BLOCK_DIMENSIONS_HEAD}}, {{BLOCK_DIMENSIONS_TRAVERSAL}}>(tgid.y * {{HEAD_DIMENSION}} + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}}, r);
      matmul_pdo_op.run(cP, mdO_{{LOOP_INDEX}}, cDV_{{LOOP_INDEX}});
      matmul_pdo_op.run(cDS, mQ_{{LOOP_INDEX}}, cDK_{{LOOP_INDEX}});
    }
)";
    }
    source += R"(
  }
)";
  }
  source += R"(
  auto dK = dK_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hk) + tgid.y * {{HEAD_DIMENSION}};
  auto dV = dV_buf + tgid.x * ({{BLOCK_DIMENSIONS_PARALLELIZATION}} * K_Hk) + tgid.y * {{HEAD_DIMENSION}};
  if (KV_C_remainder > 0 && tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}} >= KV_C_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDV_0.get_capacity(); ++k) {
      if (cDV_0.is_valid_element(k)) {
        auto idx = cDV_0.get_multidimensional_index(k);
        if (idx[1] >= (int)KV_C_remainder) {
          continue;
        }
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    source += "      dV[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hk] = ({{MEMORY_NAME_DV}})cDV_{{LOOP_INDEX}}[k];\n";
    source += "      dK[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hk] = ({{MEMORY_NAME_DK}})cDK_{{LOOP_INDEX}}[k];\n";
  }
  source += R"(
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cDV_0.get_capacity(); ++k) {
      if (cDV_0.is_valid_element(k)) {
        auto idx = cDV_0.get_multidimensional_index(k);
)";
  for (unsigned short i = 0; i < kBlocks; ++i) {
    source.SetValue("LOOP_INDEX", std::to_string(i));
    source.SetValue("LOOP_INDEX_BLOCK_DIMENSIONS_HEAD", std::to_string(i * blockDimensions[2]));
    source += "      dV[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hk] = ({{MEMORY_NAME_DV}})cDV_{{LOOP_INDEX}}[k];\n";
    source += "      dK[idx[0] + {{LOOP_INDEX_BLOCK_DIMENSIONS_HEAD}} + idx[1] * K_Hk] = ({{MEMORY_NAME_DK}})cDK_{{LOOP_INDEX}}[k];\n";
  }
  source += R"(
      }
    }
  }
)";
}


// =============================================================================
// V34 backward dQ kernel — self-contained Apple-style NAX-direct backward
// query.  Generated per V34 backward Option β sprint (Phase 1 Section B,
// post-BLK1 resolution).  Mirrors createV34Source() structure with backward
// inner loop.
//
// Algorithm (FA-2 backward dQ pattern):
//   1. Pre-loop:
//      a. Load lse[i] from device, multiply by log2(e) to get lse_log2.
//      b. Load D[i] = rowsum(dO[i] ⊙ O[i]) from device buffer (v2.38.1:
//         precomputed once on host via MLX, shared with split-dK + legacy-
//         fused-dKdV kernels; previously recomputed inline per kernel).
//   2. K-loop:
//      a. S = Q @ K^T (NAXFrag::mma).
//      b. S *= scale * log2(e) → S_log2.
//      c. row_bin_op<ExpSubOp>(lse_log2) → P = exp2(S_log2 - lse_log2)
//         = exp(S_natural - lse_natural).
//      d. dP = dO @ V^T (NAXFrag::mma).
//      e. row_bin_op<SubOp>(D_vec) on dP → dP - D.
//      f. element-wise dP *= P → dS = P * (dP - D).
//      g. dQ += dS @ K (NAXFrag::mma into FP32 accumulator).
//   3. Post-loop: dQ *= scale, cast FP32 → T, store to device.
//
// Apple-internal NAX primitives (BaseNAXFrag, NAXTile, row_reduce, row_bin_op)
// are inlined verbatim (same content as createV34Source()'s helpers block).
// Future cleanup: extract shared helpers into naxHelpersSource() (deferred to
// post-Phase-1 refactor sprint).
// =============================================================================
std::string NAAttentionKernel::createV34BackwardQuerySource() const noexcept {
  const int BQ = blockDimensions[0];
  const int BK = blockDimensions[1];
  const int BD = headDimension;
  const int WM = executionSIMDGroups;
  const int kU = 16;
  const int TQ = BQ / (WM * kU);
  const int TD = BD / kU;
  const int TK = BK / kU;
  (void)TQ; (void)TD; (void)TK;

  const bool is_bf16 =
      memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::BF16;
  const char* dtype_str = is_bf16 ? "bfloat" : "half";
  // Natural-domain scale (1/sqrt(D)); precompute log2-domain version.
  const float scale_log2e = scale * 1.4426950408889634f;

  std::ostringstream ss;
  ss << "// MFA_REQUIRE_MSL4\n";
  ss << "#include <metal_stdlib>\n";
  ss << "#include <metal_simdgroup>\n";
  ss << "#include <metal_simdgroup_matrix>\n";
  ss << "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  ss << "using namespace metal;\n";
  ss << "using namespace mpp::tensor_ops;\n";
  ss << "\n";
  ss << "// === Apple NAX helpers (shared via naxHelpersBlock(), extracted Sprint v2.38.x Phase B) ===\n";

  ss << naxHelpersBlock();

  ss << "\n// V34 backward dQ kernel — Apple-style NAX-direct\n";
  ss << "using T = " << dtype_str << ";\n";
  ss << "using namespace mlx::steel;\n";
  ss << "\n";
  ss << "// SubOp functor (needed for row_bin_op<SubOp> in backward; not in\n";
  ss << "// V34 forward helpers because forward doesn't use plain subtraction).\n";
  ss << "struct SubOp {\n";
  ss << "  template <typename U>\n";
  ss << "  METAL_FUNC static constexpr U apply(U x, U y) { return x - y; }\n";
  ss << "};\n\n";
  ss << "#define V34BWD_BQ " << BQ << "\n";
  ss << "#define V34BWD_BK " << BK << "\n";
  ss << "#define V34BWD_BD " << BD << "\n";
  ss << "#define V34BWD_WM " << WM << "\n";
  ss << "#define V34BWD_TQ " << TQ << "\n";
  ss << "#define V34BWD_TD " << TD << "\n";
  ss << "#define V34BWD_TK " << TK << "\n";
  ss << "#define V34BWD_SCALE " << scale << "f\n";
  ss << "#define V34BWD_SCALE_LOG2E " << scale_log2e << "f\n";
  // v2.50 Sprint 4 Phase 4b — causal masking baked in as compile-time
  // constant so non-causal source remains bit-identical to pre-Sprint-4.
  ss << "#define V34BWD_CAUSAL " << (isCausal ? 1 : 0) << "\n";
  ss << "\n";
  ss << R"BWDMSL(
struct V34BwdQParams {
  int qL;
  int kL;
  int gqa_factor;
  int NQ;
  int NK;
  int qL_rem;
  int kL_rem;
  // v2.50 Sprint 4 Phase 4b — causal offset.  Field order MUST match
  // V34BwdQParamsHost in v6_nax_compile.mm.
  int qL_off;
  // BHND strides (sequence stride = D, encoded in stride[2]).
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long O_strides[3];
  long L_strides[3];   // lse strides (FP32, [B, Hq, qL])
  long dO_strides[3];  // dO strides (same layout as Q)
  long dQ_strides[3];  // dQ strides (same layout as Q)
  long D_strides[3];   // D=rowsum(dO⊙O) strides (FP32, [B, Hq, qL], v2.38.1)
};

[[kernel, max_total_threads_per_threadgroup(V34BWD_WM * 32)]]
void attention_bwd_q(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device T* O [[buffer(3)]],
    const device float* L [[buffer(4)]],
    const device T* dO [[buffer(5)]],
    device T* dQ [[buffer(6)]],
    constant V34BwdQParams& params [[buffer(7)]],
    device const float* D [[buffer(8)]],  // v2.38.1: precomputed rowsum(dO⊙O), [B,Hq,qL] FP32
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {

  (void)simd_lane_id;

  // === Per-batch + per-head + per-Q-block ptr offsets ===
  ulong3 tidl{tid.x, tid.y, tid.z};
  Q  += tidl.z * params.Q_strides[0]
      + tidl.y * params.Q_strides[1]
      + tidl.x * V34BWD_BQ * params.Q_strides[2];
  ulong kv_head_idx = ulong(tid.y) / ulong(params.gqa_factor);
  K  += tidl.z * params.K_strides[0] + kv_head_idx * params.K_strides[1];
  V  += tidl.z * params.V_strides[0] + kv_head_idx * params.V_strides[1];
  O  += tidl.z * params.O_strides[0]
      + tidl.y * params.O_strides[1]
      + tidl.x * V34BWD_BQ * params.O_strides[2];
  L  += tidl.z * params.L_strides[0]
      + tidl.y * params.L_strides[1]
      + tidl.x * V34BWD_BQ * params.L_strides[2];
  dO += tidl.z * params.dO_strides[0]
      + tidl.y * params.dO_strides[1]
      + tidl.x * V34BWD_BQ * params.dO_strides[2];
  dQ += tidl.z * params.dQ_strides[0]
      + tidl.y * params.dQ_strides[1]
      + tidl.x * V34BWD_BQ * params.dQ_strides[2];
  // v2.38.1: D buffer per-batch/per-head/per-Q-block offset.
  D  += tidl.z * params.D_strides[0]
      + tidl.y * params.D_strides[1]
      + tidl.x * V34BWD_BQ * params.D_strides[2];

  // Per-SG row offset within the Q-block.
  const short tm = 16 * V34BWD_TQ * simd_group_id;
  Q  += tm * int(params.Q_strides[2]);
  O  += tm * int(params.O_strides[2]);
  L  += tm * int(params.L_strides[2]);
  dO += tm * int(params.dO_strides[2]);
  dQ += tm * int(params.dQ_strides[2]);
  D  += tm * int(params.D_strides[2]);  // v2.38.1

  // Last-block flags.
  const int NQ_aligned = params.qL / V34BWD_BQ;
  const int NK_aligned = params.kL / V34BWD_BK;
  const bool is_last_q = (int(tid.x) == NQ_aligned);
  const short lim_rows_q = (params.qL_rem > 0 ? params.qL_rem : V34BWD_BQ) - tm;
  const short lim_rows_k = (params.kL_rem > 0 ? params.kL_rem : V34BWD_BK);
  const int kb_lim = params.NK;

  // === MMA tile types ===
  using dq_accum_t = NAXTile<float, V34BWD_TQ, V34BWD_TD>;  // dQ FP32 accumulator
  using s_t       = NAXTile<float, V34BWD_TQ, V34BWD_TK>;  // S (= Q@K^T scaled)
  using dp_t      = NAXTile<float, V34BWD_TQ, V34BWD_TK>;  // dP (= dO@V^T)

  dq_accum_t dQ_accum;
  dQ_accum.clear();

  constexpr short kRowsPT = dq_accum_t::kRowsPerThread;

  // === Step 1: load lse, convert to log2 domain ===
  // lse from forward is natural-log; V34 inner-loop uses log2 domain via
  // scale*log2(e) and exp2.  Multiply lse by log2(e) once so that
  // row_bin_op<ExpSubOp>(lse_log2) below computes exp2(S_log2 - lse_log2)
  // = exp(S_natural - lse_natural) = correct softmax P.
  metal::vec<float, kRowsPT> lse_log2;
  {
    // Each lane owns kRowsPT rows in the SG; load each row's lse and scale.
    // Lane → row mapping uses get_coord() the same way row_reduce maps
    // partial-sum results.  Convention: lane with fn==0 reads lse[row].
    // Other lanes need to receive the value too (so each thread's lse_log2
    // vec is populated correctly for row_bin_op).  Use simd_shuffle_xor or
    // simd_broadcast to share.  Simpler approach: ALL lanes load lse[row]
    // for their owned rows (redundant device reads but correct).  4 lanes
    // covering the same row will read the same memory — coalesced, cheap.
    const short2 sc = dq_accum_t::NAXFrag_t::get_coord();
    constexpr short kElemRows = dq_accum_t::NAXFrag_t::kElemRows;
    constexpr short kElemRowsJump = dq_accum_t::NAXFrag_t::kElemRowsJump;
    constexpr float log2e_f = 1.4426950408889634f;
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWD_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kElemRows; i++) {
        const short local_row = iq * 16 + sc.y + i * kElemRowsJump;
        const short row_idx = iq * kElemRows + i;
        const bool in_range = (!is_last_q) || (local_row < lim_rows_q);
        // Out-of-range rows: set lse_log2 to +inf so exp2(S - inf) = 0,
        // which gives P=0, dS=0, no contribution to dQ for those rows.
        if (in_range) {
          lse_log2[row_idx] = L[local_row * int(params.L_strides[2])] * log2e_f;
        } else {
          lse_log2[row_idx] = Limits<float>::finite_max;  // ~+inf
        }
      }
    }
  }

  // === Step 2: load D[i] = rowsum(dO[i] ⊙ O[i]) from device buffer ===
  // v2.38.1: D is precomputed once on host via MLX (`mx.sum(dO*O, axis=-1)`)
  // and shared between dQ + split-dK + legacy-fused-dKdV kernels.  Replaces
  // an inline tile load + FP32 multiply + row_reduce.  Saves 1 rowsum per
  // V34 backward dQ call.
  //
  // Mirrors the lse-load pattern above (Step 1): each lane reads its owned
  // rows from the device buffer using `NAXFrag::get_coord()` + kElemRows /
  // kElemRowsJump.  Multiple lanes covering the same row → coalesced read.
  metal::vec<float, kRowsPT> D_vec;
  {
    const short2 sc = dq_accum_t::NAXFrag_t::get_coord();
    constexpr short kElemRows = dq_accum_t::NAXFrag_t::kElemRows;
    constexpr short kElemRowsJump = dq_accum_t::NAXFrag_t::kElemRowsJump;
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWD_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kElemRows; i++) {
        const short local_row = iq * 16 + sc.y + i * kElemRowsJump;
        const short row_idx = iq * kElemRows + i;
        const bool in_range = (!is_last_q) || (local_row < lim_rows_q);
        // Out-of-range rows: D=0 → contributes 0 to dS (P=0 there anyway
        // via the lse=+inf trick above).
        D_vec[row_idx] = in_range
            ? D[local_row * int(params.D_strides[2])]
            : 0.0f;
      }
    }
  }

  // === Step 3: K-loop ===
  for (int kb = 0; kb < kb_lim; kb++) {
    const bool is_last_k = (kb == NK_aligned);

    s_t Stile;
    Stile.clear();

    // QK matmul: S = Q @ K^T (NAXFrag::mma, transpose_b=true).
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWD_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWD_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWD_TD; id++) {
          NAXTile<T, 1, 1> Qfrag;
          NAXTile<T, 2, 1> Kfrag;

          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          const int K_off = ik * 16 * int(params.K_strides[2]) + id * 16;

          if (is_last_q) {
            Qfrag.load_rows(Q + Q_off, int(params.Q_strides[2]),
                            lim_rows_q - iq * 16);
          } else {
            Qfrag.load(Q + Q_off, int(params.Q_strides[2]));
          }
          if (is_last_k) {
            Kfrag.load_rows(K + K_off, int(params.K_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Kfrag.load(K + K_off, int(params.K_strides[2]));
          }

          s_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qfrag.frag_at(0, 0),
              metal::false_type{},
              Kfrag.frag_at(0, 0),
              Kfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale S into log2 domain.
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= V34BWD_SCALE_LOG2E;
    }

    // Mask out length sequence on last K block (mirrors forward, but here
    // out-of-range K columns must produce P=0 so they don't contribute to
    // dQ or D).  Setting S to -inf yields exp2(-inf - lse) = 0.
    if (is_last_k) {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc = s_t::NAXFrag_t::get_coord();
      const short sn = sc.x;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWD_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWD_TK; ik++) {
          const short col_pos = ik * 16 + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_t::kFragThrCols; jj++) {
              const auto loc = ii * s_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < lim_rows_k) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // v2.50 Sprint 4 Phase 4b — causal mask (mirror forward).  Setting
    // S[r,c] = -inf for r<c → exp2(-inf - lse) = 0 → P[r,c] = 0 → dS = 0
    // at masked positions → dQ accumulation naturally skips them.  Without
    // this, backward computes P over the unmasked S using the causal-masked
    // lse from forward, producing huge (incorrect) gradients for c>r.
#if V34BWD_CAUSAL
    {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc_c = s_t::NAXFrag_t::get_coord();
      const short sn_c = sc_c.x;   // col base within fragment
      const short sm_c = sc_c.y;   // row base within fragment
      const int base_row = int(tid.x) * V34BWD_BQ + params.qL_off + tm;
      const int base_col = kb * V34BWD_BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWD_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWD_TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_t::kFragThrCols; jj++) {
              const int r = base_row + iq * 16
                          + ii * s_t::kFragRowsJump + sm_c;
              const int c = base_col + ik * 16 + jj + sn_c;
              const auto loc = ii * s_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }
#endif

    // P = exp2(S - lse_log2)  (so P[i, j] = softmax_j(S_natural[i, .]))
    Stile.template row_bin_op<ExpSubOp>(lse_log2);
    // Stile now holds P in registers.

    // dP = dO @ V^T (NAXFrag::mma, transpose_b=true).
    dp_t dPtile;
    dPtile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWD_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWD_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWD_TD; id++) {
          NAXTile<T, 1, 1> dOfrag;
          NAXTile<T, 2, 1> Vfrag;

          const int dO_off = iq * 16 * int(params.dO_strides[2]) + id * 16;
          const int V_off = ik * 16 * int(params.V_strides[2]) + id * 16;

          if (is_last_q) {
            dOfrag.load_rows(dO + dO_off, int(params.dO_strides[2]),
                             lim_rows_q - iq * 16);
          } else {
            dOfrag.load(dO + dO_off, int(params.dO_strides[2]));
          }
          if (is_last_k) {
            Vfrag.load_rows(V + V_off, int(params.V_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Vfrag.load(V + V_off, int(params.V_strides[2]));
          }

          dp_t::NAXFrag_t::mma(
              dPtile.frag_at(iq, ik),
              dPtile.frag_at(iq, ik + 1),
              dOfrag.frag_at(0, 0),
              metal::false_type{},
              Vfrag.frag_at(0, 0),
              Vfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // dP -= D_vec (broadcast across cols of each row).
    dPtile.template row_bin_op<SubOp>(D_vec);

    // dS = P * (dP - D)  (element-wise; Stile holds P, dPtile holds dP-D).
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= dPtile.elems()[ii];
    }
    // Stile now holds dS.

    simdgroup_barrier(mem_flags::mem_none);

    // dQ_accum += dS @ K  (NAXFrag::mma, transpose_b=false).  Mirrors the
    // P @ V pattern from V34 forward.
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWD_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V34BWD_TD; id += 2) {
        if (V34BWD_BD == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWD_TK; ik++) {
          NAXTile<T, 1, 2> Kfrag2;
          const int K_off = ik * 16 * int(params.K_strides[2]) + id * 16;
          if (is_last_k) {
            Kfrag2.load_rows(K + K_off, int(params.K_strides[2]),
                             lim_rows_k - ik * 16);
          } else {
            Kfrag2.load(K + K_off, int(params.K_strides[2]));
          }
          dq_accum_t::NAXFrag_t::mma(
              dQ_accum.frag_at(iq, id),
              dQ_accum.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Kfrag2.frag_at(0, 0),
              Kfrag2.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    K += V34BWD_BK * int(params.K_strides[2]);
    V += V34BWD_BK * int(params.V_strides[2]);
  }  // end K-loop

  threadgroup_barrier(mem_flags::mem_none);

  // === Step 4: dQ_accum *= scale  (= 1/sqrt(D)) ===
  // Convention: dQ = ∇_Q (Q @ K^T * scale) backward = scale * (dS @ K).
  // We accumulated dS @ K above; multiply by scale here.
  {
    metal::vec<float, kRowsPT> scale_vec;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; i++) scale_vec[i] = V34BWD_SCALE;
    dQ_accum.template row_bin_op<MulOp>(scale_vec);
  }

  // === Step 5: store dQ ===
  // dQ_accum is FP32 NAXTile [TQ × TD]; store as T (FP16/BF16) to device.
  if (is_last_q) {
    if (lim_rows_q <= 0) return;
    dQ_accum.store_rows(dQ, int(params.dQ_strides[2]), lim_rows_q);
  } else {
    dQ_accum.store(dQ, int(params.dQ_strides[2]));
  }
}
)BWDMSL";

  return ss.str();
}



// =============================================================================
// V34 backward dK/dV kernel — single-SG (WM=1) NAX-direct implementation.
//
// Grid: (NK, H, B).  One TG per K-tile.  WM=1 (single SG per TG).
// Single SG iterates over ALL Q-tiles, accumulating partial dK + dV in
// FP32 NAX tiles in registers.  No cross-SG reduction needed.  After
// Q-loop, store dK + dV directly to device.
//
// Algorithm (FA-2 backward dK/dV per K-tile):
//   1. Pre-loop: clear dK_accum, dV_accum (FP32 NAX tiles [BK, D]).
//   2. Q-loop (q_tile = 0..NQ-1):
//      a. Load Q-tile, dO-tile, O-tile, lse-row from device.
//      b. Compute D[i] = rowsum(dO[i] ⊙ O[i]) for this Q-tile.
//      c. S = Q @ K^T (NAXFrag::mma, transpose_b=true).
//      d. S *= scale*log2(e) → log2 domain.
//      e. row_bin_op<ExpSubOp>(lse_log2) → P = exp(S_nat - lse_nat).
//      f. dP = dO @ V^T (NAXFrag::mma, transpose_b=true).
//      g. dP -= D (row_bin_op<SubOp>).
//      h. dS = P * dP (element-wise; Stile becomes dS).
//      i. dV_accum += P^T @ dO  (NAXFrag::mma, transpose_a=true).
//         WAIT: at step (h) Stile has been overwritten with dS.  Need to
//         compute dV BEFORE the P → dS transformation, OR keep P
//         separately.  Approach: compute dV first using P (in Stile),
//         then transform Stile → dS for dK accumulation.
//   3. Post-loop: cast accumulators FP32 → T, store dK + dV.
//
// Register budget (per-SG): dK_accum + dV_accum = 2 × BK × D × FP32
//   D=128, BK=32: 2 × 32 × 128 × 4 = 32 KB per SG (at register edge).
//   D=64,  BK=64: 2 × 64 × 64  × 4 = 32 KB per SG (same).
// =============================================================================
std::string NAAttentionKernel::createV34BackwardKeyValueSource() const noexcept {
  const int BQ = blockDimensions[0];
  const int BK = blockDimensions[1];
  const int BD = headDimension;
  const int WM = executionSIMDGroups;  // Should be 1 for this kernel.
  const int kU = 16;
  const int TQ = BQ / kU;            // BQ-rows-per-frag = BQ/16
  const int TD = BD / kU;
  const int TK = BK / kU;
  (void)TQ; (void)TD; (void)TK;

  const bool is_bf16 =
      memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::BF16;
  const char* dtype_str = is_bf16 ? "bfloat" : "half";
  const float scale_log2e = scale * 1.4426950408889634f;

  std::ostringstream ss;
  ss << "// MFA_REQUIRE_MSL4\n";
  ss << "#include <metal_stdlib>\n";
  ss << "#include <metal_simdgroup>\n";
  ss << "#include <metal_simdgroup_matrix>\n";
  ss << "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  ss << "using namespace metal;\n";
  ss << "using namespace mpp::tensor_ops;\n";
  ss << "\n";
  ss << naxHelpersBlock();

  ss << "\n// V34 backward dK/dV kernel\n";
  ss << "using T = " << dtype_str << ";\n";
  ss << "using namespace mlx::steel;\n";
  ss << "\n";
  ss << "struct SubOp {\n";
  ss << "  template <typename U>\n";
  ss << "  METAL_FUNC static constexpr U apply(U x, U y) { return x - y; }\n";
  ss << "};\n\n";
  ss << "#define V34BWDKV_BQ " << BQ << "\n";
  ss << "#define V34BWDKV_BK " << BK << "\n";
  ss << "#define V34BWDKV_BD " << BD << "\n";
  ss << "#define V34BWDKV_WM " << WM << "\n";
  ss << "#define V34BWDKV_TQ " << TQ << "\n";
  ss << "#define V34BWDKV_TD " << TD << "\n";
  ss << "#define V34BWDKV_TK " << TK << "\n";
  ss << "#define V34BWDKV_SCALE " << scale << "f\n";
  ss << "#define V34BWDKV_SCALE_LOG2E " << scale_log2e << "f\n";
  // v2.50 Sprint 4 Phase 4b-complete (Prompt 3): causal masking gated
  // by compile-time macro so non-causal source remains bit-identical.
  ss << "#define V34BWDKV_CAUSAL " << (isCausal ? 1 : 0) << "\n";
  ss << "\n";
  ss << R"BWDKVMSL(
struct V34BwdKVParams {
  int qL;
  int kL;
  int gqa_factor;
  int NQ;
  int NK;
  int qL_rem;
  int kL_rem;
  // v2.50 Sprint 4 Phase 4b-complete — causal offset.  Field exists
  // unconditionally so host-side V34BwdKVParamsHost layout matches.
  int qL_off;
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long O_strides[3];
  long L_strides[3];
  long dO_strides[3];
  long dK_strides[3];
  long dV_strides[3];
  long D_strides[3];   // D=rowsum(dO⊙O) strides (FP32, [B, Hq, qL], v2.38.1)
};

[[kernel, max_total_threads_per_threadgroup(V34BWDKV_WM * 32)]]
void attention_bwd_kv(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device T* O [[buffer(3)]],
    const device float* L [[buffer(4)]],
    const device T* dO [[buffer(5)]],
    device T* dK [[buffer(6)]],
    device T* dV [[buffer(7)]],
    constant V34BwdKVParams& params [[buffer(8)]],
    device const float* D [[buffer(9)]],  // v2.38.1: precomputed rowsum(dO⊙O), [B,Hq,qL] FP32
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {

  (void)simd_lane_id;
  (void)simd_group_id;  // WM=1 single-SG (WM=2 K-partition falsified Phase 2.O1)

  // === Per-batch + per-head + per-K-block ptr offsets ===
  // Grid (NK, H, B): tid.x = K-tile index, tid.y = head, tid.z = batch.
  ulong3 tidl{tid.x, tid.y, tid.z};

  // Q/O/L/dO indexed by query head (Hq).
  Q  += tidl.z * params.Q_strides[0]  + tidl.y * params.Q_strides[1];
  O  += tidl.z * params.O_strides[0]  + tidl.y * params.O_strides[1];
  L  += tidl.z * params.L_strides[0]  + tidl.y * params.L_strides[1];
  dO += tidl.z * params.dO_strides[0] + tidl.y * params.dO_strides[1];
  // v2.38.1: D buffer per-batch/per-Hq-head offset (D indexed by query head).
  D  += tidl.z * params.D_strides[0]  + tidl.y * params.D_strides[1];

  // K/V/dK/dV indexed by KV head (Hk) — apply GQA factor.
  ulong kv_head_idx = ulong(tid.y) / ulong(params.gqa_factor);
  K  += tidl.z * params.K_strides[0]  + kv_head_idx * params.K_strides[1]
      + tidl.x * V34BWDKV_BK * params.K_strides[2];
  V  += tidl.z * params.V_strides[0]  + kv_head_idx * params.V_strides[1]
      + tidl.x * V34BWDKV_BK * params.V_strides[2];

  // dK/dV outputs: with GQA, multiple Q-heads contribute to the same KV head.
  // For this Phase 1 implementation, write per Q-head (one dK/dV per (b, hq, k)
  // slice).  Caller is responsible for summing across heads if GQA reduction
  // is wanted.  Per DC12: most production v34-backward shapes are Hq==Hk.
  // We use Q-head indexing for dK/dV writes to match SDPA-vjp output layout.
  dK += tidl.z * params.dK_strides[0] + tidl.y * params.dK_strides[1]
      + tidl.x * V34BWDKV_BK * params.dK_strides[2];
  dV += tidl.z * params.dV_strides[0] + tidl.y * params.dV_strides[1]
      + tidl.x * V34BWDKV_BK * params.dV_strides[2];

  // Last-K-block flag.
  const int NK_aligned = params.kL / V34BWDKV_BK;
  const int NQ_aligned = params.qL / V34BWDKV_BQ;
  const bool is_last_k = (int(tid.x) == NK_aligned);
  const short lim_rows_k = (params.kL_rem > 0 && is_last_k)
      ? params.kL_rem : V34BWDKV_BK;
  const int nq_full = params.qL / V34BWDKV_BQ;
  const int nq_rem = params.qL % V34BWDKV_BQ;
  const int q_loop = nq_rem > 0 ? nq_full + 1 : nq_full;

  // === Accumulators ===
  // Phase 2.O1 finding (2026-05-13): WM=2 K-row partition was implemented
  // and FALSIFIED empirically (0.77-0.84× speedup = 20-25% REGRESSION vs
  // WM=1).  The redundant softmax compute across SGs taxes more than the
  // GEMM partition saves.  Reverted to WM=1 single-SG with full BK rows
  // per SG.  Q-row partition (with TGP streaming reduction) is the
  // recommended next-sprint approach — see status doc.
  using dkv_t = NAXTile<float, V34BWDKV_TK, V34BWDKV_TD>;
  dkv_t dK_accum, dV_accum;
  dK_accum.clear();
  dV_accum.clear();

  // === Q-loop ===
  // Process one Q-tile at a time.  Per iteration:
  //   - Compute S = Q @ K^T
  //   - Recompute P via softmax (using lse from forward)
  //   - Compute dP = dO @ V^T, D = rowsum(dO ⊙ O), dS = P*(dP - D)
  //   - dV_accum += P^T @ dO
  //   - dK_accum += dS^T @ Q
  for (int qb = 0; qb < q_loop; qb++) {
    const bool is_last_q = (qb == NQ_aligned);
    const short lim_rows_q = (params.qL_rem > 0 && is_last_q)
        ? params.qL_rem : V34BWDKV_BQ;

    // Per-q-tile pointers.
    const device T* Q_q  = Q  + qb * V34BWDKV_BQ * int(params.Q_strides[2]);
    const device T* O_q  = O  + qb * V34BWDKV_BQ * int(params.O_strides[2]);
    const device float* L_q = L + qb * V34BWDKV_BQ * int(params.L_strides[2]);
    const device T* dO_q = dO + qb * V34BWDKV_BQ * int(params.dO_strides[2]);
    // v2.38.1: D buffer Q-block offset (mirror L_q).
    const device float* D_q = D + qb * V34BWDKV_BQ * int(params.D_strides[2]);

    // --- Load lse, convert to log2 domain ---
    using s_q_t = NAXTile<float, V34BWDKV_TQ, V34BWDKV_TK>;
    constexpr short kRowsPT_q = s_q_t::kRowsPerThread;
    metal::vec<float, kRowsPT_q> lse_log2;
    {
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      constexpr short kEr = s_q_t::NAXFrag_t::kElemRows;
      constexpr short kErJ = s_q_t::NAXFrag_t::kElemRowsJump;
      constexpr float log2e_f = 1.4426950408889634f;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDKV_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEr; i++) {
          const short local_row = iq * 16 + sc.y + i * kErJ;
          const short row_idx = iq * kEr + i;
          const bool in_range = (!is_last_q) || (local_row < lim_rows_q);
          if (in_range) {
            lse_log2[row_idx] = L_q[local_row * int(params.L_strides[2])] * log2e_f;
          } else {
            lse_log2[row_idx] = Limits<float>::finite_max;  // P=0 for OOR rows
          }
        }
      }
    }

    // --- v2.38.1: load D[i] = rowsum(dO[i] ⊙ O[i]) from device buffer ---
    // D is precomputed on host via MLX, shared with dQ + split-dK kernels.
    // Mirrors the lse-load pattern above; coalesced reads across lanes.
    metal::vec<float, kRowsPT_q> D_vec;
    {
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      constexpr short kEr = s_q_t::NAXFrag_t::kElemRows;
      constexpr short kErJ = s_q_t::NAXFrag_t::kElemRowsJump;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDKV_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEr; i++) {
          const short local_row = iq * 16 + sc.y + i * kErJ;
          const short row_idx = iq * kEr + i;
          const bool in_range = (!is_last_q) || (local_row < lim_rows_q);
          D_vec[row_idx] = in_range
              ? D_q[local_row * int(params.D_strides[2])]
              : 0.0f;
        }
      }
    }

    // --- QK matmul: S = Q @ K^T ---
    s_q_t Stile;
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWDKV_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWDKV_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWDKV_TD; id++) {
          NAXTile<T, 1, 1> Qfrag;
          NAXTile<T, 2, 1> Kfrag;
          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          const int K_off = ik * 16 * int(params.K_strides[2]) + id * 16;
          if (is_last_q) {
            Qfrag.load_rows(Q_q + Q_off, int(params.Q_strides[2]),
                            lim_rows_q - iq * 16);
          } else {
            Qfrag.load(Q_q + Q_off, int(params.Q_strides[2]));
          }
          if (is_last_k) {
            Kfrag.load_rows(K + K_off, int(params.K_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Kfrag.load(K + K_off, int(params.K_strides[2]));
          }
          s_q_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qfrag.frag_at(0, 0),
              metal::false_type{},
              Kfrag.frag_at(0, 0),
              Kfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale into log2 domain.
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_q_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= V34BWDKV_SCALE_LOG2E;
    }

    // Mask out-of-range K columns to -inf so exp2 produces 0.
    if (is_last_k) {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      const short sn = sc.x;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDKV_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDKV_TK; ik++) {
          const short col_pos = ik * 16 + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < lim_rows_k) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // v2.50 Sprint 4 Phase 4b-complete (Prompt 3) — causal mask for dKV legacy
    // fused kernel.  WM=1 single-SG (Phase 2.O1 finding) → no SG Q-row partition.
    // K-parallel kernel: tid.x = K-block, qb in Q-loop.
    //   base_row = qb * BQ + qL_off   (Q in loop)
    //   base_col = tid.x * BK         (K parallel, no SG partition for WM=1)
    // Setting S[r,c] = -inf for r<c → exp2(-inf - lse) = 0 → P = 0 → dS = 0
    // → dK_accum + dV_accum naturally skip masked positions.
#if V34BWDKV_CAUSAL
    {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc_c = s_q_t::NAXFrag_t::get_coord();
      const short sn_c = sc_c.x;
      const short sm_c = sc_c.y;
      const int base_row = qb * V34BWDKV_BQ + params.qL_off;
      const int base_col = int(tid.x) * V34BWDKV_BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDKV_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDKV_TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const int r = base_row + iq * 16
                          + ii * s_q_t::kFragRowsJump + sm_c;
              const int c = base_col + ik * 16 + jj + sn_c;
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }
#endif

    // P = exp2(S - lse_log2)
    Stile.template row_bin_op<ExpSubOp>(lse_log2);
    // Stile holds P now.

    // --- dV_accum += P^T @ dO  (compute BEFORE dS = P*(...) overwrites P) ---
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < V34BWDKV_TK; ik++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V34BWDKV_TD; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < V34BWDKV_TQ; iq++) {
          NAXTile<T, 1, 2> dOfrag2;
          const int dO_off = iq * 16 * int(params.dO_strides[2]) + id * 16;
          if (is_last_q) {
            dOfrag2.load_rows(dO_q + dO_off, int(params.dO_strides[2]),
                              lim_rows_q - iq * 16);
          } else {
            dOfrag2.load(dO_q + dO_off, int(params.dO_strides[2]));
          }
          dkv_t::NAXFrag_t::mma(
              dV_accum.frag_at(ik, id),
              dV_accum.frag_at(ik, id + 1),
              Stile.frag_at(iq, ik),
              metal::true_type{},   // transpose_a: P^T
              dOfrag2.frag_at(0, 0),
              dOfrag2.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    // --- dP = dO @ V^T ---
    using dp_t = NAXTile<float, V34BWDKV_TQ, V34BWDKV_TK>;
    dp_t dPtile;
    dPtile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWDKV_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWDKV_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWDKV_TD; id++) {
          NAXTile<T, 1, 1> dOfrag;
          NAXTile<T, 2, 1> Vfrag;
          const int dO_off = iq * 16 * int(params.dO_strides[2]) + id * 16;
          const int V_off = ik * 16 * int(params.V_strides[2]) + id * 16;
          if (is_last_q) {
            dOfrag.load_rows(dO_q + dO_off, int(params.dO_strides[2]),
                             lim_rows_q - iq * 16);
          } else {
            dOfrag.load(dO_q + dO_off, int(params.dO_strides[2]));
          }
          if (is_last_k) {
            Vfrag.load_rows(V + V_off, int(params.V_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Vfrag.load(V + V_off, int(params.V_strides[2]));
          }
          dp_t::NAXFrag_t::mma(
              dPtile.frag_at(iq, ik),
              dPtile.frag_at(iq, ik + 1),
              dOfrag.frag_at(0, 0),
              metal::false_type{},
              Vfrag.frag_at(0, 0),
              Vfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // dP -= D
    dPtile.template row_bin_op<SubOp>(D_vec);

    // dS = P * (dP - D)  (element-wise; overwrite Stile with dS).
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_q_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= dPtile.elems()[ii];
    }
    // Stile holds dS now.

    // --- dK_accum += dS^T @ Q ---
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < V34BWDKV_TK; ik++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V34BWDKV_TD; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < V34BWDKV_TQ; iq++) {
          NAXTile<T, 1, 2> Qfrag2;
          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          if (is_last_q) {
            Qfrag2.load_rows(Q_q + Q_off, int(params.Q_strides[2]),
                             lim_rows_q - iq * 16);
          } else {
            Qfrag2.load(Q_q + Q_off, int(params.Q_strides[2]));
          }
          dkv_t::NAXFrag_t::mma(
              dK_accum.frag_at(ik, id),
              dK_accum.frag_at(ik, id + 1),
              Stile.frag_at(iq, ik),
              metal::true_type{},   // transpose_a: dS^T
              Qfrag2.frag_at(0, 0),
              Qfrag2.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    simdgroup_barrier(mem_flags::mem_none);
  }  // end Q-loop

  // === Post-loop: dK_accum *= scale (∇_K of QK^T*s = s*dS^T@Q) ===
  {
    constexpr short kRowsPT_k = dkv_t::kRowsPerThread;
    metal::vec<float, kRowsPT_k> scale_vec;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT_k; i++) scale_vec[i] = V34BWDKV_SCALE;
    dK_accum.template row_bin_op<MulOp>(scale_vec);
  }

  // === Store dK + dV (cast FP32 → T) ===
  if (is_last_k) {
    if (lim_rows_k <= 0) return;
    dK_accum.store_rows(dK, int(params.dK_strides[2]), lim_rows_k);
    dV_accum.store_rows(dV, int(params.dV_strides[2]), lim_rows_k);
  } else {
    dK_accum.store(dK, int(params.dK_strides[2]));
    dV_accum.store(dV, int(params.dV_strides[2]));
  }
}
)BWDKVMSL";

  return ss.str();
}



// =============================================================================
// V34 backward dV-only kernel — WM=4 Q-row partition with per-SG slot output.
//
// Phase 2.O2 (V34 backward optimization sprint, 2026-05-13):
// Each SG handles BQ/WM = 16 Q-rows (1 NAXFrag).  Softmax is intra-SG (no
// replication tax).  Each SG accumulates its dV partial (full BK × D) from
// its 16 Q-rows × NQ Q-tiles' contributions.  After Q-loop, each SG writes
// its partial to dV_partials[b, hq, sg_id, k, d] (FP32).  Python wrapper
// reduces via mx.sum(axis=2) and casts to T.
//
// Algorithm per Q-tile inner iteration:
//   1. Load Q[SG-rows], lse[SG-rows], dO[SG-rows].
//   2. Compute lse_log2 = lse * log2(e).
//   3. S = Q[SG-rows] @ K^T (NAXFrag::mma, transpose_b=true). Shape [16, BK].
//   4. S *= scale * log2(e). Last-K mask if needed.
//   5. P = row_bin_op<ExpSubOp>(lse_log2) → exp(S_natural - lse_natural).
//   6. dV_accum += P^T @ dO[SG-rows] (NAXFrag::mma, transpose_a=true).
//
// No D / dP / dS computation needed (dV-only).
// =============================================================================
std::string NAAttentionKernel::createV34BackwardDVSource() const noexcept {
  const int BQ = blockDimensions[0];
  const int BK = blockDimensions[1];
  const int BD = headDimension;
  const int WM = executionSIMDGroups;
  const int kU = 16;
  // Per-SG Q-row count: BQ/WM. Must equal 16 (1 NAXFrag).  Enforced by caller.
  const int TQ_per_SG = BQ / (WM * kU);  // expected = 1
  const int TD = BD / kU;
  const int TK = BK / kU;
  (void)TQ_per_SG; (void)TD; (void)TK;

  const bool is_bf16 =
      memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::BF16;
  const char* dtype_str = is_bf16 ? "bfloat" : "half";
  const float scale_log2e = scale * 1.4426950408889634f;

  std::ostringstream ss;
  ss << "// MFA_REQUIRE_MSL4\n";
  ss << "#include <metal_stdlib>\n";
  ss << "#include <metal_simdgroup>\n";
  ss << "#include <metal_simdgroup_matrix>\n";
  ss << "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  ss << "using namespace metal;\n";
  ss << "using namespace mpp::tensor_ops;\n";
  ss << "\n";

  ss << naxHelpersBlock();

  ss << "\nusing T = " << dtype_str << ";\n";
  ss << "using namespace mlx::steel;\n\n";
  ss << "#define V34BWDV_BQ " << BQ << "\n";
  ss << "#define V34BWDV_BK " << BK << "\n";
  ss << "#define V34BWDV_BD " << BD << "\n";
  ss << "#define V34BWDV_WM " << WM << "\n";
  ss << "#define V34BWDV_TQ " << TQ_per_SG << "\n";
  ss << "#define V34BWDV_TD " << TD << "\n";
  ss << "#define V34BWDV_TK " << TK << "\n";
  ss << "#define V34BWDV_SCALE_LOG2E " << scale_log2e << "f\n";
  // v2.50 Sprint 4 Phase 4b-complete (Prompt 3): causal masking macro
  ss << "#define V34BWDV_CAUSAL " << (isCausal ? 1 : 0) << "\n";
  ss << "\n";

  ss << R"BWDVMSL(
struct V34BwdVParams {
  int qL, kL;
  int gqa_factor;
  int NQ, NK;
  int qL_rem, kL_rem;
  // v2.50 Sprint 4 Phase 4b-complete — causal offset (host struct match)
  int qL_off;
  long Q_strides[3];   // [B, Hq, qL, D]
  long K_strides[3];   // [B, Hk, kL, D]
  long V_strides[3];   // [B, Hk, kL, D]
  long L_strides[3];   // [B, Hq, qL]
  long dO_strides[3];  // [B, Hq, qL, D]
  // dV_partials shape: [B, Hq, WM, kL, D] FP32. Strides:
  //   [Hq*WM*kL*D, WM*kL*D, kL*D, D] (sg-slot stride = kL*D)
  // dVp_strides indexed as: [B, Hq, sg, k] -> [3]; D-stride implicit = 1.
  long dVp_strides[4];
};

[[kernel, max_total_threads_per_threadgroup(V34BWDV_WM * 32)]]
void attention_bwd_dv(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device float* L [[buffer(3)]],
    const device T* dO [[buffer(4)]],
    device float* dV_partials [[buffer(5)]],
    constant V34BwdVParams& params [[buffer(6)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {

  (void)simd_lane_id;
  // Grid (NK, Hq, B). Each TG owns 1 K-tile, WM SGs partition Q-rows.

  ulong3 tidl{tid.x, tid.y, tid.z};
  ulong kv_head_idx = ulong(tid.y) / ulong(params.gqa_factor);

  // Base pointers.
  Q  += tidl.z * params.Q_strides[0]  + tidl.y * params.Q_strides[1];
  dO += tidl.z * params.dO_strides[0] + tidl.y * params.dO_strides[1];
  L  += tidl.z * params.L_strides[0]  + tidl.y * params.L_strides[1];
  K  += tidl.z * params.K_strides[0]  + kv_head_idx * params.K_strides[1]
      + tidl.x * V34BWDV_BK * params.K_strides[2];
  V  += tidl.z * params.V_strides[0]  + kv_head_idx * params.V_strides[1]
      + tidl.x * V34BWDV_BK * params.V_strides[2];

  // Per-SG dV_partials slot.
  // dV_partials[b, hq, sg, k_base, d] → offset = b*S0 + hq*S1 + sg*S2 + k_base*S3
  dV_partials += tidl.z * params.dVp_strides[0]
              +  tidl.y * params.dVp_strides[1]
              +  simd_group_id * params.dVp_strides[2]
              +  tidl.x * V34BWDV_BK * params.dVp_strides[3];

  // Per-SG Q-row offset within each Q-tile.
  const short sg_q_offset = 16 * V34BWDV_TQ * simd_group_id;

  // Last-K / Q bookkeeping.
  const int NQ_aligned = params.qL / V34BWDV_BQ;
  const int NK_aligned = params.kL / V34BWDV_BK;
  const bool is_last_k = (int(tid.x) == NK_aligned);
  const short lim_rows_k = (params.kL_rem > 0 && is_last_k)
      ? params.kL_rem : V34BWDV_BK;
  const int nq_full = params.qL / V34BWDV_BQ;
  const int nq_rem = params.qL % V34BWDV_BQ;
  const int q_loop = nq_rem > 0 ? nq_full + 1 : nq_full;

  // Per-SG dV accumulator (FULL BK × D, contributions only from SG's Q-rows).
  using dv_t = NAXTile<float, V34BWDV_TK, V34BWDV_TD>;
  dv_t dV_accum;
  dV_accum.clear();

  // Per-SG S tile (16 Q-rows × BK).
  using s_q_t = NAXTile<float, V34BWDV_TQ, V34BWDV_TK>;
  constexpr short kRowsPT_q = s_q_t::kRowsPerThread;

  // Q-loop.
  for (int qb = 0; qb < q_loop; qb++) {
    const bool is_last_q = (qb == NQ_aligned);
    const short lim_rows_q_full = (params.qL_rem > 0 && is_last_q)
        ? params.qL_rem : V34BWDV_BQ;
    // SG's effective limit (within its 16-row slice).
    const short sg_lim_q = (short)max(0, (int)lim_rows_q_full - (int)sg_q_offset);
    if (is_last_q && sg_lim_q <= 0) continue;

    // Per-q-tile + SG offset pointers.
    const device T* Q_qs  = Q  + qb * V34BWDV_BQ * int(params.Q_strides[2])
                              + sg_q_offset * int(params.Q_strides[2]);
    const device T* dO_qs = dO + qb * V34BWDV_BQ * int(params.dO_strides[2])
                              + sg_q_offset * int(params.dO_strides[2]);
    const device float* L_qs = L + qb * V34BWDV_BQ * int(params.L_strides[2])
                                + sg_q_offset * int(params.L_strides[2]);

    // --- Load lse for SG's 16 rows, scale to log2 domain ---
    metal::vec<float, kRowsPT_q> lse_log2;
    {
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      constexpr short kEr = s_q_t::NAXFrag_t::kElemRows;
      constexpr short kErJ = s_q_t::NAXFrag_t::kElemRowsJump;
      constexpr float log2e_f = 1.4426950408889634f;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDV_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEr; i++) {
          const short local_row = iq * 16 + sc.y + i * kErJ;
          const short row_idx = iq * kEr + i;
          const bool in_range = (!is_last_q) || (local_row < sg_lim_q);
          if (in_range) {
            lse_log2[row_idx] = L_qs[local_row * int(params.L_strides[2])] * log2e_f;
          } else {
            lse_log2[row_idx] = Limits<float>::finite_max;  // → P=0 for OOR rows
          }
        }
      }
    }

    // --- S = Q[SG-rows] @ K^T ---
    s_q_t Stile;
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWDV_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWDV_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWDV_TD; id++) {
          NAXTile<T, 1, 1> Qfrag;
          NAXTile<T, 2, 1> Kfrag;
          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          const int K_off = ik * 16 * int(params.K_strides[2]) + id * 16;
          if (is_last_q) {
            Qfrag.load_rows(Q_qs + Q_off, int(params.Q_strides[2]),
                            sg_lim_q - iq * 16);
          } else {
            Qfrag.load(Q_qs + Q_off, int(params.Q_strides[2]));
          }
          if (is_last_k) {
            Kfrag.load_rows(K + K_off, int(params.K_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Kfrag.load(K + K_off, int(params.K_strides[2]));
          }
          s_q_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qfrag.frag_at(0, 0),
              metal::false_type{},
              Kfrag.frag_at(0, 0),
              Kfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale into log2 domain.
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_q_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= V34BWDV_SCALE_LOG2E;
    }

    // Mask last-K columns to -inf.
    if (is_last_k) {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      const short sn = sc.x;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDV_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDV_TK; ik++) {
          const short col_pos = ik * 16 + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < lim_rows_k) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // v2.50 Sprint 4 Phase 4b-complete (Prompt 3) — causal mask for dV split.
    // K-parallel kernel with per-SG Q-row partition (sg_q_offset).
    //   base_row = qb * BQ + qL_off + sg_q_offset
    //   base_col = tid.x * BK
#if V34BWDV_CAUSAL
    {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc_c = s_q_t::NAXFrag_t::get_coord();
      const short sn_c = sc_c.x;
      const short sm_c = sc_c.y;
      const int base_row = qb * V34BWDV_BQ + params.qL_off + sg_q_offset;
      const int base_col = int(tid.x) * V34BWDV_BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDV_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDV_TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const int r = base_row + iq * 16
                          + ii * s_q_t::kFragRowsJump + sm_c;
              const int c = base_col + ik * 16 + jj + sn_c;
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }
#endif

    // P = exp2(S - lse_log2)
    Stile.template row_bin_op<ExpSubOp>(lse_log2);
    // Stile holds P now.

    // --- dV_accum += P^T @ dO[SG-rows] ---
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < V34BWDV_TK; ik++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V34BWDV_TD; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < V34BWDV_TQ; iq++) {
          NAXTile<T, 1, 2> dOfrag2;
          const int dO_off = iq * 16 * int(params.dO_strides[2]) + id * 16;
          if (is_last_q) {
            dOfrag2.load_rows(dO_qs + dO_off, int(params.dO_strides[2]),
                              sg_lim_q - iq * 16);
          } else {
            dOfrag2.load(dO_qs + dO_off, int(params.dO_strides[2]));
          }
          dv_t::NAXFrag_t::mma(
              dV_accum.frag_at(ik, id),
              dV_accum.frag_at(ik, id + 1),
              Stile.frag_at(iq, ik),
              metal::true_type{},   // transpose_a: P^T
              dOfrag2.frag_at(0, 0),
              dOfrag2.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    simdgroup_barrier(mem_flags::mem_none);
  }  // end Q-loop

  // === Store per-SG dV partial to its slot in dV_partials (FP32) ===
  if (is_last_k) {
    if (lim_rows_k <= 0) return;
    dV_accum.store_rows(dV_partials, int(params.dVp_strides[3]), lim_rows_k);
  } else {
    dV_accum.store(dV_partials, int(params.dVp_strides[3]));
  }
}
)BWDVMSL";

  return ss.str();
}



// =============================================================================
// V34 backward dV SPARSE kernel — Prompt 5b Section A PoC.
//
// Identical to createV34BackwardDVSource() with one structural addition: a
// per-Q-tile block_mask scan in the Q-loop.  When block_mask[qb, tid.x] is
// false, the entire Q-tile contribution to dV[k_base] is skipped (the
// inactive blocks DON'T contribute P^T @ dO since their P values are
// zero by the sparsity contract).
//
// Mask layout: 2-D (NQ, NK) bool — broadcast across (B, Hq).  This is the
// most common mask shape produced by `make_causal_block_mask` and
// `make_sliding_window_mask`.  Higher mask_ndim (3-D, 4-D) falls back to
// dense via Python-level routing.
//
// Sparse skip is uniform across the SIMD group (all threads check the same
// scalar block_mask entry) → zero warp divergence.  Pattern mirrors
// csrc/mfa_sparse_attention.cpp forward LCSA sparse scan.
//
// Output identical to dense dV kernel: dV_partials [B, Hq, WM, kL, D] FP32.
// =============================================================================
std::string NAAttentionKernel::createV34BackwardDVSparseSource() const noexcept {
  const int BQ = blockDimensions[0];
  const int BK = blockDimensions[1];
  const int BD = headDimension;
  const int WM = executionSIMDGroups;
  const int kU = 16;
  const int TQ_per_SG = BQ / (WM * kU);
  const int TD = BD / kU;
  const int TK = BK / kU;
  (void)TQ_per_SG; (void)TD; (void)TK;

  const bool is_bf16 =
      memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::BF16;
  const char* dtype_str = is_bf16 ? "bfloat" : "half";
  const float scale_log2e = scale * 1.4426950408889634f;

  std::ostringstream ss;
  ss << "// MFA_REQUIRE_MSL4\n";
  ss << "#include <metal_stdlib>\n";
  ss << "#include <metal_simdgroup>\n";
  ss << "#include <metal_simdgroup_matrix>\n";
  ss << "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  ss << "using namespace metal;\n";
  ss << "using namespace mpp::tensor_ops;\n";
  ss << "\n";

  ss << naxHelpersBlock();

  ss << "\nusing T = " << dtype_str << ";\n";
  ss << "using namespace mlx::steel;\n\n";
  ss << "#define V34BWDV_BQ " << BQ << "\n";
  ss << "#define V34BWDV_BK " << BK << "\n";
  ss << "#define V34BWDV_BD " << BD << "\n";
  ss << "#define V34BWDV_WM " << WM << "\n";
  ss << "#define V34BWDV_TQ " << TQ_per_SG << "\n";
  ss << "#define V34BWDV_TD " << TD << "\n";
  ss << "#define V34BWDV_TK " << TK << "\n";
  ss << "#define V34BWDV_SCALE_LOG2E " << scale_log2e << "f\n";
  ss << "#define V34BWDV_CAUSAL " << (isCausal ? 1 : 0) << "\n";
  ss << "#define V34BWDV_SPARSE 1\n";
  ss << "\n";

  ss << R"BWDVSPMSL(
struct V34BwdVSparseParams {
  int qL, kL;
  int gqa_factor;
  int NQ, NK;
  int qL_rem, kL_rem;
  int qL_off;
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long L_strides[3];
  long dO_strides[3];
  long dVp_strides[4];
};

[[kernel, max_total_threads_per_threadgroup(V34BWDV_WM * 32)]]
void attention_bwd_dv_sparse(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device float* L [[buffer(3)]],
    const device T* dO [[buffer(4)]],
    device float* dV_partials [[buffer(5)]],
    constant V34BwdVSparseParams& params [[buffer(6)]],
    const device bool* block_mask [[buffer(7)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {

  (void)simd_lane_id;

  ulong3 tidl{tid.x, tid.y, tid.z};
  ulong kv_head_idx = ulong(tid.y) / ulong(params.gqa_factor);

  Q  += tidl.z * params.Q_strides[0]  + tidl.y * params.Q_strides[1];
  dO += tidl.z * params.dO_strides[0] + tidl.y * params.dO_strides[1];
  L  += tidl.z * params.L_strides[0]  + tidl.y * params.L_strides[1];
  K  += tidl.z * params.K_strides[0]  + kv_head_idx * params.K_strides[1]
      + tidl.x * V34BWDV_BK * params.K_strides[2];
  V  += tidl.z * params.V_strides[0]  + kv_head_idx * params.V_strides[1]
      + tidl.x * V34BWDV_BK * params.V_strides[2];

  dV_partials += tidl.z * params.dVp_strides[0]
              +  tidl.y * params.dVp_strides[1]
              +  simd_group_id * params.dVp_strides[2]
              +  tidl.x * V34BWDV_BK * params.dVp_strides[3];

  const short sg_q_offset = 16 * V34BWDV_TQ * simd_group_id;

  const int NQ_aligned = params.qL / V34BWDV_BQ;
  const int NK_aligned = params.kL / V34BWDV_BK;
  const bool is_last_k = (int(tid.x) == NK_aligned);
  const short lim_rows_k = (params.kL_rem > 0 && is_last_k)
      ? params.kL_rem : V34BWDV_BK;
  const int nq_full = params.qL / V34BWDV_BQ;
  const int nq_rem = params.qL % V34BWDV_BQ;
  const int q_loop = nq_rem > 0 ? nq_full + 1 : nq_full;

  using dv_t = NAXTile<float, V34BWDV_TK, V34BWDV_TD>;
  dv_t dV_accum;
  dV_accum.clear();

  using s_q_t = NAXTile<float, V34BWDV_TQ, V34BWDV_TK>;
  constexpr short kRowsPT_q = s_q_t::kRowsPerThread;

  // === PROMPT 5B SECTION A SPARSE BASE: 2-D mask [NQ, NK] ===
  // Per-TG mask base pointer for this K-tile (tid.x).
  // 2-D mask broadcast across (B, Hq) → all TGs in (b, hq) see same column tid.x.
  const int NK_total = params.NK;

  for (int qb = 0; qb < q_loop; qb++) {
    // ===== SPARSE SKIP (Prompt 5b Section A PoC) =====
    // For full-blocks (qb < NQ_aligned), check block_mask[qb, tid.x].
    // For the last partial block (qb == NQ_aligned && qL_rem > 0), the mask
    // is sized for NQ_aligned and doesn't have an entry for the partial row.
    // We treat partial blocks as always-active (conservative — partial blocks
    // are at the boundary, sparse patterns rarely zero them).
    if (qb < NQ_aligned) {
      bool tile_active = block_mask[qb * NK_total + int(tid.x)];
      if (!tile_active) {
        // Skip entire Q-tile: P would be zero so P^T @ dO contributes zero.
        continue;
      }
    }
    // ===== END SPARSE SKIP =====

    const bool is_last_q = (qb == NQ_aligned);
    const short lim_rows_q_full = (params.qL_rem > 0 && is_last_q)
        ? params.qL_rem : V34BWDV_BQ;
    const short sg_lim_q = (short)max(0, (int)lim_rows_q_full - (int)sg_q_offset);
    if (is_last_q && sg_lim_q <= 0) continue;

    const device T* Q_qs  = Q  + qb * V34BWDV_BQ * int(params.Q_strides[2])
                              + sg_q_offset * int(params.Q_strides[2]);
    const device T* dO_qs = dO + qb * V34BWDV_BQ * int(params.dO_strides[2])
                              + sg_q_offset * int(params.dO_strides[2]);
    const device float* L_qs = L + qb * V34BWDV_BQ * int(params.L_strides[2])
                                + sg_q_offset * int(params.L_strides[2]);

    metal::vec<float, kRowsPT_q> lse_log2;
    {
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      constexpr short kEr = s_q_t::NAXFrag_t::kElemRows;
      constexpr short kErJ = s_q_t::NAXFrag_t::kElemRowsJump;
      constexpr float log2e_f = 1.4426950408889634f;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDV_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEr; i++) {
          const short local_row = iq * 16 + sc.y + i * kErJ;
          const short row_idx = iq * kEr + i;
          const bool in_range = (!is_last_q) || (local_row < sg_lim_q);
          if (in_range) {
            lse_log2[row_idx] = L_qs[local_row * int(params.L_strides[2])] * log2e_f;
          } else {
            lse_log2[row_idx] = Limits<float>::finite_max;
          }
        }
      }
    }

    s_q_t Stile;
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWDV_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWDV_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWDV_TD; id++) {
          NAXTile<T, 1, 1> Qfrag;
          NAXTile<T, 2, 1> Kfrag;
          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          const int K_off = ik * 16 * int(params.K_strides[2]) + id * 16;
          if (is_last_q) {
            Qfrag.load_rows(Q_qs + Q_off, int(params.Q_strides[2]),
                            sg_lim_q - iq * 16);
          } else {
            Qfrag.load(Q_qs + Q_off, int(params.Q_strides[2]));
          }
          if (is_last_k) {
            Kfrag.load_rows(K + K_off, int(params.K_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Kfrag.load(K + K_off, int(params.K_strides[2]));
          }
          s_q_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qfrag.frag_at(0, 0),
              metal::false_type{},
              Kfrag.frag_at(0, 0),
              Kfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_q_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= V34BWDV_SCALE_LOG2E;
    }

    if (is_last_k) {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      const short sn = sc.x;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDV_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDV_TK; ik++) {
          const short col_pos = ik * 16 + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < lim_rows_k) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

#if V34BWDV_CAUSAL
    {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc_c = s_q_t::NAXFrag_t::get_coord();
      const short sn_c = sc_c.x;
      const short sm_c = sc_c.y;
      const int base_row = qb * V34BWDV_BQ + params.qL_off + sg_q_offset;
      const int base_col = int(tid.x) * V34BWDV_BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDV_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDV_TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const int r = base_row + iq * 16
                          + ii * s_q_t::kFragRowsJump + sm_c;
              const int c = base_col + ik * 16 + jj + sn_c;
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }
#endif

    Stile.template row_bin_op<ExpSubOp>(lse_log2);

    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < V34BWDV_TK; ik++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V34BWDV_TD; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < V34BWDV_TQ; iq++) {
          NAXTile<T, 1, 2> dOfrag2;
          const int dO_off = iq * 16 * int(params.dO_strides[2]) + id * 16;
          if (is_last_q) {
            dOfrag2.load_rows(dO_qs + dO_off, int(params.dO_strides[2]),
                              sg_lim_q - iq * 16);
          } else {
            dOfrag2.load(dO_qs + dO_off, int(params.dO_strides[2]));
          }
          dv_t::NAXFrag_t::mma(
              dV_accum.frag_at(ik, id),
              dV_accum.frag_at(ik, id + 1),
              Stile.frag_at(iq, ik),
              metal::true_type{},
              dOfrag2.frag_at(0, 0),
              dOfrag2.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    simdgroup_barrier(mem_flags::mem_none);
  }

  if (is_last_k) {
    if (lim_rows_k <= 0) return;
    dV_accum.store_rows(dV_partials, int(params.dVp_strides[3]), lim_rows_k);
  } else {
    dV_accum.store(dV_partials, int(params.dVp_strides[3]));
  }
}
)BWDVSPMSL";

  return ss.str();
}


// =============================================================================
// V34 backward dK-only kernel — WM=4 Q-row partition with per-SG slot output.
// Phase 2.O2 sister kernel to createV34BackwardDVSource().  Adds D = rowsum(
// dO⊙O), dP = dO@V^T, dS = P*(dP-D), and dK_accum += dS^T@Q.
//
// Per-SG handles BQ/WM = 16 Q-rows (1 NAXFrag).  No softmax replication tax
// (intra-SG row-wise reductions).  Each SG writes its dK partial (full
// BK × D, contributions from its 16 Q-rows × NQ Q-tiles) to a unique slot
// in dK_partials [B, Hq, WM, kL, D] FP32.  Python wrapper reduces via
// mx.sum(axis=2) and casts to T.
//
// dK_accum is post-scaled by 1/sqrt(D) before storage (∇_K of QK^T*s = s*dS^T@Q).
// =============================================================================
std::string NAAttentionKernel::createV34BackwardDKSource() const noexcept {
  const int BQ = blockDimensions[0];
  const int BK = blockDimensions[1];
  const int BD = headDimension;
  const int WM = executionSIMDGroups;
  const int kU = 16;
  const int TQ_per_SG = BQ / (WM * kU);
  const int TD = BD / kU;
  const int TK = BK / kU;
  (void)TQ_per_SG; (void)TD; (void)TK;

  const bool is_bf16 =
      memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::BF16;
  const char* dtype_str = is_bf16 ? "bfloat" : "half";
  const float scale_log2e = scale * 1.4426950408889634f;

  std::ostringstream ss;
  ss << "// MFA_REQUIRE_MSL4\n";
  ss << "#include <metal_stdlib>\n";
  ss << "#include <metal_simdgroup>\n";
  ss << "#include <metal_simdgroup_matrix>\n";
  ss << "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  ss << "using namespace metal;\n";
  ss << "using namespace mpp::tensor_ops;\n\n";

  ss << naxHelpersBlock();

  ss << "\nusing T = " << dtype_str << ";\n";
  ss << "using namespace mlx::steel;\n\n";
  ss << "struct SubOp {\n";
  ss << "  template <typename U>\n";
  ss << "  METAL_FUNC static constexpr U apply(U x, U y) { return x - y; }\n";
  ss << "};\n\n";
  ss << "#define V34BWDK_BQ " << BQ << "\n";
  ss << "#define V34BWDK_BK " << BK << "\n";
  ss << "#define V34BWDK_BD " << BD << "\n";
  ss << "#define V34BWDK_WM " << WM << "\n";
  ss << "#define V34BWDK_TQ " << TQ_per_SG << "\n";
  ss << "#define V34BWDK_TD " << TD << "\n";
  ss << "#define V34BWDK_TK " << TK << "\n";
  ss << "#define V34BWDK_SCALE " << scale << "f\n";
  ss << "#define V34BWDK_SCALE_LOG2E " << scale_log2e << "f\n";
  // v2.50 Sprint 4 Phase 4b-complete (Prompt 3): causal masking macro
  ss << "#define V34BWDK_CAUSAL " << (isCausal ? 1 : 0) << "\n\n";

  ss << R"BWDKMSL(
struct V34BwdKParams {
  int qL, kL;
  int gqa_factor;
  int NQ, NK;
  int qL_rem, kL_rem;
  // v2.50 Sprint 4 Phase 4b-complete — causal offset (host struct match)
  int qL_off;
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long O_strides[3];
  long L_strides[3];
  long dO_strides[3];
  long dKp_strides[4];  // [B, Hq, WM, kL, D] FP32; D stride=1 implicit
  long D_strides[3];    // D=rowsum(dO⊙O) strides (FP32, [B, Hq, qL], v2.38.1)
};

[[kernel, max_total_threads_per_threadgroup(V34BWDK_WM * 32)]]
void attention_bwd_dk(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device T* O [[buffer(3)]],
    const device float* L [[buffer(4)]],
    const device T* dO [[buffer(5)]],
    device float* dK_partials [[buffer(6)]],
    constant V34BwdKParams& params [[buffer(7)]],
    device const float* D [[buffer(8)]],  // v2.38.1: precomputed rowsum(dO⊙O), [B,Hq,qL] FP32
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {

  (void)simd_lane_id;
  ulong3 tidl{tid.x, tid.y, tid.z};
  ulong kv_head_idx = ulong(tid.y) / ulong(params.gqa_factor);

  Q  += tidl.z * params.Q_strides[0]  + tidl.y * params.Q_strides[1];
  O  += tidl.z * params.O_strides[0]  + tidl.y * params.O_strides[1];
  dO += tidl.z * params.dO_strides[0] + tidl.y * params.dO_strides[1];
  L  += tidl.z * params.L_strides[0]  + tidl.y * params.L_strides[1];
  K  += tidl.z * params.K_strides[0]  + kv_head_idx * params.K_strides[1]
      + tidl.x * V34BWDK_BK * params.K_strides[2];
  V  += tidl.z * params.V_strides[0]  + kv_head_idx * params.V_strides[1]
      + tidl.x * V34BWDK_BK * params.V_strides[2];
  // v2.38.1: D buffer per-batch/per-Hq-head offset (D indexed by query head).
  D  += tidl.z * params.D_strides[0]  + tidl.y * params.D_strides[1];

  dK_partials += tidl.z * params.dKp_strides[0]
              +  tidl.y * params.dKp_strides[1]
              +  simd_group_id * params.dKp_strides[2]
              +  tidl.x * V34BWDK_BK * params.dKp_strides[3];

  const short sg_q_offset = 16 * V34BWDK_TQ * simd_group_id;

  const int NQ_aligned = params.qL / V34BWDK_BQ;
  const int NK_aligned = params.kL / V34BWDK_BK;
  const bool is_last_k = (int(tid.x) == NK_aligned);
  const short lim_rows_k = (params.kL_rem > 0 && is_last_k)
      ? params.kL_rem : V34BWDK_BK;
  const int nq_full = params.qL / V34BWDK_BQ;
  const int nq_rem = params.qL % V34BWDK_BQ;
  const int q_loop = nq_rem > 0 ? nq_full + 1 : nq_full;

  using dk_t = NAXTile<float, V34BWDK_TK, V34BWDK_TD>;
  dk_t dK_accum;
  dK_accum.clear();

  using s_q_t = NAXTile<float, V34BWDK_TQ, V34BWDK_TK>;
  constexpr short kRowsPT_q = s_q_t::kRowsPerThread;

  for (int qb = 0; qb < q_loop; qb++) {
    const bool is_last_q = (qb == NQ_aligned);
    const short lim_rows_q_full = (params.qL_rem > 0 && is_last_q)
        ? params.qL_rem : V34BWDK_BQ;
    const short sg_lim_q = (short)max(0, (int)lim_rows_q_full - (int)sg_q_offset);
    if (is_last_q && sg_lim_q <= 0) continue;

    const device T* Q_qs  = Q  + qb * V34BWDK_BQ * int(params.Q_strides[2])
                              + sg_q_offset * int(params.Q_strides[2]);
    const device T* O_qs  = O  + qb * V34BWDK_BQ * int(params.O_strides[2])
                              + sg_q_offset * int(params.O_strides[2]);
    const device T* dO_qs = dO + qb * V34BWDK_BQ * int(params.dO_strides[2])
                              + sg_q_offset * int(params.dO_strides[2]);
    const device float* L_qs = L + qb * V34BWDK_BQ * int(params.L_strides[2])
                                + sg_q_offset * int(params.L_strides[2]);
    // v2.38.1: D buffer Q-block + SG-row offset (mirror L_qs).
    const device float* D_qs = D + qb * V34BWDK_BQ * int(params.D_strides[2])
                                + sg_q_offset * int(params.D_strides[2]);

    // --- Load lse, scale to log2 domain ---
    metal::vec<float, kRowsPT_q> lse_log2;
    {
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      constexpr short kEr = s_q_t::NAXFrag_t::kElemRows;
      constexpr short kErJ = s_q_t::NAXFrag_t::kElemRowsJump;
      constexpr float log2e_f = 1.4426950408889634f;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDK_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEr; i++) {
          const short local_row = iq * 16 + sc.y + i * kErJ;
          const short row_idx = iq * kEr + i;
          const bool in_range = (!is_last_q) || (local_row < sg_lim_q);
          if (in_range) {
            lse_log2[row_idx] = L_qs[local_row * int(params.L_strides[2])] * log2e_f;
          } else {
            lse_log2[row_idx] = Limits<float>::finite_max;
          }
        }
      }
    }

    // --- v2.38.1: load D = rowsum(dO ⊙ O) from device buffer (precomputed) ---
    // D is precomputed on host via MLX, shared with dQ + legacy-fused kernels.
    // Mirrors the lse-load pattern above; coalesced reads across lanes.
    metal::vec<float, kRowsPT_q> D_vec;
    {
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      constexpr short kEr = s_q_t::NAXFrag_t::kElemRows;
      constexpr short kErJ = s_q_t::NAXFrag_t::kElemRowsJump;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDK_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEr; i++) {
          const short local_row = iq * 16 + sc.y + i * kErJ;
          const short row_idx = iq * kEr + i;
          const bool in_range = (!is_last_q) || (local_row < sg_lim_q);
          D_vec[row_idx] = in_range
              ? D_qs[local_row * int(params.D_strides[2])]
              : 0.0f;
        }
      }
    }

    // --- S = Q[SG-rows] @ K^T ---
    s_q_t Stile;
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWDK_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWDK_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWDK_TD; id++) {
          NAXTile<T, 1, 1> Qfrag;
          NAXTile<T, 2, 1> Kfrag;
          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          const int K_off = ik * 16 * int(params.K_strides[2]) + id * 16;
          if (is_last_q) {
            Qfrag.load_rows(Q_qs + Q_off, int(params.Q_strides[2]),
                            sg_lim_q - iq * 16);
          } else {
            Qfrag.load(Q_qs + Q_off, int(params.Q_strides[2]));
          }
          if (is_last_k) {
            Kfrag.load_rows(K + K_off, int(params.K_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Kfrag.load(K + K_off, int(params.K_strides[2]));
          }
          s_q_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qfrag.frag_at(0, 0),
              metal::false_type{},
              Kfrag.frag_at(0, 0),
              Kfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_q_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= V34BWDK_SCALE_LOG2E;
    }
    if (is_last_k) {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      const short sn = sc.x;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDK_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDK_TK; ik++) {
          const short col_pos = ik * 16 + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < lim_rows_k) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // v2.50 Sprint 4 Phase 4b-complete (Prompt 3) — causal mask for dK split.
    // K-parallel with per-SG Q-row partition (sg_q_offset).
#if V34BWDK_CAUSAL
    {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc_c = s_q_t::NAXFrag_t::get_coord();
      const short sn_c = sc_c.x;
      const short sm_c = sc_c.y;
      const int base_row = qb * V34BWDK_BQ + params.qL_off + sg_q_offset;
      const int base_col = int(tid.x) * V34BWDK_BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDK_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDK_TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const int r = base_row + iq * 16
                          + ii * s_q_t::kFragRowsJump + sm_c;
              const int c = base_col + ik * 16 + jj + sn_c;
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }
#endif

    Stile.template row_bin_op<ExpSubOp>(lse_log2);
    // Stile holds P.

    // --- dP = dO @ V^T ---
    using dp_t = NAXTile<float, V34BWDK_TQ, V34BWDK_TK>;
    dp_t dPtile;
    dPtile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWDK_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWDK_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWDK_TD; id++) {
          NAXTile<T, 1, 1> dOfrag;
          NAXTile<T, 2, 1> Vfrag;
          const int dO_off = iq * 16 * int(params.dO_strides[2]) + id * 16;
          const int V_off = ik * 16 * int(params.V_strides[2]) + id * 16;
          if (is_last_q) {
            dOfrag.load_rows(dO_qs + dO_off, int(params.dO_strides[2]),
                             sg_lim_q - iq * 16);
          } else {
            dOfrag.load(dO_qs + dO_off, int(params.dO_strides[2]));
          }
          if (is_last_k) {
            Vfrag.load_rows(V + V_off, int(params.V_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Vfrag.load(V + V_off, int(params.V_strides[2]));
          }
          dp_t::NAXFrag_t::mma(
              dPtile.frag_at(iq, ik),
              dPtile.frag_at(iq, ik + 1),
              dOfrag.frag_at(0, 0),
              metal::false_type{},
              Vfrag.frag_at(0, 0),
              Vfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }
    dPtile.template row_bin_op<SubOp>(D_vec);

    // dS = P * (dP - D); overwrites Stile.
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_q_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= dPtile.elems()[ii];
    }

    // --- dK_accum += dS^T @ Q[SG-rows] ---
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < V34BWDK_TK; ik++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V34BWDK_TD; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < V34BWDK_TQ; iq++) {
          NAXTile<T, 1, 2> Qfrag2;
          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          if (is_last_q) {
            Qfrag2.load_rows(Q_qs + Q_off, int(params.Q_strides[2]),
                             sg_lim_q - iq * 16);
          } else {
            Qfrag2.load(Q_qs + Q_off, int(params.Q_strides[2]));
          }
          dk_t::NAXFrag_t::mma(
              dK_accum.frag_at(ik, id),
              dK_accum.frag_at(ik, id + 1),
              Stile.frag_at(iq, ik),
              metal::true_type{},
              Qfrag2.frag_at(0, 0),
              Qfrag2.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    simdgroup_barrier(mem_flags::mem_none);
  }  // end Q-loop

  // dK *= scale
  {
    constexpr short kRowsPT_k = dk_t::kRowsPerThread;
    metal::vec<float, kRowsPT_k> scale_vec;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT_k; i++) scale_vec[i] = V34BWDK_SCALE;
    dK_accum.template row_bin_op<MulOp>(scale_vec);
  }

  if (is_last_k) {
    if (lim_rows_k <= 0) return;
    dK_accum.store_rows(dK_partials, int(params.dKp_strides[3]), lim_rows_k);
  } else {
    dK_accum.store(dK_partials, int(params.dKp_strides[3]));
  }
}
)BWDKMSL";

  return ss.str();
}


// =============================================================================
// V34 backward FUSED dK+dV kernel — Sprint v2.39.0 Phase C.1.a (Option γ).
//
// Combines split-dV (dV_accum += P^T @ dO) and split-dK (dK_accum += dS^T @ Q)
// into a single Q-loop that loads K/V tiles ONCE per K-tile across both
// gradient computations.  The structural ~10% perf win per /metal-kernel-dev
// audit (2026-05-13) is K-bandwidth amortization: split kernels each
// re-load K (split-dV) or K+V (split-dK), the fused kernel loads K+V once.
//
// CRITICAL ORDER (blueprint §"Order of operations"):
//   1. S = Q @ K^T (NAXFrag::mma into Stile FP32)
//   2. Scale to log2 domain; mask last-K columns to -inf
//   3. P = exp2(Stile - lse_log2)    ← Stile holds P
//   4. dV_accum += P^T @ dO          ← MUST use P BEFORE Stile is overwritten
//   5. dP = dO @ V^T (separate dPtile)
//   6. dPtile -= D_vec (row_bin_op<SubOp>)
//   7. dS = P * dPtile               ← overwrites Stile in place
//   8. dK_accum += dS^T @ Q
//
// Buffer map (follows v2.38.1 convention with D at last slot):
//   Q=0, K=1, V=2, L=3, dO=4, dK_partials=5, dV_partials=6, params=7, D=8
//
// D=64 only this PR; D=128 deferred to Phase C.1.b per audit staging
// (register pressure verification required via Metal frame capture).
//
// Grid (NK, Hq, B), TG size 32 * WM.  Per-SG-slot device writes to
// dK_partials + dV_partials [B, Hq, WM, kL, D] FP32 each.  Python wrapper
// reduces via mx.sum(axis=2) and casts to T.
// =============================================================================
std::string NAAttentionKernel::createV34BackwardFusedDKDVSource() const noexcept {
  const int BQ = blockDimensions[0];
  const int BK = blockDimensions[1];
  const int BD = headDimension;
  const int WM = executionSIMDGroups;
  const int kU = 16;
  const int TQ_per_SG = BQ / (WM * kU);
  const int TD = BD / kU;
  const int TK = BK / kU;
  (void)TQ_per_SG; (void)TD; (void)TK;

  const bool is_bf16 =
      memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::BF16;
  const char* dtype_str = is_bf16 ? "bfloat" : "half";
  const float scale_log2e = scale * 1.4426950408889634f;

  std::ostringstream ss;
  ss << "// MFA_REQUIRE_MSL4\n";
  ss << "#include <metal_stdlib>\n";
  ss << "#include <metal_simdgroup>\n";
  ss << "#include <metal_simdgroup_matrix>\n";
  ss << "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  ss << "using namespace metal;\n";
  ss << "using namespace mpp::tensor_ops;\n\n";

  ss << naxHelpersBlock();

  ss << "\nusing T = " << dtype_str << ";\n";
  ss << "using namespace mlx::steel;\n\n";
  ss << "struct SubOp {\n";
  ss << "  template <typename U>\n";
  ss << "  METAL_FUNC static constexpr U apply(U x, U y) { return x - y; }\n";
  ss << "};\n\n";
  ss << "#define V34BWDF_BQ " << BQ << "\n";
  ss << "#define V34BWDF_BK " << BK << "\n";
  ss << "#define V34BWDF_BD " << BD << "\n";
  ss << "#define V34BWDF_WM " << WM << "\n";
  ss << "#define V34BWDF_TQ " << TQ_per_SG << "\n";
  ss << "#define V34BWDF_TD " << TD << "\n";
  ss << "#define V34BWDF_TK " << TK << "\n";
  ss << "#define V34BWDF_SCALE " << scale << "f\n";
  ss << "#define V34BWDF_SCALE_LOG2E " << scale_log2e << "f\n";
  // v2.50 Sprint 4 Phase 4b-complete (Prompt 3): causal masking macro
  ss << "#define V34BWDF_CAUSAL " << (isCausal ? 1 : 0) << "\n\n";

  ss << R"BWDFMSL(
struct V34BwdFusedParams {
  int qL, kL;
  int gqa_factor;
  int NQ, NK;
  int qL_rem, kL_rem;
  // v2.50 Sprint 4 Phase 4b-complete — causal offset (host struct match)
  int qL_off;
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long L_strides[3];
  long dO_strides[3];
  long dKp_strides[4];  // [B, Hq, WM, kL, D] FP32; D stride=1 implicit
  long dVp_strides[4];  // [B, Hq, WM, kL, D] FP32; D stride=1 implicit
  long D_strides[3];    // D=rowsum(dO⊙O) strides (FP32, [B, Hq, qL])
};

[[kernel, max_total_threads_per_threadgroup(V34BWDF_WM * 32)]]
void attention_bwd_fused_dkdv(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device float* L [[buffer(3)]],
    const device T* dO [[buffer(4)]],
    device float* dK_partials [[buffer(5)]],
    device float* dV_partials [[buffer(6)]],
    constant V34BwdFusedParams& params [[buffer(7)]],
    device const float* D [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {

  (void)simd_lane_id;
  // Grid (NK, Hq, B). Each TG owns 1 K-tile, WM SGs partition Q-rows.

  ulong3 tidl{tid.x, tid.y, tid.z};
  ulong kv_head_idx = ulong(tid.y) / ulong(params.gqa_factor);

  Q  += tidl.z * params.Q_strides[0]  + tidl.y * params.Q_strides[1];
  dO += tidl.z * params.dO_strides[0] + tidl.y * params.dO_strides[1];
  L  += tidl.z * params.L_strides[0]  + tidl.y * params.L_strides[1];
  K  += tidl.z * params.K_strides[0]  + kv_head_idx * params.K_strides[1]
      + tidl.x * V34BWDF_BK * params.K_strides[2];
  V  += tidl.z * params.V_strides[0]  + kv_head_idx * params.V_strides[1]
      + tidl.x * V34BWDF_BK * params.V_strides[2];
  // D buffer indexed by query head (Hq), same as L.
  D  += tidl.z * params.D_strides[0]  + tidl.y * params.D_strides[1];

  // Per-SG dK_partials slot: [b, hq, sg, k_base, d] → 4 strides + D-stride=1.
  dK_partials += tidl.z * params.dKp_strides[0]
              +  tidl.y * params.dKp_strides[1]
              +  simd_group_id * params.dKp_strides[2]
              +  tidl.x * V34BWDF_BK * params.dKp_strides[3];
  dV_partials += tidl.z * params.dVp_strides[0]
              +  tidl.y * params.dVp_strides[1]
              +  simd_group_id * params.dVp_strides[2]
              +  tidl.x * V34BWDF_BK * params.dVp_strides[3];

  const short sg_q_offset = 16 * V34BWDF_TQ * simd_group_id;

  const int NQ_aligned = params.qL / V34BWDF_BQ;
  const int NK_aligned = params.kL / V34BWDF_BK;
  const bool is_last_k = (int(tid.x) == NK_aligned);
  const short lim_rows_k = (params.kL_rem > 0 && is_last_k)
      ? params.kL_rem : V34BWDF_BK;
  const int nq_full = params.qL / V34BWDF_BQ;
  const int nq_rem = params.qL % V34BWDF_BQ;
  const int q_loop = nq_rem > 0 ? nq_full + 1 : nq_full;

  // Per-SG accumulators (FP32, both persistent across Q-loop).
  using dk_t = NAXTile<float, V34BWDF_TK, V34BWDF_TD>;
  using dv_t = NAXTile<float, V34BWDF_TK, V34BWDF_TD>;
  dk_t dK_accum;
  dv_t dV_accum;
  dK_accum.clear();
  dV_accum.clear();

  using s_q_t = NAXTile<float, V34BWDF_TQ, V34BWDF_TK>;
  constexpr short kRowsPT_q = s_q_t::kRowsPerThread;

  // === Q-loop: fused dK + dV accumulation per K-tile ===
  for (int qb = 0; qb < q_loop; qb++) {
    const bool is_last_q = (qb == NQ_aligned);
    const short lim_rows_q_full = (params.qL_rem > 0 && is_last_q)
        ? params.qL_rem : V34BWDF_BQ;
    const short sg_lim_q = (short)max(0, (int)lim_rows_q_full - (int)sg_q_offset);
    if (is_last_q && sg_lim_q <= 0) continue;

    const device T* Q_qs  = Q  + qb * V34BWDF_BQ * int(params.Q_strides[2])
                              + sg_q_offset * int(params.Q_strides[2]);
    const device T* dO_qs = dO + qb * V34BWDF_BQ * int(params.dO_strides[2])
                              + sg_q_offset * int(params.dO_strides[2]);
    const device float* L_qs = L + qb * V34BWDF_BQ * int(params.L_strides[2])
                                + sg_q_offset * int(params.L_strides[2]);
    const device float* D_qs = D + qb * V34BWDF_BQ * int(params.D_strides[2])
                                + sg_q_offset * int(params.D_strides[2]);

    // --- Load lse for SG's 16 rows, scale to log2 domain ---
    metal::vec<float, kRowsPT_q> lse_log2;
    {
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      constexpr short kEr = s_q_t::NAXFrag_t::kElemRows;
      constexpr short kErJ = s_q_t::NAXFrag_t::kElemRowsJump;
      constexpr float log2e_f = 1.4426950408889634f;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDF_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEr; i++) {
          const short local_row = iq * 16 + sc.y + i * kErJ;
          const short row_idx = iq * kEr + i;
          const bool in_range = (!is_last_q) || (local_row < sg_lim_q);
          if (in_range) {
            lse_log2[row_idx] = L_qs[local_row * int(params.L_strides[2])] * log2e_f;
          } else {
            lse_log2[row_idx] = Limits<float>::finite_max;  // → P=0 for OOR
          }
        }
      }
    }

    // --- v2.38.1: load D = rowsum(dO ⊙ O) from device buffer ---
    metal::vec<float, kRowsPT_q> D_vec;
    {
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      constexpr short kEr = s_q_t::NAXFrag_t::kElemRows;
      constexpr short kErJ = s_q_t::NAXFrag_t::kElemRowsJump;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDF_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEr; i++) {
          const short local_row = iq * 16 + sc.y + i * kErJ;
          const short row_idx = iq * kEr + i;
          const bool in_range = (!is_last_q) || (local_row < sg_lim_q);
          D_vec[row_idx] = in_range
              ? D_qs[local_row * int(params.D_strides[2])]
              : 0.0f;
        }
      }
    }

    // --- S = Q[SG-rows] @ K^T ---
    s_q_t Stile;
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWDF_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWDF_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWDF_TD; id++) {
          NAXTile<T, 1, 1> Qfrag;
          NAXTile<T, 2, 1> Kfrag;
          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          const int K_off = ik * 16 * int(params.K_strides[2]) + id * 16;
          if (is_last_q) {
            Qfrag.load_rows(Q_qs + Q_off, int(params.Q_strides[2]),
                            sg_lim_q - iq * 16);
          } else {
            Qfrag.load(Q_qs + Q_off, int(params.Q_strides[2]));
          }
          if (is_last_k) {
            Kfrag.load_rows(K + K_off, int(params.K_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Kfrag.load(K + K_off, int(params.K_strides[2]));
          }
          s_q_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qfrag.frag_at(0, 0),
              metal::false_type{},
              Kfrag.frag_at(0, 0),
              Kfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale into log2 domain.
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_q_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= V34BWDF_SCALE_LOG2E;
    }

    // Mask last-K columns to -inf.
    if (is_last_k) {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      const short sn = sc.x;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDF_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDF_TK; ik++) {
          const short col_pos = ik * 16 + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < lim_rows_k) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // v2.50 Sprint 4 Phase 4b-complete (Prompt 3) — causal mask for fused
    // dKdV.  K-parallel with per-SG Q-row partition (sg_q_offset).
    // ORDER-CRITICAL preserved: mask is on Stile holding S, before
    // row_bin_op<ExpSubOp> converts to P, before P^T @ dO uses P.
#if V34BWDF_CAUSAL
    {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc_c = s_q_t::NAXFrag_t::get_coord();
      const short sn_c = sc_c.x;
      const short sm_c = sc_c.y;
      const int base_row = qb * V34BWDF_BQ + params.qL_off + sg_q_offset;
      const int base_col = int(tid.x) * V34BWDF_BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDF_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDF_TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const int r = base_row + iq * 16
                          + ii * s_q_t::kFragRowsJump + sm_c;
              const int c = base_col + ik * 16 + jj + sn_c;
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }
#endif

    // P = exp2(S - lse_log2).  Stile holds P after this.
    Stile.template row_bin_op<ExpSubOp>(lse_log2);

    // === ORDER-CRITICAL: dV_accum += P^T @ dO BEFORE Stile is overwritten ===
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < V34BWDF_TK; ik++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V34BWDF_TD; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < V34BWDF_TQ; iq++) {
          NAXTile<T, 1, 2> dOfrag2;
          const int dO_off = iq * 16 * int(params.dO_strides[2]) + id * 16;
          if (is_last_q) {
            dOfrag2.load_rows(dO_qs + dO_off, int(params.dO_strides[2]),
                              sg_lim_q - iq * 16);
          } else {
            dOfrag2.load(dO_qs + dO_off, int(params.dO_strides[2]));
          }
          dv_t::NAXFrag_t::mma(
              dV_accum.frag_at(ik, id),
              dV_accum.frag_at(ik, id + 1),
              Stile.frag_at(iq, ik),
              metal::true_type{},   // transpose_a: P^T
              dOfrag2.frag_at(0, 0),
              dOfrag2.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    // --- dP = dO @ V^T ---
    using dp_t = NAXTile<float, V34BWDF_TQ, V34BWDF_TK>;
    dp_t dPtile;
    dPtile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWDF_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWDF_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWDF_TD; id++) {
          NAXTile<T, 1, 1> dOfrag;
          NAXTile<T, 2, 1> Vfrag;
          const int dO_off = iq * 16 * int(params.dO_strides[2]) + id * 16;
          const int V_off = ik * 16 * int(params.V_strides[2]) + id * 16;
          if (is_last_q) {
            dOfrag.load_rows(dO_qs + dO_off, int(params.dO_strides[2]),
                             sg_lim_q - iq * 16);
          } else {
            dOfrag.load(dO_qs + dO_off, int(params.dO_strides[2]));
          }
          if (is_last_k) {
            Vfrag.load_rows(V + V_off, int(params.V_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Vfrag.load(V + V_off, int(params.V_strides[2]));
          }
          dp_t::NAXFrag_t::mma(
              dPtile.frag_at(iq, ik),
              dPtile.frag_at(iq, ik + 1),
              dOfrag.frag_at(0, 0),
              metal::false_type{},
              Vfrag.frag_at(0, 0),
              Vfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }
    // dP -= D_vec (per-row broadcast).
    dPtile.template row_bin_op<SubOp>(D_vec);

    // dS = P * (dP - D); overwrites Stile in place (P is consumed).
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_q_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= dPtile.elems()[ii];
    }

    // --- dK_accum += dS^T @ Q[SG-rows] ---
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < V34BWDF_TK; ik++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V34BWDF_TD; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < V34BWDF_TQ; iq++) {
          NAXTile<T, 1, 2> Qfrag2;
          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          if (is_last_q) {
            Qfrag2.load_rows(Q_qs + Q_off, int(params.Q_strides[2]),
                             sg_lim_q - iq * 16);
          } else {
            Qfrag2.load(Q_qs + Q_off, int(params.Q_strides[2]));
          }
          dk_t::NAXFrag_t::mma(
              dK_accum.frag_at(ik, id),
              dK_accum.frag_at(ik, id + 1),
              Stile.frag_at(iq, ik),
              metal::true_type{},
              Qfrag2.frag_at(0, 0),
              Qfrag2.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    simdgroup_barrier(mem_flags::mem_none);
  }  // end Q-loop

  // dK *= scale (matches split-dK; dV is not scaled — P already absorbs it).
  {
    constexpr short kRowsPT_k = dk_t::kRowsPerThread;
    metal::vec<float, kRowsPT_k> scale_vec;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT_k; i++) scale_vec[i] = V34BWDF_SCALE;
    dK_accum.template row_bin_op<MulOp>(scale_vec);
  }

  // === Per-SG-slot device write for both gradients ===
  if (is_last_k) {
    if (lim_rows_k <= 0) return;
    dK_accum.store_rows(dK_partials, int(params.dKp_strides[3]), lim_rows_k);
    dV_accum.store_rows(dV_partials, int(params.dVp_strides[3]), lim_rows_k);
  } else {
    dK_accum.store(dK_partials, int(params.dKp_strides[3]));
    dV_accum.store(dV_partials, int(params.dVp_strides[3]));
  }
}
)BWDFMSL";

  return ss.str();
}

std::string NAAttentionKernel::createV34BackwardQuerySparseSource() const noexcept {
  const int BQ = blockDimensions[0];
  const int BK = blockDimensions[1];
  const int BD = headDimension;
  const int WM = executionSIMDGroups;
  const int kU = 16;
  const int TQ = BQ / (WM * kU);
  const int TD = BD / kU;
  const int TK = BK / kU;
  (void)TQ; (void)TD; (void)TK;

  const bool is_bf16 =
      memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::BF16;
  const char* dtype_str = is_bf16 ? "bfloat" : "half";
  // Natural-domain scale (1/sqrt(D)); precompute log2-domain version.
  const float scale_log2e = scale * 1.4426950408889634f;

  std::ostringstream ss;
  ss << "// MFA_REQUIRE_MSL4\n";
  ss << "#include <metal_stdlib>\n";
  ss << "#include <metal_simdgroup>\n";
  ss << "#include <metal_simdgroup_matrix>\n";
  ss << "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  ss << "using namespace metal;\n";
  ss << "using namespace mpp::tensor_ops;\n";
  ss << "\n";
  ss << "// === Apple NAX helpers (shared via naxHelpersBlock(), extracted Sprint v2.38.x Phase B) ===\n";

  ss << naxHelpersBlock();

  ss << "\n// V34 backward dQ kernel — Apple-style NAX-direct\n";
  ss << "using T = " << dtype_str << ";\n";
  ss << "using namespace mlx::steel;\n";
  ss << "\n";
  ss << "// SubOp functor (needed for row_bin_op<SubOp> in backward; not in\n";
  ss << "// V34 forward helpers because forward doesn't use plain subtraction).\n";
  ss << "struct SubOp {\n";
  ss << "  template <typename U>\n";
  ss << "  METAL_FUNC static constexpr U apply(U x, U y) { return x - y; }\n";
  ss << "};\n\n";
  ss << "#define V34BWD_BQ " << BQ << "\n";
  ss << "#define V34BWD_BK " << BK << "\n";
  ss << "#define V34BWD_BD " << BD << "\n";
  ss << "#define V34BWD_WM " << WM << "\n";
  ss << "#define V34BWD_TQ " << TQ << "\n";
  ss << "#define V34BWD_TD " << TD << "\n";
  ss << "#define V34BWD_TK " << TK << "\n";
  ss << "#define V34BWD_SCALE " << scale << "f\n";
  ss << "#define V34BWD_SCALE_LOG2E " << scale_log2e << "f\n";
  // v2.50 Sprint 4 Phase 4b — causal masking baked in as compile-time
  // constant so non-causal source remains bit-identical to pre-Sprint-4.
  ss << "#define V34BWD_CAUSAL " << (isCausal ? 1 : 0) << "\n";
  ss << "#define V34BWDQ_SPARSE 1\n";
  ss << "\n";
  ss << R"BWDQSPMSL(
struct V34BwdQParams {
  int qL;
  int kL;
  int gqa_factor;
  int NQ;
  int NK;
  int qL_rem;
  int kL_rem;
  // v2.50 Sprint 4 Phase 4b — causal offset.  Field order MUST match
  // V34BwdQParamsHost in v6_nax_compile.mm.
  int qL_off;
  // BHND strides (sequence stride = D, encoded in stride[2]).
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long O_strides[3];
  long L_strides[3];   // lse strides (FP32, [B, Hq, qL])
  long dO_strides[3];  // dO strides (same layout as Q)
  long dQ_strides[3];  // dQ strides (same layout as Q)
  long D_strides[3];   // D=rowsum(dO⊙O) strides (FP32, [B, Hq, qL], v2.38.1)
};

[[kernel, max_total_threads_per_threadgroup(V34BWD_WM * 32)]]
void attention_bwd_q_sparse(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device T* O [[buffer(3)]],
    const device float* L [[buffer(4)]],
    const device T* dO [[buffer(5)]],
    device T* dQ [[buffer(6)]],
    constant V34BwdQParams& params [[buffer(7)]],
    device const float* D [[buffer(8)]],  // v2.38.1: precomputed rowsum(dO⊙O), [B,Hq,qL] FP32
    const device bool* block_mask [[buffer(9)]],  // v2.50 Prompt 5d: 2-D [NQ, NK] sparse mask
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {

  (void)simd_lane_id;

  // === Per-batch + per-head + per-Q-block ptr offsets ===
  ulong3 tidl{tid.x, tid.y, tid.z};
  Q  += tidl.z * params.Q_strides[0]
      + tidl.y * params.Q_strides[1]
      + tidl.x * V34BWD_BQ * params.Q_strides[2];
  ulong kv_head_idx = ulong(tid.y) / ulong(params.gqa_factor);
  K  += tidl.z * params.K_strides[0] + kv_head_idx * params.K_strides[1];
  V  += tidl.z * params.V_strides[0] + kv_head_idx * params.V_strides[1];
  O  += tidl.z * params.O_strides[0]
      + tidl.y * params.O_strides[1]
      + tidl.x * V34BWD_BQ * params.O_strides[2];
  L  += tidl.z * params.L_strides[0]
      + tidl.y * params.L_strides[1]
      + tidl.x * V34BWD_BQ * params.L_strides[2];
  dO += tidl.z * params.dO_strides[0]
      + tidl.y * params.dO_strides[1]
      + tidl.x * V34BWD_BQ * params.dO_strides[2];
  dQ += tidl.z * params.dQ_strides[0]
      + tidl.y * params.dQ_strides[1]
      + tidl.x * V34BWD_BQ * params.dQ_strides[2];
  // v2.38.1: D buffer per-batch/per-head/per-Q-block offset.
  D  += tidl.z * params.D_strides[0]
      + tidl.y * params.D_strides[1]
      + tidl.x * V34BWD_BQ * params.D_strides[2];

  // Per-SG row offset within the Q-block.
  const short tm = 16 * V34BWD_TQ * simd_group_id;
  Q  += tm * int(params.Q_strides[2]);
  O  += tm * int(params.O_strides[2]);
  L  += tm * int(params.L_strides[2]);
  dO += tm * int(params.dO_strides[2]);
  dQ += tm * int(params.dQ_strides[2]);
  D  += tm * int(params.D_strides[2]);  // v2.38.1

  // Last-block flags.
  const int NQ_aligned = params.qL / V34BWD_BQ;
  const int NK_aligned = params.kL / V34BWD_BK;
  const bool is_last_q = (int(tid.x) == NQ_aligned);
  const short lim_rows_q = (params.qL_rem > 0 ? params.qL_rem : V34BWD_BQ) - tm;
  const short lim_rows_k = (params.kL_rem > 0 ? params.kL_rem : V34BWD_BK);
  const int kb_lim = params.NK;

  // === MMA tile types ===
  using dq_accum_t = NAXTile<float, V34BWD_TQ, V34BWD_TD>;  // dQ FP32 accumulator
  using s_t       = NAXTile<float, V34BWD_TQ, V34BWD_TK>;  // S (= Q@K^T scaled)
  using dp_t      = NAXTile<float, V34BWD_TQ, V34BWD_TK>;  // dP (= dO@V^T)

  dq_accum_t dQ_accum;
  dQ_accum.clear();

  constexpr short kRowsPT = dq_accum_t::kRowsPerThread;

  // === Step 1: load lse, convert to log2 domain ===
  // lse from forward is natural-log; V34 inner-loop uses log2 domain via
  // scale*log2(e) and exp2.  Multiply lse by log2(e) once so that
  // row_bin_op<ExpSubOp>(lse_log2) below computes exp2(S_log2 - lse_log2)
  // = exp(S_natural - lse_natural) = correct softmax P.
  metal::vec<float, kRowsPT> lse_log2;
  {
    // Each lane owns kRowsPT rows in the SG; load each row's lse and scale.
    // Lane → row mapping uses get_coord() the same way row_reduce maps
    // partial-sum results.  Convention: lane with fn==0 reads lse[row].
    // Other lanes need to receive the value too (so each thread's lse_log2
    // vec is populated correctly for row_bin_op).  Use simd_shuffle_xor or
    // simd_broadcast to share.  Simpler approach: ALL lanes load lse[row]
    // for their owned rows (redundant device reads but correct).  4 lanes
    // covering the same row will read the same memory — coalesced, cheap.
    const short2 sc = dq_accum_t::NAXFrag_t::get_coord();
    constexpr short kElemRows = dq_accum_t::NAXFrag_t::kElemRows;
    constexpr short kElemRowsJump = dq_accum_t::NAXFrag_t::kElemRowsJump;
    constexpr float log2e_f = 1.4426950408889634f;
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWD_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kElemRows; i++) {
        const short local_row = iq * 16 + sc.y + i * kElemRowsJump;
        const short row_idx = iq * kElemRows + i;
        const bool in_range = (!is_last_q) || (local_row < lim_rows_q);
        // Out-of-range rows: set lse_log2 to +inf so exp2(S - inf) = 0,
        // which gives P=0, dS=0, no contribution to dQ for those rows.
        if (in_range) {
          lse_log2[row_idx] = L[local_row * int(params.L_strides[2])] * log2e_f;
        } else {
          lse_log2[row_idx] = Limits<float>::finite_max;  // ~+inf
        }
      }
    }
  }

  // === Step 2: load D[i] = rowsum(dO[i] ⊙ O[i]) from device buffer ===
  // v2.38.1: D is precomputed once on host via MLX (`mx.sum(dO*O, axis=-1)`)
  // and shared between dQ + split-dK + legacy-fused-dKdV kernels.  Replaces
  // an inline tile load + FP32 multiply + row_reduce.  Saves 1 rowsum per
  // V34 backward dQ call.
  //
  // Mirrors the lse-load pattern above (Step 1): each lane reads its owned
  // rows from the device buffer using `NAXFrag::get_coord()` + kElemRows /
  // kElemRowsJump.  Multiple lanes covering the same row → coalesced read.
  metal::vec<float, kRowsPT> D_vec;
  {
    const short2 sc = dq_accum_t::NAXFrag_t::get_coord();
    constexpr short kElemRows = dq_accum_t::NAXFrag_t::kElemRows;
    constexpr short kElemRowsJump = dq_accum_t::NAXFrag_t::kElemRowsJump;
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWD_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kElemRows; i++) {
        const short local_row = iq * 16 + sc.y + i * kElemRowsJump;
        const short row_idx = iq * kElemRows + i;
        const bool in_range = (!is_last_q) || (local_row < lim_rows_q);
        // Out-of-range rows: D=0 → contributes 0 to dS (P=0 there anyway
        // via the lse=+inf trick above).
        D_vec[row_idx] = in_range
            ? D[local_row * int(params.D_strides[2])]
            : 0.0f;
      }
    }
  }

  // === Step 3: K-loop ===
  for (int kb = 0; kb < kb_lim; kb++) {
    // === v2.50 Prompt 5d Section A sparse-skip (2-D mask [NQ, NK]) ===
    {
      const int qb_idx = int(tid.x);
      const int nk_total = params.NK;
      bool tile_active = block_mask[qb_idx * nk_total + kb];
      if (!tile_active) {
        K += V34BWD_BK * int(params.K_strides[2]);
        V += V34BWD_BK * int(params.V_strides[2]);
        continue;
      }
    }
    const bool is_last_k = (kb == NK_aligned);

    s_t Stile;
    Stile.clear();

    // QK matmul: S = Q @ K^T (NAXFrag::mma, transpose_b=true).
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWD_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWD_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWD_TD; id++) {
          NAXTile<T, 1, 1> Qfrag;
          NAXTile<T, 2, 1> Kfrag;

          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          const int K_off = ik * 16 * int(params.K_strides[2]) + id * 16;

          if (is_last_q) {
            Qfrag.load_rows(Q + Q_off, int(params.Q_strides[2]),
                            lim_rows_q - iq * 16);
          } else {
            Qfrag.load(Q + Q_off, int(params.Q_strides[2]));
          }
          if (is_last_k) {
            Kfrag.load_rows(K + K_off, int(params.K_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Kfrag.load(K + K_off, int(params.K_strides[2]));
          }

          s_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qfrag.frag_at(0, 0),
              metal::false_type{},
              Kfrag.frag_at(0, 0),
              Kfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale S into log2 domain.
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= V34BWD_SCALE_LOG2E;
    }

    // Mask out length sequence on last K block (mirrors forward, but here
    // out-of-range K columns must produce P=0 so they don't contribute to
    // dQ or D).  Setting S to -inf yields exp2(-inf - lse) = 0.
    if (is_last_k) {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc = s_t::NAXFrag_t::get_coord();
      const short sn = sc.x;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWD_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWD_TK; ik++) {
          const short col_pos = ik * 16 + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_t::kFragThrCols; jj++) {
              const auto loc = ii * s_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < lim_rows_k) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // v2.50 Sprint 4 Phase 4b — causal mask (mirror forward).  Setting
    // S[r,c] = -inf for r<c → exp2(-inf - lse) = 0 → P[r,c] = 0 → dS = 0
    // at masked positions → dQ accumulation naturally skips them.  Without
    // this, backward computes P over the unmasked S using the causal-masked
    // lse from forward, producing huge (incorrect) gradients for c>r.
#if V34BWD_CAUSAL
    {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc_c = s_t::NAXFrag_t::get_coord();
      const short sn_c = sc_c.x;   // col base within fragment
      const short sm_c = sc_c.y;   // row base within fragment
      const int base_row = int(tid.x) * V34BWD_BQ + params.qL_off + tm;
      const int base_col = kb * V34BWD_BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWD_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWD_TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_t::kFragThrCols; jj++) {
              const int r = base_row + iq * 16
                          + ii * s_t::kFragRowsJump + sm_c;
              const int c = base_col + ik * 16 + jj + sn_c;
              const auto loc = ii * s_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }
#endif

    // P = exp2(S - lse_log2)  (so P[i, j] = softmax_j(S_natural[i, .]))
    Stile.template row_bin_op<ExpSubOp>(lse_log2);
    // Stile now holds P in registers.

    // dP = dO @ V^T (NAXFrag::mma, transpose_b=true).
    dp_t dPtile;
    dPtile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWD_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWD_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWD_TD; id++) {
          NAXTile<T, 1, 1> dOfrag;
          NAXTile<T, 2, 1> Vfrag;

          const int dO_off = iq * 16 * int(params.dO_strides[2]) + id * 16;
          const int V_off = ik * 16 * int(params.V_strides[2]) + id * 16;

          if (is_last_q) {
            dOfrag.load_rows(dO + dO_off, int(params.dO_strides[2]),
                             lim_rows_q - iq * 16);
          } else {
            dOfrag.load(dO + dO_off, int(params.dO_strides[2]));
          }
          if (is_last_k) {
            Vfrag.load_rows(V + V_off, int(params.V_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Vfrag.load(V + V_off, int(params.V_strides[2]));
          }

          dp_t::NAXFrag_t::mma(
              dPtile.frag_at(iq, ik),
              dPtile.frag_at(iq, ik + 1),
              dOfrag.frag_at(0, 0),
              metal::false_type{},
              Vfrag.frag_at(0, 0),
              Vfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // dP -= D_vec (broadcast across cols of each row).
    dPtile.template row_bin_op<SubOp>(D_vec);

    // dS = P * (dP - D)  (element-wise; Stile holds P, dPtile holds dP-D).
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= dPtile.elems()[ii];
    }
    // Stile now holds dS.

    simdgroup_barrier(mem_flags::mem_none);

    // dQ_accum += dS @ K  (NAXFrag::mma, transpose_b=false).  Mirrors the
    // P @ V pattern from V34 forward.
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWD_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V34BWD_TD; id += 2) {
        if (V34BWD_BD == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWD_TK; ik++) {
          NAXTile<T, 1, 2> Kfrag2;
          const int K_off = ik * 16 * int(params.K_strides[2]) + id * 16;
          if (is_last_k) {
            Kfrag2.load_rows(K + K_off, int(params.K_strides[2]),
                             lim_rows_k - ik * 16);
          } else {
            Kfrag2.load(K + K_off, int(params.K_strides[2]));
          }
          dq_accum_t::NAXFrag_t::mma(
              dQ_accum.frag_at(iq, id),
              dQ_accum.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Kfrag2.frag_at(0, 0),
              Kfrag2.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    K += V34BWD_BK * int(params.K_strides[2]);
    V += V34BWD_BK * int(params.V_strides[2]);
  }  // end K-loop

  threadgroup_barrier(mem_flags::mem_none);

  // === Step 4: dQ_accum *= scale  (= 1/sqrt(D)) ===
  // Convention: dQ = ∇_Q (Q @ K^T * scale) backward = scale * (dS @ K).
  // We accumulated dS @ K above; multiply by scale here.
  {
    metal::vec<float, kRowsPT> scale_vec;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; i++) scale_vec[i] = V34BWD_SCALE;
    dQ_accum.template row_bin_op<MulOp>(scale_vec);
  }

  // === Step 5: store dQ ===
  // dQ_accum is FP32 NAXTile [TQ × TD]; store as T (FP16/BF16) to device.
  if (is_last_q) {
    if (lim_rows_q <= 0) return;
    dQ_accum.store_rows(dQ, int(params.dQ_strides[2]), lim_rows_q);
  } else {
    dQ_accum.store(dQ, int(params.dQ_strides[2]));
  }
}
)BWDQSPMSL";

  return ss.str();
}
std::string NAAttentionKernel::createV34BackwardDKSparseSource() const noexcept {
  const int BQ = blockDimensions[0];
  const int BK = blockDimensions[1];
  const int BD = headDimension;
  const int WM = executionSIMDGroups;
  const int kU = 16;
  const int TQ_per_SG = BQ / (WM * kU);
  const int TD = BD / kU;
  const int TK = BK / kU;
  (void)TQ_per_SG; (void)TD; (void)TK;

  const bool is_bf16 =
      memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::BF16;
  const char* dtype_str = is_bf16 ? "bfloat" : "half";
  const float scale_log2e = scale * 1.4426950408889634f;

  std::ostringstream ss;
  ss << "// MFA_REQUIRE_MSL4\n";
  ss << "#include <metal_stdlib>\n";
  ss << "#include <metal_simdgroup>\n";
  ss << "#include <metal_simdgroup_matrix>\n";
  ss << "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  ss << "using namespace metal;\n";
  ss << "using namespace mpp::tensor_ops;\n\n";

  ss << naxHelpersBlock();

  ss << "\nusing T = " << dtype_str << ";\n";
  ss << "using namespace mlx::steel;\n\n";
  ss << "struct SubOp {\n";
  ss << "  template <typename U>\n";
  ss << "  METAL_FUNC static constexpr U apply(U x, U y) { return x - y; }\n";
  ss << "};\n\n";
  ss << "#define V34BWDK_BQ " << BQ << "\n";
  ss << "#define V34BWDK_BK " << BK << "\n";
  ss << "#define V34BWDK_BD " << BD << "\n";
  ss << "#define V34BWDK_WM " << WM << "\n";
  ss << "#define V34BWDK_TQ " << TQ_per_SG << "\n";
  ss << "#define V34BWDK_TD " << TD << "\n";
  ss << "#define V34BWDK_TK " << TK << "\n";
  ss << "#define V34BWDK_SCALE " << scale << "f\n";
  ss << "#define V34BWDK_SCALE_LOG2E " << scale_log2e << "f\n";
  // v2.50 Sprint 4 Phase 4b-complete (Prompt 3): causal masking macro
  ss << "#define V34BWDK_CAUSAL " << (isCausal ? 1 : 0) << "\n\n";

  ss << R"BWDKSPMSL(
struct V34BwdKParams {
  int qL, kL;
  int gqa_factor;
  int NQ, NK;
  int qL_rem, kL_rem;
  // v2.50 Sprint 4 Phase 4b-complete — causal offset (host struct match)
  int qL_off;
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long O_strides[3];
  long L_strides[3];
  long dO_strides[3];
  long dKp_strides[4];  // [B, Hq, WM, kL, D] FP32; D stride=1 implicit
  long D_strides[3];    // D=rowsum(dO⊙O) strides (FP32, [B, Hq, qL], v2.38.1)
};

[[kernel, max_total_threads_per_threadgroup(V34BWDK_WM * 32)]]
void attention_bwd_dk_sparse(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device T* O [[buffer(3)]],
    const device float* L [[buffer(4)]],
    const device T* dO [[buffer(5)]],
    device float* dK_partials [[buffer(6)]],
    constant V34BwdKParams& params [[buffer(7)]],
    device const float* D [[buffer(8)]],
    const device bool* block_mask [[buffer(9)]],  // v2.50 Prompt 5d: 2-D sparse mask  // v2.38.1: precomputed rowsum(dO⊙O), [B,Hq,qL] FP32
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {

  (void)simd_lane_id;
  ulong3 tidl{tid.x, tid.y, tid.z};
  ulong kv_head_idx = ulong(tid.y) / ulong(params.gqa_factor);

  Q  += tidl.z * params.Q_strides[0]  + tidl.y * params.Q_strides[1];
  O  += tidl.z * params.O_strides[0]  + tidl.y * params.O_strides[1];
  dO += tidl.z * params.dO_strides[0] + tidl.y * params.dO_strides[1];
  L  += tidl.z * params.L_strides[0]  + tidl.y * params.L_strides[1];
  K  += tidl.z * params.K_strides[0]  + kv_head_idx * params.K_strides[1]
      + tidl.x * V34BWDK_BK * params.K_strides[2];
  V  += tidl.z * params.V_strides[0]  + kv_head_idx * params.V_strides[1]
      + tidl.x * V34BWDK_BK * params.V_strides[2];
  // v2.38.1: D buffer per-batch/per-Hq-head offset (D indexed by query head).
  D  += tidl.z * params.D_strides[0]  + tidl.y * params.D_strides[1];

  dK_partials += tidl.z * params.dKp_strides[0]
              +  tidl.y * params.dKp_strides[1]
              +  simd_group_id * params.dKp_strides[2]
              +  tidl.x * V34BWDK_BK * params.dKp_strides[3];

  const short sg_q_offset = 16 * V34BWDK_TQ * simd_group_id;

  const int NQ_aligned = params.qL / V34BWDK_BQ;
  const int NK_aligned = params.kL / V34BWDK_BK;
  const bool is_last_k = (int(tid.x) == NK_aligned);
  const short lim_rows_k = (params.kL_rem > 0 && is_last_k)
      ? params.kL_rem : V34BWDK_BK;
  const int nq_full = params.qL / V34BWDK_BQ;
  const int nq_rem = params.qL % V34BWDK_BQ;
  const int q_loop = nq_rem > 0 ? nq_full + 1 : nq_full;

  using dk_t = NAXTile<float, V34BWDK_TK, V34BWDK_TD>;
  dk_t dK_accum;
  dK_accum.clear();

  using s_q_t = NAXTile<float, V34BWDK_TQ, V34BWDK_TK>;
  constexpr short kRowsPT_q = s_q_t::kRowsPerThread;

  for (int qb = 0; qb < q_loop; qb++) {
    // === v2.50 Prompt 5d Section A sparse-skip ===
    if (qb < (params.qL / V34BWDK_BQ)) {
      const int nk_total = params.NK;
      bool tile_active = block_mask[qb * nk_total + int(tid.x)];
      if (!tile_active) continue;
    }
    const bool is_last_q = (qb == NQ_aligned);
    const short lim_rows_q_full = (params.qL_rem > 0 && is_last_q)
        ? params.qL_rem : V34BWDK_BQ;
    const short sg_lim_q = (short)max(0, (int)lim_rows_q_full - (int)sg_q_offset);
    if (is_last_q && sg_lim_q <= 0) continue;

    const device T* Q_qs  = Q  + qb * V34BWDK_BQ * int(params.Q_strides[2])
                              + sg_q_offset * int(params.Q_strides[2]);
    const device T* O_qs  = O  + qb * V34BWDK_BQ * int(params.O_strides[2])
                              + sg_q_offset * int(params.O_strides[2]);
    const device T* dO_qs = dO + qb * V34BWDK_BQ * int(params.dO_strides[2])
                              + sg_q_offset * int(params.dO_strides[2]);
    const device float* L_qs = L + qb * V34BWDK_BQ * int(params.L_strides[2])
                                + sg_q_offset * int(params.L_strides[2]);
    // v2.38.1: D buffer Q-block + SG-row offset (mirror L_qs).
    const device float* D_qs = D + qb * V34BWDK_BQ * int(params.D_strides[2])
                                + sg_q_offset * int(params.D_strides[2]);

    // --- Load lse, scale to log2 domain ---
    metal::vec<float, kRowsPT_q> lse_log2;
    {
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      constexpr short kEr = s_q_t::NAXFrag_t::kElemRows;
      constexpr short kErJ = s_q_t::NAXFrag_t::kElemRowsJump;
      constexpr float log2e_f = 1.4426950408889634f;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDK_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEr; i++) {
          const short local_row = iq * 16 + sc.y + i * kErJ;
          const short row_idx = iq * kEr + i;
          const bool in_range = (!is_last_q) || (local_row < sg_lim_q);
          if (in_range) {
            lse_log2[row_idx] = L_qs[local_row * int(params.L_strides[2])] * log2e_f;
          } else {
            lse_log2[row_idx] = Limits<float>::finite_max;
          }
        }
      }
    }

    // --- v2.38.1: load D = rowsum(dO ⊙ O) from device buffer (precomputed) ---
    // D is precomputed on host via MLX, shared with dQ + legacy-fused kernels.
    // Mirrors the lse-load pattern above; coalesced reads across lanes.
    metal::vec<float, kRowsPT_q> D_vec;
    {
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      constexpr short kEr = s_q_t::NAXFrag_t::kElemRows;
      constexpr short kErJ = s_q_t::NAXFrag_t::kElemRowsJump;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDK_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEr; i++) {
          const short local_row = iq * 16 + sc.y + i * kErJ;
          const short row_idx = iq * kEr + i;
          const bool in_range = (!is_last_q) || (local_row < sg_lim_q);
          D_vec[row_idx] = in_range
              ? D_qs[local_row * int(params.D_strides[2])]
              : 0.0f;
        }
      }
    }

    // --- S = Q[SG-rows] @ K^T ---
    s_q_t Stile;
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWDK_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWDK_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWDK_TD; id++) {
          NAXTile<T, 1, 1> Qfrag;
          NAXTile<T, 2, 1> Kfrag;
          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          const int K_off = ik * 16 * int(params.K_strides[2]) + id * 16;
          if (is_last_q) {
            Qfrag.load_rows(Q_qs + Q_off, int(params.Q_strides[2]),
                            sg_lim_q - iq * 16);
          } else {
            Qfrag.load(Q_qs + Q_off, int(params.Q_strides[2]));
          }
          if (is_last_k) {
            Kfrag.load_rows(K + K_off, int(params.K_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Kfrag.load(K + K_off, int(params.K_strides[2]));
          }
          s_q_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qfrag.frag_at(0, 0),
              metal::false_type{},
              Kfrag.frag_at(0, 0),
              Kfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_q_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= V34BWDK_SCALE_LOG2E;
    }
    if (is_last_k) {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      const short sn = sc.x;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDK_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDK_TK; ik++) {
          const short col_pos = ik * 16 + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < lim_rows_k) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // v2.50 Sprint 4 Phase 4b-complete (Prompt 3) — causal mask for dK split.
    // K-parallel with per-SG Q-row partition (sg_q_offset).
#if V34BWDK_CAUSAL
    {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc_c = s_q_t::NAXFrag_t::get_coord();
      const short sn_c = sc_c.x;
      const short sm_c = sc_c.y;
      const int base_row = qb * V34BWDK_BQ + params.qL_off + sg_q_offset;
      const int base_col = int(tid.x) * V34BWDK_BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDK_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDK_TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const int r = base_row + iq * 16
                          + ii * s_q_t::kFragRowsJump + sm_c;
              const int c = base_col + ik * 16 + jj + sn_c;
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }
#endif

    Stile.template row_bin_op<ExpSubOp>(lse_log2);
    // Stile holds P.

    // --- dP = dO @ V^T ---
    using dp_t = NAXTile<float, V34BWDK_TQ, V34BWDK_TK>;
    dp_t dPtile;
    dPtile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWDK_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWDK_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWDK_TD; id++) {
          NAXTile<T, 1, 1> dOfrag;
          NAXTile<T, 2, 1> Vfrag;
          const int dO_off = iq * 16 * int(params.dO_strides[2]) + id * 16;
          const int V_off = ik * 16 * int(params.V_strides[2]) + id * 16;
          if (is_last_q) {
            dOfrag.load_rows(dO_qs + dO_off, int(params.dO_strides[2]),
                             sg_lim_q - iq * 16);
          } else {
            dOfrag.load(dO_qs + dO_off, int(params.dO_strides[2]));
          }
          if (is_last_k) {
            Vfrag.load_rows(V + V_off, int(params.V_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Vfrag.load(V + V_off, int(params.V_strides[2]));
          }
          dp_t::NAXFrag_t::mma(
              dPtile.frag_at(iq, ik),
              dPtile.frag_at(iq, ik + 1),
              dOfrag.frag_at(0, 0),
              metal::false_type{},
              Vfrag.frag_at(0, 0),
              Vfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }
    dPtile.template row_bin_op<SubOp>(D_vec);

    // dS = P * (dP - D); overwrites Stile.
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_q_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= dPtile.elems()[ii];
    }

    // --- dK_accum += dS^T @ Q[SG-rows] ---
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < V34BWDK_TK; ik++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V34BWDK_TD; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < V34BWDK_TQ; iq++) {
          NAXTile<T, 1, 2> Qfrag2;
          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          if (is_last_q) {
            Qfrag2.load_rows(Q_qs + Q_off, int(params.Q_strides[2]),
                             sg_lim_q - iq * 16);
          } else {
            Qfrag2.load(Q_qs + Q_off, int(params.Q_strides[2]));
          }
          dk_t::NAXFrag_t::mma(
              dK_accum.frag_at(ik, id),
              dK_accum.frag_at(ik, id + 1),
              Stile.frag_at(iq, ik),
              metal::true_type{},
              Qfrag2.frag_at(0, 0),
              Qfrag2.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    simdgroup_barrier(mem_flags::mem_none);
  }  // end Q-loop

  // dK *= scale
  {
    constexpr short kRowsPT_k = dk_t::kRowsPerThread;
    metal::vec<float, kRowsPT_k> scale_vec;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT_k; i++) scale_vec[i] = V34BWDK_SCALE;
    dK_accum.template row_bin_op<MulOp>(scale_vec);
  }

  if (is_last_k) {
    if (lim_rows_k <= 0) return;
    dK_accum.store_rows(dK_partials, int(params.dKp_strides[3]), lim_rows_k);
  } else {
    dK_accum.store(dK_partials, int(params.dKp_strides[3]));
  }
}
)BWDKSPMSL";

  return ss.str();
}
std::string NAAttentionKernel::createV34BackwardFusedDKDVSparseSource() const noexcept {
  const int BQ = blockDimensions[0];
  const int BK = blockDimensions[1];
  const int BD = headDimension;
  const int WM = executionSIMDGroups;
  const int kU = 16;
  const int TQ_per_SG = BQ / (WM * kU);
  const int TD = BD / kU;
  const int TK = BK / kU;
  (void)TQ_per_SG; (void)TD; (void)TK;

  const bool is_bf16 =
      memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::BF16;
  const char* dtype_str = is_bf16 ? "bfloat" : "half";
  const float scale_log2e = scale * 1.4426950408889634f;

  std::ostringstream ss;
  ss << "// MFA_REQUIRE_MSL4\n";
  ss << "#include <metal_stdlib>\n";
  ss << "#include <metal_simdgroup>\n";
  ss << "#include <metal_simdgroup_matrix>\n";
  ss << "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  ss << "using namespace metal;\n";
  ss << "using namespace mpp::tensor_ops;\n\n";

  ss << naxHelpersBlock();

  ss << "\nusing T = " << dtype_str << ";\n";
  ss << "using namespace mlx::steel;\n\n";
  ss << "struct SubOp {\n";
  ss << "  template <typename U>\n";
  ss << "  METAL_FUNC static constexpr U apply(U x, U y) { return x - y; }\n";
  ss << "};\n\n";
  ss << "#define V34BWDF_BQ " << BQ << "\n";
  ss << "#define V34BWDF_BK " << BK << "\n";
  ss << "#define V34BWDF_BD " << BD << "\n";
  ss << "#define V34BWDF_WM " << WM << "\n";
  ss << "#define V34BWDF_TQ " << TQ_per_SG << "\n";
  ss << "#define V34BWDF_TD " << TD << "\n";
  ss << "#define V34BWDF_TK " << TK << "\n";
  ss << "#define V34BWDF_SCALE " << scale << "f\n";
  ss << "#define V34BWDF_SCALE_LOG2E " << scale_log2e << "f\n";
  // v2.50 Sprint 4 Phase 4b-complete (Prompt 3): causal masking macro
  ss << "#define V34BWDF_CAUSAL " << (isCausal ? 1 : 0) << "\n\n";

  ss << R"BWDFSPMSL(
struct V34BwdFusedParams {
  int qL, kL;
  int gqa_factor;
  int NQ, NK;
  int qL_rem, kL_rem;
  // v2.50 Sprint 4 Phase 4b-complete — causal offset (host struct match)
  int qL_off;
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long L_strides[3];
  long dO_strides[3];
  long dKp_strides[4];  // [B, Hq, WM, kL, D] FP32; D stride=1 implicit
  long dVp_strides[4];  // [B, Hq, WM, kL, D] FP32; D stride=1 implicit
  long D_strides[3];    // D=rowsum(dO⊙O) strides (FP32, [B, Hq, qL])
};

[[kernel, max_total_threads_per_threadgroup(V34BWDF_WM * 32)]]
void attention_bwd_fused_dkdv_sparse(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device float* L [[buffer(3)]],
    const device T* dO [[buffer(4)]],
    device float* dK_partials [[buffer(5)]],
    device float* dV_partials [[buffer(6)]],
    constant V34BwdFusedParams& params [[buffer(7)]],
    device const float* D [[buffer(8)]],
    const device bool* block_mask [[buffer(9)]],  // v2.50 Prompt 5d: 2-D sparse mask
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {

  (void)simd_lane_id;
  // Grid (NK, Hq, B). Each TG owns 1 K-tile, WM SGs partition Q-rows.

  ulong3 tidl{tid.x, tid.y, tid.z};
  ulong kv_head_idx = ulong(tid.y) / ulong(params.gqa_factor);

  Q  += tidl.z * params.Q_strides[0]  + tidl.y * params.Q_strides[1];
  dO += tidl.z * params.dO_strides[0] + tidl.y * params.dO_strides[1];
  L  += tidl.z * params.L_strides[0]  + tidl.y * params.L_strides[1];
  K  += tidl.z * params.K_strides[0]  + kv_head_idx * params.K_strides[1]
      + tidl.x * V34BWDF_BK * params.K_strides[2];
  V  += tidl.z * params.V_strides[0]  + kv_head_idx * params.V_strides[1]
      + tidl.x * V34BWDF_BK * params.V_strides[2];
  // D buffer indexed by query head (Hq), same as L.
  D  += tidl.z * params.D_strides[0]  + tidl.y * params.D_strides[1];

  // Per-SG dK_partials slot: [b, hq, sg, k_base, d] → 4 strides + D-stride=1.
  dK_partials += tidl.z * params.dKp_strides[0]
              +  tidl.y * params.dKp_strides[1]
              +  simd_group_id * params.dKp_strides[2]
              +  tidl.x * V34BWDF_BK * params.dKp_strides[3];
  dV_partials += tidl.z * params.dVp_strides[0]
              +  tidl.y * params.dVp_strides[1]
              +  simd_group_id * params.dVp_strides[2]
              +  tidl.x * V34BWDF_BK * params.dVp_strides[3];

  const short sg_q_offset = 16 * V34BWDF_TQ * simd_group_id;

  const int NQ_aligned = params.qL / V34BWDF_BQ;
  const int NK_aligned = params.kL / V34BWDF_BK;
  const bool is_last_k = (int(tid.x) == NK_aligned);
  const short lim_rows_k = (params.kL_rem > 0 && is_last_k)
      ? params.kL_rem : V34BWDF_BK;
  const int nq_full = params.qL / V34BWDF_BQ;
  const int nq_rem = params.qL % V34BWDF_BQ;
  const int q_loop = nq_rem > 0 ? nq_full + 1 : nq_full;

  // Per-SG accumulators (FP32, both persistent across Q-loop).
  using dk_t = NAXTile<float, V34BWDF_TK, V34BWDF_TD>;
  using dv_t = NAXTile<float, V34BWDF_TK, V34BWDF_TD>;
  dk_t dK_accum;
  dv_t dV_accum;
  dK_accum.clear();
  dV_accum.clear();

  using s_q_t = NAXTile<float, V34BWDF_TQ, V34BWDF_TK>;
  constexpr short kRowsPT_q = s_q_t::kRowsPerThread;

  // === Q-loop: fused dK + dV accumulation per K-tile ===
  for (int qb = 0; qb < q_loop; qb++) {
    // === v2.50 Prompt 5d Section A sparse-skip (ORDER-CRITICAL preserved) ===
    if (qb < (params.qL / V34BWDF_BQ)) {
      const int nk_total = params.NK;
      bool tile_active = block_mask[qb * nk_total + int(tid.x)];
      if (!tile_active) continue;
    }
    const bool is_last_q = (qb == NQ_aligned);
    const short lim_rows_q_full = (params.qL_rem > 0 && is_last_q)
        ? params.qL_rem : V34BWDF_BQ;
    const short sg_lim_q = (short)max(0, (int)lim_rows_q_full - (int)sg_q_offset);
    if (is_last_q && sg_lim_q <= 0) continue;

    const device T* Q_qs  = Q  + qb * V34BWDF_BQ * int(params.Q_strides[2])
                              + sg_q_offset * int(params.Q_strides[2]);
    const device T* dO_qs = dO + qb * V34BWDF_BQ * int(params.dO_strides[2])
                              + sg_q_offset * int(params.dO_strides[2]);
    const device float* L_qs = L + qb * V34BWDF_BQ * int(params.L_strides[2])
                                + sg_q_offset * int(params.L_strides[2]);
    const device float* D_qs = D + qb * V34BWDF_BQ * int(params.D_strides[2])
                                + sg_q_offset * int(params.D_strides[2]);

    // --- Load lse for SG's 16 rows, scale to log2 domain ---
    metal::vec<float, kRowsPT_q> lse_log2;
    {
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      constexpr short kEr = s_q_t::NAXFrag_t::kElemRows;
      constexpr short kErJ = s_q_t::NAXFrag_t::kElemRowsJump;
      constexpr float log2e_f = 1.4426950408889634f;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDF_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEr; i++) {
          const short local_row = iq * 16 + sc.y + i * kErJ;
          const short row_idx = iq * kEr + i;
          const bool in_range = (!is_last_q) || (local_row < sg_lim_q);
          if (in_range) {
            lse_log2[row_idx] = L_qs[local_row * int(params.L_strides[2])] * log2e_f;
          } else {
            lse_log2[row_idx] = Limits<float>::finite_max;  // → P=0 for OOR
          }
        }
      }
    }

    // --- v2.38.1: load D = rowsum(dO ⊙ O) from device buffer ---
    metal::vec<float, kRowsPT_q> D_vec;
    {
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      constexpr short kEr = s_q_t::NAXFrag_t::kElemRows;
      constexpr short kErJ = s_q_t::NAXFrag_t::kElemRowsJump;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDF_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEr; i++) {
          const short local_row = iq * 16 + sc.y + i * kErJ;
          const short row_idx = iq * kEr + i;
          const bool in_range = (!is_last_q) || (local_row < sg_lim_q);
          D_vec[row_idx] = in_range
              ? D_qs[local_row * int(params.D_strides[2])]
              : 0.0f;
        }
      }
    }

    // --- S = Q[SG-rows] @ K^T ---
    s_q_t Stile;
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWDF_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWDF_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWDF_TD; id++) {
          NAXTile<T, 1, 1> Qfrag;
          NAXTile<T, 2, 1> Kfrag;
          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          const int K_off = ik * 16 * int(params.K_strides[2]) + id * 16;
          if (is_last_q) {
            Qfrag.load_rows(Q_qs + Q_off, int(params.Q_strides[2]),
                            sg_lim_q - iq * 16);
          } else {
            Qfrag.load(Q_qs + Q_off, int(params.Q_strides[2]));
          }
          if (is_last_k) {
            Kfrag.load_rows(K + K_off, int(params.K_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Kfrag.load(K + K_off, int(params.K_strides[2]));
          }
          s_q_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qfrag.frag_at(0, 0),
              metal::false_type{},
              Kfrag.frag_at(0, 0),
              Kfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale into log2 domain.
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_q_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= V34BWDF_SCALE_LOG2E;
    }

    // Mask last-K columns to -inf.
    if (is_last_k) {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc = s_q_t::NAXFrag_t::get_coord();
      const short sn = sc.x;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDF_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDF_TK; ik++) {
          const short col_pos = ik * 16 + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < lim_rows_k) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // v2.50 Sprint 4 Phase 4b-complete (Prompt 3) — causal mask for fused
    // dKdV.  K-parallel with per-SG Q-row partition (sg_q_offset).
    // ORDER-CRITICAL preserved: mask is on Stile holding S, before
    // row_bin_op<ExpSubOp> converts to P, before P^T @ dO uses P.
#if V34BWDF_CAUSAL
    {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc_c = s_q_t::NAXFrag_t::get_coord();
      const short sn_c = sc_c.x;
      const short sm_c = sc_c.y;
      const int base_row = qb * V34BWDF_BQ + params.qL_off + sg_q_offset;
      const int base_col = int(tid.x) * V34BWDF_BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < V34BWDF_TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < V34BWDF_TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < s_q_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < s_q_t::kFragThrCols; jj++) {
              const int r = base_row + iq * 16
                          + ii * s_q_t::kFragRowsJump + sm_c;
              const int c = base_col + ik * 16 + jj + sn_c;
              const auto loc = ii * s_q_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }
#endif

    // P = exp2(S - lse_log2).  Stile holds P after this.
    Stile.template row_bin_op<ExpSubOp>(lse_log2);

    // === ORDER-CRITICAL: dV_accum += P^T @ dO BEFORE Stile is overwritten ===
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < V34BWDF_TK; ik++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V34BWDF_TD; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < V34BWDF_TQ; iq++) {
          NAXTile<T, 1, 2> dOfrag2;
          const int dO_off = iq * 16 * int(params.dO_strides[2]) + id * 16;
          if (is_last_q) {
            dOfrag2.load_rows(dO_qs + dO_off, int(params.dO_strides[2]),
                              sg_lim_q - iq * 16);
          } else {
            dOfrag2.load(dO_qs + dO_off, int(params.dO_strides[2]));
          }
          dv_t::NAXFrag_t::mma(
              dV_accum.frag_at(ik, id),
              dV_accum.frag_at(ik, id + 1),
              Stile.frag_at(iq, ik),
              metal::true_type{},   // transpose_a: P^T
              dOfrag2.frag_at(0, 0),
              dOfrag2.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    // --- dP = dO @ V^T ---
    using dp_t = NAXTile<float, V34BWDF_TQ, V34BWDF_TK>;
    dp_t dPtile;
    dPtile.clear();
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < V34BWDF_TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V34BWDF_TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < V34BWDF_TD; id++) {
          NAXTile<T, 1, 1> dOfrag;
          NAXTile<T, 2, 1> Vfrag;
          const int dO_off = iq * 16 * int(params.dO_strides[2]) + id * 16;
          const int V_off = ik * 16 * int(params.V_strides[2]) + id * 16;
          if (is_last_q) {
            dOfrag.load_rows(dO_qs + dO_off, int(params.dO_strides[2]),
                             sg_lim_q - iq * 16);
          } else {
            dOfrag.load(dO_qs + dO_off, int(params.dO_strides[2]));
          }
          if (is_last_k) {
            Vfrag.load_rows(V + V_off, int(params.V_strides[2]),
                            lim_rows_k - ik * 16);
          } else {
            Vfrag.load(V + V_off, int(params.V_strides[2]));
          }
          dp_t::NAXFrag_t::mma(
              dPtile.frag_at(iq, ik),
              dPtile.frag_at(iq, ik + 1),
              dOfrag.frag_at(0, 0),
              metal::false_type{},
              Vfrag.frag_at(0, 0),
              Vfrag.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }
    // dP -= D_vec (per-row broadcast).
    dPtile.template row_bin_op<SubOp>(D_vec);

    // dS = P * (dP - D); overwrites Stile in place (P is consumed).
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < s_q_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= dPtile.elems()[ii];
    }

    // --- dK_accum += dS^T @ Q[SG-rows] ---
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < V34BWDF_TK; ik++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V34BWDF_TD; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < V34BWDF_TQ; iq++) {
          NAXTile<T, 1, 2> Qfrag2;
          const int Q_off = iq * 16 * int(params.Q_strides[2]) + id * 16;
          if (is_last_q) {
            Qfrag2.load_rows(Q_qs + Q_off, int(params.Q_strides[2]),
                             sg_lim_q - iq * 16);
          } else {
            Qfrag2.load(Q_qs + Q_off, int(params.Q_strides[2]));
          }
          dk_t::NAXFrag_t::mma(
              dK_accum.frag_at(ik, id),
              dK_accum.frag_at(ik, id + 1),
              Stile.frag_at(iq, ik),
              metal::true_type{},
              Qfrag2.frag_at(0, 0),
              Qfrag2.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    simdgroup_barrier(mem_flags::mem_none);
  }  // end Q-loop

  // dK *= scale (matches split-dK; dV is not scaled — P already absorbs it).
  {
    constexpr short kRowsPT_k = dk_t::kRowsPerThread;
    metal::vec<float, kRowsPT_k> scale_vec;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT_k; i++) scale_vec[i] = V34BWDF_SCALE;
    dK_accum.template row_bin_op<MulOp>(scale_vec);
  }

  // === Per-SG-slot device write for both gradients ===
  if (is_last_k) {
    if (lim_rows_k <= 0) return;
    dK_accum.store_rows(dK_partials, int(params.dKp_strides[3]), lim_rows_k);
    dV_accum.store_rows(dV_partials, int(params.dVp_strides[3]), lim_rows_k);
  } else {
    dK_accum.store(dK_partials, int(params.dKp_strides[3]));
    dV_accum.store(dV_partials, int(params.dVp_strides[3]));
  }
}
)BWDFSPMSL";

  return ss.str();
}
