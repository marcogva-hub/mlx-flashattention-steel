// Derived from Draw Things' NAAttentionKernelDescriptor.hpp
// (https://github.com/liuliu/ccv/blob/unstable/lib/nnc/mfa/kernels/NAAttentionKernelDescriptor.hpp)
// Licensed under BSD 3-Clause. See LICENSE-DRAWTHINGS at repo root.
//
// mlx-mfa adaptation:
//   - Removed dependency on metal-cpp (uses MTL::Device passed as void* in
//     dispatch path; descriptor stays Metal-agnostic)
//   - Hash uses mfa_compat.h helpers (same algorithm as ccv)

#ifndef MLX_MFA_V6_NAX_NAATTENTIONKERNELDESCRIPTOR_HPP
#define MLX_MFA_V6_NAX_NAATTENTIONKERNELDESCRIPTOR_HPP

#include "../GEMMOperandPrecision.hpp"
#include "../AttentionOperand.hpp"
#include "../AttentionKernelType.hpp"
#include <simd/simd.h>

struct NAAttentionDescriptor;

/// A configuration for an Attention kernel (V6 NAX path).
struct NAAttentionKernelDescriptor {
  /// parallelization, traversal, head
  simd::ushort3 blockDimensions;

  /// Required. The problem size along the head dimension.
  unsigned short headDimension;

  unsigned short Hq;
  unsigned short Hk;

  uint16_t executionSIMDGroups;

  bool checkCEdge1;
  bool bypassThreadgroupMemory;
  bool isCausal;
  bool masked;
  bool isVarlen;
  // Sprint 3.3 — Apple-style single-Otile kernel variant.
  // Set after construction (no new constructor); defaulted to false on all paths.
  bool singleOtileMode = false;

  // V6NAX — NAX-direct rewrite (Sprint V6NAX).
  // When true, generates an Apple steel_attention_nax.h-style kernel using
  // NAXTile / NAXFrag::mma directly (no MPP cooperative_tensor at <N>).
  // Forward non-causal single-Otile only (production VSR hot path).
  // Requires BQ % (WM * 16) == 0 and BD % 16 == 0.
  bool useV6NAX = false;

  AttentionOperands<GEMMOperandPrecision> memoryPrecisions;
  AttentionKernelType type;
  float scale;

  NAAttentionKernelDescriptor() = delete;

  NAAttentionKernelDescriptor(
      simd::ushort3 blockDimensions, unsigned short headDimension,
      unsigned short Hq, unsigned short Hk, uint16_t executionSIMDGroups,
      bool checkCEdge1, AttentionOperands<GEMMOperandPrecision> memoryPrecisions,
      AttentionKernelType type, float scale) noexcept;
  NAAttentionKernelDescriptor(
      simd::ushort3 blockDimensions, unsigned short headDimension,
      unsigned short Hq, unsigned short Hk, uint16_t executionSIMDGroups,
      bool checkCEdge1, AttentionOperands<GEMMOperandPrecision> memoryPrecisions,
      AttentionKernelType type, float scale,
      bool bypassThreadgroupMemory) noexcept;
  NAAttentionKernelDescriptor(
      simd::ushort3 blockDimensions, unsigned short headDimension,
      unsigned short Hq, unsigned short Hk, uint16_t executionSIMDGroups,
      bool checkCEdge1, AttentionOperands<GEMMOperandPrecision> memoryPrecisions,
      AttentionKernelType type, float scale, bool bypassThreadgroupMemory,
      bool isCausal, bool masked) noexcept;
  NAAttentionKernelDescriptor(
      simd::ushort3 blockDimensions, unsigned short headDimension,
      unsigned short Hq, unsigned short Hk, uint16_t executionSIMDGroups,
      bool checkCEdge1, AttentionOperands<GEMMOperandPrecision> memoryPrecisions,
      AttentionKernelType type, float scale, bool bypassThreadgroupMemory,
      bool isCausal, bool masked, bool isVarlen) noexcept;

  bool operator==(const NAAttentionKernelDescriptor& rhs) const;
};

template<>
struct std::hash<NAAttentionKernelDescriptor>
{
  std::size_t operator()(const NAAttentionKernelDescriptor& hash) const noexcept;
};

#endif  // MLX_MFA_V6_NAX_NAATTENTIONKERNELDESCRIPTOR_HPP
