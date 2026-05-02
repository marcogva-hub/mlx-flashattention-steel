// Derived from Draw Things' NAAttentionKernel.hpp
// Licensed under BSD 3-Clause. See LICENSE-DRAWTHINGS at repo root.
//
// mlx-mfa adaptation:
//   - Removed metal-cpp includes (kernel produces an MSL 4 source string;
//     compilation goes through mlx-mfa's shader cache, which calls
//     newLibraryWithSource directly via MTLDevice).
//   - The kernel constructor no longer takes MTL::Device — it just generates
//     the source string. Pipeline compilation is handled by ShaderCache.

#ifndef MLX_MFA_V6_NAX_NAATTENTIONKERNEL_HPP
#define MLX_MFA_V6_NAX_NAATTENTIONKERNEL_HPP

#include "NAAttentionKernelDescriptor.hpp"
#include <simd/simd.h>
#include <string>

class CodeWriter;

struct NAAttentionKernel {
  static constexpr uint16_t computeDThreads = 32;
  static constexpr uint16_t blockMaskThreads = 256;

  std::string source;

  AttentionKernelType type;
  float scale;
  AttentionOperands<GEMMOperandPrecision> memoryPrecisions;

  /// parallelization, traversal, head
  simd::ushort3 blockDimensions;
  unsigned short headDimension;
  unsigned short Hq;
  unsigned short Hk;
  uint16_t executionSIMDGroups;

  bool bypassThreadgroupMemory;
  bool checkCEdge1;
  bool isCausal;
  bool masked;
  bool isVarlen;

  // mlx-mfa: takes a threadgroup-memory-length hint instead of a pipeline.
  // Returns the bytes to allocate at threadgroup(0) for the dispatch.
  unsigned short threadgroupMemoryAllocation() const noexcept;

  /// The number of threads per group.
  uint16_t threadgroupSize() const noexcept;

  /// Constructor: generate the MSL 4 source string. Does not require an
  /// MTL::Device (mlx-mfa's shader cache compiles the source separately).
  NAAttentionKernel(NAAttentionKernelDescriptor descriptor);

private:
  // Helpers that build operand-name and stride strings for the source.
  std::string memoryName(AttentionOperand operand) const noexcept;
  std::string sequenceLength(AttentionOperand operand) const noexcept;
  unsigned short blockSequenceLength(AttentionOperand operand) const noexcept;
  std::string operandLocationWithHeadOffsetValue(AttentionOperand operand) const noexcept;

  // Source generators.
  std::string createSource() const noexcept;
  void createConstants(CodeWriter &source) const noexcept;
  void loopForward(CodeWriter &source) const noexcept;
  void loopForwardSingleCausal(CodeWriter &source) const noexcept;
  void loopBackwardQuery(CodeWriter &source) const noexcept;
  void loopBackwardKeyValue(CodeWriter &source) const noexcept;
  std::string createComputeD() const noexcept;
  std::string createAdjustOffsets() const noexcept;
  std::string createBufferBindings() const noexcept;
};

#endif  // MLX_MFA_V6_NAX_NAATTENTIONKERNEL_HPP
