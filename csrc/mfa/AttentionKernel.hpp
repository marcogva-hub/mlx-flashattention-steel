/// AttentionKernel.hpp — Declarations for the ccv-derived attention shader generator.
///
/// AttentionKernel takes an AttentionKernelDescriptor (blocking params, dtype, head_dim,
/// causal, kernel type) and emits a complete Metal shader source string.
/// The generator supports forward, backward-query, and backward-kv loops.
///
/// Ported from liuliu/ccv lib/nnc/mfa/v2/AttentionKernel.cpp (Apache 2.0).
/// mlx-mfa changes: transposeState fix (all false), causal backward masking,
/// SEQUENCE_LENGTH unconditionally in head-offset expressions.

#ifndef AttentionKernel_hpp
#define AttentionKernel_hpp

#include "AttentionKernelDescriptor.hpp"
#include "mfa_compat.h"

class CodeWriter;

struct AttentionAccumulateDescriptor;
struct AttentionOuterProductDescriptor;

struct AttentionKernel {
  // NOTE: library member removed – we only generate source, not compile here.

  std::string source;

  AttentionKernelType type;

  AttentionOperands<bool> cacheState;

  AttentionOperands<GEMMOperandPrecision> memoryPrecisions;

  bool preferAsyncCache;

  bool preferAsyncLoad;

  AttentionOperands<GEMMOperandPrecision> registerPrecisions;

  AttentionOperands<bool> transposeState;

  /// The leading dimensions after transposed if applied.
  AttentionOperands<bool> leadingDimensions;

  /// parallelization, traversal, head
  simd::ushort3 blockDimensions;

  unsigned short headDimension;

  bool disableAsyncCopy;

  unsigned short threadgroupMemoryAllocation;

  /// The number of threads per group.
  uint16_t threadgroupSize;

  /// Construct and immediately generate shader source (no Metal device needed).
  AttentionKernel(AttentionKernelDescriptor descriptor);

private:
  /// AttentionKernel.
  std::string memoryName(AttentionOperand operand) const noexcept;
  std::string registerName(AttentionOperand operand) const noexcept;
  std::string loadFunction(AttentionOperand operand) const noexcept;
  std::string storeFunction(AttentionOperand operand) const noexcept;
  bool cached(AttentionOperand operand) const noexcept;
  bool transposed(AttentionOperand operand) const noexcept;
  std::string sequenceLength(AttentionOperand operand) const noexcept;
  unsigned short blockSequenceLength(AttentionOperand operand) const noexcept;
  std::string leadingDimension(AttentionOperand operand) const noexcept;
  unsigned short leadingBlockDimension(AttentionOperand operand) const noexcept;

  std::string parallelizationDimensionValue() const noexcept;
  std::string parallelizationGroupOffsetValue() const noexcept;
  std::string unsafeParallelizationThreadOffsetValue() const noexcept;
  std::string clampedParallelizationThreadOffsetValue() const noexcept;
  std::string traversalDimensionValue() const noexcept;
  std::string traversalOffsetValue() const noexcept;
  std::string paddedTraversalEdgeValue() const noexcept;
  unsigned short paddedHeadDimensionValue() const noexcept;
  unsigned short paddedHeadEdgeValue() const noexcept;
  unsigned short threadgroupSizeValue() const noexcept;
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string operandLocationValue(AttentionOperand operand) const noexcept;
  std::string operandLocationWithHeadOffsetValue(AttentionOperand operand) const noexcept;

  /// AttentionKernel+Source
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
  std::string createAdjustOffsets() const noexcept;
  std::string createBufferBindings() const noexcept;
  std::string loopForward() const noexcept;
  std::string loopBackwardQuery() const noexcept;
  std::string loopBackwardKeyValue() const noexcept;

  /// AttentionKernel+Accumulate
  std::string accumulate(const AttentionAccumulateDescriptor& descriptor) const noexcept;

  /// AttentionKernel+Caching
  class CachingOperationType {
  public:
    enum Value: uint16_t {
      load = 0,
      store = 1,
    };

    CachingOperationType() = default;
    constexpr CachingOperationType(Value aValue) : value(aValue) { }

    explicit operator bool() const = delete;

    constexpr bool operator==(const CachingOperationType &rhs) const { return value == rhs.value; }
    constexpr bool operator!=(const CachingOperationType &rhs) const { return value != rhs.value; }

    Value value;
  };
  std::string cache(AttentionOperand operand, CachingOperationType type) const noexcept;
  std::string createSetup() const noexcept;
  std::string createCleanup(const AttentionKernelType type) const noexcept;

  /// AttentionKernel+OuterProduct
  std::string outerProduct(const AttentionOuterProductDescriptor& descriptor) const noexcept;

  /// AttentionKernel+Softmax
  std::string computeD() const noexcept;
  std::string maskAttentionMatrixEdge() const noexcept;
  std::string causalMask() const noexcept;
  std::string causalMaskTransposed() const noexcept;
  std::string onlineReduceMaximum() const noexcept;
  std::string onlineCorrectO() const noexcept;
  std::string onlineReduceSum() const noexcept;
  std::string softmax(bool derivative) const noexcept;
};

#endif /* AttentionKernel_hpp */
