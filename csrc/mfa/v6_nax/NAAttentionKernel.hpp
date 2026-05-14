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
  // Sprint 3.3 — Apple-style single-Otile kernel variant.
  // When true, loopForward() dispatches to loopForwardSingleTile() which emits
  // a kernel with: single cS (no double-buffer), forced kBlocks=1, always-bypass
  // cP cooperative tensor (no P_buf staging), mem_none barriers, K-loop step BK.
  bool singleOtileMode;

  // V34 — see descriptor.
  bool useV34;

  // mlx-mfa: takes a threadgroup-memory-length hint instead of a pipeline.
  // Returns the bytes to allocate at threadgroup(0) for the dispatch.
  unsigned short threadgroupMemoryAllocation() const noexcept;

  /// The number of threads per group.
  uint16_t threadgroupSize() const noexcept;

  /// Constructor: generate the MSL 4 source string. Does not require an
  /// MTL::Device (mlx-mfa's shader cache compiles the source separately).
  NAAttentionKernel(NAAttentionKernelDescriptor descriptor);

  /// V34 backward dQ source generator (public — called by
  /// MFAV34BwdQuery::eval_gpu in csrc/mfa_v6_nax_primitive.cpp).
  /// Implementation in NAAttentionKernel.cpp (Phase 1 Section B of V34
  /// backward Option β sprint).
  std::string createV34BackwardQuerySource() const noexcept;
  /// V34 backward dK/dV source generator (Phase 2 of V34 backward Option β).
  /// Single-SG (WM=1) kernel; one TG per K-tile iterates over all Q-tiles
  /// and accumulates partial dK + dV in per-SG FP32 NAX tiles.  Returns
  /// (dK, dV) written to device.  No cross-SG reduction needed.
  std::string createV34BackwardKeyValueSource() const noexcept;
  /// V34 backward dV-only kernel (Phase 2.O2 — multi-SG Q-row partition).
  /// WM=4 BQ=64 BK=32 D=128: each SG handles BQ/WM=16 Q-rows.  Softmax is
  /// intra-SG (no replication tax).  Each SG writes its dV partial
  /// (BK × D, contributions from its 16 Q-rows × NQ Q-tiles) to a unique
  /// slot in dV_partials [B, Hq, WM, kL, D] FP32.  Python wrapper reduces
  /// via mx.sum(axis=2) and casts to T.
  std::string createV34BackwardDVSource() const noexcept;
  /// V34 backward dK-only kernel (Phase 2.O2 sister kernel).  Same WM=4
  /// Q-row partition architecture but adds D = rowsum(dO⊙O), dP = dO@V^T,
  /// dS = P*(dP-D), and dK_accum += dS^T @ Q.  Output: dK_partials [B, Hq,
  /// WM, kL, D] FP32; Python wrapper reduces via mx.sum(axis=2) and casts.
  std::string createV34BackwardDKSource() const noexcept;
  /// V34 backward FUSED dK+dV kernel (Option γ, Sprint v2.39.0 Phase C.1.a).
  /// Single kernel computes both gradients in one K-tile load (the structural
  /// ~10% perf win is K-bandwidth amortization, not just softmax fusion per
  /// /metal-kernel-dev audit 2026-05-13).  WM=4 Q-row partition; per-SG-slot
  /// device writes to dK_partials + dV_partials.  D=64 only; D=128 falls
  /// back to split kernels (separate PR per blueprint staging).  Order
  /// constraint: dV_accum += P^T @ dO MUST precede dS = P * dP overwriting
  /// Stile (see blueprint §"Order of operations").  Consumes v2.38.1 D_vec.
  std::string createV34BackwardFusedDKDVSource() const noexcept;

  /// V34 backward dV SPARSE kernel (Prompt 5b Section A PoC).
  /// Mirrors createV34BackwardDVSource() but adds per-Q-tile block_mask
  /// scan in the Q-loop: when block_mask[qb, k_tile] == false, skip the
  /// entire Q-tile contribution (zero divergence — uniform across SG).
  /// Pattern reference: csrc/mfa_sparse_attention.cpp forward LCSA scan.
  ///
  /// Mask layout supported: 2-D (NQ, NK) only at PoC stage (Sprint 5 v2
  /// will broaden to 3-D and 4-D layouts).  Higher mask_ndim values fall
  /// back to the dense kernel via flash_attention_sparse's routing.
  ///
  /// Output identical to dense dV kernel: dV_partials [B, Hq, WM, kL, D]
  /// FP32 — caller reduces via mx.sum(axis=2) and casts to T.
  std::string createV34BackwardDVSparseSource() const noexcept;

  // V34 backward dQ/dK/fused-dKdV SPARSE kernels — DECLARATIONS RESERVED
  // for Section A v3 follow-up.  In Prompt 5c Section A.2 (this commit),
  // Python-level orchestration uses the existing dV sparse PoC kernel
  // (consumed with sparse-LSE) + SDPA-vjp for dQ/dK gradients, which
  // delivers a CORRECT end-to-end sparse backward path while deferring
  // the 3 remaining native sparse kernels to a focused future session
  // (mechanical extension once dQ pattern is validated).  See
  // `docs/v50/sprint-5c-section-a-status.md` for the empirical
  // justification (time + risk vs. value).

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
  void loopForwardSingleTile(CodeWriter &source) const noexcept;
  std::string createV34Source() const noexcept;
  void loopBackwardQuery(CodeWriter &source) const noexcept;
  void loopBackwardKeyValue(CodeWriter &source) const noexcept;
  std::string createComputeD() const noexcept;
  std::string createAdjustOffsets() const noexcept;
  std::string createBufferBindings() const noexcept;
};

#endif  // MLX_MFA_V6_NAX_NAATTENTIONKERNEL_HPP
