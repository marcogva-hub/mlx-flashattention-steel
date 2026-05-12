/// Sprint B Sparse Attention NAX — block-sparse forward via per-Q-tile dispatch
/// with NAX matmul2d inner-loop primitives.
///
/// Sprint B Phase 1.1 scaffold: free-function entry point, mirrors Sprint D's
/// conv3d_nax_forward pattern (see design B-D2). C++ orchestration only; the
/// inner kernel is JIT-compiled via mlx::core::fast::metal_kernel.
///
/// Three-axis validation (CLAUDE_V6_NAX §7):
///   1. Output sanity: RMSE vs SDPA+float-bias oracle
///   2. Path entered: this function called (caller-level A/B perf check)
///   3. Edges preserved: all-False mask row → zero output, all-True →
///      dense-SDPA equivalence, diagonal-only → causal-style correctness
///
/// Phase 1.1 scope: 2-D mask [NQ, NK], FP16 only, head_dim ∈ {64, 128},
/// block_tile ∈ {16, 32, 64, 128}. Phase 1.2 extends to 3-D / 4-D masks +
/// causal-with-block-mask + bfloat16 + asymmetric qL ≠ kL.

#pragma once

#include <mlx/array.h>

namespace mlx_mfa {

/// Sprint B block-sparse attention forward.
///
/// Inputs:
///   Q: (B, Hq, qL, D) row-major, FP16
///   K: (B, Hk, kL, D) row-major, FP16  (Hq must be multiple of Hk for GQA)
///   V: (B, Hk, kL, D) row-major, FP16
///   block_mask: (NQ, NK) bool, where NQ = qL/BT, NK = kL/BT.
///     True means "compute this Q-tile × K-tile pair", False means skip.
///   block_tile: BT ∈ {16, 32, 64, 128}. Must evenly divide qL and kL.
///   causal: if true, apply within-tile causal mask AND skip future-K tiles
///     not already masked. Phase 1.1: causal=false only (Phase 1.2 enables).
///   scale: typically 1/sqrt(D); caller-provided.
///
/// Output:
///   O: (B, Hq, qL, D) FP16. All-False mask Q-rows → zero (per §7 edge).
///
/// Throws std::runtime_error on shape / dtype / parameter mismatch.
mlx::core::array sparse_attention_forward(
    const mlx::core::array& Q,
    const mlx::core::array& K,
    const mlx::core::array& V,
    const mlx::core::array& block_mask,
    int block_tile,
    bool causal,
    float scale);

}  // namespace mlx_mfa
