/// MFAConv3DForward — Conv3D NAX production C++ entry point.
///
/// Sprint D migration of mlx_mfa.conv_nax.conv3d_nax_forward() from
/// Python orchestrator to C++. The substantive goal is removing the
/// ~50-100 µs Python dispatch overhead per call (Phase 1.1 D15).
///
/// Implementation note: implemented as a C++ free function that uses
/// `mlx::core::fast::metal_kernel` internally to JIT-compile and
/// dispatch each chunk's im2col + matmul2d kernels. This is functionally
/// equivalent to a Primitive::eval_gpu wrapping multi-kernel dispatch
/// (each fast::metal_kernel call internally creates a CustomKernel
/// Primitive). The composition pattern keeps the implementation
/// mechanical relative to the validated Phase 1.x Python source.
///
/// See docs/conv-nax/conv-nax-prod-decisions.md D33 for the pragmatic
/// choice rationale (function-level migration vs Primitive subclass).

#pragma once

#include <mlx/array.h>
#include <array>
#include <cstdint>

namespace mlx_mfa {

/// Padding is a 6-tuple per dim: (T_left, T_right, H_left, H_right,
/// W_left, W_right). Symmetric padding is encoded as
/// (p, p, p, p, p, p). Causal pad_T is (K_T-1, 0, ...).
struct ConvPad {
  int T_left, T_right;
  int H_left, H_right;
  int W_left, W_right;
};

/// NAX-accelerated Conv3D forward, channels-last layout, FP16/BF16.
///
/// Routes:
///   - 1x1x1 with zero padding + unit stride → pointwise fast path
///   - 3x3x3 (or any K_T,K_H,K_W with K_T*K_H*K_W >= 1) → im2col + matmul2d
///
/// Inputs:
///   x: (B, T, H, W, C_in) row-major, dtype f16 or bf16
///   w: (C_out, K_T, K_H, K_W, C_in) row-major, same dtype
///   stride: (sT, sH, sW), values in {1}
///   padding: 6-tuple per dim (left, right per axis); symmetric or asymmetric
///   dilation: (dT, dH, dW), values in {1}
///   chunk_M: 0 = auto from int32-byte-offset heuristic, else override
///
/// Output:
///   y: (B, T_out, H_out, W_out, C_out), same dtype
///
/// Defensive int32 byte-offset chunking invariant (Phase 1.2 D7 lesson):
/// each chunk's im2col buffer (chunk_M × K × dtype_bytes) stays strictly
/// below 2^31 bytes × 0.875 safety margin. MPP matmul2d uses int32
/// internally for byte addresses; overflow produces NaN at row
/// 2^31 / (K × dtype_bytes).
mlx::core::array conv3d_nax_forward(
    const mlx::core::array& x,
    const mlx::core::array& w,
    const std::array<int, 3>& stride,
    const ConvPad& padding,
    const std::array<int, 3>& dilation,
    int chunk_M = 0);

}  // namespace mlx_mfa
