/// V6 NAX hardware capability detection helpers.
///
/// `device_has_neural_accelerators()` — true if the GPU has Neural Accelerators
/// (Apple GPU family 10+, M5 family). Combines two signals:
///   1. MLX 0.31.2's `mlx::core::metal::is_nax_available()` (when available)
///   2. `supportsFamily(MTLGPUFamilyApple10)` (== 1010) as fallback/validation
///
/// `device_has_nax_bf16()` — adds an `__builtin_available(macOS 26.1+)` gate
/// for bfloat16 support on NAX (MPP bf16 paths require macOS 26.1+).

#pragma once

namespace mlx_mfa {

bool device_has_neural_accelerators();
bool device_has_nax_bf16();

}  // namespace mlx_mfa
