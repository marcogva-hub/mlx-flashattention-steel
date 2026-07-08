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

/// Host macOS version (from `NSProcessInfo.operatingSystemVersion`).  This is
/// the OS-version signal for OS-aware M5+ routing: the Metal compiler shipped
/// with the OS is part of the measurement quadruple (MLX, mlx-mfa, hardware,
/// macOS/Metal-compiler).  `device_macos_major()` returns 26, 27, …; returns 0
/// if the query is unavailable (callers must treat 0 as "unknown → safe path").
int device_macos_major();
int device_macos_minor();

}  // namespace mlx_mfa
