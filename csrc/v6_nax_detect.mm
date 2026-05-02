/// Implementation of V6 NAX hardware detection.
/// Uses Objective-C++ for `__builtin_available` and `supportsFamily`.

#include "v6_nax_detect.hpp"
#include <Metal/Metal.h>
#include <mlx/backend/metal/device.h>

namespace mlx_mfa {

bool device_has_neural_accelerators() {
  // MLX 0.31.2's `mlx::core::metal::is_nax_available()` is declared in
  // device.h but NOT exported in libmlx.dylib on M5 (link error: symbol
  // not found). So we use `supportsFamily(MTLGPUFamilyApple10)` directly.
  // MTLGPUFamilyApple10 = 1010 (raw enum value; may not be in older SDKs).
  // On M5 Max this returns true; on M1–M4 it returns false.
  bool family_apple10 = false;
  @autoreleasepool {
    auto s = mlx::core::default_stream(mlx::core::Device::gpu);
    auto& d = mlx::core::metal::device(s.device);
    auto* mtl = d.mtl_device();
    if (mtl) {
      id<MTLDevice> objc_dev = (__bridge id<MTLDevice>)mtl;
      family_apple10 = [objc_dev supportsFamily:(MTLGPUFamily)1010];
    }
  }
  return family_apple10;
}

bool device_has_nax_bf16() {
  if (!device_has_neural_accelerators()) return false;
  // MPP bf16 paths require macOS 26.1+. Use __builtin_available for runtime check.
  if (@available(macOS 26.1, *)) {
    return true;
  }
  return false;
}

}  // namespace mlx_mfa
