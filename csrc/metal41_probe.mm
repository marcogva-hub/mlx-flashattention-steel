/// metal41_probe.mm — FUNCTIONAL Metal-4.1 capability probe (default build).
///
/// The macOS-27 routing seam activates on a REAL functional check, not a
/// version string: "macOS 27" guarantees nothing about the compiler (proven —
/// the sparse (long)p->NK bug PERSISTS under 4.1; the runtime JIT was at 4.0;
/// a CLT reinstall changed the compiler under the same OS). This probe asks:
/// does THIS toolchain compile AND correctly execute Metal 4.1, here?
///
/// Two required parts:
///   1. Compile-time 4.1 proof — the source carries `// MFA_REQUIRE_MSL41`
///      (ShaderCache selects MTLLanguageVersion4_1) + `#if __METAL_VERSION__
///      < 410 → #error`, so a 4.0-only compiler fails to compile → probe fails.
///   2. Functional proof — a small deterministic loop-accumulation kernel whose
///      output is read back and checked against a closed-form reference. Compile
///      success proves the 4.1 *language*; correct output proves the toolchain
///      *executes* it — catching a compiles-but-miscompiles beta (the sparse-bug
///      failure mode class).
///
/// FAIL-SAFE (external users may run this in production on beta): ANY failure —
/// compile error, dispatch error, wrong result, exception — returns false ⇒ the
/// caller falls back to the validated macOS-26 path. A false negative (miss a
/// capable toolchain) is acceptable; a false positive (activate on a broken
/// toolchain) is not. The kernel is trivial + bounded (64 threads, one
/// dispatch) — no hang risk.
///
/// This probe proves *4.1 capability*, NOT per-kernel correctness: known-broken
/// kernels (STEEL sparse D=128) keep their OWN fallbacks regardless of this.

#include "shader_cache.hpp"
#include <mlx/backend/metal/device.h>
#import <Metal/Metal.h>
#include <string>

namespace mlx_mfa {

// Returns true iff the installed Metal toolchain compiles AND correctly runs a
// Metal-4.1 kernel on this device. Fail-safe: false on ANY error.
bool probe_metal41_functional() {
  try {
    @autoreleasepool {
      // Part 1: MSL41 sentinel (→ MTLLanguageVersion4_1) + compile-time 4.1 guard.
      // Part 2: deterministic loop accumulation o[i] = sum_{j=0..i} j = i(i+1)/2
      //         (exercises loop + arithmetic codegen — the sparse-bug class —
      //         at 4.1, not just a trivial passthrough).
      const std::string src =
          "// MFA_REQUIRE_MSL41\n"
          "#if __METAL_VERSION__ < 410\n"
          "#error metal41_probe requires __METAL_VERSION__ >= 410\n"
          "#endif\n"
          "#include <metal_stdlib>\n"
          "using namespace metal;\n"
          "kernel void mfa_m41_probe(device int* o [[buffer(0)]],\n"
          "                          uint i [[thread_position_in_grid]]) {\n"
          "    int acc = 0;\n"
          "    for (uint j = 0; j <= i; ++j) acc += (int)j;\n"
          "    o[i] = acc;\n"
          "}\n";

      auto s = mlx::core::default_stream(mlx::core::Device::gpu);
      auto& d = mlx::core::metal::device(s.device);
      void* dev_raw = d.mtl_device();
      if (!dev_raw) return false;

      // compile_shader throws on compile failure (incl. the #error on 4.0).
      void* pso_raw = ShaderCache::get().compile_shader(src, "mfa_m41_probe", dev_raw);
      if (!pso_raw) return false;

      id<MTLDevice> device = (__bridge id<MTLDevice>)dev_raw;
      id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pso_raw;

      const int N = 64;
      id<MTLBuffer> out = [device newBufferWithLength:N * sizeof(int)
                                             options:MTLResourceStorageModeShared];
      if (!out) return false;
      memset([out contents], 0xFF, N * sizeof(int));  // poison → catch no-write

      id<MTLCommandQueue> q = [device newCommandQueue];
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:out offset:0 atIndex:0];
      [enc dispatchThreads:MTLSizeMake(N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(N, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if ([cb error] != nil) return false;

      const int* o = (const int*)[out contents];
      for (int i = 0; i < N; ++i) {
        if (o[i] != (i * (i + 1)) / 2) return false;  // functional proof
      }
      return true;  // compiled at 4.1 AND executed correctly
    }
  } catch (...) {
    return false;  // fail-safe: any exception ⇒ not capable ⇒ fall back
  }
}

}  // namespace mlx_mfa
