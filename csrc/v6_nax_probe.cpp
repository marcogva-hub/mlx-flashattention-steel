/// V6 NAX bring-up probe — JIT-compile a tiny MSL 4 + MPP kernel via the
/// shader cache to validate that the toolchain works on this device.
///
/// Returns a string with the compile result. Used by Phase 0 Task 0.1 to
/// gate the rest of the V6 work.

#include "shader_cache.hpp"
#include "mfa_steel_fwd_v6_nax.hpp"
#include <mlx/backend/metal/device.h>
#include <Metal/Metal.hpp>
#include <stdexcept>
#include <string>

namespace mlx_mfa {

// Returns "OK" on success, or "FAIL: <error>" if compilation fails.
std::string v6_nax_probe_msl4() {
  auto s = mlx::core::default_stream(mlx::core::Device::gpu);
  auto& d = mlx::core::metal::device(s.device);
  void* mtl_device = d.mtl_device();

  // Minimal MSL 4 stub — no MPP yet.
  std::string source = R"MSL(// MFA_REQUIRE_MSL4
#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;

kernel void v6_stub_msl4(
    device half* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    out[tid] = (half)1.0h;
}
)MSL";

  try {
    void* pipeline = ShaderCache::get().compile_shader(
        source, "v6_stub_msl4", mtl_device);
    (void)pipeline;
    return "OK";
  } catch (const std::exception& e) {
    return std::string("FAIL: ") + e.what();
  }
}

// Returns "OK" on success, or "FAIL: <error>" if compilation fails.
std::string v6_nax_probe_mpp() {
  auto s = mlx::core::default_stream(mlx::core::Device::gpu);
  auto& d = mlx::core::metal::device(s.device);
  void* mtl_device = d.mtl_device();

  // MSL 4 + MPP matmul2d header probe. Just instantiates the descriptor
  // type to confirm the headers are visible to the runtime compiler.
  std::string source = R"MSL(// MFA_REQUIRE_MSL4
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void v6_stub_mpp(
    tensor<device half, dextents<int32_t, 2>> A,
    tensor<device half, dextents<int32_t, 2>> B,
    tensor<device half, dextents<int32_t, 2>> C,
    uint3 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(
        32, 32, 64, false, false, true);
    matmul2d<desc, execution_simdgroups<1>> op;

    auto mA = A.slice(0, tgid.y * 32);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 32);
    op.run(mA, mB, mC);
}
)MSL";

  try {
    void* pipeline = ShaderCache::get().compile_shader(
        source, "v6_stub_mpp", mtl_device);
    (void)pipeline;
    return "OK";
  } catch (const std::exception& e) {
    return std::string("FAIL: ") + e.what();
  }
}

// Probe: try to compile the actual V6 NAX forward kernel for D=64, FP16.
std::string v6_nax_probe_forward_compile(int head_dim, int dtype_code) {
  auto s = mlx::core::default_stream(mlx::core::Device::gpu);
  auto& d = mlx::core::metal::device(s.device);
  void* mtl_device = d.mtl_device();

  ShaderCache::KernelKey key{};
  key.type = ShaderCache::KernelKey::KernelType::SteelForwardV6NAX;
  key.head_dim = head_dim;
  key.dtype = (uint8_t)dtype_code;

  std::string source = generate_steel_v6_nax_source(key);
  try {
    void* pipeline = ShaderCache::get().compile_shader(
        source, "v6_nax_forward", mtl_device);
    (void)pipeline;
    return "OK";
  } catch (const std::exception& e) {
    return std::string("FAIL: ") + e.what();
  }
}

}  // namespace mlx_mfa

// === Draw-Things-port forward kernel: source generation + JIT compile ===
// Test that the ported NAAttentionKernel produces a valid MSL 4 string and
// that the runtime metal compiler accepts it.

#include "mfa/v6_nax/NAAttentionKernel.hpp"

namespace mlx_mfa {

std::string v6_nax_dt_generate_source(int head_dim, int Hq, int Hk,
                                      int dtype_code) {
  // dtype_code: 0=FP16, 1=BF16
  GEMMOperandPrecision input_prec = (dtype_code == 1)
      ? GEMMOperandPrecision::BF16
      : GEMMOperandPrecision::FP16;

  // Memory precisions (Q, K, V, O all input precision; intermediates float).
  AttentionOperands<GEMMOperandPrecision> mp;
  mp[AttentionOperand::Q] = input_prec;
  mp[AttentionOperand::K] = input_prec;
  mp[AttentionOperand::V] = input_prec;
  mp[AttentionOperand::O] = input_prec;
  mp[AttentionOperand::S] = GEMMOperandPrecision::FP32;
  mp[AttentionOperand::P] = GEMMOperandPrecision::FP32;
  mp[AttentionOperand::L] = GEMMOperandPrecision::FP32;

  // Block dimensions: parallelization, traversal, head.
  // For D=64: BQ=32 BK=32; for D=128: BQ=32 BK=32. Conservative defaults.
  unsigned short BQ = 32;
  unsigned short BK = 32;
  simd::ushort3 blockDims = simd::make_ushort3(BQ, BK, (unsigned short)head_dim);

  NAAttentionKernelDescriptor desc(
      blockDims, (unsigned short)head_dim, (unsigned short)Hq,
      (unsigned short)Hk, /*executionSIMDGroups=*/4, /*checkCEdge1=*/true, mp,
      AttentionKernelType::forward, /*scale=*/1.0f / 8.0f,
      /*bypassThreadgroupMemory=*/false);

  NAAttentionKernel kernel(desc);
  return kernel.source;
}

// Forward decl from v6_nax_compile.mm
void* v6_nax_compile_with_constants(
    const std::string& source, const std::string& function_name,
    void* raw_device,
    uint32_t R, uint32_t C, uint32_t Q_bs, uint32_t K_bs,
    uint32_t V_bs, uint32_t O_bs);

std::string v6_nax_dt_compile(int head_dim, int Hq, int Hk, int dtype_code) {
  auto s = mlx::core::default_stream(mlx::core::Device::gpu);
  auto& d = mlx::core::metal::device(s.device);
  void* mtl_device = d.mtl_device();

  std::string source = v6_nax_dt_generate_source(head_dim, Hq, Hk, dtype_code);

  // Use trial values for function constants. The Draw Things kernel takes
  // R, C, and batch strides as function constants. Set them to a 64×64
  // trial to test compilation.
  uint32_t R = 64, C = 64;
  uint32_t qbs = (uint32_t)(Hq * head_dim * R);
  uint32_t kbs = (uint32_t)(Hk * head_dim * C);
  uint32_t vbs = kbs;
  uint32_t obs = qbs;

  try {
    void* pipeline = v6_nax_compile_with_constants(
        source, "attention", mtl_device, R, C, qbs, kbs, vbs, obs);
    (void)pipeline;
    return "OK";
  } catch (const std::exception& e) {
    return std::string("FAIL: ") + e.what();
  }
}

}  // namespace mlx_mfa
