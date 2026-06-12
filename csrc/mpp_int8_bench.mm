/// mpp_int8_bench.mm — Phase II-2 (campaign 2026-06) §AA.5 kill-gate
/// microbench: MPP matmul2d sustained throughput, int8 vs fp16, at
/// attention tile shapes (M=64, N=64, K=128 — the Sage QK^T tile).
///
/// Kill threshold (recorded in sprint-C-report): int8 < 1.3x fp16
/// sustained means the Sage-NAX int8 kernel sprint dies at the gate.
///
/// Sprint II-2R RECONCILIATION (2026-06-12, supersedes the II-5
/// revision note below and the II-2 verdict): int8 matmul2d IS
/// implemented on macOS 26.4 / M5 in BOTH the device-tensor forms
/// (including 64x64x128 — II-2's exact tile) AND the full-cooperative
/// form.  The II-2 probe failed on a TYPE SPELLING bug: it declared
/// operands as `char`, which in Metal C++ is a distinct type from
/// `int8_t` (= `signed char`); MPP's dispatch is keyed on
/// `__is_same_v<T, int8_t>` / `uint8_t` exactly, so `char` falls
/// through every combo list in every form.  All five II-2 variants
/// used `char` → all five failed → "unimplemented" was a probe
/// artifact, not a runtime gap.  (The full-coop variant additionally
/// violated the coop-coop dims constraint: M,N,K each in {16,32}
/// with at least one == 32 — that constraint applies ONLY when both
/// inputs are cooperative tensors; device-tensor forms accept the
/// big tiles.)  All variants below now use `int8_t`.
/// Compile matrix (verified 2026-06-12): full-coop int8 16x32x16 /
/// 32x32x16 / 32x32x32 OK, 16x16x16 + 64x64x128 dims-rejected;
/// device-tensor int8 16x32x16 / 32x32x32 / 64x64x32 / 64x64x128 OK.
///
/// Sprint II-5 historical note (diagnosis incomplete — kept for the
/// record): the II-5 probe established the working full-coop
/// configuration (matches Mininglamp-AI/cider w8a8_matmul.metal, MIT,
/// and Draw Things Metal Quantized Attention):
///   matmul2d_descriptor(16, 32, 16, false, true, true,
///                       mode::multiply_accumulate)
///   matmul2d<desc, metal::execution_simdgroup>
///   get_*_cooperative_tensor<int8_t, int8_t, int32_t>()
///   element-wise register fill (no .load() from device tensor<>).
/// Measured sustained (II-5 standalone probe, M5 Max, 320 TGs x 4 SGs):
///   int8/i32 233 TOPS vs f16/f16 124 TFLOPS = 1.88x  → GATE PASSES.
/// The cider-form variant below is tried FIRST; the historical II-2
/// variants are retained for regression tracking across macOS updates.
///
/// Dispatches via raw Metal (GPU-side timing from MTLCommandBuffer
/// GPUStart/EndTime), compiled through ShaderCache::compile_shader
/// (the MFA_REQUIRE_MSL4 sentinel selects MTLLanguageVersion4 — MPP
/// headers cannot compile through mx.fast.metal_kernel, verified
/// Sprint C).

#include "shader_cache.hpp"
#include <mlx/backend/metal/device.h>
#import <Metal/Metal.h>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

namespace mlx_mfa {

// Cider-form variant (Sprint II-5, the WORKING int8 binding): full-
// cooperative operands at fragment dims (16,32,16) — the only dims the
// MPP header's static_assert admits for coop-operand int8 — with
// element-type template args and element-wise register fill.  One
// simdgroup per threadgroup; C is a plain device pointer sink.
static std::string bench_kernel_ciderform_src(const char* in_ty,
                                              const char* acc_ty,
                                              const char* fn_name, int reps) {
  std::ostringstream ss;
  ss << R"MSL(// MFA_REQUIRE_MSL4
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void )MSL";
  ss << fn_name << R"MSL((
    const device )MSL" << in_ty << R"MSL(* A [[buffer(0)]],
    const device )MSL" << in_ty << R"MSL(* B [[buffer(1)]],
    device )MSL" << acc_ty << R"MSL(* C [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto a = op.get_left_input_cooperative_tensor<
        )MSL" << in_ty << ", " << in_ty << ", " << acc_ty << R"MSL(>();
    auto b = op.get_right_input_cooperative_tensor<
        )MSL" << in_ty << ", " << in_ty << ", " << acc_ty << R"MSL(>();
    auto c = op.get_destination_cooperative_tensor<
        decltype(a), decltype(b), )MSL" << acc_ty << R"MSL(>();
    for (ushort i = 0; i < a.get_capacity(); ++i) a[i] = A[i & 63];
    for (ushort i = 0; i < b.get_capacity(); ++i) b[i] = B[i & 63];
    for (ushort i = 0; i < c.get_capacity(); ++i) c[i] = 0;
    for (int r = 0; r < )MSL" << reps << R"MSL(; ++r) op.run(a, b, c);
    C[tgid.y] = c[0];  // sink defeats DCE
}
)MSL";
  return ss.str();
}

// Cooperative-destination variant: the int8 path may only be implemented
// for cooperative destination tensors (the Draw Things NAInt8 pattern).
static std::string bench_kernel_coop_src(const char* in_ty, const char* acc_ty,
                                         const char* fn_name, int reps) {
  std::ostringstream ss;
  ss << R"MSL(// MFA_REQUIRE_MSL4
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void )MSL";
  ss << fn_name << R"MSL((
    tensor<device )MSL" << in_ty << R"MSL(, dextents<int32_t, 2>> A,
    tensor<device )MSL" << in_ty << R"MSL(, dextents<int32_t, 2>> B,
    tensor<device )MSL" << acc_ty << R"MSL(, dextents<int32_t, 2>> C,
    uint3 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(
        64, 64, 128, false, false, true);
    matmul2d<desc, execution_simdgroups<1>> op;

    auto mA = A.slice(0, (int)tgid.y * 64);
    auto mB = B.slice((int)tgid.x * 64, 0);
    auto cC = op.get_destination_cooperative_tensor<
        decltype(mA), decltype(mB), )MSL" << acc_ty << R"MSL(>();
    for (int r = 0; r < )MSL" << reps << R"MSL(; ++r) {
        op.run(mA, mB, cC);
    }
    auto mC = C.slice((int)tgid.x * 64, (int)tgid.y * 64);
    cC.store(mC);
}
)MSL";
  return ss.str();
}

// Full-cooperative form: char inputs loaded into cooperative operand
// tensors (registers), cooperative int32 destination — the Draw Things
// NAInt8 register-level pattern.
static std::string bench_kernel_fullcoop_src(const char* fn_name, int reps) {
  std::ostringstream ss;
  ss << R"MSL(// MFA_REQUIRE_MSL4
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void )MSL";
  ss << fn_name << R"MSL((
    tensor<device int8_t, dextents<int32_t, 2>> A,
    tensor<device int8_t, dextents<int32_t, 2>> B,
    tensor<device int, dextents<int32_t, 2>> C,
    uint3 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(
        64, 64, 128, false, false, true);
    matmul2d<desc, execution_simdgroups<1>> op;

    auto ctA = op.get_left_input_cooperative_tensor<
        decltype(A), decltype(B), int>();
    auto ctB = op.get_right_input_cooperative_tensor<
        decltype(A), decltype(B), int>();
    auto ctC = op.get_destination_cooperative_tensor<
        decltype(ctA), decltype(ctB), int>();
    ctA.load(A.slice(0, (int)tgid.y * 64));
    ctB.load(B.slice((int)tgid.x * 64, 0));
    for (int r = 0; r < )MSL" << reps << R"MSL(; ++r) {
        op.run(ctA, ctB, ctC);
    }
    ctC.store(C.slice((int)tgid.x * 64, (int)tgid.y * 64));
}
)MSL";
  return ss.str();
}

static std::string bench_kernel_src(const char* in_ty, const char* acc_ty,
                                    const char* fn_name, int reps) {
  std::ostringstream ss;
  ss << R"MSL(// MFA_REQUIRE_MSL4
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void )MSL";
  ss << fn_name << R"MSL((
    tensor<device )MSL" << in_ty << R"MSL(, dextents<int32_t, 2>> A,
    tensor<device )MSL" << in_ty << R"MSL(, dextents<int32_t, 2>> B,
    tensor<device )MSL" << acc_ty << R"MSL(, dextents<int32_t, 2>> C,
    uint3 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(
        64, 64, 128, false, false, true);
    matmul2d<desc, execution_simdgroups<1>> op;

    auto mA = A.slice(0, (int)tgid.y * 64);
    auto mB = B.slice((int)tgid.x * 64, 0);
    auto mC = C.slice((int)tgid.x * 64, (int)tgid.y * 64);
    for (int r = 0; r < )MSL" << reps << R"MSL(; ++r) {
        op.run(mA, mB, mC);
    }
}
)MSL";
  return ss.str();
}

static double run_bench(void* mtl_device_raw, const std::string& src,
                        const char* fn_name, size_t elem_in, size_t elem_acc,
                        int reps, int tgs, int iters,
                        // FLOPs per rep per threadgroup; default = the
                        // historical 64x64x128 device-tensor tile.
                        double flops_per_rep_tg = 2.0 * 64 * 64 * 128) {
  id<MTLDevice> device = (__bridge id<MTLDevice>)mtl_device_raw;
  void* pso_raw = ShaderCache::get().compile_shader(src, fn_name,
                                                    mtl_device_raw);
  id<MTLComputePipelineState> pso =
      (__bridge id<MTLComputePipelineState>)pso_raw;

  // A: [64*tgs_y, 128]; B: [128, 64*tgs_x]; C: [64*tgs_x, 64*tgs_y].
  // Use a tgs x 1 grid: A rows = 64*tgs, B cols = 64.
  const size_t M = 64ull * tgs, K = 128, N = 64;
  id<MTLBuffer> bufA = [device newBufferWithLength:M * K * elem_in
                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufB = [device newBufferWithLength:K * N * elem_in
                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufC = [device newBufferWithLength:M * N * elem_acc
                                           options:MTLResourceStorageModeShared];
  memset([bufA contents], 1, M * K * elem_in);
  memset([bufB contents], 1, K * N * elem_in);

  id<MTLCommandQueue> queue = [device newCommandQueue];
  std::vector<double> times;
  for (int it = 0; it < iters + 3; ++it) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    // MSL4 tensor arguments bind as buffers in declaration order.
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(1, tgs, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb error]) {
      throw std::runtime_error(
          std::string("dispatch error: ") +
          [[[cb error] localizedDescription] UTF8String]);
    }
    if (it >= 3) times.push_back([cb GPUEndTime] - [cb GPUStartTime]);
  }
  std::sort(times.begin(), times.end());
  const double med = times[times.size() / 2];
  const double flops = flops_per_rep_tg * (double)reps * tgs;
  return flops / med / 1e12;  // TFLOPS (or TOPS for int8)
}

std::string mpp_int8_microbench() {
  auto s = mlx::core::default_stream(mlx::core::Device::gpu);
  auto& d = mlx::core::metal::device(s.device);
  void* mtl_device = d.mtl_device();

  const int reps = 512, tgs = 160, iters = 30;
  std::ostringstream out;

  // Sprint II-5: cider-form (16,32,16 full-coop) — the binding that
  // actually works for int8 on macOS 26.4.  Tried first; same form for
  // fp16 so the kill-gate ratio is apples-to-apples.  Higher reps/tgs
  // because the per-rep tile is 32x smaller than 64x64x128.
  try {
    const int cf_reps = 20000, cf_tgs = 1280, cf_iters = 9;
    const double cf_flops = 2.0 * 16 * 32 * 16;
    double cf_f16 = run_bench(
        mtl_device, bench_kernel_ciderform_src("half", "half", "cf_f16", cf_reps),
        "cf_f16", 2, 2, cf_reps, cf_tgs, cf_iters, cf_flops);
    double cf_i8 = run_bench(
        mtl_device, bench_kernel_ciderform_src("int8_t", "int32_t", "cf_i8", cf_reps),
        "cf_i8", 1, 4, cf_reps, cf_tgs, cf_iters, cf_flops);
    out << "ciderform16x32x16: fp16=" << cf_f16 << " TF int8/i32=" << cf_i8
        << " TOPS ratio=" << (cf_i8 / cf_f16) << " | legacy64x64x128: ";
  } catch (const std::exception& e) {
    out << "ciderform16x32x16=FAIL(" << std::string(e.what()).substr(0, 200)
        << ") | legacy64x64x128: ";
  }

  try {
    double tf_f16 = run_bench(
        mtl_device, bench_kernel_src("half", "float", "mm_f16", reps),
        "mm_f16", 2, 4, reps, tgs, iters);
    out << "fp16=" << tf_f16 << " TF";
    try {
      double tf_i8 = run_bench(
          mtl_device, bench_kernel_src("int8_t", "int", "mm_i8", reps),
          "mm_i8", 1, 4, reps, tgs, iters);
      out << " int8/i32=" << tf_i8 << " TOPS ratio=" << (tf_i8 / tf_f16);
    } catch (const std::exception&) {
      // int32 destination unsupported for plain device tensors on this
      // MPP version — try the mixed char x char -> float destination.
      try {
        double tf_i8f = run_bench(
            mtl_device, bench_kernel_src("int8_t", "float", "mm_i8f", reps),
            "mm_i8f", 1, 4, reps, tgs, iters);
        out << " int8/f32dest=" << tf_i8f << " TOPS ratio=" << (tf_i8f / tf_f16);
      } catch (const std::exception& e2) {
        try {
          double tf_i8h = run_bench(
              mtl_device, bench_kernel_src("int8_t", "half", "mm_i8h", reps),
              "mm_i8h", 1, 2, reps, tgs, iters);
          out << " int8/f16dest=" << tf_i8h << " TOPS ratio=" << (tf_i8h / tf_f16);
        } catch (const std::exception&) {
          // Final variant: cooperative destination (Draw Things NAInt8 form).
          try {
            double tf_i8c = run_bench(
                mtl_device,
                bench_kernel_coop_src("int8_t", "int", "mm_i8c", reps),
                "mm_i8c", 1, 4, reps, tgs, iters);
            out << " int8/coop_i32=" << tf_i8c << " TOPS ratio="
                << (tf_i8c / tf_f16);
          } catch (const std::exception&) {
            try {
              double tf_i8fc = run_bench(
                  mtl_device, bench_kernel_fullcoop_src("mm_i8fc", reps),
                  "mm_i8fc", 1, 4, reps, tgs, iters);
              out << " int8/fullcoop=" << tf_i8fc << " TOPS ratio="
                  << (tf_i8fc / tf_f16);
            } catch (const std::exception& e5) {
              out << " int8=ALL_VARIANTS_FAIL (plain i32/f32/f16, coop-dest "
                  << "i32, full-coop i32): "
                  << std::string(e5.what()).substr(0, 300);
            }
          }
        }
      }
    }
  } catch (const std::exception& e) {
    return std::string("FAIL fp16 baseline: ") + e.what();
  }
  return out.str();
}

}  // namespace mlx_mfa
