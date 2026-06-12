/// mlx-mfa nanobind bindings.

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <mlx/mlx.h>
#include <mlx/backend/metal/device.h>

#include "mfa_attention.hpp"
#include "mfa_env.hpp"
#include "shader_cache.hpp"

namespace mlx_mfa {
// estimate_gpu_cores defined in mfa_steel_fwd_v2.cpp
int estimate_gpu_cores(const std::string& device_name, int arch_gen);
// V6 NAX bring-up probes (in csrc/v6_nax_probe.cpp).
std::string v6_nax_probe_msl4();
std::string v6_nax_probe_mpp();
std::string v6_nax_probe_forward_compile(int head_dim, int dtype_code);
std::string v34_probe_source();
std::string v34_probe_compile_test(void* mtl_device_raw);
std::string mpp_int8_microbench();  // Phase II-2 kill-gate
// V6 NAX hardware detection (in csrc/v6_nax_detect.mm).
bool device_has_neural_accelerators();
bool device_has_nax_bf16();
// Draw Things port: source generation + JIT compile
std::string v6_nax_dt_generate_source(int head_dim, int Hq, int Hk, int dtype_code);
std::string v6_nax_dt_compile(int head_dim, int Hq, int Hk, int dtype_code);
// V6 NAX forward (returns O, L).  v2.37.0: optional force_v34 to route
// V34 forward path even on D=64 small-Nk shapes (used by V34 backward
// integration to obtain natural-log lse).
std::pair<mlx::core::array, mlx::core::array> v6_nax_forward(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, bool causal, bool force_v34 = false);
// V34 backward dQ (V34 backward Option β Phase 1).  Returns dQ; consumes
// O + lse from V34 forward.  Routing constraint per DC12: caller must
// ensure V34-forward-eligible shape (D=128 always; D=64 with Nk>8000).
mlx::core::array v6_nax_backward_query(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& o,
    const mlx::core::array& lse, const mlx::core::array& d_o,
    const mlx::core::array& d_vec,  // v2.38.1: precomputed rowsum(dO⊙O)
    float scale, bool causal);
// V34 backward dK/dV (V34 backward Option β Phase 2).  Single-SG WM=1
// kernel; one TG per K-tile.  Returns (dK, dV) shaped [B, Hq, kL, D]
// each (per-Q-head; GQA reduction is caller's responsibility).
std::pair<mlx::core::array, mlx::core::array> v6_nax_backward_kv(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& o,
    const mlx::core::array& lse, const mlx::core::array& d_o,
    const mlx::core::array& d_vec,  // v2.38.1: precomputed rowsum(dO⊙O)
    float scale, bool causal);
// V34 backward dV-only Phase 2.O2 multi-SG (Q-row partition).  Returns
// dV_partials [B, Hq, WM, kL, D] FP32.  Caller reduces via mx.sum(axis=2)
// and casts to T to obtain final dV [B, Hq, kL, D].
// NOTE: dV does NOT need D (= rowsum(dO⊙O)) — dV = P^T @ dO; no dS term.
mlx::core::array v6_nax_backward_dv_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& lse,
    const mlx::core::array& d_o, float scale, int wm, bool causal);
// V34 backward dV SPARSE — Prompt 5b Section A PoC.
mlx::core::array v6_nax_backward_dv_sparse_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& lse,
    const mlx::core::array& d_o, const mlx::core::array& block_mask,
    float scale, int wm, bool causal);
// v2.50 Prompt 5d Section A — dQ + dK split + fused dKdV sparse.
mlx::core::array v6_nax_backward_query_sparse_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& o,
    const mlx::core::array& lse, const mlx::core::array& d_o,
    const mlx::core::array& d_vec,
    const mlx::core::array& block_mask,
    float scale, bool causal);
mlx::core::array v6_nax_backward_dk_sparse_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& o,
    const mlx::core::array& lse, const mlx::core::array& d_o,
    const mlx::core::array& d_vec,
    const mlx::core::array& block_mask,
    float scale, int wm, bool causal);
std::pair<mlx::core::array, mlx::core::array>
v6_nax_backward_fused_dkdv_sparse_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& lse,
    const mlx::core::array& d_o, const mlx::core::array& d_vec,
    const mlx::core::array& block_mask,
    float scale, int wm, bool causal);
// V34 backward dK-only Phase 2.O2 (sister kernel).  Same shape contract
// as dV; takes additional O input (for D = rowsum(dO⊙O)).
mlx::core::array v6_nax_backward_dk_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& o,
    const mlx::core::array& lse, const mlx::core::array& d_o,
    const mlx::core::array& d_vec,  // v2.38.1: precomputed rowsum(dO⊙O)
    float scale, int wm, bool causal);
// V34 backward FUSED dK+dV (Sprint v2.39.0 Phase C.1.a, Option γ).  Single
// kernel computes both gradients in one K-tile load (K-bandwidth amortization
// per /metal-kernel-dev audit).  D=64 only this PR; D=128 deferred to
// Phase C.1.b.  Returns (dK_partials, dV_partials) both [B, Hq, WM, kL, D]
// FP32; caller reduces via mx.sum(axis=2) and casts to T.
std::pair<mlx::core::array, mlx::core::array> v6_nax_backward_fused_dkdv_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& lse,
    const mlx::core::array& d_o,
    const mlx::core::array& d_vec,
    float scale, int wm, bool causal);
}  // namespace mlx_mfa

#include "mfa_paged_gather.hpp"
#include "mfa_quantize.hpp"
#include "mfa_scatter.hpp"
#include "mfa_smooth_quant.hpp"
#include "mfa_steel_fwd_v2.hpp"
#include "mfa_conv_nax.hpp"
#include "mfa_sparse_attention.hpp"

#include <array>

namespace nb = nanobind;

NB_MODULE(_ext, m) {
  m.doc() = "mlx-mfa C++ extension: Metal Flash Attention for MLX";


  m.def(
      "mfa_attention_forward",
      &mlx_mfa::mfa_attention_forward,
      nb::arg("q"),
      nb::arg("k"),
      nb::arg("v"),
      nb::arg("scale"),
      nb::arg("causal"),
      nb::arg("softcap") = 0.0f,
      nb::arg("window_left") = -1,
      nb::arg("window_right") = -1,
      nb::arg("stream") = nb::none(),
      "Flash Attention forward (Metal). "
      "q/k/v: [B, H, N, D], float16/bfloat16/float32. "
      "softcap: tanh softcapping factor (0.0 = disabled). "
      "window_left: sliding window left radius (-1 = disabled). "
      "window_right: sliding window right radius (-1 = disabled).");

  // Debug: returns (O, L) so L (logsumexp) can be inspected from Python.
  m.def("mfa_forward_with_lse",
      [](const mlx::core::array& q,
         const mlx::core::array& k,
         const mlx::core::array& v,
         float scale, bool causal) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        // D.5: enforce row-major layout; no-op when already contiguous.
        auto qc = mlx::core::contiguous(q, false, s);
        auto kc = mlx::core::contiguous(k, false, s);
        auto vc = mlx::core::contiguous(v, false, s);
        mlx_mfa::MFAttention::Params params{
            (int)qc.shape(3), scale, causal,
            false, false, false, 0, 0.0f, false,
            /*has_attn_bias=*/false, /*attn_bias_mode=*/(uint8_t)0,
            /*window_left=*/-1, /*window_right=*/-1};
        mlx::core::Shape lse_shape = {qc.shape(0), qc.shape(1), qc.shape(2)};
        auto outs = mlx::core::array::make_arrays(
            {qc.shape(), lse_shape},
            {qc.dtype(), mlx::core::float32},
            std::make_shared<mlx_mfa::MFAttention>(s, params),
            {qc, kc, vc});
        return std::make_pair(outs[0], outs[1]);
      },
      nb::arg("q"), nb::arg("k"), nb::arg("v"),
      nb::arg("scale"), nb::arg("causal"),
      "Debug: returns (O, L) where L is the logsumexp in log2 domain.");

  // Debug: runs MFABackwardQuery kernel directly (bypasses vjp tape).
  // Takes (q, k, v, O, L, dO, scale, causal) — all pre-evaluated.
  // Returns (dQ, D_computed) so D_computed can be compared to reference.
  m.def("mfa_backward_query_debug",
      [](const mlx::core::array& q,
         const mlx::core::array& k,
         const mlx::core::array& v,
         const mlx::core::array& O,
         const mlx::core::array& L,
         const mlx::core::array& dO,
         float scale, bool causal) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        mlx_mfa::MFAttention::Params params{
            (int)q.shape(3), scale, causal,
            false, false, false, 0, 0.0f, false,
            /*has_attn_bias=*/false, /*attn_bias_mode=*/(uint8_t)0,
            /*window_left=*/-1, /*window_right=*/-1};
        mlx::core::Shape d_shape = {q.shape(0), q.shape(1), q.shape(2)};
        auto outs = mlx::core::array::make_arrays(
            {q.shape(), d_shape},
            {q.dtype(), mlx::core::float32},
            std::make_shared<mlx_mfa::MFABackwardQuery>(s, params),
            {q, k, v, O, L, dO});
        return std::make_pair(outs[0], outs[1]);
      },
      nb::arg("q"), nb::arg("k"), nb::arg("v"),
      nb::arg("O"), nb::arg("L"), nb::arg("dO"),
      nb::arg("scale"), nb::arg("causal"),
      "Debug: returns (dQ, D_computed) from MFABackwardQuery kernel directly.");

  // Debug: runs MFABackwardKeyValue kernel directly (bypasses vjp tape).
  // Takes (q, k, v, O, L, D_computed, dO, scale, causal) — all pre-evaluated.
  // Returns (dK, dV).
  m.def("mfa_backward_kv_debug",
      [](const mlx::core::array& q,
         const mlx::core::array& k,
         const mlx::core::array& v,
         const mlx::core::array& O,
         const mlx::core::array& L,
         const mlx::core::array& D,
         const mlx::core::array& dO,
         float scale, bool causal) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        mlx_mfa::MFAttention::Params params{
            (int)q.shape(3), scale, causal,
            false, false, false, 0, 0.0f, false,
            /*has_attn_bias=*/false, /*attn_bias_mode=*/(uint8_t)0,
            /*window_left=*/-1, /*window_right=*/-1};
        auto outs = mlx::core::array::make_arrays(
            {k.shape(), v.shape()},
            {k.dtype(), v.dtype()},
            std::make_shared<mlx_mfa::MFABackwardKeyValue>(s, params),
            {q, k, v, O, L, D, dO});
        return std::make_pair(outs[0], outs[1]);
      },
      nb::arg("q"), nb::arg("k"), nb::arg("v"),
      nb::arg("O"), nb::arg("L"), nb::arg("D"), nb::arg("dO"),
      nb::arg("scale"), nb::arg("causal"),
      "Debug: returns (dK, dV) from MFABackwardKeyValue kernel directly.");

  // STEEL backward: dispatches MFASteelBwdDQ + MFASteelBwdDKV.
  // Args: q,k,v,O,L,dO — all pre-evaluated on GPU (caller owns L).
  // scale, causal: forward hyperparameters.
  // Returns: tuple (dQ, dK, dV).
  // Only supports f16/bf16 with D<=128.
  m.def("mfa_steel_backward",
      [](const mlx::core::array& q,
         const mlx::core::array& k,
         const mlx::core::array& v,
         const mlx::core::array& O,
         const mlx::core::array& L,
         const mlx::core::array& dO,
         float scale, bool causal) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        // G.1: Materialise all 6 inputs before Metal kernel dispatch.
        // MLX autograd may recycle GPU buffers; mlx::core::eval() fences against aliasing.
        mlx::core::eval(std::vector<mlx::core::array>{q, k, v, O, L, dO});
        mlx_mfa::MFAttention::Params params{};
        params.head_dim    = q.shape(3);
        params.scale       = scale;
        params.causal      = causal;
        params.window_left = -1;  // disabled — steel backward has no window

        // delta = rowsum(dO * O)  [B, H, N], float32.
        // Note: the Metal kernel multiplies by p->scale internally when computing
        // dS = scale * P * (dP - delta).  Do NOT pre-multiply by scale here.
        auto dO_f32 = mlx::core::astype(dO, mlx::core::float32, s);
        auto O_f32  = mlx::core::astype(O,  mlx::core::float32, s);
        auto delta  = mlx::core::sum(
                          mlx::core::multiply(dO_f32, O_f32, s),
                          std::vector<int>{3}, false, s);

        // dQ
        auto bwd_q = mlx::core::array::make_arrays(
            {q.shape()},
            {q.dtype()},
            std::make_shared<mlx_mfa::MFASteelBwdDQ>(s, params),
            {q, k, v, O, L, dO, delta});

        // dK, dV
        auto bwd_kv = mlx::core::array::make_arrays(
            {k.shape(), v.shape()},
            {k.dtype(), v.dtype()},
            std::make_shared<mlx_mfa::MFASteelBwdDKV>(s, params),
            {q, k, v, O, L, delta, dO});

        return nb::make_tuple(bwd_q[0], bwd_kv[0], bwd_kv[1]);
      },
      nb::arg("q"), nb::arg("k"), nb::arg("v"),
      nb::arg("O"), nb::arg("L"), nb::arg("dO"),
      nb::arg("scale"), nb::arg("causal"),
      "STEEL backward: returns (dQ, dK, dV). f16/bf16, D<=128 only.");

  m.def("mfa_steel_backward_sparse",
      [](const mlx::core::array& q,
         const mlx::core::array& k,
         const mlx::core::array& v,
         const mlx::core::array& O,
         const mlx::core::array& L,
         const mlx::core::array& dO,
         const mlx::core::array& block_mask,
         float scale, bool causal) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        mlx_mfa::MFAttention::Params params{};
        params.head_dim       = q.shape(3);
        params.scale          = scale;
        params.causal         = causal;
        params.has_block_mask = true;
        params.window_left    = -1;

        // delta = rowsum(dO * O)  [B, H, N], float32.
        auto dO_f32 = mlx::core::astype(dO, mlx::core::float32, s);
        auto O_f32  = mlx::core::astype(O,  mlx::core::float32, s);
        auto delta  = mlx::core::sum(
                          mlx::core::multiply(dO_f32, O_f32, s),
                          std::vector<int>{3}, false, s);

        // dQ — inputs[7] = block_mask
        auto bwd_q = mlx::core::array::make_arrays(
            {q.shape()},
            {q.dtype()},
            std::make_shared<mlx_mfa::MFASteelBwdDQ>(s, params),
            {q, k, v, O, L, dO, delta, block_mask});

        // dK, dV — inputs[7] = block_mask
        auto bwd_kv = mlx::core::array::make_arrays(
            {k.shape(), v.shape()},
            {k.dtype(), v.dtype()},
            std::make_shared<mlx_mfa::MFASteelBwdDKV>(s, params),
            {q, k, v, O, L, delta, dO, block_mask});

        return nb::make_tuple(bwd_q[0], bwd_kv[0], bwd_kv[1]);
      },
      nb::arg("q"), nb::arg("k"), nb::arg("v"),
      nb::arg("O"), nb::arg("L"), nb::arg("dO"),
      nb::arg("block_mask"), nb::arg("scale"), nb::arg("causal"),
      "Sparse STEEL backward: block_mask skips inactive tiles. Returns (dQ, dK, dV).");

  m.def("_mlx_build_version", []() -> std::string {
#ifdef MLX_BUILD_VERSION
    return MLX_BUILD_VERSION;
#else
    return "unknown";
#endif
  }, "MLX version used at compile time (major.minor.patch).");

  m.def("shader_cache_size", []() {
    return mlx_mfa::ShaderCache::get().size();
  }, "Number of cached Metal compute pipelines.");

  m.def("shader_cache_clear", []() {
    mlx_mfa::ShaderCache::get().clear();
  }, "Clear the Metal pipeline cache.");

  // Returns a dict with:
  //   gpu_family_gen  int  — GPU silicon generation from architecture string
  //                          (e.g. "applegpu_g13s" → 13).
  //                          13=M1, 14=M2, 15=M3, 16=M4.
  //   is_m3_plus      bool — True if gen >= 15 (M3/M4: preferAsyncCache block params)
  //   device_name     str  — MTLDevice name (e.g. "Apple M1 Max")
  //   gpu_cores       int  — Estimated physical GPU core count from device name.
  //                          Correct per-variant: M1 Max=32, M1 base=8, M2 Max=38, …
  //                          Falls back to conservative gen-based estimate for
  //                          unknown names (simulator, future hardware).
  // V6 NAX bring-up probes — JIT-compile minimal MSL 4 + MPP kernels via
  // mlx-mfa's shader cache. Used by Phase 0 Task 0.1 to gate the rest of
  // the V6 NAX implementation.
  m.def("v6_nax_probe_msl4", []() -> std::string {
    return mlx_mfa::v6_nax_probe_msl4();
  }, "Probe: compile a minimal MSL 4.0 stub. Returns 'OK' or 'FAIL: <err>'.");
  m.def("v6_nax_probe_mpp", []() -> std::string {
    return mlx_mfa::v6_nax_probe_mpp();
  }, "Probe: compile MSL 4 + MPP matmul2d stub. Returns 'OK' or 'FAIL: <err>'.");
  m.def("device_has_neural_accelerators", []() -> bool {
    return mlx_mfa::device_has_neural_accelerators();
  }, "True iff the GPU has NAX (Apple GPU family 10+, M5 family).");
  m.def("device_has_nax_bf16", []() -> bool {
    return mlx_mfa::device_has_nax_bf16();
  }, "True iff NAX is available AND macOS >= 26.1 (MPP bf16 support).");
  m.def("v6_nax_probe_forward_compile",
        [](int head_dim, int dtype_code) -> std::string {
          return mlx_mfa::v6_nax_probe_forward_compile(head_dim, dtype_code);
        },
        nb::arg("head_dim"), nb::arg("dtype_code"),
        "Compile the V6 NAX forward kernel (D, dtype). Returns 'OK' or 'FAIL: <err>'.");
  m.def("v34_probe_source", []() -> std::string {
    return mlx_mfa::v34_probe_source();
  });
  m.def("v34_probe_compile", []() -> std::string {
    auto s = mlx::core::default_stream(mlx::core::Device::gpu);
    auto& d = mlx::core::metal::device(s.device);
    return mlx_mfa::v34_probe_compile_test(d.mtl_device());
  });
  m.def("mpp_int8_microbench", []() -> std::string {
    return mlx_mfa::mpp_int8_microbench();
  }, "Phase II-2 kill-gate: MPP matmul2d int8 vs fp16 sustained throughput at attention tiles.");
  m.def("v6_nax_dt_generate_source",
        [](int head_dim, int Hq, int Hk, int dtype_code) -> std::string {
          return mlx_mfa::v6_nax_dt_generate_source(head_dim, Hq, Hk, dtype_code);
        },
        nb::arg("head_dim"), nb::arg("Hq"), nb::arg("Hk"), nb::arg("dtype_code"),
        "Generate MSL 4 source from the Draw Things NAAttention port (no compile).");
  m.def("v6_nax_dt_compile",
        [](int head_dim, int Hq, int Hk, int dtype_code) -> std::string {
          return mlx_mfa::v6_nax_dt_compile(head_dim, Hq, Hk, dtype_code);
        },
        nb::arg("head_dim"), nb::arg("Hq"), nb::arg("Hk"), nb::arg("dtype_code"),
        "JIT-compile the Draw Things port. Returns 'OK' or 'FAIL: <err>'.");

  m.def("v6_nax_forward",
        [](const mlx::core::array& q, const mlx::core::array& k,
           const mlx::core::array& v, bool causal, bool force_v34) {
          return mlx_mfa::v6_nax_forward(q, k, v, causal, force_v34);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("causal") = false,
        nb::arg("force_v34") = false,
        "V6 NAX forward attention. Returns (O, L). M5+ only; D in {64,128}; FP16/BF16. "
        "v2.37.0: force_v34=True overrides default routing to ensure V34 forward "
        "path (used by V34 backward integration for natural-log lse).");

  m.def("v6_nax_backward_query",
        [](const mlx::core::array& q, const mlx::core::array& k,
           const mlx::core::array& v, const mlx::core::array& o,
           const mlx::core::array& lse, const mlx::core::array& d_o,
           const mlx::core::array& d_vec,
           float scale, bool causal) {
          return mlx_mfa::v6_nax_backward_query(q, k, v, o, lse, d_o, d_vec, scale, causal);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("o"), nb::arg("lse"), nb::arg("d_o"), nb::arg("d_vec"),
        nb::arg("scale"), nb::arg("causal") = false,
        "V34 backward dQ kernel (Option β Phase 1). Consumes O + lse from "
        "V34 forward + D=rowsum(dO⊙O) precomputed (v2.38.1).  Returns dQ.  "
        "Routing constraint per DC12: caller must ensure V34-forward-eligible "
        "shape (D=128 always; D=64 with Nk>8000).  M5+ only; D in {64,128}; "
        "FP16/BF16; no causal/sparse.");

  m.def("v6_nax_backward_kv",
        [](const mlx::core::array& q, const mlx::core::array& k,
           const mlx::core::array& v, const mlx::core::array& o,
           const mlx::core::array& lse, const mlx::core::array& d_o,
           const mlx::core::array& d_vec,
           float scale, bool causal) {
          return mlx_mfa::v6_nax_backward_kv(q, k, v, o, lse, d_o, d_vec, scale, causal);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("o"), nb::arg("lse"), nb::arg("d_o"), nb::arg("d_vec"),
        nb::arg("scale"), nb::arg("causal") = false,
        "V34 backward dK/dV kernel (Option β Phase 2). Single-SG (WM=1) "
        "design. Returns (dK, dV) shaped [B, Hq, kL, D] each (per-Q-head; "
        "GQA reduction is caller's responsibility).  Routing constraint "
        "per DC12 same as v6_nax_backward_query.  D=rowsum(dO⊙O) precomputed (v2.38.1).");

  m.def("v6_nax_backward_dv_raw",
        [](const mlx::core::array& q, const mlx::core::array& k,
           const mlx::core::array& v, const mlx::core::array& lse,
           const mlx::core::array& d_o, float scale, int wm, bool causal) {
          return mlx_mfa::v6_nax_backward_dv_raw(q, k, v, lse, d_o, scale, wm, causal);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("lse"), nb::arg("d_o"),
        nb::arg("scale"), nb::arg("wm") = 4, nb::arg("causal") = false,
        "V34 backward dV-only multi-SG kernel (Phase 2.O2).  WM=4 default "
        "with Q-row partition.  Returns dV_partials [B, Hq, WM, kL, D] FP32; "
        "caller reduces via mx.sum(axis=2) and casts to T.  "
        "Does NOT take D (= rowsum(dO⊙O)) — dV = P^T @ dO has no dS term.");

  m.def("v6_nax_backward_dv_sparse_raw",
        [](const mlx::core::array& q, const mlx::core::array& k,
           const mlx::core::array& v, const mlx::core::array& lse,
           const mlx::core::array& d_o, const mlx::core::array& block_mask,
           float scale, int wm, bool causal) {
          return mlx_mfa::v6_nax_backward_dv_sparse_raw(
              q, k, v, lse, d_o, block_mask, scale, wm, causal);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("lse"), nb::arg("d_o"), nb::arg("block_mask"),
        nb::arg("scale"), nb::arg("wm") = 4, nb::arg("causal") = false,
        "V34 backward dV SPARSE kernel (Prompt 5b Section A PoC).");

  // v2.50 Prompt 5d Section A: 3 new sparse backward kernels.
  m.def("v6_nax_backward_query_sparse_raw",
        [](const mlx::core::array& q, const mlx::core::array& k,
           const mlx::core::array& v, const mlx::core::array& o,
           const mlx::core::array& lse, const mlx::core::array& d_o,
           const mlx::core::array& d_vec,
           const mlx::core::array& block_mask,
           float scale, bool causal) {
          return mlx_mfa::v6_nax_backward_query_sparse_raw(
              q, k, v, o, lse, d_o, d_vec, block_mask, scale, causal);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("o"),
        nb::arg("lse"), nb::arg("d_o"), nb::arg("d_vec"),
        nb::arg("block_mask"),
        nb::arg("scale"), nb::arg("causal") = false,
        "V34 backward dQ sparse kernel (Prompt 5d Section A.1).");

  m.def("v6_nax_backward_dk_sparse_raw",
        [](const mlx::core::array& q, const mlx::core::array& k,
           const mlx::core::array& v, const mlx::core::array& o,
           const mlx::core::array& lse, const mlx::core::array& d_o,
           const mlx::core::array& d_vec,
           const mlx::core::array& block_mask,
           float scale, int wm, bool causal) {
          return mlx_mfa::v6_nax_backward_dk_sparse_raw(
              q, k, v, o, lse, d_o, d_vec, block_mask, scale, wm, causal);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("o"),
        nb::arg("lse"), nb::arg("d_o"), nb::arg("d_vec"),
        nb::arg("block_mask"),
        nb::arg("scale"), nb::arg("wm") = 4, nb::arg("causal") = false,
        "V34 backward dK split sparse kernel (Prompt 5d Section A.2).");

  m.def("v6_nax_backward_fused_dkdv_sparse_raw",
        [](const mlx::core::array& q, const mlx::core::array& k,
           const mlx::core::array& v, const mlx::core::array& lse,
           const mlx::core::array& d_o, const mlx::core::array& d_vec,
           const mlx::core::array& block_mask,
           float scale, int wm, bool causal) {
          auto p = mlx_mfa::v6_nax_backward_fused_dkdv_sparse_raw(
              q, k, v, lse, d_o, d_vec, block_mask, scale, wm, causal);
          return nb::make_tuple(p.first, p.second);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("lse"), nb::arg("d_o"), nb::arg("d_vec"),
        nb::arg("block_mask"),
        nb::arg("scale"), nb::arg("wm") = 4, nb::arg("causal") = false,
        "V34 backward fused dK+dV sparse kernel (Prompt 5d Section A.3).");

  m.def("v6_nax_backward_dk_raw",
        [](const mlx::core::array& q, const mlx::core::array& k,
           const mlx::core::array& v, const mlx::core::array& o,
           const mlx::core::array& lse, const mlx::core::array& d_o,
           const mlx::core::array& d_vec,
           float scale, int wm, bool causal) {
          return mlx_mfa::v6_nax_backward_dk_raw(q, k, v, o, lse, d_o, d_vec, scale, wm, causal);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("o"), nb::arg("lse"), nb::arg("d_o"), nb::arg("d_vec"),
        nb::arg("scale"), nb::arg("wm") = 4, nb::arg("causal") = false,
        "V34 backward dK-only multi-SG kernel (Phase 2.O2 sister to dV).  "
        "WM=4 Q-row partition. Returns dK_partials [B, Hq, WM, kL, D] FP32; "
        "caller reduces via mx.sum(axis=2) and casts to T.");

  m.def("v6_nax_backward_fused_dkdv_raw",
        [](const mlx::core::array& q, const mlx::core::array& k,
           const mlx::core::array& v, const mlx::core::array& lse,
           const mlx::core::array& d_o,
           const mlx::core::array& d_vec,
           float scale, int wm, bool causal) {
          return mlx_mfa::v6_nax_backward_fused_dkdv_raw(
              q, k, v, lse, d_o, d_vec, scale, wm, causal);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("lse"), nb::arg("d_o"), nb::arg("d_vec"),
        nb::arg("scale"), nb::arg("wm") = 4, nb::arg("causal") = false,
        "V34 backward FUSED dK+dV kernel (Sprint v2.39.0 Phase C.1.a, "
        "Option γ).  Single kernel computes both gradients in one K-tile "
        "load (K-bandwidth amortization per /metal-kernel-dev audit).  "
        "D=64 only this PR.  Returns (dK_partials, dV_partials) both "
        "[B, Hq, WM, kL, D] FP32; caller reduces via mx.sum(axis=2) and "
        "casts to T.  Consumes v2.38.1 D_vec precompute.");

  m.def("get_device_info", []() -> nb::dict {
    auto s = mlx::core::default_stream(mlx::core::Device::gpu);
    auto& d = mlx::core::metal::device(s.device);
    int gen = d.get_architecture_gen();
    // MTL::Device::name() returns an NS::String; utf8String() gives a C string.
    auto* mtl_dev = d.mtl_device();
    std::string dev_name = mtl_dev
        ? std::string(mtl_dev->name()->utf8String())
        : "";
    int cores = mlx_mfa::estimate_gpu_cores(dev_name, gen);
    if (dev_name.empty()) dev_name = "unknown";
    nb::dict info;
    info["gpu_family_gen"] = gen;
    info["is_m3_plus"]     = (gen >= 15);
    info["is_m5_plus"]     = (gen >= 17);
    info["has_nax"]        = (gen >= 17);
    info["device_name"]    = dev_name;
    info["gpu_cores"]      = cores;
    return info;
  }, "Return Metal GPU hardware info: silicon generation, M3+ flag, device name, gpu_cores.");

  // --- ALiBi-biased forward ---
  m.def(
      "mfa_attention_alibi_forward",
      &mlx_mfa::mfa_attention_alibi_forward,
      nb::arg("q"),
      nb::arg("k"),
      nb::arg("v"),
      nb::arg("alibi_slopes"),
      nb::arg("scale"),
      nb::arg("causal"),
      nb::arg("stream") = nb::none(),
      "Flash Attention with ALiBi per-head linear position biases.\n"
      "alibi_slopes: float32 [H], one slope per query head.\n"
      "Bias = slope_h * (k_pos - q_pos) added before softmax.\n"
      "Only f16/bf16 supported.");

  // --- Attention bias forward ---
  m.def(
      "mfa_attention_bias_forward",
      &mlx_mfa::mfa_attention_bias_forward,
      nb::arg("q"),
      nb::arg("k"),
      nb::arg("v"),
      nb::arg("attn_bias"),
      nb::arg("attn_bias_mode"),
      nb::arg("scale"),
      nb::arg("causal"),
      nb::arg("stream") = nb::none(),
      "Flash Attention with additive attention bias.\n"
      "attn_bias: float32. Mode 1: [1,1,1,Nkv]. Mode 2: [1,H,1,Nkv].\n"
      "Bias added to Q@K^T scores before softmax.\n"
      "Only f16/bf16, D=64/128/256. Modes 1-2 only.");

  // --- RoPE-fused forward ---
  m.def(
      "mfa_attention_rope_forward",
      &mlx_mfa::mfa_attention_rope_forward,
      nb::arg("q"),
      nb::arg("k"),
      nb::arg("v"),
      nb::arg("rotary_cos"),
      nb::arg("rotary_sin"),
      nb::arg("scale"),
      nb::arg("causal"),
      nb::arg("cache_seqlens"),
      nb::arg("interleaved") = true,
      nb::arg("stream") = nb::none(),
      "Flash Attention with in-kernel RoPE fusion.\n"
      "rotary_cos/sin: float32 [max_seq_len, D/2].\n"
      "cache_seqlens: KV cache length (absolute position of Q token 0).\n"
      "interleaved: True=LLaMA pairs (d*2,d*2+1); False=GPT-NeoX (d,d+D/2).\n"
      "Only f16/bf16 supported.");

  // --- Block-sparse forward ---
  m.def("mfa_attention_sparse_forward",
        [](mlx::core::array q, mlx::core::array k, mlx::core::array v,
           mlx::core::array block_mask,
           float scale, bool causal,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> mlx::core::array {
          return mlx_mfa::mfa_attention_sparse_forward(
              q, k, v, block_mask, scale, causal, stream);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("block_mask"),
        nb::arg("scale"), nb::arg("causal"),
        nb::arg("stream") = nb::none(),
        "Block-sparse forward attention.\n"
        "block_mask: uint8 [NQ_tiles, NK_tiles]. 1=compute, 0=skip.\n"
        "Returns O [B, H, N, D]. Only f16/bf16 supported.");

  // --- Block-sparse forward returning (O, L) for use by native backward ---
  m.def("mfa_attention_sparse_forward_with_lse",
        [](mlx::core::array q, mlx::core::array k, mlx::core::array v,
           mlx::core::array block_mask,
           float scale, bool causal,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> std::pair<mlx::core::array, mlx::core::array> {
          auto outs = mlx_mfa::mfa_attention_sparse_forward_with_lse(
              q, k, v, block_mask, scale, causal, stream);
          return {outs[0], outs[1]};
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("block_mask"),
        nb::arg("scale"), nb::arg("causal"),
        nb::arg("stream") = nb::none(),
        "Block-sparse forward returning (O, L) where L is logsumexp [B,H,N].\n"
        "Used by the native sparse backward pass to avoid recomputation.\n"
        "block_mask: uint8 [NQ_tiles, NK_tiles]. Only f16/bf16 supported.");

  // --- GNA (Generalized Neighborhood Attention) native forward ---
  m.def("mfa_gna_forward",
        [](mlx::core::array q, mlx::core::array k, mlx::core::array v,
           float scale,
           int dim0, int dim1, int dim2,
           int win0, int win1, int win2,
           int str0, int str1, int str2,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> mlx::core::array {
          return mlx_mfa::mfa_gna_forward(
              q, k, v, scale,
              dim0, dim1, dim2,
              win0, win1, win2,
              str0, str1, str2,
              stream);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("scale"),
        nb::arg("dim0"), nb::arg("dim1"), nb::arg("dim2"),
        nb::arg("win0"), nb::arg("win1"), nb::arg("win2"),
        nb::arg("str0"), nb::arg("str1"), nb::arg("str2"),
        nb::arg("stream") = nb::none(),
        "GNA (Generalized Neighborhood Attention) forward.\n"
        "Inline 3D window check — no block_mask allocation.\n"
        "D=128 only, f16/bf16. Returns O [B, H, N, D].");

  // --- STEEL varlen forward: packed [1, H, total_q, D] layout ---
  m.def("mfa_attention_varlen_forward",
        [](mlx::core::array q, mlx::core::array k, mlx::core::array v,
           mlx::core::array cu_seqlens_q, mlx::core::array cu_seqlens_k,
           mlx::core::array tile_offsets,
           float scale, bool causal,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> std::pair<mlx::core::array, mlx::core::array> {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_attention_varlen_forward(
              q, k, v, cu_seqlens_q, cu_seqlens_k, tile_offsets,
              scale, causal, s);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("cu_seqlens_q"), nb::arg("cu_seqlens_k"),
        nb::arg("tile_offsets"),
        nb::arg("scale"), nb::arg("causal"),
        nb::arg("stream") = nb::none(),
        "STEEL varlen attention forward.\n"
        "\n"
        "Inputs are packed: Q/O = [1, H, total_q, D], K/V = [1, H_kv, total_kv, D].\n"
        "cu_seqlens_q: int32 [num_seqs+1], cumulative query lengths.\n"
        "cu_seqlens_k: int32 [num_seqs+1], cumulative key lengths.\n"
        "tile_offsets: int32 [num_seqs+1], cumulative Q-tile counts per sequence.\n"
        "Returns (O [1,H,total_q,D], L [1,H,total_q] logsumexp in log2 domain).\n"
        "Only f16/bf16 supported.");

  m.def("mfa_paged_kv_gather",
        [](mlx::core::array pool,
           mlx::core::array block_table,
           mlx::core::array seq_lens,
           int max_kv_len,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> mlx::core::array {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_paged_kv_gather(
              pool, block_table, seq_lens, max_kv_len, s);
        },
        nb::arg("pool"),
        nb::arg("block_table"),
        nb::arg("seq_lens"),
        nb::arg("max_kv_len"),
        nb::arg("stream") = nb::none(),
        "Metal paged KV gather: pool [N_blk, BS, H, D] -> out [B, H, max_kv, D].\n"
        "pool: f16 or bf16. block_table: int32 [B, max_blocks]. seq_lens: int32 [B].\n"
        "Transposes [BS,H,D] -> [H,BS,D] (token-major -> head-major) during gather.");

  // --- Paged STEEL forward (Track FD): kernel-level paged KV ---
  m.def("mfa_paged_steel_forward",
        [](mlx::core::array q,
           mlx::core::array k_pool,
           mlx::core::array v_pool,
           mlx::core::array block_table,
           mlx::core::array seq_lens,
           float scale,
           bool  causal,
           int   window_left,
           int   window_right,
           int   block_size,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> std::pair<mlx::core::array, mlx::core::array> {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_paged_steel_forward(
              q, k_pool, v_pool, block_table, seq_lens,
              scale, causal, window_left, window_right, block_size, s);
        },
        nb::arg("q"),
        nb::arg("k_pool"),
        nb::arg("v_pool"),
        nb::arg("block_table"),
        nb::arg("seq_lens"),
        nb::arg("scale"),
        nb::arg("causal"),
        nb::arg("window_left")  = -1,
        nb::arg("window_right") = -1,
        nb::arg("block_size")   = 16,
        nb::arg("stream")       = nb::none(),
        "Paged STEEL forward attention (kernel-level paged KV, Track FD).\n"
        "\n"
        "Avoids a gather+attend round-trip by reading K/V directly from the paged\n"
        "pool inside the Metal kernel via block_table lookups.\n"
        "\n"
        "q:           [B, H, N, D]               f16 or bf16\n"
        "k_pool:      [num_blocks, block_size, H_kv, D]\n"
        "v_pool:      [num_blocks, block_size, H_kv, D]\n"
        "block_table: [B, max_blocks]             int32\n"
        "seq_lens:    [B]                         int32 (effective KV length per batch)\n"
        "\n"
        "Returns (O [B,H,N,D], L [B,H,N] logsumexp in log2 domain).\n"
        "GQA: H_q / H_kv must be integer. window_left/right=-1 disables sliding window.\n"
        "Only f16/bf16 supported.");

  // --- SageAttention forward (Track KB, CP2): fp16 Q + int8 K + fp16 V ---
  // CP2: Q is now fp16 (no external Q quantize dispatch). K stays int8.
  m.def("mfa_sage_forward",
        [](mlx::core::array q,
           mlx::core::array k_int8,
           mlx::core::array v,
           mlx::core::array k_scale,
           float scale,
           bool  causal,
           int   window_left,
           int   window_right,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> std::pair<mlx::core::array, mlx::core::array> {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_sage_forward(
              q, k_int8, v, k_scale, scale, causal,
              window_left, window_right, s);
        },
        nb::arg("q"),
        nb::arg("k_int8"),
        nb::arg("v"),
        nb::arg("k_scale"),
        nb::arg("scale"),
        nb::arg("causal")       = false,
        nb::arg("window_left")  = -1,
        nb::arg("window_right") = -1,
        nb::arg("stream")       = nb::none(),
        "SageAttention forward pass: fp16 Q + int8 K + fp16 V → fp16 O.\n"
        "\n"
        "CP2: Q is passed as fp16/bf16 directly — no external Q quantize dispatch.\n"
        "K is int8 (quantized by QuantizedKVCache at append time). K bandwidth\n"
        "reduction (2×) still applies. GEMM always runs in fp16.\n"
        "\n"
        "q:            [B, H, N, D]    fp16 or bf16 queries\n"
        "k_int8:       [B, H_kv, S, D] int8 quantized keys\n"
        "v:            [B, H_kv, S, D] fp16 or bf16 values (unquantized)\n"
        "k_scale:      [B, H_kv, NK]   float32 per-tile K dequantization scales\n"
        "window_left:  left window radius in tokens (-1 = disabled).\n"
        "window_right: right window radius in tokens (-1 = disabled).\n"
        "\n"
        "Returns (O [B,H,N,D] fp16/bf16, L [B,H,N] logsumexp in log2 domain).\n"
        "GQA: H_q / H_kv must be an integer. Only f16/bf16 V supported.");

  // --- Fused per-block INT8 quantization (Phase 4-A.1) ---
  m.def("mfa_quantize_per_block",
        [](mlx::core::array x,
           int block_size,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> std::pair<mlx::core::array, mlx::core::array> {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_quantize_per_block(x, block_size, s);
        },
        nb::arg("x"),
        nb::arg("block_size"),
        nb::arg("stream") = nb::none(),
        "Fused per-block INT8 quantization (Phase 4-A.1).\n"
        "\n"
        "Replaces the Python-side quantize_per_block() with a single Metal dispatch.\n"
        "One threadgroup per (b, h, block_idx) reduces absmax then quantizes elements.\n"
        "\n"
        "x:          [B, H, N, D]  fp16 or bf16\n"
        "block_size: positive power of 2\n"
        "\n"
        "Returns (x_int8 [B,H,N,D] int8, scale [B,H,N_blocks,1] float32).");

  // --- Fused smooth_k + quantize_per_block (Phase 1.1) ---
  m.def("mfa_smooth_quantize_k",
        [](mlx::core::array k,
           int block_size,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> std::tuple<mlx::core::array, mlx::core::array, mlx::core::array> {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_smooth_quantize_k(k, block_size, s);
        },
        nb::arg("k"),
        nb::arg("block_size"),
        nb::arg("stream") = nb::none(),
        "Fused smooth_k + quantize_per_block for SageAttention K preprocessing (Phase 1.1).\n"
        "\n"
        "Replaces smooth_k() + quantize_per_block(k_smooth) with one C++ primitive.\n"
        "Eliminates intermediate K_smooth tensor; 2 Metal dispatches instead of 3.\n"
        "\n"
        "k:          [B, H, S, D]  fp16 or bf16\n"
        "block_size: positive power of 2 (should be BK from sage_block_sizes())\n"
        "\n"
        "Returns (k_int8 [B,H,S,D] int8, k_scale [B,H,N_blocks,1] f32, k_mean [B,H,1,D] f32).");

  // --- Paged KV scatter write (Phase 4-C.1+E.2) ---
  m.def("mfa_scatter_kv",
        [](mlx::core::array pool,
           mlx::core::array tokens,
           mlx::core::array blk_ids,
           mlx::core::array blk_offs,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> mlx::core::array {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_scatter_kv(pool, tokens, blk_ids, blk_offs, s);
        },
        nb::arg("pool"),
        nb::arg("tokens"),
        nb::arg("blk_ids"),
        nb::arg("blk_offs"),
        nb::arg("stream") = nb::none(),
        "Scatter-write tokens into paged KV pool (Phase 4-C.1+E.2).\n"
        "\n"
        "Replaces the Python pool rebuild loop with a single Metal copy+scatter pass.\n"
        "Each pool element is either copied from pool_in or overwritten by a scattered\n"
        "token — whichever applies.  Optimized for small N_write (decode: 1-4 tokens).\n"
        "\n"
        "pool:     [num_blocks, block_size, H_kv, D]  fp16 or bf16\n"
        "tokens:   [N_write, H_kv, D]                 same dtype\n"
        "blk_ids:  [N_write]                           int32 (target physical block)\n"
        "blk_offs: [N_write]                           int32 (target slot within block)\n"
        "\n"
        "Returns pool_out [num_blocks, block_size, H_kv, D] with scattered writes applied.");

  // GNA native binding removed — flash_attention_gna() uses sparse path (Python-side)

  // ── PagedVarlenForward (fused packed Q + paged KV) ──────────────────────
  m.def("mfa_paged_varlen_forward",
      [](const mlx::core::array& q,
         const mlx::core::array& k_pool,
         const mlx::core::array& v_pool,
         const mlx::core::array& cu_seqlens_q,
         const mlx::core::array& tile_offsets,
         const mlx::core::array& block_table,
         const mlx::core::array& seq_lens_kv,
         float scale,
         bool causal,
         int block_size,
         nb::object stream) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        return mlx_mfa::mfa_paged_varlen_forward(
            q, k_pool, v_pool, cu_seqlens_q, tile_offsets,
            block_table, seq_lens_kv, scale, causal, block_size, s);
      },
      nb::arg("q"),
      nb::arg("k_pool"),
      nb::arg("v_pool"),
      nb::arg("cu_seqlens_q"),
      nb::arg("tile_offsets"),
      nb::arg("block_table"),
      nb::arg("seq_lens_kv"),
      nb::arg("scale"),
      nb::arg("causal"),
      nb::arg("block_size"),
      nb::arg("stream") = nb::none(),
      "Fused paged varlen forward: packed Q + paged KV in a single dispatch.");

  // ── PagedVarlenTQForward (TurboQuant fused K dequant) ──────────────────
  m.def("mfa_paged_varlen_tq_forward",
      [](const mlx::core::array& q,
         const mlx::core::array& k_pool_tq,
         const mlx::core::array& v_pool,
         const mlx::core::array& cu_seqlens_q,
         const mlx::core::array& tile_offsets,
         const mlx::core::array& block_table,
         const mlx::core::array& seq_lens_kv,
         const mlx::core::array& centroids,
         const mlx::core::array& k_scales,
         float scale,
         bool causal,
         int block_size,
         int tq_bits,
         bool tq_v_enabled,
         bool tq_wht_enabled,
         std::optional<mlx::core::array> v_pool_tq,
         std::optional<mlx::core::array> v_centroids,
         std::optional<mlx::core::array> v_scales,
         nb::object stream) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        return mlx_mfa::mfa_paged_varlen_tq_forward(
            q, k_pool_tq, v_pool, cu_seqlens_q, tile_offsets,
            block_table, seq_lens_kv, centroids, k_scales,
            scale, causal, block_size, tq_bits,
            tq_v_enabled, tq_wht_enabled, v_pool_tq, v_centroids, v_scales, s);
      },
      nb::arg("q"),
      nb::arg("k_pool_tq"),
      nb::arg("v_pool"),
      nb::arg("cu_seqlens_q"),
      nb::arg("tile_offsets"),
      nb::arg("block_table"),
      nb::arg("seq_lens_kv"),
      nb::arg("centroids"),
      nb::arg("k_scales"),
      nb::arg("scale"),
      nb::arg("causal"),
      nb::arg("block_size"),
      nb::arg("tq_bits"),
      nb::arg("tq_v_enabled") = false,
      nb::arg("tq_wht_enabled") = false,
      nb::arg("v_pool_tq") = nb::none(),
      nb::arg("v_centroids") = nb::none(),
      nb::arg("v_scales") = nb::none(),
      nb::arg("stream") = nb::none(),
      "TurboQuant fused paged varlen forward: packed uint8 K + centroid dequant, optional V-TQ + WHT fusion.");

  m.def("_invalidate_env_config", []() {
      mlx_mfa::MFAEnvConfig::invalidate();
  }, "Re-read all cached MFA_* env vars. Call after os.environ changes.");

  // ====================================================================
  // Sprint D — Conv3D NAX C++ Primitive entry point.
  // Routes mlx_mfa.conv_nax.conv3d_nax_forward through C++ instead of
  // the Phase 1.x Python orchestrator (saves ~50-100 µs Python dispatch
  // overhead per call). The Metal kernels are identical (frozen from
  // Sprint C); only the orchestration moved from Python to C++.
  // ====================================================================
  m.def("conv3d_nax_forward",
      [](const mlx::core::array& x,
         const mlx::core::array& w,
         std::array<int, 3> stride,
         std::array<int, 6> padding,
         std::array<int, 3> dilation,
         int chunk_M) {
        mlx_mfa::ConvPad pad{
            padding[0], padding[1],
            padding[2], padding[3],
            padding[4], padding[5]};
        return mlx_mfa::conv3d_nax_forward(x, w, stride, pad, dilation, chunk_M);
      },
      nb::arg("x"), nb::arg("w"),
      nb::arg("stride") = std::array<int, 3>{1, 1, 1},
      nb::arg("padding") = std::array<int, 6>{0, 0, 0, 0, 0, 0},
      nb::arg("dilation") = std::array<int, 3>{1, 1, 1},
      nb::arg("chunk_M") = 0,
      "Conv3D NAX forward via MPP matmul2d + im2col chunking. "
      "x: (B,T,H,W,C_in) f16/bf16. w: (C_out,K_T,K_H,K_W,C_in). "
      "padding: 6-tuple (T_left,T_right,H_left,H_right,W_left,W_right). "
      "chunk_M: 0 = auto from int32-byte-budget heuristic.");

  // ====================================================================
  // Sprint B Phase 1.1 — Sparse Attention NAX free-function entry point.
  // Block-skip dispatch via per-Q-tile threadgroups + per-thread Q-row
  // FA-2. Phase 1.3 swaps inner GEMMs to mpp::tensor_ops::matmul2d.
  // ====================================================================
  m.def("sparse_attention_forward",
      [](const mlx::core::array& Q,
         const mlx::core::array& K,
         const mlx::core::array& V,
         const mlx::core::array& block_mask,
         int block_tile,
         bool causal,
         float scale,
         const std::string& kernel_version) {
        return mlx_mfa::sparse_attention_forward(
            Q, K, V, block_mask, block_tile, causal, scale, kernel_version);
      },
      nb::arg("Q"), nb::arg("K"), nb::arg("V"),
      nb::arg("block_mask"),
      nb::arg("block_tile") = 32,
      nb::arg("causal") = false,
      nb::arg("scale") = 0.0f,
      nb::arg("kernel_version") = std::string(""),
      "Sprint B block-sparse attention forward (NAX). "
      "Q/K/V: (B, H, L, D) f16. block_mask: (NQ, NK) bool. "
      "Phase 1.1: D in {64, 128}, BT in {16, 32}, mask 2-D, causal=false. "
      "v2.36.1: kernel_version param overrides MFA_LCSA_KERNEL_VERSION env "
      "(thread-safe alternative for Python-side shape-aware decide_auto_version).");

  // v2.50 Prompt 5c Section A.1 — sparse forward returning (O, L).
  m.def("sparse_attention_forward_with_lse",
      [](const mlx::core::array& Q,
         const mlx::core::array& K,
         const mlx::core::array& V,
         const mlx::core::array& block_mask,
         int block_tile,
         bool causal,
         float scale) {
        auto [O, L] = mlx_mfa::sparse_attention_forward_with_lse(
            Q, K, V, block_mask, block_tile, causal, scale);
        return nb::make_tuple(O, L);
      },
      nb::arg("Q"), nb::arg("K"), nb::arg("V"),
      nb::arg("block_mask"),
      nb::arg("block_tile") = 32,
      nb::arg("causal") = false,
      nb::arg("scale") = 0.0f,
      "Block-sparse attention forward returning (O, L).  L is per-row "
      "natural-log LSE over only the active blocks (sparse-LSE), required "
      "by V34 backward sparse kernels for LSE consistency.  All-False rows "
      "write L = -INFINITY (sentinel).  V1 kernel only at PoC stage "
      "(v2.50 Prompt 5c Section A.1).");

  // _ext.__version__ removed in v2.33.1 — single SoT in mlx_mfa.__version__
  // (was hardcoded "2.22.0", 11 versions stale). See release-flow-validation-report.md §C.3.
}
