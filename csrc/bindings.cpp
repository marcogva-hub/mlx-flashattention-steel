/// mlx-mfa nanobind bindings.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <mlx/mlx.h>
#include <mlx/backend/metal/device.h>

#include "mfa_attention.hpp"
#include "shader_cache.hpp"

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
      nb::arg("stream") = nb::none(),
      "Flash Attention forward (Metal). "
      "q/k/v: [B, H, N, D], float16/bfloat16/float32.");

  // Debug: returns (O, L) so L (logsumexp) can be inspected from Python.
  m.def("mfa_forward_with_lse",
      [](const mlx::core::array& q,
         const mlx::core::array& k,
         const mlx::core::array& v,
         float scale, bool causal) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        mlx_mfa::MFAttention::Params params{q.shape(3), scale, causal};
        mlx::core::Shape lse_shape = {q.shape(0), q.shape(1), q.shape(2)};
        auto outs = mlx::core::array::make_arrays(
            {q.shape(), lse_shape},
            {q.dtype(), mlx::core::float32},
            std::make_shared<mlx_mfa::MFAttention>(s, params),
            {q, k, v});
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
        mlx_mfa::MFAttention::Params params{q.shape(3), scale, causal};
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
        mlx_mfa::MFAttention::Params params{q.shape(3), scale, causal};
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
  m.def("get_device_info", []() -> nb::dict {
    auto s = mlx::core::default_stream(mlx::core::Device::gpu);
    auto& d = mlx::core::metal::device(s.device);
    int gen = d.get_architecture_gen();
    // MTL::Device::name() returns an NS::String; utf8String() gives a C string.
    auto* mtl_dev = d.mtl_device();
    std::string dev_name = mtl_dev
        ? std::string(mtl_dev->name()->utf8String())
        : "unknown";
    nb::dict info;
    info["gpu_family_gen"] = gen;
    info["is_m3_plus"]     = (gen >= 15);
    info["device_name"]    = dev_name;
    return info;
  }, "Return Metal GPU hardware info: silicon generation, M3+ flag, device name.");

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

  m.attr("__version__") = "0.2.0";
}
