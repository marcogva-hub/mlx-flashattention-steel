/// mlx-mfa nanobind bindings.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <mlx/mlx.h>

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

  m.def("shader_cache_size", []() {
    return mlx_mfa::ShaderCache::get().size();
  }, "Number of cached Metal compute pipelines.");

  m.def("shader_cache_clear", []() {
    mlx_mfa::ShaderCache::get().clear();
  }, "Clear the Metal pipeline cache.");

  m.attr("__version__") = "0.1.0";
}
