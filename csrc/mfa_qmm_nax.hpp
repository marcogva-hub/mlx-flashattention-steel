/// V6 NAX quantized matmul expert entry point.
///
/// Computes x @ dequantize(w_q).T for MLX's transpose=True quantized-matmul
/// layout without materializing the dequantized weight tensor in device memory.

#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlx_mfa {

mlx::core::array v6_nax_quantized_matmul(
    const mlx::core::array& x,
    const mlx::core::array& w_q,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    int group_size,
    int bits,
    mlx::core::StreamOrDevice s = {});

}  // namespace mlx_mfa
