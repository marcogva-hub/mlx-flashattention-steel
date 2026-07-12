/// Expert-only V6 NAX dense Linear with optional fused GELU epilogue.

#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlx_mfa {

mlx::core::array v6_nax_linear(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    const mlx::core::array& bias,
    bool gelu,
    mlx::core::StreamOrDevice s = {});

}  // namespace mlx_mfa
