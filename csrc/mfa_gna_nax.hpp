#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlx_mfa {

mlx::core::array mfa_gna_nax_forward(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    int dim0,
    int dim1,
    int dim2,
    int window0,
    int window1,
    int window2,
    int stride0,
    int stride1,
    int stride2,
    float scale,
    mlx::core::StreamOrDevice s = {});

} // namespace mlx_mfa
