/// MFAConv3DForward — implementation.
///
/// Source generators (port of mlx_mfa/conv_nax.py f-strings):
///   - matmul2d_source(M, K, N): MPP matmul2d wrapper kernel
///   - im2col3d_source(...): channels-last im2col with per-chunk m_offset
///
/// Dispatch: mlx::core::fast::metal_kernel + chunk loop with per-chunk
/// eval (Phase 1.3 D24 pattern preserved — bounds peak GPU memory).
///
/// Tile config (32, 32, 32, sg=1): matches V6 NAX BLOCK_DIMENSIONS_* exactly
/// (Phase 1.1 D14 reference-pattern priority lesson).

#include "mfa_conv_nax.hpp"

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <mlx/utils.h>
#include <mlx/ops.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlx_mfa {

namespace {

// Validated tile config (Phase 1.1 D13: matches V6 NAX BQ=BK=BD=32, sg=1).
constexpr int M_TILE = 32;
constexpr int N_TILE = 32;
constexpr int K_TILE = 32;
constexpr int EXEC_SIMDGROUPS = 1;
constexpr int TG_THREADS = 32 * EXEC_SIMDGROUPS;

// Phase 1.2 D7 root-cause: MPP matmul2d uses int32 byte addresses.
// Each chunk's im2col buffer must stay below 2^31 × 0.875.
constexpr int64_t INT32_BYTE_BUDGET = 1LL << 31;
constexpr double SAFETY_HEADROOM = 0.875;

const std::string MATMUL_HEADER = R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;
)";

const std::string IM2COL_HEADER = R"(
#include <metal_stdlib>
using namespace metal;
)";

int dtype_bytes_for(mlx::core::Dtype dtype) {
  if (dtype == mlx::core::float16 || dtype == mlx::core::bfloat16) return 2;
  if (dtype == mlx::core::float32) return 4;
  throw std::runtime_error("conv_nax: unsupported dtype");
}

std::string matmul2d_source(int M, int K, int N) {
  // Conv3D matmul: C(M,N) = A(M,K) @ B(N,K)^T via rightT=true.
  // A is im2col buffer (M, K); B is flattened weight (N, K); both
  // row-major. matmul2d's rightT=true transposes B internally.
  // (Phase 1.1 D14 rightT bug history.)
  std::ostringstream os;
  os << "    constexpr uint M_FULL = " << M << ";\n"
     << "    constexpr uint K_FULL = " << K << ";\n"
     << "    constexpr uint N_FULL = " << N << ";\n\n"
     << "    auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(\n"
     << "        (device half*)A, dextents<int32_t, 2>(K_FULL, M_FULL));\n"
     << "    auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(\n"
     << "        (device half*)B, dextents<int32_t, 2>(K_FULL, N_FULL));\n\n"
     << "    const uint m_origin = threadgroup_position_in_grid.y * " << M_TILE << ";\n"
     << "    const uint n_origin = threadgroup_position_in_grid.x * " << N_TILE << ";\n\n"
     << "    constexpr auto desc = matmul2d_descriptor(\n"
     << "        " << M_TILE << ", " << N_TILE << ", " << K_TILE << ",\n"
     << "        false, true, true,\n"
     << "        matmul2d_descriptor::mode::multiply_accumulate);\n"
     << "    matmul2d<desc, execution_simdgroups<" << EXEC_SIMDGROUPS << ">> op;\n\n"
     << "    auto mA_init = tA.slice<" << K_TILE << ", " << M_TILE << ">(0, m_origin);\n"
     << "    auto mB_init = tB.slice<" << K_TILE << ", " << N_TILE << ">(0, n_origin);\n\n"
     << "    auto cC = op.get_destination_cooperative_tensor<\n"
     << "        decltype(mA_init), decltype(mB_init), float>();\n"
     << "    #pragma clang loop unroll(full)\n"
     << "    for (ushort k = 0; k < cC.get_capacity(); ++k) {\n"
     << "        if (cC.is_valid_element(k)) cC[k] = 0.0f;\n"
     << "    }\n\n"
     << "    for (uint k_start = 0; k_start < K_FULL; k_start += " << K_TILE << ") {\n"
     << "        auto mA_k = tA.slice<" << K_TILE << ", " << M_TILE << ">(k_start, m_origin);\n"
     << "        auto mB_k = tB.slice<" << K_TILE << ", " << N_TILE << ">(k_start, n_origin);\n"
     << "        op.run(mA_k, mB_k, cC);\n"
     << "    }\n\n"
     << "    #pragma clang loop unroll(full)\n"
     << "    for (ushort k = 0; k < cC.get_capacity(); ++k) {\n"
     << "        if (cC.is_valid_element(k)) {\n"
     << "            auto idx = cC.get_multidimensional_index(k);\n"
     << "            uint m_global = m_origin + idx[1];\n"
     << "            uint n_global = n_origin + idx[0];\n"
     << "            if (m_global < M_FULL && n_global < N_FULL) {\n"
     << "                C[m_global * N_FULL + n_global] = (half)cC[k];\n"
     << "            }\n"
     << "        }\n"
     << "    }\n";
  return os.str();
}

std::string im2col3d_source(int B, int T, int H, int W, int C_in,
                            int T_out, int H_out, int W_out,
                            int K_T, int K_H, int K_W,
                            int sT, int sH, int sW,
                            int pT_l, int pH_l, int pW_l,
                            int dT, int dH, int dW,
                            int m_offset, int m_chunk) {
  std::ostringstream os;
  os << "    constexpr uint cB    = " << B << ";\n"
     << "    constexpr uint cT    = " << T << ";\n"
     << "    constexpr uint cH    = " << H << ";\n"
     << "    constexpr uint cW    = " << W << ";\n"
     << "    constexpr uint cCin  = " << C_in << ";\n"
     << "    constexpr uint cTout = " << T_out << ";\n"
     << "    constexpr uint cHout = " << H_out << ";\n"
     << "    constexpr uint cWout = " << W_out << ";\n"
     << "    constexpr uint cKT   = " << K_T << ";\n"
     << "    constexpr uint cKH   = " << K_H << ";\n"
     << "    constexpr uint cKW   = " << K_W << ";\n"
     << "    constexpr int  csT   = " << sT << ";\n"
     << "    constexpr int  csH   = " << sH << ";\n"
     << "    constexpr int  csW   = " << sW << ";\n"
     << "    constexpr int  cpTl  = " << pT_l << ";\n"
     << "    constexpr int  cpHl  = " << pH_l << ";\n"
     << "    constexpr int  cpWl  = " << pW_l << ";\n"
     << "    constexpr int  cdT   = " << dT << ";\n"
     << "    constexpr int  cdH   = " << dH << ";\n"
     << "    constexpr int  cdW   = " << dW << ";\n"
     << "    constexpr uint cMoff = " << m_offset << ";\n"
     << "    constexpr uint cMchk = " << m_chunk << ";\n"
     << "    constexpr uint cKvol = cKT * cKH * cKW;\n"
     << "    constexpr uint cKfull = cKvol * cCin;\n\n"
     << "    uint tid = thread_position_in_grid.x;\n"
     << "    if (tid >= cMchk * cKfull) return;\n"
     << "    uint m_local = tid / cKfull;\n"
     << "    uint k = tid - m_local * cKfull;\n"
     << "    uint m_global = m_local + cMoff;\n\n"
     << "    uint rem_m = m_global;\n"
     << "    uint w_out = rem_m % cWout; rem_m /= cWout;\n"
     << "    uint h_out = rem_m % cHout; rem_m /= cHout;\n"
     << "    uint t_out = rem_m % cTout; rem_m /= cTout;\n"
     << "    uint b     = rem_m;\n\n"
     << "    uint rem_k = k;\n"
     << "    uint c_in = rem_k % cCin; rem_k /= cCin;\n"
     << "    uint k_w  = rem_k % cKW;  rem_k /= cKW;\n"
     << "    uint k_h  = rem_k % cKH;  rem_k /= cKH;\n"
     << "    uint k_t  = rem_k;\n\n"
     << "    int t_in = (int)t_out * csT + (int)k_t * cdT - cpTl;\n"
     << "    int h_in = (int)h_out * csH + (int)k_h * cdH - cpHl;\n"
     << "    int w_in = (int)w_out * csW + (int)k_w * cdW - cpWl;\n\n"
     << "    half v = (half)0.0h;\n"
     << "    if (t_in >= 0 && t_in < (int)cT &&\n"
     << "        h_in >= 0 && h_in < (int)cH &&\n"
     << "        w_in >= 0 && w_in < (int)cW) {\n"
     << "        uint in_idx = ((b * cT + (uint)t_in) * cH + (uint)h_in) * cW * cCin\n"
     << "                     + (uint)w_in * cCin + c_in;\n"
     << "        v = X[in_idx];\n"
     << "    }\n"
     << "    Im2col[m_local * cKfull + k] = v;\n";
  return os.str();
}

// Phase 1.2 D7 + Phase 1.3 D24 logic, ported from Python.
std::vector<std::pair<int, int>> compute_chunk_layout(
    int M_total, int K, int dtype_bytes) {
  int64_t max_chunk_bytes = (int64_t)(INT32_BYTE_BUDGET * SAFETY_HEADROOM);
  int64_t max_chunk_M_l = max_chunk_bytes / ((int64_t)K * dtype_bytes);
  int max_chunk_M = static_cast<int>((max_chunk_M_l / M_TILE) * M_TILE);
  if (max_chunk_M < M_TILE) {
    throw std::runtime_error(
        "conv_nax: K too large for chunking; even one M_TILE row "
        "exceeds int32 byte budget");
  }
  if (M_total <= max_chunk_M) {
    return {{0, M_total}};
  }
  int n_chunks = (M_total + max_chunk_M - 1) / max_chunk_M;
  int base = (M_total / n_chunks / M_TILE) * M_TILE;
  if (base == 0) base = M_TILE;
  std::vector<std::pair<int, int>> chunks;
  int remaining = M_total;
  int offset = 0;
  for (int i = 0; i < n_chunks; ++i) {
    int chunk_M;
    if (i == n_chunks - 1) {
      chunk_M = remaining;
    } else {
      chunk_M = base;
      remaining -= chunk_M;
    }
    chunks.emplace_back(offset, chunk_M);
    offset += chunk_M;
  }
  return chunks;
}

// Defensive int32 byte-offset chunking invariant (Phase 1.2 D7).
void enforce_int32_byte_offset_invariant(int chunk_M, int K, int dtype_bytes) {
  int64_t im2col_bytes = (int64_t)chunk_M * K * dtype_bytes;
  int64_t safety_limit =
      (int64_t)(INT32_BYTE_BUDGET * SAFETY_HEADROOM);
  if (im2col_bytes >= safety_limit) {
    std::ostringstream msg;
    msg << "conv_nax: chunk_M(" << chunk_M << ") × K(" << K
        << ") × dtype_bytes(" << dtype_bytes << ") = " << im2col_bytes
        << " >= int32-budget safety limit (" << safety_limit
        << "). MPP matmul2d would NaN at row "
        << (INT32_BYTE_BUDGET / ((int64_t)K * dtype_bytes))
        << ".";
    throw std::runtime_error(msg.str());
  }
}

// Dispatch the pointwise 1×1×1 fast path. Skip im2col; reshape input
// directly to (M, C_in) (channels-last makes this metadata-only),
// dispatch matmul2d.
mlx::core::array dispatch_pointwise_fast_path(
    const mlx::core::array& x, const mlx::core::array& w,
    int B, int T, int H, int W, int C_in, int C_out) {
  int M = B * T * H * W;
  // Reshape input (B, T, H, W, C_in) -> (M, C_in) (no copy under
  // channels-last invariant).
  auto x_flat = mlx::core::reshape(x, {M, C_in});
  auto w_flat = mlx::core::reshape(w, {C_out, C_in});

  // Chunk plan on the smaller K = C_in.
  int dtype_bytes = dtype_bytes_for(x.dtype());
  auto chunks = compute_chunk_layout(M, C_in, dtype_bytes);

  std::vector<mlx::core::array> chunk_outputs;
  chunk_outputs.reserve(chunks.size());
  bool multi_chunk = chunks.size() > 1;
  for (const auto& [m_offset, m_chunk] : chunks) {
    enforce_int32_byte_offset_invariant(m_chunk, C_in, dtype_bytes);
    mlx::core::array x_chunk =
        multi_chunk
            ? mlx::core::slice(x_flat, {m_offset, 0},
                                {m_offset + m_chunk, C_in}, {1, 1})
            : x_flat;
    std::string name = "conv3d_1x1x1_mm_" + std::to_string(m_chunk) + "_" +
                       std::to_string(C_in) + "_" + std::to_string(C_out);
    auto kernel = mlx::core::fast::metal_kernel(
        name, {"A", "B"}, {"C"},
        matmul2d_source(m_chunk, C_in, C_out), MATMUL_HEADER,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    int n_tg_x = (C_out + N_TILE - 1) / N_TILE;
    int n_tg_y = (m_chunk + M_TILE - 1) / M_TILE;
    auto outs = kernel(
        {x_chunk, w_flat},
        {mlx::core::Shape{m_chunk, C_out}},
        {x.dtype()},
        {n_tg_x * TG_THREADS, n_tg_y, 1},
        {TG_THREADS, 1, 1},
        {},
        std::nullopt,
        false,
        mlx::core::default_stream(mlx::core::Device::gpu));
    auto chunk_flat = outs[0];
    if (multi_chunk) {
      mlx::core::eval(chunk_flat);
    }
    chunk_outputs.push_back(chunk_flat);
  }
  mlx::core::array flat = (chunk_outputs.size() == 1)
      ? chunk_outputs[0]
      : mlx::core::concatenate(chunk_outputs, 0);
  return mlx::core::reshape(flat, {B, T, H, W, C_out});
}

}  // namespace

mlx::core::array conv3d_nax_forward(
    const mlx::core::array& x,
    const mlx::core::array& w,
    const std::array<int, 3>& stride,
    const ConvPad& pad,
    const std::array<int, 3>& dilation,
    int chunk_M_override) {
  // Sanity asserts (8 categories per Phase 1.1).
  if (x.ndim() != 5) {
    throw std::runtime_error("conv_nax: input must be 5D (B,T,H,W,C_in)");
  }
  if (w.ndim() != 5) {
    throw std::runtime_error("conv_nax: weight must be 5D (C_out,K_T,K_H,K_W,C_in)");
  }
  if (x.dtype() != w.dtype()) {
    throw std::runtime_error("conv_nax: x.dtype != w.dtype");
  }
  if (x.dtype() != mlx::core::float16 && x.dtype() != mlx::core::bfloat16) {
    throw std::runtime_error("conv_nax: dtype must be f16 or bf16");
  }
  int B = static_cast<int>(x.shape(0));
  int T = static_cast<int>(x.shape(1));
  int H = static_cast<int>(x.shape(2));
  int W = static_cast<int>(x.shape(3));
  int C_in = static_cast<int>(x.shape(4));
  int C_out = static_cast<int>(w.shape(0));
  int K_T = static_cast<int>(w.shape(1));
  int K_H = static_cast<int>(w.shape(2));
  int K_W = static_cast<int>(w.shape(3));
  if (static_cast<int>(w.shape(4)) != C_in) {
    throw std::runtime_error("conv_nax: C_in mismatch between x and w");
  }
  int sT = stride[0], sH = stride[1], sW = stride[2];
  int dT = dilation[0], dH = dilation[1], dW = dilation[2];
  if (sT < 1 || sH < 1 || sW < 1) {
    throw std::runtime_error("conv_nax: stride >= 1 required");
  }
  if (dT < 1 || dH < 1 || dW < 1) {
    throw std::runtime_error("conv_nax: dilation >= 1 required");
  }
  if (pad.T_left < 0 || pad.T_right < 0 || pad.H_left < 0 ||
      pad.H_right < 0 || pad.W_left < 0 || pad.W_right < 0) {
    throw std::runtime_error("conv_nax: padding values must be >= 0");
  }
  int eff_T = T + pad.T_left + pad.T_right - dT * (K_T - 1) - 1;
  int eff_H = H + pad.H_left + pad.H_right - dH * (K_H - 1) - 1;
  int eff_W = W + pad.W_left + pad.W_right - dW * (K_W - 1) - 1;
  if (eff_T < 0 || eff_H < 0 || eff_W < 0) {
    throw std::runtime_error("conv_nax: input too small for kernel after padding");
  }
  int T_out = eff_T / sT + 1;
  int H_out = eff_H / sH + 1;
  int W_out = eff_W / sW + 1;
  int M = B * T_out * H_out * W_out;
  int K = K_T * K_H * K_W * C_in;
  int N = C_out;

  // 1×1×1 fast path detection (Phase 1.4 D11).
  bool is_pointwise =
      (K_T == 1 && K_H == 1 && K_W == 1 &&
       pad.T_left == 0 && pad.T_right == 0 &&
       pad.H_left == 0 && pad.H_right == 0 &&
       pad.W_left == 0 && pad.W_right == 0 &&
       sT == 1 && sH == 1 && sW == 1);
  if (is_pointwise) {
    return dispatch_pointwise_fast_path(x, w, B, T, H, W, C_in, C_out);
  }

  // Flatten weight (C_out, K_T*K_H*K_W*C_in) row-major.
  auto w_flat = mlx::core::reshape(w, {C_out, K});

  // Plan chunks. chunk_M_override > 0 means user-specified.
  int dtype_bytes = dtype_bytes_for(x.dtype());
  std::vector<std::pair<int, int>> chunks;
  if (chunk_M_override > 0) {
    chunks.clear();
    int offset = 0;
    while (offset < M) {
      int this_chunk = std::min(chunk_M_override, M - offset);
      chunks.emplace_back(offset, this_chunk);
      offset += this_chunk;
    }
  } else {
    chunks = compute_chunk_layout(M, K, dtype_bytes);
  }

  std::vector<mlx::core::array> chunk_outputs;
  chunk_outputs.reserve(chunks.size());
  bool multi_chunk = chunks.size() > 1;
  auto stream = mlx::core::default_stream(mlx::core::Device::gpu);

  for (const auto& [m_offset, m_chunk] : chunks) {
    enforce_int32_byte_offset_invariant(m_chunk, K, dtype_bytes);

    // Build im2col kernel for this chunk.
    std::string im2col_name =
        "conv3d_im2col_" + std::to_string(B) + "_" +
        std::to_string(T) + "_" + std::to_string(H) + "_" +
        std::to_string(W) + "_" + std::to_string(C_in) + "_" +
        std::to_string(K_T) + std::to_string(K_H) + std::to_string(K_W) +
        "_s" + std::to_string(sT) + std::to_string(sH) +
        std::to_string(sW) + "_pl" + std::to_string(pad.T_left) +
        std::to_string(pad.H_left) + std::to_string(pad.W_left) +
        "_pr" + std::to_string(pad.T_right) + std::to_string(pad.H_right) +
        std::to_string(pad.W_right) + "_d" + std::to_string(dT) +
        std::to_string(dH) + std::to_string(dW) + "_off" +
        std::to_string(m_offset) + "_chk" + std::to_string(m_chunk);
    auto im2col_kernel = mlx::core::fast::metal_kernel(
        im2col_name, {"X"}, {"Im2col"},
        im2col3d_source(B, T, H, W, C_in, T_out, H_out, W_out,
                        K_T, K_H, K_W, sT, sH, sW,
                        pad.T_left, pad.H_left, pad.W_left,
                        dT, dH, dW, m_offset, m_chunk),
        IM2COL_HEADER, /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false);

    int64_t chunk_elems = (int64_t)m_chunk * K;
    constexpr int THREADS_PER_TG_IM2COL = 256;
    int64_t grid_x_im2col =
        (chunk_elems + THREADS_PER_TG_IM2COL - 1) / THREADS_PER_TG_IM2COL;
    auto im2col_outs = im2col_kernel(
        {x},
        {mlx::core::Shape{m_chunk, K}},
        {x.dtype()},
        {static_cast<int>(grid_x_im2col * THREADS_PER_TG_IM2COL), 1, 1},
        {THREADS_PER_TG_IM2COL, 1, 1},
        {},
        std::nullopt,
        false,
        stream);
    auto im2col_buf = im2col_outs[0];

    // Build matmul kernel for this chunk.
    std::string mm_name = "conv3d_matmul2d_" +
        std::to_string(m_chunk) + "_" + std::to_string(K) + "_" +
        std::to_string(N);
    auto mm_kernel = mlx::core::fast::metal_kernel(
        mm_name, {"A", "B"}, {"C"},
        matmul2d_source(m_chunk, K, N), MATMUL_HEADER,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    int n_tg_x = (N + N_TILE - 1) / N_TILE;
    int n_tg_y = (m_chunk + M_TILE - 1) / M_TILE;
    auto mm_outs = mm_kernel(
        {im2col_buf, w_flat},
        {mlx::core::Shape{m_chunk, N}},
        {x.dtype()},
        {n_tg_x * TG_THREADS, n_tg_y, 1},
        {TG_THREADS, 1, 1},
        {},
        std::nullopt,
        false,
        stream);
    auto chunk_flat = mm_outs[0];
    if (multi_chunk) {
      // Phase 1.3 D24: bound peak GPU memory by forcing per-chunk eval.
      mlx::core::eval(chunk_flat);
    }
    chunk_outputs.push_back(chunk_flat);
  }

  mlx::core::array flat = (chunk_outputs.size() == 1)
      ? chunk_outputs[0]
      : mlx::core::concatenate(chunk_outputs, 0);
  return mlx::core::reshape(flat, {B, T_out, H_out, W_out, C_out});
}

}  // namespace mlx_mfa
