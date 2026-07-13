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
#include "mfa_bool_env.hpp"

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <mlx/utils.h>
#include <mlx/ops.h>

#include <cstdint>
#include <cstdlib>
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

// ── Sprint III-6: matmul2d partial-K-tile fix ─────────────────────────
// matmul2d_source tiles the K (contraction) axis in K_TILE-wide steps and
// its K-loop reads the FINAL tile past K_FULL when K % K_TILE != 0 — the
// `tA.slice<K_TILE,…>(k_start,…)` slice exceeds the declared tensor extent
// and the cooperative load reads adjacent/OOB memory -> garbage
// accumulation (C_in=16 -> MAE/RMS 0.11; C_in=31 -> NaN; see
// docs/v50/campaign-2026-06/phase3/conv-small-channel-fix.md).  Since
// K = C_in*taps and gcd(taps,32) makes K % 32 == 0 only at C_in % 32 == 0,
// every K-unaligned shape corrupts.
//
// Fix: zero-pad BOTH matmul operands' K (last) axis up to a multiple of
// K_TILE.  Zero contraction terms contribute nothing, so C is EXACT, and
// the K-loop only ever reads in-bounds — every shape becomes the
// already-correct K % K_TILE == 0 case.  K is compile-time in the JIT
// source, so the padded K is baked into the kernel name (distinct cache
// key per Sprint A discipline).  Cost: one pad copy + a slightly wider
// matmul (at most +K_TILE-1 in K) when K is unaligned; nil when aligned.
inline int round_up_k_tile(int K) {
  return ((K + K_TILE - 1) / K_TILE) * K_TILE;
}

mlx::core::array pad_contraction_k(const mlx::core::array& a, int K, int k_pad,
                                   mlx::core::StreamOrDevice s) {
  if (k_pad == K) return a;  // already K_TILE-aligned — no-op
  int last = static_cast<int>(a.ndim()) - 1;
  return mlx::core::pad(
      a, /*axes=*/{last}, /*low_pad_size=*/{0}, /*high_pad_size=*/{k_pad - K},
      /*pad_value=*/mlx::core::array(0, a.dtype()), /*mode=*/"constant", s);
}

std::string matmul2d_source(int M, int K, int N) {
  // Conv3D matmul: C(M,N) = A(M,K) @ B(N,K)^T via rightT=true.
  // A is im2col buffer (M, K); B is flattened weight (N, K); both
  // row-major. matmul2d's rightT=true transposes B internally.
  // (Phase 1.1 D14 rightT bug history.)
  //
  // III-6 Rule-8 contract: the K-loop steps K_TILE at a time over
  // [0, K) and does NOT mask a partial final tile, so it is correct ONLY
  // when K is a multiple of K_TILE.  Callers MUST zero-pad the contraction
  // (pad_contraction_k).  Refuse to generate for an unaligned K — any
  // future dispatch site that forgets to pad fails LOUDLY here at JIT-gen
  // time rather than silently reading past the tensor extent (the
  // small-channel corruption this sprint fixed).
  if (K % K_TILE != 0) {
    throw std::runtime_error(
        "matmul2d_source: K=" + std::to_string(K) + " is not a multiple of "
        "K_TILE=" + std::to_string(K_TILE) + " — the contraction must be "
        "zero-padded (pad_contraction_k) before dispatch, else the K-loop "
        "reads past the tensor extent (silent corruption).");
  }
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
  // III-6: pad the contraction (K = C_in) to a K_TILE multiple — w_flat is
  // loop-invariant, pad once.
  int k_pad = round_up_k_tile(C_in);
  auto stream_pw = mlx::core::default_stream(mlx::core::Device::gpu);
  auto w_flat_pad = pad_contraction_k(w_flat, C_in, k_pad, stream_pw);
  for (const auto& [m_offset, m_chunk] : chunks) {
    enforce_int32_byte_offset_invariant(m_chunk, k_pad, dtype_bytes);
    mlx::core::array x_chunk =
        multi_chunk
            ? mlx::core::slice(x_flat, {m_offset, 0},
                                {m_offset + m_chunk, C_in}, {1, 1})
            : x_flat;
    auto x_chunk_pad = pad_contraction_k(x_chunk, C_in, k_pad, stream_pw);
    std::string name = "conv3d_1x1x1_mm_" + std::to_string(m_chunk) + "_" +
                       std::to_string(k_pad) + "_" + std::to_string(C_out);
    auto kernel = mlx::core::fast::metal_kernel(
        name, {"A", "B"}, {"C"},
        matmul2d_source(m_chunk, k_pad, C_out), MATMUL_HEADER,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    int n_tg_x = (C_out + N_TILE - 1) / N_TILE;
    int n_tg_y = (m_chunk + M_TILE - 1) / M_TILE;
    auto outs = kernel(
        {x_chunk_pad, w_flat_pad},
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

// ─────────────────────────────────────────────────────────────────────
// Sprint II-9 (campaign 2026-06): conv3d via the native MPP
// convolution2d primitive — eliminates the materialized im2col entirely
// (II-4: 62% of small-K time).  conv3d decomposes as K_T accumulated 2D
// convolutions (multiply_accumulate into a cooperative destination).
//
// Tiling semantics (resolved II-9 R.0; matches liuliu/ccv
// NAConv3DKernel.cpp production usage): descriptor dest dims = the
// PER-THREADGROUP tile, source dims = whole frame; destination handle
// SLICED to the tile origin; set_offsets(x0, y0) positions the source
// sampling window ((x, y) order).  Cooperative destination per the
// liuliu/example_matmul_metal4 finding (direct dest-tensor writes pass
// on macOS but are incorrect on M5 iPad).
//
// Eligibility (checked by the caller branch): the production 3x3x3/stride-1
// envelope plus the direct expert FlashVSR probe 4x3x3/temporal-stride-2.
// Both use spatial 3x3 stride/dilation 1 and C_in/C_out multiples of 16.
// Weights must be pre-packed to [K_T][K_H][K_W][C_in][C_out].
//
// Sprint III-1 (KD-7): dtype-parameterized — `mtype` is the MSL scalar
// ("half" or "bfloat").  The bf16 MPP variant
// (__tensorops_impl_convolution2d_op_run_cooperative_dv_bf_dv_bf_f32)
// was probed II-2R-style: implemented at runtime, rel err <= 0.9%
// (single bf16 store rounding), 99.9-100% bit-identical to mx.conv3d
// bf16 across the production forms.
std::string conv3d_mpp_source(int T, int H, int W, int C, int O,
                              int T_out, int H_out, int W_out,
                              int K_T, int sT, int TW, int TH,
                              const std::string& mtype,
                              int pT_left, int pH, int pW) {
  std::ostringstream ss;
  ss << R"(
  uint3 tgid = threadgroup_position_in_grid;
  const uint tiles_x = )" << (W_out / TW) << R"(;
  const uint tw = tgid.x % tiles_x;
  const uint th = tgid.x / tiles_x;
  const uint t  = tgid.y;
  const int x0 = (int)(tw * )" << TW << R"();
  const int y0 = (int)(th * )" << TH << R"();

  constexpr auto desc = convolution2d_descriptor(
      int4()" << O << ", " << TW << ", " << TH << R"(, 1),
      int4()" << C << ", " << W << ", " << H << R"(, 1),
      int2(3, 3),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(1, 1), int2(1, 1), 1, false,
      convolution2d_descriptor::mode::multiply_accumulate);
  convolution2d<desc, metal::execution_simdgroups<4>> op;
  op.set_offsets(int2(x0 + )" << (1 - pW) << ", y0 + " << (1 - pH) << R"());

  device )" << mtype << R"(* Dframe = Out + (ulong)t * )" << ((int64_t)H_out * W_out * O) << R"(;
  auto tD = tensor(Dframe, extents<int32_t, )"
     << O << ", " << W_out << ", " << H_out << R"(, 1>());
  auto tDs = tD.slice(0, x0, y0, 0);

  auto tA0 = tensor(X, extents<int32_t, )"
     << C << ", " << W << ", " << H << R"(, 1>());
  auto tW0 = tensor(Wp, extents<int32_t, )"
     << O << ", " << C << R"(, 3, 3>());
  // FLOAT cooperative destination: fp32 accumulation across the
  // kh/kw/C reduction AND the kt taps (the half-dest variant
  // accumulated in fp16 and failed the repo's 1e-5-rel parity bars
  // vs the fp32-accumulating legacy GEMM).  store() converts to the
  // half output tensor once at the end.
  auto cOut = op.get_destination_cooperative_tensor<
      decltype(tA0), decltype(tW0), float>();
  for (ushort i = 0; i < cOut.get_capacity(); ++i)
    if (cOut.is_valid_element(i)) cOut[i] = 0.0f;

  for (short kt = 0; kt < )" << K_T << R"(; ++kt) {
    const int tf = (int)t * )" << sT << R"( + kt - )" << pT_left << R"(;
    if (tf < 0 || tf >= )" << T << R"() continue;   // zero temporal pad
    auto tA = tensor(X + (ulong)tf * )" << ((int64_t)H * W * C) << R"(,
                     extents<int32_t, )"
     << C << ", " << W << ", " << H << R"(, 1>());
    auto tW = tensor(Wp + (ulong)kt * 9 * )" << ((int64_t)C * O) << R"(,
                     extents<int32_t, )"
     << O << ", " << C << R"(, 3, 3>());
    op.run(tA, tW, cOut);
  }
  // Elementwise store with (half) conversion: float coop dest cannot
  // store() directly into the half tensor view (no matching overload).
  // gmi index space matches the dest tile (channel, x, y, n).
  for (ushort i = 0; i < cOut.get_capacity(); ++i) {
    if (cOut.is_valid_element(i)) {
      auto idx = cOut.get_multidimensional_index(i);
      const int oo = idx[0];
      const int xx = idx[1];
      const int yy = idx[2];
      Dframe[((ulong)(y0 + yy) * )" << W_out << R"( + (ulong)(x0 + xx)) * )"
     << O << R"( + oo] = ()" << mtype << R"()cOut[i];
    }
  }
)";
  return ss.str();
}

const std::string CONV2D_MPP_HEADER = R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsConvolution2d.h>
using namespace metal;
using namespace mpp::tensor_ops;
)";

// w must be the PACKED weights [K_T,3,3,C_in,C_out] (caller transposes
// the repo layout (C_out,3,3,3,C_in) via {1,2,3,4,0}).
mlx::core::array conv3d_mpp_dispatch(
    const mlx::core::array& x, const mlx::core::array& w_packed,
    int T, int H, int W, int C_in, int C_out,
    int T_out, int H_out, int W_out, int K_T, int sT,
    int TW, int TH, int pT_left, int pH, int pW) {
  // III-1: MSL scalar follows the input dtype; the dtype is part of the
  // kernel name (cache-key discipline — Sprint A class). pT_left + T_out are
  // baked into the source/grid → part of the cache key (asym-pad correctness).
  const std::string mtype =
      (x.dtype() == mlx::core::bfloat16) ? "bfloat" : "half";
  std::string name = "conv3d_mpp_" + mtype + "_" + std::to_string(T) + "_" +
      std::to_string(H) + "x" + std::to_string(W) + "_" +
      std::to_string(C_in) + "_" + std::to_string(C_out) + "_t" +
      std::to_string(TW) + "x" + std::to_string(TH) +
      "_kT" + std::to_string(K_T) + "_sT" + std::to_string(sT) +
      "_p" + std::to_string(pT_left) + "x" + std::to_string(pH) + "x" +
      std::to_string(pW) + "_out" + std::to_string(T_out) + "x" +
      std::to_string(H_out) + "x" + std::to_string(W_out);
  auto kernel = mlx::core::fast::metal_kernel(
      name, {"X", "Wp"}, {"Out"},
      conv3d_mpp_source(T, H, W, C_in, C_out, T_out, H_out, W_out,
                        K_T, sT, TW, TH, mtype, pT_left, pH, pW),
      CONV2D_MPP_HEADER,
      /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
  int tiles = (W_out / TW) * (H_out / TH);
  auto outs = kernel(
      {x, w_packed},
      {mlx::core::Shape{1, T_out, H_out, W_out, C_out}},
      {x.dtype()},
      // grid is expressed in THREADS (MLX metal_kernel convention):
      // (tiles * 128, T_out, 1) with 128-thread threadgroups.
      {tiles * 128, T_out, 1},
      {128, 1, 1},
      {},
      std::nullopt,
      false,
      mlx::core::default_stream(mlx::core::Device::gpu));
  return outs[0];
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

  // Sprint II-9: MPP convolution2d path (fused — no materialized
  // im2col; II-4's 62%-of-time lever).  PROMOTED DEFAULT-ON for the
  // eligible envelope below after the three-axis gate (fp16-floor
  // parity across the shape grid, path-entered timing proof,
  // edge-cases preserved).  Measured on the production surface
  // (weight repack included): 2.36x at T8 64x64 C128 (the K=3456
  // headline cell), 1.83x at T8 32x32 C256, 1.14-1.38x small cells.
  // Opt-out: MFA_DISABLE_CONV3D_MPP=1 (build-phase opt-in flag
  // MFA_CONV3D_MPP=1 retained as a force-enable for diagnostics).
  {
    const bool mpp_enabled = !get_bool_env("MFA_DISABLE_CONV3D_MPP");
    // III-1 (KD-7): bf16 admitted after the II-2R-style runtime probe
    // (variant implemented; rel err <= 0.9%; bench 1.4-2.7x vs the
    // pre-lift public bf16 path — Apple mx.conv3d via hook fallback —
    // at the II-9 cells, 3 sessions, medians).
    if (mpp_enabled &&
        (x.dtype() == mlx::core::float16 ||
         x.dtype() == mlx::core::bfloat16) &&
        B == 1 &&
        ((K_T == 3 && sT == 1) || (K_T == 4 && sT == 2)) &&
        K_H == 3 && K_W == 3 && sH == 1 && sW == 1 &&
        dT == 1 && dH == 1 && dW == 1 &&
        // Per-axis "same"-style pad, symmetric WITHIN each axis. Temporal:
        // pad 1 ("same", T_out==T) OR pad 0 (causal — VAE pre-concats the
        // causal frames upstream, conv sees pad_T=0, T_out=T-2; the kt
        // time-loop's -pT_left offset + OOB-continue handle it). Spatial H/W:
        // pad 1 only (the convolution2d descriptor bakes int2(1,1)). Truly
        // asymmetric-within-axis (T_left!=T_right) or H/W!=1 → NOT MPP-eligible
        // (falls through; Rule 8 raise for bf16, legacy/SDPA for fp16).
        pad.T_left == pad.T_right && (pad.T_left == 0 || pad.T_left == 1) &&
        pad.H_left == pad.H_right && pad.W_left == pad.W_right &&
        ((K_T == 3 && pad.H_left == 1 && pad.W_left == 1) ||
         (K_T == 4 && pad.H_left == 0 && pad.W_left == 0)) &&
        // C=16 measured WRONG through the primitive (err 0.17-0.31 vs
        // legacy; C>=32 exact) — undocumented MPP constraint; gate at 32.
        C_in % 16 == 0 && C_in >= 32 &&
        C_out % 16 == 0 && C_out >= 32) {
      // Occupancy-aware tile pick: 16x16 amortizes best but needs
      // enough threadgroups to cover the GPU (32 TGs at the
      // T8/32x32/C256 cell measured 0.85x — underoccupied on 40
      // cores).  Prefer 16x16 only when it yields >= 64 TGs.
      int TW = 0, TH = 0;
      if (H_out % 16 == 0 && W_out % 16 == 0 &&
          (int64_t)(W_out / 16) * (H_out / 16) * T_out >= 64) {
        TW = 16; TH = 16;
      } else if (H_out % 8 == 0 && W_out % 8 == 0) {
        TW = 8; TH = 8;
      }
      if (TW != 0) {
        // Pack weights (C_out,K_T,3,3,C_in) -> [K_T][3][3][C_in][C_out].
        // transpose is a lazy view; ensure_row_contiguous in the kernel
        // forces the contiguous copy.
        auto w_packed = mlx::core::transpose(w, {1, 2, 3, 4, 0});
        // T_out = eff_T + 1 (host-computed for any pad): pad 1 → T; pad 0 → T-2.
        return conv3d_mpp_dispatch(
            x, w_packed, T, H, W, C_in, C_out,
            T_out, H_out, W_out, K_T, sT, TW, TH,
            pad.T_left, pad.H_left, pad.W_left);
      }
    }
  }

  // 1×1×1 fast path detection (Phase 1.4 D11).
  // Env-var escape hatch MFA_CONV_NAX_NO_FAST_PATH=1 (propagated to C++
  // in Sprint D so tests + diagnostics behave identically to Phase 1.4).
  bool is_pointwise =
      (K_T == 1 && K_H == 1 && K_W == 1 &&
       pad.T_left == 0 && pad.T_right == 0 &&
       pad.H_left == 0 && pad.H_right == 0 &&
       pad.W_left == 0 && pad.W_right == 0 &&
       sT == 1 && sH == 1 && sW == 1);
  const bool fast_path_disabled = get_bool_env("MFA_CONV_NAX_NO_FAST_PATH");
  if (is_pointwise && !fast_path_disabled) {
    return dispatch_pointwise_fast_path(x, w, B, T, H, W, C_in, C_out);
  }

  // Sprint III-1 (KD-7): the materialized-im2col path below uses the
  // upstream MLX im2col helper, which is broken for bf16 (utils.h:502
  // half vs bfloat16_t mismatch -> graph-eval-time "Unable to build
  // metal library").  Fail loudly at CALL time instead (Rule 8); the
  // auto-hook routes only MPP-eligible bf16 shapes here and falls back
  // upstream for the rest, so this guard is defense-in-depth for raw
  // C++ API users.
  if (x.dtype() == mlx::core::bfloat16) {
    throw std::runtime_error(
        "conv_nax: bf16 is only supported via the MPP convolution2d path "
        "(B=1, k=3x3x3, stride 1, dilation 1, pad (1,1,1), H/W % 8 == 0, "
        "C_in/C_out >= 32 and % 16 == 0, MFA_DISABLE_CONV3D_MPP unset). "
        "The legacy im2col path is fp16-only (KD-7: upstream MLX bf16 "
        "im2col bug). Use fp16 or an MPP-eligible shape.");
  }

  // Flatten weight (C_out, K_T*K_H*K_W*C_in) row-major.
  auto w_flat = mlx::core::reshape(w, {C_out, K});

  // III-6: pad the contraction K to a K_TILE multiple (matmul2d partial-
  // K-tile fix).  w_flat is loop-invariant → pad once; im2col buffers are
  // padded per chunk below.
  int k_pad = round_up_k_tile(K);
  auto w_flat_pad =
      pad_contraction_k(w_flat, K, k_pad,
                        mlx::core::default_stream(mlx::core::Device::gpu));

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
    enforce_int32_byte_offset_invariant(m_chunk, k_pad, dtype_bytes);

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
    // III-6: pad the im2col buffer's K axis to match w_flat_pad.
    auto im2col_buf_pad = pad_contraction_k(im2col_buf, K, k_pad, stream);

    // Build matmul kernel for this chunk.
    std::string mm_name = "conv3d_matmul2d_" +
        std::to_string(m_chunk) + "_" + std::to_string(k_pad) + "_" +
        std::to_string(N);
    auto mm_kernel = mlx::core::fast::metal_kernel(
        mm_name, {"A", "B"}, {"C"},
        matmul2d_source(m_chunk, k_pad, N), MATMUL_HEADER,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    int n_tg_x = (N + N_TILE - 1) / N_TILE;
    int n_tg_y = (m_chunk + M_TILE - 1) / M_TILE;
    auto mm_outs = mm_kernel(
        {im2col_buf_pad, w_flat_pad},
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
