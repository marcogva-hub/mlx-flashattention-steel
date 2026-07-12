/// Expert-only V6 NAX Linear specialized for rectangular VSR FFNs.
///
/// The NAX GEMM structure follows Apple's MIT-licensed
/// `steel_gemm_fused_nax.h`. The optional tanh-GELU epilogue is applied to
/// cooperative-tensor fragments before their only device-memory store.

#include "mfa_ffn_nax.hpp"

#include "mfa/v6_nax/NAAttentionKernel.hpp"

#include <mlx/fast.h>
#include <mlx/mlx.h>
#include <mlx/ops.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mlx_mfa {
bool device_has_neural_accelerators();

namespace {

constexpr int DEFAULT_BM = 64;
constexpr int DEFAULT_BK = 256;
constexpr int DEFAULT_WM = 2;
constexpr int SIMD_SIZE = 32;

int env_int_or_default(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (*end != '\0' || parsed <= 0 || parsed > 4096) {
    throw std::invalid_argument(std::string("v6_nax_linear: invalid ") + name);
  }
  return static_cast<int>(parsed);
}

int64_t product_leading_dims(const mlx::core::Shape& shape) {
  int64_t product = 1;
  for (int i = 0; i + 1 < static_cast<int>(shape.size()); ++i) {
    product *= shape[i];
  }
  return product;
}

std::string dtype_name(mlx::core::Dtype dtype) {
  if (dtype == mlx::core::float16) return "half";
  if (dtype == mlx::core::bfloat16) return "bfloat";
  throw std::invalid_argument("v6_nax_linear: dtype must be float16 or bfloat16");
}

std::string linear_header(
    const std::string& type,
    int M,
    int N,
    int K,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool gelu) {
  std::ostringstream os;
  os << R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;
#define Limits MfaFFNNAXLimits
)";
  os << mlx_mfa_v6_nax_helpers_block();
  os << R"(
#undef Limits
)";
  os << "typedef " << type << " FFNT;\n";
  os << "#define FFN_M " << M << "\n";
  os << "#define FFN_N " << N << "\n";
  os << "#define FFN_K " << K << "\n";
  os << "#define FFN_BM " << BM << "\n";
  os << "#define FFN_BN " << BN << "\n";
  os << "#define FFN_BK " << BK << "\n";
  os << "#define FFN_WM " << WM << "\n";
  os << "#define FFN_WN " << WN << "\n";
  os << "#define FFN_GELU " << (gelu ? 1 : 0) << "\n";
  os << R"(
using namespace mlx::steel;

template <typename T>
METAL_FUNC void ffn_mma(
    thread NAXTile<float, FFN_BM / FFN_WM / 16, FFN_BN / FFN_WN / 16>& C,
    thread NAXTile<T, FFN_BM / FFN_WM / 16, 2>& A,
    thread NAXTile<T, FFN_BN / FFN_WN / 16, 2>& B) {
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < FFN_BM / FFN_WM / 16; ++mm) {
    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < FFN_BN / FFN_WN / 16; nn += 2) {
      STEEL_PRAGMA_UNROLL
      for (short kk = 0; kk < 2; ++kk) {
        NAXTile<float, FFN_BM / FFN_WM / 16, FFN_BN / FFN_WN / 16>::NAXFrag_t::mma(
            C.frag_at(mm, nn), C.frag_at(mm, nn + 1), A.frag_at(mm, kk),
            metal::false_type{}, B.frag_at(nn, kk), B.frag_at(nn + 1, kk),
            metal::true_type{});
      }
    }
  }
}

METAL_FUNC float ffn_gelu(float x) {
  constexpr float kAlpha = 0.7978845608028654f;
  constexpr float kBeta = 0.044715f;
  return 0.5f * x * (1.0f + precise::tanh(kAlpha * (x + kBeta * x * x * x)));
}

template <typename Tile>
METAL_FUNC void ffn_epilogue(thread Tile& tile, const device FFNT* bias,
                             const int output_col) {
  const short2 coord = Tile::NAXFrag_t::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < Tile::kTileRows; ++mm) {
    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < Tile::kTileCols; ++nn) {
      thread auto& frag = tile.frag_at(mm, nn);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < Tile::NAXFrag_t::kElemRows; ++i) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < Tile::NAXFrag_t::kElemCols; ++j) {
          const int col = output_col + nn * 16 + coord.x + j;
          float value = frag[i * Tile::NAXFrag_t::kElemCols + j];
          if (col < FFN_N) value += float(bias[col]);
#if FFN_GELU
          value = ffn_gelu(value);
#endif
          frag[i * Tile::NAXFrag_t::kElemCols + j] = value;
        }
      }
    }
  }
}
)";
  return os.str();
}

std::string linear_source() {
  return R"(
  uint3 tid = threadgroup_position_in_grid;
  uint simd_gid = simdgroup_index_in_threadgroup;

  constexpr short SM = FFN_BM / FFN_WM;
  constexpr short SN = FFN_BN / FFN_WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const int tile_m = int(tid.y);
  const int tile_n = int(tid.x);
  const int output_row = tile_m * FFN_BM;
  const int output_col = tile_n * FFN_BN;
  const short tm = SM * short(simd_gid / FFN_WN);
  const short tn = SN * short(simd_gid % FFN_WN);
  const short valid_m = min(SM, short(FFN_M - (output_row + tm)));
  const short valid_n = min(SN, short(FFN_N - (output_col + tn)));

  const device FFNT* a = X + (output_row + tm) * FFN_K;
  const device FFNT* b = W + (output_col + tn) * FFN_K;
  device FFNT* y = Y + output_row * FFN_N + output_col;

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();
  for (int k = 0; k < FFN_K; k += FFN_BK) {
    for (int kk = 0; kk < FFN_BK; kk += SK) {
      NAXTile<FFNT, TM, TK> Atile;
      NAXTile<FFNT, TN, TK> Btile;
#if (FFN_M % FFN_BM) == 0
      Atile.load(a + k + kk, FFN_K);
#else
      Atile.load_safe(a + k + kk, FFN_K, short2(SK, valid_m));
#endif
#if (FFN_N % FFN_BN) == 0
      Btile.load(b + k + kk, FFN_K);
#else
      Btile.load_safe(b + k + kk, FFN_K, short2(SK, valid_n));
#endif
      ffn_mma<FFNT>(Dtile, Atile, Btile);
    }
  }

  ffn_epilogue(Dtile, Bias, output_col + tn);
#if (FFN_M % FFN_BM) == 0 && (FFN_N % FFN_BN) == 0
  Dtile.store(y + tm * FFN_N + tn, FFN_N);
#else
  Dtile.store_safe(y + tm * FFN_N + tn, FFN_N, short2(valid_n, valid_m));
#endif
)";
}

}  // namespace

mlx::core::array v6_nax_linear(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    const mlx::core::array& bias,
    bool gelu,
    mlx::core::StreamOrDevice s) {
  if (!device_has_neural_accelerators()) {
    throw std::runtime_error("v6_nax_linear: V6 NAX hardware is not available");
  }
  if (x.ndim() < 2 || weight.ndim() != 2 || bias.ndim() != 1) {
    throw std::invalid_argument("v6_nax_linear: x ndim>=2, weight 2-D, bias 1-D required");
  }
  if ((x.dtype() != mlx::core::float16 && x.dtype() != mlx::core::bfloat16) ||
      weight.dtype() != x.dtype() || bias.dtype() != x.dtype()) {
    throw std::invalid_argument("v6_nax_linear: x/weight/bias must share f16 or bf16 dtype");
  }
  const int K = static_cast<int>(x.shape(-1));
  const int N = static_cast<int>(weight.shape(0));
  if (weight.shape(1) != K || bias.shape(0) != N) {
    throw std::invalid_argument("v6_nax_linear: weight [N,K] and bias [N] shape mismatch");
  }
  const int64_t M64 = product_leading_dims(x.shape());
  if (M64 <= 0 || M64 > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("v6_nax_linear: flattened M is out of range");
  }
  const int M = static_cast<int>(M64);
  const int shape_bn = N >= K ? 64 : 256;
  const int shape_wn = N >= K ? 2 : 4;
  const int BM = env_int_or_default("MFA_FFN_NAX_BM", DEFAULT_BM);
  const int BN = env_int_or_default("MFA_FFN_NAX_BN", shape_bn);
  const int BK = env_int_or_default("MFA_FFN_NAX_BK", DEFAULT_BK);
  const int WM = env_int_or_default("MFA_FFN_NAX_WM", DEFAULT_WM);
  const int WN = env_int_or_default("MFA_FFN_NAX_WN", shape_wn);
  if (BM % (WM * 16) != 0 || BN % (WN * 32) != 0 ||
      BK < SIMD_SIZE || BK % SIMD_SIZE != 0 || K % BK != 0) {
    throw std::invalid_argument("v6_nax_linear: invalid tile override or K not divisible by BK");
  }

  auto st = mlx::core::to_stream(s);
  auto xc = mlx::core::contiguous(mlx::core::reshape(x, {M, K}), false, st);
  auto wc = mlx::core::contiguous(weight, false, st);
  auto bc = mlx::core::contiguous(bias, false, st);
  const std::string type = dtype_name(x.dtype());
  std::string name = "v6_nax_linear_" + type + "_M" + std::to_string(M) +
      "_N" + std::to_string(N) + "_K" + std::to_string(K) +
      "_bm" + std::to_string(BM) + "_bn" + std::to_string(BN) +
      "_bk" + std::to_string(BK) + "_wm" + std::to_string(WM) +
      "_wn" + std::to_string(WN) + "_gelu" + (gelu ? "1" : "0");
  auto kernel = mlx::core::fast::metal_kernel(
      name, {"X", "W", "Bias"}, {"Y"}, linear_source(),
      linear_header(type, M, N, K, BM, BN, BK, WM, WN, gelu),
      /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
  const int tiles_n = (N + BN - 1) / BN;
  const int tiles_m = (M + BM - 1) / BM;
  auto outs = kernel(
      {xc, wc, bc}, {mlx::core::Shape{M, N}}, {x.dtype()},
      {tiles_n * SIMD_SIZE, tiles_m * WN, WM}, {SIMD_SIZE, WN, WM}, {},
      std::nullopt, false, st);
  auto out_shape = x.shape();
  out_shape.back() = N;
  return mlx::core::reshape(outs[0], out_shape);
}

}  // namespace mlx_mfa
