/// V6 NAX quantized matmul expert kernel.
///
/// MIT-attributed adaptation points from MLX `quantized_nax.h`:
///   - packed-weight dequantization into threadgroup memory;
///   - MLX quantized weight/scales/biases layout for transpose=True;
///   - baseline tile shape BM=BN=BK=64, WM=WN=2.

#include "mfa_qmm_nax.hpp"

#include "mfa/v6_nax/NAAttentionKernel.hpp"

#include <mlx/fast.h>
#include <mlx/mlx.h>
#include <mlx/ops.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx_mfa {
bool device_has_neural_accelerators();

namespace {

constexpr int DEFAULT_BM = 64;
constexpr int DEFAULT_BN = 64;
constexpr int DEFAULT_BK = 64;
constexpr int DEFAULT_WM = 2;
constexpr int DEFAULT_WN = 2;
constexpr int SIMD_SIZE = 32;

std::string qmm_dtype_name(mlx::core::Dtype dtype) {
  if (dtype == mlx::core::float16) return "half";
  if (dtype == mlx::core::bfloat16) return "bfloat";
  throw std::invalid_argument("v6_nax_quantized_matmul: x dtype must be float16 or bfloat16");
}

int env_int_or_default(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (*end != '\0' || parsed <= 0 || parsed > 4096) {
    throw std::invalid_argument(std::string("v6_nax_quantized_matmul: invalid ") + name);
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

std::string qmm_header(
    const std::string& dtype_name,
    int M,
    int N,
    int K,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN) {
  const int pack_factor = 8 / bits;
  const int bytes_per_pack = 1;
  const int bk_padded = BK + 16 / 2;  // half/bfloat element size.
  std::ostringstream os;
  os << R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;
// mx.fast.metal_kernel prepends MLX utility headers which already define
// Limits<T>. The shared V6 attention helper also defines Limits<T>, so isolate
// that symbol for this embedded helper block.
#define Limits MfaQmmNAXLimits
)";
  os << mlx_mfa_v6_nax_helpers_block();
  os << R"(
#undef Limits
)";
  os << "typedef " << dtype_name << " QMMT;\n";
  os << R"(
using namespace mlx::steel;

#define QMM_M )" << M << R"(
#define QMM_N )" << N << R"(
#define QMM_K )" << K << R"(
#define QMM_GROUP_SIZE )" << group_size << R"(
#define QMM_BITS )" << bits << R"(
#define QMM_BM )" << BM << R"(
#define QMM_BN )" << BN << R"(
#define QMM_BK )" << BK << R"(
#define QMM_WM )" << WM << R"(
#define QMM_WN )" << WN << R"(
#define QMM_PACK_FACTOR )" << pack_factor << R"(
#define QMM_BYTES_PER_PACK )" << bytes_per_pack << R"(
#define QMM_BK_PADDED )" << bk_padded << R"(
#define QMM_TGP_SIZE (QMM_WM * QMM_WN * 32)

template <typename T>
METAL_FUNC void qmm_dequantize_bits4(const device uint8_t* w, T scale, T bias,
                                     threadgroup T* w_local) {
  T s0 = scale;
  T s1 = scale / static_cast<T>(16.0f);
  w_local[0] = s0 * (w[0] & 0x0f) + bias;
  w_local[1] = s1 * (w[0] & 0xf0) + bias;
}

template <typename T>
METAL_FUNC void qmm_dequantize_bits8(const device uint8_t* w, T scale, T bias,
                                     threadgroup T* w_local) {
  w_local[0] = scale * w[0] + bias;
}

template <typename T>
struct QMMQuantizedBlockLoader {
  STEEL_CONST short BROWS = QMM_BN;
  STEEL_CONST short BCOLS = QMM_BK;
  STEEL_CONST short dst_ld = QMM_BK_PADDED;
  STEEL_CONST short pack_factor = QMM_PACK_FACTOR;
  STEEL_CONST short bytes_per_pack = QMM_BYTES_PER_PACK;
  STEEL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  STEEL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < QMM_TGP_SIZE) ? 1 : (BCOLS_PACKED * BROWS) / QMM_TGP_SIZE;
  STEEL_CONST short group_steps = QMM_GROUP_SIZE / BCOLS;
  STEEL_CONST short n_groups = BCOLS / QMM_GROUP_SIZE;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const short thread_idx;
  const short bi;
  const short bj;
  const short group_id;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QMMQuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      int src_ld_,
      threadgroup T* dst_,
      uint simd_group_id,
      uint simd_lane_id)
      : src_ld(src_ld_),
        tile_stride(BCOLS_PACKED * bytes_per_pack),
        group_step_cnt(0),
        thread_idx(short(simd_group_id * 32 + simd_lane_id)),
        bi(short(n_reads * thread_idx / BCOLS_PACKED)),
        bj(short((n_reads * thread_idx) % BCOLS_PACKED)),
        group_id(short((bj * pack_factor) / QMM_GROUP_SIZE)),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor + bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / QMM_GROUP_SIZE + (QMM_GROUP_SIZE == 32 ? group_id : 0)),
        biases(biases_ + bi * src_ld / QMM_GROUP_SIZE + (QMM_GROUP_SIZE == 32 ? group_id : 0)) {}

  METAL_FUNC void dequantize_one(const device uint8_t* src_i, T scale, T bias,
                                 threadgroup T* dst_i) const {
#if QMM_BITS == 4
    qmm_dequantize_bits4<T>(src_i, scale, bias, dst_i);
#else
    qmm_dequantize_bits8<T>(src_i, scale, bias, dst_i);
#endif
  }

  METAL_FUNC void zero_local() const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < n_reads * pack_factor; ++i) {
      dst[i] = T(0);
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < QMM_TGP_SIZE && bi >= BROWS) {
      return;
    }
    if (bi >= src_tile_dim.y) {
      zero_local();
      return;
    }
    T scale = *scales;
    T bias = *biases;
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < n_reads; ++i) {
      dequantize_one(src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  METAL_FUNC void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < QMM_TGP_SIZE && bi >= BROWS) {
      return;
    }
    T scale = *scales;
    T bias = *biases;
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < n_reads; ++i) {
      dequantize_one(src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  METAL_FUNC void next() {
    src += tile_stride;
    if (QMM_GROUP_SIZE == 32) {
      scales += n_groups;
      biases += n_groups;
    } else {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    }
  }
};

template <typename T>
METAL_FUNC void qmm_mma(
    thread NAXTile<float, QMM_BM / QMM_WM / 16, QMM_BN / QMM_WN / 16>& C,
    thread NAXTile<T, QMM_BM / QMM_WM / 16, 2>& A,
    thread NAXTile<T, QMM_BN / QMM_WN / 16, 2>& B) {
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < QMM_BM / QMM_WM / 16; ++mm) {
    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < QMM_BN / QMM_WN / 16; nn += 2) {
      STEEL_PRAGMA_UNROLL
      for (short kk = 0; kk < 2; ++kk) {
        NAXTile<float, QMM_BM / QMM_WM / 16, QMM_BN / QMM_WN / 16>::NAXFrag_t::mma(
            C.frag_at(mm, nn),
            C.frag_at(mm, nn + 1),
            A.frag_at(mm, kk),
            metal::false_type{},
            B.frag_at(nn, kk),
            B.frag_at(nn + 1, kk),
            metal::true_type{});
      }
    }
  }
}
)";
  return os.str();
}

std::string qmm_source() {
  return R"(
  uint3 tid = threadgroup_position_in_grid;
  uint lid = thread_index_in_threadgroup;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;
  (void)lid;
  threadgroup QMMT Ws[QMM_BN * QMM_BK_PADDED];

  constexpr short SM = QMM_BM / QMM_WM;
  constexpr short SN = QMM_BN / QMM_WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const int K_w = QMM_K * QMM_BYTES_PER_PACK / QMM_PACK_FACTOR;
  const int K_g = QMM_K / QMM_GROUP_SIZE;
  const int y_row = int(tid.y) * QMM_BM;
  const int y_col = int(tid.x) * QMM_BN;
  const short tm = SM * short(simd_gid / QMM_WN);
  const short tn = SN * short(simd_gid % QMM_WN);
  const short sgp_sm = min(SM, short(QMM_M - (y_row + tm)));
  const short sgp_sn = min(SN, short(QMM_N - (y_col + tn)));
  const short tgp_bn = min(QMM_BN, int(QMM_N - y_col));

  const device uint8_t* wl = reinterpret_cast<const device uint8_t*>(W) + y_col * K_w;
  const device QMMT* scales = Scales + y_col * K_g;
  const device QMMT* biases = Biases + y_col * K_g;
  const device QMMT* x = X + (y_row + tm) * QMM_K;
  device QMMT* y = Y + y_row * QMM_N + y_col;

  QMMQuantizedBlockLoader<QMMT> loader_w(wl, scales, biases, QMM_K, Ws, simd_gid, simd_lid);
  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (int k = 0; k < QMM_K; k += QMM_BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if (QMM_N % QMM_BN) == 0
    loader_w.load_unsafe();
#else
    loader_w.load_safe(short2(QMM_BK, tgp_bn));
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kk1 = 0; kk1 < QMM_BK; kk1 += SK) {
      NAXTile<QMMT, TM, TK> Atile;
      NAXTile<QMMT, TN, TK> Btile;
#if (QMM_M % QMM_BM) == 0
      Atile.load(x + kk1, QMM_K);
#else
      Atile.load_safe(x + kk1, QMM_K, short2(SK, sgp_sm));
#endif
      Btile.load(Ws + tn * QMM_BK_PADDED + kk1, QMM_BK_PADDED);
      qmm_mma<QMMT>(Dtile, Atile, Btile);
    }
    x += QMM_BK;
    loader_w.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
#if (QMM_M % QMM_BM) == 0 && (QMM_N % QMM_BN) == 0
  Dtile.store(y + tm * QMM_N + tn, QMM_N);
#else
  Dtile.store_safe(y + tm * QMM_N + tn, QMM_N, short2(sgp_sn, sgp_sm));
#endif
)";
}

void validate_qmm_inputs(
    const mlx::core::array& x,
    const mlx::core::array& w_q,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    int group_size,
    int bits) {
  if (!device_has_neural_accelerators()) {
    throw std::runtime_error("v6_nax_quantized_matmul: V6 NAX hardware is not available");
  }
  if (x.ndim() < 2) {
    throw std::invalid_argument("v6_nax_quantized_matmul: x must have ndim >= 2");
  }
  if (x.dtype() != mlx::core::float16 && x.dtype() != mlx::core::bfloat16) {
    throw std::invalid_argument("v6_nax_quantized_matmul: x dtype must be float16 or bfloat16");
  }
  if (w_q.dtype() != mlx::core::uint32) {
    throw std::invalid_argument("v6_nax_quantized_matmul: w_q dtype must be uint32");
  }
  if (bits != 4 && bits != 8) {
    throw std::invalid_argument("v6_nax_quantized_matmul: only bits=4 and bits=8 are supported");
  }
  if (group_size != 32 && group_size != 64 && group_size != 128) {
    throw std::invalid_argument("v6_nax_quantized_matmul: group_size must be 32, 64, or 128");
  }
  if (w_q.ndim() != 2 || scales.ndim() != 2 || biases.ndim() != 2) {
    throw std::invalid_argument(
        "v6_nax_quantized_matmul: w_q, scales, and biases must be 2-D transpose=True tensors");
  }
  const int K = static_cast<int>(x.shape(-1));
  const int N = static_cast<int>(w_q.shape(0));
  const int pack_factor = 32 / bits;
  if (K <= 0 || N <= 0) {
    throw std::invalid_argument("v6_nax_quantized_matmul: K and output dimension must be positive");
  }
  if (K % 64 != 0) {
    throw std::invalid_argument("v6_nax_quantized_matmul: K must be a multiple of 64 for V6 NAX qmm");
  }
  if (K % group_size != 0) {
    throw std::invalid_argument("v6_nax_quantized_matmul: K must be divisible by group_size");
  }
  if (w_q.shape(1) != K / pack_factor) {
    throw std::invalid_argument("v6_nax_quantized_matmul: w_q packed dimension does not match x K/bits");
  }
  if (scales.shape(0) != N || biases.shape(0) != N ||
      scales.shape(1) != K / group_size || biases.shape(1) != K / group_size) {
    throw std::invalid_argument(
        "v6_nax_quantized_matmul: scales/biases must have shape [N, K/group_size]");
  }
}

}  // namespace

mlx::core::array v6_nax_quantized_matmul(
    const mlx::core::array& x,
    const mlx::core::array& w_q,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    int group_size,
    int bits,
    mlx::core::StreamOrDevice s) {
  validate_qmm_inputs(x, w_q, scales, biases, group_size, bits);
  auto st = mlx::core::to_stream(s);
  const int K = static_cast<int>(x.shape(-1));
  const int N = static_cast<int>(w_q.shape(0));
  const int64_t M64 = product_leading_dims(x.shape());
  if (M64 <= 0 || M64 > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("v6_nax_quantized_matmul: flattened M is out of supported int range");
  }
  const int M = static_cast<int>(M64);

  const int BM = env_int_or_default("MFA_QMM_NAX_BM", DEFAULT_BM);
  const int BN = env_int_or_default("MFA_QMM_NAX_BN", DEFAULT_BN);
  const int BK = env_int_or_default("MFA_QMM_NAX_BK", DEFAULT_BK);
  const int WM = env_int_or_default("MFA_QMM_NAX_WM", DEFAULT_WM);
  const int WN = env_int_or_default("MFA_QMM_NAX_WN", DEFAULT_WN);
  if (BM % (WM * 16) != 0 || BN % (WN * 16) != 0 || BK % SIMD_SIZE != 0 || BK < SIMD_SIZE) {
    throw std::invalid_argument("v6_nax_quantized_matmul: invalid BM/BN/BK/WM/WN tile env override");
  }
  if (K % BK != 0) {
    throw std::invalid_argument("v6_nax_quantized_matmul: K must be divisible by BK");
  }
  if (BK > group_size && group_size != 32) {
    throw std::invalid_argument("v6_nax_quantized_matmul: BK > group_size is only supported for group_size=32");
  }

  auto x_flat = mlx::core::reshape(x, {M, K});
  auto xc = mlx::core::contiguous(x_flat, false, st);
  auto wc = mlx::core::contiguous(w_q, false, st);
  auto sc = mlx::core::astype(scales, x.dtype(), st);
  auto bc = mlx::core::astype(biases, x.dtype(), st);
  sc = mlx::core::contiguous(sc, false, st);
  bc = mlx::core::contiguous(bc, false, st);

  const std::string dtype_name = qmm_dtype_name(x.dtype());
  std::string name = "v6_nax_qmm_" + dtype_name + "_M" + std::to_string(M) +
      "_N" + std::to_string(N) + "_K" + std::to_string(K) +
      "_g" + std::to_string(group_size) + "_b" + std::to_string(bits) +
      "_bm" + std::to_string(BM) + "_bn" + std::to_string(BN) +
      "_bk" + std::to_string(BK) + "_wm" + std::to_string(WM) +
      "_wn" + std::to_string(WN);
  auto kernel = mlx::core::fast::metal_kernel(
      name,
      {"W", "Scales", "Biases", "X"},
      {"Y"},
      qmm_source(),
      qmm_header(dtype_name, M, N, K, group_size, bits, BM, BN, BK, WM, WN),
      /*ensure_row_contiguous=*/true,
      /*atomic_outputs=*/false);
  const int n_tg_x = (N + BN - 1) / BN;
  const int n_tg_y = (M + BM - 1) / BM;
  auto outs = kernel(
      {wc, sc, bc, xc},
      {mlx::core::Shape{M, N}},
      {x.dtype()},
      {n_tg_x * SIMD_SIZE, n_tg_y * WN, WM},
      {SIMD_SIZE, WN, WM},
      {},
      std::nullopt,
      false,
      st);
  auto y_flat = outs[0];
  mlx::core::Shape out_shape = x.shape();
  out_shape.back() = N;
  return mlx::core::reshape(y_flat, out_shape);
}

}  // namespace mlx_mfa
