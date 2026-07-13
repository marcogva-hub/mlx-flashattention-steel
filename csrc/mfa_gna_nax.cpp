#include "mfa_gna_nax.hpp"

#include "mfa/v6_nax/NAAttentionKernel.hpp"

#include <mlx/fast.h>
#include <mlx/mlx.h>
#include <mlx/ops.h>

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mlx_mfa {

bool device_has_neural_accelerators();

namespace {

int env_int_or_default(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (!raw || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || value <= 0 || value > 4096) {
    throw std::invalid_argument(std::string(name) + " must be a positive integer");
  }
  return static_cast<int>(value);
}

int env_nonnegative_int_or_default(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (!raw || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || value < 0 || value > 4) {
    throw std::invalid_argument(std::string(name) + " must be an integer in [0,4]");
  }
  return static_cast<int>(value);
}

std::string dtype_name(mlx::core::Dtype dtype) {
  if (dtype == mlx::core::float16) {
    return "half";
  }
  if (dtype == mlx::core::bfloat16) {
    return "bfloat";
  }
  throw std::invalid_argument("GNA NAX supports float16 and bfloat16 inputs only");
}

std::string gna_nax_header(
    const std::string& dtype,
    int B,
    int Hq,
    int Hk,
    int N,
    int D,
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
    int BQ,
    int BK,
    int WM,
    bool precompute_range,
    int swizzle_log) {
  const int NQ = (N + BQ - 1) / BQ;
  const int NK = (N + BK - 1) / BK;
  const int TQ = BQ / (WM * 16);
  const int TK = BK / 16;
  const int TD = D / 16;
  const int GQA = Hq / Hk;
  const float log2_scale = scale * 1.4426950408889634f;

  std::ostringstream os;
  os << "#include <metal_stdlib>\n";
  os << "#include <metal_simdgroup>\n";
  os << "#include <metal_simdgroup_matrix>\n";
  os << "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  os << "using namespace metal;\n";
  os << "using namespace mpp::tensor_ops;\n";
  os << "#define Limits MfaGNANAXLimits\n";
  os << mlx_mfa_v6_nax_helpers_block();
  os << "#undef Limits\n";
  os << "using namespace mlx::steel;\n\n";
  os << "#define GNAT " << dtype << "\n";
  os << "#define GNA_B " << B << "\n";
  os << "#define GNA_HQ " << Hq << "\n";
  os << "#define GNA_HK " << Hk << "\n";
  os << "#define GNA_GQA " << GQA << "\n";
  os << "#define GNA_N " << N << "\n";
  os << "#define GNA_D " << D << "\n";
  os << "#define GNA_BQ " << BQ << "\n";
  os << "#define GNA_BK " << BK << "\n";
  os << "#define GNA_WM " << WM << "\n";
  os << "#define GNA_TQ " << TQ << "\n";
  os << "#define GNA_TK " << TK << "\n";
  os << "#define GNA_TD " << TD << "\n";
  os << "#define GNA_NQ " << NQ << "\n";
  os << "#define GNA_NK " << NK << "\n";
  os << "#define GNA_DIM0 " << dim0 << "\n";
  os << "#define GNA_DIM1 " << dim1 << "\n";
  os << "#define GNA_DIM2 " << dim2 << "\n";
  os << "#define GNA_WIN0 " << window0 << "\n";
  os << "#define GNA_WIN1 " << window1 << "\n";
  os << "#define GNA_WIN2 " << window2 << "\n";
  os << "#define GNA_STR0 " << stride0 << "\n";
  os << "#define GNA_STR1 " << stride1 << "\n";
  os << "#define GNA_STR2 " << stride2 << "\n";
  os << "#define GNA_LOG2_SCALE " << log2_scale << "f\n\n";
  os << "#define GNA_PRECOMPUTE_RANGE " << (precompute_range ? 1 : 0) << "\n\n";
  os << "#define GNA_SWIZZLE_LOG " << swizzle_log << "\n\n";

  os << R"MSL(
inline bool gna_pair_active(int q_idx, int k_idx) {
  if (q_idx < 0 || q_idx >= GNA_N || k_idx < 0 || k_idx >= GNA_N) {
    return false;
  }

  const int dim12 = GNA_DIM1 * GNA_DIM2;
  const int q0 = q_idx / dim12;
  const int q1 = (q_idx / GNA_DIM2) % GNA_DIM1;
  const int q2 = q_idx % GNA_DIM2;
  const int k0 = k_idx / dim12;
  const int k1 = (k_idx / GNA_DIM2) % GNA_DIM1;
  const int k2 = k_idx % GNA_DIM2;

  const int g0 = q0 / GNA_STR0;
  const int g1 = q1 / GNA_STR1;
  const int g2 = q2 / GNA_STR2;
  const int lo0 = max(0, g0 * GNA_STR0 - (GNA_WIN0 - GNA_STR0) / 2);
  const int hi0 = min(GNA_DIM0 - 1, (g0 + 1) * GNA_STR0 + (GNA_WIN0 - GNA_STR0 + 1) / 2 - 1);
  const int lo1 = max(0, g1 * GNA_STR1 - (GNA_WIN1 - GNA_STR1) / 2);
  const int hi1 = min(GNA_DIM1 - 1, (g1 + 1) * GNA_STR1 + (GNA_WIN1 - GNA_STR1 + 1) / 2 - 1);
  const int lo2 = max(0, g2 * GNA_STR2 - (GNA_WIN2 - GNA_STR2) / 2);
  const int hi2 = min(GNA_DIM2 - 1, (g2 + 1) * GNA_STR2 + (GNA_WIN2 - GNA_STR2 + 1) / 2 - 1);

  return lo0 <= k0 && k0 <= hi0 && lo1 <= k1 && k1 <= hi1 && lo2 <= k2 && k2 <= hi2;
}

inline bool gna_tile_active(int q_start, int q_end, int k_start, int k_end) {
  q_end = min(q_end, GNA_N);
  k_end = min(k_end, GNA_N);
  if (q_start >= q_end || k_start >= k_end) {
    return false;
  }

  const int dim12 = GNA_DIM1 * GNA_DIM2;
  const int q_first = q_start;
  const int q_last = q_end - 1;
  const int k_first = k_start;
  const int k_last = k_end - 1;

  int q_min0 = q_first / dim12;
  int q_max0 = q_last / dim12;
  int q_min1 = (q_first / GNA_DIM2) % GNA_DIM1;
  int q_max1 = (q_last / GNA_DIM2) % GNA_DIM1;
  int q_min2 = q_first % GNA_DIM2;
  int q_max2 = q_last % GNA_DIM2;
  if (q_max0 > q_min0) {
    q_min1 = 0; q_max1 = GNA_DIM1 - 1;
    q_min2 = 0; q_max2 = GNA_DIM2 - 1;
  } else if (q_max1 > q_min1) {
    q_min2 = 0; q_max2 = GNA_DIM2 - 1;
  }

  const int half_lo0 = (GNA_WIN0 - GNA_STR0) / 2;
  const int half_hi0 = (GNA_WIN0 - GNA_STR0 + 1) / 2;
  const int half_lo1 = (GNA_WIN1 - GNA_STR1) / 2;
  const int half_hi1 = (GNA_WIN1 - GNA_STR1 + 1) / 2;
  const int half_lo2 = (GNA_WIN2 - GNA_STR2) / 2;
  const int half_hi2 = (GNA_WIN2 - GNA_STR2 + 1) / 2;

  const int grp_min0 = q_min0 / GNA_STR0;
  const int grp_max0 = q_max0 / GNA_STR0;
  const int grp_min1 = q_min1 / GNA_STR1;
  const int grp_max1 = q_max1 / GNA_STR1;
  const int grp_min2 = q_min2 / GNA_STR2;
  const int grp_max2 = q_max2 / GNA_STR2;

  const int win_lo0 = max(0, grp_min0 * GNA_STR0 - half_lo0);
  const int win_hi0 = min(GNA_DIM0 - 1, (grp_max0 + 1) * GNA_STR0 + half_hi0 - 1);
  const int win_lo1 = max(0, grp_min1 * GNA_STR1 - half_lo1);
  const int win_hi1 = min(GNA_DIM1 - 1, (grp_max1 + 1) * GNA_STR1 + half_hi1 - 1);
  const int win_lo2 = max(0, grp_min2 * GNA_STR2 - half_lo2);
  const int win_hi2 = min(GNA_DIM2 - 1, (grp_max2 + 1) * GNA_STR2 + half_hi2 - 1);

  int k_min0 = k_first / dim12;
  int k_max0 = k_last / dim12;
  int k_min1 = (k_first / GNA_DIM2) % GNA_DIM1;
  int k_max1 = (k_last / GNA_DIM2) % GNA_DIM1;
  int k_min2 = k_first % GNA_DIM2;
  int k_max2 = k_last % GNA_DIM2;
  if (k_max0 > k_min0) {
    k_min1 = 0; k_max1 = GNA_DIM1 - 1;
    k_min2 = 0; k_max2 = GNA_DIM2 - 1;
  } else if (k_max1 > k_min1) {
    k_min2 = 0; k_max2 = GNA_DIM2 - 1;
  }

  return k_max0 >= win_lo0 && k_min0 <= win_hi0 &&
         k_max1 >= win_lo1 && k_min1 <= win_hi1 &&
         k_max2 >= win_lo2 && k_min2 <= win_hi2;
}

struct GNAWindowBounds {
  int lo0, hi0, lo1, hi1, lo2, hi2;
  int kb_begin, kb_end;
};

inline GNAWindowBounds gna_window_bounds(int q_start, int q_end) {
  q_end = min(q_end, GNA_N);
  const int dim12 = GNA_DIM1 * GNA_DIM2;
  const int q_first = q_start;
  const int q_last = q_end - 1;
  int q_min0 = q_first / dim12;
  int q_max0 = q_last / dim12;
  int q_min1 = (q_first / GNA_DIM2) % GNA_DIM1;
  int q_max1 = (q_last / GNA_DIM2) % GNA_DIM1;
  int q_min2 = q_first % GNA_DIM2;
  int q_max2 = q_last % GNA_DIM2;
  if (q_max0 > q_min0) {
    q_min1 = 0; q_max1 = GNA_DIM1 - 1;
    q_min2 = 0; q_max2 = GNA_DIM2 - 1;
  } else if (q_max1 > q_min1) {
    q_min2 = 0; q_max2 = GNA_DIM2 - 1;
  }
  const int lo0 = max(0, (q_min0 / GNA_STR0) * GNA_STR0 -
                         (GNA_WIN0 - GNA_STR0) / 2);
  const int hi0 = min(GNA_DIM0 - 1, ((q_max0 / GNA_STR0) + 1) * GNA_STR0 +
                         (GNA_WIN0 - GNA_STR0 + 1) / 2 - 1);
  const int lo1 = max(0, (q_min1 / GNA_STR1) * GNA_STR1 -
                         (GNA_WIN1 - GNA_STR1) / 2);
  const int hi1 = min(GNA_DIM1 - 1, ((q_max1 / GNA_STR1) + 1) * GNA_STR1 +
                         (GNA_WIN1 - GNA_STR1 + 1) / 2 - 1);
  const int lo2 = max(0, (q_min2 / GNA_STR2) * GNA_STR2 -
                         (GNA_WIN2 - GNA_STR2) / 2);
  const int hi2 = min(GNA_DIM2 - 1, ((q_max2 / GNA_STR2) + 1) * GNA_STR2 +
                         (GNA_WIN2 - GNA_STR2 + 1) / 2 - 1);
  const int first = lo0 * dim12 + lo1 * GNA_DIM2 + lo2;
  const int last = hi0 * dim12 + hi1 * GNA_DIM2 + hi2;
  return {lo0, hi0, lo1, hi1, lo2, hi2,
          first / GNA_BK, min(GNA_NK, (last + 1 + GNA_BK - 1) / GNA_BK)};
}

inline bool gna_tile_active_bounds(
    thread const GNAWindowBounds& window, int k_start, int k_end) {
  k_end = min(k_end, GNA_N);
  if (k_start >= k_end) return false;
  const int dim12 = GNA_DIM1 * GNA_DIM2;
  const int k_first = k_start;
  const int k_last = k_end - 1;
  int k_min0 = k_first / dim12;
  int k_max0 = k_last / dim12;
  int k_min1 = (k_first / GNA_DIM2) % GNA_DIM1;
  int k_max1 = (k_last / GNA_DIM2) % GNA_DIM1;
  int k_min2 = k_first % GNA_DIM2;
  int k_max2 = k_last % GNA_DIM2;
  if (k_max0 > k_min0) {
    k_min1 = 0; k_max1 = GNA_DIM1 - 1;
    k_min2 = 0; k_max2 = GNA_DIM2 - 1;
  } else if (k_max1 > k_min1) {
    k_min2 = 0; k_max2 = GNA_DIM2 - 1;
  }
  return k_max0 >= window.lo0 && k_min0 <= window.hi0 &&
         k_max1 >= window.lo1 && k_min1 <= window.hi1 &&
         k_max2 >= window.lo2 && k_min2 <= window.hi2;
}
)MSL";
  return os.str();
}

std::string gna_nax_source() {
  return R"MSL(
  uint3 raw_tid = threadgroup_position_in_grid;
  uint3 tid = raw_tid;
#if GNA_SWIZZLE_LOG > 0
  // Adapted from MLX steel/gemm/transforms.h BlockSwizzle (MIT, Apple Inc.).
  // This is an opt-in GNA probe; the default grid walk remains unchanged.
  const int swizzle_mask = (1 << GNA_SWIZZLE_LOG) - 1;
  tid.x = raw_tid.x >> GNA_SWIZZLE_LOG;
  tid.y = (raw_tid.y << GNA_SWIZZLE_LOG) + (raw_tid.x & swizzle_mask);
#endif
  uint simd_gid = simdgroup_index_in_threadgroup;
  if (tid.x >= GNA_NQ || tid.y >= GNA_HQ || tid.z >= GNA_B) {
    return;
  }

  constexpr short kTileRows = GNA_TQ;
  constexpr short kHeadTiles = GNA_D / 16;
  using stile_t = NAXTile<float, kTileRows, GNA_TK>;
  using otile_t = NAXTile<float, kTileRows, kHeadTiles>;
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  const int q_tile = int(tid.x);
  const int hq = int(tid.y);
  const int hk = hq / GNA_GQA;
  const int b = int(tid.z);
  const int tm = int(simd_gid) * kTileRows * 16;

  const bool is_last_q = ((GNA_N % GNA_BQ) != 0) && (q_tile == (GNA_N / GNA_BQ));
  const short lim_rows_q = short(((GNA_N % GNA_BQ) != 0 ? (GNA_N % GNA_BQ) : GNA_BQ) - tm);

  const device GNAT* Q_head = Q + ((b * GNA_HQ + hq) * GNA_N) * GNA_D;
  const device GNAT* K_head = K + ((b * GNA_HK + hk) * GNA_N) * GNA_D;
  const device GNAT* V_head = V + ((b * GNA_HK + hk) * GNA_N) * GNA_D;
  device GNAT* O_head = O + ((b * GNA_HQ + hq) * GNA_N) * GNA_D;
  const device GNAT* Q_q = Q_head + q_tile * GNA_BQ * GNA_D + tm * GNA_D;

  otile_t Otile;
  Otile.clear();
  metal::vec<float, kRowsPT> max_score;
  metal::vec<float, kRowsPT> sum_score{0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = -INFINITY;
  }

  #if GNA_PRECOMPUTE_RANGE
  const GNAWindowBounds window_bounds = gna_window_bounds(
      q_tile * GNA_BQ, (q_tile + 1) * GNA_BQ);
  const int kb_begin = window_bounds.kb_begin;
  const int kb_end = window_bounds.kb_end;
  #else
  const int kb_begin = 0;
  const int kb_end = GNA_NK;
  #endif

  for (int kb = kb_begin; kb < kb_end; ++kb) {
  #if GNA_PRECOMPUTE_RANGE
    if (!gna_tile_active_bounds(window_bounds, kb * GNA_BK, (kb + 1) * GNA_BK)) {
  #else
    if (!gna_tile_active(q_tile * GNA_BQ, (q_tile + 1) * GNA_BQ,
                         kb * GNA_BK, (kb + 1) * GNA_BK)) {
  #endif
      continue;
    }

    const bool is_last_k = ((GNA_N % GNA_BK) != 0) && (kb == (GNA_N / GNA_BK));
    const short lim_rows_k = short(((GNA_N % GNA_BK) != 0 ? (GNA_N % GNA_BK) : GNA_BK));
    const device GNAT* K_kb = K_head + kb * GNA_BK * GNA_D;
    const device GNAT* V_kb = V_head + kb * GNA_BK * GNA_D;

    stile_t Stile;
    Stile.clear();

    for (short iq = 0; iq < kTileRows; ++iq) {
      for (short ik = 0; ik < GNA_TK; ik += 2) {
        for (short id = 0; id < kHeadTiles; ++id) {
          NAXTile<GNAT, 1, 1> Qtile;
          const device GNAT* Qp = Q_q + iq * 16 * GNA_D + id * 16;
          if (is_last_q) {
            Qtile.load_rows(Qp, GNA_D, lim_rows_q - iq * 16);
          } else {
            Qtile.load(Qp, GNA_D);
          }

          NAXTile<GNAT, 2, 1> Ktile;
          const device GNAT* Kp = K_kb + ik * 16 * GNA_D + id * 16;
          if (is_last_k) {
            Ktile.load_rows(Kp, GNA_D, lim_rows_k - ik * 16);
          } else {
            Ktile.load(Kp, GNA_D);
          }

          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= GNA_LOG2_SCALE;
    }

    const short2 sc = stile_t::NAXFrag_t::get_coord();
    const short sn = sc.x;
    const short sm = sc.y;
    for (short iq = 0; iq < kTileRows; ++iq) {
      for (short ik = 0; ik < GNA_TK; ++ik) {
        thread auto& fg = Stile.frag_at(iq, ik);
        for (short ii = 0; ii < stile_t::kFragThrRows; ++ii) {
          const int row = q_tile * GNA_BQ + tm + iq * 16 +
              ii * stile_t::kFragRowsJump + int(sm);
          for (short jj = 0; jj < stile_t::kFragThrCols; ++jj) {
            const int col = kb * GNA_BK + ik * 16 + int(jj) + int(sn);
            if (!gna_pair_active(row, col)) {
              fg[ii * stile_t::kFragThrCols + jj] = -INFINITY;
            }
          }
        }
      }
    }

    metal::vec<float, kRowsPT> new_max = max_score;
    Stile.template row_reduce<MaxOp>(new_max);
    metal::vec<float, kRowsPT> correction;
    for (short i = 0; i < kRowsPT; ++i) {
      if (new_max[i] > max_score[i]) {
        correction[i] = fast::exp2(max_score[i] - new_max[i]);
        max_score[i] = new_max[i];
      } else {
        correction[i] = 1.0f;
        new_max[i] = metal::isinf(max_score[i]) ? 0.0f : max_score[i];
      }
    }

    Stile.template row_bin_op<ExpSubOp>(new_max);

    metal::vec<float, kRowsPT> sum_tmp{0};
    Stile.template row_reduce<SumOp>(sum_tmp);
    sum_score = sum_score * correction + sum_tmp;
    Otile.template row_bin_op<MulOp>(correction);

    for (short iq = 0; iq < kTileRows; ++iq) {
      for (short ik = 0; ik < GNA_TK; ++ik) {
        for (short id = 0; id < kHeadTiles; id += 2) {
          NAXTile<GNAT, 1, 2> Vtile;
          const device GNAT* Vp = V_kb + ik * 16 * GNA_D + id * 16;
          if (is_last_k) {
            Vtile.load_rows(Vp, GNA_D, lim_rows_k - ik * 16);
          } else {
            Vtile.load(Vp, GNA_D);
          }
          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }
  }

  metal::vec<float, kRowsPT> reciprocal_sum;
  for (short i = 0; i < kRowsPT; ++i) {
    reciprocal_sum[i] = sum_score[i] > 0.0f ? 1.0f / sum_score[i] : 0.0f;
  }
  Otile.template row_bin_op<MulOp>(reciprocal_sum);

  device GNAT* O_q = O_head + q_tile * GNA_BQ * GNA_D + tm * GNA_D;
  if (is_last_q) {
    if (lim_rows_q <= 0) {
      return;
    }
    Otile.store_rows(O_q, GNA_D, lim_rows_q);
  } else {
    Otile.store(O_q, GNA_D);
  }
)MSL";
}

std::string kernel_name(
    mlx::core::Dtype dtype,
    int N,
    int D,
    int Hq,
    int Hk,
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
    int BQ,
    int BK,
    int WM,
    bool precompute_range,
    int swizzle_log) {
  std::ostringstream os;
  os << "mfa_gna_nax_"
     << (dtype == mlx::core::float16 ? "f16" : "bf16")
     << "_n" << N << "_d" << D << "_hq" << Hq << "_hk" << Hk
     << "_s" << dim0 << "x" << dim1 << "x" << dim2
     << "_w" << window0 << "x" << window1 << "x" << window2
     << "_r" << stride0 << "x" << stride1 << "x" << stride2
     << "_bq" << BQ << "_bk" << BK << "_wm" << WM
     << "_pr" << (precompute_range ? 1 : 0)
     << "_swz" << swizzle_log
     << "_sc" << static_cast<int>(std::round(scale * 1000000.0f));
  return os.str();
}

} // namespace

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
    mlx::core::StreamOrDevice s) {
  auto st = mlx::core::to_stream(s);

  if (!device_has_neural_accelerators()) {
    throw std::runtime_error("GNA NAX requires Metal 4 cooperative tensor support");
  }
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument("GNA NAX expects q, k, v with shape [B, H, N, D]");
  }
  if (q.dtype() != k.dtype() || q.dtype() != v.dtype()) {
    throw std::invalid_argument("GNA NAX requires q, k, v to have matching dtypes");
  }
  const std::string dtype = dtype_name(q.dtype());

  const int B = q.shape(0);
  const int Hq = q.shape(1);
  const int N = q.shape(2);
  const int D = q.shape(3);
  const int Hk = k.shape(1);
  if (B <= 0 || Hq <= 0 || Hk <= 0 || N <= 0 || D <= 0) {
    throw std::invalid_argument("GNA NAX dimensions must be positive");
  }
  if (D != 64 && D != 128) {
    throw std::invalid_argument("GNA NAX supports D=64 and D=128 only");
  }
  if (k.shape(0) != B || v.shape(0) != B || k.shape(1) != Hk || v.shape(1) != Hk ||
      k.shape(2) != N || v.shape(2) != N || k.shape(3) != D || v.shape(3) != D) {
    throw std::invalid_argument("GNA NAX requires q/k/v to share batch, sequence, and head dimension");
  }
  if (Hq % Hk != 0) {
    throw std::invalid_argument("GNA NAX requires Hq to be divisible by Hk for GQA");
  }
  if (dim0 <= 0 || dim1 <= 0 || dim2 <= 0 || window0 <= 0 || window1 <= 0 ||
      window2 <= 0 || stride0 <= 0 || stride1 <= 0 || stride2 <= 0) {
    throw std::invalid_argument("GNA NAX dimensions, windows, and strides must be positive");
  }
  const long long expected_N = static_cast<long long>(dim0) * dim1 * dim2;
  if (expected_N != N) {
    throw std::invalid_argument("GNA NAX requires N == dim0 * dim1 * dim2");
  }
  if (!std::isfinite(scale) || scale <= 0.0f) {
    throw std::invalid_argument("GNA NAX scale must be finite and positive");
  }

  // Measured on M5/macOS 27 beta for this expert-only kernel: D=128 uses the
  // dense V6 NAX tile through N=4096, but N>=8192 benefits from the finer
  // BQ32/WM2 GNA skip granularity. Env overrides remain first-class for sweeps.
  const bool d128_large_n = (D == 128 && N >= 8192);
  const int default_BQ = (D == 64 || d128_large_n) ? 32 : 64;
  const int default_BK = 32;
  const int default_WM = (D == 64 || d128_large_n) ? 2 : 4;
  const int BQ = env_int_or_default("MFA_GNA_NAX_BQ", default_BQ);
  const int BK = env_int_or_default("MFA_GNA_NAX_BK", default_BK);
  const int WM = env_int_or_default("MFA_GNA_NAX_WM", default_WM);
  const bool precompute_range =
      env_int_or_default("MFA_GNA_NAX_PRECOMPUTE_RANGE", 0) != 0;
  const int swizzle_log =
      env_nonnegative_int_or_default("MFA_GNA_NAX_SWIZZLE_LOG", 0);
  if (BQ % (WM * 16) != 0) {
    throw std::invalid_argument("GNA NAX requires BQ to be divisible by WM*16");
  }
  if (BK % 32 != 0) {
    throw std::invalid_argument("GNA NAX requires BK to be divisible by 32");
  }
  if (D % 32 != 0) {
    throw std::invalid_argument("GNA NAX requires D to be divisible by 32");
  }

  auto q_contig = mlx::core::contiguous(q, false, st);
  auto k_contig = mlx::core::contiguous(k, false, st);
  auto v_contig = mlx::core::contiguous(v, false, st);

  auto header = gna_nax_header(
      dtype, B, Hq, Hk, N, D, dim0, dim1, dim2, window0, window1, window2,
      stride0, stride1, stride2, scale, BQ, BK, WM, precompute_range, swizzle_log);
  auto source = gna_nax_source();
  auto name = kernel_name(
      q.dtype(), N, D, Hq, Hk, dim0, dim1, dim2, window0, window1, window2,
      stride0, stride1, stride2, scale, BQ, BK, WM, precompute_range, swizzle_log);

  auto kernel = mlx::core::fast::metal_kernel(
      name,
      {"Q", "K", "V"},
      {"O"},
      source,
      header,
      true,
      false);

  return kernel(
      {q_contig, k_contig, v_contig},
      {q.shape()},
      {q.dtype()},
      {static_cast<uint32_t>(((N + BQ - 1) / BQ * WM * 32) << swizzle_log),
       static_cast<uint32_t>((Hq + (1 << swizzle_log) - 1) >> swizzle_log),
       static_cast<uint32_t>(B)},
      {static_cast<uint32_t>(WM * 32), 1, 1},
      {},
      std::nullopt,
      false,
      st)[0];
}

} // namespace mlx_mfa
