/// Sprint B Sparse Attention NAX - implementation.
///
/// Phase 1.1: per-thread-Q-row FA-2 kernel with block-mask skip. FP16, BT in
/// {16, 32}, 2-D mask, causal=false.
///
/// Phase 1.2 (this version):
///   - dtype: float16 + bfloat16
///   - block_tile: 16, 32, 64
///   - mask ndim: 2 (NQ, NK), 3 (Hq, NQ, NK), 4 (B, Hq, NQ, NK)
///   - causal: false + true (per-tile skip future + within-tile triangular)
///   - asymmetric qL != kL (cross-attention) supported
///
/// NOTE (audit B1, 2026-06-17; naming refactor 2026-07-10): the matmul2d
/// inner-GEMM swap lives in `sparse_kernel_source_v6nax` (this file, the
/// `BaseNAXFrag::mma` cooperative-tensor kernel).  The scalar generator is a
/// per-thread-Q-row fallback, not the historical dense V1 lineage.

#include "mfa_sparse_attention.hpp"

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/utils.h>

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <cmath>  // std::isfinite (raw-host scale-finite parity)
#include <string>
#include <vector>

namespace mlx_mfa {

namespace {

// One host/source truth for the routed V6NAX sparse tile.  The MSL generator
// and both host dispatch sites below consume these constants.
constexpr int kV6NAXSparseBQ = 32;
constexpr int kV6NAXSparseBK = 32;
constexpr int kV6NAXSparseWM = 2;

const std::string SPARSE_SCALAR_HEADER = R"(
#include <metal_stdlib>
using namespace metal;
)";

// bfloat is provided directly by <metal_stdlib> on Apple Silicon Metal SDK;
// no separate <metal_bf16> header. Kept as named alias for clarity.
const std::string& SPARSE_SCALAR_HEADER_BF16 = SPARSE_SCALAR_HEADER;

// ---------------------------------------------------------------------------
// V6NAX sparse header prefix — minimum #include set for Apple NAX helpers.
// The full header = prefix + V6NAX_SPARSE_APPLE_HELPERS_MSL.
// ---------------------------------------------------------------------------
const std::string V6NAX_SPARSE_HEADER_PREFIX = R"(
// MFA_REQUIRE_MSL4
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

)";


// ---------------------------------------------------------------------------
// V6NAX sparse MSL: Apple helpers (NAXFrag, NAXTile, operator structs) -
// verbatim lift
// from csrc/mfa/v6_nax/NAAttentionKernel.cpp:2336-2724 (389 LOC).
// ---------------------------------------------------------------------------
const std::string V6NAX_SPARSE_APPLE_HELPERS_MSL = R"V6SPARSE_APPLE(
// === defines.h ===
#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// === utils/type_traits.h (subset) ===
#pragma METAL internals : enable
namespace metal {
template <typename T> struct is_empty : metal::bool_constant<__is_empty(T)> {};
template <typename T> struct pointer_element {};
template <typename T> struct pointer_element<thread T*> { using type = remove_cv_t<T>; };
template <typename T> struct pointer_element<device T*> { using type = remove_cv_t<T>; };
template <typename T> struct pointer_element<constant T*> { using type = remove_cv_t<T>; };
template <typename T> struct pointer_element<threadgroup T*> { using type = remove_cv_t<T>; };
template <typename T> using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;
}
#pragma METAL internals : disable

// === utils/integral_constant.h (subset) ===
#pragma METAL internals : enable
namespace mlx { namespace steel {
template <typename T, T v> struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;
  METAL_FUNC constexpr operator value_type() const noexcept { return value; }
};
template <bool B> using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;
template <int val> using Int = integral_constant<int, val>;
#define integral_const_binop(__op__, __operator__)          \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto __operator__(                   \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv __op__ uv;                      \
    return integral_constant<decltype(res), res>{};         \
  }
integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);
template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    constexpr auto idx = Int<start>{};
    f(idx);
    const_for_loop<start + step, stop, step, F>(f);
  }
}
#undef integral_const_binop
}}
#pragma METAL internals : disable

// === Limits<> provided by MLX kernels/utils.h auto-include ===

// === Apple steel/attn/nax.h — BaseNAXFrag + NAXTile (verbatim, nax.h:27-817) ===
namespace mlx { namespace steel {

struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;
  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;
  STEEL_CONST short kElemRows = 2;
  STEEL_CONST short kElemCols = 4;
  STEEL_CONST short kElemRowsJump = 8;

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
    return short2{fn, fm};
  }

  template <typename T, typename SrcPtrType, typename StrX, typename StrY,
            typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst, SrcPtrType src,
      StrX str_x, StrY str_y, OffX off_x = {}, OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j) * str_y]);
        }
      }
    }
  }

  template <typename T, typename SrcPtrType, typename StrX, typename StrY,
            typename LimX, typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst, SrcPtrType src,
      StrX str_x, StrY str_y, LimX lim_x, OffX off_x = {}, OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j)]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j) * str_y]);
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <typename T, typename DstPtrType, typename StrX, typename StrY,
            typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src, DstPtrType dst,
      StrX str_x, StrY str_y, OffX off_x = {}, OffY off_y = {}) {
    using U = metal::pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + (c + j) * str_y] = static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <typename T, typename DstPtrType, typename StrX, typename StrY,
            typename LimX, typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_rows(
      const thread dtype_frag_t<T>& src, DstPtrType dst,
      StrX str_x, StrY str_y, LimX lim_x, OffX off_x = {}, OffY off_y = {}) {
    using U = metal::pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + (c + j) * str_y] = static_cast<U>(src[i * kElemCols + j]);
          }
        }
      }
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals, thread T* reduced_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      T thr_reduce = Op::apply(
          Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
          Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));
      T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = Op::apply(thr_reduce, qgr_reduce);
      T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);
      reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_bin_op(
      thread dtype_frag_t<T>& inp_vals, thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] = Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }

  template <typename CType, typename AType, typename BType,
            bool transpose_a = false, bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cn0, thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A, metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0, const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, transpose_a, transpose_b, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), CType>();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) ct_a[i] = A[i];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = Bn0[i];
      ct_b[kElemsPerFrag + i] = Bn1[i];
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cn0[i];
      ct_c[kElemsPerFrag + i] = Cn1[i];
    }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cn0[i] = ct_c[i];
      Cn1[i] = ct_c[kElemsPerFrag + i];
    }
  }
};

template <typename T, short kTileRows_, short kTileCols_, class NAXFrag_ = BaseNAXFrag>
struct NAXTile {
  using NAXFrag_t = NAXFrag_;
  using elem_type = T;
  STEEL_CONST short kFragRows = NAXFrag_t::kFragRows;
  STEEL_CONST short kFragCols = NAXFrag_t::kFragCols;
  STEEL_CONST short kElemsPerFrag = NAXFrag_t::kElemsPerFrag;
  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;
  STEEL_CONST short kRows = kTileRows * kFragRows;
  STEEL_CONST short kCols = kTileCols * kFragCols;
  STEEL_CONST short kNumFrags = kTileRows * kTileCols;
  STEEL_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;
  STEEL_CONST short kFragThrRows = NAXFrag_t::kElemRows;
  STEEL_CONST short kFragThrCols = NAXFrag_t::kElemCols;
  STEEL_CONST short kFragRowsJump = NAXFrag_t::kElemRowsJump;
  STEEL_CONST short kRowsPerThread = kTileRows * NAXFrag_t::kElemRows;
  STEEL_CONST short kColsPerThread = kTileCols * NAXFrag_t::kElemCols;

  typedef typename NAXFrag_t::template dtype_frag_t<T> frag_type;
  frag_type val_frags[kNumFrags];

  METAL_FUNC NAXTile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) val_frags[i] = frag_type(0);
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }
  METAL_FUNC constexpr const thread frag_type& frag_at(const short i, const short j) const {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread metal::vec<T, kRowsPerThread>& vals) const {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_reduce<Op>(frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread metal::vec<T, kRowsPerThread>& vals) {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_bin_op<Op>(frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(frag_at(idx_row.value, idx_col.value), src, ld, Int<1>{},
                        idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(frag_at(idx_row.value, idx_col.value), dst, ld, Int<1>{},
                         idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load_rows(const device U* src, const int ld, const short n_rows) {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_rows(frag_at(idx_row.value, idx_col.value), src, ld, Int<1>{},
                             n_rows, idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_rows(device U* dst, const int ld, const short n_rows) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_rows(frag_at(idx_row.value, idx_col.value), dst, ld, Int<1>{},
                              n_rows, idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }
};

}}  // namespace mlx::steel

// === Operator structs (steel_attention_nax.h:31-71) ===
struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};
struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};
struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x * y; }
};
struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); }
};

)V6SPARSE_APPLE";

// ---------------------------------------------------------------------------
// V6NAX sparse kernel body - adapted from V6NAX forward with sparse modifications:
//   - block_mask skip in outer loop (sparse iteration)
//   - K/V base + per-iteration jump pointers (NOT linear advance)
//   - is_last_q/is_last_k remainder branches dropped (front-end enforces div)
//   - all-False row -> zero output (v2.34.0 contract via rcp branch)
// Placeholders replaced per-call:
//   MASK_OFFSET_EXPR    - block_mask offset for (b, hq, qi) per mask_ndim
//   CAUSAL_WITHIN_TILE_MASK - within-tile triangular mask when causal=true
// ---------------------------------------------------------------------------
const std::string V6NAX_SPARSE_KERNEL_BODY_MSL = R"V6SPARSE_BODY(

// V6NAX sparse kernel body — adapted from V6NAX forward (NAAttentionKernel.cpp:2767-2960).
// Key sparse modifications:
//   - block_mask scan in outer loop: skip kb when mask[qi, kb] == false
//   - K/V base pointers saved; per-iteration jump via kb offset (no linear advance)
//   - is_last_q/is_last_k remainder logic dropped (front-end enforces qL/kL % BT == 0)
//   - All-False row zero-output preservation (v2.34.0 contract)

// Per-shape-emitted constants (constexpr at JIT time):
//   cB, cHq, cHk, cQL, cKL, cD, cNQ, cNK, cGQA, V6NAX_SPARSE_BQ, V6NAX_SPARSE_BK,
//   V6NAX_SPARSE_BD, V6NAX_SPARSE_WM, V6NAX_SPARSE_TQ, V6NAX_SPARSE_TD,
//   V6NAX_SPARSE_TK, V6NAX_SPARSE_DOT_SCALE

ulong3 tidl{threadgroup_position_in_grid.x,
            threadgroup_position_in_grid.y,
            threadgroup_position_in_grid.z};

// BHND strides (Q seq stride = D, head stride = qL * D, batch stride = Hq * qL * D)
const long Q_seq_stride = cD;
const long K_seq_stride = cD;
const long V_seq_stride = cD;
const long O_seq_stride = cD;
const long Q_head_stride = cQL * cD;
const long K_head_stride = cKL * cD;
const long V_head_stride = cKL * cD;
const long O_head_stride = cQL * cD;
const long Q_batch_stride = cHq * cQL * cD;
const long K_batch_stride = cHk * cKL * cD;
const long V_batch_stride = cHk * cKL * cD;
const long O_batch_stride = cHq * cQL * cD;

Q += tidl.z * Q_batch_stride
   + tidl.y * Q_head_stride
   + tidl.x * V6NAX_SPARSE_BQ * Q_seq_stride;
ulong kv_head_idx = ulong(tidl.y) / ulong(cGQA);
device const T* K_base = K + tidl.z * K_batch_stride + kv_head_idx * K_head_stride;
device const T* V_base = V + tidl.z * V_batch_stride + kv_head_idx * V_head_stride;
O += tidl.z * O_batch_stride
   + tidl.y * O_head_stride
   + tidl.x * V6NAX_SPARSE_BQ * O_seq_stride;

const uint qi = uint(tidl.x);

// Mask offset for this (b, hq, qi). Emitted per mask_ndim by source-gen.
// 2-D: mask_qrow_base = block_mask + qi * cNK
// 3-D: mask_qrow_base = block_mask + tidl.y * cNQ * cNK + qi * cNK
// 4-D: mask_qrow_base = block_mask + tidl.z * cHq * cNQ * cNK
//                                  + tidl.y * cNQ * cNK + qi * cNK
device const bool* mask_qrow_base = block_mask + MASK_OFFSET_EXPR;

const float scale2 = V6NAX_SPARSE_DOT_SCALE;

using otile_t = NAXTile<float, V6NAX_SPARSE_TQ, V6NAX_SPARSE_TD>;
otile_t Otile;
Otile.clear();

const short tm = 16 * V6NAX_SPARSE_TQ * simdgroup_index_in_threadgroup;
Q += tm * int(Q_seq_stride);

constexpr short kRowsPT = otile_t::kRowsPerThread;
metal::vec<float, kRowsPT> max_score;
metal::vec<float, kRowsPT> sum_score{0};
STEEL_PRAGMA_UNROLL
for (short i = 0; i < kRowsPT; ++i) {
  max_score[i] = Limits<float>::finite_min;
}

const int kb_lim = cNK;

// Sparse K-loop: iterate all K-blocks, skip masked.
for (int kb = 0; kb < kb_lim; kb++) {
  if (!(MASK_ACTIVE_PREDICATE)) continue;
  CAUSAL_INTER_TILE_SKIP

  // Per-iteration K and V pointers — JUMP via kb (no incremental advance).
  device const T* K_kb = K_base + kb * V6NAX_SPARSE_BK * int(K_seq_stride);
  device const T* V_kb = V_base + kb * V6NAX_SPARSE_BK * int(V_seq_stride);

  using stile_t = NAXTile<float, V6NAX_SPARSE_TQ, V6NAX_SPARSE_TK>;
  stile_t Stile;
  Stile.clear();

  // QK matmul (Apple lines 206-246; remainder branches dropped)
  STEEL_PRAGMA_UNROLL
  for (short iq = 0; iq < V6NAX_SPARSE_TQ; iq++) {
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < V6NAX_SPARSE_TK; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < V6NAX_SPARSE_TD; id++) {
        NAXTile<T, 1, 1> Qtile;
        NAXTile<T, 2, 1> Ktile;
        const int Q_load_off = iq * 16 * int(Q_seq_stride) + id * 16;
        const int K_load_off = ik * 16 * int(K_seq_stride) + id * 16;
        Qtile.load(Q + Q_load_off, int(Q_seq_stride));
        Ktile.load(K_kb + K_load_off, int(K_seq_stride));
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

  // Scale (Apple lines 248-252)
  STEEL_PRAGMA_UNROLL
  for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
    Stile.elems()[ii] *= scale2;
  }

  CAUSAL_WITHIN_TILE_MASK

  // Online softmax (Apple lines 380-409)
  metal::vec<float, kRowsPT> new_max;
  metal::vec<float, kRowsPT> factor;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
  Stile.template row_reduce<MaxOp>(new_max);
  Stile.template row_bin_op<ExpSubOp>(new_max);
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    factor[i] = fast::exp2(max_score[i] - new_max[i]);
    max_score[i] = new_max[i];
  }
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    sum_score[i] = sum_score[i] * factor[i];
  }
  Stile.template row_reduce<SumOp>(sum_score);
  Otile.template row_bin_op<MulOp>(factor);

  simdgroup_barrier(mem_flags::mem_none);

  // PV matmul (Apple lines 417-452)
  STEEL_PRAGMA_UNROLL
  for (short iq = 0; iq < V6NAX_SPARSE_TQ; iq++) {
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < V6NAX_SPARSE_TD; id += 2) {
      if (V6NAX_SPARSE_BD == 128) {
        if (id == 4) {
          threadgroup_barrier(mem_flags::mem_none);
        }
      }
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < V6NAX_SPARSE_TK; ik++) {
        NAXTile<T, 1, 2> Vtile;
        const int V_load_off = ik * 16 * int(V_seq_stride) + id * 16;
        Vtile.load(V_kb + V_load_off, int(V_seq_stride));
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

// Normalize + store (Apple lines 461-481)
threadgroup_barrier(mem_flags::mem_none);
metal::vec<float, kRowsPT> rcp;
STEEL_PRAGMA_UNROLL
for (short i = 0; i < kRowsPT; ++i) {
  // All-False row preservation (v2.34.0 contract): if sum_score == 0 → output 0
  rcp[i] = (sum_score[i] > 0.f) ? (1.f / sum_score[i]) : 0.f;
}
Otile.template row_bin_op<MulOp>(rcp);
O += tm * int(O_seq_stride);
Otile.store(O, int(O_seq_stride));

)V6SPARSE_BODY";

// Full V6NAX sparse header (prefix + Apple helpers, concatenated at static init).
const std::string V6NAX_SPARSE_HEADER =
    V6NAX_SPARSE_HEADER_PREFIX + V6NAX_SPARSE_APPLE_HELPERS_MSL;


// Scalar fallback JIT shader source. Per-thread Q-row processing inside a
// per-(b, hq, q_tile) threadgroup. Online softmax. Block mask scanned at
// K-tile granularity.
//
// dtype_str: "half" or "bfloat" - Metal Shading Language scalar type
// mask_ndim: 2 (NQ, NK), 3 (Hq, NQ, NK), 4 (B, Hq, NQ, NK)
// causal: when true emit per-tile-skip + within-tile triangular mask
std::string sparse_scalar_fallback_source(int B, int Hq, int Hk, int qL, int kL, int D,
                                           int BT, int NQ, int NK, float scale,
                                           const std::string& dtype_str,
                                           int mask_ndim, bool causal,
                                           bool emit_lse = false) {
  int gqa_factor = Hq / Hk;
  // Offset expression into block_mask for this (b, hq, q_tile).
  std::string mask_base_expr;
  if (mask_ndim == 2) {
    mask_base_expr = "block_mask + q_tile * cNK";
  } else if (mask_ndim == 3) {
    mask_base_expr = "block_mask + hq * cNQ * cNK + q_tile * cNK";
  } else {  // 4
    mask_base_expr = "block_mask + b * cHq * cNQ * cNK + hq * cNQ * cNK + q_tile * cNK";
  }

  std::ostringstream os;
  os << "    // Per-shape constants compiled in\n"
     << "    constexpr uint cB        = " << B << ";\n"
     << "    constexpr uint cHq       = " << Hq << ";\n"
     << "    constexpr uint cHk       = " << Hk << ";\n"
     << "    constexpr uint cQL       = " << qL << ";\n"
     << "    constexpr uint cKL       = " << kL << ";\n"
     << "    constexpr uint cD        = " << D << ";\n"
     << "    constexpr uint cBT       = " << BT << ";\n"
     << "    constexpr uint cNQ       = " << NQ << ";\n"
     << "    constexpr uint cNK       = " << NK << ";\n"
     << "    constexpr uint cGQA      = " << gqa_factor << ";\n"
     << "    constexpr float cSCALE   = " << scale << "f;\n"
     << "    constexpr float NEG_INF  = -1e30f;\n"
     << "\n"
     << "    const uint row_in_tile = thread_position_in_threadgroup.x;\n"
     << "    const uint hq          = threadgroup_position_in_grid.y;\n"
     << "    const uint bq_slot     = threadgroup_position_in_grid.z;\n"
     << "    const uint b           = bq_slot / cNQ;\n"
     << "    const uint q_tile      = bq_slot - b * cNQ;\n"
     << "    if (b >= cB || hq >= cHq || q_tile >= cNQ) return;\n"
     << "    if (row_in_tile >= cBT) return;\n"
     << "\n"
     << "    const uint hk          = hq / cGQA;\n"
     << "    const uint q_abs       = q_tile * cBT + row_in_tile;\n"
     << "\n"
     << "    // Base pointers (BHND row-major)\n"
     << "    device const " << dtype_str << "* Q_base = Q\n"
     << "        + b  * cHq * cQL * cD\n"
     << "        + hq *       cQL * cD\n"
     << "        + q_abs *          cD;\n"
     << "    device const " << dtype_str << "* K_b_hk = K\n"
     << "        + b  * cHk * cKL * cD\n"
     << "        + hk *       cKL * cD;\n"
     << "    device const " << dtype_str << "* V_b_hk = V\n"
     << "        + b  * cHk * cKL * cD\n"
     << "        + hk *       cKL * cD;\n"
     << "    device const bool* M_base = " << mask_base_expr << ";\n"
     << "    device "       << dtype_str << "* O_base = O\n"
     << "        + b  * cHq * cQL * cD\n"
     << "        + hq *       cQL * cD\n"
     << "        + q_abs *          cD;\n";
  if (emit_lse) {
    os << "    device float* L_base = L\n"
       << "        + b  * cHq * cQL\n"
       << "        + hq *       cQL\n"
       << "        + q_abs;\n";
  }
  os << "\n"
     << "    // Per-thread state\n"
     << "    float q_vec[cD];\n"
     << "    #pragma clang loop unroll(full)\n"
     << "    for (uint d = 0; d < cD; ++d) q_vec[d] = float(Q_base[d]);\n"
     << "\n"
     << "    float m_run = NEG_INF;\n"
     << "    float l_run = 0.0f;\n"
     << "    float o_vec[cD];\n"
     << "    #pragma clang loop unroll(full)\n"
     << "    for (uint d = 0; d < cD; ++d) o_vec[d] = 0.0f;\n"
     << "\n"
     << "    // Block-mask scan + tile inner loop\n";
  if (causal) {
    os << "    // Causal: skip future tiles entirely (k_tile > q_tile).\n"
       << "    for (uint k_tile = 0; k_tile <= q_tile; ++k_tile) {\n";
  } else {
    os << "    for (uint k_tile = 0; k_tile < cNK; ++k_tile) {\n";
  }
  os << "        if (!M_base[k_tile]) continue;\n"
     << "\n"
     << "        // (1) Score row: s[k_col] = (q . K[k_tile*BT + k_col]) * scale\n"
     << "        float s[cBT];\n"
     << "        float m_tile = NEG_INF;\n"
     << "        #pragma clang loop unroll(full)\n"
     << "        for (uint kc = 0; kc < cBT; ++kc) {\n"
     << "            device const " << dtype_str
              << "* K_row = K_b_hk + (k_tile * cBT + kc) * cD;\n"
     << "            float acc = 0.0f;\n"
     << "            #pragma clang loop unroll(full)\n"
     << "            for (uint d = 0; d < cD; ++d) {\n"
     << "                acc += q_vec[d] * float(K_row[d]);\n"
     << "            }\n"
     << "            acc *= cSCALE;\n";
  if (causal) {
    // Within-tile triangular: on the diagonal tile (k_tile == q_tile), mask
    // any (row_in_tile, kc) where kc > row_in_tile (future-of-this-row).
    os << "            if (k_tile == q_tile && kc > row_in_tile) acc = NEG_INF;\n";
  }
  os << "            s[kc] = acc;\n"
     << "            m_tile = max(m_tile, acc);\n"
     << "        }\n"
     << "\n"
     << "        // (2) FA-2 online softmax update\n"
     << "        float m_new = max(m_run, m_tile);\n"
     << "        float corr  = exp(m_run - m_new);\n"
     << "        float l_tile = 0.0f;\n"
     << "        float p[cBT];\n"
     << "        #pragma clang loop unroll(full)\n"
     << "        for (uint kc = 0; kc < cBT; ++kc) {\n"
     << "            float pv = exp(s[kc] - m_new);\n"
     << "            p[kc] = pv;\n"
     << "            l_tile += pv;\n"
     << "        }\n"
     << "        l_run = l_run * corr + l_tile;\n"
     << "\n"
     << "        // (3) O accumulator update: O = O*corr + p @ V[k_tile,:]\n"
     << "        #pragma clang loop unroll(full)\n"
     << "        for (uint d = 0; d < cD; ++d) {\n"
     << "            float acc = o_vec[d] * corr;\n"
     << "            #pragma clang loop unroll(full)\n"
     << "            for (uint kc = 0; kc < cBT; ++kc) {\n"
     << "                acc += p[kc] * float(V_b_hk[(k_tile * cBT + kc) * cD + d]);\n"
     << "            }\n"
     << "            o_vec[d] = acc;\n"
     << "        }\n"
     << "\n"
     << "        m_run = m_new;\n"
     << "    }\n"
     << "\n"
     << "    // (4) Finalize: divide by l_run; write back. All-False row -> zero.\n"
     << "    if (l_run <= 0.0f) {\n"
     << "        #pragma clang loop unroll(full)\n"
     << "        for (uint d = 0; d < cD; ++d) O_base[d] = " << dtype_str << "(0.0f);\n"
     << "    } else {\n"
     << "        float inv_l = 1.0f / l_run;\n"
     << "        #pragma clang loop unroll(full)\n"
     << "        for (uint d = 0; d < cD; ++d) O_base[d] = " << dtype_str << "(o_vec[d] * inv_l);\n"
     << "    }\n";
  if (emit_lse) {
    // v2.50 Prompt 5c Section A.1 — write per-row sparse-LSE (natural-log).
    // L[r] = m_run + log(l_run) for active rows; -INFINITY for all-False rows
    // (sentinel; consumer must handle).  Required by V6NAX backward sparse to
    // consume same convention.
    os << "    // Write sparse-LSE (natural log).  All-False rows → -INFINITY.\n"
       << "    if (l_run <= 0.0f) {\n"
       << "        L_base[0] = -INFINITY;\n"
       << "    } else {\n"
       << "        L_base[0] = m_run + log(l_run);\n"
       << "    }\n";
  }
  return os.str();
}

// ---------------------------------------------------------------------------
// V6NAX sparse source-gen — single-kernel cooperative-tensor inner-GEMM.
//
// Architecture per docs/lcsa-nax/lcsa-nax-design.md §13:
//   - Single kernel iterates K-blocks 0..NK-1 and skips masked via
//     `if (!block_mask[qi*NK+kb]) continue;` (uniform across SG → zero divergence)
//   - NAXFrag::mma cooperative-tensor inner-GEMMs (V6NAX forward pattern adapted)
//   - Per-SG Q-row partition with kU=16, BQ=BK=32, WM=2
//
// Eligibility (enforced in sparse_attention_forward before this is called):
//   D ∈ {64, 128}, BT == 32, dtype ∈ {float16, bfloat16}; causal is supported
//   by a dense V6NAX within-tile mask port plus a causal K-block skip.
//
// Apple helpers (NAXFrag, NAXTile, operator structs) embedded via
// V6NAX_SPARSE_APPLE_HELPERS_MSL; kernel body via
// V6NAX_SPARSE_KERNEL_BODY_MSL with three placeholders substituted per call:
//   MASK_OFFSET_EXPR      — per mask_ndim (2/3/4)
//   CAUSAL_INTER_TILE_SKIP — causal K-block skip before QK, empty when non-causal
//   CAUSAL_WITHIN_TILE_MASK — dense V6NAX causal mask ported to sparse NAXTile layout
// ---------------------------------------------------------------------------
std::string sparse_kernel_source_v6nax(int B, int Hq, int Hk, int qL, int kL, int D,
                                        int BT, int NQ, int NK, float scale,
                                        const std::string& dtype_str,
                                        int mask_ndim, bool causal,
                                        bool emit_lse = false,
                                        bool structured_window_probe = false,
                                        int structured_window_size = 0) {
  (void)BT;  // V6NAX sparse uses BQ=BK=32 internally; eligibility ensures BT==32
  // V6NAX sparse tile shape (DC3) — BQ=BK=32, WM=2 for both D=64 and D=128. This tile is
  // STRUCTURALLY PINNED, not a tunable (sparse-NAX-autotune, M5 Max, 2026-06-18):
  //   * BQ=BK=32 is fixed by mask-block faithfulness — one 32-wide Q/K block
  //     maps to exactly one block_mask entry (eligibility forces BT==32).
  //   * WM=2 is fixed by the cooperative-tensor inner GEMM + cross-SG reduction,
  //     which assume exactly 2 simdgroups. The divisibility rule BQ%(WM*16)==0
  //     also admits WM=1, but a measured WM=1 sweep was BOTH ~3-4x slower at high
  //     density AND incorrect (err up to 3.0e-2 > the fp16 floor) — a silent
  //     Category-A wrong-but-finite result. So WM is not exposed as a knob.
  // See .doc-archive/docs/lcsa-nax/sparse-nax-autotune-results.md (journal).
  const int V6NAX_SPARSE_BQ = kV6NAXSparseBQ;
  const int V6NAX_SPARSE_BK = kV6NAXSparseBK;
  const int V6NAX_SPARSE_BD = D;
  const int V6NAX_SPARSE_WM = kV6NAXSparseWM;
  const int kU = 16;
  const int V6NAX_SPARSE_TQ = V6NAX_SPARSE_BQ / (V6NAX_SPARSE_WM * kU);  // = 1
  const int V6NAX_SPARSE_TD = V6NAX_SPARSE_BD / kU;  // 4 for D=64, 8 for D=128
  const int V6NAX_SPARSE_TK = V6NAX_SPARSE_BK / kU;  // = 2
  const float dot_scale_log2e = scale * 1.4426950408889634f;
  const int gqa_factor = Hq / Hk;

  std::string mask_active_predicate = "mask_qrow_base[kb]";
  if (structured_window_probe) {
    // This is the same block predicate emitted by make_sliding_window_mask
    // when BQ=BK=32. The bool input remains bound so this probe changes only
    // eligibility computation, not the NAX tile body.
    mask_active_predicate =
        "((kb * V6NAX_SPARSE_BK + V6NAX_SPARSE_BK - 1) >= "
        "(int(qi) * V6NAX_SPARSE_BQ + V6NAX_SPARSE_BQ / 2 - " +
        std::to_string(structured_window_size) + ") && "
        "(kb * V6NAX_SPARSE_BK <= "
        "int(qi) * V6NAX_SPARSE_BQ + V6NAX_SPARSE_BQ / 2 + " +
        std::to_string(structured_window_size) + "))";
    if (causal) {
      mask_active_predicate = "(" + mask_active_predicate +
          " && kb * V6NAX_SPARSE_BK <= "
          "(int(qi) + 1) * V6NAX_SPARSE_BQ - 1)";
    }
  }

  // Mask offset expression (DC2)
  std::string mask_offset_expr;
  if (mask_ndim == 2) {
    mask_offset_expr = "qi * cNK";
  } else if (mask_ndim == 3) {
    mask_offset_expr = "tidl.y * cNQ * cNK + qi * cNK";
  } else {
    mask_offset_expr = "tidl.z * cHq * cNQ * cNK + "
                       "tidl.y * cNQ * cNK + qi * cNK";
  }

  std::string causal_skip;
  std::string causal_block;
  if (causal) {
    causal_skip = R"CAUSAL_SKIP(
  {
    // Dense V6NAX causal K-loop bound port: skip K-blocks whose first key
    // column is beyond the last query row in this Q block.
    const int causal_column_offset = int(cKL) - int(cQL);
    const int causal_last_column_limit =
        int(qi) * V6NAX_SPARSE_BQ + V6NAX_SPARSE_BQ - 1 + causal_column_offset;
    if (kb * V6NAX_SPARSE_BK > causal_last_column_limit) continue;
  }
)CAUSAL_SKIP";
    causal_block = R"CAUSAL_MASK(
  // Dense V6NAX causal mask port (NAAttentionKernel.cpp NAXTile form):
  // earlier tiles are fully below the diagonal, future tiles were skipped above,
  // so only diagonal-overlapping tiles need per-element triangular masking.
  {
    const int causal_column_offset = int(cKL) - int(cQL);
    const int causal_first_column_limit =
        int(qi) * V6NAX_SPARSE_BQ + causal_column_offset;
    if (kb * V6NAX_SPARSE_BK + V6NAX_SPARSE_BK - 1 > causal_first_column_limit) {
      constexpr auto neg_inf = Limits<float>::finite_min;
      const short2 sc_c = stile_t::NAXFrag_t::get_coord();
      const short sn_c = sc_c.x;
      const short sm_c = sc_c.y;
      const int base_row = int(qi) * V6NAX_SPARSE_BQ + tm;
      const int base_col = kb * V6NAX_SPARSE_BK;
      STEEL_PRAGMA_UNROLL
      for (short iq_c = 0; iq_c < V6NAX_SPARSE_TQ; iq_c++) {
        STEEL_PRAGMA_UNROLL
        for (short ik_c = 0; ik_c < V6NAX_SPARSE_TK; ik_c++) {
          thread auto& fg = Stile.frag_at(iq_c, ik_c);
          STEEL_PRAGMA_UNROLL
          for (short ii_c = 0; ii_c < stile_t::kFragThrRows; ii_c++) {
            STEEL_PRAGMA_UNROLL
            for (short jj_c = 0; jj_c < stile_t::kFragThrCols; jj_c++) {
              const int row = base_row + iq_c * 16
                            + ii_c * stile_t::kFragRowsJump + sm_c;
              const int col = base_col + ik_c * 16 + jj_c + sn_c;
              const int causal_column_limit = row + causal_column_offset;
              const auto loc = ii_c * stile_t::kFragThrCols + jj_c;
              fg[loc] = (col > causal_column_limit) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }
  }
)CAUSAL_MASK";
  } else {
    causal_skip = "";
    causal_block = "// causal=false";
  }

  std::ostringstream ss;
  // Per-shape #defines (compile-time constants for the kernel)
  ss << "using T = " << dtype_str << ";\n";
  ss << "using namespace mlx::steel;\n";
  ss << "\n";
  ss << "#define V6NAX_SPARSE_BQ " << V6NAX_SPARSE_BQ << "\n";
  ss << "#define V6NAX_SPARSE_BK " << V6NAX_SPARSE_BK << "\n";
  ss << "#define V6NAX_SPARSE_BD " << V6NAX_SPARSE_BD << "\n";
  ss << "#define V6NAX_SPARSE_WM " << V6NAX_SPARSE_WM << "\n";
  ss << "#define V6NAX_SPARSE_TQ " << V6NAX_SPARSE_TQ << "\n";
  ss << "#define V6NAX_SPARSE_TD " << V6NAX_SPARSE_TD << "\n";
  ss << "#define V6NAX_SPARSE_TK " << V6NAX_SPARSE_TK << "\n";
  ss << "#define V6NAX_SPARSE_DOT_SCALE " << dot_scale_log2e << "f\n";
  ss << "\n";
  ss << "constexpr int cB   = " << B << ";\n";
  ss << "constexpr int cHq  = " << Hq << ";\n";
  ss << "constexpr int cHk  = " << Hk << ";\n";
  ss << "constexpr int cQL  = " << qL << ";\n";
  ss << "constexpr int cKL  = " << kL << ";\n";
  ss << "constexpr int cD   = " << D << ";\n";
  ss << "constexpr int cNQ  = " << NQ << ";\n";
  ss << "constexpr int cNK  = " << NK << ";\n";
  ss << "constexpr int cGQA = " << gqa_factor << ";\n";
  ss << "\n";

  // Substitute placeholders in V6NAX_SPARSE_KERNEL_BODY_MSL
  std::string body = V6NAX_SPARSE_KERNEL_BODY_MSL;
  auto replace_all = [](std::string& s, const std::string& from,
                         const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
      s.replace(pos, from.length(), to);
      pos += to.length();
    }
  };
  replace_all(body, "MASK_OFFSET_EXPR", mask_offset_expr);
  replace_all(body, "MASK_ACTIVE_PREDICATE", mask_active_predicate);
  replace_all(body, "CAUSAL_INTER_TILE_SKIP", causal_skip);
  replace_all(body, "CAUSAL_WITHIN_TILE_MASK", causal_block);
  if (emit_lse) {
    const std::string store_marker = "Otile.store(O, int(O_seq_stride));";
    const std::string lse_store = R"LSE(

// Optional sparse LSE store.  The online softmax is in log2-domain; the
// backward contract consumes natural-log LSE, so convert once at the store.
{
  const short2 lse_coord = otile_t::NAXFrag_t::get_coord();
  if (lse_coord.x == 0) {
    device float* L_base = L
        + tidl.z * cHq * cQL
        + tidl.y * cQL
        + tidl.x * V6NAX_SPARSE_BQ;
    const short lse_row_base = short(tm + lse_coord.y);
    constexpr float ln2 = 0.69314718055994530942f;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const short row = lse_row_base + i * 8;
      L_base[row] = (sum_score[i] > 0.f)
          ? (max_score[i] + metal::log2(sum_score[i])) * ln2
          : -INFINITY;
    }
  }
}
)LSE";
    const size_t marker_pos = body.find(store_marker);
    if (marker_pos == std::string::npos ||
        body.find(store_marker, marker_pos + store_marker.size()) != std::string::npos) {
      throw std::runtime_error(
          "V6NAX sparse source: LSE store marker must occur exactly once");
    }
    body.insert(marker_pos + store_marker.size(), lse_store);
  }
  ss << body;

  return ss.str();
}

enum class SparseKernelPath {
  ScalarFallback,
  V6NAXSparse,
};

bool parse_sparse_kernel_path(const std::string& value, SparseKernelPath& path) {
  if (value == "v1" || value == "scalar_fallback" ||
      value == "sparse_scalar_fallback") {
    path = SparseKernelPath::ScalarFallback;
    return true;
  }
  if (value == "v2" || value == "v6nax_sparse" ||
      value == "v6_nax_sparse" || value == "v6-nax-sparse" ||
      value == "v6nax") {
    path = SparseKernelPath::V6NAXSparse;
    return true;
  }
  return false;
}

const char* sparse_kernel_path_cache_suffix(SparseKernelPath path) {
  switch (path) {
    case SparseKernelPath::V6NAXSparse:
      return "v6nax_sparse";
    case SparseKernelPath::ScalarFallback:
    default:
      return "scalar_fallback";
  }
}

// Read MFA_LCSA_KERNEL_VERSION env var. Legacy public aliases "v1"/"v2" remain
// valid and map to scalar fallback / V6NAX sparse respectively.
//
// v2.35.0 SHIP_OPT_IN (Section D §4-validated verdict):
//   - The cooperative-tensor path wins vs SDPA+bias and vs scalar fallback
//     across ALL tested shapes + densities (2.22-11.57× vs SDPA,
//     8.54-63.59× vs scalar fallback).
//   - Cross-session range > 10% on 5/7 shapes (3 HIGH, 2 BOUNDARY) due to
//     A/B/A pattern's scalar middle round disturbing NAX cache state.
//   - Strict criterion yields wins=2/7 → OPT-IN at the time.
//   - Historical env values are preserved: v1=scalar_fallback, v2=v6nax_sparse.
//
// See docs/lcsa-nax/lcsa-nax-coop-rewrite-results.md for full data.
SparseKernelPath read_sparse_kernel_path_env() {
  const char* env = std::getenv("MFA_LCSA_KERNEL_VERSION");
  if (env == nullptr) return SparseKernelPath::ScalarFallback;  // SHIP_OPT_IN default
  std::string v(env);
  SparseKernelPath path;
  if (parse_sparse_kernel_path(v, path)) return path;
  // "auto" or unrecognized: fall back to scalar default.
  return SparseKernelPath::ScalarFallback;
}

}  // namespace

mlx::core::array sparse_attention_forward(
    const mlx::core::array& Q,
    const mlx::core::array& K,
    const mlx::core::array& V,
    const mlx::core::array& block_mask,
    int block_tile,
    bool causal,
    float scale,
    const std::string& kernel_version,
    bool structured_window_probe,
    int structured_window_size) {
  // Sanity asserts
  if (Q.ndim() != 4 || K.ndim() != 4 || V.ndim() != 4) {
    throw std::runtime_error("sparse_attention: Q, K, V must be 4-D (B, H, L, D)");
  }
  // Phase 1.2: float16 + bfloat16
  bool is_f16 = (Q.dtype() == mlx::core::float16);
  bool is_bf16 = (Q.dtype() == mlx::core::bfloat16);
  if (!is_f16 && !is_bf16) {
    throw std::runtime_error("sparse_attention: dtype must be float16 or bfloat16");
  }
  if (K.dtype() != Q.dtype() || V.dtype() != Q.dtype()) {
    throw std::runtime_error("sparse_attention: Q, K, V dtype must match");
  }
  if (block_mask.dtype() != mlx::core::bool_) {
    throw std::runtime_error("sparse_attention: block_mask must be bool");
  }
  int mask_ndim = static_cast<int>(block_mask.ndim());
  if (mask_ndim != 2 && mask_ndim != 3 && mask_ndim != 4) {
    throw std::runtime_error("sparse_attention: block_mask.ndim must be 2, 3, or 4");
  }
  if (block_tile != 16 && block_tile != 32 && block_tile != 64) {
    throw std::runtime_error("sparse_attention: block_tile must be 16, 32, or 64");
  }
  int B  = static_cast<int>(Q.shape(0));
  int Hq = static_cast<int>(Q.shape(1));
  int qL = static_cast<int>(Q.shape(2));
  int D  = static_cast<int>(Q.shape(3));
  int Hk = static_cast<int>(K.shape(1));
  int kL = static_cast<int>(K.shape(2));
  if (static_cast<int>(K.shape(0)) != B || static_cast<int>(V.shape(0)) != B) {
    throw std::runtime_error("sparse_attention: batch dim mismatch");
  }
  if (static_cast<int>(K.shape(3)) != D || static_cast<int>(V.shape(3)) != D) {
    throw std::runtime_error("sparse_attention: head_dim mismatch");
  }
  if (static_cast<int>(V.shape(1)) != Hk || static_cast<int>(V.shape(2)) != kL) {
    throw std::runtime_error("sparse_attention: K, V shape mismatch");
  }
  if (Hq % Hk != 0) {
    throw std::runtime_error("sparse_attention: Hq must be multiple of Hk (GQA)");
  }
  if (qL % block_tile != 0 || kL % block_tile != 0) {
    throw std::runtime_error("sparse_attention: qL, kL must be multiples of block_tile");
  }
  if (D != 64 && D != 128) {
    throw std::runtime_error("sparse_attention: head_dim must be 64 or 128");
  }
  if (structured_window_probe) {
    if (block_tile != 32) {
      throw std::runtime_error(
          "sparse structured-window probe requires block_tile=32");
    }
    if (structured_window_size < 0) {
      throw std::runtime_error(
          "sparse structured-window probe requires window_size>=0");
    }
  }
  int NQ = qL / block_tile;
  int NK = kL / block_tile;
  // Mask shape check per ndim
  if (mask_ndim == 2) {
    if (static_cast<int>(block_mask.shape(0)) != NQ ||
        static_cast<int>(block_mask.shape(1)) != NK) {
      throw std::runtime_error("sparse_attention: 2-D block_mask shape != (NQ, NK)");
    }
  } else if (mask_ndim == 3) {
    if (static_cast<int>(block_mask.shape(0)) != Hq ||
        static_cast<int>(block_mask.shape(1)) != NQ ||
        static_cast<int>(block_mask.shape(2)) != NK) {
      throw std::runtime_error("sparse_attention: 3-D block_mask shape != (Hq, NQ, NK)");
    }
  } else {  // 4
    if (static_cast<int>(block_mask.shape(0)) != B ||
        static_cast<int>(block_mask.shape(1)) != Hq ||
        static_cast<int>(block_mask.shape(2)) != NQ ||
        static_cast<int>(block_mask.shape(3)) != NK) {
      throw std::runtime_error("sparse_attention: 4-D block_mask shape != (B, Hq, NQ, NK)");
    }
  }
  if (causal && qL != kL) {
    throw std::runtime_error("sparse_attention: causal=true requires qL == kL");
  }
  if (!std::isfinite(scale) || scale <= 0.0f) {
    // raw-parity sweep: nan/inf scale was baked into the Metal source as "nan"/"inf"
    // text -> cryptic 'undeclared identifier nanf' compile error, not a clear raise.
    throw std::runtime_error("sparse_attention: scale must be finite and > 0");
  }
  // Address-space precondition (Phase 1.1 carry-over)
  long long mask_bytes = 1LL;
  for (int i = 0; i < mask_ndim; ++i) mask_bytes *= static_cast<long long>(block_mask.shape(i));
  if (mask_bytes < 4096) {
    throw std::runtime_error(
        "sparse_attention: mask total bytes < 4096 (use larger qL, kL, "
        "or higher mask ndim). MLX inlines small buffers in constant "
        "address space; the JIT kernel emits device-qualified pointer.");
  }

  std::string dtype_str = is_f16 ? "half" : "bfloat";

  // Dispatch resolution order:
  //   1. Explicit `kernel_version` param (Python-side shape-aware
  //      decide_auto_version() in mlx_mfa.lcsa_nax). Highest priority.
  //   2. MFA_LCSA_KERNEL_VERSION env var (legacy aliases v1/v2 preserved).
  //   3. Internal default (scalar fallback) per read_sparse_kernel_path_env().
  SparseKernelPath path;
  if (!kernel_version.empty() && parse_sparse_kernel_path(kernel_version, path)) {
    // explicit override parsed
  } else {
    path = read_sparse_kernel_path_env();
  }

  // V6NAX sparse eligibility (per design §13 + DC3):
  //   D ∈ {64, 128}, BT == 32, dtype ∈ {float16, bfloat16},
  //   mask_ndim ∈ {2,3,4}
  // bf16 enabled 2026-06-18: the V6NAX sparse cooperative-tensor `mma` is templated on the
  // input dtype with fp32 accumulation (CType=float), and the generator already
  // emits `using T = bfloat`, so bf16 is just T=bfloat — no kernel change needed.
  // The prior `is_f16`-only gate was a Phase-1.2 deferral that silently routed bf16
  // to the scalar fallback (up to ~50x slower than plain SDPA-with-mask).
  // If V6NAX sparse was requested but not eligible, transparently fall back to scalar.
  bool v6nax_sparse_eligible = (path == SparseKernelPath::V6NAXSparse)
      && (D == 64 || D == 128)
      && (block_tile == 32)
      && (is_f16 || is_bf16);
  if (path == SparseKernelPath::V6NAXSparse && !v6nax_sparse_eligible) {
    // Silent fallback to scalar — keeps user code working when NAX sparse doesn't apply.
    path = SparseKernelPath::ScalarFallback;
  }

  const bool use_v6nax_sparse = (path == SparseKernelPath::V6NAXSparse);
  std::string name = "sparse_attn_" + std::string(sparse_kernel_path_cache_suffix(path)) +
      "_" + dtype_str + "_" +
      std::to_string(B) + "_" + std::to_string(Hq) + "_" + std::to_string(Hk) +
      "_" + std::to_string(qL) + "_" + std::to_string(kL) + "_" +
      std::to_string(D) + "_BT" + std::to_string(block_tile) +
      "_M" + std::to_string(mask_ndim) +
      (causal ? "_c" : "_nc");
  if (structured_window_probe) {
    name += "_structured_window_W" + std::to_string(structured_window_size);
  }

  std::string source;
  if (use_v6nax_sparse) {
    source = sparse_kernel_source_v6nax(B, Hq, Hk, qL, kL, D, block_tile, NQ, NK,
                                        scale, dtype_str, mask_ndim, causal,
                                        /*emit_lse=*/false,
                                        structured_window_probe,
                                        structured_window_size);
  } else {
    source = sparse_scalar_fallback_source(B, Hq, Hk, qL, kL, D, block_tile, NQ, NK,
                                            scale, dtype_str, mask_ndim, causal);
  }

  std::string header;
  if (use_v6nax_sparse) {
    header = V6NAX_SPARSE_HEADER;
  } else {
    header = is_bf16 ? SPARSE_SCALAR_HEADER_BF16 : SPARSE_SCALAR_HEADER;
  }

  auto kernel = mlx::core::fast::metal_kernel(
      name,
      {"Q", "K", "V", "block_mask"},
      {"O"},
      source,
      header,
      /*ensure_row_contiguous=*/true,
      /*atomic_outputs=*/false);

  // Dispatch grid + threadgroup size differ between scalar fallback and V6NAX sparse.
  // Scalar: TG = (BT, 1, 1), grid = (BT, Hq, B*NQ) (one Q row per thread).
  // V6NAX: TG = (WM*32, 1, 1), grid = (NQ*WM*32, Hq, B) (one TG per Q-block).
  std::tuple<int, int, int> grid, tg;
  if (use_v6nax_sparse) {
    const int tg_threads = kV6NAXSparseWM * 32;  // 64 threads per TG
    grid = std::make_tuple(NQ * tg_threads, Hq, B);
    tg = std::make_tuple(tg_threads, 1, 1);
  } else {
    grid = std::make_tuple(block_tile, Hq, B * NQ);
    tg = std::make_tuple(block_tile, 1, 1);
  }

  auto outs = kernel(
      {Q, K, V, block_mask},
      {mlx::core::Shape{B, Hq, qL, D}},
      {Q.dtype()},
      grid,
      tg,
      {},
      std::nullopt,
      false,
      mlx::core::default_stream(mlx::core::Device::gpu));
  return outs[0];
}


// =============================================================================
// v2.50 Prompt 5c Section A.1 — sparse_attention_forward returning (O, L)
//
// Sparse forward returning (O, L).  BT=32, D={64,128}, f16/bf16 uses the
// V6NAX cooperative-tensor kernel with an optional natural-log LSE store;
// other accepted block tiles retain the scalar implementation.  L is
// (B, Hq, qL) FP32 and all-False rows write -INFINITY.
// =============================================================================
std::pair<mlx::core::array, mlx::core::array>
sparse_attention_forward_with_lse(
    const mlx::core::array& Q,
    const mlx::core::array& K,
    const mlx::core::array& V,
    const mlx::core::array& block_mask,
    int block_tile,
    bool causal,
    float scale) {
  // Sanity asserts (identical to sparse_attention_forward; small inline
  // duplication is preferable to factor-out for clarity).
  if (Q.ndim() != 4 || K.ndim() != 4 || V.ndim() != 4) {
    throw std::runtime_error("sparse_attention: Q, K, V must be 4-D (B, H, L, D)");
  }
  bool is_f16 = (Q.dtype() == mlx::core::float16);
  bool is_bf16 = (Q.dtype() == mlx::core::bfloat16);
  if (!is_f16 && !is_bf16) {
    throw std::runtime_error("sparse_attention: dtype must be float16 or bfloat16");
  }
  if (K.dtype() != Q.dtype() || V.dtype() != Q.dtype()) {
    throw std::runtime_error("sparse_attention: Q, K, V dtype must match");
  }
  if (block_mask.dtype() != mlx::core::bool_) {
    throw std::runtime_error("sparse_attention: block_mask must be bool");
  }
  int mask_ndim = static_cast<int>(block_mask.ndim());
  if (mask_ndim != 2 && mask_ndim != 3 && mask_ndim != 4) {
    throw std::runtime_error("sparse_attention: block_mask.ndim must be 2, 3, or 4");
  }
  if (block_tile != 16 && block_tile != 32 && block_tile != 64) {
    throw std::runtime_error("sparse_attention: block_tile must be 16, 32, or 64");
  }
  int B  = static_cast<int>(Q.shape(0));
  int Hq = static_cast<int>(Q.shape(1));
  int qL = static_cast<int>(Q.shape(2));
  int D  = static_cast<int>(Q.shape(3));
  int Hk = static_cast<int>(K.shape(1));
  int kL = static_cast<int>(K.shape(2));
  // volet K1 (R10): batch/K↔V/head_dim mutual checks — the with_lse variant
  // omitted these (R9 sparse_attention_forward has them). Rule 8.
  if (static_cast<int>(K.shape(0)) != B || static_cast<int>(V.shape(0)) != B)
    throw std::runtime_error("sparse_attention: batch dim mismatch");
  if (static_cast<int>(K.shape(3)) != D || static_cast<int>(V.shape(3)) != D)
    throw std::runtime_error("sparse_attention: head_dim mismatch");
  if (static_cast<int>(V.shape(1)) != Hk || static_cast<int>(V.shape(2)) != kL)
    throw std::runtime_error("sparse_attention: K, V shape mismatch");
  if (Hq % Hk != 0) {
    throw std::runtime_error("sparse_attention: Hq must be multiple of Hk (GQA)");
  }
  if (qL % block_tile != 0 || kL % block_tile != 0) {
    throw std::runtime_error("sparse_attention: qL, kL must be multiples of block_tile");
  }
  if (D != 64 && D != 128) {
    throw std::runtime_error("sparse_attention: head_dim must be 64 or 128");
  }
  // SCALE value-guard (raw-parity sweep — the with_lse sibling had NONE; nan/inf/0/neg
  // scale baked into Metal source -> compile error, not a clear raise). Mirror the
  // non-LSE sibling, extended for finiteness.
  if (!std::isfinite(scale) || scale <= 0.0f) {
    throw std::runtime_error("sparse_attention: scale must be finite and > 0");
  }
  int NQ = qL / block_tile;
  int NK = kL / block_tile;
  // Mask shape check per ndim (subset of full validation; full validation
  // happens via sparse_attention_forward when called).
  if (mask_ndim == 2) {
    if (static_cast<int>(block_mask.shape(0)) != NQ ||
        static_cast<int>(block_mask.shape(1)) != NK) {
      throw std::runtime_error("sparse_attention: 2-D block_mask shape != (NQ, NK)");
    }
  } else if (mask_ndim == 3) {
    if (static_cast<int>(block_mask.shape(0)) != Hq ||
        static_cast<int>(block_mask.shape(1)) != NQ ||
        static_cast<int>(block_mask.shape(2)) != NK) {
      throw std::runtime_error(
          "sparse_attention: 3-D block_mask shape != (Hq, NQ, NK)");
    }
  } else {
    if (static_cast<int>(block_mask.shape(0)) != B ||
        static_cast<int>(block_mask.shape(1)) != Hq ||
        static_cast<int>(block_mask.shape(2)) != NQ ||
        static_cast<int>(block_mask.shape(3)) != NK) {
      throw std::runtime_error(
          "sparse_attention: 4-D block_mask shape != (B, Hq, NQ, NK)");
    }
  }
  if (causal && qL != kL) {
    throw std::runtime_error(
        "sparse_attention: causal requires qL == kL");
  }
  long long mask_bytes = 1LL;
  for (int i = 0; i < mask_ndim; ++i)
    mask_bytes *= static_cast<long long>(block_mask.shape(i));
  if (mask_bytes < 4096) {
    throw std::runtime_error(
        "sparse_attention: mask total bytes < 4096 (use larger qL, kL, "
        "or higher mask ndim).");
  }

  std::string dtype_str = is_f16 ? "half" : "bfloat";

  // V6NAX sparse LSE path: keep the same eligibility as the production
  // forward (D={64,128}, BT=32, f16/bf16) and add only the optional L store.
  // Non-LSE callers never enter this function, so their source and pipeline
  // key remain unchanged.
  const bool use_v6nax_lse =
      block_tile == 32 && (D == 64 || D == 128) && (is_f16 || is_bf16);
  if (use_v6nax_lse) {
    std::string name = "sparse_attn_v6nax_sparse_lse_" + dtype_str + "_" +
        std::to_string(B) + "_" + std::to_string(Hq) + "_" + std::to_string(Hk) +
        "_" + std::to_string(qL) + "_" + std::to_string(kL) + "_" +
        std::to_string(D) + "_BT" + std::to_string(block_tile) +
        "_M" + std::to_string(mask_ndim) + (causal ? "_c" : "_nc");

    std::string source = sparse_kernel_source_v6nax(
        B, Hq, Hk, qL, kL, D, block_tile, NQ, NK, scale,
        dtype_str, mask_ndim, causal, /*emit_lse=*/true);
    auto kernel = mlx::core::fast::metal_kernel(
        name,
        {"Q", "K", "V", "block_mask"},
        {"O", "L"},
        source,
        V6NAX_SPARSE_HEADER,
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false);

    constexpr int tg_threads = kV6NAXSparseWM * 32;
    std::tuple<int, int, int> grid = std::make_tuple(NQ * tg_threads, Hq, B);
    std::tuple<int, int, int> tg = std::make_tuple(tg_threads, 1, 1);
    auto outs = kernel(
        {Q, K, V, block_mask},
        {mlx::core::Shape{B, Hq, qL, D}, mlx::core::Shape{B, Hq, qL}},
        {Q.dtype(), mlx::core::float32},
        grid,
        tg,
        {},
        std::nullopt,
        false,
        mlx::core::default_stream(mlx::core::Device::gpu));
    return {outs[0], outs[1]};
  }

  std::string name = "sparse_attn_scalar_fallback_lse_" + dtype_str + "_" +
      std::to_string(B) + "_" + std::to_string(Hq) + "_" + std::to_string(Hk) +
      "_" + std::to_string(qL) + "_" + std::to_string(kL) + "_" +
      std::to_string(D) + "_BT" + std::to_string(block_tile) +
      "_M" + std::to_string(mask_ndim) +
      (causal ? "_c" : "_nc");

  std::string source = sparse_scalar_fallback_source(B, Hq, Hk, qL, kL, D, block_tile,
                                                      NQ, NK, scale, dtype_str,
                                                      mask_ndim, causal,
                                                      /*emit_lse=*/true);
  std::string header = is_bf16 ? SPARSE_SCALAR_HEADER_BF16 : SPARSE_SCALAR_HEADER;

  auto kernel = mlx::core::fast::metal_kernel(
      name,
      {"Q", "K", "V", "block_mask"},
      {"O", "L"},  // dual output
      source,
      header,
      /*ensure_row_contiguous=*/true,
      /*atomic_outputs=*/false);

  std::tuple<int, int, int> grid = std::make_tuple(block_tile, Hq, B * NQ);
  std::tuple<int, int, int> tg = std::make_tuple(block_tile, 1, 1);

  auto outs = kernel(
      {Q, K, V, block_mask},
      {mlx::core::Shape{B, Hq, qL, D}, mlx::core::Shape{B, Hq, qL}},
      {Q.dtype(), mlx::core::float32},  // O dtype = Q; L dtype = fp32
      grid,
      tg,
      {},
      std::nullopt,
      false,
      mlx::core::default_stream(mlx::core::Device::gpu));
  return {outs[0], outs[1]};
}

}  // namespace mlx_mfa
