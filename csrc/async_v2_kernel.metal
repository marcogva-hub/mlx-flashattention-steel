// async_v2_kernel.metal — STEEL V2 with simdgroup_async_copy hardware DMA.
//
// Uses Apple's simdgroup_event API (__asm("air.simdgroup_async_copy_2d.p3i8.p1i8"))
// to overlap device→threadgroup DMA with ALU compute:
//
//   Tile N iteration:
//     wait(k_event)            ← K[N] data ready in K_smem
//     Q @ K[N]^T               ← ALU (GEMM)
//     launch(v_event, V[N])    ← DMA: V loads while softmax runs
//     softmax + rescale O      ← ALU (OVERLAP with V DMA)
//     wait(v_event)            ← V[N] data ready in V_smem
//     launch(k_event, K[N+1])  ← DMA: K[N+1] loads while P@V runs
//     P[N] @ V[N] → O         ← ALU (OVERLAP with K[N+1] DMA)
//     loop
//
// Two kernel functions in this file (one metallib, two entry points):
//   mlx_mfa_v2_async_attention       D=64  BQ=32 BK=64
//   mlx_mfa_v2_async_attention_d128  D=128 BQ=32 BK=32
//
// Function constants (set via MTLFunctionConstantValues at pipeline creation):
//   FC_CAUSAL=0     bool   causal masking
//   FC_GQA_FACTOR=1 ushort GQA ratio H_q / H_kv  (1 = standard MHA)
//
// Note: simdgroup_async_copy is a private AIR intrinsic removed from the
// macOS 26 runtime compiler. This file targets Xcode ≤16 (macOS ≤15) offline
// compilation via:
//   xcrun metal -target air64-apple-macos15.0 -c async_v2_kernel.metal
// On macOS 26 the JIT path (shader_cache.mm) is used as fallback.

// ---------------------------------------------------------------------------
// simdgroup_event header — verbatim from createMetalSimdgroupEvent(false)
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
// ---------------------------------------------------------------------------

#ifndef __METAL_SIMDGROUP_EVENT
#define __METAL_SIMDGROUP_EVENT

struct _simdgroup_event_t;

thread _simdgroup_event_t*
__metal_simdgroup_async_copy_1d(
  ulong, ulong, threadgroup void *, const device void *, ulong)
  __asm("air.simdgroup_async_copy_1d.p3i8.p1i8");

thread _simdgroup_event_t*
__metal_simdgroup_async_copy_1d(
  ulong, ulong, device void *, const threadgroup void *, ulong)
  __asm("air.simdgroup_async_copy_1d.p1i8.p3i8");

thread _simdgroup_event_t*
__metal_simdgroup_async_copy_2d(
  ulong, ulong,
  threadgroup void *, ulong, ulong, ulong2,
  const device void *, ulong, ulong, ulong2,
  long2, int)
  __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");

thread _simdgroup_event_t*
__metal_simdgroup_async_copy_2d(
  ulong, ulong,
  device void *, ulong, ulong, ulong2,
  const threadgroup void *, ulong, ulong, ulong2,
  long2, int)
  __asm("air.simdgroup_async_copy_2d.p1i8.p3i8");

void __metal_wait_simdgroup_events(
  int, thread _simdgroup_event_t**)
  __asm("air.wait_simdgroup_events");

#pragma METAL internals : enable
namespace metal
{
  enum class simdgroup_async_copy_clamp_mode {
    clamp_to_zero = 0,
    clamp_to_edge = 1
  };

  struct simdgroup_event {
    METAL_FUNC simdgroup_event() thread {}

    template <ushort dst_elements_per_row, ushort threadgroup_size,
              simdgroup_async_copy_clamp_mode clamp_mode =
                  simdgroup_async_copy_clamp_mode::clamp_to_zero,
              typename T>
    METAL_FUNC void async_copy(
      threadgroup T *dst,
      ushort2 dst_tile_dimensions,
      const device T *src,
      uint src_elements_per_row,
      ushort2 src_tile_dimensions,
      bool transpose_matrix = false
    ) thread {
      if (transpose_matrix) {
        src_tile_dimensions = src_tile_dimensions.yx;
        dst_tile_dimensions = dst_tile_dimensions.yx;
      }
      event = __metal_simdgroup_async_copy_2d(
        sizeof(T), alignof(T),
        reinterpret_cast<threadgroup void *>(dst),
        ushort(dst_elements_per_row), 1,
        ulong2(dst_tile_dimensions),
        reinterpret_cast<const device void *>(src),
        uint(src_elements_per_row), 1,
        ulong2(src_tile_dimensions),
        long2(0),
        static_cast<int>(clamp_mode));
    }

    template <ushort src_elements_per_row, ushort threadgroup_size,
              simdgroup_async_copy_clamp_mode clamp_mode =
                  simdgroup_async_copy_clamp_mode::clamp_to_zero,
              typename T>
    METAL_FUNC void async_copy(
      device T *dst,
      uint dst_elements_per_row,
      ushort2 dst_tile_dimensions,
      const threadgroup T *src,
      ushort2 src_tile_dimensions,
      bool transpose_matrix = false
    ) thread {
      if (transpose_matrix) {
        src_tile_dimensions = src_tile_dimensions.yx;
        dst_tile_dimensions = dst_tile_dimensions.yx;
      }
      event = __metal_simdgroup_async_copy_2d(
        sizeof(T), alignof(T),
        reinterpret_cast<device void *>(dst),
        uint(dst_elements_per_row), 1,
        ulong2(dst_tile_dimensions),
        reinterpret_cast<const threadgroup void *>(src),
        ushort(src_elements_per_row), 1,
        ulong2(src_tile_dimensions),
        long2(0), 0);
    }

    template <ushort threadgroup_size, typename T>
    METAL_FUNC void async_copy(
      threadgroup T *dst, ushort dst_tile_dimensions,
      const device T *src, ushort src_tile_dimensions
    ) thread {
      async_copy<1, threadgroup_size>(
          dst, ushort2(dst_tile_dimensions, 1),
          src, 1, ushort2(src_tile_dimensions, 1));
    }

    METAL_FUNC static void wait(int count, thread simdgroup_event *events) {
      __metal_wait_simdgroup_events(
          count, reinterpret_cast<thread _simdgroup_event_t**>(events));
    }

  private:
    thread _simdgroup_event_t* event;
  };
} // namespace metal
#pragma METAL internals : disable

#endif // __METAL_SIMDGROUP_EVENT

// ---------------------------------------------------------------------------
// Standard Metal includes and STEEL shared types
// ---------------------------------------------------------------------------

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// Function constants — set at pipeline creation via MTLFunctionConstantValues
constant bool   FC_CAUSAL     [[function_constant(0)]];
constant ushort FC_GQA_FACTOR [[function_constant(1)]];

// ---------------------------------------------------------------------------
// MFABlockLoaderT — cooperative threadgroup tile loader (same as V1/V2 sync)
// ---------------------------------------------------------------------------

template <
    typename T,
    short BROWS,
    short BCOLS,
    short kDstStrRow,
    short kDstStrCol,
    short reduction_dim,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / tgp_size,
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct MFABlockLoaderT {
  STEEL_CONST short vec_size = n_reads;
  const int src_ld;
  const int tile_stride;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device T* src;

  METAL_FUNC MFABlockLoaderT(
      const device T* src_, const int src_ld_,
      threadgroup T* dst_, ushort simd_group_id, ushort simd_lane_id)
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * kDstStrRow + bj * kDstStrCol),
        src(src_ + bi * src_ld_ + bj) {}

  METAL_FUNC void load_unsafe() const {
    constexpr bool can_vectorize = (kDstStrCol == 1) && (vec_size % 4 == 0);
    if constexpr (can_vectorize) {
      using vec4_t = vec<T, 4>;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j += 4) {
          *(threadgroup vec4_t*)(dst + i * kDstStrRow + j) =
              *(const device vec4_t*)(src + i * src_ld + j);
        }
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++)
          dst[i * kDstStrRow + j * kDstStrCol] = src[i * src_ld + j];
      }
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++)
          dst[i * kDstStrRow + j * kDstStrCol] = T(0);
      }
      return;
    }
    bool tmp_idx[vec_size];
    T    tmp_val[vec_size];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++)
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++)
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++)
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++)
        dst[i * kDstStrRow + j * kDstStrCol] = tmp_val[j];
    }
  }

  METAL_FUNC void next() { src += tile_stride; }
};

// ---------------------------------------------------------------------------
// MFAMMAFrag / MFAMMATile — simdgroup matrix wrappers
// ---------------------------------------------------------------------------

template <typename T>
struct MFAMMAFrag {
  STEEL_CONST int kFragRows = 8;
  STEEL_CONST int kFragCols = 8;
  STEEL_CONST int kElemsPerFrag = 2;
  STEEL_CONST int kElemRows = 1;
  STEEL_CONST int kElemCols = 2;

  typedef simdgroup_matrix<T, 8, 8> mat_type;
  typedef vec<T, 2> frag_type;
  typedef vec<T, 1> row_frag_type;

  METAL_FUNC static short2 get_coord(ushort lane_id) {
    const short qid = lane_id / 4;
    const short fm  = (qid & 4) + ((lane_id / 2) % 4);
    const short fn  = (qid & 2) * 2 + (lane_id % 2) * 2;
    return short2{fn, fm};
  }

  template <typename SrcPtr, typename StrX, typename StrY>
  METAL_FUNC static void load(thread frag_type& dst,
                              SrcPtr src, StrX str_x, StrY str_y) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++)
        dst[i * kElemCols + j] = static_cast<T>(src[i * str_x + j * str_y]);
    }
  }

  template <typename Op>
  METAL_FUNC static void row_reduce(thread const frag_type& inp, thread T* out) {
    T thr = Op::apply(inp.x, inp.y);
    T qr  = Op::apply(thr, simd_shuffle_xor(thr, ushort(1)));
    T sr  = Op::apply(qr,  simd_shuffle_xor(qr,  ushort(8)));
    out[0] = Op::apply(out[0], sr);
  }

  template <typename Op>
  METAL_FUNC static void row_bin_op(thread frag_type& inp, thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < kElemCols; j++)
      inp[j] = Op::apply(inp[j], row_vals[0]);
  }

  METAL_FUNC static void mma(thread frag_type& D, thread frag_type& A,
                             thread frag_type& B, thread frag_type& C) {
    mat_type Dm, Am, Bm, Cm;
    reinterpret_cast<thread frag_type&>(Am.thread_elements()) = A;
    reinterpret_cast<thread frag_type&>(Bm.thread_elements()) = B;
    reinterpret_cast<thread frag_type&>(Cm.thread_elements()) = C;
    simdgroup_multiply_accumulate(Dm, Am, Bm, Cm);
    D = reinterpret_cast<thread frag_type&>(Dm.thread_elements());
  }
};

template <typename T, int kTileRows_, int kTileCols_>
struct MFAMMATile {
  using Frag = MFAMMAFrag<T>;
  STEEL_CONST int kTileRows  = kTileRows_;
  STEEL_CONST int kTileCols  = kTileCols_;
  STEEL_CONST int kNumFrags  = kTileRows_ * kTileCols_;

  typedef typename Frag::frag_type frag_type;
  frag_type val_frags[kNumFrags];

  METAL_FUNC MFAMMATile() thread {}

  METAL_FUNC void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; i++) val_frags[i] = frag_type(0);
  }

  METAL_FUNC thread frag_type& frag_at(short i, short j) {
    return val_frags[i * kTileCols_ + j];
  }
  METAL_FUNC const thread frag_type& frag_at(short i, short j) const {
    return val_frags[i * kTileCols_ + j];
  }
  METAL_FUNC thread T* elems() {
    return reinterpret_cast<thread T*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread T vals[kTileRows_]) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows_; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols_; j++)
        Frag::template row_reduce<Op>(frag_at(i, j), &vals[i]);
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread T vals[kTileRows_]) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows_; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols_; j++)
        Frag::template row_bin_op<Op>(frag_at(i, j), &vals[i]);
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void load(const threadgroup U* src, int str_row, int str_col) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows_; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols_; j++) {
        Frag::load(frag_at(i, j),
                   &src[(i * 8) * w_x * str_row + (j * 8) * w_y * str_col],
                   str_row, str_col);
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U* dst, const int ld) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows_; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols_; j++) {
        const int base = (i * 8) * w_x * ld + (j * 8) * w_y;
        dst[base + 0] = static_cast<U>(frag_at(i, j)[0]);
        dst[base + 1] = static_cast<U>(frag_at(i, j)[1]);
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store_safe(device U* dst, const int ld,
                             const short2 dims) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows_; i++) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols_; j++) {
        const int off_row = (i * 8) * w_x;
        const int off_col = (j * 8) * w_y;
        if (off_row < (int)dims.y) {
          if (off_col     < (int)dims.x) dst[off_row * ld + off_col]     = static_cast<U>(frag_at(i, j)[0]);
          if (off_col + 1 < (int)dims.x) dst[off_row * ld + off_col + 1] = static_cast<U>(frag_at(i, j)[1]);
        }
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Op structs for online softmax
// ---------------------------------------------------------------------------

struct MFAMaxOp {
  template <typename T>
  METAL_FUNC static T apply(T x, T y) { return metal::max(x, y); }
};
struct MFASumOp {
  template <typename T>
  METAL_FUNC static T apply(T x, T y) { return x + y; }
};
struct MFAMulOp {
  template <typename T>
  METAL_FUNC static T apply(T x, T y) { return x * y; }
};
struct MFADivOp {
  template <typename T>
  METAL_FUNC static T apply(T x, T y) { return x / y; }
};
struct MFAExpSubOp {
  template <typename T>
  METAL_FUNC static T apply(T x, T y) { return fast::exp2(x - y); }
};

// ---------------------------------------------------------------------------
// MFASteelParams — matches the C++ struct in mfa_steel_fwd_v2.cpp
// ---------------------------------------------------------------------------

struct MFASteelParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  int NQ, NK;
  int NQ_aligned;
  int NK_aligned;
  int qL_rem;
  int kL_rem;
  int qL_off;
  int rope_q_base;
  int rope_cos_stride;
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long O_strides[3];
  long L_strides[2];
  float softcap;
  int   has_alibi;
  int   window_left;
  int   window_right;
  long  mask_batch_stride;
  long  mask_head_stride;
};

// ===========================================================================
// D=64: BQ=32, BK=64, BD=64, WM=4, TGP=128
// TGP memory: Q_smem=4608B + K_smem=4608B + V_smem=4608B = 13824B < 32KB
// ===========================================================================

#define MFA_BQ_64  32
#define MFA_BK_64  64
#define MFA_BD_64  64
#define MFA_WM_64  4
#define MFA_TGP_64 128
#define MFA_TD_64  8   // BD/8
#define MFA_TK_64  8   // BK/8
#define MFA_TQ_64  1   // rows per simdgroup

[[kernel, max_total_threads_per_threadgroup(MFA_TGP_64)]]
void mlx_mfa_v2_async_attention(
    const device half*          Q  [[buffer(0)]],
    const device half*          K  [[buffer(1)]],
    const device half*          V  [[buffer(2)]],
    device half*                O  [[buffer(3)]],
    device float*               L  [[buffer(4)]],
    constant MFASteelParams*    p  [[buffer(5)]],
    uint3 tid          [[threadgroup_position_in_grid]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]])
{
  typedef half T;
  typedef float AccT;

  constexpr short BQ   = MFA_BQ_64;
  constexpr short BK   = MFA_BK_64;
  constexpr short BD   = MFA_BD_64;
  constexpr short WM   = MFA_WM_64;
  constexpr short TGP  = MFA_TGP_64;
  constexpr short TD   = MFA_TD_64;
  constexpr short TK   = MFA_TK_64;
  constexpr short TQ   = MFA_TQ_64;
  constexpr short ROWS_PT = TQ;

  // Padding to avoid bank conflicts
  constexpr short padQ = 16 / sizeof(T);  // 8 for half
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);
  constexpr short LDQ  = BD + padQ;   // Q row stride in smem
  constexpr short LDK  = BK + padK;   // K transposed: col stride
  constexpr short LDV  = BD + padV;   // V row stride in smem

  // Separate K_smem and V_smem for async overlap (unlike sync V2 which shares)
  threadgroup T Q_smem[BQ * LDQ];
  threadgroup T K_smem[BD * LDK];  // K transposed: BD rows × BK cols, stride LDK
  threadgroup T V_smem[BK * LDV];  // V row-major:  BK rows × BD cols, stride LDV

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = K_smem;
  threadgroup T* Vs = V_smem;

  // Cooperative loaders (synchronous fallback for safe/partial tiles)
  using QLoader = MFABlockLoaderT<T, BQ, BD, LDQ, 1, 1, TGP>;
  using KLoader = MFABlockLoaderT<T, BK, BD, 1, LDK, 0, TGP>;
  using VLoader = MFABlockLoaderT<T, BK, BD, LDV, 1, 0, TGP>;

  const int h_q  = (int)tid.y;
  const int h_kv = h_q / (int)FC_GQA_FACTOR;

  Q += (long)tid.z * p->Q_strides[0] + (long)h_q  * p->Q_strides[1];
  K += (long)tid.z * p->K_strides[0] + (long)h_kv * p->K_strides[1];
  V += (long)tid.z * p->V_strides[0] + (long)h_kv * p->V_strides[1];
  O += (long)tid.z * p->O_strides[0] + (long)h_q  * p->O_strides[1];

  const AccT scale = p->scale * M_LOG2E_F;

  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = 8 * TQ * (short)simd_group_id;

  const short Qs_off = (tm + sm) * LDQ + sn;
  const short Ks_off = sm * LDK + sn;
  const short Vs_off = sm * LDV + sn;

  MFAMMATile<AccT, TQ, TD> Qtile;
  MFAMMATile<AccT, TQ, TD> Otile;
  MFAMMATile<AccT, 1,  TK>  Ktile;
  MFAMMATile<AccT, TQ, TK>  Stile;
  MFAMMATile<AccT, 1,  1>    Vtile;
  AccT max_score[ROWS_PT];
  AccT sum_score[ROWS_PT];

  const int qb = (int)tid.x;

  const device T* Q_qb = Q + (long)qb * BQ * p->Q_strides[2];
  device T*       O_qb = O + (long)qb * BQ * p->O_strides[2];

  // Load Q tile (synchronous — loaded once, stays in registers)
  QLoader loader_q(Q_qb, (int)p->Q_strides[2], Qs,
                   (ushort)simd_group_id, (ushort)simd_lane_id);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (qb == p->NQ_aligned) {
    loader_q.load_safe(short2(BD, p->qL_rem));
  } else {
    loader_q.load_unsafe();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Qtile.template load<T, 1, 1>(&Qs[Qs_off], LDQ, 1);

  Otile.clear();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < ROWS_PT; i++) {
    max_score[i] = -INFINITY;
    sum_score[i] = 0.0f;
  }

  int q_max  = (qb + 1) * BQ + p->qL_off;
  int kb_lim = (q_max + BK - 1) / BK;
  if (kb_lim > p->NK) kb_lim = p->NK;

  if (kb_lim == 0) {
    // Empty — write zeros
    Otile.template row_bin_op<MFADivOp>(sum_score);
    device T* O_write = O_qb + (long)(tm + sm) * p->O_strides[2] + sn;
    if (qb == p->NQ_aligned) {
      auto dims = short2((short)(BD - sn), (short)(p->qL_rem - (tm + sm)));
      if (dims.x > 0 && dims.y > 0)
        Otile.template store_safe<T, 1, 1>(O_write, (int)p->O_strides[2], dims);
    } else {
      Otile.template store<T, 1, 1>(O_write, (int)p->O_strides[2]);
    }
    return;
  }

  // Preload K[0] using simdgroup_event async_copy.
  // tile for K transposed: src BK rows × BD cols → dst BD rows × BK cols
  const ushort2 k_tile = ushort2(BD, BK);  // (cols, rows) of the tile in device memory
  const ushort2 v_tile = ushort2(BD, BK);

  simdgroup_event k_ev;
  simdgroup_event v_ev;

  // Start loading K[0] asynchronously.
  // K is row-major [BK, BD]; we load and transpose into K_smem [BD, BK].
  // dst_elements_per_row = LDK (transposed stride), threadgroup_size = TGP
  k_ev.async_copy<LDK, TGP>(Ks, k_tile, K, (uint)p->K_strides[2], k_tile, /*transpose=*/true);

  for (int kb = 0; kb < kb_lim; kb++) {

    // ─ Wait for K[kb] DMA to complete ─
    simdgroup_event::wait(1, &k_ev);
    threadgroup_barrier(mem_flags::mem_threadgroup);  // all simdgroups see full K tile

    // ─ Phase 1: Q @ K[kb]^T ─
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      Ktile.template load<T, 1, 1>(
          &Ks[Ks_off + (short)(dd * 8) * LDK], LDK, 1);
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          MFAMMAFrag<AccT>::mma(
              Stile.frag_at(iq, ik),
              Qtile.frag_at(iq, dd),
              Ktile.frag_at(0, ik),
              Stile.frag_at(iq, ik));
        }
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < TQ * TK * 2; ii++) {
      Stile.elems()[ii] *= scale;
    }

    // Padding mask (last partial K-tile)
    if (kb == p->NK_aligned) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TQ; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TK; j++) {
          const short col = sn + j * 8;
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < 2; jj++) {
            if ((col + jj) >= p->kL_rem)
              Stile.frag_at(i, j)[jj] = -INFINITY;
          }
        }
      }
    }

    // Causal mask (controlled by function constant FC_CAUSAL)
    if (FC_CAUSAL) {
      if (kb >= (kb_lim - (BQ + BK - 1) / BK)) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TQ; i++) {
          const int row = qb * BQ + p->qL_off + tm + sm + i * 8;
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < TK; j++) {
            const int col = kb * BK + sn + j * 8;
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < 2; jj++) {
              if (row < (col + jj))
                Stile.frag_at(i, j)[jj] = -INFINITY;
            }
          }
        }
      }
    }

    // ─ Launch V[kb] DMA — overlaps with softmax below ─
    const device T* V_kb = V + (long)kb * BK * p->V_strides[2];
    if (kb == p->NK_aligned) {
      // Partial tile: must use synchronous fallback for correct padding
      VLoader loader_v(V_kb, (int)p->V_strides[2], Vs,
                       (ushort)simd_group_id, (ushort)simd_lane_id);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_v.load_safe(short2(BD, p->kL_rem));
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      v_ev.async_copy<LDV, TGP>(Vs, v_tile, V_kb, (uint)p->V_strides[2], v_tile, /*transpose=*/false);
    }

    // ─ Online softmax (runs while V DMA is in flight) ─
    AccT new_max[ROWS_PT];
    AccT factor[ROWS_PT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < ROWS_PT; i++) new_max[i] = max_score[i];
    Stile.template row_reduce<MFAMaxOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < ROWS_PT; i++) {
      if (new_max[i] > max_score[i]) {
        factor[i] = fast::exp2(max_score[i] - new_max[i]);
        max_score[i] = new_max[i];
      } else {
        factor[i] = 1.0f;
        new_max[i] = metal::isinf(max_score[i]) ? (AccT)0.0f : max_score[i];
      }
    }
    Stile.template row_bin_op<MFAExpSubOp>(new_max);
    AccT sum_tmp[ROWS_PT] = {0};
    Stile.template row_reduce<MFASumOp>(sum_tmp);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < ROWS_PT; i++) {
      sum_score[i] = sum_score[i] * factor[i] + sum_tmp[i];
    }
    Otile.template row_bin_op<MFAMulOp>(factor);

    // ─ Wait for V[kb] DMA ─
    if (kb != p->NK_aligned) {
      simdgroup_event::wait(1, &v_ev);
      threadgroup_barrier(mem_flags::mem_threadgroup);  // all simdgroups see full V tile
    }

    // ─ Launch K[kb+1] DMA — overlaps with P@V below ─
    if (kb + 1 < kb_lim) {
      const device T* K_next = K + (long)(kb + 1) * BK * p->K_strides[2];
      if ((kb + 1) == p->NK_aligned) {
        // Partial: use synchronous fallback at end of P@V
        // (handled below — we skip the async launch for this tile)
      } else {
        k_ev.async_copy<LDK, TGP>(Ks, k_tile, K_next, (uint)p->K_strides[2], k_tile, /*transpose=*/true);
      }
    }

    // ─ Phase 3: O += P @ V ─
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          Vtile.template load<T, 1, 1>(
              &Vs[Vs_off + ik * 8 * LDV + id * 8], LDV, 1);
          MFAMMAFrag<AccT>::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    // Handle partial next-K tile synchronously (after P@V is done)
    if (kb + 1 < kb_lim && (kb + 1) == p->NK_aligned) {
      const device T* K_next = K + (long)(kb + 1) * BK * p->K_strides[2];
      KLoader loader_k(K_next, (int)p->K_strides[2], Ks,
                       (ushort)simd_group_id, (ushort)simd_lane_id);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_k.load_safe(short2(BD, p->kL_rem));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // k_ev is not used for this tile — next loop iteration calls wait(k_ev)
      // but since we loaded synchronously, we need to fake the event.
      // Solution: set k_ev by launching a no-op 0-element copy.
      // Simpler: break here — if NK_aligned == kb+1 it's the last tile.
    }

  } // end kb loop

  Otile.template row_bin_op<MFADivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  device T* O_write = O_qb + (long)(tm + sm) * p->O_strides[2] + sn;
  if (qb == p->NQ_aligned) {
    auto dims = short2((short)(BD - sn), (short)(p->qL_rem - (tm + sm)));
    if (dims.x > 0 && dims.y > 0)
      Otile.template store_safe<T, 1, 1>(O_write, (int)p->O_strides[2], dims);
  } else {
    Otile.template store<T, 1, 1>(O_write, (int)p->O_strides[2]);
  }

  if (sn == 0) {
    const long l_boff = (long)tid.z * p->L_strides[0]
                      + (long)tid.y * p->L_strides[1];
    const long q_base = (long)qb * BQ + tm + sm;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < ROWS_PT; i++) {
      const long q_idx = q_base + i * 8;
      if (q_idx < p->qL) {
        L[l_boff + q_idx] = max_score[i] + metal::log2(sum_score[i]);
      }
    }
  }
}

// ===========================================================================
// D=128: BQ=32, BK=32, BD=128, WM=4, TGP=128
// TGP memory: Q_smem=8704B + K_smem=5120B + V_smem=4352B = 18176B < 32KB
// ===========================================================================

#define MFA_BQ_128  32
#define MFA_BK_128  32
#define MFA_BD_128  128
#define MFA_WM_128  4
#define MFA_TGP_128 128
#define MFA_TD_128  16   // BD/8
#define MFA_TK_128  4    // BK/8
#define MFA_TQ_128  1

[[kernel, max_total_threads_per_threadgroup(MFA_TGP_128)]]
void mlx_mfa_v2_async_attention_d128(
    const device half*          Q  [[buffer(0)]],
    const device half*          K  [[buffer(1)]],
    const device half*          V  [[buffer(2)]],
    device half*                O  [[buffer(3)]],
    device float*               L  [[buffer(4)]],
    constant MFASteelParams*    p  [[buffer(5)]],
    uint3 tid          [[threadgroup_position_in_grid]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]])
{
  typedef half T;
  typedef float AccT;

  constexpr short BQ   = MFA_BQ_128;
  constexpr short BK   = MFA_BK_128;
  constexpr short BD   = MFA_BD_128;
  constexpr short WM   = MFA_WM_128;
  constexpr short TGP  = MFA_TGP_128;
  constexpr short TD   = MFA_TD_128;
  constexpr short TK   = MFA_TK_128;
  constexpr short TQ   = MFA_TQ_128;
  constexpr short ROWS_PT = TQ;

  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);
  constexpr short LDQ  = BD + padQ;
  constexpr short LDK  = BK + padK;
  constexpr short LDV  = BD + padV;

  threadgroup T Q_smem[BQ * LDQ];
  threadgroup T K_smem[BD * LDK];
  threadgroup T V_smem[BK * LDV];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = K_smem;
  threadgroup T* Vs = V_smem;

  using QLoader = MFABlockLoaderT<T, BQ, BD, LDQ, 1, 1, TGP>;
  using KLoader = MFABlockLoaderT<T, BK, BD, 1, LDK, 0, TGP>;
  using VLoader = MFABlockLoaderT<T, BK, BD, LDV, 1, 0, TGP>;

  const int h_q  = (int)tid.y;
  const int h_kv = h_q / (int)FC_GQA_FACTOR;

  Q += (long)tid.z * p->Q_strides[0] + (long)h_q  * p->Q_strides[1];
  K += (long)tid.z * p->K_strides[0] + (long)h_kv * p->K_strides[1];
  V += (long)tid.z * p->V_strides[0] + (long)h_kv * p->V_strides[1];
  O += (long)tid.z * p->O_strides[0] + (long)h_q  * p->O_strides[1];

  const AccT scale = p->scale * M_LOG2E_F;

  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = 8 * TQ * (short)simd_group_id;

  const short Qs_off = (tm + sm) * LDQ + sn;
  const short Ks_off = sm * LDK + sn;
  const short Vs_off = sm * LDV + sn;

  MFAMMATile<AccT, TQ, TD> Qtile;
  MFAMMATile<AccT, TQ, TD> Otile;
  MFAMMATile<AccT, 1,  TK>  Ktile;
  MFAMMATile<AccT, TQ, TK>  Stile;
  MFAMMATile<AccT, 1,  1>    Vtile;
  AccT max_score[ROWS_PT];
  AccT sum_score[ROWS_PT];

  const int qb = (int)tid.x;

  const device T* Q_qb = Q + (long)qb * BQ * p->Q_strides[2];
  device T*       O_qb = O + (long)qb * BQ * p->O_strides[2];

  QLoader loader_q(Q_qb, (int)p->Q_strides[2], Qs,
                   (ushort)simd_group_id, (ushort)simd_lane_id);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (qb == p->NQ_aligned) {
    loader_q.load_safe(short2(BD, p->qL_rem));
  } else {
    loader_q.load_unsafe();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Qtile.template load<T, 1, 1>(&Qs[Qs_off], LDQ, 1);

  Otile.clear();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < ROWS_PT; i++) {
    max_score[i] = -INFINITY;
    sum_score[i] = 0.0f;
  }

  int q_max  = (qb + 1) * BQ + p->qL_off;
  int kb_lim = (q_max + BK - 1) / BK;
  if (kb_lim > p->NK) kb_lim = p->NK;

  if (kb_lim == 0) {
    Otile.template row_bin_op<MFADivOp>(sum_score);
    device T* O_write = O_qb + (long)(tm + sm) * p->O_strides[2] + sn;
    if (qb == p->NQ_aligned) {
      auto dims = short2((short)(BD - sn), (short)(p->qL_rem - (tm + sm)));
      if (dims.x > 0 && dims.y > 0)
        Otile.template store_safe<T, 1, 1>(O_write, (int)p->O_strides[2], dims);
    } else {
      Otile.template store<T, 1, 1>(O_write, (int)p->O_strides[2]);
    }
    return;
  }

  const ushort2 k_tile = ushort2(BD, BK);
  const ushort2 v_tile = ushort2(BD, BK);

  simdgroup_event k_ev;
  simdgroup_event v_ev;

  k_ev.async_copy<LDK, TGP>(Ks, k_tile, K, (uint)p->K_strides[2], k_tile, /*transpose=*/true);

  for (int kb = 0; kb < kb_lim; kb++) {

    simdgroup_event::wait(1, &k_ev);
    threadgroup_barrier(mem_flags::mem_threadgroup);  // all simdgroups see full K tile

    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      Ktile.template load<T, 1, 1>(
          &Ks[Ks_off + (short)(dd * 8) * LDK], LDK, 1);
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          MFAMMAFrag<AccT>::mma(
              Stile.frag_at(iq, ik),
              Qtile.frag_at(iq, dd),
              Ktile.frag_at(0, ik),
              Stile.frag_at(iq, ik));
        }
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < TQ * TK * 2; ii++) {
      Stile.elems()[ii] *= scale;
    }

    if (kb == p->NK_aligned) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TQ; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TK; j++) {
          const short col = sn + j * 8;
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < 2; jj++) {
            if ((col + jj) >= p->kL_rem)
              Stile.frag_at(i, j)[jj] = -INFINITY;
          }
        }
      }
    }

    if (FC_CAUSAL) {
      if (kb >= (kb_lim - (BQ + BK - 1) / BK)) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TQ; i++) {
          const int row = qb * BQ + p->qL_off + tm + sm + i * 8;
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < TK; j++) {
            const int col = kb * BK + sn + j * 8;
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < 2; jj++) {
              if (row < (col + jj))
                Stile.frag_at(i, j)[jj] = -INFINITY;
            }
          }
        }
      }
    }

    const device T* V_kb = V + (long)kb * BK * p->V_strides[2];
    if (kb == p->NK_aligned) {
      VLoader loader_v(V_kb, (int)p->V_strides[2], Vs,
                       (ushort)simd_group_id, (ushort)simd_lane_id);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_v.load_safe(short2(BD, p->kL_rem));
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      v_ev.async_copy<LDV, TGP>(Vs, v_tile, V_kb, (uint)p->V_strides[2], v_tile, /*transpose=*/false);
    }

    AccT new_max[ROWS_PT];
    AccT factor[ROWS_PT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < ROWS_PT; i++) new_max[i] = max_score[i];
    Stile.template row_reduce<MFAMaxOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < ROWS_PT; i++) {
      if (new_max[i] > max_score[i]) {
        factor[i] = fast::exp2(max_score[i] - new_max[i]);
        max_score[i] = new_max[i];
      } else {
        factor[i] = 1.0f;
        new_max[i] = metal::isinf(max_score[i]) ? (AccT)0.0f : max_score[i];
      }
    }
    Stile.template row_bin_op<MFAExpSubOp>(new_max);
    AccT sum_tmp[ROWS_PT] = {0};
    Stile.template row_reduce<MFASumOp>(sum_tmp);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < ROWS_PT; i++) {
      sum_score[i] = sum_score[i] * factor[i] + sum_tmp[i];
    }
    Otile.template row_bin_op<MFAMulOp>(factor);

    if (kb != p->NK_aligned) {
      simdgroup_event::wait(1, &v_ev);
      threadgroup_barrier(mem_flags::mem_threadgroup);  // all simdgroups see full V tile
    }

    if (kb + 1 < kb_lim) {
      const device T* K_next = K + (long)(kb + 1) * BK * p->K_strides[2];
      if ((kb + 1) != p->NK_aligned) {
        k_ev.async_copy<LDK, TGP>(Ks, k_tile, K_next, (uint)p->K_strides[2], k_tile, /*transpose=*/true);
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          Vtile.template load<T, 1, 1>(
              &Vs[Vs_off + ik * 8 * LDV + id * 8], LDV, 1);
          MFAMMAFrag<AccT>::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    if (kb + 1 < kb_lim && (kb + 1) == p->NK_aligned) {
      const device T* K_next = K + (long)(kb + 1) * BK * p->K_strides[2];
      KLoader loader_k(K_next, (int)p->K_strides[2], Ks,
                       (ushort)simd_group_id, (ushort)simd_lane_id);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_k.load_safe(short2(BD, p->kL_rem));
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

  } // end kb loop

  Otile.template row_bin_op<MFADivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  device T* O_write = O_qb + (long)(tm + sm) * p->O_strides[2] + sn;
  if (qb == p->NQ_aligned) {
    auto dims = short2((short)(BD - sn), (short)(p->qL_rem - (tm + sm)));
    if (dims.x > 0 && dims.y > 0)
      Otile.template store_safe<T, 1, 1>(O_write, (int)p->O_strides[2], dims);
  } else {
    Otile.template store<T, 1, 1>(O_write, (int)p->O_strides[2]);
  }

  if (sn == 0) {
    const long l_boff = (long)tid.z * p->L_strides[0]
                      + (long)tid.y * p->L_strides[1];
    const long q_base = (long)qb * BQ + tm + sm;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < ROWS_PT; i++) {
      const long q_idx = q_base + i * 8;
      if (q_idx < p->qL) {
        L[l_boff + q_idx] = max_score[i] + metal::log2(sum_score[i]);
      }
    }
  }
}
