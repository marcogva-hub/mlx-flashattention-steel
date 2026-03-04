/// mfa_steel_fwd.cpp — STEEL-style cooperative forward attention kernel generator.
///
/// Generates a self-contained Metal shader source with:
///  - BlockLoaderT: all threads cooperate to load one tile from device → threadgroup
///  - MMATile / BaseMMAFrag / tile_matmad: simdgroup matrix wrappers
///  - mlx_mfa_attention kernel: Q@K^T, softmax, P@V, writes O + L (logsumexp)
///
/// The generated source has NO #include directives; all templates are inlined.
/// Tile sizes are injected via #define substitution at generation time.

#include "mfa_steel_fwd.hpp"
#include <sstream>

namespace mlx_mfa {

// ---------------------------------------------------------------------------
// Tile config selection
// ---------------------------------------------------------------------------

SteelBlockConfig select_steel_block_config(int head_dim, bool is_low_prec) {
  // STEEL tile config rules:
  //   TQ = BQ / (WM * 8)  must be >= 1  →  BQ >= WM * 8
  //   TGP_SIZE = WM * WN * 32
  //
  // f16/bf16 (low_prec=true): can use larger tiles (smaller register footprint).
  // f32 (low_prec=false): must use smaller tiles to avoid register spill pushing
  //   threadgroup memory over the 32 KB Metal limit.
  if (!is_low_prec) {
    // f32: small tiles to control threadgroup memory incl. spill space.
    // BQ=16, WM=2 → TQ=1, TGP_SIZE=64.
    if (head_dim <= 128) {
      return {16, 16, head_dim, 2, 1, 4};
    } else {
      return {16, 16, head_dim, 2, 1, 4};
    }
  }
  // f16 / bf16 ─────────────────────────────────────────────────────────────
  //   D=64  → BQ=32, BK=32, WM=4, WN=1
  //   D=128 → BQ=32, BK=16, WM=4, WN=1
  //   D=256 → BQ=16, BK=16, WM=2, WN=1  (BQ=16 ÷ (WM=2 × 8) = TQ=1)
  //
  // Note: BK=32 for D=128 was tested and is SLOWER (0.71x vs 0.86x SDPA).
  // Larger KV tiles increase per-thread load count; 128 threads × 32 elems
  // hits worse stride patterns than 128 threads × 16 elems.
  if (head_dim <= 128) {
    int BK = (head_dim <= 64) ? 32 : 16;
    return {32, BK, head_dim, 4, 1, 8};
  } else {
    // D=256: BQ=16, WM=2 so TQ = BQ/(WM*8) = 16/16 = 1
    return {16, 16, head_dim, 2, 1, 8};
  }
}

// ---------------------------------------------------------------------------
// Metal source generator
// ---------------------------------------------------------------------------

std::string generate_steel_forward_source(const ShaderCache::KernelKey& key) {
  const int BD = key.head_dim;
  const int BQ = key.block_q;
  const int BK = key.block_k;
  const int WM = key.n_warps;  // n_warps = BQ/8; for BQ=32,WM=4; BQ=16,WM=2
  const int WN = 1;
  const bool causal = key.causal;

  // dtype string for Metal
  const char* dtype_str = "half";
  if (key.dtype == 1)      dtype_str = "bfloat";
  else if (key.dtype == 2) dtype_str = "float";

  std::ostringstream ss;

  // ── Preamble ────────────────────────────────────────────────────────────
  ss << R"MFA(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL

)MFA";

  // ── MFASteelParams struct ────────────────────────────────────────────────
  ss << R"MFA(
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
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long O_strides[3];
  long L_strides[2];
};

)MFA";

  // ── BlockLoaderT template ────────────────────────────────────────────────
  ss << R"MFA(
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
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id,
      ushort simd_lane_id)
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * kDstStrRow + bj * kDstStrCol),
        src(src_ + bi * src_ld_ + bj) {}

  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
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
        for (short j = 0; j < vec_size; j++) {
          dst[i * kDstStrRow + j * kDstStrCol] = T(0);
        }
      }
      return;
    }
    bool tmp_idx[vec_size];
    T tmp_val[vec_size];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = tmp_val[j];
      }
    }
  }

  METAL_FUNC void next() { src += tile_stride; }
};

)MFA";

  // ── BaseMMAFrag / MMATile / tile_matmad ─────────────────────────────────
  ss << R"MFA(
template <typename T>
struct MFAMMAFrag {
  STEEL_CONST int kFragRows = 8;
  STEEL_CONST int kFragCols = 8;
  STEEL_CONST int kElemsPerFrag = 2;  // (8*8)/32
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
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] = static_cast<T>(src[i * str_x + j * str_y]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC static void row_reduce(thread const frag_type& inp,
                                    thread T* out) {
    T thr = Op::apply(inp.x, inp.y);
    T qr  = Op::apply(thr, simd_shuffle_xor(thr, ushort(1)));
    T sr  = Op::apply(qr,  simd_shuffle_xor(qr,  ushort(8)));
    out[0] = Op::apply(out[0], sr);
  }

  template <typename Op>
  METAL_FUNC static void row_bin_op(thread frag_type& inp,
                                    thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < kElemCols; j++) {
      inp[j] = Op::apply(inp[j], row_vals[0]);
    }
  }

  METAL_FUNC static void mma(thread frag_type& D,
                             thread frag_type& A,
                             thread frag_type& B,
                             thread frag_type& C) {
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
  STEEL_CONST int kFragRows    = 8;
  STEEL_CONST int kFragCols    = 8;
  STEEL_CONST int kElemsPerFrag = 2;
  STEEL_CONST int kTileRows    = kTileRows_;
  STEEL_CONST int kTileCols    = kTileCols_;
  STEEL_CONST int kNumFrags    = kTileRows_ * kTileCols_;
  STEEL_CONST int kElemsPerTile = kNumFrags * 2;
  STEEL_CONST int kRowsPerThread = kTileRows_;  // kElemRows=1 per frag

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
      for (short j = 0; j < kTileCols_; j++) {
        Frag::template row_reduce<Op>(frag_at(i, j), &vals[i]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread T vals[kTileRows_]) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows_; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols_; j++) {
        Frag::template row_bin_op<Op>(frag_at(i, j), &vals[i]);
      }
    }
  }

  // Load from threadgroup.
  // str_row = stride between rows in threadgroup (e.g. LDQ for Q, LDK for K^T)
  // str_col = stride between cols (1 for row-major, LDK for K^T transposed)
  template <typename U, int w_x, int w_y>
  METAL_FUNC void load(const threadgroup U* src, int str_row, int str_col) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows_; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols_; j++) {
        Frag::load(frag_at(i, j),
                   &src[(i * kFragRows) * w_x * str_row +
                        (j * kFragCols) * w_y * str_col],
                   str_row, str_col);
      }
    }
  }

  // Store to device (row-major, ld = leading dimension)
  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U* dst, const int ld) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows_; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols_; j++) {
        const int base = (i * kFragRows) * w_x * ld + (j * kFragCols) * w_y;
        dst[base + 0 * ld + 0] = static_cast<U>(frag_at(i, j)[0]);
        dst[base + 0 * ld + 1] = static_cast<U>(frag_at(i, j)[1]);
      }
    }
  }

  // Bounded store (dims.x = cols remaining, dims.y = rows remaining)
  template <typename U, int w_x, int w_y>
  METAL_FUNC void store_safe(device U* dst, const int ld,
                             const short2 dims) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows_; i++) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols_; j++) {
        const int off_row = (i * kFragRows) * w_x;
        const int off_col = (j * kFragCols) * w_y;
        // kElemRows=1, kElemCols=2
        if (off_row < (int)dims.y) {
          if (off_col     < (int)dims.x) dst[off_row * ld + off_col]     = static_cast<U>(frag_at(i, j)[0]);
          if (off_col + 1 < (int)dims.x) dst[off_row * ld + off_col + 1] = static_cast<U>(frag_at(i, j)[1]);
        }
      }
    }
  }
};

template <typename T, int M, int N, int K>
METAL_FUNC void mfa_tile_matmad(
    thread MFAMMATile<T, M, N>& D,
    thread MFAMMATile<T, M, K>& A,
    thread MFAMMATile<T, K, N>& B,
    thread MFAMMATile<T, M, N>& C) {
  STEEL_PRAGMA_UNROLL
  for (short m = 0; m < M; m++) {
    STEEL_PRAGMA_UNROLL
    for (short n = 0; n < N; n++) {
      short n_serp = (m % 2) ? (N - 1 - n) : n;
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < K; k++) {
        MFAMMAFrag<T>::mma(D.frag_at(m, n_serp),
                           A.frag_at(m, k),
                           B.frag_at(k, n_serp),
                           C.frag_at(m, n_serp));
      }
    }
  }
}

)MFA";

  // ── Arithmetic op structs ────────────────────────────────────────────────
  ss << R"MFA(
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

)MFA";

  // ── Compile-time tile constants ──────────────────────────────────────────
  // These are injected as Metal preprocessor defines so the compiler can
  // fully unroll all loops at compile time.
  ss << "#define MFA_BQ  " << BQ  << "\n";
  ss << "#define MFA_BK  " << BK  << "\n";
  ss << "#define MFA_BD  " << BD  << "\n";
  ss << "#define MFA_WM  " << WM  << "\n";
  ss << "#define MFA_WN  " << WN  << "\n";
  ss << "#define MFA_TGP_SIZE  " << (WM * WN * 32) << "\n";
  ss << "#define MFA_DTYPE  " << dtype_str << "\n";
  ss << "\n";

  // Derived tile counts
  const int TD = BD / 8;   // head-dim frags per warp
  const int TK = BK / 8;   // K-seq frags per K tile
  const int TQ = BQ / (WM * WN * 8); // Q-seq frags per warp (must be 1)
  const int kRowsPT = TQ;  // rows per thread in S tile

  ss << "#define MFA_TD  " << TD << "\n";
  ss << "#define MFA_TK  " << TK << "\n";
  ss << "#define MFA_TQ  " << TQ << "\n";
  ss << "#define MFA_ROWS_PT  " << kRowsPT << "\n";
  ss << "\n";

  // ── Main kernel ──────────────────────────────────────────────────────────
  // We emit the kernel as a template-specialized free function to allow the
  // Metal compiler to unroll all inner loops at the concrete tile sizes.
  ss << "[[kernel, max_total_threads_per_threadgroup(MFA_TGP_SIZE)]]\n";
  ss << "void mlx_mfa_attention(\n";
  ss << "    const device MFA_DTYPE* Q    [[buffer(0)]],\n";
  ss << "    const device MFA_DTYPE* K    [[buffer(1)]],\n";
  ss << "    const device MFA_DTYPE* V    [[buffer(2)]],\n";
  ss << "    device MFA_DTYPE*       O    [[buffer(3)]],\n";
  ss << "    device float*           L    [[buffer(4)]],\n";
  ss << "    const constant MFASteelParams* p [[buffer(5)]],\n";
  ss << "    uint simd_lane_id  [[thread_index_in_simdgroup]],\n";
  ss << "    uint simd_group_id [[simdgroup_index_in_threadgroup]],\n";
  ss << "    uint3 tid          [[threadgroup_position_in_grid]])\n";
  ss << "{\n";
  ss << "  typedef MFA_DTYPE T;\n";
  ss << "  typedef float     AccT;\n";
  ss << "\n";

  // Pointer offsets: tid.x=Q-block, tid.y=head, tid.z=batch
  ss << "  const ulong boff = (ulong)tid.z * p->Q_strides[0]\n";
  ss << "                   + (ulong)tid.y * p->Q_strides[1];\n";
  ss << "  const ulong kv_head = (uint)tid.y / (uint)p->gqa_factor;\n";
  ss << "  const ulong kv_boff_k = (ulong)tid.z * p->K_strides[0]\n";
  ss << "                        + kv_head      * p->K_strides[1];\n";
  ss << "  const ulong kv_boff_v = (ulong)tid.z * p->V_strides[0]\n";
  ss << "                        + kv_head      * p->V_strides[1];\n";
  ss << "\n";
  ss << "  Q += boff         + (ulong)tid.x * MFA_BQ * p->Q_strides[2];\n";
  ss << "  K += kv_boff_k;  // K loader walks K-seq; no per-block offset\n";
  ss << "  V += kv_boff_v;\n";
  ss << "  O += (ulong)tid.z * p->O_strides[0]\n";
  ss << "     + (ulong)tid.y * p->O_strides[1]\n";
  ss << "     + (ulong)tid.x * MFA_BQ * p->O_strides[2];\n";
  ss << "\n";

  // Threadgroup memory
  ss << "  constexpr short padQ  = 16 / sizeof(T);\n";
  ss << "  constexpr short padK  = 16 / sizeof(T);\n";
  ss << "  constexpr short padV  = 16 / sizeof(T);\n";
  ss << "  constexpr short LDQ   = MFA_BD + padQ;\n";
  ss << "  constexpr short LDK   = MFA_BK + padK;\n";
  ss << "  constexpr short LDV   = MFA_BD + padV;\n";
  ss << "  constexpr short kv_s0 = (MFA_BK + padK) * MFA_BD;\n";
  ss << "  constexpr short kv_s1 = MFA_BK * (MFA_BD + padV);\n";
  ss << "  constexpr short kv_s  = kv_s0 > kv_s1 ? kv_s0 : kv_s1;\n";
  ss << "\n";
  ss << "  threadgroup T Q_smem[MFA_BQ * (MFA_BD + 16/sizeof(T))];\n";
  ss << "  threadgroup T KV_smem[kv_s];\n";
  ss << "  threadgroup T* Qs = Q_smem;\n";
  ss << "  threadgroup T* Ks = KV_smem;\n";
  ss << "  threadgroup T* Vs = KV_smem;\n";
  ss << "\n";

  // Block loaders
  ss << "  // Q loader: row-major (kDstStrRow=LDQ, kDstStrCol=1, reduction_dim=1)\n";
  ss << "  using QLoader = MFABlockLoaderT<T, MFA_BQ, MFA_BD,\n";
  ss << "      /*kDstStrRow=*/ MFA_BD + 16/sizeof(T),\n";
  ss << "      /*kDstStrCol=*/ 1,\n";
  ss << "      /*reduction_dim=*/ 1,\n";
  ss << "      /*tgp_size=*/ MFA_TGP_SIZE>;\n";
  ss << "  // K loader: transposed into tgp (kDstStrRow=1, kDstStrCol=LDK, reduction_dim=0)\n";
  ss << "  using KLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      /*kDstStrRow=*/ 1,\n";
  ss << "      /*kDstStrCol=*/ MFA_BK + 16/sizeof(T),\n";
  ss << "      /*reduction_dim=*/ 0,\n";
  ss << "      /*tgp_size=*/ MFA_TGP_SIZE>;\n";
  ss << "  // V loader: row-major (kDstStrRow=LDV, kDstStrCol=1, reduction_dim=0)\n";
  ss << "  using VLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      /*kDstStrRow=*/ MFA_BD + 16/sizeof(T),\n";
  ss << "      /*kDstStrCol=*/ 1,\n";
  ss << "      /*reduction_dim=*/ 0,\n";
  ss << "      /*tgp_size=*/ MFA_TGP_SIZE>;\n";
  ss << "\n";
  ss << "  QLoader loader_q(Q, (int)p->Q_strides[2], Qs,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "  KLoader loader_k(K, (int)p->K_strides[2], Ks,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "  VLoader loader_v(V, (int)p->V_strides[2], Vs,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "\n";

  // MMA tile declarations
  ss << "  const AccT scale = p->scale * M_LOG2E_F;\n";
  ss << "\n";
  ss << "  // Warp offset within Q tile\n";
  ss << "  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);\n";
  ss << "  const short sm = simd_coord.y;\n";
  ss << "  const short sn = simd_coord.x;\n";
  ss << "  const short tm = 8 * MFA_TQ * (short)simd_group_id;\n";
  ss << "\n";
  ss << "  const short Qs_off = (tm + sm) * LDQ + sn;\n";
  ss << "  const short Ks_off = sm * LDK + sn;\n";
  ss << "  const short Vs_off = sm * LDV + sn;\n";
  ss << "\n";
  // Qtile is TQ×TD so Q is loaded once into registers before the K loop.
  // Ktile stays 1×TK (one D-slice per DD iteration) to avoid register pressure.
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Qtile;\n";
  ss << "  MFAMMATile<AccT, 1,     MFA_TK>  Ktile;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TK> Stile;\n";
  ss << "  MFAMMATile<AccT, 1,    1>       Vtile;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Otile;\n";
  ss << "  Otile.clear();\n";
  ss << "\n";

  // Online softmax state
  ss << "  AccT max_score[MFA_ROWS_PT];\n";
  ss << "  AccT sum_score[MFA_ROWS_PT];\n";
  ss << "  STEEL_PRAGMA_UNROLL\n";
  ss << "  for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "    max_score[i] = -INFINITY;\n";
  ss << "    sum_score[i] = 0.0f;\n";
  ss << "  }\n";
  ss << "\n";

  // Cooperative Q load: device → threadgroup SRAM
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "  if ((int)tid.x == p->NQ_aligned) {\n";
  ss << "    loader_q.load_safe(short2(MFA_BD, p->qL_rem));\n";
  ss << "  } else {\n";
  ss << "    loader_q.load_unsafe();\n";
  ss << "  }\n";
  // Load Q from TGP into registers ONCE (key STEEL optimization: Q stays in
  // registers across all K-tile iterations; only K/V stream through TGP).
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "  Qtile.template load<T, 1, 1>(&Qs[Qs_off], LDQ, 1);\n";
  ss << "\n";

  // K-loop limit (causal or full)
  if (causal) {
    ss << "  int q_max = ((int)tid.x + 1) * MFA_BQ + p->qL_off;\n";
    ss << "  int kb_lim = (q_max + MFA_BK - 1) / MFA_BK;\n";
    ss << "  if (kb_lim > p->NK) kb_lim = p->NK;\n";
  } else {
    ss << "  int kb_lim = p->NK;\n";
  }
  ss << "\n";

  // Main K/V loop
  ss << "  for (int kb = 0; kb < kb_lim; kb++) {\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "    if (kb == p->NK_aligned) {\n";
  ss << "      loader_k.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "    } else {\n";
  ss << "      loader_k.load_unsafe();\n";
  ss << "    }\n";
  ss << "\n";
  // S = Q @ K^T  (Q is in registers; only K is loaded from TGP per DD slice)
  ss << "    Stile.clear();\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short dd = 0; dd < MFA_TD; dd++) {\n";
  ss << "      Ktile.template load<T, 1, 1>(\n";
  ss << "          &Ks[Ks_off + (short)(dd * 8) * LDK], LDK, 1);\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short iq = 0; iq < MFA_TQ; iq++) {\n";
  ss << "        STEEL_PRAGMA_UNROLL\n";
  ss << "        for (short ik = 0; ik < MFA_TK; ik++) {\n";
  ss << "          MFAMMAFrag<AccT>::mma(\n";
  ss << "              Stile.frag_at(iq, ik),\n";
  ss << "              Qtile.frag_at(iq, dd),\n";
  ss << "              Ktile.frag_at(0, ik),\n";
  ss << "              Stile.frag_at(iq, ik));\n";
  ss << "        }\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "\n";
  ss << "    // Apply scale (log2-domain)\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++) {\n";
  ss << "      Stile.elems()[ii] *= scale;\n";
  ss << "    }\n";
  ss << "\n";

  // K-boundary mask
  ss << "    // Mask padded K positions\n";
  ss << "    if (kb == p->NK_aligned) {\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
  ss << "        STEEL_PRAGMA_UNROLL\n";
  ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
  ss << "          const short col = sn + j * 8;\n";
  ss << "          STEEL_PRAGMA_UNROLL\n";
  ss << "          for (short jj = 0; jj < 2; jj++) {\n";
  ss << "            if ((col + jj) >= p->kL_rem)\n";
  ss << "              Stile.frag_at(i,j)[jj] = -INFINITY;\n";
  ss << "          }\n";
  ss << "        }\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "\n";

  // Causal mask
  if (causal) {
    ss << "    // Causal mask: mask k > q\n";
    ss << "    if (kb >= (kb_lim - (MFA_BQ + MFA_BK - 1) / MFA_BK)) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        const int row = (int)tid.x * MFA_BQ + p->qL_off\n";
    ss << "                      + tm + sm + i * 8;\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
    ss << "          const int col = kb * MFA_BK + sn + j * 8;\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short jj = 0; jj < 2; jj++) {\n";
    ss << "            if (row < (col + jj))\n";
    ss << "              Stile.frag_at(i,j)[jj] = -INFINITY;\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";
  }

  // Load V while doing softmax
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "    if (kb == p->NK_aligned) {\n";
  ss << "      loader_v.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "    } else {\n";
  ss << "      loader_v.load_unsafe();\n";
  ss << "    }\n";
  ss << "\n";

  // Online softmax update
  ss << "    // Online softmax\n";
  ss << "    AccT new_max[MFA_ROWS_PT];\n";
  ss << "    AccT factor[MFA_ROWS_PT];\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short i = 0; i < MFA_ROWS_PT; i++) new_max[i] = max_score[i];\n";
  ss << "    Stile.template row_reduce<MFAMaxOp>(new_max);\n";
  ss << "    Stile.template row_bin_op<MFAExpSubOp>(new_max);\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "      factor[i] = fast::exp2(max_score[i] - new_max[i]);\n";
  ss << "      max_score[i] = new_max[i];\n";
  ss << "    }\n";
  ss << "    AccT sum_tmp[MFA_ROWS_PT] = {0};\n";
  ss << "    Stile.template row_reduce<MFASumOp>(sum_tmp);\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "      sum_score[i] = sum_score[i] * factor[i] + sum_tmp[i];\n";
  ss << "    }\n";
  ss << "    Otile.template row_bin_op<MFAMulOp>(factor);\n";
  ss << "\n";

  // P @ V accumulation
  ss << "    // O += P @ V\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  // Loop order: iq → ik → id (innermost)
  // ik is the K-sequence tile (ik*8*LDV: large stride in V_smem).
  // id is the D tile (id*8: small stride = 16-byte step in V_smem).
  // With id innermost, each ik block does a sequential D-scan (cache-friendly).
  // Stile[iq][ik] stays live for all TD id iterations (saves re-load vs id-outer).
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short iq = 0; iq < MFA_TQ; iq++) {\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short ik = 0; ik < MFA_TK; ik++) {\n";
  ss << "        STEEL_PRAGMA_UNROLL\n";
  ss << "        for (short id = 0; id < MFA_TD; id++) {\n";
  ss << "          Vtile.template load<T, 1, 1>(\n";
  ss << "              &Vs[Vs_off + ik*8*LDV + id*8], LDV, 1);\n";
  ss << "          MFAMMAFrag<AccT>::mma(\n";
  ss << "              Otile.frag_at(iq, id),\n";
  ss << "              Stile.frag_at(iq, ik),\n";
  ss << "              Vtile.frag_at(0, 0),\n";
  ss << "              Otile.frag_at(iq, id));\n";
  ss << "        }\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "    loader_k.next();\n";
  ss << "    loader_v.next();\n";
  ss << "  } // end kb loop\n";
  ss << "\n";

  // Normalize + store O
  ss << "  Otile.template row_bin_op<MFADivOp>(sum_score);\n";
  ss << "  threadgroup_barrier(mem_flags::mem_none);\n";
  ss << "\n";
  ss << "  O += (long)(tm + sm) * p->O_strides[2] + sn;\n";
  ss << "  if ((int)tid.x == p->NQ_aligned) {\n";
  ss << "    auto dims = short2((short)(MFA_BD - sn),\n";
  ss << "                       (short)(p->qL_rem - (tm + sm)));\n";
  ss << "    if (dims.x > 0 && dims.y > 0)\n";
  ss << "      Otile.template store_safe<T, 1, 1>(O, (int)p->O_strides[2], dims);\n";
  ss << "  } else {\n";
  ss << "    Otile.template store<T, 1, 1>(O, (int)p->O_strides[2]);\n";
  ss << "  }\n";
  ss << "\n";

  // Write L (logsumexp): only thread with sn==0 writes
  ss << "  // Write L = max_score + log2(sum_score)  (log2-domain logsumexp)\n";
  ss << "  // Only threads with sn==0 (first column of each frag row) write.\n";
  ss << "  if (sn == 0) {\n";
  ss << "    const long l_boff = (long)tid.z * p->L_strides[0]\n";
  ss << "                      + (long)tid.y * p->L_strides[1];\n";
  ss << "    const long q_base = (long)tid.x * MFA_BQ + tm + sm;\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "      const long q_idx = q_base + i * 8;\n";
  ss << "      if (q_idx < p->qL) {\n";
  ss << "        L[l_boff + q_idx] = max_score[i] + metal::log2(sum_score[i]);\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "  }\n";
  ss << "}\n";

  return ss.str();
}

}  // namespace mlx_mfa
