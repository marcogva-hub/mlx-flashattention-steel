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

SteelBlockConfig select_steel_block_config(int head_dim, bool is_low_prec,
                                           bool is_m3_plus) {
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
  //
  // M1/M2 (is_m3_plus=false):
  //   D=64  → BQ=32, BK=32, WM=4 (TGP=~22KB)
  //   D=128 → BQ=32, BK=16, WM=4 (TGP=~22KB)  BK=32 tested: 0.71x SDPA (worse)
  //   D=256 → BQ=32, BK=16, WM=4 (TGP=29184B < 32KB)
  //
  // M3/M4 (is_m3_plus=true) — dynamic register allocation avoids peak spill:
  //   D=64  → same as M1/M2 (already optimal at BK=32)
  //   D=128 → BQ=32, BK=32, WM=4 (TGP=29696B < 32KB) — wider K tile, try larger
  //   D=256 → same tile sizes; STEEL_PRAGMA_UNROLL enabled in shader gen
  //
  // Note: BQ=64 BK=32 D=128 = 41984B → exceeds 32 KB limit, rejected.
  if (head_dim <= 64) {
    // D=64: BK=32 optimal on all gens
    return {32, 32, head_dim, 4, 1, 8};
  }
  if (head_dim <= 128) {
    int BK = (is_m3_plus) ? 32 : 16;
    // M3+: BK=32 → TGP=29696B < 32KB, fewer K-iterations (may improve pipeline)
    // M1/M2: BK=16 → tested; BK=32 gives 0.71x SDPA on M1 (register spill)
    return {32, BK, head_dim, 4, 1, 8};
  } else {
    // D=256: same tile sizes on all gens; M3+ enables unroll in shader source
    return {32, 16, head_dim, 4, 1, 8};
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
  const bool sparse = key.sparse;
  const bool is_m3_plus        = key.is_m3_plus;
  const bool has_rope          = key.has_rope;
  const bool rope_interleaved  = key.rope_interleaved; // true=LLaMA, false=GPT-NeoX
  const bool has_alibi         = key.has_alibi;

  // dtype string for Metal
  const char* dtype_str = "half";
  if (key.dtype == 1)      dtype_str = "bfloat";
  else if (key.dtype == 2) dtype_str = "float";

  // Architecture gen constant for conditional Metal code (13=M1,14=M2,15=M3,16=M4)
  const int arch_gen = is_m3_plus ? 15 : 13;

  std::ostringstream ss;

  // ── Preamble ────────────────────────────────────────────────────────────
  // STEEL_PRAGMA_UNROLL:
  //   D<=128 (TD=8/16): full unroll on all gens — loop is short (8 or 16 iters).
  //   D=256 (TD=32):
  //     M1/M2: any unroll hint causes register spill or Metal AIR code bloat → empty.
  //     M3/M4: dynamic register allocation handles peak pressure → full unroll.
  ss << R"MFA(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define STEEL_CONST static constant constexpr const
)MFA";
  // ARCHITECTURE_GEN: injected as a compile-time constant for Metal shader code.
  ss << "#define ARCHITECTURE_GEN " << arch_gen << "\n";
  ss << "#define STEEL_PRAGMA_UNROLL";
  bool enable_unroll = (key.head_dim <= 128) || is_m3_plus;
  if (enable_unroll) ss << R"MFA( _Pragma("clang loop unroll(full)")
)MFA";
  else ss << "\n";
  ss << "\n";

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
  // RoPE fusion params (present in all variants; unused when has_rope=false)
  int rope_q_base;
  int rope_cos_stride;
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long O_strides[3];
  long L_strides[2];
  // Optional features (appended for backward compat; defaults 0.0f / 0)
  float softcap;   // 0.0 = disabled; >0 = tanh(S/cap)*cap before softmax
  int   has_alibi; // 0 = disabled; 1 = ALiBi bias from buffer(9)
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
  if (sparse)
    ss << "    const device uchar* block_mask   [[buffer(6)]],\n";
  if (has_rope) {
    ss << "    const device float* rotary_cos   [[buffer(7)]],\n";
    ss << "    const device float* rotary_sin   [[buffer(8)]],\n";
  }
  if (has_alibi)
    ss << "    const device float* alibi_slopes [[buffer(9)]],\n";
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
  // Q SRAM ready.  Optionally apply RoPE to Q in SRAM before register load.
  // Each thread handles a non-overlapping slice of (Q_row, D_pair) tuples;
  // no race conditions.  The existing barrier below synchronises writes.
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  if (has_rope) {
    ss << "  {\n";
    ss << "    const uint local_id = simd_group_id * 32 + simd_lane_id;\n";
    ss << "    const int qabs_base = p->rope_q_base + (int)tid.x * MFA_BQ;\n";
    ss << "    for (int ri = (int)local_id; ri < MFA_BQ * (MFA_BD/2);\n";
    ss << "         ri += MFA_TGP_SIZE) {\n";
    ss << "      const int row  = ri / (MFA_BD/2);\n";
    ss << "      const int pair = ri % (MFA_BD/2);\n";
    ss << "      const int cos_idx = (qabs_base + row) * p->rope_cos_stride + pair;\n";
    ss << "      const float cos_v = rotary_cos[cos_idx];\n";
    ss << "      const float sin_v = rotary_sin[cos_idx];\n";
    if (rope_interleaved) {
      // LLaMA-style: pairs are adjacent (d=2*pair, d=2*pair+1)
      ss << "      const int si0 = row * LDQ + pair * 2;\n";
      ss << "      const float q0 = (float)Qs[si0];\n";
      ss << "      const float q1 = (float)Qs[si0 + 1];\n";
      ss << "      Qs[si0]     = (T)(q0 * cos_v - q1 * sin_v);\n";
      ss << "      Qs[si0 + 1] = (T)(q0 * sin_v + q1 * cos_v);\n";
    } else {
      // GPT-NeoX style: pair d maps to d and d + D/2
      ss << "      const int si0 = row * LDQ + pair;\n";
      ss << "      const int si1 = row * LDQ + pair + MFA_BD/2;\n";
      ss << "      const float q0 = (float)Qs[si0];\n";
      ss << "      const float q1 = (float)Qs[si1];\n";
      ss << "      Qs[si0] = (T)(q0 * cos_v - q1 * sin_v);\n";
      ss << "      Qs[si1] = (T)(q0 * sin_v + q1 * cos_v);\n";
    }
    ss << "    }\n";
    ss << "  }\n";
    ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  }
  // Load Q from TGP into registers ONCE (key STEEL optimization: Q stays in
  // registers across all K-tile iterations; only K/V stream through TGP).
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
  // Block-sparse: skip K-tiles where block_mask[q_tile][kb] == 0.
  // All threads in a threadgroup share tid.x and kb, so this is a
  // uniform branch — no warp divergence, just skips the barriers and math.
  if (sparse) {
    ss << "    if (!block_mask[(int)tid.x * p->NK + kb]) {\n";
    ss << "      loader_k.next();\n";
    ss << "      loader_v.next();\n";
    ss << "      continue;\n";
    ss << "    }\n";
  }
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "    if (kb == p->NK_aligned) {\n";
  ss << "      loader_k.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "    } else {\n";
  ss << "      loader_k.load_unsafe();\n";
  ss << "    }\n";
  ss << "\n";
  // Optional RoPE fusion for K.
  // K SRAM is transposed: element [k_row, d_col] stored at Ks[d_col*LDK+k_row].
  // RoPE pair (d0=2*pair, d1=2*pair+1) maps to Ks[d0*LDK+k_row] / Ks[d1*LDK+k_row].
  // Each thread handles a unique (pair, k_row) slice — no race conditions.
  // A barrier before the RoPE block ensures all cooperative K-loader writes to
  // SRAM are visible to every thread before any thread starts the RoPE reads.
  // The second barrier (after Stile.clear()) protects the subsequent GEMM.
  if (has_rope) {
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "    {\n";
    ss << "      const uint local_id = simd_group_id * 32 + simd_lane_id;\n";
    ss << "      const int kabs_base = kb * MFA_BK;\n";
    ss << "      for (int ri = (int)local_id; ri < MFA_BK * (MFA_BD/2);\n";
    ss << "           ri += MFA_TGP_SIZE) {\n";
    ss << "        const int k_row = ri % MFA_BK;\n";
    ss << "        const int pair  = ri / MFA_BK;\n";
    ss << "        const int cos_idx = (kabs_base + k_row) * p->rope_cos_stride + pair;\n";
    ss << "        const float cos_v = rotary_cos[cos_idx];\n";
    ss << "        const float sin_v = rotary_sin[cos_idx];\n";
    if (rope_interleaved) {
      // LLaMA-style: K SRAM is transposed [d_col, k_row]; pair d maps to d*2, d*2+1
      ss << "        const int si0 = pair * 2       * LDK + k_row;\n";
      ss << "        const int si1 = (pair * 2 + 1) * LDK + k_row;\n";
    } else {
      // GPT-NeoX style: pair d maps to d, d+D/2 (column offsets in transposed K SRAM)
      ss << "        const int si0 = pair              * LDK + k_row;\n";
      ss << "        const int si1 = (pair + MFA_BD/2) * LDK + k_row;\n";
    }
    ss << "        const float k0 = (float)Ks[si0];\n";
    ss << "        const float k1 = (float)Ks[si1];\n";
    ss << "        Ks[si0] = (T)(k0 * cos_v - k1 * sin_v);\n";
    ss << "        Ks[si1] = (T)(k0 * sin_v + k1 * cos_v);\n";
    ss << "      }\n";
    ss << "    }\n";
  }
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

  // Softcapping (Gemma 2 / Grok) — operates in natural-log domain
  if (key.has_softcap) {
    ss << "    // Softcapping: tanh(S_nat / cap) * cap, convert log2 <-> natural\n";
    ss << "    {\n";
    ss << "      constexpr AccT log2e = 1.4426950408889634f;\n";
    ss << "      constexpr AccT ln2   = 0.6931471805599453f;\n";
    ss << "      const AccT cap   = p->softcap;\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++) {\n";
    ss << "        AccT s_nat = Stile.elems()[ii] * ln2;   // log2 → natural\n";
    ss << "        s_nat = precise::tanh(s_nat / cap) * cap;\n";
    ss << "        Stile.elems()[ii] = s_nat * log2e;       // natural → log2\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";
  }

  // ALiBi (Attention with Linear Biases) — per-head slope × (k_pos - q_pos)
  // Added to scores in log2 domain: bias_log2 = slope * (k - q) * log2e
  if (has_alibi) {
    ss << "    // ALiBi: add per-head linear position bias to scores\n";
    ss << "    {\n";
    ss << "      constexpr AccT log2e = 1.4426950408889634f;\n";
    ss << "      const AccT slope = alibi_slopes[(int)tid.y] * log2e;\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        const int q_pos = (int)tid.x * MFA_BQ + p->qL_off + (int)tm + (int)sm + i * 8;\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
    ss << "          const int k_base = kb * MFA_BK + (int)sn + j * 8;\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short jj = 0; jj < 2; jj++) {\n";
    ss << "            Stile.frag_at(i, j)[jj] += slope * (float)(k_base + jj - q_pos);\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";
  }

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

// =========================================================================
// Flash Decoding helpers
// =========================================================================

int compute_num_splits(int kL, int BK) {
  // Target ~2*BK keys per split so each split has at least 2 K-tile iterations
  // to amortize TGP setup.  Cap at 32 splits.
  int NK = (kL + BK - 1) / BK;
  int ideal = NK / 2;   // at least 2 tiles per split
  return std::max(1, std::min(ideal, 32));
}

// Append shared Metal template code (BlockLoaderT, MMA tiles, op structs).
// Both Phase 1 and Phase 2 use the same STEEL_PRAGMA_UNROLL / Metal headers.
static void append_metal_headers_and_defines(
    std::ostringstream& ss,
    bool enable_unroll,
    int arch_gen,
    const char* dtype_str) {
  ss << R"MFA(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define STEEL_CONST static constant constexpr const
)MFA";
  ss << "#define ARCHITECTURE_GEN " << arch_gen << "\n";
  ss << "#define STEEL_PRAGMA_UNROLL";
  if (enable_unroll) ss << R"MFA( _Pragma("clang loop unroll(full)")
)MFA";
  else ss << "\n";
  ss << "\n";
  (void)dtype_str;  // may be used by caller after this
}

// Append BlockLoaderT + MMA tile templates + Op structs (identical for all kernels).
static void append_steel_shared_templates(std::ostringstream& ss) {
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
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++)
        dst[i * kDstStrRow + j * kDstStrCol] = src[i * src_ld + j];
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

)MFA";

  ss << R"MFA(
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
  STEEL_CONST int kTileRows    = kTileRows_;
  STEEL_CONST int kTileCols    = kTileCols_;
  STEEL_CONST int kNumFrags    = kTileRows_ * kTileCols_;

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
}

// =========================================================================
// Flash Decoding Phase 1: generate_flash_decode_partial_source
//
// Kernel "mlx_mfa_flash_decode_partial"
// Grid: (NQ * num_splits, H, B)
//   tid.x = split_id * NQ + q_tile_id
//   tid.y = head index
//   tid.z = batch index
//
// For each split, iterates K-tiles [kb_start, kb_end) and accumulates
// partial O and L using the same online softmax as the STEEL forward.
// =========================================================================

std::string generate_flash_decode_partial_source(const ShaderCache::KernelKey& key) {
  auto cfg = select_steel_block_config(key.head_dim, /*is_low_prec=*/(key.dtype != 2),
                                       key.is_m3_plus);
  const int BQ = cfg.BQ;
  const int BK = cfg.BK;
  const int BD = key.head_dim;
  const int WM = cfg.WM;
  const int WN = cfg.WN;
  const int TGP_SIZE = WM * WN * 32;
  const int TD = BD / 8;
  const int TK = BK / 8;
  const int TQ = BQ / (WM * WN * 8);

  const bool causal    = key.causal;
  const bool has_alibi = key.has_alibi;
  const char* dtype_str = "half";
  if (key.dtype == 1)      dtype_str = "bfloat";
  else if (key.dtype == 2) dtype_str = "float";

  const bool is_m3_plus = key.is_m3_plus;
  const int arch_gen = is_m3_plus ? 15 : 13;
  const bool enable_unroll = (key.head_dim <= 128) || is_m3_plus;

  std::ostringstream ss;

  // ── Metal includes + defines ─────────────────────────────────────────────
  append_metal_headers_and_defines(ss, enable_unroll, arch_gen, dtype_str);

  // ── MFAFlashDecodePartialParams struct ───────────────────────────────────
  ss << R"MFAP(
struct MFAFlashDecodePartialParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  int NQ, NQ_aligned;
  int qL_rem;
  int qL_off;
  int NK_total;
  int NK_aligned;
  int kL_rem;
  int num_splits;
  int NK_per_split;
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long pO_split_stride;
  long pO_batch_stride;
  long pO_head_stride;
  long pL_split_stride;
  long pL_batch_stride;
  long pL_head_stride;
  // Optional features — appended at end for backward compatibility.
  float softcap;
};

)MFAP";

  // ── Shared templates ─────────────────────────────────────────────────────
  append_steel_shared_templates(ss);

  // ── Compile-time tile constants ──────────────────────────────────────────
  ss << "#define MFA_BQ  " << BQ  << "\n";
  ss << "#define MFA_BK  " << BK  << "\n";
  ss << "#define MFA_BD  " << BD  << "\n";
  ss << "#define MFA_WM  " << WM  << "\n";
  ss << "#define MFA_WN  " << WN  << "\n";
  ss << "#define MFA_TGP_SIZE  " << TGP_SIZE << "\n";
  ss << "#define MFA_DTYPE  " << dtype_str << "\n";
  ss << "#define MFA_TD  " << TD << "\n";
  ss << "#define MFA_TK  " << TK << "\n";
  ss << "#define MFA_TQ  " << TQ << "\n";
  ss << "#define MFA_ROWS_PT  " << TQ << "\n";
  ss << "\n";

  // ── Kernel function ──────────────────────────────────────────────────────
  ss << "[[kernel, max_total_threads_per_threadgroup(MFA_TGP_SIZE)]]\n";
  ss << "void mlx_mfa_flash_decode_partial(\n";
  ss << "    const device MFA_DTYPE* Q  [[buffer(0)]],\n";
  ss << "    const device MFA_DTYPE* K  [[buffer(1)]],\n";
  ss << "    const device MFA_DTYPE* V  [[buffer(2)]],\n";
  ss << "    device MFA_DTYPE*       pO [[buffer(3)]],\n";
  ss << "    device float*           pL [[buffer(4)]],\n";
  ss << "    const constant MFAFlashDecodePartialParams* p [[buffer(5)]],\n";
  if (has_alibi)
    ss << "    const device float* alibi_slopes [[buffer(6)]],\n";
  ss << "    uint simd_lane_id  [[thread_index_in_simdgroup]],\n";
  ss << "    uint simd_group_id [[simdgroup_index_in_threadgroup]],\n";
  ss << "    uint3 tid          [[threadgroup_position_in_grid]])\n";
  ss << "{\n";
  ss << "  typedef MFA_DTYPE T;\n";
  ss << "  typedef float     AccT;\n";
  ss << "\n";

  // Extract split_id and q_tile_id from tid.x
  ss << "  // tid.x = split_id * NQ + q_tile_id\n";
  ss << "  const uint split_id  = (uint)tid.x / (uint)p->NQ;\n";
  ss << "  const uint q_tile_id = (uint)tid.x % (uint)p->NQ;\n";
  ss << "\n";

  // Pointer offsets: q_tile_id for Q/O, head+batch for K/V
  ss << "  const ulong boff = (ulong)tid.z * p->Q_strides[0]\n";
  ss << "                   + (ulong)tid.y * p->Q_strides[1];\n";
  ss << "  const ulong kv_head = (uint)tid.y / (uint)p->gqa_factor;\n";
  ss << "  const ulong kv_boff_k = (ulong)tid.z * p->K_strides[0]\n";
  ss << "                        + kv_head      * p->K_strides[1];\n";
  ss << "  const ulong kv_boff_v = (ulong)tid.z * p->V_strides[0]\n";
  ss << "                        + kv_head      * p->V_strides[1];\n";
  ss << "\n";
  ss << "  Q += boff + (ulong)q_tile_id * MFA_BQ * p->Q_strides[2];\n";
  ss << "  K += kv_boff_k;\n";
  ss << "  V += kv_boff_v;\n";
  ss << "\n";

  // pO and pL output pointers with split offset
  ss << "  pO += (long)split_id     * p->pO_split_stride\n";
  ss << "      + (long)tid.z        * p->pO_batch_stride\n";
  ss << "      + (long)tid.y        * p->pO_head_stride\n";
  ss << "      + (long)q_tile_id    * MFA_BQ * p->D;\n";
  ss << "  pL += (long)split_id     * p->pL_split_stride\n";
  ss << "      + (long)tid.z        * p->pL_batch_stride\n";
  ss << "      + (long)tid.y        * p->pL_head_stride\n";
  ss << "      + (long)q_tile_id    * MFA_BQ;\n";
  ss << "\n";

  // Threadgroup memory (same as STEEL)
  ss << "  constexpr short padQ  = 16 / sizeof(T);\n";
  ss << "  constexpr short padK  = 16 / sizeof(T);\n";
  ss << "  constexpr short padV  = 16 / sizeof(T);\n";
  ss << "  constexpr short LDQ   = MFA_BD + padQ;\n";
  ss << "  constexpr short LDK   = MFA_BK + padK;\n";
  ss << "  constexpr short LDV   = MFA_BD + padV;\n";
  ss << "  constexpr short kv_s0 = (MFA_BK + padK) * MFA_BD;\n";
  ss << "  constexpr short kv_s1 = MFA_BK * (MFA_BD + padV);\n";
  ss << "  constexpr short kv_s  = kv_s0 > kv_s1 ? kv_s0 : kv_s1;\n";
  ss << "  threadgroup T Q_smem[MFA_BQ * (MFA_BD + 16/sizeof(T))];\n";
  ss << "  threadgroup T KV_smem[kv_s];\n";
  ss << "  threadgroup T* Qs = Q_smem;\n";
  ss << "  threadgroup T* Ks = KV_smem;\n";
  ss << "  threadgroup T* Vs = KV_smem;\n";
  ss << "\n";

  // Loaders
  ss << "  using QLoader = MFABlockLoaderT<T, MFA_BQ, MFA_BD,\n";
  ss << "      /*kDstStrRow=*/ MFA_BD + 16/sizeof(T),\n";
  ss << "      /*kDstStrCol=*/ 1, /*reduction_dim=*/ 1,\n";
  ss << "      /*tgp_size=*/ MFA_TGP_SIZE>;\n";
  ss << "  using KLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      /*kDstStrRow=*/ 1,\n";
  ss << "      /*kDstStrCol=*/ MFA_BK + 16/sizeof(T), /*reduction_dim=*/ 0,\n";
  ss << "      /*tgp_size=*/ MFA_TGP_SIZE>;\n";
  ss << "  using VLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      /*kDstStrRow=*/ MFA_BD + 16/sizeof(T),\n";
  ss << "      /*kDstStrCol=*/ 1, /*reduction_dim=*/ 0,\n";
  ss << "      /*tgp_size=*/ MFA_TGP_SIZE>;\n";
  ss << "\n";
  ss << "  QLoader loader_q(Q, (int)p->Q_strides[2], Qs,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "  KLoader loader_k(K, (int)p->K_strides[2], Ks,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "  VLoader loader_v(V, (int)p->V_strides[2], Vs,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "\n";

  // MMA tile state
  ss << "  const AccT scale = p->scale * M_LOG2E_F;\n";
  ss << "  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);\n";
  ss << "  const short sm = simd_coord.y;\n";
  ss << "  const short sn = simd_coord.x;\n";
  ss << "  const short tm = 8 * MFA_TQ * (short)simd_group_id;\n";
  ss << "  const short Qs_off = (tm + sm) * LDQ + sn;\n";
  ss << "  const short Ks_off = sm * LDK + sn;\n";
  ss << "  const short Vs_off = sm * LDV + sn;\n";
  ss << "\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Qtile;\n";
  ss << "  MFAMMATile<AccT, 1,     MFA_TK>  Ktile;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TK> Stile;\n";
  ss << "  MFAMMATile<AccT, 1,    1>        Vtile;\n";
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

  // Load Q (safe for boundary tile, unsafe otherwise)
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "  if ((int)q_tile_id == p->NQ_aligned) {\n";
  ss << "    loader_q.load_safe(short2(MFA_BD, p->qL_rem));\n";
  ss << "  } else {\n";
  ss << "    loader_q.load_unsafe();\n";
  ss << "  }\n";
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "  Qtile.template load<T, 1, 1>(&Qs[Qs_off], LDQ, 1);\n";
  ss << "\n";

  // K-loop bounds: [kb_start, kb_lim) where kb_lim respects causal + split end
  ss << "  const int kb_start = (int)split_id * p->NK_per_split;\n";
  ss << "  const int kb_end   = min(kb_start + p->NK_per_split, p->NK_total);\n";
  if (causal) {
    // qL_off positions the query globally; causal mask: col <= row
    ss << "  const int q_max    = (int)q_tile_id * MFA_BQ + p->qL_off + MFA_BQ;\n";
    ss << "  const int kb_causal_lim = (q_max + MFA_BK - 1) / MFA_BK;\n";
    ss << "  const int kb_lim   = min(kb_causal_lim, kb_end);\n";
  } else {
    ss << "  const int kb_lim   = kb_end;\n";
  }
  ss << "\n";

  // Advance K/V loaders to kb_start
  ss << "  // Advance K/V loaders past the splits before ours.\n";
  ss << "  for (int kb = 0; kb < kb_start; kb++) {\n";
  ss << "    loader_k.next();\n";
  ss << "    loader_v.next();\n";
  ss << "  }\n";
  ss << "\n";

  // Main K/V loop
  ss << "  for (int kb = kb_start; kb < kb_lim; kb++) {\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "    if (kb == p->NK_aligned) {\n";
  ss << "      loader_k.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "    } else {\n";
  ss << "      loader_k.load_unsafe();\n";
  ss << "    }\n";
  ss << "\n";
  // Q @ K^T
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
  ss << "    // Scale (log2-domain)\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++)\n";
  ss << "      Stile.elems()[ii] *= scale;\n";
  ss << "\n";

  // Softcapping (Gemma 2 / Grok) — same log2↔natural conversion as STEEL fwd
  if (key.has_softcap) {
    ss << "    // Softcapping: tanh(S_nat / cap) * cap, convert log2 <-> natural\n";
    ss << "    {\n";
    ss << "      constexpr AccT log2e = 1.4426950408889634f;\n";
    ss << "      constexpr AccT ln2   = 0.6931471805599453f;\n";
    ss << "      const AccT cap   = p->softcap;\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++) {\n";
    ss << "        AccT s_nat = Stile.elems()[ii] * ln2;   // log2 → natural\n";
    ss << "        s_nat = precise::tanh(s_nat / cap) * cap;\n";
    ss << "        Stile.elems()[ii] = s_nat * log2e;       // natural → log2\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";
  }

  // ALiBi per-head position bias (same formula as STEEL fwd)
  if (has_alibi) {
    ss << "    // ALiBi: add per-head linear position bias to scores\n";
    ss << "    {\n";
    ss << "      constexpr AccT log2e = 1.4426950408889634f;\n";
    ss << "      const AccT slope = alibi_slopes[(int)tid.y] * log2e;\n";
    // q_tile_id is already defined earlier in flash decode partial as (tid.x % NQ)
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        const int q_pos = (int)q_tile_id * MFA_BQ + p->qL_off + (int)tm + (int)sm + i * 8;\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
    ss << "          const int k_base = kb * MFA_BK + (int)sn + j * 8;\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short jj = 0; jj < 2; jj++) {\n";
    ss << "            Stile.frag_at(i, j)[jj] += slope * (float)(k_base + jj - q_pos);\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";
  }

  // Mask padding in last K-tile
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
    ss << "    if (kb >= (kb_causal_lim - (MFA_BQ + MFA_BK - 1) / MFA_BK)) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        const int row = (int)q_tile_id * MFA_BQ + p->qL_off\n";
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

  // Load V during softmax
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "    if (kb == p->NK_aligned) {\n";
  ss << "      loader_v.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "    } else {\n";
  ss << "      loader_v.load_unsafe();\n";
  ss << "    }\n";
  ss << "\n";

  // Online softmax update
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
  ss << "\n";

  // Rescale O and accumulate P@V  (same loop structure as original STEEL kernel)
  // Loop order: iq → ik → id (id innermost: D scan with cache-friendly V_smem access)
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "    Otile.template row_bin_op<MFAMulOp>(factor);\n";
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
  ss << "\n";
  ss << "    loader_k.next();\n";
  ss << "    loader_v.next();\n";
  ss << "  } // end kb loop\n";
  ss << "\n";

  // Normalize pO and store
  ss << "  Otile.template row_bin_op<MFADivOp>(sum_score);\n";
  ss << "  threadgroup_barrier(mem_flags::mem_none);\n";
  ss << "\n";
  ss << "  pO += (long)(tm + sm) * p->D + sn;\n";
  ss << "  if ((int)q_tile_id == p->NQ_aligned) {\n";
  ss << "    auto dims = short2((short)(MFA_BD - sn),\n";
  ss << "                       (short)(p->qL_rem - (tm + sm)));\n";
  ss << "    if (dims.x > 0 && dims.y > 0)\n";
  ss << "      Otile.template store_safe<T, 1, 1>(pO, (int)p->D, dims);\n";
  ss << "  } else {\n";
  ss << "    Otile.template store<T, 1, 1>(pO, (int)p->D);\n";
  ss << "  }\n";
  ss << "\n";

  // Write pL (log2-domain logsumexp)
  ss << "  if (sn == 0) {\n";
  ss << "    const long q_base = (long)q_tile_id * MFA_BQ + tm + sm;\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "      const long q_idx = q_base + i * 8;\n";
  ss << "      if (q_idx < p->qL) {\n";
  ss << "        pL[q_idx] = max_score[i] + metal::log2(sum_score[i]);\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "  }\n";
  ss << "}\n";

  return ss.str();
}

// =========================================================================
// Flash Decoding Phase 2: generate_flash_decode_reduce_source
//
// Kernel "mlx_mfa_flash_decode_reduce"
// Grid: (N, H, B) — one thread-column per query position, head, batch.
// Threadgroup: reduce_tgp_size threads (= min(D, 128)) parallelize over D.
//
// Combines partial outputs pO[s] and LSE pL[s] via:
//   lse_max = max_s(pL_s)
//   w_s     = exp2(pL_s - lse_max)
//   sum_w   = Σ w_s
//   O_final = Σ (w_s / sum_w) * pO_s    (in D-parallel threads)
//   L_final = lse_max + log2(sum_w)      (written by thread 0)
// =========================================================================

std::string generate_flash_decode_reduce_source(const ShaderCache::KernelKey& key) {
  const int BD = key.head_dim;
  const char* dtype_str = "half";
  if (key.dtype == 1)      dtype_str = "bfloat";
  else if (key.dtype == 2) dtype_str = "float";

  const int reduce_tgp = std::min(BD, 128);

  std::ostringstream ss;

  ss << "#include <metal_stdlib>\n";
  ss << "using namespace metal;\n\n";

  ss << "struct MFAFlashDecodeReduceParams {\n";
  ss << "  int B, H, D;\n";
  ss << "  int qL;\n";
  ss << "  int num_splits;\n";
  ss << "  long pO_split_stride;\n";
  ss << "  long pO_batch_stride;\n";
  ss << "  long pO_head_stride;\n";
  ss << "  long pL_split_stride;\n";
  ss << "  long pL_batch_stride;\n";
  ss << "  long pL_head_stride;\n";
  ss << "  long O_batch_stride;\n";
  ss << "  long O_head_stride;\n";
  ss << "  long L_batch_stride;\n";
  ss << "  long L_head_stride;\n";
  ss << "  int reduce_tgp_size;\n";
  ss << "};\n\n";

  ss << "#define MFA_DTYPE " << dtype_str << "\n";
  ss << "#define REDUCE_TGP " << reduce_tgp << "\n\n";

  ss << "[[kernel]]\n";
  ss << "void mlx_mfa_flash_decode_reduce(\n";
  ss << "    const device MFA_DTYPE*  pO [[buffer(0)]],\n";
  ss << "    const device float*      pL [[buffer(1)]],\n";
  ss << "    device MFA_DTYPE*         O [[buffer(2)]],\n";
  ss << "    device float*             L [[buffer(3)]],\n";
  ss << "    const constant MFAFlashDecodeReduceParams* p [[buffer(4)]],\n";
  ss << "    uint3 tid [[threadgroup_position_in_grid]],\n";
  ss << "    uint  d   [[thread_index_in_threadgroup]])\n";
  ss << "{\n";
  ss << "  // tid.x = q_idx, tid.y = head_idx, tid.z = batch_idx\n";
  ss << "  const long q_idx = (long)tid.x;\n";
  ss << "  const long h_idx = (long)tid.y;\n";
  ss << "  const long b_idx = (long)tid.z;\n";
  ss << "\n";
  ss << "  if (q_idx >= p->qL) return;\n";
  ss << "\n";
  ss << "  // Per-position base offsets in pL: stride over (split, batch, head, qL)\n";
  ss << "  const long pL_bh_base = b_idx * p->pL_batch_stride\n";
  ss << "                        + h_idx * p->pL_head_stride\n";
  ss << "                        + q_idx;\n";
  ss << "  const long pO_bh_base = b_idx * p->pO_batch_stride\n";
  ss << "                        + h_idx * p->pO_head_stride\n";
  ss << "                        + q_idx * p->D;\n";
  ss << "\n";
  ss << "  // Pass 1: find max logsumexp across splits\n";
  ss << "  float lse_max = -INFINITY;\n";
  ss << "  for (int s = 0; s < p->num_splits; s++) {\n";
  ss << "    float lse = pL[s * p->pL_split_stride + pL_bh_base];\n";
  ss << "    lse_max = metal::max(lse_max, lse);\n";
  ss << "  }\n";
  ss << "\n";
  ss << "  // Pass 2: compute sum of weights\n";
  ss << "  float sum_w = 0.0f;\n";
  ss << "  for (int s = 0; s < p->num_splits; s++) {\n";
  ss << "    float lse = pL[s * p->pL_split_stride + pL_bh_base];\n";
  ss << "    sum_w += metal::exp2(lse - lse_max);\n";
  ss << "  }\n";
  ss << "\n";
  ss << "  // Pass 3: accumulate weighted pO across splits (parallelized over D)\n";
  ss << "  for (int dd = (int)d; dd < p->D; dd += REDUCE_TGP) {\n";
  ss << "    float acc = 0.0f;\n";
  ss << "    for (int s = 0; s < p->num_splits; s++) {\n";
  ss << "      float lse = pL[s * p->pL_split_stride + pL_bh_base];\n";
  ss << "      float w   = metal::exp2(lse - lse_max) / sum_w;\n";
  ss << "      long  src = s * p->pO_split_stride + pO_bh_base + dd;\n";
  ss << "      acc += w * (float)pO[src];\n";
  ss << "    }\n";
  ss << "    long dst = b_idx * p->O_batch_stride\n";
  ss << "             + h_idx * p->O_head_stride\n";
  ss << "             + q_idx * p->D + dd;\n";
  ss << "    O[dst] = (MFA_DTYPE)acc;\n";
  ss << "  }\n";
  ss << "\n";
  ss << "  // Thread 0 writes final L = lse_max + log2(sum_w)\n";
  ss << "  if (d == 0) {\n";
  ss << "    long l_dst = b_idx * p->L_batch_stride\n";
  ss << "               + h_idx * p->L_head_stride\n";
  ss << "               + q_idx;\n";
  ss << "    L[l_dst] = lse_max + metal::log2(sum_w);\n";
  ss << "  }\n";
  ss << "}\n";

  return ss.str();
}

}  // namespace mlx_mfa
