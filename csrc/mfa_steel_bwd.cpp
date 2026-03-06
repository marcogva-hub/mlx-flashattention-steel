/// mfa_steel_bwd.cpp — STEEL native backward kernel generators.
///
/// Implements generate_steel_backward_dq_source() and
/// generate_steel_backward_dkv_source() using the same BlockLoaderT /
/// MMATile / mfa_tile_matmad infrastructure as the STEEL forward kernel.
///
/// Algorithm (FlashAttention-2 backward, log2 domain throughout):
///   P   = exp2(S_log2 − L_log2)   where S_log2 = Q@K^T * scale_log2
///   dP  = dO @ V^T
///   dS  = P * (dP − delta)        delta = rowsum(dO ⊙ O), precomputed
///   dQ += scale * dS @ K          (dQ kernel, grid NQ×H×B)
///   dV += P^T @ dO                (dKV kernel, grid NK×H_kv×B)
///   dK += scale * dS^T @ Q        (dKV kernel, same grid)
///
/// TGP budget (f16/bf16):
///   dQ  kernel, D=128 BK=16: 23.0 KB  ✓
///   dQ  kernel, D=64  BK=32: 14.0 KB  ✓
///   dKV kernel, D=128 BK=16: 23.6 KB  ✓  (streaming K/V approach)
///   dKV kernel, D=64  BK=32: 18.2 KB  ✓
///   D=256 exceeds 32 KB → falls back to mx.vjp(SDPA) in Python.

#include "mfa_steel_bwd.hpp"
#include "mfa_steel_fwd.hpp"   // select_steel_block_config
#include <sstream>
#include <cmath>

namespace mlx_mfa {

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Append Metal preamble (same as forward kernel preamble).
static void append_bwd_preamble(std::ostringstream& ss,
                                const ShaderCache::KernelKey& key) {
  const int arch_gen = key.is_m3_plus ? 15 : 13;
  ss << R"MFA(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define STEEL_CONST static constant constexpr const
)MFA";
  ss << "#define ARCHITECTURE_GEN " << arch_gen << "\n";
  ss << "#define STEEL_PRAGMA_UNROLL";
  if (key.head_dim <= 128 || key.is_m3_plus)
    ss << R"MFA( _Pragma("clang loop unroll(full)")
)MFA";
  else ss << "\n";
  ss << "\n";
}

/// Append BlockLoaderT + MMATile + op-struct templates (verbatim copy from fwd).
static void append_bwd_templates(std::ostringstream& ss) {
  ss << R"MFA(
template <typename T, short BROWS, short BCOLS,
          short kDstStrRow, short kDstStrCol,
          short reduction_dim, short tgp_size,
          short n_reads = (BCOLS * BROWS) / tgp_size,
          short TCOLS   = BCOLS / n_reads,
          short TROWS   = tgp_size / TCOLS>
struct MFABlockLoaderT {
  STEEL_CONST short vec_size = n_reads;
  const int  src_ld;
  const int  tile_stride;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T*       dst;
  const device T*      src;

  METAL_FUNC MFABlockLoaderT(const device T* src_, int src_ld_,
      threadgroup T* dst_, ushort sg, ushort sl)
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
        thread_idx(sg * 32 + sl),
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

  METAL_FUNC void load_safe(short2 dim) const {
    dim = dim - short2(bj, bi);
    if (dim.x <= 0 || dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++)
          dst[i * kDstStrRow + j * kDstStrCol] = T(0);
      }
      return;
    }
    bool ok[vec_size]; T tv[vec_size];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) ok[j] = (i < dim.y) && (j < dim.x);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) tv[j]  = src[ok[j] ? i * src_ld + j : 0];
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) tv[j]  = ok[j] ? tv[j] : T(0);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) dst[i * kDstStrRow + j * kDstStrCol] = tv[j];
    }
  }
  METAL_FUNC void next() { src += tile_stride; }
};

template <typename T> struct MFAMMAFrag {
  STEEL_CONST int kFragRows = 8; STEEL_CONST int kFragCols = 8;
  STEEL_CONST int kElemsPerFrag = 2;
  STEEL_CONST int kElemRows = 1; STEEL_CONST int kElemCols = 2;
  typedef simdgroup_matrix<T,8,8> mat_type;
  typedef vec<T,2> frag_type;

  METAL_FUNC static short2 get_coord(ushort lane) {
    const short qid = lane/4;
    const short fm  = (qid&4) + ((lane/2)%4);
    const short fn  = (qid&2)*2 + (lane%2)*2;
    return short2{fn,fm};
  }
  template<typename Src,typename Sx,typename Sy>
  METAL_FUNC static void load(thread frag_type& d, Src s, Sx sx, Sy sy) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++)
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++)
        d[i*kElemCols+j] = static_cast<T>(s[i*sx + j*sy]);
  }
  template<typename Op>
  METAL_FUNC static void row_reduce(thread const frag_type& in, thread T* out) {
    T t = Op::apply(in.x, in.y);
    T q = Op::apply(t,   simd_shuffle_xor(t, ushort(1)));
    T r = Op::apply(q,   simd_shuffle_xor(q, ushort(8)));
    out[0] = Op::apply(out[0], r);
  }
  template<typename Op>
  METAL_FUNC static void row_bin_op(thread frag_type& in, thread T* v) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < kElemCols; j++) in[j] = Op::apply(in[j], v[0]);
  }
  METAL_FUNC static void mma(thread frag_type& D, thread frag_type& A,
                              thread frag_type& B, thread frag_type& C) {
    mat_type Dm,Am,Bm,Cm;
    reinterpret_cast<thread frag_type&>(Am.thread_elements()) = A;
    reinterpret_cast<thread frag_type&>(Bm.thread_elements()) = B;
    reinterpret_cast<thread frag_type&>(Cm.thread_elements()) = C;
    simdgroup_multiply_accumulate(Dm,Am,Bm,Cm);
    D = reinterpret_cast<thread frag_type&>(Dm.thread_elements());
  }
};

template<typename T, int R, int C>
struct MFAMMATile {
  using Frag = MFAMMAFrag<T>;
  STEEL_CONST int kFragRows=8; STEEL_CONST int kFragCols=8;
  STEEL_CONST int kTileRows=R; STEEL_CONST int kTileCols=C;
  STEEL_CONST int kNumFrags=R*C; STEEL_CONST int kElemsPerTile=R*C*2;
  typedef typename Frag::frag_type frag_type;
  frag_type val_frags[R*C];

  METAL_FUNC MFAMMATile() thread {}
  METAL_FUNC void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i=0;i<R*C;i++) val_frags[i] = frag_type(0);
  }
  METAL_FUNC thread frag_type& frag_at(short i,short j) { return val_frags[i*C+j]; }
  METAL_FUNC const thread frag_type& frag_at(short i,short j) const { return val_frags[i*C+j]; }
  METAL_FUNC thread T* elems() { return reinterpret_cast<thread T*>(val_frags); }

  template<typename U, int wx, int wy>
  METAL_FUNC void load(const threadgroup U* src, int sr, int sc) {
    STEEL_PRAGMA_UNROLL
    for (short i=0;i<R;i++)
      STEEL_PRAGMA_UNROLL
      for (short j=0;j<C;j++)
        Frag::load(frag_at(i,j),
            &src[(i*8)*wx*sr + (j*8)*wy*sc], sr, sc);
  }

  template<typename U, int wx, int wy>
  METAL_FUNC void store(device U* d, int ld) const {
    STEEL_PRAGMA_UNROLL
    for (short i=0;i<R;i++)
      STEEL_PRAGMA_UNROLL
      for (short j=0;j<C;j++) {
        int base = (i*8)*wx*ld + (j*8)*wy;
        d[base + 0*ld + 0] = static_cast<U>(frag_at(i,j)[0]);
        d[base + 0*ld + 1] = static_cast<U>(frag_at(i,j)[1]);
      }
  }

  template<typename U, int wx, int wy>
  METAL_FUNC void store_safe(device U* d, int ld, short2 dims) const {
    STEEL_PRAGMA_UNROLL
    for (short i=0;i<R;i++)
      STEEL_PRAGMA_UNROLL
      for (short j=0;j<C;j++) {
        int or_ = (i*8)*wx; int oc = (j*8)*wy;
        if (or_ < (int)dims.y) {
          if (oc     < (int)dims.x) d[or_*ld + oc]     = static_cast<U>(frag_at(i,j)[0]);
          if (oc + 1 < (int)dims.x) d[or_*ld + oc + 1] = static_cast<U>(frag_at(i,j)[1]);
        }
      }
  }
};

template<typename T, int M, int N, int K>
METAL_FUNC void mfa_tile_matmad(
    thread MFAMMATile<T,M,N>& D, thread MFAMMATile<T,M,K>& A,
    thread MFAMMATile<T,K,N>& B, thread MFAMMATile<T,M,N>& C) {
  STEEL_PRAGMA_UNROLL
  for (short m=0;m<M;m++) {
    STEEL_PRAGMA_UNROLL
    for (short n=0;n<N;n++) {
      short ns = (m%2) ? (N-1-n) : n;
      STEEL_PRAGMA_UNROLL
      for (short k=0;k<K;k++)
        MFAMMAFrag<T>::mma(D.frag_at(m,ns), A.frag_at(m,k),
                           B.frag_at(k,ns), C.frag_at(m,ns));
    }
  }
}

struct MFAMaxOp  { template<typename T> METAL_FUNC static T apply(T x,T y){return metal::max(x,y);} };
struct MFASumOp  { template<typename T> METAL_FUNC static T apply(T x,T y){return x+y;} };
struct MFAMulOp  { template<typename T> METAL_FUNC static T apply(T x,T y){return x*y;} };
)MFA";
}

/// Append the MFASteelBwdParams struct definition in Metal.
static void append_bwd_params_struct(std::ostringstream& ss) {
  ss << R"MFA(
struct MFASteelBwdParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  float scale_log2;
  int NQ, NK;
  int NQ_aligned, NK_aligned;
  int qL_rem, kL_rem;
  int qL_off;
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long O_strides[3];
  long dO_strides[3];
  long dQ_strides[3];
  long dK_strides[3];
  long dV_strides[3];
  long L_strides[2];
};
)MFA";
}

// ---------------------------------------------------------------------------
// BA.2  generate_steel_backward_dq_source
// ---------------------------------------------------------------------------

std::string generate_steel_backward_dq_source(
    const ShaderCache::KernelKey& key) {

  auto cfg = select_steel_block_config(key.head_dim,
                                       /*is_low_prec=*/key.dtype != 2,
                                       key.is_m3_plus);
  const int BD = key.head_dim;
  const int BQ = cfg.BQ;
  const int BK = cfg.BK;
  const int WM = cfg.WM;
  const int WN = 1;
  const bool causal = key.causal;

  const char* dtype_str = "half";
  if (key.dtype == 1)      dtype_str = "bfloat";
  else if (key.dtype == 2) dtype_str = "float";

  const int TD  = BD / 8;
  const int TK  = BK / 8;
  const int TQ  = BQ / (WM * WN * 8);   // = 1 for all configs
  const int TGP = WM * WN * 32;

  // D-split: BD>128 exceeds 32KB TGP when using full BD for Q/dO/KV smem.
  // Split into BD_HALF=128 chunks (D_SPLITS = BD/128).
  // D=256 → D_SPLITS=2 (lo/hi). D=512 → D_SPLITS=4.
  // For BD<=128: d_split=false, BD_HALF=BD, TD_HALF=TD — path is identical.
  const bool d_split   = (BD > 128);
  const int  BD_HALF   = d_split ? 128 : BD;   // fixed 128, not BD/2
  const int  TD_HALF   = BD_HALF / 8;
  const int  D_SPLITS  = d_split ? (BD / BD_HALF) : 1;

  std::ostringstream ss;

  append_bwd_preamble(ss, key);
  append_bwd_templates(ss);
  append_bwd_params_struct(ss);

  // Tile size #defines
  ss << "#define MFA_BQ  " << BQ  << "\n";
  ss << "#define MFA_BK  " << BK  << "\n";
  ss << "#define MFA_BD  " << BD  << "\n";
  ss << "#define MFA_WM  " << WM  << "\n";
  ss << "#define MFA_WN  " << WN  << "\n";
  ss << "#define MFA_TGP_SIZE  " << TGP << "\n";
  ss << "#define MFA_DTYPE  " << dtype_str << "\n";
  ss << "#define MFA_TD  " << TD << "\n";
  ss << "#define MFA_TK  " << TK << "\n";
  ss << "#define MFA_TQ  " << TQ << "\n";
  ss << "#define MFA_BD_HALF   " << BD_HALF  << "\n";
  ss << "#define MFA_TD_HALF   " << TD_HALF  << "\n";
  ss << "#define MFA_D_SPLITS  " << D_SPLITS << "\n";
  // GQA factor: baked as compile-time constant to avoid struct-field read issues.
  ss << "#define MFA_GQA_FACTOR  " << key.gqa_factor << "\n";
  ss << "\n";

  // ── Kernel signature ────────────────────────────────────────────────────
  ss << "[[kernel, max_total_threads_per_threadgroup(MFA_TGP_SIZE)]]\n";
  ss << "void mlx_mfa_bwd_dq(\n";
  ss << "    const device MFA_DTYPE* Q     [[buffer(0)]],\n";
  ss << "    const device MFA_DTYPE* K     [[buffer(1)]],\n";
  ss << "    const device MFA_DTYPE* V     [[buffer(2)]],\n";
  ss << "    const device MFA_DTYPE* _O [[buffer(3)]],\n";  // bound but not read; D precomputed
  ss << "    const device float*     L     [[buffer(4)]],\n";
  ss << "    const device MFA_DTYPE* dO    [[buffer(5)]],\n";
  ss << "    const device float*     delta [[buffer(6)]],\n";  // rowsum(dO⊙O)
  ss << "    device MFA_DTYPE*       dQ    [[buffer(7)]],\n";
  ss << "    const constant MFASteelBwdParams* p [[buffer(8)]],\n";
  ss << "    uint simd_lane_id  [[thread_index_in_simdgroup]],\n";
  ss << "    uint simd_group_id [[simdgroup_index_in_threadgroup]],\n";
  ss << "    uint3 tid          [[threadgroup_position_in_grid]])\n";
  ss << "{\n";
  ss << "  typedef MFA_DTYPE T;\n";
  ss << "  typedef float     AccT;\n\n";

  // Pointer offsets: tid.x=Q-block, tid.y=Q-head, tid.z=batch
  ss << "  const ulong kv_head = (uint)tid.y / (uint)MFA_GQA_FACTOR;\n";
  ss << "  const ulong boff_q  = (ulong)tid.z * p->Q_strides[0]\n";
  ss << "                      + (ulong)tid.y * p->Q_strides[1];\n";
  ss << "  const ulong boff_k  = (ulong)tid.z * p->K_strides[0]\n";
  ss << "                      + kv_head      * p->K_strides[1];\n";
  ss << "  const ulong boff_v  = (ulong)tid.z * p->V_strides[0]\n";
  ss << "                      + kv_head      * p->V_strides[1];\n";
  ss << "  const ulong boff_dO = (ulong)tid.z * p->dO_strides[0]\n";
  ss << "                      + (ulong)tid.y * p->dO_strides[1];\n";
  ss << "  const ulong boff_dQ = (ulong)tid.z * p->dQ_strides[0]\n";
  ss << "                      + (ulong)tid.y * p->dQ_strides[1];\n\n";

  ss << "  Q  += boff_q  + (ulong)tid.x * MFA_BQ * p->Q_strides[2];\n";
  ss << "  K  += boff_k;\n";
  ss << "  V  += boff_v;\n";
  ss << "  dO += boff_dO + (ulong)tid.x * MFA_BQ * p->dO_strides[2];\n";
  ss << "  dQ += boff_dQ + (ulong)tid.x * MFA_BQ * p->dQ_strides[2];\n";
  ss << "  const long l_boff = (long)tid.z * p->L_strides[0]\n";
  ss << "                    + (long)tid.y * p->L_strides[1];\n";
  ss << "  L     += l_boff;\n";
  ss << "  delta += l_boff;\n\n";

  // TGP memory layout
  ss << "  constexpr short padQ   = 16 / sizeof(T);\n";
  ss << "  constexpr short padDO  = 16 / sizeof(T);\n";
  ss << "  constexpr short padKt  = 16 / sizeof(T);\n";
  ss << "  constexpr short padKr  = 16 / sizeof(T);\n";
  // TGP dims use BD_HALF (= BD for non-split; BD/2 for D=256 to fit 32KB TGP)
  ss << "  constexpr short LDQ   = MFA_BD_HALF + padQ;\n";
  ss << "  constexpr short LDdO  = MFA_BD_HALF + padDO;\n";
  ss << "  constexpr short LDKt  = MFA_BK + padKt;\n";  // K transposed stride
  ss << "  constexpr short LDKr  = MFA_BD_HALF + padKr;\n";  // K row-major stride
  // kv_s: shared buffer for K_t (BD_HALF*LDKt), V_t (same), K_row (BK*LDKr)
  ss << "  constexpr int kv_s0 = MFA_BD_HALF * (MFA_BK + 16/sizeof(T));\n";
  ss << "  constexpr int kv_s1 = MFA_BK * (MFA_BD_HALF + 16/sizeof(T));\n";
  ss << "  constexpr int kv_s  = kv_s0 > kv_s1 ? kv_s0 : kv_s1;\n\n";

  ss << "  threadgroup T  Q_smem [MFA_BQ * (MFA_BD_HALF + 16/sizeof(T))];\n";
  ss << "  threadgroup T  dO_smem[MFA_BQ * (MFA_BD_HALF + 16/sizeof(T))];\n";
  ss << "  threadgroup T  KV_smem[kv_s];\n\n";

  // Block loader type aliases — all use BD_HALF for tile cols
  ss << "  // Q loader: row-major [BQ x BD_HALF], reduction_dim=1\n";
  ss << "  using QLoader = MFABlockLoaderT<T, MFA_BQ, MFA_BD_HALF,\n";
  ss << "      MFA_BD_HALF+16/sizeof(T), 1, 1, MFA_TGP_SIZE>;\n";
  ss << "  // K transposed loader: [BK x BD_HALF] → TGP transposed, reduction_dim=0\n";
  ss << "  using KtLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD_HALF,\n";
  ss << "      1, MFA_BK+16/sizeof(T), 0, MFA_TGP_SIZE>;\n";
  ss << "  // V transposed loader: same params as K transposed\n";
  ss << "  using VtLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD_HALF,\n";
  ss << "      1, MFA_BK+16/sizeof(T), 0, MFA_TGP_SIZE>;\n";
  ss << "  // K row-major loader: [BK x BD_HALF] → TGP row-major, reduction_dim=0\n";
  ss << "  using KrLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD_HALF,\n";
  ss << "      MFA_BD_HALF+16/sizeof(T), 1, 0, MFA_TGP_SIZE>;\n\n";

  if (!d_split) {
    // D<=128: single set of loaders, same as before
    ss << "  QLoader  loader_q (Q,  (int)p->Q_strides[2],  Q_smem,  (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "  QLoader  loader_dO(dO, (int)p->dO_strides[2], dO_smem, (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "  KtLoader loader_kt(K,  (int)p->K_strides[2],  KV_smem, (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "  VtLoader loader_vt(V,  (int)p->V_strides[2],  KV_smem, (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "  KrLoader loader_kr(K,  (int)p->K_strides[2],  KV_smem, (ushort)simd_group_id, (ushort)simd_lane_id);\n\n";
  } else {
    // d_split (D=256: D_SPLITS=2, D=512: D_SPLITS=4):
    // KV/KR loaders are instantiated inline per D-chunk per K-tile.
    // Q/dO persistent loaders not needed; they're also created inline in the hoist loop.
    // (No persistent loader declarations needed here.)
  }

  // Simd coords
  ss << "  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);\n";
  ss << "  const short sm = simd_coord.y;\n";
  ss << "  const short sn = simd_coord.x;\n";
  ss << "  const short tm = 8 * MFA_TQ * (short)simd_group_id;\n\n";

  ss << "  const short Qs_off  = (tm + sm) * LDQ  + sn;\n";
  ss << "  const short dOs_off = (tm + sm) * LDdO + sn;\n";
  ss << "  const short Ks_off  = sm * LDKt + sn;\n";   // transposed K/V
  ss << "  const short KRs_off = sm * LDKr + sn;\n";   // row-major K
  ss << "\n";

  if (!d_split) {
    // D<=128: load Q and dO once (single tile covers full BD)
    ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "  if ((int)tid.x == p->NQ_aligned) {\n";
    ss << "    loader_q .load_safe(short2(MFA_BD, p->qL_rem));\n";
    ss << "    loader_dO.load_safe(short2(MFA_BD, p->qL_rem));\n";
    ss << "  } else {\n";
    ss << "    loader_q .load_unsafe();\n";
    ss << "    loader_dO.load_unsafe();\n";
    ss << "  }\n";
    ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";
    // Hoist Q and dO into registers
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Qtile, dOtile, dQtile;\n";
    ss << "  MFAMMATile<AccT, 1, MFA_TK> Stile, dPtile;\n";
    ss << "  MFAMMATile<AccT, 1, MFA_TK> Ktile;\n";
    ss << "  MFAMMATile<AccT, 1, 1>      KRtile;\n\n";
    ss << "  Qtile .template load<T,1,1>(&Q_smem [Qs_off],  LDQ,  1);\n";
    ss << "  dOtile.template load<T,1,1>(&dO_smem[dOs_off], LDdO, 1);\n";
    ss << "  dQtile.clear();\n\n";
  } else {
    // d_split: Q/dO hoisted into D_SPLITS-wide tile arrays; reuse Q_smem per chunk.
    // KV_smem is reused for each D-chunk within the K-tile loop below.
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD_HALF> Qtile [MFA_D_SPLITS];\n";
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD_HALF> dOtile[MFA_D_SPLITS];\n";
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD_HALF> dQtile[MFA_D_SPLITS];\n";
    ss << "  MFAMMATile<AccT, 1, MFA_TK> Stile, dPtile;\n";
    ss << "  MFAMMATile<AccT, 1, MFA_TK> Ktile;\n";
    ss << "  MFAMMATile<AccT, 1, 1>      KRtile;\n\n";

    // Hoist Q: load each BD_HALF-wide chunk into Qtile[dh]
    ss << "  STEEL_PRAGMA_UNROLL\n";
    ss << "  for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "    QLoader loader_q_dh(Q + dh * MFA_BD_HALF, (int)p->Q_strides[2],\n";
    ss << "                        Q_smem, (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "    if ((int)tid.x == p->NQ_aligned) loader_q_dh.load_safe(short2(MFA_BD_HALF, p->qL_rem));\n";
    ss << "    else                             loader_q_dh.load_unsafe();\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "    Qtile[dh].template load<T,1,1>(&Q_smem[Qs_off], LDQ, 1);\n";
    ss << "  }\n\n";

    // Hoist dO: load each BD_HALF-wide chunk into dOtile[dh]
    ss << "  STEEL_PRAGMA_UNROLL\n";
    ss << "  for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "    QLoader loader_dO_dh(dO + dh * MFA_BD_HALF, (int)p->dO_strides[2],\n";
    ss << "                         dO_smem, (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "    if ((int)tid.x == p->NQ_aligned) loader_dO_dh.load_safe(short2(MFA_BD_HALF, p->qL_rem));\n";
    ss << "    else                             loader_dO_dh.load_unsafe();\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "    dOtile[dh].template load<T,1,1>(&dO_smem[dOs_off], LDdO, 1);\n";
    ss << "  }\n";
    ss << "  STEEL_PRAGMA_UNROLL\n";
    ss << "  for (short dh = 0; dh < MFA_D_SPLITS; dh++) dQtile[dh].clear();\n\n";
  }

  // Read per-row L and delta scalars (one per thread, indexed by Q-row = tm+sm)
  ss << "  const long q_row_idx = (long)tid.x * MFA_BQ + tm + sm;\n";
  ss << "  const float L_val     = (q_row_idx < p->qL) ? L    [q_row_idx] : 0.0f;\n";
  ss << "  const float delta_val = (q_row_idx < p->qL) ? delta[q_row_idx] : 0.0f;\n\n";

  // K-tile loop limit (causal shortcut)
  if (causal) {
    ss << "  int q_max  = ((int)tid.x + 1) * MFA_BQ + p->qL_off;\n";
    ss << "  int kb_lim = (q_max + MFA_BK - 1) / MFA_BK;\n";
    ss << "  if (kb_lim > p->NK) kb_lim = p->NK;\n";
  } else {
    ss << "  int kb_lim = p->NK;\n";
  }
  ss << "\n";

  // ── K-tile loop ──────────────────────────────────────────────────────────
  ss << "  for (int kb = 0; kb < kb_lim; kb++) {\n\n";

  if (!d_split) {
    // ── D<=128 path (existing, unchanged) ────────────────────────────────
    // Phase 1: K transposed → S = Q@K^T
    ss << "    // Phase 1: K transposed → S = Q @ K^T\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "    if (kb == p->NK_aligned) loader_kt.load_safe(short2(MFA_BD, p->kL_rem));\n";
    ss << "    else                     loader_kt.load_unsafe();\n";
    ss << "    Stile.clear();\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short dd = 0; dd < MFA_TD; dd++) {\n";
    ss << "      Ktile.template load<T,1,1>(\n";
    ss << "          &KV_smem[Ks_off + (short)(dd*8) * LDKt], LDKt, 1);\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short ik = 0; ik < MFA_TK; ik++)\n";
    ss << "        MFAMMAFrag<AccT>::mma(Stile.frag_at(0,ik),\n";
    ss << "            Qtile.frag_at(0,dd), Ktile.frag_at(0,ik), Stile.frag_at(0,ik));\n";
    ss << "    }\n\n";
  } else {
    // ── d-split path (D=256: D_SPLITS=2; D=512: D_SPLITS=4) ───────────
    // Phase 1: S += Q_dh @ K^T_dh  for each D-chunk dh
    ss << "    // Phase 1 (d-split): S = sum_dh Q_dh @ K^T_dh\n";
    ss << "    Stile.clear();\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "      KtLoader loader_kt_dh(K + dh * MFA_BD_HALF\n";
    ss << "                            + (long)kb * MFA_BK * p->K_strides[2],\n";
    ss << "                            (int)p->K_strides[2], KV_smem,\n";
    ss << "                            (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "      if (kb == p->NK_aligned) loader_kt_dh.load_safe(short2(MFA_BD_HALF, p->kL_rem));\n";
    ss << "      else                     loader_kt_dh.load_unsafe();\n";
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short dd = 0; dd < MFA_TD_HALF; dd++) {\n";
    ss << "        Ktile.template load<T,1,1>(\n";
    ss << "            &KV_smem[Ks_off + (short)(dd*8) * LDKt], LDKt, 1);\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short ik = 0; ik < MFA_TK; ik++)\n";
    ss << "          MFAMMAFrag<AccT>::mma(Stile.frag_at(0,ik),\n";
    ss << "              Qtile[dh].frag_at(0,dd), Ktile.frag_at(0,ik), Stile.frag_at(0,ik));\n";
    ss << "      }\n";
    ss << "    }\n\n";
  }

  // Scale scores (log2 domain) — same for both paths
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short ii = 0; ii < MFA_TK*2; ii++)\n";
  ss << "      Stile.elems()[ii] *= p->scale_log2;\n\n";

  // K boundary mask — same for both paths
  ss << "    if (kb == p->NK_aligned) {\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short j = 0; j < MFA_TK; j++) {\n";
  ss << "        const short col = sn + j * 8;\n";
  ss << "        if ((col    ) >= p->kL_rem) Stile.frag_at(0,j)[0] = -INFINITY;\n";
  ss << "        if ((col + 1) >= p->kL_rem) Stile.frag_at(0,j)[1] = -INFINITY;\n";
  ss << "      }\n";
  ss << "    }\n\n";

  // Causal mask — same for both paths
  if (causal) {
    ss << "    if (kb >= (kb_lim - (MFA_BQ + MFA_BK - 1) / MFA_BK)) {\n";
    ss << "      const int q_row = (int)tid.x * MFA_BQ + p->qL_off + tm + sm;\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short j = 0; j < MFA_TK; j++) {\n";
    ss << "        const int k_col = kb * MFA_BK + sn + j * 8;\n";
    ss << "        if (q_row < (k_col    )) Stile.frag_at(0,j)[0] = -INFINITY;\n";
    ss << "        if (q_row < (k_col + 1)) Stile.frag_at(0,j)[1] = -INFINITY;\n";
    ss << "      }\n";
    ss << "    }\n\n";
  }

  if (!d_split) {
    // ── D<=128: Phase 2 (V transposed → dP = dO@V^T) ─────────────────────
    ss << "    // Phase 2: V transposed → dP = dO @ V^T\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "    if (kb == p->NK_aligned) loader_vt.load_safe(short2(MFA_BD, p->kL_rem));\n";
    ss << "    else                     loader_vt.load_unsafe();\n";
    ss << "    dPtile.clear();\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short dd = 0; dd < MFA_TD; dd++) {\n";
    ss << "      Ktile.template load<T,1,1>(\n";
    ss << "          &KV_smem[Ks_off + (short)(dd*8) * LDKt], LDKt, 1);\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short ik = 0; ik < MFA_TK; ik++)\n";
    ss << "        MFAMMAFrag<AccT>::mma(dPtile.frag_at(0,ik),\n";
    ss << "            dOtile.frag_at(0,dd), Ktile.frag_at(0,ik), dPtile.frag_at(0,ik));\n";
    ss << "    }\n\n";
  } else {
    // ── d-split Phase 2: dP += dO_dh @ V^T_dh  for each D-chunk ───────────
    ss << "    // Phase 2 (d-split): dP = sum_dh dO_dh @ V^T_dh\n";
    ss << "    dPtile.clear();\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "      VtLoader loader_vt_dh(V + dh * MFA_BD_HALF\n";
    ss << "                            + (long)kb * MFA_BK * p->V_strides[2],\n";
    ss << "                            (int)p->V_strides[2], KV_smem,\n";
    ss << "                            (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "      if (kb == p->NK_aligned) loader_vt_dh.load_safe(short2(MFA_BD_HALF, p->kL_rem));\n";
    ss << "      else                     loader_vt_dh.load_unsafe();\n";
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short dd = 0; dd < MFA_TD_HALF; dd++) {\n";
    ss << "        Ktile.template load<T,1,1>(\n";
    ss << "            &KV_smem[Ks_off + (short)(dd*8) * LDKt], LDKt, 1);\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short ik = 0; ik < MFA_TK; ik++)\n";
    ss << "          MFAMMAFrag<AccT>::mma(dPtile.frag_at(0,ik),\n";
    ss << "              dOtile[dh].frag_at(0,dd), Ktile.frag_at(0,ik), dPtile.frag_at(0,ik));\n";
    ss << "      }\n";
    ss << "    }\n\n";
  }

  // Compute P = exp2(S - L); dS = P*(dP - delta) — same for both paths
  ss << "    // P = exp2(S - L); dS = P * (dP - delta)\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short j = 0; j < MFA_TK; j++) {\n";
  ss << "      Stile.frag_at(0,j)[0] = fast::exp2(Stile.frag_at(0,j)[0] - L_val);\n";
  ss << "      Stile.frag_at(0,j)[1] = fast::exp2(Stile.frag_at(0,j)[1] - L_val);\n";
  ss << "    }\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short j = 0; j < MFA_TK; j++) {\n";
  ss << "      Stile.frag_at(0,j)[0] = p->scale * Stile.frag_at(0,j)[0]\n";
  ss << "                            * (dPtile.frag_at(0,j)[0] - delta_val);\n";
  ss << "      Stile.frag_at(0,j)[1] = p->scale * Stile.frag_at(0,j)[1]\n";
  ss << "                            * (dPtile.frag_at(0,j)[1] - delta_val);\n";
  ss << "    }\n\n";

  if (!d_split) {
    // ── D<=128: Phase 3 (K row-major → dQ += dS@K) ────────────────────────
    ss << "    // Phase 3: K row-major → dQ += dS @ K\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "    if (kb == p->NK_aligned) loader_kr.load_safe(short2(MFA_BD, p->kL_rem));\n";
    ss << "    else                     loader_kr.load_unsafe();\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short iq = 0; iq < MFA_TQ; iq++) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short ik = 0; ik < MFA_TK; ik++) {\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short id = 0; id < MFA_TD; id++) {\n";
    ss << "          KRtile.template load<T,1,1>(\n";
    ss << "              &KV_smem[KRs_off + (short)(ik*8)*LDKr + (short)(id*8)],\n";
    ss << "              LDKr, 1);\n";
    ss << "          MFAMMAFrag<AccT>::mma(\n";
    ss << "              dQtile.frag_at(iq,id),\n";
    ss << "              Stile .frag_at(iq,ik),\n";
    ss << "              KRtile.frag_at(0, 0),\n";
    ss << "              dQtile.frag_at(iq,id));\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n\n";
    ss << "    loader_kt.next();\n";
    ss << "    loader_vt.next();\n";
    ss << "    loader_kr.next();\n";
  } else {
    // ── d-split Phase 3: dQ_dh += dS @ K_dh  for each D-chunk ─────────────
    ss << "    // Phase 3 (d-split): dQ_dh += dS @ K_r_dh\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "      KrLoader loader_kr_dh(K + dh * MFA_BD_HALF\n";
    ss << "                            + (long)kb * MFA_BK * p->K_strides[2],\n";
    ss << "                            (int)p->K_strides[2], KV_smem,\n";
    ss << "                            (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "      if (kb == p->NK_aligned) loader_kr_dh.load_safe(short2(MFA_BD_HALF, p->kL_rem));\n";
    ss << "      else                     loader_kr_dh.load_unsafe();\n";
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short iq = 0; iq < MFA_TQ; iq++) {\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short ik = 0; ik < MFA_TK; ik++) {\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short id = 0; id < MFA_TD_HALF; id++) {\n";
    ss << "            KRtile.template load<T,1,1>(\n";
    ss << "                &KV_smem[KRs_off + (short)(ik*8)*LDKr + (short)(id*8)],\n";
    ss << "                LDKr, 1);\n";
    ss << "            MFAMMAFrag<AccT>::mma(\n";
    ss << "                dQtile[dh].frag_at(iq,id),\n";
    ss << "                Stile     .frag_at(iq,ik),\n";
    ss << "                KRtile.frag_at(0, 0),\n";
    ss << "                dQtile[dh].frag_at(iq,id));\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n\n";
    // No .next() needed: inline loaders compute offset from kb each iteration.
  }

  ss << "  } // end kb loop\n\n";

  // Write dQ to device
  ss << "  threadgroup_barrier(mem_flags::mem_none);\n";
  ss << "  dQ += (long)(tm + sm) * p->dQ_strides[2] + sn;\n";
  if (!d_split) {
    ss << "  if ((int)tid.x == p->NQ_aligned) {\n";
    ss << "    auto dims = short2((short)(MFA_BD - sn),\n";
    ss << "                       (short)(p->qL_rem - (tm + sm)));\n";
    ss << "    if (dims.x > 0 && dims.y > 0)\n";
    ss << "      dQtile.template store_safe<T,1,1>(dQ, (int)p->dQ_strides[2], dims);\n";
    ss << "  } else {\n";
    ss << "    dQtile.template store<T,1,1>(dQ, (int)p->dQ_strides[2]);\n";
    ss << "  }\n";
  } else {
    // d-split: write each BD_HALF-wide chunk at dQ + dh*BD_HALF
    ss << "  STEEL_PRAGMA_UNROLL\n";
    ss << "  for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "    if ((int)tid.x == p->NQ_aligned) {\n";
    ss << "      short rows_rem = (short)(p->qL_rem - (tm + sm));\n";
    ss << "      if (rows_rem > 0)\n";
    ss << "        dQtile[dh].template store_safe<T,1,1>(\n";
    ss << "            dQ + dh * MFA_BD_HALF, (int)p->dQ_strides[2],\n";
    ss << "            short2((short)(MFA_BD_HALF - sn), rows_rem));\n";
    ss << "    } else {\n";
    ss << "      dQtile[dh].template store<T,1,1>(dQ + dh * MFA_BD_HALF, (int)p->dQ_strides[2]);\n";
    ss << "    }\n";
    ss << "  }\n";
  }
  ss << "}\n";

  return ss.str();
}

// ---------------------------------------------------------------------------
// BB.1  generate_steel_backward_dkv_source
// ---------------------------------------------------------------------------
//
// Design: WM=1 (one simdgroup per threadgroup, TGP_SIZE=32).
//
// With WM>1, all warps would independently accumulate into the same dK/dV
// tile positions (same K-row range, different Q-row contributions) and then
// race-write to device.  WM=1 avoids the race: the single warp owns the full
// TK×TD tile and iterates TQ=BQ/8 Q-row fragments to cover the entire BQ-
// wide Q-tile.  No inter-warp reduction needed.
//
// TGP budget (f16, WM=1):
//   D=128 BK=16:  Q(8704) + dO(8704) + KV(6144) = 23552 B  ✓
//   D=64  BK=32:  Q(4608) + dO(4608) + KV(5120) = 14336 B  ✓
//   D=256: caller falls back to mx.vjp(SDPA)
// ---------------------------------------------------------------------------

std::string generate_steel_backward_dkv_source(
    const ShaderCache::KernelKey& key) {

  // D=256 is now supported via d_split (BD_HALF=128 per Q@K^T sub-pass).
  // f32 D=256: not reached (Python filters f32 before STEEL backward).

  auto cfg = select_steel_block_config(key.head_dim,
                                       /*is_low_prec=*/key.dtype != 2,
                                       key.is_m3_plus);
  const int BD  = key.head_dim;
  const int BQ  = cfg.BQ;
  const int BK  = (BD <= 64) ? cfg.BK : 16;
  // WM=1: one simdgroup, no inter-warp race on dK/dV tile.
  const int WM  = 1;
  const int WN  = 1;
  const bool causal = key.causal;

  const bool d_split  = (BD > 128);
  const int  BD_HALF  = d_split ? 128 : BD;   // fixed 128, not BD/2
  const int  TD_HALF  = BD_HALF / 8;
  const int  D_SPLITS = d_split ? (BD / BD_HALF) : 1;

  const char* dtype_str = "half";
  if (key.dtype == 1)      dtype_str = "bfloat";
  else if (key.dtype == 2) dtype_str = "float";

  // With WM=1, WN=1: TQ = BQ/8 = 4 (single warp covers all Q-rows).
  const int TD       = BD / 8;
  const int TK       = BK / 8;
  const int TQ       = BQ / (WM * WN * 8);  // = BQ/8 = 4 for BQ=32
  const int TGP_SIZE = WM * WN * 32;         // = 32

  std::ostringstream ss;

  append_bwd_preamble(ss, key);
  append_bwd_templates(ss);
  append_bwd_params_struct(ss);

  ss << "#define MFA_BQ  " << BQ       << "\n";
  ss << "#define MFA_BK  " << BK       << "\n";
  ss << "#define MFA_BD  " << BD       << "\n";
  ss << "#define MFA_TGP_SIZE  " << TGP_SIZE << "\n";
  ss << "#define MFA_DTYPE  " << dtype_str << "\n";
  ss << "#define MFA_TD  " << TD << "\n";
  ss << "#define MFA_TK  " << TK << "\n";
  ss << "#define MFA_TQ  " << TQ << "\n";
  ss << "#define MFA_BD_HALF   " << BD_HALF  << "\n";
  ss << "#define MFA_TD_HALF   " << TD_HALF  << "\n";
  ss << "#define MFA_D_SPLITS  " << D_SPLITS << "\n";
  // GQA factor: baked as compile-time constant to avoid struct-field read issues.
  ss << "#define MFA_GQA_FACTOR  " << key.gqa_factor << "\n";
  ss << "\n";

  // ── Kernel signature ─────────────────────────────────────────────────────
  // Grid: (NK, H_kv, B) — one threadgroup per K/V tile per KV head
  // Buffers: Q(0), K(1), V(2), O(3-unused), L(4), delta(5), dO(6),
  //          dK(7), dV(8), params(9)
  ss << "[[kernel, max_total_threads_per_threadgroup(MFA_TGP_SIZE)]]\n";
  ss << "void mlx_mfa_bwd_dkv(\n";
  ss << "    const device MFA_DTYPE* Q      [[buffer(0)]],\n";
  ss << "    const device MFA_DTYPE* K      [[buffer(1)]],\n";
  ss << "    const device MFA_DTYPE* V      [[buffer(2)]],\n";
  ss << "    const device MFA_DTYPE* _O [[buffer(3)]],\n";  // bound but not read; D precomputed
  ss << "    const device float*     L      [[buffer(4)]],\n";
  ss << "    const device float*     delta  [[buffer(5)]],\n";
  ss << "    const device MFA_DTYPE* dO     [[buffer(6)]],\n";
  ss << "    device MFA_DTYPE*       dK     [[buffer(7)]],\n";
  ss << "    device MFA_DTYPE*       dV     [[buffer(8)]],\n";
  ss << "    const constant MFASteelBwdParams* p [[buffer(9)]],\n";
  ss << "    uint simd_lane_id  [[thread_index_in_simdgroup]],\n";
  ss << "    uint simd_group_id [[simdgroup_index_in_threadgroup]],\n";
  ss << "    uint3 tid          [[threadgroup_position_in_grid]])\n";
  ss << "{\n";
  ss << "  typedef MFA_DTYPE T;\n";
  ss << "  typedef float     AccT;\n\n";

  // tid.y = kv_head (grid Y = H_kv = H / gqa_factor)
  ss << "  const int kv_head = (int)tid.y;\n\n";

  // K/V pointers (fixed for all Q-head/Q-tile iterations)
  ss << "  const ulong boff_k = (ulong)tid.z * p->K_strides[0]\n";
  ss << "                     + (ulong)kv_head * p->K_strides[1];\n";
  ss << "  const ulong boff_v = (ulong)tid.z * p->V_strides[0]\n";
  ss << "                     + (ulong)kv_head * p->V_strides[1];\n";
  ss << "  K  += boff_k + (ulong)tid.x * MFA_BK * p->K_strides[2];\n";
  ss << "  V  += boff_v + (ulong)tid.x * MFA_BK * p->V_strides[2];\n\n";

  // dK/dV pointers for this K-tile
  ss << "  const ulong boff_dK = (ulong)tid.z * p->dK_strides[0]\n";
  ss << "                      + (ulong)kv_head * p->dK_strides[1];\n";
  ss << "  const ulong boff_dV = (ulong)tid.z * p->dV_strides[0]\n";
  ss << "                      + (ulong)kv_head * p->dV_strides[1];\n";
  ss << "  dK += boff_dK + (ulong)tid.x * MFA_BK * p->dK_strides[2];\n";
  ss << "  dV += boff_dV + (ulong)tid.x * MFA_BK * p->dV_strides[2];\n\n";

  // TGP memory layout — use BD_HALF for Q/dO/K_t to fit 32KB for D=256
  ss << "  constexpr short padQ  = 16 / sizeof(T);\n";
  ss << "  constexpr short padDO = 16 / sizeof(T);\n";
  ss << "  constexpr short padKt = 16 / sizeof(T);\n";
  ss << "  constexpr short LDQ   = MFA_BD_HALF + padQ;\n";
  ss << "  constexpr short LDdO  = MFA_BD_HALF + padDO;\n";
  ss << "  constexpr short LDKt  = MFA_BK + padKt;\n";
  // tmp_smem: [BK x BQ+padT] for P^T / dS^T scatter — independent of BD
  ss << "  constexpr short padT  = 16 / sizeof(T);\n";
  ss << "  constexpr short LDT   = MFA_BQ + padT;\n";
  // kv_s: max(K_t buf BD_HALF-wide, tmp buf for P^T scatter)
  ss << "  constexpr int kv_s0 = MFA_BD_HALF * (MFA_BK + 16/sizeof(T));\n";
  ss << "  constexpr int kv_s1 = MFA_BK * (MFA_BQ + 16/sizeof(T));\n";
  ss << "  constexpr int kv_s  = kv_s0 > kv_s1 ? kv_s0 : kv_s1;\n\n";

  ss << "  threadgroup T Q_smem [MFA_BQ * (MFA_BD_HALF + 16/sizeof(T))];\n";
  ss << "  threadgroup T dO_smem[MFA_BQ * (MFA_BD_HALF + 16/sizeof(T))];\n";
  ss << "  threadgroup T KV_smem[kv_s];\n\n";

  // Block loader aliases (TGP_SIZE=32 threads) — BD_HALF cols
  ss << "  using RowLoader = MFABlockLoaderT<T, MFA_BQ, MFA_BD_HALF,\n";
  ss << "      MFA_BD_HALF+16/sizeof(T), 1, 1, MFA_TGP_SIZE>;\n";
  ss << "  using KtLoader  = MFABlockLoaderT<T, MFA_BK, MFA_BD_HALF,\n";
  ss << "      1, MFA_BK+16/sizeof(T), 0, MFA_TGP_SIZE>;\n\n";

  // Simd coords — WM=1 so simd_group_id is always 0; tm is always 0.
  ss << "  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);\n";
  ss << "  const short sm  = simd_coord.y;\n";
  ss << "  const short sn  = simd_coord.x;\n";
  // With WM=1, tm=0 always; written explicitly for clarity.
  ss << "  const short tm  = (short)0;\n";
  ss << "  const short Qs_off  = (tm + sm) * LDQ  + sn;\n";
  ss << "  const short dOs_off = (tm + sm) * LDdO + sn;\n";
  ss << "  const short Kts_off = sm * LDKt + sn;\n\n";

  // dK and dV accumulators: full D or split into lo/hi halves
  if (!d_split) {
    ss << "  MFAMMATile<AccT, MFA_TK, MFA_TD> dKtile, dVtile;\n";
    ss << "  dKtile.clear();\n";
    ss << "  dVtile.clear();\n\n";
  } else {
    ss << "  MFAMMATile<AccT, MFA_TK, MFA_TD_HALF> dKtile[MFA_D_SPLITS];\n";
    ss << "  MFAMMATile<AccT, MFA_TK, MFA_TD_HALF> dVtile[MFA_D_SPLITS];\n";
    ss << "  STEEL_PRAGMA_UNROLL\n";
    ss << "  for (short dh = 0; dh < MFA_D_SPLITS; dh++) { dKtile[dh].clear(); dVtile[dh].clear(); }\n\n";
  }

  // Loop over Q heads that map to this KV head (for GQA)
  ss << "  const int q_head_start = kv_head * MFA_GQA_FACTOR;\n";
  ss << "  const int q_head_end   = q_head_start + MFA_GQA_FACTOR;\n\n";

  ss << "  for (int q_head = q_head_start; q_head < q_head_end; q_head++) {\n\n";

  // Q-tile loop lower bound (causal: skip Q-tiles fully before this K-tile)
  if (causal) {
    ss << "    int qb_min = ((int)tid.x * MFA_BK - p->qL_off) / MFA_BQ;\n";
    ss << "    if (qb_min < 0) qb_min = 0;\n";
  } else {
    ss << "    int qb_min = 0;\n";
  }
  ss << "\n";

  ss << "    // Per-Q-head strides\n";
  ss << "    const ulong boff_q  = (ulong)tid.z * p->Q_strides[0]  + (ulong)q_head * p->Q_strides[1];\n";
  ss << "    const ulong boff_dO = (ulong)tid.z * p->dO_strides[0] + (ulong)q_head * p->dO_strides[1];\n";
  ss << "    const long  l_boff  = (long)tid.z * p->L_strides[0]   + (long)q_head * p->L_strides[1];\n\n";

  ss << "    for (int qb = qb_min; qb < p->NQ; qb++) {\n\n";

  // Q and dO base for this Q-tile
  ss << "      const device T* Qptr  = Q  + boff_q  + (ulong)qb * MFA_BQ * p->Q_strides[2];\n";
  ss << "      const device T* dOptr = dO + boff_dO + (ulong)qb * MFA_BQ * p->dO_strides[2];\n\n";

  // Tile declarations — split vs non-split
  ss << "      MFAMMATile<AccT, MFA_TQ, MFA_TK> Stile;\n";
  ss << "      MFAMMATile<AccT, 1, MFA_TK>      Ktile;\n";
  if (!d_split) {
    ss << "      MFAMMATile<AccT, MFA_TQ, MFA_TD> Qtile, dOtile;\n";
  } else {
    ss << "      MFAMMATile<AccT, MFA_TQ, MFA_TD_HALF> Qtile [MFA_D_SPLITS];\n";
    ss << "      MFAMMATile<AccT, MFA_TQ, MFA_TD_HALF> dOtile[MFA_D_SPLITS];\n";
  }
  ss << "      Stile.clear();\n\n";

  if (!d_split) {
    // ── Load Q, dO, K_t (single BD-wide tiles) ────────────────────────────
    ss << "      RowLoader lq (Qptr,  (int)p->Q_strides[2],  Q_smem,  (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "      RowLoader ldO(dOptr, (int)p->dO_strides[2], dO_smem, (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "      KtLoader  lkt(K, (int)p->K_strides[2], KV_smem, (ushort)simd_group_id, (ushort)simd_lane_id);\n\n";
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "      if (qb == p->NQ_aligned) {\n";
    ss << "        lq .load_safe(short2(MFA_BD, p->qL_rem));\n";
    ss << "        ldO.load_safe(short2(MFA_BD, p->qL_rem));\n";
    ss << "      } else { lq.load_unsafe(); ldO.load_unsafe(); }\n";
    ss << "      if ((int)tid.x == p->NK_aligned) lkt.load_safe(short2(MFA_BD, p->kL_rem));\n";
    ss << "      else                             lkt.load_unsafe();\n";
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";
    ss << "      Qtile .template load<T,1,1>(&Q_smem [Qs_off],  LDQ,  1);\n";
    ss << "      dOtile.template load<T,1,1>(&dO_smem[dOs_off], LDdO, 1);\n\n";
    // S = Q @ K^T
    ss << "      // S[TQ,TK] = Q[TQ,TD] @ K_t[TD,TK]\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short dd = 0; dd < MFA_TD; dd++) {\n";
    ss << "        Ktile.template load<T,1,1>(\n";
    ss << "            &KV_smem[Kts_off + (short)(dd*8)*LDKt], LDKt, 1);\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short iq = 0; iq < MFA_TQ; iq++)\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short ik = 0; ik < MFA_TK; ik++)\n";
    ss << "            MFAMMAFrag<AccT>::mma(Stile.frag_at(iq,ik),\n";
    ss << "                Qtile.frag_at(iq,dd), Ktile.frag_at(0,ik), Stile.frag_at(iq,ik));\n";
    ss << "      }\n\n";
  } else {
    // ── d-split (D=256: D_SPLITS=2; D=512: D_SPLITS=4) ────────────────────
    // Load Q chunks: Qtile[dh] = Q[rows, dh*BD_HALF .. (dh+1)*BD_HALF]
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "        RowLoader lq_dh(Qptr + dh * MFA_BD_HALF, (int)p->Q_strides[2],\n";
    ss << "            Q_smem, (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "        if (qb == p->NQ_aligned) lq_dh.load_safe(short2(MFA_BD_HALF, p->qL_rem));\n";
    ss << "        else                     lq_dh.load_unsafe();\n";
    ss << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "        Qtile[dh].template load<T,1,1>(&Q_smem[Qs_off], LDQ, 1);\n";
    ss << "      }\n";
    // Load dO chunks
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "        RowLoader ldO_dh(dOptr + dh * MFA_BD_HALF, (int)p->dO_strides[2],\n";
    ss << "            dO_smem, (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "        if (qb == p->NQ_aligned) ldO_dh.load_safe(short2(MFA_BD_HALF, p->qL_rem));\n";
    ss << "        else                     ldO_dh.load_unsafe();\n";
    ss << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "        dOtile[dh].template load<T,1,1>(&dO_smem[dOs_off], LDdO, 1);\n";
    ss << "      }\n\n";
    // S = sum_dh Qtile[dh] @ K^T_dh
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "        KtLoader lkt_dh(K + dh * MFA_BD_HALF, (int)p->K_strides[2],\n";
    ss << "            KV_smem, (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "        if ((int)tid.x == p->NK_aligned) lkt_dh.load_safe(short2(MFA_BD_HALF, p->kL_rem));\n";
    ss << "        else                             lkt_dh.load_unsafe();\n";
    ss << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short dd = 0; dd < MFA_TD_HALF; dd++) {\n";
    ss << "          Ktile.template load<T,1,1>(&KV_smem[Kts_off + (short)(dd*8)*LDKt], LDKt, 1);\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short iq = 0; iq < MFA_TQ; iq++)\n";
    ss << "            STEEL_PRAGMA_UNROLL\n";
    ss << "            for (short ik = 0; ik < MFA_TK; ik++)\n";
    ss << "              MFAMMAFrag<AccT>::mma(Stile.frag_at(iq,ik),\n";
    ss << "                  Qtile[dh].frag_at(iq,dd), Ktile.frag_at(0,ik), Stile.frag_at(iq,ik));\n";
    ss << "        }\n";
    ss << "      }\n\n";
  }

  // Scale by scale_log2
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short ii = 0; ii < MFA_TQ*MFA_TK*2; ii++)\n";
  ss << "        Stile.elems()[ii] *= p->scale_log2;\n\n";

  // K boundary mask (last K-tile)
  ss << "      if ((int)tid.x == p->NK_aligned) {\n";
  ss << "        STEEL_PRAGMA_UNROLL\n";
  ss << "        for (short iq = 0; iq < MFA_TQ; iq++)\n";
  ss << "          STEEL_PRAGMA_UNROLL\n";
  ss << "          for (short j = 0; j < MFA_TK; j++) {\n";
  ss << "            const short col = sn + j*8;\n";
  ss << "            if ((col    ) >= p->kL_rem) Stile.frag_at(iq,j)[0] = -INFINITY;\n";
  ss << "            if ((col + 1) >= p->kL_rem) Stile.frag_at(iq,j)[1] = -INFINITY;\n";
  ss << "          }\n";
  ss << "      }\n\n";

  // Q boundary mask (last Q-tile): mask Q rows >= qL_rem
  ss << "      if (qb == p->NQ_aligned) {\n";
  ss << "        STEEL_PRAGMA_UNROLL\n";
  ss << "        for (short iq = 0; iq < MFA_TQ; iq++) {\n";
  ss << "          if ((iq*8 + sm) >= p->qL_rem) {\n";
  ss << "            STEEL_PRAGMA_UNROLL\n";
  ss << "            for (short j = 0; j < MFA_TK; j++) {\n";
  ss << "              Stile.frag_at(iq,j)[0] = -INFINITY;\n";
  ss << "              Stile.frag_at(iq,j)[1] = -INFINITY;\n";
  ss << "            }\n";
  ss << "          }\n";
  ss << "        }\n";
  ss << "      }\n\n";

  // Causal mask for dKV
  if (causal) {
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short iq = 0; iq < MFA_TQ; iq++) {\n";
    ss << "        const int q_row = qb * MFA_BQ + p->qL_off + iq*8 + sm;\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
    ss << "          const int k_col = (int)tid.x * MFA_BK + sn + j*8;\n";
    ss << "          if (q_row < (k_col    )) Stile.frag_at(iq,j)[0] = -INFINITY;\n";
    ss << "          if (q_row < (k_col + 1)) Stile.frag_at(iq,j)[1] = -INFINITY;\n";
    ss << "        }\n";
    ss << "      }\n\n";
  }

  // Read L and delta per Q-row: one float per TQ fragment (one per 8 Q-rows)
  ss << "      // L_vals[iq] and delta_vals[iq]: one scalar per 8-row Q fragment\n";
  ss << "      float L_vals[MFA_TQ], delta_vals[MFA_TQ];\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short iq = 0; iq < MFA_TQ; iq++) {\n";
  ss << "        const long q_row_idx = (long)qb * MFA_BQ + iq*8 + sm;\n";
  ss << "        L_vals[iq]     = (q_row_idx < p->qL) ? L    [l_boff + q_row_idx] : 0.0f;\n";
  ss << "        delta_vals[iq] = (q_row_idx < p->qL) ? delta[l_boff + q_row_idx] : 0.0f;\n";
  ss << "      }\n\n";

  // P = exp2(S - L): per-fragment, use L_vals[iq]
  ss << "      // P[iq,ik] = exp2(S[iq,ik] - L_vals[iq])\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short iq = 0; iq < MFA_TQ; iq++)\n";
  ss << "        STEEL_PRAGMA_UNROLL\n";
  ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
  ss << "          Stile.frag_at(iq,j)[0] = fast::exp2(Stile.frag_at(iq,j)[0] - L_vals[iq]);\n";
  ss << "          Stile.frag_at(iq,j)[1] = fast::exp2(Stile.frag_at(iq,j)[1] - L_vals[iq]);\n";
  ss << "        }\n\n";

  // dV += P^T @ dO:
  // Step 1: scatter P^T to KV_smem as [BK x BQ] row-major.
  // Lane (sm,sn) of fragment (iq,ik) writes P[iq*8+sm, ik*8+sn] to
  // KV_smem[(ik*8+sn)*LDT + iq*8+sm] = P^T[ik*8+sn, iq*8+sm].
  ss << "      // Scatter P^T → KV_smem[BK x BQ] for dV = P^T @ dO\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short iq = 0; iq < MFA_TQ; iq++)\n";
  ss << "        STEEL_PRAGMA_UNROLL\n";
  ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
  ss << "          KV_smem[(j*8 + sn    ) * LDT + iq*8 + sm] = (T)Stile.frag_at(iq,j)[0];\n";
  ss << "          KV_smem[(j*8 + sn + 1) * LDT + iq*8 + sm] = (T)Stile.frag_at(iq,j)[1];\n";
  ss << "        }\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";

  // Step 2: Pt_tile[TK x TQ] from scatter → dV += Pt @ dO
  ss << "      MFAMMATile<AccT, MFA_TK, MFA_TQ> Pt_tile;\n";
  ss << "      Pt_tile.template load<T,1,1>(&KV_smem[sm * LDT + sn], LDT, 1);\n";
  if (!d_split) {
    ss << "      // dV[TK,TD] += Pt[TK,TQ] @ dO[TQ,TD]\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short iq = 0; iq < MFA_TQ; iq++)\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short ik = 0; ik < MFA_TK; ik++)\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short id = 0; id < MFA_TD; id++)\n";
    ss << "            MFAMMAFrag<AccT>::mma(\n";
    ss << "                dVtile.frag_at(ik,id),\n";
    ss << "                Pt_tile.frag_at(ik,iq),\n";
    ss << "                dOtile .frag_at(iq,id),\n";
    ss << "                dVtile.frag_at(ik,id));\n\n";
  } else {
    // d_split: dV[dh] += Pt @ dOtile[dh]  for each chunk dh
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short iq = 0; iq < MFA_TQ; iq++)\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short ik = 0; ik < MFA_TK; ik++)\n";
    ss << "            STEEL_PRAGMA_UNROLL\n";
    ss << "            for (short id = 0; id < MFA_TD_HALF; id++)\n";
    ss << "              MFAMMAFrag<AccT>::mma(\n";
    ss << "                  dVtile[dh].frag_at(ik,id), Pt_tile.frag_at(ik,iq),\n";
    ss << "                  dOtile[dh].frag_at(iq,id), dVtile[dh].frag_at(ik,id));\n";
    ss << "      }\n\n";
  }

  // Load V transposed → KV_smem; compute dP = dO @ V^T
  ss << "      // dP = dO @ V^T\n";
  ss << "      MFAMMATile<AccT, MFA_TQ, MFA_TK> dPtile;\n";
  ss << "      dPtile.clear();\n";
  if (!d_split) {
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "      KtLoader lvt(V, (int)p->V_strides[2], KV_smem, (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "      if ((int)tid.x == p->NK_aligned) lvt.load_safe(short2(MFA_BD, p->kL_rem));\n";
    ss << "      else                             lvt.load_unsafe();\n";
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short dd = 0; dd < MFA_TD; dd++) {\n";
    ss << "        Ktile.template load<T,1,1>(&KV_smem[Kts_off + (short)(dd*8)*LDKt], LDKt, 1);\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short iq = 0; iq < MFA_TQ; iq++)\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short ik = 0; ik < MFA_TK; ik++)\n";
    ss << "            MFAMMAFrag<AccT>::mma(dPtile.frag_at(iq,ik),\n";
    ss << "                dOtile.frag_at(iq,dd), Ktile.frag_at(0,ik), dPtile.frag_at(iq,ik));\n";
    ss << "      }\n\n";
  } else {
    // d_split: dP += dOtile[dh] @ V^T_dh  for each chunk dh
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "        KtLoader lvt_dh(V + dh * MFA_BD_HALF, (int)p->V_strides[2],\n";
    ss << "            KV_smem, (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "        if ((int)tid.x == p->NK_aligned) lvt_dh.load_safe(short2(MFA_BD_HALF, p->kL_rem));\n";
    ss << "        else                             lvt_dh.load_unsafe();\n";
    ss << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short dd = 0; dd < MFA_TD_HALF; dd++) {\n";
    ss << "          Ktile.template load<T,1,1>(&KV_smem[Kts_off + (short)(dd*8)*LDKt], LDKt, 1);\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short iq = 0; iq < MFA_TQ; iq++)\n";
    ss << "            STEEL_PRAGMA_UNROLL\n";
    ss << "            for (short ik = 0; ik < MFA_TK; ik++)\n";
    ss << "              MFAMMAFrag<AccT>::mma(dPtile.frag_at(iq,ik),\n";
    ss << "                  dOtile[dh].frag_at(iq,dd), Ktile.frag_at(0,ik), dPtile.frag_at(iq,ik));\n";
    ss << "        }\n";
    ss << "      }\n\n";
  }

  // Step 2: dS = scale * P * (dP - delta); Stile holds P
  ss << "      // dS[iq,ik] = scale * P[iq,ik] * (dP[iq,ik] - delta_vals[iq])\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short iq = 0; iq < MFA_TQ; iq++)\n";
  ss << "        STEEL_PRAGMA_UNROLL\n";
  ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
  ss << "          Stile.frag_at(iq,j)[0] = p->scale * Stile.frag_at(iq,j)[0]\n";
  ss << "                                 * (dPtile.frag_at(iq,j)[0] - delta_vals[iq]);\n";
  ss << "          Stile.frag_at(iq,j)[1] = p->scale * Stile.frag_at(iq,j)[1]\n";
  ss << "                                 * (dPtile.frag_at(iq,j)[1] - delta_vals[iq]);\n";
  ss << "        }\n\n";

  // dK += dS^T @ Q:
  // Step 1: scatter dS^T to KV_smem (same pattern as P^T scatter)
  ss << "      // Scatter dS^T → KV_smem[BK x BQ] for dK = dS^T @ Q\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short iq = 0; iq < MFA_TQ; iq++)\n";
  ss << "        STEEL_PRAGMA_UNROLL\n";
  ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
  ss << "          KV_smem[(j*8 + sn    ) * LDT + iq*8 + sm] = (T)Stile.frag_at(iq,j)[0];\n";
  ss << "          KV_smem[(j*8 + sn + 1) * LDT + iq*8 + sm] = (T)Stile.frag_at(iq,j)[1];\n";
  ss << "        }\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";

  // Step 2: dSt_tile[TK x TQ] → dK += dSt @ Q
  ss << "      MFAMMATile<AccT, MFA_TK, MFA_TQ> dSt_tile;\n";
  ss << "      dSt_tile.template load<T,1,1>(&KV_smem[sm * LDT + sn], LDT, 1);\n";
  if (!d_split) {
    ss << "      // dK[TK,TD] += dSt[TK,TQ] @ Q[TQ,TD]\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short iq = 0; iq < MFA_TQ; iq++)\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short ik = 0; ik < MFA_TK; ik++)\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short id = 0; id < MFA_TD; id++)\n";
    ss << "            MFAMMAFrag<AccT>::mma(\n";
    ss << "                dKtile.frag_at(ik,id), dSt_tile.frag_at(ik,iq),\n";
    ss << "                Qtile  .frag_at(iq,id), dKtile.frag_at(ik,id));\n\n";
  } else {
    // d_split: dK[dh] += dSt @ Qtile[dh]  for each chunk dh
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short iq = 0; iq < MFA_TQ; iq++)\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short ik = 0; ik < MFA_TK; ik++)\n";
    ss << "            STEEL_PRAGMA_UNROLL\n";
    ss << "            for (short id = 0; id < MFA_TD_HALF; id++)\n";
    ss << "              MFAMMAFrag<AccT>::mma(\n";
    ss << "                  dKtile[dh].frag_at(ik,id), dSt_tile.frag_at(ik,iq),\n";
    ss << "                  Qtile[dh].frag_at(iq,id), dKtile[dh].frag_at(ik,id));\n";
    ss << "      }\n\n";
  }

  ss << "    } // end qb loop\n";
  ss << "  } // end q_head loop\n\n";

  // Write dK and dV to device
  ss << "  threadgroup_barrier(mem_flags::mem_none);\n\n";
  if (!d_split) {
    ss << "  // Write dV\n";
    ss << "  dV += sm * (long)p->dV_strides[2] + sn;\n";
    ss << "  if ((int)tid.x == p->NK_aligned) {\n";
    ss << "    auto dims = short2((short)(MFA_BD - sn), (short)(p->kL_rem - sm));\n";
    ss << "    if (dims.x > 0 && dims.y > 0)\n";
    ss << "      dVtile.template store_safe<T,1,1>(dV, (int)p->dV_strides[2], dims);\n";
    ss << "  } else {\n";
    ss << "    dVtile.template store<T,1,1>(dV, (int)p->dV_strides[2]);\n";
    ss << "  }\n\n";
    ss << "  // Write dK\n";
    ss << "  dK += sm * (long)p->dK_strides[2] + sn;\n";
    ss << "  if ((int)tid.x == p->NK_aligned) {\n";
    ss << "    auto dims = short2((short)(MFA_BD - sn), (short)(p->kL_rem - sm));\n";
    ss << "    if (dims.x > 0 && dims.y > 0)\n";
    ss << "      dKtile.template store_safe<T,1,1>(dK, (int)p->dK_strides[2], dims);\n";
    ss << "  } else {\n";
    ss << "    dKtile.template store<T,1,1>(dK, (int)p->dK_strides[2]);\n";
    ss << "  }\n";
  } else {
    // d-split: write each BD_HALF-wide chunk at offset dh*BD_HALF
    ss << "  // Write dV (d_split: MFA_D_SPLITS chunks of BD_HALF cols)\n";
    ss << "  dV += sm * (long)p->dV_strides[2] + sn;\n";
    ss << "  STEEL_PRAGMA_UNROLL\n";
    ss << "  for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "    if ((int)tid.x == p->NK_aligned) {\n";
    ss << "      short rows_rem = (short)(p->kL_rem - sm);\n";
    ss << "      if (rows_rem > 0)\n";
    ss << "        dVtile[dh].template store_safe<T,1,1>(\n";
    ss << "            dV + dh * MFA_BD_HALF, (int)p->dV_strides[2],\n";
    ss << "            short2((short)(MFA_BD_HALF - sn), rows_rem));\n";
    ss << "    } else {\n";
    ss << "      dVtile[dh].template store<T,1,1>(dV + dh * MFA_BD_HALF, (int)p->dV_strides[2]);\n";
    ss << "    }\n";
    ss << "  }\n\n";
    ss << "  // Write dK (d_split: MFA_D_SPLITS chunks of BD_HALF cols)\n";
    ss << "  dK += sm * (long)p->dK_strides[2] + sn;\n";
    ss << "  STEEL_PRAGMA_UNROLL\n";
    ss << "  for (short dh = 0; dh < MFA_D_SPLITS; dh++) {\n";
    ss << "    if ((int)tid.x == p->NK_aligned) {\n";
    ss << "      short rows_rem = (short)(p->kL_rem - sm);\n";
    ss << "      if (rows_rem > 0)\n";
    ss << "        dKtile[dh].template store_safe<T,1,1>(\n";
    ss << "            dK + dh * MFA_BD_HALF, (int)p->dK_strides[2],\n";
    ss << "            short2((short)(MFA_BD_HALF - sn), rows_rem));\n";
    ss << "    } else {\n";
    ss << "      dKtile[dh].template store<T,1,1>(dK + dh * MFA_BD_HALF, (int)p->dK_strides[2]);\n";
    ss << "    }\n";
    ss << "  }\n";
  }
  ss << "}\n";

  return ss.str();
}

}  // namespace mlx_mfa
