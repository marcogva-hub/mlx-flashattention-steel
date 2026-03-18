/// mfa_steel_gna_fwd.cpp — JIT Metal shader generator for GNA forward kernel.
///
/// Generalized Neighborhood Attention: multi-dimensional windowed attention
/// with stride-based query grouping. The kernel iterates over ALL K-tiles
/// and uses an O(ndim) ND bounding-box overlap test to skip tiles outside
/// the query's GNA window (Approach A: simple, correct first kernel).

#include "mfa_steel_gna_fwd.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace mlx_mfa {

// ── Struct layout verification ───────────────────────────────────────────
// Metal reads the constant buffer as raw bytes. If C++ and Metal disagree
// on field offsets, the kernel reads garbage → NaN output.
static_assert(offsetof(MFAGNAParams, B)            ==   0, "B offset");
static_assert(offsetof(MFAGNAParams, H)            ==   4, "H offset");
static_assert(offsetof(MFAGNAParams, D)            ==   8, "D offset");
static_assert(offsetof(MFAGNAParams, seq_len)      ==  12, "seq_len offset");
static_assert(offsetof(MFAGNAParams, scale)         ==  16, "scale offset");
static_assert(offsetof(MFAGNAParams, gqa_factor)    ==  20, "gqa_factor offset");
static_assert(offsetof(MFAGNAParams, ndim)          ==  24, "ndim offset");
static_assert(offsetof(MFAGNAParams, seq_shape)     ==  28, "seq_shape offset");
static_assert(offsetof(MFAGNAParams, seq_strides)   ==  40, "seq_strides offset");
static_assert(offsetof(MFAGNAParams, window_size)   ==  52, "window_size offset");
static_assert(offsetof(MFAGNAParams, stride)        ==  64, "stride offset");
static_assert(offsetof(MFAGNAParams, window_volume) ==  76, "window_volume offset");
static_assert(offsetof(MFAGNAParams, NQ)            ==  80, "NQ offset");
static_assert(offsetof(MFAGNAParams, NK)            ==  84, "NK offset");
static_assert(offsetof(MFAGNAParams, NQ_aligned)    ==  88, "NQ_aligned offset");
static_assert(offsetof(MFAGNAParams, NK_aligned)    ==  92, "NK_aligned offset");
static_assert(offsetof(MFAGNAParams, qL_rem)        ==  96, "qL_rem offset");
static_assert(offsetof(MFAGNAParams, kL_rem)        == 100, "kL_rem offset");
static_assert(offsetof(MFAGNAParams, Q_strides)     == 104, "Q_strides offset");
static_assert(offsetof(MFAGNAParams, K_strides)     == 128, "K_strides offset");
static_assert(offsetof(MFAGNAParams, V_strides)     == 152, "V_strides offset");
static_assert(offsetof(MFAGNAParams, O_strides)     == 176, "O_strides offset");
static_assert(offsetof(MFAGNAParams, L_strides)     == 200, "L_strides offset");
static_assert(sizeof(MFAGNAParams)                  == 216, "MFAGNAParams total size");

SteelBlockConfig select_gna_block_config(int head_dim, bool is_low_prec) {
    // Reuse STEEL V1 defaults. GNA doesn't need double-buffer or D-split
    // complexity — the window skip provides the sparsity speedup.
    if (head_dim <= 64) {
        // BQ=32 BK=32 BD=64 WM=4 WN=1 PAD=8
        return {32, 32, head_dim, 4, 1, 16 / (is_low_prec ? 2 : 4)};
    }
    // D=128 (and D=256 falls back to sparse+mask, not this kernel)
    // BQ=32 BK=16 BD=128 WM=4 WN=1 PAD=8
    return {32, 16, head_dim, 4, 1, 16 / (is_low_prec ? 2 : 4)};
}

std::string generate_gna_forward_source(const ShaderCache::KernelKey& key) {
    const int BD = key.head_dim;
    const int BQ = key.block_q;
    const int BK = key.block_k;
    const int WM = key.n_warps;
    const int WN = 1;
    const int TGP = WM * WN * 32;

    const char* dtype_str = "half";
    if (key.dtype == 1) dtype_str = "bfloat";

    const int arch_gen = key.is_m3_plus ? 15 : 13;
    const bool enable_unroll = (BD <= 128) || key.is_m3_plus;
    // Debug env vars (compile-time only, not cached in KernelKey):
    // MFA_GNA_SKIP_WINDOW=1: skip ND window test (process all K-tiles)
    // MFA_DUMP_GNA_SHADER=1: write generated shader to /tmp/gna_shader.metal

    // Tile counts
    const int TQ = BQ / (WM * 8);  // frag rows per warp
    const int TK = BK / (WN * 8);  // frag cols per warp
    const int TD = BD / 8;          // D frags
    const int kRowsPT = TQ;  // rows per thread in S tile

    std::ostringstream ss;

    // ── Preamble ──────────────────────────────────────────────────────────
    append_metal_headers_and_defines(ss, enable_unroll, arch_gen, dtype_str);

    // ── MFAGNAParams struct (Metal side) ──────────────────────────────────
    ss << R"MFA(
struct MFAGNAParams {
  int B, H, D;
  int seq_len;
  float scale;
  int gqa_factor;
  int ndim;
  int seq_shape[3];
  int seq_strides[3];
  int window_size[3];
  int stride[3];
  int window_volume;
  int NQ, NK;
  int NQ_aligned, NK_aligned;
  int qL_rem, kL_rem;
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long O_strides[3];
  long L_strides[2];
};

)MFA";

    // ── Shared STEEL templates ────────────────────────────────────────────
    append_steel_shared_templates(ss);

    // ── Compile-time constants ────────────────────────────────────────────
    ss << "#define MFA_BQ   " << BQ << "\n";
    ss << "#define MFA_BK   " << BK << "\n";
    ss << "#define MFA_BD   " << BD << "\n";
    ss << "#define MFA_WM   " << WM << "\n";
    ss << "#define MFA_WN   " << WN << "\n";
    ss << "#define MFA_TQ   " << TQ << "\n";
    ss << "#define MFA_TK   " << TK << "\n";
    ss << "#define MFA_TD   " << TD << "\n";
    ss << "#define MFA_TGP_SIZE " << TGP << "\n";
    ss << "#define MFA_ROWS_PT  " << kRowsPT << "\n";
    ss << "typedef " << dtype_str << " MFA_DTYPE;\n";
    ss << "\n";

    // ── Kernel type aliases ───────────────────────────────────────────────
    const int PAD = 16 / (key.dtype == 2 ? 4 : 2);
    const int LDQ = BD + PAD;
    const int LDK = BK + PAD;
    const int LDV = BD + PAD;

    // ── Main kernel ───────────────────────────────────────────────────────
    ss << "[[kernel, max_total_threads_per_threadgroup(MFA_TGP_SIZE)]]\n";
    ss << "void mlx_mfa_gna_attention(\n";
    ss << "    const device MFA_DTYPE* Q    [[buffer(0)]],\n";
    ss << "    const device MFA_DTYPE* K    [[buffer(1)]],\n";
    ss << "    const device MFA_DTYPE* V    [[buffer(2)]],\n";
    ss << "    device MFA_DTYPE*       O    [[buffer(3)]],\n";
    ss << "    device float*           L    [[buffer(4)]],\n";
    ss << "    const constant MFAGNAParams* p [[buffer(5)]],\n";
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
    ss << "  Q += boff;\n";
    ss << "  K += kv_boff_k;\n";
    ss << "  V += kv_boff_v;\n";
    ss << "  O += (ulong)tid.z * p->O_strides[0]\n";
    ss << "     + (ulong)tid.y * p->O_strides[1];\n";
    ss << "\n";

    // Threadgroup memory: Q_smem + K_smem(transposed) + V_smem
    // Not using shared KV_smem (V2 style) — simpler is better for first GNA kernel.
    ss << "  constexpr short padQ  = 16 / sizeof(T);\n";
    ss << "  constexpr short padK  = 16 / sizeof(T);\n";
    ss << "  constexpr short padV  = 16 / sizeof(T);\n";
    ss << "  constexpr short LDQ   = MFA_BD + padQ;\n";
    ss << "  constexpr short LDK   = MFA_BK + padK;\n";
    ss << "  constexpr short LDV   = MFA_BD + padV;\n";
    // TGP layout: [Q_smem | K_smem | V_smem]
    ss << "  constexpr short kv_s0 = (MFA_BK + padK) * MFA_BD;\n";  // K^T smem
    ss << "  constexpr short kv_s1 = MFA_BK * (MFA_BD + padV);\n";  // V smem
    ss << "  threadgroup T Q_smem[MFA_BQ * LDQ];\n";
    ss << "  threadgroup T K_smem[kv_s0];\n";
    ss << "  threadgroup T V_smem[kv_s1];\n";
    ss << "  threadgroup T* Qs = Q_smem;\n";
    ss << "  threadgroup T* Ks = K_smem;\n";
    ss << "  threadgroup T* Vs = V_smem;\n";
    ss << "\n";

    // Q loader: row-major via BlockLoaderT (standard cooperative load)
    // K/V use inline 3D window loaders (not BlockLoaderT) — see window tile loop below
    ss << "  typedef MFABlockLoaderT<T, MFA_BQ, MFA_BD, LDQ, 1, 0, MFA_TGP_SIZE> QLoader;\n";
    ss << "\n";

    // Per-thread tile coordinates
    ss << "  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);\n";
    ss << "  const short sm = simd_coord.y;\n";
    ss << "  const short sn = simd_coord.x;\n";
    ss << "  const short tm = 8 * MFA_TQ * (short)simd_group_id;\n";
    ss << "\n";
    ss << "  const short Qs_off = (tm + sm) * LDQ + sn;\n";
    ss << "  const short Ks_off = sm * LDK + sn;\n";
    ss << "  const short Vs_off = sm * LDV + sn;\n";
    ss << "\n";

    // Register tiles
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Qtile;\n";
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Otile;\n";
    ss << "  MFAMMATile<AccT, 1,      MFA_TK> Ktile;\n";
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TK> Stile;\n";
    ss << "  MFAMMATile<AccT, 1, 1>           Vtile;\n";
    ss << "  AccT max_score[MFA_ROWS_PT];\n";
    ss << "  AccT sum_score[MFA_ROWS_PT];\n";
    ss << "\n";

    // ── Q-block loop (1 block per threadgroup, no persistent) ─────────────
    ss << "  const int qb = (int)tid.x;\n";
    ss << "  if (qb >= p->NQ) return;\n";
    ss << "\n";
    ss << "  const device T* Q_qb = Q + (long)qb * MFA_BQ * p->Q_strides[2];\n";
    ss << "  device T*       O_qb = O + (long)qb * MFA_BQ * p->O_strides[2];\n";
    ss << "\n";
    ss << "  QLoader loader_q(Q_qb, (int)p->Q_strides[2], Qs,\n";
    ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "\n";

    // Reset accumulators
    ss << "  Otile.clear();\n";
    ss << "  STEEL_PRAGMA_UNROLL\n";
    ss << "  for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
    ss << "    max_score[i] = -INFINITY;\n";
    ss << "    sum_score[i] = 0.0f;\n";
    ss << "  }\n";
    ss << "\n";

    // Load Q into SRAM, then into registers
    ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "  if (qb == p->NQ_aligned) {\n";
    ss << "    loader_q.load_safe(short2(MFA_BD, p->qL_rem));\n";
    ss << "  } else {\n";
    ss << "    loader_q.load_unsafe();\n";
    ss << "  }\n";
    ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    // Load Q from SRAM into registers (hoisted out of K loop)
    // Single call loads entire TQ×TD tile from the thread's starting offset.
    ss << "  Qtile.template load<T, 1, 1>(&Qs[Qs_off], LDQ, 1);\n";
    ss << "\n";

    // ── GNA window computation ────────────────────────────────────────────
    // Compute ND bounding box of the Q-tile and the union GNA window.
    ss << "  // Compute ND coordinates of Q-tile's first and last token\n";
    ss << "  const int q_start = qb * MFA_BQ;\n";
    ss << "  const int q_end   = min(q_start + MFA_BQ - 1, p->seq_len - 1);\n";
    ss << "  int q_start_nd[3], q_end_nd[3];\n";
    ss << "  int win_lo[3], win_hi[3];\n";
    ss << "  {\n";
    ss << "    int tmp_s = q_start, tmp_e = q_end;\n";
    ss << "    for (int d = 0; d < p->ndim; d++) {\n";
    ss << "      q_start_nd[d] = tmp_s / p->seq_strides[d];\n";
    ss << "      tmp_s        %= p->seq_strides[d];\n";
    ss << "      q_end_nd[d]   = tmp_e / p->seq_strides[d];\n";
    ss << "      tmp_e        %= p->seq_strides[d];\n";
    ss << "    }\n";
    ss << "    // Union of GNA windows across all stride-groups in this Q-tile\n";
    ss << "    for (int d = 0; d < p->ndim; d++) {\n";
    ss << "      const int s = p->stride[d];\n";
    ss << "      const int w = p->window_size[d];\n";
    ss << "      const int half_lo = (w - s) / 2;\n";
    ss << "      const int half_hi = (w - s + 1) / 2;\n";
    ss << "      const int group_min = q_start_nd[d] / s;\n";
    ss << "      const int group_max = q_end_nd[d]   / s;\n";
    ss << "      win_lo[d] = max(0, group_min * s - half_lo);\n";
    ss << "      win_hi[d] = min(p->seq_shape[d] - 1,\n";
    ss << "                      (group_max + 1) * s + half_hi - 1);\n";
    ss << "    }\n";
    ss << "  }\n";
    ss << "\n";

    // ── 3D strided window loader ─────────────────────────────────────────
    // Instead of iterating all NK linear tiles and testing ND overlap,
    // iterate ONLY over the window_volume tokens in (t, h, w) order.
    // The window decomposes into wT × wH contiguous segments of wW tokens.
    // Each segment's device address is predictable (constant strides),
    // enabling hardware prefetcher to follow the access pattern.

    // Window dimensions from clamped bounds
    ss << "  // Window dimensions\n";
    // For 2D sequences (ndim=2), we treat as 1×H×W: win_lo/hi[0] is H, [1] is W.
    // For 3D: win_lo/hi[0] is T, [1] is H, [2] is W.
    // Normalize to 3D: pad with T=0..0 for 2D.
    ss << "  int wT, wH, wW;\n";
    ss << "  int t0, h0, w0;\n";  // window start coords
    ss << "  int W_seq, HW_seq;\n";  // sequence strides in tokens
    ss << "  if (p->ndim == 3) {\n";
    ss << "    wT = win_hi[0] - win_lo[0] + 1;\n";
    ss << "    wH = win_hi[1] - win_lo[1] + 1;\n";
    ss << "    wW = win_hi[2] - win_lo[2] + 1;\n";
    ss << "    t0 = win_lo[0]; h0 = win_lo[1]; w0 = win_lo[2];\n";
    ss << "    W_seq  = p->seq_shape[2];\n";
    ss << "    HW_seq = p->seq_shape[1] * p->seq_shape[2];\n";
    ss << "  } else {\n";
    ss << "    wT = 1;\n";
    ss << "    wH = win_hi[0] - win_lo[0] + 1;\n";
    ss << "    wW = win_hi[1] - win_lo[1] + 1;\n";
    ss << "    t0 = 0; h0 = win_lo[0]; w0 = win_lo[1];\n";
    ss << "    W_seq  = p->seq_shape[1];\n";
    ss << "    HW_seq = p->seq_shape[0] * p->seq_shape[1];\n";
    ss << "  }\n";
    ss << "  const int window_vol = wT * wH * wW;\n";
    ss << "  const int num_win_tiles = (window_vol + MFA_BK - 1) / MFA_BK;\n";
    ss << "  const int D_val = p->D;\n";
    ss << "  const uint lid = simd_group_id * 32 + simd_lane_id;\n";
    ss << "\n";

    // Window tile loop
    ss << "  for (int wk = 0; wk < num_win_tiles; wk++) {\n";
    ss << "    const int tile_start = wk * MFA_BK;\n";
    ss << "    const int count = min(MFA_BK, window_vol - tile_start);\n";
    ss << "\n";

    // ── Cooperative K load: window → K_smem (transposed [D, BK] layout) ──
    // K_smem layout: Ks[d * LDK + k] where d ∈ [0, D), k ∈ [0, BK)
    // Each thread loads a subset of count × D elements.
    ss << "    // Cooperative K load: 3D window → K_smem (transposed)\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    // Zero the padding region and out-of-bounds positions
    ss << "    {\n";
    ss << "      const int total_k = count * D_val;\n";
    ss << "      for (uint idx = lid; idx < (uint)(MFA_BK * MFA_BD); idx += MFA_TGP_SIZE) {\n";
    ss << "        const int lk = idx % MFA_BK;\n";
    ss << "        const int ld = idx / MFA_BK;\n";
    ss << "        if (lk < count) {\n";
    ss << "          const int wp = tile_start + lk;\n";
    ss << "          const int lw = wp % wW;\n";
    ss << "          const int lh = (wp / wW) % wH;\n";
    ss << "          const int lt = wp / (wW * wH);\n";
    ss << "          const long src_off = (long)(t0 + lt) * HW_seq * D_val\n";
    ss << "                             + (long)(h0 + lh) * W_seq  * D_val\n";
    ss << "                             + (long)(w0 + lw) * D_val + ld;\n";
    ss << "          Ks[ld * LDK + lk] = K[src_off];\n";
    ss << "        } else {\n";
    ss << "          Ks[ld * LDK + lk] = (T)0;\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "\n";

    // S = Q @ K^T (Q in registers, K in SRAM) — IDENTICAL to V1
    ss << "    Stile.clear();\n";
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

    // Apply scale (log2-domain)
    ss << "    {\n";
    ss << "      const AccT scale = p->scale * M_LOG2E_F;\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++)\n";
    ss << "        Stile.elems()[ii] *= scale;\n";
    ss << "    }\n";
    ss << "\n";

    // Mask out-of-bounds columns in partial tiles (count < BK)
    // Columns [count, BK) should be -INFINITY so softmax ignores them.
    // The simdgroup_matrix layout maps column j to specific fragment elements.
    // For simplicity, mask at the fragment level: any frag column ik where
    // ik*8 >= count needs its elements set to -INFINITY.
    // More precisely, within each 8-wide fragment column ik, elements
    // corresponding to K-positions >= count need masking.
    ss << "    if (count < MFA_BK) {\n";
    ss << "      // Mask partial tile: set S columns >= count to -INFINITY\n";
    ss << "      // Each frag column ik covers K-positions [ik*8, (ik+1)*8)\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short iq = 0; iq < MFA_TQ; iq++) {\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short ik = 0; ik < MFA_TK; ik++) {\n";
    ss << "          if ((ik + 1) * 8 > count) {\n";
    ss << "            // This fragment column has some or all out-of-bounds\n";
    ss << "            // Set entire frag to -inf (conservative; at most 7 extra masked)\n";
    ss << "            Stile.frag_at(iq, ik)[0] = -INFINITY;\n";
    ss << "            Stile.frag_at(iq, ik)[1] = -INFINITY;\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";

    // ── Cooperative V load: window → V_smem (row-major [BK, D] layout) ──
    // V_smem layout: Vs[k * LDV + d] where k ∈ [0, BK), d ∈ [0, D)
    ss << "    // Cooperative V load: 3D window → V_smem (row-major)\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "    {\n";
    ss << "      for (uint idx = lid; idx < (uint)(MFA_BK * MFA_BD); idx += MFA_TGP_SIZE) {\n";
    ss << "        const int lk = idx / MFA_BD;\n";
    ss << "        const int ld = idx % MFA_BD;\n";
    ss << "        if (lk < count) {\n";
    ss << "          const int wp = tile_start + lk;\n";
    ss << "          const int lw = wp % wW;\n";
    ss << "          const int lh = (wp / wW) % wH;\n";
    ss << "          const int lt = wp / (wW * wH);\n";
    ss << "          const long src_off = (long)(t0 + lt) * HW_seq * D_val\n";
    ss << "                             + (long)(h0 + lh) * W_seq  * D_val\n";
    ss << "                             + (long)(w0 + lw) * D_val + ld;\n";
    ss << "          Vs[lk * LDV + ld] = V[src_off];\n";
    ss << "        } else {\n";
    ss << "          Vs[lk * LDV + ld] = (T)0;\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";

    // Online softmax update (NaN-safe, identical to STEEL V1)
    ss << "    // Online softmax (NaN-safe)\n";
    ss << "    AccT new_max[MFA_ROWS_PT];\n";
    ss << "    AccT factor[MFA_ROWS_PT];\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short i = 0; i < MFA_ROWS_PT; i++) new_max[i] = max_score[i];\n";
    ss << "    Stile.template row_reduce<MFAMaxOp>(new_max);\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
    ss << "      if (new_max[i] > max_score[i]) {\n";
    ss << "        factor[i] = fast::exp2(max_score[i] - new_max[i]);\n";
    ss << "        max_score[i] = new_max[i];\n";
    ss << "      } else {\n";
    ss << "        factor[i] = 1.0f;\n";
    ss << "        new_max[i] = metal::isinf(max_score[i]) ? (AccT)0.0f : max_score[i];\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "    Stile.template row_bin_op<MFAExpSubOp>(new_max);\n";
    ss << "    AccT sum_tmp[MFA_ROWS_PT] = {0};\n";
    ss << "    Stile.template row_reduce<MFASumOp>(sum_tmp);\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
    ss << "      sum_score[i] = sum_score[i] * factor[i] + sum_tmp[i];\n";
    ss << "    }\n";
    ss << "    Otile.template row_bin_op<MFAMulOp>(factor);\n";
    ss << "\n";

    // P @ V accumulation — IDENTICAL to V1
    ss << "    // O += P @ V\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
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

    // End window tile loop
    ss << "  } // end window tile loop\n";
    ss << "\n";

    // Normalize O by sum_score
    ss << "  // O /= sum_score\n";
    ss << "  Otile.template row_bin_op<MFADivOp>(sum_score);\n";
    ss << "  threadgroup_barrier(mem_flags::mem_none);\n";
    ss << "\n";

    // Write O to device memory
    ss << "  device T* O_write = O_qb + (long)(tm + sm) * p->O_strides[2] + sn;\n";
    ss << "  if (qb == p->NQ_aligned) {\n";
    ss << "    auto dims = short2((short)(MFA_BD - sn),\n";
    ss << "                       (short)(p->qL_rem - (tm + sm)));\n";
    ss << "    if (dims.x > 0 && dims.y > 0)\n";
    ss << "      Otile.template store_safe<T, 1, 1>(O_write, (int)p->O_strides[2], dims);\n";
    ss << "  } else {\n";
    ss << "    Otile.template store<T, 1, 1>(O_write, (int)p->O_strides[2]);\n";
    ss << "  }\n";
    ss << "\n";

    // Write L (logsumexp in log2 domain)
    ss << "  // Write L = max_score + log2(sum_score)\n";
    ss << "  if (sn == 0) {\n";
    ss << "    const long l_boff = (long)tid.z * p->L_strides[0]\n";
    ss << "                      + (long)tid.y * p->L_strides[1];\n";
    ss << "    const long q_base = (long)qb * MFA_BQ + tm + sm;\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
    ss << "      const long q_idx = q_base + i * 8;\n";
    ss << "      if (q_idx < p->seq_len) {\n";
    ss << "        L[l_boff + q_idx] = max_score[i] + metal::log2(sum_score[i]);\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "  }\n";
    ss << "}\n";

    auto result = ss.str();

    // Debug: dump generated shader to /tmp for inspection
    if (std::getenv("MFA_DUMP_GNA_SHADER")) {
        FILE* f = fopen("/tmp/gna_shader.metal", "w");
        if (f) { fprintf(f, "%s", result.c_str()); fclose(f); }
    }

    return result;
}

}  // namespace mlx_mfa
