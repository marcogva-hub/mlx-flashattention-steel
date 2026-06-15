/// mfa_gna_fwd.cpp — GNA forward kernel JIT generator.
///
/// Based on STEEL V2 (mfa_steel_fwd_v2.cpp) with:
///  - Sparse mask lookup replaced by inline GNA window check
///  - Stripped: causal, RoPE, softcap, ALiBi, sliding window, split-K
///  - D=128 only (all video DiT models)
///  - Non-causal only (video attention is non-causal)
///
/// The GNA window check converts tile indices to 3D bounding boxes and tests
/// overlap with the query tile's GNA window. This is ~20 integer ops per K-tile,
/// negligible vs the GEMM compute per active tile.

#include "mfa_gna_fwd.hpp"
#include "mfa_env.hpp"
#include "mfa_steel_fwd.hpp"       // append_metal_headers_and_defines, shared templates
#include "mfa_steel_fwd_v2.hpp"    // select_steel_v2_block_config

#include <sstream>
#include <cstdlib>

namespace mlx_mfa {

std::string generate_gna_forward_source(const ShaderCache::KernelKey& key) {
  const int D = key.head_dim;
  const int gqa = key.gqa_factor;
  const char* dtype_str = (key.dtype == 1) ? "bfloat" : "half";

  // MFA_NO_PADDING=1: set all smem padding to 0 (for benchmarking bank-conflict cost).
  const bool no_padding = MFAEnvConfig::no_padding();  // load-time frozen (repo review 2026-05)
  const std::string pad_expr = no_padding ? "0" : "16 / sizeof(T)";

  auto cfg = select_steel_v2_block_config(D, key.is_m3_plus);
  const int BQ = cfg.BQ;
  const int BK = cfg.BK;
  const int WM = cfg.WM;
  const int WN = 1;
  const int TGP_SIZE = WM * WN * 32;
  const int TD = D / 8;
  const int TK = BK / 8;
  const int TQ = BQ / (WM * WN * 8);

  const bool enable_unroll = (D <= 128) || key.is_m3_plus;
  const int arch_gen = key.is_m3_plus ? 15 : 13;

  // M3+ direct reads: bypass TGP for K/V.
  const bool use_direct_reads = key.is_m3_plus;

  std::ostringstream ss;

  // ── Metal preamble ──────────────────────────────────────────────────────
  append_metal_headers_and_defines(ss, enable_unroll, arch_gen, dtype_str);

  // ── Define block-size constants ─────────────────────────────────────────
  ss << "typedef " << dtype_str << " T;\n";
  ss << "typedef float AccT;\n";
  ss << "#define MFA_BQ  " << BQ  << "\n";
  ss << "#define MFA_BK  " << BK  << "\n";
  ss << "#define MFA_BD  " << D   << "\n";
  ss << "#define MFA_WM  " << WM  << "\n";
  ss << "#define MFA_WN  " << WN  << "\n";
  ss << "#define MFA_TGP_SIZE  " << TGP_SIZE << "\n";
  ss << "#define MFA_TD  " << TD  << "\n";
  ss << "#define MFA_TK  " << TK  << "\n";
  ss << "#define MFA_TQ  " << TQ  << "\n";
  ss << "#define MFA_GQA " << gqa << "\n";
  ss << "#define MFA_ROWS_PT " << TQ << "\n";
  ss << "#define MFA_DIRECT_READS " << (use_direct_reads ? 1 : 0) << "\n";
  ss << "\n";

  // ── Shared STEEL templates ──────────────────────────────────────────────
  append_steel_shared_templates(ss);

  // ── MFASteelParams (shared with V2 — same CPU/GPU layout) ──────────────
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
  int   has_attn_bias;
  int   attn_bias_mode;
  int   attn_bias_nkv;
};

)MFA";

  // ── GNA params struct ───────────────────────────────────────────────────
  ss << R"MFA(
struct MFAGNAParams {
  int dim0, dim1, dim2;
  int win0, win1, win2;
  int str0, str1, str2;
  int dim12;  // dim1 * dim2
};

)MFA";

  // ── GNA window check function ───────────────────────────────────────────
  // For a Q-tile (q_start..q_end) and K-tile (k_start..k_end), check if the
  // K-tile's 3D bounding box overlaps the Q-tile's GNA window.
  //
  // The Q-tile's window is the UNION of windows for all stride-groups that
  // the Q-tile spans — same logic as make_gna_mask() in Python.
  ss << R"MFA(
// Convert linear index to 3D coordinate component.
// For a sequence with shape (dim0, dim1, dim2) in row-major:
//   coord0 = idx / (dim1 * dim2)
//   coord1 = (idx / dim2) % dim1
//   coord2 = idx % dim2

// Check if K-tile [k_start, k_end) overlaps the GNA window of Q-tile [q_start, q_end).
inline bool gna_tile_active(
    int q_start, int q_end,
    int k_start, int k_end,
    constant MFAGNAParams& g)
{
    // Clamp to valid range
    q_end = min(q_end, g.dim0 * g.dim12);
    k_end = min(k_end, g.dim0 * g.dim12);
    if (q_start >= q_end || k_start >= k_end) return false;

    // Q-tile 3D bounding box (min/max per dimension)
    int q_first = q_start;
    int q_last  = q_end - 1;
    int q_min0 = q_first / g.dim12;
    int q_max0 = q_last  / g.dim12;
    int q_min1 = (q_first / g.dim2) % g.dim1;
    int q_max1 = (q_last  / g.dim2) % g.dim1;
    int q_min2 = q_first % g.dim2;
    int q_max2 = q_last  % g.dim2;

    // For tiles that span multiple rows, the min/max per dimension is the
    // full range, not just first/last token. Handle wrap-around:
    if (q_max0 > q_min0) {
        // Tile spans multiple dim0 slices → full range in dim1, dim2
        q_min1 = 0; q_max1 = g.dim1 - 1;
        q_min2 = 0; q_max2 = g.dim2 - 1;
    } else if (q_max1 > q_min1) {
        // Same dim0, spans multiple dim1 rows → full range in dim2
        q_min2 = 0; q_max2 = g.dim2 - 1;
    }

    // GNA window bounds (union of stride-groups in Q-tile)
    int half_lo0 = (g.win0 - g.str0) / 2;
    int half_hi0 = (g.win0 - g.str0 + 1) / 2;
    int half_lo1 = (g.win1 - g.str1) / 2;
    int half_hi1 = (g.win1 - g.str1 + 1) / 2;
    int half_lo2 = (g.win2 - g.str2) / 2;
    int half_hi2 = (g.win2 - g.str2 + 1) / 2;

    int grp_min0 = q_min0 / g.str0;
    int grp_max0 = q_max0 / g.str0;
    int grp_min1 = q_min1 / g.str1;
    int grp_max1 = q_max1 / g.str1;
    int grp_min2 = q_min2 / g.str2;
    int grp_max2 = q_max2 / g.str2;

    int win_lo0 = max(0, grp_min0 * g.str0 - half_lo0);
    int win_hi0 = min(g.dim0 - 1, (grp_max0 + 1) * g.str0 + half_hi0 - 1);
    int win_lo1 = max(0, grp_min1 * g.str1 - half_lo1);
    int win_hi1 = min(g.dim1 - 1, (grp_max1 + 1) * g.str1 + half_hi1 - 1);
    int win_lo2 = max(0, grp_min2 * g.str2 - half_lo2);
    int win_hi2 = min(g.dim2 - 1, (grp_max2 + 1) * g.str2 + half_hi2 - 1);

    // K-tile 3D bounding box
    int k_first = k_start;
    int k_last  = k_end - 1;
    int k_min0 = k_first / g.dim12;
    int k_max0 = k_last  / g.dim12;
    int k_min1 = (k_first / g.dim2) % g.dim1;
    int k_max1 = (k_last  / g.dim2) % g.dim1;
    int k_min2 = k_first % g.dim2;
    int k_max2 = k_last  % g.dim2;

    if (k_max0 > k_min0) {
        k_min1 = 0; k_max1 = g.dim1 - 1;
        k_min2 = 0; k_max2 = g.dim2 - 1;
    } else if (k_max1 > k_min1) {
        k_min2 = 0; k_max2 = g.dim2 - 1;
    }

    // Overlap test: all 3 dimensions must overlap
    return (k_max0 >= win_lo0 && k_min0 <= win_hi0)
        && (k_max1 >= win_lo1 && k_min1 <= win_hi1)
        && (k_max2 >= win_lo2 && k_min2 <= win_hi2);
}

)MFA";

  // ── Kernel function ─────────────────────────────────────────────────────
  ss << "[[kernel, max_total_threads_per_threadgroup(MFA_TGP_SIZE)]]\n";
  ss << "void mlx_mfa_gna_attention(\n";
  ss << "    const device T*             Q         [[buffer(0)]],\n";
  ss << "    const device T*             K         [[buffer(1)]],\n";
  ss << "    const device T*             V         [[buffer(2)]],\n";
  ss << "    device T*                   O         [[buffer(3)]],\n";
  ss << "    device float*               L         [[buffer(4)]],\n";
  ss << "    constant MFASteelParams*    p         [[buffer(5)]],\n";
  ss << "    constant MFAGNAParams*      gna       [[buffer(6)]],\n";
  ss << "    uint3 tid          [[threadgroup_position_in_grid]],\n";
  ss << "    uint  simd_group_id [[simdgroup_index_in_threadgroup]],\n";
  ss << "    uint  simd_lane_id  [[thread_index_in_simdgroup]])\n";
  ss << "{\n";

  // ── GQA head remapping ──────────────────────────────────────────────────
  ss << "  const int h_q  = (int)tid.y;\n";
  ss << "  const int h_kv = h_q / p->gqa_factor;\n";
  ss << "  Q += (long)tid.z * p->Q_strides[0] + (long)h_q  * p->Q_strides[1];\n";
  ss << "  K += (long)tid.z * p->K_strides[0] + (long)h_kv * p->K_strides[1];\n";
  ss << "  V += (long)tid.z * p->V_strides[0] + (long)h_kv * p->V_strides[1];\n";
  ss << "  O += (long)tid.z * p->O_strides[0] + (long)h_q * p->O_strides[1];\n";
  ss << "\n";

  // ── Threadgroup memory (matches V2 exactly) ─────────────────────────────
  ss << "  constexpr short padQ = " << pad_expr << ";\n";
  ss << "#if !MFA_DIRECT_READS\n";
  ss << "  constexpr short padK = " << pad_expr << ";\n";
  ss << "  constexpr short padV = " << pad_expr << ";\n";
  ss << "#endif\n";
  ss << "  constexpr short LDQ  = MFA_BD + padQ;\n";
  ss << "#if !MFA_DIRECT_READS\n";
  ss << "  constexpr short LDK  = MFA_BK + padK;  // stride for transposed K\n";
  ss << "  constexpr short LDV  = MFA_BD + padV;\n";
  ss << "  constexpr short kv_s0 = (MFA_BK + padK) * MFA_BD;\n";
  ss << "  constexpr short kv_s1 = MFA_BK * (MFA_BD + padV);\n";
  ss << "  constexpr short kv_s  = kv_s0 > kv_s1 ? kv_s0 : kv_s1;\n";
  ss << "  threadgroup T KV_smem[kv_s];\n";
  ss << "  threadgroup T* Ks = KV_smem;\n";
  ss << "  threadgroup T* Vs = KV_smem;\n";
  ss << "#endif\n";
  ss << "\n";
  ss << "  threadgroup T Q_smem[MFA_BQ * (MFA_BD + padQ)];\n";
  ss << "  threadgroup T* Qs = Q_smem;\n";
  ss << "\n";

  // ── Q-tile setup ────────────────────────────────────────────────────────
  ss << "  const int qb = (int)tid.x;\n";
  ss << "  device const T* Q_qb = Q + (long)(qb * MFA_BQ) * p->Q_strides[2];\n";
  ss << "  device T* O_qb = O + (long)(qb * MFA_BQ) * p->O_strides[2];\n";
  ss << "\n";

  // Load Q tile into threadgroup memory
  ss << "  // Q tile loader (row-major)\n";
  ss << "  {\n";
  ss << "    MFABlockLoaderT<T, MFA_BQ, MFA_BD,\n";
  ss << "        MFA_BD + " << (no_padding ? "0" : "16/(int)sizeof(T)") << ", 1, 1, MFA_TGP_SIZE>\n";
  ss << "        loader_q(Q_qb, (int)p->Q_strides[2], Qs, simd_group_id, simd_lane_id);\n";
  ss << "    if (qb == p->NQ_aligned) {\n";
  ss << "      loader_q.load_safe(short2(MFA_BD, p->qL_rem));\n";
  ss << "    } else {\n";
  ss << "      loader_q.load_unsafe();\n";
  ss << "    }\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "  }\n";
  ss << "\n";

  // ── SIMD coordinates (same as V2) ───────────────────────────────────────
  ss << "  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);\n";
  ss << "  const short sm = simd_coord.y;\n";
  ss << "  const short sn = simd_coord.x;\n";
  ss << "  const short tm = 8 * MFA_TQ * (short)simd_group_id;\n";
  ss << "\n";

  // Q tile in registers (single load — load_contiguous populates all TQ×TD fragments)
  ss << "  const short Qs_off = (tm + sm) * LDQ + sn;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Qtile;\n";
  ss << "  Qtile.template load_contiguous<T, 1, 1>(&Qs[Qs_off], LDQ);\n";
  ss << "\n";

  // Score, output tiles + running max/sum
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TK> Stile;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Otile;\n";
  ss << "  MFAMMATile<AccT, 1,      MFA_TK> Ktile;\n";
  ss << "  MFAMMATile<AccT, 1,      1>       Vtile;\n";
  ss << "  AccT max_score[MFA_ROWS_PT];\n";
  ss << "  AccT sum_score[MFA_ROWS_PT];\n";
  ss << "  STEEL_PRAGMA_UNROLL\n";
  ss << "  for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "    max_score[i] = -INFINITY;\n";
  ss << "    sum_score[i] = 0.0f;\n";
  ss << "  }\n";
  ss << "  Otile.clear();\n";
  ss << "\n";

  // Scale in log2 domain
  ss << "  const AccT scale = (AccT)p->scale * 1.4426950408889634f;\n";
  ss << "\n";

  // ── K/V loader setup ────────────────────────────────────────────────────
  ss << "  const int kb_start = 0;\n";
  ss << "  const int kb_lim   = p->NK;\n";
  ss << "\n";

  ss << "#if MFA_DIRECT_READS\n";
  ss << "  const long K_stride = p->K_strides[2];\n";
  ss << "  const long V_stride = p->V_strides[2];\n";
  ss << "  const device T* K_cur = K;\n";
  ss << "  const device T* V_cur = V;\n";
  ss << "#else\n";
  ss << "  const short Ks_off = sm * LDK + sn;\n";
  ss << "  const short Vs_off = sm * LDV + sn;\n";
  ss << "  // K preload: first active tile\n";
  ss << "  // We need to find the first active tile before entering the loop.\n";
  ss << "  // For simplicity, we preload tile 0 and skip inactive tiles in the loop.\n";
  // K: transposed into TGP (kDstStrRow=1, kDstStrCol=LDK)
  ss << "  using KLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      1, MFA_BK + " << (no_padding ? "0" : "16/(int)sizeof(T)") << ", 0, MFA_TGP_SIZE>;\n";
  // V: row-major, BK×BD tiles
  ss << "  using VLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      MFA_BD + " << (no_padding ? "0" : "16/(int)sizeof(T)") << ", 1, 0, MFA_TGP_SIZE>;\n";
  ss << "  KLoader loader_k(K, (int)p->K_strides[2], Ks, simd_group_id, simd_lane_id);\n";
  ss << "  VLoader loader_v(V, (int)p->V_strides[2], Vs, simd_group_id, simd_lane_id);\n";
  ss << "\n";

  // Preload first K tile
  ss << "  if (kb_start == p->NK_aligned) {\n";
  ss << "    loader_k.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "  } else {\n";
  ss << "    loader_k.load_unsafe();\n";
  ss << "  }\n";
  ss << "  loader_k.next();\n";
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "#endif\n";
  ss << "\n";

  // ── Main K/V loop ───────────────────────────────────────────────────────
  ss << "  for (int kb = kb_start; kb < kb_lim; kb++) {\n";
  ss << "\n";

  // GNA window check (replaces sparse mask lookup)
  ss << "    // GNA: skip K-tiles outside the query's attention window\n";
  ss << "    const bool skip_tile = !gna_tile_active(\n";
  ss << "        qb * MFA_BQ, (qb + 1) * MFA_BQ,\n";
  ss << "        kb * MFA_BK, (kb + 1) * MFA_BK,\n";
  ss << "        *gna);\n";
  ss << "    if (!skip_tile) {\n";

  // Phase 1: Q@K^T
  ss << "    Stile.clear();\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short dd = 0; dd < MFA_TD; dd++) {\n";
  ss << "#if MFA_DIRECT_READS\n";
  ss << "      Ktile.template load<T, 1, 1>(\n";
  ss << "          K_cur + (long)(sm + (short)(dd * 8)) + (long)sn * K_stride,\n";
  ss << "          1, K_stride);\n";
  ss << "#else\n";
  ss << "      Ktile.template load_contiguous<T, 1, 1>(\n";
  ss << "          &Ks[Ks_off + (short)(dd * 8) * LDK], LDK);\n";
  ss << "#endif\n";
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

  // Scale
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++) {\n";
  ss << "      Stile.elems()[ii] *= scale;\n";
  ss << "    }\n";
  ss << "\n";

  // K-boundary mask (pad positions → -inf)
  ss << "    if (kb == p->NK_aligned) {\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
  ss << "        STEEL_PRAGMA_UNROLL\n";
  ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
  ss << "          const short col = sn + j * 8;\n";
  ss << "          STEEL_PRAGMA_UNROLL\n";
  ss << "          for (short jj = 0; jj < 2; jj++) {\n";
  ss << "            if ((col + jj) >= p->kL_rem)\n";
  ss << "              Stile.frag_at(i, j)[jj] = -INFINITY;\n";
  ss << "          }\n";
  ss << "        }\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "\n";

  // Per-element GNA mask: tokens outside the query's EXACT window get -inf.
  // The tile-level check is conservative (bounding box overlap), so some
  // tokens within an active tile may still be outside the window.
  ss << "    // Per-element GNA mask: mask tokens outside exact window\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short i = 0; i < MFA_TQ; i++) {\n";
  ss << "      const int q_idx = qb * MFA_BQ + (int)tm + (int)sm + i * 8;\n";
  ss << "      // Compute this query's 3D position and window bounds\n";
  ss << "      const int q_c0 = q_idx / gna->dim12;\n";
  ss << "      const int q_c1 = (q_idx / gna->dim2) % gna->dim1;\n";
  ss << "      const int q_c2 = q_idx % gna->dim2;\n";
  ss << "      // Stride group\n";
  ss << "      const int g0 = q_c0 / gna->str0;\n";
  ss << "      const int g1 = q_c1 / gna->str1;\n";
  ss << "      const int g2 = q_c2 / gna->str2;\n";
  ss << "      // Window bounds\n";
  ss << "      const int lo0 = max(0, g0 * gna->str0 - (gna->win0 - gna->str0) / 2);\n";
  ss << "      const int hi0 = min(gna->dim0 - 1, (g0 + 1) * gna->str0 + (gna->win0 - gna->str0 + 1) / 2 - 1);\n";
  ss << "      const int lo1 = max(0, g1 * gna->str1 - (gna->win1 - gna->str1) / 2);\n";
  ss << "      const int hi1 = min(gna->dim1 - 1, (g1 + 1) * gna->str1 + (gna->win1 - gna->str1 + 1) / 2 - 1);\n";
  ss << "      const int lo2 = max(0, g2 * gna->str2 - (gna->win2 - gna->str2) / 2);\n";
  ss << "      const int hi2 = min(gna->dim2 - 1, (g2 + 1) * gna->str2 + (gna->win2 - gna->str2 + 1) / 2 - 1);\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short j = 0; j < MFA_TK; j++) {\n";
  ss << "        STEEL_PRAGMA_UNROLL\n";
  ss << "        for (short jj = 0; jj < 2; jj++) {\n";
  ss << "          const int k_idx = kb * MFA_BK + (int)sn + j * 8 + jj;\n";
  ss << "          if (k_idx < p->kL) {\n";
  ss << "            const int k_c0 = k_idx / gna->dim12;\n";
  ss << "            const int k_c1 = (k_idx / gna->dim2) % gna->dim1;\n";
  ss << "            const int k_c2 = k_idx % gna->dim2;\n";
  ss << "            if (k_c0 < lo0 || k_c0 > hi0 || k_c1 < lo1 || k_c1 > hi1 || k_c2 < lo2 || k_c2 > hi2)\n";
  ss << "              Stile.frag_at(i, j)[jj] = -INFINITY;\n";
  ss << "          }\n";
  ss << "        }\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "\n";

  // Online softmax (NaN-safe)
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

  // Barrier A + V load
  ss << "#if !MFA_DIRECT_READS\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "    if (kb == p->NK_aligned) {\n";
  ss << "      loader_v.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "    } else {\n";
  ss << "      loader_v.load_unsafe();\n";
  ss << "    }\n";
  ss << "    loader_v.next();\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "#endif\n";
  ss << "\n";

  // Phase 3: O += P @ V
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short iq = 0; iq < MFA_TQ; iq++) {\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short ik = 0; ik < MFA_TK; ik++) {\n";
  ss << "        STEEL_PRAGMA_UNROLL\n";
  ss << "        for (short id = 0; id < MFA_TD; id++) {\n";
  ss << "#if MFA_DIRECT_READS\n";
  ss << "          // Partial final K-tile: clamp the key-row to the last valid\n";
  ss << "          // key so the unbounded direct device read cannot return OOB\n";
  ss << "          // NaN/stale-pool data for masked keys (P=0 there, but\n";
  ss << "          // 0*NaN=NaN would corrupt O). See III-9 V2 fix (eb68af5);\n";
  ss << "          // §AA.5.x multi-gate — same pattern as V2 single-pass.\n";
  ss << "          short v_row = sm + (short)(ik * 8);\n";
  ss << "          if (kb == p->NK_aligned && v_row >= p->kL_rem)\n";
  ss << "            v_row = p->kL_rem - 1;\n";
  ss << "          Vtile.template load<T, 1, 1>(\n";
  ss << "              V_cur + (long)v_row * V_stride\n";
  ss << "                    + sn + (short)(id * 8),\n";
  ss << "              V_stride, 1);\n";
  ss << "#else\n";
  ss << "          Vtile.template load_contiguous<T, 1, 1>(\n";
  ss << "              &Vs[Vs_off + ik*8*LDV + id*8], LDV);\n";
  ss << "#endif\n";
  ss << "          MFAMMAFrag<AccT>::mma(\n";
  ss << "              Otile.frag_at(iq, id),\n";
  ss << "              Stile.frag_at(iq, ik),\n";
  ss << "              Vtile.frag_at(0, 0),\n";
  ss << "              Otile.frag_at(iq, id));\n";
  ss << "        }\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "\n";

  // Close GNA skip block; advance VLoader in skip case
  ss << "    } else {\n";
  ss << "#if !MFA_DIRECT_READS\n";
  ss << "      loader_v.next();\n";
  ss << "#endif\n";
  ss << "    }\n";
  ss << "\n";

  // Advance K/V
  ss << "#if MFA_DIRECT_READS\n";
  ss << "    K_cur += (long)MFA_BK * K_stride;\n";
  ss << "    V_cur += (long)MFA_BK * V_stride;\n";
  ss << "#else\n";
  ss << "    if (kb + 1 < kb_lim) {\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "      if ((kb + 1) == p->NK_aligned) {\n";
  ss << "        loader_k.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "      } else {\n";
  ss << "        loader_k.load_unsafe();\n";
  ss << "      }\n";
  ss << "      loader_k.next();\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "    }\n";
  ss << "#endif\n";
  ss << "\n";

  ss << "  } // end kb loop\n";
  ss << "\n";

  // ── Normalize and write O ───────────────────────────────────────────────
  ss << "  Otile.template row_bin_op<MFADivOp>(sum_score);\n";
  ss << "  threadgroup_barrier(mem_flags::mem_none);\n";
  ss << "\n";
  ss << "  device T* O_write = O_qb + (long)(tm + sm) * p->O_strides[2] + sn;\n";
  ss << "  if (qb == p->NQ_aligned) {\n";
  ss << "    auto dims = short2((short)(MFA_BD - sn),\n";
  ss << "                       (short)(p->qL_rem - (tm + sm)));\n";
  ss << "    if (dims.x > 0 && dims.y > 0)\n";
  ss << "      Otile.template store_safe<T, 1, 1>(O_write, (int)p->O_strides[2], dims);\n";
  ss << "  } else {\n";
  ss << "    Otile.template store_contiguous<T, 1, 1>(O_write, (int)p->O_strides[2]);\n";
  ss << "  }\n";
  ss << "\n";

  // ── Write L (logsumexp) ─────────────────────────────────────────────────
  ss << "  if (sn == 0) {\n";
  ss << "    const long l_boff = (long)tid.z * p->L_strides[0]\n";
  ss << "                      + (long)tid.y * p->L_strides[1];\n";
  ss << "    const long q_base = (long)qb * MFA_BQ + tm + sm;\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "      const long q_idx = q_base + i * 8;\n";
  ss << "      if (q_idx < p->qL) {\n";
  ss << "        L[l_boff + q_idx] = max_score[i] + metal::log2(sum_score[i]);\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "  }\n";
  ss << "\n";

  ss << "}\n";

  return ss.str();
}

}  // namespace mlx_mfa
