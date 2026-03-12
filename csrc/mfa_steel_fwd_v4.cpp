/// mfa_steel_fwd_v4.cpp  –  STEEL V4 forward kernel: direct device K reads.
///
/// Key difference from V2/V3: no K_smem.  K fragments are loaded directly from
/// device memory per simdgroup in the GEMM loop.  On M3+ the L2 cache absorbs
/// the 4× (WM=4 simdgroups) redundant reads via cache hits.
///
/// Barrier schedule vs V2 (D=128 NK=128 tiles example):
///   V2:  B0 + 127×(A+B+X+C) + 2 = 510 barriers  [4/tile + 1 preload]
///   V4:  B0 + 127×(A+B) = 255 barriers            [2/tile + 1 preload]
///   Savings: ~50% fewer barriers.
///
/// V3 comparison:
///   V3 also has 2 barriers/tile but carries separate K_smem (27 KB for D=128).
///   V4 saves K_smem (17 KB for D=128) while matching V3's barrier count.
///
/// RoPE-K is NOT supported: K is read raw from device without a TGP staging
/// area.  Kernels with has_rope=true fall back to V2 (which applies RoPE in
/// K_smem before the GEMM).

#include "mfa_steel_fwd.hpp"
#include "mfa_steel_fwd_v4.hpp"
#include <sstream>

namespace mlx_mfa {

std::string generate_steel_v4_source(const ShaderCache::KernelKey& key) {
  using KK = ShaderCache::KernelKey;

  const int D            = key.head_dim;
  const bool causal      = key.causal;
  const bool has_softcap = key.has_softcap;
  const bool has_window  = key.has_window;
  const bool has_alibi   = key.has_alibi;
  const bool sparse      = key.sparse;
  const int gqa          = key.gqa_factor;

  const char* dtype_str = (key.dtype == 1) ? "bfloat" : "half";

  // V4 uses the same block config as V2.
  auto cfg = select_steel_v4_block_config(D, key.is_m3_plus);
  const int BQ = cfg.BQ;   // 32
  const int BK = cfg.BK;   // 64 (D=64) | 32 M1/M2 D=128 | 64 M3+ D=128
  const int WM = cfg.WM;   // 4
  const int WN = 1;
  const int TGP_SIZE = WM * WN * 32;  // 128
  const int TD  = D / 8;       // 8 (D=64) | 16 (D=128)
  const int TK  = BK / 8;      // 8 (D=64 BK=64) | 4 (D=128 BK=32) | 8 (D=128 BK=64)
  const int TQ  = BQ / (WM * WN * 8);  // always 1

  const bool enable_unroll = (D <= 128);
  const int  arch_gen      = key.is_m3_plus ? 15 : 13;

  std::ostringstream ss;

  // ── Metal preamble ────────────────────────────────────────────────────────
  append_metal_headers_and_defines(ss, enable_unroll, arch_gen, dtype_str);

  // ── V4 block-size constants ───────────────────────────────────────────────
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
  ss << "\n";

  // ── Shared BlockLoaderT + MMATile templates ───────────────────────────────
  append_steel_shared_templates(ss);

  // ── MFASteelParams struct (shared CPU/GPU layout) ─────────────────────────
  ss << R"MFA(
struct MFASteelParams {
  int B, H, D, qL, kL;
  int gqa_factor;
  float scale;
  int NQ, NK;
  int NQ_aligned, NK_aligned;
  int qL_rem, kL_rem;
  int qL_off;
  int rope_q_base;
  int rope_cos_stride;
  long Q_strides[3], K_strides[3], V_strides[3], O_strides[3], L_strides[2];
  float softcap;
  int has_alibi;
  int window_left, window_right;
  long mask_batch_stride, mask_head_stride;
};
)MFA";

  // ── Kernel signature (name = v4) ─────────────────────────────────────────
  ss << "kernel void mlx_mfa_v4_attention(\n";
  ss << "    const device T* Q         [[buffer(0)]],\n";
  ss << "    const device T* K         [[buffer(1)]],\n";
  ss << "    const device T* V         [[buffer(2)]],\n";
  ss << "    device T*       O         [[buffer(3)]],\n";
  ss << "    device float*   L         [[buffer(4)]],\n";
  ss << "    constant MFASteelParams* p [[buffer(5)]],\n";
  if (sparse) {
    ss << "    const device uchar* block_mask [[buffer(6)]],\n";
  }
  if (has_alibi) {
    ss << "    const device float* alibi_slopes [[buffer(9)]],\n";
  }
  ss << "    uint3 tid    [[threadgroup_position_in_grid]],\n";
  ss << "    uint  simd_group_id [[simdgroup_index_in_threadgroup]],\n";
  ss << "    uint  simd_lane_id  [[thread_index_in_simdgroup]])\n";
  ss << "{\n";

  // ── GQA head mapping ─────────────────────────────────────────────────────
  ss << "  const int h_q  = (int)tid.y;\n";
  ss << "  const int h_kv = (MFA_GQA == 1) ? h_q : (h_q / MFA_GQA);\n";
  ss << "\n";

  // Advance base pointers to (batch, head) slice
  ss << "  Q += (long)tid.z * p->Q_strides[0] + (long)h_q  * p->Q_strides[1];\n";
  ss << "  K += (long)tid.z * p->K_strides[0] + (long)h_kv * p->K_strides[1];\n";
  ss << "  V += (long)tid.z * p->V_strides[0] + (long)h_kv * p->V_strides[1];\n";
  ss << "  O += (long)tid.z * p->O_strides[0] + (long)h_q * p->O_strides[1];\n";
  ss << "\n";

  // ── Threadgroup memory: Q_smem + V_smem only (no K_smem in V4!) ──────────
  // V4 key change: K is read directly from device in the GEMM loop.
  // TGP = Q_smem + V_smem only.
  //   D=64  BK=64: Q(32×72×2=4,608) + V(64×72×2=9,216)  = 13,824 B
  //   D=128 BK=32: Q(32×136×2=8,704) + V(32×136×2=8,704) = 17,408 B
  //   D=128 BK=64: Q(32×136×2=8,704) + V(64×136×2=17,408)= 26,112 B
  ss << "  constexpr short padQ = 16 / sizeof(T);\n";
  ss << "  constexpr short padV = 16 / sizeof(T);\n";
  ss << "  constexpr short LDQ  = MFA_BD + padQ;\n";
  ss << "  constexpr short LDV  = MFA_BD + padV;\n";
  ss << "\n";
  ss << "  // V4: only Q and V in threadgroup memory — K from device.\n";
  ss << "  threadgroup T Q_smem[MFA_BQ * (MFA_BD + padQ)];\n";
  ss << "  threadgroup T V_smem[MFA_BK * (MFA_BD + padV)];\n";
  ss << "  threadgroup T* Qs = Q_smem;\n";
  ss << "  threadgroup T* Vs = V_smem;\n";
  ss << "\n";

  // ── K_stride for direct device reads ─────────────────────────────────────
  // K is [S, D] row-major; K[s, d] at offset s * K_stride + d.
  // K_stride = p->K_strides[2] = D.
  // For K^T[d, s] read: ptr = K + d + s * K_stride, row_stride=1, col_stride=K_stride.
  ss << "  // V4: K stride for direct device reads\n";
  ss << "  const int K_stride = (int)p->K_strides[2];  // = D\n";
  ss << "  // Save K base before any sliding window advance\n";
  ss << "  const device T* K_base = K;\n";
  ss << "\n";

  // ── Block loaders (Q and V only — no KLoader) ────────────────────────────
  ss << "  using QLoader = MFABlockLoaderT<T, MFA_BQ, MFA_BD,\n";
  ss << "      MFA_BD + 16/sizeof(T), 1, 1, MFA_TGP_SIZE>;\n";
  ss << "  using VLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      MFA_BD + 16/sizeof(T), 1, 0, MFA_TGP_SIZE>;\n";
  ss << "\n";

  // ── SIMD coordinates ─────────────────────────────────────────────────────
  ss << "  const AccT scale = p->scale * M_LOG2E_F;\n";
  ss << "\n";
  ss << "  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);\n";
  ss << "  const short sm = simd_coord.y;\n";
  ss << "  const short sn = simd_coord.x;\n";
  ss << "  const short tm = 8 * MFA_TQ * (short)simd_group_id;\n";
  ss << "\n";
  ss << "  const short Qs_off = (tm + sm) * LDQ + sn;\n";
  ss << "  const short Vs_off = sm * LDV + sn;\n";
  ss << "\n";

  // ── Tile registers ───────────────────────────────────────────────────────
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Qtile;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Otile;\n";
  ss << "  MFAMMATile<AccT, 1,      MFA_TK>  Ktile;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TK>  Stile;\n";
  ss << "  MFAMMATile<AccT, 1,      1>        Vtile;\n";
  ss << "  AccT max_score[MFA_ROWS_PT];\n";
  ss << "  AccT sum_score[MFA_ROWS_PT];\n";
  ss << "\n";

  // ── One Q-block per threadgroup ──────────────────────────────────────────
  ss << "  const int qb = (int)tid.x;\n";
  ss << "\n";
  ss << "  const device T* Q_qb = Q + (long)qb * MFA_BQ * p->Q_strides[2];\n";
  ss << "  device T*       O_qb = O + (long)qb * MFA_BQ * p->O_strides[2];\n";
  ss << "\n";

  // ── Sliding window: compute kb_start O(1) ────────────────────────────────
  // V advances the same as V2; K_base is saved before this so direct K reads
  // still index from the original K pointer.
  if (has_window) {
    ss << "  int kb_start  = 0;\n";
    ss << "  int kb_last_win = -1;\n";
    ss << "  if (p->window_left >= 0) {\n";
    ss << "    const int q_min = qb * MFA_BQ + p->qL_off;\n";
    ss << "    const int win_start = q_min - p->window_left;\n";
    ss << "    kb_start = win_start > 0 ? win_start / MFA_BK : 0;\n";
    ss << "    kb_last_win = (q_min + MFA_BQ - 1 > p->window_left)\n";
    ss << "                    ? (q_min + MFA_BQ - 1 - p->window_left) / MFA_BK : -1;\n";
    ss << "  }\n";
    // Advance V only (K reads use K_base directly, no advance needed).
    ss << "  V += (long)kb_start * MFA_BK * p->V_strides[2];\n";
    ss << "\n";
  } else {
    ss << "  const int kb_start  = 0;\n";
    ss << "  const int kb_last_win = -1;\n";
    ss << "\n";
  }

  // ── Create loaders (VLoader only) ────────────────────────────────────────
  ss << "  QLoader loader_q(Q_qb, (int)p->Q_strides[2], Qs,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "  VLoader loader_v(V, (int)p->V_strides[2], Vs,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "\n";

  // ── Reset accumulators ───────────────────────────────────────────────────
  ss << "  Otile.clear();\n";
  ss << "  STEEL_PRAGMA_UNROLL\n";
  ss << "  for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "    max_score[i] = -INFINITY;\n";
  ss << "    sum_score[i] = 0.0f;\n";
  ss << "  }\n";
  ss << "\n";

  // ── Load Q tile ──────────────────────────────────────────────────────────
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "  if (qb == p->NQ_aligned) {\n";
  ss << "    loader_q.load_safe(short2(MFA_BD, p->qL_rem));\n";
  ss << "  } else {\n";
  ss << "    loader_q.load_unsafe();\n";
  ss << "  }\n";
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "  Qtile.template load<T, 1, 1>(&Qs[Qs_off], LDQ, 1);\n";
  ss << "\n";

  // ── K-loop limits ────────────────────────────────────────────────────────
  if (causal) {
    ss << "  int q_max  = (qb + 1) * MFA_BQ + p->qL_off;\n";
    ss << "  int kb_lim = (q_max + MFA_BK - 1) / MFA_BK;\n";
    ss << "  if (kb_lim > p->NK) kb_lim = p->NK;\n";
  } else {
    ss << "  int kb_lim = p->NK;\n";
  }
  if (has_window) {
    ss << "  int kb_first_right;\n";
    ss << "  if (p->window_right >= 0) {\n";
    ss << "    const int q_min = qb * MFA_BQ + p->qL_off;\n";
    ss << "    int kb_right_lim = (q_min + MFA_BQ - 1 + p->window_right) / MFA_BK + 1;\n";
    ss << "    if (kb_right_lim < kb_lim) kb_lim = kb_right_lim;\n";
    ss << "    if (kb_lim < kb_start) kb_lim = kb_start;\n";
    ss << "    kb_first_right = (q_min + p->window_right + 1) / MFA_BK;\n";
    ss << "    if (kb_first_right < kb_start) kb_first_right = kb_start;\n";
    ss << "  } else {\n";
    ss << "    kb_first_right = kb_lim;\n";
    ss << "  }\n";
  } else {
    ss << "  const int kb_first_right = kb_lim;\n";
  }
  ss << "\n";

  // ── V4 Phase 0: Preload V[kb_start] before main loop ─────────────────────
  // Unlike V2/V3, we do NOT preload K (it's read from device on the fly).
  // We only preload V[kb_start] to align with the in-loop V preload pattern.
  ss << "  // V4: preload V[kb_start] into V_smem before the loop.\n";
  ss << "  // K is read directly from device — no K preload needed.\n";
  ss << "  if (kb_lim > kb_start) {\n";
  ss << "    if (kb_start == p->NK_aligned) {\n";
  ss << "      loader_v.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "    } else {\n";
  ss << "      loader_v.load_unsafe();\n";
  ss << "    }\n";
  ss << "    loader_v.next();\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // B0: V[0] visible\n";
  ss << "  }\n";
  ss << "\n";

  // ── Main K-tile loop ─────────────────────────────────────────────────────
  ss << "  for (int kb = kb_start; kb < kb_lim; kb++) {\n";
  ss << "\n";

  // V4: compute device pointer for K[kb] tile.
  // K_base is head-adjusted, K[kb] starts at kb*BK*K_stride.
  ss << "    // V4: K[kb] read directly from device (no TGP K_smem).\n";
  ss << "    const device T* K_kb = K_base + (long)kb * MFA_BK * K_stride;\n";
  ss << "\n";

  // Sparse tile-skip
  if (sparse) {
    ss << "    const bool skip_tile = !block_mask[\n";
    ss << "        (long)tid.z * p->mask_batch_stride\n";
    ss << "      + (long)tid.y * p->mask_head_stride\n";
    ss << "      + (long)qb * p->NK + kb];\n";
    ss << "    if (!skip_tile) {\n";
  }

  // ── Phase 1: Q@K^T (load K from device memory per simdgroup) ─────────────
  // Addresses K^T[d, s] in row-major K[S, D]:
  //   K[s, d] = K_kb + d + s * K_stride
  //   K^T[d, s] = same element, row_stride=1 (d), col_stride=K_stride (s)
  // Each of the WM=4 simdgroups reads K independently; L2 cache absorbs redundancy.
  ss << "    // ─ Phase 1: Q@K^T (device K, no barrier needed) ─\n";
  ss << "    Stile.clear();\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short dd = 0; dd < MFA_TD; dd++) {\n";
  ss << "      // K^T[sm+dd*8, sn]: load K fragment from device memory.\n";
  ss << "      // Row=head_dim (stride=1), Col=sequence (stride=K_stride).\n";
  ss << "      Ktile.template load<T, 1, 1>(\n";
  ss << "          K_kb + (sm + (short)(dd * 8)) + (long)sn * K_stride,\n";
  ss << "          1, K_stride);\n";
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

  // Softcap
  if (has_softcap) {
    ss << "    {\n";
    ss << "      constexpr AccT log2e = 1.4426950408889634f;\n";
    ss << "      constexpr AccT ln2   = 0.6931471805599453f;\n";
    ss << "      const AccT cap = p->softcap;\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++) {\n";
    ss << "        AccT s_nat = Stile.elems()[ii] * ln2;\n";
    ss << "        s_nat = precise::tanh(s_nat / cap) * cap;\n";
    ss << "        Stile.elems()[ii] = s_nat * log2e;\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";
  }

  // ALiBi
  if (has_alibi) {
    ss << "    {\n";
    ss << "      constexpr AccT log2e = 1.4426950408889634f;\n";
    ss << "      const AccT slope = alibi_slopes[(int)tid.y] * log2e;\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        const int q_pos = qb * MFA_BQ + p->qL_off + (int)tm + (int)sm + i * 8;\n";
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

  // Causal mask
  if (causal) {
    ss << "    if (kb >= (kb_lim - (MFA_BQ + MFA_BK - 1) / MFA_BK)) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        const int row = qb * MFA_BQ + p->qL_off + tm + sm + i * 8;\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
    ss << "          const int col = kb * MFA_BK + sn + j * 8;\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short jj = 0; jj < 2; jj++) {\n";
    ss << "            if (row < (col + jj))\n";
    ss << "              Stile.frag_at(i, j)[jj] = -INFINITY;\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";
  }

  // Sliding window masking
  if (has_window) {
    ss << "    if (kb <= kb_last_win) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        const int row = qb * MFA_BQ + p->qL_off + tm + sm + i * 8;\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
    ss << "          const int col = kb * MFA_BK + sn + j * 8;\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short jj = 0; jj < 2; jj++) {\n";
    ss << "            if ((col + jj) < row - p->window_left)\n";
    ss << "              Stile.frag_at(i,j)[jj] = -INFINITY;\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "    if (kb >= kb_first_right) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        const int row = qb * MFA_BQ + p->qL_off + tm + sm + i * 8;\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
    ss << "          const int col = kb * MFA_BK + sn + j * 8;\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short jj = 0; jj < 2; jj++) {\n";
    ss << "            if ((col + jj) > row + p->window_right)\n";
    ss << "              Stile.frag_at(i,j)[jj] = -INFINITY;\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";
  }

  // Online softmax (same as V2/V3)
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

  // ── Phase 2: P@V (V[kb] in V_smem, preloaded before loop) ───────────────
  // V4 note: V_smem has V[kb] ready (from preload for kb=kb_start, from
  // Barrier B for subsequent tiles). No transition barrier needed here.
  ss << "    // ─ Phase 2: P@V (V[kb] in V_smem, ready via B0 or prev-iter B) ─\n";
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

  // Sparse: advance VLoader on skip (keeps VLoader position in sync)
  if (sparse) {
    ss << "    } else {\n";
    ss << "      loader_v.next();  // sparse skip: advance VLoader for sync\n";
    ss << "    }\n";
    ss << "\n";
  }

  // ── V4 Barrier A + V preload for next tile ────────────────────────────────
  // Barrier A: P@V reads on V_smem are done → safe to overwrite with V[kb+1].
  // Load V[kb+1] → V_smem.
  // Barrier B: V[kb+1] written and visible for next Q@K^T/P@V.
  //
  // K[kb+1] needs no load (direct device read in next iteration).
  // Net barriers: A + B = 2 per tile (vs V2's A+B+X+C = 4 per tile).
  ss << "    // ─ V4 Barrier A: V_smem reads done → safe to overwrite with V[kb+1] ─\n";
  ss << "    // ─ V4 Barrier B: V[kb+1] written → visible for next iter ─\n";
  ss << "    if (kb + 1 < kb_lim) {\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // A: V reads done\n";
  ss << "      if ((kb + 1) == p->NK_aligned) {\n";
  ss << "        loader_v.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "      } else {\n";
  ss << "        loader_v.load_unsafe();\n";
  ss << "      }\n";
  ss << "      loader_v.next();\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // B: V[kb+1] written\n";
  ss << "    }\n";
  ss << "\n";

  ss << "  } // end kb loop\n";
  ss << "\n";

  // ── Normalize and write O ─────────────────────────────────────────────────
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
  ss << "    Otile.template store<T, 1, 1>(O_write, (int)p->O_strides[2]);\n";
  ss << "  }\n";
  ss << "\n";

  // ── Write L (logsumexp) ───────────────────────────────────────────────────
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
