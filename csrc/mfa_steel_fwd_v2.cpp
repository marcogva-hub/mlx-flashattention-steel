/// mfa_steel_fwd_v2.cpp  –  STEEL V2 forward kernel: sequential K/V phases.
///
/// Key innovation over V1:
///   - Q_smem loaded ONCE and stays in registers for all K-tile iterations.
///   - K and V share the SAME KV_smem region (sequential, not simultaneous).
///   - Enables BQ=64, BK=48 for D=128 → 31,744 bytes < 32 KB TGP.
///   - Arithmetic intensity: 32 FMAs/byte (vs 16 FMAs/byte in V1).
///
/// Barrier schedule (4 per non-last K-tile, 3 for last tile):
///   Preload K[kb_start]; barrier B0;
///   for kb:
///     Q@K^T (reads KV_smem = K)
///     barrier A  (K reads done → safe to overwrite KV_smem with V)
///     Load V into KV_smem
///     barrier B  (V written → safe to read V for P@V)
///     P@V (reads KV_smem = V)
///     if kb+1 < kb_lim:
///       barrier X  (P@V V-reads done → safe to overwrite KV_smem with K)
///       Load K[kb+1] into KV_smem
///       barrier C  (K[kb+1] written → visible for next Q@K^T)
///
/// KV_smem is shared between K and V sequentially; barriers X and A ensure
/// reads and writes never overlap within the same threadgroup.
///
/// For N=4096 causal D=128 (avg 43 K-tiles):
///   V2:  4 × 42 + 3 = 171 total barriers
///   V1:  2 × 256 = 512 total barriers (BK=16 double_buf)
///   → 67% fewer barrier stalls with 2× more compute per K-tile (BK=32 vs 16).
///
/// Supported: f16/bf16, D=64/128, causal=true/false, GQA.
/// Phase 5 extensions: RoPE, sliding window, sparse, ALiBi, softcap.

#include "mfa_steel_fwd.hpp"
#include "mfa_steel_fwd_v2.hpp"
#include <sstream>

namespace mlx_mfa {

// ---------------------------------------------------------------------------
// V2 tile config
// ---------------------------------------------------------------------------

SteelV2BlockConfig select_steel_v2_block_config(int head_dim) {
  // V2 key principle: keep BQ = V1's BQ (same grid → same occupancy), but
  // double BK using sequential K/V sharing of KV_smem.
  //
  // V1 configs: D=64  → BQ=32, BK=32, WM=4  (TGP = 4608+5120+4608 = 14336 B)
  //             D=128 → BQ=32, BK=16, WM=4  (TGP = 8704+6144+4352 = 19200 B)
  //
  // V2 configs (doubled BK, sequential KV_smem = max(K_smem, V_smem)):
  //   D=64:  BQ=32, BK=64, WM=4  → Q(4608) + KV(max(9216,9216)) = 13824 B < 14336 ✓
  //   D=128: BQ=32, BK=32, WM=4  → Q(8704) + KV(max(10240,8704)) = 18944 B < 19200 ✓
  //
  // MFABlockLoaderT constraint: n_reads = (D*BK)/TGP must divide D evenly.
  //   TGP=128 threads (WM=4). For K-transposed loader (BCOLS=D):
  //   D=64,  BK=64: n_reads=32, TCOLS=64/32=2 ✓
  //   D=128, BK=32: n_reads=32, TCOLS=128/32=4 ✓
  //
  // Both configs have same grid as V1 (BQ=32), 2× fewer K-tile iterations,
  // and slightly smaller TGP → occupancy ≥ V1.
  if (head_dim == 64)  return {32, 64,  64, 4, 1};  // TQ=1, TK=8, TD=8
  if (head_dim == 128) return {32, 32, 128, 4, 1};  // TQ=1, TK=4, TD=16
  return {0, 0, 0, 0, 0};  // unsupported
}

// ---------------------------------------------------------------------------
// V2 kernel source generator
// ---------------------------------------------------------------------------

std::string generate_steel_v2_source(const ShaderCache::KernelKey& key) {
  using KK = ShaderCache::KernelKey;

  const int D     = key.head_dim;
  const bool causal = key.causal;
  const int gqa   = key.gqa_factor;  // H_q / H_kv (1 = standard MHA)

  // V2 only supports f16/bf16
  const char* dtype_str = (key.dtype == 1) ? "bfloat" : "half";

  auto cfg = select_steel_v2_block_config(D);
  const int BQ = cfg.BQ;   // 64
  const int BK = cfg.BK;   // 48 (D=128) or 64 (D=64)
  const int WM = cfg.WM;   // 8
  const int WN = 1;
  const int TGP_SIZE = WM * WN * 32;  // 256
  const int TD  = D / 8;       // 16 (D=128) or 8 (D=64)
  const int TK  = BK / 8;      // 6 (BK=48) or 8 (BK=64)
  const int TQ  = BQ / (WM * WN * 8);  // 1  (64 / (8*8))

  // Unroll: always enable for D<=128
  const bool enable_unroll = true;
  const int  arch_gen      = 13;  // not used in V2 Metal source, placeholder

  std::ostringstream ss;

  // ── Metal preamble (shared helper from V1) ──────────────────────────────
  append_metal_headers_and_defines(ss, enable_unroll, arch_gen, dtype_str);

  // ── Define V2 block-size constants ──────────────────────────────────────
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

  // ── Shared BlockLoaderT + MMATile templates (from V1) ───────────────────
  append_steel_shared_templates(ss);

  // ── MFASteelParams struct (same as V1 — shared CPU/GPU layout) ──────────
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
};

)MFA";

  // ── V2 kernel function ───────────────────────────────────────────────────
  ss << "[[kernel, max_total_threads_per_threadgroup(MFA_TGP_SIZE)]]\n";
  ss << "void mlx_mfa_v2_attention(\n";
  ss << "    const device T*             Q         [[buffer(0)]],\n";
  ss << "    const device T*             K         [[buffer(1)]],\n";
  ss << "    const device T*             V         [[buffer(2)]],\n";
  ss << "    device T*                   O         [[buffer(3)]],\n";
  ss << "    device float*               L         [[buffer(4)]],\n";
  ss << "    constant MFASteelParams*    p         [[buffer(5)]],\n";
  ss << "    uint3 tid          [[threadgroup_position_in_grid]],\n";
  ss << "    uint  simd_group_id [[simdgroup_index_in_threadgroup]],\n";
  ss << "    uint  simd_lane_id  [[thread_index_in_simdgroup]])\n";
  ss << "{\n";

  // ── GQA head remapping ──────────────────────────────────────────────────
  ss << "  // tid: (qb_group, H_q_head, batch)\n";
  ss << "  const int h_q  = (int)tid.y;\n";
  ss << "  const int h_kv = h_q / p->gqa_factor;\n";
  ss << "  // Base pointers per (batch, head) pair\n";
  ss << "  Q += (long)tid.z * p->Q_strides[0] + (long)h_q  * p->Q_strides[1];\n";
  ss << "  K += (long)tid.z * p->K_strides[0] + (long)h_kv * p->K_strides[1];\n";
  ss << "  V += (long)tid.z * p->V_strides[0] + (long)h_kv * p->V_strides[1];\n";
  ss << "  O += (long)tid.z * p->O_strides[0] + (long)h_q * p->O_strides[1];\n";
  ss << "\n";

  // ── Threadgroup memory: Q_smem + shared KV_smem ─────────────────────────
  // Sequential K/V phases: K and V reuse the same KV_smem buffer.
  // KV_smem = max(K_smem, V_smem):
  //   K_smem (transposed): (BK+padK) * BD * sizeof(T)
  //   V_smem (row-major):  BK * (BD+padV) * sizeof(T)
  ss << "  constexpr short padQ = 16 / sizeof(T);\n";
  ss << "  constexpr short padK = 16 / sizeof(T);\n";
  ss << "  constexpr short padV = 16 / sizeof(T);\n";
  ss << "  constexpr short LDQ  = MFA_BD + padQ;\n";
  ss << "  constexpr short LDK  = MFA_BK + padK;  // stride for transposed K\n";
  ss << "  constexpr short LDV  = MFA_BD + padV;\n";
  ss << "  // KV_smem = max of K_smem and V_smem sizes:\n";
  ss << "  constexpr short kv_s0 = (MFA_BK + padK) * MFA_BD;     // K transposed\n";
  ss << "  constexpr short kv_s1 = MFA_BK * (MFA_BD + padV);      // V row-major\n";
  ss << "  constexpr short kv_s  = kv_s0 > kv_s1 ? kv_s0 : kv_s1;\n";
  ss << "\n";
  ss << "  threadgroup T Q_smem[MFA_BQ * (MFA_BD + padQ)];\n";
  ss << "  threadgroup T KV_smem[kv_s];  // K and V share this buffer sequentially\n";
  ss << "  threadgroup T* Qs = Q_smem;\n";
  ss << "  threadgroup T* Ks = KV_smem;  // K transposed into KV_smem\n";
  ss << "  threadgroup T* Vs = KV_smem;  // V row-major into same KV_smem\n";
  ss << "\n";

  // ── Block loaders ────────────────────────────────────────────────────────
  ss << "  // Q: row-major, BQ×BD tiles\n";
  ss << "  using QLoader = MFABlockLoaderT<T, MFA_BQ, MFA_BD,\n";
  ss << "      MFA_BD + 16/sizeof(T), 1, 1, MFA_TGP_SIZE>;\n";
  ss << "  // K: transposed into TGP (kDstStrRow=1, kDstStrCol=LDK)\n";
  ss << "  using KLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      1, MFA_BK + 16/sizeof(T), 0, MFA_TGP_SIZE>;\n";
  ss << "  // V: row-major, BK×BD tiles\n";
  ss << "  using VLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      MFA_BD + 16/sizeof(T), 1, 0, MFA_TGP_SIZE>;\n";
  ss << "\n";

  // ── SIMD coordinate (same pattern as V1) ─────────────────────────────────
  ss << "  const AccT scale = p->scale * M_LOG2E_F;\n";
  ss << "\n";
  ss << "  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);\n";
  ss << "  const short sm = simd_coord.y;\n";
  ss << "  const short sn = simd_coord.x;\n";
  ss << "  const short tm = 8 * MFA_TQ * (short)simd_group_id;\n";
  ss << "\n";
  ss << "  const short Qs_off = (tm + sm) * LDQ + sn;\n";
  ss << "  const short Ks_off = sm * LDK + sn;\n";
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

  // ── One Q-block per threadgroup (tid.x = Q-block index) ─────────────────
  ss << "  const int qb = (int)tid.x;\n";
  ss << "\n";

  // Per-qb source/destination pointers
  ss << "  const device T* Q_qb = Q + (long)qb * MFA_BQ * p->Q_strides[2];\n";
  ss << "  device T*       O_qb = O + (long)qb * MFA_BQ * p->O_strides[2];\n";
  ss << "\n";

  // Block loaders for this Q-block
  ss << "  QLoader loader_q(Q_qb, (int)p->Q_strides[2], Qs,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "  KLoader loader_k(K, (int)p->K_strides[2], Ks,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "  VLoader loader_v(V, (int)p->V_strides[2], Vs,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "\n";

  // Reset accumulators for this Q-block
  ss << "  Otile.clear();\n";
  ss << "  STEEL_PRAGMA_UNROLL\n";
  ss << "  for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "    max_score[i] = -INFINITY;\n";
  ss << "    sum_score[i] = 0.0f;\n";
  ss << "  }\n";
  ss << "\n";

  // ── Load Q tile → threadgroup → registers ────────────────────────────────
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "  if (qb == p->NQ_aligned) {\n";
  ss << "    loader_q.load_safe(short2(MFA_BD, p->qL_rem));\n";
  ss << "  } else {\n";
  ss << "    loader_q.load_unsafe();\n";
  ss << "  }\n";
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "  Qtile.template load<T, 1, 1>(&Qs[Qs_off], LDQ, 1);\n";
  ss << "\n";

  // ── K-loop limit (causal or full) ────────────────────────────────────────
  if (causal) {
    ss << "  int q_max  = (qb + 1) * MFA_BQ + p->qL_off;\n";
    ss << "  int kb_lim = (q_max + MFA_BK - 1) / MFA_BK;\n";
    ss << "  if (kb_lim > p->NK) kb_lim = p->NK;\n";
  } else {
    ss << "  int kb_lim = p->NK;\n";
  }
  ss << "  const int kb_start = 0;\n";
  ss << "\n";

  // ── V2 Phase 0: Preload K[kb_start] before main loop ────────────────────
  // This enables 3 barriers/tile (vs 4 for naive shared-KV) by having K ready
  // at the start of each iteration without an explicit load+barrier in the loop.
  ss << "  // V2: preload K[kb_start] into KV_smem before the loop.\n";
  ss << "  // Barrier B0 ensures K is visible at the first Q@K^T.\n";
  ss << "  if (kb_lim > kb_start) {\n";
  ss << "    if (kb_start == p->NK_aligned) {\n";
  ss << "      loader_k.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "    } else {\n";
  ss << "      loader_k.load_unsafe();\n";
  ss << "    }\n";
  ss << "    loader_k.next();\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // B0: K[0] ready\n";
  ss << "  }\n";
  ss << "\n";

  // ── Main K/V loop ────────────────────────────────────────────────────────
  ss << "  for (int kb = kb_start; kb < kb_lim; kb++) {\n";
  ss << "\n";

  // Phase 1: Q@K^T (K[kb] already in KV_smem, synced by B0 or prev-iter C)
  ss << "    // ─ Phase 1: Q@K^T ─\n";
  ss << "    // K[kb] is in KV_smem, already visible via preload barrier.\n";
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

  // Apply scale (log2 domain)
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

  // Online softmax (NaN-safe, same as V1)
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

  // Barrier A: K reads done → safe to overwrite KV_smem with V
  ss << "    // ─ Barrier A: all threads done reading K → safe to load V ─\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "\n";

  // Phase 2: Load V into KV_smem (same buffer as K)
  ss << "    // ─ Phase 2: Load V[kb] into KV_smem ─\n";
  ss << "    if (kb == p->NK_aligned) {\n";
  ss << "      loader_v.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "    } else {\n";
  ss << "      loader_v.load_unsafe();\n";
  ss << "    }\n";
  ss << "    loader_v.next();\n";
  ss << "\n";

  // Barrier B: V loaded → P@V can proceed
  ss << "    // ─ Barrier B: V fully written → P@V can read ─\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "\n";

  // Phase 3: P@V — iq outer, ik middle, id inner (same as V1)
  ss << "    // ─ Phase 3: O += P @ V ─\n";
  ss << "    // Loop order: iq → ik → id (id innermost keeps Stile[iq][ik] in regs)\n";
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

  // Barrier X: flush P@V reads of V from KV_smem before any thread writes
  // K[kb+1] into the same buffer.  K and V share KV_smem sequentially — reads
  // and writes to KV_smem cannot overlap even if issued by different SIMD groups.
  // Barrier C: K[kb+1] writes visible to all threads for next Q@K^T.
  ss << "    // ─ Barrier X: all P@V V-reads done → safe to overwrite KV_smem ─\n";
  ss << "    // ─ Barrier C: K[kb+1] fully written → visible for next Q@K^T  ─\n";
  ss << "    if (kb + 1 < kb_lim) {\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // X\n";
  ss << "      if ((kb + 1) == p->NK_aligned) {\n";
  ss << "        loader_k.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "      } else {\n";
  ss << "        loader_k.load_unsafe();\n";
  ss << "      }\n";
  ss << "      loader_k.next();\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // C\n";
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
