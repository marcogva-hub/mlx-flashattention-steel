/// mfa_steel_fwd_v5.cpp  –  STEEL V5 forward kernel: D-blocked attention.
///
/// Key differences from V2 D-split:
///   - BD_tile=32 (vs BD_HALF=128 in V2 D-split) → 4× smaller TGP
///   - BK=128 (vs BK=32/64 in V2) → 4× fewer K-tile iterations
///   - Q loaded directly from device into registers (no Q_smem)
///   - Single KV_smem buffer = max(K^T, V) = 10,240 bytes → 3 TG/CU
///
/// Barrier schedule per K-tile (D=128, D_chunks=4):
///   Q@K^T phase:   B_pre (preloaded) + (D_chunks-1) × 2 = 6 barriers
///   Transition A:  1 barrier
///   P@V phase:     load + B × D_chunks + (D_chunks-1) inter = 8 barriers
///   Preload next:  X + C = 2 barriers (conditional)
///   Total per tile: ~17   vs V2: 4 barriers × 4× more tiles = same ballpark

#include "mfa_steel_fwd.hpp"
#include "mfa_steel_fwd_v5.hpp"
#include <sstream>

namespace mlx_mfa {

std::string generate_steel_v5_source(const ShaderCache::KernelKey& key) {
  using KK = ShaderCache::KernelKey;

  const bool no_padding = (std::getenv("MFA_NO_PADDING") != nullptr);
  const std::string pad_expr = no_padding ? "0" : "16 / sizeof(T)";

  const int D       = key.head_dim;
  const bool causal = key.causal;
  const int  gqa    = key.gqa_factor;

  const char* dtype_str = (key.dtype == 1) ? "bfloat" : "half";

  auto cfg = select_steel_v5_block_config(D, key.is_m3_plus);
  const int BQ     = cfg.BQ;    // 32 (M1/M2) or 16 (M3+)
  const int BK     = cfg.BK;    // 128 for all gens
  const int BD_tile = cfg.BD_tile;  // 32 always
  const int WM     = cfg.WM;    // 4 (M1/M2) or 2 (M3+)
  const int WN     = 1;
  const int TGP_SIZE = WM * WN * 32;   // 128 or 64

  const int TQ      = BQ / (WM * WN * 8);  // always 1
  const int TK      = BK / 8;              // 128/8 = 16
  const int TD      = D / 8;               // 8 or 16
  const int BD_frags = BD_tile / 8;         // 32/8 = 4
  const int D_chunks = D / BD_tile;         // 2 (D=64) or 4 (D=128)

  const bool enable_unroll = true;  // BD_tile=32 → BD_frags=4, safe to unroll
  const int  arch_gen      = key.is_m3_plus ? 15 : 13;

  std::ostringstream ss;

  // ── Metal preamble ────────────────────────────────────────────────────────
  append_metal_headers_and_defines(ss, enable_unroll, arch_gen, dtype_str);

  // ── Type aliases + compile-time #defines ─────────────────────────────────
  ss << "typedef " << dtype_str << " T;\n";
  ss << "typedef float AccT;\n";
  ss << "#define MFA_BQ       " << BQ       << "\n";
  ss << "#define MFA_BK       " << BK       << "\n";
  ss << "#define MFA_BD       " << D        << "\n";  // full head dim
  ss << "#define MFA_BD_TILE  " << BD_tile  << "\n";  // D-chunk size
  ss << "#define MFA_BD_FRAGS " << BD_frags << "\n";  // frags per D-chunk
  ss << "#define MFA_D_CHUNKS " << D_chunks << "\n";  // D/BD_tile
  ss << "#define MFA_WM       " << WM       << "\n";
  ss << "#define MFA_WN       " << WN       << "\n";
  ss << "#define MFA_TGP_SIZE " << TGP_SIZE << "\n";
  ss << "#define MFA_TQ       " << TQ       << "\n";
  ss << "#define MFA_TK       " << TK       << "\n";
  ss << "#define MFA_TD       " << TD       << "\n";
  ss << "#define MFA_GQA      " << gqa      << "\n";
  ss << "#define MFA_ROWS_PT  " << TQ       << "\n";
  ss << "\n";

  // ── Shared templates (BlockLoaderT, MMAFrag, MMATile, ops) ───────────────
  append_steel_shared_templates(ss);

  // ── MFASteelParams (same layout as V2) ───────────────────────────────────
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

  // ── Kernel signature ──────────────────────────────────────────────────────
  ss << "[[kernel, max_total_threads_per_threadgroup(MFA_TGP_SIZE)]]\n";
  ss << "void mlx_mfa_v5_attention(\n";
  ss << "    const device T*          Q  [[buffer(0)]],\n";
  ss << "    const device T*          K  [[buffer(1)]],\n";
  ss << "    const device T*          V  [[buffer(2)]],\n";
  ss << "    device T*                O  [[buffer(3)]],\n";
  ss << "    device float*            L  [[buffer(4)]],\n";
  ss << "    constant MFASteelParams* p  [[buffer(5)]],\n";
  ss << "    uint3 tid          [[threadgroup_position_in_grid]],\n";
  ss << "    uint  simd_group_id [[simdgroup_index_in_threadgroup]],\n";
  ss << "    uint  simd_lane_id  [[thread_index_in_simdgroup]])\n";
  ss << "{\n";

  // ── GQA head mapping ─────────────────────────────────────────────────────
  ss << "  const int h_q  = (int)tid.y;\n";
  ss << "  const int h_kv = (MFA_GQA == 1) ? h_q : (h_q / MFA_GQA);\n";
  ss << "  Q += (long)tid.z * p->Q_strides[0] + (long)h_q  * p->Q_strides[1];\n";
  ss << "  K += (long)tid.z * p->K_strides[0] + (long)h_kv * p->K_strides[1];\n";
  ss << "  V += (long)tid.z * p->V_strides[0] + (long)h_kv * p->V_strides[1];\n";
  ss << "  O += (long)tid.z * p->O_strides[0] + (long)h_q  * p->O_strides[1];\n";
  ss << "\n";

  // ── Threadgroup memory: single KV_smem buffer ─────────────────────────────
  // K^T_smem layout [BD_tile, BK+pad]: BD_tile rows of (BK+pad) elements.
  // V_smem layout   [BK, BD_tile+pad]: BK rows of (BD_tile+pad) elements.
  // Buffer sized to the larger of the two:
  //   K^T: BD_tile * (BK + pad) = 32 * 136 * 2 = 8,704 bytes (half)
  //   V:   BK * (BD_tile + pad) = 128 * 40 * 2 = 10,240 bytes (half)
  //   max = V → 10,240 bytes → 3 TG/CU at 32KB threadgroup memory budget.
  ss << "  constexpr short padK = " << pad_expr << ";\n";
  ss << "  constexpr short padV = " << pad_expr << ";\n";
  ss << "  // K^T layout: [BD_tile, BK+padK]  → LDK stride = BK+padK\n";
  ss << "  constexpr short LDK  = MFA_BK + padK;\n";
  ss << "  // V layout:   [BK,     BD_tile+padV] → LDV stride = BD_tile+padV\n";
  ss << "  constexpr short LDV  = MFA_BD_TILE + padV;\n";
  ss << "\n";
  ss << "  // KV_smem holds whichever is larger: K^T or V.\n";
  ss << "  constexpr short kK_elems = MFA_BD_TILE * (MFA_BK + padK);\n";
  ss << "  constexpr short kV_elems = MFA_BK * (MFA_BD_TILE + padV);\n";
  ss << "  constexpr short kKV_elems = kK_elems > kV_elems ? kK_elems : kV_elems;\n";
  ss << "  threadgroup T KV_smem[kKV_elems];\n";
  ss << "  threadgroup T* Ks = KV_smem;  // K^T occupant\n";
  ss << "  threadgroup T* Vs = KV_smem;  // V occupant (same buffer, sequential)\n";
  ss << "\n";

  // ── Block loader types ────────────────────────────────────────────────────
  // KLoader: loads K[BK, BD_tile] transposed into K^T[BD_tile, BK+pad].
  //   src_ld = K_strides[2] = D (full K row stride, we start at dh*BD_tile offset)
  //   kDstStrRow = LDK = BK+pad (each D-row has BK+pad cols in TGP)
  //   kDstStrCol = 1 (contiguous K-index in each row)
  ss << "  using KLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD_TILE,\n";
  ss << "      1, MFA_BK + 16/sizeof(T), 0, MFA_TGP_SIZE>;\n";
  // VLoader: loads V[BK, BD_tile] row-major.
  //   src_ld = V_strides[2] = D
  //   kDstStrRow = LDV = BD_tile+pad
  //   kDstStrCol = 1
  ss << "  using VLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD_TILE,\n";
  ss << "      MFA_BD_TILE + 16/sizeof(T), 1, 0, MFA_TGP_SIZE>;\n";
  ss << "\n";

  // ── Scale ─────────────────────────────────────────────────────────────────
  ss << "  const AccT scale = p->scale * M_LOG2E_F;\n";
  ss << "\n";

  // ── SIMD coordinates (same as V2) ────────────────────────────────────────
  ss << "  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);\n";
  ss << "  const short sm = simd_coord.y;\n";  // row within 8×8 frag
  ss << "  const short sn = simd_coord.x;\n";  // col within 8×8 frag
  ss << "  const short tm = 8 * MFA_TQ * (short)simd_group_id;\n";
  ss << "\n";

  // Ks_off: position within K^T_smem for this SIMD lane.
  // K^T[BD_tile, LDK]: row=sm (D-chunk row), col=sn (K-seq col).
  ss << "  const short Ks_off = sm * LDK + sn;\n";
  // Vs_off: position within V_smem for this SIMD lane.
  // V[BK, LDV]: row=sm (K-seq row), col=sn (D-chunk col).
  ss << "  const short Vs_off = sm * LDV + sn;\n";
  ss << "\n";

  // ── Tile registers ────────────────────────────────────────────────────────
  // Qtile[TQ, TD]: full head dim in registers — loaded once from device.
  // Otile[TQ, TD]: accumulates full head dim output.
  // Stile[TQ, TK]: Q@K^T scores for one K-tile.
  // Ktile[1, TK]:  one D-row of K^T (width=BK).
  // Vtile[1, 1]:   one 8×8 V fragment.
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Qtile;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Otile;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TK> Stile;\n";
  ss << "  MFAMMATile<AccT, 1,      MFA_TK> Ktile;\n";
  ss << "  MFAMMATile<AccT, 1,      1>      Vtile;\n";
  ss << "  AccT max_score[MFA_ROWS_PT];\n";
  ss << "  AccT sum_score[MFA_ROWS_PT];\n";
  ss << "\n";

  // ── Q-block and O-block pointers ──────────────────────────────────────────
  ss << "  const int qb = (int)tid.x;\n";
  ss << "  const device T* Q_qb = Q + (long)qb * MFA_BQ * p->Q_strides[2];\n";
  ss << "  device T*       O_qb = O + (long)qb * MFA_BQ * p->O_strides[2];\n";
  ss << "\n";

  // ── Reset output accumulators ─────────────────────────────────────────────
  ss << "  Otile.clear();\n";
  ss << "  STEEL_PRAGMA_UNROLL\n";
  ss << "  for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "    max_score[i] = -INFINITY;\n";
  ss << "    sum_score[i] = 0.0f;\n";
  ss << "  }\n";
  ss << "\n";

  // ── Load Q from device into registers (no TGP, no barrier needed) ─────────
  // Each simdgroup owns rows [tm..tm+BQ/WM-1] = [tm..tm+7] (TQ=1 → 8 rows).
  // D-chunk dh contributes fragments [dh*BD_frags .. (dh+1)*BD_frags-1] to Qtile.
  // Q is [qL, D] row-major with stride p->Q_strides[2] = D.
  // Lane (sm, sn) reads Q[tm+sm, dh*BD_tile + dd*8 + sn] and [...+sn+1].
  // No barrier needed: device reads are per-simdgroup, no TGP sharing.
  ss << "  // Load Q into registers from device (no TGP, per-SIMD device read)\n";
  ss << "  {\n";
  ss << "    const int D_stride = (int)p->Q_strides[2];  // = D\n";
  ss << "    const device T* Q_lane = Q_qb + (long)(tm + sm) * D_stride + sn;\n";
  // Boundary clamp: if qb is the last (partial) Q-block, clamp row reads.
  // For out-of-bounds rows, write 0 into Qtile (will be masked out in L/O writes).
  ss << "    const bool qb_last = (qb == p->NQ_aligned);\n";
  ss << "    const int  q_row_max = qb_last ? p->qL_rem : MFA_BQ;\n";
  for (int dh = 0; dh < D_chunks; dh++) {
    for (int dd = 0; dd < BD_frags; dd++) {
      const int frag_idx = dh * BD_frags + dd;
      const int col_off  = dh * BD_tile + dd * 8;
      ss << "    // Q fragment [iq=0, " << frag_idx << "]: D-chunk " << dh
         << ", sub-frag " << dd << " (D-cols " << col_off << ".." << (col_off+7) << ")\n";
      ss << "    if ((tm + sm) < q_row_max) {\n";
      ss << "      MFAMMAFrag<AccT>::load(Qtile.frag_at(0, " << frag_idx << "),\n";
      ss << "          Q_lane + " << col_off << ",\n";
      ss << "          (int)D_stride, (int)1);\n";
      ss << "    } else {\n";
      ss << "      Qtile.frag_at(0, " << frag_idx << ") = {0, 0};\n";
      ss << "    }\n";
    }
  }
  ss << "  }\n";
  ss << "\n";

  // ── K-loop limits ─────────────────────────────────────────────────────────
  if (causal) {
    ss << "  int q_max  = (qb + 1) * MFA_BQ + p->qL_off;\n";
    ss << "  int kb_lim = (q_max + MFA_BK - 1) / MFA_BK;\n";
    ss << "  if (kb_lim > p->NK) kb_lim = p->NK;\n";
  } else {
    ss << "  int kb_lim = p->NK;\n";
  }
  ss << "  const int kb_start = 0;\n";
  ss << "\n";

  // ── Running K/V pointers ─────────────────────────────────────────────────
  ss << "  const device T* K_cur = K;\n";
  ss << "  const device T* V_cur = V;\n";
  ss << "\n";

  // ── Helper lambdas (C++-side, emit Metal code) ───────────────────────────

  // emit_load_k: emit safe/unsafe K D-chunk load.
  // ptr_expr:  device pointer to the start of K[kb, dh*BD_tile].
  // kb_expr:   Metal expression for the current K-tile index ("kb" inside the
  //            loop, or "0" for the pre-loop preload where `kb` is not yet in scope).
  auto emit_load_k = [&](const std::string& ptr_expr,
                          const std::string& kb_expr = "kb") {
    ss << "    if (" << kb_expr << " == p->NK_aligned) {\n";
    ss << "      KLoader(" << ptr_expr << ", (int)p->K_strides[2], Ks,\n";
    ss << "               (ushort)simd_group_id, (ushort)simd_lane_id)\n";
    ss << "          .load_safe(short2(MFA_BD_TILE, p->kL_rem));\n";
    ss << "    } else {\n";
    ss << "      KLoader(" << ptr_expr << ", (int)p->K_strides[2], Ks,\n";
    ss << "               (ushort)simd_group_id, (ushort)simd_lane_id)\n";
    ss << "          .load_unsafe();\n";
    ss << "    }\n";
  };

  // emit_load_v: emit safe/unsafe V D-chunk load.
  // Same kb_expr convention as emit_load_k above.
  auto emit_load_v = [&](const std::string& ptr_expr,
                          const std::string& kb_expr = "kb") {
    ss << "    if (" << kb_expr << " == p->NK_aligned) {\n";
    ss << "      VLoader(" << ptr_expr << ", (int)p->V_strides[2], Vs,\n";
    ss << "               (ushort)simd_group_id, (ushort)simd_lane_id)\n";
    ss << "          .load_safe(short2(MFA_BD_TILE, p->kL_rem));\n";
    ss << "    } else {\n";
    ss << "      VLoader(" << ptr_expr << ", (int)p->V_strides[2], Vs,\n";
    ss << "               (ushort)simd_group_id, (ushort)simd_lane_id)\n";
    ss << "          .load_unsafe();\n";
    ss << "    }\n";
  };

  // emit_qkt: emit Q@K^T sub-GEMM for D-chunk dh.
  // K^T[dh] is already in KV_smem.  Qtile.frag_at(0, dh*BD_frags+dd) for each dd.
  // LDK = BK+pad (K^T row stride).
  auto emit_v5_qkt = [&](int dh) {
    ss << "    // ─ Q@K^T: D-chunk dh=" << dh << " ─\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short dd = 0; dd < MFA_BD_FRAGS; dd++) {\n";
    ss << "      // Load K^T fragment: row dd*8 of K^T[BD_tile, LDK]\n";
    ss << "      Ktile.template load<T, 1, 1>(\n";
    ss << "          &Ks[Ks_off + (short)(dd * 8) * LDK], LDK, 1);\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short iq = 0; iq < MFA_TQ; iq++) {\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short ik = 0; ik < MFA_TK; ik++) {\n";
    ss << "          MFAMMAFrag<AccT>::mma(\n";
    ss << "              Stile.frag_at(iq, ik),\n";
    // Qtile fragment offset = dh * BD_frags + dd.  dh is C++ compile-time.
    ss << "              Qtile.frag_at(iq, " << (dh * BD_frags) << " + dd),\n";
    ss << "              Ktile.frag_at(0, ik),\n";
    ss << "              Stile.frag_at(iq, ik));\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
  };

  // emit_pv: emit P@V sub-GEMM for D-chunk dh.
  // V[dh] row-major in KV_smem.  Otile.frag_at(0, dh*BD_frags+id) for each id.
  auto emit_v5_pv = [&](int dh) {
    ss << "    // ─ P@V: D-chunk dh=" << dh << " ─\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short iq = 0; iq < MFA_TQ; iq++) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short ik = 0; ik < MFA_TK; ik++) {\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short id = 0; id < MFA_BD_FRAGS; id++) {\n";
    ss << "          Vtile.template load<T, 1, 1>(\n";
    ss << "              &Vs[Vs_off + ik*8*LDV + id*8], LDV, 1);\n";
    ss << "          MFAMMAFrag<AccT>::mma(\n";
    ss << "              Otile.frag_at(iq, " << (dh * BD_frags) << " + id),\n";
    ss << "              Stile.frag_at(iq, ik),\n";
    ss << "              Vtile.frag_at(0, 0),\n";
    ss << "              Otile.frag_at(iq, " << (dh * BD_frags) << " + id));\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
  };

  // ── Pre-loop: preload K[kb_start][dh=0] ──────────────────────────────────
  ss << "  // Pre-loop: preload K[kb_start] D-chunk 0 into KV_smem.\n";
  ss << "  if (kb_lim > kb_start) {\n";
  emit_load_k("K_cur", "0");  // pre-loop: kb=0 (kb not in scope yet)
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // B_pre: K[0][dh=0] ready\n";
  ss << "  }\n";
  ss << "\n";

  // ── Main K-tile loop ──────────────────────────────────────────────────────
  ss << "  for (int kb = kb_start; kb < kb_lim; kb++) {\n";
  ss << "\n";

  // ── Phase 1: Q@K^T (all D-chunks of K^T) ─────────────────────────────────
  ss << "    // ─ Phase 1: Q@K^T (all D-chunks) ─\n";
  ss << "    Stile.clear();\n";
  // dh=0: K[dh=0] already in KV_smem from preload (or prev-iter barrier C)
  emit_v5_qkt(0);
  for (int dh = 1; dh < D_chunks; dh++) {
    ss << "    // K[dh=" << (dh-1) << "] reads done → safe to load K[dh=" << dh << "]\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_load_k("K_cur + (long)" + std::to_string(dh) + " * MFA_BD_TILE");
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // K[dh=" << dh << "] ready\n";
    emit_v5_qkt(dh);
  }
  ss << "\n";

  // ── Scale scores ──────────────────────────────────────────────────────────
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++) {\n";
  ss << "      Stile.elems()[ii] *= scale;\n";
  ss << "    }\n";
  ss << "\n";

  // ── K-boundary mask (pad out-of-range K positions → -inf) ────────────────
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

  // ── Causal mask ───────────────────────────────────────────────────────────
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

  // ── Online softmax (NaN-safe, log2 domain) ────────────────────────────────
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
  ss << "    // Rescale entire Otile by softmax correction\n";
  ss << "    Otile.template row_bin_op<MFAMulOp>(factor);\n";
  ss << "\n";

  // ── Barrier A: K reads done → safe to use KV_smem for V ──────────────────
  ss << "    // ─ Barrier A: K phase done → start V loading ─\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "\n";

  // ── Phase 2: P@V (all D-chunks of V) ─────────────────────────────────────
  ss << "    // ─ Phase 2: P@V (all D-chunks) ─\n";
  for (int dh = 0; dh < D_chunks; dh++) {
    if (dh > 0) {
      ss << "    // V[dh=" << (dh-1) << "] reads done → safe to load V[dh=" << dh << "]\n";
      ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    }
    emit_load_v("V_cur + (long)" + std::to_string(dh) + " * MFA_BD_TILE");
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // V[dh=" << dh << "] ready\n";
    emit_v5_pv(dh);
    ss << "\n";
  }

  // ── Advance K_cur/V_cur + preload K[kb+1][dh=0] ──────────────────────────
  ss << "    K_cur += (long)MFA_BK * p->K_strides[2];\n";
  ss << "    V_cur += (long)MFA_BK * p->V_strides[2];\n";
  ss << "\n";
  ss << "    // ─ Barrier X: V reads done → preload K[kb+1][dh=0] ─\n";
  ss << "    if (kb + 1 < kb_lim) {\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // X\n";
  emit_load_k("K_cur");
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // C: K[kb+1][dh=0] ready\n";
  ss << "    }\n";
  ss << "\n";

  ss << "  } // end kb loop\n";
  ss << "\n";

  // ── Normalize Otile (divide each element by sum_score) ────────────────────
  ss << "  Otile.template row_bin_op<MFADivOp>(sum_score);\n";
  ss << "  threadgroup_barrier(mem_flags::mem_none);\n";
  ss << "\n";

  // ── Write O to device (D-blocked, same as V2 D-split) ────────────────────
  // Otile holds all D=TD fragments; write each D-chunk's fragments.
  for (int dh = 0; dh < D_chunks; dh++) {
    ss << "  // Write O D-chunk " << dh << " (D-cols "
       << (dh * BD_tile) << ".." << (dh * BD_tile + BD_tile - 1) << ")\n";
    ss << "  {\n";
    ss << "    device T* O_write = O_qb + (long)(tm + sm) * p->O_strides[2]\n";
    ss << "                       + sn + (long)" << dh << " * MFA_BD_TILE;\n";
    ss << "    if (qb == p->NQ_aligned) {\n";
    ss << "      auto dims = short2((short)(MFA_BD_TILE - sn),\n";
    ss << "                         (short)(p->qL_rem - (tm + sm)));\n";
    ss << "      if (dims.x > 0 && dims.y > 0) {\n";
    // We need to store only the BD_frags fragments for this D-chunk.
    // Otile is [TQ, TD] with TD = D/8.  Fragment index = dh*BD_frags + id.
    // Build a sub-tile by emitting a temporary 1×BD_frags tile and store.
    // Simpler: write individual fragments.
    for (int id = 0; id < BD_frags; id++) {
      const int frag_idx = dh * BD_frags + id;
      const int col_off  = id * 8;
      ss << "        if ((short)(MFA_BD_TILE - sn) > " << col_off
         << " && p->qL_rem > (tm + sm)) {\n";
      ss << "          if (" << col_off << " < dims.x)\n";
      ss << "            O_write[" << col_off << "] = static_cast<T>(Otile.frag_at(0, " << frag_idx << ")[0]);\n";
      ss << "          if (" << (col_off + 1) << " < dims.x)\n";
      ss << "            O_write[" << (col_off + 1) << "] = static_cast<T>(Otile.frag_at(0, " << frag_idx << ")[1]);\n";
      ss << "        }\n";
    }
    ss << "      }\n";
    ss << "    } else {\n";
    // Fast unsafe path: write all fragments for this D-chunk
    for (int id = 0; id < BD_frags; id++) {
      const int frag_idx = dh * BD_frags + id;
      const int col_off  = id * 8;
      ss << "      O_write[" << col_off << "] = static_cast<T>(Otile.frag_at(0, " << frag_idx << ")[0]);\n";
      ss << "      O_write[" << (col_off+1) << "] = static_cast<T>(Otile.frag_at(0, " << frag_idx << ")[1]);\n";
    }
    ss << "    }\n";
    ss << "  }\n";
    ss << "\n";
  }

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
