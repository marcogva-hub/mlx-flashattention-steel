/// mfa_steel_fwd_v5.cpp  –  STEEL V5 forward kernel: D-blocked attention.
///
/// Key differences from V2 D-split:
///   - BD_tile=32 (vs BD_HALF=128 in V2 D-split) → 4× smaller TGP
///   - BK=128 (vs BK=32/64 in V2) → 4× fewer K-tile iterations
///   - Q loaded directly from device into registers (no Q_smem)
///   - Single KV_smem buffer = max(K^T, V) = 8,192 bytes → 4 TG/CU (no padding needed)
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

  // V5 always uses pad_expr="0". The KLoader/VLoader LD_DST template params
  // are compile-time constants (MFA_BK and MFA_BD_TILE), so smem stride must
  // equal BK or BD_tile exactly — padding would create a loader/stride mismatch.
  // On M3+, device reads bypass threadgroup entirely so padding is irrelevant.
  // On M1/M2, bank-conflict risk is real but the loader constraint prevents
  // adding stride padding without restructuring the loader template.
  const std::string pad_expr = "0";

  const int D           = key.head_dim;
  const bool causal     = key.causal;
  const bool has_window = key.has_window;
  const bool has_softcap = key.has_softcap;
  const bool has_alibi   = key.has_alibi;
  const bool sparse      = key.sparse;
  const int  gqa         = key.gqa_factor;

  const char* dtype_str = (key.dtype == 1) ? "bfloat" : "half";

  auto cfg = select_steel_v5_block_config(D, key.is_m3_plus);
  const int BQ     = cfg.BQ;    // 32 (all gens)
  const int BK     = cfg.BK;    // 128 for all gens
  const int BD_tile = cfg.BD_tile;  // 32 always
  const int WM     = cfg.WM;    // 4 (all gens)
  const int WN     = 1;
  const int TGP_SIZE = WM * WN * 32;   // 128

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
  ss << "#define MFA_DIRECT_READS " << (key.is_m3_plus ? 1 : 0) << "\n";
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
  if (sparse)
    ss << "    const device uchar* block_mask [[buffer(6)]],\n";
  if (has_alibi)
    ss << "    const device float* alibi_slopes [[buffer(9)]],\n";
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

  // ── Threadgroup memory + loaders (TGP path only) ─────────────────────────
  // M3+ (MFA_DIRECT_READS=1): K and V read directly from device — no KV_smem needed.
  //   KLoader/VLoader excluded on M3+ (direct device reads bypass TGP).
  // TGP path: KV_smem holds whichever is larger: K^T or V.
  //   K^T: BD_tile * BK * sizeof(T) = 32 * 128 * 2 = 8,192 bytes (half)
  //   V:   BK * BD_tile * sizeof(T) = 128 * 32 * 2 = 8,192 bytes (half)
  //   max = 8,192 bytes → 4 TG/CU on M1/M2 (no padding needed).
  ss << "#if !MFA_DIRECT_READS\n";
  ss << "  constexpr short padK = " << pad_expr << ";\n";
  ss << "  constexpr short padV = " << pad_expr << ";\n";
  ss << "  // K^T layout: [BD_tile, BK+padK]  → LDK stride = BK+padK\n";
  ss << "  constexpr short LDK  = MFA_BK + padK;\n";
  ss << "  // V layout:   [BK,     BD_tile+padV] → LDV stride = BD_tile+padV\n";
  ss << "  constexpr short LDV  = MFA_BD_TILE + padV;\n";
  ss << "\n";
  ss << "  constexpr short kK_elems = MFA_BD_TILE * (MFA_BK + padK);\n";
  ss << "  constexpr short kV_elems = MFA_BK * (MFA_BD_TILE + padV);\n";
  ss << "  constexpr short kKV_elems = kK_elems > kV_elems ? kK_elems : kV_elems;\n";
  ss << "  threadgroup T KV_smem[kKV_elems];\n";
  ss << "  threadgroup T* Ks = KV_smem;  // K^T occupant\n";
  ss << "  threadgroup T* Vs = KV_smem;  // V occupant (same buffer, sequential)\n";
  ss << "\n";
  // KLoader: loads K[BK, BD_tile] transposed into K^T[BD_tile, BK+pad].
  ss << "  using KLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD_TILE,\n";
  ss << "      1, MFA_BK, 0, MFA_TGP_SIZE>;\n";  // LDK = BK (no pad)
  // VLoader: loads V[BK, BD_tile] row-major.
  ss << "  using VLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD_TILE,\n";
  ss << "      MFA_BD_TILE, 1, 0, MFA_TGP_SIZE>;\n";  // LDV = BD_tile (no pad)
  ss << "#endif  // !MFA_DIRECT_READS\n";
  ss << "\n";

  // Strides for M3+ direct device reads.
  ss << "  const int K_stride = (int)p->K_strides[2];  // = D\n";
  ss << "  const int V_stride = (int)p->V_strides[2];  // = D\n";
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

  // Ks_off/Vs_off: only needed for TGP path (inside #else branches).
  ss << "#if !MFA_DIRECT_READS\n";
  ss << "  const short Ks_off = sm * LDK + sn;\n";
  ss << "  const short Vs_off = sm * LDV + sn;\n";
  ss << "#endif\n";
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

  // ── K-loop limits (causal + sliding window) ──────────────────────────────
  if (causal) {
    ss << "  int q_max  = (qb + 1) * MFA_BQ + p->qL_off;\n";
    ss << "  int kb_lim = (q_max + MFA_BK - 1) / MFA_BK;\n";
    ss << "  if (kb_lim > p->NK) kb_lim = p->NK;\n";
  } else {
    ss << "  int kb_lim = p->NK;\n";
  }

  // Sliding window left bound → kb_start + K_cur/V_cur advance
  if (has_window) {
    ss << "  int kb_start   = 0;\n";
    ss << "  int kb_last_win = -1;  // last tile needing left-boundary masking\n";
    ss << "  if (p->window_left >= 0) {\n";
    ss << "    const int q_min   = qb * MFA_BQ + p->qL_off;\n";
    ss << "    const int ws      = q_min - p->window_left;\n";
    ss << "    kb_start = ws > 0 ? ws / MFA_BK : 0;\n";
    ss << "    kb_last_win = (q_min + MFA_BQ - 1 > p->window_left)\n";
    ss << "                    ? (q_min + MFA_BQ - 1 - p->window_left) / MFA_BK : -1;\n";
    ss << "  }\n";
  } else {
    ss << "  const int kb_start   = 0;\n";
    ss << "  const int kb_last_win = -1;\n";
  }
  ss << "\n";

  // ── Running K/V pointers (advanced to kb_start tile for sliding window) ──
  ss << "  const device T* K_cur = K + (long)kb_start * MFA_BK * p->K_strides[2];\n";
  ss << "  const device T* V_cur = V + (long)kb_start * MFA_BK * p->V_strides[2];\n";
  ss << "\n";

  // Sliding window right bound: clamp kb_lim + track first tile needing masking
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
    ss << "    kb_first_right = kb_lim;  // no right masking\n";
    ss << "  }\n";
  } else {
    ss << "  const int kb_first_right = kb_lim;  // no window\n";
  }
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

  // emit_v5_qkt: emit Q@K^T sub-GEMM for D-chunk dh.
  // TGP path (MFA_DIRECT_READS=0): K^T[dh] already in KV_smem.
  // Device path (MFA_DIRECT_READS=1): load K^T fragment directly from K_cur.
  //   K^T[dh*BD_tile + sm + dd*8, sn] = K_cur[(dh*BD_tile + sm + dd*8) + sn*K_stride]
  //   row_stride=1 (D-axis), col_stride=K_stride=D (S-axis)  — same as V4.
  auto emit_v5_qkt = [&](int dh) {
    ss << "    // ─ Q@K^T: D-chunk dh=" << dh << " ─\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short dd = 0; dd < MFA_BD_FRAGS; dd++) {\n";
    ss << "#if MFA_DIRECT_READS\n";
    ss << "      // M3+: direct device read — no TGP copy, no barrier.\n";
    ss << "      Ktile.template load<T, 1, 1>(\n";
    ss << "          K_cur + (long)(" << (dh * BD_tile) << " + sm + (short)(dd * 8))\n";
    ss << "                + (long)sn * K_stride,\n";
    ss << "          1, K_stride);\n";
    ss << "#else\n";
    ss << "      // TGP path: K^T fragment from KV_smem.\n";
    ss << "      Ktile.template load<T, 1, 1>(\n";
    ss << "          &Ks[Ks_off + (short)(dd * 8) * LDK], LDK, 1);\n";
    ss << "#endif\n";
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

  // emit_v5_pv: emit P@V sub-GEMM for D-chunk dh.
  // TGP path (MFA_DIRECT_READS=0): V[dh] row-major in KV_smem.
  // Device path (MFA_DIRECT_READS=1): load V fragment directly from V_cur.
  //   V[sm + ik*8, dh*BD_tile + sn + id*8] = V_cur[(sm+ik*8)*V_stride + dh*BD_tile + sn + id*8]
  //   row_stride=V_stride=D (S-axis), col_stride=1 (D-axis).
  //   Out-of-bounds rows (kb=NK_aligned): P values are 0 after K-boundary mask → safe.
  auto emit_v5_pv = [&](int dh) {
    ss << "    // ─ P@V: D-chunk dh=" << dh << " ─\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short iq = 0; iq < MFA_TQ; iq++) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short ik = 0; ik < MFA_TK; ik++) {\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short id = 0; id < MFA_BD_FRAGS; id++) {\n";
    ss << "#if MFA_DIRECT_READS\n";
    ss << "          // M3+: direct device read — no TGP copy, no barrier.\n";
    ss << "          Vtile.template load<T, 1, 1>(\n";
    ss << "              V_cur + (long)(sm + (short)(ik * 8)) * V_stride\n";
    ss << "                    + " << (dh * BD_tile) << " + sn + (short)(id * 8),\n";
    ss << "              V_stride, 1);\n";
    ss << "#else\n";
    ss << "          // TGP path: V fragment from KV_smem.\n";
    ss << "          Vtile.template load<T, 1, 1>(\n";
    ss << "              &Vs[Vs_off + (short)(ik*8)*LDV + (short)(id*8)], LDV, 1);\n";
    ss << "#endif\n";
    ss << "          MFAMMAFrag<AccT>::mma(\n";
    ss << "              Otile.frag_at(iq, " << (dh * BD_frags) << " + id),\n";
    ss << "              Stile.frag_at(iq, ik),\n";
    ss << "              Vtile.frag_at(0, 0),\n";
    ss << "              Otile.frag_at(iq, " << (dh * BD_frags) << " + id));\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
  };

  // ── Pre-loop: preload K[kb_start][dh=0] (TGP path only) ─────────────────
  // M3+ direct reads: K is loaded from device on-the-fly in the loop; no preload needed.
  ss << "#if !MFA_DIRECT_READS\n";
  ss << "  // Pre-loop: preload K[kb_start] D-chunk 0 into KV_smem.\n";
  ss << "  if (kb_lim > kb_start) {\n";
  emit_load_k("K_cur", "kb_start");  // pre-loop: kb_start tile (kb not in scope yet)
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // B_pre: K[0][dh=0] ready\n";
  ss << "  }\n";
  ss << "#endif  // !MFA_DIRECT_READS\n";
  ss << "\n";

  // ── Main K-tile loop ──────────────────────────────────────────────────────
  ss << "  for (int kb = kb_start; kb < kb_lim; kb++) {\n";
  ss << "\n";

  // Sparse tile-skip: uniform branch (zero warp divergence since all TG threads share tid.x, kb)
  if (sparse) {
    ss << "    // Block-sparse: skip tiles where block_mask==0 (uniform branch)\n";
    ss << "    const bool skip_tile = !block_mask[\n";
    ss << "        (long)tid.z * p->mask_batch_stride\n";
    ss << "      + (long)tid.y * p->mask_head_stride\n";
    ss << "      + (long)qb * p->NK + kb];\n";
    ss << "    if (!skip_tile) {\n";
  }

  // ── Phase 1: Q@K^T (all D-chunks of K^T) ─────────────────────────────────
  // M3+ (MFA_DIRECT_READS=1): all K^T fragments loaded directly from device —
  //   no barriers needed between D-chunks.
  // TGP path (MFA_DIRECT_READS=0): each D-chunk loads into KV_smem, fenced with barriers.
  ss << "    // ─ Phase 1: Q@K^T (all D-chunks) ─\n";
  ss << "    Stile.clear();\n";
  // dh=0: TGP path uses K[dh=0] from pre-loop preload (or prev-iter barrier C).
  //        Device path reads directly from K_cur — no prior load needed.
  emit_v5_qkt(0);
  for (int dh = 1; dh < D_chunks; dh++) {
    ss << "#if !MFA_DIRECT_READS\n";
    ss << "    // K[dh=" << (dh-1) << "] reads done → safe to load K[dh=" << dh << "]\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_load_k("K_cur + (long)" + std::to_string(dh) + " * MFA_BD_TILE");
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // K[dh=" << dh << "] ready\n";
    ss << "#endif  // !MFA_DIRECT_READS\n";
    emit_v5_qkt(dh);
  }
  ss << "\n";

  // ── Scale scores ──────────────────────────────────────────────────────────
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++) {\n";
  ss << "      Stile.elems()[ii] *= scale;\n";
  ss << "    }\n";
  ss << "\n";

  // Softcap (Gemma 2 / Grok): tanh(S_nat / cap) * cap, in log2 domain
  if (has_softcap) {
    ss << "    // Softcapping: convert log2→nat, tanh, nat→log2\n";
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

  // ALiBi: per-head linear position bias added to scores in log2 domain
  if (has_alibi) {
    ss << "    // ALiBi: add per-head linear position bias to scores\n";
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

  // ── Sliding window masking (left + right boundaries) ─────────────────────
  if (has_window) {
    ss << "    // Window left boundary: mask col < row - window_left\n";
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
    ss << "              Stile.frag_at(i, j)[jj] = -INFINITY;\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "    // Window right boundary: mask col > row + window_right\n";
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

  // ── Barrier A + Phase 2: P@V ─────────────────────────────────────────────
  // M3+ (MFA_DIRECT_READS=1): no barrier A (KV_smem not used for K), V loaded from device.
  // TGP path: barrier A gates KV_smem reuse; each V D-chunk loads into KV_smem.
  ss << "#if !MFA_DIRECT_READS\n";
  ss << "    // ─ Barrier A: K phase done → start V loading ─\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "#endif  // !MFA_DIRECT_READS\n";
  ss << "\n";

  // ── Phase 2: P@V (all D-chunks of V) ─────────────────────────────────────
  ss << "    // ─ Phase 2: P@V (all D-chunks) ─\n";
  for (int dh = 0; dh < D_chunks; dh++) {
    if (dh > 0) {
      ss << "#if !MFA_DIRECT_READS\n";
      ss << "    // V[dh=" << (dh-1) << "] reads done → safe to load V[dh=" << dh << "]\n";
      ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      ss << "#endif\n";
    }
    ss << "#if !MFA_DIRECT_READS\n";
    emit_load_v("V_cur + (long)" + std::to_string(dh) + " * MFA_BD_TILE");
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // V[dh=" << dh << "] ready\n";
    ss << "#endif  // !MFA_DIRECT_READS\n";
    emit_v5_pv(dh);
    ss << "\n";
  }

  // Close sparse if(!skip_tile) block
  if (sparse)
    ss << "    }  // end if (!skip_tile)\n\n";

  // ── Advance K_cur/V_cur + preload K[kb+1][dh=0] (TGP path only) ──────────
  ss << "    K_cur += (long)MFA_BK * p->K_strides[2];\n";
  ss << "    V_cur += (long)MFA_BK * p->V_strides[2];\n";
  ss << "\n";
  ss << "#if !MFA_DIRECT_READS\n";
  ss << "    // ─ Barrier X: V reads done → preload K[kb+1][dh=0] ─\n";
  ss << "    if (kb + 1 < kb_lim) {\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // X\n";
  emit_load_k("K_cur");
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // C: K[kb+1][dh=0] ready\n";
  ss << "    }\n";
  ss << "#endif  // !MFA_DIRECT_READS\n";
  ss << "\n";

  ss << "  } // end kb loop\n";
  ss << "\n";

  // ── Normalize Otile (divide each element by sum_score) ────────────────────
  ss << "  Otile.template row_bin_op<MFADivOp>(sum_score);\n";
  ss << "  threadgroup_barrier(mem_flags::mem_none);\n";
  ss << "\n";

  // ── Write O to device — vectorized (same pattern as V2) ─────────────────
  // Otile is MFAMMATile<AccT, TQ=1, TD> with TD=D/8 fragments spanning the
  // full head dimension. store<T,1,1> / store_safe<T,1,1> write all TD fragments
  // in a single vectorized call, matching the V2 pattern.
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
