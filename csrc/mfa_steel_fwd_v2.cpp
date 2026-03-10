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
  // BQ=32, WM=4, TGP=128 (default): baseline V2 tile config.
  //
  // Both BQ=64 options were fully evaluated (M1 Max, B=2 H=8 f16):
  //   Option A: BQ=64, WM=4, TQ=2 (128 threads) — regressed ~2× on D=64:
  //     doubling Q_smem (4.6→9.2 KB) reduces TGs/core 2→1. Reverted.
  //   Option B: BQ=64, WM=8, TQ=1 (256 threads) — evaluated via MFA_V2_BQ64:
  //     D=64:  TGP=18,432B → TGs/core 2→1. Neutral large N, small-N noisy.
  //     D=128: TGP=27,648B → same 1 TG/core. Results (BQ64/BQ32 ratio):
  //       N=1024 causal:  0.62× (REGRESSION: 38% slower, register pressure)
  //       N=2048-4096:    0.98–0.99× (neutral)
  //       N=8192 causal:  1.06× (marginal win, within noise)
  //     Decision: N=1024 regression fails threshold. BQ=32 WM=4 stays default.
  //     MFA_V2_BQ64=1 env var retains BQ=64 WM=8 for research use.
  //
  // TGP memory (BQ=32, WM=4, TGP=128 threads):
  //   D=64:  Q=32×72×2=4,608B  KV=max(64×72,64×72)×2=9,216B  → 13,824B
  //   D=128: Q=32×136×2=8,704B KV=max(128×40,32×136)×2=10,240B → 18,944B
  // TGP memory (BQ=64, WM=8, TGP=256 threads):
  //   D=64:  Q=64×72×2=9,216B  KV=max(64×72,64×72)×2=9,216B   → 18,432B
  //   D=128: Q=64×136×2=17,408B KV=max(128×40,32×136)×2=10,240B → 27,648B
  //
  // D=256: BQ=16 retained for source-completeness; routes to V1 in eval_gpu().
  const bool use_bq64 = (std::getenv("MFA_V2_BQ64") != nullptr);
  if (use_bq64) {
    // BQ=64, WM=8, TGP=256 — Option B (256 threads, TQ=1 per simdgroup)
    // MFABlockLoaderT constraints verified (n_reads integer for all loaders):
    //   D=64:  Q(64×64/256=16), K(64×64/256=16), V(64×64/256=16) — all ok
    //   D=128: Q(64×128/256=32), K(128×32/256=16), V(32×128/256=16) — all ok
    if (head_dim == 64)  return {64, 64,  64, 8, 1};  // TQ=1, TK=8, TD=8
    if (head_dim == 128) return {64, 32, 128, 8, 1};  // TQ=1, TK=4, TD=16
    if (head_dim == 256) return {16, 32, 256, 2, 1};  // not dispatched
    return {0, 0, 0, 0, 0};
  }
  if (head_dim == 64)  return {32, 64,  64, 4, 1};  // BQ=32, TQ=1, TK=8, TD=8
  if (head_dim == 128) return {32, 32, 128, 4, 1};  // BQ=32, TQ=1, TK=4, TD=16
  if (head_dim == 256) return {16, 32, 256, 2, 1};  // BQ=16, TQ=1 (not dispatched)
  return {0, 0, 0, 0, 0};  // unsupported (D=512+ needs BD-split)
}

// ---------------------------------------------------------------------------
// V2 kernel source generator
// ---------------------------------------------------------------------------

std::string generate_steel_v2_source(const ShaderCache::KernelKey& key) {
  using KK = ShaderCache::KernelKey;

  const int D          = key.head_dim;
  const bool causal    = key.causal;
  const bool has_softcap = key.has_softcap;
  const bool has_window  = key.has_window;
  const int gqa   = key.gqa_factor;  // H_q / H_kv (1 = standard MHA)

  // V2 only supports f16/bf16
  const char* dtype_str = (key.dtype == 1) ? "bfloat" : "half";

  auto cfg = select_steel_v2_block_config(D);
  const int BQ = cfg.BQ;   // 32 (default) or 64 (MFA_V2_BQ64)
  const int BK = cfg.BK;   // 64 (D=64) or 32 (D=128)
  const int WM = cfg.WM;   // 4 (default, TGP=128) or 8 (MFA_V2_BQ64, TGP=256)
  const int WN = 1;
  const int TGP_SIZE = WM * WN * 32;  // 128 (default) or 256 (MFA_V2_BQ64)
  const int TD  = D / 8;       // 8 (D=64) or 16 (D=128)
  const int TK  = BK / 8;      // 8 (D=64) or 4 (D=128, BK=32)
  const int TQ  = BQ / (WM * WN * 8);  // always 1

  // Unroll: safe for D<=128 (TD=8/16); D=256 (TD=32) causes register spill.
  const bool enable_unroll = (D <= 128) || key.is_m3_plus;
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

  // ── Sliding window: compute kb_start (left bound) + O(1) K/V advance ────
  // Must be done BEFORE loader creation so KLoader/VLoader start at the
  // correct K-tile (no advance-by-loop).
  if (has_window) {
    ss << "  int kb_start  = 0;\n";
    ss << "  int kb_last_win = -1;  // last tile with left-boundary masking\n";
    ss << "  if (p->window_left >= 0) {\n";
    ss << "    const int q_min = qb * MFA_BQ + p->qL_off;\n";
    ss << "    const int win_start = q_min - p->window_left;\n";
    ss << "    kb_start = win_start > 0 ? win_start / MFA_BK : 0;\n";
    ss << "    kb_last_win = (q_min + MFA_BQ - 1 > p->window_left)\n";
    ss << "                    ? (q_min + MFA_BQ - 1 - p->window_left) / MFA_BK : -1;\n";
    ss << "  }\n";
    ss << "  // O(1) K/V advance: shift base pointers to kb_start tile\n";
    ss << "  K += (long)kb_start * MFA_BK * p->K_strides[2];\n";
    ss << "  V += (long)kb_start * MFA_BK * p->V_strides[2];\n";
    ss << "\n";
  } else {
    ss << "  const int kb_start  = 0;\n";
    ss << "  const int kb_last_win = -1;\n";
    ss << "\n";
  }

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
  // kb_start is already declared above (before loaders).
  // Sliding window right bound: clamp kb_lim + track first tile needing masking.
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

  // Sliding window masking (left + right boundaries)
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
    ss << "              Stile.frag_at(i,j)[jj] = -INFINITY;\n";
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
    ss << "              Stile.frag_at(i,j)[jj] = -INFINITY;\n";
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

// ---------------------------------------------------------------------------
// V2 Split-K: num_splits heuristic  (Phase 3)
// ---------------------------------------------------------------------------

int estimate_gpu_cores(const std::string& device_name, int arch_gen) {
  // Longest-prefix-first matching (Ultra > Max > Pro > base).
  // "Apple M1 Max" → 32, "Apple M1" → 8, etc.
  // Falls back to conservative gen-based estimate for unknown names / simulator.
  if (!device_name.empty()) {
    // M1 family
    if (device_name.find("M1 Ultra") != std::string::npos) return 48;
    if (device_name.find("M1 Max")   != std::string::npos) return 32;
    if (device_name.find("M1 Pro")   != std::string::npos) return 16;
    if (device_name.find("M1")       != std::string::npos) return 8;
    // M2 family
    if (device_name.find("M2 Ultra") != std::string::npos) return 60;
    if (device_name.find("M2 Max")   != std::string::npos) return 38;
    if (device_name.find("M2 Pro")   != std::string::npos) return 19;
    if (device_name.find("M2")       != std::string::npos) return 10;
    // M3 family
    if (device_name.find("M3 Ultra") != std::string::npos) return 60;
    if (device_name.find("M3 Max")   != std::string::npos) return 40;
    if (device_name.find("M3 Pro")   != std::string::npos) return 18;
    if (device_name.find("M3")       != std::string::npos) return 10;
    // M4 family
    if (device_name.find("M4 Ultra") != std::string::npos) return 64;
    if (device_name.find("M4 Max")   != std::string::npos) return 40;
    if (device_name.find("M4 Pro")   != std::string::npos) return 20;
    if (device_name.find("M4")       != std::string::npos) return 10;
    // A-series (iPad)
    if (device_name.find("A17")      != std::string::npos) return 6;
    if (device_name.find("A16")      != std::string::npos) return 5;
    if (device_name.find("A15")      != std::string::npos) return 5;
  }
  // Fallback: conservative gen-based estimate (base chip, not Max variant).
  if (arch_gen >= 17) return 40;
  if (arch_gen >= 16) return 20;
  if (arch_gen >= 15) return 18;
  if (arch_gen >= 14) return 19;
  return 8;  // M1 base
}

int compute_v2_num_splits(int total_tgs, int kL, int BK, int gpu_cores) {
  // gpu_cores is the actual estimated core count from estimate_gpu_cores().
  // Skip if already well-occupied
  if (total_tgs >= (int)(0.8f * (float)gpu_cores)) return 1;

  const int NK_total = (kL + BK - 1) / BK;
  if (NK_total < 2) return 1;  // Can't meaningfully split

  // FA2 heuristic: find smallest s s.t. total_tgs*s >= gpu_cores.
  // Hard constraint: each split must cover >= 2 K-tiles (avoid tiny partial sums).
  // Cap at 32 (same as flash decode).
  const int max_by_kv = NK_total / 2;
  const int max_splits = std::min(32, max_by_kv);
  if (max_splits < 2) return 1;

  for (int s = 2; s <= max_splits; s++) {
    if (total_tgs * s >= gpu_cores) return s;
  }
  return max_splits;  // best we can do
}

// ---------------------------------------------------------------------------
// V2 Split-K: partial kernel source  (Phase 3)
// ---------------------------------------------------------------------------
//
// Kernel grid: (NQ * num_splits, H, B)
//   split_id = tid.x / NQ;  qb = tid.x % NQ
// K-loop range: [kb_split_start, min(kb_split_end, kb_causal_lim))
// Empty split: writes pO=0, pL=-inf and returns early.
// Output: pO[split,B,H,qb*BQ:qb*BQ+BQ, D] (normalized partial O, dtype T)
//         pL[split,B,H,qb*BQ:qb*BQ+BQ]     (log2-domain logsumexp, float32)
// Phase 2 reduce: reuses the existing FlashDecodeReduce kernel unchanged.

std::string generate_steel_v2_splitk_partial_source(const ShaderCache::KernelKey& key) {
  using KK = ShaderCache::KernelKey;

  const int D      = key.head_dim;
  const bool causal    = key.causal;
  const bool has_softcap = key.has_softcap;
  const bool has_window  = key.has_window;
  const int gqa    = key.gqa_factor;
  const char* dtype_str = (key.dtype == 1) ? "bfloat" : "half";

  auto cfg = select_steel_v2_block_config(D);
  const int BQ      = cfg.BQ;   // 32
  const int BK      = cfg.BK;   // 64 (D=64) or 32 (D=128)
  const int WM      = cfg.WM;   // 4
  const int WN      = 1;
  const int TGP_SIZE = WM * WN * 32;  // 128
  const int TD      = D  / 8;
  const int TK      = BK / 8;
  const int TQ      = BQ / (WM * WN * 8);  // 1

  const bool enable_unroll = (D <= 128) || key.is_m3_plus;
  const int  arch_gen      = 13;  // placeholder; not used in V2 Metal shader

  std::ostringstream ss;

  // ── Metal preamble ──────────────────────────────────────────────────────
  append_metal_headers_and_defines(ss, enable_unroll, arch_gen, dtype_str);

  // ── Block-size constants ─────────────────────────────────────────────────
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

  // ── Shared templates ─────────────────────────────────────────────────────
  append_steel_shared_templates(ss);

  // ── FlashDecodePartialParams Metal struct ────────────────────────────────
  // Layout MUST match C++ FlashDecodePartialParams in mfa_steel_fwd.hpp.
  ss << R"MFA(
struct MFAFlashDecodePartialParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  int NQ;
  int NQ_aligned;
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
  float softcap;
  int window_left;
  int window_right;
};

)MFA";

  // ── Kernel function ──────────────────────────────────────────────────────
  ss << "[[kernel, max_total_threads_per_threadgroup(MFA_TGP_SIZE)]]\n";
  ss << "void mlx_mfa_v2_splitk_partial(\n";
  ss << "    const device T*                           Q   [[buffer(0)]],\n";
  ss << "    const device T*                           K   [[buffer(1)]],\n";
  ss << "    const device T*                           V   [[buffer(2)]],\n";
  ss << "    device T*                                 pO  [[buffer(3)]],\n";
  ss << "    device float*                             pL  [[buffer(4)]],\n";
  ss << "    constant MFAFlashDecodePartialParams*     p   [[buffer(5)]],\n";
  ss << "    uint3 tid          [[threadgroup_position_in_grid]],\n";
  ss << "    uint  simd_group_id [[simdgroup_index_in_threadgroup]],\n";
  ss << "    uint  simd_lane_id  [[thread_index_in_simdgroup]])\n";
  ss << "{\n";

  // ── Split + Q-block indexing ─────────────────────────────────────────────
  ss << "  // tid.x encodes (split_id, qb): split_id = tid.x / NQ, qb = tid.x % NQ\n";
  ss << "  const int split_id = (int)tid.x / p->NQ;\n";
  ss << "  const int qb       = (int)tid.x % p->NQ;\n";
  ss << "  const int h_q  = (int)tid.y;\n";
  ss << "  const int h_kv = h_q / p->gqa_factor;\n";
  ss << "\n";

  // ── Base pointers per (batch, head) ──────────────────────────────────────
  ss << "  Q  += (long)tid.z * p->Q_strides[0] + (long)h_q  * p->Q_strides[1];\n";
  ss << "  K  += (long)tid.z * p->K_strides[0] + (long)h_kv * p->K_strides[1];\n";
  ss << "  V  += (long)tid.z * p->V_strides[0] + (long)h_kv * p->V_strides[1];\n";
  ss << "  pO += (long)split_id * p->pO_split_stride\n";
  ss << "      + (long)tid.z   * p->pO_batch_stride\n";
  ss << "      + (long)h_q     * p->pO_head_stride\n";
  ss << "      + (long)qb      * MFA_BQ * p->D;\n";
  ss << "  pL += (long)split_id * p->pL_split_stride\n";
  ss << "      + (long)tid.z   * p->pL_batch_stride\n";
  ss << "      + (long)h_q     * p->pL_head_stride\n";
  ss << "      + (long)qb      * MFA_BQ;\n";
  ss << "\n";

  // ── K-loop range for this split ───────────────────────────────────────────
  ss << "  const int kb_split_start = split_id * p->NK_per_split;\n";
  ss << "  const int kb_split_end   = min(kb_split_start + p->NK_per_split, p->NK_total);\n";
  if (causal) {
    ss << "  const int q_max       = (qb + 1) * MFA_BQ + p->qL_off;\n";
    ss << "  const int kb_causal_lim = min((q_max + MFA_BK - 1) / MFA_BK, p->NK_total);\n";
    ss << "  const int kb_lim      = min(kb_split_end, kb_causal_lim);\n";
  } else {
    ss << "  const int kb_lim      = kb_split_end;\n";
  }
  ss << "\n";

  // ── Threadgroup memory ────────────────────────────────────────────────────
  ss << "  constexpr short padQ = 16 / sizeof(T);\n";
  ss << "  constexpr short padK = 16 / sizeof(T);\n";
  ss << "  constexpr short padV = 16 / sizeof(T);\n";
  ss << "  constexpr short LDQ  = MFA_BD + padQ;\n";
  ss << "  constexpr short LDK  = MFA_BK + padK;\n";
  ss << "  constexpr short LDV  = MFA_BD + padV;\n";
  ss << "  constexpr short kv_s0 = (MFA_BK + padK) * MFA_BD;\n";
  ss << "  constexpr short kv_s1 = MFA_BK * (MFA_BD + padV);\n";
  ss << "  constexpr short kv_s  = kv_s0 > kv_s1 ? kv_s0 : kv_s1;\n";
  ss << "  threadgroup T Q_smem[MFA_BQ * (MFA_BD + 16/sizeof(T))];\n";
  ss << "  threadgroup T KV_smem[kv_s];  // K and V share sequentially\n";
  ss << "  threadgroup T* Qs = Q_smem;\n";
  ss << "  threadgroup T* Ks = KV_smem;\n";
  ss << "  threadgroup T* Vs = KV_smem;\n";
  ss << "\n";

  // ── Block loaders ─────────────────────────────────────────────────────────
  ss << "  using QLoader = MFABlockLoaderT<T, MFA_BQ, MFA_BD,\n";
  ss << "      MFA_BD + 16/sizeof(T), 1, 1, MFA_TGP_SIZE>;\n";
  ss << "  using KLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      1, MFA_BK + 16/sizeof(T), 0, MFA_TGP_SIZE>;\n";
  ss << "  using VLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      MFA_BD + 16/sizeof(T), 1, 0, MFA_TGP_SIZE>;\n";
  ss << "\n";

  // ── SIMD coordinate ───────────────────────────────────────────────────────
  ss << "  const AccT scale = p->scale * M_LOG2E_F;\n";
  ss << "  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);\n";
  ss << "  const short sm = simd_coord.y;\n";
  ss << "  const short sn = simd_coord.x;\n";
  ss << "  const short tm = 8 * MFA_TQ * (short)simd_group_id;\n";
  ss << "  const short Qs_off = (tm + sm) * LDQ + sn;\n";
  ss << "  const short Ks_off = sm * LDK + sn;\n";
  ss << "  const short Vs_off = sm * LDV + sn;\n";
  ss << "\n";

  // ── Tile registers ────────────────────────────────────────────────────────
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Qtile;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Otile;\n";
  ss << "  MFAMMATile<AccT, 1,      MFA_TK>  Ktile;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TK>  Stile;\n";
  ss << "  MFAMMATile<AccT, 1,      1>        Vtile;\n";
  ss << "  AccT max_score[MFA_ROWS_PT];\n";
  ss << "  AccT sum_score[MFA_ROWS_PT];\n";
  ss << "\n";

  // ── Empty split early exit ────────────────────────────────────────────────
  // When kb_split_start >= kb_lim this TG covers no K-tiles.
  // Write pO=0 (pool allocator may not zero-init) and pL=-inf.
  ss << "  if (kb_split_start >= kb_lim) {\n";
  ss << "    // Write pO = 0 cooperatively (flat D-major loop)\n";
  ss << "    const uint tgp_tid = (uint)simd_group_id * 32u + (uint)simd_lane_id;\n";
  ss << "    const uint n_rows_empty = (qb == p->NQ_aligned) ? (uint)p->qL_rem : MFA_BQ;\n";
  ss << "    for (uint kk = tgp_tid; kk < n_rows_empty * MFA_BD; kk += MFA_TGP_SIZE)\n";
  ss << "      pO[kk] = T(0);\n";
  ss << "    // Write pL = -INFINITY for all valid q positions\n";
  ss << "    if (sn == 0) {\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "        const long abs_q = (long)qb * MFA_BQ + tm + sm + i * 8;\n";
  ss << "        if (abs_q < p->qL) pL[tm + sm + i * 8] = -INFINITY;\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "    return;\n";
  ss << "  }\n";
  ss << "\n";

  // ── Offset K/V to split start (O(1) vs advance-by-loop) ──────────────────
  ss << "  // Offset K/V pointers to kb_split_start — O(1) vs V1 advance-by-loop.\n";
  ss << "  K += (long)kb_split_start * MFA_BK * p->K_strides[2];\n";
  ss << "  V += (long)kb_split_start * MFA_BK * p->V_strides[2];\n";
  ss << "\n";

  // ── Block loaders (from offset pointers) ─────────────────────────────────
  ss << "  const device T* Q_qb = Q + (long)qb * MFA_BQ * p->Q_strides[2];\n";
  ss << "  QLoader loader_q(Q_qb, (int)p->Q_strides[2], Qs,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "  KLoader loader_k(K, (int)p->K_strides[2], Ks,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "  VLoader loader_v(V, (int)p->V_strides[2], Vs,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "\n";

  // ── Init accumulators ─────────────────────────────────────────────────────
  ss << "  Otile.clear();\n";
  ss << "  STEEL_PRAGMA_UNROLL\n";
  ss << "  for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "    max_score[i] = -INFINITY;\n";
  ss << "    sum_score[i] = 0.0f;\n";
  ss << "  }\n";
  ss << "\n";

  // ── Load Q tile ───────────────────────────────────────────────────────────
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "  if (qb == p->NQ_aligned) {\n";
  ss << "    loader_q.load_safe(short2(MFA_BD, p->qL_rem));\n";
  ss << "  } else {\n";
  ss << "    loader_q.load_unsafe();\n";
  ss << "  }\n";
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "  Qtile.template load<T, 1, 1>(&Qs[Qs_off], LDQ, 1);\n";
  ss << "\n";

  // ── V2 preload K[kb_split_start] (Barrier B0) ────────────────────────────
  ss << "  // V2: preload first K tile of split before main loop.\n";
  ss << "  if (kb_split_start == p->NK_aligned) {\n";
  ss << "    loader_k.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "  } else {\n";
  ss << "    loader_k.load_unsafe();\n";
  ss << "  }\n";
  ss << "  loader_k.next();\n";
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);  // B0\n";
  ss << "\n";

  // ── Main K/V loop [kb_split_start, kb_lim) ───────────────────────────────
  ss << "  for (int kb = kb_split_start; kb < kb_lim; kb++) {\n";
  ss << "\n";

  // Phase 1: Q@K^T
  ss << "    // ─ Phase 1: Q@K^T ─\n";
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
  ss << "              Stile.frag_at(iq, ik), Qtile.frag_at(iq, dd),\n";
  ss << "              Ktile.frag_at(0, ik),  Stile.frag_at(iq, ik));\n";
  ss << "        }\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "\n";

  // Scale
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++)\n";
  ss << "      Stile.elems()[ii] *= scale;\n";
  ss << "\n";

  // Softcap (Gemma 2 / Grok)
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
    ss << "            if (row < (col + jj)) Stile.frag_at(i, j)[jj] = -INFINITY;\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";
  }

  // Online softmax
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

  // Barrier A: K reads done → safe to load V
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // A\n";
  ss << "\n";

  // Load V
  ss << "    if (kb == p->NK_aligned) {\n";
  ss << "      loader_v.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "    } else {\n";
  ss << "      loader_v.load_unsafe();\n";
  ss << "    }\n";
  ss << "    loader_v.next();\n";
  ss << "\n";

  // Barrier B: V loaded
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // B\n";
  ss << "\n";

  // Phase 3: P@V
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short iq = 0; iq < MFA_TQ; iq++) {\n";
  ss << "      STEEL_PRAGMA_UNROLL\n";
  ss << "      for (short ik = 0; ik < MFA_TK; ik++) {\n";
  ss << "        STEEL_PRAGMA_UNROLL\n";
  ss << "        for (short id = 0; id < MFA_TD; id++) {\n";
  ss << "          Vtile.template load<T, 1, 1>(\n";
  ss << "              &Vs[Vs_off + ik*8*LDV + id*8], LDV, 1);\n";
  ss << "          MFAMMAFrag<AccT>::mma(\n";
  ss << "              Otile.frag_at(iq, id), Stile.frag_at(iq, ik),\n";
  ss << "              Vtile.frag_at(0, 0),   Otile.frag_at(iq, id));\n";
  ss << "        }\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "\n";

  // Barriers X + C: preload K[kb+1] (V2 preload optimization)
  ss << "    // ─ X: P@V V-reads done → safe to overwrite KV_smem ─\n";
  ss << "    // ─ C: K[kb+1] visible for next iteration's Q@K^T  ─\n";
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

  // ── Normalize O and write pO ──────────────────────────────────────────────
  ss << "  Otile.template row_bin_op<MFADivOp>(sum_score);\n";
  ss << "  threadgroup_barrier(mem_flags::mem_none);\n";
  ss << "\n";
  ss << "  device T* pO_write = pO + (long)(tm + sm) * p->D + sn;\n";
  ss << "  if (qb == p->NQ_aligned) {\n";
  ss << "    auto dims = short2((short)(MFA_BD - sn),\n";
  ss << "                       (short)(p->qL_rem - (tm + sm)));\n";
  ss << "    if (dims.x > 0 && dims.y > 0)\n";
  ss << "      Otile.template store_safe<T, 1, 1>(pO_write, (int)p->D, dims);\n";
  ss << "  } else {\n";
  ss << "    Otile.template store<T, 1, 1>(pO_write, (int)p->D);\n";
  ss << "  }\n";
  ss << "\n";

  // ── Write pL (log2-domain logsumexp) ─────────────────────────────────────
  // Use LOCAL tile index (tm+sm+i*8) as the write address: pL is already
  // advanced to the qb-tile base (pL += qb*MFA_BQ) at kernel entry.
  // abs_q is the BOUNDS CHECK only — not the write address.
  ss << "  if (sn == 0) {\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "      const long abs_q = (long)qb * MFA_BQ + tm + sm + (long)i * 8;\n";
  ss << "      if (abs_q < p->qL)\n";
  ss << "        pL[tm + sm + (long)i * 8] = max_score[i] + metal::log2(sum_score[i]);\n";
  ss << "    }\n";
  ss << "  }\n";
  ss << "\n";

  ss << "}\n";

  return ss.str();
}

}  // namespace mlx_mfa
