/// mfa_steel_fwd_v2.cpp  –  STEEL V2 forward kernel: sequential K/V phases.
///
/// Key innovation over V1:
///   - Q_smem loaded ONCE and stays in registers for all K-tile iterations.
///   - K and V share the SAME KV_smem region (sequential, not simultaneous).
///   - Default: BK=32 for D=128 (18,944B TGP). BK=64 evaluated: regression at N≥8192.
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
/// BK=64 for D=128 was evaluated and REVERTED: TK=8 doubles K/P fragment registers,
/// causing register spill at N≥8192 (−27% vs BK=32). BK=32 remains the default.
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

SteelV2BlockConfig select_steel_v2_block_config(int head_dim, bool is_m3_plus) {
  // BQ=32, WM=4, TGP=128 (default): baseline V2 tile config.
  //
  // D=128 BK selection is gen-aware:
  //   M1/M2 (is_m3_plus=false): BK=32 — BK=64 spills registers at N≥8192 on M1 Max.
  //     Root cause: TK=8 doubles K/P accumulator registers alongside pinned BQ×D Q-regs.
  //   M3+  (is_m3_plus=true):  BK=64 — dynamic register allocation provides more headroom.
  //     TGP: Q(8,704B) + KV(max(18,432,17,408)=18,432B) = 27,136B < 32KB ✓
  //     Loader constraints (TGP=128): Q(n_reads=32,TCOLS=4) K(64,1) V(64,2) all ok ✓
  //     Note: unconfirmed on M3+ HW — override with MFA_V2_FORCE_BK=32 if regression.
  //
  // MFA_V2_FORCE_BK=<32|64>: override D=128 BK selection for testing/debugging.
  // MFA_V2_BQ64=1: use BQ=64,WM=8 (research path, all gens — see comment below).
  //
  // TGP memory (BQ=32, WM=4, TGP=128 threads):
  //   D=64  BK=64: Q=32×72×2=4,608B  KV=max(64×72,64×72)×2=9,216B    → 13,824B
  //   D=128 BK=32: Q=32×136×2=8,704B KV=max(128×40,32×136)×2=10,240B → 18,944B  [M1/M2]
  //   D=128 BK=64: Q=32×136×2=8,704B KV=max(128×72,64×136)×2=18,432B → 27,136B  [M3+]
  // TGP memory (BQ=64, WM=8, TGP=256 threads, MFA_V2_BQ64):
  //   D=64:  Q=64×72×2=9,216B  KV=max(64×72,64×72)×2=9,216B            → 18,432B
  //   D=128: Q=64×136×2=17,408B KV=max(128×40,32×136)×2=10,240B        → 27,648B
  //   Both BQ=64 options fully evaluated; N=1024 regression 0.62× → BQ=32 stays default.
  //
  // D=256: BQ=16 retained for source-completeness; routes to V1 in eval_gpu().

  // MFA_V2_FORCE_BK=32|64 — override gen-based BK selection (debug/testing).
  const char* force_bk_env = std::getenv("MFA_V2_FORCE_BK");
  int forced_bk = 0;
  if (force_bk_env) {
    forced_bk = std::atoi(force_bk_env);
    if (forced_bk != 32 && forced_bk != 64) forced_bk = 0;  // ignore invalid values
  }

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
  if (head_dim == 128) {
    // Gen-aware: BK=64 on M3+ (dynamic register file); BK=32 on M1/M2.
    const int bk = forced_bk ? forced_bk : (is_m3_plus ? 64 : 32);
    return {32, bk, 128, 4, 1};  // TQ=1, TD=16; TK=bk/8
  }
  if (head_dim == 256) return {16, 32, 256, 2, 1};  // BQ=16, TQ=1 (not dispatched)
  return {0, 0, 0, 0, 0};  // unsupported (D=512+ needs BD-split)
}

SteelV2BlockConfig select_steel_v2_dsplit_block_config(bool is_m3_plus) {
  // D=256/512 is a separate family: keep BK policy independent from D=128.
  // Global MFA_V2_FORCE_BK (used by D=128 calibration) must not leak here.
  // Optional debug override for this family only:
  //   MFA_V2_FORCE_BK_D256=8|16|32|64
  int forced_bk = 0;
  if (const char* env = std::getenv("MFA_V2_FORCE_BK_D256")) {
    const int parsed = std::atoi(env);
    if (parsed == 8 || parsed == 16 || parsed == 32 || parsed == 64) forced_bk = parsed;
  }

  const int bk = forced_bk ? forced_bk : (is_m3_plus ? 64 : 8);
  return {32, bk, 128, 4, 1};
}

SteelV2BlockConfig select_steel_v2_d512_block_config(bool is_m3_plus) {
  // D=512 ONLY: decoupled from D=256 so autoresearch can iterate independently.
  // MFA_V2_FORCE_BK_D512=<4|8|12|16|20|24|32>: override BK for D=512 testing.
  int forced_bk = 0;
  if (const char* env = std::getenv("MFA_V2_FORCE_BK_D512")) {
    const int parsed = std::atoi(env);
    if (parsed == 4 || parsed == 8 || parsed == 12 || parsed == 16 ||
        parsed == 20 || parsed == 24 || parsed == 32) forced_bk = parsed;
  }
  const int bk = forced_bk ? forced_bk : (is_m3_plus ? 32 : 8);
  return {32, bk, 128, 4, 1};
}

// ---------------------------------------------------------------------------
// V2 kernel source generator
// ---------------------------------------------------------------------------

std::string generate_steel_v2_source(const ShaderCache::KernelKey& key) {
  using KK = ShaderCache::KernelKey;

  // MFA_NO_PADDING=1: set all smem padding to 0 (for benchmarking bank-conflict cost).
  const bool no_padding = (std::getenv("MFA_NO_PADDING") != nullptr);
  const std::string pad_expr = no_padding ? "0" : "16 / sizeof(T)";

  const int D          = key.head_dim;
  const bool causal    = key.causal;
  const bool has_softcap = key.has_softcap;
  const bool has_window  = key.has_window;
  const bool has_alibi   = key.has_alibi;
  const bool sparse      = key.sparse;
  const bool has_rope    = key.has_rope;
  const bool rope_interleaved = key.rope_interleaved;
  const int gqa   = key.gqa_factor;  // H_q / H_kv (1 = standard MHA)

  // V2 only supports f16/bf16
  const char* dtype_str = (key.dtype == 1) ? "bfloat" : "half";

  auto cfg = select_steel_v2_block_config(D, key.is_m3_plus);
  const int BQ = cfg.BQ;   // 32 (default) or 64 (MFA_V2_BQ64)
  const int BK = cfg.BK;   // 64 (D=64) | M1/M2 D=128:32, M3+ D=128:64
  const int WM = cfg.WM;   // 4 (default, TGP=128) or 8 (MFA_V2_BQ64, TGP=256)
  const int WN = 1;
  const int TGP_SIZE = WM * WN * 32;  // 128 (default) or 256 (MFA_V2_BQ64)
  const int TD  = D / 8;       // 8 (D=64) or 16 (D=128)
  const int TK  = BK / 8;      // 8 (D=64) or 4 (D=128)
  const int TQ  = BQ / (WM * WN * 8);  // always 1

  // Unroll: safe for D<=128 (TD=8/16); D=256 (TD=32) causes register spill.
  const bool enable_unroll = (D <= 128) || key.is_m3_plus;
  const int  arch_gen      = key.is_m3_plus ? 15 : 13;

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
  // M3+ direct device reads: bypass TGP for K/V. Disabled when RoPE is active
  // (RoPE requires in-place K modification in threadgroup memory).
  const bool use_direct_reads = key.is_m3_plus && !key.has_rope;
  ss << "#define MFA_DIRECT_READS " << (use_direct_reads ? 1 : 0) << "\n";
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
  if (sparse)
    ss << "    const device uchar* block_mask   [[buffer(6)]],\n";
  if (has_rope) {
    ss << "    const device float* rotary_cos   [[buffer(7)]],\n";
    ss << "    const device float* rotary_sin   [[buffer(8)]],\n";
  }
  if (has_alibi)
    ss << "    const device float* alibi_slopes [[buffer(9)]],\n";
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
  // M3+ (MFA_DIRECT_READS=1): K/V read directly from device; no KV_smem needed.
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

  // ── Block loaders ────────────────────────────────────────────────────────
  ss << "  // Q: row-major, BQ×BD tiles\n";
  ss << "  using QLoader = MFABlockLoaderT<T, MFA_BQ, MFA_BD,\n";
  ss << "      MFA_BD + 16/sizeof(T), 1, 1, MFA_TGP_SIZE>;\n";
  ss << "#if !MFA_DIRECT_READS\n";
  ss << "  // K: transposed into TGP (kDstStrRow=1, kDstStrCol=LDK)\n";
  ss << "  using KLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      1, MFA_BK + 16/sizeof(T), 0, MFA_TGP_SIZE>;\n";
  ss << "  // V: row-major, BK×BD tiles\n";
  ss << "  using VLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      MFA_BD + 16/sizeof(T), 1, 0, MFA_TGP_SIZE>;\n";
  ss << "#endif  // !MFA_DIRECT_READS\n";
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
  ss << "#if !MFA_DIRECT_READS\n";
  ss << "  const short Ks_off = sm * LDK + sn;\n";
  ss << "  const short Vs_off = sm * LDV + sn;\n";
  ss << "#endif\n";
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
  ss << "#if !MFA_DIRECT_READS\n";
  ss << "  KLoader loader_k(K, (int)p->K_strides[2], Ks,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "  VLoader loader_v(V, (int)p->V_strides[2], Vs,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  ss << "#else\n";
  ss << "  // M3+ direct reads: K/V pointers for device memory access.\n";
  ss << "  const int K_stride = (int)p->K_strides[2];  // = D\n";
  ss << "  const int V_stride = (int)p->V_strides[2];  // = D\n";
  ss << "  const device T* K_cur = K;  // advances by BK*K_stride per tile\n";
  ss << "  const device T* V_cur = V;\n";
  ss << "#endif\n";
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
  // RoPE-Q: apply in-place to Q_smem before loading into Qtile registers
  if (has_rope) {
    ss << "  // RoPE-Q: apply rotary embeddings to Q in smem\n";
    ss << "  {\n";
    ss << "    const uint local_id = simd_group_id * 32 + simd_lane_id;\n";
    ss << "    const int qabs_base = p->rope_q_base + qb * MFA_BQ;\n";
    ss << "    for (int ri = (int)local_id; ri < MFA_BQ * (MFA_BD/2);\n";
    ss << "         ri += MFA_TGP_SIZE) {\n";
    ss << "      const int row  = ri / (MFA_BD/2);\n";
    ss << "      const int pair = ri % (MFA_BD/2);\n";
    ss << "      const int cos_idx = (qabs_base + row) * p->rope_cos_stride + pair;\n";
    ss << "      const float cos_v = rotary_cos[cos_idx];\n";
    ss << "      const float sin_v = rotary_sin[cos_idx];\n";
    if (rope_interleaved) {
      ss << "      const int si0 = row * LDQ + pair * 2;\n";
      ss << "      const float q0 = (float)Qs[si0];\n";
      ss << "      const float q1 = (float)Qs[si0 + 1];\n";
      ss << "      Qs[si0]     = (T)(q0 * cos_v - q1 * sin_v);\n";
      ss << "      Qs[si0 + 1] = (T)(q0 * sin_v + q1 * cos_v);\n";
    } else {
      ss << "      const int si0 = row * LDQ + pair;\n";
      ss << "      const int si1 = row * LDQ + pair + MFA_BD/2;\n";
      ss << "      const float q0 = (float)Qs[si0];\n";
      ss << "      const float q1 = (float)Qs[si1];\n";
      ss << "      Qs[si0] = (T)(q0 * cos_v - q1 * sin_v);\n";
      ss << "      Qs[si1] = (T)(q0 * sin_v + q1 * cos_v);\n";
    }
    ss << "    }\n";
    ss << "  }\n";
    ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);  // RoPE-Q writes visible\n";
  }
  ss << "  Qtile.template load_contiguous<T, 1, 1>(&Qs[Qs_off], LDQ);\n";
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
  // Inline lambda to emit RoPE-K for a given kabs_base variable name.
  // Applied after K tile is written to KV_smem and a barrier ensures visibility.
  // Must be followed by another barrier (to make RoPE writes visible for Q@K^T).
  auto emit_rope_k = [&](const std::string& kabs_expr) {
    ss << "    // RoPE-K: apply rotary embeddings to K in KV_smem (transposed)\n";
    ss << "    {\n";
    ss << "      const uint local_id = simd_group_id * 32 + simd_lane_id;\n";
    ss << "      const int kabs_base = " << kabs_expr << ";\n";
    ss << "      for (int ri = (int)local_id; ri < MFA_BK * (MFA_BD/2);\n";
    ss << "           ri += MFA_TGP_SIZE) {\n";
    ss << "        const int k_row = ri % MFA_BK;\n";
    ss << "        const int pair  = ri / MFA_BK;\n";
    ss << "        const int cos_idx = (kabs_base + k_row) * p->rope_cos_stride + pair;\n";
    ss << "        const float cos_v = rotary_cos[cos_idx];\n";
    ss << "        const float sin_v = rotary_sin[cos_idx];\n";
    if (rope_interleaved) {
      ss << "        const int ci0 = pair * 2 * LDK + k_row;\n";
      ss << "        const int ci1 = (pair * 2 + 1) * LDK + k_row;\n";
    } else {
      ss << "        const int ci0 = pair * LDK + k_row;\n";
      ss << "        const int ci1 = (pair + MFA_BD/2) * LDK + k_row;\n";
    }
    ss << "        const float k0 = (float)Ks[ci0];\n";
    ss << "        const float k1 = (float)Ks[ci1];\n";
    ss << "        Ks[ci0] = (T)(k0 * cos_v - k1 * sin_v);\n";
    ss << "        Ks[ci1] = (T)(k0 * sin_v + k1 * cos_v);\n";
    ss << "      }\n";
    ss << "    }\n";
  };

  ss << "#if !MFA_DIRECT_READS\n";
  ss << "  if (kb_lim > kb_start) {\n";
  ss << "    if (kb_start == p->NK_aligned) {\n";
  ss << "      loader_k.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "    } else {\n";
  ss << "      loader_k.load_unsafe();\n";
  ss << "    }\n";
  ss << "    loader_k.next();\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // B0: K[0] visible\n";
  if (has_rope) {
    emit_rope_k("kb_start * MFA_BK");
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // RoPE-K[0] visible\n";
  }
  ss << "  }\n";
  ss << "#endif  // !MFA_DIRECT_READS — skip K preload on M3+\n";
  ss << "\n";

  // ── Main K/V loop ────────────────────────────────────────────────────────
  ss << "  for (int kb = kb_start; kb < kb_lim; kb++) {\n";
  ss << "\n";

  // Sparse tile-skip: uniform branch (all threads in TG share tid.x, kb).
  if (sparse) {
    ss << "    // Block-sparse: skip tiles where block_mask==0 (uniform branch)\n";
    ss << "    const bool skip_tile = !block_mask[\n";
    ss << "        (long)tid.z * p->mask_batch_stride\n";
    ss << "      + (long)tid.y * p->mask_head_stride\n";
    ss << "      + (long)qb * p->NK + kb];\n";
    ss << "    if (!skip_tile) {\n";
  }

  // Phase 1: Q@K^T
  ss << "    // ─ Phase 1: Q@K^T ─\n";
  ss << "#if MFA_DIRECT_READS\n";
  ss << "    // M3+ direct: K_cur points to K[kb*BK, 0] in device memory.\n";
  ss << "    // K is [S, D] row-major. K^T fragment at (d, s): K_cur[s*K_stride + d].\n";
  ss << "    // load<T, row_stride=1, col_stride=K_stride> reads K^T sub-tile.\n";
  ss << "#else\n";
  ss << "    // K[kb] is in KV_smem, already visible via preload barrier.\n";
  ss << "#endif\n";
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

  ss << "#if !MFA_DIRECT_READS\n";
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
  ss << "#endif  // !MFA_DIRECT_READS — no A/B barriers on M3+\n";
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
  ss << "#if MFA_DIRECT_READS\n";
  ss << "          // M3+: V[s, d] row-major. V_cur[(s)*V_stride + d].\n";
  ss << "          Vtile.template load<T, 1, 1>(\n";
  ss << "              V_cur + (long)(sm + (short)(ik * 8)) * V_stride\n";
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

  // Close sparse if(!skip_tile) block; in skip case just advance VLoader
  if (sparse) {
    ss << "    } else {\n";
    ss << "#if !MFA_DIRECT_READS\n";
    ss << "      loader_v.next();  // sparse skip: keep VLoader in sync\n";
    ss << "#endif\n";
    ss << "    }\n";
    ss << "\n";
  }

  // End of K-tile iteration: advance K/V pointers or preload next tile.
  ss << "#if MFA_DIRECT_READS\n";
  ss << "    // M3+: advance device pointers to next K-tile. No barriers needed.\n";
  ss << "    K_cur += (long)MFA_BK * K_stride;\n";
  ss << "    V_cur += (long)MFA_BK * V_stride;\n";
  ss << "#else\n";
  // Barrier X: flush V-reads before K[kb+1] write.
  // Barrier C: K[kb+1] written → visible.
  ss << "    if (kb + 1 < kb_lim) {\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // X\n";
  ss << "      if ((kb + 1) == p->NK_aligned) {\n";
  ss << "        loader_k.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "      } else {\n";
  ss << "        loader_k.load_unsafe();\n";
  ss << "      }\n";
  ss << "      loader_k.next();\n";
  if (has_rope) {
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // C_load: K visible for RoPE\n";
    emit_rope_k("(kb + 1) * MFA_BK");
  }
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // C\n";
  ss << "    }\n";
  ss << "#endif  // !MFA_DIRECT_READS\n";
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
  ss << "    Otile.template store_contiguous<T, 1, 1>(O_write, (int)p->O_strides[2]);\n";
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
// V2 D-split kernel (CP1/CP2): D=256 (2 passes) and D=512 (4 passes)
// ---------------------------------------------------------------------------
//
// Kernel name: "mlx_mfa_v2_dsplit_attention"
// Grid: (NQ, H, B)  — same as V2 single-pass.
//
// Design: BD_HALF=128, D_SPLITS=D/128 (2 for D=256, 4 for D=512).
// Block config: select_steel_v2_dsplit_block_config(is_m3_plus) for BK.
//   M1/M2: BK=8, TK=1   M3+: BK=64, TK=8
// TGP: BQ=32, WM=4, TGP_SIZE=128, TD_HALF=16, TQ=1.
//
// Barrier pattern per K-tile (D=256 / D_SPLITS=2):
//   B_k1 (done K[0]→write K[1]) + K_vis1 + A + V_vis0 + B_v1 + V_vis1 + X + C = 8
//
// Limitations vs V2 single-pass:
//   - No RoPE (GPT-NeoX style spans D-halves — incompatible with D-split smem).
//   - No sparse block_mask (mask sized for V1 BK ≠ D-split BK).
//   - Softcap, ALiBi, sliding window, causal, GQA: fully supported.

std::string generate_steel_v2_dsplit_source(const ShaderCache::KernelKey& key) {
  using KK = ShaderCache::KernelKey;

  const bool no_padding = (std::getenv("MFA_NO_PADDING") != nullptr);
  const std::string pad_expr = no_padding ? "0" : "16 / sizeof(T)";

  const int D          = key.head_dim;  // 256 or 512
  const bool causal    = key.causal;
  const bool has_softcap = key.has_softcap;
  const bool has_window  = key.has_window;
  const bool has_alibi   = key.has_alibi;
  // NOTE: RoPE is NOT supported in D-split (GPT-NeoX pairs d with d+D/2 across D-halves)
  const int gqa   = key.gqa_factor;

  const char* dtype_str = (key.dtype == 1) ? "bfloat" : "half";

  // D-split parameters
  const int BD_HALF  = 128;
  const int D_SPLITS = D / BD_HALF;   // 2 for D=256, 4 for D=512

  // Block config: use D=128 V2 tile config for each BD_HALF pass
  auto cfg = (D == 512)
      ? select_steel_v2_d512_block_config(key.is_m3_plus)
      : select_steel_v2_dsplit_block_config(key.is_m3_plus);
  const int BQ = cfg.BQ;   // 32
  const int BK = cfg.BK;
  const int WM = cfg.WM;   // 4
  const int WN = 1;
  const int TGP_SIZE = WM * WN * 32;  // 128
  const int TD_HALF  = BD_HALF / 8;   // 16
  const int TK       = BK / 8;        // 4 (BK=32) or 8 (BK=64)
  const int TQ       = BQ / (WM * WN * 8);  // always 1

  // TD_HALF=16 is safe to unroll (no register spill like TD=32)
  const bool enable_unroll = true;
  const int  arch_gen      = key.is_m3_plus ? 15 : 13;

  std::ostringstream ss;

  // ── Metal preamble ────────────────────────────────────────────────────────
  append_metal_headers_and_defines(ss, enable_unroll, arch_gen, dtype_str);

  // ── V2 D-split defines ────────────────────────────────────────────────────
  ss << "typedef " << dtype_str << " T;\n";
  ss << "typedef float AccT;\n";
  ss << "#define MFA_BQ  " << BQ       << "\n";
  ss << "#define MFA_BK  " << BK       << "\n";
  ss << "#define MFA_BD  " << D        << "\n";  // full head dim (used for strides only)
  ss << "#define MFA_BD_HALF  " << BD_HALF  << "\n";
  ss << "#define MFA_D_SPLITS " << D_SPLITS << "\n";
  ss << "#define MFA_WM  " << WM       << "\n";
  ss << "#define MFA_WN  " << WN       << "\n";
  ss << "#define MFA_TGP_SIZE  " << TGP_SIZE << "\n";
  ss << "#define MFA_TD_HALF  " << TD_HALF  << "\n";
  ss << "#define MFA_TK  " << TK       << "\n";
  ss << "#define MFA_TQ  " << TQ       << "\n";
  ss << "#define MFA_GQA " << gqa      << "\n";
  ss << "#define MFA_ROWS_PT " << TQ   << "\n";
  ss << "\n";

  // ── Shared templates (BlockLoaderT, MMAFrag, MMATile) ────────────────────
  append_steel_shared_templates(ss);

  // ── MFASteelParams struct (same layout as V2 single-pass) ────────────────
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

  // ── Kernel function signature ─────────────────────────────────────────────
  ss << "[[kernel, max_total_threads_per_threadgroup(MFA_TGP_SIZE)]]\n";
  ss << "void mlx_mfa_v2_dsplit_attention(\n";
  ss << "    const device T*             Q         [[buffer(0)]],\n";
  ss << "    const device T*             K         [[buffer(1)]],\n";
  ss << "    const device T*             V         [[buffer(2)]],\n";
  ss << "    device T*                   O         [[buffer(3)]],\n";
  ss << "    device float*               L         [[buffer(4)]],\n";
  ss << "    constant MFASteelParams*    p         [[buffer(5)]],\n";
  // buffer(6) = block_mask: unused (sparse not supported in D-split)
  // buffer(7/8) = rope: unused (RoPE incompatible with D-split)
  if (has_alibi)
    ss << "    const device float* alibi_slopes [[buffer(9)]],\n";
  ss << "    uint3 tid          [[threadgroup_position_in_grid]],\n";
  ss << "    uint  simd_group_id [[simdgroup_index_in_threadgroup]],\n";
  ss << "    uint  simd_lane_id  [[thread_index_in_simdgroup]])\n";
  ss << "{\n";

  // ── GQA head remapping ────────────────────────────────────────────────────
  ss << "  // tid: (qb_group, H_q_head, batch)\n";
  ss << "  const int h_q  = (int)tid.y;\n";
  ss << "  const int h_kv = h_q / p->gqa_factor;\n";
  ss << "  Q += (long)tid.z * p->Q_strides[0] + (long)h_q  * p->Q_strides[1];\n";
  ss << "  K += (long)tid.z * p->K_strides[0] + (long)h_kv * p->K_strides[1];\n";
  ss << "  V += (long)tid.z * p->V_strides[0] + (long)h_kv * p->V_strides[1];\n";
  ss << "  O += (long)tid.z * p->O_strides[0] + (long)h_q  * p->O_strides[1];\n";
  ss << "\n";

  // ── Threadgroup memory ────────────────────────────────────────────────────
  // Q_smem: BQ × BD_HALF (reused per D-half; smaller than full-D smem)
  // KV_smem: max(K_transposed_smem, V_rowmajor_smem) — BD_HALF wide
  //   K_smem (transposed): (BK+padK) × BD_HALF × sizeof(T)
  //   V_smem (row-major):  BK × (BD_HALF+padV) × sizeof(T)
  ss << "  constexpr short padQ = " << pad_expr << ";\n";
  ss << "  constexpr short padK = " << pad_expr << ";\n";
  ss << "  constexpr short padV = " << pad_expr << ";\n";
  ss << "  constexpr short LDQ  = MFA_BD_HALF + padQ;\n";
  ss << "  constexpr short LDK  = MFA_BK + padK;         // stride for transposed K\n";
  ss << "  constexpr short LDV  = MFA_BD_HALF + padV;\n";
  ss << "  constexpr short kv_s0 = (MFA_BK + padK) * MFA_BD_HALF;   // K transposed\n";
  ss << "  constexpr short kv_s1 = MFA_BK * (MFA_BD_HALF + padV);   // V row-major\n";
  ss << "  constexpr short kv_s  = kv_s0 > kv_s1 ? kv_s0 : kv_s1;\n";
  ss << "\n";
  ss << "  threadgroup T Q_smem[MFA_BQ * (MFA_BD_HALF + padQ)];  // BD_HALF-wide Q tile\n";
  ss << "  threadgroup T KV_smem[kv_s];  // K and V share this buffer sequentially\n";
  ss << "  threadgroup T* Qs = Q_smem;\n";
  ss << "  threadgroup T* Ks = KV_smem;\n";
  ss << "  threadgroup T* Vs = KV_smem;\n";
  ss << "\n";

  // ── Block loader aliases (BD_HALF-wide) ───────────────────────────────────
  ss << "  using QLoader = MFABlockLoaderT<T, MFA_BQ, MFA_BD_HALF,\n";
  ss << "      MFA_BD_HALF + 16/sizeof(T), 1, 1, MFA_TGP_SIZE>;\n";
  ss << "  using KLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD_HALF,\n";
  ss << "      1, MFA_BK + 16/sizeof(T), 0, MFA_TGP_SIZE>;\n";
  ss << "  using VLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD_HALF,\n";
  ss << "      MFA_BD_HALF + 16/sizeof(T), 1, 0, MFA_TGP_SIZE>;\n";
  ss << "\n";

  // ── SIMD coordinate ───────────────────────────────────────────────────────
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

  // ── Tile register arrays (D-split: one tile per D-half) ───────────────────
  ss << "  // D-split: Qtile[D_SPLITS] and Otile[D_SPLITS] hold all D-halves.\n";
  ss << "  // TD_HALF=BD_HALF/8=16 fragments per tile (same as V2 D=128).\n";
  for (int dh = 0; dh < D_SPLITS; dh++) {
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD_HALF> Qtile" << dh << ";\n";
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD_HALF> Otile" << dh << ";\n";
  }
  ss << "  MFAMMATile<AccT, 1,      MFA_TK>       Ktile;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TK>       Stile;\n";
  ss << "  MFAMMATile<AccT, 1,      1>             Vtile;\n";
  ss << "  AccT max_score[MFA_ROWS_PT];\n";
  ss << "  AccT sum_score[MFA_ROWS_PT];\n";
  ss << "\n";

  // ── One Q-block per threadgroup ───────────────────────────────────────────
  ss << "  const int qb = (int)tid.x;\n";
  ss << "  const device T* Q_qb = Q + (long)qb * MFA_BQ * p->Q_strides[2];\n";
  ss << "  device T*       O_qb = O + (long)qb * MFA_BQ * p->O_strides[2];\n";
  ss << "\n";

  // ── Sliding window: O(1) K/V advance ─────────────────────────────────────
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
    ss << "  K += (long)kb_start * MFA_BK * p->K_strides[2];\n";
    ss << "  V += (long)kb_start * MFA_BK * p->V_strides[2];\n";
    ss << "\n";
  } else {
    ss << "  const int kb_start  = 0;\n";
    ss << "  const int kb_last_win = -1;\n";
    ss << "\n";
  }

  // Running K/V tile pointers (advanced manually each K-tile iteration)
  ss << "  const device T* K_cur = K;\n";
  ss << "  const device T* V_cur = V;\n";
  ss << "\n";

  // ── Initialize output accumulators ────────────────────────────────────────
  for (int dh = 0; dh < D_SPLITS; dh++) {
    ss << "  Otile" << dh << ".clear();\n";
  }
  ss << "  STEEL_PRAGMA_UNROLL\n";
  ss << "  for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "    max_score[i] = -INFINITY;\n";
  ss << "    sum_score[i] = 0.0f;\n";
  ss << "  }\n";
  ss << "\n";

  // ── Load Q for each D-half into Qtile registers ──────────────────────────
  // Q_smem is reused per D-half (BD_HALF-wide). Two barriers per dh:
  //   1. Before write: ensure previous dh's Qtile.load is done by all threads.
  //   2. After write: Q data visible in smem for Qtile.load.
  ss << "  // Load all D-halves of Q into Qtile registers\n";
  for (int dh = 0; dh < D_SPLITS; dh++) {
    ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "  if (qb == p->NQ_aligned) {\n";
    ss << "    QLoader(Q_qb + (long)" << dh << " * MFA_BD_HALF,\n";
    ss << "            (int)p->Q_strides[2], Qs,\n";
    ss << "            (ushort)simd_group_id, (ushort)simd_lane_id)\n";
    ss << "        .load_safe(short2(MFA_BD_HALF, p->qL_rem));\n";
    ss << "  } else {\n";
    ss << "    QLoader(Q_qb + (long)" << dh << " * MFA_BD_HALF,\n";
    ss << "            (int)p->Q_strides[2], Qs,\n";
    ss << "            (ushort)simd_group_id, (ushort)simd_lane_id)\n";
    ss << "        .load_unsafe();\n";
    ss << "  }\n";
    ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "  Qtile" << dh << ".template load_contiguous<T, 1, 1>(&Qs[Qs_off], LDQ);\n";
    ss << "\n";
  }

  // ── K-loop limit ──────────────────────────────────────────────────────────
  if (causal) {
    ss << "  int q_max  = (qb + 1) * MFA_BQ + p->qL_off;\n";
    ss << "  int kb_lim = (q_max + MFA_BK - 1) / MFA_BK;\n";
    ss << "  if (kb_lim > p->NK) kb_lim = p->NK;\n";
  } else {
    ss << "  int kb_lim = p->NK;\n";
  }

  // Sliding window right bound
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

  // ── Pre-loop: preload K[kb_start][dh=0] into KV_smem ─────────────────────
  // Barrier B0 ensures K[0] is visible before entering the Q@K^T phase.
  ss << "  // Pre-loop: preload K[kb_start][dh=0] before main loop.\n";
  ss << "  if (kb_lim > kb_start) {\n";
  ss << "    if (kb_start == p->NK_aligned) {\n";
  ss << "      KLoader(K_cur, (int)p->K_strides[2], Ks,\n";
  ss << "              (ushort)simd_group_id, (ushort)simd_lane_id)\n";
  ss << "          .load_safe(short2(MFA_BD_HALF, p->kL_rem));\n";
  ss << "    } else {\n";
  ss << "      KLoader(K_cur, (int)p->K_strides[2], Ks,\n";
  ss << "              (ushort)simd_group_id, (ushort)simd_lane_id)\n";
  ss << "          .load_unsafe();\n";
  ss << "    }\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // B0: K[0] visible\n";
  ss << "  }\n";
  ss << "\n";

  // ── Helper lambdas for repeated K/V load patterns ─────────────────────────
  // emit_load_k: emit safe/unsafe K load from given pointer expression
  auto emit_load_k = [&](const std::string& ptr_expr) {
    ss << "    if (kb == p->NK_aligned) {\n";
    ss << "      KLoader(" << ptr_expr << ", (int)p->K_strides[2], Ks,\n";
    ss << "               (ushort)simd_group_id, (ushort)simd_lane_id)\n";
    ss << "          .load_safe(short2(MFA_BD_HALF, p->kL_rem));\n";
    ss << "    } else {\n";
    ss << "      KLoader(" << ptr_expr << ", (int)p->K_strides[2], Ks,\n";
    ss << "               (ushort)simd_group_id, (ushort)simd_lane_id)\n";
    ss << "          .load_unsafe();\n";
    ss << "    }\n";
  };

  auto emit_load_v = [&](const std::string& ptr_expr) {
    ss << "    if (kb == p->NK_aligned) {\n";
    ss << "      VLoader(" << ptr_expr << ", (int)p->V_strides[2], Vs,\n";
    ss << "               (ushort)simd_group_id, (ushort)simd_lane_id)\n";
    ss << "          .load_safe(short2(MFA_BD_HALF, p->kL_rem));\n";
    ss << "    } else {\n";
    ss << "      VLoader(" << ptr_expr << ", (int)p->V_strides[2], Vs,\n";
    ss << "               (ushort)simd_group_id, (ushort)simd_lane_id)\n";
    ss << "          .load_unsafe();\n";
    ss << "    }\n";
  };

  // emit_qkt: Q@K^T contribution from D-half dh (K already in KV_smem)
  auto emit_qkt = [&](int dh) {
    ss << "    // Q@K^T for dh=" << dh << "\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short dd = 0; dd < MFA_TD_HALF; dd++) {\n";
    ss << "      Ktile.template load_contiguous<T, 1, 1>(\n";
    ss << "          &Ks[Ks_off + (short)(dd * 8) * LDK], LDK);\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short iq = 0; iq < MFA_TQ; iq++) {\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short ik = 0; ik < MFA_TK; ik++) {\n";
    ss << "          MFAMMAFrag<AccT>::mma(\n";
    ss << "              Stile.frag_at(iq, ik),\n";
    ss << "              Qtile" << dh << ".frag_at(iq, dd),\n";
    ss << "              Ktile.frag_at(0, ik),\n";
    ss << "              Stile.frag_at(iq, ik));\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
  };

  // emit_pv: P@V contribution from D-half dh (V already in KV_smem) → Otile[dh]
  auto emit_pv = [&](int dh) {
    ss << "    // P@V for dh=" << dh << "\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short iq = 0; iq < MFA_TQ; iq++) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short ik = 0; ik < MFA_TK; ik++) {\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short id = 0; id < MFA_TD_HALF; id++) {\n";
    ss << "          Vtile.template load_contiguous<T, 1, 1>(\n";
    ss << "              &Vs[Vs_off + ik*8*LDV + id*8], LDV);\n";
    ss << "          MFAMMAFrag<AccT>::mma(\n";
    ss << "              Otile" << dh << ".frag_at(iq, id),\n";
    ss << "              Stile.frag_at(iq, ik),\n";
    ss << "              Vtile.frag_at(0, 0),\n";
    ss << "              Otile" << dh << ".frag_at(iq, id));\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
  };

  // ── Main K-V loop ─────────────────────────────────────────────────────────
  ss << "  for (int kb = kb_start; kb < kb_lim; kb++) {\n";
  ss << "\n";

  // Phase 1: Q@K^T for all D-halves
  // dh=0: K already in KV_smem from pre-loop preload or prev-iter barrier C.
  // dh=1..D_SPLITS-1: load K[kb][dh] into KV_smem before accumulation.
  ss << "    // ─ Phase 1: Q@K^T (all D-halves) ─\n";
  ss << "    Stile.clear();\n";
  emit_qkt(0);  // K[kb][dh=0] is already in KV_smem
  for (int dh = 1; dh < D_SPLITS; dh++) {
    ss << "    // Barrier: done reading K[dh=" << (dh-1) << "], safe to write K[dh=" << dh << "]\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_load_k("K_cur + (long)" + std::to_string(dh) + " * MFA_BD_HALF");
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // K[dh=" << dh << "] visible\n";
    emit_qkt(dh);
  }
  ss << "\n";

  // Apply QK scale (log2 domain)
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++) {\n";
  ss << "      Stile.elems()[ii] *= scale;\n";
  ss << "    }\n";
  ss << "\n";

  // Softcap (Gemma 2 / Grok)
  if (has_softcap) {
    ss << "    // Softcapping: log2→nat, tanh, nat→log2\n";
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

  // K-boundary mask (last K-tile: pad positions → -inf)
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

  // Online softmax (NaN-safe, log2 domain — identical to V2 single-pass)
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
  ss << "    // Rescale ALL D-half output tiles by softmax correction factor\n";
  for (int dh = 0; dh < D_SPLITS; dh++) {
    ss << "    Otile" << dh << ".template row_bin_op<MFAMulOp>(factor);\n";
  }
  ss << "\n";

  // Barrier A: all K reads done (last Q@K^T + register softmax ops)
  // Safe to overwrite KV_smem with V[dh=0].
  ss << "    // ─ Barrier A: K phase complete → safe to load V ─\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "\n";

  // Phase 2: V loading and P@V for all D-halves
  // Between D-halves: barrier to signal all threads done reading V[dh-1]
  // before overwriting with V[dh].
  ss << "    // ─ Phase 2: V loading + P@V for all D-halves ─\n";
  for (int dh = 0; dh < D_SPLITS; dh++) {
    if (dh > 0) {
      ss << "    // Barrier: done reading V[dh=" << (dh-1) << "], safe to write V[dh=" << dh << "]\n";
      ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    }
    emit_load_v("V_cur + (long)" + std::to_string(dh) + " * MFA_BD_HALF");
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);  // V[dh=" << dh << "] visible\n";
    emit_pv(dh);
    ss << "\n";
  }

  // Advance K_cur and V_cur to next K-tile (for next iteration + preload)
  ss << "    K_cur += (long)MFA_BK * p->K_strides[2];\n";
  ss << "    V_cur += (long)MFA_BK * p->V_strides[2];\n";
  ss << "\n";

  // Barrier X + preload K[kb+1][dh=0] for next iteration
  ss << "    // ─ Barrier X: V reads done → preload K[kb+1][dh=0] ─\n";
  ss << "    if (kb + 1 < kb_lim) {\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // X\n";
  ss << "      if ((kb + 1) == p->NK_aligned) {\n";
  ss << "        KLoader(K_cur, (int)p->K_strides[2], Ks,\n";
  ss << "                (ushort)simd_group_id, (ushort)simd_lane_id)\n";
  ss << "            .load_safe(short2(MFA_BD_HALF, p->kL_rem));\n";
  ss << "      } else {\n";
  ss << "        KLoader(K_cur, (int)p->K_strides[2], Ks,\n";
  ss << "                (ushort)simd_group_id, (ushort)simd_lane_id)\n";
  ss << "            .load_unsafe();\n";
  ss << "      }\n";
  ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // C: K[kb+1][0] visible\n";
  ss << "    }\n";
  ss << "\n";

  ss << "  } // end kb loop\n";
  ss << "\n";

  // ── Normalize O (divide by sum_score) ────────────────────────────────────
  ss << "  // Normalize: O /= sum_score for all D-halves\n";
  for (int dh = 0; dh < D_SPLITS; dh++) {
    ss << "  Otile" << dh << ".template row_bin_op<MFADivOp>(sum_score);\n";
  }
  ss << "  threadgroup_barrier(mem_flags::mem_none);\n";
  ss << "\n";

  // ── Write O for each D-half ───────────────────────────────────────────────
  ss << "  // Write output: each D-half at offset dh*BD_HALF within the row\n";
  for (int dh = 0; dh < D_SPLITS; dh++) {
    ss << "  {\n";
    ss << "    device T* O_write = O_qb + (long)(tm + sm) * p->O_strides[2]\n";
    ss << "                       + sn + (long)" << dh << " * MFA_BD_HALF;\n";
    ss << "    if (qb == p->NQ_aligned) {\n";
    ss << "      auto dims = short2((short)(MFA_BD_HALF - sn),\n";
    ss << "                         (short)(p->qL_rem - (tm + sm)));\n";
    ss << "      if (dims.x > 0 && dims.y > 0)\n";
    ss << "        Otile" << dh << ".template store_safe<T, 1, 1>(O_write, (int)p->O_strides[2], dims);\n";
    ss << "    } else {\n";
    ss << "      Otile" << dh << ".template store_contiguous<T, 1, 1>(O_write, (int)p->O_strides[2]);\n";
    ss << "    }\n";
    ss << "  }\n";
  }
  ss << "\n";

  // ── Write L (logsumexp, log2 domain) ────────────────────────────────────
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

int compute_v2_num_splits(int total_tgs, int kL, int BK, int gpu_cores, bool force_splitk) {
  // gpu_cores is the actual estimated core count from estimate_gpu_cores().
  // Skip if already well-occupied
  if (!force_splitk && total_tgs >= (int)(0.8f * (float)gpu_cores)) return 1;

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

  const bool no_padding = (std::getenv("MFA_NO_PADDING") != nullptr);
  const std::string pad_expr = no_padding ? "0" : "16 / sizeof(T)";

  const int D      = key.head_dim;
  const bool causal    = key.causal;
  const bool has_softcap = key.has_softcap;
  const bool has_window  = key.has_window;
  const bool has_alibi   = key.has_alibi;
  const bool has_rope    = key.has_rope;
  const bool rope_interleaved = key.rope_interleaved;
  const int gqa    = key.gqa_factor;
  const char* dtype_str = (key.dtype == 1) ? "bfloat" : "half";

  auto cfg = select_steel_v2_block_config(D, key.is_m3_plus);
  const int BQ      = cfg.BQ;   // 32
  const int BK      = cfg.BK;   // 64 (D=64) | M1/M2 D=128:32, M3+ D=128:64
  const int WM      = cfg.WM;   // 4
  const int WN      = 1;
  const int TGP_SIZE = WM * WN * 32;  // 128
  const int TD      = D  / 8;
  const int TK      = BK / 8;
  const int TQ      = BQ / (WM * WN * 8);  // 1

  const bool enable_unroll = (D <= 128) || key.is_m3_plus;
  const int  arch_gen      = key.is_m3_plus ? 15 : 13;

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
  int rope_q_base;
  int rope_cos_stride;
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
  if (has_rope) {
    ss << "    const device float* rotary_cos   [[buffer(6)]],\n";
    ss << "    const device float* rotary_sin   [[buffer(7)]],\n";
  }
  if (has_alibi) {
    ss << "    const device float* alibi_slopes [[buffer(9)]],\n";
  }
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
  ss << "  const int q_min = qb * MFA_BQ + p->qL_off;\n";
  ss << "  const int q_max = q_min + MFA_BQ;\n";
  ss << "  int kb_start = kb_split_start;\n";
  ss << "  int kb_lim   = kb_split_end;\n";
  if (causal) {
    ss << "  const int kb_causal_lim = min((q_max + MFA_BK - 1) / MFA_BK, p->NK_total);\n";
    ss << "  if (kb_lim > kb_causal_lim) kb_lim = kb_causal_lim;\n";
  }
  if (has_window) {
    ss << "  int kb_last_win = -1;\n";
    ss << "  int kb_first_right = kb_lim;\n";
    ss << "  if (p->window_left >= 0) {\n";
    ss << "    const int win_start = q_min - p->window_left;\n";
    ss << "    const int kb_left_start = (win_start > 0) ? (win_start / MFA_BK) : 0;\n";
    ss << "    if (kb_start < kb_left_start) kb_start = kb_left_start;\n";
    ss << "    kb_last_win = (q_max - 1 > p->window_left)\n";
    ss << "                    ? (q_max - 1 - p->window_left) / MFA_BK : -1;\n";
    ss << "  }\n";
    ss << "  if (p->window_right >= 0) {\n";
    ss << "    const int kb_right_lim = (q_max - 1 + p->window_right) / MFA_BK + 1;\n";
    ss << "    if (kb_lim > kb_right_lim) kb_lim = kb_right_lim;\n";
    ss << "    kb_first_right = (q_min + p->window_right + 1) / MFA_BK;\n";
    ss << "    if (kb_first_right < kb_start) kb_first_right = kb_start;\n";
    ss << "  }\n";
    ss << "  if (kb_lim < kb_start) kb_lim = kb_start;\n";
  }
  ss << "\n";

  // ── Threadgroup memory ────────────────────────────────────────────────────
  ss << "  constexpr short padQ = " << pad_expr << ";\n";
  ss << "  constexpr short padK = " << pad_expr << ";\n";
  ss << "  constexpr short padV = " << pad_expr << ";\n";
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
  // When kb_start >= kb_lim this TG covers no K-tiles.
  // Write pO=0 (pool allocator may not zero-init) and pL=-inf.
  ss << "  if (kb_start >= kb_lim) {\n";
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

  // ── Offset K/V to split+window start (O(1) vs advance-by-loop) ───────────
  ss << "  // Offset K/V pointers to kb_start — O(1) vs V1 advance-by-loop.\n";
  ss << "  K += (long)kb_start * MFA_BK * p->K_strides[2];\n";
  ss << "  V += (long)kb_start * MFA_BK * p->V_strides[2];\n";
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
  // RoPE-Q: apply rotary embeddings in-place to Q_smem before loading into registers
  if (has_rope) {
    ss << "  // RoPE-Q: apply rotary embeddings to Q in smem\n";
    ss << "  {\n";
    ss << "    const uint local_id = simd_group_id * 32 + simd_lane_id;\n";
    ss << "    const int qabs_base = p->rope_q_base + qb * MFA_BQ;\n";
    ss << "    for (int ri = (int)local_id; ri < MFA_BQ * (MFA_BD/2);\n";
    ss << "         ri += MFA_TGP_SIZE) {\n";
    ss << "      const int row  = ri / (MFA_BD/2);\n";
    ss << "      const int pair = ri % (MFA_BD/2);\n";
    ss << "      const int cos_idx = (qabs_base + row) * p->rope_cos_stride + pair;\n";
    ss << "      const float cos_v = rotary_cos[cos_idx];\n";
    ss << "      const float sin_v = rotary_sin[cos_idx];\n";
    if (rope_interleaved) {
      ss << "      const int si0 = row * LDQ + pair * 2;\n";
      ss << "      const float q0 = (float)Qs[si0];\n";
      ss << "      const float q1 = (float)Qs[si0 + 1];\n";
      ss << "      Qs[si0]     = (T)(q0 * cos_v - q1 * sin_v);\n";
      ss << "      Qs[si0 + 1] = (T)(q0 * sin_v + q1 * cos_v);\n";
    } else {
      ss << "      const int si0 = row * LDQ + pair;\n";
      ss << "      const int si1 = row * LDQ + pair + MFA_BD/2;\n";
      ss << "      const float q0 = (float)Qs[si0];\n";
      ss << "      const float q1 = (float)Qs[si1];\n";
      ss << "      Qs[si0] = (T)(q0 * cos_v - q1 * sin_v);\n";
      ss << "      Qs[si1] = (T)(q0 * sin_v + q1 * cos_v);\n";
    }
    ss << "    }\n";
    ss << "  }\n";
    ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);  // RoPE-Q writes visible\n";
  }
  ss << "  Qtile.template load_contiguous<T, 1, 1>(&Qs[Qs_off], LDQ);\n";
  ss << "\n";

  // Inline lambda to emit RoPE-K after a K tile is loaded into KV_smem.
  auto emit_rope_k = [&](const std::string& kabs_expr) {
    ss << "    // RoPE-K: apply rotary embeddings to K in KV_smem (transposed layout)\n";
    ss << "    {\n";
    ss << "      const uint local_id = simd_group_id * 32 + simd_lane_id;\n";
    ss << "      const int kabs_base = " << kabs_expr << ";\n";
    ss << "      for (int ri = (int)local_id; ri < MFA_BK * (MFA_BD/2);\n";
    ss << "           ri += MFA_TGP_SIZE) {\n";
    ss << "        const int k_row = ri % MFA_BK;\n";
    ss << "        const int pair  = ri / MFA_BK;\n";
    ss << "        const int cos_idx = (kabs_base + k_row) * p->rope_cos_stride + pair;\n";
    ss << "        const float cos_v = rotary_cos[cos_idx];\n";
    ss << "        const float sin_v = rotary_sin[cos_idx];\n";
    if (rope_interleaved) {
      ss << "        const int ci0 = pair * 2 * LDK + k_row;\n";
      ss << "        const int ci1 = (pair * 2 + 1) * LDK + k_row;\n";
    } else {
      ss << "        const int ci0 = pair * LDK + k_row;\n";
      ss << "        const int ci1 = (pair + MFA_BD/2) * LDK + k_row;\n";
    }
    ss << "        const float k0 = (float)Ks[ci0];\n";
    ss << "        const float k1 = (float)Ks[ci1];\n";
    ss << "        Ks[ci0] = (T)(k0 * cos_v - k1 * sin_v);\n";
    ss << "        Ks[ci1] = (T)(k0 * sin_v + k1 * cos_v);\n";
    ss << "      }\n";
    ss << "    }\n";
  };

  // ── V2 preload K[kb_start] (Barrier B0) ──────────────────────────────────
  ss << "  // V2: preload first K tile of split before main loop.\n";
  ss << "  if (kb_start == p->NK_aligned) {\n";
  ss << "    loader_k.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "  } else {\n";
  ss << "    loader_k.load_unsafe();\n";
  ss << "  }\n";
  ss << "  loader_k.next();\n";
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);  // B0\n";
  if (has_rope) {
    emit_rope_k("kb_start * MFA_BK");
    ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);  // RoPE-K[0] visible\n";
  }
  ss << "\n";

  // ── Main K/V loop [kb_start, kb_lim) ─────────────────────────────────────
  ss << "  for (int kb = kb_start; kb < kb_lim; kb++) {\n";
  ss << "\n";

  // Phase 1: Q@K^T
  ss << "    // ─ Phase 1: Q@K^T ─\n";
  ss << "    Stile.clear();\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short dd = 0; dd < MFA_TD; dd++) {\n";
  ss << "      Ktile.template load_contiguous<T, 1, 1>(\n";
  ss << "          &Ks[Ks_off + (short)(dd * 8) * LDK], LDK);\n";
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

  if (has_alibi) {
    ss << "    // ALiBi: add per-head linear position bias to scores\n";
    ss << "    {\n";
    ss << "      constexpr AccT log2e = 1.4426950408889634f;\n";
    ss << "      const AccT slope = alibi_slopes[(int)tid.y] * log2e;\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        const int q_pos = q_min + (int)tm + (int)sm + i * 8;\n";
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
    ss << "            if (row < (col + jj)) Stile.frag_at(i, j)[jj] = -INFINITY;\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";
  }

  if (has_window) {
    ss << "    // Window left boundary: mask col < row - window_left\n";
    ss << "    if (kb <= kb_last_win) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        const int row = q_min + tm + sm + i * 8;\n";
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
    ss << "        const int row = q_min + tm + sm + i * 8;\n";
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
  ss << "          Vtile.template load_contiguous<T, 1, 1>(\n";
  ss << "              &Vs[Vs_off + ik*8*LDV + id*8], LDV);\n";
  ss << "          MFAMMAFrag<AccT>::mma(\n";
  ss << "              Otile.frag_at(iq, id), Stile.frag_at(iq, ik),\n";
  ss << "              Vtile.frag_at(0, 0),   Otile.frag_at(iq, id));\n";
  ss << "        }\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "\n";

  // Barriers X + C: preload K[kb+1] (V2 preload optimization)
  // With RoPE: barrier C is split into C_load + RoPE-K + C_rope.
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
  if (has_rope) {
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // C_load: K[kb+1] in smem\n";
    emit_rope_k("(kb + 1) * MFA_BK");
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // C_rope: RoPE-K visible\n";
  } else {
    ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);  // C\n";
  }
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
  ss << "    Otile.template store_contiguous<T, 1, 1>(pO_write, (int)p->D);\n";
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
