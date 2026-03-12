/// mfa_sage_fwd.cpp — SageAttention forward Metal shader generator.
///
/// Implements generate_sage_forward_source() declared in mfa_sage_fwd.hpp.
/// See mfa_sage_fwd.hpp for MFASageParams layout.
///
/// Key differences from STEEL forward (mfa_steel_fwd.cpp):
///   - Q is fp16/bf16 (CP2: no Q quantize dispatch; direct load into TGP).
///   - K is const device char* (int8); dequantized cooperatively in TGP.
///   - V remains fp16; uses standard MFABlockLoaderT (unchanged from STEEL).
///   - Non-persistent grid: (NQ, H, B) — one TG per Q-tile (no 4-tile loop).
///   - No d_split, RoPE, ALiBi, sparse, double_buf in v1.2.0.
///   - Sliding window (has_window=true): same kb_start/kb_lim logic as STEEL.
///   - Buffer layout: Q=0, K=1, V=2, O=3, L=4, params=5, K_scale=6.

#include "mfa_sage_fwd.hpp"
#include "mfa_steel_fwd.hpp"   // append_metal_headers_and_defines, append_steel_shared_templates
#include <sstream>

namespace mlx_mfa {

// =========================================================================
// generate_sage_forward_source
// =========================================================================

std::string generate_sage_forward_source(const ShaderCache::KernelKey& key) {
  const int BD = key.head_dim;
  const int BQ = key.block_q;
  const int BK = key.block_k;
  const int WM = key.n_warps;
  const int WN = 1;
  const bool causal     = key.causal;
  const bool has_window = key.has_window;
  const bool is_m3_plus = key.is_m3_plus;

  // No d_split in Sage v1.2.0 (D <= 256 only)
  const int TD = BD / 8;       // head-dim frags per warp
  const int TK = BK / 8;       // K-seq frags per K tile
  const int TQ = BQ / (WM * WN * 8);  // Q-seq frags per warp (must be 1)
  const int kRowsPT = TQ;

  const char* dtype_str = "half";
  if (key.dtype == 1)      dtype_str = "bfloat";
  else if (key.dtype == 2) dtype_str = "float";

  const int arch_gen        = is_m3_plus ? 15 : 13;
  const bool enable_unroll  = (BD <= 128) || is_m3_plus;

  std::ostringstream ss;

  // ── Preamble ──────────────────────────────────────────────────────────────
  append_metal_headers_and_defines(ss, enable_unroll, arch_gen, dtype_str);

  // ── MFASageParams struct ──────────────────────────────────────────────────
  // Must EXACTLY match the C++ MFASageParams in mfa_sage_fwd.hpp.
  // CP2: q_scale_stride_b/h removed; Q_scale buffer eliminated.
  ss << R"SAGE(
struct MFASageParams {
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
  int rope_q_base;       // unused — kept for layout compatibility
  int rope_cos_stride;   // unused — kept for layout compatibility
  long Q_strides[3];     // [B,H,N] fp16 Q strides (element units; CP2: Q is fp16)
  long K_strides[3];     // [B,H,S] int8 K strides
  long V_strides[3];     // [B,H_kv,S] fp16 strides
  long O_strides[3];     // [B,H,N] fp16 strides
  long L_strides[2];     // [B,H] f32 strides
  float softcap;         // 0.0 = disabled
  int   has_alibi;       // always 0 in Sage
  int   window_left;     // -1 = disabled; >=0 = left window radius (tokens)
  int   window_right;    // -1 = disabled; >=0 = right window radius (tokens)
  // Sage-specific scale index strides (K only; Q_scale eliminated CP2)
  int NQ_blocks;
  int NK_blocks;
  int k_scale_stride_b;  // H_kv * NK_blocks
  int k_scale_stride_h;  // NK_blocks
};

)SAGE";

  // ── Shared Metal templates ────────────────────────────────────────────────
  // MFABlockLoaderT, MFAMMAFrag, MFAMMATile, mfa_tile_matmad, op structs.
  // Identical to the STEEL forward kernel — proven to compile on all gen.
  append_steel_shared_templates(ss);

  // ── Compile-time tile constants ───────────────────────────────────────────
  ss << "#define MFA_BQ  " << BQ  << "\n";
  ss << "#define MFA_BK  " << BK  << "\n";
  ss << "#define MFA_BD  " << BD  << "\n";
  ss << "#define MFA_WM  " << WM  << "\n";
  ss << "#define MFA_WN  " << WN  << "\n";
  ss << "#define MFA_TGP_SIZE  " << (WM * WN * 32) << "\n";
  ss << "#define MFA_DTYPE  " << dtype_str << "\n";
  ss << "#define MFA_TD  " << TD << "\n";
  ss << "#define MFA_TK  " << TK << "\n";
  ss << "#define MFA_TQ  " << TQ << "\n";
  ss << "#define MFA_ROWS_PT  " << kRowsPT << "\n";
  ss << "\n";

  // ── Kernel function ───────────────────────────────────────────────────────
  ss << "[[kernel, max_total_threads_per_threadgroup(MFA_TGP_SIZE)]]\n";
  // CP2: Q is now fp16 (same dtype as V/O). Q_scale buffer eliminated.
  //      K_scale moves from buffer(7) to buffer(6).
  ss << "void mlx_mfa_sage_attention(\n";
  ss << "    const device MFA_DTYPE*       Q       [[buffer(0)]],\n";
  ss << "    const device char*            K       [[buffer(1)]],\n";
  ss << "    const device MFA_DTYPE*       V       [[buffer(2)]],\n";
  ss << "    device MFA_DTYPE*             O       [[buffer(3)]],\n";
  ss << "    device float*                 L       [[buffer(4)]],\n";
  ss << "    const constant MFASageParams* p       [[buffer(5)]],\n";
  ss << "    const device float*           K_scale [[buffer(6)]],\n";
  ss << "    uint simd_lane_id  [[thread_index_in_simdgroup]],\n";
  ss << "    uint simd_group_id [[simdgroup_index_in_threadgroup]],\n";
  ss << "    uint3 tid          [[threadgroup_position_in_grid]])\n";
  ss << "{\n";
  ss << "  typedef MFA_DTYPE T;\n";
  ss << "  typedef float     AccT;\n";
  ss << "\n";

  // Non-persistent: one TG per Q-tile; qb = tid.x
  ss << "  const int qb = (int)tid.x;\n";
  ss << "\n";

  // Pointer offsets into Q, K (int8), V, O (fp16)
  ss << "  const ulong boff = (ulong)tid.z * (ulong)p->Q_strides[0]\n";
  ss << "                   + (ulong)tid.y * (ulong)p->Q_strides[1];\n";
  ss << "  const ulong kv_head = (ulong)tid.y / (ulong)p->gqa_factor;\n";
  ss << "  const ulong kv_boff_k = (ulong)tid.z * (ulong)p->K_strides[0]\n";
  ss << "                        + kv_head      * (ulong)p->K_strides[1];\n";
  ss << "  const ulong kv_boff_v = (ulong)tid.z * (ulong)p->V_strides[0]\n";
  ss << "                        + kv_head      * (ulong)p->V_strides[1];\n";
  ss << "\n";
  ss << "  Q += boff;\n";
  ss << "  K += kv_boff_k;\n";
  ss << "  V += kv_boff_v;\n";
  ss << "  O += (ulong)tid.z * (ulong)p->O_strides[0]\n";
  ss << "     + (ulong)tid.y * (ulong)p->O_strides[1];\n";
  ss << "\n";

  // K_scale pointer offset (per [B, H_kv] slice). Q_scale eliminated (CP2).
  ss << "  const device float* K_scale_bh =\n";
  ss << "      K_scale + (long)tid.z * p->k_scale_stride_b\n";
  ss << "              + (long)kv_head * p->k_scale_stride_h;\n";
  ss << "\n";

  // Threadgroup memory: Q_smem (row-major) + KV_smem (K transposed, V row-major)
  ss << "  constexpr short padQ = 16 / sizeof(T);\n";
  ss << "  constexpr short padK = 16 / sizeof(T);\n";
  ss << "  constexpr short padV = 16 / sizeof(T);\n";
  ss << "  constexpr short LDQ  = MFA_BD + padQ;\n";
  ss << "  constexpr short LDK  = MFA_BK + padK;\n";
  ss << "  constexpr short LDV  = MFA_BD + padV;\n";
  // kv_s0 = K transposed smem (d×BK), kv_s1 = V row-major smem (BK×d)
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

  // V loader (fp16, row-major, reduction_dim=0 → tile_stride = BK * V_stride)
  ss << "  // V stays fp16; use standard block loader (unchanged from STEEL forward)\n";
  ss << "  using VLoader = MFABlockLoaderT<T, MFA_BK, MFA_BD,\n";
  ss << "      /*kDstStrRow=*/ MFA_BD + 16/sizeof(T),\n";
  ss << "      /*kDstStrCol=*/ 1,\n";
  ss << "      /*reduction_dim=*/ 0,\n";
  ss << "      /*tgp_size=*/ MFA_TGP_SIZE>;\n";
  ss << "\n";

  // MMA tile layout vars
  ss << "  const AccT att_scale = p->scale * M_LOG2E_F;\n";
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

  // Tile register declarations
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Qtile;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Otile;\n";
  ss << "  MFAMMATile<AccT, 1,      MFA_TK> Ktile;\n";
  ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TK> Stile;\n";
  ss << "  MFAMMATile<AccT, 1,      1>      Vtile;\n";
  ss << "  AccT max_score[MFA_ROWS_PT];\n";
  ss << "  AccT sum_score[MFA_ROWS_PT];\n";
  ss << "\n";
  ss << "  Otile.clear();\n";
  ss << "  STEEL_PRAGMA_UNROLL\n";
  ss << "  for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
  ss << "    max_score[i] = -INFINITY;\n";
  ss << "    sum_score[i] = 0.0f;\n";
  ss << "  }\n";
  ss << "\n";

  // ── Q fp16 cooperative load (CP2: no quantize/dequantize) ────────────────
  // All TGP_SIZE threads cooperate: thread t handles elements
  // t, t+TGP_SIZE, t+2*TGP_SIZE, ... of the flat BQ*BD tile.
  // Q is now fp16/bf16 — load directly without any int8 conversion.
  ss << "  // Q: cooperative fp16 direct load into Q_smem (CP2: no int8 round-trip)\n";
  ss << "  {\n";
  ss << "    const device T* Q_qb = Q + (long)qb * MFA_BQ * p->Q_strides[2];\n";
  ss << "    const uint local_id = simd_group_id * 32 + simd_lane_id;\n";
  ss << "    for (int elem = (int)local_id; elem < MFA_BQ * MFA_BD;\n";
  ss << "         elem += MFA_TGP_SIZE) {\n";
  ss << "      const int row = elem / MFA_BD;\n";
  ss << "      const int col = elem % MFA_BD;\n";
  // Boundary check: pad rows beyond qL_rem with 0 in the last Q-tile.
  ss << "      const bool valid = (qb < p->NQ_aligned) || (row < p->qL_rem);\n";
  ss << "      Qs[row * LDQ + col] = valid\n";
  ss << "                             ? Q_qb[(long)row * p->Q_strides[2] + col]\n";
  ss << "                             : T(0.0f);\n";
  ss << "    }\n";
  ss << "  }\n";
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "\n";

  // Load dequantized Q tile from TGP into registers (stays there for all K iters)
  ss << "  // Load Q tile from TGP into registers (hoisted outside K loop)\n";
  ss << "  Qtile.template load<T, 1, 1>(&Qs[Qs_off], LDQ, 1);\n";
  ss << "\n";

  // K-loop limit (causal: only iterate K-tiles that Q can attend to)
  if (causal) {
    ss << "  int q_max = (qb + 1) * MFA_BQ + p->qL_off;\n";
    ss << "  int kb_lim = (q_max + MFA_BK - 1) / MFA_BK;\n";
    ss << "  if (kb_lim > p->NK) kb_lim = p->NK;\n";
  } else {
    ss << "  int kb_lim = p->NK;\n";
  }
  // Sliding window: kb_start (left skip), kb_lim clamp (right skip),
  // kb_last_win / kb_first_right for per-element masking in boundary tiles.
  // Mirrors STEEL forward window logic exactly.
  if (has_window) {
    ss << "  int q_min = qb * MFA_BQ + p->qL_off;\n";
    ss << "  int kb_start, kb_last_win;\n";
    ss << "  if (p->window_left >= 0) {\n";
    ss << "    int win_start = q_min - p->window_left;\n";
    ss << "    kb_start = win_start > 0 ? win_start / MFA_BK : 0;\n";
    ss << "    kb_last_win = (q_min + MFA_BQ - 1 > p->window_left)\n";
    ss << "                    ? (q_min + MFA_BQ - 1 - p->window_left) / MFA_BK : -1;\n";
    ss << "  } else {\n";
    ss << "    kb_start = 0;\n";
    ss << "    kb_last_win = -1;\n";
    ss << "  }\n";
    ss << "  int kb_first_right;\n";
    ss << "  if (p->window_right >= 0) {\n";
    ss << "    int kb_right_lim = (q_min + MFA_BQ - 1 + p->window_right) / MFA_BK + 1;\n";
    ss << "    if (kb_right_lim < kb_lim) kb_lim = kb_right_lim;\n";
    ss << "    if (kb_lim < kb_start) kb_lim = kb_start;\n";
    ss << "    kb_first_right = (q_min + p->window_right + 1) / MFA_BK;\n";
    ss << "    if (kb_first_right < kb_start) kb_first_right = kb_start;\n";
    ss << "  } else {\n";
    ss << "    kb_first_right = kb_lim;\n";
    ss << "  }\n";
  } else {
    ss << "  const int kb_start = 0;\n";
    ss << "  const int kb_last_win = -1;\n";
    ss << "  const int kb_first_right = p->NK;\n";
  }
  ss << "\n";

  // V loader (positioned at start of KV sequence for this BH slice)
  ss << "  VLoader loader_v(V, (int)p->V_strides[2], Vs,\n";
  ss << "                   (ushort)simd_group_id, (ushort)simd_lane_id);\n";
  if (has_window) {
    // Advance loader_v to kb_start to keep K/V indices in sync.
    ss << "  for (int _kf = 0; _kf < kb_start; _kf++) loader_v.next();\n";
  }
  ss << "\n";

  // ── Main K/V loop ─────────────────────────────────────────────────────────
  ss << "  for (int kb = kb_start; kb < kb_lim; kb++) {\n";
  ss << "\n";

  // K int8 dequantize + transpose cooperative load into Ks
  // K is stored row-major [S, D] in device memory (int8).
  // In TGP we need it transposed [D, BK] for GEMM: Ks[d_col * LDK + k_row].
  ss << "    // K: cooperative int8 → fp16 dequantize + transpose into Ks\n";
  ss << "    {\n";
  ss << "      const float k_sc = K_scale_bh[kb];\n";
  ss << "      const device char* Kb = K + (long)kb * MFA_BK * p->K_strides[2];\n";
  ss << "      const uint local_id = simd_group_id * 32 + simd_lane_id;\n";
  ss << "      for (int elem = (int)local_id; elem < MFA_BK * MFA_BD;\n";
  ss << "           elem += MFA_TGP_SIZE) {\n";
  ss << "        const int row = elem / MFA_BD;   // token index in K-tile\n";
  ss << "        const int col = elem % MFA_BD;   // head dim index\n";
  ss << "        const bool valid = (kb < p->NK_aligned) || (row < p->kL_rem);\n";
  ss << "        const int8_t raw = valid\n";
  ss << "                           ? Kb[(long)row * p->K_strides[2] + col]\n";
  ss << "                           : (int8_t)0;\n";
  ss << "        Ks[col * LDK + row] = (T)((float)raw * k_sc);\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "\n";

  // S = Q @ K^T
  ss << "    // S = Q @ K^T  (Qtile in registers, Ks transposed in TGP)\n";
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

  // Apply scale (log2 domain for numerically stable softmax)
  ss << "    // Apply scale (log2-domain: scale = 1/sqrt(D) * log2e)\n";
  ss << "    STEEL_PRAGMA_UNROLL\n";
  ss << "    for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++) {\n";
  ss << "      Stile.elems()[ii] *= att_scale;\n";
  ss << "    }\n";
  ss << "\n";

  // K-boundary mask (last K-tile may be partial)
  ss << "    // Mask padded positions in the last (partial) K-tile\n";
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

  // Causal mask (only last few K-tiles need masking)
  if (causal) {
    ss << "    // Causal mask: position k > position q → mask -∞\n";
    ss << "    if (kb >= (kb_lim - (MFA_BQ + MFA_BK - 1) / MFA_BK)) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        const int row = qb * MFA_BQ + p->qL_off\n";
    ss << "                      + tm + sm + i * 8;\n";
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

  // Sliding window masking: boundary tiles only.
  // Left boundary [kb_start, kb_last_win]: mask k < q - window_left.
  // Right boundary [kb_first_right, kb_lim): mask k > q + window_right.
  if (has_window) {
    ss << "    // Window left boundary: mask k < q - window_left\n";
    ss << "    if (kb <= kb_last_win) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        const int row = qb * MFA_BQ + p->qL_off\n";
    ss << "                      + tm + sm + i * 8;\n";
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
    ss << "    // Window right boundary: mask k > q + window_right\n";
    ss << "    if (kb >= kb_first_right) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        const int row = qb * MFA_BQ + p->qL_off\n";
    ss << "                      + tm + sm + i * 8;\n";
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

  // V load: fp16, shares KV_smem with Ks (safe after K-GEMM is done)
  ss << "    // Load V fp16 — barrier ensures K-GEMM finished before overwriting KV_smem\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "    if (kb == p->NK_aligned) {\n";
  ss << "      loader_v.load_safe(short2(MFA_BD, p->kL_rem));\n";
  ss << "    } else {\n";
  ss << "      loader_v.load_unsafe();\n";
  ss << "    }\n";
  ss << "\n";

  // Online softmax (NaN-safe, identical to STEEL forward)
  ss << "    // Online softmax (NaN-safe: handles all-masked tiles)\n";
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

  // P @ V accumulation
  ss << "    // O += P @ V  (V in TGP row-major; identical to STEEL forward)\n";
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
  ss << "    loader_v.next();\n";
  ss << "  } // end kb loop\n";
  ss << "\n";

  // Normalize O and store
  ss << "  // Normalize O by softmax denominator, then write to device memory\n";
  ss << "  Otile.template row_bin_op<MFADivOp>(sum_score);\n";
  ss << "  threadgroup_barrier(mem_flags::mem_none);\n";
  ss << "\n";
  ss << "  device T* O_qb    = O + (long)qb * MFA_BQ * p->O_strides[2];\n";
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

  // Write L (logsumexp in log2 domain — same convention as STEEL forward)
  ss << "  // Write L = max_score + log2(sum_score)  (log2-domain logsumexp)\n";
  ss << "  // Only threads with sn==0 write (first column of each frag row).\n";
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
  ss << "}\n";

  return ss.str();
}

}  // namespace mlx_mfa
