// mfa_steel_paged_varlen_fwd.cpp
// ===========================================================================
// JIT Metal shader generator for the fused PagedVarlenForward kernel.
//
// Combines:
//   • Varlen grid scheduling from generate_steel_varlen_forward_source():
//     one threadgroup per (global_q_tile, head, batch=0). The global_q_tile
//     maps to a (seq_id, local_tile_id) pair via tile_offsets[].
//   • Paged KV gather from generate_paged_steel_forward_source():
//     K and V tokens are fetched from a physical pool via block_table lookup
//     instead of a contiguous K/V tensor.
//
// Grid: (total_q_tiles, H, 1)
//   tid.x = global Q-tile index   (maps to seq_id + local Q-tile within seq)
//   tid.y = head index             (0 .. H-1)
//   tid.z = unused (always 0)
//
// Buffer layout:
//   buffer(0): Q          packed [total_q, H, D] — stride = Q_head_stride per head
//   buffer(1): k_pool     [num_blocks, block_size, H_kv, D]
//   buffer(2): v_pool     [num_blocks, block_size, H_kv, D]
//   buffer(3): O          packed [total_q, H, D]
//   buffer(4): L          [H, total_q]  float32 logsumexp
//   buffer(5): params     MFAPagedVarlenParams (constant)
//   buffer(6): cu_seqlens_q  [num_seqs+1] int32 cumulative Q lengths
//   buffer(7): tile_offsets  [num_seqs+1] int32 cumulative Q-tile counts
//   buffer(8): block_table   [num_seqs, max_blocks] int32
//   buffer(9): seq_lens_kv   [num_seqs] int32 effective KV length per sequence
// ===========================================================================

#include "mfa_steel_paged_varlen_fwd.hpp"
#include "mfa_steel_fwd.hpp"  // SteelBlockConfig, select_steel_block_config,
                              // append_metal_headers_and_defines,
                              // append_steel_shared_templates
#include <sstream>

namespace mlx_mfa {

std::string generate_paged_varlen_forward_source(const ShaderCache::KernelKey& key) {
    const int BD      = key.head_dim;
    const int BQ      = key.block_q;
    const int BK      = key.block_k;
    const int WM      = key.n_warps;
    const int TGP_SIZE = WM * 32;
    const bool causal  = key.causal;
    const bool is_m3_plus = key.is_m3_plus;

    const char* dtype_str = "half";
    if (key.dtype == 1)      dtype_str = "bfloat";
    else if (key.dtype == 2) dtype_str = "float";

    const int arch_gen     = is_m3_plus ? 15 : 13;
    const bool enable_unroll = (BD <= 128) || is_m3_plus;

    // Tile fragment counts
    const int TD      = BD / 8;
    const int TK      = BK / 8;
    const int TQ      = BQ / (WM * 8);   // rows-per-warp (ROWS_PT)
    const int ROWS_PT = TQ;

    // How many elements each thread writes during the cooperative KV gather.
    // Each gather covers BK * BD elements; there are TGP_SIZE threads.
    const int kv_tile_elems    = BK * BD;
    const int elems_per_thread = (kv_tile_elems + TGP_SIZE - 1) / TGP_SIZE;

    std::ostringstream ss;

    // ── Metal preamble + STEEL_PRAGMA_UNROLL ─────────────────────────────────
    append_metal_headers_and_defines(ss, enable_unroll, arch_gen, dtype_str);

    // ── MFAPagedVarlenParams struct (Metal side) ──────────────────────────────
    // Layout MUST match the C++ struct MFAPagedVarlenParams above.
    ss << R"MFA(
struct MFAPagedVarlenParams {
  int H, D;
  int gqa_factor;
  int num_seqs;
  int total_q;
  int total_q_tiles;
  float scale;
  float softcap;
  long Q_head_stride;
  int block_size;
  int max_blocks;
  int pool_block_stride;
  int pool_tok_stride;
  int H_kv;
  int window_left;
  int window_right;
  int num_blocks;
};

)MFA";

    // ── Shared templates: BlockLoaderT, MMAFrag, MMATile, op structs ─────────
    append_steel_shared_templates(ss);

    // ── Compile-time tile constants ───────────────────────────────────────────
    ss << "#define MFA_DTYPE    " << dtype_str   << "\n";
    ss << "#define MFA_BQ       " << BQ          << "\n";
    ss << "#define MFA_BK       " << BK          << "\n";
    ss << "#define MFA_BD       " << BD          << "\n";
    ss << "#define MFA_PAD      (16/sizeof(MFA_DTYPE))\n";
    ss << "#define MFA_TGP_SIZE " << TGP_SIZE    << "\n";
    ss << "#define MFA_TQ       " << TQ          << "\n";
    ss << "#define MFA_TK       " << TK          << "\n";
    ss << "#define MFA_TD       " << TD          << "\n";
    ss << "#define MFA_ROWS_PT  " << ROWS_PT     << "\n";
    ss << "#define CAUSAL       " << (causal ? 1 : 0) << "\n";
    ss << "\n";

    // ── Kernel signature ──────────────────────────────────────────────────────
    ss << "[[kernel, max_total_threads_per_threadgroup(MFA_TGP_SIZE)]]\n";
    ss << "void mlx_mfa_paged_varlen_forward(\n";
    ss << "    const device MFA_DTYPE* Q                        [[buffer(0)]],\n";
    ss << "    const device MFA_DTYPE* k_pool                   [[buffer(1)]],\n";
    ss << "    const device MFA_DTYPE* v_pool                   [[buffer(2)]],\n";
    ss << "    device MFA_DTYPE*       O                        [[buffer(3)]],\n";
    ss << "    device float*           L                        [[buffer(4)]],\n";
    ss << "    const constant MFAPagedVarlenParams* p           [[buffer(5)]],\n";
    ss << "    const device int*       cu_seqlens_q             [[buffer(6)]],\n";
    ss << "    const device int*       tile_offsets             [[buffer(7)]],\n";
    ss << "    const device int*       block_table              [[buffer(8)]],\n";
    ss << "    const device int*       seq_lens_kv              [[buffer(9)]],\n";
    ss << "    uint simd_lane_id  [[thread_index_in_simdgroup]],\n";
    ss << "    uint simd_group_id [[simdgroup_index_in_threadgroup]],\n";
    ss << "    uint3 tid          [[threadgroup_position_in_grid]])\n";
    ss << "{\n";
    ss << "  typedef MFA_DTYPE T;\n";
    ss << "  typedef float     AccT;\n";
    ss << "\n";

    // ── Seq-ID resolution (varlen pattern) ───────────────────────────────────
    // Each threadgroup covers one Q-tile. We binary-search tile_offsets[] to
    // find which sequence this global tile belongs to.
    ss << "  const int global_q_tile = (int)tid.x;\n";
    ss << "  int seq_id = p->num_seqs - 1;\n";
    ss << "  for (int s = 0; s < p->num_seqs - 1; s++) {\n";
    ss << "    if (global_q_tile < tile_offsets[s + 1]) { seq_id = s; break; }\n";
    ss << "  }\n";
    ss << "  const int local_tile_id = global_q_tile - tile_offsets[seq_id];\n";
    ss << "  const int q_start  = cu_seqlens_q[seq_id];\n";
    ss << "  const int q_end    = cu_seqlens_q[seq_id + 1];\n";
    ss << "  const int qL_local = q_end - q_start;\n";
    ss << "  const int kL_local = seq_lens_kv[seq_id];\n";
    ss << "  const int NK_local   = (kL_local + MFA_BK - 1) / MFA_BK;\n";
    ss << "  const int NK_aligned = kL_local / MFA_BK;\n";
    ss << "  const int kL_rem     = kL_local % MFA_BK;\n";
    ss << "  const int NQ_aligned = qL_local / MFA_BQ;\n";
    ss << "  const int qL_rem_local = qL_local % MFA_BQ;\n";
    ss << "\n";

    // ── Pointer setup ─────────────────────────────────────────────────────────
    // Q/O are packed as [total_q, H, D]; stride per head = Q_head_stride = total_q * D.
    // Within a head, token i is at offset i * D.
    // This tile starts at token (q_start + local_tile_id * BQ).
    ss << "  const int kv_head = (int)tid.y / p->gqa_factor;\n";
    ss << "  Q += (long)tid.y * p->Q_head_stride\n";
    ss << "     + (long)(q_start + local_tile_id * MFA_BQ) * MFA_BD;\n";
    ss << "  O += (long)tid.y * p->Q_head_stride\n";
    ss << "     + (long)(q_start + local_tile_id * MFA_BQ) * MFA_BD;\n";
    ss << "  L += (long)tid.y * p->total_q + q_start;\n";
    ss << "\n";

    // ── Scale (log2 domain for fast exp2) ────────────────────────────────────
    ss << "  const AccT scale_v = p->scale * M_LOG2E_F;\n";
    ss << "\n";

    // ── MMA thread coordinates ────────────────────────────────────────────────
    ss << "  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);\n";
    ss << "  const short sm = simd_coord.y;\n";
    ss << "  const short sn = simd_coord.x;\n";
    ss << "  const short tm = 8 * MFA_TQ * (short)simd_group_id;\n";
    ss << "  const int thread_idx = (int)simd_group_id * 32 + (int)simd_lane_id;\n";
    ss << "\n";

    // ── Threadgroup SMEM layout ───────────────────────────────────────────────
    // Q_smem: [BQ, BD+pad] row-major
    // KV_smem: shared between:
    //   K (transposed): [BD, BK+pad]  — K_smem[d*LDK + t]
    //   V (row-major):  [BK, BD+pad]  — V_smem[t*LDV + d]
    // Take max of both to size a single shared buffer.
    ss << "  constexpr short padQ  = MFA_PAD;\n";
    ss << "  constexpr short LDQ   = MFA_BD + padQ;\n";
    ss << "  constexpr short LDK   = MFA_BK + (short)(16/sizeof(T));\n";
    ss << "  constexpr short LDV   = MFA_BD + (short)(16/sizeof(T));\n";
    ss << "  constexpr short kv_s0 = LDK * MFA_BD;      // transposed K size\n";
    ss << "  constexpr short kv_s1 = MFA_BK * LDV;      // row-major V size\n";
    ss << "  constexpr short kv_s  = kv_s0 > kv_s1 ? kv_s0 : kv_s1;\n";
    ss << "\n";
    ss << "  threadgroup T Q_smem[MFA_BQ * (MFA_BD + MFA_PAD)];\n";
    ss << "  threadgroup T KV_smem[kv_s];\n";
    ss << "  threadgroup T* Qs = Q_smem;\n";
    ss << "  threadgroup T* Ks = KV_smem;\n";
    ss << "  threadgroup T* Vs = KV_smem;\n";
    ss << "\n";

    // ── Q loader (BlockLoaderT, row-major into Q_smem) ────────────────────────
    // Template params: <T, BROWS=BQ, BCOLS=BD, kDstStrRow=BD+pad, kDstStrCol=1,
    //                     reduction_dim=1 (Q is loaded along D), tgp_size>
    ss << "  using QLoader = MFABlockLoaderT<T, MFA_BQ, MFA_BD,\n";
    ss << "      /*kDstStrRow=*/ MFA_BD + MFA_PAD,\n";
    ss << "      /*kDstStrCol=*/ 1,\n";
    ss << "      /*reduction_dim=*/ 1,\n";
    ss << "      /*tgp_size=*/ MFA_TGP_SIZE>;\n";
    ss << "  QLoader loader_q(Q, MFA_BD, Qs, (ushort)simd_group_id, (ushort)simd_lane_id);\n";
    ss << "\n";

    // ── Load Q into SMEM → registers ─────────────────────────────────────────
    ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "  if (local_tile_id == NQ_aligned && qL_rem_local > 0) {\n";
    ss << "    loader_q.load_safe(short2(MFA_BD, (short)qL_rem_local));\n";
    ss << "  } else {\n";
    ss << "    loader_q.load_unsafe();\n";
    ss << "  }\n";
    ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "\n";

    // SMEM-to-register offsets
    ss << "  const short Qs_off = (tm + sm) * LDQ + sn;\n";
    ss << "  const short Ks_off = sm * LDK + sn;\n";
    ss << "  const short Vs_off = sm * LDV + sn;\n";
    ss << "\n";

    // ── Register tile declarations ────────────────────────────────────────────
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Qtile;\n";
    ss << "  MFAMMATile<AccT, 1,      MFA_TK>  Ktile;\n";
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TK> Stile;\n";
    ss << "  MFAMMATile<AccT, 1,      1>       Vtile;\n";
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Otile;\n";
    ss << "\n";

    ss << "  Otile.clear();\n";
    ss << "  Qtile.template load<T, 1, 1>(&Qs[Qs_off], LDQ, 1);\n";
    ss << "\n";

    // ── Online softmax state ──────────────────────────────────────────────────
    ss << "  AccT max_score[MFA_ROWS_PT];\n";
    ss << "  AccT sum_score[MFA_ROWS_PT];\n";
    ss << "  STEEL_PRAGMA_UNROLL\n";
    ss << "  for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
    ss << "    max_score[i] = -INFINITY;\n";
    ss << "    sum_score[i] = 0.0f;\n";
    ss << "  }\n";
    ss << "\n";

    // ── Causal K-loop limit setup ─────────────────────────────────────────────
    // qL_off: when qL_local < kL_local (cross-seq causal), shift query positions
    // so that local Q row 0 corresponds to absolute position (kL_local - qL_local).
    if (causal) {
        ss << "  const int qL_off = (qL_local < kL_local) ? (kL_local - qL_local) : 0;\n";
        ss << "  const int q_max_pos = qL_off + (local_tile_id + 1) * MFA_BQ;\n";
        ss << "  const int kb_lim = min(NK_local, (q_max_pos + MFA_BK - 1) / MFA_BK);\n";
    } else {
        ss << "  const int qL_off = 0;\n";
        ss << "  const int kb_lim = NK_local;\n";
    }
    ss << "\n";

    // ── Main K-tile loop ──────────────────────────────────────────────────────
    ss << "  for (int kb = 0; kb < kb_lim; kb++) {\n";
    ss << "\n";

    // Barrier: ensure previous iteration's P@V reads from KV_smem (V) are done
    // before we overwrite KV_smem with the new K tile.
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "\n";

    // ── Paged K gather: transposed layout K_smem[d * LDK + t] ────────────────
    // Each thread fetches elems_per_thread elements. Slot = thread_idx + ei*TGP_SIZE.
    // Decompose slot as: t = slot % BK  (seq position within tile)
    //                    d = slot / BK  (head-dim index)
    // Physical page: block_table[seq_id * max_blocks + blk_idx]
    ss << "    {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short ei = 0; ei < " << elems_per_thread << "; ei++) {\n";
    ss << "        const int slot = thread_idx + (int)ei * MFA_TGP_SIZE;\n";
    ss << "        if (slot < MFA_BK * MFA_BD) {\n";
    ss << "          const int t = slot % MFA_BK;\n";
    ss << "          const int d = slot / MFA_BK;\n";
    ss << "          const int global_tok = kb * MFA_BK + t;\n";
    ss << "          T val = T(0);\n";
    ss << "          if (global_tok < kL_local) {\n";
    ss << "            const int blk_idx    = global_tok / p->block_size;\n";
    ss << "            const int tok_in_blk = global_tok % p->block_size;\n";
    ss << "            // OOB guards (CC-02): blk_idx within block_table, phys within pool.\n";
    ss << "            if (blk_idx < p->max_blocks) {\n";
    ss << "              const int phys = block_table[seq_id * p->max_blocks + blk_idx];\n";
    ss << "              if (phys >= 0 && phys < p->num_blocks) {\n";
    ss << "                val = k_pool[(long)phys * p->pool_block_stride\n";
    ss << "                          + tok_in_blk * p->pool_tok_stride\n";
    ss << "                          + kv_head * p->D + d];\n";
    ss << "              }\n";
    ss << "            }\n";
    ss << "          }\n";
    ss << "          Ks[d * LDK + t] = val;\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "\n";

    // ── Q @ K^T → Stile ──────────────────────────────────────────────────────
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

    // ── Scale (log2 domain) ───────────────────────────────────────────────────
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++) {\n";
    ss << "      Stile.elems()[ii] *= scale_v;\n";
    ss << "    }\n";
    ss << "\n";

    // ── K-boundary mask: positions >= kL_local → -inf ────────────────────────
    // Prevents zero-padded K rows from contributing to normalization.
    ss << "    if ((kb + 1) * MFA_BK > kL_local) {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
    ss << "        STEEL_PRAGMA_UNROLL\n";
    ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
    ss << "          const int k_base = kb * MFA_BK + sn + j * 8;\n";
    ss << "          STEEL_PRAGMA_UNROLL\n";
    ss << "          for (short jj = 0; jj < 2; jj++) {\n";
    ss << "            if ((k_base + jj) >= kL_local)\n";
    ss << "              Stile.frag_at(i,j)[jj] = -INFINITY;\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";

    // ── Causal mask ───────────────────────────────────────────────────────────
    // Only applied near the diagonal (last few K-tiles) for efficiency.
    // Row = absolute query position; Col = absolute key position.
    if (causal) {
        ss << "    if (kb >= (kb_lim - (MFA_BQ + MFA_BK - 1) / MFA_BK)) {\n";
        ss << "      STEEL_PRAGMA_UNROLL\n";
        ss << "      for (short i = 0; i < MFA_TQ; i++) {\n";
        ss << "        const int row = qL_off + local_tile_id * MFA_BQ + tm + sm + i * 8;\n";
        ss << "        STEEL_PRAGMA_UNROLL\n";
        ss << "        for (short j = 0; j < MFA_TK; j++) {\n";
        ss << "          const int col = kb * MFA_BK + sn + j * 8;\n";
        ss << "          STEEL_PRAGMA_UNROLL\n";
        ss << "          for (short jj = 0; jj < 2; jj++) {\n";
        ss << "            if (row < (col + jj))\n";
        ss << "              Stile.frag_at(i,j)[jj] = -INFINITY;\n";
        ss << "          }\n";
        ss << "        }\n";
        ss << "      }\n";
        ss << "    }\n";
        ss << "\n";
    }

    // ── Barrier before V load ─────────────────────────────────────────────────
    // K GEMM is done; KV_smem can now be overwritten with V data.
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "\n";

    // ── Paged V gather: row-major layout V_smem[t * LDV + d] ─────────────────
    // Same cooperative gather as K, but decomposed as t = slot / BD, d = slot % BD
    // to produce row-major storage (each row = one KV token, columns = head dim).
    ss << "    {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short ei = 0; ei < " << elems_per_thread << "; ei++) {\n";
    ss << "        const int slot = thread_idx + (int)ei * MFA_TGP_SIZE;\n";
    ss << "        if (slot < MFA_BK * MFA_BD) {\n";
    ss << "          const int t = slot / MFA_BD;\n";
    ss << "          const int d = slot % MFA_BD;\n";
    ss << "          const int global_tok = kb * MFA_BK + t;\n";
    ss << "          T val = T(0);\n";
    ss << "          if (global_tok < kL_local) {\n";
    ss << "            const int blk_idx    = global_tok / p->block_size;\n";
    ss << "            const int tok_in_blk = global_tok % p->block_size;\n";
    ss << "            // OOB guards (CC-02): blk_idx within block_table, phys within pool.\n";
    ss << "            if (blk_idx < p->max_blocks) {\n";
    ss << "              const int phys = block_table[seq_id * p->max_blocks + blk_idx];\n";
    ss << "              if (phys >= 0 && phys < p->num_blocks) {\n";
    ss << "                val = v_pool[(long)phys * p->pool_block_stride\n";
    ss << "                          + tok_in_blk * p->pool_tok_stride\n";
    ss << "                          + kv_head * p->D + d];\n";
    ss << "              }\n";
    ss << "            }\n";
    ss << "          }\n";
    ss << "          Vs[t * LDV + d] = val;\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";

    // ── Online softmax (NaN-safe version from paged kernel) ───────────────────
    // Two-path: if new_max > old_max, rescale; otherwise factor=1 and guard isinf.
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

    // ── P @ V → Otile ────────────────────────────────────────────────────────
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
    ss << "\n";

    ss << "  } // end kb loop\n";
    ss << "\n";

    // ── Normalize O: divide each row by sum_score ─────────────────────────────
    ss << "  Otile.template row_bin_op<MFADivOp>(sum_score);\n";
    ss << "  threadgroup_barrier(mem_flags::mem_none);\n";
    ss << "\n";

    // ── Write O ───────────────────────────────────────────────────────────────
    // O is already offset to the first token of this tile's row in Q/O.
    // Advance by (tm + sm) rows and sn columns within the tile.
    ss << "  device T* O_write = O + (long)(tm + sm) * MFA_BD + sn;\n";
    ss << "  if (local_tile_id == NQ_aligned && qL_rem_local > 0) {\n";
    ss << "    auto dims = short2((short)(MFA_BD - sn),\n";
    ss << "                       (short)(qL_rem_local - (tm + sm)));\n";
    ss << "    if (dims.x > 0 && dims.y > 0)\n";
    ss << "      Otile.template store_safe<T, 1, 1>(O_write, MFA_BD, dims);\n";
    ss << "  } else {\n";
    ss << "    Otile.template store<T, 1, 1>(O_write, MFA_BD);\n";
    ss << "  }\n";
    ss << "\n";

    // ── Write L (log2-domain logsumexp) ───────────────────────────────────────
    // L is [H, total_q]; for head tid.y, the base is L + tid.y * total_q.
    // We already applied L += (long)tid.y * p->total_q + q_start at pointer setup,
    // so here we just index relative to q_start.
    ss << "  if (sn == 0) {\n";
    ss << "    const long q_base = (long)(local_tile_id * MFA_BQ + tm + sm);\n";
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
    ss << "      const long q_idx = q_base + i * 8;\n";
    ss << "      if ((q_start + q_idx) < q_end) {\n";
    ss << "        L[q_idx] = max_score[i] + metal::log2(sum_score[i]);\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "  }\n";
    ss << "}\n";

    return ss.str();
}

}  // namespace mlx_mfa
