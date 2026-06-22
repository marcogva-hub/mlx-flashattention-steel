// mfa_steel_paged_varlen_tq_fwd.cpp
// ===========================================================================
// JIT Metal shader generator for the TurboQuant fused PagedVarlenForward kernel.
//
// Identical to mfa_steel_paged_varlen_fwd.cpp except:
//   • K is stored as packed uint8 (2 quantization indices per byte)
//   • K gather: unpack indices → centroid lookup → multiply by per-vector scale
//   • V remains fp16 (unchanged)
//   • Additional buffers: centroids (buffer 10), k_scales (buffer 11)
//
// Buffer layout:
//   buffer(0):  Q             packed [total_q, H, D]
//   buffer(1):  k_pool_tq     [num_blocks, block_size, H_kv, packed_D] uint8
//   buffer(2):  v_pool        [num_blocks, block_size, H_kv, D] fp16
//   buffer(3):  O             packed [total_q, H, D]
//   buffer(4):  L             [H, total_q] float32 logsumexp
//   buffer(5):  params        MFAPagedVarlenTQParams (constant)
//   buffer(6):  cu_seqlens_q  [num_seqs+1] int32
//   buffer(7):  tile_offsets  [num_seqs+1] int32
//   buffer(8):  block_table   [num_seqs, max_blocks] int32
//   buffer(9):  seq_lens_kv   [num_seqs] int32
//   buffer(10): centroids     [n_centroids] fp16 (e.g. 8 for 3-bit)
//   buffer(11): k_scales      [num_blocks, block_size, H_kv] float32
// ===========================================================================
//
// ── Dequant-in-GEMM analysis (v2.24.1) ──────────────────────────────────
//
// Considered restructuring K gather to dequantify DURING the GEMM (write
// packed uint8 indices to K_smem, centroid lookup inline in GEMM loop)
// instead of BEFORE it (current: unpack → centroid → fp16 → K_smem → GEMM).
//
// This would halve the K phase of KV_smem from fp16 to uint8 — saving
// 512 B (D=64 BK=32) to 1792 B (D=128 BK=16).
//
// TGP budget analysis (select_steel_block_config values):
//
//   Config      | Q_smem  | KV_smem | Centroids | Total   | % 32KB | Headroom
//   ------------|---------|---------|-----------|---------|--------|----------
//   M1  D=64    | 4,608 B | 5,120 B |     64 B  | 9,792 B |  30%   | 22,976 B
//   M1  D=128   | 8,704 B | 6,144 B |     64 B  | 14,912B |  46%   | 17,856 B
//   M3+ D=128   | 8,704 B | 10,240B |     64 B  | 19,008B |  58%   | 13,760 B
//
// Decision: SKIP.  All configs sit well below the 32 KB hardware limit.
// Max usage is 19 KB (M3+ D=128) with 13.7 KB headroom.  The uint8 savings
// (0.5–1.8 KB) cannot change occupancy tiers (next threshold at 32 KB).
// Additional complexity (MFAMMAFrag inline centroid lookup, uint8 smem
// reinterpretation) is not justified for zero occupancy benefit.
//
// Revisit if: new block configs push TGP past 28 KB, or D=256 TQ support
// is added (larger BQ/BD tiles).
// ─────────────────────────────────────────────────────────────────────────

#include "mfa_steel_paged_varlen_tq_fwd.hpp"
#include "mfa_steel_fwd.hpp"
#include <sstream>

namespace mlx_mfa {

std::string generate_paged_varlen_tq_forward_source(const ShaderCache::KernelKey& key) {
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

    const int TD      = BD / 8;
    const int TK      = BK / 8;
    const int TQ      = BQ / (WM * 8);
    const int ROWS_PT = TQ;

    // K gather: each element is a packed byte → produces 2 fp16 values.
    // We still gather BK * BD fp16 values into K_smem, but read from packed source.
    const int kv_tile_elems    = BK * BD;
    const int elems_per_thread = (kv_tile_elems + TGP_SIZE - 1) / TGP_SIZE;

    std::ostringstream ss;

    // ── Metal preamble ───────────────────────────────────────────────────────
    append_metal_headers_and_defines(ss, enable_unroll, arch_gen, dtype_str);

    // ── MFAPagedVarlenTQParams struct ────────────────────────────────────────
    ss << R"MFA(
struct MFAPagedVarlenTQParams {
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
  int pool_block_stride_v;
  int pool_tok_stride_v;
  int pool_block_stride_k;
  int pool_tok_stride_k;
  int H_kv;
  int packed_D;
  int tq_bits;
  int n_centroids;
  int window_left;
  int window_right;
  int tq_v_enabled;
  int tq_v_pool_block_stride;
  int tq_v_pool_tok_stride;
  int tq_wht_enabled;
  int num_blocks;
};

)MFA";

    // ── Shared templates ─────────────────────────────────────────────────────
    append_steel_shared_templates(ss);

    // ── Compile-time constants ────────────────────────────────────────────────
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

    // ── Kernel signature ─────────────────────────────────────────────────────
    ss << "[[kernel, max_total_threads_per_threadgroup(MFA_TGP_SIZE)]]\n";
    ss << "void mlx_mfa_paged_varlen_tq_forward(\n";
    ss << "    const device MFA_DTYPE* Q                        [[buffer(0)]],\n";
    ss << "    const device uchar*     k_pool_tq                [[buffer(1)]],\n";
    ss << "    const device MFA_DTYPE* v_pool                   [[buffer(2)]],\n";
    ss << "    device MFA_DTYPE*       O                        [[buffer(3)]],\n";
    ss << "    device float*           L                        [[buffer(4)]],\n";
    ss << "    const constant MFAPagedVarlenTQParams* p         [[buffer(5)]],\n";
    ss << "    const device int*       cu_seqlens_q             [[buffer(6)]],\n";
    ss << "    const device int*       tile_offsets             [[buffer(7)]],\n";
    ss << "    const device int*       block_table              [[buffer(8)]],\n";
    ss << "    const device int*       seq_lens_kv              [[buffer(9)]],\n";
    ss << "    const device MFA_DTYPE* centroids                [[buffer(10)]],\n";
    ss << "    const device float*     k_scales                 [[buffer(11)]],\n";
    ss << "    const device uchar*     v_pool_tq                [[buffer(12)]],\n";
    ss << "    const device MFA_DTYPE* v_centroids              [[buffer(13)]],\n";
    ss << "    const device float*     v_scales                 [[buffer(14)]],\n";
    ss << "    uint simd_lane_id  [[thread_index_in_simdgroup]],\n";
    ss << "    uint simd_group_id [[simdgroup_index_in_threadgroup]],\n";
    ss << "    uint3 tid          [[threadgroup_position_in_grid]])\n";
    ss << "{\n";
    ss << "  typedef MFA_DTYPE T;\n";
    ss << "  typedef float     AccT;\n";
    ss << "\n";

    // ── Seq-ID resolution (identical to non-TQ) ──────────────────────────────
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

    // ── Pointer setup ────────────────────────────────────────────────────────
    ss << "  const int kv_head = (int)tid.y / p->gqa_factor;\n";
    ss << "  Q += (long)tid.y * p->Q_head_stride\n";
    ss << "     + (long)(q_start + local_tile_id * MFA_BQ) * MFA_BD;\n";
    ss << "  O += (long)tid.y * p->Q_head_stride\n";
    ss << "     + (long)(q_start + local_tile_id * MFA_BQ) * MFA_BD;\n";
    ss << "  L += (long)tid.y * p->total_q + q_start;\n";
    ss << "\n";

    // ── TQ bit extraction mask ───────────────────────────────────────────────
    ss << "  const int tq_bits = p->tq_bits;\n";
    ss << "  const uchar tq_mask = (uchar)((1 << tq_bits) - 1);\n";
    ss << "\n";

    // ── Scale (log2 domain) ──────────────────────────────────────────────────
    // When WHT is fused in kernel, fold 1/sqrt(D) normalization into scale.
    ss << "  const AccT base_scale = p->tq_wht_enabled\n";
    ss << "      ? (p->scale * rsqrt((AccT)p->D))\n";
    ss << "      : p->scale;\n";
    ss << "  const AccT scale_v = base_scale * M_LOG2E_F;\n";
    ss << "\n";

    // ── MMA thread coordinates ───────────────────────────────────────────────
    ss << "  const short2 simd_coord = MFAMMAFrag<AccT>::get_coord((ushort)simd_lane_id);\n";
    ss << "  const short sm = simd_coord.y;\n";
    ss << "  const short sn = simd_coord.x;\n";
    ss << "  const short tm = 8 * MFA_TQ * (short)simd_group_id;\n";
    ss << "  const int thread_idx = (int)simd_group_id * 32 + (int)simd_lane_id;\n";
    ss << "\n";

    // ── SMEM layout (same as non-TQ — K is dequantified into fp16 in SMEM) ──
    ss << "  constexpr short padQ  = MFA_PAD;\n";
    ss << "  constexpr short LDQ   = MFA_BD + padQ;\n";
    ss << "  constexpr short LDK   = MFA_BK + (short)(16/sizeof(T));\n";
    ss << "  constexpr short LDV   = MFA_BD + (short)(16/sizeof(T));\n";
    ss << "  constexpr short kv_s0 = LDK * MFA_BD;\n";
    ss << "  constexpr short kv_s1 = MFA_BK * LDV;\n";
    ss << "  constexpr short kv_s  = kv_s0 > kv_s1 ? kv_s0 : kv_s1;\n";
    ss << "\n";
    ss << "  threadgroup T Q_smem[MFA_BQ * (MFA_BD + MFA_PAD)];\n";
    ss << "  threadgroup T KV_smem[kv_s];\n";
    ss << "  threadgroup T* Qs = Q_smem;\n";
    ss << "  threadgroup T* Ks = KV_smem;\n";
    ss << "  threadgroup T* Vs = KV_smem;\n";
    ss << "\n";

    // Phase 3C: Centroid cache in threadgroup memory.
    // Both K and V centroids are loaded once and read from smem in every gather.
    // Max 16 centroids (4-bit); actual count = p->n_centroids at runtime.
    ss << "  // Centroid TGP cache (Phase 3C): load once, read many\n";
    ss << "  constexpr short MAX_CENTROIDS = 16;\n";
    ss << "  threadgroup T k_centroids_smem[MAX_CENTROIDS];\n";
    ss << "  threadgroup T v_centroids_smem[MAX_CENTROIDS];\n";
    ss << "  if (thread_idx < p->n_centroids) {\n";
    ss << "    k_centroids_smem[thread_idx] = centroids[thread_idx];\n";
    ss << "    if (p->tq_v_enabled) {\n";
    ss << "      v_centroids_smem[thread_idx] = v_centroids[thread_idx];\n";
    ss << "    }\n";
    ss << "  }\n";
    ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "\n";

    // ── Q loader ─────────────────────────────────────────────────────────────
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

    // ── WHT butterfly on Q_smem (when tq_wht_enabled) ────────────────────────
    // Walsh-Hadamard transform: log2(D) passes of butterfly add/subtract.
    // Each pass at stride h: for pairs (i, i+h), replace with (a+b, a-b).
    // Applied in-place on Q_smem rows. Normalization 1/sqrt(D) folded into scale.
    ss << "  if (p->tq_wht_enabled) {\n";
    {
        // Total elements in Q_smem = BQ * D; distribute across TGP threads.
        // Each butterfly pass processes BQ * D/2 pairs.
        const int log2_D = (key.head_dim == 64) ? 6 : (key.head_dim == 128) ? 7 : 8;
        ss << "    const int total_q_elems = MFA_BQ * MFA_BD;\n";
        ss << "    const int n_pairs = total_q_elems / 2;\n";
        ss << "    const int pairs_per_thread = (n_pairs + MFA_TGP_SIZE - 1) / MFA_TGP_SIZE;\n";
        for (int pass = 0; pass < log2_D; pass++) {
            int h = 1 << pass;
            ss << "    {\n";
            ss << "      const int h = " << h << ";\n";
            ss << "      STEEL_PRAGMA_UNROLL\n";
            ss << "      for (int pi = 0; pi < pairs_per_thread; pi++) {\n";
            ss << "        const int pair_id = thread_idx + pi * MFA_TGP_SIZE;\n";
            ss << "        if (pair_id < n_pairs) {\n";
            // pair_id addresses the pair within the flattened BQ*D space.
            // Within each row of D elements, pairs at stride h:
            // element indices: group = pair_id_in_row / h, lo = group*2h + (pair_id_in_row % h)
            ss << "          const int row = pair_id / (MFA_BD / 2);\n";
            ss << "          const int pair_in_row = pair_id % (MFA_BD / 2);\n";
            ss << "          const int group = pair_in_row / h;\n";
            ss << "          const int lo_d = group * (2 * h) + (pair_in_row % h);\n";
            ss << "          const int hi_d = lo_d + h;\n";
            ss << "          const int lo_idx = row * LDQ + lo_d;\n";
            ss << "          const int hi_idx = row * LDQ + hi_d;\n";
            ss << "          T a = Qs[lo_idx];\n";
            ss << "          T b = Qs[hi_idx];\n";
            ss << "          Qs[lo_idx] = a + b;\n";
            ss << "          Qs[hi_idx] = a - b;\n";
            ss << "        }\n";
            ss << "      }\n";
            ss << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
            ss << "    }\n";
        }
    }
    ss << "  }\n";
    ss << "\n";

    // SMEM-to-register offsets
    ss << "  const short Qs_off = (tm + sm) * LDQ + sn;\n";
    ss << "  const short Ks_off = sm * LDK + sn;\n";
    ss << "  const short Vs_off = sm * LDV + sn;\n";
    ss << "\n";

    // ── Register tile declarations ───────────────────────────────────────────
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Qtile;\n";
    ss << "  MFAMMATile<AccT, 1,      MFA_TK>  Ktile;\n";
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TK> Stile;\n";
    ss << "  MFAMMATile<AccT, 1,      1>       Vtile;\n";
    ss << "  MFAMMATile<AccT, MFA_TQ, MFA_TD> Otile;\n";
    ss << "\n";

    ss << "  Otile.clear();\n";
    ss << "  Qtile.template load<T, 1, 1>(&Qs[Qs_off], LDQ, 1);\n";
    ss << "\n";

    // ── Online softmax state ─────────────────────────────────────────────────
    ss << "  AccT max_score[MFA_ROWS_PT];\n";
    ss << "  AccT sum_score[MFA_ROWS_PT];\n";
    ss << "  STEEL_PRAGMA_UNROLL\n";
    ss << "  for (short i = 0; i < MFA_ROWS_PT; i++) {\n";
    ss << "    max_score[i] = -INFINITY;\n";
    ss << "    sum_score[i] = 0.0f;\n";
    ss << "  }\n";
    ss << "\n";

    // ── Causal K-loop limit ──────────────────────────────────────────────────
    if (causal) {
        ss << "  const int qL_off = (qL_local < kL_local) ? (kL_local - qL_local) : 0;\n";
        ss << "  const int q_max_pos = qL_off + (local_tile_id + 1) * MFA_BQ;\n";
        ss << "  const int kb_lim = min(NK_local, (q_max_pos + MFA_BK - 1) / MFA_BK);\n";
    } else {
        ss << "  const int qL_off = 0;\n";
        ss << "  const int kb_lim = NK_local;\n";
    }
    ss << "\n";

    // ── Main K-tile loop ─────────────────────────────────────────────────────
    ss << "  for (int kb = 0; kb < kb_lim; kb++) {\n";
    ss << "\n";

    // Barrier before K write
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "\n";

    // ── TQ K gather: read packed uint8, dequantify to fp16 ───────────────────
    // Each thread handles elems_per_thread (d,t) pairs.
    // For each pair: read packed byte from k_pool_tq, extract index for this d,
    // look up centroid, multiply by k_scales[phys_tok], write to K_smem[d*LDK+t].
    ss << "    {\n";
    ss << "      STEEL_PRAGMA_UNROLL\n";
    ss << "      for (short ei = 0; ei < " << elems_per_thread << "; ei++) {\n";
    ss << "        const int slot = thread_idx + (int)ei * MFA_TGP_SIZE;\n";
    ss << "        if (slot < MFA_BK * MFA_BD) {\n";
    ss << "          const int t = slot % MFA_BK;\n";    // seq position within tile
    ss << "          const int d = slot / MFA_BK;\n";    // head-dim index
    ss << "          const int global_tok = kb * MFA_BK + t;\n";
    ss << "          T val = T(0);\n";
    ss << "          if (global_tok < kL_local) {\n";
    ss << "            const int blk_idx    = global_tok / p->block_size;\n";
    ss << "            const int tok_in_blk = global_tok % p->block_size;\n";
    ss << "            // OOB guards (CC-02): blk_idx is checked BEFORE the block_table\n";
    ss << "            // read, then phys BEFORE any pool read — matches the non-TQ sibling\n";
    ss << "            // mfa_steel_paged_varlen_fwd.cpp (no device read precedes its guard).\n";
    ss << "            if (blk_idx < p->max_blocks) {\n";
    ss << "            const int phys = block_table[seq_id * p->max_blocks + blk_idx];\n";
    ss << "            if (phys >= 0 && phys < p->num_blocks) {\n";
    ss << "\n";
    // Index extraction per bit-width.  Sprint III-2 FIX: this block was
    // the 3-bit bit-planar form UNCONDITIONALLY — tq_bits=2/4 read the
    // pool with the wrong layout and produced silently-wrong K since the
    // kernel landed (arbitrated vs Python ground-truth dequant: fused
    // 0.147-0.150 max-abs wrong at unit scale, growing with magnitude).
    //   3-bit: bit-planar, 32 indices -> 3 planes x 4 B = 12 B/group
    //   2-bit: 4 indices/byte;  4-bit: 2 indices/byte (pack_k_for_metal)
    ss << "            const long row_off = (long)phys * p->pool_block_stride_k\n";
    ss << "                              + tok_in_blk * p->pool_tok_stride_k\n";
    ss << "                              + kv_head * p->packed_D;\n";
    ss << "            uchar idx;\n";
    ss << "            if (tq_bits == 3) {\n";
    ss << "              const int group = d / 32;\n";
    ss << "              const int lane  = d % 32;\n";
    ss << "              const int byte_in_lane = lane / 8;\n";
    ss << "              const int bit_in_byte  = lane % 8;\n";
    ss << "              const long base_off = row_off + group * 12;\n";
    ss << "              const uchar b0 = k_pool_tq[base_off + 0 * 4 + byte_in_lane];\n";
    ss << "              const uchar b1 = k_pool_tq[base_off + 1 * 4 + byte_in_lane];\n";
    ss << "              const uchar b2 = k_pool_tq[base_off + 2 * 4 + byte_in_lane];\n";
    ss << "              idx = ((b0 >> bit_in_byte) & 1)\n";
    ss << "                  | (((b1 >> bit_in_byte) & 1) << 1)\n";
    ss << "                  | (((b2 >> bit_in_byte) & 1) << 2);\n";
    ss << "            } else if (tq_bits == 2) {\n";
    ss << "              const uchar pbyte = k_pool_tq[row_off + d / 4];\n";
    ss << "              idx = (pbyte >> ((d % 4) * 2)) & 3;\n";
    ss << "            } else {\n";
    ss << "              const uchar pbyte = k_pool_tq[row_off + d / 2];\n";
    ss << "              idx = (pbyte >> ((d % 2) * 4)) & 15;\n";
    ss << "            }\n";
    // Centroid lookup → fp16 value
    ss << "            const T centroid_val = k_centroids_smem[idx];\n";
    // Per-vector scale: k_scales[phys * block_size * H_kv + tok_in_blk * H_kv + kv_head]
    ss << "            const float kscale = k_scales[\n";
    ss << "                (long)phys * p->block_size * p->H_kv\n";
    ss << "                + tok_in_blk * p->H_kv + kv_head];\n";
    ss << "            val = T((float)centroid_val * kscale);\n";
    ss << "            }\n";  // close phys-in-pool guard (CC-02)
    ss << "            }\n";  // close blk_idx-in-table guard (CC-02)
    ss << "          }\n";
    ss << "          Ks[d * LDK + t] = val;\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "\n";

    // ── Q @ K^T → Stile (identical to non-TQ) ───────────────────────────────
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

    // ── Scale ────────────────────────────────────────────────────────────────
    ss << "    STEEL_PRAGMA_UNROLL\n";
    ss << "    for (short ii = 0; ii < MFA_TQ * MFA_TK * 2; ii++) {\n";
    ss << "      Stile.elems()[ii] *= scale_v;\n";
    ss << "    }\n";
    ss << "\n";

    // ── K-boundary mask ──────────────────────────────────────────────────────
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

    // ── Causal mask ──────────────────────────────────────────────────────────
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

    // ── Barrier before V load ────────────────────────────────────────────────
    ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    ss << "\n";

    // ── Paged V gather (fp16 or TQ depending on tq_v_enabled) ──────────────
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
    ss << "            // OOB guards (CC-02): blk_idx is checked BEFORE the block_table\n";
    ss << "            // read, then phys BEFORE any pool read — matches the non-TQ sibling\n";
    ss << "            // mfa_steel_paged_varlen_fwd.cpp (no device read precedes its guard).\n";
    ss << "            if (blk_idx < p->max_blocks) {\n";
    ss << "            const int phys = block_table[seq_id * p->max_blocks + blk_idx];\n";
    ss << "            if (phys >= 0 && phys < p->num_blocks) {\n";
    // V-TQ branch: uniform branch (all threads take same path), zero divergence cost
    ss << "            if (p->tq_v_enabled) {\n";
    // Sprint III-2 FIX: same 3-bit-only-layout bug as the K path —
    // branch on tq_bits (V packing mirrors pack_k_for_metal layouts).
    ss << "              const long vrow_off = (long)phys * p->tq_v_pool_block_stride\n";
    ss << "                                   + tok_in_blk * p->tq_v_pool_tok_stride\n";
    ss << "                                   + kv_head * p->packed_D;\n";
    ss << "              uchar v_idx;\n";
    ss << "              if (tq_bits == 3) {\n";
    ss << "                const int vgroup = d / 32;\n";
    ss << "                const int vlane  = d % 32;\n";
    ss << "                const int vbyte_in_lane = vlane / 8;\n";
    ss << "                const int vbit_in_byte  = vlane % 8;\n";
    ss << "                const long vbase_off = vrow_off + vgroup * 12;\n";
    ss << "                const uchar vb0 = v_pool_tq[vbase_off + 0 * 4 + vbyte_in_lane];\n";
    ss << "                const uchar vb1 = v_pool_tq[vbase_off + 1 * 4 + vbyte_in_lane];\n";
    ss << "                const uchar vb2 = v_pool_tq[vbase_off + 2 * 4 + vbyte_in_lane];\n";
    ss << "                v_idx = ((vb0 >> vbit_in_byte) & 1)\n";
    ss << "                      | (((vb1 >> vbit_in_byte) & 1) << 1)\n";
    ss << "                      | (((vb2 >> vbit_in_byte) & 1) << 2);\n";
    ss << "              } else if (tq_bits == 2) {\n";
    ss << "                const uchar vpbyte = v_pool_tq[vrow_off + d / 4];\n";
    ss << "                v_idx = (vpbyte >> ((d % 4) * 2)) & 3;\n";
    ss << "              } else {\n";
    ss << "                const uchar vpbyte = v_pool_tq[vrow_off + d / 2];\n";
    ss << "                v_idx = (vpbyte >> ((d % 2) * 4)) & 15;\n";
    ss << "              }\n";
    ss << "              const T v_centroid_val = v_centroids_smem[v_idx];\n";
    ss << "              const float vscale = v_scales[\n";
    ss << "                  (long)phys * p->block_size * p->H_kv\n";
    ss << "                  + tok_in_blk * p->H_kv + kv_head];\n";
    ss << "              val = T((float)v_centroid_val * vscale);\n";
    ss << "            } else {\n";
    ss << "              val = v_pool[(long)phys * p->pool_block_stride_v\n";
    ss << "                          + tok_in_blk * p->pool_tok_stride_v\n";
    ss << "                          + kv_head * p->D + d];\n";
    ss << "            }\n";
    ss << "            }\n";  // close phys-in-pool guard (CC-02)
    ss << "            }\n";  // close blk_idx-in-table guard (CC-02)
    ss << "          }\n";
    ss << "          Vs[t * LDV + d] = val;\n";
    ss << "        }\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "\n";

    // ── Online softmax ───────────────────────────────────────────────────────
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

    // ── Normalize O ──────────────────────────────────────────────────────────
    ss << "  Otile.template row_bin_op<MFADivOp>(sum_score);\n";
    ss << "  threadgroup_barrier(mem_flags::mem_none);\n";
    ss << "\n";

    // ── Write O ──────────────────────────────────────────────────────────────
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

    // ── Write L ──────────────────────────────────────────────────────────────
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
