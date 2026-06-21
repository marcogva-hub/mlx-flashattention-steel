#pragma once
#include "shader_cache.hpp"
#include <cstdint>
#include <string>

namespace mlx_mfa {

/// Parameters for the TurboQuant fused paged-varlen kernel.
/// Layout MUST exactly match MFAPagedVarlenTQParams in the Metal source string.
///
/// K is stored as packed uint8 indices (2 per byte) with per-vector scales.
/// V can be fp16 (tq_v_enabled=0) or packed TQ (tq_v_enabled=1).
/// Centroids are passed as small fp16 lookup tables.
struct MFAPagedVarlenTQParams {
    int H, D;
    int gqa_factor;        // H / H_kv
    int num_seqs;
    int total_q;
    int total_q_tiles;
    float scale;           // attention scale (1/sqrt(D))
    float softcap;         // 0.0 = disabled
    int64_t Q_head_stride; // = total_q * D
    int block_size;        // tokens per pool block
    int max_blocks;        // columns in block_table
    int pool_block_stride_v; // V pool fp16: block_size * H_kv * D
    int pool_tok_stride_v;   // V pool fp16: H_kv * D
    int pool_block_stride_k; // K TQ pool: block_size * H_kv * packed_D
    int pool_tok_stride_k;   // K TQ pool: H_kv * packed_D
    int H_kv;
    int packed_D;          // D / 2  (2 indices per byte)
    int tq_bits;           // 2, 3, or 4
    int n_centroids;       // 2^tq_bits
    int window_left;       // -1 = disabled
    int window_right;      // -1 = disabled
    // V-TQ fields (Phase 3A)
    int tq_v_enabled;      // 0 = V is fp16, 1 = V is packed TQ
    int tq_v_pool_block_stride; // V TQ pool: block_size * H_kv * packed_D
    int tq_v_pool_tok_stride;   // V TQ pool: H_kv * packed_D
    // WHT fusion (Phase 4)
    int tq_wht_enabled;    // 0 = Q pre-rotated in Python, 1 = WHT applied in kernel
    int num_blocks;        // pool.shape(0) — upper bound for phys (OOB guard, CC-02)
};

/// Generate the Metal shader source for the TurboQuant paged-varlen kernel.
/// K gather dequantifies packed uint8 indices inline via centroid lookup.
/// Kernel function name: "mlx_mfa_paged_varlen_tq_forward".
std::string generate_paged_varlen_tq_forward_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
