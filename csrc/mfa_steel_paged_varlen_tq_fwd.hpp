#pragma once
#include "shader_cache.hpp"
#include <cstdint>
#include <string>

namespace mlx_mfa {

/// Parameters for the TurboQuant fused paged-varlen kernel.
/// Layout MUST exactly match MFAPagedVarlenTQParams in the Metal source string.
///
/// K is stored as packed uint8 indices (2 per byte) with per-vector scales.
/// V remains fp16. Centroids are passed as a small fp16 lookup table.
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
    int pool_block_stride_v; // V pool: block_size * H_kv * D
    int pool_tok_stride_v;   // V pool: H_kv * D
    int pool_block_stride_k; // K TQ pool: block_size * H_kv * packed_D
    int pool_tok_stride_k;   // K TQ pool: H_kv * packed_D
    int H_kv;
    int packed_D;          // D / 2  (2 indices per byte)
    int tq_bits;           // 2, 3, or 4
    int n_centroids;       // 2^tq_bits
    int window_left;       // -1 = disabled
    int window_right;      // -1 = disabled
};

/// Generate the Metal shader source for the TurboQuant paged-varlen kernel.
/// K gather dequantifies packed uint8 indices inline via centroid lookup.
/// Kernel function name: "mlx_mfa_paged_varlen_tq_forward".
std::string generate_paged_varlen_tq_forward_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
