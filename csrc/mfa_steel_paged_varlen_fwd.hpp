#pragma once
#include "shader_cache.hpp"
#include <cstdint>
#include <string>

namespace mlx_mfa {

/// Parameters passed from C++ to the Metal paged-varlen kernel.
/// Layout MUST exactly match MFAPagedVarlenParams in the Metal source string.
///
/// Packed Q/O layout: [1, H, total_q, D]
/// Pool layout: k_pool/v_pool each [num_blocks, block_size, H_kv, D]
///   token pos → phys = block_table[seq_id * max_blocks + pos/block_size]
struct MFAPagedVarlenParams {
    int H, D;
    int gqa_factor;       // H / H_kv
    int num_seqs;         // number of independent sequences
    int total_q;          // sum of all q lengths
    int total_q_tiles;    // sum of Q-tiles = tile_offsets[num_seqs]
    float scale;          // attention scale (1/sqrt(D))
    float softcap;        // 0.0 = disabled (reserved for future use)
    int64_t Q_head_stride; // = total_q * D
    int block_size;        // tokens per pool block (page size)
    int max_blocks;        // columns in block_table (per-sequence max pages)
    int pool_block_stride; // = block_size * H_kv * D
    int pool_tok_stride;   // = H_kv * D
    int H_kv;              // kv head count = H / gqa_factor
    int window_left;       // -1 = disabled; >=0 = sliding window left radius
    int window_right;      // -1 = disabled; >=0 = sliding window right radius
};

/// Generate the Metal shader source for the fused paged-varlen forward kernel.
/// Combines varlen grid scheduling (one threadgroup per Q-tile per sequence)
/// with paged KV gather (block_table lookup per token).
/// Kernel function name: "mlx_mfa_paged_varlen_forward".
std::string generate_paged_varlen_forward_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
