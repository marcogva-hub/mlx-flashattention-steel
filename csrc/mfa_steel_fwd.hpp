/// mfa_steel_fwd.hpp  –  STEEL-style cooperative forward attention kernel.
///
/// Replaces the ccv-derived forward kernel for better performance:
///  • Cooperative K/V tile loading into threadgroup SRAM (like MLX SDPA)
///  • All threads collaboratively load each K/V tile (no per-warp repetition)
///  • 128-bit coalesced vectorized reads vs. per-lane simdgroup_matrix::load
///
/// The kernel function name in the generated Metal source is "mlx_mfa_attention".

#pragma once

#include "shader_cache.hpp"
#include <cstdint>
#include <string>

namespace mlx_mfa {

/// Parameters passed from C++ to the Metal kernel.
/// Layout MUST exactly match MFASteelParams in the Metal source string.
///
/// C++ type → Metal type mapping:
///   int      → int   (4 bytes)
///   float    → float (4 bytes)
///   int64_t  → long  (8 bytes on both sides)
struct MFASteelParams {
    // Tensor dimensions
    int B, H, D;
    int qL, kL;
    int gqa_factor;    // kv_heads: q_heads / kv_heads (1 for standard MHA)
    float scale;       // attention scale factor
    // Block counts
    int NQ, NK;        // total threadgroup blocks along Q / K
    int NQ_aligned;    // last aligned (full) Q block index = NQ - (qL%BQ ? 1 : 0)
    int NK_aligned;    // last aligned (full) K block index = NK - (kL%BK ? 1 : 0)
    // Boundary remainders
    int qL_rem;        // elements in last Q block = qL % BQ  (0 if aligned)
    int kL_rem;        // elements in last K block = kL % BK
    int qL_off;        // query sequence offset for cross-attention (0 for self-attn)
    // RoPE fusion params (only used when has_rope kernel variant is compiled)
    int rope_q_base;   // absolute position of Q token 0 = cache_seqlens
    int rope_cos_stride; // stride of rotary_cos along seq dim = D/2
    // Strides: [B, H, S] for Q/K/V/O  (D=1 stride is implicit)
    int64_t Q_strides[3];
    int64_t K_strides[3];
    int64_t V_strides[3];
    int64_t O_strides[3];
    // L (logsumexp) strides: [B, H]  (qL=1 stride is implicit)
    int64_t L_strides[2];
    // Optional features — appended at end for backward compatibility.
    // Defaults: softcap=0.0f (disabled), has_alibi=0 (disabled).
    float softcap;     // 0.0 = disabled; >0 → tanh(S/cap)*cap before softmax
    int   has_alibi;   // 0 = disabled; 1 = ALiBi per-head bias (buffer(9))
};

/// Select BQ/BK/BD/WM/WN based on head_dim and precision.
/// Returns a struct used both for dispatch sizing and kernel source generation.
struct SteelBlockConfig {
    int BQ;     // query tile rows
    int BK;     // key/value tile rows
    int BD;     // head dimension (== full head_dim, not a sub-tile)
    int WM;     // SIMD groups along query (= total warps since WN=1)
    int WN;     // SIMD groups along key (always 1 for attention)
    int PAD;    // threadgroup bank-conflict padding per row
};

SteelBlockConfig select_steel_block_config(int head_dim, bool is_low_prec,
                                           bool is_m3_plus = false);

/// Generate the complete Metal shader source for the STEEL-style forward kernel.
/// The source defines the kernel function "mlx_mfa_attention".
std::string generate_steel_forward_source(const ShaderCache::KernelKey& key);

// ── Flash Decoding (Split-KV) ─────────────────────────────────────────────
//
// Flash Decoding improves GPU utilization during autoregressive decode (N_q<=4)
// by splitting the KV sequence into num_splits chunks dispatched in parallel.
//
// Phase 1 kernel ("mlx_mfa_flash_decode_partial"):
//   Grid (NQ * num_splits, H, B) — each threadgroup covers one Q-tile of one KV split.
//   Outputs: pO[num_splits, B, H, qL, D]  (normalized partial output)
//            pL[num_splits, B, H, qL]     (log2-domain logsumexp per split)
//
// Phase 2 kernel ("mlx_mfa_flash_decode_reduce"):
//   Grid (N, H, B) — one thread-column per query position, head, batch.
//   Reads pO/pL and writes final O and L via log-sum-exp combining.

/// Parameters for the Flash Decode Phase 1 (partial attention) kernel.
/// Must exactly match the MFAFlashDecodePartialParams struct in Metal source.
struct FlashDecodePartialParams {
    // Tensor dimensions
    int B, H, D;
    int qL, kL;
    int gqa_factor;
    float scale;
    // Q tile counts
    int NQ, NQ_aligned;
    int qL_rem;
    int qL_off;          // query position offset for causal (S - N for decode)
    // K (full sequence)
    int NK_total;        // = ceil(kL / BK)
    int NK_aligned;      // last fully-filled K-tile index
    int kL_rem;          // kL % BK (0 if aligned)
    // Split config
    int num_splits;
    int NK_per_split;    // K-tiles per split = ceil(NK_total / num_splits)
    // Input strides: [B, H, seqLen] in elements
    int64_t Q_strides[3];
    int64_t K_strides[3];
    int64_t V_strides[3];
    // pO strides: [B, H, qL] per split (split offset computed separately)
    int64_t pO_split_stride;   // B * H * qL * D
    int64_t pO_batch_stride;   // H * qL * D
    int64_t pO_head_stride;    // qL * D
    // pL strides: [B, H, qL] per split
    int64_t pL_split_stride;   // B * H * qL
    int64_t pL_batch_stride;   // H * qL
    int64_t pL_head_stride;    // qL
    // Optional features — appended at end for backward compatibility.
    float softcap;             // 0.0 = disabled; >0 → tanh(S/cap)*cap before softmax
};

/// Parameters for the Flash Decode Phase 2 (reduce) kernel.
struct FlashDecodeReduceParams {
    int B, H, D;
    int qL;
    int num_splits;
    // pO: [num_splits, B, H, qL, D]
    int64_t pO_split_stride;
    int64_t pO_batch_stride;
    int64_t pO_head_stride;
    // pL: [num_splits, B, H, qL]
    int64_t pL_split_stride;
    int64_t pL_batch_stride;
    int64_t pL_head_stride;
    // O: [B, H, qL, D]
    int64_t O_batch_stride;
    int64_t O_head_stride;
    // L: [B, H, qL]
    int64_t L_batch_stride;
    int64_t L_head_stride;
    // Reduce threadgroup size (= min(D, 128))
    int reduce_tgp_size;
};

/// Compute the number of KV splits for Flash Decoding.
/// Target: ~128 keys per split for good SM occupancy.
int compute_num_splits(int kL, int BK);

/// Generate Phase 1 Metal source for the Flash Decode partial kernel.
std::string generate_flash_decode_partial_source(const ShaderCache::KernelKey& key);

/// Generate Phase 2 Metal source for the Flash Decode reduce kernel.
std::string generate_flash_decode_reduce_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa

// =========================================================================
// MFASteelVarlenParams — STEEL varlen forward kernel params
// =========================================================================
// Layout MUST exactly match MFASteelVarlenParams in the Metal source string.
// Packed layout: Q/O = [1, H, total_q, D], K/V = [1, H_kv, total_kv, D]
struct MFASteelVarlenParams {
  int H, D;
  int gqa_factor;       // H / H_kv
  int num_seqs;         // number of independent sequences
  int total_q;          // sum of all q lengths
  int total_kv;         // sum of all kv lengths
  int total_q_tiles;    // sum of Q-tiles = tile_offsets[num_seqs]
  float scale;          // attention scale (1/sqrt(D))
  float softcap;        // 0.0 = disabled
  long Q_head_stride;   // = total_q * D
  long K_head_stride;   // = total_kv * D
};

// Generator function declaration (inside mlx_mfa namespace)
namespace mlx_mfa {
std::string generate_steel_varlen_forward_source(const ShaderCache::KernelKey& key);
}  // namespace mlx_mfa
