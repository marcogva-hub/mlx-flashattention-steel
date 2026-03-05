/// mfa_steel_bwd.hpp — STEEL native backward kernel declarations.
///
/// Provides JIT Metal source generators for dQ and dK/dV backward passes
/// using the same tile infrastructure as the STEEL forward kernel.
///
/// Algorithm: FlashAttention-2 backward, log2-domain throughout.
///
///   P = exp2(S_log2 - L_log2)   where S_log2 = Q@K^T * scale * log2(e)
///   D = scale * rowsum(O ⊙ dO)  (delta, one float per Q row)
///   dS = P * (dO @ V^T - D)
///   dQ += scale * dS @ K        (parallelize over Q tiles)
///   dK += scale * dS^T @ Q      (parallelize over K tiles)
///   dV += P^T @ dO              (parallelize over K tiles)
///
/// Only f16/bf16 are dispatched to STEEL backward; f32 uses the ccv path.

#pragma once

#include "shader_cache.hpp"
#include <string>

namespace mlx_mfa {

/// Params for the STEEL backward dQ kernel.
///
/// Layout MUST exactly match the Metal-side MFASteelBwdParams struct
/// emitted by generate_steel_backward_dq_source() and
/// generate_steel_backward_dkv_source().
struct MFASteelBackwardParams {
    // Tensor dimensions
    int B, H, D;
    int qL, kL;
    int gqa_factor;      // H / H_kv  (1 = standard MHA)
    float scale;         // attention scale (1/sqrt(D)), NOT log2-multiplied
    float scale_log2;    // scale * log2(e) — used for score computation
    // Tile counts
    int NQ, NK;
    int NQ_aligned;      // = NQ - (qL%BQ ? 1 : 0)
    int NK_aligned;      // = NK - (kL%BK ? 1 : 0)
    int qL_rem;          // qL % BQ  (0 if aligned)
    int kL_rem;          // kL % BK
    int qL_off;          // cross-attention offset (S - N for decode, 0 otherwise)
    // Strides: [B stride, H stride, seq stride] for 4-D tensors [B, H, seq, D]
    int64_t Q_strides[3];
    int64_t K_strides[3];
    int64_t V_strides[3];
    int64_t O_strides[3];
    int64_t dO_strides[3];
    int64_t dQ_strides[3];
    int64_t dK_strides[3];
    int64_t dV_strides[3];
    // L (logsumexp) strides: [B stride, H stride]  (qL stride=1 implicit)
    int64_t L_strides[2];
};

/// Generate Metal source for the STEEL backward dQ kernel.
/// Grid: (NQ, H, B) — one threadgroup per Q-tile.
/// Buffers: Q(0), K(1), V(2), O(3), L(4), dO(5), D(6), dQ(7), params(8)
std::string generate_steel_backward_dq_source(
    const ShaderCache::KernelKey& key);

/// Generate Metal source for the STEEL backward dK/dV kernel.
/// Grid: (NK, H, B) — one threadgroup per K/V-tile.
/// Buffers: Q(0), K(1), V(2), O(3), L(4), D(5), dO(6), dK(7), dV(8), params(9)
std::string generate_steel_backward_dkv_source(
    const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
