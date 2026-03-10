/// mfa_steel_fwd_v2.hpp  –  STEEL V2 forward kernel: sequential K/V phases.
///
/// V2 keeps BQ=V1's BQ (same grid, same occupancy) but doubles BK by sharing
/// K_smem and V_smem in a single KV_smem buffer (sequential, not simultaneous):
///   D=64:  BQ=32, BK=64, WM=4 → TGP=13,824 B (V1: 14,336 B)
///   D=128: BQ=32, BK=32, WM=4 → TGP=18,944 B (V1: 19,200 B)
///
/// Both configs fit in V1's footprint, so occupancy is ≥ V1. 2× larger BK →
/// 2× fewer K-tile iterations → 2× more compute per barrier stall.

#pragma once

#include "shader_cache.hpp"
#include <string>

namespace mlx_mfa {

struct SteelV2BlockConfig {
  int BQ;   // query tile rows  (= 64 for D=64/128)
  int BK;   // KV tile rows     (= 64 for D=64, 32 for D=128) — 2× V1
  int BD;   // head dimension   (= D)
  int WM;   // SIMD groups      (= 8)
  int WN;   // always 1
};

/// Select V2 tile config for f16/bf16 inputs.
/// Returns {0,0,0,0,0} for unsupported head dims (D>128).
SteelV2BlockConfig select_steel_v2_block_config(int head_dim);

/// Generate the complete Metal shader source for the STEEL V2 forward kernel.
/// Kernel function name: "mlx_mfa_v2_attention".
/// Supports: f16/bf16, D=64/128, causal/non-causal, GQA.
std::string generate_steel_v2_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
