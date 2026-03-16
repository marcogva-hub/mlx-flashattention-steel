/// mfa_steel_fwd_v5.hpp  –  STEEL V5 forward kernel: D-blocked attention.
///
/// V5 key design: D is tiled into BD_tile=32 chunks.  Q+O stay in registers for
/// the entire K-loop (Q loaded from device once per Q-block; no Q_smem).
/// A single KV_smem buffer is reused for K^T and V chunks.
///
/// V5 TGP vs V2 (D=128, M1/M2):
///   V2 (BK=32): Q_smem(8704) + KV_smem(10240)  = 18,944 B  → 1 TG/CU
///   V5 (BK=128): KV_smem only = max(8704, 10240) = 10,240 B  → 3 TG/CU
///
/// V5 K-tile count vs V2 (D=128, N=4096):
///   V2 (M1/M2 BK=32): 128 K-tiles
///   V5 (BK=128): 32 K-tiles  (4× fewer, amortizing barrier overhead)
///
/// Total barriers per sequence length are comparable to V2:
///   V2 D=128:  128 tiles × 4 barriers = 512
///   V5 D=128:   32 tiles × ~17 barriers ≈ 544  (6% more, for 3× occupancy)
///
/// Q loading without TGP:
///   Each simdgroup owns BQ/WM = 8 unique Q rows — no cross-simdgroup sharing
///   needed, so Q is read directly from device memory into registers per SIMD.
///   This eliminates the Q_smem that V2 needed (8,704 bytes).
///
/// Block configs:
///   M1/M2: BQ=32, BK=128, BD_tile=32, WM=4  → TGP=10,240 B, 3 TG/CU
///   M3+:   BQ=16, BK=128, BD_tile=32, WM=2  → TGP=10,240 B, 3 TG/CU

#pragma once

#include "shader_cache.hpp"
#include <string>

namespace mlx_mfa {

struct SteelV5BlockConfig {
  int BQ;       // Query block size (sequence dim)
  int BK;       // Key block size (sequence dim) — 4× V2's BK
  int BD_tile;  // D-blocking tile size (head dim chunk)
  int WM;       // SIMD groups per threadgroup
};

/// Check if V5 is eligible for a given head_dim.
inline bool v5_eligible(int head_dim) {
  // D=512 is intentionally excluded here; D=512 stays in the dedicated
  // D-split family and is currently SDPA-default in auto-dispatch policy.
  return (head_dim == 64 || head_dim == 128);
}

/// Block config for V5.
/// BK=128 (4× V2's BK=32 for M1/M2) with BD_tile=32.
/// TGP: max(K^T_smem=8704, V_smem=10240) = 10,240 bytes → 3 TG/CU.
inline SteelV5BlockConfig select_steel_v5_block_config(int /*head_dim*/,
                                                       bool is_m3_plus) {
  // M3+ dynamic register allocation makes BQ=32 WM=4 viable even with
  // direct device reads. BQ=16 WM=2 halved occupancy and prevented
  // FP16/FP32 inter-simdgroup parallelism.
  (void)is_m3_plus;
  return {.BQ = 32, .BK = 128, .BD_tile = 32, .WM = 4};
}

/// Generate the Metal shader source for the STEEL V5 forward kernel.
/// Kernel function name: "mlx_mfa_v5_attention".
///
/// Supported: f16/bf16, D=64/128, causal/non-causal, GQA.
/// CP1 scope: dense (non-causal only) dispatch.  Causal + window + ALiBi
/// added in CP2.
std::string generate_steel_v5_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
