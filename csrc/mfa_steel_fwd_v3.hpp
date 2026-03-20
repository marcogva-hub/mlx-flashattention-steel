/// mfa_steel_fwd_v3.hpp  –  STEEL V3 forward kernel: separate K_smem + V_smem.
///
/// V3 eliminates 2 of 4 barriers per K-tile that V2 requires to safely share
/// a single KV_smem buffer.  By allocating K_smem and V_smem separately the
/// K→V and V→K[next] transition barriers are not needed:
///
///   V2 barrier schedule (4/iter non-last, 2/iter last):
///     Q@K^T → barrier A (K→V) → load V → barrier B → P@V →
///     barrier X (V→K) → load K[next] → barrier C
///
///   V3 barrier schedule (2/iter non-last, 1/iter last):
///     Q@K^T → P@V → barrier A (reads done) →
///     load K[next]+V[next] → barrier B (writes done)
///
/// TGP budget (separate K+V, not max(K,V)):
///   D=64  BK=64 all gens:  Q(4,608) + K(9,216) + V(9,216) = 23,040 B  ✅
///   D=128 BK=32 M1/M2:    Q(8,704) + K(10,240) + V(8,704) = 27,648 B  ✅
///   D=128 BK=64 M3+:      Q(8,704) + K(18,432) + V(17,408)= 44,544 B  ❌ OVER
///
/// V3 is dispatched BEFORE V2 in eval_gpu().  M3+ D=128 falls back to V2.
/// Set MFA_DISABLE_V3=1 to bypass V3 (forces V2 path).

#pragma once

#include "shader_cache.hpp"
#include "mfa_steel_fwd_v2.hpp"  // reuse SteelV2BlockConfig
#include <string>

namespace mlx_mfa {

/// Check whether V3 is eligible for a given (D, is_m3_plus) pair.
/// V3 TGP: Q + separate K_smem + separate V_smem (vs V2: Q + max(K,V)).
/// Eligible: D=64 (all gens) and D=128 M1/M2 only (BK=32, TGP=27,648 B < 32 KB).
inline bool v3_tgp_eligible(int head_dim, bool is_m3_plus) {
  if (head_dim == 64)  return true;            // TGP 23,040 B — all gens OK
  if (head_dim == 128) return !is_m3_plus;     // BK=32 only; M3+ BK=64 → 44 KB ❌
  return false;
}

/// Return the block config to use for V3 (same values as V2 since same BQ/BK/WM).
/// Only call when v3_tgp_eligible() is true.
inline SteelV2BlockConfig select_steel_v3_block_config(int head_dim, bool is_m3_plus) {
  // V3 uses the same BQ/BK/WM values as V2; the difference is the smem layout.
  // For D=128 we always use BK=32 in V3 (M3+ BK=64 doesn't fit).
  if (head_dim == 64)  return {32, 32,  64, 4, 1};
  if (head_dim == 128) return {32, 32, 128, 4, 1};  // BK=32 regardless of gen
  return {0, 0, 0, 0, 0};
}

/// Generate the complete Metal shader source for the STEEL V3 forward kernel.
/// Kernel function name: "mlx_mfa_v3_attention".
/// Supports: f16/bf16, D=64/128 (M1/M2 only for D=128), causal/non-causal,
/// GQA, softcap, ALiBi, sliding window, sparse (block_mask), RoPE.
std::string generate_steel_v3_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
