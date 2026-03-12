/// mfa_steel_fwd_v4.hpp  –  STEEL V4 forward kernel: direct device K reads.
///
/// V4 eliminates K threadgroup memory entirely.  On M3+ the per-lane L2-cached
/// device reads for K are fast enough to avoid the cooperative TGP transpose.
///
/// V4 barrier schedule vs V2:
///   V2 (4/iter): Q@K^T(TGP K) → A(K done) → load V → B(V ready) → P@V →
///                X(V done) → load K[next] → C(K ready)
///   V4 (2/iter): Q@K^T(device K) → P@V → A(V reads done) →
///                load V[next] → B(V ready)
///   Savings: 2 barriers/tile × NK tiles (X + C eliminated entirely).
///
/// TGP budget (no K_smem):
///   D=64  BK=64 all gens:  Q(4,608) + V(9,216)  = 13,824 B  ✅  (2 TG/CU)
///   D=128 BK=32 M1/M2:    Q(8,704) + V(8,704)  = 17,408 B  ✅  (1 TG/CU)
///   D=128 BK=64 M3+:      Q(8,704) + V(17,408) = 26,112 B  ✅  (1 TG/CU)
///
/// V4 is dispatched only on M3+ for D=64/128.  Set MFA_ENABLE_V4=1 to opt in
/// (disabled by default until benchmarks confirm benefit).
/// V4 does NOT support RoPE-K (K is read raw from device; no TGP for in-place
/// RoPE rotation).  RoPE falls back to V2.

#pragma once

#include "shader_cache.hpp"
#include "mfa_steel_fwd_v2.hpp"  // reuse SteelV2BlockConfig
#include <string>

namespace mlx_mfa {

/// Check whether V4 is TGP-eligible for a given (D, is_m3_plus) pair.
/// All configs have TGP < 32 KB, so V4 is always eligible when supported.
inline bool v4_tgp_eligible(int head_dim, bool /*is_m3_plus*/) {
  return (head_dim == 64 || head_dim == 128);
}

/// Block config for V4: same BQ/BK/WM as V2 (V4 changes smem layout only).
/// Uses the V2 config (which may use BK=64 for D=128 on M3+).
inline SteelV2BlockConfig select_steel_v4_block_config(int head_dim, bool is_m3_plus) {
  return select_steel_v2_block_config(head_dim, is_m3_plus);
}

/// Generate the complete Metal shader source for the STEEL V4 forward kernel.
/// Kernel function name: "mlx_mfa_v4_attention".
///
/// Supported: f16/bf16, D=64/128, causal/non-causal, GQA, softcap, ALiBi,
///            sliding window, sparse (block_mask).
/// NOT supported: RoPE (K not buffered in TGP; falls back to V2).
std::string generate_steel_v4_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
