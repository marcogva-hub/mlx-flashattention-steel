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
/// Block configs (post-autoresearch):
///   M1/M2: BQ=32, BK=32, BD_tile=64, WM=4  → TGP=4,096 B, 8 TG/CU
///   (was:  BQ=32, BK=128, BD_tile=32, WM=4 → TGP=10,240 B, 3 TG/CU)

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
/// autoresearch (16 iters, M1 Max, 2026-03-20): BK=32 BD_tile=64
///   TGP = 32×64×2 = 4,096B → 8 TGs/CU (was 10,240B → 3 TGs/CU)
///   V5/SDPA: 1.57x geomean (was 1.30x)
///   V5/V3:   0.97x geomean (D=128: 1.01-1.08x wins; D=64: 0.89-0.96x loses)
///   D=128 B*H≥16: V5 beats V3 by 2-8% (more TGs/CU, fewer D-chunks)
inline SteelV5BlockConfig select_steel_v5_block_config(int head_dim,
                                                       bool is_m3_plus) {
  (void)is_m3_plus;
  // Autoresearch overrides (env vars take precedence over code changes)
  auto get_int = [](const char* env, int def) -> int {
    if (const char* v = std::getenv(env)) {
      const int p = std::atoi(v);
      if (p > 0) return p;
    }
    return def;
  };
  const int bk      = get_int("MFA_V5_FORCE_BK",      32);
  const int bd_tile = get_int("MFA_V5_FORCE_BD_TILE",  64);
  const int bq      = get_int("MFA_V5_FORCE_BQ",       32);
  const int wm      = get_int("MFA_V5_FORCE_WM",        4);
  (void)head_dim;
  return {.BQ = bq, .BK = bk, .BD_tile = bd_tile, .WM = wm};
}

/// Generate the Metal shader source for the STEEL V5 forward kernel.
/// Kernel function name: "mlx_mfa_v5_attention".
///
/// Supported: f16/bf16, D=64/128, causal/non-causal, GQA.
/// CP1 scope: dense (non-causal only) dispatch.  Causal + window + ALiBi
/// added in CP2.
std::string generate_steel_v5_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
