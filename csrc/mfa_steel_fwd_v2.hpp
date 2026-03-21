/// mfa_steel_fwd_v2.hpp  –  STEEL V2 forward kernel: sequential K/V phases.
///
/// V2 doubles BK vs V1 by sharing K_smem and V_smem in a single KV_smem
/// buffer (sequential phases), reducing K-tile iterations by 2×:
///   D=64:  BQ=32, BK=64, WM=4 → TGP=13,824 B (V1: 14,336 B)  +22–85% vs V1 (causal)
///   D=128: BQ=32, BK=32, WM=4 → TGP=18,944 B (V1: 19,200 B)  +22–78% vs V1 (causal)
///
/// D=256 NOT dispatched: BQ must halve (→16) to fit 32KB TGP, which also halves
/// WM (4→2). Fewer warps/TG makes each K tile load slower; net regression vs V1
/// (0.62–0.84× causal, 0.58–0.62× non-causal).  D=256 routes to V1 in eval_gpu().
/// The D=256 kernel source and config are retained for future research.
///
/// 2× larger BK → 2× fewer K-tile iterations → 2× more compute per barrier stall.
/// D=256: pragma unroll disabled (TD=32 → register spill on M1/M2); M3+ still unrolls.

#pragma once

#include "shader_cache.hpp"
#include <string>

namespace mlx_mfa {

struct SteelV2BlockConfig {
  int BQ;   // query tile rows  (32 for D=64/128, 16 for D=256)
  int BK;   // KV tile rows     (64 for D=64, 32 for D=128/256) — 2× V1
  int BD;   // head dimension   (= D)
  int WM;   // SIMD groups      (4 for D=64/128, 2 for D=256)
  int WN;   // always 1
};

/// Select V2 tile config for f16/bf16 inputs.
/// Returns {0,0,0,0,0} for unsupported head dims (D>256).
SteelV2BlockConfig select_steel_v2_block_config(int head_dim, bool is_m3_plus);

/// Select D-split tile config for the large-D family (D=256/512).
/// This is intentionally separate from the D=64/128 selector so D=256 policy
/// can evolve independently. Uses MFA_V2_FORCE_BK_D256=32|64 when set.
SteelV2BlockConfig select_steel_v2_dsplit_block_config(bool is_m3_plus);

/// Select D-split tile config for D=512 ONLY.
/// Decoupled from D=256 so autoresearch can iterate independently.
/// Uses MFA_V2_FORCE_BK_D512=4|8|12|16|20|24|32 when set.
SteelV2BlockConfig select_steel_v2_d512_block_config(bool is_m3_plus);

/// Estimate actual GPU core count from MTLDevice name + fallback arch_gen.
/// Uses longest-prefix matching (Ultra > Max > Pro > base) so "M1 Max" matches
/// before "M1". Falls back to conservative gen-based estimate for unknown names.
/// If name is empty / unavailable (simulator, CI), uses arch_gen estimate only.
int estimate_gpu_cores(const std::string& device_name, int arch_gen);

/// Generate the complete Metal shader source for the STEEL V2 forward kernel.
/// Kernel function name: "mlx_mfa_v2_attention".
/// Supports: f16/bf16, D=64/128/256, causal/non-causal, GQA.
std::string generate_steel_v2_source(const ShaderCache::KernelKey& key);

// ── V2 Split-K (Phase 3) ─────────────────────────────────────────────────────
//
// Two-phase split-K for under-occupied grids (total_tgs < 0.8 * gpu_cores):
//   Phase 1: SteelV2SplitKPartial — each TG handles one Q-tile + K-range
//   Phase 2: FlashDecodeReduce    — reused as-is (no new kernel needed)
//
// Activation: v2_eligible && total_tgs < 0.8 * gpu_cores && S >= 2*BK
// Params struct: FlashDecodePartialParams (same as flash decode — reused)

/// Compute num_splits for V2 split-K.
/// FA2-inspired heuristic: find smallest s s.t. total_tgs*s >= gpu_cores.
/// Returns 1 if split is not beneficial (already well-occupied or too few K-tiles).
/// gpu_cores: actual core count from estimate_gpu_cores(); NOT arch_gen internally.
/// force_splitk=true bypasses the occupancy short-circuit and always tries to
/// find s>=2 when NK_total permits (used by MFA_FORCE_SPLITK=1 debug override).
int compute_v2_num_splits(int total_tgs, int kL, int BK, int gpu_cores, bool force_splitk = false);

/// Generate the Metal shader source for the V2 split-K partial kernel.
/// Kernel function name: "mlx_mfa_v2_splitk_partial".
/// Takes FlashDecodePartialParams; outputs pO (normalized, dtype T) and pL (float32 log2).
/// Phase 2 reduce uses the existing FlashDecodeReduce kernel (type 5).
std::string generate_steel_v2_splitk_partial_source(const ShaderCache::KernelKey& key);

// ── V2 D-split (CP1/CP2) ─────────────────────────────────────────────────────
//
// D-split attention for D=256 (D_SPLITS=2) and D=512 (D_SPLITS=4):
//   BD_HALF=128; each pass processes one BD_HALF chunk of the head dimension.
//   BK from select_steel_v2_dsplit_block_config(is_m3_plus) — separate large-D policy.
//   Q loaded into named register tiles (Qtile0, Qtile1, …) before the main loop.
//   K/V pointers advanced by K_strides[2] each K-tile; dh offset via + dh*BD_HALF.
//   No RoPE support (GPT-NeoX pairs cross BD_HALF boundary). All other features OK.

/// Generate Metal shader source for D-split V2 kernel (D=256/512).
/// Kernel function name: "mlx_mfa_v2_dsplit_attention".
/// BD_HALF=128; D_SPLITS = D/128 (2 for D=256, 4 for D=512).
/// Supports: f16/bf16, causal, softcap, ALiBi, sliding window, GQA. No RoPE.
std::string generate_steel_v2_dsplit_source(const ShaderCache::KernelKey& key);

}  // namespace mlx_mfa
