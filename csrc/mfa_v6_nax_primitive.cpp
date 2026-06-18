/// MFAV6Forward — MLX Primitive that wraps the Draw Things NAAttention port.
///
/// Used to make the V6 forward kernel callable from Python via array::make_arrays
/// (the standard MLX pattern). Once correctness is validated, this can be
/// merged into MFAttention::eval_gpu() in mfa_attention.cpp as a fast-path.

#include "shader_cache.hpp"
#include "mfa_key_tie.hpp"
#include "mfa_env_aliases.hpp"
#include "mfa/v6_nax/NAAttentionKernel.hpp"

#include <mlx/mlx.h>
#include <mlx/primitives.h>
#include <mlx/backend/metal/device.h>
#include <mlx/allocator.h>
#include <mlx/utils.h>
#include <Metal/Metal.hpp>

#include <cmath>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <mutex>
#include <tuple>
#include <utility>
#include <CoreFoundation/CoreFoundation.h>  // CFRelease for pipeline-cache race handling

namespace mlx_mfa {

// Forward decls (defined in v6_nax_compile.mm).
void* v6_nax_compile_with_constants(
    const std::string& source, const std::string& function_name,
    void* raw_device,
    uint32_t R, uint32_t C, uint32_t Q_bs, uint32_t K_bs,
    uint32_t V_bs, uint32_t O_bs);

void v6_nax_dispatch(
    void* pipeline_raw,
    void* enc_raw,
    void* /*q_buf*/, uint64_t /*q_offset*/,
    void* /*k_buf*/, uint64_t /*k_offset*/,
    void* /*v_buf*/, uint64_t /*v_offset*/,
    void* /*o_buf*/, uint64_t /*o_offset*/,
    void* /*l_buf*/, uint64_t /*l_offset*/,
    uint32_t R, uint32_t Hq, uint32_t batchDimension,
    unsigned short BQ, uint16_t executionSIMDGroups,
    unsigned short tgmem_bytes);

void* v6nax_compile(const std::string& source, const std::string& function_name, void* raw_device);
void v6nax_dispatch(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

namespace {

uint32_t ceil_log2_u32(uint32_t x) {
  if (x <= 1) return 0;
  x -= 1;
  uint32_t b = 0;
  while (x > 0) { x >>= 1; ++b; }
  return b;
}

// Cache pipelines.
struct V6Key {
  int head_dim, Hq, Hk, dtype;
  bool isCausal;
  uint32_t R, C, qbs, kbs, vbs, obs;
  // Repo review 2026-05: tile/config params as dedicated fields.  These were
  // previously bit-packed into the high bits of R/C/qbs/kbs (e.g.
  // `qbs + (SG << 24)`), which collides once the stride exceeds 2^24 —
  // routine at production shapes (H=8, N=16384, D=128 → qbs = 2^24 exactly).
  // A collision returns a pipeline compiled for different tile sizes →
  // silently wrong kernel.
  uint16_t cfg_BQ = 0, cfg_BK = 0, cfg_SG = 0, cfg_BD = 0;
  // Campaign 2026-06 Sprint A (A-1): uint16_t, NOT uint8_t — axis_flags
  // accumulates up to bit 11 (MFA_V6_MAX_THREADS buckets use bits 7-9,
  // MFA_V6_MATMUL_EXEC_SG uses bits 10-11).  A uint8_t silently truncated
  // bits 8-11 to zero, aliasing distinct pipeline configs to one key
  // (e.g. MATMUL_EXEC_SG=4 reused the EXEC_SG=1 pipeline).  The loss
  // predates the 2026-05 bit-packing fix: the old `axis_flags << 24`
  // encoding overflowed the same bits out of the 32-bit word.
  uint16_t cfg_axis_flags = 0;
  bool     cfg_bypass_tgp = false;
  // V6NAX — dedicated cache-key fields (no bit-packing).
  bool use_v6nax = false;
  uint16_t v6nax_BQ = 0;
  uint16_t v6nax_BK = 0;
  uint16_t v6nax_WM = 0;
  // F-2 (Change 3): scale is baked into the source (#define V6NAX_DOT_SCALE), so a
  // distinct scale is a distinct pipeline.  Without this key field, a custom-scale
  // call would reuse the default-scale pipeline → silently wrong scale.
  float v6_scale = 0.0f;
  // Track 6: single declaration of the affecting-input set.
  auto tie() const {
    return std::tie(head_dim, Hq, Hk, dtype, isCausal, R, C, qbs, kbs, vbs,
                    obs, cfg_BQ, cfg_BK, cfg_SG, cfg_BD, cfg_axis_flags,
                    cfg_bypass_tgp, use_v6nax, v6nax_BQ, v6nax_BK, v6nax_WM,
                    v6_scale);
  }
  bool operator==(const V6Key& o) const { return tie() == o.tie(); }
};
struct V6KeyHash {
  size_t operator()(const V6Key& k) const { return mlx_mfa_keys::hash_tie(k.tie()); }
};

std::mutex v6_mtx;
std::unordered_map<V6Key, void*, V6KeyHash> v6_pipelines;

// Repo review 2026-05: race-safe pipeline cache insert.  The double-checked
// pattern (probe under lock → compile WITHOUT lock → store under lock) lets
// two threads compile the same key concurrently; the second store used to
// overwrite the first without releasing it, leaking the displaced
// MTLComputePipelineState (held via CFBridgingRetain in v6_nax_compile.mm).
// emplace + CFRelease-on-loss makes the insert race-safe and leak-free.
template <typename Map, typename Key>
static void* cache_insert_or_release(Map& map, std::mutex& mtx,
                                     const Key& key, void* pipeline) {
  std::lock_guard<std::mutex> lock(mtx);
  auto [it, inserted] = map.emplace(key, pipeline);
  if (!inserted) {
    CFRelease((CFTypeRef)pipeline);  // another thread won; drop our copy
    return it->second;
  }
  return pipeline;
}

std::string generate_v6_source(int head_dim, int Hq, int Hk, int dtype_code,
                                bool isCausal, bool bhnd, int R = 0,
                                bool use_v6nax_override = false,
                                bool use_v6nax_explicit = false,
                                float scale_override = -1.0f) {
  // F-2 (Change 3): scale is BAKED into the source as `#define V6NAX_DOT_SCALE`
  // (NAAttentionKernel.cpp createV6NAXSource, via descriptor.scale).  A custom
  // scale therefore produces a DISTINCT kernel — the dispatch cache key (V6Key)
  // MUST include the resolved scale or a different-scale call silently reuses the
  // wrong baked pipeline.  scale_override <= 0 means "use the default 1/sqrt(D)".
  const float resolved_scale =
      (scale_override > 0.0f) ? scale_override : 1.0f / std::sqrt((float)head_dim);
  GEMMOperandPrecision input_prec = (dtype_code == 1)
      ? GEMMOperandPrecision::BF16
      : GEMMOperandPrecision::FP16;
  AttentionOperands<GEMMOperandPrecision> mp;
  mp[AttentionOperand::Q] = input_prec;
  mp[AttentionOperand::K] = input_prec;
  mp[AttentionOperand::V] = input_prec;
  mp[AttentionOperand::O] = input_prec;
  mp[AttentionOperand::S] = GEMMOperandPrecision::FP32;
  mp[AttentionOperand::P] = GEMMOperandPrecision::FP32;
  mp[AttentionOperand::L] = GEMMOperandPrecision::FP32;

  // Tile dimensions: env vars override defaults (autoresearch).
  //   MFA_V6_BLOCK_R   — parallelization (rows per simdgroup) default 32
  //   MFA_V6_BLOCK_C   — traversal block (K cols) default 32
  //   MFA_V6_EXEC_SG   — simdgroups per threadgroup default 4
  //   MFA_V6_BLOCK_D   — head sub-tile default = head_dim (full)
  //   MFA_V6_BYPASS_TGP — Path A (cooperative→cooperative) default 0
  // Post-generation source string overrides:
  //   MFA_V6_FORCE_DYNAMIC_K — force dynamic_length_v<int> for K constants
  //   MFA_V6_RELAXED_PRECISION — 0 disables relaxed_precision in matmul2d
  //   MFA_V6_UNROLL_MODE — full | none | 2 | 4 — pragma loop unroll setting
  // Sprint 3.3 + autoresearch — auto-tuned per-D defaults.
  // Bench at the original BQ=32 default showed bimodal: D=64 wins, D=128 loses
  // under single-Otile. Autoresearch sweep (BQ ∈ {16,32,64} × BK ∈ {32,64} ×
  // SG ∈ {2,4,8}, single-Otile=on) revealed BQ=16 wins universally — the
  // BQ=32 default was the dominant bottleneck, not the kernel structure:
  //   D=64  best:  BQ=16 BK=64 SG=2  (FlashVSR-dense 1.11ms = 1.22×SDPA;
  //                                    LTX2-cross 1.59ms = 1.20×SDPA)
  //   D=128 best:  BQ=16 BK=32 SG=8  (SeedVR2-small 276ms = 1.49×SDPA;
  //                                    vs 1129ms at the BQ=32 default)
  // BQ=64 is uniformly catastrophic (4× slower) — coop_tensor row-count too
  // large for register packing. Default to single-Otile + BQ=16 + per-D
  // BK/SG, with env var overrides preserved.
  unsigned short BQ = 16;
  // Dispatch v5 (REVERTED from v2.30 dispatch v6 — thermal-controlled
  // re-bench showed v6 regresses SeedVR2-large +14.3% and SeedVR2-small
  // +5.9% vs v5; Sprint G's "wins" were within-session pipeline-cache
  // artifacts that didn't replicate cross-session).
  unsigned short BK = (head_dim == 64) ? 64 : 32;
  uint16_t exec_sg;
  if (head_dim == 64) {
    exec_sg = 2;
  } else {
    exec_sg = (R >= 50000) ? 16 : 8;
  }
  bool bypass_tgp = false;
  // Sprint B (v2.30): BHND rewriter now handles GQA — single-Otile is the
  // default for both non-GQA and GQA-divisible (Hq % Hk == 0).
  bool single_otile = (Hq == Hk) || (Hk > 0 && Hq % Hk == 0);
  unsigned short BD = (unsigned short)head_dim;
  if (const char* env_r = std::getenv("MFA_V6_BLOCK_R")) BQ = (unsigned short)std::atoi(env_r);
  if (const char* env_c = std::getenv("MFA_V6_BLOCK_C")) BK = (unsigned short)std::atoi(env_c);
  if (const char* env_sg = std::getenv("MFA_V6_EXEC_SG")) exec_sg = (uint16_t)std::atoi(env_sg);
  if (const char* env_b = std::getenv("MFA_V6_BYPASS_TGP")) bypass_tgp = (std::atoi(env_b) != 0);
  // Explicit env override (set 0 or 1) wins over the auto-default above.
  if (const char* env_so = std::getenv("MFA_V6_NAX_SINGLE_OTILE")) single_otile = (std::atoi(env_so) != 0);
  if (const char* env_d = std::getenv("MFA_V6_BLOCK_D")) BD = (unsigned short)std::atoi(env_d);
  // Sprint 3.3: single-Otile mode forces bypass on (the new path always uses cP).
  if (single_otile) bypass_tgp = true;

  // V6NAX — NAX-direct rewrite. Caller (eval_gpu) decides via use_v6nax_explicit
  // based on shape (D, Nk). Default fallback when not explicit: D=128 → ON.
  // Forward, non-causal, single-Otile-eligible only.
  bool use_v6nax;
  if (use_v6nax_explicit) {
    use_v6nax = use_v6nax_override;
  } else {
    use_v6nax = (head_dim == 128);  // source-gen-only default (without Nk info)
    if (const char* env_v6nax = mlx_mfa::getenv_aliased("MFA_V6_USE_NAX"))
      use_v6nax = (std::atoi(env_v6nax) != 0);
  }
  // v2.50 Prompt 4 Section B: lift `isCausal` constraint.  Prompt 2
  // Phase 4a added V6NAX forward causal kernel support but missed this
  // dispatch-side gate — causal was silently routing to STEEL legacy
  // (log2-domain lse) instead of V6NAX (natural-log lse), making
  // V6NAX backward consume wrong-domain lse and produce wrong gradients.
  // Now V6NAX forward + V6NAX backward causal both engage when force_v6nax=True.
  if (use_v6nax && !single_otile) use_v6nax = false;
  // V6NAX needs BQ % (WM * 16) == 0 and BD % 16 == 0.
  // Per-D defaults: D=64 → WM=2, BQ=32, BK=64; D=128 → WM=4, BQ=64, BK=32.
  // Override via env vars below.
  unsigned short v6nax_BQ = (head_dim == 64) ? 32 : 64;
  // NAX-autotune (research/nax-autotune-m5, M5 Max): BK=32 for ALL D. The old
  // D=64 default BK=64 was inherited, not tuned for the post-F-3 NAX kernel; a
  // hardware sweep over the (only) LIVE tile knobs found BK=32 is robustly faster
  // for D=64 (−2..−15% across 6 shapes × {fp16,bf16}; larger N benefits most) and
  // is the existing D=128 default. (Shader side — must match the eval_gpu dispatch.)
  unsigned short v6nax_BK = 32;
  uint16_t v6nax_WM = (head_dim == 64) ? 2 : 4;
  if (use_v6nax) {
    if (const char* env_bq = mlx_mfa::getenv_aliased("MFA_V6_NAX_BQ")) v6nax_BQ = (unsigned short)std::atoi(env_bq);
    if (const char* env_bk = mlx_mfa::getenv_aliased("MFA_V6_NAX_BK")) v6nax_BK = (unsigned short)std::atoi(env_bk);
    if (const char* env_wm = mlx_mfa::getenv_aliased("MFA_V6_NAX_WM")) v6nax_WM = (uint16_t)std::atoi(env_wm);
    // Validate: BQ % (WM*16) == 0
    if (v6nax_BQ % (v6nax_WM * 16) != 0 || head_dim % 16 != 0) {
      use_v6nax = false;  // fall back to legacy if invalid config
    }
    // Phase II-8 addendum (Pattern #9, THIRD site): the V6NAX forward
    // generator emits the QK matmul as a PAIRED 16x32x16 MMA
    // (`for (ik = 0; ik < V6NAX_TK; ik += 2)` — NAAttentionKernel.cpp
    // ~line 2885), so V6NAX_TK = BK/16 must be even.  MFA_V6_NAX_BK=16
    // (or any non-multiple of 32) would reproduce the II-6 backward
    // out-of-bounds corruption in the FORWARD.  Loud failure per
    // Rule 8 (env override only — defaults 64/32 are valid).
    if (v6nax_BK == 0 || v6nax_BK % 32 != 0) {
      throw std::runtime_error(
          "V6NAX forward: BK must be a positive multiple of 32 (paired "
          "16x32x16 MMA requires TK = BK/16 even). Got BK=" +
          std::to_string((int)v6nax_BK) + " (MFA_V6_NAX_BK).");
    }
  }

  simd::ushort3 blockDims = use_v6nax
      ? simd::make_ushort3(v6nax_BQ, v6nax_BK, BD)
      : simd::make_ushort3(BQ, BK, BD);
  uint16_t exec_sg_for_desc = use_v6nax ? v6nax_WM : exec_sg;

  NAAttentionKernelDescriptor desc(
      blockDims, (unsigned short)head_dim, (unsigned short)Hq,
      (unsigned short)Hk, /*executionSIMDGroups=*/exec_sg_for_desc,
      /*checkCEdge1=*/true, mp, AttentionKernelType::forward,
      /*scale=*/resolved_scale,
      /*bypassThreadgroupMemory=*/bypass_tgp,
      /*isCausal=*/isCausal, /*masked=*/false);
  desc.singleOtileMode = single_otile;
  desc.useV6NAX = use_v6nax;

  NAAttentionKernel kern(desc);
  std::string source = kern.source;

  // ── Post-generation substitutions for Axes 4, 5, 6 ──────────────────────
  // Helper: replace ALL occurrences of `from` with `to` in `s`.
  auto replace_all = [](std::string& s, const std::string& from,
                        const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
      s.replace(pos, from.size(), to);
      pos += to.size();
    }
  };

  // Axe 4: force dynamic_length_v even when K%32==0 (paradox test).
  if (const char* env_dk = std::getenv("MFA_V6_FORCE_DYNAMIC_K")) {
    if (std::atoi(env_dk) != 0) {
      // The static K values appear inside matmul2d_descriptor(R, C, K, ...).
      // We swap any static numeric K (28, 32, 48, 64, 80, 96, 128, ...) for
      // `dynamic_length_v<int>` only inside matmul2d_descriptor calls. Doing
      // a coarse regex-like replace is fragile; instead we look for the
      // already-substituted K constants and replace them.
      // For simplicity, we replace the BLOCK_C value (BK) and BD/HEAD_DIM
      // when they appear as the third arg of matmul2d_descriptor.
      // (The remainder qk_desc uses HEAD_DIMENSION_REMAINDER which is small
      // and may not be a multiple of 32 — leave it alone.)
      std::string bk_str = std::to_string(BK);
      std::string bd_str = std::to_string(BD);
      // Only swap if BK or BD are multiples of 32 (otherwise dynamic is
      // already in use). NB: the SUBSTITUTION targets must be unique in
      // the source — they appear inside `matmul2d_descriptor(R, C, K, ...)`
      // which is precisely where we want them.
      if (BK % 32 == 0) {
        // Substring " " + bk_str + ", false, false," is the PV descriptor's
        // K position; ", false, true, true" is the QK descriptor's flags.
        // Use safer marker — replace " <BK>, false, false, true," etc.
        std::string find_pv = ", " + bk_str + ", false, false, true,";
        std::string find_qk = ", " + bk_str + ", false, true, true,";
        replace_all(source, find_pv, ", dynamic_length_v<int>, false, false, true,");
        replace_all(source, find_qk, ", dynamic_length_v<int>, false, true, true,");
      }
      if (BD % 32 == 0 && BD != BK) {
        std::string find_qk_d = ", " + bd_str + ", false, true, true,";
        replace_all(source, find_qk_d, ", dynamic_length_v<int>, false, true, true,");
      }
    }
  }

  // Axe 5: relaxed_precision toggle.
  // matmul2d_descriptor(R, C, K, leftT, rightT, /*relaxed*/ true, ...)
  if (const char* env_rp = std::getenv("MFA_V6_RELAXED_PRECISION")) {
    if (std::atoi(env_rp) == 0) {
      // Find ", true, true, matmul2d_descriptor::mode" → ", true, false, ..."
      // and ", false, true, true, matmul2d_descriptor::mode" → ", false, true, false, ..."
      // and ", false, false, true, matmul2d_descriptor::mode" → ", false, false, false, ..."
      replace_all(source,
                  ", true, true, matmul2d_descriptor::mode",
                  ", true, false, matmul2d_descriptor::mode");
      replace_all(source,
                  ", false, true, true, matmul2d_descriptor::mode",
                  ", false, true, false, matmul2d_descriptor::mode");
      replace_all(source,
                  ", false, false, true, matmul2d_descriptor::mode",
                  ", false, false, false, matmul2d_descriptor::mode");
    }
  }

  // Axe 6: K-loop unroll mode override.
  if (const char* env_un = std::getenv("MFA_V6_UNROLL_MODE")) {
    std::string mode = env_un;
    std::string replacement;
    if (mode == "full") replacement = "#pragma clang loop unroll(full)";
    else if (mode == "none") replacement = "#pragma clang loop unroll(disable)";
    else if (mode == "2") replacement = "#pragma clang loop unroll_count(2)";
    else if (mode == "4") replacement = "#pragma clang loop unroll_count(4)";
    else replacement = "#pragma clang loop unroll(full)";  // fallback
    replace_all(source, "#pragma clang loop unroll(full)", replacement);
  }

  // Campaign 2026-06 Sprint A: the v2.30 MFA_V6_MATMUL_EXEC_SG experiment
  // knob (blind replace_all of execution_simdgroups<1> with <N>) is REMOVED.
  // Current MetalPerformancePrimitives headers statically require single-SG
  // scope for the cooperative-tensor map_index/operand-layout patterns this
  // source uses — <N> for N>1 fails compilation with static_asserts.  The
  // knob only ever "worked" post-v2.30 because the cache key truncated its
  // axis_flags bits (10-11) to zero, silently aliasing every override to
  // the <1> pipeline (a Pattern #8-style ghost: the knob was a no-op).
  // Fixing the key truncation (A-1) surfaced the incompatibility loudly.

  // ── Sprint 2A: BHND layout migration (MFA_V6_BHND=1) ─────────────────────
  // Rewrites the kernel to read Q/K/V/O in [B, H, N, D] layout (MLX native)
  // instead of [B, N, H, D] (Draw Things native). Eliminates the host-side
  // transpose+contiguous overhead (3 dispatches + 3× peak memory).
  //
  // Strategy:
  //  1. Per-batch base offset gains a per-head offset:
  //       Q_buf += tgid.z * Q_batch_stride           (BNHD)
  //     becomes
  //       Q_buf += tgid.z * Q_batch_stride + tgid.y * R * D  (BHND)
  //     (and analogous for K, V, O — using C and Hk for K/V)
  //  2. Tensor declarations: dextents(K_Hq, R) → dextents(D, R) (per-head view)
  //  3. Slice args: drop the `tgid.y * D + ` head offset (head is in Q_buf base)
  //  4. Output writeback: drop `+ tgid.y * D` and replace `K_Hq` with `D`
  //
  // Limitation: Forward path only, non-GQA only (Hq == Hk). For GQA (LTX2-cross),
  // K/V's per-head offset uses `(tgid.y / ratio)` which has different syntax we
  // don't rewrite here — falls back to BNHD path.
  // Sprint 2A: BHND mode is now the DEFAULT (gated by Params, not env var).
  // Caller decides per-call. GQA shapes (Hq != Hk) auto-fall-back to BNHD.
  // Legacy BNHD path can be force-enabled via MFA_V6_BNHD_LEGACY=1 in caller.
  if (bhnd) {
    // Sprint B (v2.30) — GQA support extension. The original BHND rewriter
    // (Sprint 2A) only handled Hq == Hk because the patterns for K/V slice
    // args (`tgid.y / ratio * D`) differ from Q/O patterns (`tgid.y * D`).
    // The new branch below handles Hq != Hk by emitting per-KV-head offsets
    // `(tgid.y / ratio) * C * D` in the K/V buffer bases.
    if (Hq != Hk && Hq % Hk == 0) {
      const int ratio = Hq / Hk;
      const std::string D_str = std::to_string(head_dim);
      const std::string ratio_str = std::to_string(ratio);
      // Q, O patterns use raw tgid.y (Q-head index, Hq heads).
      const std::string head_y_D = "tgid.y * " + D_str;
      // K, V patterns from the source emitter look like `tgid.y / RATIO* D`
      // (note: no space before '*'; the H_HK_RATIO substitution
      // is "/ <ratio>" so the literal becomes "tgid.y / 4* 128").
      const std::string head_y_div_D = "tgid.y / " + ratio_str + "* " + D_str;

      // Step 1: per-batch base offset → add per-head offset
      replace_all(source,
                  "Q_buf = Q_buf + tgid.z * Q_batch_stride;",
                  "Q_buf = Q_buf + tgid.z * Q_batch_stride + tgid.y * R * "
                  + D_str + ";");
      replace_all(source,
                  "O_buf = O_buf + tgid.z * O_batch_stride;",
                  "O_buf = O_buf + tgid.z * O_batch_stride + tgid.y * R * "
                  + D_str + ";");
      // K/V use the KV-head index (tgid.y / ratio).
      replace_all(source,
                  "K_buf = K_buf + tgid.z * K_batch_stride;",
                  "K_buf = K_buf + tgid.z * K_batch_stride + (tgid.y / "
                  + ratio_str + ") * C * " + D_str + ";");
      replace_all(source,
                  "V_buf = V_buf + tgid.z * V_batch_stride;",
                  "V_buf = V_buf + tgid.z * V_batch_stride + (tgid.y / "
                  + ratio_str + ") * C * " + D_str + ";");

      // Step 2: tensor extents → per-head (D, seq).
      replace_all(source,
                  "dextents<int32_t, 2>(K_Hq, R)",
                  "dextents<int32_t, 2>(" + D_str + ", R)");
      replace_all(source,
                  "dextents<int32_t, 2>(K_Hk, C)",
                  "dextents<int32_t, 2>(" + D_str + ", C)");

      // Step 3: drop head offset in slice args. Order matters: the longer
      // K/V pattern (`tgid.y / 4* 128`) is a SUBSTRING-shape that must be
      // matched BEFORE the Q/O bare pattern (`tgid.y * 128`); otherwise the
      // bare match would catch the leading `tgid.y` and mangle the GQA
      // expression. Match-with-trailing-plus first to handle composite
      // expressions inside slice args.
      replace_all(source, head_y_div_D + " + ", "");  // K/V slices with `+ k` follow-on
      replace_all(source, head_y_div_D, "0");          // bare K/V slice arg
      replace_all(source, head_y_D + " + ", "");      // Q/O slices with `+ k`
      replace_all(source, head_y_D, "0");              // bare Q/O slice arg

      // Step 4: output writeback (same as non-GQA).
      replace_all(source, "+ 0;\n", ";\n");
      replace_all(source, " + 0)", ")");
      replace_all(source, "* K_Hq)", "* " + D_str + ")");
      replace_all(source, "idx[1] * K_Hq", "idx[1] * " + D_str);
    }
    else if (Hq == Hk) {  // non-GQA only for now
      const std::string D_str = std::to_string(head_dim);
      const std::string head_y_D = "tgid.y * " + D_str;

      // Step 1: per-batch base offset → add per-head offset
      // For Q, O (use R = sequence length, Hq heads):
      replace_all(source,
                  "Q_buf = Q_buf + tgid.z * Q_batch_stride;",
                  "Q_buf = Q_buf + tgid.z * Q_batch_stride + tgid.y * R * "
                  + D_str + ";");
      replace_all(source,
                  "O_buf = O_buf + tgid.z * O_batch_stride;",
                  "O_buf = O_buf + tgid.z * O_batch_stride + tgid.y * R * "
                  + D_str + ";");
      // For K, V (use C = KV-length, Hk heads — non-GQA so tgid.y == K-head):
      replace_all(source,
                  "K_buf = K_buf + tgid.z * K_batch_stride;",
                  "K_buf = K_buf + tgid.z * K_batch_stride + tgid.y * C * "
                  + D_str + ";");
      replace_all(source,
                  "V_buf = V_buf + tgid.z * V_batch_stride;",
                  "V_buf = V_buf + tgid.z * V_batch_stride + tgid.y * C * "
                  + D_str + ";");

      // Step 2: tensor extents → per-head (D, seq) instead of (Hq*D, seq)
      replace_all(source,
                  "dextents<int32_t, 2>(K_Hq, R)",
                  "dextents<int32_t, 2>(" + D_str + ", R)");
      replace_all(source,
                  "dextents<int32_t, 2>(K_Hk, C)",
                  "dextents<int32_t, 2>(" + D_str + ", C)");

      // Step 3: drop head offset in slice args
      // Order matters: replace "tgid.y * D + " (with trailing space+plus) BEFORE
      // replacing bare "tgid.y * D", otherwise the bare match would catch the
      // prefix and leave " + " orphaned.
      replace_all(source, head_y_D + " + ", "");
      replace_all(source, head_y_D, "0");

      // Step 4: output writeback
      // Pattern: "O_buf + tgid.x * (BR * K_Hq) + 0" — note that the `+ tgid.y * 64`
      // already became `+ 0` from Step 3's bare-match replacement. Now collapse:
      // `+ 0` and replace `K_Hq` with `D`. We do the collapse first.
      replace_all(source, "+ 0;\n", ";\n");      // statement-end + 0
      replace_all(source, " + 0)", ")");          // inside parens
      // Replace remaining K_Hq usages (now only in output base + idx[1]*K_Hq)
      // with D. K_Hq is also defined at line 37 as `constant uint K_Hq = 64 * Hq;`
      // but redefining it via search is too risky — leave the constant declaration,
      // just rewrite the USES.
      // Skip the constant declaration line (which is `constant uint K_Hq = D * Hq`).
      // The remaining uses are:
      //   `tgid.x * (BR * K_Hq)`   — output base, want BR * D
      //   `idx[1] * K_Hq`          — output cell store row stride, want D
      replace_all(source, "* K_Hq)", "* " + D_str + ")");
      replace_all(source, "idx[1] * K_Hq", "idx[1] * " + D_str);
    }
  }

  return source;
}

}  // namespace

// MFAV6Forward — Primitive for V6 NAX forward attention.
class MFAV6Forward : public mlx::core::Primitive {
public:
  struct Params {
    bool causal;
    bool bhnd;  // Sprint 2A: layout flag, decided by caller per-call.
    // v2.37.0 V6NAX backward integration: force V6NAX forward routing even
    // on D=64 small-Nk shapes (which by default route to legacy v6_nax).
    // Caller passes true when V6NAX backward will consume the lse.
    bool force_v6nax = false;
    // F-2 (Change 3): custom QK scale (baked into the kernel source). <=0 sentinel
    // means "use the default 1/sqrt(D)". Plumbed from the binding so the dense NAX
    // forward works at ALL scales (no custom-scale footgun).
    float scale = -1.0f;
  };

  MFAV6Forward(mlx::core::Stream stream, Params params)
      : mlx::core::Primitive(stream), params_(params) {}

  const char* name() const override { return "MFAV6Forward"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("V6 NAX is GPU only");
  }

  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override {
    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    auto& out = outputs[0];
    auto& lse = outputs[1];

    // Layout selection (Sprint 2A): per-call via Params, decided by wrapper.
    // BHND is the default; BNHD is opt-in via MFA_V6_BNHD_LEGACY=1 (caller-side)
    // or auto-fallback for GQA shapes (Hq != Hk).
    const bool bhnd = params_.bhnd;
    int B  = q.shape(0);
    int N  = bhnd ? q.shape(2) : q.shape(1);
    int Hq = bhnd ? q.shape(1) : q.shape(2);
    int D  = q.shape(3);
    int Nk = bhnd ? k.shape(2) : k.shape(1);
    int Hk = bhnd ? k.shape(1) : k.shape(2);

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V6: only FP16/BF16");

    out.set_data(mlx::core::allocator::malloc(out.nbytes()));
    lse.set_data(mlx::core::allocator::malloc(lse.nbytes()));

    // DIAGNOSTIC ONLY (MFA_V6_SENTINEL_FILL=1): host-fill the output
    // buffer with a sentinel pattern before kernel dispatch. Apple
    // Silicon unified memory: host writes to data<T>() are visible to
    // the GPU once the encoder is committed. Any cell still equal to
    // the sentinel after dispatch is provably *not* written by the
    // kernel — direct detection of the Day J `tensor_inline + matmul2d`
    // partial-output bug.
    //   FP16 0x7E00 = signaling NaN; mathematically impossible from
    //   correct softmax(QK^T)·V on finite inputs.
    //   FP32 LSE: 0x7FC00000 = FP32 quiet NaN.
    if (std::getenv("MFA_V6_SENTINEL_FILL")) {
      const uint16_t fp16_sentinel = 0x7E00;
      uint16_t* o_ptr = out.data<uint16_t>();
      const size_t o_n = out.nbytes() / sizeof(uint16_t);
      for (size_t i = 0; i < o_n; ++i) o_ptr[i] = fp16_sentinel;
      const uint32_t fp32_sentinel = 0x7FC00000u;
      uint32_t* l_ptr = lse.data<uint32_t>();
      const size_t l_n = lse.nbytes() / sizeof(uint32_t);
      for (size_t i = 0; i < l_n; ++i) l_ptr[i] = fp32_sentinel;
    }

    auto& d = mlx::core::metal::device(stream().device);
    void* mtl_device = d.mtl_device();

    uint32_t R = (uint32_t)N;
    uint32_t C = (uint32_t)Nk;
    // Repo review 2026-05: compute strides in int64 before narrowing —
    // `Hq * N * D` in int32 is UB for products > 2^31; the wrapped value
    // would address wrong GPU memory silently.  Guard the uint32 ceiling.
    const int64_t qbs64 = (int64_t)Hq * N * D;
    const int64_t kbs64 = (int64_t)Hk * Nk * D;
    if (qbs64 > (int64_t)UINT32_MAX || kbs64 > (int64_t)UINT32_MAX)
      throw std::runtime_error(
          "V6 NAX: batch stride exceeds uint32 (H*N*D too large)");
    uint32_t qbs = (uint32_t)qbs64;
    uint32_t kbs = (uint32_t)kbs64;
    uint32_t vbs = kbs;
    uint32_t obs = qbs;

    // Tile params — auto-tuned defaults (mirror the source-gen path above).
    // Dispatch v6 (Sprint C+G): D=64 SG=4; D=128 3-way N-conditional with
    // BK shift at N>=100000.
    // Dispatch v5 (REVERTED from v2.30 dispatch v6).
    unsigned short BQ = 16;
    unsigned short BK = (D == 64) ? 64 : 32;
    uint16_t executionSIMDGroups;
    if (D == 64) {
      executionSIMDGroups = 2;
    } else {
      executionSIMDGroups = ((int)R >= 50000) ? 16 : 8;
    }
    bool bypass_tgp = false;
    unsigned short BD = (unsigned short)D;
    int axis_flags = 0;
    if (const char* env_r = std::getenv("MFA_V6_BLOCK_R")) BQ = (unsigned short)std::atoi(env_r);
    if (const char* env_c = std::getenv("MFA_V6_BLOCK_C")) BK = (unsigned short)std::atoi(env_c);
    if (const char* env_sg = std::getenv("MFA_V6_EXEC_SG")) executionSIMDGroups = (uint16_t)std::atoi(env_sg);
    if (const char* env_b = std::getenv("MFA_V6_BYPASS_TGP")) bypass_tgp = (std::atoi(env_b) != 0);
    if (const char* env_d = std::getenv("MFA_V6_BLOCK_D")) BD = (unsigned short)std::atoi(env_d);
    // Axes 4-6 affect the kernel source — fold them into a flag for cache.
    if (const char* env_dk = std::getenv("MFA_V6_FORCE_DYNAMIC_K"))
      if (std::atoi(env_dk) != 0) axis_flags |= 0x01;
    if (const char* env_rp = std::getenv("MFA_V6_RELAXED_PRECISION"))
      if (std::atoi(env_rp) == 0) axis_flags |= 0x02;
    if (const char* env_un = std::getenv("MFA_V6_UNROLL_MODE")) {
      std::string m(env_un);
      if (m == "none") axis_flags |= 0x04;
      else if (m == "2") axis_flags |= 0x08;
      else if (m == "4") axis_flags |= 0x10;
    }
    if (params_.bhnd) axis_flags |= 0x20;  // Sprint 2A — BHND layout
    // Sprint 3.3 — single-Otile cache key. Mirror the auto-default logic from
    // the source-generation path so the cache key matches whichever variant was
    // actually compiled.
    {
      // Sprint B (v2.30): single-Otile default mirrors the GQA-supporting
      // logic in the source-gen path. Both non-GQA and GQA-divisible.
      bool so_for_key = (Hq == Hk) || (Hk > 0 && Hq % Hk == 0);
      if (const char* env_so = std::getenv("MFA_V6_NAX_SINGLE_OTILE"))
        so_for_key = (std::atoi(env_so) != 0);
      if (so_for_key) axis_flags |= 0x40;
    }
    // Sprint E (v2.30) — MFA_V6_MAX_THREADS env var changes pipeline state
    // attribute (maxTotalThreadsPerThreadgroup). Different settings produce
    // different compiled pipelines; encode in the cache key.
    if (const char* env_mt = std::getenv("MFA_V6_MAX_THREADS")) {
      int v = std::atoi(env_mt);
      // Encode with 3 bits in axis_flags 0x80/0x100/0x200 — discrete buckets.
      if (v > 0 && v <= 256) axis_flags |= 0x80;
      else if (v > 256 && v <= 384) axis_flags |= 0x100;
      else if (v > 384 && v <= 512) axis_flags |= 0x180;
      else if (v > 512 && v <= 768) axis_flags |= 0x200;
      // 769-1024 maps to default (0) — no bit set.
    }
    // MFA_V6_MATMUL_EXEC_SG encoding removed (campaign 2026-06 Sprint A) —
    // the substitution it keyed is gone (statically illegal on current MPP;
    // see the note in the source-substitution section above).

    // Audit F-3 (2026-06-18): V6 is PURE NAX.  The simdgroup-within-V6 fallback
    // (the old `use_v6nax=false` path) was a DIVERGED, BROKEN duplicate of the
    // standalone simdgroup family — it produced garbage at D=64 (e.g. D=64 N=4096
    // gave max-abs-err ≈ 512 vs fp32) and was UNREACHABLE from production Python
    // (every MFAV6Forward entry forces NAX — the F-2 dense route and the backward
    // recompute — and NAX-ineligible dense shapes route to the EXISTING dispatch:
    // D=64 dense → SDPA, F-2).  It is removed.  MFAV6Forward now serves ONLY NAX
    // (D ∈ {64,128}, valid GQA); the only legacy escape (MFA_V6_USE_NAX=0 → broken
    // simdgroup) is gone too (the env name is retained as a deprecated no-op alias).
    bool use_v6nax = true;  // D is constrained to {64,128} by v6_nax_forward()
    const bool valid_gqa = (Hq == Hk) || (Hk > 0 && Hq % Hk == 0);
    if (!valid_gqa) {
      // NAX requires Hq % Hk == 0.  The removed simdgroup fallback was the only
      // (broken) path for invalid GQA — fail loudly (Rule 8) instead of silently
      // dispatching garbage; such shapes belong on flash_attention (SDPA).
      throw std::runtime_error(
          "v6_nax_forward: invalid GQA (Hq=" + std::to_string(Hq) +
          " not a multiple of Hk=" + std::to_string(Hk) +
          "); NAX requires Hq % Hk == 0.  The simdgroup-within-V6 fallback was "
          "removed in audit F-3 (broken diverged duplicate) — route such shapes "
          "through flash_attention (SDPA) instead.");
    }
    unsigned short v6nax_BQ = (D == 64) ? 32 : 64;
    // NAX-autotune (M5 Max): BK=32 for all D — D=64 BK=64 was an untuned inherited
    // default; the hardware sweep found BK=32 robustly faster for D=64 (−2..−15%).
    // (Dispatch side — must match the generate_v6_source shader default above.)
    unsigned short v6nax_BK = 32;
    uint16_t v6nax_WM = (D == 64) ? 2 : 4;
    if (use_v6nax) {
      if (const char* env_bq = mlx_mfa::getenv_aliased("MFA_V6_NAX_BQ")) v6nax_BQ = (unsigned short)std::atoi(env_bq);
      if (const char* env_bk = mlx_mfa::getenv_aliased("MFA_V6_NAX_BK")) v6nax_BK = (unsigned short)std::atoi(env_bk);
      if (const char* env_wm = mlx_mfa::getenv_aliased("MFA_V6_NAX_WM")) v6nax_WM = (uint16_t)std::atoi(env_wm);
      if (v6nax_BQ % (v6nax_WM * 16) != 0 || D % 16 != 0) {
        use_v6nax = false;
      }
      // Phase II-8 addendum (Pattern #9, third site — see the guard in
      // the other dispatch path above): paired-MMA forward requires
      // BK % 32 == 0.
      if (use_v6nax && (v6nax_BK == 0 || v6nax_BK % 32 != 0)) {
        throw std::runtime_error(
            "V6NAX forward: BK must be a positive multiple of 32 (paired "
            "16x32x16 MMA requires TK = BK/16 even). Got BK=" +
            std::to_string((int)v6nax_BK) + " (MFA_V6_NAX_BK).");
      }
    }

    // Include all tile + flag params in cache key.
    // Repo review 2026-05: tile/config params moved from bit-packed high
    // bits of R/C/qbs/kbs to dedicated key fields (see V6Key comment).
    // F-2 (Change 3): resolve the baked scale (default 1/sqrt(D) when caller
    // passes the <=0 sentinel) and key the pipeline on it.
    const float resolved_scale =
        (params_.scale > 0.0f) ? params_.scale : 1.0f / std::sqrt((float)D);
    V6Key key{D, Hq, Hk, dtype_code, params_.causal,
              R, C, qbs, kbs, vbs, obs,
              BQ, BK, executionSIMDGroups, BD,
              (uint16_t)axis_flags, bypass_tgp,
              use_v6nax, v6nax_BQ, v6nax_BK, v6nax_WM,
              resolved_scale};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v6_mtx);
      auto it = v6_pipelines.find(key);
      if (it != v6_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      std::string src = generate_v6_source(
          D, Hq, Hk, dtype_code, params_.causal, params_.bhnd, (int)R,
          /*use_v6nax_override=*/use_v6nax, /*use_v6nax_explicit=*/true,
          /*scale_override=*/resolved_scale);
      // F-3: V6 forward is PURE NAX — always the matmul2d kernel (no FCs;
      // params via struct buffer).  The simdgroup `v6_nax_compile_with_constants`
      // fallback is removed (broken diverged duplicate; that compile helper is
      // retained only for the diagnostic probe in v6_nax_probe.cpp).
      if (mlx_mfa::getenv_aliased("MFA_V6_DUMP_SOURCE")) {
        fprintf(stderr, "=== V6NAX source for BQ=%d BK=%d BD=%d WM=%d ===\n",
                (int)v6nax_BQ, (int)v6nax_BK, (int)D, (int)v6nax_WM);
        auto pos = src.find("// === lse write");
        if (pos != std::string::npos) {
          fprintf(stderr, "%s\n=== ===\n", src.substr(pos, 800).c_str());
        } else {
          fprintf(stderr, "(lse write marker not found!)\n");
        }
      }
      pipeline = v6nax_compile(src, "attention", mtl_device);
      pipeline = cache_insert_or_release(v6_pipelines, v6_mtx, key, pipeline);
    }

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_output_array(out, 3);
    // F-3: pure NAX — lse always at buffer 5 (buffer 4 holds the V6NAXParams
    // struct via set_bytes).  Per v6nax-backward-decisions.md DC0 — lse is
    // required input infrastructure for the V6NAX backward dQ/dK/dV kernels.
    enc.set_output_array(lse, 5);

    v6nax_dispatch(
        pipeline, &enc,
        (int)N, (int)Nk, (int)Hq, (int)Hk, (int)B, (int)D,
        v6nax_BQ, v6nax_BK, v6nax_WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV6Forward*>(&other);
    // Repo review 2026-05: force_v6nax MUST participate — a force_v6nax=true
    // forward emits natural-log LSE (consumed by V6NAX backward) while the
    // default path emits log2-domain LSE.  Without this term, MLX graph
    // dedup could conflate the two nodes, feeding log2 LSE into a backward
    // expecting natural log (silently wrong gradients).
    return p && p->params_.causal == params_.causal
             && p->params_.bhnd == params_.bhnd
             && p->params_.force_v6nax == params_.force_v6nax
             // F-2: distinct scale → distinct kernel; must not graph-dedup.
             && p->params_.scale == params_.scale;
  }

  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    return {inputs[0].shape(),
            mlx::core::Shape{inputs[0].shape(0), inputs[0].shape(1),
                             inputs[0].shape(2)}};
  }

private:
  Params params_;
};

// Public Python-callable forward.
//
// MLX layout: [B, H, N, D]
// Draw Things kernel layout: [B, N, H, D] (heads interleaved per token)
// We transpose Q/K/V into kernel layout, dispatch, then transpose O back.
std::pair<mlx::core::array, mlx::core::array> v6_nax_forward(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, bool causal, bool force_v6nax,
    float scale) {
  if (q.ndim() != 4) throw std::runtime_error("V6: Q must be 4D [B,H,N,D]");
  int D = q.shape(3);
  if (D != 64 && D != 128) throw std::runtime_error("V6: D must be 64 or 128");

  auto s = mlx::core::default_stream(mlx::core::Device::gpu);

  // Sprint 2A — Layout selection (BHND default since 2026-05-04):
  //   - BNHD legacy path forced via MFA_V6_BNHD_LEGACY=1
  //   - GQA shapes (Hq != Hk) auto-fallback to BNHD (rewriter only handles
  //     non-GQA; LTX2-cross is non-GQA so this rarely triggers)
  const int Hq_from_input = q.shape(1);
  const int Hk_from_input = k.shape(1);
  const bool legacy_opt_in = (std::getenv("MFA_V6_BNHD_LEGACY") != nullptr);
  // Sprint B (v2.30): BHND rewriter now handles GQA (Hq % Hk == 0).
  const bool can_bhnd = (Hq_from_input == Hk_from_input) ||
                        (Hk_from_input > 0 && Hq_from_input % Hk_from_input == 0);
  const bool bhnd = !legacy_opt_in && can_bhnd;
  MFAV6Forward::Params params{causal, bhnd, force_v6nax, scale};

  if (bhnd) {
    // Pass Q/K/V directly in MLX-native [B, H, N, D] layout.
    // The post-gen-rewritten kernel reads this layout natively. Input
    // arrays must be contiguous (no strided views from upstream).
    auto qc = mlx::core::contiguous(q, false, s);
    auto kc = mlx::core::contiguous(k, false, s);
    auto vc = mlx::core::contiguous(v, false, s);
    // Output O in [B, H, N, D] layout — same as input, no return transpose.
    mlx::core::Shape o_shape{qc.shape(0), qc.shape(1), qc.shape(2), qc.shape(3)};
    mlx::core::Shape lse_shape{q.shape(0), q.shape(1), q.shape(2)};
    auto outs = mlx::core::array::make_arrays(
        {o_shape, lse_shape},
        {q.dtype(), mlx::core::float32},
        std::make_shared<MFAV6Forward>(s, params),
        {qc, kc, vc});
    return {outs[0], outs[1]};
  }

  // Legacy / GQA-fallback: transpose [B,H,N,D] -> [B,N,H,D] for Draw Things
  // kernel layout. Used when MFA_V6_BNHD_LEGACY=1 or when shape is GQA.
  auto q_bnhd = mlx::core::transpose(q, std::vector<int>{0, 2, 1, 3}, s);
  auto k_bnhd = mlx::core::transpose(k, std::vector<int>{0, 2, 1, 3}, s);
  auto v_bnhd = mlx::core::transpose(v, std::vector<int>{0, 2, 1, 3}, s);
  auto qc = mlx::core::contiguous(q_bnhd, false, s);
  auto kc = mlx::core::contiguous(k_bnhd, false, s);
  auto vc = mlx::core::contiguous(v_bnhd, false, s);

  // Output O in kernel layout [B, N, Hq, D]; will transpose back at the end.
  mlx::core::Shape o_shape{qc.shape(0), qc.shape(1), qc.shape(2), qc.shape(3)};
  // L is [B, Hq, N] in mlx layout (kernel writes it that way directly).
  mlx::core::Shape lse_shape{q.shape(0), q.shape(1), q.shape(2)};
  auto outs = mlx::core::array::make_arrays(
      {o_shape, lse_shape},
      {q.dtype(), mlx::core::float32},
      std::make_shared<MFAV6Forward>(s, params),
      {qc, kc, vc});
  // Transpose O back: [B, N, H, D] -> [B, H, N, D]
  auto o_bhnd = mlx::core::transpose(outs[0], std::vector<int>{0, 2, 1, 3}, s);
  return {o_bhnd, outs[1]};
}

// =============================================================================
// V6NAX backward dQ — minimum-viable Primitive (Phase 1 Section B).
//
// Per DC13: standalone Primitive for dQ alone.  dK/dV gets a separate
// Primitive in Phase 2.  Combined dispatcher (v6_nax_backward returning
// (dQ, dK, dV)) lands in Phase 2 Section E.
//
// Caching: minimal — single pipeline per (D, dtype) cell, cached in a
// dedicated map distinct from v6_pipelines (different kernel function).
// Phase 1 trades compile-once-per-cell for code simplicity.
// =============================================================================

// -----------------------------------------------------------------------------
// v2.40.x-internal Sprint C (P3-HIGH-01): V6NAX backward pipeline-compile helper.
// Consolidates the ~30-40 LOC of pipeline-cache-miss boilerplate duplicated
// across all 5 V6NAX backward Primitives (MFAV6NAXBwdQuery, MFAV6NAXBwdKeyValue,
// MFAV6NAXBwdDV, MFAV6NAXBwdDK, MFAV6NAXBwdFusedDKDV) into a single helper.
// Pure refactor: produces byte-identical generated source as before; only
// the C++ boilerplate around source-gen + compile is consolidated.
//
// Each caller still owns its own pipeline-cache mutex + map (the cache keys
// differ per Primitive).  The helper handles:
//   1. AttentionOperands precision setup (FP16/BF16 inputs, FP32 S/P/L)
//   2. NAAttentionKernelDescriptor construction (singleOtileMode + useV6NAX)
//   3. Optional source-dump hook (env-gated via MFA_V6BWD*_DUMP_SOURCE +
//      optional MFA_V6BWD*_DUMP_PATH for file output)
//   4. Source string generation via caller-provided lambda
//   5. Final v6nax_compile() invocation
// -----------------------------------------------------------------------------
namespace {

template <typename SourceGenFn>
void* compile_v6nax_backward_pipeline(
    int D, int Hq, int Hk, int dtype_code,
    unsigned short BQ, unsigned short BK, uint16_t WM,
    float scale,
    SourceGenFn source_gen_fn,
    const char* kernel_fn_name,
    void* mtl_device,
    bool isCausal = false,  // v2.50 Phase 4b-complete (Prompt 3): plumbed through
    const char* dump_env_var = nullptr,
    const char* dump_label = nullptr,
    const char* dump_path_env_var = nullptr,
    bool generator_handles_odd_tk = false) {
  // Phase II-6 (campaign 2026-06): paired-MMA TK guard.  Every V6NAX
  // backward generator emits the S-recompute as a PAIRED 16x32x16 MMA
  // (`for (ik = 0; ik < TK; ik += 2)` writing frag_at(iq, ik) AND
  // frag_at(iq, ik+1)).  MPP cooperative matmul2d has no 16x16x16 form
  // (header static_assert: at least one of M,N,K must be 32), so TK
  // (= BK/16) MUST be even.  BK=16 (TK=1) reads 16 K-rows past the
  // tile AND writes one fragment out of bounds — silent gradient
  // corruption that scales exponentially with score magnitude (II-6
  // finding: fused dKdV default BK=16 since v2.39.1 produced dV errors
  // 4x the gradient magnitude at unit-scale inputs and inf at std>=2).
  // Loud failure per Rule 8 — this also guards the MFA_V6BWD*_BK env
  // overrides on every backward Primitive.
  if (BK == 0 || BK % 16 != 0 ||
      (BK % 32 != 0 && !generator_handles_odd_tk)) {
    throw std::runtime_error(
        std::string("V6NAX backward '") + kernel_fn_name +
        "': BK must be a positive multiple of 32 (paired 16x32x16 MMA "
        "requires TK = BK/16 even; MPP has no 16x16x16 cooperative "
        "matmul). Got BK=" + std::to_string((int)BK) +
        ". The v2.39.1 fused-kernel BK=16 default was numerically "
        "invalid and is withdrawn (Phase II-6).  BK%16 configs are "
        "accepted ONLY for generators with the II-8 odd-TK tail.");
  }
  // Build memoryPrecisions (FP16/BF16 inputs, FP32 intermediates).
  GEMMOperandPrecision input_prec = (dtype_code == 1)
      ? GEMMOperandPrecision::BF16
      : GEMMOperandPrecision::FP16;
  AttentionOperands<GEMMOperandPrecision> mp;
  mp[AttentionOperand::Q] = input_prec;
  mp[AttentionOperand::K] = input_prec;
  mp[AttentionOperand::V] = input_prec;
  mp[AttentionOperand::O] = input_prec;
  mp[AttentionOperand::S] = GEMMOperandPrecision::FP32;
  mp[AttentionOperand::P] = GEMMOperandPrecision::FP32;
  mp[AttentionOperand::L] = GEMMOperandPrecision::FP32;

  // Kernel descriptor (12-arg constructor; AttentionKernelType ignored by
  // V6NAX backward source generators which switch on the source-gen method).
  // v2.50 Phase 4b-complete (Prompt 3): isCausal now plumbed through so
  // V6NAXBWD*_CAUSAL macros get the correct compile-time value.  Pre-Prompt-3
  // this was hardcoded to false — a latent bug that silently made my
  // Prompt 2 Phase 4b dQ causal mask a no-op in production.
  simd::ushort3 blockDims =
      simd::make_ushort3(BQ, BK, (unsigned short)D);
  NAAttentionKernelDescriptor desc(
      blockDims, (unsigned short)D, (unsigned short)Hq,
      (unsigned short)Hk, /*executionSIMDGroups=*/WM,
      /*checkCEdge1=*/false, mp,
      AttentionKernelType::forward,  // placeholder; ignored by V6NAX backward
      /*scale=*/scale,
      /*bypassThreadgroupMemory=*/false,
      /*isCausal=*/isCausal, /*masked=*/false);
  desc.singleOtileMode = true;
  desc.useV6NAX = true;

  // Source generation via caller's lambda.
  NAAttentionKernel ker(desc);
  std::string src = source_gen_fn(ker);

  // Optional source-dump hook.
  if (dump_env_var && mlx_mfa::getenv_aliased(dump_env_var)) {
    const char* path = dump_path_env_var ? mlx_mfa::getenv_aliased(dump_path_env_var) : nullptr;
    const char* label = dump_label ? dump_label : kernel_fn_name;
    if (path) {
      FILE* f = fopen(path, "w");
      if (f) {
        fwrite(src.data(), 1, src.size(), f);
        fclose(f);
        fprintf(stderr,
                "[v2.40.x] %s source dumped to %s "
                "(D=%d BQ=%d BK=%d WM=%d, %zu bytes)\n",
                label, path, D, (int)BQ, (int)BK, (int)WM, src.size());
      }
    } else {
      fprintf(stderr,
              "=== %s source (D=%d BQ=%d BK=%d WM=%d) length=%zu bytes ===\n%s\n",
              label, D, (int)BQ, (int)BK, (int)WM, src.size(), src.c_str());
    }
  }

  return v6nax_compile(src, kernel_fn_name, mtl_device);
}

}  // namespace


void v6nax_dispatch_bwd_query(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

struct V6NAXBwdQKey {
  int D;
  int Hq, Hk;
  int dtype_code;  // 0=fp16, 1=bf16
  unsigned short v6nax_BQ, v6nax_BK;
  uint16_t v6nax_WM;
  bool causal;  // v2.50 Phase 4b-complete: separate pipeline cache per causal flag
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V6NAXBWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  auto tie() const { return std::tie(D, Hq, Hk, dtype_code, v6nax_BQ, v6nax_BK, v6nax_WM, causal, scale); }  // Track 6: single declaration
  bool operator==(const V6NAXBwdQKey& o) const { return tie() == o.tie(); }
};
struct V6NAXBwdQKeyHash {
  size_t operator()(const V6NAXBwdQKey& k) const { return mlx_mfa_keys::hash_tie(k.tie()); }
};
namespace {
std::mutex v6nax_bwdq_mtx;
std::unordered_map<V6NAXBwdQKey, void*, V6NAXBwdQKeyHash> v6nax_bwdq_pipelines;
}

class MFAV6NAXBwdQuery : public mlx::core::Primitive {
 public:
  // v2.50 Phase 4b-complete (Prompt 3): causal added to constructor.
  // Default false preserves prior signature; new code should pass causal.
  MFAV6NAXBwdQuery(mlx::core::Stream s, float scale, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), causal_(causal) {}

  const char* name() const override { return "MFAV6NAXBwdQuery"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV6NAXBwdQuery: CPU eval not supported");
  }

  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override {
    // inputs: [Q, K, V, O, L, dO, D]  — D added v2.38.1
    const auto& q   = inputs[0];
    const auto& k   = inputs[1];
    const auto& v   = inputs[2];
    const auto& o   = inputs[3];
    const auto& lse = inputs[4];
    const auto& d_o = inputs[5];
    const auto& d_vec = inputs[6];  // v2.38.1: precomputed rowsum(dO⊙O)
    auto& dq        = outputs[0];

    if (q.ndim() != 4)
      throw std::runtime_error("V6NAX bwd dQ: Q must be 4D [B,H,N,D]");
    const int B  = q.shape(0);
    const int Hq = q.shape(1);
    const int N  = q.shape(2);
    const int D  = q.shape(3);
    const int Hk = k.shape(1);
    const int Nk = k.shape(2);

    if (D != 64 && D != 128)
      throw std::runtime_error("V6NAX bwd dQ: D must be 64 or 128");

    // M5-tuned defaults per DC7 (matches V6NAX forward defaults).
    unsigned short v6nax_BQ = (D == 64) ? 32 : 64;
    unsigned short v6nax_BK = (D == 64) ? 64 : 32;
    uint16_t v6nax_WM = (D == 64) ? 2 : 4;
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWD_BQ"))
      v6nax_BQ = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWD_BK"))
      v6nax_BK = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWD_WM"))
      v6nax_WM = (uint16_t)std::atoi(e);

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V6NAX bwd dQ: only FP16/BF16");

    dq.set_data(mlx::core::allocator::malloc(dq.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V6NAXBwdQKey key{D, Hq, Hk, dtype_code, v6nax_BQ, v6nax_BK, v6nax_WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v6nax_bwdq_mtx);
      auto it = v6nax_bwdq_pipelines.find(key);
      if (it != v6nax_bwdq_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      // v2.40.x-internal Sprint C: consolidated via compile_v6nax_backward_pipeline.
      // v2.50 Phase 4b-complete (Prompt 3): causal_ plumbed through.
      pipeline = compile_v6nax_backward_pipeline(
          D, Hq, Hk, dtype_code, v6nax_BQ, v6nax_BK, v6nax_WM, scale_,
          [](NAAttentionKernel& k) { return k.createV6NAXBackwardQuerySource(); },
          "attention_bwd_q", mtl_device, causal_,
          "MFA_V6BWD_DUMP_SOURCE", "V6NAX bwd dQ", nullptr);
      pipeline = cache_insert_or_release(v6nax_bwdq_pipelines, v6nax_bwdq_mtx, key, pipeline);
    }

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_input_array(o, 3);
    enc.set_input_array(lse, 4);
    enc.set_input_array(d_o, 5);
    enc.set_output_array(dq, 6);
    // params at buffer 7 via enc.set_bytes in dispatcher.
    enc.set_input_array(d_vec, 8);  // v2.38.1: D=rowsum(dO⊙O), [B,Hq,qL] FP32

    v6nax_dispatch_bwd_query(
        pipeline, &enc,
        (int)N, (int)Nk, (int)Hq, (int)Hk, (int)B, (int)D,
        v6nax_BQ, v6nax_BK, v6nax_WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV6NAXBwdQuery*>(&other);
    return p && p->scale_ == scale_ && p->causal_ == causal_;
  }

  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  float scale_;
  bool causal_;  // v2.50 Phase 4b-complete (Prompt 3)
};

// Public Python-callable: V6NAX backward dQ.
//
// Args: Q [B,Hq,N,D], K [B,Hk,Nk,D], V [B,Hk,Nk,D] (T),
//       O [B,Hq,N,D] (T),
//       L [B,Hq,N] (FP32),
//       dO [B,Hq,N,D] (T),
//       scale (float).
//
// Returns: dQ [B,Hq,N,D] (T).
//
// Routing constraint per DC12: callers must ensure V6NAX-forward-eligible
// shapes (D=128 always; D=64 with Nk>8000).  V6NAX backward will produce
// garbage on shapes that routed through legacy v6_nax forward (lse
// convention mismatch).  flash_attention() VJP layer enforces this in
// Phase 2 Section E.
mlx::core::array v6_nax_backward_query(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& o,
    const mlx::core::array& lse, const mlx::core::array& d_o,
    const mlx::core::array& d_vec,  // v2.38.1: precomputed rowsum(dO⊙O)
    float scale, bool causal) {
  if (q.ndim() != 4) throw std::runtime_error("V6NAX bwd dQ: Q must be 4D");
  if (k.shape(1) <= 0 || q.shape(1) % k.shape(1) != 0)
    throw std::runtime_error("V6NAX bwd dQ: Hq must be multiple of Hk");

  auto s = mlx::core::default_stream(mlx::core::Device::gpu);

  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);
  auto oc = mlx::core::contiguous(o, false, s);
  auto lsec = mlx::core::contiguous(lse, false, s);
  auto dOc = mlx::core::contiguous(d_o, false, s);
  auto dvc = mlx::core::contiguous(d_vec, false, s);  // v2.38.1

  mlx::core::Shape dq_shape{qc.shape(0), qc.shape(1), qc.shape(2), qc.shape(3)};
  auto outs = mlx::core::array::make_arrays(
      {dq_shape},
      {q.dtype()},
      std::make_shared<MFAV6NAXBwdQuery>(s, scale, causal),
      {qc, kc, vc, oc, lsec, dOc, dvc});
  return outs[0];
}

// =============================================================================
// V6NAX backward dK/dV — Phase 2 Primitive (single-SG WM=1 design).
// =============================================================================

void v6nax_dispatch_bwd_kv(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

struct V6NAXBwdKVKey {
  int D;
  int Hq, Hk;
  int dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;  // v2.50 Phase 4b-complete: separate pipeline per causal
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V6NAXBWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  auto tie() const { return std::tie(D, Hq, Hk, dtype_code, BQ, BK, WM, causal, scale); }  // Track 6: single declaration
  bool operator==(const V6NAXBwdKVKey& o) const { return tie() == o.tie(); }
};
struct V6NAXBwdKVKeyHash {
  size_t operator()(const V6NAXBwdKVKey& k) const { return mlx_mfa_keys::hash_tie(k.tie()); }
};
namespace {
std::mutex v6nax_bwdkv_mtx;
std::unordered_map<V6NAXBwdKVKey, void*, V6NAXBwdKVKeyHash> v6nax_bwdkv_pipelines;
}

class MFAV6NAXBwdKeyValue : public mlx::core::Primitive {
 public:
  MFAV6NAXBwdKeyValue(mlx::core::Stream s, float scale, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), causal_(causal) {}

  const char* name() const override { return "MFAV6NAXBwdKeyValue"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV6NAXBwdKeyValue: CPU eval not supported");
  }

  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override {
    // inputs: [Q, K, V, O, L, dO, D]  — D added v2.38.1
    // outputs: [dK, dV]
    const auto& q   = inputs[0];
    const auto& k   = inputs[1];
    const auto& v   = inputs[2];
    const auto& o   = inputs[3];
    const auto& lse = inputs[4];
    const auto& d_o = inputs[5];
    const auto& d_vec = inputs[6];  // v2.38.1: precomputed rowsum(dO⊙O)
    auto& dk = outputs[0];
    auto& dv = outputs[1];

    const int B  = q.shape(0);
    const int Hq = q.shape(1);
    const int N  = q.shape(2);
    const int D  = q.shape(3);
    const int Hk = k.shape(1);
    const int Nk = k.shape(2);

    if (D != 64 && D != 128)
      throw std::runtime_error("V6NAX bwd dKdV: D must be 64 or 128");

    // Phase 2 defaults: WM=1 single-SG; BQ=32; BK=(D==64?64:32).
    // Phase 2.O1 (2026-05-13): WM=2 K-row partition was attempted and
    // FALSIFIED empirically (0.77-0.84× regression vs WM=1).  The
    // redundant softmax compute across SGs taxed more than the GEMM
    // partition saved.  Reverted to WM=1.  See v6nax-backward-status.md
    // §"Phase 2.O1 falsified" for next-attempt design (Q-row partition
    // + TGP streaming reduction).
    unsigned short BQ = 32;
    unsigned short BK = (D == 64) ? 64 : 32;
    uint16_t WM = 1;
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDKV_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDKV_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDKV_WM"))
      WM = (uint16_t)std::atoi(e);

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V6NAX bwd dKdV: only FP16/BF16");

    dk.set_data(mlx::core::allocator::malloc(dk.nbytes()));
    dv.set_data(mlx::core::allocator::malloc(dv.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V6NAXBwdKVKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v6nax_bwdkv_mtx);
      auto it = v6nax_bwdkv_pipelines.find(key);
      if (it != v6nax_bwdkv_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      // v2.40.x-internal Sprint C: consolidated via compile_v6nax_backward_pipeline.
      pipeline = compile_v6nax_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV6NAXBackwardKeyValueSource(); },
          "attention_bwd_kv", mtl_device, causal_,
          nullptr, nullptr, nullptr);  // no dump hook in legacy fused
      pipeline = cache_insert_or_release(v6nax_bwdkv_pipelines, v6nax_bwdkv_mtx, key, pipeline);
    }

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_input_array(o, 3);
    enc.set_input_array(lse, 4);
    enc.set_input_array(d_o, 5);
    enc.set_output_array(dk, 6);
    enc.set_output_array(dv, 7);
    // params at buffer 8 via enc.set_bytes in dispatcher.
    enc.set_input_array(d_vec, 9);  // v2.38.1: D=rowsum(dO⊙O), [B,Hq,qL] FP32

    v6nax_dispatch_bwd_kv(pipeline, &enc, (int)N, (int)Nk, (int)Hq, (int)Hk,
                        (int)B, (int)D, BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV6NAXBwdKeyValue*>(&other);
    return p && p->scale_ == scale_ && p->causal_ == causal_;
  }

  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    // dK/dV produced per Q-head: shape [B, Hq, kL, D] each.
    auto qs = inputs[0].shape();
    auto ks = inputs[1].shape();
    mlx::core::Shape dk_shape{qs[0], qs[1], ks[2], qs[3]};
    return {dk_shape, dk_shape};
  }

 private:
  float scale_;
  bool causal_;  // v2.50 Phase 4b-complete (Prompt 3)
};

std::pair<mlx::core::array, mlx::core::array> v6_nax_backward_kv(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& o,
    const mlx::core::array& lse, const mlx::core::array& d_o,
    const mlx::core::array& d_vec,  // v2.38.1: precomputed rowsum(dO⊙O)
    float scale, bool causal) {
  if (q.ndim() != 4) throw std::runtime_error("V6NAX bwd dKdV: Q must be 4D");
  auto s = mlx::core::default_stream(mlx::core::Device::gpu);

  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);
  auto oc = mlx::core::contiguous(o, false, s);
  auto lsec = mlx::core::contiguous(lse, false, s);
  auto dOc = mlx::core::contiguous(d_o, false, s);
  auto dvc = mlx::core::contiguous(d_vec, false, s);  // v2.38.1

  // dK/dV shape: [B, Hq, kL, D] per Q-head (matches SDPA-vjp output).
  mlx::core::Shape dk_shape{qc.shape(0), qc.shape(1), kc.shape(2), qc.shape(3)};
  auto outs = mlx::core::array::make_arrays(
      {dk_shape, dk_shape},
      {q.dtype(), q.dtype()},
      std::make_shared<MFAV6NAXBwdKeyValue>(s, scale, causal),
      {qc, kc, vc, oc, lsec, dOc, dvc});
  return {outs[0], outs[1]};
}

// =============================================================================
// V6NAX backward dV-only — Phase 2.O2 multi-SG Q-row partition Primitive.
// Emits per-SG dV partial to a [B, Hq, WM, kL, D] FP32 intermediate buffer.
// Python wrapper reduces via mx.sum(axis=2) and casts to T.
// =============================================================================

void v6nax_dispatch_bwd_dv(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

struct V6NAXBwdVKey {
  int D;
  int Hq, Hk;
  int dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;  // v2.50 Phase 4b-complete: separate pipeline per causal
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V6NAXBWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  auto tie() const { return std::tie(D, Hq, Hk, dtype_code, BQ, BK, WM, causal, scale); }  // Track 6: single declaration
  bool operator==(const V6NAXBwdVKey& o) const { return tie() == o.tie(); }
};
struct V6NAXBwdVKeyHash {
  size_t operator()(const V6NAXBwdVKey& k) const { return mlx_mfa_keys::hash_tie(k.tie()); }
};
namespace {
std::mutex v6nax_bwdv_mtx;
std::unordered_map<V6NAXBwdVKey, void*, V6NAXBwdVKeyHash> v6nax_bwdv_pipelines;
}

class MFAV6NAXBwdDV : public mlx::core::Primitive {
 public:
  MFAV6NAXBwdDV(mlx::core::Stream s, float scale, uint16_t wm, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), wm_(wm), causal_(causal) {}

  const char* name() const override { return "MFAV6NAXBwdDV"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV6NAXBwdDV: CPU eval not supported");
  }

  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override {
    // inputs: [Q, K, V, L, dO]; outputs: [dV_partials [B, Hq, WM, kL, D] FP32]
    const auto& q   = inputs[0];
    const auto& k   = inputs[1];
    const auto& v   = inputs[2];
    const auto& lse = inputs[3];
    const auto& d_o = inputs[4];
    auto& dvp = outputs[0];

    const int B  = q.shape(0);
    const int Hq = q.shape(1);
    const int N  = q.shape(2);
    const int D  = q.shape(3);
    const int Hk = k.shape(1);
    const int Nk = k.shape(2);

    if (D != 64 && D != 128)
      throw std::runtime_error("V6NAX bwd dV: D must be 64 or 128");

    // Phase 2.O2 defaults: WM=4 Q-row partition. BQ = WM*16 = 64.
    unsigned short BQ = 64;
    unsigned short BK = 32;
    uint16_t WM = wm_;
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDV_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDV_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDV_WM"))
      WM = (uint16_t)std::atoi(e);

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V6NAX bwd dV: only FP16/BF16");

    dvp.set_data(mlx::core::allocator::malloc(dvp.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V6NAXBwdVKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v6nax_bwdv_mtx);
      auto it = v6nax_bwdv_pipelines.find(key);
      if (it != v6nax_bwdv_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      // v2.40.x-internal Sprint C: consolidated via compile_v6nax_backward_pipeline.
      pipeline = compile_v6nax_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV6NAXBackwardDVSource(); },
          "attention_bwd_dv", mtl_device, causal_,
          nullptr, nullptr, nullptr);  // no dump hook in split-dV
      pipeline = cache_insert_or_release(v6nax_bwdv_pipelines, v6nax_bwdv_mtx, key, pipeline);
    }

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_input_array(lse, 3);
    enc.set_input_array(d_o, 4);
    enc.set_output_array(dvp, 5);

    v6nax_dispatch_bwd_dv(pipeline, &enc, (int)N, (int)Nk, (int)Hq, (int)Hk,
                        (int)B, (int)D, BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV6NAXBwdDV*>(&other);
    return p && p->scale_ == scale_ && p->wm_ == wm_ && p->causal_ == causal_;
  }

  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    auto qs = inputs[0].shape();
    auto ks = inputs[1].shape();
    // dV_partials: [B, Hq, WM, kL, D] FP32
    mlx::core::Shape s{qs[0], qs[1], (int)wm_, ks[2], qs[3]};
    return {s};
  }

 private:
  float scale_;
  uint16_t wm_;
  bool causal_;
};

// ============================================================================
// V6NAX backward dV SPARSE Primitive — Prompt 5b Section A PoC.
// Identical to MFAV6NAXBwdDV but accepts block_mask input and routes to the
// sparse source generator.  Cache key extended with is_sparse flag.
// ============================================================================

void v6nax_dispatch_bwd_dv_sparse(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

struct V6NAXBwdVSparseKey {
  int D;
  int Hq, Hk;
  int dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V6NAXBWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  // is_sparse implicit (this struct only used for sparse kernels) but
  // included for future-proofing if a single cache holds both variants.
  auto tie() const { return std::tie(D, Hq, Hk, dtype_code, BQ, BK, WM, causal, scale); }  // Track 6: single declaration
  bool operator==(const V6NAXBwdVSparseKey& o) const { return tie() == o.tie(); }
};
struct V6NAXBwdVSparseKeyHash {
  size_t operator()(const V6NAXBwdVSparseKey& k) const { return mlx_mfa_keys::hash_tie(k.tie()); }
};
namespace {
std::mutex v6nax_bwdv_sparse_mtx;
std::unordered_map<V6NAXBwdVSparseKey, void*, V6NAXBwdVSparseKeyHash>
    v6nax_bwdv_sparse_pipelines;
}

class MFAV6NAXBwdDVSparse : public mlx::core::Primitive {
 public:
  MFAV6NAXBwdDVSparse(mlx::core::Stream s, float scale, uint16_t wm, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), wm_(wm), causal_(causal) {}

  const char* name() const override { return "MFAV6NAXBwdDVSparse"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV6NAXBwdDVSparse: CPU eval not supported");
  }

  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override {
    // inputs: [Q, K, V, L, dO, block_mask]; outputs: [dV_partials FP32]
    const auto& q   = inputs[0];
    const auto& k   = inputs[1];
    const auto& v   = inputs[2];
    const auto& lse = inputs[3];
    const auto& d_o = inputs[4];
    const auto& block_mask = inputs[5];
    auto& dvp = outputs[0];

    const int B  = q.shape(0);
    const int Hq = q.shape(1);
    const int N  = q.shape(2);
    const int D  = q.shape(3);
    const int Hk = k.shape(1);
    const int Nk = k.shape(2);

    if (D != 64 && D != 128)
      throw std::runtime_error("V6NAX bwd dV sparse: D must be 64 or 128");
    if (block_mask.ndim() != 2)
      throw std::runtime_error("V6NAX bwd dV sparse: block_mask must be 2-D [NQ, NK] (Section A PoC)");

    unsigned short BQ = 64;
    unsigned short BK = 32;
    uint16_t WM = wm_;
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDV_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDV_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDV_WM"))
      WM = (uint16_t)std::atoi(e);

    // v2.50 Prompt 5f Phase A — KD-1 fix: enforce mask shape match.
    // Python orchestrator (`_convert_mask_for_v6nax_bwd_kernel` in attention.py)
    // converts BT-block masks to this kernel's tile geometry before dispatch.
    // Runtime check guards against future regressions or direct callers.
    {
      const int expected_NQ = (N + BQ - 1) / BQ;
      const int expected_NK = (Nk + BK - 1) / BK;
      const int mask_NQ = block_mask.shape(-2);
      const int mask_NK = block_mask.shape(-1);
      if (mask_NQ != expected_NQ || mask_NK != expected_NK) {
        std::ostringstream oss;
        oss << "V6NAX bwd dV sparse: block_mask shape ["
            << mask_NQ << ", " << mask_NK << "] does not match expected ["
            << expected_NQ << ", " << expected_NK << "] for tile geometry "
            << "(BQ=" << BQ << ", BK=" << BK << ") at qL=" << N
            << " kL=" << Nk << ".  See _convert_mask_for_v6nax_bwd_kernel "
            << "in mlx_mfa/attention.py (KD-1 resolution).";
        throw std::runtime_error(oss.str());
      }
    }

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V6NAX bwd dV sparse: only FP16/BF16");

    dvp.set_data(mlx::core::allocator::malloc(dvp.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V6NAXBwdVSparseKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v6nax_bwdv_sparse_mtx);
      auto it = v6nax_bwdv_sparse_pipelines.find(key);
      if (it != v6nax_bwdv_sparse_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      pipeline = compile_v6nax_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV6NAXBackwardDVSparseSource(); },
          "attention_bwd_dv_sparse", mtl_device, causal_,
          nullptr, nullptr, nullptr);
      pipeline = cache_insert_or_release(v6nax_bwdv_sparse_pipelines, v6nax_bwdv_sparse_mtx, key, pipeline);
    }

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_input_array(lse, 3);
    enc.set_input_array(d_o, 4);
    enc.set_output_array(dvp, 5);
    // buffer(6) = params set inside dispatch helper via enc.set_bytes
    enc.set_input_array(block_mask, 7);

    // Phase II-14: the sparse generator's active-qb list is a fixed
    // 1024-entry threadgroup array; reject qL beyond it loudly.
    if ((N + BQ - 1) / BQ > 1024) {
      throw std::runtime_error(
          "V6NAX sparse dV: qL/BQ exceeds the 1024-entry active-list "
          "capacity (qL=" + std::to_string((int)N) +
          ", BQ=" + std::to_string((int)BQ) + ") — II-14 restructure.");
    }
    v6nax_dispatch_bwd_dv_sparse(pipeline, &enc, (int)N, (int)Nk,
                               (int)Hq, (int)Hk, (int)B, (int)D, BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV6NAXBwdDVSparse*>(&other);
    return p && p->scale_ == scale_ && p->wm_ == wm_ && p->causal_ == causal_;
  }

  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    auto qs = inputs[0].shape();
    auto ks = inputs[1].shape();
    mlx::core::Shape s{qs[0], qs[1], (int)wm_, ks[2], qs[3]};
    return {s};
  }

 private:
  float scale_;
  uint16_t wm_;
  bool causal_;
};

// Returns dV_partials FP32 [B, Hq, WM, kL, D] from sparse kernel.
// Block mask must be 2-D [NQ, NK] bool.  Caller reduces via mx.sum(axis=2)
// + cast to T.
mlx::core::array v6_nax_backward_dv_sparse_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& lse,
    const mlx::core::array& d_o, const mlx::core::array& block_mask,
    float scale, int wm, bool causal) {
  if (q.ndim() != 4) throw std::runtime_error("V6NAX bwd dV sparse: Q must be 4D");
  if (block_mask.ndim() != 2)
    throw std::runtime_error("V6NAX bwd dV sparse: block_mask must be 2-D (Section A PoC)");
  auto s = mlx::core::default_stream(mlx::core::Device::gpu);

  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);
  auto lsec = mlx::core::contiguous(lse, false, s);
  auto dOc = mlx::core::contiguous(d_o, false, s);
  auto bmc = mlx::core::contiguous(block_mask, false, s);

  mlx::core::Shape dvp_shape{qc.shape(0), qc.shape(1), wm,
                              kc.shape(2), qc.shape(3)};
  auto outs = mlx::core::array::make_arrays(
      {dvp_shape},
      {mlx::core::float32},
      std::make_shared<MFAV6NAXBwdDVSparse>(s, scale, (uint16_t)wm, causal),
      {qc, kc, vc, lsec, dOc, bmc});
  return outs[0];
}


// Returns dV_partials FP32 [B, Hq, WM, kL, D].  Caller must mx.sum(axis=2)
// and cast to T to get final dV.
mlx::core::array v6_nax_backward_dv_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& lse,
    const mlx::core::array& d_o, float scale, int wm, bool causal) {
  if (q.ndim() != 4) throw std::runtime_error("V6NAX bwd dV: Q must be 4D");
  auto s = mlx::core::default_stream(mlx::core::Device::gpu);

  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);
  auto lsec = mlx::core::contiguous(lse, false, s);
  auto dOc = mlx::core::contiguous(d_o, false, s);

  mlx::core::Shape dvp_shape{qc.shape(0), qc.shape(1), wm,
                              kc.shape(2), qc.shape(3)};
  auto outs = mlx::core::array::make_arrays(
      {dvp_shape},
      {mlx::core::float32},
      std::make_shared<MFAV6NAXBwdDV>(s, scale, (uint16_t)wm, causal),
      {qc, kc, vc, lsec, dOc});
  return outs[0];
}

// =============================================================================
// V6NAX backward dK-only — Phase 2.O2 multi-SG Q-row partition Primitive.
// Sister to MFAV6NAXBwdDV.  Emits per-SG dK partial to dK_partials [B, Hq, WM,
// kL, D] FP32.  Python wrapper reduces via mx.sum(axis=2) and casts to T.
// =============================================================================

void v6nax_dispatch_bwd_dk(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

struct V6NAXBwdKKey {
  int D;
  int Hq, Hk;
  int dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;  // v2.50 Phase 4b-complete: separate pipeline per causal
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V6NAXBWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  auto tie() const { return std::tie(D, Hq, Hk, dtype_code, BQ, BK, WM, causal, scale); }  // Track 6: single declaration
  bool operator==(const V6NAXBwdKKey& o) const { return tie() == o.tie(); }
};
struct V6NAXBwdKKeyHash {
  size_t operator()(const V6NAXBwdKKey& k) const { return mlx_mfa_keys::hash_tie(k.tie()); }
};
namespace {
std::mutex v6nax_bwdk_mtx;
std::unordered_map<V6NAXBwdKKey, void*, V6NAXBwdKKeyHash> v6nax_bwdk_pipelines;
}

class MFAV6NAXBwdDK : public mlx::core::Primitive {
 public:
  MFAV6NAXBwdDK(mlx::core::Stream s, float scale, uint16_t wm, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), wm_(wm), causal_(causal) {}

  const char* name() const override { return "MFAV6NAXBwdDK"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV6NAXBwdDK: CPU eval not supported");
  }

  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override {
    // inputs: [Q, K, V, O, L, dO, D]  — D added v2.38.1
    // outputs: [dK_partials [B, Hq, WM, kL, D] FP32]
    const auto& q   = inputs[0];
    const auto& k   = inputs[1];
    const auto& v   = inputs[2];
    const auto& o   = inputs[3];
    const auto& lse = inputs[4];
    const auto& d_o = inputs[5];
    const auto& d_vec = inputs[6];  // v2.38.1: precomputed rowsum(dO⊙O)
    auto& dkp = outputs[0];

    const int B  = q.shape(0);
    const int Hq = q.shape(1);
    const int N  = q.shape(2);
    const int D  = q.shape(3);
    const int Hk = k.shape(1);
    const int Nk = k.shape(2);

    if (D != 64 && D != 128)
      throw std::runtime_error("V6NAX bwd dK: D must be 64 or 128");

    unsigned short BQ = 64;
    unsigned short BK = 32;
    uint16_t WM = wm_;
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDK_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDK_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDK_WM"))
      WM = (uint16_t)std::atoi(e);

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V6NAX bwd dK: only FP16/BF16");

    dkp.set_data(mlx::core::allocator::malloc(dkp.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V6NAXBwdKKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v6nax_bwdk_mtx);
      auto it = v6nax_bwdk_pipelines.find(key);
      if (it != v6nax_bwdk_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      // v2.40.x-internal Sprint C: consolidated via compile_v6nax_backward_pipeline.
      pipeline = compile_v6nax_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV6NAXBackwardDKSource(); },
          "attention_bwd_dk", mtl_device, causal_,
          nullptr, nullptr, nullptr);  // no dump hook in split-dK
      pipeline = cache_insert_or_release(v6nax_bwdk_pipelines, v6nax_bwdk_mtx, key, pipeline);
    }

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_input_array(o, 3);
    enc.set_input_array(lse, 4);
    enc.set_input_array(d_o, 5);
    enc.set_output_array(dkp, 6);
    // params at buffer 7 via enc.set_bytes in dispatcher.
    enc.set_input_array(d_vec, 8);  // v2.38.1: D=rowsum(dO⊙O), [B,Hq,qL] FP32


    v6nax_dispatch_bwd_dk(pipeline, &enc, (int)N, (int)Nk, (int)Hq, (int)Hk,
                        (int)B, (int)D, BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV6NAXBwdDK*>(&other);
    return p && p->scale_ == scale_ && p->wm_ == wm_ && p->causal_ == causal_;
  }

  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    auto qs = inputs[0].shape();
    auto ks = inputs[1].shape();
    mlx::core::Shape s{qs[0], qs[1], (int)wm_, ks[2], qs[3]};
    return {s};
  }

 private:
  float scale_;
  uint16_t wm_;
  bool causal_;
};

mlx::core::array v6_nax_backward_dk_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& o,
    const mlx::core::array& lse, const mlx::core::array& d_o,
    const mlx::core::array& d_vec,  // v2.38.1: precomputed rowsum(dO⊙O)
    float scale, int wm, bool causal) {
  if (q.ndim() != 4) throw std::runtime_error("V6NAX bwd dK: Q must be 4D");
  auto s = mlx::core::default_stream(mlx::core::Device::gpu);

  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);
  auto oc = mlx::core::contiguous(o, false, s);
  auto lsec = mlx::core::contiguous(lse, false, s);
  auto dOc = mlx::core::contiguous(d_o, false, s);
  auto dvc = mlx::core::contiguous(d_vec, false, s);  // v2.38.1

  mlx::core::Shape dkp_shape{qc.shape(0), qc.shape(1), wm,
                              kc.shape(2), qc.shape(3)};
  auto outs = mlx::core::array::make_arrays(
      {dkp_shape},
      {mlx::core::float32},
      std::make_shared<MFAV6NAXBwdDK>(s, scale, (uint16_t)wm, causal),
      {qc, kc, vc, oc, lsec, dOc, dvc});
  return outs[0];
}


// =============================================================================
// V6NAX backward FUSED dK+dV Primitive (Sprint v2.39.0 Phase C.1.a, Option γ).
// Combines split-dV + split-dK into a single kernel dispatch.  Per-SG-slot
// outputs to dK_partials + dV_partials [B, Hq, WM, kL, D] FP32 each;
// caller reduces via mx.sum(axis=2) and casts to T.
// =============================================================================

void v6nax_dispatch_bwd_fused_dkdv(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

struct V6NAXBwdFusedKey {
  int D;
  int Hq, Hk;
  int dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;  // v2.50 Phase 4b-complete: separate pipeline per causal
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V6NAXBWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  auto tie() const { return std::tie(D, Hq, Hk, dtype_code, BQ, BK, WM, causal, scale); }  // Track 6: single declaration
  bool operator==(const V6NAXBwdFusedKey& o) const { return tie() == o.tie(); }
};
struct V6NAXBwdFusedKeyHash {
  size_t operator()(const V6NAXBwdFusedKey& k) const { return mlx_mfa_keys::hash_tie(k.tie()); }
};
namespace {
std::mutex v6nax_bwd_fused_mtx;
std::unordered_map<V6NAXBwdFusedKey, void*, V6NAXBwdFusedKeyHash> v6nax_bwd_fused_pipelines;
}

class MFAV6NAXBwdFusedDKDV : public mlx::core::Primitive {
 public:
  MFAV6NAXBwdFusedDKDV(mlx::core::Stream s, float scale, uint16_t wm, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), wm_(wm), causal_(causal) {}

  const char* name() const override { return "MFAV6NAXBwdFusedDKDV"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV6NAXBwdFusedDKDV: CPU eval not supported");
  }

  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override {
    // inputs: [Q, K, V, L, dO, D]  — note: no O input (fused kernel uses
    // precomputed D from v2.38.1 and does not need O for D recompute).
    // outputs: [dK_partials, dV_partials] both [B, Hq, WM, kL, D] FP32.
    const auto& q   = inputs[0];
    const auto& k   = inputs[1];
    const auto& v   = inputs[2];
    const auto& lse = inputs[3];
    const auto& d_o = inputs[4];
    const auto& d_vec = inputs[5];
    auto& dkp = outputs[0];
    auto& dvp = outputs[1];

    const int B  = q.shape(0);
    const int Hq = q.shape(1);
    const int N  = q.shape(2);
    const int D  = q.shape(3);
    const int Hk = k.shape(1);
    const int Nk = k.shape(2);

    // Phase C.1.a (v2.39.0/.1): D=64 only.
    // v2.40.0-internal (Sprint B): D=128 added.  Source generator was
    // already D-parameterized; gate lifted after empirical bench at
    // D=128 BK=16 confirmed no register-pressure regression vs split.
    // See docs/v6-nax/v40-0-internal-decisions.md.
    if (D != 64 && D != 128)
      throw std::runtime_error(
          "V6NAX bwd fused dKdV: D must be 64 or 128 (Phase C.1.a + C.1.b)");

    // Phase II-6 (campaign 2026-06): default BK restored to 32.
    //
    // HISTORY + CORRECTION: v2.39.1 lowered the default to BK=16 to fix
    // the v2.39.0 register-spill regression (H1) and measured
    // "1.01-1.12x vs split".  That configuration was NUMERICALLY
    // INVALID: every backward generator emits the S-recompute as a
    // paired 16x32x16 MMA (ik += 2 over TK), so BK=16 (TK=1) read 16
    // K-rows past the tile and wrote one S-fragment out of bounds —
    // silent gradient corruption scaling exponentially with score
    // magnitude (invisible at the test fixtures' 0.1-scale inputs;
    // dV errors 4x gradient magnitude at unit scale; inf at std>=2).
    // The v2.39.1 perf number was therefore measured on corrupt math
    // and is WITHDRAWN.  Pattern #9 instance: Primitive changed the
    // dispatch constant; generator's even-TK assumption not re-audited.
    //
    // BK=32 is the minimum valid block (TK=2).  At D=64 this is the
    // v2.39.0 spill-regression config — which is why `auto` now routes
    // to the split kernels (attention.py `_v6nax_backward_vjp`); fused
    // remains reachable via MFA_V6_BWD_KERNEL=fused for benchmarking.
    // A true TK=1 generator variant (scratch second fragment) is a
    // Marco-gated future item.  compile_v6nax_backward_pipeline() now
    // rejects BK % 32 != 0 loudly for ALL backward Primitives.
    unsigned short BQ = 64;
    unsigned short BK = 32;
    uint16_t WM = wm_;
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDF_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDF_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDF_WM"))
      WM = (uint16_t)std::atoi(e);

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V6NAX bwd fused dKdV: only FP16/BF16");

    dkp.set_data(mlx::core::allocator::malloc(dkp.nbytes()));
    dvp.set_data(mlx::core::allocator::malloc(dvp.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V6NAXBwdFusedKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v6nax_bwd_fused_mtx);
      auto it = v6nax_bwd_fused_pipelines.find(key);
      if (it != v6nax_bwd_fused_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      // v2.40.x-internal Sprint C: consolidated via compile_v6nax_backward_pipeline.
      // Fused path keeps source-dump hook (set MFA_V6BWDF_DUMP_SOURCE=1;
      // optional MFA_V6BWDF_DUMP_PATH=<file> for file output).
      pipeline = compile_v6nax_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV6NAXBackwardFusedDKDVSource(); },
          "attention_bwd_fused_dkdv", mtl_device, causal_,
          "MFA_V6BWDF_DUMP_SOURCE", "V6NAX bwd fused dKdV", "MFA_V6BWDF_DUMP_PATH",
          /*generator_handles_odd_tk=*/true);  // II-8 item 3 tail
      pipeline = cache_insert_or_release(v6nax_bwd_fused_pipelines, v6nax_bwd_fused_mtx, key, pipeline);
    }

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_input_array(lse, 3);
    enc.set_input_array(d_o, 4);
    enc.set_output_array(dkp, 5);
    enc.set_output_array(dvp, 6);
    // params at buffer 7 via enc.set_bytes in dispatcher.
    enc.set_input_array(d_vec, 8);

    v6nax_dispatch_bwd_fused_dkdv(pipeline, &enc, (int)N, (int)Nk, (int)Hq, (int)Hk,
                                 (int)B, (int)D, BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV6NAXBwdFusedDKDV*>(&other);
    return p && p->scale_ == scale_ && p->wm_ == wm_ && p->causal_ == causal_;
  }

  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    auto qs = inputs[0].shape();
    auto ks = inputs[1].shape();
    mlx::core::Shape s{qs[0], qs[1], (int)wm_, ks[2], qs[3]};
    return {s, s};  // dKp + dVp same shape
  }

 private:
  float scale_;
  uint16_t wm_;
  bool causal_;
};

std::pair<mlx::core::array, mlx::core::array> v6_nax_backward_fused_dkdv_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& lse,
    const mlx::core::array& d_o,
    const mlx::core::array& d_vec,
    float scale, int wm, bool causal) {
  if (q.ndim() != 4)
    throw std::runtime_error("V6NAX bwd fused dKdV: Q must be 4D");
  if (q.shape(3) != 64 && q.shape(3) != 128)
    throw std::runtime_error(
        "V6NAX bwd fused dKdV: D must be 64 or 128 (Phase C.1.a + C.1.b)");

  auto s = mlx::core::default_stream(mlx::core::Device::gpu);

  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);
  auto lsec = mlx::core::contiguous(lse, false, s);
  auto dOc = mlx::core::contiguous(d_o, false, s);
  auto dvc = mlx::core::contiguous(d_vec, false, s);

  mlx::core::Shape partial_shape{qc.shape(0), qc.shape(1), wm,
                                  kc.shape(2), qc.shape(3)};
  auto outs = mlx::core::array::make_arrays(
      {partial_shape, partial_shape},
      {mlx::core::float32, mlx::core::float32},
      std::make_shared<MFAV6NAXBwdFusedDKDV>(s, scale, (uint16_t)wm, causal),
      {qc, kc, vc, lsec, dOc, dvc});
  return {outs[0], outs[1]};  // dK_partials, dV_partials
}


// =============================================================================
// v2.50 Prompt 5d Section A — Sparse plumbing for dQ, dK split, fused dKdV.
// =============================================================================

void v6nax_dispatch_bwd_query_sparse(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);
void v6nax_dispatch_bwd_dk_sparse(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);
void v6nax_dispatch_bwd_fused_dkdv_sparse(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

// ─────────────────────────────────────────────────────────────────────
// MFAV6NAXBwdQuerySparse
// ─────────────────────────────────────────────────────────────────────
struct V6NAXBwdQSparseKey {
  int D, Hq, Hk, dtype_code;
  unsigned short v6nax_BQ, v6nax_BK;
  uint16_t v6nax_WM;
  bool causal;
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V6NAXBWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  auto tie() const { return std::tie(D, Hq, Hk, dtype_code, v6nax_BQ, v6nax_BK, v6nax_WM, causal, scale); }  // Track 6: single declaration
  bool operator==(const V6NAXBwdQSparseKey& o) const { return tie() == o.tie(); }
};
struct V6NAXBwdQSparseKeyHash {
  size_t operator()(const V6NAXBwdQSparseKey& k) const { return mlx_mfa_keys::hash_tie(k.tie()); }
};
namespace {
std::mutex v6nax_bwdq_sparse_mtx;
std::unordered_map<V6NAXBwdQSparseKey, void*, V6NAXBwdQSparseKeyHash>
    v6nax_bwdq_sparse_pipelines;
}

class MFAV6NAXBwdQuerySparse : public mlx::core::Primitive {
 public:
  MFAV6NAXBwdQuerySparse(mlx::core::Stream s, float scale, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), causal_(causal) {}

  const char* name() const override { return "MFAV6NAXBwdQuerySparse"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV6NAXBwdQuerySparse: CPU eval not supported");
  }

  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override {
    const auto& q   = inputs[0];
    const auto& k   = inputs[1];
    const auto& v   = inputs[2];
    const auto& o   = inputs[3];
    const auto& lse = inputs[4];
    const auto& d_o = inputs[5];
    const auto& d_vec = inputs[6];
    const auto& block_mask = inputs[7];
    auto& dq  = outputs[0];

    if (q.ndim() != 4)
      throw std::runtime_error("V6NAX bwd dQ sparse: Q must be 4D");
    const int B  = q.shape(0);
    const int Hq = q.shape(1);
    const int N  = q.shape(2);
    const int D  = q.shape(3);
    const int Hk = k.shape(1);
    const int Nk = k.shape(2);
    if (D != 64 && D != 128)
      throw std::runtime_error("V6NAX bwd dQ sparse: D must be 64 or 128");
    if (block_mask.ndim() != 2)
      throw std::runtime_error("V6NAX bwd dQ sparse: block_mask must be 2-D [NQ, NK]");

    unsigned short v6nax_BQ = (D == 64) ? 32 : 64;
    unsigned short v6nax_BK = (D == 64) ? 64 : 32;
    uint16_t v6nax_WM = (D == 64) ? 2 : 4;
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWD_BQ"))
      v6nax_BQ = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWD_BK"))
      v6nax_BK = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWD_WM"))
      v6nax_WM = (uint16_t)std::atoi(e);

    // v2.50 Prompt 5f Phase A — KD-1 fix: enforce mask shape match.
    {
      const int expected_NQ = (N + v6nax_BQ - 1) / v6nax_BQ;
      const int expected_NK = (Nk + v6nax_BK - 1) / v6nax_BK;
      const int mask_NQ = block_mask.shape(-2);
      const int mask_NK = block_mask.shape(-1);
      if (mask_NQ != expected_NQ || mask_NK != expected_NK) {
        std::ostringstream oss;
        oss << "V6NAX bwd dQ sparse: block_mask shape ["
            << mask_NQ << ", " << mask_NK << "] does not match expected ["
            << expected_NQ << ", " << expected_NK << "] for tile geometry "
            << "(BQ=" << v6nax_BQ << ", BK=" << v6nax_BK << ") at qL=" << N
            << " kL=" << Nk << ".  See _convert_mask_for_v6nax_bwd_kernel "
            << "in mlx_mfa/attention.py (KD-1 resolution).";
        throw std::runtime_error(oss.str());
      }
    }

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V6NAX bwd dQ sparse: only FP16/BF16");

    dq.set_data(mlx::core::allocator::malloc(dq.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V6NAXBwdQSparseKey key{D, Hq, Hk, dtype_code, v6nax_BQ, v6nax_BK, v6nax_WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v6nax_bwdq_sparse_mtx);
      auto it = v6nax_bwdq_sparse_pipelines.find(key);
      if (it != v6nax_bwdq_sparse_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      pipeline = compile_v6nax_backward_pipeline(
          D, Hq, Hk, dtype_code, v6nax_BQ, v6nax_BK, v6nax_WM, scale_,
          [](NAAttentionKernel& k) { return k.createV6NAXBackwardQuerySparseSource(); },
          "attention_bwd_q_sparse", mtl_device, causal_,
          "MFA_V6BWD_DUMP_SOURCE", "V6NAX bwd dQ sparse", nullptr);
      pipeline = cache_insert_or_release(v6nax_bwdq_sparse_pipelines, v6nax_bwdq_sparse_mtx, key, pipeline);
    }

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_input_array(o, 3);
    enc.set_input_array(lse, 4);
    enc.set_input_array(d_o, 5);
    enc.set_output_array(dq, 6);
    enc.set_input_array(d_vec, 8);
    enc.set_input_array(block_mask, 9);

    // Phase II-14: the sparse generator's active-kb list is a fixed
    // 1024-entry threadgroup array; reject kL beyond it loudly.
    if ((Nk + v6nax_BK - 1) / v6nax_BK > 1024) {
      throw std::runtime_error(
          "V6NAX sparse dQ: kL/BK exceeds the 1024-entry active-list "
          "capacity (kL=" + std::to_string((int)Nk) +
          ", BK=" + std::to_string((int)v6nax_BK) + ") — II-14 restructure.");
    }
    v6nax_dispatch_bwd_query_sparse(
        pipeline, &enc, (int)N, (int)Nk, (int)Hq, (int)Hk,
        (int)B, (int)D, v6nax_BQ, v6nax_BK, v6nax_WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV6NAXBwdQuerySparse*>(&other);
    return p && p->scale_ == scale_ && p->causal_ == causal_;
  }

  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  float scale_;
  bool causal_;
};

mlx::core::array v6_nax_backward_query_sparse_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& o,
    const mlx::core::array& lse, const mlx::core::array& d_o,
    const mlx::core::array& d_vec,
    const mlx::core::array& block_mask,
    float scale, bool causal) {
  if (q.ndim() != 4) throw std::runtime_error("V6NAX bwd dQ sparse: Q must be 4D");
  if (block_mask.ndim() != 2)
    throw std::runtime_error("V6NAX bwd dQ sparse: block_mask must be 2-D");
  auto s = mlx::core::default_stream(mlx::core::Device::gpu);

  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);
  auto oc = mlx::core::contiguous(o, false, s);
  auto lsec = mlx::core::contiguous(lse, false, s);
  auto dOc = mlx::core::contiguous(d_o, false, s);
  auto dvc = mlx::core::contiguous(d_vec, false, s);
  auto bmc = mlx::core::contiguous(block_mask, false, s);

  auto outs = mlx::core::array::make_arrays(
      {qc.shape()},
      {qc.dtype()},
      std::make_shared<MFAV6NAXBwdQuerySparse>(s, scale, causal),
      {qc, kc, vc, oc, lsec, dOc, dvc, bmc});
  return outs[0];
}


// ─────────────────────────────────────────────────────────────────────
// MFAV6NAXBwdDKSparse — dK split sparse
// ─────────────────────────────────────────────────────────────────────
struct V6NAXBwdKSparseKey {
  int D, Hq, Hk, dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V6NAXBWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  auto tie() const { return std::tie(D, Hq, Hk, dtype_code, BQ, BK, WM, causal, scale); }  // Track 6: single declaration
  bool operator==(const V6NAXBwdKSparseKey& o) const { return tie() == o.tie(); }
};
struct V6NAXBwdKSparseKeyHash {
  size_t operator()(const V6NAXBwdKSparseKey& k) const { return mlx_mfa_keys::hash_tie(k.tie()); }
};
namespace {
std::mutex v6nax_bwdk_sparse_mtx;
std::unordered_map<V6NAXBwdKSparseKey, void*, V6NAXBwdKSparseKeyHash>
    v6nax_bwdk_sparse_pipelines;
}

class MFAV6NAXBwdDKSparse : public mlx::core::Primitive {
 public:
  MFAV6NAXBwdDKSparse(mlx::core::Stream s, float scale, uint16_t wm, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), wm_(wm), causal_(causal) {}

  const char* name() const override { return "MFAV6NAXBwdDKSparse"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV6NAXBwdDKSparse: CPU eval not supported");
  }

  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override {
    const auto& q   = inputs[0];
    const auto& k   = inputs[1];
    const auto& v   = inputs[2];
    const auto& o   = inputs[3];
    const auto& lse = inputs[4];
    const auto& d_o = inputs[5];
    const auto& d_vec = inputs[6];
    const auto& block_mask = inputs[7];
    auto& dkp = outputs[0];

    const int B  = q.shape(0);
    const int Hq = q.shape(1);
    const int N  = q.shape(2);
    const int D  = q.shape(3);
    const int Hk = k.shape(1);
    const int Nk = k.shape(2);

    if (D != 64 && D != 128)
      throw std::runtime_error("V6NAX bwd dK sparse: D must be 64 or 128");
    if (block_mask.ndim() != 2)
      throw std::runtime_error("V6NAX bwd dK sparse: block_mask must be 2-D");

    unsigned short BQ = 64;
    unsigned short BK = 32;
    uint16_t WM = wm_;
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDK_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDK_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDK_WM"))
      WM = (uint16_t)std::atoi(e);
    // v2.50 Prompt 5f Phase A — KD-1 fix: enforce mask shape match.
    {
      const int expected_NQ = (N + BQ - 1) / BQ;
      const int expected_NK = (Nk + BK - 1) / BK;
      const int mask_NQ = block_mask.shape(-2);
      const int mask_NK = block_mask.shape(-1);
      if (mask_NQ != expected_NQ || mask_NK != expected_NK) {
        std::ostringstream oss;
        oss << "V6NAX bwd dK sparse: block_mask shape ["
            << mask_NQ << ", " << mask_NK << "] does not match expected ["
            << expected_NQ << ", " << expected_NK << "] for tile geometry "
            << "(BQ=" << BQ << ", BK=" << BK << ") at qL=" << N
            << " kL=" << Nk << ".  See _convert_mask_for_v6nax_bwd_kernel "
            << "in mlx_mfa/attention.py (KD-1 resolution).";
        throw std::runtime_error(oss.str());
      }
    }

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V6NAX bwd dK sparse: only FP16/BF16");

    dkp.set_data(mlx::core::allocator::malloc(dkp.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V6NAXBwdKSparseKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v6nax_bwdk_sparse_mtx);
      auto it = v6nax_bwdk_sparse_pipelines.find(key);
      if (it != v6nax_bwdk_sparse_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      pipeline = compile_v6nax_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV6NAXBackwardDKSparseSource(); },
          "attention_bwd_dk_sparse", mtl_device, causal_,
          nullptr, nullptr, nullptr);
      pipeline = cache_insert_or_release(v6nax_bwdk_sparse_pipelines, v6nax_bwdk_sparse_mtx, key, pipeline);
    }

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_input_array(o, 3);
    enc.set_input_array(lse, 4);
    enc.set_input_array(d_o, 5);
    enc.set_output_array(dkp, 6);
    enc.set_input_array(d_vec, 8);
    enc.set_input_array(block_mask, 9);

    // Phase II-14: the sparse generator's active-qb list is a fixed
    // 1024-entry threadgroup array; reject qL beyond it loudly.
    if ((N + BQ - 1) / BQ > 1024) {
      throw std::runtime_error(
          "V6NAX sparse dK: qL/BQ exceeds the 1024-entry active-list "
          "capacity (qL=" + std::to_string((int)N) +
          ", BQ=" + std::to_string((int)BQ) + ") — II-14 restructure.");
    }
    v6nax_dispatch_bwd_dk_sparse(pipeline, &enc, (int)N, (int)Nk, (int)Hq, (int)Hk,
                                (int)B, (int)D, BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV6NAXBwdDKSparse*>(&other);
    return p && p->scale_ == scale_ && p->wm_ == wm_ && p->causal_ == causal_;
  }

  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    auto qs = inputs[0].shape();
    auto ks = inputs[1].shape();
    mlx::core::Shape s{qs[0], qs[1], (int)wm_, ks[2], qs[3]};
    return {s};
  }

 private:
  float scale_;
  uint16_t wm_;
  bool causal_;
};

mlx::core::array v6_nax_backward_dk_sparse_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& o,
    const mlx::core::array& lse, const mlx::core::array& d_o,
    const mlx::core::array& d_vec,
    const mlx::core::array& block_mask,
    float scale, int wm, bool causal) {
  if (q.ndim() != 4) throw std::runtime_error("V6NAX bwd dK sparse: Q must be 4D");
  if (block_mask.ndim() != 2)
    throw std::runtime_error("V6NAX bwd dK sparse: block_mask must be 2-D");
  auto s = mlx::core::default_stream(mlx::core::Device::gpu);

  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);
  auto oc = mlx::core::contiguous(o, false, s);
  auto lsec = mlx::core::contiguous(lse, false, s);
  auto dOc = mlx::core::contiguous(d_o, false, s);
  auto dvc = mlx::core::contiguous(d_vec, false, s);
  auto bmc = mlx::core::contiguous(block_mask, false, s);

  mlx::core::Shape dkp_shape{qc.shape(0), qc.shape(1), wm,
                              kc.shape(2), qc.shape(3)};
  auto outs = mlx::core::array::make_arrays(
      {dkp_shape},
      {mlx::core::float32},
      std::make_shared<MFAV6NAXBwdDKSparse>(s, scale, (uint16_t)wm, causal),
      {qc, kc, vc, oc, lsec, dOc, dvc, bmc});
  return outs[0];
}


// ─────────────────────────────────────────────────────────────────────
// MFAV6NAXBwdFusedDKDVSparse — fused dKdV sparse
// ─────────────────────────────────────────────────────────────────────
struct V6NAXBwdFSparseKey {
  int D, Hq, Hk, dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V6NAXBWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  auto tie() const { return std::tie(D, Hq, Hk, dtype_code, BQ, BK, WM, causal, scale); }  // Track 6: single declaration
  bool operator==(const V6NAXBwdFSparseKey& o) const { return tie() == o.tie(); }
};
struct V6NAXBwdFSparseKeyHash {
  size_t operator()(const V6NAXBwdFSparseKey& k) const { return mlx_mfa_keys::hash_tie(k.tie()); }
};
namespace {
std::mutex v6nax_bwdf_sparse_mtx;
std::unordered_map<V6NAXBwdFSparseKey, void*, V6NAXBwdFSparseKeyHash>
    v6nax_bwdf_sparse_pipelines;
}

class MFAV6NAXBwdFusedDKDVSparse : public mlx::core::Primitive {
 public:
  MFAV6NAXBwdFusedDKDVSparse(mlx::core::Stream s, float scale, uint16_t wm,
                          bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), wm_(wm), causal_(causal) {}

  const char* name() const override { return "MFAV6NAXBwdFusedDKDVSparse"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV6NAXBwdFusedDKDVSparse: CPU eval not supported");
  }

  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override {
    const auto& q   = inputs[0];
    const auto& k   = inputs[1];
    const auto& v   = inputs[2];
    const auto& lse = inputs[3];
    const auto& d_o = inputs[4];
    const auto& d_vec = inputs[5];
    const auto& block_mask = inputs[6];
    auto& dkp = outputs[0];
    auto& dvp = outputs[1];

    const int B  = q.shape(0);
    const int Hq = q.shape(1);
    const int N  = q.shape(2);
    const int D  = q.shape(3);
    const int Hk = k.shape(1);
    const int Nk = k.shape(2);

    if (D != 64 && D != 128)
      throw std::runtime_error("V6NAX bwd fused-dKdV sparse: D must be 64 or 128");
    if (block_mask.ndim() != 2)
      throw std::runtime_error("V6NAX bwd fused-dKdV sparse: block_mask must be 2-D");

    unsigned short BQ = 64;
    unsigned short BK = 32;
    uint16_t WM = wm_;
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDF_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDF_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = mlx_mfa::getenv_aliased("MFA_V6BWDF_WM"))
      WM = (uint16_t)std::atoi(e);

    // v2.50 Prompt 5f Phase A — KD-1 fix: enforce mask shape match.
    {
      const int expected_NQ = (N + BQ - 1) / BQ;
      const int expected_NK = (Nk + BK - 1) / BK;
      const int mask_NQ = block_mask.shape(-2);
      const int mask_NK = block_mask.shape(-1);
      if (mask_NQ != expected_NQ || mask_NK != expected_NK) {
        std::ostringstream oss;
        oss << "V6NAX bwd fused-dKdV sparse: block_mask shape ["
            << mask_NQ << ", " << mask_NK << "] does not match expected ["
            << expected_NQ << ", " << expected_NK << "] for tile geometry "
            << "(BQ=" << BQ << ", BK=" << BK << ") at qL=" << N
            << " kL=" << Nk << ".  See _convert_mask_for_v6nax_bwd_kernel "
            << "in mlx_mfa/attention.py (KD-1 resolution).";
        throw std::runtime_error(oss.str());
      }
    }

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V6NAX bwd fused-dKdV sparse: only FP16/BF16");

    dkp.set_data(mlx::core::allocator::malloc(dkp.nbytes()));
    dvp.set_data(mlx::core::allocator::malloc(dvp.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V6NAXBwdFSparseKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v6nax_bwdf_sparse_mtx);
      auto it = v6nax_bwdf_sparse_pipelines.find(key);
      if (it != v6nax_bwdf_sparse_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      pipeline = compile_v6nax_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV6NAXBackwardFusedDKDVSparseSource(); },
          "attention_bwd_fused_dkdv_sparse", mtl_device, causal_,
          nullptr, nullptr, nullptr);
      pipeline = cache_insert_or_release(v6nax_bwdf_sparse_pipelines, v6nax_bwdf_sparse_mtx, key, pipeline);
    }

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_input_array(lse, 3);
    enc.set_input_array(d_o, 4);
    enc.set_output_array(dkp, 5);
    enc.set_output_array(dvp, 6);
    enc.set_input_array(d_vec, 8);
    enc.set_input_array(block_mask, 9);

    // Phase II-14: the sparse generator's active-qb list is a fixed
    // 1024-entry threadgroup array; reject qL beyond it loudly.
    if ((N + BQ - 1) / BQ > 1024) {
      throw std::runtime_error(
          "V6NAX sparse fused dKdV: qL/BQ exceeds the 1024-entry active-"
          "list capacity (qL=" + std::to_string((int)N) +
          ", BQ=" + std::to_string((int)BQ) + ") — II-14 restructure.");
    }
    v6nax_dispatch_bwd_fused_dkdv_sparse(pipeline, &enc, (int)N, (int)Nk,
                                        (int)Hq, (int)Hk, (int)B, (int)D,
                                        BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV6NAXBwdFusedDKDVSparse*>(&other);
    return p && p->scale_ == scale_ && p->wm_ == wm_ && p->causal_ == causal_;
  }

  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    auto qs = inputs[0].shape();
    auto ks = inputs[1].shape();
    mlx::core::Shape s{qs[0], qs[1], (int)wm_, ks[2], qs[3]};
    return {s, s};
  }

 private:
  float scale_;
  uint16_t wm_;
  bool causal_;
};

std::pair<mlx::core::array, mlx::core::array>
v6_nax_backward_fused_dkdv_sparse_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& lse,
    const mlx::core::array& d_o, const mlx::core::array& d_vec,
    const mlx::core::array& block_mask,
    float scale, int wm, bool causal) {
  if (q.ndim() != 4) throw std::runtime_error("V6NAX bwd fused-dKdV sparse: Q must be 4D");
  if (block_mask.ndim() != 2)
    throw std::runtime_error("V6NAX bwd fused-dKdV sparse: block_mask must be 2-D");
  auto s = mlx::core::default_stream(mlx::core::Device::gpu);

  auto qc = mlx::core::contiguous(q, false, s);
  auto kc = mlx::core::contiguous(k, false, s);
  auto vc = mlx::core::contiguous(v, false, s);
  auto lsec = mlx::core::contiguous(lse, false, s);
  auto dOc = mlx::core::contiguous(d_o, false, s);
  auto dvc = mlx::core::contiguous(d_vec, false, s);
  auto bmc = mlx::core::contiguous(block_mask, false, s);

  mlx::core::Shape partials_shape{qc.shape(0), qc.shape(1), wm,
                                   kc.shape(2), qc.shape(3)};
  auto outs = mlx::core::array::make_arrays(
      {partials_shape, partials_shape},
      {mlx::core::float32, mlx::core::float32},
      std::make_shared<MFAV6NAXBwdFusedDKDVSparse>(s, scale, (uint16_t)wm, causal),
      {qc, kc, vc, lsec, dOc, dvc, bmc});
  return {outs[0], outs[1]};
}

}  // namespace mlx_mfa
