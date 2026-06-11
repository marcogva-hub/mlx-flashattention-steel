/// MFAV6Forward — MLX Primitive that wraps the Draw Things NAAttention port.
///
/// Used to make the V6 forward kernel callable from Python via array::make_arrays
/// (the standard MLX pattern). Once correctness is validated, this can be
/// merged into MFAttention::eval_gpu() in mfa_attention.cpp as a fast-path.

#include "shader_cache.hpp"
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

void* v34_compile(const std::string& source, const std::string& function_name, void* raw_device);
void v34_dispatch(
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
  // V34 — dedicated cache-key fields (no bit-packing).
  bool use_v34 = false;
  uint16_t v34_BQ = 0;
  uint16_t v34_BK = 0;
  uint16_t v34_WM = 0;
  bool operator==(const V6Key& o) const {
    return head_dim == o.head_dim && Hq == o.Hq && Hk == o.Hk &&
           dtype == o.dtype && isCausal == o.isCausal &&
           R == o.R && C == o.C &&
           qbs == o.qbs && kbs == o.kbs && vbs == o.vbs && obs == o.obs &&
           cfg_BQ == o.cfg_BQ && cfg_BK == o.cfg_BK && cfg_SG == o.cfg_SG &&
           cfg_BD == o.cfg_BD && cfg_axis_flags == o.cfg_axis_flags &&
           cfg_bypass_tgp == o.cfg_bypass_tgp &&
           use_v34 == o.use_v34 && v34_BQ == o.v34_BQ &&
           v34_BK == o.v34_BK && v34_WM == o.v34_WM;
  }
};
struct V6KeyHash {
  size_t operator()(const V6Key& k) const {
    size_t h = std::hash<int>{}(k.head_dim);
    h ^= std::hash<int>{}(k.Hq) << 1;
    h ^= std::hash<int>{}(k.Hk) << 2;
    h ^= std::hash<int>{}(k.dtype) << 3;
    h ^= std::hash<bool>{}(k.isCausal) << 4;
    h ^= std::hash<uint32_t>{}(k.R) << 5;
    h ^= std::hash<uint32_t>{}(k.C) << 6;
    h ^= std::hash<uint32_t>{}(k.qbs) << 7;
    // Repo review 2026-05: kbs/vbs/obs participate in operator== but were
    // absent from the hash — GQA shapes (kbs != qbs) clustered into one
    // bucket, degrading lookups to a linear scan over colliders.
    h ^= std::hash<uint32_t>{}(k.kbs) << 12;
    h ^= std::hash<uint32_t>{}(k.vbs) << 13;
    h ^= std::hash<uint32_t>{}(k.obs) << 14;
    h ^= std::hash<uint16_t>{}(k.cfg_BQ) << 15;
    h ^= std::hash<uint16_t>{}(k.cfg_BK) << 16;
    h ^= std::hash<uint16_t>{}(k.cfg_SG) << 17;
    h ^= std::hash<uint16_t>{}(k.cfg_BD) << 18;
    h ^= std::hash<uint16_t>{}(k.cfg_axis_flags) << 19;
    h ^= std::hash<bool>{}(k.cfg_bypass_tgp) << 20;
    h ^= std::hash<bool>{}(k.use_v34) << 8;
    h ^= std::hash<uint16_t>{}(k.v34_BQ) << 9;
    h ^= std::hash<uint16_t>{}(k.v34_BK) << 10;
    h ^= std::hash<uint16_t>{}(k.v34_WM) << 11;
    return h;
  }
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
                                bool use_v34_override = false,
                                bool use_v34_explicit = false) {
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

  // V34 — NAX-direct rewrite. Caller (eval_gpu) decides via use_v34_explicit
  // based on shape (D, Nk). Default fallback when not explicit: D=128 → ON.
  // Forward, non-causal, single-Otile-eligible only.
  bool use_v34;
  if (use_v34_explicit) {
    use_v34 = use_v34_override;
  } else {
    use_v34 = (head_dim == 128);  // source-gen-only default (without Nk info)
    if (const char* env_v34 = std::getenv("MFA_V6_USE_V34"))
      use_v34 = (std::atoi(env_v34) != 0);
  }
  // v2.50 Prompt 4 Section B: lift `isCausal` constraint.  Prompt 2
  // Phase 4a added V34 forward causal kernel support but missed this
  // dispatch-side gate — causal was silently routing to STEEL legacy
  // (log2-domain lse) instead of V34 (natural-log lse), making
  // V34 backward consume wrong-domain lse and produce wrong gradients.
  // Now V34 forward + V34 backward causal both engage when force_v34=True.
  if (use_v34 && !single_otile) use_v34 = false;
  // V34 needs BQ % (WM * 16) == 0 and BD % 16 == 0.
  // Per-D defaults: D=64 → WM=2, BQ=32, BK=64; D=128 → WM=4, BQ=64, BK=32.
  // Override via env vars below.
  unsigned short v34_BQ = (head_dim == 64) ? 32 : 64;
  unsigned short v34_BK = (head_dim == 64) ? 64 : 32;
  uint16_t v34_WM = (head_dim == 64) ? 2 : 4;
  if (use_v34) {
    if (const char* env_bq = std::getenv("MFA_V6_V34_BQ")) v34_BQ = (unsigned short)std::atoi(env_bq);
    if (const char* env_bk = std::getenv("MFA_V6_V34_BK")) v34_BK = (unsigned short)std::atoi(env_bk);
    if (const char* env_wm = std::getenv("MFA_V6_V34_WM")) v34_WM = (uint16_t)std::atoi(env_wm);
    // Validate: BQ % (WM*16) == 0
    if (v34_BQ % (v34_WM * 16) != 0 || head_dim % 16 != 0) {
      use_v34 = false;  // fall back to legacy if invalid config
    }
  }

  simd::ushort3 blockDims = use_v34
      ? simd::make_ushort3(v34_BQ, v34_BK, BD)
      : simd::make_ushort3(BQ, BK, BD);
  uint16_t exec_sg_for_desc = use_v34 ? v34_WM : exec_sg;

  NAAttentionKernelDescriptor desc(
      blockDims, (unsigned short)head_dim, (unsigned short)Hq,
      (unsigned short)Hk, /*executionSIMDGroups=*/exec_sg_for_desc,
      /*checkCEdge1=*/true, mp, AttentionKernelType::forward,
      /*scale=*/1.0f / std::sqrt((float)head_dim),
      /*bypassThreadgroupMemory=*/bypass_tgp,
      /*isCausal=*/isCausal, /*masked=*/false);
  desc.singleOtileMode = single_otile;
  desc.useV34 = use_v34;

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
    // v2.37.0 V34 backward integration: force V34 forward routing even
    // on D=64 small-Nk shapes (which by default route to legacy v6_nax).
    // Caller passes true when V34 backward will consume the lse.
    bool force_v34 = false;
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

    // V34 dispatch — mirror source-gen default logic.
    // Default: ON for D=128 (cross-session bench shows +33-40% vs legacy,
    //   3 shapes reach SDPA parity).
    // Default: OFF for D=64 small-N (FlashVSR-style regresses -39%).
    // Default: ON for D=64 with N_kv > 8000 (LTX2-style asymmetric wins +18%).
    // Override via env var MFA_V6_USE_V34={0,1}.
    bool use_v34;
    if (params_.force_v34) {
      // v2.37.0: caller (V34 backward integration) requires V34 forward
      // to produce natural-log lse.  Override default routing.
      use_v34 = true;
    } else if (D == 128) {
      use_v34 = true;
    } else if (D == 64 && Nk > 8000) {
      // LTX2-cross style asymmetric: V34 wins ~+18%.
      use_v34 = true;
    } else {
      use_v34 = false;
    }
    if (const char* env_v34 = std::getenv("MFA_V6_USE_V34"))
      use_v34 = (std::atoi(env_v34) != 0);
    unsigned short v34_BQ = (D == 64) ? 32 : 64;
    unsigned short v34_BK = (D == 64) ? 64 : 32;
    uint16_t v34_WM = (D == 64) ? 2 : 4;
    {
      bool so_for_v34 = (Hq == Hk) || (Hk > 0 && Hq % Hk == 0);
      if (const char* env_so = std::getenv("MFA_V6_NAX_SINGLE_OTILE"))
        so_for_v34 = (std::atoi(env_so) != 0);
      // v2.50 Prompt 4 Section B: lift causal constraint here too.
      // Prompt 2 Phase 4a added V34 forward causal kernel support but
      // missed this dispatch gate — causal was silently routing to
      // STEEL legacy (which emits log2-domain lse) instead of V34
      // (which emits natural-log lse).  V34 backward consumed wrong-
      // domain lse and produced wrong gradients.  See
      // docs/v50/phase-4b-complete-dv-residual-decisions.md.
      if (use_v34 && !so_for_v34) use_v34 = false;
    }
    if (use_v34) {
      if (const char* env_bq = std::getenv("MFA_V6_V34_BQ")) v34_BQ = (unsigned short)std::atoi(env_bq);
      if (const char* env_bk = std::getenv("MFA_V6_V34_BK")) v34_BK = (unsigned short)std::atoi(env_bk);
      if (const char* env_wm = std::getenv("MFA_V6_V34_WM")) v34_WM = (uint16_t)std::atoi(env_wm);
      if (v34_BQ % (v34_WM * 16) != 0 || D % 16 != 0) {
        use_v34 = false;
      }
    }

    // Include all tile + flag params in cache key.
    // Repo review 2026-05: tile/config params moved from bit-packed high
    // bits of R/C/qbs/kbs to dedicated key fields (see V6Key comment).
    V6Key key{D, Hq, Hk, dtype_code, params_.causal,
              R, C, qbs, kbs, vbs, obs,
              BQ, BK, executionSIMDGroups, BD,
              (uint16_t)axis_flags, bypass_tgp,
              use_v34, v34_BQ, v34_BK, v34_WM};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v6_mtx);
      auto it = v6_pipelines.find(key);
      if (it != v6_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      std::string src = generate_v6_source(
          D, Hq, Hk, dtype_code, params_.causal, params_.bhnd, (int)R,
          /*use_v34_override=*/use_v34, /*use_v34_explicit=*/true);
      if (use_v34) {
        // V34 uses no FCs (params via struct buffer).
        if (std::getenv("MFA_V34_DUMP_SOURCE")) {
          fprintf(stderr, "=== V34 source for BQ=%d BK=%d BD=%d WM=%d ===\n",
                  (int)v34_BQ, (int)v34_BK, (int)D, (int)v34_WM);
          auto pos = src.find("// === lse write");
          if (pos != std::string::npos) {
            fprintf(stderr, "%s\n=== ===\n",
                    src.substr(pos, 800).c_str());
          } else {
            fprintf(stderr, "(lse write marker not found!)\n");
          }
        }
        pipeline = v34_compile(src, "attention", mtl_device);
      } else {
        pipeline = v6_nax_compile_with_constants(
            src, "attention", mtl_device, R, C, qbs, kbs, vbs, obs);
      }
      pipeline = cache_insert_or_release(v6_pipelines, v6_mtx, key, pipeline);
    }

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_output_array(out, 3);
    if (!use_v34) {
      enc.set_output_array(lse, 4);  // Legacy path: buffer 4 is lse.
    } else {
      // v2.36.x BLK1 patch: V34 forward now writes lse to buffer 5
      // (buffer 4 holds the V34Params struct via set_bytes).  Per
      // docs/v6-nax/v34-backward-decisions.md DC0 — lse is required
      // input infrastructure for V34 backward dQ/dK/dV kernels.
      enc.set_output_array(lse, 5);
    }

    if (use_v34) {
      v34_dispatch(
          pipeline, &enc,
          (int)N, (int)Nk, (int)Hq, (int)Hk, (int)B, (int)D,
          v34_BQ, v34_BK, v34_WM);
    } else {
      unsigned short elem_size = 2;  // FP16/BF16 = 2 bytes
      unsigned short tgmem = BQ * BK * executionSIMDGroups * elem_size;
      v6_nax_dispatch(
          pipeline, &enc,
          nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
          R, (uint32_t)Hq, (uint32_t)B,
          BQ, executionSIMDGroups, tgmem);
    }
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV6Forward*>(&other);
    // Repo review 2026-05: force_v34 MUST participate — a force_v34=true
    // forward emits natural-log LSE (consumed by V34 backward) while the
    // default path emits log2-domain LSE.  Without this term, MLX graph
    // dedup could conflate the two nodes, feeding log2 LSE into a backward
    // expecting natural log (silently wrong gradients).
    return p && p->params_.causal == params_.causal
             && p->params_.bhnd == params_.bhnd
             && p->params_.force_v34 == params_.force_v34;
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
    const mlx::core::array& v, bool causal, bool force_v34) {
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
  MFAV6Forward::Params params{causal, bhnd, force_v34};

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
// V34 backward dQ — minimum-viable Primitive (Phase 1 Section B).
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
// v2.40.x-internal Sprint C (P3-HIGH-01): V34 backward pipeline-compile helper.
// Consolidates the ~30-40 LOC of pipeline-cache-miss boilerplate duplicated
// across all 5 V34 backward Primitives (MFAV34BwdQuery, MFAV34BwdKeyValue,
// MFAV34BwdDV, MFAV34BwdDK, MFAV34BwdFusedDKDV) into a single helper.
// Pure refactor: produces byte-identical generated source as before; only
// the C++ boilerplate around source-gen + compile is consolidated.
//
// Each caller still owns its own pipeline-cache mutex + map (the cache keys
// differ per Primitive).  The helper handles:
//   1. AttentionOperands precision setup (FP16/BF16 inputs, FP32 S/P/L)
//   2. NAAttentionKernelDescriptor construction (singleOtileMode + useV34)
//   3. Optional source-dump hook (env-gated via MFA_V34BWD*_DUMP_SOURCE +
//      optional MFA_V34BWD*_DUMP_PATH for file output)
//   4. Source string generation via caller-provided lambda
//   5. Final v34_compile() invocation
// -----------------------------------------------------------------------------
namespace {

template <typename SourceGenFn>
void* compile_v34_backward_pipeline(
    int D, int Hq, int Hk, int dtype_code,
    unsigned short BQ, unsigned short BK, uint16_t WM,
    float scale,
    SourceGenFn source_gen_fn,
    const char* kernel_fn_name,
    void* mtl_device,
    bool isCausal = false,  // v2.50 Phase 4b-complete (Prompt 3): plumbed through
    const char* dump_env_var = nullptr,
    const char* dump_label = nullptr,
    const char* dump_path_env_var = nullptr) {
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
  // V34 backward source generators which switch on the source-gen method).
  // v2.50 Phase 4b-complete (Prompt 3): isCausal now plumbed through so
  // V34BWD*_CAUSAL macros get the correct compile-time value.  Pre-Prompt-3
  // this was hardcoded to false — a latent bug that silently made my
  // Prompt 2 Phase 4b dQ causal mask a no-op in production.
  simd::ushort3 blockDims =
      simd::make_ushort3(BQ, BK, (unsigned short)D);
  NAAttentionKernelDescriptor desc(
      blockDims, (unsigned short)D, (unsigned short)Hq,
      (unsigned short)Hk, /*executionSIMDGroups=*/WM,
      /*checkCEdge1=*/false, mp,
      AttentionKernelType::forward,  // placeholder; ignored by V34 backward
      /*scale=*/scale,
      /*bypassThreadgroupMemory=*/false,
      /*isCausal=*/isCausal, /*masked=*/false);
  desc.singleOtileMode = true;
  desc.useV34 = true;

  // Source generation via caller's lambda.
  NAAttentionKernel ker(desc);
  std::string src = source_gen_fn(ker);

  // Optional source-dump hook.
  if (dump_env_var && std::getenv(dump_env_var)) {
    const char* path = dump_path_env_var ? std::getenv(dump_path_env_var) : nullptr;
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

  return v34_compile(src, kernel_fn_name, mtl_device);
}

}  // namespace


void v34_dispatch_bwd_query(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

struct V34BwdQKey {
  int D;
  int Hq, Hk;
  int dtype_code;  // 0=fp16, 1=bf16
  unsigned short v34_BQ, v34_BK;
  uint16_t v34_WM;
  bool causal;  // v2.50 Phase 4b-complete: separate pipeline cache per causal flag
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V34BWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  bool operator==(const V34BwdQKey& o) const {
    return D == o.D && Hq == o.Hq && Hk == o.Hk
        && dtype_code == o.dtype_code
        && v34_BQ == o.v34_BQ && v34_BK == o.v34_BK && v34_WM == o.v34_WM
        && causal == o.causal
        && scale == o.scale;
  }
};
struct V34BwdQKeyHash {
  size_t operator()(const V34BwdQKey& k) const {
    size_t h = std::hash<int>{}(k.D);
    h ^= std::hash<int>{}(k.Hq) << 1;
    h ^= std::hash<int>{}(k.Hk) << 2;
    h ^= std::hash<int>{}(k.dtype_code) << 3;
    h ^= std::hash<uint16_t>{}(k.v34_BQ) << 4;
    h ^= std::hash<uint16_t>{}(k.v34_BK) << 5;
    h ^= std::hash<uint16_t>{}(k.v34_WM) << 6;
    h ^= std::hash<bool>{}(k.causal) << 7;
    h ^= std::hash<float>{}(k.scale) << 16;
    return h;
  }
};
namespace {
std::mutex v34_bwdq_mtx;
std::unordered_map<V34BwdQKey, void*, V34BwdQKeyHash> v34_bwdq_pipelines;
}

class MFAV34BwdQuery : public mlx::core::Primitive {
 public:
  // v2.50 Phase 4b-complete (Prompt 3): causal added to constructor.
  // Default false preserves prior signature; new code should pass causal.
  MFAV34BwdQuery(mlx::core::Stream s, float scale, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), causal_(causal) {}

  const char* name() const override { return "MFAV34BwdQuery"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV34BwdQuery: CPU eval not supported");
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
      throw std::runtime_error("V34 bwd dQ: Q must be 4D [B,H,N,D]");
    const int B  = q.shape(0);
    const int Hq = q.shape(1);
    const int N  = q.shape(2);
    const int D  = q.shape(3);
    const int Hk = k.shape(1);
    const int Nk = k.shape(2);

    if (D != 64 && D != 128)
      throw std::runtime_error("V34 bwd dQ: D must be 64 or 128");

    // M5-tuned defaults per DC7 (matches V34 forward defaults).
    unsigned short v34_BQ = (D == 64) ? 32 : 64;
    unsigned short v34_BK = (D == 64) ? 64 : 32;
    uint16_t v34_WM = (D == 64) ? 2 : 4;
    if (const char* e = std::getenv("MFA_V34BWD_BQ"))
      v34_BQ = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWD_BK"))
      v34_BK = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWD_WM"))
      v34_WM = (uint16_t)std::atoi(e);

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V34 bwd dQ: only FP16/BF16");

    dq.set_data(mlx::core::allocator::malloc(dq.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V34BwdQKey key{D, Hq, Hk, dtype_code, v34_BQ, v34_BK, v34_WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v34_bwdq_mtx);
      auto it = v34_bwdq_pipelines.find(key);
      if (it != v34_bwdq_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      // v2.40.x-internal Sprint C: consolidated via compile_v34_backward_pipeline.
      // v2.50 Phase 4b-complete (Prompt 3): causal_ plumbed through.
      pipeline = compile_v34_backward_pipeline(
          D, Hq, Hk, dtype_code, v34_BQ, v34_BK, v34_WM, scale_,
          [](NAAttentionKernel& k) { return k.createV34BackwardQuerySource(); },
          "attention_bwd_q", mtl_device, causal_,
          "MFA_V34BWD_DUMP_SOURCE", "V34 bwd dQ", nullptr);
      pipeline = cache_insert_or_release(v34_bwdq_pipelines, v34_bwdq_mtx, key, pipeline);
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

    v34_dispatch_bwd_query(
        pipeline, &enc,
        (int)N, (int)Nk, (int)Hq, (int)Hk, (int)B, (int)D,
        v34_BQ, v34_BK, v34_WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV34BwdQuery*>(&other);
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

// Public Python-callable: V34 backward dQ.
//
// Args: Q [B,Hq,N,D], K [B,Hk,Nk,D], V [B,Hk,Nk,D] (T),
//       O [B,Hq,N,D] (T),
//       L [B,Hq,N] (FP32),
//       dO [B,Hq,N,D] (T),
//       scale (float).
//
// Returns: dQ [B,Hq,N,D] (T).
//
// Routing constraint per DC12: callers must ensure V34-forward-eligible
// shapes (D=128 always; D=64 with Nk>8000).  V34 backward will produce
// garbage on shapes that routed through legacy v6_nax forward (lse
// convention mismatch).  flash_attention() VJP layer enforces this in
// Phase 2 Section E.
mlx::core::array v6_nax_backward_query(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& o,
    const mlx::core::array& lse, const mlx::core::array& d_o,
    const mlx::core::array& d_vec,  // v2.38.1: precomputed rowsum(dO⊙O)
    float scale, bool causal) {
  if (q.ndim() != 4) throw std::runtime_error("V34 bwd dQ: Q must be 4D");
  if (k.shape(1) <= 0 || q.shape(1) % k.shape(1) != 0)
    throw std::runtime_error("V34 bwd dQ: Hq must be multiple of Hk");

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
      std::make_shared<MFAV34BwdQuery>(s, scale, causal),
      {qc, kc, vc, oc, lsec, dOc, dvc});
  return outs[0];
}

// =============================================================================
// V34 backward dK/dV — Phase 2 Primitive (single-SG WM=1 design).
// =============================================================================

void v34_dispatch_bwd_kv(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

struct V34BwdKVKey {
  int D;
  int Hq, Hk;
  int dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;  // v2.50 Phase 4b-complete: separate pipeline per causal
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V34BWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  bool operator==(const V34BwdKVKey& o) const {
    return D == o.D && Hq == o.Hq && Hk == o.Hk
        && dtype_code == o.dtype_code
        && BQ == o.BQ && BK == o.BK && WM == o.WM
        && causal == o.causal
        && scale == o.scale;
  }
};
struct V34BwdKVKeyHash {
  size_t operator()(const V34BwdKVKey& k) const {
    size_t h = std::hash<int>{}(k.D);
    h ^= std::hash<int>{}(k.Hq) << 1;
    h ^= std::hash<int>{}(k.Hk) << 2;
    h ^= std::hash<int>{}(k.dtype_code) << 3;
    h ^= std::hash<uint16_t>{}(k.BQ) << 4;
    h ^= std::hash<uint16_t>{}(k.BK) << 5;
    h ^= std::hash<uint16_t>{}(k.WM) << 6;
    h ^= std::hash<bool>{}(k.causal) << 7;
    h ^= std::hash<float>{}(k.scale) << 16;
    return h;
  }
};
namespace {
std::mutex v34_bwdkv_mtx;
std::unordered_map<V34BwdKVKey, void*, V34BwdKVKeyHash> v34_bwdkv_pipelines;
}

class MFAV34BwdKeyValue : public mlx::core::Primitive {
 public:
  MFAV34BwdKeyValue(mlx::core::Stream s, float scale, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), causal_(causal) {}

  const char* name() const override { return "MFAV34BwdKeyValue"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV34BwdKeyValue: CPU eval not supported");
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
      throw std::runtime_error("V34 bwd dKdV: D must be 64 or 128");

    // Phase 2 defaults: WM=1 single-SG; BQ=32; BK=(D==64?64:32).
    // Phase 2.O1 (2026-05-13): WM=2 K-row partition was attempted and
    // FALSIFIED empirically (0.77-0.84× regression vs WM=1).  The
    // redundant softmax compute across SGs taxed more than the GEMM
    // partition saved.  Reverted to WM=1.  See v34-backward-status.md
    // §"Phase 2.O1 falsified" for next-attempt design (Q-row partition
    // + TGP streaming reduction).
    unsigned short BQ = 32;
    unsigned short BK = (D == 64) ? 64 : 32;
    uint16_t WM = 1;
    if (const char* e = std::getenv("MFA_V34BWDKV_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDKV_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDKV_WM"))
      WM = (uint16_t)std::atoi(e);

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V34 bwd dKdV: only FP16/BF16");

    dk.set_data(mlx::core::allocator::malloc(dk.nbytes()));
    dv.set_data(mlx::core::allocator::malloc(dv.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V34BwdKVKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v34_bwdkv_mtx);
      auto it = v34_bwdkv_pipelines.find(key);
      if (it != v34_bwdkv_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      // v2.40.x-internal Sprint C: consolidated via compile_v34_backward_pipeline.
      pipeline = compile_v34_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV34BackwardKeyValueSource(); },
          "attention_bwd_kv", mtl_device, causal_,
          nullptr, nullptr, nullptr);  // no dump hook in legacy fused
      pipeline = cache_insert_or_release(v34_bwdkv_pipelines, v34_bwdkv_mtx, key, pipeline);
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

    v34_dispatch_bwd_kv(pipeline, &enc, (int)N, (int)Nk, (int)Hq, (int)Hk,
                        (int)B, (int)D, BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV34BwdKeyValue*>(&other);
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
  if (q.ndim() != 4) throw std::runtime_error("V34 bwd dKdV: Q must be 4D");
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
      std::make_shared<MFAV34BwdKeyValue>(s, scale, causal),
      {qc, kc, vc, oc, lsec, dOc, dvc});
  return {outs[0], outs[1]};
}

// =============================================================================
// V34 backward dV-only — Phase 2.O2 multi-SG Q-row partition Primitive.
// Emits per-SG dV partial to a [B, Hq, WM, kL, D] FP32 intermediate buffer.
// Python wrapper reduces via mx.sum(axis=2) and casts to T.
// =============================================================================

void v34_dispatch_bwd_dv(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

struct V34BwdVKey {
  int D;
  int Hq, Hk;
  int dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;  // v2.50 Phase 4b-complete: separate pipeline per causal
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V34BWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  bool operator==(const V34BwdVKey& o) const {
    return D == o.D && Hq == o.Hq && Hk == o.Hk
        && dtype_code == o.dtype_code
        && BQ == o.BQ && BK == o.BK && WM == o.WM
        && causal == o.causal
        && scale == o.scale;
  }
};
struct V34BwdVKeyHash {
  size_t operator()(const V34BwdVKey& k) const {
    size_t h = std::hash<int>{}(k.D);
    h ^= std::hash<int>{}(k.Hq) << 1;
    h ^= std::hash<int>{}(k.Hk) << 2;
    h ^= std::hash<int>{}(k.dtype_code) << 3;
    h ^= std::hash<uint16_t>{}(k.BQ) << 4;
    h ^= std::hash<uint16_t>{}(k.BK) << 5;
    h ^= std::hash<uint16_t>{}(k.WM) << 6;
    h ^= std::hash<bool>{}(k.causal) << 7;
    h ^= std::hash<float>{}(k.scale) << 16;
    return h;
  }
};
namespace {
std::mutex v34_bwdv_mtx;
std::unordered_map<V34BwdVKey, void*, V34BwdVKeyHash> v34_bwdv_pipelines;
}

class MFAV34BwdDV : public mlx::core::Primitive {
 public:
  MFAV34BwdDV(mlx::core::Stream s, float scale, uint16_t wm, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), wm_(wm), causal_(causal) {}

  const char* name() const override { return "MFAV34BwdDV"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV34BwdDV: CPU eval not supported");
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
      throw std::runtime_error("V34 bwd dV: D must be 64 or 128");

    // Phase 2.O2 defaults: WM=4 Q-row partition. BQ = WM*16 = 64.
    unsigned short BQ = 64;
    unsigned short BK = 32;
    uint16_t WM = wm_;
    if (const char* e = std::getenv("MFA_V34BWDV_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDV_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDV_WM"))
      WM = (uint16_t)std::atoi(e);

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V34 bwd dV: only FP16/BF16");

    dvp.set_data(mlx::core::allocator::malloc(dvp.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V34BwdVKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v34_bwdv_mtx);
      auto it = v34_bwdv_pipelines.find(key);
      if (it != v34_bwdv_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      // v2.40.x-internal Sprint C: consolidated via compile_v34_backward_pipeline.
      pipeline = compile_v34_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV34BackwardDVSource(); },
          "attention_bwd_dv", mtl_device, causal_,
          nullptr, nullptr, nullptr);  // no dump hook in split-dV
      pipeline = cache_insert_or_release(v34_bwdv_pipelines, v34_bwdv_mtx, key, pipeline);
    }

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_input_array(lse, 3);
    enc.set_input_array(d_o, 4);
    enc.set_output_array(dvp, 5);

    v34_dispatch_bwd_dv(pipeline, &enc, (int)N, (int)Nk, (int)Hq, (int)Hk,
                        (int)B, (int)D, BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV34BwdDV*>(&other);
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
// V34 backward dV SPARSE Primitive — Prompt 5b Section A PoC.
// Identical to MFAV34BwdDV but accepts block_mask input and routes to the
// sparse source generator.  Cache key extended with is_sparse flag.
// ============================================================================

void v34_dispatch_bwd_dv_sparse(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

struct V34BwdVSparseKey {
  int D;
  int Hq, Hk;
  int dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V34BWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  // is_sparse implicit (this struct only used for sparse kernels) but
  // included for future-proofing if a single cache holds both variants.
  bool operator==(const V34BwdVSparseKey& o) const {
    return D == o.D && Hq == o.Hq && Hk == o.Hk
        && dtype_code == o.dtype_code
        && BQ == o.BQ && BK == o.BK && WM == o.WM
        && causal == o.causal
        && scale == o.scale;
  }
};
struct V34BwdVSparseKeyHash {
  size_t operator()(const V34BwdVSparseKey& k) const {
    size_t h = std::hash<int>{}(k.D);
    h ^= std::hash<int>{}(k.Hq) << 1;
    h ^= std::hash<int>{}(k.Hk) << 2;
    h ^= std::hash<int>{}(k.dtype_code) << 3;
    h ^= std::hash<uint16_t>{}(k.BQ) << 4;
    h ^= std::hash<uint16_t>{}(k.BK) << 5;
    h ^= std::hash<uint16_t>{}(k.WM) << 6;
    h ^= std::hash<bool>{}(k.causal) << 7;
    h ^= std::hash<float>{}(k.scale) << 16;
    return h;
  }
};
namespace {
std::mutex v34_bwdv_sparse_mtx;
std::unordered_map<V34BwdVSparseKey, void*, V34BwdVSparseKeyHash>
    v34_bwdv_sparse_pipelines;
}

class MFAV34BwdDVSparse : public mlx::core::Primitive {
 public:
  MFAV34BwdDVSparse(mlx::core::Stream s, float scale, uint16_t wm, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), wm_(wm), causal_(causal) {}

  const char* name() const override { return "MFAV34BwdDVSparse"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV34BwdDVSparse: CPU eval not supported");
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
      throw std::runtime_error("V34 bwd dV sparse: D must be 64 or 128");
    if (block_mask.ndim() != 2)
      throw std::runtime_error("V34 bwd dV sparse: block_mask must be 2-D [NQ, NK] (Section A PoC)");

    unsigned short BQ = 64;
    unsigned short BK = 32;
    uint16_t WM = wm_;
    if (const char* e = std::getenv("MFA_V34BWDV_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDV_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDV_WM"))
      WM = (uint16_t)std::atoi(e);

    // v2.50 Prompt 5f Phase A — KD-1 fix: enforce mask shape match.
    // Python orchestrator (`_convert_mask_for_v34_bwd_kernel` in attention.py)
    // converts BT-block masks to this kernel's tile geometry before dispatch.
    // Runtime check guards against future regressions or direct callers.
    {
      const int expected_NQ = (N + BQ - 1) / BQ;
      const int expected_NK = (Nk + BK - 1) / BK;
      const int mask_NQ = block_mask.shape(-2);
      const int mask_NK = block_mask.shape(-1);
      if (mask_NQ != expected_NQ || mask_NK != expected_NK) {
        std::ostringstream oss;
        oss << "V34 bwd dV sparse: block_mask shape ["
            << mask_NQ << ", " << mask_NK << "] does not match expected ["
            << expected_NQ << ", " << expected_NK << "] for tile geometry "
            << "(BQ=" << BQ << ", BK=" << BK << ") at qL=" << N
            << " kL=" << Nk << ".  See _convert_mask_for_v34_bwd_kernel "
            << "in mlx_mfa/attention.py (KD-1 resolution).";
        throw std::runtime_error(oss.str());
      }
    }

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V34 bwd dV sparse: only FP16/BF16");

    dvp.set_data(mlx::core::allocator::malloc(dvp.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V34BwdVSparseKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v34_bwdv_sparse_mtx);
      auto it = v34_bwdv_sparse_pipelines.find(key);
      if (it != v34_bwdv_sparse_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      pipeline = compile_v34_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV34BackwardDVSparseSource(); },
          "attention_bwd_dv_sparse", mtl_device, causal_,
          nullptr, nullptr, nullptr);
      pipeline = cache_insert_or_release(v34_bwdv_sparse_pipelines, v34_bwdv_sparse_mtx, key, pipeline);
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

    v34_dispatch_bwd_dv_sparse(pipeline, &enc, (int)N, (int)Nk,
                               (int)Hq, (int)Hk, (int)B, (int)D, BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV34BwdDVSparse*>(&other);
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
  if (q.ndim() != 4) throw std::runtime_error("V34 bwd dV sparse: Q must be 4D");
  if (block_mask.ndim() != 2)
    throw std::runtime_error("V34 bwd dV sparse: block_mask must be 2-D (Section A PoC)");
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
      std::make_shared<MFAV34BwdDVSparse>(s, scale, (uint16_t)wm, causal),
      {qc, kc, vc, lsec, dOc, bmc});
  return outs[0];
}


// Returns dV_partials FP32 [B, Hq, WM, kL, D].  Caller must mx.sum(axis=2)
// and cast to T to get final dV.
mlx::core::array v6_nax_backward_dv_raw(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& lse,
    const mlx::core::array& d_o, float scale, int wm, bool causal) {
  if (q.ndim() != 4) throw std::runtime_error("V34 bwd dV: Q must be 4D");
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
      std::make_shared<MFAV34BwdDV>(s, scale, (uint16_t)wm, causal),
      {qc, kc, vc, lsec, dOc});
  return outs[0];
}

// =============================================================================
// V34 backward dK-only — Phase 2.O2 multi-SG Q-row partition Primitive.
// Sister to MFAV34BwdDV.  Emits per-SG dK partial to dK_partials [B, Hq, WM,
// kL, D] FP32.  Python wrapper reduces via mx.sum(axis=2) and casts to T.
// =============================================================================

void v34_dispatch_bwd_dk(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

struct V34BwdKKey {
  int D;
  int Hq, Hk;
  int dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;  // v2.50 Phase 4b-complete: separate pipeline per causal
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V34BWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  bool operator==(const V34BwdKKey& o) const {
    return D == o.D && Hq == o.Hq && Hk == o.Hk
        && dtype_code == o.dtype_code
        && BQ == o.BQ && BK == o.BK && WM == o.WM
        && causal == o.causal
        && scale == o.scale;
  }
};
struct V34BwdKKeyHash {
  size_t operator()(const V34BwdKKey& k) const {
    size_t h = std::hash<int>{}(k.D);
    h ^= std::hash<int>{}(k.Hq) << 1;
    h ^= std::hash<int>{}(k.Hk) << 2;
    h ^= std::hash<int>{}(k.dtype_code) << 3;
    h ^= std::hash<uint16_t>{}(k.BQ) << 4;
    h ^= std::hash<uint16_t>{}(k.BK) << 5;
    h ^= std::hash<uint16_t>{}(k.WM) << 6;
    h ^= std::hash<bool>{}(k.causal) << 7;
    h ^= std::hash<float>{}(k.scale) << 16;
    return h;
  }
};
namespace {
std::mutex v34_bwdk_mtx;
std::unordered_map<V34BwdKKey, void*, V34BwdKKeyHash> v34_bwdk_pipelines;
}

class MFAV34BwdDK : public mlx::core::Primitive {
 public:
  MFAV34BwdDK(mlx::core::Stream s, float scale, uint16_t wm, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), wm_(wm), causal_(causal) {}

  const char* name() const override { return "MFAV34BwdDK"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV34BwdDK: CPU eval not supported");
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
      throw std::runtime_error("V34 bwd dK: D must be 64 or 128");

    unsigned short BQ = 64;
    unsigned short BK = 32;
    uint16_t WM = wm_;
    if (const char* e = std::getenv("MFA_V34BWDK_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDK_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDK_WM"))
      WM = (uint16_t)std::atoi(e);

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V34 bwd dK: only FP16/BF16");

    dkp.set_data(mlx::core::allocator::malloc(dkp.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V34BwdKKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v34_bwdk_mtx);
      auto it = v34_bwdk_pipelines.find(key);
      if (it != v34_bwdk_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      // v2.40.x-internal Sprint C: consolidated via compile_v34_backward_pipeline.
      pipeline = compile_v34_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV34BackwardDKSource(); },
          "attention_bwd_dk", mtl_device, causal_,
          nullptr, nullptr, nullptr);  // no dump hook in split-dK
      pipeline = cache_insert_or_release(v34_bwdk_pipelines, v34_bwdk_mtx, key, pipeline);
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


    v34_dispatch_bwd_dk(pipeline, &enc, (int)N, (int)Nk, (int)Hq, (int)Hk,
                        (int)B, (int)D, BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV34BwdDK*>(&other);
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
  if (q.ndim() != 4) throw std::runtime_error("V34 bwd dK: Q must be 4D");
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
      std::make_shared<MFAV34BwdDK>(s, scale, (uint16_t)wm, causal),
      {qc, kc, vc, oc, lsec, dOc, dvc});
  return outs[0];
}


// =============================================================================
// V34 backward FUSED dK+dV Primitive (Sprint v2.39.0 Phase C.1.a, Option γ).
// Combines split-dV + split-dK into a single kernel dispatch.  Per-SG-slot
// outputs to dK_partials + dV_partials [B, Hq, WM, kL, D] FP32 each;
// caller reduces via mx.sum(axis=2) and casts to T.
// =============================================================================

void v34_dispatch_bwd_fused_dkdv(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

struct V34BwdFusedKey {
  int D;
  int Hq, Hk;
  int dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;  // v2.50 Phase 4b-complete: separate pipeline per causal
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V34BWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  bool operator==(const V34BwdFusedKey& o) const {
    return D == o.D && Hq == o.Hq && Hk == o.Hk
        && dtype_code == o.dtype_code
        && BQ == o.BQ && BK == o.BK && WM == o.WM
        && causal == o.causal
        && scale == o.scale;
  }
};
struct V34BwdFusedKeyHash {
  size_t operator()(const V34BwdFusedKey& k) const {
    size_t h = std::hash<int>{}(k.D);
    h ^= std::hash<int>{}(k.Hq) << 1;
    h ^= std::hash<int>{}(k.Hk) << 2;
    h ^= std::hash<int>{}(k.dtype_code) << 3;
    h ^= std::hash<uint16_t>{}(k.BQ) << 4;
    h ^= std::hash<uint16_t>{}(k.BK) << 5;
    h ^= std::hash<uint16_t>{}(k.WM) << 6;
    h ^= std::hash<bool>{}(k.causal) << 7;
    h ^= std::hash<float>{}(k.scale) << 16;
    return h;
  }
};
namespace {
std::mutex v34_bwd_fused_mtx;
std::unordered_map<V34BwdFusedKey, void*, V34BwdFusedKeyHash> v34_bwd_fused_pipelines;
}

class MFAV34BwdFusedDKDV : public mlx::core::Primitive {
 public:
  MFAV34BwdFusedDKDV(mlx::core::Stream s, float scale, uint16_t wm, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), wm_(wm), causal_(causal) {}

  const char* name() const override { return "MFAV34BwdFusedDKDV"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV34BwdFusedDKDV: CPU eval not supported");
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
          "V34 bwd fused dKdV: D must be 64 or 128 (Phase C.1.a + C.1.b)");

    // v2.39.1: default BK=16 (was 32 in v2.39.0).  The v2.39.0 BK=32
    // default caused per-SG register spilling at D=64 (H1 confirmed by
    // Sprint v2.39.1 investigation — see docs/v6-nax/v39-1-investigation-
    // synthesis.md).  BK=16 halves dK_accum + dV_accum register
    // footprint at D=64 and brings the kernel below the M5 NAX
    // compiler's spill threshold, recovering 1.01-1.12× speedup vs
    // split across qL ∈ {2048, 16384}.
    //
    // v2.40.0-internal: D=128 reuses BK=16 default.  At D=128 BK=16
    // the per-lane accumulator footprint doubles vs D=64 BK=16 (~512 B
    // vs ~256 B per lane combined dK_accum + dV_accum), matching the
    // v2.39.0 spill-boundary footprint in accumulator B/lane terms.
    // Empirical bench (Sprint B Phase B.5) characterizes whether the
    // higher arithmetic intensity at D=128 amortizes any residual
    // spill cost.  Override via MFA_V34BWDF_BK if benchmarking
    // alternatives.
    unsigned short BQ = 64;
    unsigned short BK = 16;
    uint16_t WM = wm_;
    if (const char* e = std::getenv("MFA_V34BWDF_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDF_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDF_WM"))
      WM = (uint16_t)std::atoi(e);

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V34 bwd fused dKdV: only FP16/BF16");

    dkp.set_data(mlx::core::allocator::malloc(dkp.nbytes()));
    dvp.set_data(mlx::core::allocator::malloc(dvp.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V34BwdFusedKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v34_bwd_fused_mtx);
      auto it = v34_bwd_fused_pipelines.find(key);
      if (it != v34_bwd_fused_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      // v2.40.x-internal Sprint C: consolidated via compile_v34_backward_pipeline.
      // Fused path keeps source-dump hook (set MFA_V34BWDF_DUMP_SOURCE=1;
      // optional MFA_V34BWDF_DUMP_PATH=<file> for file output).
      pipeline = compile_v34_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV34BackwardFusedDKDVSource(); },
          "attention_bwd_fused_dkdv", mtl_device, causal_,
          "MFA_V34BWDF_DUMP_SOURCE", "V34 bwd fused dKdV", "MFA_V34BWDF_DUMP_PATH");
      pipeline = cache_insert_or_release(v34_bwd_fused_pipelines, v34_bwd_fused_mtx, key, pipeline);
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

    v34_dispatch_bwd_fused_dkdv(pipeline, &enc, (int)N, (int)Nk, (int)Hq, (int)Hk,
                                 (int)B, (int)D, BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV34BwdFusedDKDV*>(&other);
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
    throw std::runtime_error("V34 bwd fused dKdV: Q must be 4D");
  if (q.shape(3) != 64 && q.shape(3) != 128)
    throw std::runtime_error(
        "V34 bwd fused dKdV: D must be 64 or 128 (Phase C.1.a + C.1.b)");

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
      std::make_shared<MFAV34BwdFusedDKDV>(s, scale, (uint16_t)wm, causal),
      {qc, kc, vc, lsec, dOc, dvc});
  return {outs[0], outs[1]};  // dK_partials, dV_partials
}


// =============================================================================
// v2.50 Prompt 5d Section A — Sparse plumbing for dQ, dK split, fused dKdV.
// =============================================================================

void v34_dispatch_bwd_query_sparse(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);
void v34_dispatch_bwd_dk_sparse(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);
void v34_dispatch_bwd_fused_dkdv_sparse(
    void* pipeline_raw, void* enc_raw,
    int qL, int kL, int Hq, int Hk, int batchDimension, int head_dim,
    unsigned short BQ, unsigned short BK, uint16_t WM);

// ─────────────────────────────────────────────────────────────────────
// MFAV34BwdQuerySparse
// ─────────────────────────────────────────────────────────────────────
struct V34BwdQSparseKey {
  int D, Hq, Hk, dtype_code;
  unsigned short v34_BQ, v34_BK;
  uint16_t v34_WM;
  bool causal;
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V34BWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  bool operator==(const V34BwdQSparseKey& o) const {
    return D == o.D && Hq == o.Hq && Hk == o.Hk
        && dtype_code == o.dtype_code
        && v34_BQ == o.v34_BQ && v34_BK == o.v34_BK && v34_WM == o.v34_WM
        && causal == o.causal
        && scale == o.scale;
  }
};
struct V34BwdQSparseKeyHash {
  size_t operator()(const V34BwdQSparseKey& k) const {
    size_t h = std::hash<int>{}(k.D);
    h ^= std::hash<int>{}(k.Hq) << 1;
    h ^= std::hash<int>{}(k.Hk) << 2;
    h ^= std::hash<int>{}(k.dtype_code) << 3;
    h ^= std::hash<uint16_t>{}(k.v34_BQ) << 4;
    h ^= std::hash<uint16_t>{}(k.v34_BK) << 5;
    h ^= std::hash<uint16_t>{}(k.v34_WM) << 6;
    h ^= std::hash<bool>{}(k.causal) << 7;
    h ^= std::hash<float>{}(k.scale) << 16;
    return h;
  }
};
namespace {
std::mutex v34_bwdq_sparse_mtx;
std::unordered_map<V34BwdQSparseKey, void*, V34BwdQSparseKeyHash>
    v34_bwdq_sparse_pipelines;
}

class MFAV34BwdQuerySparse : public mlx::core::Primitive {
 public:
  MFAV34BwdQuerySparse(mlx::core::Stream s, float scale, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), causal_(causal) {}

  const char* name() const override { return "MFAV34BwdQuerySparse"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV34BwdQuerySparse: CPU eval not supported");
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
      throw std::runtime_error("V34 bwd dQ sparse: Q must be 4D");
    const int B  = q.shape(0);
    const int Hq = q.shape(1);
    const int N  = q.shape(2);
    const int D  = q.shape(3);
    const int Hk = k.shape(1);
    const int Nk = k.shape(2);
    if (D != 64 && D != 128)
      throw std::runtime_error("V34 bwd dQ sparse: D must be 64 or 128");
    if (block_mask.ndim() != 2)
      throw std::runtime_error("V34 bwd dQ sparse: block_mask must be 2-D [NQ, NK]");

    unsigned short v34_BQ = (D == 64) ? 32 : 64;
    unsigned short v34_BK = (D == 64) ? 64 : 32;
    uint16_t v34_WM = (D == 64) ? 2 : 4;
    if (const char* e = std::getenv("MFA_V34BWD_BQ"))
      v34_BQ = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWD_BK"))
      v34_BK = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWD_WM"))
      v34_WM = (uint16_t)std::atoi(e);

    // v2.50 Prompt 5f Phase A — KD-1 fix: enforce mask shape match.
    {
      const int expected_NQ = (N + v34_BQ - 1) / v34_BQ;
      const int expected_NK = (Nk + v34_BK - 1) / v34_BK;
      const int mask_NQ = block_mask.shape(-2);
      const int mask_NK = block_mask.shape(-1);
      if (mask_NQ != expected_NQ || mask_NK != expected_NK) {
        std::ostringstream oss;
        oss << "V34 bwd dQ sparse: block_mask shape ["
            << mask_NQ << ", " << mask_NK << "] does not match expected ["
            << expected_NQ << ", " << expected_NK << "] for tile geometry "
            << "(BQ=" << v34_BQ << ", BK=" << v34_BK << ") at qL=" << N
            << " kL=" << Nk << ".  See _convert_mask_for_v34_bwd_kernel "
            << "in mlx_mfa/attention.py (KD-1 resolution).";
        throw std::runtime_error(oss.str());
      }
    }

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V34 bwd dQ sparse: only FP16/BF16");

    dq.set_data(mlx::core::allocator::malloc(dq.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V34BwdQSparseKey key{D, Hq, Hk, dtype_code, v34_BQ, v34_BK, v34_WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v34_bwdq_sparse_mtx);
      auto it = v34_bwdq_sparse_pipelines.find(key);
      if (it != v34_bwdq_sparse_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      pipeline = compile_v34_backward_pipeline(
          D, Hq, Hk, dtype_code, v34_BQ, v34_BK, v34_WM, scale_,
          [](NAAttentionKernel& k) { return k.createV34BackwardQuerySparseSource(); },
          "attention_bwd_q_sparse", mtl_device, causal_,
          "MFA_V34BWD_DUMP_SOURCE", "V34 bwd dQ sparse", nullptr);
      pipeline = cache_insert_or_release(v34_bwdq_sparse_pipelines, v34_bwdq_sparse_mtx, key, pipeline);
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

    v34_dispatch_bwd_query_sparse(
        pipeline, &enc, (int)N, (int)Nk, (int)Hq, (int)Hk,
        (int)B, (int)D, v34_BQ, v34_BK, v34_WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV34BwdQuerySparse*>(&other);
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
  if (q.ndim() != 4) throw std::runtime_error("V34 bwd dQ sparse: Q must be 4D");
  if (block_mask.ndim() != 2)
    throw std::runtime_error("V34 bwd dQ sparse: block_mask must be 2-D");
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
      std::make_shared<MFAV34BwdQuerySparse>(s, scale, causal),
      {qc, kc, vc, oc, lsec, dOc, dvc, bmc});
  return outs[0];
}


// ─────────────────────────────────────────────────────────────────────
// MFAV34BwdDKSparse — dK split sparse
// ─────────────────────────────────────────────────────────────────────
struct V34BwdKSparseKey {
  int D, Hq, Hk, dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V34BWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  bool operator==(const V34BwdKSparseKey& o) const {
    return D == o.D && Hq == o.Hq && Hk == o.Hk
        && dtype_code == o.dtype_code
        && BQ == o.BQ && BK == o.BK && WM == o.WM
        && causal == o.causal
        && scale == o.scale;
  }
};
struct V34BwdKSparseKeyHash {
  size_t operator()(const V34BwdKSparseKey& k) const {
    size_t h = std::hash<int>{}(k.D);
    h ^= std::hash<int>{}(k.Hq) << 1;
    h ^= std::hash<int>{}(k.Hk) << 2;
    h ^= std::hash<int>{}(k.dtype_code) << 3;
    h ^= std::hash<uint16_t>{}(k.BQ) << 4;
    h ^= std::hash<uint16_t>{}(k.BK) << 5;
    h ^= std::hash<uint16_t>{}(k.WM) << 6;
    h ^= std::hash<bool>{}(k.causal) << 7;
    h ^= std::hash<float>{}(k.scale) << 16;
    return h;
  }
};
namespace {
std::mutex v34_bwdk_sparse_mtx;
std::unordered_map<V34BwdKSparseKey, void*, V34BwdKSparseKeyHash>
    v34_bwdk_sparse_pipelines;
}

class MFAV34BwdDKSparse : public mlx::core::Primitive {
 public:
  MFAV34BwdDKSparse(mlx::core::Stream s, float scale, uint16_t wm, bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), wm_(wm), causal_(causal) {}

  const char* name() const override { return "MFAV34BwdDKSparse"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV34BwdDKSparse: CPU eval not supported");
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
      throw std::runtime_error("V34 bwd dK sparse: D must be 64 or 128");
    if (block_mask.ndim() != 2)
      throw std::runtime_error("V34 bwd dK sparse: block_mask must be 2-D");

    unsigned short BQ = 64;
    unsigned short BK = 32;
    uint16_t WM = wm_;
    if (const char* e = std::getenv("MFA_V34BWDK_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDK_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDK_WM"))
      WM = (uint16_t)std::atoi(e);
    // v2.50 Prompt 5f Phase A — KD-1 fix: enforce mask shape match.
    {
      const int expected_NQ = (N + BQ - 1) / BQ;
      const int expected_NK = (Nk + BK - 1) / BK;
      const int mask_NQ = block_mask.shape(-2);
      const int mask_NK = block_mask.shape(-1);
      if (mask_NQ != expected_NQ || mask_NK != expected_NK) {
        std::ostringstream oss;
        oss << "V34 bwd dK sparse: block_mask shape ["
            << mask_NQ << ", " << mask_NK << "] does not match expected ["
            << expected_NQ << ", " << expected_NK << "] for tile geometry "
            << "(BQ=" << BQ << ", BK=" << BK << ") at qL=" << N
            << " kL=" << Nk << ".  See _convert_mask_for_v34_bwd_kernel "
            << "in mlx_mfa/attention.py (KD-1 resolution).";
        throw std::runtime_error(oss.str());
      }
    }

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V34 bwd dK sparse: only FP16/BF16");

    dkp.set_data(mlx::core::allocator::malloc(dkp.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V34BwdKSparseKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v34_bwdk_sparse_mtx);
      auto it = v34_bwdk_sparse_pipelines.find(key);
      if (it != v34_bwdk_sparse_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      pipeline = compile_v34_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV34BackwardDKSparseSource(); },
          "attention_bwd_dk_sparse", mtl_device, causal_,
          nullptr, nullptr, nullptr);
      pipeline = cache_insert_or_release(v34_bwdk_sparse_pipelines, v34_bwdk_sparse_mtx, key, pipeline);
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

    v34_dispatch_bwd_dk_sparse(pipeline, &enc, (int)N, (int)Nk, (int)Hq, (int)Hk,
                                (int)B, (int)D, BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV34BwdDKSparse*>(&other);
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
  if (q.ndim() != 4) throw std::runtime_error("V34 bwd dK sparse: Q must be 4D");
  if (block_mask.ndim() != 2)
    throw std::runtime_error("V34 bwd dK sparse: block_mask must be 2-D");
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
      std::make_shared<MFAV34BwdDKSparse>(s, scale, (uint16_t)wm, causal),
      {qc, kc, vc, oc, lsec, dOc, dvc, bmc});
  return outs[0];
}


// ─────────────────────────────────────────────────────────────────────
// MFAV34BwdFusedDKDVSparse — fused dKdV sparse
// ─────────────────────────────────────────────────────────────────────
struct V34BwdFSparseKey {
  int D, Hq, Hk, dtype_code;
  unsigned short BQ, BK;
  uint16_t WM;
  bool causal;
  // Repo review 2026-05: scale is baked into the Metal source
  // (DOT_SCALE / V34BWD_SCALE #defines) — it MUST be part of the
  // cache key or a second scale silently reuses the first's kernel.
  float scale;
  bool operator==(const V34BwdFSparseKey& o) const {
    return D == o.D && Hq == o.Hq && Hk == o.Hk
        && dtype_code == o.dtype_code
        && BQ == o.BQ && BK == o.BK && WM == o.WM
        && causal == o.causal
        && scale == o.scale;
  }
};
struct V34BwdFSparseKeyHash {
  size_t operator()(const V34BwdFSparseKey& k) const {
    size_t h = std::hash<int>{}(k.D);
    h ^= std::hash<int>{}(k.Hq) << 1;
    h ^= std::hash<int>{}(k.Hk) << 2;
    h ^= std::hash<int>{}(k.dtype_code) << 3;
    h ^= std::hash<uint16_t>{}(k.BQ) << 4;
    h ^= std::hash<uint16_t>{}(k.BK) << 5;
    h ^= std::hash<uint16_t>{}(k.WM) << 6;
    h ^= std::hash<bool>{}(k.causal) << 7;
    h ^= std::hash<float>{}(k.scale) << 16;
    return h;
  }
};
namespace {
std::mutex v34_bwdf_sparse_mtx;
std::unordered_map<V34BwdFSparseKey, void*, V34BwdFSparseKeyHash>
    v34_bwdf_sparse_pipelines;
}

class MFAV34BwdFusedDKDVSparse : public mlx::core::Primitive {
 public:
  MFAV34BwdFusedDKDVSparse(mlx::core::Stream s, float scale, uint16_t wm,
                          bool causal = false)
      : mlx::core::Primitive(s), scale_(scale), wm_(wm), causal_(causal) {}

  const char* name() const override { return "MFAV34BwdFusedDKDVSparse"; }

  void eval_cpu(const std::vector<mlx::core::array>&,
                std::vector<mlx::core::array>&) override {
    throw std::runtime_error("MFAV34BwdFusedDKDVSparse: CPU eval not supported");
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
      throw std::runtime_error("V34 bwd fused-dKdV sparse: D must be 64 or 128");
    if (block_mask.ndim() != 2)
      throw std::runtime_error("V34 bwd fused-dKdV sparse: block_mask must be 2-D");

    unsigned short BQ = 64;
    unsigned short BK = 32;
    uint16_t WM = wm_;
    if (const char* e = std::getenv("MFA_V34BWDF_BQ"))
      BQ = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDF_BK"))
      BK = (unsigned short)std::atoi(e);
    if (const char* e = std::getenv("MFA_V34BWDF_WM"))
      WM = (uint16_t)std::atoi(e);

    // v2.50 Prompt 5f Phase A — KD-1 fix: enforce mask shape match.
    {
      const int expected_NQ = (N + BQ - 1) / BQ;
      const int expected_NK = (Nk + BK - 1) / BK;
      const int mask_NQ = block_mask.shape(-2);
      const int mask_NK = block_mask.shape(-1);
      if (mask_NQ != expected_NQ || mask_NK != expected_NK) {
        std::ostringstream oss;
        oss << "V34 bwd fused-dKdV sparse: block_mask shape ["
            << mask_NQ << ", " << mask_NK << "] does not match expected ["
            << expected_NQ << ", " << expected_NK << "] for tile geometry "
            << "(BQ=" << BQ << ", BK=" << BK << ") at qL=" << N
            << " kL=" << Nk << ".  See _convert_mask_for_v34_bwd_kernel "
            << "in mlx_mfa/attention.py (KD-1 resolution).";
        throw std::runtime_error(oss.str());
      }
    }

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V34 bwd fused-dKdV sparse: only FP16/BF16");

    dkp.set_data(mlx::core::allocator::malloc(dkp.nbytes()));
    dvp.set_data(mlx::core::allocator::malloc(dvp.nbytes()));

    auto& dev = mlx::core::metal::device(stream().device);
    void* mtl_device = dev.mtl_device();

    V34BwdFSparseKey key{D, Hq, Hk, dtype_code, BQ, BK, WM, causal_, scale_};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v34_bwdf_sparse_mtx);
      auto it = v34_bwdf_sparse_pipelines.find(key);
      if (it != v34_bwdf_sparse_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      pipeline = compile_v34_backward_pipeline(
          D, Hq, Hk, dtype_code, BQ, BK, WM, scale_,
          [](NAAttentionKernel& k) { return k.createV34BackwardFusedDKDVSparseSource(); },
          "attention_bwd_fused_dkdv_sparse", mtl_device, causal_,
          nullptr, nullptr, nullptr);
      pipeline = cache_insert_or_release(v34_bwdf_sparse_pipelines, v34_bwdf_sparse_mtx, key, pipeline);
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

    v34_dispatch_bwd_fused_dkdv_sparse(pipeline, &enc, (int)N, (int)Nk,
                                        (int)Hq, (int)Hk, (int)B, (int)D,
                                        BQ, BK, WM);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV34BwdFusedDKDVSparse*>(&other);
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
  if (q.ndim() != 4) throw std::runtime_error("V34 bwd fused-dKdV sparse: Q must be 4D");
  if (block_mask.ndim() != 2)
    throw std::runtime_error("V34 bwd fused-dKdV sparse: block_mask must be 2-D");
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
      std::make_shared<MFAV34BwdFusedDKDVSparse>(s, scale, (uint16_t)wm, causal),
      {qc, kc, vc, lsec, dOc, dvc, bmc});
  return {outs[0], outs[1]};
}

}  // namespace mlx_mfa
