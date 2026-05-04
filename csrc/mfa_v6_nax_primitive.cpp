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
#include <string>
#include <unordered_map>
#include <mutex>

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
  bool operator==(const V6Key& o) const {
    return head_dim == o.head_dim && Hq == o.Hq && Hk == o.Hk &&
           dtype == o.dtype && isCausal == o.isCausal &&
           R == o.R && C == o.C &&
           qbs == o.qbs && kbs == o.kbs && vbs == o.vbs && obs == o.obs;
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
    return h;
  }
};

std::mutex v6_mtx;
std::unordered_map<V6Key, void*, V6KeyHash> v6_pipelines;

std::string generate_v6_source(int head_dim, int Hq, int Hk, int dtype_code,
                                bool isCausal, bool bhnd) {
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
  unsigned short BQ = 32, BK = 32;
  uint16_t exec_sg = 4;
  bool bypass_tgp = false;
  // Sprint 3.3 — Apple-style single-Otile dispatch.
  // Bench (M5 Max, 5 production shapes) showed a bimodal pattern:
  //   D=64  → single-Otile is faster: -25% on FlashVSR-dense, -44% on LTX2-cross
  //   D=128 → single-Otile regresses: +16-23% on SeedVR2-small/CogVideoX/SeedVR2-large
  // Root cause: double-buffer (cS_0/cS_1) hides PV-matmul latency for D=128 long
  // sequences (836+ K-tile iters); for D=64 short cross-attention the buffer
  // overhead dominates. Default: enable single-Otile only for D=64 non-GQA.
  bool single_otile = (head_dim == 64 && Hq == Hk);
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
  simd::ushort3 blockDims =
      simd::make_ushort3(BQ, BK, BD);

  NAAttentionKernelDescriptor desc(
      blockDims, (unsigned short)head_dim, (unsigned short)Hq,
      (unsigned short)Hk, /*executionSIMDGroups=*/exec_sg,
      /*checkCEdge1=*/true, mp, AttentionKernelType::forward,
      /*scale=*/1.0f / std::sqrt((float)head_dim),
      /*bypassThreadgroupMemory=*/bypass_tgp,
      /*isCausal=*/isCausal, /*masked=*/false);
  desc.singleOtileMode = single_otile;

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
    if (Hq == Hk) {  // non-GQA only for now
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
    uint32_t qbs = (uint32_t)(Hq * N * D);
    uint32_t kbs = (uint32_t)(Hk * Nk * D);
    uint32_t vbs = kbs;
    uint32_t obs = qbs;

    // Tile params (env vars override default for autoresearch).
    unsigned short BQ = 32, BK = 32;
    uint16_t executionSIMDGroups = 4;
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
      bool so_for_key = (D == 64 && Hq == Hk);
      if (const char* env_so = std::getenv("MFA_V6_NAX_SINGLE_OTILE"))
        so_for_key = (std::atoi(env_so) != 0);
      if (so_for_key) axis_flags |= 0x40;
    }

    // Include all tile + flag params in cache key.
    V6Key key{D, Hq, Hk, dtype_code, params_.causal,
              R + ((uint32_t)BQ << 24), C + ((uint32_t)BK << 24),
              qbs + ((uint32_t)executionSIMDGroups << 24) +
                    ((uint32_t)(bypass_tgp ? 1 : 0) << 31),
              kbs + ((uint32_t)BD << 16) + ((uint32_t)axis_flags << 24),
              vbs, obs};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v6_mtx);
      auto it = v6_pipelines.find(key);
      if (it != v6_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      std::string src = generate_v6_source(
          D, Hq, Hk, dtype_code, params_.causal, params_.bhnd);
      pipeline = v6_nax_compile_with_constants(
          src, "attention", mtl_device, R, C, qbs, kbs, vbs, obs);
      std::lock_guard<std::mutex> lock(v6_mtx);
      v6_pipelines[key] = pipeline;
    }

    unsigned short elem_size = 2;  // FP16/BF16 = 2 bytes
    unsigned short tgmem = BQ * BK * executionSIMDGroups * elem_size;

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_input_array(q, 0);
    enc.set_input_array(k, 1);
    enc.set_input_array(v, 2);
    enc.set_output_array(out, 3);
    enc.set_output_array(lse, 4);

    v6_nax_dispatch(
        pipeline, &enc,
        nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
        R, (uint32_t)Hq, (uint32_t)B,
        BQ, executionSIMDGroups, tgmem);
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto p = dynamic_cast<const MFAV6Forward*>(&other);
    return p && p->params_.causal == params_.causal
             && p->params_.bhnd == params_.bhnd;
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
    const mlx::core::array& v, bool causal) {
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
  const bool can_bhnd = (Hq_from_input == Hk_from_input);
  const bool bhnd = !legacy_opt_in && can_bhnd;
  MFAV6Forward::Params params{causal, bhnd};

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

}  // namespace mlx_mfa
