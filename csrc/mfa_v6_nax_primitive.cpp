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
                                bool isCausal) {
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

  // Tile dimensions: Phase 3B autoresearch overrides via env vars.
  // BLOCK_R = parallelization (rows per simdgroup) - default 32
  // BLOCK_C = traversal block (K columns) - default 32
  // executionSIMDGroups - default 4
  // BLOCK_D = head dimension (always full HEAD_DIM in v1)
  unsigned short BQ = 32, BK = 32;
  uint16_t exec_sg = 4;
  if (const char* env_r = std::getenv("MFA_V6_BLOCK_R")) BQ = (unsigned short)std::atoi(env_r);
  if (const char* env_c = std::getenv("MFA_V6_BLOCK_C")) BK = (unsigned short)std::atoi(env_c);
  if (const char* env_sg = std::getenv("MFA_V6_EXEC_SG")) exec_sg = (uint16_t)std::atoi(env_sg);
  simd::ushort3 blockDims =
      simd::make_ushort3(BQ, BK, (unsigned short)head_dim);

  NAAttentionKernelDescriptor desc(
      blockDims, (unsigned short)head_dim, (unsigned short)Hq,
      (unsigned short)Hk, /*executionSIMDGroups=*/exec_sg,
      /*checkCEdge1=*/true, mp, AttentionKernelType::forward,
      /*scale=*/1.0f / std::sqrt((float)head_dim),
      /*bypassThreadgroupMemory=*/false,
      /*isCausal=*/isCausal, /*masked=*/false);

  NAAttentionKernel kern(desc);
  return kern.source;
}

}  // namespace

// MFAV6Forward — Primitive for V6 NAX forward attention.
class MFAV6Forward : public mlx::core::Primitive {
public:
  struct Params {
    bool causal;
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

    // Inputs arrive in kernel layout [B, N, H, D] (transposed by caller).
    int B = q.shape(0);
    int N = q.shape(1);
    int Hq = q.shape(2);
    int D = q.shape(3);
    int Nk = k.shape(1);
    int Hk = k.shape(2);

    int dtype_code;
    if (q.dtype() == mlx::core::float16) dtype_code = 0;
    else if (q.dtype() == mlx::core::bfloat16) dtype_code = 1;
    else throw std::runtime_error("V6: only FP16/BF16");

    out.set_data(mlx::core::allocator::malloc(out.nbytes()));
    lse.set_data(mlx::core::allocator::malloc(lse.nbytes()));

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
    if (const char* env_r = std::getenv("MFA_V6_BLOCK_R")) BQ = (unsigned short)std::atoi(env_r);
    if (const char* env_c = std::getenv("MFA_V6_BLOCK_C")) BK = (unsigned short)std::atoi(env_c);
    if (const char* env_sg = std::getenv("MFA_V6_EXEC_SG")) executionSIMDGroups = (uint16_t)std::atoi(env_sg);

    // Include tile params in cache key so different configs get different pipelines.
    V6Key key{D, Hq, Hk, dtype_code, params_.causal,
              R + ((uint32_t)BQ << 24), C + ((uint32_t)BK << 24),
              qbs + ((uint32_t)executionSIMDGroups << 24),
              kbs, vbs, obs};
    void* pipeline = nullptr;
    {
      std::lock_guard<std::mutex> lock(v6_mtx);
      auto it = v6_pipelines.find(key);
      if (it != v6_pipelines.end()) pipeline = it->second;
    }
    if (!pipeline) {
      std::string src = generate_v6_source(
          D, Hq, Hk, dtype_code, params_.causal);
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
    return p && p->params_.causal == params_.causal;
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
  // Transpose [B,H,N,D] -> [B,N,H,D] for Draw Things kernel layout
  auto q_bnhd = mlx::core::transpose(q, std::vector<int>{0, 2, 1, 3}, s);
  auto k_bnhd = mlx::core::transpose(k, std::vector<int>{0, 2, 1, 3}, s);
  auto v_bnhd = mlx::core::transpose(v, std::vector<int>{0, 2, 1, 3}, s);
  auto qc = mlx::core::contiguous(q_bnhd, false, s);
  auto kc = mlx::core::contiguous(k_bnhd, false, s);
  auto vc = mlx::core::contiguous(v_bnhd, false, s);

  MFAV6Forward::Params params{causal};
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
