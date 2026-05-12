/// Sprint B Sparse Attention NAX - implementation.
///
/// Phase 1.1: per-thread-Q-row FA-2 kernel with block-mask skip. FP16, BT in
/// {16, 32}, 2-D mask, causal=false.
///
/// Phase 1.2 (this version):
///   - dtype: float16 + bfloat16
///   - block_tile: 16, 32, 64
///   - mask ndim: 2 (NQ, NK), 3 (Hq, NQ, NK), 4 (B, Hq, NQ, NK)
///   - causal: false + true (per-tile skip future + within-tile triangular)
///   - asymmetric qL != kL (cross-attention) supported
///
/// Phase 1.3 will swap inner GEMMs to mpp::tensor_ops::matmul2d for BT > 64
/// and to lift register-pressure constraints.

#include "mfa_sparse_attention.hpp"

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/utils.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace mlx_mfa {

namespace {

const std::string SPARSE_HEADER = R"(
#include <metal_stdlib>
using namespace metal;
)";

// bfloat is provided directly by <metal_stdlib> on Apple Silicon Metal SDK;
// no separate <metal_bf16> header. Kept as named alias for clarity.
const std::string& SPARSE_HEADER_BF16 = SPARSE_HEADER;

// JIT shader source. Per-thread Q-row processing inside a per-(b, hq, q_tile)
// threadgroup. Online softmax. Block mask scanned at K-tile granularity.
//
// dtype_str: "half" or "bfloat" - Metal Shading Language scalar type
// mask_ndim: 2 (NQ, NK), 3 (Hq, NQ, NK), 4 (B, Hq, NQ, NK)
// causal: when true emit per-tile-skip + within-tile triangular mask
std::string sparse_kernel_source(int B, int Hq, int Hk, int qL, int kL, int D,
                                  int BT, int NQ, int NK, float scale,
                                  const std::string& dtype_str,
                                  int mask_ndim, bool causal) {
  int gqa_factor = Hq / Hk;
  // Offset expression into block_mask for this (b, hq, q_tile).
  std::string mask_base_expr;
  if (mask_ndim == 2) {
    mask_base_expr = "block_mask + q_tile * cNK";
  } else if (mask_ndim == 3) {
    mask_base_expr = "block_mask + hq * cNQ * cNK + q_tile * cNK";
  } else {  // 4
    mask_base_expr = "block_mask + b * cHq * cNQ * cNK + hq * cNQ * cNK + q_tile * cNK";
  }

  std::ostringstream os;
  os << "    // Per-shape constants compiled in\n"
     << "    constexpr uint cB        = " << B << ";\n"
     << "    constexpr uint cHq       = " << Hq << ";\n"
     << "    constexpr uint cHk       = " << Hk << ";\n"
     << "    constexpr uint cQL       = " << qL << ";\n"
     << "    constexpr uint cKL       = " << kL << ";\n"
     << "    constexpr uint cD        = " << D << ";\n"
     << "    constexpr uint cBT       = " << BT << ";\n"
     << "    constexpr uint cNQ       = " << NQ << ";\n"
     << "    constexpr uint cNK       = " << NK << ";\n"
     << "    constexpr uint cGQA      = " << gqa_factor << ";\n"
     << "    constexpr float cSCALE   = " << scale << "f;\n"
     << "    constexpr float NEG_INF  = -1e30f;\n"
     << "\n"
     << "    const uint row_in_tile = thread_position_in_threadgroup.x;\n"
     << "    const uint hq          = threadgroup_position_in_grid.y;\n"
     << "    const uint bq_slot     = threadgroup_position_in_grid.z;\n"
     << "    const uint b           = bq_slot / cNQ;\n"
     << "    const uint q_tile      = bq_slot - b * cNQ;\n"
     << "    if (b >= cB || hq >= cHq || q_tile >= cNQ) return;\n"
     << "    if (row_in_tile >= cBT) return;\n"
     << "\n"
     << "    const uint hk          = hq / cGQA;\n"
     << "    const uint q_abs       = q_tile * cBT + row_in_tile;\n"
     << "\n"
     << "    // Base pointers (BHND row-major)\n"
     << "    device const " << dtype_str << "* Q_base = Q\n"
     << "        + b  * cHq * cQL * cD\n"
     << "        + hq *       cQL * cD\n"
     << "        + q_abs *          cD;\n"
     << "    device const " << dtype_str << "* K_b_hk = K\n"
     << "        + b  * cHk * cKL * cD\n"
     << "        + hk *       cKL * cD;\n"
     << "    device const " << dtype_str << "* V_b_hk = V\n"
     << "        + b  * cHk * cKL * cD\n"
     << "        + hk *       cKL * cD;\n"
     << "    device const bool* M_base = " << mask_base_expr << ";\n"
     << "    device "       << dtype_str << "* O_base = O\n"
     << "        + b  * cHq * cQL * cD\n"
     << "        + hq *       cQL * cD\n"
     << "        + q_abs *          cD;\n"
     << "\n"
     << "    // Per-thread state\n"
     << "    float q_vec[cD];\n"
     << "    #pragma clang loop unroll(full)\n"
     << "    for (uint d = 0; d < cD; ++d) q_vec[d] = float(Q_base[d]);\n"
     << "\n"
     << "    float m_run = NEG_INF;\n"
     << "    float l_run = 0.0f;\n"
     << "    float o_vec[cD];\n"
     << "    #pragma clang loop unroll(full)\n"
     << "    for (uint d = 0; d < cD; ++d) o_vec[d] = 0.0f;\n"
     << "\n"
     << "    // Block-mask scan + tile inner loop\n";
  if (causal) {
    os << "    // Causal: skip future tiles entirely (k_tile > q_tile).\n"
       << "    for (uint k_tile = 0; k_tile <= q_tile; ++k_tile) {\n";
  } else {
    os << "    for (uint k_tile = 0; k_tile < cNK; ++k_tile) {\n";
  }
  os << "        if (!M_base[k_tile]) continue;\n"
     << "\n"
     << "        // (1) Score row: s[k_col] = (q . K[k_tile*BT + k_col]) * scale\n"
     << "        float s[cBT];\n"
     << "        float m_tile = NEG_INF;\n"
     << "        #pragma clang loop unroll(full)\n"
     << "        for (uint kc = 0; kc < cBT; ++kc) {\n"
     << "            device const " << dtype_str
              << "* K_row = K_b_hk + (k_tile * cBT + kc) * cD;\n"
     << "            float acc = 0.0f;\n"
     << "            #pragma clang loop unroll(full)\n"
     << "            for (uint d = 0; d < cD; ++d) {\n"
     << "                acc += q_vec[d] * float(K_row[d]);\n"
     << "            }\n"
     << "            acc *= cSCALE;\n";
  if (causal) {
    // Within-tile triangular: on the diagonal tile (k_tile == q_tile), mask
    // any (row_in_tile, kc) where kc > row_in_tile (future-of-this-row).
    os << "            if (k_tile == q_tile && kc > row_in_tile) acc = NEG_INF;\n";
  }
  os << "            s[kc] = acc;\n"
     << "            m_tile = max(m_tile, acc);\n"
     << "        }\n"
     << "\n"
     << "        // (2) FA-2 online softmax update\n"
     << "        float m_new = max(m_run, m_tile);\n"
     << "        float corr  = exp(m_run - m_new);\n"
     << "        float l_tile = 0.0f;\n"
     << "        float p[cBT];\n"
     << "        #pragma clang loop unroll(full)\n"
     << "        for (uint kc = 0; kc < cBT; ++kc) {\n"
     << "            float pv = exp(s[kc] - m_new);\n"
     << "            p[kc] = pv;\n"
     << "            l_tile += pv;\n"
     << "        }\n"
     << "        l_run = l_run * corr + l_tile;\n"
     << "\n"
     << "        // (3) O accumulator update: O = O*corr + p @ V[k_tile,:]\n"
     << "        #pragma clang loop unroll(full)\n"
     << "        for (uint d = 0; d < cD; ++d) {\n"
     << "            float acc = o_vec[d] * corr;\n"
     << "            #pragma clang loop unroll(full)\n"
     << "            for (uint kc = 0; kc < cBT; ++kc) {\n"
     << "                acc += p[kc] * float(V_b_hk[(k_tile * cBT + kc) * cD + d]);\n"
     << "            }\n"
     << "            o_vec[d] = acc;\n"
     << "        }\n"
     << "\n"
     << "        m_run = m_new;\n"
     << "    }\n"
     << "\n"
     << "    // (4) Finalize: divide by l_run; write back. All-False row -> zero.\n"
     << "    if (l_run <= 0.0f) {\n"
     << "        #pragma clang loop unroll(full)\n"
     << "        for (uint d = 0; d < cD; ++d) O_base[d] = " << dtype_str << "(0.0f);\n"
     << "    } else {\n"
     << "        float inv_l = 1.0f / l_run;\n"
     << "        #pragma clang loop unroll(full)\n"
     << "        for (uint d = 0; d < cD; ++d) O_base[d] = " << dtype_str << "(o_vec[d] * inv_l);\n"
     << "    }\n";
  return os.str();
}

}  // namespace

mlx::core::array sparse_attention_forward(
    const mlx::core::array& Q,
    const mlx::core::array& K,
    const mlx::core::array& V,
    const mlx::core::array& block_mask,
    int block_tile,
    bool causal,
    float scale) {
  // Sanity asserts
  if (Q.ndim() != 4 || K.ndim() != 4 || V.ndim() != 4) {
    throw std::runtime_error("sparse_attention: Q, K, V must be 4-D (B, H, L, D)");
  }
  // Phase 1.2: float16 + bfloat16
  bool is_f16 = (Q.dtype() == mlx::core::float16);
  bool is_bf16 = (Q.dtype() == mlx::core::bfloat16);
  if (!is_f16 && !is_bf16) {
    throw std::runtime_error("sparse_attention: dtype must be float16 or bfloat16");
  }
  if (K.dtype() != Q.dtype() || V.dtype() != Q.dtype()) {
    throw std::runtime_error("sparse_attention: Q, K, V dtype must match");
  }
  if (block_mask.dtype() != mlx::core::bool_) {
    throw std::runtime_error("sparse_attention: block_mask must be bool");
  }
  int mask_ndim = static_cast<int>(block_mask.ndim());
  if (mask_ndim != 2 && mask_ndim != 3 && mask_ndim != 4) {
    throw std::runtime_error("sparse_attention: block_mask.ndim must be 2, 3, or 4");
  }
  if (block_tile != 16 && block_tile != 32 && block_tile != 64) {
    throw std::runtime_error("sparse_attention: block_tile must be 16, 32, or 64");
  }
  int B  = static_cast<int>(Q.shape(0));
  int Hq = static_cast<int>(Q.shape(1));
  int qL = static_cast<int>(Q.shape(2));
  int D  = static_cast<int>(Q.shape(3));
  int Hk = static_cast<int>(K.shape(1));
  int kL = static_cast<int>(K.shape(2));
  if (static_cast<int>(K.shape(0)) != B || static_cast<int>(V.shape(0)) != B) {
    throw std::runtime_error("sparse_attention: batch dim mismatch");
  }
  if (static_cast<int>(K.shape(3)) != D || static_cast<int>(V.shape(3)) != D) {
    throw std::runtime_error("sparse_attention: head_dim mismatch");
  }
  if (static_cast<int>(V.shape(1)) != Hk || static_cast<int>(V.shape(2)) != kL) {
    throw std::runtime_error("sparse_attention: K, V shape mismatch");
  }
  if (Hq % Hk != 0) {
    throw std::runtime_error("sparse_attention: Hq must be multiple of Hk (GQA)");
  }
  if (qL % block_tile != 0 || kL % block_tile != 0) {
    throw std::runtime_error("sparse_attention: qL, kL must be multiples of block_tile");
  }
  if (D != 64 && D != 128) {
    throw std::runtime_error("sparse_attention: head_dim must be 64 or 128");
  }
  int NQ = qL / block_tile;
  int NK = kL / block_tile;
  // Mask shape check per ndim
  if (mask_ndim == 2) {
    if (static_cast<int>(block_mask.shape(0)) != NQ ||
        static_cast<int>(block_mask.shape(1)) != NK) {
      throw std::runtime_error("sparse_attention: 2-D block_mask shape != (NQ, NK)");
    }
  } else if (mask_ndim == 3) {
    if (static_cast<int>(block_mask.shape(0)) != Hq ||
        static_cast<int>(block_mask.shape(1)) != NQ ||
        static_cast<int>(block_mask.shape(2)) != NK) {
      throw std::runtime_error("sparse_attention: 3-D block_mask shape != (Hq, NQ, NK)");
    }
  } else {  // 4
    if (static_cast<int>(block_mask.shape(0)) != B ||
        static_cast<int>(block_mask.shape(1)) != Hq ||
        static_cast<int>(block_mask.shape(2)) != NQ ||
        static_cast<int>(block_mask.shape(3)) != NK) {
      throw std::runtime_error("sparse_attention: 4-D block_mask shape != (B, Hq, NQ, NK)");
    }
  }
  if (causal && qL != kL) {
    throw std::runtime_error("sparse_attention: causal=true requires qL == kL");
  }
  if (scale <= 0.0f) {
    throw std::runtime_error("sparse_attention: scale must be > 0");
  }
  // Address-space precondition (Phase 1.1 carry-over)
  long long mask_bytes = 1LL;
  for (int i = 0; i < mask_ndim; ++i) mask_bytes *= static_cast<long long>(block_mask.shape(i));
  if (mask_bytes < 4096) {
    throw std::runtime_error(
        "sparse_attention: mask total bytes < 4096 (use larger qL, kL, "
        "or higher mask ndim). MLX inlines small buffers in constant "
        "address space; the JIT kernel emits device-qualified pointer.");
  }

  std::string dtype_str = is_f16 ? "half" : "bfloat";
  std::string name = "sparse_attn_" + dtype_str + "_" +
      std::to_string(B) + "_" + std::to_string(Hq) + "_" + std::to_string(Hk) +
      "_" + std::to_string(qL) + "_" + std::to_string(kL) + "_" +
      std::to_string(D) + "_BT" + std::to_string(block_tile) +
      "_M" + std::to_string(mask_ndim) +
      (causal ? "_c" : "_nc");

  const std::string& header = is_bf16 ? SPARSE_HEADER_BF16 : SPARSE_HEADER;
  auto kernel = mlx::core::fast::metal_kernel(
      name,
      {"Q", "K", "V", "block_mask"},
      {"O"},
      sparse_kernel_source(B, Hq, Hk, qL, kL, D, block_tile, NQ, NK, scale,
                            dtype_str, mask_ndim, causal),
      header,
      /*ensure_row_contiguous=*/true,
      /*atomic_outputs=*/false);

  int bt = block_tile;
  int b_nq = B * NQ;
  auto outs = kernel(
      {Q, K, V, block_mask},
      {mlx::core::Shape{B, Hq, qL, D}},
      {Q.dtype()},
      {bt, Hq, b_nq},
      {bt, 1, 1},
      {},
      std::nullopt,
      false,
      mlx::core::default_stream(mlx::core::Device::gpu));
  return outs[0];
}

}  // namespace mlx_mfa
