/// Sprint B Sparse Attention NAX — implementation.
///
/// Phase 1.1 scaffold: per-thread-Q-row FA-2 kernel with block-mask skip.
/// Each threadgroup processes one (b, hq, q_tile); each thread handles one
/// Q row within the tile. Online softmax in distributed registers. No
/// matmul2d yet (Phase 1.3 swaps in matmul2d once correctness is locked).
///
/// Layout assumptions (verified at entry):
///   Q: (B, Hq, qL, D) row-major
///   K, V: (B, Hk, kL, D) row-major, Hq % Hk == 0
///   block_mask: (NQ, NK) bool where NQ=qL/BT, NK=kL/BT
///
/// Phase 1.1 limits (relaxed in 1.2 / 1.3):
///   - dtype: float16 only
///   - block_tile: 16 or 32 only (register pressure)
///   - mask: 2-D only
///   - causal: false only

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

// JIT shader source. Per-thread Q-row processing inside a per-(b, hq, q_tile)
// threadgroup. Online softmax. Block mask scanned at K-tile granularity.
std::string sparse_kernel_source(int B, int Hq, int Hk, int qL, int kL, int D,
                                  int BT, int NQ, int NK, float scale) {
  int gqa_factor = Hq / Hk;
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
     << "    device const half* Q_base = Q\n"
     << "        + b  * cHq * cQL * cD\n"
     << "        + hq *       cQL * cD\n"
     << "        + q_abs *          cD;\n"
     << "    device const half* K_b_hk = K\n"
     << "        + b  * cHk * cKL * cD\n"
     << "        + hk *       cKL * cD;\n"
     << "    device const half* V_b_hk = V\n"
     << "        + b  * cHk * cKL * cD\n"
     << "        + hk *       cKL * cD;\n"
     << "    device const bool*  M_base = block_mask + q_tile * cNK;\n"
     << "    device half*        O_base = O\n"
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
     << "    // Block-mask scan + tile inner loop\n"
     << "    for (uint k_tile = 0; k_tile < cNK; ++k_tile) {\n"
     << "        if (!M_base[k_tile]) continue;\n"
     << "\n"
     << "        // (1) Score row: s[k_col] = (q · K[k_tile*BT + k_col]) * scale\n"
     << "        float s[cBT];\n"
     << "        float m_tile = NEG_INF;\n"
     << "        #pragma clang loop unroll(full)\n"
     << "        for (uint kc = 0; kc < cBT; ++kc) {\n"
     << "            device const half* K_row = K_b_hk + (k_tile * cBT + kc) * cD;\n"
     << "            float acc = 0.0f;\n"
     << "            #pragma clang loop unroll(full)\n"
     << "            for (uint d = 0; d < cD; ++d) {\n"
     << "                acc += q_vec[d] * float(K_row[d]);\n"
     << "            }\n"
     << "            acc *= cSCALE;\n"
     << "            s[kc] = acc;\n"
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
     << "    // (4) Finalize: divide by l_run; write back. All-False row → zero.\n"
     << "    if (l_run <= 0.0f) {\n"
     << "        #pragma clang loop unroll(full)\n"
     << "        for (uint d = 0; d < cD; ++d) O_base[d] = half(0.0f);\n"
     << "    } else {\n"
     << "        float inv_l = 1.0f / l_run;\n"
     << "        #pragma clang loop unroll(full)\n"
     << "        for (uint d = 0; d < cD; ++d) O_base[d] = half(o_vec[d] * inv_l);\n"
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
  // Sanity asserts (8 categories per design §4)
  if (Q.ndim() != 4 || K.ndim() != 4 || V.ndim() != 4) {
    throw std::runtime_error("sparse_attention: Q, K, V must be 4-D (B, H, L, D)");
  }
  if (Q.dtype() != mlx::core::float16 ||
      K.dtype() != mlx::core::float16 ||
      V.dtype() != mlx::core::float16) {
    throw std::runtime_error("sparse_attention: Phase 1.1 supports float16 only");
  }
  if (block_mask.dtype() != mlx::core::bool_) {
    throw std::runtime_error("sparse_attention: block_mask must be bool");
  }
  if (block_mask.ndim() != 2) {
    throw std::runtime_error("sparse_attention: Phase 1.1 supports 2-D mask only");
  }
  if (causal) {
    throw std::runtime_error("sparse_attention: Phase 1.1 supports causal=false only");
  }
  if (block_tile != 16 && block_tile != 32) {
    throw std::runtime_error("sparse_attention: Phase 1.1 supports BT ∈ {16, 32} only");
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
    throw std::runtime_error("sparse_attention: Phase 1.1 supports D ∈ {64, 128} only");
  }
  int NQ = qL / block_tile;
  int NK = kL / block_tile;
  if (static_cast<int>(block_mask.shape(0)) != NQ ||
      static_cast<int>(block_mask.shape(1)) != NK) {
    throw std::runtime_error("sparse_attention: block_mask shape != (NQ, NK)");
  }
  if (scale <= 0.0f) {
    throw std::runtime_error("sparse_attention: scale must be > 0");
  }
  // Phase 1.1 requires the bool mask to land in Metal device (not constant)
  // address space. MLX inlines buffers < ~4 KB as constant. Production shapes
  // (lcsa_small_seq4k and beyond) produce masks >= 16 KB -> device. Reject
  // smaller masks loudly so users hit a clear error instead of MSL compile
  // failures. Phase 1.2 will support both via emitted-qualifier branching.
  int mask_bytes = NQ * NK;  // bool = 1 byte per element
  if (mask_bytes < 4096) {
    throw std::runtime_error(
        "sparse_attention: Phase 1.1 requires NQ*NK >= 4096 (use qL, kL >= "
        "2048 with BT=32, or qL, kL >= 1024 with BT=16). Smaller masks "
        "trigger MLX constant-address-space inlining which Phase 1.1 does "
        "not yet handle.");
  }

  std::string name = "sparse_attn_" +
      std::to_string(B) + "_" + std::to_string(Hq) + "_" + std::to_string(Hk) +
      "_" + std::to_string(qL) + "_" + std::to_string(kL) + "_" +
      std::to_string(D) + "_BT" + std::to_string(block_tile);

  auto kernel = mlx::core::fast::metal_kernel(
      name,
      {"Q", "K", "V", "block_mask"},
      {"O"},
      sparse_kernel_source(B, Hq, Hk, qL, kL, D, block_tile, NQ, NK, scale),
      SPARSE_HEADER,
      /*ensure_row_contiguous=*/true,
      /*atomic_outputs=*/false);

  int bt = block_tile;
  int b_nq = B * NQ;
  // grid total threads = (BT, Hq, B*NQ); threadgroup = (BT, 1, 1)
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
