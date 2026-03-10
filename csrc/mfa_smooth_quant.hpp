// mfa_smooth_quant.hpp — Fused smooth_k + quantize_per_block Metal kernel (Phase 1.1)
//
// Fuses two Python-level ops into one C++ primitive:
//   1. smooth_k:           k_mean = mean(K, axis=S);  K_smooth = K - k_mean
//   2. quantize_per_block: absmax per block → scale → round → clip → int8
//
// Benefits vs the two-call path:
//   - Eliminates the intermediate K_smooth [B,H,S,D] fp16 tensor (saves bandwidth).
//   - Reduces Python-level dispatch overhead (2 MLX ops → 0; 1 C++ → 1 C++).
//   - Returns k_mean alongside (k_int8, k_scale) so callers don't need a separate mean().
//
// API:
//   mfa_smooth_quantize_k(k, block_size) -> (k_int8, k_scale, k_mean)
//
// Input:  k  [B, H, S, D]   fp16 or bf16
// Output: k_int8  [B, H, S, D]         int8
//         k_scale [B, H, S_blocks, 1]  float32
//         k_mean  [B, H, 1, D]         float32
//
// GPU implementation: 2 Metal dispatches per eval_gpu call.
//   Pass 1 — mfa_smooth_k_mean:  reduce mean over S for each (B, H, D) channel.
//   Pass 2 — mfa_smooth_k_quant: subtract mean, absmax per block, quantize.
#pragma once

#include <mlx/mlx.h>
#include <mlx/primitives.h>
#include <string>
#include <vector>

namespace mlx_mfa {

// Metal source generators.
std::string generate_smooth_k_mean_source(const std::string& dtype_str);
std::string generate_smooth_k_quant_source(const std::string& dtype_str);

// MLX Primitive: fused smooth+quantize.
//
// Inputs:  [0] k   [B, H, S, D]  fp16 or bf16
// Outputs: [0] k_int8  [B, H, S, D]      int8
//          [1] k_scale [B, H, N_blocks, 1] float32
//          [2] k_mean  [B, H, 1, D]       float32
class MFASmoothQuantizeK : public mlx::core::Primitive {
public:
    explicit MFASmoothQuantizeK(
        mlx::core::Stream stream,
        int B, int H, int S, int D, int block_size, int N_blocks)
        : mlx::core::Primitive(stream),
          B_(B), H_(H), S_(S), D_(D),
          block_size_(block_size), N_blocks_(N_blocks) {}

    void eval_cpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    void eval_gpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    const char* name() const override { return "MFASmoothQuantizeK"; }

    bool is_equivalent(const mlx::core::Primitive& other) const override {
        const auto* o = dynamic_cast<const MFASmoothQuantizeK*>(&other);
        return o && o->B_ == B_ && o->H_ == H_ && o->S_ == S_
            && o->D_ == D_ && o->block_size_ == block_size_
            && o->N_blocks_ == N_blocks_;
    }

private:
    int B_, H_, S_, D_, block_size_, N_blocks_;
};

// C++ API: [B,H,S,D] fp16/bf16 → (k_int8, k_scale, k_mean)
std::tuple<mlx::core::array, mlx::core::array, mlx::core::array>
mfa_smooth_quantize_k(
    const mlx::core::array& k,
    int block_size,
    mlx::core::StreamOrDevice s = {});

} // namespace mlx_mfa
