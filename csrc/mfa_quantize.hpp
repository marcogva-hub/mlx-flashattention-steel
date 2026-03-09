// mfa_quantize.hpp — Fused per-block INT8 quantization Metal kernel (Phase 4-A.1)
//
// Replaces the Python-side quantize_per_block() which dispatches 12+ separate
// Metal kernels per call with a single fused Metal dispatch.
//
// API:
//   mfa_quantize_per_block(x, block_size) -> (x_int8, scale)
//
// Input : [B, H, N, D]       fp16 or bf16
// Output: [B, H, N, D]       int8  — quantized values
//         [B, H, N_blocks, 1] float32 — per-block absmax / 127.0 scale
//
// Algorithm (per (b, h, block_idx) threadgroup):
//   1. Reduce absmax over (block_size, D) elements in threadgroup memory
//   2. scale = max(absmax / 127.0, 1e-8)
//   3. Write scale to scale buffer
//   4. Quantize: round(x / scale), clip to [-128, 127], cast to int8
//
// One kernel supports both f16 and bf16 (dtype selected at JIT compile time).
#pragma once

#include <mlx/mlx.h>
#include <mlx/primitives.h>
#include <vector>
#include <string>

namespace mlx_mfa {

// Parameters packed into Metal buffer(3).
struct QuantizeBlockParams {
    int B;
    int H;
    int N;
    int D;
    int block_size;
    int N_blocks;
};

// Generate the Metal source for the quantize_per_block kernel.
// dtype_str: "half" or "bfloat" (Metal type names)
std::string generate_quantize_per_block_source(const std::string& dtype_str);

// MLX Primitive: fused per-block INT8 quantization.
//
// Inputs:
//   [0] x  [B, H, N, D]  fp16 or bf16
//
// Outputs:
//   [0] x_int8  [B, H, N, D]      int8
//   [1] scale   [B, H, N_blocks, 1] float32
class MFAQuantizePerBlock : public mlx::core::Primitive {
public:
    explicit MFAQuantizePerBlock(
        mlx::core::Stream stream,
        int B, int H, int N, int D, int block_size, int N_blocks)
        : mlx::core::Primitive(stream),
          B_(B), H_(H), N_(N), D_(D),
          block_size_(block_size), N_blocks_(N_blocks) {}

    void eval_cpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    void eval_gpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    const char* name() const override { return "MFAQuantizePerBlock"; }

    bool is_equivalent(const mlx::core::Primitive& other) const override {
        const auto* o = dynamic_cast<const MFAQuantizePerBlock*>(&other);
        return o && o->B_ == B_ && o->H_ == H_ && o->N_ == N_
            && o->D_ == D_ && o->block_size_ == block_size_
            && o->N_blocks_ == N_blocks_;
    }

private:
    int B_, H_, N_, D_, block_size_, N_blocks_;
};

// C++ API: [B,H,N,D] fp16/bf16 → ([B,H,N,D] int8, [B,H,N_blocks,1] f32)
std::pair<mlx::core::array, mlx::core::array> mfa_quantize_per_block(
    const mlx::core::array& x,
    int block_size,
    mlx::core::StreamOrDevice s = {});

} // namespace mlx_mfa
