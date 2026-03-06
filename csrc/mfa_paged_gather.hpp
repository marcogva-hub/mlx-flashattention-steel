// mfa_paged_gather.hpp — Metal paged KV gather kernel (Track EB, v0.9.3)
//
// Materialises contiguous K/V tensors from a paged block pool in a single
// Metal dispatch, replacing the Python-level for-loop gather.
//
// Pool layout  : [num_blocks, block_size, H_kv, D]   (token-major within block)
// Output layout: [B, H_kv, max_kv_len, D]             (BHND — STEEL-ready)
//
// The kernel transposes [block_size, H_kv, D] → [H_kv, block_size, D] during
// the copy.  One thread per output element; 1D grid.
#pragma once

#include <mlx/mlx.h>
#include <mlx/primitives.h>
#include <vector>
#include <string>

namespace mlx_mfa {

// Parameters packed into Metal buffer(4)
struct PagedGatherParams {
    int B;
    int H;
    int D;
    int block_size;
    int max_blocks;
    int max_kv_len;
    int out_batch_stride;    // H * max_kv_len * D
    int out_head_stride;     // max_kv_len * D
    int pool_block_stride;   // block_size * H * D
    int pool_tok_stride;     // H * D
};

// Generate the Metal source for the paged KV gather kernel.
std::string generate_paged_kv_gather_source(bool is_f16);

// MLX Primitive: gathers K or V from the page pool into a contiguous BHND tensor.
//
// Inputs:
//   [0] pool         [num_blocks, block_size, H, D] f16 or bf16
//   [1] block_table  [B, max_blocks] int32
//   [2] seq_lens     [B] int32
//
// Output:
//   [0] out          [B, H, max_kv_len, D] same dtype as pool
class MFAPagedKVGather : public mlx::core::Primitive {
public:
    explicit MFAPagedKVGather(
        mlx::core::Stream stream,
        int B, int H, int D, int block_size, int max_blocks, int max_kv_len)
        : mlx::core::Primitive(stream),
          B_(B), H_(H), D_(D), block_size_(block_size),
          max_blocks_(max_blocks), max_kv_len_(max_kv_len) {}

    void eval_cpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    void eval_gpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    const char* name() const override { return "MFAPagedKVGather"; }

    bool is_equivalent(const mlx::core::Primitive& other) const override {
        const auto* o = dynamic_cast<const MFAPagedKVGather*>(&other);
        return o && o->B_ == B_ && o->H_ == H_ && o->D_ == D_
            && o->block_size_ == block_size_ && o->max_blocks_ == max_blocks_
            && o->max_kv_len_ == max_kv_len_;
    }

private:
    int B_, H_, D_, block_size_, max_blocks_, max_kv_len_;
};

// C++ API: gathers pool → contiguous [B, H, max_kv_len, D] tensor.
mlx::core::array mfa_paged_kv_gather(
    const mlx::core::array& pool,
    const mlx::core::array& block_table,
    const mlx::core::array& seq_lens,
    int max_kv_len,
    mlx::core::StreamOrDevice s = {});

} // namespace mlx_mfa
