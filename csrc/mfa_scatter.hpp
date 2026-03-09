// mfa_scatter.hpp — In-place scatter write for paged KV pool (Phase 4-C.1+E.2)
//
// Replaces the Python-side pool rebuild loop (O(num_blocks) MLX ops per step)
// with a single Metal dispatch that copy-writes the pool in one pass.
//
// API:
//   mfa_scatter_kv(pool, tokens, blk_ids, blk_offs) -> new_pool
//
// pool:     [num_blocks, block_size, H_kv, D]  fp16 or bf16
// tokens:   [N_write, H_kv, D]                 same dtype
// blk_ids:  [N_write]                           int32  (physical block index)
// blk_offs: [N_write]                           int32  (slot within block)
// Output:   [num_blocks, block_size, H_kv, D]  copy of pool with scattered writes
//
// Metal kernel: one thread per pool element.
//   - Default: copy pool_in → pool_out
//   - Override: for any n in [0, N_write), if (blk, off) matches (blk_ids[n], blk_offs[n]),
//     write tokens[n, h, d] instead.
//
// Optimized for small N_write (decode: N_write = 1-4).
// For large N_write (prefill), the inner scan is O(N_write) per thread — use Python fallback.
#pragma once

#include <mlx/mlx.h>
#include <mlx/primitives.h>
#include <vector>
#include <string>

namespace mlx_mfa {

// Parameters packed into Metal buffer(5).
struct ScatterKVParams {
    int num_blocks;
    int block_size;
    int H;       // H_kv
    int D;
    int N_write;
};

// Generate the Metal source for the scatter_kv kernel.
// dtype_str: "half" or "bfloat" (Metal type names)
std::string generate_scatter_kv_source(const std::string& dtype_str);

// MLX Primitive: in-place scatter write into paged KV pool.
//
// Inputs:
//   [0] pool    [num_blocks, block_size, H_kv, D]  fp16 or bf16
//   [1] tokens  [N_write, H_kv, D]                 same dtype
//   [2] blk_ids  [N_write]                          int32
//   [3] blk_offs [N_write]                          int32
//
// Output:
//   [0] pool_out [num_blocks, block_size, H_kv, D]  same dtype
class MFAScatterKV : public mlx::core::Primitive {
public:
    explicit MFAScatterKV(
        mlx::core::Stream stream,
        int num_blocks, int block_size, int H, int D, int N_write)
        : mlx::core::Primitive(stream),
          num_blocks_(num_blocks), block_size_(block_size),
          H_(H), D_(D), N_write_(N_write) {}

    void eval_cpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    void eval_gpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    const char* name() const override { return "MFAScatterKV"; }

    bool is_equivalent(const mlx::core::Primitive& other) const override {
        const auto* o = dynamic_cast<const MFAScatterKV*>(&other);
        return o && o->num_blocks_ == num_blocks_ && o->block_size_ == block_size_
            && o->H_ == H_ && o->D_ == D_ && o->N_write_ == N_write_;
    }

private:
    int num_blocks_, block_size_, H_, D_, N_write_;
};

// C++ API:
//   pool:     [num_blocks, block_size, H_kv, D]  fp16 or bf16
//   tokens:   [N_write, H_kv, D]                 same dtype
//   blk_ids:  [N_write]                           int32
//   blk_offs: [N_write]                           int32
// Returns: [num_blocks, block_size, H_kv, D]
mlx::core::array mfa_scatter_kv(
    const mlx::core::array& pool,
    const mlx::core::array& tokens,
    const mlx::core::array& blk_ids,
    const mlx::core::array& blk_offs,
    mlx::core::StreamOrDevice s = {});

} // namespace mlx_mfa
