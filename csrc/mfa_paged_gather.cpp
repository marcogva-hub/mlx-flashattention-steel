// mfa_paged_gather.cpp — Metal paged KV gather kernel (Track EB, v0.9.3)
#include "mfa_paged_gather.hpp"
#include "shader_cache.hpp"

#include <mlx/mlx.h>
#include <mlx/backend/metal/device.h>

#include <sstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>

using namespace mlx::core;
using namespace mlx_mfa;

// ---------------------------------------------------------------------------
// Metal source generation
// ---------------------------------------------------------------------------

std::string mlx_mfa::generate_paged_kv_gather_source(bool is_f16) {
    const char* dtype_str = is_f16 ? "half" : "bfloat16_t";
    std::ostringstream ss;

    ss << "#include <metal_stdlib>\n"
       << "using namespace metal;\n\n"
       << "typedef " << dtype_str << " T;\n\n";

    ss << R"(struct PagedGatherParams {
    int B;
    int H;
    int D;
    int block_size;
    int max_blocks;
    int max_kv_len;
    int out_batch_stride;
    int out_head_stride;
    int pool_block_stride;
    int pool_tok_stride;
};

// One thread per output element.
// Decodes (b, h, kv_t, d) from flat gid, looks up the physical block, copies.
// Writes T(0) for padding positions (kv_t >= seq_lens[b]) and sentinel blocks.
kernel void paged_kv_gather(
    const device T*   pool         [[buffer(0)]],
    device T*         out          [[buffer(1)]],
    const device int* block_table  [[buffer(2)]],
    const device int* seq_lens     [[buffer(3)]],
    const constant PagedGatherParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int total = p.B * p.H * p.max_kv_len * p.D;
    if ((int)gid >= total) return;

    int tmp  = (int)gid;
    int d    = tmp % p.D;             tmp /= p.D;
    int kv_t = tmp % p.max_kv_len;   tmp /= p.max_kv_len;
    int h    = tmp % p.H;             tmp /= p.H;
    int b    = tmp;

    if (kv_t >= seq_lens[b]) {
        out[gid] = T(0.0f);
        return;
    }

    int log_blk  = kv_t / p.block_size;
    int tok_off  = kv_t % p.block_size;
    int phys_blk = block_table[b * p.max_blocks + log_blk];
    if (phys_blk < 0) {
        out[gid] = T(0.0f);
        return;
    }

    // pool layout: [phys_blk][tok_off][h][d]
    int src = phys_blk * p.pool_block_stride
            + tok_off  * p.pool_tok_stride
            + h        * p.D
            + d;
    out[gid] = pool[src];
}
)";
    return ss.str();
}

// ---------------------------------------------------------------------------
// CPU fallback (needed for Primitive base class)
// ---------------------------------------------------------------------------

void MFAPagedKVGather::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    const auto& pool        = inputs[0];
    const auto& block_table = inputs[1];
    const auto& seq_lens    = inputs[2];
    auto&       out         = outputs[0];

    out.set_data(allocator::malloc(out.nbytes()));
    std::memset(out.data<uint8_t>(), 0, out.nbytes());

    // Works for both f16 and bf16 (both are 16-bit types stored as uint16)
    const auto* pool_ptr  = pool.data<uint16_t>();
    const auto* table_ptr = block_table.data<int32_t>();
    const auto* lens_ptr  = seq_lens.data<int32_t>();
    auto*       out_ptr   = out.data<uint16_t>();

    for (int b = 0; b < B_; b++) {
        int kv_len = lens_ptr[b];
        for (int kv_t = 0; kv_t < kv_len; kv_t++) {
            int log_blk  = kv_t / block_size_;
            int tok_off  = kv_t % block_size_;
            int phys_blk = table_ptr[b * max_blocks_ + log_blk];
            if (phys_blk < 0) continue;
            for (int h = 0; h < H_; h++) {
                for (int d = 0; d < D_; d++) {
                    int src = phys_blk * (block_size_ * H_ * D_)
                            + tok_off  * (H_ * D_)
                            + h        * D_
                            + d;
                    int dst = b * (H_ * max_kv_len_ * D_)
                            + h * (max_kv_len_ * D_)
                            + kv_t * D_
                            + d;
                    out_ptr[dst] = pool_ptr[src];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU dispatch (MLX Metal encoder)
// ---------------------------------------------------------------------------

void MFAPagedKVGather::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    const auto& pool        = inputs[0];
    const auto& block_table = inputs[1];
    const auto& seq_lens    = inputs[2];
    auto&       out         = outputs[0];

    out.set_data(allocator::malloc(out.nbytes()));

    const bool is_f16 = (pool.dtype() == float16);

    using KK = ShaderCache::KernelKey;
    // The kernel type uniquely identifies the paged gather; all other fields
    // are used only for the cache hash/equality check (not for shader gen).
    KK key{
        KK::KernelType::PagedKVGather,
        D_,          // head_dim
        block_size_, // repurposed as block_size for cache key
        max_blocks_, // repurposed as max_blocks for cache key
        0,           // block_d
        0,           // n_warps
        false,       // causal
        false,       // sparse
        false,       // is_m3_plus
        false,       // has_rope
        false,       // rope_interleaved
        false,       // has_softcap
        false,       // has_alibi
        is_f16 ? uint8_t(0) : uint8_t(1),
        1            // gqa_factor (unused)
    };

    auto& dev = metal::device(stream().device);
    void* raw = ShaderCache::get().get_or_compile(key, dev.mtl_device());
    auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw);

    PagedGatherParams params{};
    params.B               = B_;
    params.H               = H_;
    params.D               = D_;
    params.block_size      = block_size_;
    params.max_blocks      = max_blocks_;
    params.max_kv_len      = max_kv_len_;
    params.out_batch_stride  = H_ * max_kv_len_ * D_;
    params.out_head_stride   = max_kv_len_ * D_;
    params.pool_block_stride = block_size_ * H_ * D_;
    params.pool_tok_stride   = H_ * D_;

    auto& enc = dev.get_command_encoder(stream().index);
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_array(pool,        0);
    enc.set_output_array(out,        1);
    enc.set_input_array(block_table, 2);
    enc.set_input_array(seq_lens,    3);
    enc.set_bytes(params,            4);

    int total = B_ * H_ * max_kv_len_ * D_;
    int tgp   = std::min(256, (int)pipeline->maxTotalThreadsPerThreadgroup());
    int ngrp  = (total + tgp - 1) / tgp;
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)ngrp, 1, 1),
        MTL::Size::Make((size_t)tgp, 1, 1));
}

// ---------------------------------------------------------------------------
// Public C++ API
// ---------------------------------------------------------------------------

array mlx_mfa::mfa_paged_kv_gather(
    const array& pool,
    const array& block_table,
    const array& seq_lens,
    int max_kv_len,
    StreamOrDevice s)
{
    if (pool.ndim() != 4) {
        throw std::invalid_argument(
            "mfa_paged_kv_gather: pool must be 4-D [num_blocks, block_size, H, D]");
    }
    if (block_table.ndim() != 2) {
        throw std::invalid_argument(
            "mfa_paged_kv_gather: block_table must be 2-D [B, max_blocks]");
    }
    if (seq_lens.ndim() != 1) {
        throw std::invalid_argument(
            "mfa_paged_kv_gather: seq_lens must be 1-D [B]");
    }

    const int block_size = pool.shape(1);
    const int H          = pool.shape(2);
    const int D          = pool.shape(3);
    const int B          = block_table.shape(0);
    const int max_blocks = block_table.shape(1);

    mlx::core::Shape out_shape = {B, H, max_kv_len, D};
    auto st = to_stream(s);

    auto outputs = array::make_arrays(
        {out_shape},
        {pool.dtype()},
        std::make_shared<MFAPagedKVGather>(
            st, B, H, D, block_size, max_blocks, max_kv_len),
        {pool, block_table, seq_lens});
    return outputs[0];
}
