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
    // CC Batch-1 Class B: this C++-extension kernel compiles with `#include
    // <metal_stdlib>` only — MLX's `bfloat16_t` typedef is NOT in scope here
    // (that's the mx.fast.metal_kernel surface). The native Metal-4 bf16 type is
    // `bfloat` (as used by every other C++-ext kernel: conv_nax/sage/sparse/gna).
    // The old `bfloat16_t` late-compile-failed ("unknown type name 'bfloat16_t'")
    // for the reachable bf16 paged-gather decode config.
    const char* dtype_str = is_f16 ? "half" : "bfloat";
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
    int num_blocks;
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
    // OOB guard: a logical block index past block_table's columns would read
    // garbage from beyond the block_table allocation (CC-03 secondary).
    if (log_blk >= p.max_blocks) {
        out[gid] = T(0.0f);
        return;
    }
    int tok_off  = kv_t % p.block_size;
    int phys_blk = block_table[b * p.max_blocks + log_blk];
    // OOB guard: -1 padding (unallocated page) AND out-of-range physical block
    // ids both contribute zero — never index the pool out of bounds (CC-03).
    if (phys_blk < 0 || phys_blk >= p.num_blocks) {
        out[gid] = T(0.0f);
        return;
    }

    // pool layout: [phys_blk][tok_off][h][d].  64-bit arithmetic: a large pool
    // (num_blocks * pool_block_stride) can exceed INT32_MAX (CC-03 secondary).
    long src = (long)phys_blk * p.pool_block_stride
            + (long)tok_off  * p.pool_tok_stride
            + (long)h        * p.D
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
            if (log_blk >= max_blocks_) continue;           // OOB guard (CC-03)
            int tok_off  = kv_t % block_size_;
            int phys_blk = table_ptr[b * max_blocks_ + log_blk];
            if (phys_blk < 0 || phys_blk >= num_blocks_) continue;  // -1 padding + OOB
            for (int h = 0; h < H_; h++) {
                for (int d = 0; d < D_; d++) {
                    long src = (long)phys_blk * (block_size_ * H_ * D_)
                            + (long)tok_off  * (H_ * D_)
                            + (long)h        * D_
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
        false,       // has_attn_bias
        0,           // attn_bias_mode
        false,       // has_window
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
    params.num_blocks      = num_blocks_;
    params.out_batch_stride  = H_ * max_kv_len_ * D_;
    params.out_head_stride   = max_kv_len_ * D_;
    params.pool_block_stride = block_size_ * H_ * D_;
    params.pool_tok_stride   = H_ * D_;

    auto& enc = mlx::core::metal::get_command_encoder(stream());
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

    const int num_blocks = pool.shape(0);
    const int block_size = pool.shape(1);
    const int H          = pool.shape(2);
    const int D          = pool.shape(3);
    const int B          = block_table.shape(0);
    const int max_blocks = block_table.shape(1);

    // CX-02 (volet C, host half): the kernel reads seq_lens[b] for b in [0,B)
    // where B = block_table.shape(0).  If seq_lens is shorter than B it reads
    // out of bounds (silent finite-wrong / fault).  Enforce the host invariant.
    if (seq_lens.shape(0) != B) {
        throw std::invalid_argument(
            "mfa_paged_kv_gather: seq_lens length must equal block_table batch "
            "size B (seq_lens.shape[0]=" + std::to_string(seq_lens.shape(0)) +
            " vs B=" + std::to_string(B) + ").");
    }
    // CX-05 (volet C): the kernel reinterprets block_table / seq_lens buffers as
    // int32 (data<int32_t>()).  A float/int64 array's bits read as int32 →
    // silent finite-wrong indices.  Require int32 metadata.
    if (block_table.dtype() != int32) {
        throw std::invalid_argument(
            "mfa_paged_kv_gather: block_table must be int32 (the kernel reads it "
            "as int32; a different dtype reads garbage indices).");
    }
    if (seq_lens.dtype() != int32) {
        throw std::invalid_argument(
            "mfa_paged_kv_gather: seq_lens must be int32 (the kernel reads it as "
            "int32; a different dtype reads garbage lengths).");
    }
    // CAPACITY-SEAM (Phase 2 grid, wrapper-vs-raw parity): the public wrapper applies
    // the logical capacity guard (seq_lens <= max_blocks*block_size) before its gather
    // call; the raw gather did not — an over-capacity seq_lens silently zero-filled
    // the tail (finite-wrong dilution). Same cheap structural contract as the shared
    // assert_paged_capacity helper (inlined here: separate translation unit).
    if (seq_lens.size() > 0 && max_blocks > 0 && block_size > 0) {
        array sl = seq_lens; mlx::core::eval(sl);
        const int32_t* p = sl.data<int32_t>();
        const int n = static_cast<int>(sl.shape(0));
        int32_t smin = p[0], smax = p[0];
        for (int i = 1; i < n; ++i) { if (p[i] < smin) smin = p[i]; if (p[i] > smax) smax = p[i]; }
        if (smin < 0)
            throw std::invalid_argument("mfa_paged_kv_gather: seq_lens must be >= 0; got min="
                                        + std::to_string(smin) + ".");
        const long long cap = static_cast<long long>(max_blocks) * static_cast<long long>(block_size);
        if (static_cast<long long>(smax) > cap)
            throw std::invalid_argument(
                "mfa_paged_kv_gather: seq_lens max (" + std::to_string(smax) +
                ") exceeds max_blocks*block_size = " + std::to_string(max_blocks) + "*" +
                std::to_string(block_size) + " = " + std::to_string(cap) +
                " (a logical block index would run past the block_table columns).");
    }

    mlx::core::Shape out_shape = {B, H, max_kv_len, D};
    auto st = to_stream(s);

    auto outputs = array::make_arrays(
        {out_shape},
        {pool.dtype()},
        std::make_shared<MFAPagedKVGather>(
            st, B, H, D, block_size, max_blocks, max_kv_len, num_blocks),
        {pool, block_table, seq_lens});
    return outputs[0];
}
