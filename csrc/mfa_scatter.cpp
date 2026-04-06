// mfa_scatter.cpp — In-place scatter write for paged KV pool (Phase 4-C.1+E.2)
#include "mfa_scatter.hpp"
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

std::string mlx_mfa::generate_scatter_kv_source(const std::string& dtype_str) {
    std::ostringstream ss;

    ss << "#include <metal_stdlib>\n"
       << "using namespace metal;\n\n";

    ss << "typedef " << dtype_str << " SKVDtype;\n\n";

    ss << R"(struct ScatterKVParams {
    int num_blocks;
    int block_size;
    int H;
    int D;
    int N_write;
};

// ---------------------------------------------------------------------------
// mfa_scatter_kv kernel
//
// Grid  : (ceil(total_pool_elems / THREADS), 1, 1)  threadgroups
// TG    : (THREADS, 1, 1)  threads  — use 256
//
// Each thread handles one element of the output pool.
// Default path: copy from pool_in.
// Scatter path: if (blk, off) matches any (blk_ids[n], blk_offs[n]),
//   write tokens[n, h, d] instead.
//
// Optimized for small N_write (decode: 1-4).
// ---------------------------------------------------------------------------
kernel void mfa_scatter_kv(
    device SKVDtype*        pool_out  [[buffer(0)]],
    const device SKVDtype*  pool_in   [[buffer(1)]],
    const device SKVDtype*  tokens    [[buffer(2)]],
    const device int*       blk_ids   [[buffer(3)]],
    const device int*       blk_offs  [[buffer(4)]],
    constant ScatterKVParams& p       [[buffer(5)]],
    uint3 gid_v [[thread_position_in_grid]])
{
    const int elem = (int)gid_v.x;
    const int total = p.num_blocks * p.block_size * p.H * p.D;
    if (elem >= total) return;

    // Decode linear element index → (blk, off, h, d)
    const int d   = elem % p.D;
    const int h   = (elem / p.D) % p.H;
    const int off = (elem / (p.D * p.H)) % p.block_size;
    const int blk = (elem / (p.D * p.H * p.block_size));

    // Default: copy from pool_in
    SKVDtype val = pool_in[elem];

    // Scatter override: scan N_write entries (small for decode)
    for (int n = 0; n < p.N_write; n++) {
        if (blk_ids[n] == blk && blk_offs[n] == off) {
            val = tokens[n * p.H * p.D + h * p.D + d];
            break;
        }
    }

    pool_out[elem] = val;
}
)";

    return ss.str();
}

// ---------------------------------------------------------------------------
// CPU fallback (reference implementation)
// ---------------------------------------------------------------------------

void MFAScatterKV::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    const auto& pool     = inputs[0];  // [num_blocks, block_size, H, D]
    const auto& tokens   = inputs[1];  // [N_write, H, D]
    const auto& blk_ids  = inputs[2];  // [N_write] int32
    const auto& blk_offs = inputs[3];  // [N_write] int32
    auto& pool_out       = outputs[0];

    const size_t nbytes = pool_out.nbytes();
    pool_out.set_data(allocator::malloc(nbytes));

    // Copy pool → pool_out
    std::memcpy(pool_out.data<void>(), pool.data<void>(), nbytes);

    const int H = H_, D = D_, bs = block_size_;

    // Scatter write each token
    for (int n = 0; n < N_write_; n++) {
        int blk = blk_ids.data<int32_t>()[n];
        int off = blk_offs.data<int32_t>()[n];
        if (blk < 0 || blk >= num_blocks_ || off < 0 || off >= bs) continue;

        // pool_out[blk, off, :, :] = tokens[n, :, :]
        size_t pool_offset = (static_cast<size_t>(blk) * bs + off) * H * D;
        size_t tok_offset  = static_cast<size_t>(n) * H * D;
        size_t elem_bytes  = static_cast<size_t>(H) * D * pool.itemsize();

        // Copy one [H, D] slice
        const uint8_t* src = reinterpret_cast<const uint8_t*>(tokens.data<void>())
                           + tok_offset * pool.itemsize();
        uint8_t*       dst = reinterpret_cast<uint8_t*>(pool_out.data<void>())
                           + pool_offset * pool.itemsize();
        std::memcpy(dst, src, elem_bytes);
    }
}

// ---------------------------------------------------------------------------
// GPU dispatch
// ---------------------------------------------------------------------------

void MFAScatterKV::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    const auto& pool     = inputs[0];
    const auto& tokens   = inputs[1];
    const auto& blk_ids  = inputs[2];
    const auto& blk_offs = inputs[3];
    auto& pool_out       = outputs[0];

    pool_out.set_data(allocator::malloc(pool_out.nbytes()));

    const bool is_f16 = (pool.dtype() == float16);

    using KK = ShaderCache::KernelKey;
    KK key{
        KK::KernelType::ScatterKV,
        D_,            // head_dim = D
        N_write_,      // block_q  = N_write (cache key dimension)
        block_size_,   // block_k  = block_size
        0, 0,          // block_d, n_warps — unused
        false, false, false, false, false, false, false,
        /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
        false,
        is_f16 ? uint8_t(0) : uint8_t(1),
        1
    };

    auto& dev = metal::device(stream().device);
    void* raw = ShaderCache::get().get_or_compile(key, dev.mtl_device());
    auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw);

    ScatterKVParams params{num_blocks_, block_size_, H_, D_, N_write_};

    const int total_elems = num_blocks_ * block_size_ * H_ * D_;
    constexpr int THREADS = 256;
    const int num_tg = (total_elems + THREADS - 1) / THREADS;

    auto& enc = dev.get_command_encoder(stream().index);
    enc.set_compute_pipeline_state(pipeline);
    enc.set_output_array(pool_out, 0);
    enc.set_input_array(pool,     1);
    enc.set_input_array(tokens,   2);
    enc.set_input_array(blk_ids,  3);
    enc.set_input_array(blk_offs, 4);
    enc.set_bytes(params,         5);

    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)num_tg, 1, 1),
        MTL::Size::Make(THREADS, 1, 1));
}

// ---------------------------------------------------------------------------
// Public C++ API
// ---------------------------------------------------------------------------

mlx::core::array mlx_mfa::mfa_scatter_kv(
    const array& pool,
    const array& tokens,
    const array& blk_ids,
    const array& blk_offs,
    StreamOrDevice s)
{
    if (pool.ndim() != 4)
        throw std::invalid_argument(
            "mfa_scatter_kv: pool must be 4-D [num_blocks, block_size, H_kv, D]");
    if (tokens.ndim() != 3)
        throw std::invalid_argument(
            "mfa_scatter_kv: tokens must be 3-D [N_write, H_kv, D]");
    if (blk_ids.ndim() != 1 || blk_offs.ndim() != 1)
        throw std::invalid_argument(
            "mfa_scatter_kv: blk_ids and blk_offs must be 1-D [N_write]");
    if (pool.dtype() != float16 && pool.dtype() != bfloat16)
        throw std::invalid_argument(
            "mfa_scatter_kv: only float16 and bfloat16 are supported");
    if (blk_ids.dtype() != int32 || blk_offs.dtype() != int32)
        throw std::invalid_argument(
            "mfa_scatter_kv: blk_ids and blk_offs must be int32");

    const int num_blocks  = pool.shape(0);
    const int block_size  = pool.shape(1);
    const int H           = pool.shape(2);
    const int D           = pool.shape(3);
    const int N_write     = tokens.shape(0);

    if (tokens.shape(1) != H || tokens.shape(2) != D)
        throw std::invalid_argument(
            "mfa_scatter_kv: tokens shape[1:] must match pool shape[2:]");
    if (blk_ids.shape(0) != N_write || blk_offs.shape(0) != N_write)
        throw std::invalid_argument(
            "mfa_scatter_kv: blk_ids/blk_offs length must match tokens.shape(0)");

    auto st = to_stream(s);

    // Ensure row-major layout.
    auto pool_c     = mlx::core::contiguous(pool,     false, st);
    auto tokens_c   = mlx::core::contiguous(tokens,   false, st);
    auto blk_ids_c  = mlx::core::contiguous(blk_ids,  false, st);
    auto blk_offs_c = mlx::core::contiguous(blk_offs, false, st);

    mlx::core::Shape out_shape = {num_blocks, block_size, H, D};

    auto outs = array::make_arrays(
        {out_shape},
        {pool.dtype()},
        std::make_shared<MFAScatterKV>(st, num_blocks, block_size, H, D, N_write),
        {pool_c, tokens_c, blk_ids_c, blk_offs_c});
    return outs[0];
}
