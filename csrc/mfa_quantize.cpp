// mfa_quantize.cpp — Fused per-block INT8 quantization Metal kernel (Phase 4-A.1)
#include "mfa_quantize.hpp"
#include "shader_cache.hpp"

#include <mlx/mlx.h>
#include <mlx/backend/metal/device.h>

#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace mlx::core;
using namespace mlx_mfa;

// ---------------------------------------------------------------------------
// Metal source generation
// ---------------------------------------------------------------------------

std::string mlx_mfa::generate_quantize_per_block_source(const std::string& dtype_str) {
    std::ostringstream ss;

    ss << "#include <metal_stdlib>\n"
       << "using namespace metal;\n\n";

    ss << "typedef " << dtype_str << " QPBDtype;\n\n";

    ss << R"(struct QuantizeBlockParams {
    int B;
    int H;
    int N;
    int D;
    int block_size;
    int N_blocks;
};

// ---------------------------------------------------------------------------
// quantize_per_block kernel
//
// Grid  : (N_blocks, H, B)  threadgroups
// TG    : (THREADS, 1, 1)   threads — THREADS must be power of 2 (use 256)
//
// Phase 1: threadgroup tree-reduce to find absmax of the block.
// Phase 2: each thread quantizes its elements, writes int8 output.
// ---------------------------------------------------------------------------
kernel void mfa_quantize_per_block(
    const device QPBDtype*   x      [[buffer(0)]],
    device char*             x_int8 [[buffer(1)]],
    device float*            scale  [[buffer(2)]],
    constant QuantizeBlockParams& p [[buffer(3)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid_v [[thread_position_in_threadgroup]],
    uint3 tgs_v [[threads_per_threadgroup]])
{
    const int block_idx = (int)tgid.x;
    const int h         = (int)tgid.y;
    const int b         = (int)tgid.z;
    const uint tid = tid_v.x;
    const uint tgs = tgs_v.x;

    if (b >= p.B || h >= p.H || block_idx >= p.N_blocks) return;

    const int token_start = block_idx * p.block_size;
    const int n_tokens = min(p.block_size, p.N - token_start);
    const int n_elems  = n_tokens * p.D;
    const int base     = (b * p.H + h) * p.N * p.D + token_start * p.D;

    threadgroup float smem[256];

    float local_max = 0.0f;
    for (int i = (int)tid; i < n_elems; i += (int)tgs) {
        local_max = max(local_max, abs(float(x[base + i])));
    }
    smem[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = max(smem[tid], smem[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float absmax  = smem[0];
    const float scale_v = max(absmax / 127.0f, 1.0e-8f);

    if (tid == 0) {
        scale[(b * p.H + h) * p.N_blocks + block_idx] = scale_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = (int)tid; i < n_elems; i += (int)tgs) {
        float q = round(float(x[base + i]) / scale_v);
        x_int8[base + i] = char(int(clamp(q, -128.0f, 127.0f)));
    }
}
)";

    return ss.str();
}

// ---------------------------------------------------------------------------
// CPU fallback (reference implementation — used in non-GPU environments)
// ---------------------------------------------------------------------------

void MFAQuantizePerBlock::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    const auto& x = inputs[0];
    auto& out_int8  = outputs[0];
    auto& out_scale = outputs[1];

    out_int8.set_data(allocator::malloc(out_int8.nbytes()));
    out_scale.set_data(allocator::malloc(out_scale.nbytes()));

    auto* scale_ptr = out_scale.data<float>();
    auto* int8_ptr  = out_int8.data<int8_t>();

    // Cast to float32 for reference computation
    auto x_f32 = astype(x, float32);
    mlx::core::eval(x_f32);
    const auto* xf = x_f32.data<float>();

    for (int b = 0; b < B_; b++) {
        for (int h = 0; h < H_; h++) {
            for (int bi = 0; bi < N_blocks_; bi++) {
                int tok_start = bi * block_size_;
                int n_tok     = std::min(block_size_, N_ - tok_start);
                int n_elem    = n_tok * D_;
                int base      = (b * H_ + h) * N_ * D_ + tok_start * D_;

                float absmax = 0.0f;
                for (int i = 0; i < n_elem; i++) {
                    absmax = std::max(absmax, std::abs(xf[base + i]));
                }
                float scale = std::max(absmax / 127.0f, 1e-8f);
                int si = (b * H_ + h) * N_blocks_ + bi;
                scale_ptr[si] = scale;

                for (int i = 0; i < n_elem; i++) {
                    float q = std::round(xf[base + i] / scale);
                    q = std::max(-128.0f, std::min(127.0f, q));
                    int8_ptr[base + i] = static_cast<int8_t>(static_cast<int>(q));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU dispatch
// ---------------------------------------------------------------------------

void MFAQuantizePerBlock::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    const auto& x = inputs[0];
    auto& out_int8  = outputs[0];
    auto& out_scale = outputs[1];

    out_int8.set_data(allocator::malloc(out_int8.nbytes()));
    out_scale.set_data(allocator::malloc(out_scale.nbytes()));

    const bool is_f16 = (x.dtype() == float16);

    using KK = ShaderCache::KernelKey;
    KK key{
        KK::KernelType::QuantizePerBlock,
        D_,          // head_dim  (= D)
        block_size_, // block_q   (= block_size)
        0, 0, 0,     // block_k, block_d, n_warps — unused
        false, false, false, false, false, false, false,
        /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
        false,
        is_f16 ? uint8_t(0) : uint8_t(1),
        1
    };

    auto& dev = metal::device(stream().device);
    void* raw = ShaderCache::get().get_or_compile(key, dev.mtl_device());
    auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(raw);

    QuantizeBlockParams params{B_, H_, N_, D_, block_size_, N_blocks_};

    auto& enc = mlx::core::metal::get_command_encoder(stream());
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_array(x,          0);
    enc.set_output_array(out_int8,  1);
    enc.set_output_array(out_scale, 2);
    enc.set_bytes(params,           3);

    // Grid: one TG per (block_idx, h, b); TG: 256 threads (power-of-2 for tree reduce)
    enc.dispatch_threadgroups(
        MTL::Size::Make((size_t)N_blocks_, (size_t)H_, (size_t)B_),
        MTL::Size::Make(256, 1, 1));
}

// ---------------------------------------------------------------------------
// Public C++ API
// ---------------------------------------------------------------------------

std::pair<array, array> mlx_mfa::mfa_quantize_per_block(
    const array& x,
    int block_size,
    StreamOrDevice s)
{
    if (x.ndim() != 4)
        throw std::invalid_argument(
            "mfa_quantize_per_block: input must be 4-D [B, H, N, D]");
    if (x.dtype() != float16 && x.dtype() != bfloat16)
        throw std::invalid_argument(
            "mfa_quantize_per_block: only float16 and bfloat16 are supported");
    if (block_size <= 0 || (block_size & (block_size - 1)) != 0)
        throw std::invalid_argument(
            "mfa_quantize_per_block: block_size must be a positive power of 2");

    const int B        = x.shape(0);
    const int H        = x.shape(1);
    const int N        = x.shape(2);
    const int D        = x.shape(3);
    const int N_blocks = (N + block_size - 1) / block_size;

    // Enforce row-major layout at the C++ entry point
    auto xc = mlx::core::contiguous(x, false, to_stream(s));

    mlx::core::Shape int8_shape  = {B, H, N, D};
    mlx::core::Shape scale_shape = {B, H, N_blocks, 1};

    auto st   = to_stream(s);
    auto outs = array::make_arrays(
        {int8_shape, scale_shape},
        {int8, float32},
        std::make_shared<MFAQuantizePerBlock>(st, B, H, N, D, block_size, N_blocks),
        {xc});

    return {outs[0], outs[1]};
}
