// mfa_smooth_quant.cpp — Fused smooth_k + quantize_per_block (Phase 1.1)
//
// See mfa_smooth_quant.hpp for algorithm description.
#include "mfa_smooth_quant.hpp"
#include "shader_cache.hpp"

#include <mlx/mlx.h>
#include <mlx/backend/metal/device.h>

#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

using namespace mlx::core;
using namespace mlx_mfa;

// ---------------------------------------------------------------------------
// Metal source: Pass 1 — per-channel mean over S
// ---------------------------------------------------------------------------
// Grid:    (ceil(D/256), H, B)   threadgroups
// Threads: (256, 1, 1)
//
// Each thread handles one D-channel: accumulates K[b, h, :, d] over S, writes mean.
// Output stored in flat [B*H*D] buffer (no keepdims overhead in Metal).
// ---------------------------------------------------------------------------

std::string mlx_mfa::generate_smooth_k_mean_source(const std::string& dtype_str) {
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n"
       << "using namespace metal;\n\n";
    ss << "typedef " << dtype_str << " SQDtype;\n\n";

    ss << R"(struct SmoothMeanParams {
    int B;
    int H;
    int S;
    int D;
};

// One TG per D-tile of 256 channels, per (H, B).
// Each thread accumulates one D-channel mean over all S tokens.
kernel void mfa_smooth_k_mean(
    const device SQDtype* k      [[buffer(0)]],
    device float*         k_mean [[buffer(1)]],
    constant SmoothMeanParams& p [[buffer(2)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid_v [[thread_position_in_threadgroup]])
{
    const int h   = (int)tgid.y;
    const int b   = (int)tgid.z;
    const int tid = (int)tid_v.x;

    if (b >= p.B || h >= p.H) return;

    // D channel handled by this thread.
    const int d = (int)tgid.x * 256 + tid;
    if (d >= p.D) return;

    // K[b, h, 0, 0] base offset
    const int bh_base = (b * p.H + h) * p.S * p.D;

    // Accumulate sum over S dimension.
    float acc = 0.0f;
    for (int s = 0; s < p.S; ++s) {
        acc += float(k[bh_base + s * p.D + d]);
    }

    // Write mean into flat [B, H, D] buffer (caller reshapes to [B,H,1,D]).
    k_mean[(b * p.H + h) * p.D + d] = acc / float(p.S);
}
)";
    return ss.str();
}

// ---------------------------------------------------------------------------
// Metal source: Pass 2 — subtract mean + quantize per block
// ---------------------------------------------------------------------------
// Grid:    (N_blocks, H, B)   threadgroups
// Threads: (256, 1, 1)
//
// Each TG handles one K-block of size (block_size * D) elements:
//   1. Load per-channel mean from k_mean buffer (flat [B*H*D]).
//   2. Subtract mean; tree-reduce absmax across block.
//   3. scale = absmax / 127.0  (clamped >= 1e-8).
//   4. Write scale to k_scale buffer.
//   5. Quantize: round((k - mean) / scale), clip, cast to int8.
// ---------------------------------------------------------------------------

std::string mlx_mfa::generate_smooth_k_quant_source(const std::string& dtype_str) {
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n"
       << "using namespace metal;\n\n";
    ss << "typedef " << dtype_str << " SQDtype;\n\n";

    ss << R"(struct SmoothQuantParams {
    int B;
    int H;
    int S;
    int D;
    int block_size;
    int N_blocks;
};

kernel void mfa_smooth_k_quant(
    const device SQDtype* k       [[buffer(0)]],
    const device float*   k_mean  [[buffer(1)]],
    device char*          k_int8  [[buffer(2)]],
    device float*         k_scale [[buffer(3)]],
    constant SmoothQuantParams& p [[buffer(4)]],
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
    const int n_tokens    = min(p.block_size, p.S - token_start);
    const int n_elems     = n_tokens * p.D;
    const int bh_base     = (b * p.H + h) * p.S * p.D + token_start * p.D;
    const int mean_base   = (b * p.H + h) * p.D;  // flat [B*H*D] layout

    threadgroup float smem[256];

    // Phase A: tree-reduce absmax of (k - mean) for this block.
    float local_max = 0.0f;
    for (int i = (int)tid; i < n_elems; i += (int)tgs) {
        int d       = i % p.D;
        float val   = float(k[bh_base + i]) - k_mean[mean_base + d];
        local_max   = max(local_max, abs(val));
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
        k_scale[(b * p.H + h) * p.N_blocks + block_idx] = scale_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase B: subtract mean + quantize to int8.
    for (int i = (int)tid; i < n_elems; i += (int)tgs) {
        int d     = i % p.D;
        float val = float(k[bh_base + i]) - k_mean[mean_base + d];
        float q   = round(val / scale_v);
        k_int8[bh_base + i] = char(int(clamp(q, -128.0f, 127.0f)));
    }
}
)";
    return ss.str();
}

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

void MFASmoothQuantizeK::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    const auto& k   = inputs[0];
    auto& out_int8  = outputs[0];
    auto& out_scale = outputs[1];
    auto& out_mean  = outputs[2];

    out_int8.set_data(allocator::malloc(out_int8.nbytes()));
    out_scale.set_data(allocator::malloc(out_scale.nbytes()));
    out_mean.set_data(allocator::malloc(out_mean.nbytes()));

    // Convert to float32 for reference computation.
    auto k_f32 = astype(k, float32);
    mlx::core::eval({k_f32});   // materialise k_f32 on CPU
    const auto* kf  = k_f32.data<float>();
    auto* int8_ptr  = out_int8.data<int8_t>();
    auto* scale_ptr = out_scale.data<float>();
    auto* mean_ptr  = out_mean.data<float>();

    for (int b = 0; b < B_; b++) {
        for (int h = 0; h < H_; h++) {
            const int bh = b * H_ + h;

            // Pass 1: per-channel mean over S.
            for (int d = 0; d < D_; d++) {
                double acc = 0.0;
                for (int s = 0; s < S_; s++) {
                    acc += static_cast<double>(kf[(bh * S_ + s) * D_ + d]);
                }
                mean_ptr[bh * D_ + d] = static_cast<float>(acc / S_);
            }

            // Pass 2: subtract mean + per-block quantize.
            for (int bi = 0; bi < N_blocks_; bi++) {
                int tok_start = bi * block_size_;
                int n_tok     = std::min(block_size_, S_ - tok_start);
                int n_elem    = n_tok * D_;
                int base      = bh * S_ * D_ + tok_start * D_;

                float absmax = 0.0f;
                for (int i = 0; i < n_elem; i++) {
                    int d   = i % D_;
                    float v = kf[base + i] - mean_ptr[bh * D_ + d];
                    absmax  = std::max(absmax, std::abs(v));
                }
                float scale = std::max(absmax / 127.0f, 1e-8f);
                scale_ptr[bh * N_blocks_ + bi] = scale;

                for (int i = 0; i < n_elem; i++) {
                    int d   = i % D_;
                    float v = kf[base + i] - mean_ptr[bh * D_ + d];
                    float q = std::round(v / scale);
                    q = std::max(-128.0f, std::min(127.0f, q));
                    int8_ptr[base + i] = static_cast<int8_t>(static_cast<int>(q));
                }
            }
        }
    }

    // Reshape out_mean data from flat [B*H*D] to [B,H,1,D].
    // The data is already laid out correctly (bh*D + d maps to [b,h,0,d]).
    // No reshape needed — just the stride interpretation changes at the Python level.
}

// ---------------------------------------------------------------------------
// GPU dispatch
// ---------------------------------------------------------------------------

void MFASmoothQuantizeK::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs)
{
    const auto& k   = inputs[0];
    auto& out_int8  = outputs[0];
    auto& out_scale = outputs[1];
    auto& out_mean  = outputs[2];

    out_int8.set_data(allocator::malloc(out_int8.nbytes()));
    out_scale.set_data(allocator::malloc(out_scale.nbytes()));
    // Allocate out_mean separately so we can get its raw buffer pointer.
    auto mean_buf = allocator::malloc(out_mean.nbytes());
    out_mean.set_data(mean_buf);

    const bool is_f16   = (k.dtype() == float16);
    const uint8_t dtype = is_f16 ? 0 : 1;

    using KK = ShaderCache::KernelKey;
    auto& dev = metal::device(stream().device);
    auto& enc = dev.get_command_encoder(stream().index);

    // Get the raw MTL buffer pointer for k_mean intermediate.
    // This allows both pass 1 (write) and pass 2 (read) to access it
    // without MLX fence interference, similar to the flash decode pattern.
    auto* mean_mtl = reinterpret_cast<MTL::Buffer*>(mean_buf.ptr());

    // ── Pass 1: compute per-channel mean ───────────────────────────────────
    {
        KK mean_key{
            KK::KernelType::SmoothQuantizeMean,
            D_, 0, 0, 0, 0,
            false, false, false, false, false, false, false,
            /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
            false,
            dtype, 1
        };
        void* raw = ShaderCache::get().get_or_compile(mean_key, dev.mtl_device());
        auto* pl  = reinterpret_cast<MTL::ComputePipelineState*>(raw);

        struct SmoothMeanParams { int B, H, S, D; };
        SmoothMeanParams pm{B_, H_, S_, D_};

        enc.set_compute_pipeline_state(pl);
        enc.set_input_array(k,     0);
        enc.set_buffer(mean_mtl,   1, 0);
        enc.set_bytes(pm,          2);

        // Grid: one TG per 256 D-channels, per (H, B).
        size_t d_tgs = ((size_t)D_ + 255) / 256;
        enc.dispatch_threadgroups(
            MTL::Size::Make(d_tgs, (size_t)H_, (size_t)B_),
            MTL::Size::Make(256, 1, 1));

        // Barrier: pass 2 reads k_mean written by pass 1.
        // Use enc.barrier() (not maybeInsertBarrier — we used set_buffer, not
        // set_output_array, so needs_barrier_ flag is not set by MLX).
        enc.barrier();
    }

    // ── Pass 2: subtract mean + quantize ───────────────────────────────────
    {
        KK quant_key{
            KK::KernelType::SmoothQuantizeK,
            D_, block_size_, 0, 0, 0,
            false, false, false, false, false, false, false,
            /*has_attn_bias=*/false, /*attn_bias_mode=*/0,
            false,
            dtype, 1
        };
        void* raw = ShaderCache::get().get_or_compile(quant_key, dev.mtl_device());
        auto* pl  = reinterpret_cast<MTL::ComputePipelineState*>(raw);

        struct SmoothQuantParams { int B, H, S, D, block_size, N_blocks; };
        SmoothQuantParams pq{B_, H_, S_, D_, block_size_, N_blocks_};

        enc.set_compute_pipeline_state(pl);
        enc.set_input_array(k,          0);
        enc.set_buffer(mean_mtl,        1, 0);
        enc.set_output_array(out_int8,  2);
        enc.set_output_array(out_scale, 3);
        enc.set_bytes(pq,               4);

        enc.dispatch_threadgroups(
            MTL::Size::Make((size_t)N_blocks_, (size_t)H_, (size_t)B_),
            MTL::Size::Make(256, 1, 1));
    }
    // Note: out_mean data is already populated from pass 1 via mean_buf.
    // MLX tracks out_mean as an output of this primitive because it is in
    // the outputs[] vector; the data pointer was set via set_data(mean_buf).
}

// ---------------------------------------------------------------------------
// Public C++ API
// ---------------------------------------------------------------------------

std::tuple<array, array, array> mlx_mfa::mfa_smooth_quantize_k(
    const array& k,
    int block_size,
    StreamOrDevice s)
{
    if (k.ndim() != 4)
        throw std::invalid_argument(
            "mfa_smooth_quantize_k: input must be 4-D [B, H, S, D]");
    if (k.dtype() != float16 && k.dtype() != bfloat16)
        throw std::invalid_argument(
            "mfa_smooth_quantize_k: only float16 and bfloat16 are supported");
    if (block_size <= 0 || (block_size & (block_size - 1)) != 0)
        throw std::invalid_argument(
            "mfa_smooth_quantize_k: block_size must be a positive power of 2");

    const int B        = k.shape(0);
    const int H        = k.shape(1);
    const int S        = k.shape(2);
    const int D        = k.shape(3);
    const int N_blocks = (S + block_size - 1) / block_size;

    // Enforce contiguous row-major layout at the C++ boundary.
    auto kc = mlx::core::contiguous(k, false, to_stream(s));

    Shape int8_shape  = {B, H, S, D};
    Shape scale_shape = {B, H, N_blocks, 1};
    Shape mean_shape  = {B, H, 1, D};    // [B, H, D] flat in memory, exposed as [B,H,1,D]

    auto st   = to_stream(s);
    auto outs = array::make_arrays(
        {int8_shape, scale_shape, mean_shape},
        {int8, float32, float32},
        std::make_shared<MFASmoothQuantizeK>(st, B, H, S, D, block_size, N_blocks),
        {kc});

    return {outs[0], outs[1], outs[2]};
}
