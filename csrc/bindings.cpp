/// mlx-mfa nanobind bindings.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <mlx/mlx.h>
#include <mlx/backend/metal/device.h>

#include "mfa_attention.hpp"
#include "mfa_env.hpp"
#include "mfa_paged_gather.hpp"
#include "mfa_quantize.hpp"
#include "mfa_scatter.hpp"
#include "mfa_smooth_quant.hpp"
#include "mfa_steel_fwd_v2.hpp"
#include "shader_cache.hpp"

namespace nb = nanobind;

NB_MODULE(_ext, m) {
  m.doc() = "mlx-mfa C++ extension: Metal Flash Attention for MLX";


  m.def(
      "mfa_attention_forward",
      &mlx_mfa::mfa_attention_forward,
      nb::arg("q"),
      nb::arg("k"),
      nb::arg("v"),
      nb::arg("scale"),
      nb::arg("causal"),
      nb::arg("softcap") = 0.0f,
      nb::arg("window_left") = -1,
      nb::arg("window_right") = -1,
      nb::arg("stream") = nb::none(),
      "Flash Attention forward (Metal). "
      "q/k/v: [B, H, N, D], float16/bfloat16/float32. "
      "softcap: tanh softcapping factor (0.0 = disabled). "
      "window_left: sliding window left radius (-1 = disabled). "
      "window_right: sliding window right radius (-1 = disabled).");

  // Debug: returns (O, L) so L (logsumexp) can be inspected from Python.
  m.def("mfa_forward_with_lse",
      [](const mlx::core::array& q,
         const mlx::core::array& k,
         const mlx::core::array& v,
         float scale, bool causal) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        // D.5: enforce row-major layout; no-op when already contiguous.
        auto qc = mlx::core::contiguous(q, false, s);
        auto kc = mlx::core::contiguous(k, false, s);
        auto vc = mlx::core::contiguous(v, false, s);
        mlx_mfa::MFAttention::Params params{
            (int)qc.shape(3), scale, causal,
            false, false, false, 0, 0.0f, false, /*window_left=*/-1,
            /*window_right=*/-1};
        mlx::core::Shape lse_shape = {qc.shape(0), qc.shape(1), qc.shape(2)};
        auto outs = mlx::core::array::make_arrays(
            {qc.shape(), lse_shape},
            {qc.dtype(), mlx::core::float32},
            std::make_shared<mlx_mfa::MFAttention>(s, params),
            {qc, kc, vc});
        return std::make_pair(outs[0], outs[1]);
      },
      nb::arg("q"), nb::arg("k"), nb::arg("v"),
      nb::arg("scale"), nb::arg("causal"),
      "Debug: returns (O, L) where L is the logsumexp in log2 domain.");

  // Debug: runs MFABackwardQuery kernel directly (bypasses vjp tape).
  // Takes (q, k, v, O, L, dO, scale, causal) — all pre-evaluated.
  // Returns (dQ, D_computed) so D_computed can be compared to reference.
  m.def("mfa_backward_query_debug",
      [](const mlx::core::array& q,
         const mlx::core::array& k,
         const mlx::core::array& v,
         const mlx::core::array& O,
         const mlx::core::array& L,
         const mlx::core::array& dO,
         float scale, bool causal) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        mlx_mfa::MFAttention::Params params{
            (int)q.shape(3), scale, causal,
            false, false, false, 0, 0.0f, false, /*window_left=*/-1,
            /*window_right=*/-1};
        mlx::core::Shape d_shape = {q.shape(0), q.shape(1), q.shape(2)};
        auto outs = mlx::core::array::make_arrays(
            {q.shape(), d_shape},
            {q.dtype(), mlx::core::float32},
            std::make_shared<mlx_mfa::MFABackwardQuery>(s, params),
            {q, k, v, O, L, dO});
        return std::make_pair(outs[0], outs[1]);
      },
      nb::arg("q"), nb::arg("k"), nb::arg("v"),
      nb::arg("O"), nb::arg("L"), nb::arg("dO"),
      nb::arg("scale"), nb::arg("causal"),
      "Debug: returns (dQ, D_computed) from MFABackwardQuery kernel directly.");

  // Debug: runs MFABackwardKeyValue kernel directly (bypasses vjp tape).
  // Takes (q, k, v, O, L, D_computed, dO, scale, causal) — all pre-evaluated.
  // Returns (dK, dV).
  m.def("mfa_backward_kv_debug",
      [](const mlx::core::array& q,
         const mlx::core::array& k,
         const mlx::core::array& v,
         const mlx::core::array& O,
         const mlx::core::array& L,
         const mlx::core::array& D,
         const mlx::core::array& dO,
         float scale, bool causal) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        mlx_mfa::MFAttention::Params params{
            (int)q.shape(3), scale, causal,
            false, false, false, 0, 0.0f, false, /*window_left=*/-1,
            /*window_right=*/-1};
        auto outs = mlx::core::array::make_arrays(
            {k.shape(), v.shape()},
            {k.dtype(), v.dtype()},
            std::make_shared<mlx_mfa::MFABackwardKeyValue>(s, params),
            {q, k, v, O, L, D, dO});
        return std::make_pair(outs[0], outs[1]);
      },
      nb::arg("q"), nb::arg("k"), nb::arg("v"),
      nb::arg("O"), nb::arg("L"), nb::arg("D"), nb::arg("dO"),
      nb::arg("scale"), nb::arg("causal"),
      "Debug: returns (dK, dV) from MFABackwardKeyValue kernel directly.");

  // STEEL backward: dispatches MFASteelBwdDQ + MFASteelBwdDKV.
  // Args: q,k,v,O,L,dO — all pre-evaluated on GPU (caller owns L).
  // scale, causal: forward hyperparameters.
  // Returns: tuple (dQ, dK, dV).
  // Only supports f16/bf16 with D<=128.
  m.def("mfa_steel_backward",
      [](const mlx::core::array& q,
         const mlx::core::array& k,
         const mlx::core::array& v,
         const mlx::core::array& O,
         const mlx::core::array& L,
         const mlx::core::array& dO,
         float scale, bool causal) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        // G.1: Materialise all 6 inputs before Metal kernel dispatch.
        // MLX autograd may recycle GPU buffers; mlx::core::eval() fences against aliasing.
        mlx::core::eval(std::vector<mlx::core::array>{q, k, v, O, L, dO});
        mlx_mfa::MFAttention::Params params{};
        params.head_dim    = q.shape(3);
        params.scale       = scale;
        params.causal      = causal;
        params.window_left = -1;  // disabled — steel backward has no window

        // delta = rowsum(dO * O)  [B, H, N], float32.
        // Note: the Metal kernel multiplies by p->scale internally when computing
        // dS = scale * P * (dP - delta).  Do NOT pre-multiply by scale here.
        auto dO_f32 = mlx::core::astype(dO, mlx::core::float32, s);
        auto O_f32  = mlx::core::astype(O,  mlx::core::float32, s);
        auto delta  = mlx::core::sum(
                          mlx::core::multiply(dO_f32, O_f32, s),
                          std::vector<int>{3}, false, s);

        // dQ
        auto bwd_q = mlx::core::array::make_arrays(
            {q.shape()},
            {q.dtype()},
            std::make_shared<mlx_mfa::MFASteelBwdDQ>(s, params),
            {q, k, v, O, L, dO, delta});

        // dK, dV
        auto bwd_kv = mlx::core::array::make_arrays(
            {k.shape(), v.shape()},
            {k.dtype(), v.dtype()},
            std::make_shared<mlx_mfa::MFASteelBwdDKV>(s, params),
            {q, k, v, O, L, delta, dO});

        return nb::make_tuple(bwd_q[0], bwd_kv[0], bwd_kv[1]);
      },
      nb::arg("q"), nb::arg("k"), nb::arg("v"),
      nb::arg("O"), nb::arg("L"), nb::arg("dO"),
      nb::arg("scale"), nb::arg("causal"),
      "STEEL backward: returns (dQ, dK, dV). f16/bf16, D<=128 only.");

  m.def("mfa_steel_backward_sparse",
      [](const mlx::core::array& q,
         const mlx::core::array& k,
         const mlx::core::array& v,
         const mlx::core::array& O,
         const mlx::core::array& L,
         const mlx::core::array& dO,
         const mlx::core::array& block_mask,
         float scale, bool causal) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        mlx_mfa::MFAttention::Params params{};
        params.head_dim       = q.shape(3);
        params.scale          = scale;
        params.causal         = causal;
        params.has_block_mask = true;
        params.window_left    = -1;

        // delta = rowsum(dO * O)  [B, H, N], float32.
        auto dO_f32 = mlx::core::astype(dO, mlx::core::float32, s);
        auto O_f32  = mlx::core::astype(O,  mlx::core::float32, s);
        auto delta  = mlx::core::sum(
                          mlx::core::multiply(dO_f32, O_f32, s),
                          std::vector<int>{3}, false, s);

        // dQ — inputs[7] = block_mask
        auto bwd_q = mlx::core::array::make_arrays(
            {q.shape()},
            {q.dtype()},
            std::make_shared<mlx_mfa::MFASteelBwdDQ>(s, params),
            {q, k, v, O, L, dO, delta, block_mask});

        // dK, dV — inputs[7] = block_mask
        auto bwd_kv = mlx::core::array::make_arrays(
            {k.shape(), v.shape()},
            {k.dtype(), v.dtype()},
            std::make_shared<mlx_mfa::MFASteelBwdDKV>(s, params),
            {q, k, v, O, L, delta, dO, block_mask});

        return nb::make_tuple(bwd_q[0], bwd_kv[0], bwd_kv[1]);
      },
      nb::arg("q"), nb::arg("k"), nb::arg("v"),
      nb::arg("O"), nb::arg("L"), nb::arg("dO"),
      nb::arg("block_mask"), nb::arg("scale"), nb::arg("causal"),
      "Sparse STEEL backward: block_mask skips inactive tiles. Returns (dQ, dK, dV).");

  m.def("_mlx_build_version", []() -> std::string {
#ifdef MLX_BUILD_VERSION
    return MLX_BUILD_VERSION;
#else
    return "unknown";
#endif
  }, "MLX version used at compile time (major.minor.patch).");

  m.def("shader_cache_size", []() {
    return mlx_mfa::ShaderCache::get().size();
  }, "Number of cached Metal compute pipelines.");

  m.def("shader_cache_clear", []() {
    mlx_mfa::ShaderCache::get().clear();
  }, "Clear the Metal pipeline cache.");

  // Returns a dict with:
  //   gpu_family_gen  int  — GPU silicon generation from architecture string
  //                          (e.g. "applegpu_g13s" → 13).
  //                          13=M1, 14=M2, 15=M3, 16=M4.
  //   is_m3_plus      bool — True if gen >= 15 (M3/M4: preferAsyncCache block params)
  //   device_name     str  — MTLDevice name (e.g. "Apple M1 Max")
  //   gpu_cores       int  — Estimated physical GPU core count from device name.
  //                          Correct per-variant: M1 Max=32, M1 base=8, M2 Max=38, …
  //                          Falls back to conservative gen-based estimate for
  //                          unknown names (simulator, future hardware).
  m.def("get_device_info", []() -> nb::dict {
    auto s = mlx::core::default_stream(mlx::core::Device::gpu);
    auto& d = mlx::core::metal::device(s.device);
    int gen = d.get_architecture_gen();
    // MTL::Device::name() returns an NS::String; utf8String() gives a C string.
    auto* mtl_dev = d.mtl_device();
    std::string dev_name = mtl_dev
        ? std::string(mtl_dev->name()->utf8String())
        : "";
    int cores = mlx_mfa::estimate_gpu_cores(dev_name, gen);
    if (dev_name.empty()) dev_name = "unknown";
    nb::dict info;
    info["gpu_family_gen"] = gen;
    info["is_m3_plus"]     = (gen >= 15);
    info["device_name"]    = dev_name;
    info["gpu_cores"]      = cores;
    return info;
  }, "Return Metal GPU hardware info: silicon generation, M3+ flag, device name, gpu_cores.");

  // --- ALiBi-biased forward ---
  m.def(
      "mfa_attention_alibi_forward",
      &mlx_mfa::mfa_attention_alibi_forward,
      nb::arg("q"),
      nb::arg("k"),
      nb::arg("v"),
      nb::arg("alibi_slopes"),
      nb::arg("scale"),
      nb::arg("causal"),
      nb::arg("stream") = nb::none(),
      "Flash Attention with ALiBi per-head linear position biases.\n"
      "alibi_slopes: float32 [H], one slope per query head.\n"
      "Bias = slope_h * (k_pos - q_pos) added before softmax.\n"
      "Only f16/bf16 supported.");

  // --- RoPE-fused forward ---
  m.def(
      "mfa_attention_rope_forward",
      &mlx_mfa::mfa_attention_rope_forward,
      nb::arg("q"),
      nb::arg("k"),
      nb::arg("v"),
      nb::arg("rotary_cos"),
      nb::arg("rotary_sin"),
      nb::arg("scale"),
      nb::arg("causal"),
      nb::arg("cache_seqlens"),
      nb::arg("interleaved") = true,
      nb::arg("stream") = nb::none(),
      "Flash Attention with in-kernel RoPE fusion.\n"
      "rotary_cos/sin: float32 [max_seq_len, D/2].\n"
      "cache_seqlens: KV cache length (absolute position of Q token 0).\n"
      "interleaved: True=LLaMA pairs (d*2,d*2+1); False=GPT-NeoX (d,d+D/2).\n"
      "Only f16/bf16 supported.");

  // --- Block-sparse forward ---
  m.def("mfa_attention_sparse_forward",
        [](mlx::core::array q, mlx::core::array k, mlx::core::array v,
           mlx::core::array block_mask,
           float scale, bool causal,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> mlx::core::array {
          return mlx_mfa::mfa_attention_sparse_forward(
              q, k, v, block_mask, scale, causal, stream);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("block_mask"),
        nb::arg("scale"), nb::arg("causal"),
        nb::arg("stream") = nb::none(),
        "Block-sparse forward attention.\n"
        "block_mask: uint8 [NQ_tiles, NK_tiles]. 1=compute, 0=skip.\n"
        "Returns O [B, H, N, D]. Only f16/bf16 supported.");

  // --- Block-sparse forward returning (O, L) for use by native backward ---
  m.def("mfa_attention_sparse_forward_with_lse",
        [](mlx::core::array q, mlx::core::array k, mlx::core::array v,
           mlx::core::array block_mask,
           float scale, bool causal,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> std::pair<mlx::core::array, mlx::core::array> {
          auto outs = mlx_mfa::mfa_attention_sparse_forward_with_lse(
              q, k, v, block_mask, scale, causal, stream);
          return {outs[0], outs[1]};
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("block_mask"),
        nb::arg("scale"), nb::arg("causal"),
        nb::arg("stream") = nb::none(),
        "Block-sparse forward returning (O, L) where L is logsumexp [B,H,N].\n"
        "Used by the native sparse backward pass to avoid recomputation.\n"
        "block_mask: uint8 [NQ_tiles, NK_tiles]. Only f16/bf16 supported.");

  // --- STEEL varlen forward: packed [1, H, total_q, D] layout ---
  m.def("mfa_attention_varlen_forward",
        [](mlx::core::array q, mlx::core::array k, mlx::core::array v,
           mlx::core::array cu_seqlens_q, mlx::core::array cu_seqlens_k,
           mlx::core::array tile_offsets,
           float scale, bool causal,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> std::pair<mlx::core::array, mlx::core::array> {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_attention_varlen_forward(
              q, k, v, cu_seqlens_q, cu_seqlens_k, tile_offsets,
              scale, causal, s);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"),
        nb::arg("cu_seqlens_q"), nb::arg("cu_seqlens_k"),
        nb::arg("tile_offsets"),
        nb::arg("scale"), nb::arg("causal"),
        nb::arg("stream") = nb::none(),
        "STEEL varlen attention forward.\n"
        "\n"
        "Inputs are packed: Q/O = [1, H, total_q, D], K/V = [1, H_kv, total_kv, D].\n"
        "cu_seqlens_q: int32 [num_seqs+1], cumulative query lengths.\n"
        "cu_seqlens_k: int32 [num_seqs+1], cumulative key lengths.\n"
        "tile_offsets: int32 [num_seqs+1], cumulative Q-tile counts per sequence.\n"
        "Returns (O [1,H,total_q,D], L [1,H,total_q] logsumexp in log2 domain).\n"
        "Only f16/bf16 supported.");

  m.def("mfa_paged_kv_gather",
        [](mlx::core::array pool,
           mlx::core::array block_table,
           mlx::core::array seq_lens,
           int max_kv_len,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> mlx::core::array {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_paged_kv_gather(
              pool, block_table, seq_lens, max_kv_len, s);
        },
        nb::arg("pool"),
        nb::arg("block_table"),
        nb::arg("seq_lens"),
        nb::arg("max_kv_len"),
        nb::arg("stream") = nb::none(),
        "Metal paged KV gather: pool [N_blk, BS, H, D] -> out [B, H, max_kv, D].\n"
        "pool: f16 or bf16. block_table: int32 [B, max_blocks]. seq_lens: int32 [B].\n"
        "Transposes [BS,H,D] -> [H,BS,D] (token-major -> head-major) during gather.");

  // --- Paged STEEL forward (Track FD): kernel-level paged KV ---
  m.def("mfa_paged_steel_forward",
        [](mlx::core::array q,
           mlx::core::array k_pool,
           mlx::core::array v_pool,
           mlx::core::array block_table,
           mlx::core::array seq_lens,
           float scale,
           bool  causal,
           int   window_left,
           int   window_right,
           int   block_size,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> std::pair<mlx::core::array, mlx::core::array> {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_paged_steel_forward(
              q, k_pool, v_pool, block_table, seq_lens,
              scale, causal, window_left, window_right, block_size, s);
        },
        nb::arg("q"),
        nb::arg("k_pool"),
        nb::arg("v_pool"),
        nb::arg("block_table"),
        nb::arg("seq_lens"),
        nb::arg("scale"),
        nb::arg("causal"),
        nb::arg("window_left")  = -1,
        nb::arg("window_right") = -1,
        nb::arg("block_size")   = 16,
        nb::arg("stream")       = nb::none(),
        "Paged STEEL forward attention (kernel-level paged KV, Track FD).\n"
        "\n"
        "Avoids a gather+attend round-trip by reading K/V directly from the paged\n"
        "pool inside the Metal kernel via block_table lookups.\n"
        "\n"
        "q:           [B, H, N, D]               f16 or bf16\n"
        "k_pool:      [num_blocks, block_size, H_kv, D]\n"
        "v_pool:      [num_blocks, block_size, H_kv, D]\n"
        "block_table: [B, max_blocks]             int32\n"
        "seq_lens:    [B]                         int32 (effective KV length per batch)\n"
        "\n"
        "Returns (O [B,H,N,D], L [B,H,N] logsumexp in log2 domain).\n"
        "GQA: H_q / H_kv must be integer. window_left/right=-1 disables sliding window.\n"
        "Only f16/bf16 supported.");

  // --- SageAttention forward (Track KB, CP2): fp16 Q + int8 K + fp16 V ---
  // CP2: Q is now fp16 (no external Q quantize dispatch). K stays int8.
  m.def("mfa_sage_forward",
        [](mlx::core::array q,
           mlx::core::array k_int8,
           mlx::core::array v,
           mlx::core::array k_scale,
           float scale,
           bool  causal,
           int   window_left,
           int   window_right,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> std::pair<mlx::core::array, mlx::core::array> {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_sage_forward(
              q, k_int8, v, k_scale, scale, causal,
              window_left, window_right, s);
        },
        nb::arg("q"),
        nb::arg("k_int8"),
        nb::arg("v"),
        nb::arg("k_scale"),
        nb::arg("scale"),
        nb::arg("causal")       = false,
        nb::arg("window_left")  = -1,
        nb::arg("window_right") = -1,
        nb::arg("stream")       = nb::none(),
        "SageAttention forward pass: fp16 Q + int8 K + fp16 V → fp16 O.\n"
        "\n"
        "CP2: Q is passed as fp16/bf16 directly — no external Q quantize dispatch.\n"
        "K is int8 (quantized by QuantizedKVCache at append time). K bandwidth\n"
        "reduction (2×) still applies. GEMM always runs in fp16.\n"
        "\n"
        "q:            [B, H, N, D]    fp16 or bf16 queries\n"
        "k_int8:       [B, H_kv, S, D] int8 quantized keys\n"
        "v:            [B, H_kv, S, D] fp16 or bf16 values (unquantized)\n"
        "k_scale:      [B, H_kv, NK]   float32 per-tile K dequantization scales\n"
        "window_left:  left window radius in tokens (-1 = disabled).\n"
        "window_right: right window radius in tokens (-1 = disabled).\n"
        "\n"
        "Returns (O [B,H,N,D] fp16/bf16, L [B,H,N] logsumexp in log2 domain).\n"
        "GQA: H_q / H_kv must be an integer. Only f16/bf16 V supported.");

  // --- Fused per-block INT8 quantization (Phase 4-A.1) ---
  m.def("mfa_quantize_per_block",
        [](mlx::core::array x,
           int block_size,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> std::pair<mlx::core::array, mlx::core::array> {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_quantize_per_block(x, block_size, s);
        },
        nb::arg("x"),
        nb::arg("block_size"),
        nb::arg("stream") = nb::none(),
        "Fused per-block INT8 quantization (Phase 4-A.1).\n"
        "\n"
        "Replaces the Python-side quantize_per_block() with a single Metal dispatch.\n"
        "One threadgroup per (b, h, block_idx) reduces absmax then quantizes elements.\n"
        "\n"
        "x:          [B, H, N, D]  fp16 or bf16\n"
        "block_size: positive power of 2\n"
        "\n"
        "Returns (x_int8 [B,H,N,D] int8, scale [B,H,N_blocks,1] float32).");

  // --- Fused smooth_k + quantize_per_block (Phase 1.1) ---
  m.def("mfa_smooth_quantize_k",
        [](mlx::core::array k,
           int block_size,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> std::tuple<mlx::core::array, mlx::core::array, mlx::core::array> {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_smooth_quantize_k(k, block_size, s);
        },
        nb::arg("k"),
        nb::arg("block_size"),
        nb::arg("stream") = nb::none(),
        "Fused smooth_k + quantize_per_block for SageAttention K preprocessing (Phase 1.1).\n"
        "\n"
        "Replaces smooth_k() + quantize_per_block(k_smooth) with one C++ primitive.\n"
        "Eliminates intermediate K_smooth tensor; 2 Metal dispatches instead of 3.\n"
        "\n"
        "k:          [B, H, S, D]  fp16 or bf16\n"
        "block_size: positive power of 2 (should be BK from sage_block_sizes())\n"
        "\n"
        "Returns (k_int8 [B,H,S,D] int8, k_scale [B,H,N_blocks,1] f32, k_mean [B,H,1,D] f32).");

  // --- Paged KV scatter write (Phase 4-C.1+E.2) ---
  m.def("mfa_scatter_kv",
        [](mlx::core::array pool,
           mlx::core::array tokens,
           mlx::core::array blk_ids,
           mlx::core::array blk_offs,
           std::optional<mlx::core::StreamOrDevice> stream)
            -> mlx::core::array {
          auto s = mlx::core::to_stream(stream.value_or(mlx::core::default_device()));
          return mlx_mfa::mfa_scatter_kv(pool, tokens, blk_ids, blk_offs, s);
        },
        nb::arg("pool"),
        nb::arg("tokens"),
        nb::arg("blk_ids"),
        nb::arg("blk_offs"),
        nb::arg("stream") = nb::none(),
        "Scatter-write tokens into paged KV pool (Phase 4-C.1+E.2).\n"
        "\n"
        "Replaces the Python pool rebuild loop with a single Metal copy+scatter pass.\n"
        "Each pool element is either copied from pool_in or overwritten by a scattered\n"
        "token — whichever applies.  Optimized for small N_write (decode: 1-4 tokens).\n"
        "\n"
        "pool:     [num_blocks, block_size, H_kv, D]  fp16 or bf16\n"
        "tokens:   [N_write, H_kv, D]                 same dtype\n"
        "blk_ids:  [N_write]                           int32 (target physical block)\n"
        "blk_offs: [N_write]                           int32 (target slot within block)\n"
        "\n"
        "Returns pool_out [num_blocks, block_size, H_kv, D] with scattered writes applied.");

  // GNA native binding removed — flash_attention_gna() uses sparse path (Python-side)

  // ── PagedVarlenForward (fused packed Q + paged KV) ──────────────────────
  m.def("mfa_paged_varlen_forward",
      [](const mlx::core::array& q,
         const mlx::core::array& k_pool,
         const mlx::core::array& v_pool,
         const mlx::core::array& cu_seqlens_q,
         const mlx::core::array& tile_offsets,
         const mlx::core::array& block_table,
         const mlx::core::array& seq_lens_kv,
         float scale,
         bool causal,
         int block_size,
         nb::object stream) {
        auto s = mlx::core::default_stream(mlx::core::Device::gpu);
        return mlx_mfa::mfa_paged_varlen_forward(
            q, k_pool, v_pool, cu_seqlens_q, tile_offsets,
            block_table, seq_lens_kv, scale, causal, block_size, s);
      },
      nb::arg("q"),
      nb::arg("k_pool"),
      nb::arg("v_pool"),
      nb::arg("cu_seqlens_q"),
      nb::arg("tile_offsets"),
      nb::arg("block_table"),
      nb::arg("seq_lens_kv"),
      nb::arg("scale"),
      nb::arg("causal"),
      nb::arg("block_size"),
      nb::arg("stream") = nb::none(),
      "Fused paged varlen forward: packed Q + paged KV in a single dispatch.");

  m.def("_invalidate_env_config", []() {
      mlx_mfa::MFAEnvConfig::invalidate();
  }, "Re-read all cached MFA_* env vars. Call after os.environ changes.");

  m.attr("__version__") = "2.20.0";
}
