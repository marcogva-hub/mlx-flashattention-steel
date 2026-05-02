/// V6 NAX kernel generator — MPP-accelerated flash-attention forward (M5+).
///
/// Produces MSL 4.0 + MetalPerformancePrimitives source. Patterned after
/// MLX's bundled steel_attention_nax.h (Apple's reference V6-class kernel)
/// but lightweight — just enough to evaluate gate G7 (does V6 beat V2 on
/// M5 Max?) before investing in the full Draw Things ccv port.

#include "mfa_steel_fwd_v6_nax.hpp"
#include <sstream>

namespace mlx_mfa {

std::string generate_steel_v6_nax_source(const ShaderCache::KernelKey& key) {
  const int BD = key.head_dim;       // 64 or 128
  const bool is_bf16 = (key.dtype == 1);
  const char* dtype_str = is_bf16 ? "bfloat" : "half";

  // Tile params:
  //   BQ = query block (rows per threadgroup)
  //   BK = key block (cols per threadgroup)
  //   WM = number of simdgroups per threadgroup (cooperative scope of matmul2d)
  // For D=64:  BQ=64, BK=32, WM=2
  // For D=128: BQ=64, BK=32, WM=4
  const int BQ = 64;
  const int BK = 32;
  const int WM = (BD == 64) ? 2 : 4;

  std::ostringstream ss;
  ss << "// MFA_REQUIRE_MSL4\n";  // shader_cache.mm marker for MSL 4.0
  ss << "#include <metal_stdlib>\n";
  ss << "#include <metal_tensor>\n";
  ss << "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  ss << "using namespace metal;\n";
  ss << "using namespace mpp::tensor_ops;\n";
  ss << "\n";
  ss << "#define V6_BD " << BD << "\n";
  ss << "#define V6_BQ " << BQ << "\n";
  ss << "#define V6_BK " << BK << "\n";
  ss << "#define V6_WM " << WM << "\n";
  ss << "#define V6_TGP_SIZE (" << WM << " * 32)\n";
  ss << "using T = " << dtype_str << ";\n";
  ss << "\n";

  // Params struct — must match MFAV6NaxParams.
  ss << R"MSL(
struct MFAV6NaxParams {
  int B;
  int H;
  int gqa_factor;
  int N;
  int D;
  float scale;
  long Q_strides[3];
  long K_strides[3];
  long V_strides[3];
  long O_strides[3];
};
)MSL";

  // Kernel body.
  // Grid: (NQ, H, B) where NQ = ceil(N / BQ).
  // Each TG produces one BQ-row tile of O (one head, one batch).
  ss << R"MSL(
[[kernel, max_total_threads_per_threadgroup(V6_TGP_SIZE)]]
void v6_nax_forward(
    tensor<device T, dextents<int32_t, 2>> Qmat,   // [N, D]   per-head slice
    tensor<device T, dextents<int32_t, 2>> Kmat,   // [N, D]
    tensor<device T, dextents<int32_t, 2>> Vmat,   // [N, D]
    tensor<device T, dextents<int32_t, 2>> Omat,   // [N, D]
    constant MFAV6NaxParams& p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]])
{
    const int qb        = (int)tgid.x;          // Q-tile index
    const int q_row_lo  = qb * V6_BQ;
    const int N         = p.N;
    const int kL        = N;
    const int NK        = (kL + V6_BK - 1) / V6_BK;

    // Q @ K^T : [BQ x BD] x [BD x BK]^T = [BQ x BK]
    constexpr auto qk_desc = matmul2d_descriptor(
        V6_BQ, V6_BK, V6_BD,
        false,        // transpose_left
        true,         // transpose_right (K^T)
        true);        // relaxed_precision (FP16/BF16 friendly)
    matmul2d<qk_desc, execution_simdgroups<V6_WM>> qk_op;

    // P @ V : [BQ x BK] x [BK x BD] = [BQ x BD]  (mode = multiply_accumulate)
    constexpr auto pv_desc = matmul2d_descriptor(
        V6_BQ, V6_BD, V6_BK,
        false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroups<V6_WM>> pv_op;

    // Cooperative tensors (live in register file, distributed over WM simdgroups).
    auto S_coop = qk_op.get_destination_cooperative_tensor<
        decltype(Qmat.slice(0, 0)),
        decltype(Kmat.slice(0, 0)),
        float>();
    auto O_coop = pv_op.get_destination_cooperative_tensor<
        decltype(Qmat.slice(0, 0)),  // shape proxy - same row count as P
        decltype(Vmat.slice(0, 0)),
        float>();

    // Initialize O to zero, max to -inf, sum to 0.
    // kRowsPerThread (rows-per-thread) is implementation-defined; use a small
    // upper bound and rely on .get_capacity() / .get_mask() for correctness.
    constexpr short kMaxRowsPT = 8;
    float row_max[kMaxRowsPT];
    float row_sum[kMaxRowsPT];
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < O_coop.get_capacity(); ++i) {
        if (O_coop.get_mask(i)) O_coop[i] = 0.0f;
    }
    #pragma clang loop unroll(full)
    for (short i = 0; i < kMaxRowsPT; ++i) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
    }

    // Pre-bind Q row tile: same Q used for every K tile.
    auto Q_tile = Qmat.slice(0, q_row_lo);

    // Loop over K-tiles.
    const float log2e_scale = p.scale * 1.4426950408889634f;
    for (int kb = 0; kb < NK; ++kb) {
        const int k_row_lo = kb * V6_BK;
        // Skip past the boundary if N is not a multiple of BK
        // (assumed aligned in v1; will add ragged-tile handling later).
        if (k_row_lo >= kL) break;

        // Tile views.
        auto K_tile = Kmat.slice(0, k_row_lo);   // [BK, BD] -> we transpose in matmul
        auto V_tile = Vmat.slice(0, k_row_lo);   // [BK, BD]

        // S_coop = Q @ K^T  (relaxed precision, accumulator float)
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < S_coop.get_capacity(); ++i) {
            if (S_coop.get_mask(i)) S_coop[i] = 0.0f;
        }
        qk_op.run(Q_tile, K_tile, S_coop);

        // Apply scale (log2 domain): S *= scale * log2e
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < S_coop.get_capacity(); ++i) {
            if (S_coop.get_mask(i)) S_coop[i] *= log2e_scale;
        }

        // Online softmax: rowmax + factor + rowsum (each thread sees its
        // own row-fragment; cross-thread reduction via simd_shuffle_xor in
        // a fully-fledged port. For the v1 minimal kernel we iterate over
        // S_coop.get_multidimensional_index(i) to find row index).
        float thr_new_max[kMaxRowsPT];
        #pragma clang loop unroll(full)
        for (short i = 0; i < kMaxRowsPT; ++i) thr_new_max[i] = row_max[i];

        // Pass 1: per-thread row max
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < S_coop.get_capacity(); ++i) {
            if (!S_coop.get_mask(i)) continue;
            auto idx = S_coop.get_multidimensional_index(i);
            short r = (short)idx[0];   // local row within the BQ tile
            if (r >= 0 && r < kMaxRowsPT) {
                thr_new_max[r] = max(thr_new_max[r], (float)S_coop[i]);
            }
        }

        // Cross-thread (simdgroup) reduction over row-max
        // Note: with matmul2d cooperative tensors, the data layout across
        // threads is implementation-defined. We use simd_max as a coarse
        // reduction; for production this should use the row-wise pattern
        // from steel_attention_nax.h.
        #pragma clang loop unroll(full)
        for (short r = 0; r < kMaxRowsPT; ++r) {
            thr_new_max[r] = simd_max(thr_new_max[r]);
        }

        // factor = exp2(old_max - new_max); sum *= factor
        float factor[kMaxRowsPT];
        #pragma clang loop unroll(full)
        for (short r = 0; r < kMaxRowsPT; ++r) {
            factor[r] = fast::exp2(row_max[r] - thr_new_max[r]);
            // Sentinel: if old max was -inf, factor = 0 (correct).
            if (isinf(row_max[r])) factor[r] = 0.0f;
            row_max[r] = thr_new_max[r];
            row_sum[r] = row_sum[r] * factor[r];
        }

        // Pass 2: P = exp2(S - new_max); accumulate row_sum
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < S_coop.get_capacity(); ++i) {
            if (!S_coop.get_mask(i)) continue;
            auto idx = S_coop.get_multidimensional_index(i);
            short r = (short)idx[0];
            float p_val = fast::exp2((float)S_coop[i] - row_max[r]);
            S_coop[i] = p_val;
        }

        // Sum P along row (per-thread partial then simd_sum)
        float thr_row_partial[kMaxRowsPT] = {0};
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < S_coop.get_capacity(); ++i) {
            if (!S_coop.get_mask(i)) continue;
            auto idx = S_coop.get_multidimensional_index(i);
            short r = (short)idx[0];
            if (r >= 0 && r < kMaxRowsPT) {
                thr_row_partial[r] += (float)S_coop[i];
            }
        }
        #pragma clang loop unroll(full)
        for (short r = 0; r < kMaxRowsPT; ++r) {
            row_sum[r] += simd_sum(thr_row_partial[r]);
        }

        // Scale O by factor before P@V accumulate
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < O_coop.get_capacity(); ++i) {
            if (!O_coop.get_mask(i)) continue;
            auto idx = O_coop.get_multidimensional_index(i);
            short r = (short)idx[0];
            if (r >= 0 && r < kMaxRowsPT) {
                O_coop[i] *= factor[r];
            }
        }

        // O += P @ V  (P is in register cooperative tensor; need a tile view)
        // For the minimal kernel, we materialize P to threadgroup memory
        // then run P@V via tensor_inline view.
        threadgroup T P_smem[V6_BQ * V6_BK];
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < S_coop.get_capacity(); ++i) {
            if (!S_coop.get_mask(i)) continue;
            auto idx = S_coop.get_multidimensional_index(i);
            short r = (short)idx[0];
            short c = (short)idx[1];
            P_smem[r * V6_BK + c] = (T)S_coop[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto P_tile = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>(
            P_smem, dextents<int32_t, 2>(V6_BQ, V6_BK));
        pv_op.run(P_tile, V_tile, O_coop);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize O by row_sum
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < O_coop.get_capacity(); ++i) {
        if (!O_coop.get_mask(i)) continue;
        auto idx = O_coop.get_multidimensional_index(i);
        short r = (short)idx[0];
        if (r >= 0 && r < kMaxRowsPT) {
            O_coop[i] /= row_sum[r];
        }
    }

    // Store O
    auto O_tile = Omat.slice(0, q_row_lo);
    O_coop.store(O_tile);
}
)MSL";

  return ss.str();
}

}  // namespace mlx_mfa
