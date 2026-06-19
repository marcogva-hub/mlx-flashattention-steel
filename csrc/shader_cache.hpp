/// shader_cache.hpp — Thread-safe cache of compiled Metal pipeline states.
///
/// Metal shader compilation (newLibraryWithSource + newComputePipelineState)
/// is ~10-50ms.  ShaderCache is a process-wide singleton that compiles each
/// (KernelType, head_dim, block_dims, causal, dtype, m3_plus) combination
/// once and caches the resulting id<MTLComputePipelineState> for reuse.
///
/// The cache uses void* with __bridge_retained so the Obj-C++ Metal objects
/// are heap-managed outside the C++ type system (ARC-safe).

#pragma once
#include <tuple>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mlx_mfa {

/// Cache for compiled Metal compute pipeline states.
///
/// MFA generates Metal shaders at runtime (JIT) parameterized by
/// head_dim, dtype, block dims, causal mask, device caps.
/// Compilation is ~10-50ms so we cache results keyed by KernelKey.
class ShaderCache {
 public:
  static ShaderCache& get();

  struct KernelKey {
    enum class KernelType : uint8_t {
      AttentionForward = 0,
      AttentionBackwardDQ = 1,
      AttentionBackwardDKV = 2,
      SteelForward = 3,          // STEEL-style cooperative forward kernel
      FlashDecodePartial = 4,    // Flash Decoding Phase 1: partial attn per split
      FlashDecodeReduce  = 5,    // Flash Decoding Phase 2: LSE reduce over splits
      SteelBackwardDQ  = 6,      // STEEL native backward dQ (f16/bf16, D<=128)
      SteelBackwardDKV = 7,      // STEEL native backward dK/dV (f16/bf16, D<=128)
      SteelVarlenForward = 8,    // STEEL varlen forward: (total_q_tiles, H, 1) grid
      PagedKVGather      = 9,    // Paged KV gather: pool → contiguous BHND (Track EB)
      PagedSteelForward  = 10,   // STEEL forward with kernel-level paged KV (Track FD)
      SageForward        = 11,   // SageAttention: int8 Q/K loads + fp16 GEMM (Track KB)
      QuantizePerBlock   = 12,   // Fused per-block INT8 quantization (Phase 4-A.1)
      ScatterKV          = 13,   // In-place scatter write for paged KV pool (Phase 4-C.1/E.2)
      SmoothQuantizeMean = 14,   // Fused smooth_k pass 1: per-channel mean over S (Phase 1.1)
      SmoothQuantizeK    = 15,   // Fused smooth_k pass 2: subtract mean + quantize (Phase 1.1)
      SteelForwardV2         = 16,  // STEEL V2: sequential K/V phases, BQ=32 BK=64/32
      SteelV2SplitKPartial   = 17,  // STEEL V2 split-K Phase 1: partial attn per K-range
      // SteelV2SplitKReduce reuses FlashDecodeReduce (type 5) — no new enum needed.
      // --- CP1/CP2: D-split V2 (D=256 2-pass, D=512 4-pass, BD_HALF=128) ---
      // Dispatched by v2_dsplit_eligible in eval_gpu() for D=256/512, f16/bf16.
      // generate_steel_v2_dsplit_source() in mfa_steel_fwd_v2.cpp.
      // Performance: ~1.0× SDPA (vs old V1 ~0.57× for D=256). No RoPE support.
      SteelV2DSplit256       = 18,  // D=256 V2 D-split: 2 BD_HALF=128 inner-loop passes
      SteelV2DSplit512       = 19,  // D=512 V2 D-split: 4 BD_HALF=128 inner-loop passes
      // --- V3: separate K_smem + V_smem, 2 barriers/iter (was 4 in V2) ---
      // D=64 all gens, D=128 M1/M2 only (BK=32; M3+ BK=64 exceeds 32KB TGP).
      // generate_steel_v3_source() in mfa_steel_fwd_v3.cpp.
      SteelForwardV3             = 20,  // STEEL V3: separate K/V smem, 2 barriers/iter
      // --- V4 / V5: RETIRED — removed from the build (Lot-2 chore) ---
      // Both were experimental, opt-in-only (MFA_ENABLE_V4 / MFA_ENABLE_V5),
      // never auto-routed.  Retired: mfa_steel_fwd_v{4,5}.cpp dropped from
      // CMakeLists.txt, the eval_gpu dispatch blocks removed from
      // mfa_attention.cpp, and the env knobs removed from _knobs.py
      // KNOWN_KNOBS.  V5 was M5-validated before removal and showed no
      // advantage (3.1–4.4× slower than the routed NAX/SDPA default).  The two
      // enum constants below are RETAINED ONLY to avoid renumbering the rest of
      // the enum — there is no dispatch path, no generator, and no env var for
      // them.  Source recoverable via the archive/v4-v5-prototypes tag.
      SteelForwardV4             = 21,  // RETIRED (Lot-2) — unused; kept to avoid renumber
      SteelForwardV5             = 23,  // RETIRED (Lot-2) — unused; kept to avoid renumber
      GNAForward                 = 24,  // GNA: inline 3D window check, no block_mask
      // 25-26: reserved
      PagedVarlenForward         = 27,  // Fused packed varlen Q + paged KV gather
      PagedVarlenTQForward       = 28,  // TurboQuant: packed uint8 K + centroid dequant
      // --- V6 NAX: MPP cooperative tensor attention (M5+/Apple10) ---
      // Uses MetalPerformancePrimitives matmul2d for Q@K^T and P@V GEMMs.
      // Compiled with MSL 4.0 (required for `<metal_tensor>` and MPP).
      // Strict support matrix: D ∈ {64,128}, FP16/BF16, dense forward only,
      // no causal/mask/bias/softcap/window/sparse/RoPE in v1.
      // See docs/v6-nax/ for the architecture and implementation notes.
      SteelForwardV6NAX          = 22,  // V6 NAX: MPP-accelerated forward (M5+, MSL 4)
    };

    KernelType type;
    int  head_dim;
    int  block_q;   // parallelization block (8*n_warps)
    int  block_k;   // traversal block
    int  block_d;   // head sub-tile
    int  n_warps;   // SIMD groups per threadgroup
    bool causal;
    bool sparse;     // true = block-sparse path (device uchar* block_mask at buffer(6))
    bool is_m3_plus; // GPUFamily(1009): preferAsyncCache vs preferAsyncLoad
    bool has_rope;           // true = in-kernel RoPE fusion; rotary_cos/sin at buffer(7/8)
    bool rope_interleaved;   // true = LLaMA-style (pair d*2,d*2+1); false = GPT-NeoX (d,d+D/2)
    bool has_softcap;        // true = tanh softcapping (Gemma 2 / Grok); p->softcap > 0
    bool has_alibi;          // true = ALiBi per-head bias; slopes at buffer(9)
    bool has_attn_bias;      // true = additive bias tensor at buffer(10)
    uint8_t attn_bias_mode;  // 0=[B,H,Nq,Nkv], 1=[1,1,1,Nkv], 2=[1,H,1,Nkv], 3=[1,H,Nq,Nkv]
    bool has_window;         // true = native sliding window (window_left in params)
    uint8_t dtype;       // 0=f16, 1=bf16, 2=f32
    int  gqa_factor;     // H_q / H_kv; 1 = standard MHA (baked into shader as #define)

    // Track 6: single declaration of the affecting-input set.
    auto tie() const {
      return std::tie(type, head_dim, block_q, block_k, block_d, n_warps, causal, sparse, is_m3_plus, has_rope, rope_interleaved, has_softcap, has_alibi, has_attn_bias, attn_bias_mode, has_window, dtype, gqa_factor);
    }
    bool operator==(const KernelKey& other) const;
  };

  struct KernelKeyHash {
    size_t operator()(const KernelKey& k) const;
  };

  /// Get or compile a pipeline state. Thread-safe.
  /// device: id<MTLDevice> as void* (ARC-unmanaged, caller keeps alive).
  /// Returns: id<MTLComputePipelineState> as void* (__bridge_retained).
  void* get_or_compile(const KernelKey& key, void* device);

  void   clear();
  size_t size() { std::lock_guard<std::mutex> lock(mtx_); return cache_.size(); }

  // Public for V6 NAX bring-up probe. Caller owns the returned pipeline.
  void* compile_shader(
      const std::string& source,
      const std::string& function_name,
      void* device);

 private:
  ShaderCache() = default;

  std::mutex mtx_;
  std::unordered_map<KernelKey, void*, KernelKeyHash> cache_;
};

}  // namespace mlx_mfa
