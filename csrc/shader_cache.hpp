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
      // --- M5+ / Metal 4 stubs (A19+, gen >= 17) ---
      // TensorOpsForward = 6,   // Reserved: Metal 4 cooperative tensor API
      //                         // Not yet implemented; M5+ hardware required.
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
    bool has_rope;   // true = in-kernel RoPE fusion; rotary_cos/sin at buffer(7/8)
    uint8_t dtype;   // 0=f16, 1=bf16, 2=f32

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
  size_t size() const { return cache_.size(); }

 private:
  ShaderCache() = default;

  void* compile_shader(
      const std::string& source,
      const std::string& function_name,
      void* device);

  std::mutex mtx_;
  std::unordered_map<KernelKey, void*, KernelKeyHash> cache_;
};

}  // namespace mlx_mfa
