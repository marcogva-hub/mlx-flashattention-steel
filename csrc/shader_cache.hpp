#pragma once

#include <string>
#include <unordered_map>

// Forward declare Metal types (avoid requiring metal-cpp in headers)
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLComputePipelineState;
#else
namespace MTL { class Device; class ComputePipelineState; }
#endif

namespace mlx_mfa {

/// Cache for compiled Metal compute pipeline states.
///
/// MFA generates Metal shaders at runtime (JIT) parameterized by
/// head_dim, dtype, block dims, causal mask, device caps.
/// Compilation is ~10-50ms so we cache results.
class ShaderCache {
 public:
  static ShaderCache& get();

  struct KernelKey {
    enum class KernelType : uint8_t {
      AttentionForward = 0,
      AttentionBackwardDQ = 1,
      AttentionBackwardDKV = 2,
    };

    KernelType type;
    int head_dim;
    int block_q;
    int block_k;
    int block_d;
    int n_warps;
    bool causal;
    bool bf16_emulation;
    uint8_t dtype;  // 0=f16, 1=bf16, 2=f32

    bool operator==(const KernelKey& other) const;
  };

  struct KernelKeyHash {
    size_t operator()(const KernelKey& k) const;
  };

  /// Get or compile a kernel. Thread-safe via simple mutex.
  void* get_or_compile(const KernelKey& key, void* device);

  void clear();
  size_t size() const { return cache_.size(); }

 private:
  ShaderCache() = default;

  std::string generate_shader_source(const KernelKey& key);
  void* compile_shader(
      const std::string& source,
      const std::string& function_name,
      void* device);

  std::unordered_map<KernelKey, void*, KernelKeyHash> cache_;
};

}  // namespace mlx_mfa
