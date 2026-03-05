/// shader_cache.mm — Objective-C++ implementation of ShaderCache.
///
/// Uses native Metal API (NSError, MTLDevice, MTLLibrary) rather than
/// metal-cpp to keep the ARC lifetime model simple.  All MTL objects are
/// held as void* with __bridge_retained in the C++ map; they are released
/// via __bridge_transfer when the cache is cleared.
///
/// Set env MFA_DEBUG_SHADERS=1 to dump generated Metal source to stderr
/// (gated so zero overhead in production).

#include "shader_cache.hpp"
#include "mfa_shader_gen.hpp"
#include "mfa_steel_fwd.hpp"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

namespace mlx_mfa {

// Singleton
ShaderCache& ShaderCache::get() {
  static ShaderCache instance;
  return instance;
}

// ---------------------------------------------------------------------------
// KernelKey equality and hash
// ---------------------------------------------------------------------------

bool ShaderCache::KernelKey::operator==(const KernelKey& other) const {
  return type      == other.type      &&
         head_dim  == other.head_dim  &&
         block_q   == other.block_q   &&
         block_k   == other.block_k   &&
         block_d   == other.block_d   &&
         n_warps   == other.n_warps   &&
         causal    == other.causal    &&
         sparse    == other.sparse    &&
         is_m3_plus == other.is_m3_plus &&
         has_rope     == other.has_rope     &&
         has_softcap  == other.has_softcap  &&
         has_alibi    == other.has_alibi    &&
         dtype        == other.dtype;
}

size_t ShaderCache::KernelKeyHash::operator()(const KernelKey& k) const {
  // FNV-1a mix
  size_t h = 14695981039346656037ULL;
  auto mix = [&h](uint64_t val) {
    h ^= val;
    h *= 1099511628211ULL;
  };
  mix(static_cast<uint64_t>(k.type));
  mix(static_cast<uint64_t>(k.head_dim));
  mix(static_cast<uint64_t>(k.block_q));
  mix(static_cast<uint64_t>(k.block_k));
  mix(static_cast<uint64_t>(k.block_d));
  mix(static_cast<uint64_t>(k.n_warps));
  mix(static_cast<uint64_t>(k.causal));
  mix(static_cast<uint64_t>(k.sparse));
  mix(static_cast<uint64_t>(k.is_m3_plus));
  mix(static_cast<uint64_t>(k.has_rope));
  mix(static_cast<uint64_t>(k.has_softcap));
  mix(static_cast<uint64_t>(k.has_alibi));
  mix(static_cast<uint64_t>(k.dtype));
  return h;
}

// ---------------------------------------------------------------------------
// get_or_compile (thread-safe)
// ---------------------------------------------------------------------------

void* ShaderCache::get_or_compile(const KernelKey& key, void* device) {
  {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
  }

  std::string fn_name;
  std::string source;

  using KT = KernelKey::KernelType;
  if (key.type == KT::SteelForward) {
    fn_name = "mlx_mfa_attention";
    source  = generate_steel_forward_source(key);
  } else if (key.type == KT::FlashDecodePartial) {
    fn_name = "mlx_mfa_flash_decode_partial";
    source  = generate_flash_decode_partial_source(key);
  } else if (key.type == KT::FlashDecodeReduce) {
    fn_name = "mlx_mfa_flash_decode_reduce";
    source  = generate_flash_decode_reduce_source(key);
  } else {
    // ccv-derived kernels (AttentionForward, BackwardDQ, BackwardDKV)
    fn_name = "attention";
    source  = generate_attention_source(key);
  }

  // Debug: set MFA_DEBUG_SHADERS=1 to dump generated Metal source to stderr.
  if (const char* dbg = getenv("MFA_DEBUG_SHADERS")) {
    (void)dbg;
    const char* type_str = "forward";
    if (key.type == KT::AttentionBackwardDQ)  type_str = "backwardDQ";
    if (key.type == KT::AttentionBackwardDKV) type_str = "backwardDKV";
    if (key.type == KT::SteelForward)         type_str = "steel_fwd";
    if (key.type == KT::FlashDecodePartial)   type_str = "flash_decode_partial";
    if (key.type == KT::FlashDecodeReduce)    type_str = "flash_decode_reduce";
    fprintf(stderr,
            "\n=== MFA Shader [%s D=%d bq=%d bk=%d bd=%d m3=%d dtype=%d] ===\n"
            "%s\n=== END MFA Shader ===\n",
            type_str, key.head_dim, key.block_q, key.block_k, key.block_d,
            (int)key.is_m3_plus, (int)key.dtype,
            source.c_str());
    fflush(stderr);
  }

  void* pipeline = compile_shader(source, fn_name, device);

  {
    std::lock_guard<std::mutex> lock(mtx_);
    cache_.emplace(key, pipeline);
  }
  return pipeline;
}

// ---------------------------------------------------------------------------
// Metal compilation (Objective-C)
// ---------------------------------------------------------------------------

void* ShaderCache::compile_shader(
    const std::string& source,
    const std::string& function_name,
    void* raw_device) {
  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)raw_device;
    NSError* error = nil;

    NSString* src = [NSString stringWithUTF8String:source.c_str()];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    // Allow the compiler to see the full Metal standard library.
    // 3.1+ required: bfloat4 vector type (bfloat scalar added in 3.0,
    // bfloat2/4 vectors added in 3.1 / macOS 14).
    opts.languageVersion = MTLLanguageVersion3_1;

    id<MTLLibrary> library = [device newLibraryWithSource:src
                                                  options:opts
                                                    error:&error];
    if (!library) {
      std::string msg = "MFA Metal compilation failed";
      if (error) {
        msg += ": ";
        msg += [[error localizedDescription] UTF8String];
      }
      throw std::runtime_error(msg);
    }

    NSString* fnName = [NSString stringWithUTF8String:function_name.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:fnName];
    if (!function) {
      throw std::runtime_error(
          "MFA Metal function '" + function_name + "' not found in library");
    }

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
      std::string msg = "MFA pipeline creation failed";
      if (error) {
        msg += ": ";
        msg += [[error localizedDescription] UTF8String];
      }
      throw std::runtime_error(msg);
    }

    // Explicitly retain: caller owns the object; ShaderCache::clear() calls CFRelease.
    // CFBridgingRetain works in both ARC and MRC (no-ARC) contexts.
    return (void*)CFBridgingRetain(pipeline);
  }
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

void ShaderCache::clear() {
  std::lock_guard<std::mutex> lock(mtx_);
  for (auto& [_, pipeline] : cache_) {
    if (pipeline) {
      CFRelease(pipeline);
    }
  }
  cache_.clear();
}

}  // namespace mlx_mfa
