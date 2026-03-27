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
#include "mfa_steel_bwd.hpp"
#include "mfa_paged_gather.hpp"
#include "mfa_sage_fwd.hpp"
#include "mfa_quantize.hpp"
#include "mfa_scatter.hpp"
#include "mfa_smooth_quant.hpp"
#include "mfa_steel_fwd_v2.hpp"
#include "mfa_steel_fwd_v3.hpp"
#include "mfa_steel_fwd_v4.hpp"
#include "mfa_steel_fwd_v5.hpp"
// GNA native kernel removed (sparse path is faster) — include was here
#include "mfa_steel_paged_varlen_fwd.hpp"
#include "mfa_steel_paged_varlen_tq_fwd.hpp"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
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
         has_rope          == other.has_rope          &&
         rope_interleaved  == other.rope_interleaved  &&
         has_softcap       == other.has_softcap       &&
         has_alibi         == other.has_alibi         &&
         has_window        == other.has_window        &&
         dtype        == other.dtype        &&
         gqa_factor   == other.gqa_factor;
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
  mix(static_cast<uint64_t>(k.rope_interleaved));
  mix(static_cast<uint64_t>(k.has_softcap));
  mix(static_cast<uint64_t>(k.has_alibi));
  mix(static_cast<uint64_t>(k.has_window));
  mix(static_cast<uint64_t>(k.dtype));
  mix(static_cast<uint64_t>(k.gqa_factor));
  return h;
}

// ---------------------------------------------------------------------------
// CP4c: Async metallib fast path (ships with package in mlx_mfa/precompiled/)
// ---------------------------------------------------------------------------

/// Try to load the async V2 metallib (uses simdgroup_async_copy hardware DMA).
/// Checks mlx_mfa/precompiled/async_v2.metallib relative to the installed
/// package dylib.  Uses MTLFunctionConstantValues for FC_CAUSAL and
/// FC_GQA_FACTOR so one metallib serves all flag combinations.
/// Set MFA_DISABLE_ASYNC=1 to skip this path entirely.
/// Returns a retained id<MTLComputePipelineState> (as void*) or nullptr.
static void* try_async_pipeline(const ShaderCache::KernelKey& key,
                                void* raw_device) {
  using KT = ShaderCache::KernelKey::KernelType;
  const bool ir_debug = (std::getenv("MFA_IR_INVESTIGATE") != nullptr);

  // Only D=64/128 SteelForwardV2, f16 only, no extra features
  if (key.type != KT::SteelForwardV2) return nullptr;
  if (key.head_dim != 64 && key.head_dim != 128) return nullptr;
  if (key.dtype != 0) return nullptr;  // f16 only
  if (key.sparse || key.has_rope || key.has_softcap ||
      key.has_alibi || key.has_window) return nullptr;
  if (std::getenv("MFA_DISABLE_ASYNC")) {
    if (ir_debug) {
      NSLog(@"[MFA-IR-INVESTIGATE] Async pipeline: disabled by MFA_DISABLE_ASYNC");
    }
    return nullptr;
  }

  @autoreleasepool {
    // Resolve precompiled dir relative to our shared library's location.
    // The dylib lives at mlx_mfa/_ext.cpython-*.so and the metallib is at
    // mlx_mfa/precompiled/async_v2.metallib — same parent directory.
    Dl_info dl_info;
    if (dladdr((void*)&ShaderCache::get, &dl_info) == 0) {
      if (ir_debug) {
        NSLog(@"[MFA-IR-INVESTIGATE] Async pipeline: dladdr failed");
      }
      return nullptr;
    }
    NSString* dylib_path = [NSString stringWithUTF8String:dl_info.dli_fname];
    NSString* pkg_dir    = [dylib_path stringByDeletingLastPathComponent];
    NSURL* metallib_url  = [NSURL fileURLWithPath:
        [[pkg_dir stringByAppendingPathComponent:@"precompiled"]
                  stringByAppendingPathComponent:@"async_v2.metallib"]];

    if (ir_debug) {
      NSLog(@"[MFA-IR-INVESTIGATE] Async pipeline: loading %@", [metallib_url path]);
    }
    if (![[NSFileManager defaultManager] fileExistsAtPath:[metallib_url path]]) {
      if (ir_debug) {
        NSLog(@"[MFA-IR-INVESTIGATE] Async pipeline: FAILED (metallib missing)");
      }
      return nullptr;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)raw_device;
    NSError* error = nil;

    id<MTLLibrary> library = [device newLibraryWithURL:metallib_url error:&error];
    if (!library) {
      if (ir_debug) {
        NSLog(@"[MFA-IR-INVESTIGATE] Async pipeline: FAILED library load (%@)",
              error ? [error localizedDescription] : @"unknown");
      }
      return nullptr;
    }

    NSString* fn_name = (key.head_dim == 64)
        ? @"mlx_mfa_v2_async_attention"
        : @"mlx_mfa_v2_async_attention_d128";

    // Set function constants: index 0 = FC_CAUSAL (bool), index 1 = FC_GQA_FACTOR (ushort)
    MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
    bool   causal_val = key.causal;
    ushort gqa_val    = (ushort)key.gqa_factor;
    [constants setConstantValue:&causal_val type:MTLDataTypeBool   atIndex:0];
    [constants setConstantValue:&gqa_val    type:MTLDataTypeUShort atIndex:1];

    id<MTLFunction> function = [library newFunctionWithName:fn_name
                                            constantValues:constants
                                                     error:&error];
    if (!function) {
      if (ir_debug) {
        NSLog(@"[MFA-IR-INVESTIGATE] Async pipeline: FAILED function load (%@)",
              error ? [error localizedDescription] : @"unknown");
      }
      return nullptr;
    }

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
      if (ir_debug) {
        NSLog(@"[MFA-IR-INVESTIGATE] Async pipeline: FAILED pipeline creation (%@)",
              error ? [error localizedDescription] : @"unknown");
      }
      return nullptr;
    }
    if (ir_debug) {
      NSLog(@"[MFA-IR-INVESTIGATE] Async pipeline: SUCCESS (%@)", fn_name);
    }

    return (void*)CFBridgingRetain(pipeline);
  }
}

// ---------------------------------------------------------------------------
// CP9: Precompiled metallib fast path
// ---------------------------------------------------------------------------

/// Try to load a precompiled .metallib for SteelForwardV2 or V2 D-split keys.
/// Returns a retained id<MTLComputePipelineState> (as void*) on success,
/// or nullptr when no matching file exists (caller falls through to JIT).
///
/// Filename schemes (must match mlx_mfa/compile_metallib.py):
///   Standard V2:  v2_D{D}_BK{BK}_M{is_m3_plus}_dtype{dtype_code}_causal{0|1}.metallib
///   D-split V2:   v2_dsplit_D{D}_BK{BK}_M{is_m3_plus}_dtype{dtype_code}_causal{0|1}.metallib
/// Located in: ~/.mlx_mfa/metallib/
static void* try_precompiled_pipeline(const ShaderCache::KernelKey& key,
                                      void* raw_device) {
  using KT = ShaderCache::KernelKey::KernelType;

  const bool is_std_v2    = (key.type == KT::SteelForwardV2);
  const bool is_dsplit256 = (key.type == KT::SteelV2DSplit256);
  const bool is_dsplit512 = (key.type == KT::SteelV2DSplit512);

  if (!is_std_v2 && !is_dsplit256 && !is_dsplit512) return nullptr;

  // Only precompile standard single-head MHA without extra features.
  if (key.sparse || key.has_rope || key.has_softcap ||
      key.has_alibi || key.has_window || key.gqa_factor != 1) return nullptr;

  @autoreleasepool {
    NSString* fname;
    NSString* fn_name;
    if (is_std_v2) {
      fname   = [NSString stringWithFormat:
          @"v2_D%d_BK%d_M%d_dtype%d_causal%d.metallib",
          key.head_dim, key.block_k,
          (int)key.is_m3_plus, (int)key.dtype, (int)key.causal];
      fn_name = @"mlx_mfa_v2_attention";
    } else {
      fname   = [NSString stringWithFormat:
          @"v2_dsplit_D%d_BK%d_M%d_dtype%d_causal%d.metallib",
          key.head_dim, key.block_k,
          (int)key.is_m3_plus, (int)key.dtype, (int)key.causal];
      fn_name = @"mlx_mfa_v2_dsplit_attention";
    }

    NSString* home = NSHomeDirectory();
    NSURL* dir_url  = [NSURL fileURLWithPath:
        [home stringByAppendingPathComponent:@".mlx_mfa/metallib"]];
    NSURL* file_url = [dir_url URLByAppendingPathComponent:fname];

    // Bail out quickly when the file is absent (no Metal exception thrown).
    if (![[NSFileManager defaultManager] fileExistsAtPath:[file_url path]]) {
      return nullptr;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)raw_device;
    NSError* error = nil;

    id<MTLLibrary> library = [device newLibraryWithURL:file_url error:&error];
    if (!library) return nullptr;  // fall through to JIT

    id<MTLFunction> function = [library newFunctionWithName:fn_name];
    if (!function) return nullptr;

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) return nullptr;

    return (void*)CFBridgingRetain(pipeline);
  }
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

  // CP4c: async metallib (hardware DMA, ships with package) — best throughput.
  if (void* async = try_async_pipeline(key, device)) {
    std::lock_guard<std::mutex> lock(mtx_);
    cache_.emplace(key, async);
    return async;
  }

  // CP9: try to load a precompiled metallib — skips ~50ms JIT compilation.
  if (void* pre = try_precompiled_pipeline(key, device)) {
    std::lock_guard<std::mutex> lock(mtx_);
    cache_.emplace(key, pre);
    return pre;
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
  } else if (key.type == KT::SteelBackwardDQ) {
    fn_name = "mlx_mfa_bwd_dq";
    source  = generate_steel_backward_dq_source(key);
  } else if (key.type == KT::SteelBackwardDKV) {
    fn_name = "mlx_mfa_bwd_dkv";
    source  = generate_steel_backward_dkv_source(key);
  } else if (key.type == KT::SteelVarlenForward) {
    fn_name = "mlx_mfa_steel_varlen_forward";
    source  = generate_steel_varlen_forward_source(key);
  } else if (key.type == KT::PagedKVGather) {
    fn_name = "paged_kv_gather";
    source  = generate_paged_kv_gather_source(key.dtype == 0);
  } else if (key.type == KT::PagedSteelForward) {
    fn_name = "mlx_mfa_paged_attention";
    source  = generate_paged_steel_forward_source(key);
  } else if (key.type == KT::SageForward) {
    fn_name = "mlx_mfa_sage_attention";
    source  = generate_sage_forward_source(key);
  } else if (key.type == KT::QuantizePerBlock) {
    fn_name = "mfa_quantize_per_block";
    source  = generate_quantize_per_block_source(key.dtype == 0 ? "half" : "bfloat");
  } else if (key.type == KT::ScatterKV) {
    fn_name = "mfa_scatter_kv";
    source  = generate_scatter_kv_source(key.dtype == 0 ? "half" : "bfloat");
  } else if (key.type == KT::SmoothQuantizeMean) {
    fn_name = "mfa_smooth_k_mean";
    source  = generate_smooth_k_mean_source(key.dtype == 0 ? "half" : "bfloat");
  } else if (key.type == KT::SmoothQuantizeK) {
    fn_name = "mfa_smooth_k_quant";
    source  = generate_smooth_k_quant_source(key.dtype == 0 ? "half" : "bfloat");
  } else if (key.type == KT::SteelForwardV2) {
    fn_name = "mlx_mfa_v2_attention";
    source  = generate_steel_v2_source(key);
  } else if (key.type == KT::SteelV2SplitKPartial) {
    fn_name = "mlx_mfa_v2_splitk_partial";
    source  = generate_steel_v2_splitk_partial_source(key);
  } else if (key.type == KT::SteelV2DSplit256 ||
             key.type == KT::SteelV2DSplit512) {
    fn_name = "mlx_mfa_v2_dsplit_attention";
    source  = generate_steel_v2_dsplit_source(key);
  } else if (key.type == KT::SteelForwardV3) {
    fn_name = "mlx_mfa_v3_attention";
    source  = generate_steel_v3_source(key);
  } else if (key.type == KT::SteelForwardV4) {
    fn_name = "mlx_mfa_v4_attention";
    source  = generate_steel_v4_source(key);
  } else if (key.type == KT::SteelForwardV5) {
    fn_name = "mlx_mfa_v5_attention";
    source  = generate_steel_v5_source(key);
  } else if (key.type == KT::PagedVarlenForward) {
    fn_name = "mlx_mfa_paged_varlen_forward";
    source  = generate_paged_varlen_forward_source(key);
  } else if (key.type == KT::PagedVarlenTQForward) {
    fn_name = "mlx_mfa_paged_varlen_tq_forward";
    source  = generate_paged_varlen_tq_forward_source(key);
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
    if (key.type == KT::SteelBackwardDQ)      type_str = "steel_bwd_dq";
    if (key.type == KT::SteelBackwardDKV)     type_str = "steel_bwd_dkv";
    if (key.type == KT::SteelVarlenForward)   type_str = "steel_varlen_fwd";
    if (key.type == KT::PagedKVGather)        type_str = "paged_kv_gather";
    if (key.type == KT::PagedSteelForward)   type_str = "paged_steel_fwd";
    if (key.type == KT::SageForward)         type_str = "sage_fwd";
    if (key.type == KT::QuantizePerBlock)    type_str = "quantize_per_block";
    if (key.type == KT::ScatterKV)           type_str = "scatter_kv";
    if (key.type == KT::SmoothQuantizeMean)  type_str = "smooth_k_mean";
    if (key.type == KT::SmoothQuantizeK)     type_str = "smooth_k_quant";
    if (key.type == KT::SteelForwardV2)         type_str = "steel_fwd_v2";
    if (key.type == KT::SteelV2SplitKPartial)  type_str = "steel_v2_splitk_partial";
    if (key.type == KT::SteelV2DSplit256)       type_str = "steel_v2_dsplit256";
    if (key.type == KT::SteelV2DSplit512)       type_str = "steel_v2_dsplit512";
    if (key.type == KT::SteelForwardV3)         type_str = "steel_fwd_v3";
    if (key.type == KT::SteelForwardV4)         type_str = "steel_fwd_v4";
    if (key.type == KT::SteelForwardV5)         type_str = "steel_fwd_v5";
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
