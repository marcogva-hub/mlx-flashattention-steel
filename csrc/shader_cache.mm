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
#include "mfa_key_tie.hpp"
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
#include "mfa_gna_fwd.hpp"
#include "mfa_steel_paged_varlen_fwd.hpp"
#include "mfa_steel_paged_varlen_tq_fwd.hpp"
#include "mfa_bool_env.hpp"

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
  // Track 6: derived from the tie() declaration — cannot diverge.
  return tie() == other.tie();
}

size_t ShaderCache::KernelKeyHash::operator()(const KernelKey& k) const {
  return mlx_mfa_keys::hash_tie(k.tie());
}

// ---------------------------------------------------------------------------
// CP9 / R8: precompiled (AOT) metallib cache — content-addressed
// ---------------------------------------------------------------------------
// The CP4c async_v2.metallib fast path (simdgroup_async_copy, tried BEFORE the JIT
// on macOS 14/15) was RETIRED in 2.62.2 (review 2026-09, BLD-04, decision Marco):
// its source had been frozen since 2026-03-11 (pre-RC-A causal zone, ungated
// non-causal K-loop limit) so every later kernel fix bypassed it on that floor.
// csrc/async_v2_kernel.metal is kept as a historical reference only.

#ifndef MLX_MFA_VERSION
#define MLX_MFA_VERSION "unversioned"
#endif

/// FNV-1a 64-bit of the generated MSL source, hex.  A content hash, not security.
static std::string fnv1a64_hex(const std::string& s) {
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
  return std::string(buf);
}

/// R8 (review 2026-09): canonical AOT filename for an AOT-eligible key, "" otherwise.
/// The old name (D, BK, is_m3_plus, dtype, causal) omitted block_q / n_warps /
/// steel_msl_mode, the mlx-mfa version and the source, while the loaded pipeline was
/// cached under the FULL KernelKey: a BQ32 build served BQ64 host geometry
/// (MFA_V2_BQ64=1, err 0.95) and metallibs built before a kernel fix kept loading
/// after an upgrade.  Now: full geometry + version + FNV-1a-64(source) — a metallib
/// compiled from any other source / version / geometry is never matched (no user
/// file is deleted).  compile_metallib.py reads this exact name from the
/// MFA_DEBUG_SHADERS header: one source of truth.
static std::string aot_metallib_filename(const ShaderCache::KernelKey& key,
                                         const std::string& source) {
  using KT = ShaderCache::KernelKey::KernelType;
  const bool is_std_v2 = (key.type == KT::SteelForwardV2);
  const bool is_dsplit = (key.type == KT::SteelV2DSplit256 ||
                          key.type == KT::SteelV2DSplit512);
  if (!is_std_v2 && !is_dsplit) return "";
  // Only standard single-head MHA without extra features is precompiled.
  if (key.sparse || key.has_rope || key.has_softcap || key.has_alibi ||
      key.has_attn_bias || key.has_window || key.gqa_factor != 1) return "";
  char buf[320];
  snprintf(buf, sizeof(buf),
           "%s_D%d_BQ%d_BK%d_BD%d_W%d_M%d_dtype%d_causal%d_msl%d_v%s_h%s.metallib",
           is_std_v2 ? "v2" : "v2_dsplit", key.head_dim, key.block_q,
           key.block_k, key.block_d, key.n_warps, (int)key.is_m3_plus,
           (int)key.dtype, (int)key.causal, (int)key.steel_msl_mode,
           MLX_MFA_VERSION, fnv1a64_hex(source).c_str());
  return std::string(buf);
}

/// Load ~/.mlx_mfa/metallib/<aot_name> when it exists.  Returns a retained
/// id<MTLComputePipelineState> (as void*) or nullptr (caller JIT-compiles).
static void* try_precompiled_pipeline(const std::string& aot_name,
                                      const std::string& fn_name,
                                      void* raw_device) {
  if (aot_name.empty()) return nullptr;
  @autoreleasepool {
    NSString* home = NSHomeDirectory();
    NSURL* dir_url  = [NSURL fileURLWithPath:
        [home stringByAppendingPathComponent:@".mlx_mfa/metallib"]];
    NSURL* file_url = [dir_url URLByAppendingPathComponent:
        [NSString stringWithUTF8String:aot_name.c_str()]];
    // Bail out quickly when the file is absent (no Metal exception thrown).
    if (![[NSFileManager defaultManager] fileExistsAtPath:[file_url path]]) {
      return nullptr;
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)raw_device;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:file_url error:&error];
    if (!library) return nullptr;  // fall through to JIT
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:fn_name.c_str()]];
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
  } else if (key.type == KT::GNAForward) {
    fn_name = "mlx_mfa_gna_attention";
    source  = generate_gna_forward_source(key);
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

  // CP9 / R8: content-addressed AOT metallib (skips ~50 ms of JIT).  Looked up
  // AFTER source generation because the filename embeds the source hash.
  const std::string aot_name = aot_metallib_filename(key, source);
  if (void* pre = try_precompiled_pipeline(aot_name, fn_name, device)) {
    std::lock_guard<std::mutex> lock(mtx_);
    cache_.emplace(key, pre);
    return pre;
  }

  // Debug: set MFA_DEBUG_SHADERS=1 to dump generated Metal source to stderr
  // (JIT path only; `aot=` is the canonical precompiled filename, or "-").
  if (get_bool_env("MFA_DEBUG_SHADERS")) {
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
    if (key.type == KT::GNAForward)             type_str = "gna_fwd";
    fprintf(stderr,
            "\n=== MFA Shader [%s D=%d bq=%d bk=%d bd=%d m3=%d dtype=%d aot=%s] ===\n"
            "%s\n=== END MFA Shader ===\n",
            type_str, key.head_dim, key.block_q, key.block_k, key.block_d,
            (int)key.is_m3_plus, (int)key.dtype,
            aot_name.empty() ? "-" : aot_name.c_str(),
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
    // Default: MSL 3.1 — bfloat2/4 vectors (added in macOS 14, 3.1+).
    // V6 NAX kernels need MSL 4.0 for `<metal_tensor>` + MPP cooperative
    // tensor APIs. Detect via the marker `// MFA_REQUIRE_MSL4` injected at
    // the top of the source by the V6 generator.
    if (source.find("// MFA_REQUIRE_MSL41") != std::string::npos) {
      // MTLLanguageVersion4_1 (macOS 27 / metal4.1): fp8/fp4 packed format
      // types (metal_fp8_e4m3_format / metal_fp4_e2m1_format) + their
      // matmul2d low-precision support. Probe/characterization only today —
      // no shipping shader uses this marker (checked FIRST since "…MSL41"
      // contains the "…MSL4" substring).
      opts.languageVersion = (MTLLanguageVersion)((4 << 16) + 1);
    } else if (source.find("// MFA_REQUIRE_MSL4") != std::string::npos) {
      // MTLLanguageVersion4_0 (M5+, macOS 26 / iOS 19+).
      // Use the integer encoding to stay compatible with older SDKs that
      // may not have MTLLanguageVersion4_0 in their headers.
      opts.languageVersion = (MTLLanguageVersion)((4 << 16) + 0);
    } else {
      opts.languageVersion = MTLLanguageVersion3_1;
    }

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
