# M5 Max — Threadgroup Memory Budget Verification

**Date:** 2026-05-03
**Hardware:** Apple M5 Max (40 GPU cores, 64 NAX, applegpu_g17s)
**Source of probe:** Direct Metal API call via Obj-C++ helper

---

## TL;DR

**`maxThreadgroupMemoryLength` = 32,768 bytes (32 KB) on M5 Max.**

The "Apple9+ dynamic shader core memory" idea — that newer architectures
might allow larger TGP allocations than the legacy 32 KB cap — does NOT
apply to attention kernels on M5 Max. The hardware/driver still enforces
the 32 KB ceiling for `setThreadgroupMemoryLength()` calls. Our
autoresearch's 32 KB constraint was correct.

---

## Verification (programmatic Metal API call)

Written `/tmp/probe_device.mm`:
```objc
#import <Metal/Metal.h>
#include <stdio.h>
int main() {
  @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    printf("name: %s\n", [d.name UTF8String]);
    printf("maxThreadgroupMemoryLength: %lu bytes (%.1f KB)\n",
           (unsigned long)d.maxThreadgroupMemoryLength,
           d.maxThreadgroupMemoryLength / 1024.0);
    printf("maxBufferLength: %llu bytes (%.1f GB)\n", ...);
    MTLSize maxGroup = d.maxThreadsPerThreadgroup;
    printf("maxThreadsPerThreadgroup: %lux%lux%lu\n", ...);
    printf("supportsFamily(Apple9): %d\n", [d supportsFamily:MTLGPUFamilyApple9]);
    printf("argumentBuffersSupport: %ld\n", (long)d.argumentBuffersSupport);
    printf("hasUnifiedMemory: %d\n", d.hasUnifiedMemory);
    printf("recommendedMaxWorkingSetSize: %llu (%.1f GB)\n", ...);
  }
  return 0;
}
```

Build: `clang++ -fobjc-arc -framework Metal -framework Foundation`.

### Output (M5 Max)

```
name: Apple M5 Max
maxThreadgroupMemoryLength: 32768 bytes (32.0 KB)
maxBufferLength: 86586540032 bytes (80.6 GB)
maxThreadsPerThreadgroup: 1024x1024x1024
supportsFamily(Apple9): 1
argumentBuffersSupport: 1
hasUnifiedMemory: 1
recommendedMaxWorkingSetSize: 115448725504 (107.5 GB)
```

---

## Interpretation

| Limit | M5 Max value | Implication for V6 NAX |
|-------|-------------|------------------------|
| `maxThreadgroupMemoryLength` | **32 KB** | Hardware ceiling. Our `is_valid_config` correctly rejects configs exceeding 32 KB. No new config opens up. |
| `maxThreadsPerThreadgroup` | 1024 (cube) | We currently use 512 (16 SG × 32). Could use 1024 (32 SG × 32). Tested in autoresearch — SG=32 regresses (register spill / scheduling). |
| `maxBufferLength` | 80.6 GB | No relevance for kernel; per-buffer cap. |
| `supportsFamily(Apple9)` | true | Confirms Apple9+ feature set (cooperative tensors, NAX MPP). |

**Note**: `MTLGPUFamilyApple10` (M5 family / A19 base) is not yet
exposed as a constant in macOS 26.x SDK headers. The `Apple9` query
covers M3 → M5 inclusive.

---

## Why "dynamic shader core memory" does not apply

Apple's marketing for M3+ "dynamic shader core memory" refers to the
GPU core's ability to **dynamically partition** L1 cache and shader
core memory between vertex/fragment/compute work as needed. It does
NOT increase the per-threadgroup TGP allocation visible to the
programmer — the `MTLDevice.maxThreadgroupMemoryLength` API gives
the actual ceiling.

The 32 KB on M5 is consistent with M2/M3/M4. Apple has not raised
this limit in any post-2023 architecture.

---

## What our autoresearch already did right

The Phase 3B autoresearch (`bench/v6_nax_autoresearch_v2.py`) uses:

```python
# is_valid_config in bench script
elem_size = 2  # FP16
tgmem = block_r * block_c * exec_sg * elem_size
if tgmem > 32768:
    return False, f"tgmem {tgmem}B > 32KB"
```

Verified against actual hardware. Our 245-config sweep had **166 valid
configs** (the rest excluded by this 32 KB check). All winning configs
land at TG memory ≤ 24 KB (75% of cap), giving headroom for compiler-
allocated stack/spills.

---

## What COULD change the budget

1. **Apple announces new device family** — would need to retest on M6+.
   No public roadmap suggests imminent change.
2. **Per-pipeline-state TG memory** — Some Metal compute pipelines
   advertise *higher* `staticThreadgroupMemoryLength` if all TGP
   allocations are static (compile-time-sized). Our V6 uses dynamic
   `setThreadgroupMemoryLength()`. Could potentially get more if
   restructured to static, but no documented post-32KB ceiling for
   compute pipelines on Apple Silicon.
3. **`MTLCompileOptions.maxTotalThreadgroupMemory`** — does not exist
   in the public Metal API.

---

## Action

**No action needed.** The 32 KB bound is real, hardware-enforced, and
correctly reflected in our autoresearch's `is_valid_config()` filter.

The discussion of larger tile configs in the original Sprint prompt
(R=16, C=128, SG=16 = 65,536 bytes, currently impossible) is correctly
ruled out by hardware. We do NOT have unexplored space to autoresearch
on this dimension.
