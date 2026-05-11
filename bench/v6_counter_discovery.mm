// V6 NAX — MTLCounterSampleBuffer counter discovery
//
// Enumerates all MTLCounterSet objects exposed by MTLDevice and lists
// the MTLCounter objects within each. Output documents what GPU counters
// Apple makes available on M5 Max for programmatic profiling.
//
// Build:
//   clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation \
//     bench/v6_counter_discovery.mm -o /tmp/v6_counter_discovery
//
// Usage: /tmp/v6_counter_discovery [--json]

#import <Metal/Metal.h>
#include <stdio.h>
#include <string>

int main(int argc, const char** argv) {
  bool json_mode = (argc > 1 && std::string(argv[1]) == "--json");

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "ERROR: no Metal device\n");
      return 1;
    }

    if (!json_mode) {
      printf("Device: %s\n", [device.name UTF8String]);
      printf("supportsFamily(Apple9): %d\n", [device supportsFamily:MTLGPUFamilyApple9]);
      printf("maxThreadgroupMemoryLength: %lu B (%.1f KB)\n",
             (unsigned long)device.maxThreadgroupMemoryLength,
             device.maxThreadgroupMemoryLength / 1024.0);
      printf("\n");
    }

    NSArray<id<MTLCounterSet>>* counterSets = device.counterSets;
    if (json_mode) printf("[\n");
    NSUInteger n_sets = counterSets ? counterSets.count : 0;
    if (!json_mode) printf("Counter sets exposed by device: %lu\n", (unsigned long)n_sets);

    for (NSUInteger i = 0; i < n_sets; i++) {
      id<MTLCounterSet> set = counterSets[i];
      NSArray<id<MTLCounter>>* counters = set.counters;
      if (json_mode) {
        if (i > 0) printf(",\n");
        printf("  {\"set_name\": \"%s\", \"counters\": [",
               [set.name UTF8String]);
        for (NSUInteger j = 0; j < counters.count; j++) {
          if (j > 0) printf(", ");
          printf("\"%s\"", [counters[j].name UTF8String]);
        }
        printf("]}");
      } else {
        printf("  Counter Set #%lu: %s (%lu counters)\n",
               (unsigned long)i, [set.name UTF8String], (unsigned long)counters.count);
        for (NSUInteger j = 0; j < counters.count; j++) {
          printf("    [%lu] %s\n",
                 (unsigned long)j, [counters[j].name UTF8String]);
        }
      }
    }
    if (json_mode) printf("\n]\n");

    // Probe sample buffer support per counter set
    if (!json_mode) {
      printf("\nSample buffer support test:\n");
      for (NSUInteger i = 0; i < n_sets; i++) {
        id<MTLCounterSet> set = counterSets[i];
        MTLCounterSampleBufferDescriptor* desc =
            [[MTLCounterSampleBufferDescriptor alloc] init];
        desc.counterSet = set;
        desc.label = @"probe";
        desc.sampleCount = 2;
        // Try storage modes
        for (auto mode : {MTLStorageModeShared, MTLStorageModePrivate}) {
          desc.storageMode = mode;
          NSError* error = nil;
          id<MTLCounterSampleBuffer> buf =
              [device newCounterSampleBufferWithDescriptor:desc error:&error];
          const char* mode_str =
              mode == MTLStorageModeShared ? "shared" : "private";
          if (buf) {
            printf("  %s [%s]: OK\n", [set.name UTF8String], mode_str);
          } else {
            printf("  %s [%s]: FAIL — %s\n",
                   [set.name UTF8String], mode_str,
                   error ? [[error localizedDescription] UTF8String] : "(nil error)");
          }
        }
      }
    }
  }
  return 0;
}
