// II-10 — PASS-1: matmul-grade top-K index selection (Approach 5 refined).
// Structure adapted from the II-2R attention prototype: per-SG 16 Q rows,
// fp16 (16,32,16) MMA QK^T tiles, fp32 dest. Selection: each of lanes
// 0..15 OWNS one row's top-K state (unsorted array + running min) in
// registers; per tile, owner lanes gather the 32 tile scores for their
// row via simd_shuffle and insert candidates. Output: int32 indices
// [B,H,N,K]. Kill gate: >8ms at B1 H16 N=S=4096 D=128 K=64.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant uint N_CONST [[function_constant(0)]];
constant uint S_CONST [[function_constant(1)]];

#define D_DIM 128
#define K_TOP 64
#define BK 32

kernel void topk_pass1(
    const device half* Q  [[buffer(0)]],
    const device half* K  [[buffer(1)]],
    device int*  IDX       [[buffer(2)]],   // [H, N, K_TOP]
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lid  [[thread_index_in_simdgroup]])
{
  // Per-SG staging tile: scores laid out [row][col] so owner lanes read
  // their row contiguously (the 32-way broadcast-poll variant measured
  // 66ms — shuffle-grade scalar work).
  threadgroup float stage_all[4][16][BK];
  threadgroup float (&stage)[16][BK] = stage_all[sgid];
  const uint N = N_CONST, S = S_CONST;
  const uint head = tgid.y;
  const uint q0 = tgid.x * 64 + sgid * 16;
  if (q0 >= N) return;
  const device half* Qh = Q + ((ulong)head * N + q0) * D_DIM;
  const device half* Kh = K + (ulong)head * S * D_DIM;

  constexpr auto dQK = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<dQK, metal::execution_simdgroup> op;
  using cta_t = decltype(op.get_left_input_cooperative_tensor<half,half,float>());
  using ctb_t = decltype(op.get_right_input_cooperative_tensor<half,half,float>());
  using ctc_t = decltype(op.get_destination_cooperative_tensor<cta_t,ctb_t,float>());

  // Q fp16 in plain registers (8 d-chunks x 8 elems).
  half Qreg[8][8];
  short qa_col[8], qa_row[8];
  {
    cta_t f = op.get_left_input_cooperative_tensor<half,half,float>();
    for (ushort i = 0; i < f.get_capacity() && i < 8; ++i) {
      auto ix = f.get_multidimensional_index(i);
      qa_col[i] = ix[0]; qa_row[i] = ix[1];
    }
  }
  #pragma unroll
  for (short id = 0; id < 8; ++id)
    for (short i = 0; i < 8; ++i)
      Qreg[id][i] = Qh[(ulong)qa_row[i] * D_DIM + id * 16 + qa_col[i]];

  // coordinate tables
  short kb_col[16], kb_row[16];
  {
    ctb_t f = op.get_right_input_cooperative_tensor<half,half,float>();
    for (ushort i = 0; i < f.get_capacity() && i < 16; ++i) {
      auto ix = f.get_multidimensional_index(i);
      kb_row[i] = ix[1]; kb_col[i] = ix[0];
    }
  }
  short sa_col[16], sa_row[16];
  {
    ctc_t f = op.get_destination_cooperative_tensor<cta_t,ctb_t,float>();
    for (ushort i = 0; i < f.get_capacity() && i < 16; ++i) {
      auto ix = f.get_multidimensional_index(i);
      sa_col[i] = ix[0]; sa_row[i] = ix[1];
    }
  }

  // Per-owner-lane top-K state: lanes 0..15 own row (q0 + lid).
  float heap_s[K_TOP];
  int   heap_i[K_TOP];
  float run_min = -INFINITY;
  short run_min_pos = 0;
  short filled = 0;
  const bool owner = (lid < 16);

  for (uint kb = 0; kb < S; kb += BK) {
    // S-tile = Q @ K^T (fp32 dest)
    ctc_t Sacc = op.get_destination_cooperative_tensor<cta_t,ctb_t,float>();
    #pragma unroll
    for (ushort i = 0; i < 16; ++i) Sacc[i] = 0;
    #pragma unroll
    for (short id = 0; id < 8; ++id) {
      cta_t Qf = op.get_left_input_cooperative_tensor<half,half,float>();
      #pragma unroll
      for (ushort i = 0; i < 8; ++i) Qf[i] = Qreg[id][i];
      ctb_t Kf = op.get_right_input_cooperative_tensor<half,half,float>();
      #pragma unroll
      for (ushort i = 0; i < 16; ++i)
        Kf[i] = Kh[(ulong)(kb + kb_row[i]) * D_DIM + id * 16 + kb_col[i]];
      op.run(Qf, Kf, Sacc);
    }

    // Stage scores into TGM by coordinates; owners consume their row.
    #pragma unroll
    for (ushort i = 0; i < 16; ++i)
      stage[sa_row[i]][sa_col[i]] = Sacc[i];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (owner) {
      #pragma unroll
      for (ushort c = 0; c < BK; ++c) {
        const float v = stage[lid][c];
        const int col_global = (int)kb + (int)c;
        if (filled < K_TOP) {
          heap_s[filled] = v; heap_i[filled] = col_global;
          ++filled;
          if (filled == K_TOP) {
            run_min = INFINITY;
            #pragma unroll
            for (short j = 0; j < K_TOP; ++j)
              if (heap_s[j] < run_min) { run_min = heap_s[j]; run_min_pos = j; }
          }
        } else if (v > run_min) {
          heap_s[run_min_pos] = v; heap_i[run_min_pos] = col_global;
          run_min = INFINITY;
          #pragma unroll
          for (short j = 0; j < K_TOP; ++j)
            if (heap_s[j] < run_min) { run_min = heap_s[j]; run_min_pos = j; }
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (owner && (q0 + lid) < N) {
    device int* out = IDX + ((ulong)head * N + (q0 + lid)) * K_TOP;
    for (short j = 0; j < K_TOP; ++j) out[j] = heap_i[j];
  }
}
)MSL";

int main(int argc, char** argv) {
  @autoreleasepool {
    const int H = (argc > 1) ? atoi(argv[1]) : 16;
    const int N = (argc > 2) ? atoi(argv[2]) : 4096;
    const int S = N, D = 128, K = 64;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__fp16> q((size_t)H*N*D), k((size_t)H*S*D);
    for (auto& x : q) x = (__fp16)nd(rng);
    for (auto& x : k) x = (__fp16)nd(rng);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion4_0;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
        [NSString stringWithUTF8String:kSrc] options:opts error:&err];
    if (!lib) { printf("COMPILE FAIL:\n%s\n", [[err localizedDescription] UTF8String]); return 1; }
    MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
    uint nN = N, nS = S;
    [fc setConstantValue:&nN type:MTLDataTypeUInt atIndex:0];
    [fc setConstantValue:&nS type:MTLDataTypeUInt atIndex:1];
    id<MTLFunction> fn = [lib newFunctionWithName:@"topk_pass1" constantValues:fc error:&err];
    if (!fn) { printf("FN FAIL: %s\n", [[err localizedDescription] UTF8String]); return 1; }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { printf("PSO FAIL: %s\n", [[err localizedDescription] UTF8String]); return 1; }

    id<MTLBuffer> bQ = [dev newBufferWithBytes:q.data() length:q.size()*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bK = [dev newBufferWithBytes:k.data() length:k.size()*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bI = [dev newBufferWithLength:(size_t)H*N*K*4 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> qd = [dev newCommandQueue];
    auto dispatch = [&]() -> double {
      id<MTLCommandBuffer> cb = [qd commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:bQ offset:0 atIndex:0];
      [enc setBuffer:bK offset:0 atIndex:1];
      [enc setBuffer:bI offset:0 atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake((N + 63)/64, H, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [enc endEncoding];
      [cb commit]; [cb waitUntilCompleted];
      if (cb.error) { printf("EXEC FAIL\n"); exit(1); }
      return [cb GPUEndTime] - [cb GPUStartTime];
    };
    dispatch();

    // correctness: set-parity vs CPU top-K (FP16-tie tolerant: compare
    // the SCORE multiset, not the index set).
    if (N <= 1024) {
      int bad_rows = 0;
      const int* idx = (const int*)bI.contents;
      for (int h = 0; h < H && bad_rows < 5; ++h)
        for (int n = 0; n < N && bad_rows < 5; ++n) {
          std::vector<float> row(S);
          for (int s = 0; s < S; ++s) {
            float acc = 0;
            for (int d = 0; d < D; ++d)
              acc += (float)q[((size_t)h*N+n)*D+d] * (float)k[((size_t)h*S+s)*D+d];
            row[s] = acc;
          }
          std::vector<float> sorted = row;
          std::nth_element(sorted.begin(), sorted.begin() + (S-K), sorted.end());
          std::vector<float> ref(sorted.begin() + (S-K), sorted.end());
          std::sort(ref.begin(), ref.end());
          std::vector<float> got;
          for (int j = 0; j < K; ++j) got.push_back(row[idx[((size_t)h*N+n)*K + j]]);
          std::sort(got.begin(), got.end());
          float maxd = 0;
          for (int j = 0; j < K; ++j) maxd = std::max(maxd, std::fabs(got[j]-ref[j]));
          if (maxd > 1e-2f) { bad_rows++; if (bad_rows<=2) printf("  row h%d n%d score-set maxd=%.4f\n", h, n, maxd); }
        }
      printf("correctness H%d N%d: %s\n", H, N, bad_rows == 0 ? "** SCORE-SET PARITY **" : "FAIL");
    }
    std::vector<double> ts;
    for (int i = 0; i < 12; ++i) ts.push_back(dispatch());
    std::sort(ts.begin(), ts.end());
    printf("PASS-1 H%d N=S=%d D128 K64: median %.3f ms (kill gate: 8ms)\n",
           H, N, ts[ts.size()/2]*1e3);
    return 0;
  }
}
