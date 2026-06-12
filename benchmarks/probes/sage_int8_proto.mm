// II-2R Phase R.2 — Sage-NAX int8 attention forward prototype (D=128).
// Recipe (R.1 gate): per-row symmetric int8 Q,K; per-channel symmetric
// int8 V; S = int8 QK^T -> int32 -> fp32 dequant -> online softmax fp32;
// P quantized at fixed 127 (P <= 1); PV int8 -> int32 accumulated into
// fp32 O with softmax corrections; v_scale[d]/127 applied at writeout.
//
// Tiles: BQ=64 (WM=4 SGs x 16 rows), BK=32. QK mma: desc(16,32,16) x8
// d-chunks. PV mma: desc(16,32,32) x4 n-chunks. Full-cooperative
// register form only (the 2.00x form; device-tensor int8 is fp16-speed).
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

static int g_ablate = 0;
static const char* kSrc = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant uint  N_CONST  [[function_constant(0)]];   // q length
constant uint  S_CONST  [[function_constant(1)]];   // kv length
constant float SM_SCALE [[function_constant(2)]];   // 1/sqrt(D) * log2e

#define D_DIM 128
#define BQ 64
#define BK 32
#define WM 4
#ifndef ABLATE
#define ABLATE 0
#endif

kernel void sage_int8_fwd(
    const device int8_t*  Q   [[buffer(0)]],   // [B,H,N,D] int8
    const device int8_t*  K   [[buffer(1)]],   // [B,H,S,D] int8
    const device half*    V   [[buffer(2)]],   // [B,H,S,D] fp16
    const device float*   qs  [[buffer(3)]],   // [B,H,N] per-row scale
    const device float*   ks  [[buffer(4)]],   // [B,H,S]
    const device float*   vs  [[buffer(5)]],   // [B,H,D] per-channel scale
    device half*          O   [[buffer(6)]],   // [B,H,N,D]
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lid  [[thread_index_in_simdgroup]])
{
  const uint N = N_CONST, S = S_CONST;
  const uint head = tgid.y, batch = tgid.z;
  const uint q0 = tgid.x * BQ + sgid * 16;       // this SG's first Q row
  if (q0 >= N) return;

  const ulong qk_head_off = ((ulong)batch * 0 + head) * (ulong)N * D_DIM;  // B=1
  const ulong kv_head_off = ((ulong)head) * (ulong)S * D_DIM;
  const device int8_t* Qh = Q + qk_head_off + (ulong)q0 * D_DIM;
  const device int8_t* Kh = K + kv_head_off;
  const device half* Vh = V + kv_head_off;
  const device float*  qsh = qs + (ulong)head * N + q0;
  const device float*  ksh = ks + (ulong)head * S;
  const device float*  vsh = vs + (ulong)head * D_DIM;

  // ---- ops ----
  constexpr auto dQK = matmul2d_descriptor(16, 32, 16, false, true, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dQK, metal::execution_simdgroup> opQK;
  // PV in fp16 (variant A): int8-PV measured 8.2 ms of the 12.3 ms
  // kernel regardless of shape/transpose (the dest create/zero/readback
  // structure dominates); fp16 PV deletes the P-quantization tax, the
  // V quant pass, and the /127 + v_scale epilogue, and improves
  // accuracy (sim cos 0.999967 vs 0.999950).
  constexpr auto dPV = matmul2d_descriptor(16, 32, 16, false, false, true,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dPV, metal::execution_simdgroup> opPV;

  using cta_t = decltype(opQK.get_left_input_cooperative_tensor<int8_t,int8_t,int32_t>());
  using ctb_t = decltype(opQK.get_right_input_cooperative_tensor<int8_t,int8_t,int32_t>());
  using ctc_t = decltype(opQK.get_destination_cooperative_tensor<cta_t,ctb_t,int32_t>());
  using pva_t = decltype(opPV.get_left_input_cooperative_tensor<half,half,float>());
  using pvb_t = decltype(opPV.get_right_input_cooperative_tensor<half,half,float>());
  using pvc_t = decltype(opPV.get_destination_cooperative_tensor<pva_t,pvb_t,float>());

  // ---- per-thread persistent state ----
  // Q int8 in plain registers (coop tensors cannot be arrayed); the
  // coordinate map of ct_a is instance-independent, so capture once.
  int8_t Qreg[8][8];
  short  qa_row[8];   // gmi row per slot (same across chunks)
  short  qa_col[8];   // gmi k_inner per slot
  {
    cta_t f = opQK.get_left_input_cooperative_tensor<int8_t,int8_t,int32_t>();
    for (ushort i = 0; i < f.get_capacity() && i < 8; ++i) {
      auto ix = f.get_multidimensional_index(i);    // (k_inner, m_row)
      qa_col[i] = ix[0]; qa_row[i] = ix[1];
    }
  }
  #pragma unroll
  for (short id = 0; id < 8; ++id)
    for (short i = 0; i < 8; ++i)
      Qreg[id][i] = Qh[(ulong)qa_row[i] * D_DIM + id * 16 + qa_col[i]];
  // O accumulator fp32 via PV dest layout: 4 chunks of (16x32).
  float Oacc[4][16];
  #pragma unroll
  for (short c = 0; c < 4; ++c)
    for (short i = 0; i < 16; ++i) Oacc[c][i] = 0.0f;

  // ---- loop-invariant coordinate tables (gmi is lane-pure; capture once) ----
  short kb_col[16], kb_row[16];        // Kf: (k_inner, n_col)
  {
    ctb_t f = opQK.get_right_input_cooperative_tensor<int8_t,int8_t,int32_t>();
    for (ushort i = 0; i < f.get_capacity() && i < 16; ++i) {
      auto ix = f.get_multidimensional_index(i);
      kb_row[i] = ix[1]; kb_col[i] = ix[0];
    }
  }
  short sa_col[16], sa_row[16], sa_slot[16];
  short my_rows[2] = {-1, -1};
  {
    ctc_t f = opQK.get_destination_cooperative_tensor<cta_t,ctb_t,int32_t>();
    for (ushort i = 0; i < f.get_capacity() && i < 16; ++i) {
      auto ix = f.get_multidimensional_index(i);
      sa_col[i] = ix[0]; sa_row[i] = ix[1];
      short slot = (my_rows[0] == sa_row[i]) ? 0 :
                   (my_rows[1] == sa_row[i]) ? 1 :
                   (my_rows[0] < 0 ? (my_rows[0] = sa_row[i], 0)
                                   : (my_rows[1] = sa_row[i], 1));
      sa_slot[i] = slot;
    }
  }
  short pf_perm0[8], pf_perm1[8];      // Pf half-tile slots -> Sacc slots
  {
    pva_t pf = opPV.get_left_input_cooperative_tensor<half,half,float>();
    for (ushort i = 0; i < pf.get_capacity() && i < 8; ++i) {
      auto ixp = pf.get_multidimensional_index(i);   // (k_inner 0..15, m_row)
      pf_perm0[i] = 0; pf_perm1[i] = 0;
      for (ushort j = 0; j < 16; ++j) {
        if (sa_col[j] == ixp[0] && sa_row[j] == ixp[1]) pf_perm0[i] = (short)j;
        if (sa_col[j] == ixp[0] + 16 && sa_row[j] == ixp[1]) pf_perm1[i] = (short)j;
      }
    }
  }
  short vb_kin[16], vb_ncol[16];       // fp16 Vf: gmi (n_col d, k_row s)
  ushort vb_cap2;
  {
    pvb_t f = opPV.get_right_input_cooperative_tensor<half,half,float>();
    vb_cap2 = f.get_capacity();
    for (ushort i = 0; i < f.get_capacity() && i < 16; ++i) {
      auto ix = f.get_multidimensional_index(i);
      vb_ncol[i] = ix[0]; vb_kin[i] = ix[1];
    }
  }
  // packed-load eligibility: groups of 4 slots with same row + consecutive
  // cols (the nax fragment layout) -> single 4-byte reads.
  bool k_pack = true, v_pack = true;
  #pragma unroll
  for (short g = 0; g < 4; ++g) {
    #pragma unroll
    for (short j = 1; j < 4; ++j) {
      k_pack = k_pack && (kb_row[g*4+j] == kb_row[g*4]) &&
               (kb_col[g*4+j] == kb_col[g*4] + j);
    }
  }
  (void)v_pack;

  short oc_col[16], oc_row[16], oc_slot[16];   // PVc dest coords
  {
    pvc_t f = opPV.get_destination_cooperative_tensor<pva_t,pvb_t,float>();
    for (ushort i = 0; i < f.get_capacity() && i < 16; ++i) {
      auto ix = f.get_multidimensional_index(i);
      oc_col[i] = ix[0]; oc_row[i] = ix[1];
      oc_slot[i] = (my_rows[0] == oc_row[i]) ? 0 : 1;
    }
  }

  // Persistent PV accumulators (one per 32-d chunk) — created ONCE.
  // The per-tile create/zero/readback lifecycle measured ~10 us per
  // dest cycle (TGM staging); persistent dests with in-place softmax
  // rescale eliminate 3/4 of those cycles AND the Oacc shadow copy.
  pvc_t PVc0 = opPV.get_destination_cooperative_tensor<pva_t,pvb_t,float>();
  pvc_t PVc1 = opPV.get_destination_cooperative_tensor<pva_t,pvb_t,float>();
  pvc_t PVc2 = opPV.get_destination_cooperative_tensor<pva_t,pvb_t,float>();
  pvc_t PVc3 = opPV.get_destination_cooperative_tensor<pva_t,pvb_t,float>();
  #pragma unroll
  for (ushort i = 0; i < 16; ++i) { PVc0[i]=0; PVc1[i]=0; PVc2[i]=0; PVc3[i]=0; }

  // online softmax state
  float m_run[2] = {-INFINITY, -INFINITY};
  float l_run[2] = {0.0f, 0.0f};

  // ---- K loop ----
  for (uint kb = 0; kb < S; kb += BK) {
    // S = Q @ K^T  (int32, 16x32)
    ctc_t Sacc = opQK.get_destination_cooperative_tensor<cta_t,ctb_t,int32_t>();
    for (ushort i = 0; i < Sacc.get_capacity(); ++i) Sacc[i] = 0;
    #pragma unroll
    for (short id = 0; id < 8; ++id) {
      cta_t Qf = opQK.get_left_input_cooperative_tensor<int8_t,int8_t,int32_t>();
      for (ushort i = 0; i < Qf.get_capacity() && i < 8; ++i) Qf[i] = Qreg[id][i];
      ctb_t Kf = opQK.get_right_input_cooperative_tensor<int8_t,int8_t,int32_t>();
      #pragma unroll
      for (ushort i = 0; i < 16; ++i)
        Kf[i] = Kh[(ulong)(kb + kb_row[i]) * D_DIM + id * 16 + kb_col[i]];
      opQK.run(Qf, Kf, Sacc);
    }

    // dequant + per-row max over this tile (fp32)
    float Sf[16];
#if ABLATE >= 2
    // mma-floor ablation: skip dequant/softmax; constant P
    #pragma unroll
    for (ushort i = 0; i < 16; ++i) Sf[i] = (float)Sacc[i] * 1e-9f;
    float corr[2] = {1.0f, 1.0f};
    float tile_sum[2] = {0.0f, 0.0f};
    l_run[0] += 1.0f; l_run[1] += 1.0f;
    pva_t Pf0 = opPV.get_left_input_cooperative_tensor<half,half,float>();
    pva_t Pf1 = opPV.get_left_input_cooperative_tensor<half,half,float>();
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) { Pf0[i] = (half)(Sf[i] != 12345.f); Pf1[i] = Pf0[i]; }
#else
    float tile_max[2] = {-INFINITY, -INFINITY};
    #pragma unroll
    for (ushort i = 0; i < 16; ++i) {
      float v = (float)Sacc[i] * qsh[sa_row[i]] * ksh[kb + sa_col[i]] * SM_SCALE;
      Sf[i] = v;
      tile_max[sa_slot[i]] = max(tile_max[sa_slot[i]], v);
    }
    // reduce row max across the 4 lanes sharing each row (xor 1 and 8)
    #pragma unroll
    for (short s2 = 0; s2 < 2; ++s2) {
      float v = tile_max[s2];
      v = max(v, simd_shuffle_xor(v, (ushort)1));
      v = max(v, simd_shuffle_xor(v, (ushort)8));
      tile_max[s2] = v;
    }

    // new running max + corrections
    float corr[2], m_new[2];
    #pragma unroll
    for (short s2 = 0; s2 < 2; ++s2) {
      m_new[s2] = max(m_run[s2], tile_max[s2]);
      corr[s2] = (m_run[s2] == -INFINITY) ? 0.0f
                 : fast::exp2(m_run[s2] - m_new[s2]);
      l_run[s2] *= corr[s2];
      m_run[s2] = m_new[s2];
    }

    // P = exp2(Sf - m) in fp16 fragments (no quantization)
    pva_t Pf0 = opPV.get_left_input_cooperative_tensor<half,half,float>();
    pva_t Pf1 = opPV.get_left_input_cooperative_tensor<half,half,float>();
    float tile_sum[2] = {0.0f, 0.0f};
    #pragma unroll
    for (ushort i = 0; i < 16; ++i) {
      float p = fast::exp2(Sf[i] - m_run[sa_slot[i]]);
      tile_sum[sa_slot[i]] += p;
      Sf[i] = p;
    }
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) {
      Pf0[i] = (half)Sf[pf_perm0[i]];
      Pf1[i] = (half)Sf[pf_perm1[i]];
    }
#endif
    // row-sum reduce across lanes
    #pragma unroll
    for (short s2 = 0; s2 < 2; ++s2) {
      float v = tile_sum[s2];
      v += simd_shuffle_xor(v, (ushort)1);
      v += simd_shuffle_xor(v, (ushort)8);
      l_run[s2] += v;
    }

    // Persistent-dest PV: rescale accumulators in place, then mma.
#if ABLATE >= 1
    Oacc[0][0] += (float)Pf0[0] + (float)Pf1[0];   // keep live
#else
    #pragma unroll
    for (ushort i = 0; i < 16; ++i) {
      const float cr = corr[oc_slot[i]];
      PVc0[i] *= cr; PVc1[i] *= cr; PVc2[i] *= cr; PVc3[i] *= cr;
    }
    #pragma unroll
    for (short c = 0; c < 4; ++c) {
      pvb_t Vf0 = opPV.get_right_input_cooperative_tensor<half,half,float>();
      pvb_t Vf1 = opPV.get_right_input_cooperative_tensor<half,half,float>();
      #pragma unroll
      for (ushort i = 0; i < 16; ++i) {
        Vf0[i] = Vh[(ulong)(kb + vb_kin[i]) * D_DIM + c * 32 + vb_ncol[i]];
        Vf1[i] = Vh[(ulong)(kb + 16 + vb_kin[i]) * D_DIM + c * 32 + vb_ncol[i]];
      }
      thread pvc_t& PVc = (c == 0) ? PVc0 : (c == 1) ? PVc1 : (c == 2) ? PVc2 : PVc3;
      opPV.run(Pf0, Vf0, PVc);
      opPV.run(Pf1, Vf1, PVc);
    }
#endif
  }

  // ---- writeout: O / l ----
  {
    device half* Oh = O + qk_head_off + (ulong)q0 * D_DIM;
    const float rl0 = 1.0f / max(l_run[0], 1e-20f);
    const float rl1 = 1.0f / max(l_run[1], 1e-20f);
    #pragma unroll
    for (ushort i = 0; i < 16; ++i) {
      const float rl = oc_slot[i] == 0 ? rl0 : rl1;
      if ((uint)(q0 + oc_row[i]) < N) {
        const float v0 = (ABLATE >= 1) ? Oacc[0][i] : (float)PVc0[i];
        Oh[(ulong)oc_row[i] * D_DIM + 0  + oc_col[i]] = (half)(v0 * rl);
        Oh[(ulong)oc_row[i] * D_DIM + 32 + oc_col[i]] = (half)((float)PVc1[i] * rl);
        Oh[(ulong)oc_row[i] * D_DIM + 64 + oc_col[i]] = (half)((float)PVc2[i] * rl);
        Oh[(ulong)oc_row[i] * D_DIM + 96 + oc_col[i]] = (half)((float)PVc3[i] * rl);
      }
    }
  }
}
)MSL";

// ---------- host ----------
struct Tensors {
  int B = 1, H, N, S, D = 128;
  std::vector<float> q, k, v;          // fp32 reference inputs
  std::vector<int8_t> qi, ki, vi;
  std::vector<float> qs, ks, vs;       // scales
};

static void quantize(Tensors& t) {
  auto rowq = [&](const std::vector<float>& x, std::vector<int8_t>& xi,
                  std::vector<float>& sc, int rows) {
    xi.resize((size_t)rows * t.D); sc.resize(rows);
    for (int r = 0; r < rows; ++r) {
      float amax = 1e-8f;
      for (int d = 0; d < t.D; ++d) amax = std::max(amax, std::fabs(x[(size_t)r*t.D+d]));
      sc[r] = amax / 127.0f;
      for (int d = 0; d < t.D; ++d)
        xi[(size_t)r*t.D+d] = (int8_t)std::lrint(x[(size_t)r*t.D+d] / sc[r]);
    }
  };
  rowq(t.q, t.qi, t.qs, t.H * t.N);
  rowq(t.k, t.ki, t.ks, t.H * t.S);
  // V per-channel (per head)
  t.vi.resize(t.v.size()); t.vs.resize((size_t)t.H * t.D);
  for (int h = 0; h < t.H; ++h)
    for (int d = 0; d < t.D; ++d) {
      float amax = 1e-8f;
      for (int s = 0; s < t.S; ++s)
        amax = std::max(amax, std::fabs(t.v[((size_t)h*t.S+s)*t.D+d]));
      t.vs[(size_t)h*t.D+d] = amax / 127.0f;
      for (int s = 0; s < t.S; ++s)
        t.vi[((size_t)h*t.S+s)*t.D+d] =
            (int8_t)std::lrint(t.v[((size_t)h*t.S+s)*t.D+d] / t.vs[(size_t)h*t.D+d]);
    }
}

// fp32 reference attention (from ORIGINAL fp inputs)
static std::vector<float> reference(const Tensors& t) {
  std::vector<float> out((size_t)t.H * t.N * t.D, 0.f);
  const float scale = 1.0f / std::sqrt((float)t.D);
  std::vector<float> row(t.S);
  for (int h = 0; h < t.H; ++h)
    for (int n = 0; n < t.N; ++n) {
      float mx_ = -1e30f;
      for (int s = 0; s < t.S; ++s) {
        float acc = 0;
        for (int d = 0; d < t.D; ++d)
          acc += t.q[((size_t)h*t.N+n)*t.D+d] * t.k[((size_t)h*t.S+s)*t.D+d];
        row[s] = acc * scale; mx_ = std::max(mx_, row[s]);
      }
      float l = 0;
      for (int s = 0; s < t.S; ++s) { row[s] = std::exp(row[s]-mx_); l += row[s]; }
      for (int s = 0; s < t.S; ++s) {
        const float w = row[s] / l;
        for (int d = 0; d < t.D; ++d)
          out[((size_t)h*t.N+n)*t.D+d] += w * t.v[((size_t)h*t.S+s)*t.D+d];
      }
    }
  return out;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    const int H = (argc > 1) ? atoi(argv[1]) : 2;
    const int N = (argc > 2) ? atoi(argv[2]) : 256;
    g_ablate = (argc > 3) ? atoi(argv[3]) : 0;
    Tensors t; t.H = H; t.N = N; t.S = N;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    t.q.resize((size_t)H*N*t.D); t.k.resize(t.q.size()); t.v.resize(t.q.size());
    for (auto& x : t.q) x = nd(rng);
    for (auto& x : t.k) x = nd(rng);
    for (auto& x : t.v) x = nd(rng);
    quantize(t);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion4_0;
    NSError* err = nil;
    char defbuf[64];
    snprintf(defbuf, sizeof(defbuf), "#define ABLATE %d\n", g_ablate);
    std::string full = std::string(defbuf) + kSrc;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
        [NSString stringWithUTF8String:full.c_str()] options:opts error:&err];
    if (!lib) { printf("COMPILE FAIL:\n%s\n", [[err localizedDescription] UTF8String]); return 1; }
    MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
    uint nN = N, nS = N;
    float smsc = (1.0f / std::sqrt(128.0f)) * 1.4426950408889634f;
    [fc setConstantValue:&nN type:MTLDataTypeUInt atIndex:0];
    [fc setConstantValue:&nS type:MTLDataTypeUInt atIndex:1];
    [fc setConstantValue:&smsc type:MTLDataTypeFloat atIndex:2];
    id<MTLFunction> fn = [lib newFunctionWithName:@"sage_int8_fwd"
                                   constantValues:fc error:&err];
    if (!fn) { printf("FN FAIL: %s\n", [[err localizedDescription] UTF8String]); return 1; }
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { printf("PSO FAIL: %s\n", [[err localizedDescription] UTF8String]); return 1; }

    auto mkbuf = [&](const void* p, size_t n) {
      return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> bQ = mkbuf(t.qi.data(), t.qi.size());
    id<MTLBuffer> bK = mkbuf(t.ki.data(), t.ki.size());
    std::vector<__fp16> vh(t.v.size());
    for (size_t i = 0; i < t.v.size(); ++i) vh[i] = (__fp16)t.v[i];
    id<MTLBuffer> bV = mkbuf(vh.data(), vh.size() * 2);
    id<MTLBuffer> bqs = mkbuf(t.qs.data(), t.qs.size()*4);
    id<MTLBuffer> bks = mkbuf(t.ks.data(), t.ks.size()*4);
    id<MTLBuffer> bvs = mkbuf(t.vs.data(), t.vs.size()*4);
    id<MTLBuffer> bO = [dev newBufferWithLength:(size_t)H*N*128*2
                                        options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue];

    auto dispatch = [&]() -> double {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:bQ offset:0 atIndex:0];
      [enc setBuffer:bK offset:0 atIndex:1];
      [enc setBuffer:bV offset:0 atIndex:2];
      [enc setBuffer:bqs offset:0 atIndex:3];
      [enc setBuffer:bks offset:0 atIndex:4];
      [enc setBuffer:bvs offset:0 atIndex:5];
      [enc setBuffer:bO offset:0 atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake((N + 63) / 64, H, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * 4, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) { printf("EXEC FAIL\n"); exit(1); }
      return [cb GPUEndTime] - [cb GPUStartTime];
    };
    dispatch();

    // correctness (CPU reference is single-threaded; cap at N<=1024)
    if (N > 1024) {
      std::vector<double> ts;
      for (int i = 0; i < 16; ++i) ts.push_back(dispatch());
      std::sort(ts.begin(), ts.end());
      printf("H=%d N=%d kernel median: %.3f ms (no ref at this size)\n",
             H, N, ts[ts.size()/2] * 1e3);
      return 0;
    }
    auto ref = reference(t);
    const __fp16* o = (const __fp16*)bO.contents;
    double se = 0, mx_ = 0; double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      double d = (double)o[i] - ref[i];
      se += d * d; mx_ = std::max(mx_, std::fabs(d));
      dot += (double)o[i] * ref[i]; na += (double)o[i]*o[i]; nb += ref[i]*ref[i];
    }
    double rmse = std::sqrt(se / ref.size());
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    printf("H=%d N=%d: rmse=%.5f max=%.4f cos=%.6f\n", H, N, rmse, mx_, cos);

    // timing (if large enough)
    if (N >= 2048) {
      std::vector<double> ts;
      for (int i = 0; i < 12; ++i) ts.push_back(dispatch());
      std::sort(ts.begin(), ts.end());
      printf("kernel median: %.3f ms\n", ts[ts.size()/2] * 1e3);
    }
    return 0;
  }
}
