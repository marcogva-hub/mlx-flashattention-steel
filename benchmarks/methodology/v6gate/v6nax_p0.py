"""V6 NAX Phase 0 — P0.1 (denominator: effective attention TFLOPS of the current
paths on M5) + P0.5 (INT8 attention accuracy vs fp32). Throwaway probe."""
import time, math
import numpy as np
import mlx.core as mx
import mlx_mfa
from mlx_mfa import flash_attention

def bench(fn, q, k, v, warm=10, it=50):
    for _ in range(warm): mx.eval(fn(q, k, v))
    ts = []
    for _ in range(it):
        t0 = time.perf_counter(); mx.eval(fn(q, k, v)); ts.append(time.perf_counter()-t0)
    return sorted(ts)[len(ts)//2]

NAX_PEAK_FP16 = 51.8  # TFLOPS, Day-J M5 Max NAX matmul2d peak (M=128 sweet spot)
print(f"mlx_mfa {mlx_mfa.__version__} | NAX fp16 peak (Day-J) = {NAX_PEAK_FP16} TFLOPS\n")

print("=== P0.1 — effective attention TFLOPS of the CURRENT paths (the denominator) ===")
print(f"{'shape':<22}{'SDPA(auto) ms':>14}{'TFLOPS':>9}{'%NAXpeak':>10}  | {'mfa ms':>9}{'TFLOPS':>9}")
for B,H,N,D in [(2,8,2048,128),(2,8,4096,128),(2,8,8192,128),(1,32,4096,128),(2,8,4096,64)]:
    scale=1/math.sqrt(D)
    q=(mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16)
    k=(mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16)
    v=(mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16); mx.eval(q,k,v)
    flops = 4.0*B*H*N*N*D  # QK^T (2 N^2 D) + P@V (2 N^2 D), causal-agnostic upper bound
    t_sdpa = bench(lambda q,k,v: flash_attention(q,k,v,scale=scale,causal=True), q,k,v)
    t_mfa  = bench(lambda q,k,v: flash_attention(q,k,v,scale=scale,causal=True,backend="mfa"), q,k,v)
    tf_sdpa = flops/t_sdpa/1e12; tf_mfa = flops/t_mfa/1e12
    print(f"B{B}H{H}N{N}D{D:<8}{t_sdpa*1e3:>14.3f}{tf_sdpa:>9.1f}{100*tf_sdpa/NAX_PEAK_FP16:>9.0f}%  | {t_mfa*1e3:>9.3f}{tf_mfa:>9.1f}")

print("\n=== P0.5 — INT8 attention accuracy vs fp32 (gates the dequant-in-GEMM lever) ===")
def q8_perrow(x):  # symmetric per-row(last-dim) int8
    s = mx.max(mx.abs(x), axis=-1, keepdims=True) / 127.0
    s = mx.maximum(s, 1e-8)
    return mx.round(x / s).astype(mx.int8), s
for D,N in [(128,2048),(64,2048),(128,4096)]:
    B,H=1,8; scale=1/math.sqrt(D); mx.random.seed(3)
    q=(mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float32)
    k=(mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float32)
    v=(mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float32); mx.eval(q,k,v)
    # fp32 reference attention (causal)
    def attn_fp32(qf,kf,vf):
        S=(qf@kf.transpose(0,1,3,2))*scale; i=mx.arange(N)[:,None]; j=mx.arange(N)[None,:]
        S=mx.where(j>i, mx.array(-1e9), S); return mx.softmax(S,axis=-1)@vf
    ref=attn_fp32(q,k,v); mx.eval(ref)
    # INT8 QK^T: quantize q,k -> int8 matmul -> dequant by row scales; softmax; P@V in fp16
    qi,qs=q8_perrow(q); ki,ks=q8_perrow(k)
    S8=(qi.astype(mx.float32)@ki.astype(mx.float32).transpose(0,1,3,2))*qs*ks.transpose(0,1,3,2)*scale
    i=mx.arange(N)[:,None]; j=mx.arange(N)[None,:]; S8=mx.where(j>i, mx.array(-1e9), S8)
    out8=mx.softmax(S8,axis=-1)@v; mx.eval(out8)
    cos=float((mx.sum(out8*ref)/(mx.sqrt(mx.sum(out8*out8))*mx.sqrt(mx.sum(ref*ref)))).item())
    rel=float((mx.max(mx.abs(out8-ref))/ (mx.max(mx.abs(ref))+1e-9)).item())
    print(f"  D={D} N={N}: INT8-QK^T attention vs fp32 — cosine={cos:.5f}  max_rel_err={rel:.4f}")
