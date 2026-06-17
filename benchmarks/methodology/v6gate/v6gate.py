"""V6 measurement gate: Q1 (non-dense effective TFLOPS, verify the cited 18-25) +
Q2 (kernel-switch cost, in-graph vs raw). 3-replicate median. Throwaway probe.
Effective TFLOPS = real (active) FLOPs / time (no causal-½ on throughput; count active work)."""
import time, math
import numpy as np
import mlx.core as mx
import mlx_mfa
from mlx_mfa import flash_attention, flash_attention_sparse, flash_attention_gna, make_causal_block_mask

def med3(fn):  # 3 replicate medians of (warm 8, time 30)
    reps=[]
    for _ in range(3):
        for _ in range(8): mx.eval(fn())
        ts=[]
        for _ in range(30):
            t0=time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter()-t0)
        reps.append(sorted(ts)[15])
    reps.sort(); return reps[1], (max(reps)-min(reps))/reps[1]  # median, CV-ish range

NAX_CEIL="26-46 TFLOPS (SDPA register cooperative_tensor, DEDUCED ceiling per re-scope)"
print(f"mlx_mfa {mlx_mfa.__version__} | NAX non-dense ceiling = {NAX_CEIL}\n")
print("=== Q1 — current non-dense effective TFLOPS (active-FLOP; verify cited 18-25) ===")

B,H,D=2,8,128; scale=1/math.sqrt(D)
# windowed V3 (auto), causal, N=4096, window W=1024: active keys ~ band
N=4096; W=1024
q=(mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16); k=q*1.0; v=q*1.0
q=(mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16)
k=(mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16)
v=(mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16); mx.eval(q,k,v)
# active QK+PV FLOPs for causal sliding window W: sum_i min(i+1,W) ~ N*W - W^2/2
active_kv = N*W - W*W/2
flops_win = 4.0*B*H*active_kv*D
t,_=med3(lambda: flash_attention(q,k,v,scale=scale,causal=True,window_size=(W,0)))
print(f"  windowed V3 N={N} W={W} D={D}: {t*1e3:.3f}ms  eff={flops_win/t/1e12:.1f} TFLOPS (active-FLOP approx)")

# block-sparse, density ~0.25 (prebuilt mask), N=4096
mask=make_causal_block_mask(N, D); mx.eval(mask)  # causal block mask (triangular ~ density 0.5)
dens=float(mx.mean(mask.astype(mx.float32)).item())
flops_sp=4.0*B*H*N*N*D*dens
t,_=med3(lambda: flash_attention_sparse(q,k,v,mask,scale=scale,causal=True))
print(f"  block-sparse N={N} dens={dens:.2f} D={D}: {t*1e3:.3f}ms  eff={flops_sp/t/1e12:.1f} TFLOPS (active-FLOP)")

# GNA native, 3D window
ss=(8,8,8); Ng=8*8*8
gq=(mx.random.uniform(-1,1,(1,4,Ng,D))*0.1).astype(mx.float16)
gk=(mx.random.uniform(-1,1,(1,4,Ng,D))*0.1).astype(mx.float16)
gv=(mx.random.uniform(-1,1,(1,4,Ng,D))*0.1).astype(mx.float16); mx.eval(gq,gk,gv)
# GNA window (3,3,3) over (8,8,8): active per query ~ 27 neighbors; FLOPs ~ 4*1*4*Ng*27*D
flops_gna=4.0*1*4*Ng*27*D
t,_=med3(lambda: flash_attention_gna(gq,gk,gv,ss,(3,3,3),(3,3,3),scale=scale))
print(f"  GNA N={Ng} win(3,3,3) D={D}: {t*1e3:.3f}ms  eff={flops_gna/t/1e12:.1f} TFLOPS (active-FLOP approx)")

# dense backend=mfa (the shared STEEL MMA, re-confirm P0.1)
t,_=med3(lambda: flash_attention(q,k,v,scale=scale,causal=False,backend="mfa"))
print(f"  [shared STEEL MMA] dense backend=mfa N={N} D={D}: {t*1e3:.3f}ms  eff={4.0*B*H*N*N*D/t/1e12:.1f} TFLOPS")

print("\n=== Q2 — kernel-switch cost (in-graph lazy: SDPA + mlx-mfa flash both Primitives) ===")
# 12 'layers'; compare all-SDPA, all-sparse, alternating. In-graph -> one eval at end.
def run_layers(pattern):  # pattern: list of 'd'(SDPA dense) / 's'(sparse)
    x=q
    for p in pattern:
        if p=='d': x=flash_attention(x,k,v,scale=scale,causal=True)
        else: x=flash_attention_sparse(x,k,v,mask,scale=scale,causal=True)
    return x
L=12
for name,pat in [("all-dense(SDPA)",'d'*L),("all-sparse(mfa)",'s'*L),("ALTERNATING d/s",'ds'*(L//2))]:
    t,cv=med3(lambda: run_layers(pat))
    print(f"  {name:<20} {L} layers: {t*1e3:.3f}ms  ({t*1e3/L:.3f}ms/layer)  rangeCV={cv:.2f}")
print("  (alternating ~ mean(all-dense,all-sparse) => switch cost sub-material; both in-graph, no forced eval)")
