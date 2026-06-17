"""IV-OPT R.1 — non-decode regime profile on M5/26.6.

Per regime: t_call (flash_attention/op Python call = dispatch decision + lazy graph build,
no GPU) vs t_eval (mx.eval = kernel). Python-dispatch % of wall-clock = the reducible-overhead
lever. <~few% => kernel-dominated => at the floor. Also times mask construction (recompute class).
"""
import time
import numpy as np
import mlx.core as mx
import mlx_mfa
from mlx_mfa import (flash_attention, flash_attention_sparse, flash_attention_gna,
                     make_lcsa_mask, make_causal_block_mask)

ITERS, WARMUP = 60, 15
def md(xs): return sorted(xs)[len(xs)//2]


def seg(fn, *a):
    for _ in range(WARMUP):
        o = fn(*a); mx.eval(o)
    c, e = [], []
    for _ in range(ITERS):
        t0 = time.perf_counter(); o = fn(*a); t1 = time.perf_counter()
        mx.eval(o); t2 = time.perf_counter()
        c.append((t1-t0)*1e6); e.append((t2-t1)*1e6)
    return md(c), md(e)


def line(label, c, e):
    tot = c + e
    print(f"  {label:<34} call={c:8.1f}us  eval={e:9.1f}us  tot={tot:9.1f}us  py={100*c/tot:5.1f}%")


print(f"mlx_mfa {mlx_mfa.__version__} | mlx {mx.__version__}\n")

# ---- 1. Prefill / large-N dense forward (causal) ----
print("=== prefill / large-N dense forward (B=2 H=8 D=128 causal) ===")
for N in (2048, 4096, 8192):
    B, H, D = 2, 8, 128; scale = 1/np.sqrt(D)
    q = (mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16)
    k = (mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16)
    v = (mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16)
    mx.eval(q,k,v)
    line(f"N={N} auto", *seg(lambda q,k,v: flash_attention(q,k,v,scale=scale,causal=True), q,k,v))

# ---- 2. Windowed-causal (the V3 production path on M5) ----
print("\n=== windowed-causal forward (B=2 H=8 D=128, window=1024) ===")
for N in (4096, 8192):
    B, H, D = 2, 8, 128; scale = 1/np.sqrt(D)
    q = (mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16)
    k = (mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16)
    v = (mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16)
    mx.eval(q,k,v)
    line(f"N={N} window auto", *seg(lambda q,k,v: flash_attention(q,k,v,scale=scale,causal=True,window_size=(1024,0)), q,k,v))

# ---- 3. Sparse: mask construction cost + attend ----
print("\n=== sparse (LCSA) — mask build vs attend (B=1 H=8 D=128) ===")
for N in (4096, 8192):
    B, H, D = 1, 8, 128; scale = 1/np.sqrt(D)
    q = (mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16)
    k = (mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16)
    v = (mx.random.uniform(-1,1,(B,H,N,D))*0.1).astype(mx.float16)
    mx.eval(q,k,v)
    # mask build cost (Python-side; recompute class if rebuilt per call)
    H_, W_ = 64, N // 64   # spatial grid s.t. H*W = N
    tb = []
    for _ in range(WARMUP): m = make_lcsa_mask(q, k, H_, W_, 8, 256); mx.eval(m)
    for _ in range(ITERS):
        t0 = time.perf_counter(); m = make_lcsa_mask(q, k, H_, W_, 8, 256); mx.eval(m)
        tb.append((time.perf_counter()-t0)*1e6)
    mask = make_lcsa_mask(q, k, H_, W_, 8, 256); mx.eval(mask)
    c, e = seg(lambda q,k,v: flash_attention_sparse(q,k,v,mask,scale=scale), q,k,v)
    print(f"  N={N}: mask_build={md(tb):8.1f}us | attend call={c:.1f}us eval={e:.1f}us py={100*c/(c+e):.1f}%")

# ---- 4. GNA ----
print("\n=== GNA native (B=1 Hq=4 D=128, 3D window) ===")
seq_shape = (4, 8, 8); N = 4*8*8; D = 128; scale = 1/np.sqrt(D)
q = (mx.random.uniform(-1,1,(1,4,N,D))*0.1).astype(mx.float16)
k = (mx.random.uniform(-1,1,(1,4,N,D))*0.1).astype(mx.float16)
v = (mx.random.uniform(-1,1,(1,4,N,D))*0.1).astype(mx.float16)
mx.eval(q,k,v)
line(f"GNA N={N}", *seg(lambda q,k,v: flash_attention_gna(q,k,v,seq_shape,(2,3,3),(2,3,3),scale=scale), q,k,v))

# ---- 5. conv3d NAX ----
print("\n=== conv3d NAX forward ===")
try:
    from mlx_mfa import conv3d_nax_forward
    x = (mx.random.uniform(-1,1,(1,16,32,32,64))*0.1).astype(mx.float16)
    w = (mx.random.uniform(-1,1,(64,3,3,3,64))*0.1).astype(mx.float16)
    mx.eval(x,w)
    line("conv3d_nax T16 32x32 C64", *seg(lambda x,w,_v: conv3d_nax_forward(x,w,padding=(1,1,1)), x, w, None))
except Exception as ex:
    print(f"  conv3d note: {ex!r}")
