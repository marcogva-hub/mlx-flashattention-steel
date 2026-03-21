#!/usr/bin/env python3
"""D=512 autoresearch bench — outputs SPEEDUP_RATIO on last line."""
import math, os, sys, time
import mlx.core as mx

D, SEED = 512, 42
DTYPE = mx.float16
PROFILES = [
    (2, 8, 1024, True),
    (2, 8, 2048, True),
    (2, 8, 4096, True),
    (2, 8, 8192, True),
]
WARMUP, ITERS = 3, 12

def med(fn):
    for _ in range(WARMUP): mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    ts.sort(); return ts[len(ts)//2]

def main():
    from mlx_mfa import flash_attention, is_mfa_available, get_device_info
    if not is_mfa_available():
        print("SPEEDUP_RATIO: 0.000000"); sys.exit(1)
    dev = get_device_info()
    print(f"Device: {dev.get('chip_name','?')} M3+={dev.get('is_m3_plus',False)}")
    print(f"D={D} f16  BK_D512={os.environ.get('MFA_V2_FORCE_BK_D512','code default')}  BD_HALF_D512={os.environ.get('MFA_V2_BD_HALF_D512','code default')}")
    print("-"*60)
    scale = 1/math.sqrt(D)
    ratios = []
    for B,H,N,causal in PROFILES:
        mx.random.seed(SEED)
        q=mx.random.normal([B,H,N,D]).astype(DTYPE)
        k=mx.random.normal([B,H,N,D]).astype(DTYPE)
        v=mx.random.normal([B,H,N,D]).astype(DTYPE)
        mx.eval(q,k,v)
        mfa_ms  = med(lambda: flash_attention(q,k,v,scale=scale,causal=causal,backend="mfa"))
        sdpa_ms = med(lambda: flash_attention(q,k,v,scale=scale,causal=causal,backend="sdpa"))
        r = sdpa_ms/mfa_ms if mfa_ms>0 else 0.0
        ratios.append(r)
        print(f"N={N:5d} {'causal' if causal else 'dense ':6s}  mfa={mfa_ms:7.2f}ms  sdpa={sdpa_ms:7.2f}ms  ratio={r:6.3f}x")
    print("-"*60)
    import math as m
    geo = m.exp(sum(m.log(max(r,1e-9)) for r in ratios)/len(ratios))
    wins = sum(1 for r in ratios if r>=1.02)
    print(f"Wins(>=1.02): {wins}/{len(ratios)}  Geomean: {geo:.4f}x")
    print(f"SPEEDUP_RATIO: {geo:.6f}")

if __name__=="__main__": main()
