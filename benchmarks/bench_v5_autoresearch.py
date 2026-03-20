#!/usr/bin/env python3
"""V5 autoresearch bench — SPEEDUP_RATIO: geomean V5/SDPA, causal profils."""
import math, os, time, sys
import mlx.core as mx

SEED, WARMUP, ITERS = 42, 3, 12
DTYPE = mx.float16

# Profils causal uniquement (là où V5 a du potentiel d'après le triage)
# prod_b2h8 = profil production standard
PROFILES = [
    (2, 8, 2048,  64, True),
    (2, 8, 4096,  64, True),
    (2, 8, 8192,  64, True),
    (2, 8, 2048, 128, True),
    (2, 8, 4096, 128, True),
    (2, 8, 8192, 128, True),
]

def med(fn):
    for _ in range(WARMUP): mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    ts.sort(); return ts[len(ts)//2]

def main():
    from mlx_mfa import flash_attention, is_mfa_available, get_device_info
    if not is_mfa_available():
        print("SPEEDUP_RATIO: 0.000000"); sys.exit(1)
    dev = get_device_info()
    bk      = os.environ.get("MFA_V5_FORCE_BK",      "128")
    bd_tile = os.environ.get("MFA_V5_FORCE_BD_TILE",  "32")
    print(f"Device: {dev.get('chip_name','?')} M3+={dev.get('is_m3_plus',False)} | "
          f"BK={bk} BD_tile={bd_tile} MFA_ENABLE_V5={os.environ.get('MFA_ENABLE_V5','not set')}")
    print("-"*70)
    print(f"{'Config':32s}  {'V5 ms':>7}  {'SDPA ms':>8}  {'V3 ms':>7}  {'V5/SDPA':>8}  {'V5/V3':>7}")
    print("-"*70)

    ratios_sdpa, ratios_v3 = [], []
    for B,H,N,D,causal in PROFILES:
        scale = 1/math.sqrt(D)
        mx.random.seed(SEED)
        q=mx.random.normal([B,H,N,D]).astype(DTYPE)
        k=mx.random.normal([B,H,N,D]).astype(DTYPE)
        v=mx.random.normal([B,H,N,D]).astype(DTYPE)
        mx.eval(q,k,v)

        v5_ms   = med(lambda: flash_attention(q,k,v,scale=scale,causal=causal,backend="mfa"))
        sdpa_ms = med(lambda: flash_attention(q,k,v,scale=scale,causal=causal,backend="sdpa"))
        # V3 via disable V5 temporarily
        prev = os.environ.get("MFA_ENABLE_V5")
        os.environ.pop("MFA_ENABLE_V5", None)
        v3_ms = med(lambda: flash_attention(q,k,v,scale=scale,causal=causal,backend="mfa"))
        if prev: os.environ["MFA_ENABLE_V5"] = prev

        r_sdpa = sdpa_ms/v5_ms if v5_ms>0 else 0
        r_v3   = v3_ms/v5_ms   if v5_ms>0 else 0
        ratios_sdpa.append(r_sdpa)
        ratios_v3.append(r_v3)

        lbl = f"B={B} H={H} N={N:5d} D={D:3d} causal"
        print(f"{lbl:32s}  {v5_ms:7.2f}  {sdpa_ms:8.2f}  {v3_ms:7.2f}  "
              f"{r_sdpa:8.3f}x  {r_v3:7.3f}x")

    print("-"*70)
    import math as m
    geo_sdpa = m.exp(sum(m.log(max(r,1e-9)) for r in ratios_sdpa)/len(ratios_sdpa))
    geo_v3   = m.exp(sum(m.log(max(r,1e-9)) for r in ratios_v3)/len(ratios_v3))
    wins_sdpa = sum(1 for r in ratios_sdpa if r>=1.02)
    wins_v3   = sum(1 for r in ratios_v3   if r>=1.02)
    print(f"Wins vs SDPA (>=1.02x): {wins_sdpa}/{len(ratios_sdpa)}")
    print(f"Wins vs V3   (>=1.02x): {wins_v3}/{len(ratios_v3)}")
    print(f"Geomean V5/SDPA: {geo_sdpa:.4f}x | Geomean V5/V3: {geo_v3:.4f}x")
    print(f"SPEEDUP_RATIO: {geo_sdpa:.6f}")

if __name__=="__main__": main()
