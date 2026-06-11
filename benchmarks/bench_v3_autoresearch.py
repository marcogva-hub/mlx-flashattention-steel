#!/usr/bin/env python3
"""V3 autoresearch bench — geomean V3/SDPA on D=64/128 causal profiles.
Last line output: SPEEDUP_RATIO: X.XXXXXX  (maximize)
"""
import math, os, sys
from functools import partial
import mlx.core as mx
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).parent))  # repo review 2026-05: allow `python benchmarks/<f>.py` from repo root
from bench_utils import med as _med

SEED = 42
DTYPE = mx.float16
med = partial(_med, warmup=3, iters=12)

# Profiles: D=64 and D=128, causal only (where V3 has potential)
# B=2 H=8 = production profile; B=1 H=1 = under-occupied profile
PROFILES = [
    (2, 8,  2048, 64,  True),
    (2, 8,  4096, 64,  True),
    (2, 8,  8192, 64,  True),
    (1, 1,  2048, 64,  True),
    (2, 8,  2048, 128, True),
    (2, 8,  4096, 128, True),
    (1, 1,  2048, 128, True),
]

def main():
    try:
        from mlx_mfa import flash_attention, is_mfa_available, get_device_info
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)
    if not is_mfa_available():
        print("SPEEDUP_RATIO: 0.000000"); sys.exit(1)

    dev = get_device_info()
    chip = dev.get('chip_name', dev.get('name', '?'))
    print(f"Device: {chip} M3+={dev.get('is_m3_plus',False)} | MFA_ENABLE_V3={os.environ.get('MFA_ENABLE_V3','not set')}")
    print("-"*64)
    print(f"{'Config':32s}  {'V3 ms':>7}  {'SDPA ms':>8}  {'ratio':>7}")
    print("-"*64)

    ratios = []
    for B,H,N,D,causal in PROFILES:
        scale = 1/math.sqrt(D)
        mx.random.seed(SEED)
        q=mx.random.normal([B,H,N,D]).astype(DTYPE)
        k=mx.random.normal([B,H,N,D]).astype(DTYPE)
        v=mx.random.normal([B,H,N,D]).astype(DTYPE)
        mx.eval(q,k,v)
        v3_ms   = med(lambda: flash_attention(q,k,v,scale=scale,causal=causal,backend="mfa"))
        sdpa_ms = med(lambda: flash_attention(q,k,v,scale=scale,causal=causal,backend="sdpa"))
        r = sdpa_ms/v3_ms if v3_ms>0 else 0.0
        ratios.append(r)
        lbl = f"B={B} H={H} N={N:5d} D={D} causal"
        print(f"{lbl:32s}  {v3_ms:7.2f}  {sdpa_ms:8.2f}  {r:7.3f}x")

    print("-"*64)
    geo = math.exp(sum(math.log(max(r,1e-9)) for r in ratios)/len(ratios))
    wins = sum(1 for r in ratios if r>=1.02)
    print(f"Wins(>=1.02x): {wins}/{len(ratios)}  Geomean: {geo:.4f}x")
    print(f"SPEEDUP_RATIO: {geo:.6f}")

if __name__=="__main__": main()
