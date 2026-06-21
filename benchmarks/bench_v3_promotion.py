#!/usr/bin/env python3
"""V3 promotion verification: compare V3 (default) vs V2 (MFA_DISABLE_V3=1) vs SDPA."""
import math, os, sys
import mlx.core as mx
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).parent))  # repo review 2026-05: allow `python benchmarks/<f>.py` from repo root
from bench_utils import med
# audit H7/H-09 phantom-bench gate (run-at-import bench)
from _bench_guard import require_accel_or_die as _phantom_gate
_phantom_gate(__file__)

SEED = 42
DTYPE = mx.float16

PROFILES = [
    # Winning regime (should route V3 by default)
    (2, 8,  4096,  64, True,  "V3 expected"),
    (2, 8,  8192,  64, True,  "V3 expected"),
    (2, 8,  2048, 128, True,  "V3 expected"),
    (2, 8,  4096, 128, True,  "V3 expected"),
    # Losing regime (should route V2 by default)
    (2, 8,  2048,  64, True,  "V2 expected"),
    (2, 8,  1024, 128, True,  "V2 expected"),
    (1, 1,  4096,  64, True,  "V2 expected (B*H<16)"),
    (2, 8,  4096,  64, False, "V2 expected (non-causal)"),
]

def main():
    from mlx_mfa import flash_attention, is_mfa_available, get_device_info
    if not is_mfa_available():
        print("MFA extension not available"); return
    dev = get_device_info()
    print(f"Device: {dev.get('chip_name','?')} M3+={dev.get('is_m3_plus',False)}")
    print(f"{'Config':38s}  {'Auto ms':>8}  {'SDPA ms':>8}  {'V2 ms':>8}  "
          f"{'auto/SDPA':>10}  {'auto/V2':>8}  note")
    print("-"*100)

    for B,H,N,D,causal,note in PROFILES:
        scale = 1/math.sqrt(D)
        mx.random.seed(SEED)
        q=mx.random.normal([B,H,N,D]).astype(DTYPE)
        k=mx.random.normal([B,H,N,D]).astype(DTYPE)
        v=mx.random.normal([B,H,N,D]).astype(DTYPE)
        mx.eval(q,k,v)

        auto_ms = med(lambda: flash_attention(q,k,v,scale=scale,causal=causal,backend="auto"))
        sdpa_ms = med(lambda: flash_attention(q,k,v,scale=scale,causal=causal,backend="sdpa"))
        # Force V2 via env var
        prev = os.environ.get("MFA_DISABLE_V3")
        os.environ["MFA_DISABLE_V3"] = "1"
        v2_ms = med(lambda: flash_attention(q,k,v,scale=scale,causal=causal,backend="mfa"))
        if prev is None: del os.environ["MFA_DISABLE_V3"]
        else: os.environ["MFA_DISABLE_V3"] = prev

        r_sdpa = sdpa_ms/auto_ms if auto_ms>0 else 0
        r_v2   = v2_ms/auto_ms if auto_ms>0 else 0
        flag = "✅" if r_sdpa >= 1.02 else ("⚠" if r_sdpa >= 0.98 else "❌")
        lbl = f"B={B} H={H} N={N:5d} D={D} {'causal' if causal else 'dense':7s}"
        print(f"{lbl:38s}  {auto_ms:8.2f}  {sdpa_ms:8.2f}  {v2_ms:8.2f}  "
              f"{r_sdpa:9.3f}x  {r_v2:7.3f}x  {flag} {note}")

    print("-"*100)
    print("✅=V3 winning (≥1.02x vs SDPA)  ⚠=neutral  ❌=regression")
    print("auto/V2 > 1.0 = V3 routed and winning vs V2 for that config")

if __name__=="__main__": main()
