#!/usr/bin/env python3
"""D=512 VAE-path MFA vs SDPA bench — autoresearch metric script (v2).

Measures forced MFA vs forced SDPA (same wrapper overhead, pure kernel signal).

The ratio sdpa_ms / mfa_ms per config is what matters.
Agent modifies dispatch_policy.py to route winning configs to MFA.

Profiles (B=1, H=8, D=512, f16):
  N=64/128/256/512 non-causal, N=256/512 causal

Output last line: SPEEDUP_RATIO: X.XXXXXX  (geomean sdpa/mfa, maximize it)
"""
from __future__ import annotations
import math, os, sys, time
import mlx.core as mx

D = 512
DTYPE = mx.float16
SEED = 42
PROFILES = [
    (1, 8,  64, False),
    (1, 8, 128, False),
    (1, 8, 256, False),
    (1, 8, 512, False),
    (1, 8, 256, True),
    (1, 8, 512, True),
]
WARMUP = 3
ITERS  = 15

def _median_ms(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup): mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]

def main():
    try:
        from mlx_mfa import flash_attention, is_mfa_available, get_device_info
        from mlx_mfa.dispatch_policy import should_use_mfa
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); sys.exit(1)

    if not is_mfa_available():
        print("WARNING: MFA extension not compiled.")
        print("SPEEDUP_RATIO: 1.000000"); sys.exit(0)

    dev = get_device_info()
    is_m3_plus = bool(dev.get("is_m3_plus", False))
    gpu_cores  = int(dev.get("gpu_cores", 0))
    chip = dev.get("chip_name", dev.get("name", "?"))

    print(f"Device: {chip} gen={dev.get('gen','?')} M3+={is_m3_plus} cores={gpu_cores} | D={D} f16 profiles={len(PROFILES)}")
    print("-" * 74)
    print(f"{'Config':30s}  {'MFA ms':>8}  {'SDPA ms':>8}  {'ratio':>7}  {'policy':>7}  occ")
    print("-" * 74)

    scale = 1.0 / math.sqrt(D)
    ratios = []

    for (B, H, N, causal) in PROFILES:
        policy_mfa = should_use_mfa(D, N, causal, is_m3_plus, dtype=DTYPE, backend="auto")
        mx.random.seed(SEED)
        q = mx.random.normal([B, H, N, D]).astype(DTYPE)
        k = mx.random.normal([B, H, N, D]).astype(DTYPE)
        v = mx.random.normal([B, H, N, D]).astype(DTYPE)
        mx.eval(q, k, v)

        mfa_ms  = _median_ms(lambda: flash_attention(q, k, v, scale=scale, causal=causal, backend="mfa"))
        sdpa_ms = _median_ms(lambda: flash_attention(q, k, v, scale=scale, causal=causal, backend="sdpa"))
        ratio = sdpa_ms / mfa_ms if mfa_ms > 0 else 1.0
        ratios.append(ratio)

        total_tgs = ((N + 31) // 32) * H * B
        occ = total_tgs / max(gpu_cores, 1)
        lbl = f"N={N:4d} {'causal' if causal else 'dense ':6s}"
        print(f"B={B} H={H} {lbl:18s}  {mfa_ms:8.3f}  {sdpa_ms:8.3f}  {ratio:7.3f}x  {'→MFA' if policy_mfa else '→SDPA':>7}  {occ:.2f}x")

    print("-" * 74)
    geomean = math.exp(sum(math.log(max(r, 1e-9)) for r in ratios) / len(ratios))
    wins   = sum(1 for r in ratios if r >= 1.02)
    losses = sum(1 for r in ratios if r < 0.98)

    # Highlight near-win candidates for the agent
    print("Near-win candidates (ratio >= 0.95x, currently →SDPA):")
    found = False
    for i, ((B, H, N, causal), r) in enumerate(zip(PROFILES, ratios)):
        policy_mfa = should_use_mfa(D, N, causal, is_m3_plus, dtype=DTYPE, backend="auto")
        if r >= 0.95 and not policy_mfa:
            print(f"  N={N:4d} causal={str(causal):<5} ratio={r:.3f}x  ← lower _d512_min_n threshold to activate")
            found = True
    if not found:
        print("  None. All configs either routed to MFA already or ratio < 0.95x.")

    print(f"Wins  (>=1.02x): {wins}/{len(ratios)}  |  Losses (<0.98x): {losses}/{len(ratios)}")
    print(f"Geometric mean (sdpa/mfa kernel): {geomean:.4f}x")
    print(f"SPEEDUP_RATIO: {geomean:.6f}")

if __name__ == "__main__":
    main()
