#!/usr/bin/env python3
"""D=256 kernel bench — autoresearch metric script.

Measures forced MFA V2 D-split vs forced SDPA for D=256 causal.
Both paths via flash_attention() wrapper — identical overhead, pure kernel signal.

The near-miss regime on M1 Max (from decision pass 2026-03-12):
  N=2048 causal: ~0.94x  (3 iterations baseline)
  N=4096 causal: ~0.98x  (3 iterations baseline)
  N=8192 causal: ~1.01x  (3 iterations baseline)

The agent modifies csrc/mfa_steel_fwd_v2.cpp (select_steel_v2_dsplit_block_config)
then rebuilds. Goal: push N=4096 / N=8192 geomean above 1.02x consistently.

Output last line:  SPEEDUP_RATIO: X.XXXXXX   (geomean V2/SDPA, maximize it)

NOTE: Run with project venv:
    ~/code/mlx-mfa-v2/.venv/bin/python benchmarks/bench_dispatch_d256_kernel.py
"""
from __future__ import annotations
import math, os, sys, time
import mlx.core as mx

D      = 256
DTYPE  = mx.float16
SEED   = 42
B, H   = 2, 8          # production-like profile matching decision pass

# Focus on the near-miss regime — the region where kernel changes have leverage.
# N=1024 is a consistent loss (0.73x); N=16384 is already a solid win (1.14x).
# The actionable question: can we push N=2048–8192 above 1.02x?
PROFILES = [
    (B, H, 2048, True),
    (B, H, 4096, True),
    (B, H, 8192, True),
]

WARMUP = 5
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
        print("WARNING: MFA extension not compiled — rebuild first.")
        print("SPEEDUP_RATIO: 0.000000"); sys.exit(1)

    dev = get_device_info()
    is_m3_plus = bool(dev.get("is_m3_plus", False))
    chip = dev.get("chip_name", dev.get("name", "?"))

    # Show current dsplit BK in effect (for log readability)
    bk_override = os.environ.get("MFA_V2_FORCE_BK_D256", "not set → code default")
    print(f"Device: {chip} gen={dev.get('gen','?')} M3+={is_m3_plus} | "
          f"D={D} f16 B={B} H={H} | MFA_V2_FORCE_BK_D256={bk_override}")
    print("-" * 68)
    print(f"{'Config':28s}  {'V2 ms':>8}  {'SDPA ms':>8}  {'ratio':>7}  policy")
    print("-" * 68)

    scale = 1.0 / math.sqrt(D)
    ratios = []

    for (Bi, Hi, N, causal) in PROFILES:
        policy_mfa = should_use_mfa(D, N, causal, is_m3_plus, dtype=DTYPE, backend="auto")

        mx.random.seed(SEED)
        q = mx.random.normal([Bi, Hi, N, D]).astype(DTYPE)
        k = mx.random.normal([Bi, Hi, N, D]).astype(DTYPE)
        v = mx.random.normal([Bi, Hi, N, D]).astype(DTYPE)
        mx.eval(q, k, v)

        # Force V2 D-split (MFA_DISABLE_V2 unset = V2 active)
        v2_ms   = _median_ms(lambda: flash_attention(q, k, v, scale=scale, causal=causal, backend="mfa"))
        sdpa_ms = _median_ms(lambda: flash_attention(q, k, v, scale=scale, causal=causal, backend="sdpa"))

        ratio = sdpa_ms / v2_ms if v2_ms > 0 else 0.0
        ratios.append(ratio)

        lbl = f"D=256 N={N} {'causal' if causal else 'dense':6s}"
        print(f"{lbl:28s}  {v2_ms:8.2f}  {sdpa_ms:8.2f}  {ratio:7.3f}x  "
              f"{'→MFA' if policy_mfa else '→SDPA'}")

    print("-" * 68)

    geomean = math.exp(sum(math.log(max(r, 1e-9)) for r in ratios) / len(ratios))
    wins    = sum(1 for r in ratios if r >= 1.02)
    losses  = sum(1 for r in ratios if r < 0.98)

    print(f"Wins  (>=1.02x): {wins}/{len(ratios)}  |  Losses (<0.98x): {losses}/{len(ratios)}")
    print(f"Geometric mean V2/SDPA: {geomean:.4f}x")
    print(f"SPEEDUP_RATIO: {geomean:.6f}")

if __name__ == "__main__":
    main()
