#!/usr/bin/env python3
"""Auto-dispatch validation: flash_attention(backend='auto') must be >= SDPA.

Verifies the Phase 1 smart dispatch guarantee:
  - Below threshold: routes to SDPA → 1.0x effective
  - Above threshold: routes to MFA → wins (>= 1.0x)

Usage:
    .venv/bin/python benchmarks/bench_auto_dispatch_validation.py
"""
from __future__ import annotations
import math, sys, time
from datetime import date
import numpy as np
sys.path.insert(0, ".")
import mlx.core as mx
from mlx_mfa import flash_attention, get_device_info

_materialize = mx.eval

WARMUP = 5
TIMED  = 20
BATCH  = 1
HEADS  = 8

# Methodology fix (audit H-09): a SINGLE explicit accept threshold — the old code
# accepted >= 0.90 but labelled failures/summaries as "0.95", an ambiguous mismatch.
# 0.90 = the documented noise floor (±10% at sub-2ms).
ACCEPT_RATIO = 0.90

MATRIX = {
    "D":      [64, 128, 256, 512],
    "N":      [512, 1024, 2048, 4096, 8192],
    "causal": [True, False],
}


def _timed_ms(fn) -> float:
    for _ in range(WARMUP):
        _materialize(fn())
    mx.synchronize()
    ts = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        _materialize(fn())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def bench_one(B, H, N, D, causal):
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(42)
    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H, N, D)).astype(mx.float16)
    v = mx.random.normal((B, H, N, D)).astype(mx.float16)
    _materialize(q, k, v)
    auto_ms = _timed_ms(lambda: flash_attention(q, k, v, scale=scale,
                                                causal=causal, backend="auto"))
    sdpa_ms = _timed_ms(lambda: flash_attention(q, k, v, scale=scale,
                                                causal=causal, backend="sdpa"))
    ratio = sdpa_ms / auto_ms if auto_ms > 0 else float("nan")
    return dict(D=D, N=N, causal=causal,
                auto_ms=auto_ms, sdpa_ms=sdpa_ms, ratio=ratio)


def main():
    # Phantom-bench gate (audit H7/H-09): refuse to run if mlx-mfa acceleration
    # isn't live — else `auto` and `sdpa` would BOTH be SDPA and report fake 1.0x.
    from _bench_guard import require_accel_or_die
    require_accel_or_die("bench_auto_dispatch_validation.py")

    dev = get_device_info()
    print(f"mlx-mfa auto-dispatch validation  --  {date.today()}")
    print(f"Device: {dev.get('device_name','?')}  M3+={dev.get('is_m3_plus',False)}")
    print(f"Config: B={BATCH} H={HEADS}  warmup={WARMUP}  timed={TIMED}")
    print()

    failures = []
    all_results = []

    for causal in MATRIX["causal"]:
        cs = "causal" if causal else "non-causal"
        print(f"=== {cs} " + "=" * 55)
        print(f"{'D':>5} {'N':>6} {'auto ms':>9} {'sdpa ms':>9} {'ratio':>8}  status")
        print("-" * 60)
        for D in MATRIX["D"]:
            for N in MATRIX["N"]:
                r = bench_one(BATCH, HEADS, N, D, causal)
                all_results.append(r)
                ok = r["ratio"] >= ACCEPT_RATIO
                status = "OK" if ok else f"FAIL ({r['ratio']:.2f}x < {ACCEPT_RATIO:.2f}x)"
                if not ok:
                    failures.append(r)
                print(f"  D={D:<4} N={N:<5} "
                      f"{r['auto_ms']:8.2f}ms {r['sdpa_ms']:8.2f}ms "
                      f"  {r['ratio']:5.2f}x  {status}")
        print()

    wins = [r for r in all_results if r["ratio"] >= 1.0]
    print(f"Summary: auto >= 1.0x SDPA: {len(wins)}/{len(all_results)}")
    print(f"         auto >= {ACCEPT_RATIO:.2f}x SDPA: {len(all_results)-len(failures)}/{len(all_results)}")

    print()
    print("Note: sub-2ms benchmarks have ±10% Metal scheduling jitter (Python overhead: ~2μs/call).")
    if failures:
        print(f"\nFAILED configs (auto < {ACCEPT_RATIO:.2f}x SDPA):")
        for r in failures:
            c = "causal" if r["causal"] else "non-causal"
            print(f"  D={r['D']} N={r['N']} {c}: {r['ratio']:.2f}x")
    else:
        print(f"\nAll auto-dispatch cases >= {ACCEPT_RATIO:.2f}x SDPA ✓")


if __name__ == "__main__":
    main()
