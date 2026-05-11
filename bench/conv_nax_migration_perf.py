#!/usr/bin/env python3
"""Sprint D Track D.2 — C++ Primitive perf parity sanity vs Phase 1.5 numbers.

Compares the C++-routed conv3d_nax_forward (default after Sprint D)
against the Phase 1.5 documented medians from ship-shelve-decision.md §2.
Single-session sanity check (NOT a re-sweep). Bar: ±5% drift.

If the C++ Primitive shows ≥ 5% improvement on mid_resnet (the expected
gain from removing ~50-100 µs Python dispatch overhead), note in results.
"""
import os, time, statistics, json
import mlx.core as mx
from mlx_mfa.conv_nax import conv3d_nax_forward

# Bookend shapes per prompt §6.2: smallest + largest production shapes.
SHAPES = [
    ("mid_resnet",            1,  5, 64,  64,  512, 512, 3, 3, 3,
     {"phase1_5_nax_ms": 8.7, "phase1_5_mlx_ms": 19.7, "phase1_5_ratio": 2.26}),
    ("up2_resnet0_peakflops", 1, 17, 256, 256, 512, 256, 3, 3, 3,
     {"phase1_5_nax_ms": 332.4, "phase1_5_mlx_ms": 524.5, "phase1_5_ratio": 1.54}),
]


def bench_shape(label, B, T, H, W, C_in, C_out, K_T, K_H, K_W, phase15, n_runs=5):
    pad = (K_T // 2, K_H // 2, K_W // 2)
    mx.random.seed(0)
    x = (mx.random.uniform(shape=(B, T, H, W, C_in)) * 0.1).astype(mx.float16)
    w = (mx.random.uniform(shape=(C_out, K_T, K_H, K_W, C_in)) * 0.1).astype(mx.float16)
    mx.async_eval(x, w); mx.synchronize()

    def call_nax():
        return conv3d_nax_forward(x, w, stride=(1,1,1), padding=pad, dilation=(1,1,1))
    def call_mlx():
        return mx.conv_general(x, w, stride=[1,1,1], padding=list(pad),
                                kernel_dilation=[1,1,1])

    # Warmup
    for _ in range(3):
        y = call_nax(); mx.async_eval(y); mx.synchronize()
    nax_times = []
    for _ in range(n_runs):
        mx.synchronize()
        t0 = time.perf_counter()
        y = call_nax(); mx.async_eval(y); mx.synchronize()
        nax_times.append((time.perf_counter()-t0)*1000)

    for _ in range(3):
        y = call_mlx(); mx.async_eval(y); mx.synchronize()
    mlx_times = []
    for _ in range(n_runs):
        mx.synchronize()
        t0 = time.perf_counter()
        y = call_mlx(); mx.async_eval(y); mx.synchronize()
        mlx_times.append((time.perf_counter()-t0)*1000)

    nax_med = statistics.median(nax_times)
    mlx_med = statistics.median(mlx_times)
    ratio = mlx_med / nax_med if nax_med > 0 else 0
    p15_nax = phase15["phase1_5_nax_ms"]
    p15_ratio = phase15["phase1_5_ratio"]
    drift_nax = (nax_med - p15_nax) / p15_nax * 100
    drift_ratio = (ratio - p15_ratio) / p15_ratio * 100
    return {
        "shape": label,
        "nax_median_ms": nax_med,
        "mlx_median_ms": mlx_med,
        "ratio": ratio,
        "phase1_5_nax_ms": p15_nax,
        "phase1_5_ratio": p15_ratio,
        "nax_ms_drift_pct": drift_nax,
        "ratio_drift_pct": drift_ratio,
    }


def main():
    print("Sprint D Track D.2 — C++ Primitive perf parity sanity")
    print("="*80)
    results = []
    for spec in SHAPES:
        r = bench_shape(*spec)
        results.append(r)
        print(f"{r['shape']:<28}  D nax={r['nax_median_ms']:>7.2f}ms  "
              f"P1.5 nax={r['phase1_5_nax_ms']:>7.2f}ms  "
              f"drift={r['nax_ms_drift_pct']:>+6.2f}%  "
              f"ratio={r['ratio']:>5.2f}× (P1.5 {r['phase1_5_ratio']:.2f}× drift "
              f"{r['ratio_drift_pct']:>+6.2f}%)")
    # Check ±5% drift bar
    all_within_5pct = all(abs(r["nax_ms_drift_pct"]) < 5.0 for r in results)
    print(f"\nAll shapes within ±5% drift: {all_within_5pct}")
    out_path = "docs/conv-nax/conv-nax-prod-perf-sanity.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "all_within_5pct": all_within_5pct}, f, indent=2)
    print(f"data -> {out_path}")


if __name__ == "__main__":
    main()
