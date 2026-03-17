#!/usr/bin/env python3
import json
import os
import statistics
import time

os.environ["MFA_DISABLE_ASYNC"] = "1"
os.environ.setdefault("MFA_IR_INVESTIGATE", "1")

import mlx.core as mx
import mlx_mfa


def run_once(q, k, v):
    out = mlx_mfa.flash_attention(q, k, v, causal=True)
    mx.eval(out)


def main():
    b, h, n, d = 1, 8, 8192, 128
    warmup = 3
    iters = 10

    q = mx.random.normal([b, h, n, d]).astype(mx.float16)
    k = mx.random.normal([b, h, n, d]).astype(mx.float16)
    v = mx.random.normal([b, h, n, d]).astype(mx.float16)
    mx.eval(q, k, v)

    for _ in range(warmup):
        run_once(q, k, v)

    times = []
    for _ in range(iters):
        mx.eval(q)
        t0 = time.perf_counter()
        run_once(q, k, v)
        times.append(time.perf_counter() - t0)

    result = {
        "path": "sync",
        "shape": {"B": b, "H": h, "N": n, "D": d, "causal": True},
        "warmup": warmup,
        "iters": iters,
        "median_ms": statistics.median(times) * 1000.0,
        "mean_ms": statistics.mean(times) * 1000.0,
        "min_ms": min(times) * 1000.0,
        "max_ms": max(times) * 1000.0,
        "samples_ms": [t * 1000.0 for t in times],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
