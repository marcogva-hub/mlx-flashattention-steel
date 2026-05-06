"""Inner runner for v32_kernel_sweep — benches one (shape × backend) in subprocess isolation."""
import argparse
import json
import math
import os
import statistics
import sys
import time

import mlx.core as mx
import mlx_mfa


# Shape registry (mirrored in outer sweep script)
NICHE_SHAPES = {
    "whisper-base":     dict(B=1, Hq=12, Hk=12, qL=1500,   kL=1500,    D=80,  causal=False),
    "gpt-neo-d96":      dict(B=1, Hq=16, Hk=16, qL=2048,   kL=2048,    D=96,  causal=True),
    "codestral-d192":   dict(B=1, Hq=32, Hk=8,  qL=2048,   kL=2048,    D=192, causal=True),
    "custom-d256":      dict(B=1, Hq=8,  Hk=8,  qL=2048,   kL=2048,    D=256, causal=False),
    "llama-decode-8k":  dict(B=1, Hq=32, Hk=8,  qL=1,      kL=8192,    D=128, causal=False),
    "llama-decode-32k": dict(B=1, Hq=32, Hk=8,  qL=1,      kL=32768,   D=128, causal=False),
    "flashvsr-dense":   dict(B=1, Hq=10, Hk=10, qL=4096,   kL=4096,    D=64,  causal=False),
    "llama-prefill-2k": dict(B=1, Hq=32, Hk=8,  qL=2048,   kL=2048,    D=128, causal=True),
    "llama-prefill-4k": dict(B=1, Hq=32, Hk=8,  qL=4096,   kL=4096,    D=128, causal=True),
    "llama-prefill-8k": dict(B=1, Hq=32, Hk=8,  qL=8192,   kL=8192,    D=128, causal=True),
    "ltx2-cross":       dict(B=1, Hq=8,  Hk=8,  qL=2048,   kL=14000,   D=64,  causal=False),
    "seedvr2-small":    dict(B=1, Hq=20, Hk=20, qL=26730,  kL=26730,   D=128, causal=False),
    "cogvideox":        dict(B=1, Hq=30, Hk=30, qL=70200,  kL=70200,   D=128, causal=False),
    "canonical-d128-4k": dict(B=1, Hq=20, Hk=20, qL=4096,  kL=4096,    D=128, causal=False),
    "canonical-d64-8k":  dict(B=1, Hq=20, Hk=20, qL=8192,  kL=8192,    D=64,  causal=False),
}


WARMUP = 3


def make(s, dtype=mx.float16):
    mx.random.seed(42)
    q = mx.random.normal((s["B"], s["Hq"], s["qL"], s["D"]), dtype=dtype)
    k = mx.random.normal((s["B"], s["Hk"], s["kL"], s["D"]), dtype=dtype)
    v = mx.random.normal((s["B"], s["Hk"], s["kL"], s["D"]), dtype=dtype)
    mx.synchronize()
    return q, k, v


def correctness(s, dtype):
    q, k, v = make(s, dtype)
    scale = 1.0 / math.sqrt(s["D"])
    out = mlx_mfa.flash_attention(q, k, v, scale=scale, causal=s["causal"])
    ref = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale,
        mask=("causal" if s["causal"] else None),
    )
    mx.synchronize()
    diff = (out.astype(mx.float32) - ref.astype(mx.float32))
    rmse = float(mx.sqrt(mx.mean(diff * diff)))
    finite = bool(mx.all(mx.isfinite(out)).item())
    return rmse, finite


def time_backend(s, backend, runs):
    q, k, v = make(s, mx.float16)
    scale = 1.0 / math.sqrt(s["D"])

    if backend == "sdpa":
        def call():
            mask = "causal" if s["causal"] else None
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    elif backend == "mfa":
        def call():
            return mlx_mfa.flash_attention(q, k, v, scale=scale, causal=s["causal"], backend="mfa")
    elif backend == "auto":
        def call():
            return mlx_mfa.flash_attention(q, k, v, scale=scale, causal=s["causal"], backend="auto")
    else:
        raise ValueError(f"Unknown backend: {backend}")

    for _ in range(WARMUP):
        out = call(); mx.synchronize()
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = call(); mx.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=list(NICHE_SHAPES.keys()))
    ap.add_argument("--backend", required=True, choices=["sdpa", "mfa", "auto"])
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    s = NICHE_SHAPES[args.shape]

    rmse = None
    finite = True
    if args.backend in ("mfa", "auto"):
        try:
            rmse, finite = correctness(s, mx.float16)
        except Exception as e:
            print(json.dumps({"shape": args.shape, "backend": args.backend,
                              "skipped": str(e)[:200], "supported": False}), flush=True)
            return

    try:
        timings = time_backend(s, args.backend, args.runs)
    except Exception as e:
        print(json.dumps({"shape": args.shape, "backend": args.backend,
                          "error": str(e)[:200]}), flush=True)
        return

    median = statistics.median(timings)
    record = {
        "shape": args.shape,
        "shape_dims": s,
        "backend": args.backend,
        "median_ms": median,
        "runs_ms": timings,
        "rmse": rmse,
        "finite": finite,
        "supported": True,
    }
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
