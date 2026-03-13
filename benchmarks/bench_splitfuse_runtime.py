#!/usr/bin/env python3
"""Focused benchmark for runtime-integrated splitfuse."""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import mlx.core as mx

from mlx_mfa import (
    __version__ as mlx_mfa_version,
    create_decode_runtime,
    flash_attention_splitfuse,
    get_device_info,
)


def _eval_any(x):
    if isinstance(x, dict):
        for y in x.values():
            _eval_any(y)
        return
    if isinstance(x, (list, tuple)):
        for y in x:
            _eval_any(y)
        return
    mx.eval(x)


def _measure_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(max(0, warmup)):
        _eval_any(fn())
        mx.synchronize()
    samples = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        _eval_any(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return float(samples[len(samples) // 2])


def bench_splitfuse_dense(D: int, warmup: int, iters: int) -> dict:
    scale = 1.0 / math.sqrt(D)

    mx.random.seed(81000 + D)
    q_pre = mx.random.normal((1, 4, 16, D)).astype(mx.float16)
    mx.random.seed(81100 + D)
    k_pre = mx.random.normal((1, 4, 16, D)).astype(mx.float16)
    mx.random.seed(81200 + D)
    v_pre = mx.random.normal((1, 4, 16, D)).astype(mx.float16)

    mx.random.seed(81300 + D)
    q_dec = mx.random.normal((1, 4, 2, D)).astype(mx.float16)
    mx.random.seed(81400 + D)
    k_hist = mx.random.normal((1, 4, 48, D)).astype(mx.float16)
    mx.random.seed(81500 + D)
    v_hist = mx.random.normal((1, 4, 48, D)).astype(mx.float16)
    mx.eval(q_pre, k_pre, v_pre, q_dec, k_hist, v_hist)

    rt = create_decode_runtime(
        backend="dense",
        quantized_kv=False,
        B=1,
        H_q=4,
        H_kv=4,
        D=D,
        max_seq_len=256,
    )
    rt.register_prefix("p0", q_pre, k_pre, v_pre, overwrite=True)

    def helper_path():
        return flash_attention_splitfuse(
            q_pre,
            k_pre,
            v_pre,
            q_dec,
            k_hist,
            v_hist,
            scale=scale,
            causal=True,
        )

    def runtime_manual_splitfuse():
        return rt.splitfuse(
            q_pre,
            k_pre,
            v_pre,
            q_dec,
            k_hist,
            v_hist,
            scale=scale,
            causal=True,
        )

    def runtime_step_cached():
        rt.reset()
        _ = rt.prefill(q_pre, k_pre, v_pre, scale=scale, causal=True)
        return rt.splitfuse_step(q_dec, scale=scale, causal=True)

    def runtime_step_prefix():
        rt.seed_prefix(prefix_id="p0", reset=True)
        return rt.splitfuse_step(
            q_dec,
            use_registered_prefix=True,
            prefix_id="p0",
            scale=scale,
            causal=True,
        )

    ms_helper = _measure_ms(helper_path, warmup=warmup, iters=iters)
    ms_runtime_manual = _measure_ms(runtime_manual_splitfuse, warmup=warmup, iters=iters)
    ms_runtime_step = _measure_ms(runtime_step_cached, warmup=warmup, iters=iters)
    ms_runtime_prefix = _measure_ms(runtime_step_prefix, warmup=warmup, iters=iters)

    return {
        "scenario": "dense_splitfuse",
        "D": D,
        "helper_ms": ms_helper,
        "runtime_splitfuse_ms": ms_runtime_manual,
        "runtime_splitfuse_step_ms": ms_runtime_step,
        "runtime_splitfuse_step_prefix_ms": ms_runtime_prefix,
        "runtime_step_vs_helper": ms_runtime_step / ms_helper if ms_helper > 0 else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Splitfuse runtime integration benchmark")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument(
        "--output",
        type=str,
        default="notes/splitfuse_runtime_matrix_latest.json",
    )
    args = parser.parse_args()

    rows = [
        bench_splitfuse_dense(64, args.warmup, args.iters),
        bench_splitfuse_dense(128, args.warmup, args.iters),
    ]

    payload = {
        "version": mlx_mfa_version,
        "device": get_device_info(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": args.warmup,
        "iters": args.iters,
        "results": rows,
    }

    out_path = args.output
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
