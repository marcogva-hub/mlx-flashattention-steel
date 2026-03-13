#!/usr/bin/env python3
"""Hybrid KV cache smoke benchmark.

Goal: validate that real hybrid tier behavior is operational and quantify
baseline-vs-hybrid overhead in representative serving flows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any

import mlx.core as mx

from mlx_mfa import __version__ as mlx_mfa_version
from mlx_mfa import create_decode_runtime, get_device_info


def _eval_any(x: Any) -> None:
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


def _make_runtime(
    *,
    backend: str,
    D: int,
    hybrid_cache: bool,
    hybrid_hot_seq_capacity: int,
    paged: bool = False,
    B: int = 1,
):
    kwargs = {
        "backend": backend,
        "paged": paged,
        "quantized_kv": False,
        "hybrid_cache": hybrid_cache,
        "hybrid_with_secondary": True,
        "hybrid_hot_seq_capacity": hybrid_hot_seq_capacity,
        "B": B,
        "H_q": 4,
        "H_kv": 4,
        "D": D,
        "max_seq_len": 512,
    }
    if paged:
        kwargs.update({"num_blocks": 128, "block_size": 16})
    return create_decode_runtime(**kwargs)


def bench_dense_decode(D: int, warmup: int, iters: int) -> dict[str, Any]:
    scale = 1.0 / math.sqrt(D)

    mx.random.seed(5000 + D)
    q_pre = mx.random.normal((1, 4, 64, D)).astype(mx.float16)
    mx.random.seed(5100 + D)
    k_pre = mx.random.normal((1, 4, 64, D)).astype(mx.float16)
    mx.random.seed(5200 + D)
    v_pre = mx.random.normal((1, 4, 64, D)).astype(mx.float16)

    mx.random.seed(5300 + D)
    q_step = mx.random.normal((1, 4, 1, D)).astype(mx.float16)
    mx.random.seed(5400 + D)
    k_step = mx.random.normal((1, 4, 1, D)).astype(mx.float16)
    mx.random.seed(5500 + D)
    v_step = mx.random.normal((1, 4, 1, D)).astype(mx.float16)
    mx.eval(q_pre, k_pre, v_pre, q_step, k_step, v_step)

    rt_base = _make_runtime(
        backend="dense",
        D=D,
        hybrid_cache=False,
        hybrid_hot_seq_capacity=1,
    )
    rt_hybrid = _make_runtime(
        backend="dense",
        D=D,
        hybrid_cache=True,
        hybrid_hot_seq_capacity=1,
    )

    def run_base():
        rt_base.reset()
        _ = rt_base.prefill(q_pre, k_pre, v_pre, scale=scale, causal=True)
        out = None
        for _ in range(8):
            out = rt_base.step(q_step, k_step, v_step, scale=scale)
        return out

    def run_hybrid():
        rt_hybrid.reset()
        _ = rt_hybrid.prefill(q_pre, k_pre, v_pre, scale=scale, causal=True)
        rt_hybrid.hybrid_prefetch([0], pin=False, reason="bench_dense")
        out = None
        for _ in range(8):
            out = rt_hybrid.step(q_step, k_step, v_step, scale=scale)
        return out

    ms_base = _measure_ms(run_base, warmup=warmup, iters=iters)
    ms_hybrid = _measure_ms(run_hybrid, warmup=warmup, iters=iters)

    return {
        "scenario": "dense_prefill_plus_8_decode_steps",
        "D": D,
        "baseline_ms": ms_base,
        "hybrid_ms": ms_hybrid,
        "ratio_hybrid_vs_baseline": (ms_hybrid / ms_base) if ms_base > 0 else float("inf"),
    }


def bench_paged_decode(D: int, warmup: int, iters: int) -> dict[str, Any]:
    scale = 1.0 / math.sqrt(D)

    mx.random.seed(6000 + D)
    q = mx.random.normal((2, 4, 1, D)).astype(mx.float16)
    mx.random.seed(6100 + D)
    k = mx.random.normal((2, 4, 1, D)).astype(mx.float16)
    mx.random.seed(6200 + D)
    v = mx.random.normal((2, 4, 1, D)).astype(mx.float16)
    mx.eval(q, k, v)

    seq_ids = [10, 11]

    rt_base = _make_runtime(
        backend="paged",
        D=D,
        hybrid_cache=False,
        hybrid_hot_seq_capacity=1,
        paged=True,
        B=2,
    )
    rt_hybrid = _make_runtime(
        backend="paged",
        D=D,
        hybrid_cache=True,
        hybrid_hot_seq_capacity=2,
        paged=True,
        B=2,
    )

    def run_base():
        rt_base.reset()
        out = None
        for _ in range(16):
            out = rt_base.paged_step_batch(
                q,
                k,
                v,
                seq_ids=seq_ids,
                scale=scale,
                causal=True,
            )
        return out

    def run_hybrid():
        rt_hybrid.reset()
        rt_hybrid.hybrid_prefetch(seq_ids, pin=False, reason="bench_paged_start")
        out = None
        for _ in range(16):
            out = rt_hybrid.paged_step_batch(
                q,
                k,
                v,
                seq_ids=seq_ids,
                scale=scale,
                causal=True,
            )
        return out

    ms_base = _measure_ms(run_base, warmup=warmup, iters=iters)
    ms_hybrid = _measure_ms(run_hybrid, warmup=warmup, iters=iters)

    return {
        "scenario": "paged_batch_2x16_decode_steps",
        "D": D,
        "baseline_ms": ms_base,
        "hybrid_ms": ms_hybrid,
        "ratio_hybrid_vs_baseline": (ms_hybrid / ms_base) if ms_base > 0 else float("inf"),
    }


def bench_prefix_reuse(D: int, warmup: int, iters: int) -> dict[str, Any]:
    scale = 1.0 / math.sqrt(D)

    mx.random.seed(7000 + D)
    q_pre = mx.random.normal((1, 4, 24, D)).astype(mx.float16)
    mx.random.seed(7100 + D)
    k_pre = mx.random.normal((1, 4, 24, D)).astype(mx.float16)
    mx.random.seed(7200 + D)
    v_pre = mx.random.normal((1, 4, 24, D)).astype(mx.float16)

    mx.random.seed(7300 + D)
    q_suf = mx.random.normal((1, 4, 8, D)).astype(mx.float16)
    mx.random.seed(7400 + D)
    k_suf = mx.random.normal((1, 4, 8, D)).astype(mx.float16)
    mx.random.seed(7500 + D)
    v_suf = mx.random.normal((1, 4, 8, D)).astype(mx.float16)
    mx.eval(q_pre, k_pre, v_pre, q_suf, k_suf, v_suf)

    rt_base = _make_runtime(
        backend="dense",
        D=D,
        hybrid_cache=False,
        hybrid_hot_seq_capacity=1,
    )
    rt_hybrid = _make_runtime(
        backend="dense",
        D=D,
        hybrid_cache=True,
        hybrid_hot_seq_capacity=1,
    )

    rt_base.register_prefix("p0", q_pre, k_pre, v_pre, scale=scale, overwrite=True)
    rt_hybrid.register_prefix("p0", q_pre, k_pre, v_pre, scale=scale, overwrite=True)

    def run_base():
        return rt_base.prefill_with_prefix(
            q_suf,
            k_suf,
            v_suf,
            prefix_id="p0",
            chunk_size=4,
            scale=scale,
            causal=True,
            reset=True,
        )

    def run_hybrid():
        rt_hybrid.hybrid_prefetch([0], pin=True, reason="bench_prefix")
        return rt_hybrid.prefill_with_prefix(
            q_suf,
            k_suf,
            v_suf,
            prefix_id="p0",
            chunk_size=4,
            scale=scale,
            causal=True,
            reset=True,
        )

    ms_base = _measure_ms(run_base, warmup=warmup, iters=iters)
    ms_hybrid = _measure_ms(run_hybrid, warmup=warmup, iters=iters)

    return {
        "scenario": "dense_prefix_reuse_prefill_with_prefix",
        "D": D,
        "baseline_ms": ms_base,
        "hybrid_ms": ms_hybrid,
        "ratio_hybrid_vs_baseline": (ms_hybrid / ms_base) if ms_base > 0 else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid KV cache smoke benchmark")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument(
        "--output",
        type=str,
        default="notes/hybrid_kv_cache_bench_latest.json",
    )
    args = parser.parse_args()

    results = []
    for D in (64, 128):
        results.append(bench_dense_decode(D, args.warmup, args.iters))
        results.append(bench_paged_decode(D, args.warmup, args.iters))
        results.append(bench_prefix_reuse(D, args.warmup, args.iters))

    payload = {
        "version": mlx_mfa_version,
        "device": get_device_info(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
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
