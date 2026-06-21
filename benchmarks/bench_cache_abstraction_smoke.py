#!/usr/bin/env python3
"""Smoke benchmark for KV cache abstraction overhead.

Goal: ensure the cache adapter layer is operational and does not introduce
obvious regressions in representative serving flows.

Run in a separate process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import mlx.core as mx

from mlx_mfa import (
    __version__ as mlx_mfa_version,
    DenseKVCache,
    PagedKVCache,
    InferenceContext,
    create_decode_runtime,
    adapt_kv_cache,
    flash_attention_speculative_verify,
    get_device_info,
)


def _eval_any(x):
    if isinstance(x, (tuple, list)):
        for y in x:
            _eval_any(y)
        return
    if isinstance(x, dict):
        for y in x.values():
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


def bench_dense_cache_adapter(D: int, warmup: int, iters: int) -> dict:
    mx.random.seed(8100 + D)
    k = mx.random.normal((1, 4, 16, D)).astype(mx.float16)
    mx.random.seed(8200 + D)
    v = mx.random.normal((1, 4, 16, D)).astype(mx.float16)
    mx.eval(k, v)

    cache = DenseKVCache(B=1, H=4, D=D, max_seq_len=128)
    adapter = adapt_kv_cache(cache)

    def direct():
        cache.reset()
        cache.append(k, v)
        return cache.k_for_attention(0), cache.v_for_attention(0)

    def through_adapter():
        adapter.reset(seq_id=0)
        adapter.append(k, v, seq_id=0)
        return adapter.attention_k(0), adapter.attention_v(0)

    ms_direct = _measure_ms(direct, warmup=warmup, iters=iters)
    ms_adapter = _measure_ms(through_adapter, warmup=warmup, iters=iters)
    return {
        "scenario": "dense_cache_append_attention_view",
        "D": D,
        "direct_ms": ms_direct,
        "adapter_ms": ms_adapter,
        "ratio_adapter_vs_direct": (ms_adapter / ms_direct) if ms_direct > 0 else float("inf"),
    }


def bench_paged_cache_adapter(D: int, warmup: int, iters: int) -> dict:
    mx.random.seed(9100 + D)
    k = mx.random.normal((1, 4, 12, D)).astype(mx.float16)
    mx.random.seed(9200 + D)
    v = mx.random.normal((1, 4, 12, D)).astype(mx.float16)
    mx.eval(k, v)

    cache = PagedKVCache(num_blocks=64, block_size=16, H=4, D=D)
    adapter = adapt_kv_cache(cache)

    def direct():
        cache.reset(seq_id=11)
        cache.append(k, v, seq_id=11)
        return cache.get_block_table([11]), cache.get_seq_lens([11])

    def through_adapter():
        adapter.reset(seq_id=11)
        adapter.append(k, v, seq_id=11)
        return adapter.paged_tables([11])

    ms_direct = _measure_ms(direct, warmup=warmup, iters=iters)
    ms_adapter = _measure_ms(through_adapter, warmup=warmup, iters=iters)
    return {
        "scenario": "paged_cache_append_table_view",
        "D": D,
        "direct_ms": ms_direct,
        "adapter_ms": ms_adapter,
        "ratio_adapter_vs_direct": (ms_adapter / ms_direct) if ms_direct > 0 else float("inf"),
    }


def bench_runtime_dense_step(D: int, warmup: int, iters: int) -> dict:
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(10100 + D)
    q_pre = mx.random.normal((1, 4, 32, D)).astype(mx.float16)
    mx.random.seed(10200 + D)
    k_pre = mx.random.normal((1, 4, 32, D)).astype(mx.float16)
    mx.random.seed(10300 + D)
    v_pre = mx.random.normal((1, 4, 32, D)).astype(mx.float16)
    mx.random.seed(10400 + D)
    q_step = mx.random.normal((1, 4, 1, D)).astype(mx.float16)
    mx.random.seed(10500 + D)
    k_step = mx.random.normal((1, 4, 1, D)).astype(mx.float16)
    mx.random.seed(10600 + D)
    v_step = mx.random.normal((1, 4, 1, D)).astype(mx.float16)
    mx.eval(q_pre, k_pre, v_pre, q_step, k_step, v_step)

    ctx = InferenceContext(B=1, H_kv=4, D=D, max_seq_len=128)
    rt = create_decode_runtime(
        backend="dense",
        quantized_kv=False,
        B=1,
        H_q=4,
        H_kv=4,
        D=D,
        max_seq_len=128,
    )

    def direct_ctx():
        ctx.reset()
        _ = ctx.prefill(q_pre, k_pre, v_pre, scale=scale, causal=True)
        return ctx.step(q_step, k_step, v_step, scale=scale)

    def runtime_path():
        rt.reset()
        _ = rt.prefill(q_pre, k_pre, v_pre, scale=scale, causal=True)
        return rt.step(q_step, k_step, v_step, scale=scale)

    ms_ctx = _measure_ms(direct_ctx, warmup=warmup, iters=iters)
    ms_rt = _measure_ms(runtime_path, warmup=warmup, iters=iters)
    return {
        "scenario": "dense_runtime_prefill_step",
        "D": D,
        "context_ms": ms_ctx,
        "runtime_ms": ms_rt,
        "ratio_runtime_vs_context": (ms_rt / ms_ctx) if ms_ctx > 0 else float("inf"),
    }


def bench_runtime_prefix_and_speculative(D: int, warmup: int, iters: int) -> dict:
    scale = 1.0 / math.sqrt(D)
    rt = create_decode_runtime(
        backend="dense",
        quantized_kv=False,
        B=1,
        H_q=4,
        H_kv=4,
        D=D,
        max_seq_len=256,
    )

    mx.random.seed(11100 + D)
    q_pre = mx.random.normal((1, 4, 16, D)).astype(mx.float16)
    mx.random.seed(11200 + D)
    k_pre = mx.random.normal((1, 4, 16, D)).astype(mx.float16)
    mx.random.seed(11300 + D)
    v_pre = mx.random.normal((1, 4, 16, D)).astype(mx.float16)

    mx.random.seed(11400 + D)
    q_s = mx.random.normal((1, 4, 8, D)).astype(mx.float16)
    mx.random.seed(11500 + D)
    k_s = mx.random.normal((1, 4, 8, D)).astype(mx.float16)
    mx.random.seed(11600 + D)
    v_s = mx.random.normal((1, 4, 8, D)).astype(mx.float16)

    mx.random.seed(11700 + D)
    q_verify = mx.random.normal((1, 4, 4, D)).astype(mx.float16)
    draft_ids = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    mx.eval(q_pre, k_pre, v_pre, q_s, k_s, v_s, q_verify, draft_ids)

    rt.register_prefix("p0", q_pre, k_pre, v_pre, scale=scale, overwrite=True)

    def prefix_flow():
        return rt.prefill_with_prefix(
            q_s,
            k_s,
            v_s,
            prefix_id="p0",
            chunk_size=4,
            scale=scale,
            causal=True,
            reset=True,
        )

    def runtime_spec_step():
        rt.reset()
        _ = rt.prefill(q_pre, k_pre, v_pre, scale=scale, causal=True)
        return rt.speculative_step(
            q_verify,
            draft_ids,
            accept_logprob_delta=0.0,
        )

    def helper_spec_verify():
        rt.reset()
        _ = rt.prefill(q_pre, k_pre, v_pre, scale=scale, causal=True)
        k_hist = rt.context.k_cache
        v_hist = rt.context.v_cache
        return flash_attention_speculative_verify(
            q_verify,
            k_hist,
            v_hist,
            draft_ids,
            scale=scale,
            causal=True,
        )

    ms_prefix = _measure_ms(prefix_flow, warmup=warmup, iters=iters)
    ms_spec_runtime = _measure_ms(runtime_spec_step, warmup=warmup, iters=iters)
    ms_spec_helper = _measure_ms(helper_spec_verify, warmup=warmup, iters=iters)

    return {
        "scenario": "runtime_prefix_and_speculative",
        "D": D,
        "prefix_runtime_ms": ms_prefix,
        "spec_runtime_ms": ms_spec_runtime,
        "spec_helper_ms": ms_spec_helper,
        "ratio_runtime_spec_vs_helper": (
            ms_spec_runtime / ms_spec_helper if ms_spec_helper > 0 else float("inf")
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="KV cache abstraction smoke benchmark")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument(
        "--output",
        type=str,
        default="devnotes/cache_abstraction_smoke_latest.json",
    )
    args = parser.parse_args()

    results = []
    for D in (64, 128):
        results.append(bench_dense_cache_adapter(D, args.warmup, args.iters))
        results.append(bench_paged_cache_adapter(D, args.warmup, args.iters))
        results.append(bench_runtime_dense_step(D, args.warmup, args.iters))
        results.append(bench_runtime_prefix_and_speculative(D, args.warmup, args.iters))

    payload = {
        "version": mlx_mfa_version,
        "device": get_device_info(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }

    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
