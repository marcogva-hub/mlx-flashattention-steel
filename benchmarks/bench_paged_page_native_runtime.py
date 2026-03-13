#!/usr/bin/env python3
"""Focused paged runtime benchmark for page-native bridge reductions."""

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
    flash_attention_speculative_verify,
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
    vals = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        _eval_any(fn())
        mx.synchronize()
        vals.append((time.perf_counter() - t0) * 1000.0)
    vals.sort()
    return float(vals[len(vals) // 2])


def bench_splitfuse_paged_decode_only(D: int, warmup: int, iters: int) -> dict:
    scale = 1.0 / math.sqrt(D)
    rt = create_decode_runtime(
        backend="paged",
        paged=True,
        query_layout="batched",
        quantized_kv=False,
        B=1,
        H_q=4,
        H_kv=4,
        D=D,
        num_blocks=64,
        block_size=16,
    )
    sid = 5
    q_pre = mx.random.normal((1, 4, 24, D)).astype(mx.float16)
    k_pre = mx.random.normal((1, 4, 24, D)).astype(mx.float16)
    v_pre = mx.random.normal((1, 4, 24, D)).astype(mx.float16)
    q_dec = mx.random.normal((1, 4, 1, D)).astype(mx.float16)
    mx.eval(q_pre, k_pre, v_pre, q_dec)

    def manual_bridge():
        rt.reset(seq_id=sid)
        _ = rt.prefill(q_pre, k_pre, v_pre, seq_id=sid, scale=scale)
        k_hist = rt._cache_adapter().attention_k(sid)
        v_hist = rt._cache_adapter().attention_v(sid)
        return flash_attention_splitfuse(
            None,
            None,
            None,
            q_dec,
            k_hist,
            v_hist,
            scale=scale,
            causal=True,
        )

    def runtime_page_native():
        rt.reset(seq_id=sid)
        _ = rt.prefill(q_pre, k_pre, v_pre, seq_id=sid, scale=scale)
        return rt.splitfuse_step(q_dec, seq_id=sid, scale=scale)

    ms_manual = _measure_ms(manual_bridge, warmup=warmup, iters=iters)
    ms_runtime = _measure_ms(runtime_page_native, warmup=warmup, iters=iters)
    return {
        "scenario": "paged_splitfuse_decode_only",
        "D": D,
        "manual_bridge_ms": ms_manual,
        "runtime_page_native_ms": ms_runtime,
        "runtime_vs_manual": ms_runtime / ms_manual if ms_manual > 0 else float("inf"),
    }


def bench_spec_verify_paged(D: int, warmup: int, iters: int) -> dict:
    scale = 1.0 / math.sqrt(D)
    rt = create_decode_runtime(
        backend="paged",
        paged=True,
        query_layout="batched",
        quantized_kv=False,
        B=2,
        H_q=4,
        H_kv=4,
        D=D,
        num_blocks=96,
        block_size=16,
    )
    seq_ids = [3, 7]
    q_pre = mx.random.normal((2, 4, 20, D)).astype(mx.float16)
    k_pre = mx.random.normal((2, 4, 20, D)).astype(mx.float16)
    v_pre = mx.random.normal((2, 4, 20, D)).astype(mx.float16)
    q_target = mx.random.normal((2, 4, 3, D)).astype(mx.float16)
    draft_ids = mx.zeros((2, 3), dtype=mx.int32)
    mx.eval(q_pre, k_pre, v_pre, q_target, draft_ids)

    def manual_dense_bridge():
        rt.reset()
        _ = rt.paged_prefill_batch(q_pre, k_pre, v_pre, seq_ids=seq_ids, causal=True)
        k0 = rt._cache_adapter().attention_k(seq_ids[0])
        v0 = rt._cache_adapter().attention_v(seq_ids[0])
        k1 = rt._cache_adapter().attention_k(seq_ids[1])
        v1 = rt._cache_adapter().attention_v(seq_ids[1])
        k_dense = mx.concatenate([k0, k1], axis=0)
        v_dense = mx.concatenate([v0, v1], axis=0)
        return flash_attention_speculative_verify(
            q_target,
            k_dense,
            v_dense,
            draft_ids,
            scale=scale,
            causal=True,
        )

    def runtime_page_native():
        rt.reset()
        _ = rt.paged_prefill_batch(q_pre, k_pre, v_pre, seq_ids=seq_ids, causal=True)
        return rt.speculative_verify(
            q_target,
            draft_ids,
            seq_ids=seq_ids,
            scale=scale,
            causal=True,
        )

    ms_manual = _measure_ms(manual_dense_bridge, warmup=warmup, iters=iters)
    ms_runtime = _measure_ms(runtime_page_native, warmup=warmup, iters=iters)
    return {
        "scenario": "paged_speculative_verify",
        "D": D,
        "manual_bridge_ms": ms_manual,
        "runtime_page_native_ms": ms_runtime,
        "runtime_vs_manual": ms_runtime / ms_manual if ms_manual > 0 else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paged runtime page-native benchmark")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--output",
        type=str,
        default="devnotes/paged_page_native_runtime_latest.json",
    )
    args = parser.parse_args()

    rows = []
    for D in (64, 128):
        rows.append(bench_splitfuse_paged_decode_only(D, args.warmup, args.iters))
        rows.append(bench_spec_verify_paged(D, args.warmup, args.iters))

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
