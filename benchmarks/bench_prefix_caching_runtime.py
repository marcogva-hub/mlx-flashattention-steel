#!/usr/bin/env python3
"""Runtime-integrated prefix caching benchmark matrix.

Compares three serving-oriented paths:
  1) no prefix reuse baseline,
  2) explicit helper/manual prefix orchestration,
  3) runtime-managed prefix caching.

Run in a separate process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import mlx.core as mx

from mlx_mfa import (
    __version__ as mlx_mfa_version,
    create_decode_runtime,
    make_shared_prefix_cache,
    get_device_info,
)


@dataclass(frozen=True)
class PrefixScenario:
    name: str
    backend: str
    D: int
    H_q: int
    H_kv: int
    prefix_len: int
    suffix_len: int
    chunk_size: int
    requests: int


def _measure_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(max(0, warmup)):
        out = fn()
        mx.eval(out)
        mx.synchronize()

    samples: list[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return float(samples[len(samples) // 2])


def _build_inputs(sc: PrefixScenario):
    scale = 1.0 / math.sqrt(sc.D)

    mx.random.seed(7000 + sc.D)
    q_pre = mx.random.normal((1, sc.H_q, sc.prefix_len, sc.D)).astype(mx.float16)
    mx.random.seed(7100 + sc.D)
    k_pre = mx.random.normal((1, sc.H_kv, sc.prefix_len, sc.D)).astype(mx.float16)
    mx.random.seed(7200 + sc.D)
    v_pre = mx.random.normal((1, sc.H_kv, sc.prefix_len, sc.D)).astype(mx.float16)

    q_sufs = []
    k_sufs = []
    v_sufs = []
    for i in range(sc.requests):
        mx.random.seed(7300 + sc.D * 10 + i)
        q_sufs.append(mx.random.normal((1, sc.H_q, sc.suffix_len, sc.D)).astype(mx.float16))
        mx.random.seed(7400 + sc.D * 10 + i)
        k_sufs.append(mx.random.normal((1, sc.H_kv, sc.suffix_len, sc.D)).astype(mx.float16))
        mx.random.seed(7500 + sc.D * 10 + i)
        v_sufs.append(mx.random.normal((1, sc.H_kv, sc.suffix_len, sc.D)).astype(mx.float16))

    mx.eval(q_pre, k_pre, v_pre, *q_sufs, *k_sufs, *v_sufs)
    return scale, q_pre, k_pre, v_pre, q_sufs, k_sufs, v_sufs


def _run_dense_no_reuse(
    sc: PrefixScenario,
    scale: float,
    q_pre: mx.array,
    k_pre: mx.array,
    v_pre: mx.array,
    q_sufs: list[mx.array],
    k_sufs: list[mx.array],
    v_sufs: list[mx.array],
):
    rt = create_decode_runtime(
        backend="dense",
        quantized_kv=False,
        B=1,
        H_q=sc.H_q,
        H_kv=sc.H_kv,
        D=sc.D,
        max_seq_len=max(8192, sc.prefix_len + sc.suffix_len + 64),
    )
    last = None
    for q_s, k_s, v_s in zip(q_sufs, k_sufs, v_sufs):
        q_full = mx.concatenate([q_pre, q_s], axis=2)
        k_full = mx.concatenate([k_pre, k_s], axis=2)
        v_full = mx.concatenate([v_pre, v_s], axis=2)
        last = rt.prefill(q_full, k_full, v_full, scale=scale, causal=True)
    return last


def _run_dense_explicit(
    sc: PrefixScenario,
    scale: float,
    q_pre: mx.array,
    k_pre: mx.array,
    v_pre: mx.array,
    q_sufs: list[mx.array],
    k_sufs: list[mx.array],
    v_sufs: list[mx.array],
):
    _, kp, vp = make_shared_prefix_cache(q_pre, k_pre, v_pre, scale=scale)
    rt = create_decode_runtime(
        backend="dense",
        quantized_kv=False,
        B=1,
        H_q=sc.H_q,
        H_kv=sc.H_kv,
        D=sc.D,
        max_seq_len=max(8192, sc.prefix_len + sc.suffix_len + 64),
    )
    last = None
    for q_s, k_s, v_s in zip(q_sufs, k_sufs, v_sufs):
        rt.reset()
        rt.context._cache.append(kp, vp)
        last = rt.chunked_prefill(
            q_s,
            k_s,
            v_s,
            chunk_size=sc.chunk_size,
            scale=scale,
            causal=True,
            reset=False,
        )
    return last


def _run_dense_runtime_managed(
    sc: PrefixScenario,
    scale: float,
    q_pre: mx.array,
    k_pre: mx.array,
    v_pre: mx.array,
    q_sufs: list[mx.array],
    k_sufs: list[mx.array],
    v_sufs: list[mx.array],
):
    rt = create_decode_runtime(
        backend="dense",
        quantized_kv=False,
        B=1,
        H_q=sc.H_q,
        H_kv=sc.H_kv,
        D=sc.D,
        max_seq_len=max(8192, sc.prefix_len + sc.suffix_len + 64),
    )
    rt.register_prefix("shared", q_pre, k_pre, v_pre, scale=scale)
    last = None
    for q_s, k_s, v_s in zip(q_sufs, k_sufs, v_sufs):
        last = rt.prefill_with_prefix(
            q_s,
            k_s,
            v_s,
            prefix_id="shared",
            chunk_size=sc.chunk_size,
            scale=scale,
            causal=True,
            reset=True,
        )
    return last


def _run_paged_no_reuse(
    sc: PrefixScenario,
    scale: float,
    q_pre: mx.array,
    k_pre: mx.array,
    v_pre: mx.array,
    q_sufs: list[mx.array],
    k_sufs: list[mx.array],
    v_sufs: list[mx.array],
):
    rt = create_decode_runtime(
        backend="paged",
        paged=True,
        query_layout="batched",
        quantized_kv=False,
        B=1,
        H_q=sc.H_q,
        H_kv=sc.H_kv,
        D=sc.D,
        num_blocks=256,
        block_size=16,
        max_seq_len=max(8192, sc.prefix_len + sc.suffix_len + 64),
    )
    last = None
    for i, (q_s, k_s, v_s) in enumerate(zip(q_sufs, k_sufs, v_sufs)):
        sid = 100 + i
        q_full = mx.concatenate([q_pre, q_s], axis=2)
        k_full = mx.concatenate([k_pre, k_s], axis=2)
        v_full = mx.concatenate([v_pre, v_s], axis=2)
        last = rt.chunked_prefill(
            q_full,
            k_full,
            v_full,
            chunk_size=sc.chunk_size,
            seq_ids=[sid],
            scale=scale,
            causal=True,
            reset=True,
        )
    return last


def _run_paged_explicit(
    sc: PrefixScenario,
    scale: float,
    q_pre: mx.array,
    k_pre: mx.array,
    v_pre: mx.array,
    q_sufs: list[mx.array],
    k_sufs: list[mx.array],
    v_sufs: list[mx.array],
):
    _, kp, vp = make_shared_prefix_cache(q_pre, k_pre, v_pre, scale=scale)
    rt = create_decode_runtime(
        backend="paged",
        paged=True,
        query_layout="batched",
        quantized_kv=False,
        B=1,
        H_q=sc.H_q,
        H_kv=sc.H_kv,
        D=sc.D,
        num_blocks=256,
        block_size=16,
        max_seq_len=max(8192, sc.prefix_len + sc.suffix_len + 64),
    )
    last = None
    for i, (q_s, k_s, v_s) in enumerate(zip(q_sufs, k_sufs, v_sufs)):
        sid = 100 + i
        rt.context.cache.reset(seq_id=sid)
        rt.context.cache.append(kp, vp, seq_id=sid)
        last = rt.chunked_prefill(
            q_s,
            k_s,
            v_s,
            chunk_size=sc.chunk_size,
            seq_ids=[sid],
            scale=scale,
            causal=True,
            reset=False,
        )
    return last


def _run_paged_runtime_managed(
    sc: PrefixScenario,
    scale: float,
    q_pre: mx.array,
    k_pre: mx.array,
    v_pre: mx.array,
    q_sufs: list[mx.array],
    k_sufs: list[mx.array],
    v_sufs: list[mx.array],
):
    rt = create_decode_runtime(
        backend="paged",
        paged=True,
        query_layout="batched",
        quantized_kv=False,
        B=1,
        H_q=sc.H_q,
        H_kv=sc.H_kv,
        D=sc.D,
        num_blocks=256,
        block_size=16,
        max_seq_len=max(8192, sc.prefix_len + sc.suffix_len + 64),
    )
    rt.register_prefix("shared", q_pre, k_pre, v_pre, scale=scale)
    last = None
    for i, (q_s, k_s, v_s) in enumerate(zip(q_sufs, k_sufs, v_sufs)):
        sid = 100 + i
        last = rt.prefill_with_prefix(
            q_s,
            k_s,
            v_s,
            prefix_id="shared",
            seq_id=sid,
            chunk_size=sc.chunk_size,
            scale=scale,
            causal=True,
            reset=True,
        )
    return last


def run_one(sc: PrefixScenario, warmup: int, iters: int) -> dict:
    scale, q_pre, k_pre, v_pre, q_sufs, k_sufs, v_sufs = _build_inputs(sc)

    if sc.backend == "dense":
        fn_no = lambda: _run_dense_no_reuse(sc, scale, q_pre, k_pre, v_pre, q_sufs, k_sufs, v_sufs)
        fn_exp = lambda: _run_dense_explicit(sc, scale, q_pre, k_pre, v_pre, q_sufs, k_sufs, v_sufs)
        fn_rt = lambda: _run_dense_runtime_managed(sc, scale, q_pre, k_pre, v_pre, q_sufs, k_sufs, v_sufs)
    elif sc.backend == "paged":
        fn_no = lambda: _run_paged_no_reuse(sc, scale, q_pre, k_pre, v_pre, q_sufs, k_sufs, v_sufs)
        fn_exp = lambda: _run_paged_explicit(sc, scale, q_pre, k_pre, v_pre, q_sufs, k_sufs, v_sufs)
        fn_rt = lambda: _run_paged_runtime_managed(sc, scale, q_pre, k_pre, v_pre, q_sufs, k_sufs, v_sufs)
    else:
        raise ValueError(f"unknown backend {sc.backend!r}")

    ms_no = _measure_ms(fn_no, warmup=warmup, iters=iters)
    ms_exp = _measure_ms(fn_exp, warmup=warmup, iters=iters)
    ms_rt = _measure_ms(fn_rt, warmup=warmup, iters=iters)

    out_no = fn_no()
    out_exp = fn_exp()
    out_rt = fn_rt()
    mx.eval(out_no, out_exp, out_rt)

    err_rt_vs_exp = float(mx.abs(out_rt.astype(mx.float32) - out_exp.astype(mx.float32)).max().item())

    return {
        "scenario": sc.name,
        "backend": sc.backend,
        "D": sc.D,
        "H_q": sc.H_q,
        "H_kv": sc.H_kv,
        "prefix_len": sc.prefix_len,
        "suffix_len": sc.suffix_len,
        "chunk_size": sc.chunk_size,
        "requests": sc.requests,
        "no_reuse_ms": ms_no,
        "explicit_helper_ms": ms_exp,
        "runtime_managed_ms": ms_rt,
        "ratio_runtime_vs_no_reuse": (ms_no / ms_rt) if ms_rt > 0 else float("inf"),
        "ratio_runtime_vs_explicit": (ms_exp / ms_rt) if ms_rt > 0 else float("inf"),
        "max_err_runtime_vs_explicit": err_rt_vs_exp,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Runtime prefix caching benchmark matrix")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=7)
    p.add_argument(
        "--output",
        type=str,
        default="notes/prefix_caching_runtime_matrix_latest.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    scenarios = [
        PrefixScenario(
            name="dense_prefix_reuse_chunked",
            backend="dense",
            D=64,
            H_q=8,
            H_kv=4,
            prefix_len=512,
            suffix_len=256,
            chunk_size=128,
            requests=4,
        ),
        PrefixScenario(
            name="dense_prefix_reuse_chunked",
            backend="dense",
            D=128,
            H_q=8,
            H_kv=4,
            prefix_len=512,
            suffix_len=256,
            chunk_size=128,
            requests=4,
        ),
        PrefixScenario(
            name="paged_prefix_reuse_chunked",
            backend="paged",
            D=64,
            H_q=8,
            H_kv=4,
            prefix_len=512,
            suffix_len=256,
            chunk_size=128,
            requests=4,
        ),
        PrefixScenario(
            name="paged_prefix_reuse_chunked",
            backend="paged",
            D=128,
            H_q=8,
            H_kv=4,
            prefix_len=512,
            suffix_len=256,
            chunk_size=128,
            requests=4,
        ),
    ]

    rows = [run_one(sc, warmup=args.warmup, iters=args.iters) for sc in scenarios]

    out = {
        "meta": {
            "device": get_device_info(),
            "mlx_mfa_version": mlx_mfa_version,
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "python": os.sys.version.split()[0],
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "rows": rows,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
