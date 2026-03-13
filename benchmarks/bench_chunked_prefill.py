#!/usr/bin/env python3
"""Chunked prefill benchmark matrix (serving-oriented).

Compares monolithic prefill paths against explicit chunked-prefill runtime paths
for dense and paged modes, and reports chunk latency profile stats.

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
    PagedKVCache,
    create_decode_runtime,
    flash_attention_paged_varlen,
    get_device_info,
)


@dataclass(frozen=True)
class DenseScenario:
    name: str
    B: int
    H: int
    N: int


@dataclass(frozen=True)
class PagedBatchedScenario:
    name: str
    B: int
    H_q: int
    H_kv: int
    N: int
    seq_ids: tuple[int, ...]


@dataclass(frozen=True)
class PagedPackedScenario:
    name: str
    H_q: int
    H_kv: int
    q_lens: tuple[int, ...]
    seq_ids: tuple[int, ...]


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


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    vals = sorted(values)
    i50 = int(0.50 * (len(vals) - 1))
    i95 = int(0.95 * (len(vals) - 1))
    return {
        "p50_ms": float(vals[i50]),
        "p95_ms": float(vals[i95]),
        "max_ms": float(vals[-1]),
    }


def _pack_queries(q_seqs: list[mx.array]):
    offsets = [0]
    for q in q_seqs:
        offsets.append(offsets[-1] + int(q.shape[2]))
    total_q = offsets[-1]
    q_pack = (
        mx.concatenate(q_seqs, axis=2)
        if total_q > 0
        else q_seqs[0][:, :, :0, :]
    )
    return q_pack, mx.array(offsets, dtype=mx.int32), offsets


def bench_dense_one(
    scenario: DenseScenario,
    D: int,
    *,
    chunk_size: int,
    warmup: int,
    iters: int,
) -> dict:
    mx.random.seed(3000 + D + scenario.N)
    q = mx.random.normal((scenario.B, scenario.H, scenario.N, D)).astype(mx.float16)
    k = mx.random.normal((scenario.B, scenario.H, scenario.N, D)).astype(mx.float16)
    v = mx.random.normal((scenario.B, scenario.H, scenario.N, D)).astype(mx.float16)
    mx.eval(q, k, v)
    scale = 1.0 / math.sqrt(D)

    def run_monolithic():
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=scenario.B,
            H_q=scenario.H,
            H_kv=scenario.H,
            D=D,
            max_seq_len=max(8192, scenario.N * 2),
        )
        return rt.prefill(q, k, v, scale=scale, causal=True)

    def run_chunked():
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=scenario.B,
            H_q=scenario.H,
            H_kv=scenario.H,
            D=D,
            max_seq_len=max(8192, scenario.N * 2),
        )
        return rt.chunked_prefill(
            q,
            k,
            v,
            chunk_size=chunk_size,
            scale=scale,
            causal=True,
        )

    def chunk_profile_once():
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=scenario.B,
            H_q=scenario.H,
            H_kv=scenario.H,
            D=D,
            max_seq_len=max(8192, scenario.N * 2),
        )
        step_ms: list[float] = []
        out_parts = []
        for s in range(0, scenario.N, chunk_size):
            e = min(scenario.N, s + chunk_size)
            t0 = time.perf_counter()
            out = rt.step(
                q[:, :, s:e, :],
                k[:, :, s:e, :],
                v[:, :, s:e, :],
                scale=scale,
            )
            mx.eval(out)
            mx.synchronize()
            step_ms.append((time.perf_counter() - t0) * 1000.0)
            out_parts.append(out)
        out_ref = (
            mx.concatenate(out_parts, axis=2)
            if out_parts
            else mx.zeros((scenario.B, scenario.H, 0, D), dtype=q.dtype)
        )
        return out_ref, step_ms

    ms_monolithic = _measure_ms(run_monolithic, warmup=warmup, iters=iters)
    ms_chunked = _measure_ms(run_chunked, warmup=warmup, iters=iters)

    out_mono = run_monolithic()
    out_chunk = run_chunked()
    out_ref, chunk_steps = chunk_profile_once()
    mx.eval(out_mono, out_chunk, out_ref)

    err_chunk_vs_mono = float(
        mx.abs(out_chunk.astype(mx.float32) - out_mono.astype(mx.float32)).max().item()
    )
    err_chunk_vs_ref = float(
        mx.abs(out_chunk.astype(mx.float32) - out_ref.astype(mx.float32)).max().item()
    )

    return {
        "group": "dense",
        "scenario": scenario.name,
        "D": D,
        "dtype": "f16",
        "N": scenario.N,
        "B": scenario.B,
        "chunk_size": chunk_size,
        "num_chunks": int((scenario.N + chunk_size - 1) // chunk_size),
        "monolithic_ms": ms_monolithic,
        "chunked_ms": ms_chunked,
        "ratio_chunked_vs_monolithic": (ms_chunked / ms_monolithic) if ms_monolithic > 0 else float("inf"),
        "max_err_chunked_vs_monolithic": err_chunk_vs_mono,
        "max_err_chunked_vs_manual_incremental": err_chunk_vs_ref,
        "chunk_profile": _quantiles(chunk_steps),
    }


def bench_paged_batched_one(
    scenario: PagedBatchedScenario,
    D: int,
    *,
    chunk_size: int,
    warmup: int,
    iters: int,
    block_size: int,
) -> dict:
    mx.random.seed(4000 + D + scenario.N)
    q = mx.random.normal((scenario.B, scenario.H_q, scenario.N, D)).astype(mx.float16)
    k = mx.random.normal((scenario.B, scenario.H_kv, scenario.N, D)).astype(mx.float16)
    v = mx.random.normal((scenario.B, scenario.H_kv, scenario.N, D)).astype(mx.float16)
    mx.eval(q, k, v)
    scale = 1.0 / math.sqrt(D)

    def _new_runtime():
        return create_decode_runtime(
            backend="paged",
            paged=True,
            query_layout="batched",
            quantized_kv=False,
            B=1,
            H_q=scenario.H_q,
            H_kv=scenario.H_kv,
            D=D,
            num_blocks=512,
            block_size=block_size,
            max_seq_len=max(8192, scenario.N * 2),
        )

    def run_monolithic():
        rt = _new_runtime()
        return rt.paged_prefill_batch(
            q,
            k,
            v,
            seq_ids=scenario.seq_ids,
            scale=scale,
            causal=True,
        )

    def run_chunked():
        rt = _new_runtime()
        return rt.chunked_prefill(
            q,
            k,
            v,
            chunk_size=chunk_size,
            seq_ids=scenario.seq_ids,
            scale=scale,
            causal=True,
        )

    def manual_chunk_once():
        rt = _new_runtime()
        chunk_ms: list[float] = []
        parts = []
        for s in range(0, scenario.N, chunk_size):
            e = min(scenario.N, s + chunk_size)
            t0 = time.perf_counter()
            out = rt.paged_step_batch(
                q[:, :, s:e, :],
                k[:, :, s:e, :],
                v[:, :, s:e, :],
                seq_ids=scenario.seq_ids,
                scale=scale,
                causal=True,
            )
            mx.eval(out)
            mx.synchronize()
            chunk_ms.append((time.perf_counter() - t0) * 1000.0)
            parts.append(out)
        out_ref = (
            mx.concatenate(parts, axis=2)
            if parts
            else mx.zeros((scenario.B, scenario.H_q, 0, D), dtype=q.dtype)
        )
        return out_ref, chunk_ms

    ms_monolithic = _measure_ms(run_monolithic, warmup=warmup, iters=iters)
    ms_chunked = _measure_ms(run_chunked, warmup=warmup, iters=iters)

    out_mono = run_monolithic()
    out_chunk = run_chunked()
    out_ref, chunk_steps = manual_chunk_once()
    mx.eval(out_mono, out_chunk, out_ref)

    err_chunk_vs_mono = float(
        mx.abs(out_chunk.astype(mx.float32) - out_mono.astype(mx.float32)).max().item()
    )
    err_chunk_vs_ref = float(
        mx.abs(out_chunk.astype(mx.float32) - out_ref.astype(mx.float32)).max().item()
    )

    return {
        "group": "paged_batched",
        "scenario": scenario.name,
        "D": D,
        "dtype": "f16",
        "N": scenario.N,
        "B": scenario.B,
        "H_q": scenario.H_q,
        "H_kv": scenario.H_kv,
        "chunk_size": chunk_size,
        "num_chunks": int((scenario.N + chunk_size - 1) // chunk_size),
        "monolithic_ms": ms_monolithic,
        "chunked_ms": ms_chunked,
        "ratio_chunked_vs_monolithic": (ms_chunked / ms_monolithic) if ms_monolithic > 0 else float("inf"),
        "max_err_chunked_vs_monolithic": err_chunk_vs_mono,
        "max_err_chunked_vs_manual_incremental": err_chunk_vs_ref,
        "chunk_profile": _quantiles(chunk_steps),
    }


def bench_paged_packed_one(
    scenario: PagedPackedScenario,
    D: int,
    *,
    chunk_size: int,
    warmup: int,
    iters: int,
    block_size: int,
) -> dict:
    mx.random.seed(5000 + D + sum(scenario.q_lens))

    q_seqs = [mx.random.normal((1, scenario.H_q, ql, D)).astype(mx.float16) for ql in scenario.q_lens]
    k_seqs = [mx.random.normal((1, scenario.H_kv, ql, D)).astype(mx.float16) for ql in scenario.q_lens]
    v_seqs = [mx.random.normal((1, scenario.H_kv, ql, D)).astype(mx.float16) for ql in scenario.q_lens]
    mx.eval(*q_seqs, *k_seqs, *v_seqs)

    q_pack, cu_q, offsets = _pack_queries(q_seqs)
    k_pack = mx.concatenate(k_seqs, axis=2)
    v_pack = mx.concatenate(v_seqs, axis=2)
    mx.eval(q_pack, k_pack, v_pack, cu_q)
    scale = 1.0 / math.sqrt(D)

    def _new_runtime():
        return create_decode_runtime(
            backend="paged",
            paged=True,
            query_layout="packed",
            quantized_kv=False,
            B=1,
            H_q=scenario.H_q,
            H_kv=scenario.H_kv,
            D=D,
            num_blocks=512,
            block_size=block_size,
            max_seq_len=max(8192, sum(scenario.q_lens) * 2),
        )

    def run_monolithic_varlen():
        rt = _new_runtime()
        for sid, k_i, v_i in zip(scenario.seq_ids, k_seqs, v_seqs):
            rt.context.cache.append(k_i, v_i, seq_id=sid)
        return rt.paged_varlen(q_pack, cu_q, seq_ids=scenario.seq_ids, scale=scale, causal=True)

    def run_chunked_runtime():
        rt = _new_runtime()
        return rt.chunked_prefill(
            q_pack,
            k_pack,
            v_pack,
            chunk_size=chunk_size,
            seq_ids=scenario.seq_ids,
            cu_seqlens_q=cu_q,
            scale=scale,
            causal=True,
        )

    def manual_chunk_once():
        cache = PagedKVCache(num_blocks=512, block_size=block_size, H=scenario.H_kv, D=D)
        lengths = [offsets[i + 1] - offsets[i] for i in range(len(scenario.seq_ids))]
        consumed = [0] * len(scenario.seq_ids)
        out_parts = [[] for _ in scenario.seq_ids]
        chunk_ms: list[float] = []

        while any(consumed[i] < lengths[i] for i in range(len(scenario.seq_ids))):
            active_rows = [i for i in range(len(scenario.seq_ids)) if consumed[i] < lengths[i]]
            active_seq_ids = [scenario.seq_ids[i] for i in active_rows]
            q_parts = []
            chunk_offsets = [0]
            for i in active_rows:
                s = offsets[i] + consumed[i]
                e = min(offsets[i + 1], s + chunk_size)
                q_parts.append(q_pack[:, :, s:e, :])
                cache.append(k_pack[:, :, s:e, :], v_pack[:, :, s:e, :], seq_id=scenario.seq_ids[i])
                chunk_offsets.append(chunk_offsets[-1] + (e - s))

            q_chunk = mx.concatenate(q_parts, axis=2)
            cu_chunk = mx.array(chunk_offsets, dtype=mx.int32)
            table = cache.get_block_table(active_seq_ids)
            lens = cache.get_seq_lens(active_seq_ids)

            t0 = time.perf_counter()
            out_step = flash_attention_paged_varlen(
                q_chunk,
                cache.k_pool,
                cache.v_pool,
                table,
                lens,
                cu_chunk,
                scale=scale,
                causal=True,
                block_size=block_size,
            )
            mx.eval(out_step)
            mx.synchronize()
            chunk_ms.append((time.perf_counter() - t0) * 1000.0)

            for local_idx, i in enumerate(active_rows):
                s = chunk_offsets[local_idx]
                e = chunk_offsets[local_idx + 1]
                out_parts[i].append(out_step[:, :, s:e, :])
                consumed[i] += e - s

        out_ref = mx.concatenate(
            [
                mx.concatenate(parts, axis=2)
                if parts
                else mx.zeros((1, scenario.H_q, 0, D), dtype=q_pack.dtype)
                for parts in out_parts
            ],
            axis=2,
        )
        return out_ref, chunk_ms

    ms_monolithic = _measure_ms(run_monolithic_varlen, warmup=warmup, iters=iters)
    ms_chunked = _measure_ms(run_chunked_runtime, warmup=warmup, iters=iters)

    out_mono = run_monolithic_varlen()
    out_chunk = run_chunked_runtime()
    out_ref, chunk_steps = manual_chunk_once()
    mx.eval(out_mono, out_chunk, out_ref)

    err_chunk_vs_mono = float(
        mx.abs(out_chunk.astype(mx.float32) - out_mono.astype(mx.float32)).max().item()
    )
    err_chunk_vs_ref = float(
        mx.abs(out_chunk.astype(mx.float32) - out_ref.astype(mx.float32)).max().item()
    )

    return {
        "group": "paged_packed",
        "scenario": scenario.name,
        "D": D,
        "dtype": "f16",
        "total_q": int(sum(scenario.q_lens)),
        "B_packed": len(scenario.q_lens),
        "q_lens": list(scenario.q_lens),
        "chunk_size": chunk_size,
        "monolithic_ms": ms_monolithic,
        "chunked_ms": ms_chunked,
        "ratio_chunked_vs_monolithic": (ms_chunked / ms_monolithic) if ms_monolithic > 0 else float("inf"),
        "max_err_chunked_vs_monolithic": err_chunk_vs_mono,
        "max_err_chunked_vs_manual_incremental": err_chunk_vs_ref,
        "chunk_profile": _quantiles(chunk_steps),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunked prefill benchmark matrix")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=7)
    p.add_argument("--chunk-sizes", type=str, default="128,256,512")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument(
        "--output",
        type=str,
        default="notes/chunked_prefill_matrix_latest.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    chunk_sizes = tuple(int(x.strip()) for x in args.chunk_sizes.split(",") if x.strip())

    dense_scenarios = [
        DenseScenario(name="dense_b1h8_n4096", B=1, H=8, N=4096),
        DenseScenario(name="dense_b1h8_n8192", B=1, H=8, N=8192),
    ]
    paged_batched_scenarios = [
        PagedBatchedScenario(
            name="paged_b2_gqa_n4096",
            B=2,
            H_q=8,
            H_kv=4,
            N=4096,
            seq_ids=(101, 202),
        ),
    ]
    paged_packed_scenarios = [
        PagedPackedScenario(
            name="paged_packed_hetero_prefill",
            H_q=8,
            H_kv=4,
            q_lens=(2048, 1024, 3072),
            seq_ids=(11, 22, 33),
        ),
    ]

    rows = []
    for D in (64, 128):
        for chunk_size in chunk_sizes:
            for sc in dense_scenarios:
                rows.append(
                    bench_dense_one(
                        sc,
                        D,
                        chunk_size=chunk_size,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                )
            for sc in paged_batched_scenarios:
                rows.append(
                    bench_paged_batched_one(
                        sc,
                        D,
                        chunk_size=chunk_size,
                        warmup=args.warmup,
                        iters=args.iters,
                        block_size=args.block_size,
                    )
                )
            for sc in paged_packed_scenarios:
                rows.append(
                    bench_paged_packed_one(
                        sc,
                        D,
                        chunk_size=chunk_size,
                        warmup=args.warmup,
                        iters=args.iters,
                        block_size=args.block_size,
                    )
                )

    rows_sorted = sorted(rows, key=lambda r: (r["group"], r["D"], r["scenario"], r["chunk_size"]))

    out = {
        "meta": {
            "device": get_device_info(),
            "mlx_mfa_version": mlx_mfa_version,
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "python": os.sys.version.split()[0],
            "warmup": args.warmup,
            "iters": args.iters,
            "chunk_sizes": list(chunk_sizes),
            "block_size": args.block_size,
        },
        "rows": rows_sorted,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
