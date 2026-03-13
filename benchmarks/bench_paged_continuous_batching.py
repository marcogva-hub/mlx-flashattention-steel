#!/usr/bin/env python3
"""Scheduler-style paged continuous batching benchmark.

Compares remap-aware paged runtime/API paths with manual row-reordered baselines.
Runs in a separate process.
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
    flash_attention_paged,
    flash_attention_paged_varlen,
    get_device_info,
)


@dataclass(frozen=True)
class SchedulerScenario:
    name: str
    slot_seq_ids: tuple[int, ...]
    active_steps: tuple[tuple[int, ...], ...]


def _measure_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(max(0, warmup)):
        out = fn()
        mx.eval(out)
        mx.synchronize()
    samples = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return float(samples[len(samples) // 2])


def _init_cache_state(
    slot_seq_ids: tuple[int, ...],
    prefill_len: int,
    H_kv: int,
    D: int,
    *,
    seed_base: int,
):
    k_prefill = {}
    v_prefill = {}
    for i, sid in enumerate(slot_seq_ids):
        mx.random.seed(seed_base + i)
        k_prefill[sid] = mx.random.normal((1, H_kv, prefill_len, D)).astype(mx.float16)
        mx.random.seed(seed_base + 100 + i)
        v_prefill[sid] = mx.random.normal((1, H_kv, prefill_len, D)).astype(mx.float16)
    mx.eval(*k_prefill.values(), *v_prefill.values())
    return k_prefill, v_prefill


def _build_seq_lengths_for_step(active_slots: tuple[int, ...], base: int):
    return [base + (i % 3) for i in range(len(active_slots))]


def _pack_queries(q_parts: list[mx.array]):
    offsets = [0]
    for q in q_parts:
        offsets.append(offsets[-1] + int(q.shape[2]))
    total_q = offsets[-1]
    if total_q == 0:
        H_q = q_parts[0].shape[1]
        D = q_parts[0].shape[3]
        q_pack = mx.zeros((1, H_q, 0, D), dtype=q_parts[0].dtype)
    else:
        q_pack = mx.concatenate(q_parts, axis=2)
    cu = mx.array(offsets, dtype=mx.int32)
    return q_pack, cu


def bench_paged_step_batch(
    scenario: SchedulerScenario,
    D: int,
    *,
    warmup: int,
    iters: int,
    block_size: int,
    prefill_len: int,
):
    H_q, H_kv = 8, 4
    scale = 1.0 / math.sqrt(D)
    slot_seq_ids = scenario.slot_seq_ids

    k_prefill, v_prefill = _init_cache_state(
        slot_seq_ids,
        prefill_len,
        H_kv,
        D,
        seed_base=900,
    )

    q_steps = []
    k_steps = []
    v_steps = []
    for step_idx, active_slots in enumerate(scenario.active_steps):
        B_active = len(active_slots)
        mx.random.seed(1000 + step_idx)
        q_steps.append(mx.random.normal((B_active, H_q, 1, D)).astype(mx.float16))
        mx.random.seed(1100 + step_idx)
        k_steps.append(mx.random.normal((B_active, H_kv, 1, D)).astype(mx.float16))
        mx.random.seed(1200 + step_idx)
        v_steps.append(mx.random.normal((B_active, H_kv, 1, D)).astype(mx.float16))
    mx.eval(*q_steps, *k_steps, *v_steps)

    def run_manual():
        cache = PagedKVCache(num_blocks=256, block_size=block_size, H=H_kv, D=D)
        for sid in slot_seq_ids:
            cache.append(k_prefill[sid], v_prefill[sid], seq_id=sid)

        last_out = None
        for active_slots, q, k_new, v_new in zip(scenario.active_steps, q_steps, k_steps, v_steps):
            active_seq_ids = [slot_seq_ids[i] for i in active_slots]
            for b, sid in enumerate(active_seq_ids):
                cache.append(k_new[b : b + 1], v_new[b : b + 1], seq_id=sid)
            table = cache.get_block_table(active_seq_ids)
            lens = cache.get_seq_lens(active_seq_ids)
            last_out = flash_attention_paged(
                q,
                cache.k_pool,
                cache.v_pool,
                table,
                lens,
                scale=scale,
                causal=True,
                block_size=block_size,
            )
        return last_out

    def run_runtime_remap():
        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            quantized_kv=False,
            query_layout="batched",
            B=1,
            H_q=H_q,
            H_kv=H_kv,
            D=D,
            num_blocks=256,
            block_size=block_size,
        )
        for sid in slot_seq_ids:
            rt.context.cache.append(k_prefill[sid], v_prefill[sid], seq_id=sid)

        last_out = None
        for active_slots, q, k_new, v_new in zip(scenario.active_steps, q_steps, k_steps, v_steps):
            idx = mx.array(active_slots, dtype=mx.int32)
            last_out = rt.paged_step_batch(
                q,
                k_new,
                v_new,
                seq_ids=slot_seq_ids,
                cache_batch_idx=idx,
                scale=scale,
                causal=True,
            )
        return last_out

    ms_manual = _measure_ms(run_manual, warmup=warmup, iters=iters)
    ms_runtime = _measure_ms(run_runtime_remap, warmup=warmup, iters=iters)

    out_manual = run_manual()
    out_runtime = run_runtime_remap()
    mx.eval(out_manual, out_runtime)
    err = float(mx.max(mx.abs(out_manual.astype(mx.float32) - out_runtime.astype(mx.float32))).item())

    return {
        "scenario": scenario.name,
        "kind": "paged_step_batch_remap",
        "D": D,
        "dtype": "f16",
        "steps": len(scenario.active_steps),
        "manual_ms": ms_manual,
        "runtime_remap_ms": ms_runtime,
        "ratio_runtime_vs_manual": (ms_manual / ms_runtime) if ms_runtime > 0 else float("inf"),
        "max_err": err,
    }


def bench_paged_varlen_remap(
    scenario: SchedulerScenario,
    D: int,
    *,
    warmup: int,
    iters: int,
    block_size: int,
    prefill_len: int,
):
    H_q, H_kv = 8, 4
    scale = 1.0 / math.sqrt(D)
    slot_seq_ids = scenario.slot_seq_ids

    k_prefill, v_prefill = _init_cache_state(
        slot_seq_ids,
        prefill_len,
        H_kv,
        D,
        seed_base=1300,
    )

    q_pack_steps = []
    cu_steps = []
    for step_idx, active_slots in enumerate(scenario.active_steps):
        q_lens = _build_seq_lengths_for_step(active_slots, base=1)
        q_parts = []
        for i, ql in enumerate(q_lens):
            mx.random.seed(1400 + step_idx * 10 + i)
            q_parts.append(mx.random.normal((1, H_q, ql, D)).astype(mx.float16))
        q_pack, cu = _pack_queries(q_parts)
        q_pack_steps.append(q_pack)
        cu_steps.append(cu)
    mx.eval(*q_pack_steps, *cu_steps)

    def run_manual():
        cache = PagedKVCache(num_blocks=256, block_size=block_size, H=H_kv, D=D)
        for sid in slot_seq_ids:
            cache.append(k_prefill[sid], v_prefill[sid], seq_id=sid)

        last_out = None
        for active_slots, q_pack, cu in zip(scenario.active_steps, q_pack_steps, cu_steps):
            active_seq_ids = [slot_seq_ids[i] for i in active_slots]
            table = cache.get_block_table(active_seq_ids)
            lens = cache.get_seq_lens(active_seq_ids)
            cu_list = [int(x) for x in cu.tolist()]
            max_q = max((cu_list[i + 1] - cu_list[i]) for i in range(len(cu_list) - 1))
            last_out = flash_attention_paged_varlen(
                q_pack,
                cache.k_pool,
                cache.v_pool,
                table,
                lens,
                cu,
                max_seqlen_q=max_q,
                scale=scale,
                causal=True,
                block_size=block_size,
            )
        return last_out

    def run_runtime_remap():
        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            quantized_kv=False,
            query_layout="packed",
            B=1,
            H_q=H_q,
            H_kv=H_kv,
            D=D,
            num_blocks=256,
            block_size=block_size,
        )
        for sid in slot_seq_ids:
            rt.context.cache.append(k_prefill[sid], v_prefill[sid], seq_id=sid)

        last_out = None
        for active_slots, q_pack, cu in zip(scenario.active_steps, q_pack_steps, cu_steps):
            idx = mx.array(active_slots, dtype=mx.int32)
            last_out = rt.paged_varlen(
                q_pack,
                cu,
                seq_ids=slot_seq_ids,
                cache_batch_idx=idx,
                scale=scale,
                causal=True,
            )
        return last_out

    ms_manual = _measure_ms(run_manual, warmup=warmup, iters=iters)
    ms_runtime = _measure_ms(run_runtime_remap, warmup=warmup, iters=iters)

    out_manual = run_manual()
    out_runtime = run_runtime_remap()
    mx.eval(out_manual, out_runtime)
    err = float(mx.max(mx.abs(out_manual.astype(mx.float32) - out_runtime.astype(mx.float32))).item())

    return {
        "scenario": scenario.name,
        "kind": "paged_varlen_remap",
        "D": D,
        "dtype": "f16",
        "steps": len(scenario.active_steps),
        "manual_ms": ms_manual,
        "runtime_remap_ms": ms_runtime,
        "ratio_runtime_vs_manual": (ms_manual / ms_runtime) if ms_runtime > 0 else float("inf"),
        "max_err": err,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--prefill-len", type=int, default=512)
    ap.add_argument(
        "--output",
        type=str,
        default="notes/paged_continuous_batching_latest.json",
    )
    args = ap.parse_args()

    scenario = SchedulerScenario(
        name="scheduler_reorder_active3_of6",
        slot_seq_ids=(101, 102, 103, 104, 105, 106),
        active_steps=((0, 2, 5), (5, 1, 2), (4, 0, 3), (2, 5, 1), (3, 4, 0)),
    )

    rows = []
    for D in (64, 128):
        row_step = bench_paged_step_batch(
            scenario,
            D,
            warmup=args.warmup,
            iters=args.iters,
            block_size=args.block_size,
            prefill_len=args.prefill_len,
        )
        rows.append(row_step)
        print(
            f"{row_step['kind']:>26} D={D:<3} "
            f"runtime={row_step['runtime_remap_ms']:.3f}ms "
            f"manual={row_step['manual_ms']:.3f}ms "
            f"ratio={row_step['ratio_runtime_vs_manual']:.3f} "
            f"err={row_step['max_err']:.3e}"
        )

        row_varlen = bench_paged_varlen_remap(
            scenario,
            D,
            warmup=args.warmup,
            iters=args.iters,
            block_size=args.block_size,
            prefill_len=args.prefill_len,
        )
        rows.append(row_varlen)
        print(
            f"{row_varlen['kind']:>26} D={D:<3} "
            f"runtime={row_varlen['runtime_remap_ms']:.3f}ms "
            f"manual={row_varlen['manual_ms']:.3f}ms "
            f"ratio={row_varlen['ratio_runtime_vs_manual']:.3f} "
            f"err={row_varlen['max_err']:.3e}"
        )

    payload = {
        "date": time.strftime("%Y-%m-%d"),
        "mlx_mfa_version": mlx_mfa_version,
        "device": get_device_info(),
        "warmup": args.warmup,
        "iters": args.iters,
        "block_size": args.block_size,
        "prefill_len": args.prefill_len,
        "rows": rows,
        "notes": [
            "Scheduler-style active-order remap benchmark for paged batched and packed-varlen runtime paths.",
            "Primary win target is operational capability with explicit mapping; performance deltas may be modest.",
        ],
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved ->", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
