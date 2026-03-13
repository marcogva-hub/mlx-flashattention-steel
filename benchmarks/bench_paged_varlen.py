#!/usr/bin/env python3
"""vLLM-oriented benchmark for paged KV + packed varlen queries.

Compares:
  1) Padded batched paged baseline (q padded to max_q)
  2) New flash_attention_paged_varlen path
  3) Sequence-by-sequence paged fallback

Run in a separate process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import mlx.core as mx

from mlx_mfa import __version__ as mlx_mfa_version
from mlx_mfa import flash_attention_paged, flash_attention_paged_varlen, get_device_info


@dataclass(frozen=True)
class Scenario:
    name: str
    B: int
    H_q: int
    H_kv: int
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]


def _build_pool(k_seqs: list[mx.array], v_seqs: list[mx.array], block_size: int):
    B = len(k_seqs)
    H_kv = k_seqs[0].shape[1]
    D = k_seqs[0].shape[3]

    blocks_per_seq = [(int(k.shape[2]) + block_size - 1) // block_size for k in k_seqs]
    total_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq) if blocks_per_seq else 0

    pool_k = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float16)
    pool_v = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float16)
    table = np.full((B, max_blocks), -1, dtype=np.int32)
    lens = np.zeros((B,), dtype=np.int32)

    blk_base = 0
    for b in range(B):
        k_np = np.array(k_seqs[b]).astype(np.float16)[0].transpose(1, 0, 2)  # [S,H,D]
        v_np = np.array(v_seqs[b]).astype(np.float16)[0].transpose(1, 0, 2)
        S = k_np.shape[0]
        lens[b] = S
        n_blk = blocks_per_seq[b]
        for lb in range(n_blk):
            table[b, lb] = blk_base + lb
            s0 = lb * block_size
            s1 = min(S, s0 + block_size)
            pool_k[blk_base + lb, : s1 - s0] = k_np[s0:s1]
            pool_v[blk_base + lb, : s1 - s0] = v_np[s0:s1]
        blk_base += n_blk

    return (
        mx.array(pool_k),
        mx.array(pool_v),
        mx.array(table, dtype=mx.int32),
        mx.array(lens, dtype=mx.int32),
    )


def _pack_queries(q_seqs: list[mx.array]):
    offsets = [0]
    for q in q_seqs:
        offsets.append(offsets[-1] + int(q.shape[2]))
    total_q = offsets[-1]
    q_pack = mx.concatenate(q_seqs, axis=2) if total_q > 0 else q_seqs[0][:, :, :0, :]
    cu = mx.array(offsets, dtype=mx.int32)
    return q_pack, cu


def _to_batched_padded(q_seqs: list[mx.array]):
    B = len(q_seqs)
    H_q = q_seqs[0].shape[1]
    D = q_seqs[0].shape[3]
    q_lens = [int(q.shape[2]) for q in q_seqs]
    max_q = max(q_lens) if q_lens else 0
    q_b = np.zeros((B, H_q, max_q, D), dtype=np.float16)
    for b, q in enumerate(q_seqs):
        q_np = np.array(q).astype(np.float16)
        q_b[b, :, : q_lens[b], :] = q_np[0]
    return mx.array(q_b), q_lens


def _pack_output_from_batched(out_batched: mx.array, q_lens: list[int]) -> mx.array:
    parts = [out_batched[b:b+1, :, :ql, :] for b, ql in enumerate(q_lens) if ql > 0]
    if not parts:
        H = out_batched.shape[1]
        D = out_batched.shape[3]
        return mx.zeros((1, H, 0, D), dtype=out_batched.dtype)
    return mx.concatenate(parts, axis=2)


def _measure(fn, warmup: int, iters: int) -> float:
    for _ in range(max(0, warmup)):
        out = fn()
        mx.eval(out)
        mx.synchronize()
    ts = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return float(ts[len(ts) // 2])


def _seq_loop_paged(
    q_pack: mx.array,
    cu_q: mx.array,
    pool_k: mx.array,
    pool_v: mx.array,
    table: mx.array,
    lens: mx.array,
    *,
    scale: float,
    causal: bool,
    block_size: int,
):
    cu = [int(x) for x in cu_q.tolist()]
    B = table.shape[0]
    outs = []
    for i in range(B):
        qs, qe = cu[i], cu[i + 1]
        if qe == qs:
            continue
        out_i = flash_attention_paged(
            q_pack[:, :, qs:qe, :],
            pool_k,
            pool_v,
            table[i:i+1, :],
            lens[i:i+1],
            scale=scale,
            causal=causal,
            block_size=block_size,
        )
        outs.append(out_i)
    if not outs:
        H_q, D = q_pack.shape[1], q_pack.shape[3]
        return mx.zeros((1, H_q, 0, D), dtype=q_pack.dtype)
    return mx.concatenate(outs, axis=2)


def run_one(
    scenario: Scenario,
    D: int,
    dtype: mx.Dtype,
    *,
    warmup: int,
    iters: int,
    causal: bool,
    block_size: int,
):
    mx.random.seed(123)

    q_seqs = [mx.random.normal((1, scenario.H_q, ql, D)).astype(dtype) for ql in scenario.q_lens]
    k_seqs = [mx.random.normal((1, scenario.H_kv, kl, D)).astype(dtype) for kl in scenario.kv_lens]
    v_seqs = [mx.random.normal((1, scenario.H_kv, kl, D)).astype(dtype) for kl in scenario.kv_lens]
    mx.eval(*q_seqs, *k_seqs, *v_seqs)

    q_pack, cu_q = _pack_queries(q_seqs)
    q_batched, q_lens = _to_batched_padded(q_seqs)
    pool_k, pool_v, table, lens = _build_pool(k_seqs, v_seqs, block_size)

    scale = 1.0 / math.sqrt(D)

    def run_padded_batched():
        return flash_attention_paged(
            q_batched,
            pool_k,
            pool_v,
            table,
            lens,
            scale=scale,
            causal=causal,
            block_size=block_size,
        )

    def run_varlen():
        return flash_attention_paged_varlen(
            q_pack,
            pool_k,
            pool_v,
            table,
            lens,
            cu_q,
            max_seqlen_q=max(q_lens),
            scale=scale,
            causal=causal,
            block_size=block_size,
        )

    def run_seq_loop():
        return _seq_loop_paged(
            q_pack,
            cu_q,
            pool_k,
            pool_v,
            table,
            lens,
            scale=scale,
            causal=causal,
            block_size=block_size,
        )

    ms_padded = _measure(run_padded_batched, warmup=warmup, iters=iters)
    ms_varlen = _measure(run_varlen, warmup=warmup, iters=iters)
    ms_loop = _measure(run_seq_loop, warmup=warmup, iters=iters)

    out_padded = run_padded_batched()
    out_varlen = run_varlen()
    out_loop = run_seq_loop()
    out_packed_from_padded = _pack_output_from_batched(out_padded, q_lens)
    mx.eval(out_varlen, out_loop, out_packed_from_padded)

    err_varlen_vs_loop = float(mx.max(mx.abs(
        out_varlen.astype(mx.float32) - out_loop.astype(mx.float32)
    )).item())
    err_varlen_vs_padded = float(mx.max(mx.abs(
        out_varlen.astype(mx.float32) - out_packed_from_padded.astype(mx.float32)
    )).item())

    return {
        "scenario": scenario.name,
        "B": scenario.B,
        "H_q": scenario.H_q,
        "H_kv": scenario.H_kv,
        "gqa_ratio": scenario.H_q // scenario.H_kv,
        "D": D,
        "dtype": "bf16" if dtype == mx.bfloat16 else "f16",
        "causal": causal,
        "total_q": int(sum(scenario.q_lens)),
        "max_q": int(max(scenario.q_lens)),
        "max_kv": int(max(scenario.kv_lens)),
        "q_lens": list(scenario.q_lens),
        "kv_lens": list(scenario.kv_lens),
        "padded_paged_ms": ms_padded,
        "paged_varlen_ms": ms_varlen,
        "seq_loop_paged_ms": ms_loop,
        "ratio_varlen_vs_padded": (ms_padded / ms_varlen) if ms_varlen > 0 else float("inf"),
        "ratio_varlen_vs_seq_loop": (ms_loop / ms_varlen) if ms_varlen > 0 else float("inf"),
        "max_err_varlen_vs_seq_loop": err_varlen_vs_loop,
        "max_err_varlen_vs_padded_packed": err_varlen_vs_padded,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--output", type=str, default="notes/paged_varlen_matrix_latest.json")
    args = ap.parse_args()

    scenarios = [
        Scenario(
            name="gqa_b8_hq8_hkv4_hetero",
            B=8,
            H_q=8,
            H_kv=4,
            q_lens=(1, 2, 4, 1, 3, 2, 1, 4),
            kv_lens=(1024, 1536, 2048, 3072, 4096, 8192, 6144, 3584),
        ),
        Scenario(
            name="mqa_b8_hq16_hkv1_hetero",
            B=8,
            H_q=16,
            H_kv=1,
            q_lens=(1, 1, 2, 4, 1, 3, 2, 1),
            kv_lens=(1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192),
        ),
    ]

    rows = []
    dtypes = [mx.float16]

    for sc in scenarios:
        for D in (64, 128):
            for dtype in dtypes:
                row = run_one(
                    sc,
                    D,
                    dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                    causal=True,
                    block_size=args.block_size,
                )
                rows.append(row)
                print(
                    f"{sc.name:>28} D={D:<3} {row['dtype']} "
                    f"varlen={row['paged_varlen_ms']:.3f}ms "
                    f"padded={row['padded_paged_ms']:.3f}ms "
                    f"loop={row['seq_loop_paged_ms']:.3f}ms "
                    f"var/padded={row['ratio_varlen_vs_padded']:.3f}"
                )

    payload = {
        "date": time.strftime("%Y-%m-%d"),
        "mlx_mfa_version": mlx_mfa_version,
        "device": get_device_info(),
        "warmup": args.warmup,
        "iters": args.iters,
        "block_size": args.block_size,
        "rows": rows,
        "notes": [
            "vLLM-oriented paged + packed varlen query matrix.",
            "Compares padded batched paged baseline vs paged_varlen vs per-sequence loop.",
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
