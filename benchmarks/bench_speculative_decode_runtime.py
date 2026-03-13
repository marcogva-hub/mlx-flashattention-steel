#!/usr/bin/env python3
"""Speculative decode runtime benchmark matrix.

Compares:
  1) manual helper-level speculative flow (verify + caller-side accept/reject),
  2) runtime-integrated flow via DecodeRuntime.speculative_step().

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
import numpy as np

from mlx_mfa import (
    __version__ as mlx_mfa_version,
    create_decode_runtime,
    get_device_info,
)


@dataclass(frozen=True)
class SpecScenario:
    name: str
    backend: str
    D: int
    H_q: int
    H_kv: int
    cache_len: int
    n_draft: int
    seq_id: int = 0


@dataclass(frozen=True)
class AcceptMode:
    name: str
    delta: float
    use_draft_logprobs: bool


def _eval_any(x):
    if isinstance(x, dict):
        for v in x.values():
            _eval_any(v)
        return
    if isinstance(x, (tuple, list)):
        for v in x:
            _eval_any(v)
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


def _accepted_prefix_lens(mask: mx.array) -> mx.array:
    lens = []
    for row in mask.tolist():
        n = 0
        for flag in row:
            if bool(flag):
                n += 1
            else:
                break
        lens.append(n)
    return mx.array(lens, dtype=mx.int32)


def _manual_speculative_step(
    rt,
    q_target: mx.array,
    draft_ids: mx.array,
    *,
    accept_delta: float,
    draft_logprobs: mx.array | None,
    seq_id: int | None,
):
    kwargs = {}
    if seq_id is not None:
        kwargs["seq_id"] = seq_id

    out, lse, target_logprobs = rt.speculative_verify(q_target, draft_ids, **kwargs)
    if draft_logprobs is None:
        accept_mask = target_logprobs >= float(accept_delta)
    else:
        accept_mask = (
            target_logprobs.astype(mx.float32)
            - draft_logprobs.astype(mx.float32)
        ) >= float(accept_delta)

    accepted_prefix_lens = _accepted_prefix_lens(accept_mask)
    token_idx = mx.arange(int(draft_ids.shape[1]), dtype=mx.int32)[None, :]
    prefix_mask = token_idx < accepted_prefix_lens[:, None]
    minus_one = mx.full(draft_ids.shape, -1, dtype=draft_ids.dtype)
    accepted_ids = mx.where(prefix_mask, draft_ids, minus_one)
    rejected_ids = mx.where(prefix_mask, minus_one, draft_ids)

    return {
        "out": out,
        "lse": lse,
        "target_logprobs": target_logprobs,
        "accept_mask": accept_mask,
        "accepted_prefix_lens": accepted_prefix_lens,
        "accepted_ids": accepted_ids,
        "rejected_ids": rejected_ids,
    }


def _build_runtime(sc: SpecScenario):
    scale = 1.0 / math.sqrt(sc.D)
    if sc.backend == "dense":
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_q=sc.H_q,
            H_kv=sc.H_kv,
            D=sc.D,
            max_seq_len=max(4096, sc.cache_len + 32),
        )
        mx.random.seed(5100 + sc.D)
        q_pre = mx.random.normal((1, sc.H_q, sc.cache_len, sc.D)).astype(mx.float16)
        mx.random.seed(5200 + sc.D)
        k_pre = mx.random.normal((1, sc.H_kv, sc.cache_len, sc.D)).astype(mx.float16)
        mx.random.seed(5300 + sc.D)
        v_pre = mx.random.normal((1, sc.H_kv, sc.cache_len, sc.D)).astype(mx.float16)
        rt.prefill(q_pre, k_pre, v_pre, scale=scale, causal=True)
        seq_id = None
    elif sc.backend == "paged":
        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            query_layout="batched",
            quantized_kv=False,
            B=1,
            H_q=sc.H_q,
            H_kv=sc.H_kv,
            D=sc.D,
            num_blocks=max(256, (sc.cache_len // 16) * 4),
            block_size=16,
            max_seq_len=max(4096, sc.cache_len + 32),
        )
        mx.random.seed(6100 + sc.D)
        q_pre = mx.random.normal((1, sc.H_q, sc.cache_len, sc.D)).astype(mx.float16)
        mx.random.seed(6200 + sc.D)
        k_pre = mx.random.normal((1, sc.H_kv, sc.cache_len, sc.D)).astype(mx.float16)
        mx.random.seed(6300 + sc.D)
        v_pre = mx.random.normal((1, sc.H_kv, sc.cache_len, sc.D)).astype(mx.float16)
        rt.prefill(q_pre, k_pre, v_pre, scale=scale, causal=True, seq_id=sc.seq_id)
        seq_id = sc.seq_id
    else:
        raise ValueError(f"unsupported backend: {sc.backend}")

    mx.random.seed(7100 + sc.D)
    q_target = mx.random.normal((1, sc.H_q, sc.n_draft, sc.D)).astype(mx.float16)
    draft_ids = mx.array([list(range(sc.n_draft))], dtype=mx.int32)
    mx.eval(q_target, draft_ids)

    return rt, q_target, draft_ids, seq_id


def _build_draft_logprobs(rt, q_target, draft_ids, mode: AcceptMode, seq_id: int | None):
    if not mode.use_draft_logprobs:
        return None

    kwargs = {}
    if seq_id is not None:
        kwargs["seq_id"] = seq_id
    _, _, lp = rt.speculative_verify(q_target, draft_ids, **kwargs)
    lp_np = np.array(lp.astype(mx.float32))
    n = lp_np.shape[1]
    cut = max(1, n // 2)

    # First segment easy accept; tail hard reject.
    draft_lp_np = lp_np.copy()
    draft_lp_np[:, :cut] = lp_np[:, :cut] - 0.5
    draft_lp_np[:, cut:] = lp_np[:, cut:] + 5.0
    return mx.array(draft_lp_np.astype(np.float32))


def run_case(sc: SpecScenario, mode: AcceptMode, warmup: int, iters: int) -> dict:
    rt, q_target, draft_ids, seq_id = _build_runtime(sc)
    draft_logprobs = _build_draft_logprobs(rt, q_target, draft_ids, mode, seq_id)

    def manual_fn():
        return _manual_speculative_step(
            rt,
            q_target,
            draft_ids,
            accept_delta=mode.delta,
            draft_logprobs=draft_logprobs,
            seq_id=seq_id,
        )

    def runtime_fn():
        kwargs = {}
        if seq_id is not None:
            kwargs["seq_id"] = seq_id
        return rt.speculative_step(
            q_target,
            draft_ids,
            draft_logprobs=draft_logprobs,
            accept_logprob_delta=mode.delta,
            **kwargs,
        )

    ms_manual = _measure_ms(manual_fn, warmup=warmup, iters=iters)
    ms_runtime = _measure_ms(runtime_fn, warmup=warmup, iters=iters)

    out_manual = manual_fn()
    out_runtime = runtime_fn()
    _eval_any(out_manual)
    _eval_any(out_runtime)

    accepted_manual = np.array(out_manual["accepted_prefix_lens"]).astype(np.int32)
    accepted_runtime = np.array(out_runtime["accepted_prefix_lens"]).astype(np.int32)
    mask_manual = np.array(out_manual["accept_mask"]).astype(np.int32)
    mask_runtime = np.array(out_runtime["accept_mask"]).astype(np.int32)

    return {
        "scenario": sc.name,
        "backend": sc.backend,
        "D": sc.D,
        "cache_len": sc.cache_len,
        "n_draft": sc.n_draft,
        "accept_mode": mode.name,
        "manual_ms": ms_manual,
        "runtime_ms": ms_runtime,
        "ratio_runtime_vs_manual": (ms_manual / ms_runtime) if ms_runtime > 0 else float("inf"),
        "accepted_prefix_lens": accepted_runtime.tolist(),
        "accept_mask": mask_runtime.tolist(),
        "manual_runtime_prefix_match": bool(np.array_equal(accepted_manual, accepted_runtime)),
        "manual_runtime_mask_match": bool(np.array_equal(mask_manual, mask_runtime)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark runtime speculative decode flow")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--output",
        type=str,
        default="devnotes/speculative_decode_runtime_matrix_latest.json",
    )
    args = parser.parse_args()

    scenarios = [
        SpecScenario("dense_short", "dense", D=64, H_q=8, H_kv=4, cache_len=1024, n_draft=4),
        SpecScenario("dense_micro", "dense", D=128, H_q=8, H_kv=4, cache_len=2048, n_draft=8),
        SpecScenario("paged_short", "paged", D=64, H_q=8, H_kv=4, cache_len=1024, n_draft=4, seq_id=7),
    ]
    modes = [
        AcceptMode("full_accept", -1e9, False),
        AcceptMode("partial_accept", 0.0, True),
        AcceptMode("reject_all", 1e6, False),
    ]

    results = []
    for sc in scenarios:
        for mode in modes:
            results.append(run_case(sc, mode, args.warmup, args.iters))

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
    main()
