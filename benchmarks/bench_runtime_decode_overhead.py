#!/usr/bin/env python3
"""Micro-benchmark: legacy inference context vs unified DecodeRuntime."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from mlx_mfa import (
    __version__ as mlx_mfa_version,
    create_decode_runtime,
    create_inference_context,
)


def _time_decode_loop(*, use_runtime: bool, steps: int, scale: float, q_pre, k_pre, v_pre, q_tok, k_tok, v_tok) -> float:
    t0 = time.perf_counter()
    if use_runtime:
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=q_pre.shape[0],
            H_kv=k_pre.shape[1],
            D=q_pre.shape[-1],
            max_seq_len=4096,
        )
        out = rt.prefill(q_pre, k_pre, v_pre, scale=scale)
        mx.eval(out)
        for _ in range(steps):
            out = rt.step(q_tok, k_tok, v_tok, scale=scale)
            mx.eval(out)
    else:
        ctx = create_inference_context(
            backend="dense",
            quantized_kv=False,
            B=q_pre.shape[0],
            H_kv=k_pre.shape[1],
            D=q_pre.shape[-1],
            max_seq_len=4096,
        )
        out = ctx.prefill(q_pre, k_pre, v_pre, scale=scale)
        mx.eval(out)
        for _ in range(steps):
            out = ctx.step(q_tok, k_tok, v_tok, scale=scale)
            mx.eval(out)
    return (time.perf_counter() - t0) * 1000.0


def _time_factory(*, use_runtime: bool, n_iters: int, B: int, H_kv: int, D: int) -> float:
    t0 = time.perf_counter()
    if use_runtime:
        for _ in range(n_iters):
            _ = create_decode_runtime(
                backend="dense",
                quantized_kv=False,
                B=B,
                H_kv=H_kv,
                D=D,
                max_seq_len=4096,
            )
    else:
        for _ in range(n_iters):
            _ = create_inference_context(
                backend="dense",
                quantized_kv=False,
                B=B,
                H_kv=H_kv,
                D=D,
                max_seq_len=4096,
            )
    return (time.perf_counter() - t0) * 1e6 / float(n_iters)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--output",
        type=str,
        default="devnotes/runtime_unification_overhead_latest.json",
    )
    args = parser.parse_args()

    mx.random.seed(42)
    B, H_kv, H_q, N_pre, D = 1, 4, 4, 64, 64
    scale = 1.0 / (D ** 0.5)

    q_pre = mx.random.normal((B, H_q, N_pre, D)).astype(mx.float16)
    k_pre = mx.random.normal((B, H_kv, N_pre, D)).astype(mx.float16)
    v_pre = mx.random.normal((B, H_kv, N_pre, D)).astype(mx.float16)
    q_tok = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
    k_tok = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
    v_tok = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
    mx.eval(q_pre, k_pre, v_pre, q_tok, k_tok, v_tok)

    for _ in range(args.warmup):
        _time_decode_loop(
            use_runtime=False,
            steps=args.steps,
            scale=scale,
            q_pre=q_pre,
            k_pre=k_pre,
            v_pre=v_pre,
            q_tok=q_tok,
            k_tok=k_tok,
            v_tok=v_tok,
        )
        _time_decode_loop(
            use_runtime=True,
            steps=args.steps,
            scale=scale,
            q_pre=q_pre,
            k_pre=k_pre,
            v_pre=v_pre,
            q_tok=q_tok,
            k_tok=k_tok,
            v_tok=v_tok,
        )

    legacy = []
    unified = []
    for _ in range(args.iters):
        legacy.append(
            _time_decode_loop(
                use_runtime=False,
                steps=args.steps,
                scale=scale,
                q_pre=q_pre,
                k_pre=k_pre,
                v_pre=v_pre,
                q_tok=q_tok,
                k_tok=k_tok,
                v_tok=v_tok,
            )
        )
        unified.append(
            _time_decode_loop(
                use_runtime=True,
                steps=args.steps,
                scale=scale,
                q_pre=q_pre,
                k_pre=k_pre,
                v_pre=v_pre,
                q_tok=q_tok,
                k_tok=k_tok,
                v_tok=v_tok,
            )
        )

    legacy_mean = sum(legacy) / len(legacy)
    unified_mean = sum(unified) / len(unified)

    factory_legacy_us = _time_factory(
        use_runtime=False,
        n_iters=300,
        B=B,
        H_kv=H_kv,
        D=D,
    )
    factory_unified_us = _time_factory(
        use_runtime=True,
        n_iters=300,
        B=B,
        H_kv=H_kv,
        D=D,
    )

    out = {
        "date": time.strftime("%Y-%m-%d"),
        "mlx_mfa_version": mlx_mfa_version,
        "shape": {
            "B": B,
            "H_q": H_q,
            "H_kv": H_kv,
            "N_pre": N_pre,
            "steps": args.steps,
            "D": D,
            "dtype": "float16",
        },
        "decode_loop_ms": {
            "legacy_context_mean": legacy_mean,
            "unified_runtime_mean": unified_mean,
            "unified_over_legacy": unified_mean / legacy_mean if legacy_mean else None,
        },
        "factory_overhead_us": {
            "legacy_context": factory_legacy_us,
            "unified_runtime": factory_unified_us,
            "unified_over_legacy": factory_unified_us / factory_legacy_us if factory_legacy_us else None,
        },
        "notes": [
            "Measured in separate process.",
            "Dense decode path only (production default).",
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    raise SystemExit(main())
