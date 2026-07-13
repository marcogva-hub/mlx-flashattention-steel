"""Correctness and isolated timing harness for sparse forward LSE.

The scalar arm uses the same BT32 mask expanded 2x2 to BT16.  That makes it
an independent implementation with identical block semantics, while keeping
the old scalar LSE generator selectable.  Each invocation is one fresh Python
process; use separate invocations for the two arm orders and sessions.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import mlx.core as mx
import numpy as np

from mlx_mfa import _dispatch_trace
from mlx_mfa.lcsa_nax import sparse_attention_nax, sparse_attention_nax_with_lse


def _make_inputs(n: int, d: int, dtype, density: float, seed: int, source_bt: int):
    mx.random.seed(seed)
    q = (mx.random.uniform(-1, 1, (1, 2, n, d)) * 0.1).astype(dtype)
    k = (mx.random.uniform(-1, 1, (1, 2, n, d)) * 0.1).astype(dtype)
    v = (mx.random.uniform(-1, 1, (1, 2, n, d)) * 0.1).astype(dtype)
    if source_bt not in (32, 64):
        raise ValueError("source_bt must be 32 or 64")
    nb = n // source_bt
    rng = np.random.default_rng(seed)
    mask = rng.random((nb, nb)) < density
    # Keep every Q tile non-empty so finite LSE comparisons are meaningful.
    mask[np.arange(nb), np.arange(nb)] = True
    source_mask = mx.array(mask)
    if source_bt == 64:
        source_mask = mx.repeat(mx.repeat(source_mask, 2, axis=-2), 2, axis=-1)
    return q, k, v, source_mask


def _scalar_mask(mask32: mx.array) -> mx.array:
    return mx.repeat(mx.repeat(mask32, 2, axis=-2), 2, axis=-1)


def _call(arm: str, q, k, v, mask32, scale: float, causal: bool, scalar_bt: int = 16):
    if arm == "nax_lse":
        return sparse_attention_nax_with_lse(
            q, k, v, mask32, block_tile=32, scale=scale, causal=causal
        )
    if arm == "scalar_lse":
        scalar_mask = mask32 if scalar_bt == 32 else _scalar_mask(mask32)
        return sparse_attention_nax_with_lse(
            q, k, v, scalar_mask, block_tile=scalar_bt,
            scale=scale, causal=causal
        )
    if arm == "nax_no_lse":
        return sparse_attention_nax(
            q, k, v, mask32, block_tile=32, scale=scale, causal=causal
        )
    raise ValueError(f"unknown arm: {arm}")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    return 1.0 if denom == 0 else float(np.dot(af, bf) / denom)


def _timed(
    arm: str, q, k, v, mask, scale, causal, repeats: int, runs: int, scalar_bt: int
):
    for _ in range(2):
        out = _call(arm, q, k, v, mask, scale, causal, scalar_bt)
        mx.eval(*out) if isinstance(out, tuple) else mx.eval(out)
    mx.synchronize()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        for _ in range(repeats):
            out = _call(arm, q, k, v, mask, scale, causal, scalar_bt)
            mx.eval(*out) if isinstance(out, tuple) else mx.eval(out)
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0 / repeats)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("correctness", "perf"), required=True)
    parser.add_argument("--arm", choices=("nax_lse", "scalar_lse", "nax_no_lse"))
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--density", type=float, default=0.1)
    parser.add_argument("--source-bt", type=int, choices=(32, 64), default=32)
    parser.add_argument("--scalar-bt", type=int, choices=(16, 32), default=16)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    dtype = mx.float16 if args.dtype == "fp16" else mx.bfloat16
    q, k, v, mask = _make_inputs(
        args.n, args.d, dtype, args.density, args.seed, args.source_bt
    )
    scale = 1.0 / math.sqrt(args.d)
    mx.eval(q, k, v, mask)

    if args.mode == "correctness":
        if args.arm not in ("nax_lse", "scalar_lse"):
            raise SystemExit("correctness requires --arm nax_lse or scalar_lse")
        with _dispatch_trace.capture() as trace:
            out = _call(args.arm, q, k, v, mask, scale, args.causal, args.scalar_bt)
            mx.eval(*out)
        o, l = out
        scalar_out = _call(
            "scalar_lse", q, k, v, mask, scale, args.causal, args.scalar_bt
        )
        mx.eval(*scalar_out)
        o_np = np.array(o.astype(mx.float32))
        l_np = np.array(l)
        ref_o = np.array(scalar_out[0].astype(mx.float32))
        ref_l = np.array(scalar_out[1])
        finite = np.isfinite(ref_l)
        result = {
            "mode": "correctness",
            "arm": args.arm,
            "n": args.n,
            "d": args.d,
            "dtype": args.dtype,
            "density": args.density,
            "source_bt": args.source_bt,
            "causal": args.causal,
            "trace": trace,
            "cos_o_vs_scalar": _cosine(o_np, ref_o),
            "max_abs_o_vs_scalar": float(np.max(np.abs(o_np - ref_o))),
            "max_abs_lse_vs_scalar_finite": float(
                np.max(np.abs(l_np[finite] - ref_l[finite]))
            ),
            "all_finite_lse_match": bool(
                np.array_equal(np.isneginf(l_np), np.isneginf(ref_l))
            ),
        }
        if result["cos_o_vs_scalar"] < 0.999 or not result["all_finite_lse_match"]:
            raise AssertionError(json.dumps(result, indent=2))
    else:
        if args.arm is None:
            raise SystemExit("perf requires --arm")
        with _dispatch_trace.capture() as trace:
            samples = _timed(
                args.arm, q, k, v, mask, scale, args.causal,
                args.repeats, args.runs, args.scalar_bt,
            )
        result = {
            "mode": "perf",
            "arm": args.arm,
            "n": args.n,
            "d": args.d,
            "dtype": args.dtype,
            "density": args.density,
            "source_bt": args.source_bt,
            "causal": args.causal,
            "repeats_per_sample": args.repeats,
            "runs": args.runs,
            "samples_ms": samples,
            "median_ms": float(np.median(samples)),
            "trace": trace,
        }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
