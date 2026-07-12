#!/usr/bin/env python3
"""Safety reproducer for packed-varlen V6 NAX, not a benchmark.

Each process creates exactly one geometry, evaluates the expert kernel twice,
and emits finite/determinism evidence.  A process crash is deliberately left
visible to the caller so shell-level fresh-process campaigns can count it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform

import mlx.core as mx

from mlx_mfa import _ext


TARGET_TOTAL = 35018
TARGET_SEGMENTS = 20
TILE = (32, 32, 2)


def _lengths(segments: int, total: int, heterogeneous: bool) -> list[int]:
    if total < segments:
        raise ValueError("total must be >= segments")
    if not heterogeneous:
        base, remainder = divmod(total, segments)
        return [base] * (segments - 1) + [base + remainder]
    # Strictly positive, intentionally tail-heavy segments with the same total.
    lengths = [1 + (index * 137) % 1700 for index in range(segments)]
    delta = total - sum(lengths)
    if lengths[-1] + delta <= 0:
        raise ValueError("heterogeneous construction underflow")
    lengths[-1] += delta
    return lengths


def _prefix(values: list[int]) -> list[int]:
    out = [0]
    for value in values:
        out.append(out[-1] + value)
    return out


def _tile_offsets(lengths: list[int], bq: int) -> mx.array:
    return mx.array(_prefix([(length + bq - 1) // bq for length in lengths]), dtype=mx.int32)


def _finite(x: mx.array) -> bool:
    return not bool(mx.any(mx.isnan(x))) and not bool(mx.any(mx.isinf(x)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=TARGET_SEGMENTS)
    parser.add_argument("--total", type=int, default=TARGET_TOTAL)
    parser.add_argument("--gqa", type=int, choices=(1, 2, 8), default=2)
    parser.add_argument("--heterogeneous", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    args = parser.parse_args()

    os.environ["MFA_V6_NAX_BQ"] = str(TILE[0])
    os.environ["MFA_V6_NAX_BK"] = str(TILE[1])
    os.environ["MFA_V6_NAX_WM"] = str(TILE[2])
    dtype = mx.float16 if args.dtype == "fp16" else mx.bfloat16
    lengths = _lengths(args.segments, args.total, args.heterogeneous)
    offsets = _prefix(lengths)
    hq, hk, d = args.gqa, 1, 128
    mx.random.seed(20260712)
    q = mx.random.normal((1, hq, args.total, d)).astype(dtype)
    k = mx.random.normal((1, hk, args.total, d)).astype(dtype)
    v = mx.random.normal((1, hk, args.total, d)).astype(dtype)
    cu = mx.array(offsets, dtype=mx.int32)
    tiles = _tile_offsets(lengths, TILE[0])
    mx.eval(q, k, v, cu, tiles)

    def call():
        return _ext.v6_nax_varlen_forward(
            q, k, v, cu, cu, tiles, 1.0 / math.sqrt(d), args.causal
        )[0]

    first = call()
    mx.eval(first)
    second = call()
    mx.eval(second)
    first_finite, second_finite = _finite(first), _finite(second)
    max_abs = float(mx.max(mx.abs(first.astype(mx.float32) - second.astype(mx.float32))))
    print(json.dumps({
        "case": {
            "segments": args.segments,
            "total": args.total,
            "gqa": args.gqa,
            "heterogeneous": args.heterogeneous,
            "causal": args.causal,
            "dtype": args.dtype,
            "tile": TILE,
        },
        "finite": {"first": first_finite, "second": second_finite},
        "run_twice_max_abs": max_abs,
        "byte_identical": max_abs == 0.0,
        "mlx": mx.__version__,
        "platform": platform.platform(),
    }, sort_keys=True))
    if not first_finite or not second_finite or max_abs != 0.0:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
