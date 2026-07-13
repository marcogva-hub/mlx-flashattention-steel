#!/usr/bin/env python3
"""One process-isolated sparse-routing benchmark arm.

The script refuses to emit timing unless the public arm terminates in
``v6nax_sparse``, the baseline terminates in ``sdpa``, both use the requested
dtype, and both are correct against an untimed fp32 masked-attention oracle.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import flash_attention, flash_attention_sparse
from mlx_mfa.attention import make_sliding_window_mask


SESSIONS = 5
WARMUPS = 2
DISPATCHES_PER_SAMPLE = 20
CORRECTION_COS_MIN = 0.999
BLOCK_TILE = 32


def evaluate(value):
    mx.eval(*value) if isinstance(value, (tuple, list)) else mx.eval(value)
    mx.synchronize()


def cosine(a, b):
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    value = mx.sum(af * bf) / mx.sqrt(mx.sum(af * af) * mx.sum(bf * bf))
    mx.eval(value)
    return float(value.item())


def stats(samples):
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
        "n": len(samples),
    }


def terminal(trace):
    return [item for item in trace if not item[1].startswith("[reentrant]")]


def make_random_mask(n, density, seed, causal):
    rng = np.random.default_rng(seed)
    blocks = n // BLOCK_TILE
    if causal:
        # Reserve the diagonal inside the requested density budget. This keeps
        # the *actual* block density at or below the public causal gate's 0.30
        # ceiling instead of accidentally crossing it after diagonal repair.
        mask = np.eye(blocks, dtype=np.bool_)
        target = max(blocks, int(np.floor(density * blocks * blocks)))
        candidates = np.flatnonzero(~mask.reshape(-1))
        chosen = rng.choice(candidates, size=target - blocks, replace=False)
        mask.reshape(-1)[chosen] = True
    else:
        mask = rng.random((blocks, blocks)) < density
    # Every query block needs at least one active key block for a finite oracle.
    empty = np.flatnonzero(~mask.any(axis=1))
    mask[empty, np.minimum(empty, blocks - 1)] = True
    return mx.array(mask)


def token_bias(block_mask, n, dtype, causal):
    expanded = mx.repeat(mx.repeat(block_mask, BLOCK_TILE, -2), BLOCK_TILE, -1)
    expanded = expanded[..., :n, :n]
    if causal:
        rows = mx.arange(n)[:, None]
        cols = mx.arange(n)[None, :]
        expanded = expanded & (cols <= rows)
    fp32 = mx.where(
        expanded,
        mx.array(0.0, dtype=mx.float32),
        mx.array(-1e30, dtype=mx.float32),
    )
    return fp32, fp32.astype(dtype)


def build_cell(args):
    dtype = mx.float16 if args.dtype == "fp16" else mx.bfloat16
    scale = 1.0 / math.sqrt(args.D)
    mx.random.seed(args.seed)
    q = mx.random.normal((args.B, args.H, args.N, args.D)).astype(dtype)
    k = mx.random.normal((args.B, args.H, args.N, args.D)).astype(dtype)
    v = mx.random.normal((args.B, args.H, args.N, args.D)).astype(dtype)

    if args.mask_kind == "sliding":
        block_mask = make_sliding_window_mask(
            args.N, args.window, head_dim=args.D, causal=False
        ).astype(mx.bool_)
    else:
        block_mask = make_random_mask(
            args.N, args.density, args.seed + 17, args.causal
        )
    oracle_bias, timed_bias = token_bias(block_mask, args.N, dtype, args.causal)

    def public():
        return flash_attention_sparse(
            q, k, v, block_mask, scale=scale, causal=args.causal
        )

    def baseline():
        # Causality is already represented in the additive mask. This keeps the
        # baseline a single, fingerprintable same-dtype SDPA call.
        return flash_attention(
            q,
            k,
            v,
            scale=scale,
            causal=False,
            attn_bias=timed_bias,
            backend="sdpa",
        )

    def oracle():
        return mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=scale,
            mask=oracle_bias,
        )

    return q, block_mask, public, baseline, oracle


def preflight(args, q, block_mask, public, baseline, oracle):
    with dtrace.capture() as public_trace:
        public_out = public()
        evaluate(public_out)
    with dtrace.capture() as baseline_trace:
        baseline_out = baseline()
        evaluate(baseline_out)
    oracle_out = oracle()
    public_again = public()
    evaluate((oracle_out, public_again))

    public_terminal = terminal(list(public_trace))
    baseline_terminal = terminal(list(baseline_trace))
    failures = []
    if not public_terminal or public_terminal[-1][0] != "v6nax_sparse":
        failures.append(f"public terminal={public_terminal!r}")
    if not baseline_terminal or baseline_terminal[-1][0] != "sdpa":
        failures.append(f"baseline terminal={baseline_terminal!r}")
    expected_dtype = str(q.dtype)
    for name, output in (("public", public_out), ("baseline", baseline_out)):
        if str(output.dtype) != expected_dtype:
            failures.append(f"{name} dtype={output.dtype}, expected={expected_dtype}")

    correction = {
        "public_cos_vs_fp32": cosine(public_out, oracle_out),
        "baseline_cos_vs_fp32": cosine(baseline_out, oracle_out),
        "public_finite": bool(mx.all(mx.isfinite(public_out)).item()),
        "baseline_finite": bool(mx.all(mx.isfinite(baseline_out)).item()),
        "public_run_twice_max_abs": float(
            mx.max(mx.abs(public_out.astype(mx.float32) - public_again.astype(mx.float32))).item()
        ),
        "public_vs_baseline_max_abs": float(
            mx.max(mx.abs(public_out.astype(mx.float32) - baseline_out.astype(mx.float32))).item()
        ),
    }
    if correction["public_cos_vs_fp32"] < CORRECTION_COS_MIN:
        failures.append(f"public cos={correction['public_cos_vs_fp32']}")
    if correction["baseline_cos_vs_fp32"] < CORRECTION_COS_MIN:
        failures.append(f"baseline cos={correction['baseline_cos_vs_fp32']}")
    if not correction["public_finite"] or not correction["baseline_finite"]:
        failures.append("non-finite output")
    if correction["public_run_twice_max_abs"] != 0.0:
        failures.append(
            f"public run-twice delta={correction['public_run_twice_max_abs']}"
        )
    if correction["public_vs_baseline_max_abs"] == 0.0:
        failures.append("distinct public/SDPA paths produced byte-identical output")
    if failures:
        raise RuntimeError(
            "refusing sparse gate benchmark without engagement/correction: "
            + "; ".join(failures)
        )

    return {
        "input_dtype": expected_dtype,
        "mask_dtype": str(block_mask.dtype),
        "public": {"trace": list(public_trace), "terminal": public_terminal[-1]},
        "baseline": {
            "trace": list(baseline_trace),
            "terminal": baseline_terminal[-1],
            "framework_path": "mlx-mfa backend=sdpa -> mx.fast.scaled_dot_product_attention",
        },
        "correction": correction,
    }


def time_arm(fn):
    for _ in range(WARMUPS):
        for _ in range(DISPATCHES_PER_SAMPLE):
            evaluate(fn())
    samples = []
    for _ in range(SESSIONS):
        started = time.perf_counter()
        for _ in range(DISPATCHES_PER_SAMPLE):
            evaluate(fn())
        samples.append((time.perf_counter() - started) * 1000.0 / DISPATCHES_PER_SAMPLE)
    return stats(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("public", "sdpa"), required=True)
    parser.add_argument("--mask-kind", choices=("sliding", "random"), required=True)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--density", type=float, default=0.15)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.N % BLOCK_TILE:
        raise ValueError("N must be divisible by BT=32")

    q, block_mask, public, baseline, oracle = build_cell(args)
    fingerprint = preflight(args, q, block_mask, public, baseline, oracle)
    timing = time_arm(public if args.arm == "public" else baseline)
    row = {
        "cell": {
            "mask_kind": args.mask_kind,
            "window": args.window if args.mask_kind == "sliding" else None,
            "requested_density": args.density if args.mask_kind == "random" else None,
            "actual_block_density": float(mx.mean(block_mask.astype(mx.float32)).item()),
            "causal": args.causal,
            "B": args.B,
            "H": args.H,
            "BH": args.B * args.H,
            "N": args.N,
            "D": args.D,
            "dtype": args.dtype,
        },
        "arm": args.arm,
        "which_binary": fingerprint,
        "timing": timing,
    }
    payload = {
        "schema": "mlx-mfa.sparse-gate-remap.arm.v1",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "mlx": importlib.metadata.version("mlx"),
        "platform": platform.platform(),
        "method": {
            "sessions": SESSIONS,
            "warmups": WARMUPS,
            "dispatches_per_sample": DISPATCHES_PER_SAMPLE,
            "process_isolation": "one fresh process per arm/order",
        },
        "row": row,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"{args.arm} {row['cell']} median={timing['median_ms']:.6f}ms "
        f"public={fingerprint['public']['terminal'][0]} "
        f"baseline={fingerprint['baseline']['terminal'][0]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
