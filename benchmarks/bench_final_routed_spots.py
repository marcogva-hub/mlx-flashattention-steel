#!/usr/bin/env python3
"""Public routed-path spot checks with blocking two-arm fingerprints.

The timed SDPA arm always receives the same q/k/v dtype as the routed arm.
Float32 is used only for the untimed correctness oracle.  A timing sample is
refused unless both public and baseline terminals are captured and validated.
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
from mlx_mfa import flash_attention, flash_attention_gna, flash_attention_sparse
from mlx_mfa.attention import make_sliding_window_mask
from benchmarks.bench_gna_nax import make_gna_mask


SESSIONS = 5
WARMUPS = 2
DISPATCHES_PER_SAMPLE = 20
CORRECTION_COS_MIN = 0.999


def evaluate(value):
    mx.eval(*value) if isinstance(value, (tuple, list)) else mx.eval(value)
    mx.synchronize()


def stats(samples):
    return {"median_ms": statistics.median(samples), "p95_ms": float(np.percentile(samples, 95)),
            "samples_ms": samples, "n": len(samples)}


def cosine(a, b):
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    x = mx.sum(af * bf) / mx.sqrt(mx.sum(af * af) * mx.sum(bf * bf))
    mx.eval(x)
    return float(x.item())


def _dtype_name(value):
    return str(value.dtype)


def _terminal(trace):
    return [item for item in trace if not item[1].startswith("[reentrant]")]


def _require_fingerprints(*, label, input_dtype, public_trace, baseline_trace,
                          expected_public, public_output, baseline_output,
                          oracle_output):
    """Validate both arms before allowing the caller to report timing."""
    public_terminal = _terminal(public_trace)
    baseline_terminal = _terminal(baseline_trace)
    failures = []
    if not public_terminal or public_terminal[-1][0] != expected_public:
        failures.append(
            f"public terminal must be {expected_public!r}, got {public_trace!r}")
    if not baseline_terminal or baseline_terminal[-1][0] != "sdpa":
        failures.append(
            f"baseline terminal must be 'sdpa', got {baseline_trace!r}")
    if _dtype_name(public_output) != input_dtype:
        failures.append(
            f"public output dtype {_dtype_name(public_output)} != input {input_dtype}")
    if _dtype_name(baseline_output) != input_dtype:
        failures.append(
            f"baseline output dtype {_dtype_name(baseline_output)} != input {input_dtype}")

    checks = {}
    for arm_name, output in (("public", public_output), ("baseline", baseline_output)):
        check = {
            "cos_vs_fp32_oracle": cosine(output, oracle_output),
            "finite": bool(mx.all(mx.isfinite(output)).item()),
        }
        checks[arm_name] = check
        if check["cos_vs_fp32_oracle"] < CORRECTION_COS_MIN or not check["finite"]:
            failures.append(f"{arm_name} correction failed: {check}")

    public_vs_baseline = float(mx.max(mx.abs(
        public_output.astype(mx.float32) - baseline_output.astype(mx.float32)
    )).item())
    if public_vs_baseline == 0.0:
        failures.append(
            "public and baseline outputs are byte-identical; distinct-path "
            "engagement is not corroborated")
    if failures:
        raise RuntimeError(
            f"{label}: refusing benchmark ratio because fingerprint/correction "
            f"is incomplete: {'; '.join(failures)}")
    return {
        "input_dtype": input_dtype,
        "public": {
            "trace": public_trace,
            "terminal": public_terminal[-1],
            "output_dtype": _dtype_name(public_output),
        },
        "baseline": {
            "trace": baseline_trace,
            "terminal": baseline_terminal[-1],
            "output_dtype": _dtype_name(baseline_output),
            "framework_path": "mlx-mfa backend=sdpa -> mx.fast.scaled_dot_product_attention",
        },
        "public_vs_baseline_max_abs": public_vs_baseline,
        "correction": checks,
    }


def time_arm(fn):
    for _ in range(WARMUPS):
        for _ in range(DISPATCHES_PER_SAMPLE):
            evaluate(fn())
    values = []
    for _ in range(SESSIONS):
        started = time.perf_counter()
        for _ in range(DISPATCHES_PER_SAMPLE):
            evaluate(fn())
        values.append((time.perf_counter() - started) * 1000.0 / DISPATCHES_PER_SAMPLE)
    return stats(values)


def run(kind, arm):
    d, n, dtype = 128, 4096, mx.float16
    scale = 1.0 / math.sqrt(d)
    mx.random.seed(20260713 + len(kind))
    if kind == "dense":
        q = mx.random.normal((1, 4, n, d)).astype(dtype)
        k = mx.random.normal((1, 4, n, d)).astype(dtype)
        v = mx.random.normal((1, 4, n, d)).astype(dtype)
        def public():
            return flash_attention(q, k, v, scale=scale, causal=False)
        expected = "nax_dense"
        def baseline():
            return flash_attention(q, k, v, scale=scale, causal=False, backend="sdpa")
        oracle = lambda: mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=scale)
        label = "dense_d128_n4096"
    elif kind == "sparse":
        q = mx.random.normal((1, 1, n, d)).astype(dtype)
        k = mx.random.normal((1, 1, n, d)).astype(dtype)
        v = mx.random.normal((1, 1, n, d)).astype(dtype)
        block_mask = make_sliding_window_mask(n, 128, head_dim=d)
        block_mask = block_mask.astype(mx.bool_)
        expanded = mx.repeat(mx.repeat(block_mask.astype(mx.float32), 32, -2), 32, -1)
        oracle_bias = mx.where(
            expanded, mx.array(0.0, mx.float32), mx.array(-1e30, mx.float32))
        bias = oracle_bias.astype(dtype)
        def public():
            return flash_attention_sparse(q, k, v, block_mask, scale=scale, causal=False)
        expected = "v6nax_sparse"
        def baseline():
            return flash_attention(
                q, k, v, scale=scale, causal=False, attn_bias=bias, backend="sdpa")
        oracle = lambda: mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            scale=scale, mask=oracle_bias)
        label = "sparse_bt32_d128_n4096"
    elif kind == "gna":
        seq_shape = (4, 32, 32)
        q = mx.random.normal((1, 1, n, d)).astype(dtype)
        k = mx.random.normal((1, 1, n, d)).astype(dtype)
        v = mx.random.normal((1, 1, n, d)).astype(dtype)
        window, stride = (1, 7, 7), (1, 1, 1)
        oracle_mask = make_gna_mask(seq_shape, window, stride).astype(mx.float32)
        mask = oracle_mask.astype(dtype)
        def public():
            return flash_attention_gna(q, k, v, seq_shape, window, stride, scale=scale)
        expected = "gna_v6nax"
        def baseline():
            return flash_attention(
                q, k, v, scale=scale, causal=False, attn_bias=mask, backend="sdpa")
        oracle = lambda: mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            scale=scale, mask=oracle_mask)
        label = "gna_d128_3d_n4096"
    else:
        raise ValueError(kind)

    with dtrace.capture() as public_trace:
        public_probe = public()
        evaluate(public_probe)
    with dtrace.capture() as baseline_trace:
        baseline_probe = baseline()
        evaluate(baseline_probe)
    oracle_probe = oracle()
    evaluate((public_probe, baseline_probe, oracle_probe))
    fingerprint = _require_fingerprints(
        label=label,
        input_dtype=_dtype_name(q),
        public_trace=list(public_trace),
        baseline_trace=list(baseline_trace),
        expected_public=expected,
        public_output=public_probe,
        baseline_output=baseline_probe,
        oracle_output=oracle_probe,
    )

    timing = time_arm(public if arm == "public" else baseline)
    print(
        f"{label} arm={arm}: median={timing['median_ms']:.3f}ms "
        f"public={fingerprint['public']['terminal']} "
        f"baseline={fingerprint['baseline']['terminal']} "
        f"dtype={fingerprint['input_dtype']}",
        flush=True,
    )
    return {"label": label, "kind": kind, "shape": {"N": n, "D": d},
            "arm": arm, "which_binary": fingerprint,
            "correction": fingerprint["correction"], "timing": timing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("dense", "sparse", "gna"), required=True)
    parser.add_argument("--arm", choices=("public", "sdpa"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = run(args.kind, args.arm)
    payload = {"schema": "mlx-mfa.final-routed-spots.v2",
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "mlx": importlib.metadata.version("mlx"), "platform": platform.platform(),
               "arm": args.arm,
               "method": {"sessions": SESSIONS, "warmups": WARMUPS,
                          "samples_per_session": 1,
                          "dispatches_per_sample": DISPATCHES_PER_SAMPLE},
               "row": row}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
