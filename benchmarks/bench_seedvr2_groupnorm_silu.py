#!/usr/bin/env python3
"""Probe a fused per-frame GroupNorm+SiLU kernel on SeedVR2 VAE shapes."""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
from mlx import nn


CASES = {
    "d512_t3_h108_w132": (1, 3, 108, 132, 512),
    "d128_t5_h432_w528": (1, 5, 432, 528, 128),
}
GROUPS = 32
EPS = 1e-6


def _eval(value: Any) -> None:
    if isinstance(value, mx.array):
        mx.eval(value)
    else:
        mx.eval(*value)
    mx.synchronize()


def _stats(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(statistics.median(samples)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(statistics.mean(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "sample_count": len(samples),
        "samples_ms": samples,
    }


def _time_arm(
    fn: Callable[[], mx.array], warmup: int, sessions: int, cooldown_s: float
) -> dict[str, Any]:
    for _ in range(warmup):
        _eval(fn())
    samples: list[float] = []
    for session in range(sessions):
        mx.synchronize()
        start = time.perf_counter()
        _eval(fn())
        samples.append((time.perf_counter() - start) * 1000.0)
        if cooldown_s > 0 and session + 1 < sessions:
            time.sleep(cooldown_s)
    return _stats(samples)


@functools.lru_cache(maxsize=8)
def _kernel(height: int, width: int, channels: int):
    group_size = channels // GROUPS
    elements_per_group = height * width * group_size
    elements_per_frame = height * width * channels
    source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint simd_id = simdgroup_index_in_threadgroup;
        uint workgroup = threadgroup_position_in_grid.x;
        uint frame = workgroup / {GROUPS}u;
        uint group = workgroup % {GROUPS}u;

        threadgroup float partial_sum[8];
        threadgroup float partial_sq[8];
        threadgroup float stats[2];

        float local_sum = 0.0f;
        float local_sq = 0.0f;
        for (uint i = tid; i < {elements_per_group}u; i += 256u) {{
            uint pixel = i / {group_size}u;
            uint channel_in_group = i % {group_size}u;
            uint channel = group * {group_size}u + channel_in_group;
            uint offset = frame * {elements_per_frame}u + pixel * {channels}u + channel;
            float value = float(x[offset]);
            local_sum += value;
            local_sq += value * value;
        }}
        local_sum = simd_sum(local_sum);
        local_sq = simd_sum(local_sq);
        if (lane == 0u) {{
            partial_sum[simd_id] = local_sum;
            partial_sq[simd_id] = local_sq;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u) {{
            float sum_value = lane < 8u ? partial_sum[lane] : 0.0f;
            float sq_value = lane < 8u ? partial_sq[lane] : 0.0f;
            sum_value = simd_sum(sum_value);
            sq_value = simd_sum(sq_value);
            if (lane == 0u) {{
                float mean = sum_value / float({elements_per_group});
                float variance = metal::max(
                    sq_value / float({elements_per_group}) - mean * mean, 0.0f);
                stats[0] = mean;
                stats[1] = metal::rsqrt(variance + {EPS}f);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float mean = stats[0];
        float inv_std = stats[1];
        for (uint i = tid; i < {elements_per_group}u; i += 256u) {{
            uint pixel = i / {group_size}u;
            uint channel_in_group = i % {group_size}u;
            uint channel = group * {group_size}u + channel_in_group;
            uint offset = frame * {elements_per_frame}u + pixel * {channels}u + channel;
            float value = (float(x[offset]) - mean) * inv_std;
            value = value * float(weight[channel]) + float(bias[channel]);
            value = value / (1.0f + metal::exp(-value));
            out[offset] = T(value);
        }}
    """
    return mx.fast.metal_kernel(
        name=f"seedvr2_groupnorm_silu_h{height}_w{width}_c{channels}",
        input_names=["x", "weight", "bias"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def fused_groupnorm_silu(
    x: mx.array, weight: mx.array, bias: mx.array
) -> mx.array:
    batch, frames, height, width, channels = x.shape
    if batch != 1 or channels % GROUPS:
        raise ValueError("probe supports B=1 and channels divisible by 32")
    kernel = _kernel(int(height), int(width), int(channels))
    (out,) = kernel(
        inputs=[x, weight, bias],
        template=[("T", x.dtype)],
        grid=(int(frames) * GROUPS * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


def cosine_and_max_abs(actual: mx.array, reference: mx.array) -> tuple[float, float]:
    af = actual.astype(mx.float32)
    rf = reference.astype(mx.float32)
    dot = mx.sum(af * rf)
    denominator = mx.sqrt(mx.sum(af * af) * mx.sum(rf * rf))
    max_abs = mx.max(mx.abs(af - rf))
    mx.eval(dot, denominator, max_abs)
    return float(dot.item()) / float(denominator.item()), float(max_abs.item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--order", choices=("baseline-target", "target-baseline"), default="baseline-target"
    )
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    results: dict[str, Any] = {}
    for index, (name, shape) in enumerate(CASES.items()):
        mx.random.seed(20260712 + index)
        x = (mx.random.normal(shape) * 0.1).astype(mx.float16)
        channels = shape[-1]
        weight = (1.0 + mx.random.normal((channels,)) * 0.05).astype(mx.float16)
        bias = (mx.random.normal((channels,)) * 0.05).astype(mx.float16)
        norm = nn.GroupNorm(
            num_groups=GROUPS,
            dims=channels,
            eps=EPS,
            pytorch_compatible=True,
        )
        norm.weight = weight
        norm.bias = bias
        mx.eval(x, weight, bias)

        def baseline():
            batch, frames, height, width, _ = x.shape
            x4 = x.reshape(batch * frames, height, width, channels).astype(mx.float32)
            return nn.silu(norm(x4)).astype(x.dtype).reshape(x.shape)

        def target():
            return fused_groupnorm_silu(x, weight, bias)

        reference = baseline()
        actual = target()
        _eval((reference, actual))
        cosine, max_abs = cosine_and_max_abs(actual, reference)
        if cosine < 0.999 or not np.isfinite(np.asarray(actual.astype(mx.float32))).all():
            raise RuntimeError(f"{name}: correctness failed: cosine={cosine}")
        arms = {"baseline": baseline, "target": target}
        timings: dict[str, Any] = {}
        for arm_name in args.order.split("-"):
            timings[arm_name] = _time_arm(
                arms[arm_name], args.warmup, args.sessions, args.cooldown
            )
        results[name] = {
            "shape": list(shape),
            "correctness": {"cosine": cosine, "max_abs": max_abs},
            "engagement": {
                "baseline": "MLX GroupNorm(pytorch_compatible)+SiLU",
                "target": "direct mx.fast.metal_kernel fused reduction+affine+SiLU",
                "different_paths": True,
            },
            "timings": timings,
            "speedup_baseline_over_target": (
                timings["baseline"]["median_ms"] / timings["target"]["median_ms"]
            ),
        }
    from mlx_mfa import _ext

    payload = {
        "schema": "mlx-mfa.seedvr2-groupnorm-silu.probe.v1",
        "stamp": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_mfa": importlib.metadata.version("mlx-mfa"),
            "git_head": os.popen("git rev-parse HEAD").read().strip(),
            "device": dict(_ext.get_device_info()),
        },
        "methodology": {
            "order": args.order,
            "sessions_per_arm": args.sessions,
            "warmup": args.warmup,
            "cooldown_s": args.cooldown,
            "expert_probe_only": True,
            "seedvr2_unchanged": True,
        },
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
