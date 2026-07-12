#!/usr/bin/env python3
"""Benchmark fused GroupNorm+SiLU on real SeedVR2 VAE activations.

The external SeedVR2 checkout is imported read-only. ``grid`` captures every
eligible GroupNorm input from a production-weight encode/decode unit, validates
the fused Metal kernel on every call, then benchmarks one representative per
shape family. ``unit`` applies an in-memory monkeypatch to the ResNet helpers
and the two final norms so the VAE can be timed without editing SeedVR2.
"""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tomllib
import types
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import mlx.core as mx
import numpy as np
from mlx import nn


DEFAULT_SEEDVR2_ROOT = Path(
    "/Users/marcomarcelino/code/ComfyUI-SeedVR2_VideoUpscaler-ORIGINAL"
)
DEFAULT_VIDEO = DEFAULT_SEEDVR2_ROOT / "example_workflows/example_inputs/Eyes_212x120.mp4"
DEFAULT_HEIGHT = 432
DEFAULT_WIDTH = 528
DEFAULT_SESSIONS = 5
DEFAULT_WARMUP = 3
DEFAULT_COOLDOWN_S = 1.0
GROUPS = 32
EPS = 1e-6


def _arrays(value: Any) -> list[mx.array]:
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (tuple, list)):
        result: list[mx.array] = []
        for item in value:
            result.extend(_arrays(item))
        return result
    return []


def _eval(value: Any) -> None:
    arrays = _arrays(value)
    if arrays:
        mx.eval(*arrays)
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


def _cosine_and_max_abs(actual: mx.array, reference: mx.array) -> tuple[float, float]:
    actual_f = actual.astype(mx.float32)
    reference_f = reference.astype(mx.float32)
    dot = mx.sum(actual_f * reference_f)
    denominator = mx.sqrt(mx.sum(actual_f * actual_f) * mx.sum(reference_f * reference_f))
    max_abs = mx.max(mx.abs(actual_f - reference_f))
    mx.eval(dot, denominator, max_abs)
    denominator_value = float(denominator.item())
    cosine = float(dot.item()) / denominator_value if denominator_value else 1.0
    return cosine, float(max_abs.item())


@functools.lru_cache(maxsize=64)
def _kernel(height: int, width: int, channels: int, output_fp32: bool):
    group_size = channels // GROUPS
    elements_per_group = height * width * group_size
    elements_per_frame = height * width * channels
    output_cast = "float(value)" if output_fp32 else "T(value)"
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
            out[offset] = {output_cast};
        }}
    """
    return mx.fast.metal_kernel(
        name=(
            f"seedvr2_groupnorm_silu_h{height}_w{width}_c{channels}_"
            f"{'f32' if output_fp32 else 'native'}"
        ),
        input_names=["x", "weight", "bias"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def fused_groupnorm_silu(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    *,
    output_dtype: mx.Dtype | None = None,
) -> mx.array:
    if x.ndim == 4:
        frames, height, width, channels = x.shape
        output_shape = x.shape
    elif x.ndim == 5:
        batch, frames, height, width, channels = x.shape
        if batch != 1:
            raise ValueError("probe supports B=1")
        output_shape = x.shape
    else:
        raise ValueError(f"expected 4D/5D channels-last input, got {x.shape}")
    if channels % GROUPS:
        raise ValueError("channels must be divisible by 32")
    out_dtype = output_dtype or x.dtype
    output_fp32 = out_dtype == mx.float32
    kernel = _kernel(int(height), int(width), int(channels), output_fp32)
    (out,) = kernel(
        inputs=[x, weight, bias],
        template=[("T", x.dtype)],
        grid=(int(frames) * GROUPS * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[out_dtype],
    )
    return out


def _setup_seedvr2(root: Path) -> None:
    sys.path.insert(0, str(root / "src" / "mlx_native"))
    sys.path.insert(0, str(root / "src"))
    if "toml" not in sys.modules:
        shim = types.ModuleType("toml")

        def load_toml(path: str | Path) -> dict[str, Any]:
            with Path(path).open("rb") as handle:
                return tomllib.load(handle)

        shim.load = load_toml  # type: ignore[attr-defined]
        sys.modules["toml"] = shim


def _load_vae(root: Path):
    from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE

    weights_path = root / "models" / "SEEDVR2" / "ema_vae_fp16.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    vae = SeedVR2VAE()
    raw = mx.load(str(weights_path))
    weights: dict[str, mx.array] = {}
    for key, value in raw.items():
        weights[key] = value.transpose(0, 2, 3, 4, 1) if value.ndim == 5 else value
    vae.load_weights(list(weights.items()))
    replaced = vae.replace_conv1x1_with_linear()
    mx.eval(vae.parameters())
    return vae, len(weights), replaced


def _load_video(path: Path, frames: int, height: int, width: int) -> mx.array:
    if not path.is_file():
        raise FileNotFoundError(path)
    command = [
        "/opt/homebrew/bin/ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"scale={width}:{height}:flags=bicubic",
        "-frames:v",
        str(frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    expected = frames * height * width * 3
    if len(completed.stdout) != expected:
        raise RuntimeError(
            f"ffmpeg returned {len(completed.stdout)} bytes, expected {expected}"
        )
    video = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
        frames, height, width, 3
    )
    video_f = video.astype(np.float32) / 127.5 - 1.0
    return mx.array(video_f).transpose(3, 0, 1, 2)[None].astype(mx.float16)


def _collect_eligible_norms(vae: nn.Module) -> dict[int, str]:
    eligible: dict[int, str] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, nn.GroupNorm):
            leaf = path.rsplit(".", 1)[-1]
            if leaf in {"norm1", "norm2", "conv_norm_out"}:
                eligible[id(value)] = path
            return
        if isinstance(value, nn.Module):
            for name, child in value.children().items():
                visit(child, f"{path}.{name}" if path else name)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, dict):
            for name, child in value.items():
                visit(child, f"{path}.{name}" if path else str(name))

    visit(vae, "vae")
    return eligible


def _prepare_real_unit(args: argparse.Namespace):
    root = args.seedvr2_root.resolve()
    _setup_seedvr2(root)
    vae, weight_count, replaced = _load_vae(root)
    video9 = _load_video(args.video.resolve(), 9, args.height, args.width)
    video5 = video9[:, :, :5]
    latent3 = vae.encode(video9)
    _eval(latent3)
    if latent3.shape[2] != 3:
        raise RuntimeError(f"expected a 3-frame real latent, got {latent3.shape}")
    return root, vae, video5, latent3, weight_count, replaced


def _stamp() -> dict[str, Any]:
    from mlx_mfa import _ext

    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_mfa": importlib.metadata.version("mlx-mfa"),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "device": dict(_ext.get_device_info()),
    }


def _family_name(shape: tuple[int, ...]) -> str:
    return "x".join(str(int(value)) for value in shape)


def grid(args: argparse.Namespace) -> dict[str, Any]:
    root, vae, video5, latent3, weight_count, replaced = _prepare_real_unit(args)
    eligible = _collect_eligible_norms(vae)
    if not eligible:
        raise RuntimeError("no eligible SeedVR2 GroupNorm modules found")

    original_call = nn.GroupNorm.__call__
    representatives: dict[str, tuple[mx.array, nn.GroupNorm]] = {}
    call_rows: list[dict[str, Any]] = []
    phase = "unknown"

    def traced_call(module: nn.GroupNorm, x: mx.array):
        output = original_call(module, x)
        path = eligible.get(id(module))
        if path is None:
            return output
        x_native = x.astype(mx.float16)
        target = fused_groupnorm_silu(x_native, module.weight, module.bias)
        reference = nn.silu(output).astype(mx.float16)
        _eval((x_native, target, reference))
        cosine, max_abs = _cosine_and_max_abs(target, reference)
        x_min = mx.min(x_native)
        x_max = mx.max(x_native)
        x_rms = mx.sqrt(mx.mean(x_native.astype(mx.float32) ** 2))
        mx.eval(x_min, x_max, x_rms)
        family = f"{phase}:{_family_name(tuple(int(value) for value in x_native.shape))}"
        representatives.setdefault(family, (x_native, module))
        call_rows.append(
            {
                "phase": phase,
                "path": path,
                "family": family,
                "shape": list(x_native.shape),
                "input_min": float(x_min.item()),
                "input_max": float(x_max.item()),
                "input_rms": float(x_rms.item()),
                "cosine": cosine,
                "max_abs": max_abs,
                "finite": bool(np.isfinite(np.asarray(target.astype(mx.float32))).all()),
            }
        )
        return output

    nn.GroupNorm.__call__ = traced_call
    try:
        phase = "encode"
        _eval(vae.encode(video5))
        phase = "decode"
        _eval(vae.decode(latent3))
    finally:
        nn.GroupNorm.__call__ = original_call

    if len(representatives) != 18:
        raise RuntimeError(
            f"expected 18 GroupNorm shape families, found {len(representatives)}"
        )
    if any(row["cosine"] < 0.999 or not row["finite"] for row in call_rows):
        failed = [row for row in call_rows if row["cosine"] < 0.999 or not row["finite"]]
        raise RuntimeError(f"real-activation correctness failed: {failed[:3]}")

    calls_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in call_rows:
        calls_by_family[row["family"]].append(row)

    family_rows: list[dict[str, Any]] = []
    for family, (x_native, module) in representatives.items():
        shape = tuple(int(value) for value in x_native.shape)

        def baseline() -> mx.array:
            return nn.silu(
                original_call(module, x_native.astype(mx.float32))
            ).astype(mx.float16)

        def target() -> mx.array:
            return fused_groupnorm_silu(x_native, module.weight, module.bias)

        baseline_out = baseline()
        target_out = target()
        _eval((baseline_out, target_out))
        cosine, max_abs = _cosine_and_max_abs(target_out, baseline_out)
        if cosine < 0.999:
            raise RuntimeError(f"{family}: representative cosine={cosine}")
        timings: dict[str, Any] = {}
        arms = {"baseline": baseline, "target": target}
        for arm in args.order.split("-"):
            timings[arm] = _time_arm(
                arms[arm], args.warmup, args.sessions, args.cooldown
            )
        speedup = timings["baseline"]["median_ms"] / timings["target"]["median_ms"]
        family_calls = calls_by_family[family]
        family_rows.append(
            {
                "family": family,
                "shape": list(shape),
                "calls_per_unit": len(family_calls),
                "paths": sorted({row["path"] for row in family_calls}),
                "phases": sorted({row["phase"] for row in family_calls}),
                "real_activation": {
                    "min": min(row["input_min"] for row in family_calls),
                    "max": max(row["input_max"] for row in family_calls),
                    "max_rms": max(row["input_rms"] for row in family_calls),
                    "min_cosine_all_calls": min(row["cosine"] for row in family_calls),
                    "max_abs_all_calls": max(row["max_abs"] for row in family_calls),
                    "all_finite": all(row["finite"] for row in family_calls),
                },
                "representative_correctness": {
                    "cosine": cosine,
                    "max_abs": max_abs,
                },
                "engagement": {
                    "baseline": "SeedVR2 fp32 cast + MLX GroupNorm + SiLU + fp16 cast",
                    "target": "direct mx.fast.metal_kernel fused GroupNorm+SiLU",
                    "target_direct_kernel_calls": 1,
                    "different_paths": True,
                },
                "timings": timings,
                "speedup_baseline_over_target": speedup,
                "baseline_weighted_ms": timings["baseline"]["median_ms"] * len(family_calls),
                "target_weighted_ms": timings["target"]["median_ms"] * len(family_calls),
                "candidate": speedup > 1.05,
            }
        )
    family_rows.sort(key=lambda row: row["baseline_weighted_ms"], reverse=True)
    baseline_weighted = sum(row["baseline_weighted_ms"] for row in family_rows)
    target_weighted = sum(row["target_weighted_ms"] for row in family_rows)
    return {
        "schema": "mlx-mfa.seedvr2-groupnorm-silu.grid.v2",
        "stamp": _stamp(),
        "seedvr2": {
            "root": str(root),
            "git_head": subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "video": str(args.video.resolve()),
            "weight_count": weight_count,
            "linear_1x1_replaced": replaced,
        },
        "methodology": {
            "order": args.order,
            "sessions_per_arm": args.sessions,
            "warmup": args.warmup,
            "cooldown_s": args.cooldown,
            "real_video_input": True,
            "real_latent_from_production_vae": True,
            "production_weights": True,
            "eligible_norm_rule": "norm1/norm2/conv_norm_out; Attention3D excluded",
        },
        "validation": {
            "eligible_module_count": len(eligible),
            "captured_calls": len(call_rows),
            "family_count": len(family_rows),
            "min_cosine_all_real_calls": min(row["cosine"] for row in call_rows),
            "max_abs_all_real_calls": max(row["max_abs"] for row in call_rows),
            "all_finite": all(row["finite"] for row in call_rows),
        },
        "weighted_projection": {
            "baseline_ms": baseline_weighted,
            "target_ms": target_weighted,
            "speedup": baseline_weighted / target_weighted,
        },
        "families": family_rows,
        "real_call_correctness": call_rows,
    }


@contextmanager
def _groupnorm_patch(vae: nn.Module, enabled: bool) -> Iterator[dict[str, int]]:
    counter = {"fused_calls": 0}
    if not enabled:
        yield counter
        return
    from mflux.models.seedvr2.model.seedvr2_vae.encoder import resnet_block_3d as enc_resnet
    from mflux.models.seedvr2.model.seedvr2_vae.decoder import decoder_resnet_block_3d as dec_resnet

    original_enc = enc_resnet._fused_norm_silu
    original_dec = dec_resnet._fused_norm_silu
    original_groupnorm = nn.GroupNorm.__call__
    original_silu = nn.silu
    final_norms = {id(vae.encoder.conv_norm_out), id(vae.decoder.conv_norm_out)}
    already_activated: dict[int, mx.array] = {}

    def fused_helper(x: mx.array, norm: nn.GroupNorm) -> mx.array:
        counter["fused_calls"] += 1
        return fused_groupnorm_silu(x, norm.weight, norm.bias)

    def patched_groupnorm(module: nn.GroupNorm, x: mx.array) -> mx.array:
        if id(module) not in final_norms:
            return original_groupnorm(module, x)
        counter["fused_calls"] += 1
        result = fused_groupnorm_silu(
            x.astype(mx.float16), module.weight, module.bias, output_dtype=mx.float16
        )
        already_activated[id(result)] = result
        return result

    def patched_silu(x: mx.array) -> mx.array:
        sentinel = already_activated.pop(id(x), None)
        return x if sentinel is x else original_silu(x)

    enc_resnet._fused_norm_silu = fused_helper
    dec_resnet._fused_norm_silu = fused_helper
    nn.GroupNorm.__call__ = patched_groupnorm
    nn.silu = patched_silu
    try:
        yield counter
    finally:
        nn.silu = original_silu
        nn.GroupNorm.__call__ = original_groupnorm
        dec_resnet._fused_norm_silu = original_dec
        enc_resnet._fused_norm_silu = original_enc


def unit(args: argparse.Namespace) -> dict[str, Any]:
    root, vae, video5, latent3, weight_count, replaced = _prepare_real_unit(args)
    import mlx_mfa

    def set_conv(enabled: bool) -> None:
        if enabled:
            os.environ["MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE"] = "1"
        else:
            os.environ.pop("MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE", None)

    def run(arm: str, phase: str) -> tuple[mx.array, int]:
        is_target = arm == "target"
        conv_mode = args.target_conv if is_target else args.baseline_conv
        set_conv(conv_mode == "all")
        with _groupnorm_patch(vae, enabled=is_target and args.target_groupnorm) as counter:
            value = vae.encode(video5) if phase == "encode" else vae.decode(latent3)
            _eval(value)
        return value, counter["fused_calls"]

    baseline_encode, baseline_encode_fused = run("baseline", "encode")
    baseline_decode, baseline_decode_fused = run("baseline", "decode")
    target_encode, target_encode_fused = run("target", "encode")
    target_decode, target_decode_fused = run("target", "decode")
    encode_cosine, encode_max_abs = _cosine_and_max_abs(target_encode, baseline_encode)
    decode_cosine, decode_max_abs = _cosine_and_max_abs(target_decode, baseline_decode)
    if encode_cosine < 0.999 or decode_cosine < 0.999:
        raise RuntimeError(
            f"unit correctness failed: encode={encode_cosine}, decode={decode_cosine}"
        )
    if baseline_encode_fused or baseline_decode_fused:
        raise RuntimeError("baseline unexpectedly called fused GroupNorm kernel")
    if args.target_groupnorm and (target_encode_fused <= 0 or target_decode_fused <= 0):
        raise RuntimeError("target did not engage fused GroupNorm kernel")

    engagement: dict[str, Any] = {}
    for arm in ("baseline", "target"):
        mlx_mfa.reset_hook_stats()
        _, encode_fused = run(arm, "encode")
        _, decode_fused = run(arm, "decode")
        engagement[arm] = {
            "hook_stats": mlx_mfa.get_hook_stats(),
            "groupnorm_fused_calls": encode_fused + decode_fused,
        }

    timings: dict[str, Any] = {}
    if args.settle > 0:
        time.sleep(args.settle)
    timed_arms = [args.only_arm] if args.only_arm else args.order.split("-")
    for arm in timed_arms:
        timings[arm] = {}
        for phase in ("encode", "decode"):
            timings[arm][phase] = _time_arm(
                lambda arm=arm, phase=phase: run(arm, phase)[0],
                args.warmup,
                args.sessions,
                args.cooldown,
            )
    set_conv(False)
    return {
        "schema": "mlx-mfa.seedvr2-groupnorm-silu.unit.v2",
        "stamp": _stamp(),
        "seedvr2": {
            "root": str(root),
            "git_head": subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "video": str(args.video.resolve()),
            "weight_count": weight_count,
            "linear_1x1_replaced": replaced,
        },
        "methodology": {
            "order": args.order,
            "only_arm": args.only_arm,
            "sessions_per_arm_phase": args.sessions,
            "warmup": args.warmup,
            "cooldown_s": args.cooldown,
            "settle_s": args.settle,
            "isolated_process_required": True,
            "real_video_input": True,
            "real_latent_from_production_vae": True,
            "baseline_conv": args.baseline_conv,
            "target_conv": args.target_conv,
            "target_groupnorm": args.target_groupnorm,
        },
        "correctness": {
            "encode": {"cosine": encode_cosine, "max_abs": encode_max_abs},
            "decode": {"cosine": decode_cosine, "max_abs": decode_max_abs},
        },
        "engagement": engagement,
        "timings": timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("grid", "unit"), default="grid")
    parser.add_argument("--seedvr2-root", type=Path, default=DEFAULT_SEEDVR2_ROOT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_S)
    parser.add_argument("--settle", type=float, default=10.0)
    parser.add_argument(
        "--order",
        choices=("baseline-target", "target-baseline"),
        default="baseline-target",
    )
    parser.add_argument("--only-arm", choices=("baseline", "target"))
    parser.add_argument("--baseline-conv", choices=("off", "all"), default="all")
    parser.add_argument("--target-conv", choices=("off", "all"), default="all")
    parser.add_argument(
        "--target-groupnorm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    mx.random.seed(20260712)
    payload = grid(args) if args.mode == "grid" else unit(args)
    text = json.dumps(payload, indent=2)
    if not args.quiet:
        print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
