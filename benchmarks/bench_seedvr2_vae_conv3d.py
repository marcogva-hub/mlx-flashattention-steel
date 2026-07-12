#!/usr/bin/env python3
"""Inventory SeedVR2 VAE Conv3D routing and cost by runtime family.

The SeedVR2 repository is imported read-only. This harness loads production VAE
weights, executes the same 5-frame encode + 3-latent decode units as the E2E
profile, and attributes every CausalConv3d call to its actual mlx-mfa route.

The instrumentation intentionally synchronizes each call. Its family shares
are diagnostic cost attribution, not a reconstruction of the lazy production
wall clock. Comparative probe timings live behind a separate mode added only
after this inventory identifies the dominant fallback family.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
import tomllib
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np


DEFAULT_SEEDVR2_ROOT = Path(
    "/Users/marcomarcelino/code/ComfyUI-SeedVR2_VideoUpscaler-ORIGINAL"
)
DEFAULT_HEIGHT = 432
DEFAULT_WIDTH = 528
DEFAULT_SESSIONS = 5
DEFAULT_WARMUP = 1
DEFAULT_COOLDOWN_S = 1.0
PROBE_CASES = {
    "d512_t5_h108_w132": (1, 5, 108, 132, 512),
    "d512_t4_h108_w132": (1, 4, 108, 132, 512),
}


def _arrays(value: Any) -> list[mx.array]:
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (tuple, list)):
        arrays: list[mx.array] = []
        for item in value:
            arrays.extend(_arrays(item))
        return arrays
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
    fn: Callable[[], Any], *, warmup: int, sessions: int, cooldown_s: float
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
    af = actual.astype(mx.float32)
    rf = reference.astype(mx.float32)
    dot = mx.sum(af * rf)
    denominator = mx.sqrt(mx.sum(af * af) * mx.sum(rf * rf))
    max_abs = mx.max(mx.abs(af - rf))
    mx.eval(dot, denominator, max_abs)
    denominator_value = float(denominator.item())
    cosine = float(dot.item()) / denominator_value if denominator_value else 1.0
    return cosine, float(max_abs.item())


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


def _load_vae(weights_path: Path):
    from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE

    vae = SeedVR2VAE()
    raw = mx.load(str(weights_path))
    weights: dict[str, mx.array] = {}
    for key, value in raw.items():
        weights[key] = value.transpose(0, 2, 3, 4, 1) if value.ndim == 5 else value
    vae.load_weights(list(weights.items()))
    replaced = vae.replace_conv1x1_with_linear()
    mx.eval(vae.parameters())
    return vae, len(weights), replaced


def _tuple3(value: Any) -> tuple[int, int, int] | None:
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return (int(value[0]),) * 3
        if len(value) == 3:
            return tuple(int(item) for item in value)
    return None


def _precise_gate_reason(
    hooks: Any,
    input_array: mx.array,
    weight: mx.array,
    stride: Any,
    padding: Any,
    kernel_dilation: Any,
    input_dilation: Any,
    groups: int,
    flip: bool,
) -> tuple[str, tuple[int, ...] | None]:
    if not hooks._is_m5_plus():
        return "not_m5_plus", None
    if len(weight.shape) != 5:
        return "weight_not_5d", None
    if weight.dtype not in (mx.float16, mx.bfloat16):
        return f"weight_dtype_{weight.dtype}", None
    kernel = tuple(int(item) for item in weight.shape[1:4])
    stride3 = _tuple3(stride)
    kd3 = _tuple3(kernel_dilation)
    id3 = _tuple3(input_dilation)
    if kernel not in hooks._ELIGIBLE_KERNEL_SIZES:
        return f"kernel_{kernel}_not_eligible", None
    if groups != 1:
        return f"groups_{groups}", None
    if flip:
        return "flip_true", None
    if stride3 != (1, 1, 1):
        return f"stride_{stride3}", None
    if kd3 != (1, 1, 1):
        return f"kernel_dilation_{kd3}", None
    if id3 != (1, 1, 1):
        return f"input_dilation_{id3}", None
    pad = hooks._normalize_padding_to_6tuple(padding)
    if pad is None or any(item < 0 for item in pad):
        return f"padding_{padding}", pad
    if input_array.shape[0] != 1:
        return f"batch_{input_array.shape[0]}", pad
    if pad not in ((1, 1, 1, 1, 1, 1), (0, 0, 1, 1, 1, 1)):
        return f"mpp_padding_{pad}", pad
    if input_array.shape[2] % 8 or input_array.shape[3] % 8:
        return f"spatial_alignment_{input_array.shape[2]}x{input_array.shape[3]}", pad
    cin = int(input_array.shape[-1])
    cout = int(weight.shape[0])
    if cin < 32 or cin % 16:
        return f"cin_{cin}_outside_mpp", pad
    if cout < 32 or cout % 16:
        return f"cout_{cout}_outside_mpp", pad
    return "eligible_mpp", pad


def _family_key(record: dict[str, Any]) -> str:
    return "|".join(
        [
            record["phase"],
            "x=" + "x".join(map(str, record["input_shape"])),
            "w=" + "x".join(map(str, record["weight_shape"])),
            "s=" + "x".join(map(str, record["stride"])),
            "p=" + "x".join(map(str, record["padding_6"] or [])),
            record["route"],
            record["gate_reason"],
        ]
    )


def _stamp() -> dict[str, Any]:
    from mlx_mfa import _ext

    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_mfa": importlib.metadata.version("mlx-mfa"),
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "device": dict(_ext.get_device_info()),
    }


def inventory(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("SEEDVR2_MLX_METAL_CONV_ENABLED", "0").lower() in {
        "1",
        "true",
        "yes",
    }:
        raise RuntimeError("SeedVR2 WIP Metal conv must be disabled for mlx-mfa inventory")

    root = args.seedvr2_root.resolve()
    _setup_seedvr2(root)
    import mlx_mfa
    from mlx_mfa import _auto_hooks as hooks
    from mflux.models.seedvr2.model.seedvr2_vae.common import conv3d as conv_module

    weights_path = root / "models" / "SEEDVR2" / "ema_vae_fp16.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if not getattr(mx.conv_general, "__mlx_mfa_hook__", False):
        raise RuntimeError("mlx-mfa conv_general hook is not installed")

    vae, weight_count, replaced = _load_vae(weights_path)
    x = mx.random.normal((1, 3, 5, args.height, args.width)).astype(mx.float16)
    z = mx.random.normal((1, 16, 3, args.height // 8, args.width // 8)).astype(
        mx.float16
    )
    mx.eval(x, z)

    def encode_unit():
        return vae.encode(x)

    def decode_unit():
        return vae.decode(z)

    for _ in range(args.warmup):
        _eval((encode_unit(), decode_unit()))

    original_conv_call = conv_module.CausalConv3d.__call__
    hooked_conv_general = mx.conv_general
    records: list[dict[str, Any]] = []
    current_phase = "unknown"
    current_session = -1

    def traced_conv_general(
        input_array,
        weight,
        stride=1,
        padding=0,
        kernel_dilation=1,
        input_dilation=1,
        groups=1,
        flip=False,
        **kwargs,
    ):
        before = mlx_mfa.get_hook_stats()
        start = time.perf_counter()
        result = hooked_conv_general(
            input_array,
            weight,
            stride=stride,
            padding=padding,
            kernel_dilation=kernel_dilation,
            input_dilation=input_dilation,
            groups=groups,
            flip=flip,
            **kwargs,
        )
        _eval(result)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        after = mlx_mfa.get_hook_stats()
        executed_delta = (
            after["executed"].get("conv3d_nax_forward", 0)
            - before["executed"].get("conv3d_nax_forward", 0)
        )
        pad_slice_delta = (
            after["executed"].get("conv3d_nax_pad_slice", 0)
            - before["executed"].get("conv3d_nax_pad_slice", 0)
        )
        fallback_delta = (
            after["fallback"].get("conv3d_nax_forward", 0)
            - before["fallback"].get("conv3d_nax_forward", 0)
        )
        if executed_delta == 1 and fallback_delta == 0:
            route = "nax_mpp"
        elif pad_slice_delta == 1 and fallback_delta == 0:
            route = "nax_pad_slice"
        elif fallback_delta == 1 and executed_delta == 0 and pad_slice_delta == 0:
            route = "mlx_pre_hook_fallback"
        else:
            raise RuntimeError(
                "ambiguous per-call engagement: "
                f"executed={executed_delta}, pad_slice={pad_slice_delta}, "
                f"fallback={fallback_delta}"
            )
        reason, pad = _precise_gate_reason(
            hooks,
            input_array,
            weight,
            stride,
            padding,
            kernel_dilation,
            input_dilation,
            groups,
            flip,
        )
        record = {
            "session": current_session,
            "phase": current_phase,
            "input_shape": list(input_array.shape),
            "weight_shape": list(weight.shape),
            "output_shape": list(result.shape),
            "dtype": str(input_array.dtype),
            "stride": list(_tuple3(stride) or ()),
            "padding_raw": list(padding) if isinstance(padding, tuple) else padding,
            "padding_6": list(pad) if pad is not None else None,
            "kernel_dilation": list(_tuple3(kernel_dilation) or ()),
            "input_dilation": list(_tuple3(input_dilation) or ()),
            "groups": groups,
            "flip": flip,
            "route": route,
            "gate_reason": reason,
            "core_conv_ms": elapsed_ms,
        }
        records.append(record)
        return result

    def traced_causal_call(module, *call_args, **call_kwargs):
        start_index = len(records)
        start = time.perf_counter()
        result = original_conv_call(module, *call_args, **call_kwargs)
        _eval(result)
        total_ms = (time.perf_counter() - start) * 1000.0
        if len(records) != start_index + 1:
            raise RuntimeError(
                "expected exactly one mx.conv_general per CausalConv3d call, got "
                f"{len(records) - start_index}"
            )
        records[-1]["causal_call_ms"] = total_ms
        records[-1]["module_kernel"] = list(module.kernel_size)
        records[-1]["module_stride"] = list(module.stride)
        records[-1]["module_padding"] = list(module.padding)
        return result

    mx.conv_general = traced_conv_general
    conv_module.CausalConv3d.__call__ = traced_causal_call
    session_walls: list[dict[str, float]] = []
    try:
        for session in range(args.sessions):
            current_session = session
            current_phase = "encode"
            start = time.perf_counter()
            _eval(encode_unit())
            encode_ms = (time.perf_counter() - start) * 1000.0
            current_phase = "decode"
            start = time.perf_counter()
            _eval(decode_unit())
            decode_ms = (time.perf_counter() - start) * 1000.0
            session_walls.append(
                {"encode_ms": encode_ms, "decode_ms": decode_ms, "total_ms": encode_ms + decode_ms}
            )
            if args.cooldown > 0 and session + 1 < args.sessions:
                time.sleep(args.cooldown)
    finally:
        conv_module.CausalConv3d.__call__ = original_conv_call
        mx.conv_general = hooked_conv_general

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        families[_family_key(record)].append(record)
    family_rows: list[dict[str, Any]] = []
    total_conv_ms = sum(record["causal_call_ms"] for record in records)
    for key, items in families.items():
        template = items[0]
        causal_samples = [item["causal_call_ms"] for item in items]
        core_samples = [item["core_conv_ms"] for item in items]
        row = {
            "family": key,
            "phase": template["phase"],
            "input_shape": template["input_shape"],
            "weight_shape": template["weight_shape"],
            "output_shape": template["output_shape"],
            "dtype": template["dtype"],
            "stride": template["stride"],
            "padding_6": template["padding_6"],
            "route": template["route"],
            "gate_reason": template["gate_reason"],
            "calls_total": len(items),
            "calls_per_session": len(items) / args.sessions,
            "causal_call": _stats(causal_samples),
            "core_conv": _stats(core_samples),
            "per_session_total_ms": sum(causal_samples) / args.sessions,
            "share_of_instrumented_conv": sum(causal_samples) / total_conv_ms,
        }
        family_rows.append(row)
    family_rows.sort(key=lambda item: item["per_session_total_ms"], reverse=True)

    route_totals: dict[str, dict[str, float]] = {}
    for route in sorted({record["route"] for record in records}):
        selected = [record for record in records if record["route"] == route]
        route_totals[route] = {
            "calls_per_session": len(selected) / args.sessions,
            "per_session_total_ms": sum(item["causal_call_ms"] for item in selected)
            / args.sessions,
            "share_of_instrumented_conv": sum(
                item["causal_call_ms"] for item in selected
            )
            / total_conv_ms,
        }

    expected_calls = args.sessions * 91
    if len(records) != expected_calls:
        raise RuntimeError(f"expected {expected_calls} Conv3D calls, got {len(records)}")
    for session in range(args.sessions):
        session_records = [record for record in records if record["session"] == session]
        routes = defaultdict(int)
        for record in session_records:
            routes[record["route"]] += 1
        if routes["nax_mpp"] != 36 or routes["mlx_pre_hook_fallback"] != 55:
            raise RuntimeError(f"session {session}: route lock failed: {dict(routes)}")

    return {
        "schema": "mlx-mfa.seedvr2-vae-conv3d.inventory.v1",
        "stamp": _stamp(),
        "seedvr2": {
            "root": str(root),
            "git_head": os.popen(f"git -C {root} rev-parse HEAD").read().strip(),
            "weight_count": weight_count,
            "linear_1x1_replaced": replaced,
        },
        "methodology": {
            "unit": "encode 5x432x528 + decode 3x54x66",
            "sessions": args.sessions,
            "warmup": args.warmup,
            "cooldown_s": args.cooldown,
            "per_call_sync_changes_lazy_schedule": True,
            "cost_basis": "CausalConv3d call wall including padding/transpose/bias",
            "seedvr2_wip_metal_conv_enabled": False,
        },
        "validation": {
            "records": len(records),
            "expected_records": expected_calls,
            "per_session_nax_mpp": 36,
            "per_session_fallback": 55,
            "routes_locked": True,
        },
        "session_walls": session_walls,
        "route_totals": route_totals,
        "families": family_rows,
        "raw_records": records,
    }


def _spatial_pad_slice_probe(x: mx.array, weight: mx.array) -> mx.array:
    from mlx_mfa import _ext

    height = int(x.shape[2])
    width = int(x.shape[3])
    padded_height = ((height + 7) // 8) * 8
    padded_width = ((width + 7) // 8) * 8
    if padded_height == height and padded_width == width:
        raise RuntimeError("spatial probe requires a non-aligned input")
    padded = mx.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (0, padded_height - height),
            (0, padded_width - width),
            (0, 0),
        ),
    )
    output = _ext.conv3d_nax_forward(
        padded,
        weight,
        stride=(1, 1, 1),
        padding=(0, 0, 1, 1, 1, 1),
        dilation=(1, 1, 1),
        chunk_M=0,
    )
    return output[:, :, :height, :width, :]


def probe(args: argparse.Namespace) -> dict[str, Any]:
    import mlx_mfa
    from mlx_mfa import _auto_hooks as hooks
    from mlx_mfa import _ext

    if hooks._ORIGINAL_CONV_GENERAL is None:
        raise RuntimeError("pre-hook mx.conv_general baseline is unavailable")
    original_native = _ext.conv3d_nax_forward
    case_results: dict[str, Any] = {}
    for case_index, (name, shape) in enumerate(PROBE_CASES.items()):
        mx.random.seed(20260712 + case_index)
        x = (mx.random.normal(shape) * 0.05).astype(mx.float16)
        weight = (
            mx.random.normal((512, 3, 3, 3, 512)) * (1.0 / (27 * 512) ** 0.5)
        ).astype(mx.float16)
        mx.eval(x, weight)

        def baseline():
            return hooks._ORIGINAL_CONV_GENERAL(
                x,
                weight,
                stride=(1, 1, 1),
                padding=(0, 1, 1),
            )

        def target():
            return _spatial_pad_slice_probe(x, weight)

        oracle = hooks._ORIGINAL_CONV_GENERAL(
            x.astype(mx.float32),
            weight.astype(mx.float32),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
        )
        baseline_probe = baseline()
        target_probe = target()
        _eval((oracle, baseline_probe, target_probe))
        cosine, max_abs = _cosine_and_max_abs(target_probe, oracle)
        baseline_cosine, baseline_max_abs = _cosine_and_max_abs(baseline_probe, oracle)
        finite = bool(np.isfinite(np.asarray(target_probe.astype(mx.float32))).all())
        if cosine < 0.999 or not finite:
            raise RuntimeError(
                f"{name}: correctness gate failed: cosine={cosine}, finite={finite}"
            )

        native_calls = 0

        def counted_native(*call_args, **call_kwargs):
            nonlocal native_calls
            native_calls += 1
            return original_native(*call_args, **call_kwargs)

        _ext.conv3d_nax_forward = counted_native
        try:
            _eval(baseline())
            baseline_native_calls = native_calls
            native_calls = 0
            _eval(target())
            target_native_calls = native_calls
        finally:
            _ext.conv3d_nax_forward = original_native
        if baseline_native_calls != 0 or target_native_calls != 1:
            raise RuntimeError(
                f"{name}: which-binary failed: baseline={baseline_native_calls}, "
                f"target={target_native_calls}"
            )

        arms = {"baseline": baseline, "target": target}
        timings: dict[str, Any] = {}
        for arm_name in args.order.split("-"):
            timings[arm_name] = _time_arm(
                arms[arm_name],
                warmup=args.warmup,
                sessions=args.sessions,
                cooldown_s=args.cooldown,
            )
        ratio = timings["baseline"]["median_ms"] / timings["target"]["median_ms"]
        case_results[name] = {
            "input_shape": list(shape),
            "weight_shape": list(weight.shape),
            "padded_spatial": [112, 136],
            "padding_work_ratio": (112 * 136) / (108 * 132),
            "correctness": {
                "target_vs_fp32": {"cosine": cosine, "max_abs": max_abs},
                "baseline_vs_fp32": {
                    "cosine": baseline_cosine,
                    "max_abs": baseline_max_abs,
                },
                "finite": finite,
            },
            "engagement": {
                "baseline": "captured pre-hook mx.conv_general",
                "baseline_native_calls": baseline_native_calls,
                "target": "zero-pad -> direct _ext.conv3d_nax_forward -> slice",
                "target_native_calls": target_native_calls,
                "symbols_distinct": original_native is not hooks._ORIGINAL_CONV_GENERAL,
            },
            "timings": timings,
            "speedup_baseline_over_target": ratio,
            "target_wins": ratio > 1.05,
        }

    return {
        "schema": "mlx-mfa.seedvr2-vae-conv3d.probe.v1",
        "stamp": _stamp(),
        "methodology": {
            "order": args.order,
            "sessions_per_arm": args.sessions,
            "warmup": args.warmup,
            "cooldown_s": args.cooldown,
            "per_arm_sustained": True,
            "baseline_pre_hook": True,
            "probe_is_expert_only": True,
            "public_gate_unchanged": True,
        },
        "cases": case_results,
    }


def unit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.seedvr2_root.resolve()
    _setup_seedvr2(root)
    import mlx_mfa

    weights_path = root / "models" / "SEEDVR2" / "ema_vae_fp16.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    vae, weight_count, replaced = _load_vae(weights_path)
    x = mx.random.normal((1, 3, 5, args.height, args.width)).astype(mx.float16)
    z = mx.random.normal((1, 16, 3, args.height // 8, args.width // 8)).astype(
        mx.float16
    )
    mx.eval(x, z)

    def set_spatial_opt_in(enabled: bool) -> None:
        if enabled:
            os.environ["MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE"] = "1"
        else:
            os.environ.pop("MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE", None)

    def encode(enabled: bool):
        set_spatial_opt_in(enabled)
        return vae.encode(x)

    def decode(enabled: bool):
        set_spatial_opt_in(enabled)
        return vae.decode(z)

    baseline_encode = encode(False)
    baseline_decode = decode(False)
    target_encode = encode(True)
    target_decode = decode(True)
    _eval((baseline_encode, baseline_decode, target_encode, target_decode))
    encode_cosine, encode_max_abs = _cosine_and_max_abs(target_encode, baseline_encode)
    decode_cosine, decode_max_abs = _cosine_and_max_abs(target_decode, baseline_decode)
    if encode_cosine < 0.999 or decode_cosine < 0.999:
        raise RuntimeError(
            "unit correctness failed: "
            f"encode_cosine={encode_cosine}, decode_cosine={decode_cosine}"
        )

    engagement: dict[str, Any] = {}
    for arm_name, enabled in (("baseline", False), ("target", True)):
        mlx_mfa.reset_hook_stats()
        _eval((encode(enabled), decode(enabled)))
        engagement[arm_name] = mlx_mfa.get_hook_stats()
    baseline_executed = engagement["baseline"]["executed"]
    baseline_fallback = engagement["baseline"]["fallback"]
    target_executed = engagement["target"]["executed"]
    target_fallback = engagement["target"]["fallback"]
    if (
        baseline_executed.get("conv3d_nax_forward", 0) != 36
        or baseline_executed.get("conv3d_nax_spatial_pad_slice", 0) != 0
        or baseline_fallback.get("conv3d_nax_forward", 0) != 55
    ):
        raise RuntimeError(f"baseline engagement failed: {engagement['baseline']}")
    if (
        target_executed.get("conv3d_nax_forward", 0) != 36
        or target_executed.get("conv3d_nax_spatial_pad_slice", 0) != 17
        or target_fallback.get("conv3d_nax_forward", 0) != 38
    ):
        raise RuntimeError(f"target engagement failed: {engagement['target']}")

    arms = {
        "baseline": {
            "encode": lambda: encode(False),
            "decode": lambda: decode(False),
        },
        "target": {
            "encode": lambda: encode(True),
            "decode": lambda: decode(True),
        },
    }
    timings: dict[str, Any] = {}
    timed_arms = [args.only_arm] if args.only_arm else args.order.split("-")
    if args.settle > 0:
        time.sleep(args.settle)
    try:
        for arm_name in timed_arms:
            timings[arm_name] = {
                phase: _time_arm(
                    fn,
                    warmup=args.warmup,
                    sessions=args.sessions,
                    cooldown_s=args.cooldown,
                )
                for phase, fn in arms[arm_name].items()
            }
    finally:
        set_spatial_opt_in(False)

    speedup = None
    archive_projection = None
    if set(timings) == {"baseline", "target"}:
        encode_speedup = (
            timings["baseline"]["encode"]["median_ms"]
            / timings["target"]["encode"]["median_ms"]
        )
        decode_speedup = (
            timings["baseline"]["decode"]["median_ms"]
            / timings["target"]["decode"]["median_ms"]
        )
        encode_scale = 897 / 5
        decode_scale = 225 / 3
        baseline_archive_ms = (
            timings["baseline"]["encode"]["median_ms"] * encode_scale
            + timings["baseline"]["decode"]["median_ms"] * decode_scale
        )
        target_archive_ms = (
            timings["target"]["encode"]["median_ms"] * encode_scale
            + timings["target"]["decode"]["median_ms"] * decode_scale
        )
        vae_speedup = baseline_archive_ms / target_archive_ms
        speedup = {"encode": encode_speedup, "decode": decode_speedup}
        archive_projection = {
            "encode_scale": encode_scale,
            "decode_scale": decode_scale,
            "baseline_ms": baseline_archive_ms,
            "target_ms": target_archive_ms,
            "vae_speedup": vae_speedup,
            "vae_time_reduction": 1.0 - target_archive_ms / baseline_archive_ms,
            "e2e_speedup_using_vae_share_0_72": 1.0 / (0.28 + 0.72 / vae_speedup),
        }
    return {
        "schema": "mlx-mfa.seedvr2-vae-conv3d.unit.v1",
        "stamp": _stamp(),
        "seedvr2": {
            "root": str(root),
            "git_head": os.popen(f"git -C {root} rev-parse HEAD").read().strip(),
            "weight_count": weight_count,
            "linear_1x1_replaced": replaced,
        },
        "methodology": {
            "order": args.order,
            "only_arm": args.only_arm,
            "sessions_per_arm_phase": args.sessions,
            "warmup": args.warmup,
            "cooldown_s": args.cooldown,
            "per_arm_sustained": True,
            "production_weights": True,
            "shape_faithful_synthetic_activations": True,
            "baseline_pre_opt_in": True,
            "settle_before_timing_s": args.settle,
        },
        "correctness": {
            "target_encode_vs_baseline": {
                "cosine": encode_cosine,
                "max_abs": encode_max_abs,
            },
            "target_decode_vs_baseline": {
                "cosine": decode_cosine,
                "max_abs": decode_max_abs,
            },
        },
        "engagement": engagement,
        "timings": timings,
        "speedup": speedup,
        "archive_projection": archive_projection,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("inventory", "probe", "unit"), default="inventory"
    )
    parser.add_argument("--seedvr2-root", type=Path, default=DEFAULT_SEEDVR2_ROOT)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_S)
    parser.add_argument(
        "--order",
        choices=("baseline-target", "target-baseline"),
        default="baseline-target",
    )
    parser.add_argument("--only-arm", choices=("baseline", "target"))
    parser.add_argument("--settle", type=float, default=0.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    mx.random.seed(20260712)
    if args.mode == "inventory":
        result = inventory(args)
    elif args.mode == "probe":
        result = probe(args)
    else:
        result = unit(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.mode == "inventory":
        default_name = "seedvr2_vae_conv3d_inventory.json"
    else:
        default_name = (
            f"seedvr2_vae_conv3d_{args.mode}_{args.order.replace('-', '_')}.json"
        )
    out = args.out or Path("benchmarks/results") / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
