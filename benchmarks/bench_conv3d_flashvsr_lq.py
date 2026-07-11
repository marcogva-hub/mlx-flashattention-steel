#!/usr/bin/env python3
"""Benchmark the exact FlashVSR LQ Conv3D calls against MLX.

The two cases correspond to the three calls made for an eight-frame input:
conv1 executes twice and conv2 executes once. Arms are sustained rather than
interleaved, and both orderings are recorded to expose thermal/order drift.
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
import types
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_mfa import _ext
from mlx_mfa import _auto_hooks as hooks


ROOT = Path("/Users/marcomarcelino/code/FlashVSR")
MODEL_DIR = ROOT / "model"
STRIDE = (2, 1, 1)
PAD = (0, 0, 0, 0, 0, 0)


def cosine(a: mx.array, b: mx.array) -> float:
    af = np.asarray(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    bf = np.asarray(b.astype(mx.float32)).reshape(-1).astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))


def time_arm(fn, sessions: int, iters: int) -> dict:
    for _ in range(2):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(sessions):
        start = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0 / iters)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def load_weights(dtype):
    sys.path.insert(0, str(ROOT))
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from run_flashvsr_FINAL import LQAdapter, load_weights as load_model_weights

    model = LQAdapter(target_dim=1536, num_layers=1)
    load_model_weights(model, "lq_proj_mlx", str(MODEL_DIR))
    def convert(tree):
        if isinstance(tree, mx.array):
            return tree.astype(dtype) if tree.dtype == mx.float32 else tree
        if isinstance(tree, dict):
            return {key: convert(value) for key, value in tree.items()}
        if isinstance(tree, list):
            return [convert(value) for value in tree]
        return tree

    model.update(convert(model.parameters()))
    return model.conv1.weight, model.conv2.weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    dtype = mx.float16 if args.dtype == "fp16" else mx.bfloat16
    w1, w2 = load_weights(dtype)
    mx.random.seed(712)
    cases = {
        "conv1": ((1, 6, 18, 18, 768), w1, 2),
        "conv2": ((1, 4, 18, 18, 2048), w2, 1),
    }
    results = {}
    for name, (shape, weight, multiplicity) in cases.items():
        x = (mx.random.normal(shape) * 0.05).astype(dtype)
        baseline = lambda: hooks._ORIGINAL_CONV_GENERAL(
            x, weight, stride=STRIDE, padding=0
        )
        target = lambda: _ext.conv3d_nax_forward(
            x, weight, stride=STRIDE, padding=PAD,
            dilation=(1, 1, 1), chunk_M=0
        )
        oracle = hooks._ORIGINAL_CONV_GENERAL(
            x.astype(mx.float32), weight.astype(mx.float32),
            stride=STRIDE, padding=0
        )
        y_target, y_base = target(), baseline()
        mx.eval(oracle, y_target, y_base)
        cos = cosine(y_target, oracle)
        if cos < 0.999 or not np.isfinite(np.asarray(y_target.astype(mx.float32))).all():
            raise RuntimeError(f"{name}: correctness failed, cos={cos}")
        first_base = time_arm(baseline, args.sessions, args.iters)
        first_nax = time_arm(target, args.sessions, args.iters)
        second_nax = time_arm(target, args.sessions, args.iters)
        second_base = time_arm(baseline, args.sessions, args.iters)
        results[name] = {
            "input_shape": shape,
            "weight_shape": tuple(weight.shape),
            "multiplicity": multiplicity,
            "cos_vs_fp32": cos,
            "cos_vs_mlx_dtype": cosine(y_target, y_base),
            "baseline_then_nax": {"mlx": first_base, "nax": first_nax},
            "nax_then_baseline": {"nax": second_nax, "mlx": second_base},
            "speedup_first": first_base["median_ms"] / first_nax["median_ms"],
            "speedup_second": second_base["median_ms"] / second_nax["median_ms"],
        }
    payload = {
        "stamp": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "git": os.popen("git rev-parse HEAD").read().strip(),
            "mlx": importlib.metadata.version("mlx"),
            "python": platform.python_version(),
            "device": dict(_ext.get_device_info()),
        },
        "engagement": {
            "target": "direct _ext.conv3d_nax_forward",
            "baseline": "captured pre-hook mx.conv_general",
            "different_symbols": _ext.conv3d_nax_forward is not hooks._ORIGINAL_CONV_GENERAL,
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
