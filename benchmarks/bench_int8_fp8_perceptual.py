#!/usr/bin/env python3
"""SeedVR2 fake-int8 Pareto probe.

This is a read-only overlay around the external SeedVR2 repository.  It loads
the production classes and weights through the existing audit harness, then
replaces selected Linear/attention boundaries in-process with symmetric
quantize-dequantize operations.  No external-repo file is modified.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np
from mlx import nn

from benchmarks.bench_audit_e2e_seedvr2 import (
    DEFAULT_SEEDVR2_ROOT,
    _cosine,
    _eval,
    _load_dit,
    _load_vae,
    _setup_seedvr2,
)


def _fake_int8(x: mx.array, granularity: str, *, weight: bool = False) -> mx.array:
    original = x.dtype
    xf = x.astype(mx.float32)
    if granularity == "per_tensor":
        amax = mx.max(mx.abs(xf))
        scale = mx.maximum(amax, mx.array(1e-8, dtype=mx.float32)) / 127.0
        q = mx.clip(mx.round(xf / scale), -127.0, 127.0)
        return (q * scale).astype(original)
    if granularity.startswith("per_block"):
        group = int(granularity.removeprefix("per_block"))
        if xf.shape[-1] % group:
            raise ValueError(f"last dimension {xf.shape[-1]} is not divisible by {group}")
        flat = xf.reshape(-1, xf.shape[-1] // group, group)
        amax = mx.max(mx.abs(flat), axis=-1, keepdims=True)
        scale = mx.maximum(amax, mx.array(1e-8, dtype=mx.float32)) / 127.0
        q = mx.clip(mx.round(flat / scale), -127.0, 127.0)
        return (q * scale).reshape(xf.shape).astype(original)
    if granularity == "per_channel":
        if weight:
            # Linear weights are [out, in]: one scale per output row.
            amax = mx.max(mx.abs(xf), axis=1, keepdims=True)
        else:
            axes = tuple(range(max(xf.ndim - 1, 0)))
            amax = mx.max(mx.abs(xf), axis=axes, keepdims=True) if axes else mx.abs(xf)
        scale = mx.maximum(amax, mx.array(1e-8, dtype=mx.float32)) / 127.0
        q = mx.clip(mx.round(xf / scale), -127.0, 127.0)
        return (q * scale).astype(original)
    raise ValueError(f"unsupported granularity {granularity}")


class _QuantLinear(nn.Module):
    def __init__(self, module: object, granularity: str):
        super().__init__()
        self.weight = _fake_int8(module.weight, granularity, weight=True)  # type: ignore[attr-defined]
        bias = getattr(module, "bias", None)
        if bias is not None:
            self.bias = bias
        self.granularity = granularity

    def __call__(self, x: mx.array) -> mx.array:
        qx = _fake_int8(x, self.granularity)
        result = mx.matmul(qx, self.weight.T)
        return result + self.bias if hasattr(self, "bias") else result


class _QuantNorm(nn.Module):
    def __init__(self, base: object, granularity: str):
        super().__init__()
        self.base = base
        self.granularity = granularity

    def __call__(self, x: mx.array) -> mx.array:
        return _fake_int8(self.base(x), self.granularity)


class _QuantActivation(nn.Module):
    def __init__(self, base: object, granularity: str):
        super().__init__()
        self.base = base
        self.granularity = granularity

    def __call__(self, x: mx.array) -> mx.array:
        return self.base(_fake_int8(x, self.granularity))


def _patch_ffn(model: object, granularity: str) -> int:
    count = 0
    for block in model.blocks:  # type: ignore[attr-defined]
        for name in ("vid", "txt", "all"):
            mlp = getattr(block.mlp, name, None)
            if mlp is None:
                continue
            for proj_name in ("proj_in", "proj_in_gate", "proj_out"):
                proj = getattr(mlp, proj_name, None)
                if proj is not None:
                    setattr(mlp, proj_name, _QuantLinear(proj, granularity))
                    count += 1
    return count


def _patch_attention(model: object, granularity: str, site: str) -> int:
    count = 0
    for block in model.blocks:  # type: ignore[attr-defined]
        attn = block.attn
        if site in ("attention_qk", "combined"):
            for name in ("norm_q_vid", "norm_k_vid", "norm_q_txt", "norm_k_txt"):
                module = getattr(attn, name)
                setattr(attn, name, _QuantNorm(module, granularity))
                count += 1
        if site in ("attention_pv", "combined"):
            for name in ("proj_out_vid", "proj_out_txt"):
                module = getattr(attn, name)
                setattr(attn, name, _QuantActivation(module, granularity))
                count += 1
    return count


def _ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    c1, c2 = 1e-4, 9e-4
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = np.mean((a - mu_a) * (b - mu_b))
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a * mu_a + mu_b * mu_b + c1) *
                  (var_a + var_b + c2)))


def _decoded_metrics(ref: mx.array, out: mx.array) -> dict:
    ref_np = np.asarray(ref.astype(mx.float32))
    out_np = np.asarray(out.astype(mx.float32))
    if ref_np.shape != out_np.shape:
        raise ValueError(f"decoded shape mismatch: {ref_np.shape} vs {out_np.shape}")
    if ref_np.ndim == 5:
        # SeedVR2 VAE layout is [B,C,T,H,W].
        frames = [(_ssim_global(ref_np[0, :, t], out_np[0, :, t]),
                   float(np.sqrt(np.mean((ref_np[0, :, t] - out_np[0, :, t]) ** 2))))
                  for t in range(ref_np.shape[2])]
    else:
        frames = [(_ssim_global(ref_np, out_np),
                   float(np.sqrt(np.mean((ref_np - out_np) ** 2))))]
    rmse = float(np.sqrt(np.mean((ref_np - out_np) ** 2)))
    data_range = max(float(ref_np.max() - ref_np.min()), 1e-8)
    psnr = 20.0 * np.log10(data_range / max(rmse, 1e-12))
    return {
        "shape": list(ref_np.shape),
        "ssim_min": min(x[0] for x in frames),
        "ssim_median": float(np.median([x[0] for x in frames])),
        "psnr_db": float(psnr),
        "frame_ssim": [x[0] for x in frames],
        "rmse": rmse,
    }


def _stamp(root: Path) -> dict:
    from mlx_mfa import _ext

    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_mfa": importlib.metadata.version("mlx-mfa"),
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "device_info": dict(_ext.get_device_info()),
        "seedvr2_root": str(root),
        "seedvr2_git_head": os.popen(f"git -C {root} rev-parse HEAD").read().strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=("none", "attention_qk", "attention_pv", "ffn", "combined"), default="none")
    parser.add_argument("--granularity", choices=("per_tensor", "per_block32", "per_block64", "per_block128", "per_channel"), default="per_tensor")
    parser.add_argument("--seedvr2-root", type=Path, default=DEFAULT_SEEDVR2_ROOT)
    parser.add_argument("--t-lat", type=int, default=4)
    parser.add_argument("--text-tokens", type=int, default=58)
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.seedvr2_root.resolve()
    weights_dir = root / "models" / "SEEDVR2"
    dit_weights = weights_dir / "seedvr2_ema_3b_fp16.safetensors"
    vae_weights = weights_dir / "ema_vae_fp16.safetensors"
    _setup_seedvr2(root)
    mx.random.seed(20260713)
    model, _, _, _, _ = _load_dit(dit_weights, mx.bfloat16, 8192)
    vae = None if args.skip_vae else _load_vae(vae_weights)[0]

    vid = mx.random.normal((1, 33, args.t_lat, 54, 66)).astype(mx.bfloat16)
    txt = mx.random.normal((1, args.text_tokens, 5120)).astype(mx.bfloat16)
    timestep = mx.array([0.5], dtype=mx.float32)
    mx.eval(vid, txt, timestep)

    def run_model() -> mx.array:
        value = model(vid, txt, timestep)
        mx.eval(value)
        return value

    baseline = run_model()
    baseline_latent = baseline[:, :, :min(3, baseline.shape[2])]
    baseline_decoded = None
    if vae is not None:
        baseline_decoded = vae.decode(baseline_latent)
        mx.eval(baseline_decoded)

    patched = 0
    if args.site in ("ffn", "combined"):
        patched += _patch_ffn(model, args.granularity)
    if args.site in ("attention_qk", "attention_pv", "combined"):
        patched += _patch_attention(model, args.granularity, args.site)
    if patched:
        model.clear_cond_cache()
    quantized = run_model()
    _eval((baseline, quantized))
    quant_decoded = None
    if baseline_decoded is not None:
        quant_decoded = vae.decode(quantized[:, :, :min(3, quantized.shape[2])])
        mx.eval(quant_decoded)
    result = {
        "stage": "ITEM6-stage2-fake-int8",
        "site": args.site,
        "granularity": args.granularity,
        "proxy": {
            "real_seedvr2_classes_and_production_weights": True,
            "synthetic_shape_faithful_inputs": True,
            "t_lat": args.t_lat,
            "text_tokens": args.text_tokens,
            "full_archival_clip": False,
        },
        "engagement": {
            "patched_module_boundaries": patched,
            "method": "MLX fake quantize->dequantize in the selected site; no fast quant kernel claimed",
        },
        "intermediate": {
            "finite": bool(np.isfinite(np.asarray(quantized.astype(mx.float32))).all()),
            "cosine": _cosine(baseline, quantized),
            "max_abs": float(mx.max(mx.abs(baseline.astype(mx.float32) - quantized.astype(mx.float32)))),
        },
        "decoded": None if baseline_decoded is None else _decoded_metrics(
            baseline_decoded, quant_decoded
        ),
        "stamp": _stamp(root),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
