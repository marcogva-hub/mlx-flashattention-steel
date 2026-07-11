#!/usr/bin/env python3
"""Profile SeedVR2 with real MLX classes/weights and shape-faithful inputs.

The archival portfolio run is too long for five complete repetitions. This
harness measures sustained production units (VAE chunks and one DiT batch),
then extrapolates the 895-frame reference workload. Component timings insert
MLX eval barriers and are diagnostic rather than an additive reconstruction of
the lazy production graph.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
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
DEFAULT_FRAMES = 895
DEFAULT_PADDED_FRAMES = 897
DEFAULT_WIDTH = 528
DEFAULT_HEIGHT = 432
DEFAULT_T_LAT = 225
DEFAULT_DIT_BATCH_T = 38
DEFAULT_TEXT_TOKENS = 58
DEFAULT_SESSIONS = 5
DEFAULT_WARMUP = 1
DEFAULT_CHUNK_TOKENS = 8192


def _stats(samples: list[float]) -> dict[str, Any]:
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(statistics.median(samples)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(statistics.mean(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "sample_count": len(samples),
        "samples_ms": samples,
    }


def _arrays(value: Any) -> list[mx.array]:
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (tuple, list)):
        result: list[mx.array] = []
        for item in value:
            result.extend(_arrays(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_arrays(item))
        return result
    return []


def _eval(value: Any) -> None:
    arrays = _arrays(value)
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


def _finite(value: Any) -> bool:
    return all(
        bool(np.isfinite(np.asarray(x.astype(mx.float32))).all())
        for x in _arrays(value)
    )


def _cosine(a: mx.array, b: mx.array) -> float:
    af = np.asarray(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    bf = np.asarray(b.astype(mx.float32)).reshape(-1).astype(np.float64)
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / denom) if denom else 1.0


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


def _stamp() -> dict[str, Any]:
    from mlx_mfa import _ext

    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_mfa": importlib.metadata.version("mlx-mfa"),
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "device_info": dict(_ext.get_device_info()),
        "has_nax": bool(_ext.device_has_neural_accelerators()),
    }


def _setup_seedvr2(root: Path) -> None:
    sys.path.insert(0, str(root / "src" / "mlx_native"))
    sys.path.insert(0, str(root / "src"))
    # SeedVR2's weight package imports its model-saving utility, which imports
    # third-party ``toml`` only to read a version string. The canonical mlx-mfa
    # venv intentionally lacks that optional package. Provide the read-only
    # subset through Python 3.11's stdlib tomllib; no dependency is installed.
    if "toml" not in sys.modules:
        toml_shim = types.ModuleType("toml")

        def load_toml(path: str | Path) -> dict[str, Any]:
            with Path(path).open("rb") as handle:
                return tomllib.load(handle)

        toml_shim.load = load_toml  # type: ignore[attr-defined]
        sys.modules["toml"] = toml_shim


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


def _load_dit(weights_path: Path, dtype: mx.Dtype, chunk_tokens: int):
    from mflux.models.seedvr2.model.seedvr2_transformer.transformer import (
        SeedVR2Transformer,
        apply_chunked_linears,
    )
    from mflux.models.seedvr2.weights.seedvr2_weight_mapping import (
        SeedVR2WeightMapping,
    )

    model = SeedVR2Transformer()
    raw = mx.load(str(weights_path))
    mapped: dict[str, mx.array] = {}
    for target in SeedVR2WeightMapping.get_transformer_mapping():
        if "{block}" in target.to_pattern:
            limit = target.max_blocks if target.max_blocks is not None else model.num_layers
            block_ids = range(limit)
        else:
            block_ids = (None,)
        for block_id in block_ids:
            to_key = target.to_pattern
            if block_id is not None:
                to_key = to_key.replace("{block}", str(block_id))
            for pattern in target.from_pattern:
                from_key = pattern
                if block_id is not None:
                    from_key = from_key.replace("{block}", str(block_id))
                if from_key in raw:
                    value = raw[from_key]
                    if target.transform is not None:
                        value = target.transform(value)
                    mapped[to_key] = value
                    break
    model.load_weights(list(mapped.items()), strict=False)

    def cast_tree(tree: Any) -> Any:
        if isinstance(tree, mx.array) and tree.dtype in (mx.float16, mx.float32):
            return tree.astype(dtype)
        if isinstance(tree, dict):
            return {key: cast_tree(value) for key, value in tree.items()}
        if isinstance(tree, list):
            return [cast_tree(value) for value in tree]
        return tree

    model.update(cast_tree(model.parameters()))
    wrapped = apply_chunked_linears(model, chunk_tokens)
    representative_linear = model.blocks[0].attn.proj_qkv_vid
    instance_dunder_only = "__call__" in getattr(representative_linear, "__dict__", {})
    mx.eval(model.parameters())
    return model, len(raw), len(mapped), wrapped, instance_dunder_only


def _archive_dit_extents(total_t: int, core_t: int = 30, overlap: int = 4) -> list[int]:
    extents: list[int] = []
    for index in range(math.ceil(total_t / core_t)):
        core_start = index * core_t
        core_end = min(core_start + core_t, total_t)
        ext_start = max(0, core_start - overlap)
        ext_end = min(total_t, core_end + overlap)
        extents.append(ext_end - ext_start)
    return extents


def _profile_vae(
    vae: Any,
    *,
    height: int,
    width: int,
    sessions: int,
    warmup: int,
    cooldown_s: float,
) -> dict[str, Any]:
    import mlx_mfa
    from mlx_mfa import _ext
    from mflux.models.seedvr2.model.seedvr2_vae.common import attention_3d as attn_mod
    from mflux.models.seedvr2.model.seedvr2_vae.common import conv3d as conv_mod
    from mflux.models.seedvr2.model.seedvr2_vae.common import linear_conv1x1 as lin_mod
    from mflux.models.seedvr2.model.seedvr2_vae.decoder import decoder_resnet_block_3d as dec_resnet
    from mflux.models.seedvr2.model.seedvr2_vae.encoder import resnet_block_3d as enc_resnet

    x = mx.random.normal((1, 3, 5, height, width)).astype(mx.float16)
    z = mx.random.normal((1, 16, 3, height // 8, width // 8)).astype(mx.float16)
    mx.eval(x, z)

    def encode_unit():
        return vae.encode(x)

    def decode_unit():
        return vae.decode(z)

    encoded = encode_unit()
    decoded = decode_unit()
    _eval((encoded, decoded))
    encoded_repeat = encode_unit()
    decoded_repeat = decode_unit()
    _eval((encoded_repeat, decoded_repeat))

    sdpa_calls = 0
    v6_calls = 0
    original_sdpa = mx.fast.scaled_dot_product_attention
    original_v6 = _ext.v6_nax_forward

    def counted_sdpa(*args, **kwargs):
        nonlocal sdpa_calls
        sdpa_calls += 1
        return original_sdpa(*args, **kwargs)

    def counted_v6(*args, **kwargs):
        nonlocal v6_calls
        v6_calls += 1
        return original_v6(*args, **kwargs)

    mlx_mfa.reset_hook_stats()
    mx.fast.scaled_dot_product_attention = counted_sdpa
    _ext.v6_nax_forward = counted_v6
    try:
        _eval((encode_unit(), decode_unit()))
    finally:
        mx.fast.scaled_dot_product_attention = original_sdpa
        _ext.v6_nax_forward = original_v6
    hook_stats = mlx_mfa.get_hook_stats()

    wall = {
        "encode_5_frames": _time_arm(
            encode_unit, warmup=warmup, sessions=sessions, cooldown_s=cooldown_s
        ),
        "decode_3_latents": _time_arm(
            decode_unit, warmup=warmup, sessions=sessions, cooldown_s=cooldown_s
        ),
    }

    categories: dict[str, list[float]] = defaultdict(list)
    originals = {
        "conv": conv_mod.CausalConv3d.__call__,
        "linear": lin_mod.LinearConv1x1x1.__call__,
        "attention": attn_mod.Attention3D.__call__,
        "dec_norm_silu": dec_resnet._fused_norm_silu,
        "enc_norm_silu": enc_resnet._fused_norm_silu,
    }

    def wrap(category: str, original: Callable[..., Any]):
        def timed(module, *args, **kwargs):
            mx.synchronize()
            start = time.perf_counter()
            result = original(module, *args, **kwargs)
            _eval(result)
            categories[category].append((time.perf_counter() - start) * 1000.0)
            return result

        return timed

    conv_mod.CausalConv3d.__call__ = wrap("conv3d", originals["conv"])
    lin_mod.LinearConv1x1x1.__call__ = wrap("linear_1x1", originals["linear"])
    attn_mod.Attention3D.__call__ = wrap("attention3d", originals["attention"])
    def timed_norm_silu(original: Callable[..., Any], *args, **kwargs):
        mx.synchronize()
        start = time.perf_counter()
        result = original(*args, **kwargs)
        _eval(result)
        categories["groupnorm_silu"].append((time.perf_counter() - start) * 1000.0)
        return result

    dec_resnet._fused_norm_silu = lambda *args, **kwargs: timed_norm_silu(
        originals["dec_norm_silu"], *args, **kwargs
    )
    enc_resnet._fused_norm_silu = lambda *args, **kwargs: timed_norm_silu(
        originals["enc_norm_silu"], *args, **kwargs
    )
    diagnostic_walls: list[float] = []
    try:
        for _ in range(sessions):
            start = time.perf_counter()
            _eval(encode_unit())
            _eval(decode_unit())
            diagnostic_walls.append((time.perf_counter() - start) * 1000.0)
    finally:
        conv_mod.CausalConv3d.__call__ = originals["conv"]
        lin_mod.LinearConv1x1x1.__call__ = originals["linear"]
        attn_mod.Attention3D.__call__ = originals["attention"]
        dec_resnet._fused_norm_silu = originals["dec_norm_silu"]
        enc_resnet._fused_norm_silu = originals["enc_norm_silu"]

    category_totals = {
        key: {
            "total_ms": float(sum(values)),
            "per_session_ms": float(sum(values) / sessions),
            "call_count": len(values),
            "calls_per_session": len(values) / sessions,
        }
        for key, values in categories.items()
    }
    diagnostic_attributed_ms = sum(
        item["per_session_ms"] for item in category_totals.values()
    )
    diagnostic_wall = _stats(diagnostic_walls)
    diagnostic_other_ms = max(
        0.0, diagnostic_wall["median_ms"] - diagnostic_attributed_ms
    )
    category_totals["other_unattributed"] = {
        "total_ms": diagnostic_other_ms * sessions,
        "per_session_ms": diagnostic_other_ms,
        "call_count": 0,
        "calls_per_session": 0.0,
        "note": "Residual of the instrumented encode+decode wall after non-overlapping wrappers.",
    }
    diagnostic_basis = diagnostic_attributed_ms + diagnostic_other_ms
    diagnostic_shares = {
        key: value["per_session_ms"] / diagnostic_basis
        for key, value in category_totals.items()
    }

    return {
        "wall": wall,
        "validation": {
            "encode_finite": _finite(encoded),
            "decode_finite": _finite(decoded),
            "encode_repeat_cos": _cosine(encoded, encoded_repeat),
            "decode_repeat_cos": _cosine(decoded, decoded_repeat),
        },
        "which_binary": {
            "attention3d_public_flash_calls_expected": 3,
            "sdpa_calls": sdpa_calls,
            "v6_nax_forward_calls": v6_calls,
            "interpretation": "Attention3D D=512 enters mlx-mfa public API then falls back to MLX SDPA.",
            "conv_hook_stats": hook_stats,
        },
        "diagnostic_categories_encode_plus_decode": category_totals,
        "diagnostic_wall_encode_plus_decode": diagnostic_wall,
        "diagnostic_category_share": diagnostic_shares,
        "shapes": {
            "encode": list(x.shape),
            "decode": list(z.shape),
            "attention3d_head_dim": 512,
        },
    }


def _profile_dit(
    model: Any,
    *,
    t_lat: int,
    text_tokens: int,
    dtype: mx.Dtype,
    sessions: int,
    warmup: int,
    cooldown_s: float,
) -> dict[str, Any]:
    import sys

    from mlx_mfa import _ext
    from mflux.models.seedvr2.model.seedvr2_transformer.transformer_block import (
        _cached_rms_norm,
    )

    vid = mx.random.normal((1, 33, t_lat, 54, 66)).astype(dtype)
    txt = mx.random.normal((1, text_tokens, 5120)).astype(dtype)
    timestep = mx.array([0.5], dtype=mx.float32)
    mx.eval(vid, txt, timestep)

    def full_dit():
        return model(vid, txt, timestep)

    varlen_calls = 0
    sdpa_calls = 0
    public_varlen_calls = 0
    public_qkv_dtypes: list[str] = []
    original_varlen = _ext.mfa_attention_varlen_forward
    original_sdpa = mx.fast.scaled_dot_product_attention
    attention_module = sys.modules[model.blocks[0].attn.__class__.__module__]
    original_public_varlen = attention_module._flash_attention_varlen_qkv_packed

    def counted_varlen(*args, **kwargs):
        nonlocal varlen_calls
        varlen_calls += 1
        return original_varlen(*args, **kwargs)

    def counted_sdpa(*args, **kwargs):
        nonlocal sdpa_calls
        sdpa_calls += 1
        return original_sdpa(*args, **kwargs)

    def counted_public_varlen(qkv, *args, **kwargs):
        nonlocal public_varlen_calls
        public_varlen_calls += 1
        public_qkv_dtypes.append(str(qkv.dtype))
        return original_public_varlen(qkv, *args, **kwargs)

    _ext.mfa_attention_varlen_forward = counted_varlen
    mx.fast.scaled_dot_product_attention = counted_sdpa
    attention_module._flash_attention_varlen_qkv_packed = counted_public_varlen
    try:
        probe = full_dit()
        _eval(probe)
    finally:
        _ext.mfa_attention_varlen_forward = original_varlen
        mx.fast.scaled_dot_product_attention = original_sdpa
        attention_module._flash_attention_varlen_qkv_packed = original_public_varlen
    repeat = full_dit()
    _eval(repeat)
    if public_varlen_calls != model.num_layers:
        raise RuntimeError(
            "which-binary failed: SeedVR2 did not enter the expected public varlen API; "
            f"expected {model.num_layers}, got {public_varlen_calls}"
        )

    full_stats = _time_arm(
        full_dit, warmup=warmup, sessions=sessions, cooldown_s=cooldown_s
    )

    vid_emb, vid_shape = model.vid_in(vid)
    txt_emb = model.txt_in(txt)
    txt_shape = mx.full((1, 1), text_tokens, dtype=mx.int32)
    emb = model.emb_in(timestep).reshape(-1, model.vid_dim, 2, 3)
    mx.eval(vid_emb, vid_shape, txt_emb, txt_shape, emb)
    block = model.blocks[0]
    vid_norm = _cached_rms_norm(vid_emb, block.norm_eps)
    txt_norm = _cached_rms_norm(txt_emb, block.norm_eps)
    vid_mod = block.ada.modulate_vid(vid_norm, emb, layer="attn", mode="in")
    txt_mod = block.ada.modulate_txt(txt_norm, emb, layer="attn", mode="in")
    mx.eval(vid_mod, txt_mod)

    def block_full():
        return block(vid_emb, txt_emb, emb, vid_shape, txt_shape)

    def attention_full():
        return block.attn(vid_mod, txt_mod, vid_shape, txt_shape)

    def ffn_full():
        return block.mlp(vid_mod, txt_mod)

    def qkv_bundle():
        return (
            block.attn.proj_qkv_vid(vid_mod.reshape(-1, model.vid_dim)),
            block.attn.proj_qkv_txt(txt_mod.reshape(-1, model.txt_dim)),
        )

    def norm_ada_bundle():
        vn1 = _cached_rms_norm(vid_emb, block.norm_eps)
        tn1 = _cached_rms_norm(txt_emb, block.norm_eps)
        vn2 = _cached_rms_norm(vid_emb, block.norm_eps)
        tn2 = _cached_rms_norm(txt_emb, block.norm_eps)
        return (
            block.ada.modulate_vid(vn1, emb, layer="attn", mode="in"),
            block.ada.modulate_txt(tn1, emb, layer="attn", mode="in"),
            block.ada.modulate_vid(vn2, emb, layer="mlp", mode="in"),
            block.ada.modulate_txt(tn2, emb, layer="mlp", mode="in"),
        )

    block_stats = {
        "block_full": _time_arm(
            block_full, warmup=warmup, sessions=sessions, cooldown_s=cooldown_s
        ),
        "joint_attention_total": _time_arm(
            attention_full, warmup=warmup, sessions=sessions, cooldown_s=cooldown_s
        ),
        "ffn_swiglu_total": _time_arm(
            ffn_full, warmup=warmup, sessions=sessions, cooldown_s=cooldown_s
        ),
        "qkv_projection_bundle": _time_arm(
            qkv_bundle, warmup=warmup, sessions=sessions, cooldown_s=cooldown_s
        ),
        "norm_ada_bundle": _time_arm(
            norm_ada_bundle, warmup=warmup, sessions=sessions, cooldown_s=cooldown_s
        ),
    }

    kernel_samples: list[float] = []

    def timed_varlen(*args, **kwargs):
        mx.synchronize()
        start = time.perf_counter()
        result = original_varlen(*args, **kwargs)
        _eval(result)
        kernel_samples.append((time.perf_counter() - start) * 1000.0)
        return result

    if varlen_calls > 0:
        _ext.mfa_attention_varlen_forward = timed_varlen
        try:
            for _ in range(sessions):
                _eval(attention_full())
        finally:
            _ext.mfa_attention_varlen_forward = original_varlen
        kernel_label = "varlen_native_kernel_calls_diagnostic"
    else:
        def timed_sdpa(*args, **kwargs):
            mx.synchronize()
            start = time.perf_counter()
            result = original_sdpa(*args, **kwargs)
            _eval(result)
            kernel_samples.append((time.perf_counter() - start) * 1000.0)
            return result

        mx.fast.scaled_dot_product_attention = timed_sdpa
        try:
            for _ in range(sessions):
                _eval(attention_full())
        finally:
            mx.fast.scaled_dot_product_attention = original_sdpa
        kernel_label = "sdpa_window_kernel_calls_diagnostic"
    block_stats[kernel_label] = _stats(kernel_samples)

    return {
        "wall": {"one_dit_batch": full_stats},
        "block": block_stats,
        "validation": {
            "finite": _finite(probe),
            "repeat_cos": _cosine(probe, repeat),
            "dtype_trace": {
                "raw_video": str(vid.dtype),
                "video_embedding": str(vid_emb.dtype),
                "time_embedding": str(emb.dtype),
                "video_norm": str(vid_norm.dtype),
                "video_after_ada": str(vid_mod.dtype),
                "public_packed_qkv": sorted(set(public_qkv_dtypes)),
            },
        },
        "which_binary": {
            "seedvr2_public_varlen_calls": public_varlen_calls,
            "mfa_attention_varlen_forward_calls": varlen_calls,
            "sdpa_calls": sdpa_calls,
            "expected_calls": model.num_layers,
            "public_qkv_dtypes": sorted(set(public_qkv_dtypes)),
            "kernel": (
                "STEEL mfa_attention_varlen_forward (D=128)"
                if varlen_calls > 0
                else "MLX SDPA split-concat fallback inside public varlen API"
            ),
        },
        "shapes": {
            "raw_video": list(vid.shape),
            "video_tokens": int(t_lat * 27 * 33),
            "text_tokens": text_tokens,
            "heads": 20,
            "head_dim": 128,
            "width": 2560,
            "swiglu_hidden": 6912,
            "layers": model.num_layers,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seedvr2-root", type=Path, default=DEFAULT_SEEDVR2_ROOT)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--padded-frames", type=int, default=DEFAULT_PADDED_FRAMES)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--total-t-lat", type=int, default=DEFAULT_T_LAT)
    parser.add_argument("--dit-t-lat", type=int, default=DEFAULT_DIT_BATCH_T)
    parser.add_argument("--text-tokens", type=int, default=DEFAULT_TEXT_TOKENS)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--profile", choices=("smoke", "final"), default="smoke")
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--skip-dit", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.profile == "smoke":
        args.sessions = 1
        args.warmup = 0
        args.width = 64
        args.height = 64
        args.dit_t_lat = 1
        args.text_tokens = 8

    root = args.seedvr2_root.resolve()
    weights_dir = root / "models" / "SEEDVR2"
    dit_weights = weights_dir / "seedvr2_ema_3b_fp16.safetensors"
    vae_weights = weights_dir / "ema_vae_fp16.safetensors"
    for path in (dit_weights, vae_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    _setup_seedvr2(root)
    mx.random.seed(20260711)

    result: dict[str, Any] = {
        "schema": "mlx-mfa.e2e-profile.v1",
        "model": "SeedVR2-3B",
        "scope": {
            "kind": "real SeedVR2 MLX classes and production weights; synthetic shape-faithful activations",
            "proxy": True,
            "reference_archival_workload": {
                "frames": args.frames,
                "padded_frames": args.padded_frames,
                "resolution": [args.width, args.height],
                "total_t_lat": args.total_t_lat,
                "one_step": True,
                "vae_dtype": "fp16",
                "dit_dtype": "bf16",
            },
            "not_included": [
                "video I/O",
                "color correction",
                "quality metrics",
                "full 895-frame graph repeated five times",
            ],
            "seedvr2_root": str(root),
            "seedvr2_git_head": os.popen(f"git -C {root} rev-parse HEAD").read().strip(),
        },
        "methodology": {
            "sessions": args.sessions,
            "warmup": args.warmup,
            "cooldown_s": args.cooldown,
            "per_arm_sustained": True,
            "component_eval_barriers_change_lazy_schedule": True,
            "phase_extrapolation_from_production_units": True,
        },
        "stamp": _stamp(),
        "weights": {
            "dit": str(dit_weights),
            "vae": str(vae_weights),
        },
    }

    if not args.skip_vae:
        vae, vae_weight_count, replaced = _load_vae(vae_weights)
        result["vae"] = _profile_vae(
            vae,
            height=args.height,
            width=args.width,
            sessions=args.sessions,
            warmup=args.warmup,
            cooldown_s=args.cooldown,
        )
        result["vae"]["loader"] = {
            "weight_count": vae_weight_count,
            "conv1x1_replaced": replaced,
        }
        del vae
        mx.clear_cache()

    if not args.skip_dit:
        model, raw_count, mapped_count, wrapped, instance_dunder_only = _load_dit(
            dit_weights, mx.bfloat16, args.chunk_tokens
        )
        result["dit"] = _profile_dit(
            model,
            t_lat=args.dit_t_lat,
            text_tokens=args.text_tokens,
            dtype=mx.bfloat16,
            sessions=args.sessions,
            warmup=args.warmup,
            cooldown_s=args.cooldown,
        )
        result["dit"]["loader"] = {
            "raw_weight_count": raw_count,
            "mapped_weight_count": mapped_count,
            "chunked_linears_reported": wrapped,
            "chunked_linears_engaged": not instance_dunder_only,
            "chunking_note": (
                "apply_chunked_linears assigns instance __call__; Python special-method lookup bypasses it."
                if instance_dunder_only
                else "wrapper engagement verified by replacement type"
            ),
            "chunk_tokens": args.chunk_tokens,
        }
        del model
        mx.clear_cache()

    if "vae" in result and "dit" in result:
        extents = _archive_dit_extents(args.total_t_lat)
        encode_unit = result["vae"]["wall"]["encode_5_frames"]["median_ms"]
        decode_unit = result["vae"]["wall"]["decode_3_latents"]["median_ms"]
        dit_unit = result["dit"]["wall"]["one_dit_batch"]["median_ms"]
        encode_scale = args.padded_frames / 5.0
        decode_scale = args.total_t_lat / 3.0
        dit_scale = sum(extents) / args.dit_t_lat
        phases = {
            "vae_encode": encode_unit * encode_scale,
            "dit_one_step": dit_unit * dit_scale,
            "vae_decode": decode_unit * decode_scale,
        }
        total = sum(phases.values())
        result["archive_extrapolation"] = {
            "dit_batch_extents_t_lat": extents,
            "processed_dit_t_lat_with_overlap": sum(extents),
            "scale_factors": {
                "vae_encode": encode_scale,
                "dit_one_step": dit_scale,
                "vae_decode": decode_scale,
            },
            "phase_median_ms": phases,
            "phase_share": {key: value / total for key, value in phases.items()},
            "total_median_ms": total,
            "caveat": "Linear extrapolation from sustained units; not a full-graph wall-clock reconstruction.",
        }

    validations: list[bool] = []
    if "vae" in result:
        validations.extend(
            [
                result["vae"]["validation"]["encode_finite"],
                result["vae"]["validation"]["decode_finite"],
                result["vae"]["validation"]["encode_repeat_cos"] >= 0.999,
                result["vae"]["validation"]["decode_repeat_cos"] >= 0.999,
                result["vae"]["which_binary"]["sdpa_calls"] > 0,
                result["vae"]["which_binary"]["v6_nax_forward_calls"] == 0,
            ]
        )
    if "dit" in result:
        validations.extend(
            [
                result["dit"]["validation"]["finite"],
                result["dit"]["validation"]["repeat_cos"] >= 0.999,
                result["dit"]["which_binary"]["seedvr2_public_varlen_calls"]
                == result["dit"]["which_binary"]["expected_calls"],
            ]
        )
    result["all_validation_gates_passed"] = all(validations)
    if not result["all_validation_gates_passed"]:
        raise RuntimeError("SeedVR2 profile validation or which-binary gate failed")

    out = args.out
    if out is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = Path("benchmarks/results") / f"audit_e2e_seedvr2_{args.profile}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
