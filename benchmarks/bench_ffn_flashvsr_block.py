#!/usr/bin/env python3
"""Real-weight FlashVSR block A/B for expert fused NAX FFN."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import types
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx_mfa
from mlx_mfa import _ext


ROOT = Path("/Users/marcomarcelino/code/FlashVSR")


def cosine(a, b):
    af = np.asarray(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    bf = np.asarray(b.astype(mx.float32)).reshape(-1).astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))


def timing(fn, sessions, iters):
    for _ in range(5):
        mx.eval(fn())
    samples = []
    for _ in range(sessions):
        start = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / iters)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples), "samples_ms": samples}


def to_fp16(model):
    def cv(tree):
        if isinstance(tree, mx.array):
            return tree.astype(mx.float16) if tree.dtype == mx.float32 else tree
        if isinstance(tree, dict):
            return {k: cv(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [cv(v) for v in tree]
        return tree
    model.update(cv(model.parameters()))


class NAXFFN(nn.Module):
    def __init__(self, original):
        super().__init__()
        up, _gelu, down = original.net.layers
        self.up_weight, self.up_bias = up.weight, up.bias
        self.down_weight, self.down_bias = down.weight, down.bias

    def __call__(self, x):
        x = x.astype(self.up_weight.dtype) if x.dtype != self.up_weight.dtype else x
        hidden = _ext.v6_nax_linear(x, self.up_weight, self.up_bias, True)
        return _ext.v6_nax_linear(hidden, self.down_weight, self.down_bias, False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--ffn-iters", type=int, default=10)
    parser.add_argument("--block-iters", type=int, default=3)
    parser.add_argument("--order", choices=("baseline-first", "nax-first"), default="baseline-first")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from mlx_wan_dit import WanModel, modulate, sinusoidal_embedding_1d

    model = WanModel(dim=1536, num_layers=30, num_heads=12)
    model.load_weights(str(ROOT / "model/wan_dit_mlx_v9.safetensors"))
    to_fp16(model)
    block = model.blocks[0]
    mx.random.seed(12001)
    x = mx.random.normal((1, 8, 32, 32, 16)).astype(mx.float16)
    context = mx.random.normal((1, 226, 4096)).astype(mx.float16)
    timestep = mx.array([1000.0], dtype=mx.float32)
    t_emb = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, timestep))
    t_mod = model.time_projection(t_emb).reshape(1, 6, model.dim)
    ctx = model.text_embedding(context)
    block.cross_attn.init_context(ctx)
    x_block = model.patchify(x)
    cos_rope, sin_rope = model.rope.get_freqs_3d(8, 16, 16, 128, f_offset=0)
    mx.eval(x, context, t_emb, t_mod, ctx, x_block, cos_rope, sin_rope)

    params = t_mod + block.modulation
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, _gate_mlp = [
        params[:, i, None, None, None, :] for i in range(6)
    ]
    x_mod = modulate(block.norm1(x_block), shift_msa, scale_msa)
    self_out = block.self_attn(x_mod, cos_rope, sin_rope, is_stream=False, topk_ratio=2.0, local_range=11, mfa_lcsa_mode="on")
    x_after_self = x_block + gate_msa * self_out
    x_after_cross = x_after_self + block.cross_attn(block.norm3(x_after_self))
    ffn_input = modulate(block.norm2(x_after_cross), shift_mlp, scale_mlp)
    mx.eval(ffn_input)

    original_ffn = block.ffn
    nax_ffn = NAXFFN(original_ffn)
    up, _gelu, down = original_ffn.net.layers
    ffn_input_fp16 = ffn_input.astype(up.weight.dtype) if ffn_input.dtype != up.weight.dtype else ffn_input
    baseline_ffn = original_ffn(ffn_input_fp16)
    target_ffn = nax_ffn(ffn_input_fp16)
    fp32 = nn.gelu_approx(ffn_input_fp16.astype(mx.float32) @ up.weight.astype(mx.float32).T + up.bias.astype(mx.float32))
    fp32 = fp32 @ down.weight.astype(mx.float32).T + down.bias.astype(mx.float32)
    mx.eval(baseline_ffn, target_ffn, fp32)
    ffn_cos = cosine(target_ffn, fp32)
    if ffn_cos < 0.999:
        raise RuntimeError(f"FFN cosine failed: {ffn_cos}")

    if args.order == "nax-first":
        target_ffn_time = timing(lambda: nax_ffn(ffn_input_fp16), args.sessions, args.ffn_iters)
        baseline_ffn_time = timing(lambda: original_ffn(ffn_input_fp16), args.sessions, args.ffn_iters)
    else:
        baseline_ffn_time = timing(lambda: original_ffn(ffn_input_fp16), args.sessions, args.ffn_iters)
        target_ffn_time = timing(lambda: nax_ffn(ffn_input_fp16), args.sessions, args.ffn_iters)

    def baseline_block():
        block.ffn = original_ffn
        return block(x_block, t_mod, cos_rope, sin_rope, is_stream=False, topk_ratio=2.0, local_range=11, mfa_lcsa_mode="on")

    def target_block():
        block.ffn = nax_ffn
        return block(x_block, t_mod, cos_rope, sin_rope, is_stream=False, topk_ratio=2.0, local_range=11, mfa_lcsa_mode="on")

    base_block_out, target_block_out = baseline_block(), target_block()
    mx.eval(base_block_out, target_block_out)
    block_cos = cosine(target_block_out, base_block_out)
    if block_cos < 0.999:
        raise RuntimeError(f"block cosine failed: {block_cos}")

    counts = {"linear": 0, "sparse": 0}
    real_linear = _ext.v6_nax_linear
    import mlx_mfa.lcsa_nax as lcsa_nax
    real_sparse = lcsa_nax._ext.sparse_attention_forward
    def counted_linear(*a, **kw):
        counts["linear"] += 1
        return real_linear(*a, **kw)
    def counted_sparse(*a, **kw):
        counts["sparse"] += 1
        return real_sparse(*a, **kw)
    _ext.v6_nax_linear = counted_linear
    lcsa_nax._ext.sparse_attention_forward = counted_sparse
    try:
        mx.eval(target_block())
    finally:
        _ext.v6_nax_linear = real_linear
        lcsa_nax._ext.sparse_attention_forward = real_sparse
    if counts != {"linear": 2, "sparse": 1}:
        raise RuntimeError(f"which-binary counts failed: {counts}")

    if args.order == "nax-first":
        target_block_time = timing(target_block, args.sessions, args.block_iters)
        baseline_block_time = timing(baseline_block, args.sessions, args.block_iters)
    else:
        baseline_block_time = timing(baseline_block, args.sessions, args.block_iters)
        target_block_time = timing(target_block, args.sessions, args.block_iters)
    payload = {
        "mlx": mx.__version__, "mlx_mfa": mlx_mfa.__version__, "device": mlx_mfa.get_device_info(),
        "workload": "FlashVSR block0 real weights, synthetic shape-faithful activations", "order": args.order,
        "dtypes": {"block_input": str(x_block.dtype), "ffn_input_before_cast": str(ffn_input.dtype), "weight": str(up.weight.dtype)},
        "correctness": {"ffn_cos_fp32": ffn_cos, "block_cos_baseline": block_cos},
        "which_binary": counts,
        "ffn": {"baseline": baseline_ffn_time, "nax": target_ffn_time, "baseline_over_nax": baseline_ffn_time["median_ms"] / target_ffn_time["median_ms"]},
        "block": {"baseline": baseline_block_time, "nax": target_block_time, "baseline_over_nax": baseline_block_time["median_ms"] / target_block_time["median_ms"]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
