"""SVDQuant model quantization — replace nn.Linear with SVDQuantLinear.

Offline calibration: quantize weights to INT4 and optionally compute
SVD low-rank correction on the quantization residual.

Usage:
    from mlx_mfa.svdquant import quantize_model

    stats = quantize_model(model, bits=4, group_size=64, rank=32)
    print(f"Compression: {stats['overall_compression']:.1f}x")
"""

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx
from mlx import nn

from mlx_mfa.svdquant.linear import SVDQuantLinear

# Minimum dimension for quantization — small layers stay FP16.
_MIN_DIM = 256


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 64,
    rank: int = 0,
    calibration_data: Optional[mx.array] = None,
    smooth_alpha: float = 0.5,
    class_predicate: Optional[Callable] = None,
) -> dict:
    """Quantize all linear layers in a model to SVDQuantLinear.

    Walks the nn.Module tree, replaces matching nn.Linear layers with
    SVDQuantLinear (W4A16 + optional SVD low-rank correction).

    Args:
        model: nn.Module with nn.Linear layers.
        bits: quantization bits (4).
        group_size: group size for quantization (32, 64, or 128).
        rank: SVD correction rank (0=no correction, 16/32/64 typical).
        calibration_data: optional input for smoothing calibration.
            Not yet implemented — reserved for Phase 3 autosearch.
        smooth_alpha: SmoothQuant alpha (reserved for Phase 3).
        class_predicate: function(path, module) -> bool selecting which
            layers to quantize. Default: all nn.Linear with both dims >= 256.

    Returns:
        dict with per-layer stats and overall compression ratio.
    """
    if class_predicate is None:
        class_predicate = _default_predicate

    stats: dict = {
        "layers": [],
        "total_fp16_bytes": 0,
        "total_quant_bytes": 0,
    }

    def _quantize_layer(path: str, linear: nn.Linear) -> SVDQuantLinear:
        # Repo review 2026-05: idempotence guard.  A custom class_predicate
        # matching SVDQuantLinear (already-quantized) would re-quantize the
        # packed int4 weight — silently corrupting it.  The default predicate
        # excludes these, but custom predicates must be safe too.
        if isinstance(linear, SVDQuantLinear):
            return linear
        W = linear.weight  # [M, K]
        M, K = W.shape
        has_bias = hasattr(linear, "bias") and linear.bias is not None

        # Step 1: Quantize weights via MLX native quantization
        W_q, scales, biases = mx.quantize(W, group_size=group_size, bits=bits)

        # Step 2: Optional SVD correction on quantization residual
        proj_down = None
        proj_up = None
        if rank > 0:
            W_dequant = mx.dequantize(
                W_q, scales, biases, group_size=group_size, bits=bits
            )
            residual = W - W_dequant
            # Materialize residual before numpy conversion
            mx.synchronize()

            # SVD on residual — numpy for SVD (one-time offline cost)
            import numpy as np

            R_np = np.array(residual.astype(mx.float32))
            U, S, Vt = np.linalg.svd(R_np, full_matrices=False)

            # Truncated rank-r approximation with balanced norm split
            S_sqrt = np.sqrt(S[:rank])
            L1 = U[:, :rank] * S_sqrt[None, :]  # [M, rank]
            L2 = Vt[:rank, :] * S_sqrt[:, None]  # [rank, K]

            proj_up = mx.array(L1.astype(np.float16))  # [M, rank]
            proj_down = mx.array(L2.astype(np.float16))  # [rank, K]

            # Measure error reduction
            reconstructed = W_dequant + mx.array((L1 @ L2).astype(np.float16))
            err_before = float(mx.max(mx.abs(W - W_dequant)))
            err_after = float(mx.max(mx.abs(W - reconstructed)))
        else:
            W_dequant = mx.dequantize(
                W_q, scales, biases, group_size=group_size, bits=bits
            )
            err_before = float(mx.max(mx.abs(W - W_dequant)))
            err_after = err_before

        # Step 3: Create SVDQuantLinear and populate weights
        svdq = SVDQuantLinear(
            in_features=K,
            out_features=M,
            bias=has_bias,
            group_size=group_size,
            bits=bits,
            rank=rank,
        )
        svdq.weight = W_q
        svdq.scales = scales
        svdq.biases = biases
        if has_bias:
            svdq.bias = linear.bias
        if rank > 0:
            svdq.proj_down = proj_down
            svdq.proj_up = proj_up

        fp16_bytes = M * K * 2
        stats["layers"].append(
            {
                "path": path,
                "shape": (M, K),
                "rank": rank,
                "err_before": err_before,
                "err_after": err_after,
                "compression": fp16_bytes / svdq.memory_bytes,
            }
        )
        stats["total_fp16_bytes"] += fp16_bytes
        stats["total_quant_bytes"] += svdq.memory_bytes

        return svdq

    _replace_layers(model, class_predicate, _quantize_layer)

    stats["overall_compression"] = (
        stats["total_fp16_bytes"] / stats["total_quant_bytes"]
        if stats["total_quant_bytes"] > 0
        else 1.0
    )
    return stats


def _default_predicate(path: str, module: nn.Module) -> bool:
    """Default predicate: quantize nn.Linear layers with both dims >= 256."""
    if not isinstance(module, nn.Linear):
        return False
    M, K = module.weight.shape
    return M >= _MIN_DIM and K >= _MIN_DIM


def _replace_layers(
    model: nn.Module,
    predicate: Callable,
    replacer: Callable,
    prefix: str = "",
) -> None:
    """Walk nn.Module tree and replace layers matching predicate in-place.

    Uses module.children().items() for discovery, but mutates via getattr/
    setattr on the original model — because MLX children() returns COPIES
    (not references), so mutating the returned dict/list has no effect.
    """
    for name, child in model.children().items():
        full_path = f"{prefix}.{name}" if prefix else name

        # III-4 pass-7 F7-1 FIX: `nn.Module` IS a `dict` subclass
        # (issubclass(nn.Linear, dict) is True), so the dict branch must
        # NOT run first — it would treat a direct `nn.Linear` attribute
        # (the most common model structure) as a container, iterate its
        # weight/bias arrays (not Modules), and replace NOTHING — while
        # quantize_model reported success (silent no-op).  Check the
        # nn.Module branch FIRST; the dict/list branches then handle only
        # genuine non-Module containers of submodules.
        if isinstance(child, nn.Module):
            if predicate(full_path, child):
                setattr(model, name, replacer(full_path, child))
            else:
                _replace_layers(child, predicate, replacer, prefix=full_path)
        elif isinstance(child, dict):
            # Dict children (e.g., named submodules)
            real_dict = getattr(model, name)
            for k, v in real_dict.items():
                path = f"{full_path}.{k}"
                if isinstance(v, nn.Module) and predicate(path, v):
                    real_dict[k] = replacer(path, v)
                elif isinstance(v, nn.Module):
                    _replace_layers(v, predicate, replacer, prefix=path)
        elif isinstance(child, list):
            # List children (e.g., nn.Sequential.layers)
            real_list = getattr(model, name)
            for i, v in enumerate(real_list):
                path = f"{full_path}.{i}"
                if isinstance(v, nn.Module) and predicate(path, v):
                    real_list[i] = replacer(path, v)
                elif isinstance(v, nn.Module):
                    _replace_layers(v, predicate, replacer, prefix=path)
