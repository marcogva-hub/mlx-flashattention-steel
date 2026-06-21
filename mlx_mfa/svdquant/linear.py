"""SVDQuantLinear — W4A16 linear layer with optional SVD low-rank correction.

Drop-in replacement for nn.Linear. Quantizes weights to INT4 (via
mx.quantized_matmul) and optionally adds an FP16 low-rank residual branch
to recover the most significant quantization error modes.

Forward pass:
    y = quantized_matmul(x, W_q, scales, biases) + x @ L2.T @ L1.T + bias

When rank=0, equivalent to nn.QuantizedLinear (no low-rank overhead).

The low-rank correction is computed offline by SVD on the quantization
residual R = W - dequant(quantize(W)). See quantize.py for calibration.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn


class SVDQuantLinear(nn.Module):
    """W4A16 linear layer with optional SVD low-rank correction.

    Args:
        in_features: input dimension K
        out_features: output dimension M
        bias: whether to add a bias term
        group_size: quantization group size (32, 64, or 128)
        bits: quantization bits (4)
        rank: SVD low-rank correction rank (0=disabled, 16/32/64 typical)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        rank: int = 0,
    ):
        super().__init__()
        # CC-06 (audit): validate at the boundary.  Previously invalid args were
        # silently absorbed — non-divisible in_features DROPPED channels (wrong
        # math, no error), bits/group_size==0 raised a bare ZeroDivisionError, and
        # rank<0 was accepted.  Fail loudly and clearly instead (RULE 8).
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"SVDQuantLinear: in_features and out_features must be > 0, got "
                f"in_features={in_features}, out_features={out_features}."
            )
        if bits <= 0 or 32 % bits != 0:
            raise ValueError(
                f"SVDQuantLinear: bits must be a positive divisor of 32 (2/4/8), got {bits}."
            )
        if group_size <= 0:
            raise ValueError(f"SVDQuantLinear: group_size must be > 0, got {group_size}.")
        elems_per_32bits = 32 // bits
        if in_features % elems_per_32bits != 0:
            raise ValueError(
                f"SVDQuantLinear: in_features ({in_features}) must be divisible by "
                f"32//bits = {elems_per_32bits} (else quantized channels are dropped)."
            )
        if in_features % group_size != 0:
            raise ValueError(
                f"SVDQuantLinear: in_features ({in_features}) must be divisible by "
                f"group_size ({group_size}) (else scale/bias channels are dropped)."
            )
        if rank < 0:
            raise ValueError(f"SVDQuantLinear: rank must be >= 0, got {rank}.")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bits = bits
        self.rank = rank

        # Quantized weight — placeholder shapes; overwritten by quantize_model
        # or load_weights. mx.quantize packs into uint32.
        self.weight = mx.zeros(
            (out_features, in_features // elems_per_32bits), dtype=mx.uint32
        )
        self.scales = mx.zeros(
            (out_features, in_features // group_size), dtype=mx.float16
        )
        self.biases = mx.zeros(
            (out_features, in_features // group_size), dtype=mx.float16
        )

        if bias:
            self.bias = mx.zeros((out_features,), dtype=mx.float16)

        # Low-rank correction (FP16)
        if rank > 0:
            self.proj_down = mx.zeros((rank, in_features), dtype=mx.float16)
            self.proj_up = mx.zeros((out_features, rank), dtype=mx.float16)

        # Optional per-channel smoothing scale (set during calibration)
        self.smooth_scale = None

    def __call__(self, x: mx.array) -> mx.array:
        # Optional channel smoothing
        if self.smooth_scale is not None:
            x = x * self.smooth_scale

        # Main INT4 branch — dequant happens in-register, no FP16 buffer
        y = mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            bits=self.bits,
            group_size=self.group_size,
            transpose=True,
        )

        # Low-rank FP16 correction: y += x @ L2.T @ L1.T
        if self.rank > 0:
            hidden = x @ self.proj_down.T  # [*, K] @ [K, rank] -> [*, rank]
            y = y + hidden @ self.proj_up.T  # [*, rank] @ [rank, M] -> [*, M]

        if "bias" in self:
            y = y + self.bias

        return y

    @property
    def memory_bytes(self) -> int:
        """Total memory footprint in bytes."""
        # INT4 packed weights: out * in * bits / 8
        w_bytes = self.out_features * self.in_features * self.bits // 8
        # Scales + biases: each [out, in/gs] in FP16 (2 bytes)
        n_groups = self.in_features // self.group_size
        sb_bytes = self.out_features * n_groups * 2 * 2  # scales + biases
        # Low-rank: (rank * in + out * rank) * 2 bytes
        lr_bytes = 0
        if self.rank > 0:
            lr_bytes = (self.rank * self.in_features + self.out_features * self.rank) * 2
        # Bias
        b_bytes = self.out_features * 2 if "bias" in self else 0
        return w_bytes + sb_bytes + lr_bytes + b_bytes

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs FP16 nn.Linear."""
        fp16_bytes = self.out_features * self.in_features * 2
        if "bias" in self:
            fp16_bytes += self.out_features * 2
        return fp16_bytes / self.memory_bytes if self.memory_bytes > 0 else 1.0

    def __repr__(self) -> str:
        bias_str = "bias=True" if "bias" in self else "bias=False"
        return (
            f"SVDQuantLinear(in={self.in_features}, out={self.out_features}, "
            f"{bias_str}, bits={self.bits}, gs={self.group_size}, "
            f"rank={self.rank}, compress={self.compression_ratio:.1f}x)"
        )
