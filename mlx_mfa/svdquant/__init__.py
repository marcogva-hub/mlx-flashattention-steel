"""mlx_mfa.svdquant — W4A16 quantization with optional SVD low-rank correction.

Phase 1: Unfused baseline using mx.quantized_matmul + FP16 low-rank matmuls.
Phase 2 (future): Fused Metal kernel via mx.fast.metal_kernel.

Quick start::

    from mlx_mfa.svdquant import SVDQuantLinear, quantize_model

    # Quantize all large linear layers to W4 with rank-32 SVD correction
    stats = quantize_model(model, bits=4, group_size=64, rank=32)

    # Or use SVDQuantLinear directly as a drop-in for nn.Linear
    layer = SVDQuantLinear(2560, 2560, rank=32)
"""

from mlx_mfa.svdquant.linear import SVDQuantLinear
from mlx_mfa.svdquant.quantize import quantize_model

__all__ = ["SVDQuantLinear", "quantize_model"]
