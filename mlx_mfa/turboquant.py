"""TurboQuant KV cache compression — Phase 1 (non-fused).

Two-stage vector quantization for KV cache compression:
  Stage 1 (PolarQuant/MSE): random rotation + Lloyd-Max scalar quantization
  Stage 2 (QJL): 1-bit random projection of residual for dot-product bias correction

Reference: TurboQuant (Google, ICLR 2026) — https://arxiv.org/abs/2504.19874

Phase 1 decompresses to fp16 before attention — memory savings only, no speed gain.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx

# ---------------------------------------------------------------------------
# Step 1.1 — Rotation transforms
# ---------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    """Smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _walsh_hadamard_transform(x: mx.array) -> mx.array:
    """Normalized WHT along last dimension via iterative butterfly. O(d log d).

    The WHT is orthogonal and self-inverse: WHT(WHT(x)) == x.

    Args:
        x: [..., d] where d must be a power of 2.

    Returns:
        WHT(x) / sqrt(d)  — normalized so the transform is orthogonal.
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"d must be power of 2, got {d}"

    batch_shape = list(x.shape[:-1])
    result = x
    h = 1
    while h < d:
        # Reshape: [..., d] → [..., d/(2h), 2, h]
        result = result.reshape(batch_shape + [d // (2 * h), 2, h])
        a = result[..., 0, :]  # even half
        b = result[..., 1, :]  # odd half
        result = mx.stack([a + b, a - b], axis=-2)
        result = result.reshape(batch_shape + [d])
        h *= 2
    return result * (1.0 / math.sqrt(d))


def _random_rotation_matrix(d: int, seed: int = 42) -> mx.array:
    """Fixed random orthogonal matrix via QR decomposition.

    O(d²) memory and compute, but a single matmul on Apple Silicon is very
    fast for D<=256. The matrix is deterministic for a given (d, seed).

    Args:
        d: dimension (head_dim)
        seed: random seed for reproducibility

    Returns:
        [d, d] orthogonal float32 matrix (R @ R.T == I).
    """
    key = mx.random.key(seed)
    G = mx.random.normal((d, d), key=key)
    mx.synchronize()
    # QR decomposition runs on CPU only in MLX.
    Q, R = mx.linalg.qr(G, stream=mx.cpu)
    # Fix sign ambiguity so the result is unique for a given seed.
    diag_sign = mx.sign(mx.diag(R, stream=mx.cpu), stream=mx.cpu)
    Q = Q * diag_sign[None, :]
    mx.synchronize()
    return Q


def apply_rotation(x: mx.array, rotation: str = "wht", seed: int = 42) -> mx.array:
    """Apply a random rotation to decorrelate coordinates.

    Args:
        x: [..., D] float32 tensor
        rotation: "wht" (Walsh-Hadamard) or "qr" (random orthogonal)
        seed: seed for QR rotation matrix

    Returns:
        Rotated tensor, same shape as x.
    """
    if rotation == "wht":
        return _walsh_hadamard_transform(x)
    elif rotation == "qr":
        D = x.shape[-1]
        R = _random_rotation_matrix(D, seed=seed)
        return x @ R
    else:
        raise ValueError(f"Unknown rotation: {rotation!r}. Use 'wht' or 'qr'.")


def apply_inverse_rotation(
    x: mx.array, rotation: str = "wht", seed: int = 42
) -> mx.array:
    """Invert the rotation applied by apply_rotation.

    WHT is self-inverse; QR uses R.T (orthogonal inverse).
    """
    if rotation == "wht":
        return _walsh_hadamard_transform(x)
    elif rotation == "qr":
        D = x.shape[-1]
        R = _random_rotation_matrix(D, seed=seed)
        return x @ R.T
    else:
        raise ValueError(f"Unknown rotation: {rotation!r}. Use 'wht' or 'qr'.")
