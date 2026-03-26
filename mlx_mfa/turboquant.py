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


# ---------------------------------------------------------------------------
# Step 1.2 — Lloyd-Max centroids for N(0,1)
# ---------------------------------------------------------------------------
# Pre-computed via 200 iterations of the Lloyd-Max algorithm on the standard
# normal distribution. These are the MSE-optimal scalar quantizer levels.
# Symmetric around 0 (the distribution is symmetric).

_CENTROIDS_2B = [-1.51041761, -0.45278003, 0.45278003, 1.51041761]
_BOUNDARIES_2B = [-0.98159882, 0.0, 0.98159882]

_CENTROIDS_3B = [
    -2.1519457, -1.34390928, -0.75600528, -0.24509418,
    0.24509418, 0.75600528, 1.34390928, 2.1519457,
]
_BOUNDARIES_3B = [
    -1.74792749, -1.04995728, -0.50054973, 0.0,
    0.50054973, 1.04995728, 1.74792749,
]

_CENTROIDS_4B = [
    -2.73322996, -2.06974067, -1.61878445, -1.2569267,
    -0.94294213, -0.65722317, -0.38834098, -0.12849502,
    0.12849502, 0.38834098, 0.65722317, 0.94294213,
    1.2569267, 1.61878445, 2.06974067, 2.73322996,
]
_BOUNDARIES_4B = [
    -2.40148531, -1.84426256, -1.43785557, -1.09993441,
    -0.80008265, -0.52278207, -0.258418, -0.0,
    0.258418, 0.52278207, 0.80008265, 1.09993441,
    1.43785557, 1.84426256, 2.40148531,
]

# Keyed by bits for fast lookup.
_CENTROID_TABLE: dict[int, tuple[list[float], list[float]]] = {
    2: (_BOUNDARIES_2B, _CENTROIDS_2B),
    3: (_BOUNDARIES_3B, _CENTROIDS_3B),
    4: (_BOUNDARIES_4B, _CENTROIDS_4B),
}

# Cached MLX arrays (created on first use to avoid import-time GPU work).
_centroid_cache: dict[int, tuple[mx.array, mx.array]] = {}


def _get_centroids(bits: int) -> tuple[mx.array, mx.array]:
    """Return (boundaries, centroids) as MLX float32 arrays for a given bitwidth.

    The boundaries array has 2^bits - 1 elements (decision thresholds).
    The centroids array has 2^bits elements (reconstruction values).
    """
    if bits not in _CENTROID_TABLE:
        raise ValueError(f"Unsupported bits={bits}. Use 2, 3, or 4.")
    if bits not in _centroid_cache:
        bounds, cents = _CENTROID_TABLE[bits]
        _centroid_cache[bits] = (
            mx.array(bounds, dtype=mx.float32),
            mx.array(cents, dtype=mx.float32),
        )
    return _centroid_cache[bits]


def quantize_to_indices(x: mx.array, bits: int) -> mx.array:
    """Quantize float32 values to Lloyd-Max bin indices.

    Args:
        x: [...] float32 tensor (rotated coordinates, ~N(0, sigma))
        bits: 2, 3, or 4

    Returns:
        [...] uint8 tensor with values in [0, 2^bits - 1].
    """
    boundaries, _ = _get_centroids(bits)
    # MLX has no searchsorted. For small boundary arrays (3-15 elements),
    # broadcasting comparison is efficient: sum(x > b_i) gives the bin index.
    # Shape: x=[...], boundaries=[K] → compare=[..., K] → sum → [...]
    flat = x.reshape(-1, 1)  # [N, 1]
    indices = (flat >= boundaries[None, :]).sum(axis=-1)  # [N]
    return indices.reshape(x.shape).astype(mx.uint8)


def dequantize_from_indices(indices: mx.array, bits: int) -> mx.array:
    """Map quantization indices back to centroid values.

    Args:
        indices: [...] uint8 tensor with values in [0, 2^bits - 1]
        bits: 2, 3, or 4

    Returns:
        [...] float32 tensor of centroid reconstruction values.
    """
    _, centroids = _get_centroids(bits)
    return centroids[indices.astype(mx.int32)]
