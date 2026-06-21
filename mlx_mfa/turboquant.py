"""TurboQuant KV cache compression.

Two-stage vector quantization for KV cache compression:
  Stage 1 (PolarQuant/MSE): random rotation + Lloyd-Max scalar quantization
  Stage 2 (QJL): 1-bit random projection of residual for dot-product bias correction

Reference: TurboQuant (Google, ICLR 2026) — https://arxiv.org/abs/2504.19874

Execution paths
~~~~~~~~~~~~~~~

**Phase 1 — decompress path** (``turboquant_compress`` → ``turboquant_decompress``):
    Decompresses to fp16 before attention.  Supports QJL 1-bit residual correction
    (``use_qjl=True``) for improved 2-bit quality.  No speed gain over fp16 attention;
    benefits are memory-only.

**Phase 2 — fused K dequant** (``flash_attention_paged_varlen_turboquant``):
    Reads packed uint8 K indices inline during the K gather; centroid lookup +
    per-vector rescaling fused into the attention kernel.  Eliminates the
    ~18ms decompress overhead.  V stays fp16 by default.  QJL is **not fused** —
    the fused kernel uses PolarQuant/MSE only.

**Phase 3 — fused K+V dequant** (``tq_v_enabled=True``):
    Both K and V are TQ-packed and dequantified inline, achieving ~8× KV compression.
    Use via ``TurboQuantPagedInferenceContext`` or
    ``create_decode_runtime(turboquant=True, tq_v=True)``.

QJL note
~~~~~~~~
QJL correction (``use_qjl=True`` in ``turboquant_compress``) is a Phase 1 path only.
It applies a 1-bit random-projected residual bias to the attention scores, which
requires access to the full decomposed residual — incompatible with fused kernel
streaming.  For 2-bit quantization where QJL matters most, use the Phase 1 decompress
path.  For 3-bit and above, PolarQuant/MSE alone (fused path) is sufficient.
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


# ---------------------------------------------------------------------------
# Step 1.4 — Bit packing / unpacking
# ---------------------------------------------------------------------------


def _pack_1bit(bits_arr: mx.array) -> mx.array:
    """Pack 1-bit values (0 or 1) into uint8. 8 values per byte."""
    flat = bits_arr.reshape(-1).astype(mx.uint8)
    n = flat.shape[0]
    pad_n = (8 - n % 8) % 8
    if pad_n:
        flat = mx.concatenate([flat, mx.zeros((pad_n,), dtype=mx.uint8)])
    flat = flat.reshape(-1, 8)
    packed = flat[:, 0]
    for i in range(1, 8):
        packed = packed | (flat[:, i] << i)
    return packed


def _unpack_1bit(packed: mx.array, n_values: int) -> mx.array:
    """Unpack 1-bit packed bytes back to uint8 (0 or 1)."""
    bits_out = []
    for i in range(8):
        bits_out.append((packed >> i) & mx.array(0x01, dtype=mx.uint8))
    interleaved = mx.stack(bits_out, axis=-1).reshape(-1)
    return interleaved[:n_values]


def _pack_2bit(indices: mx.array) -> mx.array:
    """Pack 2-bit indices (0-3) into uint8. 4 values per byte."""
    shape = indices.shape
    flat = indices.reshape(-1).astype(mx.uint8)
    n = flat.shape[0]
    # Pad to multiple of 4
    pad_n = (4 - n % 4) % 4
    if pad_n:
        flat = mx.concatenate([flat, mx.zeros((pad_n,), dtype=mx.uint8)])
    flat = flat.reshape(-1, 4)
    packed = flat[:, 0] | (flat[:, 1] << 2) | (flat[:, 2] << 4) | (flat[:, 3] << 6)
    return packed  # [n_packed] uint8


def _unpack_2bit(packed: mx.array, n_values: int) -> mx.array:
    """Unpack 2-bit packed bytes back to uint8 indices 0-3."""
    b0 = packed & mx.array(0x03, dtype=mx.uint8)
    b1 = (packed >> 2) & mx.array(0x03, dtype=mx.uint8)
    b2 = (packed >> 4) & mx.array(0x03, dtype=mx.uint8)
    b3 = (packed >> 6) & mx.array(0x03, dtype=mx.uint8)
    interleaved = mx.stack([b0, b1, b2, b3], axis=-1).reshape(-1)
    return interleaved[:n_values]


def _pack_3bit(indices: mx.array) -> mx.array:
    """Pack 3-bit indices (0-7) into uint8. 8 values → 3 bytes.

    Layout: 8 values v0..v7 packed little-endian into 3 bytes:
      byte0 = v0 | (v1<<3) | (v2<<6)          [v2 contributes 2 low bits]
      byte1 = (v2>>2) | (v3<<1) | (v4<<4) | (v5<<7)  [v5 contributes 1 low bit]
      byte2 = (v5>>1) | (v6<<2) | (v7<<5)
    """
    flat = indices.reshape(-1).astype(mx.uint8)
    n = flat.shape[0]
    pad_n = (8 - n % 8) % 8
    if pad_n:
        flat = mx.concatenate([flat, mx.zeros((pad_n,), dtype=mx.uint8)])
    flat = flat.reshape(-1, 8)  # [groups, 8]
    v = [flat[:, i] for i in range(8)]

    byte0 = v[0] | (v[1] << 3) | (v[2] << 6)
    byte1 = (v[2] >> 2) | (v[3] << 1) | (v[4] << 4) | (v[5] << 7)
    byte2 = (v[5] >> 1) | (v[6] << 2) | (v[7] << 5)

    packed = mx.stack([byte0, byte1, byte2], axis=-1).reshape(-1)
    return packed  # [groups * 3] uint8


def _unpack_3bit(packed: mx.array, n_values: int) -> mx.array:
    """Unpack 3-bit packed bytes back to uint8 indices 0-7."""
    mask3 = mx.array(0x07, dtype=mx.uint8)
    packed = packed.reshape(-1, 3)
    b0, b1, b2 = packed[:, 0], packed[:, 1], packed[:, 2]

    v0 = b0 & mask3
    v1 = (b0 >> 3) & mask3
    v2 = ((b0 >> 6) | (b1 << 2)) & mask3
    v3 = (b1 >> 1) & mask3
    v4 = (b1 >> 4) & mask3
    v5 = ((b1 >> 7) | (b2 << 1)) & mask3
    v6 = (b2 >> 2) & mask3
    v7 = (b2 >> 5) & mask3

    interleaved = mx.stack([v0, v1, v2, v3, v4, v5, v6, v7], axis=-1).reshape(-1)
    return interleaved[:n_values]


def _pack_4bit(indices: mx.array) -> mx.array:
    """Pack 4-bit indices (0-15) into uint8. 2 values per byte."""
    flat = indices.reshape(-1).astype(mx.uint8)
    n = flat.shape[0]
    pad_n = n % 2
    if pad_n:
        flat = mx.concatenate([flat, mx.zeros((1,), dtype=mx.uint8)])
    flat = flat.reshape(-1, 2)
    packed = flat[:, 0] | (flat[:, 1] << 4)
    return packed


def _unpack_4bit(packed: mx.array, n_values: int) -> mx.array:
    """Unpack 4-bit packed bytes back to uint8 indices 0-15."""
    low = packed & mx.array(0x0F, dtype=mx.uint8)
    high = (packed >> 4) & mx.array(0x0F, dtype=mx.uint8)
    interleaved = mx.stack([low, high], axis=-1).reshape(-1)
    return interleaved[:n_values]


_PACK_FNS = {2: _pack_2bit, 3: _pack_3bit, 4: _pack_4bit}
_UNPACK_FNS = {2: _unpack_2bit, 3: _unpack_3bit, 4: _unpack_4bit}


def _packed_nbytes(n_values: int, bits: int) -> int:
    """Number of uint8 bytes ``pack_indices`` produces for ``n_values`` at ``bits``.

    Mirrors the on-disk pack layouts: 2/4-bit are dense (``ceil(n*bits/8)``); 3-bit
    is bit-planar in groups of 8 values → 3 bytes per 8-value group.
    """
    if bits == 3:
        return ((n_values + 7) // 8) * 3
    return (n_values * bits + 7) // 8


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack quantization indices into bit-packed uint8 bytes."""
    if bits not in _PACK_FNS:
        raise ValueError(f"pack_indices: unsupported bits={bits}; use 2, 3, or 4.")
    return _PACK_FNS[bits](indices)


def unpack_indices(packed: mx.array, n_values: int, bits: int) -> mx.array:
    """Unpack bit-packed uint8 bytes to quantization indices.

    CC-05 (audit): the packed buffer carries no bit-width, so packing at one
    width and unpacking at another silently produced garbage.  Assert the packed
    length matches ``(n_values, bits)`` (like ``unpack_3bit_optimal``) so a
    bit-width / length mismatch fails loudly (RULE 8) instead of corrupting.
    """
    if bits not in _UNPACK_FNS:
        raise ValueError(f"unpack_indices: unsupported bits={bits}; use 2, 3, or 4.")
    expected = _packed_nbytes(n_values, bits)
    if packed.size != expected:
        raise ValueError(
            f"unpack_indices: packed length {packed.size} != expected {expected} "
            f"for n_values={n_values}, bits={bits}. The packed buffer carries no "
            f"bit-width — this usually means pack/unpack used mismatched bits or "
            f"n_values (would silently corrupt indices)."
        )
    return _UNPACK_FNS[bits](packed, n_values)


# ---------------------------------------------------------------------------
# Step 1.3 — Compress / Decompress core
# ---------------------------------------------------------------------------

_DTYPE_STR_MAP = {
    mx.float16: "float16",
    mx.bfloat16: "bfloat16",
    mx.float32: "float32",
}
_STR_DTYPE_MAP = {v: k for k, v in _DTYPE_STR_MAP.items()}


def _require_dtype_str(dtype) -> str:
    """Map an MLX dtype to its stored string, or raise (CC-08 — no silent fp16)."""
    s = _DTYPE_STR_MAP.get(dtype)
    if s is None:
        raise ValueError(
            f"turboquant: unsupported dtype {dtype}; supported are "
            f"{tuple(_DTYPE_STR_MAP.values())}. (Previously coerced silently to float16.)"
        )
    return s


def turboquant_compress(
    x: mx.array,
    bits: int = 3,
    *,
    use_qjl: bool = True,
    rotation: str = "wht",
    seed: int = 42,
) -> dict:
    """Compress a KV tensor with TurboQuant (PolarQuant MSE + optional QJL).

    The algorithm:
      1. Convert to float32
      2. Apply random rotation (WHT or QR) to decorrelate coordinates
      3. Per-vector L2 normalization → coordinates become ~N(0, 1/sqrt(d))
      4. Scalar quantize each coordinate with Lloyd-Max centroids
      5. (Optional) QJL: store 1-bit sign of random-projected residual

    Args:
        x: [B, H, S, D] fp16/bf16/f32 KV tensor
        bits: quantization bits (2, 3, or 4)
        use_qjl: apply QJL 1-bit residual correction
        rotation: "wht" or "qr"
        seed: random seed for rotation and QJL projection

    Returns:
        dict with keys: x_q, scales, bits, rotation, seed, dtype, shape,
        and optionally qjl_signs, qjl_norms, qjl_proj_seed.
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
    if x.ndim != 4:
        raise ValueError(f"Expected [B,H,S,D] input, got ndim={x.ndim}")

    original_dtype = x.dtype
    B, H, S, D = x.shape

    # 1. Work in float32 for precision
    x_f32 = x.astype(mx.float32)

    # 2. Apply rotation to decorrelate
    x_rot = apply_rotation(x_f32, rotation, seed)

    # 3. Per-vector scale: normalize so coordinates are ~N(0, 1)
    # scale = ||x_rot||_2 / sqrt(D) per vector → shape [B, H, S, 1]
    norms = mx.sqrt((x_rot * x_rot).sum(axis=-1, keepdims=True))
    scale = norms / math.sqrt(D)
    # Avoid division by zero for zero vectors.
    safe_scale = mx.maximum(scale, 1e-10)
    x_normalized = x_rot / safe_scale  # ~N(0, 1) per coordinate

    # 4. Quantize each coordinate
    x_q = quantize_to_indices(x_normalized, bits)  # [B, H, S, D] uint8

    # Bit-pack for storage efficiency (3-bit: 5.3× compression ratio)
    x_q_packed = pack_indices(x_q, bits)
    n_values = B * H * S * D  # needed for unpacking

    result = {
        "x_q_packed": x_q_packed,
        "n_values": n_values,
        "scales": scale.squeeze(-1),  # [B, H, S]
        "bits": bits,
        "rotation": rotation,
        "seed": seed,
        # CC-08 (audit): an unsupported input dtype was silently recorded as
        # "float16", so decompress returned the wrong dtype with no signal.
        # Raise instead (RULE 8); supported = fp16/bf16/fp32.
        "dtype": _require_dtype_str(original_dtype),
        "shape": (B, H, S, D),
    }

    # 5. QJL 1-bit residual correction
    if use_qjl:
        x_recon_normalized = dequantize_from_indices(x_q, bits)
        residual = x_normalized - x_recon_normalized  # [B, H, S, D]

        # Per-vector residual L2 norm (needed for reconstruction)
        residual_norms = mx.sqrt(
            (residual * residual).sum(axis=-1, keepdims=True)
        ).squeeze(-1)  # [B, H, S]

        # Random Gaussian projection matrix (deterministic from seed)
        qjl_seed = seed + 7  # offset to avoid correlation with rotation seed
        key = mx.random.key(qjl_seed)
        # Rademacher matrix (±1) is cheaper and has same guarantees as Gaussian
        # for sign recovery. Shape: [D, D].
        S_proj = (
            2.0 * mx.random.bernoulli(p=0.5, shape=(D, D), key=key).astype(mx.float32)
            - 1.0
        ) / math.sqrt(D)

        # Project residual and store only the sign (1 bit per dimension)
        proj = residual @ S_proj  # [B, H, S, D]
        qjl_signs = proj >= 0  # bool [B, H, S, D]

        # Pack signs as 1-bit: 8 bools per byte
        result["qjl_signs_packed"] = _pack_1bit(qjl_signs.astype(mx.uint8))
        result["qjl_n_signs"] = B * H * S * D
        result["qjl_norms"] = residual_norms
        result["qjl_proj_seed"] = qjl_seed

    return result


def turboquant_decompress(compressed: dict) -> mx.array:
    """Decompress a TurboQuant-compressed KV tensor back to fp16/bf16.

    For Phase 1: full vector reconstruction (approximate). The QJL correction
    is applied as a vector-level adjustment — this is an approximation of the
    true QJL inner-product estimator (which operates on scores, not vectors).

    Args:
        compressed: dict from turboquant_compress()

    Returns:
        [B, H, S, D] tensor in original dtype.
    """
    bits = compressed["bits"]
    scales = compressed["scales"]  # [B, H, S]
    rotation = compressed["rotation"]
    seed = compressed["seed"]
    B, H, S, D = compressed["shape"]

    # 1. Unpack and dequantize: packed bytes → indices → centroid values
    x_q = unpack_indices(
        compressed["x_q_packed"], compressed["n_values"], bits
    ).reshape(B, H, S, D)
    x_normalized = dequantize_from_indices(x_q, bits)  # [B, H, S, D] f32

    # 2. QJL correction (approximate vector reconstruction)
    if "qjl_signs_packed" in compressed:
        qjl_signs = _unpack_1bit(
            compressed["qjl_signs_packed"], compressed["qjl_n_signs"]
        ).reshape(B, H, S, D)
        qjl_norms = compressed["qjl_norms"]  # [B, H, S]
        qjl_seed = compressed["qjl_proj_seed"]

        # Reconstruct the same projection matrix
        key = mx.random.key(qjl_seed)
        S_proj = (
            2.0 * mx.random.bernoulli(p=0.5, shape=(D, D), key=key).astype(mx.float32)
            - 1.0
        ) / math.sqrt(D)

        # Approximate residual direction from signs: sign(S @ r) @ S^T
        # Scaled by ||r|| * sqrt(pi/2) / D for unbiased estimation.
        sign_vals = 2.0 * qjl_signs.astype(mx.float32) - 1.0  # ±1
        correction_dir = sign_vals @ S_proj.T  # [B, H, S, D]
        # Scale by residual norm and the unbiased factor sqrt(pi/2)/D
        correction_scale = (
            qjl_norms[..., None] * math.sqrt(math.pi / 2.0) / D
        )
        x_normalized = x_normalized + correction_dir * correction_scale

    # 3. Rescale by per-vector norm
    x_rot = x_normalized * scales[..., None]  # [B, H, S, D]

    # 4. Inverse rotation
    x_decompressed = apply_inverse_rotation(x_rot, rotation, seed)

    # 5. Cast to original dtype (CC-08: raise on a corrupted/unknown dtype tag
    # instead of silently returning fp16).
    target_dtype = _STR_DTYPE_MAP.get(compressed.get("dtype"))
    if target_dtype is None:
        raise ValueError(
            f"turboquant_decompress: unknown dtype tag {compressed.get('dtype')!r}; "
            f"expected one of {tuple(_STR_DTYPE_MAP)}."
        )
    return x_decompressed.astype(target_dtype)


# ---------------------------------------------------------------------------
# Step 2.1 — TurboQuantKVCache
# ---------------------------------------------------------------------------


class TurboQuantKVCache:
    """KV cache with TurboQuant compression.

    Stores K (and optionally V) in TurboQuant compressed format.
    Decompresses to fp16/bf16 transparently when accessed for attention.

    The compression is applied per-append: each new token chunk is compressed
    immediately, storing only the compressed representation.

    Usage::

        cache = TurboQuantKVCache(bits=3, use_qjl=True)
        cache.append(k_new, v_new)              # compresses K immediately
        k_fp16 = cache.k_decompressed()          # decompresses for attention
        v_fp16 = cache.v_decompressed()
        print(cache.compression_ratio)            # ~3.5-5× vs fp16
    """

    def __init__(
        self,
        *,
        bits: int = 3,
        use_qjl: bool = True,
        rotation: str = "wht",
        seed: int = 42,
        compress_v: bool = False,
    ):
        """
        Args:
            bits: quantization bits for K (and V if compress_v). 2, 3, or 4.
            use_qjl: QJL 1-bit correction for K (recommended for dot-product accuracy).
            rotation: "wht" (Walsh-Hadamard) or "qr" (random orthogonal).
            seed: random seed for rotation matrix and QJL projection.
            compress_v: also compress V (MSE-only, no QJL — V errors are linear).
        """
        self.bits = bits
        self.use_qjl = use_qjl
        self.rotation = rotation
        self.seed = seed
        self.compress_v = compress_v

        # Storage: list of compressed dicts (one per append call)
        self._k_chunks: list[dict] = []
        self._v_chunks: list[dict | mx.array] = []  # compressed or raw fp16
        self._seq_len = 0
        self._dtype: Optional[str] = None

    def append(self, k_new: mx.array, v_new: mx.array) -> None:
        """Append new K/V tokens, compressing K (and optionally V) immediately.

        Args:
            k_new: [B, H, S_new, D] fp16/bf16
            v_new: [B, H, S_new, D] fp16/bf16
        """
        if k_new.ndim != 4:
            raise ValueError(f"Expected [B,H,S,D], got ndim={k_new.ndim}")

        S_new = k_new.shape[2]

        # Compress K
        k_compressed = turboquant_compress(
            k_new,
            bits=self.bits,
            use_qjl=self.use_qjl,
            rotation=self.rotation,
            seed=self.seed,
        )
        self._k_chunks.append(k_compressed)

        # V: compress or store raw
        if self.compress_v:
            v_compressed = turboquant_compress(
                v_new,
                bits=self.bits,
                use_qjl=False,  # V doesn't benefit from QJL
                rotation=self.rotation,
                seed=self.seed,
            )
            self._v_chunks.append(v_compressed)
        else:
            self._v_chunks.append(v_new)

        self._seq_len += S_new
        if self._dtype is None:
            self._dtype = _require_dtype_str(k_new.dtype)  # CC-08: no silent fp16

    def k_decompressed(self) -> mx.array:
        """Return full K in original dtype, decompressed from all chunks."""
        if not self._k_chunks:
            raise RuntimeError("Cache is empty — call append() first")
        parts = [turboquant_decompress(c) for c in self._k_chunks]
        return mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0]

    def v_decompressed(self) -> mx.array:
        """Return full V in original dtype."""
        if not self._v_chunks:
            raise RuntimeError("Cache is empty — call append() first")
        if self.compress_v:
            parts = [turboquant_decompress(c) for c in self._v_chunks]
            return mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0]
        else:
            chunks = self._v_chunks
            return (
                mx.concatenate(chunks, axis=2) if len(chunks) > 1 else chunks[0]
            )

    @property
    def seq_length(self) -> int:
        return self._seq_len

    @property
    def memory_bytes(self) -> int:
        """Actual memory usage of compressed cache."""
        total = 0
        for c in self._k_chunks:
            total += c["x_q_packed"].nbytes + c["scales"].nbytes
            if "qjl_signs_packed" in c:
                total += c["qjl_signs_packed"].nbytes + c["qjl_norms"].nbytes
        for vc in self._v_chunks:
            if isinstance(vc, dict):
                total += vc["x_q_packed"].nbytes + vc["scales"].nbytes
                if "qjl_signs_packed" in vc:
                    total += vc["qjl_signs_packed"].nbytes + vc["qjl_norms"].nbytes
            else:
                total += vc.nbytes
        return total

    @property
    def memory_bytes_fp16(self) -> int:
        """What the fp16 equivalent would use."""
        if not self._k_chunks:
            return 0
        B, H, _, D = self._k_chunks[0]["shape"]
        # K + V, both [B, H, seq_len, D] in fp16 (2 bytes)
        return 2 * B * H * self._seq_len * D * 2

    @property
    def compression_ratio(self) -> float:
        """memory_bytes_fp16 / memory_bytes."""
        mem = self.memory_bytes
        return self.memory_bytes_fp16 / mem if mem > 0 else 0.0

    def reset(self) -> None:
        """Clear all cached data."""
        self._k_chunks.clear()
        self._v_chunks.clear()
        self._seq_len = 0


# ---------------------------------------------------------------------------
# Step 2.2 — Adapter for KVCacheAdapter interface
# ---------------------------------------------------------------------------
# Imported lazily to avoid circular imports (kv_cache.py may import turboquant).


def _make_adapter(cache: "TurboQuantKVCache"):
    """Create a TurboQuantKVCacheAdapter wrapping a TurboQuantKVCache.

    Returns a KVCacheAdapter subclass instance.
    """
    from mlx_mfa.kv_cache import KVCacheAdapter, KVCacheCapabilities

    class TurboQuantKVCacheAdapter(KVCacheAdapter):
        """Adapter: TurboQuantKVCache → KVCacheAdapter interface."""

        kind = "turboquant"

        @property
        def capabilities(self) -> KVCacheCapabilities:
            return KVCacheCapabilities(
                append=True,
                reset=True,
                seq_length=True,
                attention_view=True,
                multi_seq=False,
            )

        def append(self, k_new, v_new, *, seq_id: int = 0) -> None:
            self.cache.append(k_new, v_new)

        def attention_k(self, seq_id: int = 0):
            return self.cache.k_decompressed()

        def attention_v(self, seq_id: int = 0):
            return self.cache.v_decompressed()

        def seq_length(self, seq_id: int = 0) -> int:
            return self.cache.seq_length

        def reset(self, *, seq_id: Optional[int] = None) -> None:
            self.cache.reset()

    return TurboQuantKVCacheAdapter(cache)


# ---------------------------------------------------------------------------
# Bit-planar optimal packing: 32 indices × 3 bits → 12 bytes (5.33× compress)
# ---------------------------------------------------------------------------


def _compute_packed_d(D: int, bits: int) -> int:
    """Return packed dimension: bytes per D elements at given bit-width."""
    if bits == 3:
        assert D % 32 == 0, f"D must be multiple of 32 for 3-bit packing, got {D}"
        return (D // 32) * 12  # 12 bytes per group of 32
    elif bits == 2:
        return D // 4  # 4 indices per byte
    elif bits == 4:
        return D // 2  # 2 indices per byte
    else:
        raise ValueError(f"Unsupported bits={bits}")


def pack_3bit_optimal(indices: mx.array) -> mx.array:
    """Pack 3-bit indices in bit-planar layout: 32 indices → 12 bytes.

    Layout per group of 32: bytes 0-3 = bit-plane 0, bytes 4-7 = bit-plane 1,
    bytes 8-11 = bit-plane 2. Within each 4-byte plane, byte i contains bits
    for indices [i*8 .. i*8+7], packed LSB-first.

    Args:
        indices: [..., D] uint8 with values 0-7. D must be multiple of 32.

    Returns:
        [..., D * 3 // 8] uint8 packed bytes. (48 bytes for D=128)
    """
    *prefix, D = indices.shape
    assert D % 32 == 0, f"D must be multiple of 32, got {D}"
    n_groups = D // 32

    idx = indices.reshape(*prefix, n_groups, 32)

    bit0 = (idx & 1).astype(mx.uint8)
    bit1 = ((idx >> 1) & 1).astype(mx.uint8)
    bit2 = ((idx >> 2) & 1).astype(mx.uint8)

    powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint8)

    def pack_plane(bits_arr):
        # bits_arr: [..., n_groups, 32] → [..., n_groups, 4, 8] → sum → [..., n_groups, 4]
        b = bits_arr.reshape(*prefix, n_groups, 4, 8)
        return (b * powers).sum(axis=-1).astype(mx.uint8)

    p0 = pack_plane(bit0)
    p1 = pack_plane(bit1)
    p2 = pack_plane(bit2)

    packed = mx.concatenate([p0, p1, p2], axis=-1)  # [..., n_groups, 12]
    return packed.reshape(*prefix, n_groups * 12)


def unpack_3bit_optimal(packed: mx.array, D: int) -> mx.array:
    """Unpack bit-planar 3-bit packed bytes back to uint8 indices 0-7.

    Args:
        packed: [..., D * 3 // 8] uint8
        D: original dimension (must be multiple of 32)

    Returns:
        [..., D] uint8 with values 0-7
    """
    *prefix, packed_D = packed.shape
    n_groups = D // 32
    assert packed_D == n_groups * 12

    packed = packed.reshape(*prefix, n_groups, 12)
    p0 = packed[..., 0:4]
    p1 = packed[..., 4:8]
    p2 = packed[..., 8:12]

    shifts = mx.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=mx.uint8)

    def unpack_plane(p):
        # p: [..., n_groups, 4] → [..., n_groups, 4, 1] → shift → [..., n_groups, 4, 8]
        p_exp = mx.expand_dims(p, axis=-1)
        bits = (p_exp >> shifts) & 1
        return bits.reshape(*prefix, n_groups, 32)

    b0 = unpack_plane(p0)
    b1 = unpack_plane(p1)
    b2 = unpack_plane(p2)

    indices = (b0 | (b1 << 1) | (b2 << 2)).astype(mx.uint8)
    return indices.reshape(*prefix, D)


# ---------------------------------------------------------------------------
# Phase 2 — Metal packing helpers for fused TQ kernel
# ---------------------------------------------------------------------------


def pack_k_for_metal(
    k: mx.array,
    bits: int = 3,
    *,
    rotation: str = "wht",
    seed: int = 42,
) -> tuple[mx.array, mx.array, mx.array]:
    """Rotate, normalize, quantize K and pack for the fused Metal TQ kernel.

    Packing layout depends on bit-width:
    - 3-bit: bit-planar — 32 indices → 12 bytes. packed_D = D*3/8.
    - 2-bit: 4 indices per byte. packed_D = D/4.
    - 4-bit: 2 indices per byte. packed_D = D/2.

    Args:
        k: [B, H, S, D] fp16/bf16 K tensor.
        bits: 2, 3, or 4.
        rotation: "wht" or "qr".
        seed: random seed for rotation.

    Returns:
        (k_packed, scales, centroids_fp16):
          k_packed: [B, H, S, packed_D] uint8.
          scales: [B, H, S] float32 — per-vector L2 scale.
          centroids_fp16: [2^bits] float16 — centroid lookup table for Metal.
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
    if k.ndim != 4:
        raise ValueError(f"Expected [B,H,S,D], got ndim={k.ndim}")

    B, H, S, D = k.shape
    if D % 32 != 0:
        raise ValueError(f"D must be multiple of 32 for Metal packing, got D={D}")

    # 1. Rotate + normalize (same as turboquant_compress)
    k_f32 = k.astype(mx.float32)
    k_rot = apply_rotation(k_f32, rotation, seed)
    norms = mx.sqrt((k_rot * k_rot).sum(axis=-1, keepdims=True))
    scale = norms / math.sqrt(D)
    safe_scale = mx.maximum(scale, 1e-10)
    k_normalized = k_rot / safe_scale  # ~N(0,1)

    # 2. Quantize to indices
    k_indices = quantize_to_indices(k_normalized, bits)  # [B,H,S,D] uint8

    # 3. Pack indices
    if bits == 3:
        k_packed = pack_3bit_optimal(k_indices)
    elif bits == 2:
        # 4 indices per byte: [B,H,S, D/4, 4] → pack
        k_groups = k_indices.reshape(B, H, S, D // 4, 4)
        k_packed = (k_groups[..., 0]
                    | (k_groups[..., 1] << 2)
                    | (k_groups[..., 2] << 4)
                    | (k_groups[..., 3] << 6))
    else:  # bits == 4
        # 2 indices per byte
        k_pairs = k_indices.reshape(B, H, S, D // 2, 2)
        k_packed = k_pairs[..., 0] | (k_pairs[..., 1] << 4)

    # 4. Centroids as fp16 for Metal buffer
    _, centroids = _get_centroids(bits)
    centroids_fp16 = centroids.astype(mx.float16)

    return k_packed, scale.squeeze(-1), centroids_fp16


def build_tq_paged_k_pool(
    k_pool_fp16: mx.array,
    bits: int = 3,
    *,
    rotation: str = "wht",
    seed: int = 42,
) -> tuple[mx.array, mx.array, mx.array]:
    """Convert a paged KV pool's K from fp16 to TQ-packed format.

    Args:
        k_pool_fp16: [num_pages, block_size, H_kv, D] fp16 K pool.
        bits: quantization bits (2, 3, or 4).
        rotation: "wht" or "qr".
        seed: random seed.

    Returns:
        (k_pool_tq, scales, centroids_fp16):
          k_pool_tq: [num_pages, block_size, H_kv, packed_D] uint8.
          scales: [num_pages, block_size, H_kv] float32.
          centroids_fp16: [2^bits] float16.
    """
    if k_pool_fp16.ndim != 4:
        raise ValueError(
            f"Expected [num_pages, block_size, H_kv, D], got ndim={k_pool_fp16.ndim}"
        )

    num_pages, block_size, H_kv, D = k_pool_fp16.shape

    # Reshape to [1, num_pages*block_size*H_kv, 1, D] for pack_k_for_metal
    # (it expects [B,H,S,D])
    flat = k_pool_fp16.reshape(1, 1, num_pages * block_size * H_kv, D)
    k_packed, scales, centroids_fp16 = pack_k_for_metal(
        flat, bits=bits, rotation=rotation, seed=seed
    )

    # Reshape back to pool layout
    packed_D = _compute_packed_d(D, bits)
    k_pool_tq = k_packed.reshape(num_pages, block_size, H_kv, packed_D)
    scales = scales.reshape(num_pages, block_size, H_kv)

    return k_pool_tq, scales, centroids_fp16


def pack_v_for_metal(
    v: mx.array,
    bits: int = 3,
    *,
    rotation: str = "wht",
    seed: int = 42,
) -> tuple[mx.array, mx.array, mx.array]:
    """Rotate, normalize, quantize V and pack for the fused Metal TQ kernel.

    Same packing scheme as ``pack_k_for_metal``.

    Args:
        v: [B, H, S, D] fp16/bf16 V tensor.
        bits: 2, 3, or 4.
        rotation: "wht" or "qr".
        seed: random seed for rotation.

    Returns:
        (v_packed, scales, centroids_fp16):
          v_packed: [B, H, S, packed_D] uint8.
          scales: [B, H, S] float32 — per-vector L2 scale.
          centroids_fp16: [2^bits] float16 — centroid lookup table for Metal.
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
    if v.ndim != 4:
        raise ValueError(f"Expected [B,H,S,D], got ndim={v.ndim}")

    B, H, S, D = v.shape
    if D % 32 != 0:
        raise ValueError(f"D must be multiple of 32 for Metal packing, got D={D}")

    v_f32 = v.astype(mx.float32)
    v_rot = apply_rotation(v_f32, rotation, seed)
    norms = mx.sqrt((v_rot * v_rot).sum(axis=-1, keepdims=True))
    scale = norms / math.sqrt(D)
    safe_scale = mx.maximum(scale, 1e-10)
    v_normalized = v_rot / safe_scale

    v_indices = quantize_to_indices(v_normalized, bits)

    if bits == 3:
        v_packed = pack_3bit_optimal(v_indices)
    elif bits == 2:
        v_groups = v_indices.reshape(B, H, S, D // 4, 4)
        v_packed = (v_groups[..., 0]
                    | (v_groups[..., 1] << 2)
                    | (v_groups[..., 2] << 4)
                    | (v_groups[..., 3] << 6))
    else:  # bits == 4
        v_pairs = v_indices.reshape(B, H, S, D // 2, 2)
        v_packed = v_pairs[..., 0] | (v_pairs[..., 1] << 4)

    _, centroids = _get_centroids(bits)
    centroids_fp16 = centroids.astype(mx.float16)

    return v_packed, scale.squeeze(-1), centroids_fp16


def build_tq_paged_v_pool(
    v_pool_fp16: mx.array,
    bits: int = 3,
    *,
    rotation: str = "wht",
    seed: int = 42,
) -> tuple[mx.array, mx.array, mx.array]:
    """Convert a paged KV pool's V from fp16 to TQ-packed format.

    Args:
        v_pool_fp16: [num_pages, block_size, H_kv, D] fp16 V pool.
        bits: quantization bits (2, 3, or 4).
        rotation: "wht" or "qr".
        seed: random seed.

    Returns:
        (v_pool_tq, scales, centroids_fp16):
          v_pool_tq: [num_pages, block_size, H_kv, packed_D] uint8.
          scales: [num_pages, block_size, H_kv] float32.
          centroids_fp16: [2^bits] float16.
    """
    if v_pool_fp16.ndim != 4:
        raise ValueError(
            f"Expected [num_pages, block_size, H_kv, D], got ndim={v_pool_fp16.ndim}"
        )

    num_pages, block_size, H_kv, D = v_pool_fp16.shape
    flat = v_pool_fp16.reshape(1, 1, num_pages * block_size * H_kv, D)
    v_packed, scales, centroids_fp16 = pack_v_for_metal(
        flat, bits=bits, rotation=rotation, seed=seed
    )

    packed_D = _compute_packed_d(D, bits)
    v_pool_tq = v_packed.reshape(num_pages, block_size, H_kv, packed_D)
    scales = scales.reshape(num_pages, block_size, H_kv)

    return v_pool_tq, scales, centroids_fp16
