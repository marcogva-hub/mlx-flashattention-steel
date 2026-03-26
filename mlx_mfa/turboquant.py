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


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack quantization indices into bit-packed uint8 bytes."""
    return _PACK_FNS[bits](indices)


def unpack_indices(packed: mx.array, n_values: int, bits: int) -> mx.array:
    """Unpack bit-packed uint8 bytes to quantization indices."""
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
        "dtype": _DTYPE_STR_MAP.get(original_dtype, "float16"),
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

    # 5. Cast to original dtype
    target_dtype = _STR_DTYPE_MAP.get(compressed["dtype"], mx.float16)
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
            self._dtype = _DTYPE_STR_MAP.get(k_new.dtype, "float16")

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
