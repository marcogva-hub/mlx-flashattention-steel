"""Video/spatial attention block-mask construction helpers.

All functions return ``bool mx.array [NQ_tiles, NK_tiles]`` compatible with
``flash_attention_sparse()``.  Masks are built in numpy for speed (tile-level
arithmetic, not token-level) then converted to ``mx.array``.

Public API (re-exported from ``mlx_mfa``):
    make_spatial_2d_mask
    make_spatial_3d_mask
    make_topk_spatial_mask
    make_segment_mask
    make_causal_segment_mask
    make_adaptive_window_mask
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import mlx.core as mx

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bq_bk(head_dim: int) -> tuple[int, int]:
    """Return (BQ, BK) tile sizes — must match _steel_block_config in attention.py."""
    if head_dim <= 64:
        return 32, 32
    elif head_dim <= 128:
        return 32, 16
    else:
        return 32, 16  # D=256


def _tile_bboxes_1d(tile_size: int, num_tiles: int, total_tokens: int) -> np.ndarray:
    """Return [num_tiles, 2] array of (start, end) token ranges per tile."""
    starts = np.arange(num_tiles) * tile_size
    ends = np.minimum(starts + tile_size, total_tokens) - 1
    return np.stack([starts, ends], axis=1)  # [T, 2]


def _token_to_xy(token_idx: np.ndarray, pW: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert flat token index to (x, y) patch coordinates."""
    y = token_idx // pW
    x = token_idx % pW
    return x, y


def _token_to_xyt(
    token_idx: np.ndarray, pW: int, pHW: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert flat token index (T, H, W) order to (x, y, t) patch coordinates."""
    t = token_idx // pHW
    spatial = token_idx % pHW
    y = spatial // pW
    x = spatial % pW
    return x, y, t


def _tile_spatial_bboxes_2d(
    tile_size: int,
    num_tiles: int,
    total_tokens: int,
    pW: int,
) -> np.ndarray:
    """Return [num_tiles, 4] of (min_x, max_x, min_y, max_y) per tile."""
    starts = np.arange(num_tiles) * tile_size
    ends = np.minimum(starts + tile_size, total_tokens)

    # For each tile, sample ALL token indices in [start, end)
    # Then compute min/max x and y.
    # Vectorised: enumerate all possible offsets 0..tile_size-1, clip.
    offsets = np.arange(tile_size)  # [tile_size]
    # [num_tiles, tile_size]
    token_indices = starts[:, None] + offsets[None, :]
    valid_mask = token_indices < total_tokens
    # Replace out-of-range with last valid token (to avoid garbage coords)
    token_indices_clamped = np.where(valid_mask, token_indices, ends[:, None] - 1)
    token_indices_clamped = np.clip(token_indices_clamped, 0, total_tokens - 1)

    x, y = _token_to_xy(token_indices_clamped, pW)  # both [num_tiles, tile_size]

    # Mask out-of-range tokens from min/max
    INF = 10**9
    x_for_min = np.where(valid_mask, x, INF)
    x_for_max = np.where(valid_mask, x, -INF)
    y_for_min = np.where(valid_mask, y, INF)
    y_for_max = np.where(valid_mask, y, -INF)

    return np.stack([
        x_for_min.min(axis=1),
        x_for_max.max(axis=1),
        y_for_min.min(axis=1),
        y_for_max.max(axis=1),
    ], axis=1).astype(np.int32)  # [num_tiles, 4]


def _tile_spatial_bboxes_3d(
    tile_size: int,
    num_tiles: int,
    total_tokens: int,
    pW: int,
    pHW: int,
) -> np.ndarray:
    """Return [num_tiles, 6] of (min_x, max_x, min_y, max_y, min_t, max_t) per tile."""
    starts = np.arange(num_tiles) * tile_size
    ends = np.minimum(starts + tile_size, total_tokens)

    offsets = np.arange(tile_size)
    token_indices = starts[:, None] + offsets[None, :]
    valid_mask = token_indices < total_tokens
    token_indices_clamped = np.where(valid_mask, token_indices, ends[:, None] - 1)
    token_indices_clamped = np.clip(token_indices_clamped, 0, total_tokens - 1)

    x, y, t = _token_to_xyt(token_indices_clamped, pW, pHW)

    INF = 10**9
    x_for_min = np.where(valid_mask, x, INF)
    x_for_max = np.where(valid_mask, x, -INF)
    y_for_min = np.where(valid_mask, y, INF)
    y_for_max = np.where(valid_mask, y, -INF)
    t_for_min = np.where(valid_mask, t, INF)
    t_for_max = np.where(valid_mask, t, -INF)

    return np.stack([
        x_for_min.min(axis=1),
        x_for_max.max(axis=1),
        y_for_min.min(axis=1),
        y_for_max.max(axis=1),
        t_for_min.min(axis=1),
        t_for_max.max(axis=1),
    ], axis=1).astype(np.int32)  # [num_tiles, 6]


# ---------------------------------------------------------------------------
# Track O — Spatial 2D / 3D masks
# ---------------------------------------------------------------------------

def make_spatial_2d_mask(
    height: int,
    width: int,
    spatial_radius: int,
    head_dim: int = 128,
    patch_size: int = 1,
) -> mx.array:
    """2D spatial locality block mask for image/video-frame attention.

    Tokens are flattened in row-major order: token_idx = y * W + x (patch
    coordinates).  A Q-tile and K-tile are active when ANY token pair in
    those tiles satisfies:

        |qx - kx| <= spatial_radius  AND  |qy - ky| <= spatial_radius

    (Chebyshev / square window.)

    Args:
        height:         Number of rows in pixels (divided by patch_size if > 1).
        width:          Number of columns in pixels (divided by patch_size if > 1).
        spatial_radius: Radius in *patch* units.
        head_dim:       Head dimension (determines BQ, BK tile sizes).
        patch_size:     Pixels per patch (default 1 = already in patch units).

    Returns:
        bool mx.array [NQ_tiles, NK_tiles].

    Example::

        # 720p frame, 2×2 patches, attend within 8-patch radius
        mask = make_spatial_2d_mask(360, 640, spatial_radius=8, patch_size=2)
        out = flash_attention_sparse(q, k, v, mask)
    """
    pH = height // patch_size
    pW = width // patch_size
    N = pH * pW

    BQ, BK = _bq_bk(head_dim)
    NQ = (N + BQ - 1) // BQ
    NK = (N + BK - 1) // BK

    # Compute bounding box of each tile in (x, y) patch space
    q_bboxes = _tile_spatial_bboxes_2d(BQ, NQ, N, pW)  # [NQ, 4]
    k_bboxes = _tile_spatial_bboxes_2d(BK, NK, N, pW)  # [NK, 4]

    # Vectorised distance check: broadcast [NQ, 4] vs [NK, 4]
    # q_bboxes: [NQ, 1, 4], k_bboxes: [1, NK, 4]
    q = q_bboxes[:, None, :]   # [NQ, 1, 4]
    k = k_bboxes[None, :, :]   # [1, NK, 4]

    # Segment gap along x: max(0, q_min_x - k_max_x, k_min_x - q_max_x)
    dist_x = np.maximum(0, np.maximum(q[..., 0] - k[..., 1], k[..., 0] - q[..., 1]))
    dist_y = np.maximum(0, np.maximum(q[..., 2] - k[..., 3], k[..., 2] - q[..., 3]))

    mask_np = (dist_x <= spatial_radius) & (dist_y <= spatial_radius)  # [NQ, NK]
    return mx.array(mask_np)


def make_spatial_3d_mask(
    height: int,
    width: int,
    num_frames: int,
    spatial_radius: int,
    temporal_radius: int,
    head_dim: int = 128,
    patch_size: int = 1,
    temporal_patch_size: int = 1,
) -> mx.array:
    """3D spatio-temporal locality block mask for video attention.

    Tokens are flattened in (T, H, W) order:
    ``token_idx = t * (pH * pW) + y * pW + x``.

    A tile pair is active when ANY token pair satisfies:

        |qx - kx| <= spatial_radius  AND
        |qy - ky| <= spatial_radius  AND
        |qt - kt| <= temporal_radius

    Args:
        height, width:       Spatial dimensions (pixels; divided by patch_size).
        num_frames:          Temporal length (frames; divided by temporal_patch_size).
        spatial_radius:      Spatial radius in patch units.
        temporal_radius:     Temporal radius in frame/patch units.
        head_dim:            Head dimension for tile sizes.
        patch_size:          Pixels per spatial patch.
        temporal_patch_size: Frames per temporal patch.

    Returns:
        bool mx.array [NQ_tiles, NK_tiles].

    Example::

        # 16 frames of 360×640 (2×2 patches), local spatial + adjacent frames
        mask = make_spatial_3d_mask(360, 640, 16,
                                    spatial_radius=8, temporal_radius=2,
                                    patch_size=2)
        out = flash_attention_sparse(q, k, v, mask)
    """
    pH = height // patch_size
    pW = width // patch_size
    pT = num_frames // temporal_patch_size
    pHW = pH * pW
    N = pT * pHW

    BQ, BK = _bq_bk(head_dim)
    NQ = (N + BQ - 1) // BQ
    NK = (N + BK - 1) // BK

    q_bboxes = _tile_spatial_bboxes_3d(BQ, NQ, N, pW, pHW)  # [NQ, 6]
    k_bboxes = _tile_spatial_bboxes_3d(BK, NK, N, pW, pHW)  # [NK, 6]

    q = q_bboxes[:, None, :]
    k = k_bboxes[None, :, :]

    dist_x = np.maximum(0, np.maximum(q[..., 0] - k[..., 1], k[..., 0] - q[..., 1]))
    dist_y = np.maximum(0, np.maximum(q[..., 2] - k[..., 3], k[..., 2] - q[..., 3]))
    dist_t = np.maximum(0, np.maximum(q[..., 4] - k[..., 5], k[..., 4] - q[..., 5]))

    mask_np = (
        (dist_x <= spatial_radius) &
        (dist_y <= spatial_radius) &
        (dist_t <= temporal_radius)
    )  # [NQ, NK]
    return mx.array(mask_np)


def make_topk_spatial_mask(
    q: mx.array,
    k: mx.array,
    top_k: int,
    head_dim: int = 128,
) -> mx.array:
    """Content-aware top-k block mask via coarse attention scoring.

    For each Q-tile, keep only the ``top_k`` K-tiles with the highest mean
    coarse attention score (average pooled Q × average pooled K^T, averaged
    over batch and heads).

    Args:
        q:        Query tensor [B, H, N, D].
        k:        Key tensor [B, H, S, D].
        top_k:    Number of K-tiles to keep per Q-tile.
        head_dim: Head dimension for tile sizes.

    Returns:
        bool mx.array [NQ_tiles, NK_tiles].

    Example::

        mask = make_topk_spatial_mask(q, k, top_k=16)
        out = flash_attention_sparse(q, k, v, mask)
    """
    BQ, BK = _bq_bk(head_dim)
    N = q.shape[2]
    S = k.shape[2]
    NQ = (N + BQ - 1) // BQ
    NK = (S + BK - 1) // BK

    actual_top_k = min(top_k, NK)

    # Average-pool Q and K to tile-level representations [B, H, NQ/NK, D]
    import numpy as _np
    q_np = _np.array(q.astype(mx.float32))  # [B, H, N, D]
    k_np = _np.array(k.astype(mx.float32))  # [B, H, S, D]

    # Pool: tile averages
    def pool_tiles(x_np, tile_size, num_tiles, total):
        B, H, _, D = x_np.shape
        pooled = _np.zeros((B, H, num_tiles, D), dtype=_np.float32)
        for ti in range(num_tiles):
            t_start = ti * tile_size
            t_end = min(t_start + tile_size, total)
            pooled[:, :, ti, :] = x_np[:, :, t_start:t_end, :].mean(axis=2)
        return pooled

    q_pooled = pool_tiles(q_np, BQ, NQ, N)  # [B, H, NQ, D]
    k_pooled = pool_tiles(k_np, BK, NK, S)  # [B, H, NK, D]

    # Coarse scores: [B, H, NQ, NK]
    scores = _np.einsum("bhqd,bhkd->bhqk", q_pooled, k_pooled)

    # Average across batch and heads: [NQ, NK]
    scores_avg = scores.mean(axis=(0, 1))

    # For each Q-tile, find the top_k K-tile indices
    # argpartition gives the top_k in O(N) — then sort to get true top_k
    mask_np = _np.zeros((NQ, NK), dtype=bool)
    for qi in range(NQ):
        row = scores_avg[qi]
        if actual_top_k >= NK:
            mask_np[qi, :] = True
        else:
            topk_idx = _np.argpartition(row, -actual_top_k)[-actual_top_k:]
            mask_np[qi, topk_idx] = True

    return mx.array(mask_np)


# ---------------------------------------------------------------------------
# Track P — Segment / document masks
# ---------------------------------------------------------------------------

def make_segment_mask(
    segment_lengths: list[int],
    head_dim: int = 128,
) -> mx.array:
    """Block-diagonal mask: tokens attend only within their own segment.

    Tokens from different segments cannot attend to each other.  A tile pair
    (qi, ki) is active iff any token in qi's range and any token in ki's range
    share the same segment.

    Args:
        segment_lengths: List of per-segment token counts.  Sum = N.
        head_dim:        Head dimension for tile sizes.

    Returns:
        bool mx.array [NQ_tiles, NK_tiles].

    Example::

        # 3 video clips packed: 128 + 256 + 128 = 512 tokens
        mask = make_segment_mask([128, 256, 128])
        out = flash_attention_sparse(q, k, v, mask)
    """
    N = sum(segment_lengths)
    BQ, BK = _bq_bk(head_dim)
    NQ = (N + BQ - 1) // BQ
    NK = (N + BK - 1) // BK

    # Build a token→segment array using np.repeat
    seg_ids = np.repeat(
        np.arange(len(segment_lengths), dtype=np.int32),
        segment_lengths
    )  # [N]

    # For each tile, the set of segments it contains can be described by the
    # range [seg_of_first_token, seg_of_last_token].  Two tiles overlap iff
    # their segment ranges are not disjoint.
    #
    # Because segment_lengths are contiguous, each tile covers a contiguous
    # range of segment IDs.

    tile_q_starts = np.arange(NQ) * BQ
    tile_q_ends = np.minimum(tile_q_starts + BQ, N) - 1
    tile_k_starts = np.arange(NK) * BK
    tile_k_ends = np.minimum(tile_k_starts + BK, N) - 1

    # seg range for each tile
    q_seg_min = seg_ids[tile_q_starts]      # [NQ]
    q_seg_max = seg_ids[tile_q_ends]        # [NQ]
    k_seg_min = seg_ids[tile_k_starts]      # [NK]
    k_seg_max = seg_ids[tile_k_ends]        # [NK]

    # Broadcast: [NQ, NK]
    # Two ranges [a, b] and [c, d] overlap iff a <= d AND c <= b
    mask_np = (
        (q_seg_min[:, None] <= k_seg_max[None, :]) &
        (k_seg_min[None, :] <= q_seg_max[:, None])
    )
    return mx.array(mask_np)


def make_causal_segment_mask(
    segment_lengths: list[int],
    head_dim: int = 128,
) -> mx.array:
    """Block-diagonal + causal within each segment.

    Combines segment isolation with causal masking.  Use alongside
    ``flash_attention_sparse(..., causal=True)`` for exact token-level
    causal masking within active blocks.

    Returns:
        bool mx.array [NQ_tiles, NK_tiles].
    """
    from mlx_mfa.attention import make_causal_block_mask
    N = sum(segment_lengths)
    seg_mask = make_segment_mask(segment_lengths, head_dim)
    causal_mask = make_causal_block_mask(N, head_dim)
    # Both are bool arrays — AND them
    return seg_mask & causal_mask


# ---------------------------------------------------------------------------
# Track Q — Adaptive window mask (SeedVR2 scaling)
# ---------------------------------------------------------------------------

def _make_asymmetric_spatial_mask(
    pH: int,
    pW: int,
    pT: int,
    radius_h: int,
    radius_w: int,
    radius_t: int,
    head_dim: int,
) -> mx.array:
    """Internal: spatial_3d_mask with separate H/W/T radii."""
    pHW = pH * pW
    N = pT * pHW

    BQ, BK = _bq_bk(head_dim)
    NQ = (N + BQ - 1) // BQ
    NK = (N + BK - 1) // BK

    q_bboxes = _tile_spatial_bboxes_3d(BQ, NQ, N, pW, pHW)
    k_bboxes = _tile_spatial_bboxes_3d(BK, NK, N, pW, pHW)

    q = q_bboxes[:, None, :]
    k = k_bboxes[None, :, :]

    dist_x = np.maximum(0, np.maximum(q[..., 0] - k[..., 1], k[..., 0] - q[..., 1]))
    dist_y = np.maximum(0, np.maximum(q[..., 2] - k[..., 3], k[..., 2] - q[..., 3]))
    dist_t = np.maximum(0, np.maximum(q[..., 4] - k[..., 5], k[..., 4] - q[..., 5]))

    mask_np = (dist_x <= radius_w) & (dist_y <= radius_h) & (dist_t <= radius_t)
    return mx.array(mask_np)


def make_adaptive_window_mask(
    height: int,
    width: int,
    num_frames: int = 1,
    base_window_h: int = 16,
    base_window_w: int = 16,
    base_window_t: int = 4,
    train_resolution: tuple[int, int] = (256, 256),
    inference_resolution: tuple[int, int] = (512, 512),
    head_dim: int = 128,
    patch_size: int = 1,
) -> mx.array:
    """Adaptive window mask that scales window size with resolution ratio.

    Prevents RoPE aliasing when inference resolution exceeds training resolution
    by constraining the attention window so that positional encoding ranges stay
    consistent with training (SeedVR2 strategy).

    Window scaling rule:
        effective_window = base_window × (train_resolution / inference_resolution)

    At 2× the training resolution the window halves, keeping RoPE ranges
    identical to training.

    Args:
        height, width:        Inference spatial dims in pixels (or patches if patch_size=1).
        num_frames:           Number of frames (1 for image attention).
        base_window_h/w/t:    Window *diameter* (not radius) at training resolution.
        train_resolution:     (H, W) the model was trained at.
        inference_resolution: (H, W) being used for inference.
        head_dim:             Head dimension for tile sizes.
        patch_size:           Pixels per spatial patch.

    Returns:
        bool mx.array [NQ_tiles, NK_tiles].

    Example::

        # SeedVR2: base 16×16 window at 256 training, inferring at 1024
        mask = make_adaptive_window_mask(1024, 1024, num_frames=4,
                                         base_window_h=16, base_window_w=16,
                                         train_resolution=(256, 256),
                                         inference_resolution=(1024, 1024))
    """
    scale_h = train_resolution[0] / inference_resolution[0]
    scale_w = train_resolution[1] / inference_resolution[1]

    eff_window_h = max(1, int(base_window_h * scale_h))
    eff_window_w = max(1, int(base_window_w * scale_w))
    eff_window_t = base_window_t  # temporal window does not scale with spatial res

    radius_h = eff_window_h // 2
    radius_w = eff_window_w // 2
    radius_t = eff_window_t // 2

    pH = max(1, height // patch_size)
    pW = max(1, width // patch_size)

    return _make_asymmetric_spatial_mask(pH, pW, num_frames,
                                         radius_h, radius_w, radius_t,
                                         head_dim)
