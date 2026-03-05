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
    make_lcsa_mask
    make_axial_spatial_mask
    make_axial_temporal_mask
    make_dilated_temporal_mask
    make_sink_window_mask
    make_reference_frame_mask
    make_cross_stream_mask
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


# ---------------------------------------------------------------------------
# Track U — LCSA Composite Mask (FlashVSR)
# ---------------------------------------------------------------------------

def make_lcsa_mask(
    q: mx.array,
    k: mx.array,
    height: int,
    width: int,
    spatial_radius: int,
    top_k: int,
    head_dim: int = 128,
    num_frames: int = 1,
    temporal_radius: "Optional[int]" = None,
    patch_size: int = 1,
) -> mx.array:
    """Locality-Constrained Sparse Attention mask (FlashVSR LCSA).

    Two-stage composition:
      1. Spatial locality window: tiles within ``spatial_radius`` (Chebyshev)
         via ``make_spatial_2d/3d_mask``.
      2. Top-k scoring: within that window, keep the ``top_k`` K-tiles with
         the highest coarse QK^T scores per Q-tile.

    Composes existing primitives — does NOT duplicate their logic.

    Args:
        q:               Query tensor [B, H, N, D].
        k:               Key tensor [B, H, S, D].
        height:          Frame height in pixels.
        width:           Frame width in pixels.
        spatial_radius:  Chebyshev radius in patch units.
        top_k:           Maximum K-tiles to retain per Q-tile.
        head_dim:        Head dimension (tile sizes).
        num_frames:      Number of frames (1 = 2D, >1 = 3D).
        temporal_radius: Temporal radius in frames (3D mode only).
        patch_size:      Pixels per patch.

    Returns:
        bool mx.array [NQ_tiles, NK_tiles].  LCSA ⊆ spatial_mask.
    """
    # Stage 1: spatial locality mask
    if num_frames > 1:
        tr = temporal_radius if temporal_radius is not None else num_frames
        spatial_mask = make_spatial_3d_mask(
            height, width, num_frames,
            spatial_radius=spatial_radius,
            temporal_radius=tr,
            head_dim=head_dim,
            patch_size=patch_size,
        )
    else:
        spatial_mask = make_spatial_2d_mask(
            height, width,
            spatial_radius=spatial_radius,
            head_dim=head_dim,
            patch_size=patch_size,
        )

    spatial_np = np.array(spatial_mask)  # [NQ, NK]
    NQ, NK = spatial_np.shape
    actual_top_k = min(top_k, NK)

    # Stage 2: coarse QK^T scores — average-pooled tile representations
    BQ, BK = _bq_bk(head_dim)
    N = q.shape[2]
    S = k.shape[2]

    def pool_tiles(x_np: np.ndarray, tile_size: int, num_tiles: int, total: int) -> np.ndarray:
        B, H, _, D = x_np.shape
        pooled = np.zeros((B, H, num_tiles, D), dtype=np.float32)
        for ti in range(num_tiles):
            t_start = ti * tile_size
            t_end = min(t_start + tile_size, total)
            pooled[:, :, ti, :] = x_np[:, :, t_start:t_end, :].mean(axis=2)
        return pooled

    q_np = np.array(q.astype(mx.float32))
    k_np = np.array(k.astype(mx.float32))
    q_pooled = pool_tiles(q_np, BQ, NQ, N)  # [B, H, NQ, D]
    k_pooled = pool_tiles(k_np, BK, NK, S)  # [B, H, NK, D]

    scores_avg = np.einsum("bhqd,bhkd->bhqk", q_pooled, k_pooled).mean(axis=(0, 1))  # [NQ, NK]

    # Mask out tiles outside the spatial window with -inf, then take top_k
    scores_masked = np.where(spatial_np, scores_avg, -np.inf)

    mask_np = np.zeros((NQ, NK), dtype=bool)
    for qi in range(NQ):
        row = scores_masked[qi]
        active_count = int(spatial_np[qi].sum())
        k_to_keep = min(actual_top_k, active_count)
        if k_to_keep == 0:
            continue
        topk_idx = np.argpartition(row, -k_to_keep)[-k_to_keep:]
        # Only keep those that are within the spatial window
        valid = topk_idx[spatial_np[qi, topk_idx]]
        mask_np[qi, valid] = True

    return mx.array(mask_np)


# ---------------------------------------------------------------------------
# Track V — Axial / Factored Attention Masks
# ---------------------------------------------------------------------------

def make_axial_spatial_mask(
    height: int,
    width: int,
    num_frames: int,
    head_dim: int = 128,
    patch_size: int = 1,
    spatial_radius: "Optional[int]" = None,
) -> mx.array:
    """Attention restricted to the same frame (spatial axis only).

    Token (x, y, t) attends to token (x', y', t') only when t == t'.
    Optionally further restricted to ``spatial_radius`` within each frame.

    Token layout: (T, H, W) row-major — token_idx = t*(pH*pW) + y*pW + x.

    Args:
        height:         Frame height in pixels.
        width:          Frame width in pixels.
        num_frames:     Number of frames T.
        head_dim:       Head dimension (tile sizes).
        patch_size:     Pixels per patch.
        spatial_radius: Optional Chebyshev radius within each frame.

    Returns:
        bool mx.array [NQ_tiles, NK_tiles].
    """
    pH = height // patch_size
    pW = width // patch_size
    pHW = pH * pW
    N = pHW * num_frames

    BQ, BK = _bq_bk(head_dim)
    NQ = (N + BQ - 1) // BQ
    NK = (N + BK - 1) // BK

    # Frame index for each tile: the frame of the first token in the tile
    def tile_frame_range(tile_size, num_tiles, pHW_):
        """Return (min_frame, max_frame) per tile."""
        frames_min = np.zeros(num_tiles, dtype=np.int32)
        frames_max = np.zeros(num_tiles, dtype=np.int32)
        for ti in range(num_tiles):
            t_start = ti * tile_size
            t_end = min(t_start + tile_size, N) - 1
            frames_min[ti] = t_start // pHW_
            frames_max[ti] = t_end // pHW_
        return frames_min, frames_max

    q_fmin, q_fmax = tile_frame_range(BQ, NQ, pHW)
    k_fmin, k_fmax = tile_frame_range(BK, NK, pHW)

    # Two tiles are in the same-frame region only if they overlap on one frame
    # (conservative: active if ANY q-token and k-token share a frame)
    q_fmin_2d = q_fmin[:, None]
    q_fmax_2d = q_fmax[:, None]
    k_fmin_2d = k_fmin[None, :]
    k_fmax_2d = k_fmax[None, :]
    frame_overlap = (q_fmin_2d <= k_fmax_2d) & (k_fmin_2d <= q_fmax_2d)  # [NQ, NK]

    if spatial_radius is not None:
        # Restrict further: build 3D mask with temporal_radius=0 (same frame only)
        sp_mask = make_spatial_3d_mask(
            height, width, num_frames,
            spatial_radius=spatial_radius, temporal_radius=0,
            head_dim=head_dim, patch_size=patch_size,
        )
        frame_overlap = frame_overlap & np.array(sp_mask)

    return mx.array(frame_overlap)


def make_axial_temporal_mask(
    height: int,
    width: int,
    num_frames: int,
    head_dim: int = 128,
    patch_size: int = 1,
    temporal_radius: "Optional[int]" = None,
    causal: bool = False,
) -> mx.array:
    """Attention restricted to the same spatial position across frames.

    Token (x, y, t) attends to (x', y', t') only when x==x' and y==y'.
    Optional ``temporal_radius`` and causal masking.

    Token layout: (T, H, W) row-major — token_idx = t*(pH*pW) + y*pW + x.

    Args:
        height:          Frame height in pixels.
        width:           Frame width in pixels.
        num_frames:      Number of frames T.
        head_dim:        Head dimension (tile sizes).
        patch_size:      Pixels per patch.
        temporal_radius: Max frame distance (None = no limit).
        causal:          If True, only attend to previous/same frames.

    Returns:
        bool mx.array [NQ_tiles, NK_tiles].
    """
    pH = height // patch_size
    pW = width // patch_size
    pHW = pH * pW
    N = pHW * num_frames

    BQ, BK = _bq_bk(head_dim)
    NQ = (N + BQ - 1) // BQ
    NK = (N + BK - 1) // BK

    # Spatial position (x+y*pW) for each tile — min/max range
    def tile_spatial_range(tile_size, num_tiles):
        smin = np.zeros(num_tiles, dtype=np.int32)
        smax = np.zeros(num_tiles, dtype=np.int32)
        fmin = np.zeros(num_tiles, dtype=np.int32)
        fmax = np.zeros(num_tiles, dtype=np.int32)
        for ti in range(num_tiles):
            t_start = ti * tile_size
            t_end = min(t_start + tile_size, N) - 1
            smin[ti] = t_start % pHW
            smax[ti] = t_end % pHW
            fmin[ti] = t_start // pHW
            fmax[ti] = t_end // pHW
        return smin, smax, fmin, fmax

    q_smin, q_smax, q_fmin, q_fmax = tile_spatial_range(BQ, NQ)
    k_smin, k_smax, k_fmin, k_fmax = tile_spatial_range(BK, NK)

    # Same spatial position: ranges overlap (conservative)
    spatial_overlap = (
        (q_smin[:, None] <= k_smax[None, :]) &
        (k_smin[None, :] <= q_smax[:, None])
    )

    # Temporal distance
    if temporal_radius is not None:
        frame_dist = np.maximum(
            0,
            np.maximum(
                q_fmin[:, None] - k_fmax[None, :],
                k_fmin[None, :] - q_fmax[:, None],
            ),
        )
        temporal_ok = frame_dist <= temporal_radius
        spatial_overlap = spatial_overlap & temporal_ok

    if causal:
        # q_min_frame >= k_max_frame (q can only attend to earlier/same frames)
        causal_ok = q_fmin[:, None] >= k_fmax[None, :]
        # Include same-tile (diagonal)
        same_tile = (q_fmin[:, None] <= k_fmax[None, :]) & (k_fmin[None, :] <= q_fmax[:, None])
        spatial_overlap = spatial_overlap & (causal_ok | same_tile)

    return mx.array(spatial_overlap)


# ---------------------------------------------------------------------------
# Track W — Dilated Temporal Mask
# ---------------------------------------------------------------------------

def make_dilated_temporal_mask(
    height: int,
    width: int,
    num_frames: int,
    dilation_rate: int,
    local_window: int = 1,
    head_dim: int = 128,
    patch_size: int = 1,
) -> mx.array:
    """Dilated temporal attention mask for long videos.

    Token at frame t attends to:
      - Frames in [t - local_window, t + local_window]  (local context)
      - Frames t ± k * dilation_rate for all k ≥ 1      (dilated long-range)

    Token layout: (T, H, W) row-major.

    Args:
        height:        Frame height in pixels.
        width:         Frame width in pixels.
        num_frames:    Total number of frames T.
        dilation_rate: Frame stride for dilated connections.
        local_window:  Half-size of local frame window.
        head_dim:      Head dimension (tile sizes).
        patch_size:    Pixels per patch.

    Returns:
        bool mx.array [NQ_tiles, NK_tiles].
    """
    pH = height // patch_size
    pW = width // patch_size
    pHW = pH * pW
    N = pHW * num_frames

    BQ, BK = _bq_bk(head_dim)
    NQ = (N + BQ - 1) // BQ
    NK = (N + BK - 1) // BK

    # For each Q-frame t, build the set of allowed K-frames
    # Set of allowed frames: local_window + dilated
    allowed: list[set] = []
    for t in range(num_frames):
        a: set = set()
        for d in range(-local_window, local_window + 1):
            ft = t + d
            if 0 <= ft < num_frames:
                a.add(ft)
        k = 1
        while True:
            fp = t + k * dilation_rate
            fm = t - k * dilation_rate
            added = False
            if 0 <= fp < num_frames:
                a.add(fp)
                added = True
            if 0 <= fm < num_frames:
                a.add(fm)
                added = True
            if not added:
                break
            k += 1
        allowed.append(a)

    # Build tile-level frame ranges
    def tile_frame_range(tile_size, num_tiles):
        fmin = np.zeros(num_tiles, dtype=np.int32)
        fmax = np.zeros(num_tiles, dtype=np.int32)
        for ti in range(num_tiles):
            t_start = ti * tile_size
            t_end = min(t_start + tile_size, N) - 1
            fmin[ti] = t_start // pHW
            fmax[ti] = t_end // pHW
        return fmin, fmax

    q_fmin, q_fmax = tile_frame_range(BQ, NQ)
    k_fmin, k_fmax = tile_frame_range(BK, NK)

    # Precompute allowed frame sets per Q-frame range
    mask_np = np.zeros((NQ, NK), dtype=bool)
    for qi in range(NQ):
        # Union of allowed K-frames for all Q-frames in this tile
        q_allowed: set = set()
        for f in range(q_fmin[qi], q_fmax[qi] + 1):
            q_allowed |= allowed[f]
        for ki in range(NK):
            # Active if any K-frame in this K-tile is in q_allowed
            for kf in range(k_fmin[ki], k_fmax[ki] + 1):
                if kf in q_allowed:
                    mask_np[qi, ki] = True
                    break

    return mx.array(mask_np)


# ---------------------------------------------------------------------------
# Track X — Sink Tokens + Reference Frame Masks
# ---------------------------------------------------------------------------

def make_sink_window_mask(
    seq_len: int,
    window_size: int,
    num_sink_tokens: int,
    head_dim: int = 128,
    causal: bool = False,
) -> mx.array:
    """Sliding window attention with global sink tokens (StreamingLLM style).

    Each query attends to:
      - The first ``num_sink_tokens`` tokens (always visible, "attention sinks")
      - The local window of the last ``window_size`` tokens

    Args:
        seq_len:         Sequence length N.
        window_size:     Sliding window half-width (tokens).
        num_sink_tokens: Number of leading sink tokens always visible.
        head_dim:        Head dimension (tile sizes).
        causal:          If True, each query only sees previous tokens.

    Returns:
        bool mx.array [NQ_tiles, NK_tiles].
    """
    BQ, BK = _bq_bk(head_dim)
    NQ = (seq_len + BQ - 1) // BQ
    NK = (seq_len + BK - 1) // BK

    # Tile ranges
    q_start = np.arange(NQ) * BQ
    q_end = np.minimum(q_start + BQ - 1, seq_len - 1)
    k_start = np.arange(NK) * BK
    k_end = np.minimum(k_start + BK - 1, seq_len - 1)

    # Sink tiles: K-tiles that contain at least one sink token
    num_sink_tiles = max(1, (num_sink_tokens + BK - 1) // BK) if num_sink_tokens > 0 else 0
    sink_ok = np.zeros(NK, dtype=bool)
    sink_ok[:num_sink_tiles] = True  # [NK]

    # Window: Q-tile's last token is at q_end; window covers [q_end - window_size, q_end]
    window_lo = np.maximum(0, q_end[:, None] - window_size)  # [NQ, 1]
    # K-tile active in window if k_start <= q_end AND k_end >= window_lo
    window_ok = (k_start[None, :] <= q_end[:, None]) & (k_end[None, :] >= window_lo)

    mask_np = sink_ok[None, :] | window_ok  # [NQ, NK]

    if causal:
        # Only attend to K-tiles ending before/at the Q-tile start
        causal_ok = k_start[None, :] <= q_end[:, None]
        mask_np = mask_np & causal_ok

    return mx.array(mask_np)


def make_reference_frame_mask(
    height: int,
    width: int,
    num_frames: int,
    reference_frames: "list[int]",
    spatial_radius: "Optional[int]" = None,
    temporal_radius: int = 1,
    head_dim: int = 128,
    patch_size: int = 1,
) -> mx.array:
    """Video attention with global reference frames + local context.

    Each token can attend to:
      - All tokens in any reference frame (always visible)
      - Local temporal context (±temporal_radius frames)
      - Optionally restricted by spatial_radius within each frame

    Args:
        height:           Frame height in pixels.
        width:            Frame width in pixels.
        num_frames:       Total number of frames.
        reference_frames: List of frame indices that are always visible.
        spatial_radius:   Optional spatial Chebyshev radius.
        temporal_radius:  Local temporal window radius.
        head_dim:         Head dimension.
        patch_size:       Pixels per patch.

    Returns:
        bool mx.array [NQ_tiles, NK_tiles].
    """
    pH = height // patch_size
    pW = width // patch_size
    pHW = pH * pW
    N = pHW * num_frames

    BQ, BK = _bq_bk(head_dim)
    NQ = (N + BQ - 1) // BQ
    NK = (N + BK - 1) // BK

    ref_set = set(int(f) for f in reference_frames)

    def tile_frame_range(tile_size, num_tiles):
        fmin = np.zeros(num_tiles, dtype=np.int32)
        fmax = np.zeros(num_tiles, dtype=np.int32)
        for ti in range(num_tiles):
            t_start = ti * tile_size
            t_end = min(t_start + tile_size, N) - 1
            fmin[ti] = t_start // pHW
            fmax[ti] = t_end // pHW
        return fmin, fmax

    q_fmin, q_fmax = tile_frame_range(BQ, NQ)
    k_fmin, k_fmax = tile_frame_range(BK, NK)

    # K-tile is a reference tile if it contains ANY reference frame
    k_is_ref = np.array([
        any(kf in ref_set for kf in range(k_fmin[ki], k_fmax[ki] + 1))
        for ki in range(NK)
    ])

    # Local temporal window
    frame_dist = np.maximum(
        0,
        np.maximum(
            q_fmin[:, None] - k_fmax[None, :],
            k_fmin[None, :] - q_fmax[:, None],
        ),
    )
    local_ok = frame_dist <= temporal_radius

    mask_np = k_is_ref[None, :] | local_ok  # [NQ, NK]

    if spatial_radius is not None:
        sp_mask_np = np.array(
            make_spatial_3d_mask(
                height, width, num_frames,
                spatial_radius=spatial_radius, temporal_radius=temporal_radius,
                head_dim=head_dim, patch_size=patch_size,
            )
        )
        # Reference tiles bypass spatial restriction; local context is spatially restricted
        mask_np = k_is_ref[None, :] | (local_ok & sp_mask_np)

    return mx.array(mask_np)


# ---------------------------------------------------------------------------
# Track Y — Cross-Stream Attention Mask
# ---------------------------------------------------------------------------

def make_cross_stream_mask(
    n_tokens_q: int,
    n_tokens_kv: int,
    head_dim: int = 128,
    pattern: str = "full",
    q_segments: "Optional[list[int]]" = None,
    kv_segments: "Optional[list[int]]" = None,
    q_frames: "Optional[int]" = None,
    kv_frames: "Optional[int]" = None,
) -> mx.array:
    """Cross-stream block mask for multi-modal models (e.g. LTX-2 dual-stream DiT).

    Q and KV can have different token counts (rectangular mask).

    Patterns:
      ``"full"``     — every Q-tile attends to every KV-tile (dense cross-attention)
      ``"temporal"`` — Q frame t → KV frame t only  (requires q_frames, kv_frames)
      ``"segment"``  — matching segment pairs only   (requires q_segments, kv_segments)

    Args:
        n_tokens_q:   Number of query tokens.
        n_tokens_kv:  Number of key/value tokens.
        head_dim:     Head dimension (tile sizes).
        pattern:      One of "full", "temporal", "segment".
        q_segments:   List of Q segment lengths (``"segment"`` pattern).
        kv_segments:  List of KV segment lengths (``"segment"`` pattern).
        q_frames:     Number of Q frames (``"temporal"`` pattern).
        kv_frames:    Number of KV frames (``"temporal"`` pattern).

    Returns:
        bool mx.array [NQ_tiles, NK_tiles]  (rectangular when n_tokens_q != n_tokens_kv).
    """
    BQ, BK = _bq_bk(head_dim)
    NQ = (n_tokens_q + BQ - 1) // BQ
    NK = (n_tokens_kv + BK - 1) // BK

    if pattern == "full":
        return mx.array(np.ones((NQ, NK), dtype=bool))

    if pattern == "temporal":
        if q_frames is None or kv_frames is None:
            raise ValueError("make_cross_stream_mask with pattern='temporal' requires q_frames and kv_frames.")
        q_tpf = n_tokens_q // q_frames   # tokens per Q-frame
        kv_tpf = n_tokens_kv // kv_frames  # tokens per KV-frame

        def tile_frame(tile_size, num_tiles, tpf, total):
            fmin = np.zeros(num_tiles, dtype=np.int32)
            fmax = np.zeros(num_tiles, dtype=np.int32)
            for ti in range(num_tiles):
                t_start = ti * tile_size
                t_end = min(t_start + tile_size, total) - 1
                fmin[ti] = t_start // tpf
                fmax[ti] = t_end // tpf
            return fmin, fmax

        q_fmin, q_fmax = tile_frame(BQ, NQ, q_tpf, n_tokens_q)
        k_fmin, k_fmax = tile_frame(BK, NK, kv_tpf, n_tokens_kv)

        # Frame overlap (same frame)
        mask_np = (
            (q_fmin[:, None] <= k_fmax[None, :]) &
            (k_fmin[None, :] <= q_fmax[:, None])
        )
        return mx.array(mask_np)

    if pattern == "segment":
        if q_segments is None or kv_segments is None:
            raise ValueError("make_cross_stream_mask with pattern='segment' requires q_segments and kv_segments.")
        if len(q_segments) != len(kv_segments):
            raise ValueError("q_segments and kv_segments must have the same number of segments.")

        q_seg_ids = np.repeat(np.arange(len(q_segments)), q_segments)    # [n_tokens_q]
        kv_seg_ids = np.repeat(np.arange(len(kv_segments)), kv_segments)  # [n_tokens_kv]

        def tile_seg_range(tile_size, num_tiles, seg_ids, total):
            smin = np.zeros(num_tiles, dtype=np.int32)
            smax = np.zeros(num_tiles, dtype=np.int32)
            for ti in range(num_tiles):
                t_start = ti * tile_size
                t_end = min(t_start + tile_size, total)
                smin[ti] = seg_ids[t_start:t_end].min()
                smax[ti] = seg_ids[t_start:t_end].max()
            return smin, smax

        q_smin, q_smax = tile_seg_range(BQ, NQ, q_seg_ids, n_tokens_q)
        k_smin, k_smax = tile_seg_range(BK, NK, kv_seg_ids, n_tokens_kv)

        mask_np = (
            (q_smin[:, None] <= k_smax[None, :]) &
            (k_smin[None, :] <= q_smax[:, None])
        )
        return mx.array(mask_np)

    raise ValueError(f"Unknown pattern '{pattern}'. Use 'full', 'temporal', or 'segment'.")
