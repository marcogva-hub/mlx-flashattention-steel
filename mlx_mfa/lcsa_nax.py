"""Sprint B Sparse Attention NAX public Python API.

Per design doc docs/lcsa-nax/lcsa-nax-design.md, the Sprint B path produces
block-sparse attention via:
  - Per-Q-tile threadgroup dispatch
  - Block-mask scan at K-tile granularity (zero-warp-divergence skip)
  - Online softmax (FA-2) across kept tiles

Phase 1.2 capabilities:
  - dtype: float16 OR bfloat16
  - D in {64, 128}
  - block_tile in {16, 32, 64}  (Phase 1.3 BT autoresearch refines)
  - block_mask: 2-D (NQ, NK), 3-D (Hq, NQ, NK), or 4-D (B, Hq, NQ, NK) bool
  - causal: False OR True (with within-tile triangular)
  - asymmetric qL != kL (cross-attention)

Constraint (Phase 1.1 carry-over):
  - block_mask total bytes must be >= 4096 (MLX inlines smaller buffers in
    constant address space which the kernel does not yet handle).

Phase 1.5 will introduce sparse_attention_dispatch() as a router that wraps
this and falls back to v2.33.1 cached-SDPA fallback for shapes outside
Sprint B's envelope.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import mlx.core as mx

try:
    from . import _ext  # nanobind extension
    _HAS_EXT = hasattr(_ext, "sparse_attention_forward")
except ImportError:
    _ext = None
    _HAS_EXT = False


# ---------------------------------------------------------------------------
# v2.36.1 — shape-aware V2 sparse default per canonical methodology
# (`docs/methodology/canonical-protocol.md`).
#
# Calibrated from 3-session canonical re-bench
# (`docs/methodology/canonical-bench-results.md`):
#   - 7/7 tested shapes graduate V2-eligible (6 CONFIDENT + 1 BOUNDARY,
#     0 HIGH_VARIANCE)
#   - Smallest tested work product: qL=4096, kL=4096, D=128 -> 2.15e9
#   - Below this work product, no canonical-protocol data exists. To
#     honor DC9 (empirical calibration, not extrapolation), we keep
#     V1 default for shapes smaller than the smallest tested.
#
# Users can override via `MFA_LCSA_KERNEL_VERSION=v1` or `=v2` env var.
# ---------------------------------------------------------------------------
_V2_DEFAULT_WORK_THRESHOLD = 2_147_483_648  # = 4096 * 4096 * 128


def decide_auto_version(
    density: float, qL: int, kL: int, D: int = 128
) -> str:
    """Shape-aware V2 sparse attention default per canonical methodology.

    v2.36.1: V2 sparse graduates to default for shapes where canonical
    benchmark methodology (docs/methodology/canonical-protocol.md) yields
    CONFIDENT or BOUNDARY cross-session ratio. Shapes smaller than the
    smallest tested work product keep V1 conservatively (no canonical
    data to validate them).

    Decision order:
      1. Env override: MFA_LCSA_KERNEL_VERSION=v1 or =v2 wins unconditionally
      2. Shape-aware threshold: qL * kL * D >= 2.15e9 -> "v2"
      3. Otherwise -> "v1"

    Args:
        density: block-mask density (currently unused in the threshold
            but accepted for future refinement per DC9 note).
        qL: query sequence length.
        kL: key sequence length.
        D: per-head dimension. Default 128 (production V2 set).

    Returns:
        "v1" or "v2".

    See docs/methodology/canonical-bench-results.md for calibration data.
    """
    # Env override has highest priority (preserves v2.35.0 SHIP_OPT_IN
    # contract for users who already set the env var).
    env = os.environ.get("MFA_LCSA_KERNEL_VERSION", "").strip().lower()
    if env in ("v1", "v2"):
        return env

    # Shape-aware default per canonical-methodology calibration.
    work_product = qL * kL * D
    if work_product >= _V2_DEFAULT_WORK_THRESHOLD:
        return "v2"
    return "v1"


def sparse_attention_nax(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    block_mask: mx.array,
    *,
    block_tile: int = 32,
    scale: Optional[float] = None,
    causal: bool = False,
) -> mx.array:
    """Block-sparse attention via NAX per-Q-tile dispatch.

    Args:
        Q: (B, Hq, qL, D) float16 or bfloat16. Hq must be multiple of Hk for GQA.
        K: (B, Hk, kL, D) same dtype as Q.
        V: (B, Hk, kL, D) same dtype as Q.
        block_mask: bool tensor. ndim ∈ {2, 3, 4} for layouts:
            2-D: (NQ, NK)  -- shared across batch and heads
            3-D: (Hq, NQ, NK)  -- per-head sparsity
            4-D: (B, Hq, NQ, NK)  -- per-batch per-head sparsity
            where NQ = qL // BT, NK = kL // BT.
            True = compute that Q-tile/K-tile pair; False = skip.
        block_tile: BT in {16, 32, 64}, must evenly divide qL and kL.
        scale: query scale; default 1/sqrt(D).
        causal: if True, skip tiles with k_tile > q_tile AND apply within-
            tile triangular mask on diagonal tiles. Requires qL == kL.

    Returns:
        O: (B, Hq, qL, D) same dtype as Q. All-False mask Q-rows → zero output.

    Raises:
        RuntimeError: if extension unavailable or shape/dtype constraint
            violated. Constraint violations are caught at C++ entry and
            surfaced verbatim.
    """
    if not _HAS_EXT:
        raise RuntimeError(
            "sparse_attention_nax requires the C++ extension. "
            "Rebuild with: CMAKE_ARGS='-DPython_EXECUTABLE=.venv/bin/python' "
            ".venv/bin/python -m pip install --no-build-isolation -e ."
        )
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    # v2.36.1: shape-aware default routing via explicit kernel_version
    # param.  Empty string falls back to MFA_LCSA_KERNEL_VERSION env var
    # (legacy v2.35.0 path).  Density is not part of the threshold yet
    # (DC9 note); pass 1.0 as a placeholder.
    kernel_version = decide_auto_version(
        density=1.0,
        qL=Q.shape[2],
        kL=K.shape[2],
        D=Q.shape[-1],
    )
    return _ext.sparse_attention_forward(
        Q, K, V, block_mask,
        block_tile=block_tile,
        causal=causal,
        scale=float(scale),
        kernel_version=kernel_version,
    )

# --------------------------------------------------------------------------
# Density-thresholded dispatcher.
#
# Default kernel is V1 (per-thread FA-2). V2 cooperative-tensor kernel is
# opt-in via `MFA_LCSA_KERNEL_VERSION=v2` per v2.35.0 SHIP_OPT_IN verdict
# (docs/lcsa-nax/lcsa-nax-coop-rewrite-results.md). V2 wins vs SDPA+bias
# across all tested cells but cross-session range > 10% on 5/7 shapes →
# conservative opt-in to let users explicitly test V2 in their environment.
#
# Threshold 0.02 reflects V1's break-even density (Phase 1.4 sweep). With
# V2 opt-in: users wanting V2's broad envelope should ALSO raise the
# threshold via the dispatcher's `density_threshold=` parameter.
# --------------------------------------------------------------------------

# V1 break-even density (Phase 1.4 sweep). Conservative for default V1 path.
# Users opting into V2 (MFA_LCSA_KERNEL_VERSION=v2) get broad envelope and
# should pass `density_threshold=0.95` to capture V2's full win range.
DEFAULT_DENSITY_THRESHOLD = 0.02


def _bool_mask_to_float_bias(block_mask, BT, qL, kL, target_dtype):
    """Expand bool block_mask to (..., qL, kL) float bias (0 / -inf)."""
    # Each block_mask[..., q, k] gates a BT x BT submatrix.
    expanded = mx.repeat(block_mask, BT, axis=-2)
    expanded = mx.repeat(expanded, BT, axis=-1)
    neg_inf = mx.array(-float("inf"), dtype=target_dtype)
    zero = mx.array(0.0, dtype=target_dtype)
    bias = mx.where(expanded, zero, neg_inf)
    return bias


def sparse_attention_dispatch(
    Q,
    K,
    V,
    block_mask,
    *,
    block_tile=16,
    scale=None,
    causal=False,
    density_threshold=DEFAULT_DENSITY_THRESHOLD,
    density=None,
    precomputed_bias=None,
):
    """Density-thresholded sparse attention dispatcher.

    Routes to one of two implementations based on block_mask density:
      - density < density_threshold: Sprint B NAX kernel (sparse_attention_nax)
      - density >= density_threshold: MLX SDPA + expanded float bias

    Args:
        Q, K, V, block_mask: same as sparse_attention_nax.
        block_tile: defaults to 16 (Phase 1.3 winner).
        scale: query scale; default 1/sqrt(D).
        causal: only supported on the Sprint B path. If True and the dispatcher
            chooses the SDPA path, an explicit causal bias is added to the
            block_mask bias before SDPA dispatch.
        density_threshold: route boundary. Default 0.02 from Phase 1.4 data.
        density: optional pre-computed density (avoids a reduction per call).
        precomputed_bias: optional pre-built (qL, kL) float bias - if the
            caller already has the float bias (cache-HIT pattern from v2.33.1),
            passing it skips the internal bias expansion. Used only on the
            SDPA route; ignored when Sprint B path is taken.

    Returns:
        O: same shape/dtype as Q.
    """
    if density is None:
        d_arr = mx.mean(block_mask.astype(mx.float32))
        mx.async_eval(d_arr); mx.synchronize()
        density = float(d_arr)
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    if density < density_threshold:
        return sparse_attention_nax(
            Q, K, V, block_mask,
            block_tile=block_tile,
            scale=scale,
            causal=causal,
        )
    # SDPA + float bias path.
    qL = Q.shape[2]
    kL = K.shape[2]
    if precomputed_bias is not None:
        bias = precomputed_bias
    else:
        bias = _bool_mask_to_float_bias(block_mask, block_tile, qL, kL, Q.dtype)
    if causal:
        # Combine with causal mask. MLX 0.31 SDPA mask='causal' and float bias
        # are mutually exclusive; emit a manual causal bias and sum.
        q_idx = mx.arange(qL).reshape(-1, 1)
        k_idx = mx.arange(kL).reshape(1, -1)
        causal_bias = mx.where(k_idx > q_idx,
                                mx.array(-float("inf"), dtype=Q.dtype),
                                mx.array(0.0, dtype=Q.dtype))
        if bias.ndim > 2:
            for _ in range(bias.ndim - 2):
                causal_bias = mx.expand_dims(causal_bias, 0)
        bias = bias + causal_bias
    return mx.fast.scaled_dot_product_attention(
        Q, K, V, scale=scale, mask=bias)


__all__ = [
    "sparse_attention_nax",
    "sparse_attention_dispatch",
    "decide_auto_version",
    "DEFAULT_DENSITY_THRESHOLD",
]
