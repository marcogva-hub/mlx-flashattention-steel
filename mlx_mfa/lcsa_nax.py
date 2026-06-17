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

`sparse_attention_dispatch()` (below) is the router that wraps this and falls
back to the cached-SDPA path for shapes outside
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


def sparse_attention_nax_with_lse(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    block_mask: mx.array,
    *,
    block_tile: int = 32,
    scale: Optional[float] = None,
    causal: bool = False,
):
    """v2.50 Prompt 5c Section A.1 — sparse forward returning (O, L_sparse).

    L is per-row natural-log LSE computed over ONLY active blocks
    (sparse-LSE).  All-False rows return L = -INFINITY (sentinel; consumer
    must handle).

    Required by V6NAX backward sparse kernels for LSE consistency
    (Pattern #5 — dense LSE + sparse skip in backward gives wrong
    gradients; consistent sparse-LSE forward + sparse backward gives
    correct gradients).

    Constraints (the with_lse path uses the V1 generator only — LSE not yet
    emitted by the V2 matmul2d kernel; production, not PoC):
      - 2-D mask (NQ, NK) bool — 3-D / 4-D fall back to dense
        sparse_attention_nax (no LSE return)
      - D in {64, 128}, BT in {16, 32, 64}, fp16/bf16
      - mask total bytes >= 4096 (MLX inlines small buffers)

    Returns:
      (O, L): O is (B, Hq, qL, D) same dtype as Q; L is (B, Hq, qL) FP32
    """
    if not _HAS_EXT:
        raise RuntimeError(
            "sparse_attention_nax_with_lse requires the C++ extension."
        )
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    return _ext.sparse_attention_forward_with_lse(
        Q, K, V, block_mask,
        block_tile=block_tile,
        causal=causal,
        scale=float(scale),
    )

# --------------------------------------------------------------------------
# Density-thresholded dispatcher.
#
# Routing logic:
#   density < density_threshold:  sparse_attention_nax (NAX kernel, LCSA path)
#   density >= density_threshold: SDPA + expanded float bias
#
# Default kernel is V1 (per-thread FA-2). V2 cooperative-tensor kernel is
# opt-in via `MFA_LCSA_KERNEL_VERSION=v2` per v2.35.0 SHIP_OPT_IN verdict
# (docs/lcsa-nax/lcsa-nax-coop-rewrite-results.md). V2 wins vs SDPA+bias
# across all tested cells but cross-session range > 10% on 5/7 shapes →
# conservative opt-in to let users explicitly test V2 in their environment.
#
# v2.50-Sprint1 empirical recalibration (M5+ NAX hardware):
# The v2.50-NAX-coverage audit (docs/audits/v50-nax-coverage/) measured
# flash_attention_sparse 1.26× slower than dense SDPA at density 0.023 on
# M5 Max — root-caused to the dispatcher routing density >= 0.02 to the
# SDPA+bias path on M5+, where the bias expansion overhead exceeds the
# compute savings.
#
# Sprint 1 density sweep (M5 Max, B=1 H=12 qL=kL=4096 D=128 fp16 BT=32):
#
#   density | NAX (ms) | SDPA+bias (ms) | dense SDPA (ms) | NAX wins?
#   0.0156  |   0.77   |   2.63         |   2.44          | YES (NAX/dense 0.32×)
#   0.0233  |   0.38   |   2.62         |   2.38          | YES (NAX/dense 0.16×)
#   0.0463  |   0.43   |   2.63         |   2.44          | YES (NAX/dense 0.18×)
#   0.0841  |   0.51   |   2.63         |   2.42          | YES (NAX/dense 0.21×)
#   0.1573  |   0.64   |   2.57         |   2.39          | YES (NAX/dense 0.27×)
#   0.2947  |   0.91   |   2.60         |   2.39          | YES (NAX/dense 0.38×)
#   0.5327  |   1.39   |   2.59         |   2.39          | YES (NAX/dense 0.58×)
#   0.7539  |   1.83   |   2.63         |   2.40          | YES (NAX/dense 0.76×)
#   0.9019  |   2.18   |     —          |   2.42          | YES (NAX/dense 0.90×)
#   0.9515  |   2.24   |     —          |   2.38          | YES (NAX/dense 0.94×)
#   0.9906  |   2.32   |     —          |   2.39          | YES (NAX/dense 0.97×)
#   1.0000  |   2.33   |     —          |   2.40          | YES (NAX/dense 0.97×)
#
# LCSA NAX wins at EVERY density level on M5+.  The SDPA+bias path is
# never optimal on M5+ NAX hardware.  Threshold raised from 0.02 to 1.01
# to always route through NAX on M5+ (preserves dispatcher interface for
# non-M5 callers; M1/M3 callers continue to use threshold=0.02 explicit if
# they invoke this dispatcher).
#
# Historical context: the 0.02 threshold was V1's break-even on older
# hardware (Phase 1.4 sweep, M1/M3 V1 sparse STEEL kernel).  M5+ NAX V2
# inverts the trade-off — per-tile dispatch overhead is fully amortized by
# the cooperative-tensor MMA primitives, and tile-skip via block_mask is
# nearly free.
# --------------------------------------------------------------------------

# M5+ NAX optimal default: always route to NAX (1.01 > 1.0 means density
# never exceeds the threshold).  v2.50-Sprint1 empirical recalibration.
# Non-NAX callers should pass `density_threshold=0.02` explicitly to
# preserve the V1 break-even semantics for M1/M3 STEEL paths.
DEFAULT_DENSITY_THRESHOLD = 1.01


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
    # III-4 R10: this dispatcher defaults block_tile=16 (FlashVSR LCSA
    # convention), whereas the lower-level sparse_attention_nax* helpers
    # default to 32.  Intentional — callers pass an explicit block_tile
    # matching their mask; the default only applies to the FlashVSR path.
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
        density_threshold: route boundary. Default
            ``DEFAULT_DENSITY_THRESHOLD`` (1.01 since v2.50-Sprint1 — always
            route to NAX on M5+; pass 0.02 explicitly for the M1/M3 STEEL
            V1 break-even semantics). (III-4 R10: was documented as 0.02.)
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
    out = mx.fast.scaled_dot_product_attention(
        Q, K, V, scale=scale, mask=bias)
    # III-4 pass-3 F2: a fully-masked query row makes the bias row all
    # -inf; mx.fast.scaled_dot_product_attention returns NaN there.  The
    # NAX kernel branch (above) emits ZEROS for empty rows; match it (the
    # II-6 empty-row contract) so the two dispatch branches agree.  A row
    # is active iff it has at least one unmasked (bias == 0) key.  The
    # [..., qL, 1] activity mask broadcasts over the value dim of
    # out [..., qL, D].
    row_active = (mx.max(bias, axis=-1, keepdims=True) >= 0)
    out = mx.where(row_active, out, mx.zeros_like(out))
    return out


__all__ = [
    "sparse_attention_nax",
    "sparse_attention_dispatch",
    "decide_auto_version",
    "DEFAULT_DENSITY_THRESHOLD",
]
