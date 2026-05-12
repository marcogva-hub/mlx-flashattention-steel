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
from typing import Optional

import mlx.core as mx

try:
    from . import _ext  # nanobind extension
    _HAS_EXT = hasattr(_ext, "sparse_attention_forward")
except ImportError:
    _ext = None
    _HAS_EXT = False


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
    return _ext.sparse_attention_forward(
        Q, K, V, block_mask,
        block_tile=block_tile,
        causal=causal,
        scale=float(scale),
    )


__all__ = ["sparse_attention_nax"]
