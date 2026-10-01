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
import warnings
import contextlib
import contextvars
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
# v2.36.1 — shape-aware sparse default per canonical methodology
# (`docs/methodology/canonical-protocol.md`).
#
# Calibrated from 3-session canonical re-bench
# (`docs/methodology/canonical-bench-results.md`):
#   - 7/7 tested shapes graduate V6NAX-sparse eligible (6 CONFIDENT + 1 BOUNDARY,
#     0 HIGH_VARIANCE)
#   - Smallest tested work product: qL=4096, kL=4096, D=128 -> 2.15e9
#   - Below this work product, no canonical-protocol data exists. To
#     honor DC9 (empirical calibration, not extrapolation), we keep
#     scalar-fallback default for shapes smaller than the smallest tested.
#
# Users can override via `MFA_LCSA_KERNEL_VERSION=v1` or `=v2` env var.  These
# legacy public aliases are retained: v1=scalar_fallback, v2=v6nax_sparse.
# ---------------------------------------------------------------------------
_V2_DEFAULT_WORK_THRESHOLD = 2_147_483_648  # = 4096 * 4096 * 128

SPARSE_KERNEL_SCALAR_FALLBACK = "scalar_fallback"
SPARSE_KERNEL_V6NAX = "v6nax_sparse"

_KERNEL_VERSION_ALIASES = {
    "v1": "v1",
    SPARSE_KERNEL_SCALAR_FALLBACK: "v1",
    "sparse_scalar_fallback": "v1",
    "v2": "v2",
    SPARSE_KERNEL_V6NAX: "v2",
    "v6_nax_sparse": "v2",
    "v6-nax-sparse": "v2",
    "v6nax": "v2",
}


def _normalize_kernel_version_alias(value: str) -> Optional[str]:
    """Return the legacy public alias understood by the C++ binding.

    `kernel_version` and `MFA_LCSA_KERNEL_VERSION` are public surfaces, so
    "v1"/"v2" remain stable aliases even though the internal paths are now named
    scalar_fallback / v6nax_sparse.
    """
    return _KERNEL_VERSION_ALIASES.get(value.strip().lower())


def decide_auto_version(
    density: float, qL: int, kL: int, D: int = 128
) -> str:
    """Capability-based V6NAX sparse attention default (audit Phase F).

    Routes the V6NAX-sparse-capable head dims (D in {64, 128}) to the V6NAX
    matmul2d kernel. The public return value intentionally remains the legacy
    alias "v2"; the C++ binding normalizes it to the V6NAX sparse path.
    The old v2.36.1 `qL*kL*D >= 2^31` work-product threshold is RETIRED — Phase E
    measured the scalar fallback is never fastest (V6NAX sparse is 19-59x faster), so the
    threshold only mis-routed D=64 (always < 2^31) and D=128 small-N to the slow
    scalar fallback. The C++ sparse_attention_forward falls v2->v1 internally
    when V6NAX sparse is ineligible (D outside {64,128}, block_tile!=32, or
    dtype outside f16/bf16); causal is supported, so scalar remains
    the genuine fallback — never the default for a V6NAX-capable shape.

    Decision order:
      1. Env override: MFA_LCSA_KERNEL_VERSION=v1/v2 or canonical alias wins
      2. D in {64, 128} -> "v2"   (legacy alias for V6NAX sparse)
      3. Otherwise (e.g. D=256) -> "v1"   (legacy alias for scalar fallback)

    Args:
        density: block-mask density (currently unused in the threshold
            but accepted for future refinement per DC9 note).
        qL: query sequence length.
        kL: key sequence length.
        D: per-head dimension. Default 128 (production V6NAX sparse set).

    Returns:
        "v1" or "v2" for backward compatibility.

    See docs/methodology/canonical-bench-results.md for calibration data.
    """
    # Env override has highest priority (preserves v2.35.0 SHIP_OPT_IN
    # contract for users who already set the env var).
    env = os.environ.get("MFA_LCSA_KERNEL_VERSION", "").strip().lower()
    env_alias = _normalize_kernel_version_alias(env)
    if env_alias is not None:
        return env_alias

    # Audit Phase F (2026-06-18): route by V6NAX-sparse capability (head_dim), NOT the old
    # `qL*kL*D >= _V2_DEFAULT_WORK_THRESHOLD (2^31)` work-product gate.  Phase E
    # measured the scalar fallback is NEVER fastest (V6NAX sparse is 19-59x faster
    # than scalar and 1.5-3.9x faster than SDPA at low density); the 2^31 threshold
    # mis-routed D=64 (work always < 2^31) and D=128 N<4096 to the slow scalar
    # path.  The C++ `sparse_attention_forward` falls v2->v1 internally when
    # V6NAX sparse is ineligible (D outside {64,128}, BT!=32, or non-f16/bf16),
    # while causal is eligible. Returning "v2" for
    # the V6NAX-capable head-dims selects V6NAX wherever it can run and keeps
    # scalar fallback only as the genuine fallback.
    # (Old threshold const retained above for provenance; no longer gates routing.)
    if D in (64, 128):
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
    # Shape-aware default routing via explicit kernel_version param.  The public
    # legacy aliases remain "v1" / "v2"; C++ maps them to scalar_fallback /
    # v6nax_sparse. Empty string falls back to MFA_LCSA_KERNEL_VERSION env var
    # (legacy v2.35.0 path).  Density is not part of the threshold yet
    # (DC9 note); pass 1.0 as a placeholder.
    kernel_version = decide_auto_version(
        density=1.0,
        qL=Q.shape[2],
        kL=K.shape[2],
        D=Q.shape[-1],
    )
    if kernel_version == "v2":
        block_mask, block_tile, _ = _expand_bt64_for_v6nax(
            Q, K, block_mask, block_tile, causal=causal
        )
    from mlx_mfa import _dispatch_trace as _dtrace
    if _dtrace.recording():
        path = (SPARSE_KERNEL_V6NAX if kernel_version == "v2"
                and block_tile == SPARSE_NAX_KERNEL_BLOCK_TILE
                and Q.dtype in (mx.float16, mx.bfloat16)
                and Q.shape[-1] in SPARSE_NAX_VIABLE_HEAD_DIMS
                else SPARSE_KERNEL_SCALAR_FALLBACK)
        _dtrace.record(path, f"sparse_attention_nax BT={block_tile}")
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

    Constraints:
      - BT=32, D in {64, 128}, fp16/bf16 use the V6NAX sparse cooperative-
        tensor kernel and emit LSE in the same launch. Other accepted BTs
        retain the scalar LSE implementation.
      - 2-D, 3-D, or 4-D bool masks are accepted with the corresponding
        (NQ, NK), (Hq, NQ, NK), or (B, Hq, NQ, NK) layout.
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
    from mlx_mfa import _dispatch_trace as _dtrace
    if _dtrace.recording():
        use_v6nax_lse = (
            block_tile == SPARSE_NAX_KERNEL_BLOCK_TILE
            and Q.dtype in (mx.float16, mx.bfloat16)
            and Q.shape[-1] in SPARSE_NAX_VIABLE_HEAD_DIMS
        )
        _dtrace.record(
            "v6nax_sparse_lse" if use_v6nax_lse else "scalar_fallback_lse",
            f"sparse_attention_nax_with_lse BT={block_tile}",
        )
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
# Historical note: the original public aliases were V1 (per-thread FA-2) and V2
# (cooperative-tensor).  They are now treated as scalar_fallback / v6nax_sparse
# aliases at the C++ boundary; `MFA_LCSA_KERNEL_VERSION=v1` and `=v2` remain
# supported.
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
# Historical single-geometry result only. The hardened same-dtype remap dated
# 2026-07-13 supersedes it for public routing: shape/load/mask regime matters,
# and measured-loss or unmeasured cells now delegate to SDPA. The 1.01 argument
# remains only as a backwards-compatible secondary ceiling.
#
# Historical context: the 0.02 threshold was V1's break-even on older
# hardware (Phase 1.4 sweep, M1/M3 V1 sparse STEEL kernel).
# --------------------------------------------------------------------------

# Backwards-compatible secondary ceiling. The hardened shape/dtype gate below
# is authoritative; 1.01 means this legacy argument does not narrow it unless
# a caller explicitly supplies a lower value.
DEFAULT_DENSITY_THRESHOLD = 1.01

# ── BT-aware NAX-sparse win window (sparse-NAX victory map, engagement-proven) ──
# The native NAX sparse kernel (matmul2d cooperative tensors = Metal 4 / macOS
# 26.0 — STABLE, NOT the macOS-27 beta track) beats dense SDPA only in a specific
# BLOCK-TILE window; the dispatcher used to route on density alone and IGNORE the
# tile, mis-routing BT=16 into a ~5.5× slowdown (footgun). These are the measured
# boundaries; TILE VIABILITY is the PRIMARY gate so the default is safe regardless
# of any hand-tuned density threshold.
#
# β3-INDICATIVE (macOS 27 β3 / Metal 32023.918, M5 Max, MLX 0.31.2,
# 2026-07-13) — RE-VALIDATE on stable macOS. The hardened same-dtype map routes
# ONLY cells that won beyond the 7.53% null floor in both arm orders. B·H is
# restricted to the measured values rather than interpolating the interval.
SPARSE_NAX_VIABLE_BLOCK_TILES = frozenset({32})
SPARSE_NAX_VIABLE_HEAD_DIMS = frozenset({64, 128})
# ── 2.64 B4 (Marco D3, 2026-10-01): the extended envelope is the DEFAULT non-causal law.
# N in [SPARSE_NAX_MIN_N, SPARSE_NAX_MAX_N], ANY B*H, fp16/bf16, the density ceilings
# kept as measured.  MAX_N 200000: engaged + row-correct + 5.3-9.7x at N=200000 and
# B*H 32/40/56 (devnotes/production_shapes_2026-10.md §2; Volet A 16k-144k B1H40).
# MIN_N 2048 (decision D2) — evidence = ONE shape (FlashVSR B1H12, 2048x8192).  The
# B*H allowlist is gone: every measured B*H won at d <= 0.30 (the COVERAGE below —
# documentation, not a gate: DAY-3 Block 1 B*H16 204 cells 0 loss, Phase 2, Volet A).
SPARSE_NAX_MEASURED_BH_COVERAGE = frozenset({1, 4, 12, 16, 32, 40, 56})
SPARSE_NAX_MIN_N = 2048
SPARSE_NAX_MAX_N = 200_000
SPARSE_NAX_DENSITY_CEILING = 0.30
# The lower measured ceilings below held at N < 8192 (the 2.63 map: N=8192 won all
# 36 fp16 cells at d <= 0.30); they are kept in exactly that range.
SPARSE_NAX_LOWER_CEILING_BELOW_N = 8192
# 2.63 policy, kept one release behind MFA_SPARSE_NAX_LEGACY_POLICY=1.
_LEGACY_MEASURED_BH = frozenset({1, 4, 12})
_LEGACY_MIN_N = 4096
_LEGACY_MAX_N = 8192
SPARSE_NAX_MEASURED_BH = _LEGACY_MEASURED_BH       # 2.63 name (legacy policy only)
SPARSE_NAX_D64_BH12_DENSITY_CEILING = 0.25
SPARSE_NAX_D128_BH4_DENSITY_CEILING = 0.05
SPARSE_NAX_CAUSAL_MIN_N = 4096
SPARSE_NAX_CAUSAL_MAX_N = 8192
SPARSE_NAX_CAUSAL_MAX_BH = 12
SPARSE_NAX_CAUSAL_DENSITY_CEILING = 0.30
SPARSE_NAX_CAUSAL_BH4_DENSITY_CEILING = 0.10
SPARSE_NAX_KERNEL_BLOCK_TILE = 32                   # V6NAX sparse BQ=BK is structurally pinned at 32
SPARSE_NAX_EXPANDABLE_BLOCK_TILE = 64               # One BT64 block maps exactly to 2x2 BT32 blocks

# Volet A Phase 1 (spec §1, item 4): at/above this block density the skip's
# per-block loop-control overhead outweighs its shrinking skip benefit, so the
# wrapper diverts to the dense masked route. This dispatch — NOT anything in the
# kernel — is what buys the "<=5% overhead at ~zero sparsity" gate (gate 7).
# Initial 0.85, to be calibrated by the d=0.95 measurement cell; override via
# MFA_SPARSE_D_DENSE_CUTOFF.
SPARSE_NAX_D_DENSE_CUTOFF = 0.85


def _d_dense_cutoff() -> float:
    """spec §1 item 4 — block density at/above which sparse routing diverts to
    the dense masked path (env override: MFA_SPARSE_D_DENSE_CUTOFF).

    RC 2.63.0 (decision D2): an invalid value is REFUSED (it silently fell back to
    0.85, and accepted nan — disabling the cutoff — or a negative value — sending
    everything dense).  Valid: a finite number > 0; > 1 disables the diversion.
    """
    raw = os.environ.get("MFA_SPARSE_D_DENSE_CUTOFF")
    if raw is None:
        return SPARSE_NAX_D_DENSE_CUTOFF
    try:
        val = float(raw)
    except ValueError:
        val = float("nan")
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(
            f"[mlx-mfa] MFA_SPARSE_D_DENSE_CUTOFF must be a finite block density > 0 "
            f"(> 1 disables the dense diversion); got {raw!r}.")
    return val


# RC 2.63.0 (decision D4): a CONTEXT-LOCAL override of the opt-in, so callers such as
# sla_attention never mutate the process-wide os.environ (not thread-safe; a
# concurrent C++ getenv during setenv is undefined behaviour) and can force it OFF.
_EXTENDED_OVERRIDE: "contextvars.ContextVar[bool | None]" = contextvars.ContextVar(
    "mlx_mfa_sparse_extended", default=None)


@contextlib.contextmanager
def _extended_override(value: bool):
    """Force the extended opt-in ON/OFF for the current context (thread / task)."""
    token = _EXTENDED_OVERRIDE.set(bool(value))
    try:
        yield
    finally:
        _EXTENDED_OVERRIDE.reset(token)


def _sparse_extended_enabled() -> bool:
    """2.64 (D3): ``MFA_SPARSE_NAX_EXTENDED`` is a documented, DEPRECATED no-op — the
    extended envelope is the default law (``_nax_sparse_route_viable``).  Always False;
    setting the knob (strict 0/1, still validated) emits a DeprecationWarning.  The
    context override (`_extended_override`) is inert as well."""
    from mlx_mfa._knobs import get_bool_env
    if get_bool_env("MFA_SPARSE_NAX_EXTENDED"):
        warnings.warn(
            "MFA_SPARSE_NAX_EXTENDED is a no-op since mlx-mfa 2.64: the extended sparse "
            "envelope is the default routing law (N in [2048, 200000], any B*H, density "
            "ceilings kept).  MFA_SPARSE_NAX_LEGACY_POLICY=1 restores the 2.63 policy for "
            "one release.", DeprecationWarning, stacklevel=2)
    return False


def _legacy_policy() -> bool:
    from mlx_mfa._knobs import get_bool_env
    return bool(get_bool_env("MFA_SPARSE_NAX_LEGACY_POLICY"))


# The V6NAX sparse kernel refuses block masks smaller than this (C++ sparse_attention:
# "mask total bytes < 4096").
SPARSE_NAX_MIN_MASK_BYTES = 4096


def _mask_bytes(block_mask) -> int:
    n = 1
    for dim in block_mask.shape:
        n *= int(dim)
    return n


def _extended_prepare(D: int, block_mask, N: int, S: int):
    """RC 2.63.0 (decisions D3/D5): the opt-in rules shared by flash_attention_sparse
    and sparse_attention_dispatch.  Loud refusals (spec §1: no silent downgrade) —
    pre-M5, D outside {64, 128}, a mask that is not at 32-block granularity — after
    expanding a 64-block mask EXACTLY to 32 blocks (each block -> 2x2).  Returns the
    (possibly expanded) mask."""
    from mlx_mfa.attention import _get_is_m5_plus_cached
    if not _get_is_m5_plus_cached():
        raise RuntimeError(
            "MFA_SPARSE_NAX_EXTENDED=1 requires M5+ (NAX) hardware; this chip "
            "is pre-M5 where STEEL sparse is backlog (spec §1/§5). Unset the "
            "env for the default routing (graceful SDPA fallback).")
    if D not in (64, 128):
        raise ValueError(
            f"MFA_SPARSE_NAX_EXTENDED=1: head_dim must be 64 or 128 (spec §1 "
            f"v1 matrix); got D={D} (D=256/512 are outside the extended "
            f"envelope).")
    nq32, nk32 = -(-N // 32), -(-S // 32)
    shape = tuple(block_mask.shape[-2:])
    if shape != (nq32, nk32) and shape == (-(-N // 64), -(-S // 64)):
        block_mask = mx.repeat(mx.repeat(block_mask, 2, axis=-2), 2, axis=-1)[
            ..., :nq32, :nk32]
        shape = (nq32, nk32)
    if shape != (nq32, nk32):
        raise ValueError(
            f"MFA_SPARSE_NAX_EXTENDED=1: block tile must be 32 (spec §1, "
            f"structural); a [{shape[0]}, {shape[1]}] mask is not at 32-block (or "
            f"64-block) granularity for N={N}, S={S} (expected [{nq32}, {nk32}]). "
            f"Either rebuild the mask at 32-block granularity, or use the default "
            f"path (MFA_SPARSE_NAX_EXTENDED=0 / sla_attention(extended=False)), which "
            f"accepts other geometries such as the 32x16 STEEL masks.")
    return block_mask


def _expand_bt64_exact(D: int, block_mask, N: int, S: int):
    """2.64 (D3): the extended envelope's EXACT 64 -> 32 block expansion, on the default
    path (no refusals).  A ceil-granular 64-token mask [.., ceil(N/64), ceil(S/64)] —
    LongCat BSA / VSA tiles — becomes the 32-token mask the V6NAX kernel takes (each
    block -> identical 2x2 sub-blocks, sliced to ceil(N/32)), BEFORE auto_pad so a
    non-aligned sequence reaches the padded kernel.  Same conditions as the router's
    aligned BT64 expansion: M5+, D in {64, 128}, the V6 sparse backward not opted in
    (it needs bt >= 64), not under MFA_SPARSE_NAX_LEGACY_POLICY (2.63 refused these)."""
    from mlx_mfa.attention import _get_is_m5_plus_cached
    from mlx_mfa._knobs import get_bool_env
    from mlx_mfa._env_aliases import get_bool_env_aliased
    nq32, nk32 = -(-N // 32), -(-S // 32)
    shape = tuple(block_mask.shape[-2:])
    if (shape == (nq32, nk32) or shape != (-(-N // 64), -(-S // 64))
            or D not in SPARSE_NAX_VIABLE_HEAD_DIMS or not _get_is_m5_plus_cached()
            or get_bool_env_aliased("MFA_ENABLE_V6_BACKWARD") or _legacy_policy()
            or get_bool_env("MFA_DISABLE_AUTO_HOOKS")):
        return block_mask
    return mx.repeat(mx.repeat(block_mask, 2, axis=-2), 2, axis=-1)[..., :nq32, :nk32]


def _nax_sparse_route_viable(Q, K, block_tile, density, *, causal=False, V=None) -> bool:
    """Whether the V6NAX sparse route serves this call under the DEFAULT policy.

    Capacity first (32-token tile, fp16/bf16, D in {64, 128}, V matching Q/K), then:
      * 2.64 non-causal law (B4/B5): qL, kL in [SPARSE_NAX_MIN_N, SPARSE_NAX_MAX_N],
        qL != kL allowed (the kernel documents non-causal rectangular), any B*H,
        density <= 0.30 — and the measured lower ceilings (D128 B*H4 0.05, D64 B*H12
        0.25) below N=8192, where they were measured.
      * causal: the 2.63 exact cells, unchanged (qL == kL — U2).
    ``MFA_SPARSE_NAX_LEGACY_POLICY=1`` restores the complete 2.63 decision."""
    if block_tile not in SPARSE_NAX_VIABLE_BLOCK_TILES:
        return False
    if Q.dtype not in (mx.float16, mx.bfloat16):
        return False
    D = int(Q.shape[3])
    if D not in SPARSE_NAX_VIABLE_HEAD_DIMS:
        return False
    # API-04 (review 2026-09): the NAX kernel requires V to share Q/K's head_dim,
    # dtype and K's length (C++ raises "head_dim mismatch" otherwise).  A V the
    # kernel cannot serve (e.g. documented-valid asymmetric D_v) takes the SDPA
    # route instead of crashing.  V=None keeps the historical Q/K-only decision.
    if V is not None and (int(V.shape[3]) != D or V.dtype != Q.dtype
                          or int(V.shape[2]) != int(K.shape[2])):
        return False
    if _legacy_policy():
        return _route_viable_263(Q, K, density, causal=causal)
    qL, kL = int(Q.shape[2]), int(K.shape[2])
    bh = int(Q.shape[0]) * int(Q.shape[1])
    if causal:
        return qL == kL and _causal_cells_263(qL, D, bh, Q.dtype, density)
    if not (SPARSE_NAX_MIN_N <= min(qL, kL) and max(qL, kL) <= SPARSE_NAX_MAX_N):
        return False
    ceiling = SPARSE_NAX_DENSITY_CEILING
    if max(qL, kL) < SPARSE_NAX_LOWER_CEILING_BELOW_N:
        if bh == 4 and D == 128:
            ceiling = SPARSE_NAX_D128_BH4_DENSITY_CEILING
        elif bh == 12 and D == 64:
            ceiling = SPARSE_NAX_D64_BH12_DENSITY_CEILING
    return density <= ceiling


def _causal_cells_263(qL, D, bh, dtype, density) -> bool:
    """The 2.63 causal cells (unchanged in 2.64)."""
    # bf16 has one measured winning causal cell; keep it exact.
    if dtype == mx.bfloat16:
        return (qL == 4096 and D == 128 and bh == 4
                and density <= SPARSE_NAX_CAUSAL_BH4_DENSITY_CEILING)
    if qL == 4096 and D == 128 and bh == 4:
        return density <= SPARSE_NAX_CAUSAL_BH4_DENSITY_CEILING
    if qL == 4096 and D == 128 and bh == 12:
        return density <= SPARSE_NAX_CAUSAL_DENSITY_CEILING
    if qL == 8192 and D in (64, 128) and bh == 12:
        return density <= SPARSE_NAX_CAUSAL_DENSITY_CEILING
    return False


def _route_viable_263(Q, K, density, *, causal=False) -> bool:
    """The complete 2.63 policy (capacity already checked) — MFA_SPARSE_NAX_LEGACY_POLICY=1."""
    D = int(Q.shape[3])
    qL, kL = int(Q.shape[2]), int(K.shape[2])
    if qL != kL:                       # square-only in v1 (capacity-adjacent)
        return False
    if qL > _LEGACY_MAX_N:
        return False
    bh = int(Q.shape[0]) * int(Q.shape[1])
    if causal:
        return _causal_cells_263(qL, D, bh, Q.dtype, density)
    if Q.dtype == mx.bfloat16:
        return (_LEGACY_MIN_N <= qL <= _LEGACY_MAX_N
                and D == 128 and bh == 12
                and density <= SPARSE_NAX_DENSITY_CEILING)
    if (qL == _LEGACY_MAX_N and bh in _LEGACY_MEASURED_BH
            and density <= SPARSE_NAX_DENSITY_CEILING):
        return True
    if not (_LEGACY_MIN_N <= qL <= _LEGACY_MAX_N):
        return False
    if bh == 12 and D == 128:
        return density <= SPARSE_NAX_DENSITY_CEILING
    if bh == 12 and D == 64:
        return density <= SPARSE_NAX_D64_BH12_DENSITY_CEILING
    if bh == 4 and D == 128:
        return density <= SPARSE_NAX_D128_BH4_DENSITY_CEILING
    return False


def _quasi_dense_nax_viable(Q, K, V, block_mask, block_tile, causal) -> bool:
    """2.64 B6 (D2): at density >= D_DENSE_CUTOFF the best measured arm is the V6NAX
    kernel (0.42x of the 2.63 path on the FlashVSR shapes; ~= dense SDPA at d 0.999)
    when it can serve the call: non-causal (the evidence), 32-token tile, the capacity
    checks, lengths within the law's N bounds.  Otherwise SDPA + bool keep-mask."""
    if causal or block_tile != SPARSE_NAX_KERNEL_BLOCK_TILE or _legacy_policy():
        return False
    qL, kL = int(Q.shape[2]), int(K.shape[2])
    if not (SPARSE_NAX_MIN_N <= min(qL, kL) and max(qL, kL) <= SPARSE_NAX_MAX_N):
        return False
    from mlx_mfa.attention import _nax_sparse_capacity_ok
    return _nax_sparse_capacity_ok(Q, K, V, block_mask, causal)


def _expand_bt64_for_v6nax(
    Q,
    K,
    block_mask,
    block_tile,
    *,
    causal=False,
    require_public_route=True,
):
    """Return a viable BT64 mask at the pinned BT32 V6NAX granularity.

    Repeating each source block 2x2 preserves the token-level mask exactly.
    The expanded representation is normally revalidated against the public NAX
    gate; malformed and out-of-window inputs retain their prior fallback path.
    Internal opt-in backward orchestrators may set `require_public_route=False`:
    they have a separate eligibility gate but still require the V6NAX LSE
    forward rather than the scalar BT64 implementation.
    """
    if block_tile != SPARSE_NAX_EXPANDABLE_BLOCK_TILE:
        return block_mask, block_tile, False
    q_len, k_len = int(Q.shape[2]), int(K.shape[2])
    if q_len % block_tile or k_len % block_tile:
        return block_mask, block_tile, False
    if tuple(block_mask.shape[-2:]) != (q_len // block_tile, k_len // block_tile):
        return block_mask, block_tile, False
    expanded = mx.repeat(mx.repeat(block_mask, 2, axis=-2), 2, axis=-1)
    density = mask_density(expanded)                       # 2.64 B3: no fp32 copy
    if require_public_route and not _nax_sparse_route_viable(
        Q, K, SPARSE_NAX_KERNEL_BLOCK_TILE, density, causal=causal
    ):
        return block_mask, block_tile, False
    return expanded, SPARSE_NAX_KERNEL_BLOCK_TILE, True


def mask_density(block_mask) -> float:
    """Exact block density (fraction of True / non-zero entries) WITHOUT an fp32 copy.

    2.64 B3: the routers used ``mx.mean(mask.astype(mx.float32))`` — a transient fp32
    copy of the whole per-head mask (4x its bytes: 3.6 GB at N=168,960 x H32 x BT32),
    the bulk of the Phase-2 2-5x peak-memory excess.  Row counts are reduced first
    (int, [..., NQ] — no full-size intermediate), then summed in int64 (no int32
    overflow above 2^31 entries).  Exact: count / size."""
    m = block_mask if block_mask.dtype == mx.bool_ else (block_mask != 0)
    total = mx.sum(mx.sum(m, axis=-1).astype(mx.int64))
    return int(total.item()) / max(int(m.size), 1)


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
    """Shape-gated sparse attention dispatcher.

    Routes to V6NAX only inside `_nax_sparse_route_viable`; all other cells use
    MLX SDPA with an expanded float bias. `density_threshold` is retained as a
    backwards-compatible, further-restrict-only ceiling inside that envelope.

    Args:
        Q, K, V, block_mask: same as sparse_attention_nax.
        block_tile: defaults to 16 (Phase 1.3 winner).
        scale: query scale; default 1/sqrt(D).
        causal: supported on the V6NAX sparse path inside the β3-measured
            causal window. If True and the dispatcher chooses the SDPA path,
            an explicit causal bias is added to the block_mask bias before
            SDPA dispatch.
        density_threshold: optional secondary route ceiling. Default
            ``DEFAULT_DENSITY_THRESHOLD`` (1.01) leaves the canonical gate
            unchanged; lower values may only narrow the V6NAX region.
        density: optional pre-computed density (avoids a reduction per call).
        precomputed_bias: optional pre-built (qL, kL) float bias - if the
            caller already has the float bias (cache-HIT pattern from v2.33.1),
            passing it skips the internal bias expansion. Used only on the
            SDPA route; ignored when Sprint B path is taken.

    Returns:
        O: same shape/dtype as Q.
    """
    # CX-R8-01 (volet L): validate Q/K/V at the dispatcher entry, BEFORE the
    # native-vs-SDPA route split, so BOTH routes are guarded. The SDPA route
    # previously accepted a V whose kv-seq disagreed with K → finite-wrong / NaN
    # (and a dtype-mismatched V).  Dtype is required EQUAL (matching the native
    # route's contract) but NOT restricted to f16/bf16 at the ENTRY — f32 is
    # valid and is routed to SDPA below (CX-R9-02; the native kernel is
    # f16/bf16-only, so f32 never reaches it).
    # V head_dim is left unconstrained: asymmetric D_v is valid on the SDPA route.
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise ValueError(
            "sparse_attention_dispatch: Q, K, V must be 4-D [B, H, N, D]")
    if Q.shape[0] != K.shape[0] or Q.shape[0] != V.shape[0]:
        raise ValueError(
            "sparse_attention_dispatch: Q, K, V must share the batch dim")
    if K.shape[2] != V.shape[2]:
        raise ValueError(
            "sparse_attention_dispatch: K and V must share the kv sequence length "
            f"(Sk={K.shape[2]}, Sv={V.shape[2]})")
    if K.shape[1] != V.shape[1]:
        raise ValueError(
            "sparse_attention_dispatch: K and V must have the same number of heads "
            f"(Hk={K.shape[1]}, Hv={V.shape[1]})")
    if Q.shape[3] != K.shape[3]:
        raise ValueError(
            "sparse_attention_dispatch: Q and K must share head_dim for Q@K^T "
            f"(Dq={Q.shape[3]}, Dk={K.shape[3]})")
    if K.shape[1] <= 0 or Q.shape[1] % K.shape[1] != 0:
        raise ValueError(
            "sparse_attention_dispatch: Q heads must be a positive multiple of "
            f"KV heads (Hq={Q.shape[1]}, Hk={K.shape[1]}) for GQA")
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError(
            "sparse_attention_dispatch: Q, K, V must share dtype")

    if density is None:
        # 2.64 B3: exact count, no fp32 copy.  NEPB-05: the synchronous read is
        # allowed inside a graph transformation (async_eval is not).
        density = mask_density(block_mask)
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    # CX-R9-02 (volet M): the native sparse kernel is f16/bf16-only — route any
    # non-f16/bf16 dtype (e.g. float32) to the SDPA path REGARDLESS of density so
    # f32 produces correct attention consistently. Previously f32 was
    # density-dependent (ran on SDPA, raised on the native route).
    _force_sdpa = Q.dtype not in (mx.float16, mx.bfloat16)
    _small_mask = False
    if not _force_sdpa and _sparse_extended_enabled():
        # RC 2.63.0 (D5): the opt-in rules shared with flash_attention_sparse — loud
        # refusals, exact 64 -> 32 block expansion, masks below the kernel's 4096-byte
        # minimum to the dense route (the kernel would raise on them).
        _expanded = _extended_prepare(Q.shape[3], block_mask, Q.shape[2], K.shape[2])
        if _expanded is not block_mask:
            block_mask, block_tile = _expanded, SPARSE_NAX_KERNEL_BLOCK_TILE
            density = mask_density(block_mask)                # 2.64 B3
        elif tuple(block_mask.shape[-2:]) == (-(-Q.shape[2] // 32), -(-K.shape[2] // 32)):
            block_tile = SPARSE_NAX_KERNEL_BLOCK_TILE
        _small_mask = _mask_bytes(block_mask) < SPARSE_NAX_MIN_MASK_BYTES
    if not _force_sdpa:
        block_mask, block_tile, _ = _expand_bt64_for_v6nax(
            Q, K, block_mask, block_tile, causal=causal
        )
    # Hardened same-dtype β3 map: route only the exact measured B·H/dtype/
    # causal regions encoded in `_nax_sparse_route_viable`. N=2048 and every
    # unmeasured region fall through to SDPA regardless of density_threshold.
    # This removes the BT=16 ~5.5× mis-route footgun: routing correctness no longer
    # depends on a caller hand-tuning the density threshold. density_threshold is
    # retained as a secondary (further-restrict-only) tunable within the window.
    _route = ((not _force_sdpa) and not _small_mask
              and _nax_sparse_route_viable(Q, K, block_tile, density, causal=causal, V=V)
              and density < density_threshold
              and density < _d_dense_cutoff())
    # 2.64 B6 (D2): at/above the dense cutoff the V6NAX kernel is the best measured arm
    # whenever it can serve the call (same rule as flash_attention_sparse).
    if (not _route and not _force_sdpa and not _small_mask
            and density >= _d_dense_cutoff() and density < density_threshold
            and _quasi_dense_nax_viable(Q, K, V, block_mask, block_tile, causal)):
        _route = True
    if _route:
        return sparse_attention_nax(
            Q, K, V, block_mask,
            block_tile=block_tile,
            scale=scale,
            causal=causal,
        )
    # SDPA fallback.  2.64 B2 (D4): a BOOL keep-mask built from the block mask — the
    # float bias survives only for a caller-supplied real additive bias
    # (``precomputed_bias``).  Empty rows -> zeros (II-6; the NAX branch above emits
    # zeros), decided per element row; the causal rule is the canonical zero-clamped
    # one (NAMING.md; review DSP-12).
    from mlx_mfa import _dispatch_trace as _dtrace

    _dtrace.record("sdpa", "extended: mask < 4096 B (below the V6NAX minimum) -> dense"
                   if _small_mask else "sparse_attention_dispatch outside hardened gate")
    qL = Q.shape[2]
    kL = K.shape[2]
    if precomputed_bias is not None:
        bias = precomputed_bias
        if causal:
            q_idx = mx.arange(qL).reshape(-1, 1) + max(0, kL - qL)
            k_idx = mx.arange(kL).reshape(1, -1)
            causal_bias = mx.where(k_idx > q_idx,
                                    mx.array(-float("inf"), dtype=Q.dtype),
                                    mx.array(0.0, dtype=Q.dtype))
            bias = bias + causal_bias
        out = mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale, mask=bias)
        row_active = (mx.max(bias, axis=-1, keepdims=True) >= 0)   # unchanged 2.63 rule
        return mx.where(row_active, out, mx.zeros_like(out))
    from mlx_mfa.attention import _sparse_keep_mask, _sparse_sdpa_rows
    # Same geometry as the former _bool_mask_to_float_bias: block_tile on both axes,
    # no trimming (a non-aligned length raises in SDPA exactly as before).
    keep, row_active = _sparse_keep_mask(
        block_mask, block_mask.shape[-2] * block_tile, block_mask.shape[-1] * block_tile,
        block_tile, block_tile, causal, where="sparse_attention_dispatch (SDPA fallback)")
    return _sparse_sdpa_rows(Q, K, V, scale, keep, row_active)


__all__ = [
    "sparse_attention_nax",
    "sparse_attention_dispatch",
    "decide_auto_version",
    "DEFAULT_DENSITY_THRESHOLD",
]
