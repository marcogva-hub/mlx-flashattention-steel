"""Shape-aware dispatch policy for MFA vs SDPA backend selection.

The dispatch decision is based on empirical crossover points derived from
bench_dispatch_matrix.py on Apple M1 Max (M1 Max, f16, B=1 H=8).

Conservative principle: only activate MFA when it is EXPECTED to be faster
than MLX SDPA.  It is always better to route to SDPA (and be ~1.0x) than to
route to MFA incorrectly (and be 0.5x).

Baseline crossover data (decision passes through 2026-03-12):
  - D=64  causal: MFA wins with low-N crossover on current M1/M2 defaults
  - D=128 causal: MFA wins with low-N crossover on current M1/M2 defaults
  - D=256 causal: narrow win only for f16 on M1/M2 at long N (>=4096)
  - D=256 bf16 and all non-causal dense routes remain conservative SDPA
  - D=512 dense remains conservative SDPA (0/32 wins in decision pass);
    post-autoresearch ceiling: 0.80x geomean (74 iters, BK=128 BD_HALF=32
    optimal). Autoresearch (2026-03-20, M1 Max) exhaustively explored
    BK∈{4..256}, BD_HALF∈{16..256}, BQ∈{16..64}, WM∈{1..8}, plus exotic
    approaches (direct device reads, lazy Q, no-unroll, half-padding).
    Root cause: 64 barriers/K-tile (16 D-split passes × ~4 barriers each),
    ~6% constant ALU overhead vs SDPA that cannot be eliminated with D-split
    architecture. Asymptotic: 0.96x at N=32k — approaches but never crosses
    1.0x. Large-batch profiles (B=4 H=8 N=8192) reach 0.97x.

Override the dispatch table at runtime::

    import os
    os.environ["MLX_MFA_DISPATCH_TABLE"] = "/path/to/custom_thresholds.json"

Enable verbose dispatch logging::

    os.environ["MLX_MFA_VERBOSE_DISPATCH"] = "1"
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional

import mlx.core as mx


def _invalidate_cached_env():
    """Notify C++ MFAEnvConfig that cached env vars changed.

    Must be called after any os.environ mutation of cached MFA_* vars
    (MFA_V2_FORCE_BK, MFA_V2_BQ64, MFA_FORCE_GEN, MFA_V3_FORCE_BK_*,
    MFA_V5_FORCE_*, MFA_V2_BD_HALF_D512, etc.).

    Dispatch gate vars (MFA_ENABLE_V3, MFA_ENABLE_V5, etc.) are live-read
    static methods and do NOT require this call.
    """
    try:
        from mlx_mfa._ext import _invalidate_env_config
        _invalidate_env_config()
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Default thresholds: minimum N to activate MFA for (D, causal) pair.
# Derived from M1 Max dispatch matrix baseline.  999_999 effectively disables.
# ---------------------------------------------------------------------------

# D=512 autoresearch ceiling (2026-03-20, M1 Max, 74 iterations):
#   Best achievable: BD_HALF=32 BK=128 BQ=32 WM=4 → 0.80x geomean SDPA
#   (B=2 H=8, N=1024–8192, f16, causal).
#   Asymptotic: 0.96x at N=32768 — approaches but never crosses 1.0x.
#   Root cause: 16 D-split passes × ~4 barriers/pass = 64 barriers/K-tile,
#   yielding ~6% constant ALU overhead vs SDPA's fused single-pass approach.
#   This overhead is intrinsic to the D-split architecture on M1/M2 and
#   cannot be eliminated by block-config tuning alone.
_D512_CONSERVATIVE_MIN_N = 999_999


_DEFAULT_THRESHOLDS: dict[tuple[int, bool], int] = {
    # D=64 causal: V2 kernel (BK=64) raises crossover. Measured (M1 Max, 12 trials):
    #   N=512→1.18x, N=1024→1.14x, N=2048→1.25x, N=4096→1.86x, N=16384→2.20x.
    # Threshold N=1024 (conservatively above break-even, stable across runs).
    (64,  True):  1024,
    # D=64 non-causal: V2 wins on M1/M2 (1.06x at N=2048, 1.43x at N=8192).
    # Old measurements (0.98x) predated V2 BK=64 optimization or used different MLX.
    (64,  False): 2048,
    # D=128 causal: V2 delivers 1.33x at N=2048, 1.60x at N=4096, 1.76x at N=16384.
    # Old V1 threshold was N=8192 (too conservative). New threshold: N=2048.
    (128, True):  2048,
    # D=128 non-causal: V2 wins on M1/M2 (1.12x at N=2048, 1.51x at N=8192).
    # Old measurements (0.90x) predated V2 BK=64 optimization or used different MLX.
    (128, False): 2048,
    # D=256 causal: dtype-specific logic now lives in should_use_mfa():
    #   - M1/M2 + f16: promote at N>=4096
    #   - M1/M2 + bf16: keep SDPA
    # This table entry remains as a conservative fallback when dtype is unknown.
    (256, True):  8192,
    # D=256 non-causal remains a clear loss (~0.55x in decision pass).
    (256, False): 999_999,
    # D=512 is handled by _d512_min_n() as a separate decision family.
    # Keep conservative fallback table entries for custom/legacy callers.
    (512, True):  _D512_CONSERVATIVE_MIN_N,
    (512, False): _D512_CONSERVATIVE_MIN_N,
}

# M3+ thresholds: V1 double-buffer (2 barriers/tile) replaces V2 for D<=128 causal.
# On M3+ hardware, reduced TGP bandwidth makes V2's 3-4 barriers/tile a net loss.
# V1 wins 1.5-3.7x over V2 at D<=128 causal on M4 Max (see mfa_attention.cpp).
_M3_THRESHOLDS: dict[tuple[int, bool], int] = {
    # D=64 causal: V1 routed on M3+ (dispatch guard). V1 wins from N=512.
    (64,  True):  512,
    # D=64 non-causal: M3+ SDPA wins (0.60x at N=8192 on M4 Max). Disabled.
    (64,  False): 999_999,
    # D=128 causal: V1 routed on M3+ (dispatch guard). V1 wins from N=1024.
    (128, True):  1024,
    # D=128 non-causal: M3+ SDPA wins (0.68x at N=8192 on M4 Max). Disabled.
    (128, False): 999_999,
    # D=256/512: V2 D-split still used (not affected by V1-over-V2 routing).
    (256, True):  999_999,
    (256, False): 999_999,
    (512, True):  _D512_CONSERVATIVE_MIN_N,
    (512, False): _D512_CONSERVATIVE_MIN_N,
}

# M5+ NAX thresholds: Apple's `steel_attention_nax.h` is the optimal forward
# path on canonical shapes (D∈{64,128}, qL>8, no exotic features). v2.32.0
# routes forward to MLX SDPA on those shapes — V34 NAX-direct matches it but
# does not beat it cross-session. Non-canonical D and decode (qL≤8) keep
# mlx-mfa kernels because SDPA's NAX path doesn't cover them on M5+.
#
# 999_999 means "always route to SDPA at this (D, causal) regardless of N".
# V34-backward training carve-out (env-var opt-in, D=64 qL≥4096) is in
# `_v34_backward_carveout()` further down.  The earlier
# `_should_use_mfa_m5_nax_carveout()` canonical-path hook was deleted in
# v2.38.0 (dormant since v2.32.0; no Sprint A.6 carve-outs materialized).
_M5_NAX_THRESHOLDS: dict[tuple[int, bool], int] = {
    # D=64 canonical: SDPA NAX wins. Route SDPA. Carve-outs handle specific
    # MFA-winning shapes (e.g., FlashVSR-dense) inline.
    (64,  True):  999_999,
    (64,  False): 999_999,
    # D=128 canonical: SDPA NAX wins. v2.31.0 V34 results showed parity
    # within session; cross-session diagnostic showed legacy/MFA path
    # depends on environmental conditions whereas SDPA NAX is stable.
    (128, True):  999_999,
    (128, False): 999_999,
    # D=256/512: SDPA's NAX path doesn't cover these. Keep V2 D-split etc.
    # (Defer to non-NAX thresholds for these head_dims.)
    (256, True):  999_999,
    (256, False): 999_999,
    (512, True):  _D512_CONSERVATIVE_MIN_N,
    (512, False): _D512_CONSERVATIVE_MIN_N,
}

_verbose: bool = os.environ.get("MLX_MFA_VERBOSE_DISPATCH", "0") == "1"

# Native STEEL backward policy (targeted, benchmark-backed only).
# 2026-03-12 targeted pass on M1 Max found 0/16 winning configs for:
#   D in {64, 128}, N in {2048, 4096, 8192, 16384}, causal=True, f16/bf16.
# Keep auto-dispatch disabled until a measured winning regime appears.
_NATIVE_BWD_MIN_N: dict[tuple[int, str], int] = {
    (64, "float16"): 999_999,
    (64, "bfloat16"): 999_999,
    (128, "float16"): 999_999,
    (128, "bfloat16"): 999_999,
}

# Sage decode auto-routing policy (specialized, benchmark-backed only).
# 2026-03-12 decode matrix (post-bwd pass):
#   - 13/240 wins overall, with most rows losing vs dense STEEL.
#   - Production-like GQA wins appeared only in very narrow D=128 windowed
#     decode regimes at N_cache=4096.
# Auto-promotion is intentionally strict to avoid broad regressions.

# Cached custom dispatch table (loaded once from MLX_MFA_DISPATCH_TABLE env var).
_custom_thresholds: Optional[dict[tuple[int, bool], int]] = None
_custom_table_loaded = False


def _dispatch_dtype_key(dtype) -> Optional[str]:
    """Normalize dtype objects/strings for dispatch policy lookup."""
    if dtype is None:
        return None
    dtype_str = str(dtype)
    if dtype_str in {"float16", "mlx.core.float16"}:
        return "float16"
    if dtype_str in {"bfloat16", "mlx.core.bfloat16"}:
        return "bfloat16"
    return None


def _d256_min_n(
    *,
    head_dim: int,
    causal: bool,
    is_m3_plus: bool,
    dtype_key: Optional[str],
    has_custom_table: bool,
) -> Optional[int]:
    """Return D=256 family threshold when a dedicated rule applies.

    D=256 is handled as a separate design family from D=64/128:
    - M3+ f16 and bf16 causal: promote from N>=2048 (1.58-1.68x on M4 Max)
    - M1/M2 f16 causal: promote from N>=2048 (1.09x@2048, 1.22x@4096 post-BK=8)
    - M1/M2 bf16 causal: keep SDPA (0.65-0.88x on M1 Max -- emulation cost)
    - non-causal: defer to global table (already SDPA default)
    """
    if head_dim != 256 or not causal or has_custom_table:
        return None
    if is_m3_plus:
        # M4 Max D=256 causal (B=2 H=8): f16 1.64-1.66x, bf16 1.58-1.68x.
        return 2048
    # M1/M2: only f16 promoted (bf16 D-split emulation is too expensive)
    # BK=8 default (527f9d3): N=1024 0.84x, N=2048 1.09x, N=4096 1.22x (M1 Max)
    if dtype_key == "float16":
        return 2048
    if dtype_key == "bfloat16":
        return 999_999  # 0.65-0.88x on M1 Max
    return None


def _d512_min_n(
    *,
    head_dim: int,
    has_custom_table: bool,
) -> Optional[int]:
    """Return D=512 threshold when dedicated policy should be applied.

    D=512 is intentionally modeled as a separate family from D=64/128/256.
    Current benchmark evidence keeps dense D=512 on conservative SDPA default
    across causal and non-causal modes.
    """
    if head_dim != 512 or has_custom_table:
        return None
    return _D512_CONSERVATIVE_MIN_N


def _forced_d256_auto_decision(head_dim: int, *, backend: str) -> Optional[bool]:
    """Return forced D=256 auto-route override when explicitly requested.

    Env:
      MFA_FORCE_D256_PATH=1|mfa   -> force MFA for D=256 in backend='auto'
      MFA_FORCE_D256_PATH=0|sdpa  -> force SDPA for D=256 in backend='auto'
    """
    if backend != "auto" or head_dim != 256:
        return None
    raw = os.environ.get("MFA_FORCE_D256_PATH")
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw in {"1", "mfa"}:
        return True
    if raw in {"0", "sdpa"}:
        return False
    return None


def _forced_d512_auto_decision(head_dim: int, *, backend: str) -> Optional[bool]:
    """Return forced D=512 auto-route override when explicitly requested.

    Env:
      MFA_FORCE_D512_PATH=1|mfa   -> force MFA for D=512 in backend='auto'
      MFA_FORCE_D512_PATH=0|sdpa  -> force SDPA for D=512 in backend='auto'
    """
    if backend != "auto" or head_dim != 512:
        return None
    raw = os.environ.get("MFA_FORCE_D512_PATH")
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw in {"1", "mfa"}:
        return True
    if raw in {"0", "sdpa"}:
        return False
    return None


def _splitk_env_key(
    head_dim: int,
    causal: bool,
    *,
    has_alibi: bool,
    has_window: bool,
) -> str:
    """Return the split-K calibration env key for this shape family."""
    return (
        f"MFA_SPLITK_MAX_N_D{int(head_dim)}"
        f"_C{1 if causal else 0}"
        f"_A{1 if has_alibi else 0}"
        f"_W{1 if has_window else 0}"
    )


def _load_custom_table() -> Optional[dict[tuple[int, bool], int]]:
    """Load a JSON dispatch table if MLX_MFA_DISPATCH_TABLE is set."""
    global _custom_thresholds, _custom_table_loaded
    if _custom_table_loaded:
        return _custom_thresholds
    _custom_table_loaded = True
    path = os.environ.get("MLX_MFA_DISPATCH_TABLE", "")
    if not path:
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
        table: dict[tuple[int, bool], int] = {}
        for entry in data.get("thresholds", []):
            key = (int(entry["D"]), bool(entry["causal"]))
            table[key] = int(entry["min_N"])
        _custom_thresholds = table
        if _verbose:
            print(f"[MFA dispatch] loaded custom table: {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[MFA dispatch] WARNING: failed to load {path!r}: {exc}")
    return _custom_thresholds


def _v34_backward_carveout(
    head_dim: int,
    seq_len: int,
    causal: bool,
    dtype_key: Optional[str],
) -> bool:
    """**flash_attention()-level** V34-backward eligibility carve-out
    (v2.37.2 → v2.38.x consolidated).

    Called ONLY from `flash_attention()` body when `use_mfa` would
    otherwise be False.  Caller is responsible for the outer guards
    (`softcap == 0`, `alibi_slopes is None`, `not return_lse`, K/V
    same-dtype, `backend == "auto"`) — this function tests only the
    shape + env predicate.

    Returns True if the shape qualifies for the V34-backward NAX-direct
    path (1.81-1.82× faster than SDPA-vjp at D=64 qL=4096-8192 per
    v2.37.2 perf finding).  Without the carve-out + outer guards, the
    public `flash_attention()` autograd path silently falls back to
    SDPA-vjp (v2.37.0/v2.37.1 silent integration bug).

    **History:**
    - v2.37.2: narrow predicate inline in `flash_attention()` body
    - v2.38.x (consolidation): extracted here per Sprint 2 audit
      M5-HIGH-01.  Single source of truth for V34-backward routing.
    - v2.39.2-internal (Sprint A): broadened threshold from
      `qL >= 4096` to `qL >= 2048` after v2.39.1 BK=16 fix made the
      fused kernel reach parity-with-SDPA-vjp at qL=2048 (3-session
      cross-session variance 1.004; see `docs/v6-nax/v39-2-internal-
      decisions.md`).  Below qL=2048, fused regresses vs SDPA-vjp
      (qL=1024: 0.85×, qL=512: 0.50×) — kept out of the carve-out.

    **Currently active predicate:**
    D=64, qL ≥ 2048, non-causal, fp16/bf16, `MFA_ENABLE_V34_BACKWARD=1`.
    Causal kept GATED OUT in production until the K-parallel kernel dV
    residual is resolved (v2.50 Phase 4b-complete Prompt 3 partial state).

    Future broadening (e.g., D=128 if Option γ proves out at v2.40.0-
    internal) extends this function rather than introducing new inline
    overrides.
    """
    # dtype_key values: "float16" / "bfloat16" / None per
    # _dispatch_dtype_key().  NOT "fp16" / "bf16".
    #
    # v2.39.2-internal: qL floor lowered from 4096 to 2048.
    #
    # v2.50 Phase 4b-complete (Prompt 3) — PARTIAL.  Critical compile_v34_
    # backward_pipeline isCausal=false hardcoded bug FIXED (was making
    # Prompt 2 Phase 4b dQ a silent no-op).  dQ kernel now produces
    # correct causal gradients (RMSE 8.7e-6 at qL=2048 D=64 fp16,
    # well within bounds).  The 4 K-parallel kernels (dV split, dK split,
    # dKV legacy fused, dKdV fused) have causal mask blocks compiled
    # in but produce dV with structural ~25× under-counting residual
    # (RMSE 2.7e-3 vs 1e-3 bound).  Gate kept on `causal=True` until
    # the residual is resolved in a focused future session.
    # See `docs/v50/phase-4b-complete-decisions.md`.
    if (
        head_dim == 64
        and seq_len >= 2048
        and not causal
        and dtype_key in ("float16", "bfloat16")
        and os.environ.get("MFA_ENABLE_V34_BACKWARD") == "1"
    ):
        return True
    return False


def should_use_mfa(
    head_dim: int,
    seq_len: int,
    causal: bool,
    is_m3_plus: bool,
    *,
    dtype=None,
    kv_seq_len: Optional[int] = None,
    window_size: Optional[tuple] = None,
    sparse: bool = False,
    backend: str = "auto",
    has_nax: bool = False,
) -> bool:
    """Decide whether the MFA Metal kernel should be used for this config.

    Returns ``True`` only when MFA is *expected* to be faster than SDPA.

    Window and sparse attention always route to MFA because tile-skipping
    provides a guaranteed speedup over dense SDPA regardless of shape.

    Parameters
    ----------
    head_dim : int
        Head dimension D.
    seq_len : int
        Query sequence length N.
    causal : bool
        Whether causal masking is applied.
    is_m3_plus : bool
        True on M3/M4/M5+ Apple Silicon (better block configs available).
    dtype : optional
        Input dtype (``mx.float16`` / ``mx.bfloat16``) when available. Used
        for D=256 narrow routing, where f16 and bf16 regimes differ.
    kv_seq_len : int, optional
        KV sequence length S (when different from N, i.e. cross-attention).
        When None, assumed equal to seq_len (self-attention).
    window_size : tuple, optional
        ``(left, right)`` sliding-window radii.  Non-negative left enables
        the window path, which always benefits from MFA tile-skipping.
    sparse : bool
        True for block-sparse attention (always benefits from tile-skipping).
    backend : str
        ``"auto"`` (shape-aware), ``"mfa"`` (force MFA), ``"sdpa"`` (force SDPA).
    """
    if backend == "mfa":
        if _verbose:
            print(f"[MFA dispatch] backend=mfa forced -> MFA")
        return True
    if backend == "sdpa":
        if _verbose:
            print(f"[MFA dispatch] backend=sdpa forced -> SDPA")
        return False

    # v2.32.0 — explicit SDPA-routing overrides (highest priority after backend=).
    force_sdpa = os.environ.get("MFA_FORCE_SDPA_ROUTE")
    if force_sdpa == "1":
        if _verbose:
            print(f"[MFA dispatch] MFA_FORCE_SDPA_ROUTE=1 -> SDPA")
        return False
    disable_sdpa = os.environ.get("MFA_DISABLE_SDPA_ROUTE")
    if disable_sdpa == "1":
        # Disable the v2.32.0 strategic SDPA routing; fall through to
        # M3+/legacy thresholds. Mainly for benchmarking / regression checks.
        has_nax = False
        if _verbose:
            print(f"[MFA dispatch] MFA_DISABLE_SDPA_ROUTE=1 -> falling through to legacy thresholds")

    # Sliding-window and block-sparse ALWAYS use MFA: tile-skip guarantees speedup.
    # window_size=(left, right): MFA when either dimension is set (>=0).
    # Right-only window (left=-1, right>=0) still requires the MFA kernel.
    if window_size is not None:
        left  = window_size[0] if len(window_size) > 0 else -1
        right = window_size[1] if len(window_size) > 1 else -1
        if left >= 0 or right >= 0:
            if _verbose:
                print(f"[MFA dispatch] window={window_size} -> MFA (windowed)")
            return True
    if sparse:
        if _verbose:
            print(f"[MFA dispatch] sparse -> MFA (tile-skip)")
        return True

    forced_d256 = _forced_d256_auto_decision(head_dim, backend=backend)
    if forced_d256 is not None:
        if _verbose:
            print(
                f"[MFA dispatch] D=256 force override MFA_FORCE_D256_PATH "
                f"-> {'MFA' if forced_d256 else 'SDPA'}"
            )
        return forced_d256

    forced_d512 = _forced_d512_auto_decision(head_dim, backend=backend)
    if forced_d512 is not None:
        if _verbose:
            print(
                f"[MFA dispatch] D=512 force override MFA_FORCE_D512_PATH "
                f"-> {'MFA' if forced_d512 else 'SDPA'}"
            )
        return forced_d512

    # Cross-attention routing (benchmarked M1 Max, 2026-04-06, DiT/UNet audit).
    #
    # When N_kv is much smaller than N_q, MFA's per-tile overhead dominates
    # because there are too few K-tiles to amortize fixed costs:
    #   N_kv≤512, N_q≥8192: SDPA wins 0.70-0.82x across D=64/128
    #   N_kv≤77,  N_q=4096, D=128: MFA wins 1.60x (small N_q OK)
    #   N_kv≤77,  N_q=4096, D=64:  SDPA wins 0.70x
    #
    # Conversely, when N_kv >> N_q (e.g. LTX-2 audio→video), MFA wins big
    # (8.59x) because flash attention processes Q rows in tiles while SDPA
    # materializes the full N_q × N_kv attention matrix.
    _kv_len = kv_seq_len if kv_seq_len is not None else seq_len
    if _kv_len != seq_len:
        # Cross-attention: small N_kv with large N_q → SDPA
        if _kv_len <= 512 and seq_len > 8192:
            if _verbose:
                print(
                    f"[MFA dispatch] cross-attn small KV: N_q={seq_len} "
                    f"N_kv={_kv_len} -> SDPA (few K-tiles, tile overhead dominates)"
                )
            return False
        # Cross-attention: large N_kv with small N_q → MFA wins big
        # (flash attention iterates K-tiles per Q-tile; fewer Q-tiles = less work)
        if _kv_len >= 4096 and seq_len <= 4096:
            # v2.32.0 — M5+ NAX qualification: pure-decode patterns (qL ≤ 16,
            # e.g. llama generation qL=1) win on SDPA's sdpa_vector path
            # rather than MFA's flash-decode kernel. Sprint A measured 1.9-2.6×
            # SDPA wins on llama-decode-8k/32k. Cross-attn (qL > 16, e.g.
            # ltx2-cross qL=2048 kL=14000) stays on MFA where it wins ~11%.
            if has_nax and seq_len <= 16:
                if _verbose:
                    print(
                        f"[MFA dispatch] M5+ NAX decode pattern: N_q={seq_len} "
                        f"N_kv={_kv_len} -> falling through to NAX SDPA route"
                    )
                # Fall through to the has_nax block below.
            else:
                if _verbose:
                    print(
                        f"[MFA dispatch] cross-attn large KV: N_q={seq_len} "
                        f"N_kv={_kv_len} -> MFA (few Q-tiles, flash attention wins)"
                    )
                return True

    # v2.32.0 — M5+ NAX SDPA routing.
    # Apple's `steel_attention_nax.h` is the optimal forward path on canonical
    # shapes (D∈{64,128}, no exotic features). Route to SDPA on those.
    # Carve-outs (Sprint A empirical findings) keep mlx-mfa for specific
    # shape-corners where mlx-mfa wins despite the canonical match.
    #
    # Decode patterns (qL ≤ 8, kL >> qL) are caught by the cross-attn rule
    # above (kL ≥ 4096 and qL ≤ 4096 → MFA). Short symmetric small-N (e.g.
    # N=8, S=64) falls through to standard thresholds — for D=64 non-causal
    # those say SDPA on M3+/M5+, which is the right answer (small SDPA call
    # is fine, no need to invoke MFA flash-decode for N=8 self-attn).
    if has_nax:
        # Canonical D=64 / D=128 (any N >= ~16 not handled by cross-attn rule):
        # v2.32.0 default = SDPA on M5+ NAX (Apple's steel_attention_nax.h is
        # optimal there).  The `_should_use_mfa_m5_nax_carveout()` placeholder
        # was deleted in v2.38.0 (dormant since v2.32.0; no Sprint A.6
        # carve-outs ever materialized).  If a future Sprint A.6 surfaces
        # empirically-validated MFA-winning shapes on M5+ NAX canonical D,
        # re-introduce a named function (not inline conditionals) and
        # call it here.
        if head_dim in (64, 128):
            if _verbose:
                print(f"[MFA dispatch] M5+ NAX canonical D={head_dim} N={seq_len} causal={causal} -> SDPA (Apple's steel_attention_nax.h is optimal)")
            return False
        # D=256/512 not covered by SDPA NAX — fall through to standard table.

    # Dense attention: check empirical crossover threshold.
    custom = _load_custom_table()
    if custom is not None:
        thresholds = custom
    elif has_nax:
        thresholds = _M5_NAX_THRESHOLDS
    elif is_m3_plus:
        thresholds = _M3_THRESHOLDS
    else:
        thresholds = _DEFAULT_THRESHOLDS

    dtype_key = _dispatch_dtype_key(dtype)

    d512_min_n = _d512_min_n(
        head_dim=head_dim,
        has_custom_table=(custom is not None),
    )

    d256_min_n = _d256_min_n(
        head_dim=head_dim,
        causal=causal,
        is_m3_plus=is_m3_plus,
        dtype_key=dtype_key,
        has_custom_table=(custom is not None),
    )
    if d512_min_n is not None:
        min_n = d512_min_n
    elif d256_min_n is not None:
        min_n = d256_min_n
    else:
        min_n = thresholds.get((head_dim, causal), 999_999)

    use_mfa = seq_len >= min_n

    if _verbose:
        src = "custom" if custom else ("M3+" if is_m3_plus else "M1")
        print(
            f"[MFA dispatch] D={head_dim} N={seq_len} causal={causal} "
            f"m3+={is_m3_plus} dtype={dtype_key or 'unknown'} "
            f"threshold={min_n} ({src}) "
            f"-> {'MFA' if use_mfa else 'SDPA'}"
        )
    return use_mfa


def should_use_splitk(
    head_dim: int,
    seq_len: int,
    causal: bool,
    *,
    has_alibi: bool = False,
    has_window: bool = False,
) -> bool:
    """Return whether split-K should be enabled for this shape family.

    Priority:
      1) ``MFA_FORCE_SPLITK=0|1`` hard override.
      2) calibrated max-N env threshold (if present).
      3) fallback to C++ occupancy heuristic (return ``True`` here).
    """
    force = os.environ.get("MFA_FORCE_SPLITK")
    if force == "0":
        return False
    if force == "1":
        return True

    key = _splitk_env_key(
        head_dim,
        causal,
        has_alibi=has_alibi,
        has_window=has_window,
    )
    raw = os.environ.get(key)
    if raw is None or raw == "":
        # No calibration entry -> let C++ occupancy heuristic decide.
        return True
    try:
        max_n = int(raw)
    except ValueError:
        return True
    if max_n < 0:
        return True
    return seq_len <= max_n


def _native_bwd_dtype_key(dtype) -> Optional[str]:
    """Normalize dtype objects/strings for native-bwd policy lookup."""
    if dtype == mx.float16 or dtype == "float16":
        return "float16"
    if dtype == mx.bfloat16 or dtype == "bfloat16":
        return "bfloat16"
    return None


def should_use_native_backward(
    head_dim: int,
    seq_len: int,
    causal: bool,
    *,
    dtype,
) -> bool:
    """Return whether native STEEL backward should be used for this shape.

    Priority:
      1) ``MFA_FORCE_NATIVE_BWD=0|1`` hard override (for supported shapes).
      2) benchmark-backed narrow policy table.

    Safety constraints (always enforced):
      - causal only
      - D in {64, 128}
      - dtype in {float16, bfloat16}
    """
    dtype_key = _native_bwd_dtype_key(dtype)
    supported = causal and (head_dim in (64, 128)) and (dtype_key is not None)

    force = os.environ.get("MFA_FORCE_NATIVE_BWD")
    if force == "0":
        return False
    if force == "1":
        return supported

    if not supported:
        return False

    min_n = _NATIVE_BWD_MIN_N.get((head_dim, dtype_key), 999_999)
    return seq_len >= min_n


def _window_enabled(window_size: Optional[tuple]) -> bool:
    """Return True when either window side is enabled."""
    if window_size is None:
        return False
    left = window_size[0] if len(window_size) > 0 else -1
    right = window_size[1] if len(window_size) > 1 else -1
    return left >= 0 or right >= 0


def should_use_sage_decode(
    head_dim: int,
    n_q: int,
    cache_len: int,
    causal: bool,
    *,
    has_quantized_kv: bool,
    window_size: Optional[tuple] = None,
    gqa_factor: int = 1,
    dtype=None,
) -> bool:
    """Return whether decode auto mode should route to Sage.

    Priority:
      1) ``MFA_FORCE_SAGE_DECODE=0|1`` hard override.
      2) benchmark-backed narrow policy.

    Safety constraints (always enforced):
      - decode shape only (``n_q <= 4``)
      - causal only
      - ``head_dim in {64, 128}``
      - quantized KV cache available
    """
    decode_shape = n_q <= 4
    supported = (
        has_quantized_kv
        and causal
        and decode_shape
        and (head_dim in (64, 128))
    )

    force = os.environ.get("MFA_FORCE_SAGE_DECODE")
    if force == "0":
        return False
    if force == "1":
        return supported

    if not supported:
        return False

    # Narrow promotion only: production-like GQA decode windows.
    if head_dim != 128:
        return False
    if not _window_enabled(window_size):
        return False
    if gqa_factor != 2:
        return False
    if cache_len != 4096:
        return False

    dtype_key = _dispatch_dtype_key(dtype)
    if dtype_key == "float16":
        return n_q == 4
    if dtype_key == "bfloat16":
        return n_q == 1
    return False


def calibrate_dispatch(
    head_dims: Optional[list] = None,
    save_path: Optional[str] = None,
    *,
    warmup: int = 5,
    n_iters: int = 20,
    calibrate_kernel_configs: bool = True,
    calibrate_splitk: bool = True,
) -> dict[tuple[int, bool], int]:
    """Run micro-benchmarks to find optimal MFA/SDPA crossover points.

    Creates a device-specific dispatch table and optionally saves it to
    ``~/.mlx_mfa/dispatch_table.json``.  Pass that path as
    ``MLX_MFA_DISPATCH_TABLE`` to activate the calibrated table.

    Parameters
    ----------
    head_dims : list of int, optional
        Head dimensions to calibrate.  Defaults to ``[64, 128, 256]``.
    save_path : str, optional
        Where to save the JSON table.  Defaults to
        ``~/.mlx_mfa/dispatch_table.json``.
    warmup : int
        Warmup iterations per config (default 5).
    n_iters : int
        Timed iterations per config (default 20).
    calibrate_kernel_configs : bool
        When True (default), also benchmark D=128 BK=32 vs BK=64 and save
        the optimal BK to ``kernel_configs.d128_optimal_bk`` in the JSON.
        BK=64 is chosen only if it wins at BOTH N=4096 and N=8192 (i.e.,
        BK=64 time < 0.95 × BK=32 time at both points).
    calibrate_splitk : bool
        When True (default), benchmark V2 split-K on/off for representative
        production families (D=64/128, causal dense, causal+ALiBi,
        causal+window) and save per-family ``max_N`` crossover entries.

    Returns
    -------
    dict
        Mapping ``(D, causal) -> min_N`` where MFA first beats SDPA.
    """
    import math
    import time

    import numpy as np
    import mlx.core as mx
    from mlx_mfa import flash_attention
    from mlx_mfa.attention import _fallback_sdpa

    _materialize = mx.eval

    if head_dims is None:
        head_dims = [64, 128, 256]

    N_CANDIDATES = [512, 1024, 2048, 4096, 8192, 16384]
    results: dict[tuple[int, bool], int] = {}
    thresholds_list = []

    print("Calibrating MFA dispatch thresholds...")
    for D in head_dims:
        for causal in [True, False]:
            min_N = 999_999
            for N in N_CANDIDATES:
                scale = 1.0 / math.sqrt(D)
                q = mx.zeros([1, 8, N, D], dtype=mx.float16)
                k = mx.zeros([1, 8, N, D], dtype=mx.float16)
                v = mx.zeros([1, 8, N, D], dtype=mx.float16)
                _materialize(q, k, v)

                def _run(fn):
                    for _ in range(warmup):
                        _materialize(fn())
                    mx.synchronize()
                    ts = []
                    for _ in range(n_iters):
                        t0 = time.perf_counter()
                        _materialize(fn())
                        mx.synchronize()
                        ts.append((time.perf_counter() - t0) * 1000.0)
                    return float(np.median(ts))

                mfa_ms  = _run(lambda: flash_attention(
                    q, k, v, scale=scale, causal=causal, backend="mfa"))
                sdpa_ms = _run(lambda: _fallback_sdpa(q, k, v, scale, causal))
                ratio = sdpa_ms / mfa_ms if mfa_ms > 0 else 0.0
                c_str = "causal" if causal else "non-causal"
                print(f"  D={D} N={N} {c_str}: {ratio:.2f}x "
                      f"({'MFA' if ratio >= 1.0 else 'SDPA'})")
                if ratio >= 1.0 and min_N == 999_999:
                    min_N = N

            results[(D, causal)] = min_N
            thresholds_list.append({"D": D, "causal": causal, "min_N": min_N})
            c_str = "causal" if causal else "non-causal"
            print(f"  => D={D} {c_str}: min_N={min_N}")

    # ── Kernel config calibration (BK=32 vs BK=64 for D=128) ───────────────
    kernel_configs: dict = {}
    splitk_thresholds: list[dict] = []
    if calibrate_kernel_configs:
        print("\nCalibrating D=128 BK selection (BK=32 vs BK=64)...")
        bk_results: dict[int, dict[int, float]] = {32: {}, 64: {}}
        for bk in (32, 64):
            for N in (4096, 8192):
                scale = 1.0 / math.sqrt(128)
                q = mx.zeros([1, 8, N, 128], dtype=mx.float16)
                k = mx.zeros([1, 8, N, 128], dtype=mx.float16)
                v = mx.zeros([1, 8, N, 128], dtype=mx.float16)
                _materialize(q, k, v)

                def _run_bk(fn):  # noqa: E306
                    for _ in range(warmup):
                        _materialize(fn())
                    mx.synchronize()
                    ts = []
                    for _ in range(n_iters):
                        t0 = time.perf_counter()
                        _materialize(fn())
                        mx.synchronize()
                        ts.append((time.perf_counter() - t0) * 1000.0)
                    return float(np.median(ts))

                prev = os.environ.get("MFA_V2_FORCE_BK")
                try:
                    os.environ["MFA_V2_FORCE_BK"] = str(bk)
                    _invalidate_cached_env()
                    ms = _run_bk(lambda: flash_attention(  # noqa: B023
                        q, k, v, scale=scale, causal=True, backend="mfa"))
                finally:
                    if prev is None:
                        os.environ.pop("MFA_V2_FORCE_BK", None)
                    else:
                        os.environ["MFA_V2_FORCE_BK"] = prev
                    _invalidate_cached_env()
                bk_results[bk][N] = ms
                print(f"  D=128 BK={bk} N={N}: {ms:.2f} ms")

        # BK=64 wins only if faster at BOTH N=4096 AND N=8192
        wins_4096 = bk_results[64][4096] < 0.95 * bk_results[32][4096]
        wins_8192 = bk_results[64][8192] < 0.95 * bk_results[32][8192]
        from mlx_mfa import get_device_info as _gdi
        _dev_info = _gdi()
        _hw_m3_plus = bool(_dev_info.get("is_m3_plus", False))
        optimal_bk = 64 if (_hw_m3_plus and wins_4096 and wins_8192) else 32
        kernel_configs["d128_optimal_bk"] = optimal_bk
        print(f"  => D=128 optimal BK={optimal_bk} "
              f"(BK=64 wins N=4096: {wins_4096}, N=8192: {wins_8192})")

    # ── Split-K calibration (on/off crossover) ─────────────────────────────
    if calibrate_splitk:
        print("\nCalibrating split-K on/off crossover (D=64/128, causal families)...")
        splitk_profiles = [
            {"causal": True, "has_alibi": False, "window_size": None},
            {"causal": True, "has_alibi": True, "window_size": None},
            {"causal": True, "has_alibi": False, "window_size": (256, 0)},
            {"causal": True, "has_alibi": False, "window_size": (512, 0)},
        ]
        splitk_candidates = [256, 512, 1024, 2048, 4096, 8192]

        for D in (64, 128):
            for profile in splitk_profiles:
                causal = profile["causal"]
                has_alibi = profile["has_alibi"]
                window_size = profile["window_size"]
                has_window = window_size is not None
                max_n = -1

                for N in splitk_candidates:
                    scale = 1.0 / math.sqrt(D)
                    q = mx.zeros([1, 1, N, D], dtype=mx.float16)
                    k = mx.zeros([1, 1, N, D], dtype=mx.float16)
                    v = mx.zeros([1, 1, N, D], dtype=mx.float16)
                    slopes = mx.array([-0.1], dtype=mx.float32)
                    _materialize(q, k, v, slopes)

                    def _run_mode(force_splitk: str) -> float:
                        prev = os.environ.get("MFA_FORCE_SPLITK")
                        try:
                            os.environ["MFA_FORCE_SPLITK"] = force_splitk
                            for _ in range(warmup):
                                _materialize(flash_attention(
                                    q, k, v,
                                    scale=scale,
                                    causal=causal,
                                    backend="mfa",
                                    alibi_slopes=(slopes if has_alibi else None),
                                    window_size=window_size,
                                ))
                            mx.synchronize()
                            ts = []
                            for _ in range(n_iters):
                                t0 = time.perf_counter()
                                _materialize(flash_attention(
                                    q, k, v,
                                    scale=scale,
                                    causal=causal,
                                    backend="mfa",
                                    alibi_slopes=(slopes if has_alibi else None),
                                    window_size=window_size,
                                ))
                                mx.synchronize()
                                ts.append((time.perf_counter() - t0) * 1000.0)
                            return float(np.median(ts))
                        finally:
                            if prev is None:
                                os.environ.pop("MFA_FORCE_SPLITK", None)
                            else:
                                os.environ["MFA_FORCE_SPLITK"] = prev

                    splitk_ms = _run_mode("1")
                    nosplit_ms = _run_mode("0")
                    speedup = nosplit_ms / splitk_ms if splitk_ms > 0 else 0.0

                    prof = (
                        "dense"
                        if not has_alibi and not has_window
                        else ("alibi" if has_alibi else f"window={window_size[0]}")
                    )
                    print(
                        f"  D={D} N={N} {prof}: split-K {splitk_ms:.2f} ms, "
                        f"no-split {nosplit_ms:.2f} ms, speedup {speedup:.2f}x"
                    )
                    # Require a small margin to avoid noise-driven toggles.
                    if speedup >= 1.02:
                        max_n = N

                entry = {
                    "D": D,
                    "causal": causal,
                    "has_alibi": has_alibi,
                    "has_window": has_window,
                    "max_N": max_n,
                }
                splitk_thresholds.append(entry)
                print(
                    f"  => split-K max_N D={D} C={int(causal)} A={int(has_alibi)} "
                    f"W={int(has_window)}: {max_n}"
                )

    # ── Save ─────────────────────────────────────────────────────────────────
    if save_path is None:
        save_path = os.path.expanduser("~/.mlx_mfa/dispatch_table.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload: dict = {
        "generated": "mlx_mfa.dispatch_policy.calibrate_dispatch",
        "thresholds": thresholds_list,
    }
    if kernel_configs:
        payload["kernel_configs"] = kernel_configs
    if splitk_thresholds:
        payload["splitk_thresholds"] = splitk_thresholds
    with open(save_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved dispatch table -> {save_path}")
    return results


# ---------------------------------------------------------------------------
# Auto-load calibrated kernel config at import time
# ---------------------------------------------------------------------------

def _load_calibrated_kernel_config() -> None:
    """Read kernel_configs from dispatch_table.json and apply via os.environ.

    Uses os.environ.setdefault so an explicit MFA_V2_FORCE_BK set by the
    user before import still takes precedence.
    Called once at mlx_mfa import time.
    """
    table_path = os.environ.get(
        "MLX_MFA_DISPATCH_TABLE",
        os.path.expanduser("~/.mlx_mfa/dispatch_table.json"),
    )
    if not os.path.exists(table_path):
        return
    try:
        with open(table_path) as fh:
            data = json.load(fh)
        bk = data.get("kernel_configs", {}).get("d128_optimal_bk")
        if bk in (32, 64):
            if bk == 64:
                from mlx_mfa import get_device_info as _gdi
                if not bool(_gdi().get("is_m3_plus", False)):
                    bk = 32  # downgrade: M1/M2 cannot use BK=64 safely
            os.environ.setdefault("MFA_V2_FORCE_BK", str(bk))
            _invalidate_cached_env()
            if _verbose:
                print(f"[MFA dispatch] loaded calibrated BK={bk} from {table_path}")

        for entry in data.get("splitk_thresholds", []):
            d = int(entry.get("D"))
            c = bool(entry.get("causal"))
            a = bool(entry.get("has_alibi"))
            w = bool(entry.get("has_window"))
            max_n = int(entry.get("max_N"))
            env_key = _splitk_env_key(d, c, has_alibi=a, has_window=w)
            os.environ.setdefault(env_key, str(max_n))
            if _verbose:
                print(f"[MFA dispatch] loaded {env_key}={max_n} from {table_path}")
    except Exception:  # noqa: BLE001
        pass  # silently skip — calibration is advisory
