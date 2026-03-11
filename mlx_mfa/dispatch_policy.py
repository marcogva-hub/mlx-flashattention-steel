"""Shape-aware dispatch policy for MFA vs SDPA backend selection.

The dispatch decision is based on empirical crossover points derived from
bench_dispatch_matrix.py on Apple M1 Max (M1 Max, f16, B=1 H=8).

Conservative principle: only activate MFA when it is EXPECTED to be faster
than MLX SDPA.  It is always better to route to SDPA (and be ~1.0x) than to
route to MFA incorrectly (and be 0.5x).

Baseline crossover data (dispatch_matrix.json, 2026-03-10):
  - D=64  causal  N>=2048: MFA wins (1.06x at 2048, 1.41x at 8192)
  - D=128 causal  N>=8192: MFA wins (1.25x at 8192)
  - All non-causal: MFA never wins (best 0.92x at D=64 N=8192)
  - D=256/512: MFA never wins causal or non-causal

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

# ---------------------------------------------------------------------------
# Default thresholds: minimum N to activate MFA for (D, causal) pair.
# Derived from M1 Max dispatch matrix baseline.  999_999 effectively disables.
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[tuple[int, bool], int] = {
    # D=64 causal: V2 kernel (BK=64) raises crossover. Measured (M1 Max, 12 trials):
    #   N=512→1.18x, N=1024→1.14x, N=2048→1.25x, N=4096→1.86x, N=16384→2.20x.
    # Threshold N=1024 (conservatively above break-even, stable across runs).
    (64,  True):  1024,
    # D=64 non-causal: 0.98x at N=16384, 0.97x at N=32768 — MFA never wins.
    (64,  False): 999_999,
    # D=128 causal: V2 delivers 1.33x at N=2048, 1.60x at N=4096, 1.76x at N=16384.
    # Old V1 threshold was N=8192 (too conservative). New threshold: N=2048.
    (128, True):  2048,
    # D=128 non-causal: 0.90x at N=16384, 0.88x at N=32768 — disable.
    (128, False): 999_999,
    # D=256 causal: V1 kernel wins at large N (CP7 benchmark — M1 Max, f16):
    #   N=8192→0.80x, N=16384→1.03x, N=32768→1.28x.
    # Threshold N=16384 (stable 1.03x+; N<16384 routes to SDPA).
    (256, True):  16384,
    (256, False): 999_999,
    # D=512: best 0.49x at N=16384 — MFA never wins.
    (512, True):  999_999,
    (512, False): 999_999,
}

# M3+ thresholds: D=128 BK=64 (doubled vs M1 BK=32) → larger per-tile speedup.
# V2 BK=64 on M3+ provides ~2× K-tile size → can activate at lower N than M1.
_M3_THRESHOLDS: dict[tuple[int, bool], int] = {
    # D=64 causal: BK=64 same on all gens; lower threshold conservatively to 512.
    (64,  True):  512,
    (64,  False): 999_999,
    # D=128 causal: M3+ BK=64 (~2× tile vs M1 BK=32) → threshold N=1024.
    # On M1/M2 threshold is N=2048; M3+ wins earlier due to larger tile.
    (128, True):  1024,
    (128, False): 999_999,
    # D=256 causal: M3+ likely wins at smaller N due to register file headroom.
    # Conservative estimate N=8192 (not measured; override with MFA_DISPATCH_TABLE).
    (256, True):  8192,
    (256, False): 999_999,
    (512, True):  999_999,
    (512, False): 999_999,
}

_verbose: bool = os.environ.get("MLX_MFA_VERBOSE_DISPATCH", "0") == "1"

# Cached custom dispatch table (loaded once from MLX_MFA_DISPATCH_TABLE env var).
_custom_thresholds: Optional[dict[tuple[int, bool], int]] = None
_custom_table_loaded = False


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


def should_use_mfa(
    head_dim: int,
    seq_len: int,
    causal: bool,
    is_m3_plus: bool,
    *,
    window_size: Optional[tuple] = None,
    sparse: bool = False,
    backend: str = "auto",
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

    # Dense attention: check empirical crossover threshold.
    custom = _load_custom_table()
    if custom is not None:
        thresholds = custom
    elif is_m3_plus:
        thresholds = _M3_THRESHOLDS
    else:
        thresholds = _DEFAULT_THRESHOLDS

    min_n = thresholds.get((head_dim, causal), 999_999)
    use_mfa = seq_len >= min_n

    if _verbose:
        src = "custom" if custom else ("M3+" if is_m3_plus else "M1")
        print(
            f"[MFA dispatch] D={head_dim} N={seq_len} causal={causal} "
            f"m3+={is_m3_plus} threshold={min_n} ({src}) "
            f"-> {'MFA' if use_mfa else 'SDPA'}"
        )
    return use_mfa


def calibrate_dispatch(
    head_dims: Optional[list] = None,
    save_path: Optional[str] = None,
    *,
    warmup: int = 5,
    n_iters: int = 20,
    calibrate_kernel_configs: bool = True,
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
                    ms = _run_bk(lambda: flash_attention(  # noqa: B023
                        q, k, v, scale=scale, causal=True, backend="mfa"))
                finally:
                    if prev is None:
                        os.environ.pop("MFA_V2_FORCE_BK", None)
                    else:
                        os.environ["MFA_V2_FORCE_BK"] = prev
                bk_results[bk][N] = ms
                print(f"  D=128 BK={bk} N={N}: {ms:.2f} ms")

        # BK=64 wins only if faster at BOTH N=4096 AND N=8192
        wins_4096 = bk_results[64][4096] < 0.95 * bk_results[32][4096]
        wins_8192 = bk_results[64][8192] < 0.95 * bk_results[32][8192]
        optimal_bk = 64 if (wins_4096 and wins_8192) else 32
        kernel_configs["d128_optimal_bk"] = optimal_bk
        print(f"  => D=128 optimal BK={optimal_bk} "
              f"(BK=64 wins N=4096: {wins_4096}, N=8192: {wins_8192})")

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
            os.environ.setdefault("MFA_V2_FORCE_BK", str(bk))
            if _verbose:
                print(f"[MFA dispatch] loaded calibrated BK={bk} from {table_path}")
    except Exception:  # noqa: BLE001
        pass  # silently skip — calibration is advisory
