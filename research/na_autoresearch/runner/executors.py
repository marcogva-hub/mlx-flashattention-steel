"""Cell executors for the NA-autoresearch funnel.

A *cell* is one (kernel, shape, tile/knob config, mode) point. An executor runs
ONE cell and returns a result dict. Stage-1 screening executors are in-process,
1 order, few reps (rank-only). Correctness (fp32/SDPA oracle) and engagement
(byteΔ-vs-SDPA + no-raise) are ALWAYS checked — a fast cell is not a green cell.

Design facts established at source + empirically (Phase 0):
  - Dense NAX tiles MFA_V6_NAX_{BQ,BK,WM} (+ D_SUBTILE) are read via getenv per
    dispatch (csrc/mfa_v6_nax_primitive.cpp:354-356) → env-sweepable in-process.
  - Tiles are a PURE perf axis: cos == 1.0 across configs. So the screening oracle
    is cos-vs-SDPA >= 0.999, and the tile-was-used proof is (a) it did not RAISE
    (invalid BQ%(WM*16)!=0 / BK%32!=0 fail loud, Rule 8) and (b) byteΔ-vs-SDPA > 0
    (NAX engaged, not the SDPA fallback).
  - Screening cell wall-clock ~0.08 s (cached) .. ~1.2 s (fresh JIT + slow tile).

No fabricated numbers: every field here is a measured value or a config echo.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import numpy as np
import mlx.core as mx

# adaptive-timer thresholds (RULE 3): per-iter ms above which reps shrink (rank-only)
SLOW_ITER_MS = float(os.environ.get("NAAR_SLOW_ITER_MS", "25"))   # >this → 3 reps, coarse
MED_ITER_MS = float(os.environ.get("NAAR_MED_ITER_MS", "5"))      # >this → 6 reps


# ----- timing -----------------------------------------------------------------

def _med(fn: Callable[[], Any], warmup: int, iters: int) -> float:
    """Median ms of fn(); warmup not timed. Includes any first-call JIT on warmup[0]."""
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def _med_adaptive(fn: Callable[[], Any]):
    """Cost-bounded screening timer. One JIT/device-warm eval, estimate per-iter, then scale
    reps: a tile at ~2.7 s/iter (D128 N32768) is OBVIOUSLY bad — 3 reps rank it, 12 waste ~30 s.
    Returns (median_ms, reps, coarse). This is the 'no silent cap' guard — `coarse` is recorded.
    Thresholds are module-top (RULE 3)."""
    mx.eval(fn())            # includes first-call JIT for a new tile hash
    mx.synchronize()
    t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
    est = (time.perf_counter() - t0) * 1000.0
    if est > SLOW_ITER_MS:       extra_warm, reps, coarse = 0, 3, True
    elif est > MED_ITER_MS:      extra_warm, reps, coarse = 1, 6, False
    else:                        extra_warm, reps, coarse = 3, 12, False
    for _ in range(extra_warm):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); mx.eval(fn()); mx.synchronize()
        ts.append((time.perf_counter() - t) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], reps, coarse


def _cos(a: mx.array, b: mx.array) -> float:
    af = np.asarray(a.astype(mx.float32)).ravel().astype(np.float64)
    bf = np.asarray(b.astype(mx.float32)).ravel().astype(np.float64)
    d = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / d) if d else 0.0


def _maxabs(a: mx.array, b: mx.array) -> float:
    return float(np.abs(np.asarray(a.astype(mx.float32)) - np.asarray(b.astype(mx.float32))).max())


_DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}


def _mk_qkv(shape: dict, seed: int = 0):
    mx.random.seed(seed)
    B, H, N, D = shape["B"], shape["H"], shape["N"], shape["D"]
    dt = _DTYPES[shape.get("dtype", "float16")]
    f = lambda: (mx.random.normal((B, H, N, D)) * 0.1).astype(dt)
    q, k, v = f(), f(), f()
    mx.eval(q, k, v)
    return q, k, v, D


# ----- dense forward screening ------------------------------------------------

def screen_dense(cell: dict, warmup: int = 4, iters: int = 12, adaptive: bool = True) -> dict:
    """Stage-1 screening of one dense-forward tile config.

    cell = {kernel:'dense', shape:{B,H,N,D,dtype,causal}, tiles:{BQ,BK,WM[,BD]}}
    Returns a result dict with median_ms, cos_vs_sdpa, byteΔ_vs_sdpa (engagement),
    correct/engaged booleans, and the exact tile echo. Raises are caught and
    reported as engaged=False + error (loud-fail = invalid tile, still legible).
    """
    import mlx_mfa
    from mlx_mfa import _ext
    os.environ["MFA_SILENCE_NAX_WARNING"] = "1"
    shape = cell["shape"]
    tiles = cell.get("tiles", {}) or {}
    # orthogonal correctness-neutral perf knobs (RELAXED_PRECISION, UNROLL_MODE, MAX_THREADS, …),
    # each env-swept per dispatch (mfa_v6_nax_primitive.cpp:467/485/947). {} = kernel defaults.
    knobs = cell.get("knobs", {}) or {}
    prev = {}
    for var, key in (("MFA_V6_NAX_BQ", "BQ"), ("MFA_V6_NAX_BK", "BK"),
                     ("MFA_V6_NAX_WM", "WM"), ("MFA_V6_NAX_D_SUBTILE", "BD")):
        prev[var] = os.environ.get(var)
        os.environ.pop(var, None)
        if key in tiles and tiles[key] is not None:
            os.environ[var] = str(tiles[key])
    for var, val in knobs.items():
        prev[var] = os.environ.get(var)
        os.environ.pop(var, None)
        if val is not None:
            os.environ[var] = str(val)
    t_cell0 = time.perf_counter()
    res: dict[str, Any] = {"cell_id": cell["cell_id"], "kernel": "dense", "mode": cell.get("mode", "screening"),
                           "shape": shape, "tiles": tiles, "knobs": knobs}
    if os.environ.get("NAAR_DAY_AMBIENT"):   # day-mode: intra-batch rankings valid, absolutes re-judged in Stage-2
        res["day_ambient"] = True
    try:
        q, k, v, D = _mk_qkv(shape)
        causal = bool(shape.get("causal", False))
        scale = 1.0 / (D ** 0.5)
        # FORCE the NAX kernel so tile knobs engage at ANY D (public flash_attention
        # routes D=64 dense -> SDPA, which would leave tiles inert). Autoresearch tunes
        # the KERNEL, so we run the kernel directly and prove engagement (byteΔ vs SDPA > 0).
        run = lambda: _ext.v6_nax_forward(q, k, v, causal, True, scale)[0]
        sdpa_run = lambda: mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask="causal" if causal else None)
        if adaptive:                                    # screening: cost-bounded reps
            med, reps, coarse = _med_adaptive(run)
            sdpa_ms, _, _ = _med_adaptive(sdpa_run)
        else:                                           # floors/sentinel: precise
            med, reps, coarse = _med(run, warmup, iters), iters, False
            sdpa_ms = _med(sdpa_run, warmup, iters)
        o = run(); mx.eval(o)
        sdpa = sdpa_run(); mx.eval(sdpa)
        res.update({
            "median_ms": round(med, 5),                 # NAX kernel @ this tile
            "sdpa_ms": round(sdpa_ms, 5),               # production competitor at this shape
            "ratio_vs_sdpa": round(sdpa_ms / med, 4),   # >1 => NAX-tile beats SDPA
            "reps": reps, "coarse": coarse,             # coarse=True => rank-only (slow tile)
            "cos_vs_sdpa": round(_cos(o, sdpa), 6),
            "byteDelta_vs_sdpa": _maxabs(o, sdpa),
            "engaged": _maxabs(o, sdpa) > 0.0,          # NAX ran, not the SDPA fallback
            "correct": _cos(o, sdpa) >= 0.999,
            "finite": bool(np.isfinite(np.asarray(o.astype(mx.float32))).all()),
            "error": None,
        })
    except Exception as e:                                # loud-fail (invalid tile) is data, not a crash
        res.update({"median_ms": None, "engaged": False, "correct": False,
                    "error": str(e)[:200]})
    finally:
        for var, val in prev.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val
    res["cell_wall_s"] = round(time.perf_counter() - t_cell0, 3)
    res["ts_epoch"] = int(time.time())
    return res


# ----- sparse BT32 screening --------------------------------------------------

def _block_mask(nq: int, nk: int, density: float, seed: int) -> mx.array:
    """Symmetric block mask (NQ,NK) bool at ~`density`, diagonal always on (seeded)."""
    mx.random.seed(seed)
    m = (mx.random.uniform(shape=(nq, nk)) < density)
    m = (m | m.T) | (mx.eye(nq, dtype=mx.float32) > 0.5)
    return m.astype(mx.bool_)


def _causal_block_mask(nq: int, nk: int, density: float, seed: int) -> mx.array:
    """Causal block mask: a density-subset of the LOWER-triangular blocks (j<=i), diagonal on.
    density=1.0 → the full causal block set (calibration reference)."""
    mx.random.seed(seed)
    rand = (mx.random.uniform(shape=(nq, nk)) < density)
    ii = mx.arange(nq)[:, None]
    jj = mx.arange(nk)[None, :]
    lower = (jj <= ii)
    m = (rand & lower) | (mx.eye(nq, dtype=mx.float32) > 0.5)
    return m.astype(mx.bool_)


def screen_sparse(cell: dict, warmup: int = 4, iters: int = 12, adaptive: bool = True) -> dict:
    """Stage-1 screening of the v6nax_sparse BT-block kernel vs SDPA+expanded-mask.

    Sparse tiles are pinned (BT=32, BQ=BK=32) — the swept axes are density × shape, not tiles.
    Oracle: cos vs SDPA with the block mask expanded to element level. Engagement: byteΔ vs
    'scalar_fallback' > 0 (v6nax_sparse is a distinct kernel; 0 would be a silent fallback).
    Validated Phase-0 afternoon: cos=1.0, byteΔ_vs_scalar=7.6e-6 at D{64,128} N4096 BT32.
    """
    from mlx_mfa import _ext
    os.environ["MFA_SILENCE_NAX_WARNING"] = "1"
    shape = cell["shape"]
    BT = int(cell.get("BT", 32))
    density = float(cell.get("density", 0.3))
    causal = bool(shape.get("causal", False))
    res: dict[str, Any] = {"cell_id": cell["cell_id"], "kernel": "sparse", "mode": cell.get("mode", "screening"),
                           "shape": shape, "BT": BT, "density": density}
    if os.environ.get("NAAR_DAY_AMBIENT"):
        res["day_ambient"] = True
    t0 = time.perf_counter()
    try:
        q, k, v, D = _mk_qkv(shape)
        scale = 1.0 / (D ** 0.5)
        nq = nk = shape["N"] // BT
        bm = _causal_block_mask(nq, nk, density, seed=0) if causal else _block_mask(nq, nk, density, seed=0)
        mx.eval(bm)
        nax = lambda: _ext.sparse_attention_forward(q, k, v, bm, BT, causal, scale, "v6nax_sparse", False, 0)
        if adaptive:
            med, reps, coarse = _med_adaptive(nax)
        else:
            med, reps, coarse = _med(nax, warmup, iters), iters, False
        o = nax(); mx.eval(o)
        sca = _ext.sparse_attention_forward(q, k, v, bm, BT, causal, scale, "scalar_fallback", False, 0)
        mx.eval(sca)
        em = mx.repeat(mx.repeat(bm, BT, axis=0), BT, axis=1)   # (N,N) block-active
        if causal:                                             # AND element-level causal triangle
            idx = mx.arange(shape["N"])
            em = em & (idx[None, :] <= idx[:, None])           # key j <= query i
        qdt = q.dtype                                          # mask dtype must match input (bf16 fix)
        addm = mx.where(em, mx.array(0.0, qdt), mx.array(-6e4, qdt))
        sdpa_run = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=addm)
        sdpa = sdpa_run(); mx.eval(sdpa)
        sdpa_ms = _med_adaptive(sdpa_run)[0] if adaptive else _med(sdpa_run, warmup, iters)
        res.update({
            "median_ms": round(med, 5), "sdpa_ms": round(sdpa_ms, 5),
            "ratio_vs_sdpa": round(sdpa_ms / med, 4), "reps": reps, "coarse": coarse,
            "effective_density": round(float(bm.astype(mx.float32).mean()), 4),
            "cos_vs_sdpa": round(_cos(o, sdpa), 6), "byteDelta_vs_scalar": _maxabs(o, sca),
            "engaged": _maxabs(o, sca) > 0.0, "correct": _cos(o, sdpa) >= 0.999,
            "finite": bool(np.isfinite(np.asarray(o.astype(mx.float32))).all()), "error": None,
        })
    except Exception as e:
        res.update({"median_ms": None, "engaged": False, "correct": False, "error": str(e)[:200]})
    res["cell_wall_s"] = round(time.perf_counter() - t0, 3)
    res["ts_epoch"] = int(time.time())
    return res


# ----- backward-D64 screening -------------------------------------------------

def screen_backward(cell: dict, warmup: int = 4, iters: int = 12, adaptive: bool = True) -> dict:
    """Stage-1 screening of the V6 split backward (prod default-on at D=64) vs SDPA-vjp.

    Which-binary: byteΔ of dQ vs the SDPA-vjp gradient (MFA_DISABLE_V6_BACKWARD=1) > 0 → v6 ran.
    Oracle: dQ cos vs SDPA-vjp >= 0.999. Ratio: time(SDPA-vjp)/time(v6). Tile knobs (BQ/BK/WM)
    are swept like dense (same primitive reads getenv). Validated: ratio 2.16× at D64 N4096.
    """
    import mlx_mfa
    os.environ["MFA_SILENCE_NAX_WARNING"] = "1"
    shape = cell["shape"]
    tiles = cell.get("tiles", {}) or {}
    causal = bool(shape.get("causal", False))
    prev = {}
    for var, key in (("MFA_V6_NAX_BQ", "BQ"), ("MFA_V6_NAX_BK", "BK"), ("MFA_V6_NAX_WM", "WM")):
        prev[var] = os.environ.get(var)
        os.environ.pop(var, None)
        if tiles.get(key) is not None:
            os.environ[var] = str(tiles[key])
    prev["MFA_DISABLE_V6_BACKWARD"] = os.environ.get("MFA_DISABLE_V6_BACKWARD")
    res: dict[str, Any] = {"cell_id": cell["cell_id"], "kernel": "backward", "mode": cell.get("mode", "screening"),
                           "shape": shape, "tiles": tiles}
    if os.environ.get("NAAR_DAY_AMBIENT"):
        res["day_ambient"] = True
    t0 = time.perf_counter()
    try:
        q, k, v, D = _mk_qkv(shape)
        def loss(a, b, c):
            return mlx_mfa.flash_attention(a, b, c, causal=causal).sum()
        gfn = mx.grad(loss, argnums=(0, 1, 2))
        os.environ.pop("MFA_DISABLE_V6_BACKWARD", None)          # v6 split backward (default-on)
        v6_run = lambda: gfn(q, k, v)
        if adaptive:
            med, reps, coarse = _med_adaptive(v6_run)
        else:
            med, reps, coarse = _med(v6_run, warmup, iters), iters, False
        g6 = gfn(q, k, v); mx.eval(g6)
        os.environ["MFA_DISABLE_V6_BACKWARD"] = "1"              # SDPA-vjp reference
        sdpa_run = lambda: gfn(q, k, v)
        sdpa_ms = _med_adaptive(sdpa_run)[0] if adaptive else _med(sdpa_run, warmup, iters)
        gs = gfn(q, k, v); mx.eval(gs)
        dq6, dqs = g6[0], gs[0]
        res.update({
            "median_ms": round(med, 5), "sdpa_vjp_ms": round(sdpa_ms, 5),
            "ratio_vs_sdpa_vjp": round(sdpa_ms / med, 4), "reps": reps, "coarse": coarse,
            "cos_vs_sdpa_vjp": round(_cos(dq6, dqs), 6), "byteDelta_vs_sdpa_vjp": _maxabs(dq6, dqs),
            "engaged": _maxabs(dq6, dqs) > 0.0, "correct": _cos(dq6, dqs) >= 0.999,
            "finite": bool(np.isfinite(np.asarray(dq6.astype(mx.float32))).all()), "error": None,
        })
    except Exception as e:
        res.update({"median_ms": None, "engaged": False, "correct": False, "error": str(e)[:200]})
    finally:
        for var, val in prev.items():
            os.environ.pop(var, None)
            if val is not None:
                os.environ[var] = val
    res["cell_wall_s"] = round(time.perf_counter() - t0, 3)
    res["ts_epoch"] = int(time.time())
    return res


# ----- decode (narrow qL, GQA) screening skeleton -----------------------------

def screen_decode(cell: dict, warmup: int = 4, iters: int = 12, adaptive: bool = True) -> dict:
    """Stage-1 characterization of the narrow-qL GQA decode carveout vs SDPA.

    Decode has NO own NAX kernel today — the qL≤8/16 carveout routes to STEEL V2 (parity-by-
    identity) only in narrow corners; most configs FALL BACK to SDPA. `engaged = byteΔ vs SDPA > 0`
    is the which-binary gate: a decode cell that falls back to SDPA (byteΔ=0) is NOT an MFA win,
    only a distinct-kernel corner is. Candidates = engaged AND ratio>1 (census: kL≥16384 ∧ GQA≥8).
    shape = {B, Hq, Hk, Nq, S, D, dtype}.
    """
    import mlx_mfa
    os.environ["MFA_SILENCE_NAX_WARNING"] = "1"
    s = cell["shape"]
    causal = bool(s.get("causal", False))
    B, Hq, Hk, Nq, S, D = s["B"], s["Hq"], s["Hk"], s["Nq"], s["S"], s["D"]
    dt = _DTYPES[s.get("dtype", "float16")]
    res: dict[str, Any] = {"cell_id": cell["cell_id"], "kernel": "decode", "mode": cell.get("mode", "screening"),
                           "shape": s, "gqa": (Hq // Hk if Hk else 0)}
    if os.environ.get("NAAR_DAY_AMBIENT"):
        res["day_ambient"] = True
    t0 = time.perf_counter()
    try:
        mx.random.seed(0)
        q = (mx.random.normal((B, Hq, Nq, D)) * 0.1).astype(dt)
        k = (mx.random.normal((B, Hk, S, D)) * 0.1).astype(dt)
        v = (mx.random.normal((B, Hk, S, D)) * 0.1).astype(dt)
        mx.eval(q, k, v)
        scale = 1.0 / (D ** 0.5)
        mfa = lambda: mlx_mfa.flash_attention(q, k, v, causal=causal)
        sdpa_run = lambda: mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask="causal" if causal else None)
        if adaptive:
            med, reps, coarse = _med_adaptive(mfa)
            sdpa_ms, _, _ = _med_adaptive(sdpa_run)
        else:
            med, reps, coarse = _med(mfa, warmup, iters), iters, False
            sdpa_ms = _med(sdpa_run, warmup, iters)
        o = mfa(); mx.eval(o)
        sd = sdpa_run(); mx.eval(sd)
        res.update({
            "median_ms": round(med, 5), "sdpa_ms": round(sdpa_ms, 5),
            "ratio_vs_sdpa": round(sdpa_ms / med, 4), "reps": reps, "coarse": coarse,
            "cos_vs_sdpa": round(_cos(o, sd), 6), "byteDelta_vs_sdpa": _maxabs(o, sd),
            "engaged": _maxabs(o, sd) > 0.0,          # distinct MFA/STEEL kernel, not the SDPA fallback
            "correct": _cos(o, sd) >= 0.999,
            "finite": bool(np.isfinite(np.asarray(o.astype(mx.float32))).all()), "error": None,
        })
    except Exception as e:
        res.update({"median_ms": None, "engaged": False, "correct": False, "error": str(e)[:200]})
    res["cell_wall_s"] = round(time.perf_counter() - t0, 3)
    res["ts_epoch"] = int(time.time())
    return res


# executor registry: kernel -> screening fn. Extended per family as the census lands.
SCREEN_EXECUTORS: dict[str, Callable[[dict], dict]] = {
    "dense": screen_dense,
    "sparse": screen_sparse,
    "backward": screen_backward,
    "decode": screen_decode,
}


def run_cell(cell: dict) -> dict:
    ex = SCREEN_EXECUTORS.get(cell["kernel"])
    if ex is None:
        return {"cell_id": cell["cell_id"], "kernel": cell["kernel"], "error": "no executor",
                "engaged": False, "correct": False, "ts_epoch": int(time.time())}
    return ex(cell)
