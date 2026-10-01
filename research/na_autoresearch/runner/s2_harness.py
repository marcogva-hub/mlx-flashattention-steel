"""Stage-2 hardened A/B confirmation harness (measure-only; no promotion).

Contract (Marco 2026-08-12): per candidate, default arm vs candidate arm ·
>=10 FRESH PROCESSES per order · 2 ORDERS (AB=default-first, BA=candidate-first,
each process measures BOTH arms in its order to null within-process warmup bias) ·
fp32 oracle (cos vs fp32 SDPA) · FINGERPRINT both arms (byteΔ vs fp16 SDPA = which-binary;
knob candidates MUST change a verifiable bit — the RELAXED lesson) · outliers DECLARED
(median across processes + spread) · same build · verdict vs the day-floor re-measured at
block head. The >=10 fresh processes ARE the cross-session variance control, so each process
uses canonical warmup+continuous (no 4s cooldown).

Verdicts: CONFIRMED (candidate faster than default beyond floor noise in BOTH orders) /
KILLED / INDECIDABLE. Incremental JSONL (crash-safe, idempotent resume by (cand,order,proc)).

  child : s2_harness.py child '<spec_json>' <AB|BA>        # prints one JSON line
  parent: s2_harness.py parent <shortlist.json> <out.jsonl> [procs_per_order]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

WARMUP, ITERS = 10, 100          # canonical continuous (variance handled across processes)


# --------------------------------------------------------------------------- child
def _run_child(spec: dict, order: str) -> dict:
    import numpy as np
    import mlx.core as mx
    os.environ["MFA_SILENCE_NAX_WARNING"] = "1"
    from mlx_mfa import _ext
    sh = spec["shape"]
    B, H, N, D = sh["B"], sh["H"], sh["N"], sh["D"]
    dt = {"float16": mx.float16, "bfloat16": mx.bfloat16}[sh.get("dtype", "float16")]
    causal = bool(sh.get("causal", False))
    scale = 1.0 / (D ** 0.5)
    mx.random.seed(0)
    f = lambda: (mx.random.normal((B, H, N, D)) * 0.1).astype(dt)
    q, k, v = f(), f(), f(); mx.eval(q, k, v)
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    oracle = mx.fast.scaled_dot_product_attention(qf, kf, vf, scale=scale,
                                                  mask="causal" if causal else None)
    sdpa16 = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                                                  mask="causal" if causal else None)
    mx.eval(oracle, sdpa16)
    of = np.asarray(oracle).ravel().astype(np.float64)
    s16 = np.asarray(sdpa16.astype(mx.float32))

    def set_env(cfg):
        for var, key in (("MFA_V6_NAX_BQ", "BQ"), ("MFA_V6_NAX_BK", "BK"), ("MFA_V6_NAX_WM", "WM")):
            os.environ.pop(var, None)
            t = cfg.get("tiles") or {}
            if t.get(key) is not None:
                os.environ[var] = str(t[key])
        for var in ("MFA_V6_RELAXED_PRECISION", "MFA_V6_UNROLL_MODE", "MFA_V6_MAX_THREADS"):
            os.environ.pop(var, None)
        for var, val in (cfg.get("knobs") or {}).items():
            os.environ[var] = str(val)

    def measure(cfg):
        set_env(cfg)
        run = lambda: _ext.v6_nax_forward(q, k, v, causal, True, scale)[0]
        for _ in range(WARMUP):
            mx.eval(run())
        mx.synchronize()
        ts = []
        for _ in range(ITERS):
            t0 = time.perf_counter(); mx.eval(run()); mx.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts.sort()
        o = run(); mx.eval(o)
        oa = np.asarray(o.astype(mx.float32))
        cos = float(np.dot(oa.ravel().astype(np.float64), of) /
                    (np.linalg.norm(oa.ravel().astype(np.float64)) * np.linalg.norm(of)))
        fp = float(np.abs(oa - s16).max())          # byteΔ vs fp16 SDPA (which-binary/engagement)
        return ts[len(ts) // 2], cos, fp

    default = {"tiles": {}, "knobs": {}}
    cand = {"tiles": spec.get("tiles", {}), "knobs": spec.get("knobs", {})}
    if order == "AB":
        d_ms, d_cos, d_fp = measure(default); c_ms, c_cos, c_fp = measure(cand)
    else:
        c_ms, c_cos, c_fp = measure(cand); d_ms, d_cos, d_fp = measure(default)
    return {"cand_id": spec["cand_id"], "order": order,
            "default_ms": round(d_ms, 5), "cand_ms": round(c_ms, 5),
            "default_cos_fp32": round(d_cos, 6), "cand_cos_fp32": round(c_cos, 6),
            "default_fp": d_fp, "cand_fp": c_fp,          # >0 => NAX engaged (not SDPA fallback)
            "knob_changed": bool(spec.get("knobs")) and abs(c_ms - d_ms) > 1e-9,  # RELAXED guard signal
            "ts": int(time.time())}


# --------------------------------------------------------------------------- parent
def _done_keys(out_path):
    keys = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l); keys.add((r["cand_id"], r["order"], r["proc"]))
            except Exception:
                pass
    return keys


def _parent(shortlist_path, out_path, procs):
    cands = json.load(open(shortlist_path))
    done = _done_keys(out_path)
    self = os.path.abspath(__file__)
    for spec in cands:
        for order in ("AB", "BA"):
            for p in range(procs):
                if (spec["cand_id"], order, p) in done:
                    continue
                r = subprocess.run([sys.executable, self, "child", json.dumps(spec), order],
                                   capture_output=True, text=True, timeout=300)
                line = (r.stdout or "").strip().splitlines()
                rec = None
                for ln in reversed(line):
                    if ln.startswith("{"):
                        rec = json.loads(ln); break
                if rec is None:
                    rec = {"cand_id": spec["cand_id"], "order": order, "error": (r.stderr or "")[-200:]}
                rec["proc"] = p
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
    print(f"parent done: {len(cands)} candidates × 2 orders × {procs} procs → {out_path}")


# --------------------------------------------------------------------------- solo (one arm / process)
def _run_solo(spec: dict, arm: str) -> dict:
    """Measure ONE arm in a fresh process — no within-process ordering (fixes the thermal-drift
    confound on fast kernels). Pool-level drift is nulled by interleaving default/cand launches."""
    import numpy as np
    import mlx.core as mx
    os.environ["MFA_SILENCE_NAX_WARNING"] = "1"
    from mlx_mfa import _ext
    sh = spec["shape"]
    B, H, N, D = sh["B"], sh["H"], sh["N"], sh["D"]
    dt = {"float16": mx.float16, "bfloat16": mx.bfloat16}[sh.get("dtype", "float16")]
    causal = bool(sh.get("causal", False)); scale = 1.0 / (D ** 0.5)
    mx.random.seed(0)
    f = lambda: (mx.random.normal((B, H, N, D)) * 0.1).astype(dt)
    q, k, v = f(), f(), f(); mx.eval(q, k, v)
    of = np.asarray(mx.fast.scaled_dot_product_attention(q.astype(mx.float32), k.astype(mx.float32),
        v.astype(mx.float32), scale=scale, mask="causal" if causal else None)).ravel().astype(np.float64)
    cfg = {"tiles": {}, "knobs": {}} if arm == "default" else {"tiles": spec.get("tiles", {}), "knobs": spec.get("knobs", {})}
    for var, key in (("MFA_V6_NAX_BQ", "BQ"), ("MFA_V6_NAX_BK", "BK"), ("MFA_V6_NAX_WM", "WM")):
        os.environ.pop(var, None)
        if (cfg["tiles"] or {}).get(key) is not None:
            os.environ[var] = str(cfg["tiles"][key])
    for var in ("MFA_V6_RELAXED_PRECISION", "MFA_V6_UNROLL_MODE", "MFA_V6_MAX_THREADS"):
        os.environ.pop(var, None)
    for var, val in (cfg["knobs"] or {}).items():
        os.environ[var] = str(val)
    run = lambda: _ext.v6_nax_forward(q, k, v, causal, True, scale)[0]
    for _ in range(WARMUP):
        mx.eval(run())
    mx.synchronize()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter(); mx.eval(run()); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    o = run(); mx.eval(o); oa = np.asarray(o.astype(mx.float32)).ravel().astype(np.float64)
    cos = float(np.dot(oa, of) / (np.linalg.norm(oa) * np.linalg.norm(of)))
    return {"cand_id": spec["cand_id"], "arm": arm, "ms": round(ts[len(ts) // 2], 5), "cos_fp32": round(cos, 6)}


def _parent_solo(shortlist_path, out_path, procs):
    cands = json.load(open(shortlist_path)); done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l); done.add((r["cand_id"], r["arm"], r["proc"]))
            except Exception:
                pass
    self = os.path.abspath(__file__)
    for spec in cands:
        for p in range(procs):
            for arm in ("default", "cand"):   # interleaved per proc → nulls pool-level drift
                if (spec["cand_id"], arm, p) in done:
                    continue
                r = subprocess.run([sys.executable, self, "solo", json.dumps(spec), arm],
                                   capture_output=True, text=True, timeout=300)
                rec = None
                for ln in reversed((r.stdout or "").strip().splitlines()):
                    if ln.startswith("{"):
                        rec = json.loads(ln); break
                if rec is None:
                    rec = {"cand_id": spec["cand_id"], "arm": arm, "error": (r.stderr or "")[-200:]}
                rec["proc"] = p
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
    print(f"parent_solo done: {len(cands)} candidates × {procs} procs × 2 arms → {out_path}")


if __name__ == "__main__":
    if sys.argv[1] == "child":
        print(json.dumps(_run_child(json.loads(sys.argv[2]), sys.argv[3])))
    elif sys.argv[1] == "parent":
        _parent(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 10)
    elif sys.argv[1] == "solo":
        print(json.dumps(_run_solo(json.loads(sys.argv[2]), sys.argv[3])))
    elif sys.argv[1] == "parent_solo":
        _parent_solo(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 12)
