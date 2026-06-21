#!/usr/bin/env python3
"""Axes 2/4/5/6 per-axis empirical sweep on production VSR shapes.

For each axis, we test variants against the CURRENT-BEST (R, C, SG) config
from dispatch-table-final.json. Each case runs in a subprocess to honor
env-var compile-time effects, with warmup=3 + iters=15.
"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Current best (R, C, SG) per shape — from v6-dispatch-table-final.json.
BEST = {
    "FlashVSR-dense": {"B":1,"H":10,"N":4096,"D":64,  "R":16,"C":64,"SG":16, "p50_ref":1.32},
    "SeedVR2-small":  {"B":1,"H":20,"N":26730,"D":128,"R":16,"C":48,"SG":16, "p50_ref":241.95},
    "CogVideoX":      {"B":1,"H":30,"N":70200,"D":128,"R":16,"C":48,"SG":16, "p50_ref":2817.20},
    "SeedVR2-large":  {"B":1,"H":20,"N":111375,"D":128,"R":16,"C":48,"SG":16,"p50_ref":4706.28},
}

CHILD = """
import sys, time, math
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward

B, H, N, D = __B__, __H__, __N__, __D__
warmup, iters = 3, 15

mx.random.seed(0)
q = mx.random.normal((B,H,N,D)).astype(mx.float16)
k = mx.random.normal((B,H,N,D)).astype(mx.float16)
v = mx.random.normal((B,H,N,D)).astype(mx.float16)
mx.async_eval(q,k,v); mx.synchronize()

try:
    for _ in range(warmup):
        o,_ = v6_nax_forward(q,k,v,False); mx.async_eval(o); mx.synchronize()
    ts=[]
    for _ in range(iters):
        mx.synchronize(); t0=time.perf_counter()
        o,_ = v6_nax_forward(q,k,v,False); mx.async_eval(o); mx.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    ts.sort()
    print("OK,%.4f,%.4f,%.4f" % (ts[len(ts)//2], ts[0], ts[-1]))
except Exception as e:
    err = str(e).replace(chr(10),' ')[:200]
    print("ERR," + type(e).__name__ + ": " + err)
"""

def run_case(shape_name, overrides, label, timeout_s=300):
    s = BEST[shape_name]
    env = os.environ.copy()
    env["MFA_V6_BLOCK_R"] = str(s["R"])
    env["MFA_V6_BLOCK_C"] = str(s["C"])
    env["MFA_V6_EXEC_SG"] = str(s["SG"])
    for k, v in overrides.items(): env[k] = v
    src = (CHILD.replace("__B__", str(s["B"]))
                .replace("__H__", str(s["H"]))
                .replace("__N__", str(s["N"]))
                .replace("__D__", str(s["D"])))
    t0 = time.perf_counter()
    try:
        r = subprocess.run([".venv/bin/python","-c",src], env=env,
                          capture_output=True, text=True, timeout=timeout_s)
        out = r.stdout.strip()
    except subprocess.TimeoutExpired:
        return None, f"timeout {timeout_s}s", time.perf_counter()-t0
    dt = time.perf_counter()-t0
    if out.startswith("OK,"):
        parts = out[3:].split(",")
        return {"p50":float(parts[0]),"min":float(parts[1]),"max":float(parts[2])}, "ok", dt
    return None, out[4:200] if out.startswith("ERR,") else out[:200], dt


def main():
    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M"), "axes": {}}

    # ── Axe 2: BLOCK_D ─────────────────────────────────────────────────────
    print("=" * 78); print("AXE 2 — BLOCK_D"); print("=" * 78)
    a2 = {}
    for name, s in BEST.items():
        D = s["D"]
        bd_grid = [32, 64] if D == 64 else [32, 64, 128]
        a2[name] = {}
        # Baseline first (no override → BD = head_dim = D)
        m, st, dt = run_case(name, {}, "baseline")
        if m: a2[name]["baseline"] = m["p50"]
        print(f"  {name:<18} baseline (BD={D}): {m['p50'] if m else 'FAIL':>10}  ({dt:.1f}s)")
        for bd in bd_grid:
            if bd == D and "baseline" in a2[name]: continue  # already measured
            m2, st2, dt2 = run_case(name, {"MFA_V6_BLOCK_D": str(bd)}, f"BD={bd}")
            tag = f"{m2['p50']:.2f}" if m2 else f"FAIL ({st2[:40]})"
            a2[name][f"BD={bd}"] = m2["p50"] if m2 else None
            print(f"  {name:<18} BD={bd:<4}        : {tag:>10}  ({dt2:.1f}s)")
        print()
    results["axes"]["axe2_block_d"] = a2

    # ── Axe 4: FORCE_DYNAMIC_K ─────────────────────────────────────────────
    print("=" * 78); print("AXE 4 — FORCE_DYNAMIC_K"); print("=" * 78)
    a4 = {}
    for name in BEST:
        m, _, dt = run_case(name, {"MFA_V6_FORCE_DYNAMIC_K":"1"}, "FORCE_DYN")
        base = a2.get(name, {}).get("baseline")
        delta = ((m["p50"]/base - 1.0)*100) if (m and base) else None
        a4[name] = {"force_dyn_k": m["p50"] if m else None, "delta_pct": delta}
        tag = f"{m['p50']:.2f} ({delta:+.1f}%)" if (m and delta is not None) else "FAIL"
        print(f"  {name:<18} {tag:>22}  ({dt:.1f}s)")
    results["axes"]["axe4_force_dynamic_k"] = a4

    # ── Axe 5: RELAXED_PRECISION ───────────────────────────────────────────
    print()
    print("=" * 78); print("AXE 5 — RELAXED_PRECISION=0"); print("=" * 78)
    a5 = {}
    for name in BEST:
        m, _, dt = run_case(name, {"MFA_V6_RELAXED_PRECISION":"0"}, "RELAXED=0")
        base = a2.get(name, {}).get("baseline")
        delta = ((m["p50"]/base - 1.0)*100) if (m and base) else None
        a5[name] = {"relaxed_off": m["p50"] if m else None, "delta_pct": delta}
        tag = f"{m['p50']:.2f} ({delta:+.1f}%)" if (m and delta is not None) else "FAIL"
        print(f"  {name:<18} {tag:>22}  ({dt:.1f}s)")
    results["axes"]["axe5_relaxed_precision_off"] = a5

    # ── Axe 6: UNROLL_MODE  (4 modes; 1 D=64 + 1 D=128 shape to limit time) 
    print()
    print("=" * 78); print("AXE 6 — UNROLL_MODE"); print("=" * 78)
    a6 = {}
    for name in ["FlashVSR-dense", "SeedVR2-small"]:  # smallest of each D
        a6[name] = {}
        base = a2.get(name, {}).get("baseline")
        a6[name]["full"] = base
        for mode in ["none", "2", "4"]:
            m, _, dt = run_case(name, {"MFA_V6_UNROLL_MODE": mode}, f"UNROLL={mode}")
            delta = ((m["p50"]/base - 1.0)*100) if (m and base) else None
            a6[name][mode] = {"p50": m["p50"] if m else None, "delta_pct": delta}
            tag = f"{m['p50']:.2f} ({delta:+.1f}%)" if (m and delta is not None) else "FAIL"
            print(f"  {name:<18} UNROLL={mode:<5}: {tag:>22}  ({dt:.1f}s)")
    results["axes"]["axe6_unroll_mode"] = a6

    out_path = ROOT / "docs/v6-nax/axes_2456_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote → {out_path}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
