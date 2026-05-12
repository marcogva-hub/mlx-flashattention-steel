"""BHND vs BNHD benchmark: timing + peak memory."""
import os, gc, math, statistics, subprocess, sys, time
from pathlib import Path
import json

SHAPES = [
    {"name": "FlashVSR-dense", "B":1, "H":10, "Nq":4096,  "Nkv":4096,  "D":64,  "R":16,"C":64,"SG":16},
    {"name": "SeedVR2-small",  "B":1, "H":20, "Nq":26730, "Nkv":26730, "D":128, "R":16,"C":48,"SG":16},
    {"name": "CogVideoX",      "B":1, "H":30, "Nq":70200, "Nkv":70200, "D":128, "R":16,"C":48,"SG":16},
    {"name": "SeedVR2-large",  "B":1, "H":20, "Nq":111375,"Nkv":111375,"D":128, "R":16,"C":48,"SG":16},
    {"name": "LTX2-cross",     "B":1, "H":8,  "Nq":2048,  "Nkv":14000, "D":64,  "R":16,"C":64,"SG":8},
]

CHILD = '''
import os, time, gc, math
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward

B, H, Nq, Nkv, D = __B__, __H__, __NQ__, __NKV__, __D__
WARMUP, ITERS = 3, 15
mx.random.seed(0)
q = mx.random.normal((B,H,Nq,D)).astype(mx.float16)
k = mx.random.normal((B,H,Nkv,D)).astype(mx.float16)
v = mx.random.normal((B,H,Nkv,D)).astype(mx.float16)
mx.async_eval(q,k,v); mx.synchronize()

# Timing
for _ in range(WARMUP):
    o, _ = v6_nax_forward(q, k, v, False)
    mx.async_eval(o); mx.synchronize()
ts = []
for _ in range(ITERS):
    mx.synchronize()
    t0 = time.perf_counter()
    o, _ = v6_nax_forward(q, k, v, False)
    mx.async_eval(o); mx.synchronize()
    ts.append((time.perf_counter() - t0) * 1000)
ts.sort()
p50 = ts[len(ts)//2]

# Peak memory
del o; gc.collect(); mx.clear_cache()
base = mx.metal.get_active_memory()
mx.metal.reset_peak_memory()
o2, _ = v6_nax_forward(q, k, v, False)
mx.async_eval(o2); mx.synchronize()
peak_delta = mx.metal.get_peak_memory() - base
print(f"RESULT:p50={p50:.4f},peak_delta={peak_delta}")
'''

results = []
print(f"{'shape':<18} {'mode':<6} {'p50_ms':>10} {'peak_MB':>9} {'vs BNHD':>10}")
print("-" * 60)

for s in SHAPES:
    src = (CHILD.replace("__B__", str(s["B"])).replace("__H__", str(s["H"]))
                .replace("__NQ__", str(s["Nq"])).replace("__NKV__", str(s["Nkv"]))
                .replace("__D__", str(s["D"])))
    bnhd_p50 = bnhd_peak = None
    bhnd_p50 = bhnd_peak = None
    for mode in ["bnhd", "bhnd"]:
        env = os.environ.copy()
        env["MFA_V6_BLOCK_R"] = str(s["R"])
        env["MFA_V6_BLOCK_C"] = str(s["C"])
        env["MFA_V6_EXEC_SG"] = str(s["SG"])
        if mode == "bhnd":
            env["MFA_V6_BHND"] = "1"
        try:
            r = subprocess.run([".venv/bin/python","-c",src], env=env,
                              capture_output=True, text=True, timeout=300)
            out = r.stdout.strip()
            for line in out.split("\n"):
                if line.startswith("RESULT:"):
                    parts = line[7:].split(",")
                    p50 = float(parts[0].split("=")[1])
                    peak = int(parts[1].split("=")[1])
                    peak_mb = peak / 1e6
                    if mode == "bnhd":
                        bnhd_p50 = p50; bnhd_peak = peak_mb
                    else:
                        bhnd_p50 = p50; bhnd_peak = peak_mb
                    delta = ""
                    if mode == "bhnd" and bnhd_p50:
                        delta = f"{(p50/bnhd_p50 - 1)*100:+.1f}%"
                    print(f"{s['name']:<18} {mode:<6} {p50:>10.3f} {peak_mb:>9.2f} {delta:>10}")
                    break
            else:
                err = (r.stderr or out)[-200:]
                print(f"{s['name']:<18} {mode:<6} ERROR: {err[:60]}")
        except subprocess.TimeoutExpired:
            print(f"{s['name']:<18} {mode:<6} TIMEOUT")
    if bnhd_peak and bhnd_peak:
        mem_ratio = bnhd_peak / bhnd_peak
        time_delta = (bhnd_p50 / bnhd_p50 - 1) * 100
        results.append({**s, "bnhd_p50":bnhd_p50, "bhnd_p50":bhnd_p50,
                        "bnhd_peak_mb":bnhd_peak, "bhnd_peak_mb":bhnd_peak,
                        "time_delta_pct":time_delta, "mem_ratio":mem_ratio})
        print(f"  → memory: {mem_ratio:.2f}× reduction ({bnhd_peak:.1f} → {bhnd_peak:.1f} MB)")
    print()

with open("docs/v6-nax/bhnd-bench-results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to docs/v6-nax/bhnd-bench-results.json")
