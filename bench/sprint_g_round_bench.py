"""Single-round bench for Sprint G re-bench A/B/A.

Calls _ext.v6_nax_forward DIRECTLY (V6 NAX kernel) — NOT mlx_mfa.attention
which routes via flash_attention to STEEL/SDPA.

Multi-run: 3 runs × 6-10 iters median, median-of-medians.
Internal cooldowns between shapes for thermal stability.
"""
import argparse, json, math, os, statistics, time
from pathlib import Path

# Wipe overrides so we test the dispatch baked into the build.
for k in ("MFA_V6_BLOCK_R","MFA_V6_BLOCK_C","MFA_V6_EXEC_SG",
          "MFA_V6_NAX_SINGLE_OTILE","MFA_V6_BYPASS_TGP","MFA_V6_BLOCK_D",
          "MFA_V6_BNHD_LEGACY","MFA_V6_MAX_THREADS","MFA_V6_MATMUL_EXEC_SG"):
    os.environ.pop(k, None)

import mlx.core as mx
from mlx_mfa import _ext

_force = getattr(mx, "eval")

SHAPES = [
    {"name": "FlashVSR-dense", "B":1, "H":10, "N_q":4096,   "N_kv":4096,   "D":64,  "iters":8},
    {"name": "LTX2-cross",     "B":1, "H":8,  "N_q":2048,   "N_kv":14000,  "D":64,  "iters":8},
    {"name": "SeedVR2-small",  "B":1, "H":20, "N_q":26730,  "N_kv":26730,  "D":128, "iters":6},
    {"name": "CogVideoX",      "B":1, "H":30, "N_q":70200,  "N_kv":70200,  "D":128, "iters":4},
    {"name": "SeedVR2-large",  "B":1, "H":20, "N_q":111375, "N_kv":111375, "D":128, "iters":4},
]
WARMUP = 3
INTER_SHAPE_COOL = 30  # seconds between shapes within a round


def make(s):
    mx.random.seed(42)
    q = mx.random.normal((s["B"], s["H"], s["N_q"], s["D"]), dtype=mx.float16)
    k = mx.random.normal((s["B"], s["H"], s["N_kv"], s["D"]), dtype=mx.float16)
    v = mx.random.normal((s["B"], s["H"], s["N_kv"], s["D"]), dtype=mx.float16)
    _force(q, k, v)
    return q, k, v


def time_v6(s, iters):
    q, k, v = make(s)
    for _ in range(WARMUP):
        o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(timings)


def time_sdpa(s, iters):
    q, k, v = make(s)
    scale = 1.0 / math.sqrt(s["D"])
    for _ in range(WARMUP):
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale); _force(o)
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale); _force(o)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(timings)


def correctness(s):
    q, k, v = make(s)
    out, _ = _ext.v6_nax_forward(q, k, v, False); _force(out)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0/math.sqrt(s["D"]))
    _force(ref)
    diff = (out.astype(mx.float32) - ref.astype(mx.float32))
    rmse = float(mx.sqrt(mx.mean(diff * diff)))
    finite = bool(mx.all(mx.isfinite(out)).item())
    return rmse, finite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pre-cooldown", type=int, default=0)
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()

    if args.pre_cooldown > 0:
        print(f"[{args.label}] pre-cooldown {args.pre_cooldown}s", flush=True)
        time.sleep(args.pre_cooldown)

    print(f"[{args.label}] starting (V6 NAX direct via _ext.v6_nax_forward)", flush=True)
    round_data = {"label": args.label, "shapes": {}}
    for shape_idx, s in enumerate(SHAPES):
        rmse, finite = correctness(s)
        if not finite or rmse > 5e-3:
            print(f"  {s['name']:<20}: BAD CORRECTNESS rmse={rmse:.2e} finite={finite}", flush=True)
            round_data["shapes"][s["name"]] = {"correctness_ok": False, "rmse": rmse}
            continue

        v6_runs = [time_v6(s, s["iters"]) for _ in range(args.runs)]
        v6_med = statistics.median(v6_runs)
        sdpa_runs = [time_sdpa(s, s["iters"]) for _ in range(args.runs)]
        sdpa_med = statistics.median(sdpa_runs)
        ratio = v6_med / sdpa_med
        round_data["shapes"][s["name"]] = {
            "correctness_ok": True, "rmse": rmse,
            "v6_runs_ms": v6_runs, "v6_median_ms": v6_med,
            "sdpa_runs_ms": sdpa_runs, "sdpa_median_ms": sdpa_med,
            "v6_over_sdpa": ratio,
        }
        print(f"  {s['name']:<20}: v6={v6_med:8.2f}ms sdpa={sdpa_med:8.2f}ms ratio={ratio:.2f}× "
              f"(v6 runs: {[f'{x:.2f}' for x in v6_runs]})", flush=True)
        # Inter-shape cooldown for thermal stability (skip after last shape)
        if shape_idx < len(SHAPES) - 1:
            print(f"  [inter-shape cooldown {INTER_SHAPE_COOL}s]", flush=True)
            time.sleep(INTER_SHAPE_COOL)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        if "rounds" not in existing:
            existing = {"rounds": []}
    else:
        existing = {"rounds": []}
    existing["rounds"].append(round_data)
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"[{args.label}] done; appended to {out_path}", flush=True)


if __name__ == "__main__":
    main()
