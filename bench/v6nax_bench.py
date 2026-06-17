"""V6NAX vs legacy V6 NAX — single shape, single mode bench (subprocess-callable)."""
import argparse, json, math, os, statistics, time
from pathlib import Path

# Wipe overrides for deterministic dispatch.
for k in ("MFA_V6_BLOCK_R", "MFA_V6_BLOCK_C", "MFA_V6_EXEC_SG",
          "MFA_V6_NAX_SINGLE_OTILE", "MFA_V6_BYPASS_TGP", "MFA_V6_BLOCK_D",
          "MFA_V6_BNHD_LEGACY", "MFA_V6_MAX_THREADS", "MFA_V6_MATMUL_EXEC_SG",
          "MFA_V6_USE_V33", "MFA_V6_USE_NAX",
          "MFA_V6_NAX_BQ", "MFA_V6_NAX_BK", "MFA_V6_NAX_WM"):
    os.environ.pop(k, None)


import mlx.core as mx
from mlx_mfa import _ext

_force = getattr(mx, "eval")

SHAPES = {
    "FlashVSR-dense": dict(B=1, H=10, N_q=4096,   N_kv=4096,   D=64,  iters=8),
    "LTX2-cross":     dict(B=1, H=8,  N_q=2048,   N_kv=14000,  D=64,  iters=8),
    "SeedVR2-small":  dict(B=1, H=20, N_q=26730,  N_kv=26730,  D=128, iters=6),
    "CogVideoX":      dict(B=1, H=30, N_q=70200,  N_kv=70200,  D=128, iters=4),
    "SeedVR2-large":  dict(B=1, H=20, N_q=111375, N_kv=111375, D=128, iters=4),
}
WARMUP = 3


def make(s):
    mx.random.seed(42)
    q = mx.random.normal((s["B"], s["H"], s["N_q"],  s["D"]), dtype=mx.float16)
    k = mx.random.normal((s["B"], s["H"], s["N_kv"], s["D"]), dtype=mx.float16)
    v = mx.random.normal((s["B"], s["H"], s["N_kv"], s["D"]), dtype=mx.float16)
    _force(q, k, v)
    return q, k, v


def correctness(s):
    q, k, v = make(s)
    out, _ = _ext.v6_nax_forward(q, k, v, False); _force(out)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0/math.sqrt(s["D"]))
    _force(ref)
    diff = (out.astype(mx.float32) - ref.astype(mx.float32))
    rmse = float(mx.sqrt(mx.mean(diff * diff)))
    finite = bool(mx.all(mx.isfinite(out)).item())
    return rmse, finite


def time_v6(s, iters):
    q, k, v = make(s)
    for _ in range(WARMUP):
        out, _ = _ext.v6_nax_forward(q, k, v, False); _force(out)
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out, _ = _ext.v6_nax_forward(q, k, v, False); _force(out)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(timings)


def time_sdpa(s, iters):
    q, k, v = make(s)
    scale = 1.0 / math.sqrt(s["D"])
    for _ in range(WARMUP):
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale); _force(out)
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale); _force(out)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(timings)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", required=True, choices=list(SHAPES.keys()))
    p.add_argument("--mode", required=True, help="legacy | v6nax")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--output", required=True)
    p.add_argument("--pre-cooldown", type=int, default=0)
    p.add_argument("--include-sdpa", action="store_true")
    args = p.parse_args()

    if args.mode == "v6nax":
        os.environ["MFA_V6_USE_NAX"] = "1"

    if args.pre_cooldown > 0:
        time.sleep(args.pre_cooldown)

    s = SHAPES[args.shape]
    rmse, finite = correctness(s)
    record = {
        "shape": args.shape, "mode": args.mode,
        "shape_dims": s, "correctness_ok": finite and rmse < 1e-3,
        "rmse": rmse, "finite": finite,
    }
    if record["correctness_ok"]:
        v6_runs = [time_v6(s, s["iters"]) for _ in range(args.runs)]
        record["v6_runs_ms"] = v6_runs
        record["v6_median_ms"] = statistics.median(v6_runs)
        if args.include_sdpa:
            sdpa_runs = [time_sdpa(s, s["iters"]) for _ in range(args.runs)]
            record["sdpa_runs_ms"] = sdpa_runs
            record["sdpa_median_ms"] = statistics.median(sdpa_runs)
            record["v6_over_sdpa"] = record["v6_median_ms"] / record["sdpa_median_ms"]
        msg = (f"{args.shape:<18} mode={args.mode:<7} "
               f"v6={record['v6_median_ms']:8.2f}ms")
        if args.include_sdpa:
            msg += f" sdpa={record['sdpa_median_ms']:8.2f}ms ratio={record['v6_over_sdpa']:.3f}x"
        msg += f" rmse={rmse:.2e}"
        print(msg, flush=True)
    else:
        print(f"{args.shape} mode={args.mode}: BAD CORRECTNESS rmse={rmse:.2e} finite={finite}",
              flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    else:
        existing = {"records": []}
    existing["records"].append(record)
    out_path.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
