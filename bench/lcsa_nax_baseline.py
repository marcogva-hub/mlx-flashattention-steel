#!/usr/bin/env python3
"""Sprint B Phase 0 — LCSA / block-sparse attention baseline bench.

Compares the current mlx-mfa sparse path against MLX SDPA-with-float-mask
on representative FlashVSR LCSA shapes. The current mlx-mfa path on
M5+ is `_sparse_fallback_sdpa_perhead()` (per attention.py:2228) which
expands the block mask to float bias and calls SDPA — i.e. dense compute
+ mask, no block-skip. The MLX SDPA reference is essentially the same
operation called from outside mlx-mfa.

Methodology (Sprint C precedent §4):
- 3 sessions sequential, §4 cooldowns (90/60/180s)
- A/B/A pattern per shape per session
- Smoke gate (Phase 1.1 lesson): correctness check before timing
- Conditions sidecar per Artifact #5

Output: docs/lcsa-nax/lcsa-nax-phase0-baseline-data.json
"""
import argparse, json, math, os, platform, statistics, subprocess
import sys, time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_mfa.attention import flash_attention_sparse, _steel_block_config
from mlx_mfa.attention import make_sliding_window_mask


# Representative FlashVSR-style LCSA shapes per §3.4. Real FlashVSR uses
# head_dim=128, num_heads=12 (Wan2.1 DiT dim=1536). Sequence lengths span
# small-clip to long-clip latent token counts.
#
# Tuple: (label, B, H, N=K, D, sparsity_label, local_range_tokens)
# local_range is window in TOKENS (BK-block mask uses tiles of size 16).
SHAPES = [
    # Small VSR clip (~10 latent frames × 16×32 spatial = 5120 tokens; round to 4096)
    ("lcsa_small_seq4k",    1, 12,  4096, 128, "dense_window_512",  512),
    ("lcsa_small_seq4k_sparse", 1, 12,  4096, 128, "sparse_window_128", 128),
    # Medium VSR clip (~16 latent frames × 16×32 = 8192)
    ("lcsa_mid_seq8k",      1, 12,  8192, 128, "dense_window_512",  512),
    ("lcsa_mid_seq8k_sparse",   1, 12,  8192, 128, "sparse_window_128", 128),
    # Larger VSR clip (~24 latent frames × 32×32 = ~24k → 16k for bench tractability)
    ("lcsa_large_seq16k",   1, 12, 16384, 128, "dense_window_1024", 1024),
    ("lcsa_large_seq16k_sparse",1, 12, 16384, 128, "sparse_window_256", 256),
]

# Smoke gate shape: small enough that FP16 noise is negligible.
SMOKE_CFG = (1, 4, 256, 64)
SMOKE_REL_BAR = 1e-2


def make_inputs(B, H, N, D, seed=0):
    mx.random.seed(seed)
    q = (mx.random.uniform(shape=(B, H, N, D)) * 0.1).astype(mx.float16)
    k = (mx.random.uniform(shape=(B, H, N, D)) * 0.1).astype(mx.float16)
    v = (mx.random.uniform(shape=(B, H, N, D)) * 0.1).astype(mx.float16)
    mx.async_eval(q, k, v); mx.synchronize()
    return q, k, v


def sliding_mask_with_density(N, D, window_size):
    """Return (block_mask, density) for sliding-window pattern."""
    mask = make_sliding_window_mask(N, window_size=window_size, head_dim=D)
    # Density = fraction of True entries
    mx.async_eval(mask); mx.synchronize()
    import numpy as np
    arr = np.array(mask)
    density = float(arr.mean())
    return mask, density


def smoke_gate():
    """Phase 1.1 lesson: correctness BEFORE timing. Run on tiny shape."""
    B, H, N, D = SMOKE_CFG
    q, k, v = make_inputs(B, H, N, D, seed=0)
    mask, density = sliding_mask_with_density(N, D, window_size=64)

    # Reference: dense SDPA + float bias. BQ/BK from steel block config.
    BQ, BK = _steel_block_config(D)
    NQ = (N + BQ - 1) // BQ
    NK = (N + BK - 1) // BK
    full_mask = mx.broadcast_to(mask[None, None, :, :], (B, H, NQ, NK))
    expanded = full_mask[:, :, :, None, :, None]
    expanded = mx.broadcast_to(expanded, (B, H, NQ, BQ, NK, BK))
    expanded = expanded.reshape(B, H, NQ*BQ, NK*BK)[:, :, :N, :N]
    neg_inf = mx.array(float("-inf"), dtype=q.dtype)
    zero = mx.array(0.0, dtype=q.dtype)
    bias = mx.where(expanded, zero, neg_inf)
    scale = 1.0 / math.sqrt(D)

    y_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=bias)
    y_mfa = flash_attention_sparse(q, k, v, mask, scale=scale, causal=False)
    mx.async_eval(y_ref, y_mfa); mx.synchronize()

    err = mx.abs(y_ref.astype(mx.float32) - y_mfa.astype(mx.float32))
    rmse = float(mx.sqrt(mx.mean(err*err)))
    mag = float(mx.max(mx.abs(y_ref.astype(mx.float32))))
    rel = rmse / mag if mag > 0 else 0.0
    nan = int(mx.sum(mx.isnan(y_mfa.astype(mx.float32))))
    inf = int(mx.sum(mx.isinf(y_mfa.astype(mx.float32))))
    passed = rel < SMOKE_REL_BAR and nan == 0 and inf == 0
    return passed, {"rel_err": rel, "rmse": rmse, "mag": mag, "n_nan": nan,
                    "n_inf": inf, "density": density,
                    "shape": f"B={B} H={H} N={N} D={D}",
                    "bar": SMOKE_REL_BAR, "passed": passed}


def time_call(func, n_runs):
    # Warmup
    y = func(); mx.async_eval(y); mx.synchronize()
    y = func(); mx.async_eval(y); mx.synchronize()
    times = []
    for _ in range(n_runs):
        mx.synchronize()
        t0 = time.perf_counter()
        y = func()
        mx.async_eval(y); mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times


def run_shape(label, B, H, N, D, sparsity_label, window_size, *, n_runs):
    q, k, v = make_inputs(B, H, N, D, seed=42)
    mask, density = sliding_mask_with_density(N, D, window_size)
    BQ, BK = _steel_block_config(D)
    NQ = (N + BQ - 1) // BQ
    NK = (N + BK - 1) // BK
    scale = 1.0 / math.sqrt(D)

    # Build the equivalent float bias for SDPA comparison.
    full_mask = mx.broadcast_to(mask[None, None, :, :], (B, H, NQ, NK))
    expanded = full_mask[:, :, :, None, :, None]
    expanded = mx.broadcast_to(expanded, (B, H, NQ, BQ, NK, BK))
    expanded = expanded.reshape(B, H, NQ*BQ, NK*BK)[:, :, :N, :N]
    neg_inf = mx.array(float("-inf"), dtype=q.dtype)
    zero = mx.array(0.0, dtype=q.dtype)
    bias = mx.where(expanded, zero, neg_inf)
    mx.async_eval(bias); mx.synchronize()

    def call_mfa():
        return flash_attention_sparse(q, k, v, mask, scale=scale, causal=False)
    def call_sdpa():
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=bias)

    # A/B/A pattern (Sprint C precedent)
    mfa_a = time_call(call_mfa, n_runs)
    sdpa  = time_call(call_sdpa, n_runs)
    mfa_b = time_call(call_mfa, n_runs)

    mfa_all = sorted(mfa_a + mfa_b)
    mfa_med = statistics.median(mfa_all)
    sdpa_med = statistics.median(sdpa)
    aba_drift = (
        abs(statistics.median(mfa_a) - statistics.median(mfa_b))
        / statistics.median(mfa_a) * 100 if mfa_a else 0
    )

    # Dense FLOPs (full attention)
    flops_dense = 2.0 * B * H * N * N * D
    # Effective sparse FLOPs (just the unmasked Q-K dot products)
    flops_sparse = flops_dense * density

    return {
        "shape": label,
        "B": B, "H": H, "N": N, "D": D,
        "sparsity_label": sparsity_label,
        "window_size_tokens": window_size,
        "BQ": BQ, "BK": BK, "NQ": NQ, "NK": NK,
        "density": density,
        "flops_dense": flops_dense,
        "flops_sparse_effective": flops_sparse,
        "mfa_times_a_ms": mfa_a,
        "mfa_times_b_ms": mfa_b,
        "sdpa_times_ms": sdpa,
        "mfa_median_ms": mfa_med,
        "sdpa_median_ms": sdpa_med,
        "aba_drift_pct": aba_drift,
        "mfa_TFLOPS_dense": flops_dense / (mfa_med * 1e-3) / 1e12,
        "mfa_TFLOPS_sparse_effective": flops_sparse / (mfa_med * 1e-3) / 1e12,
        "sdpa_TFLOPS_dense": flops_dense / (sdpa_med * 1e-3) / 1e12,
        "mfa_vs_sdpa_ratio": sdpa_med / mfa_med if mfa_med > 0 else 0,
    }


def capture_conditions():
    out = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "platform": platform.platform()}
    for n, c in [("sw_vers", ["sw_vers"]), ("uptime", ["uptime"]),
                 ("uname", ["uname", "-a"])]:
        try:
            out[n] = subprocess.run(c, check=True, capture_output=True,
                                    text=True, timeout=5).stdout.strip()
        except Exception as e:
            out[f"{n}_error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_label")
    ap.add_argument("--data_path",
                    default="docs/lcsa-nax/lcsa-nax-phase0-baseline-data.json")
    ap.add_argument("--cooldown_shape", type=float, default=60.0)
    ap.add_argument("--cooldown_initial", type=float, default=180.0)
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--skip_initial_cooldown", action="store_true")
    args = ap.parse_args()

    print(f"[lcsa-nax phase 0] session={args.session_label}")

    # Smoke gate
    print("[lcsa-nax phase 0] correctness smoke gate...")
    passed, diag = smoke_gate()
    print(f"  smoke: rel_err={diag['rel_err']:.4e}  density={diag['density']:.3f}  "
          f"NaN={diag['n_nan']}  -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("[lcsa-nax phase 0] STATUS: SMOKE_FAILED", file=sys.stderr)
        print(f"  diag: {json.dumps(diag, indent=2)}", file=sys.stderr)
        sys.exit(2)

    if not args.skip_initial_cooldown:
        print(f"[lcsa-nax phase 0] initial cooldown {args.cooldown_initial}s")
        time.sleep(args.cooldown_initial)

    record = {
        "session_label": args.session_label,
        "phase": "B-0 LCSA baseline (mlx-mfa flash_attention_sparse vs MLX SDPA+float-bias)",
        "cooldowns": {"shape_s": args.cooldown_shape,
                      "initial_s": args.cooldown_initial,
                      "skip_initial": args.skip_initial_cooldown},
        "n_runs": args.n_runs,
        "smoke_gate": diag,
        "conditions": capture_conditions(),
        "results": [],
    }

    for spec in SHAPES:
        label = spec[0]
        try:
            res = run_shape(*spec, n_runs=args.n_runs)
        except Exception as e:
            res = {"shape": label, "error": str(e)[:300]}
        record["results"].append(res)
        if "error" in res:
            print(f"  {label:<32} ERROR: {res['error'][:80]}")
        else:
            print(f"  {label:<32} N={res['N']:>5} H={res['H']:>2} "
                  f"density={res['density']:>5.2f}  "
                  f"MFA={res['mfa_median_ms']:>7.2f}ms  "
                  f"SDPA={res['sdpa_median_ms']:>7.2f}ms  "
                  f"ratio={res['mfa_vs_sdpa_ratio']:>4.2f}× "
                  f"drift={res['aba_drift_pct']:>4.1f}%")
        time.sleep(args.cooldown_shape)

    p = Path(args.data_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(p.read_text()) if p.exists() else []
    existing.append(record)
    p.write_text(json.dumps(existing, indent=2))
    print(f"\n[lcsa-nax phase 0] session '{args.session_label}' → {p}")


if __name__ == "__main__":
    main()
