#!/usr/bin/env python3
"""V32 niche-shape kernel sweep — empirically determine MFA vs SDPA winner per shape.

Usage:
    .venv/bin/python bench/v32_kernel_sweep.py --output docs/v6-nax/v32-kernel-sweep.json

Procedure:
- For each niche shape, run 3 backends (sdpa, mfa, auto) in subprocess isolation.
- 5 runs per (shape, backend) per CLAUDE_V6_NAX.md §3 multi-run requirement.
- Initial 3-min cooldown, 30s inter-config, 60s inter-shape.
- Total: ~16 shapes × 3 backends = 48 subprocesses, ~30-90s each → 30-60 min wall clock.

The MFA backend lets the internal mlx-mfa dispatch pick the best sub-kernel
(V2/V3/V4/V5/V6/V6NAX) for each shape. Sprint A's question is "does MFA-best
beat SDPA on this shape?" — we don't need to force individual sub-kernels.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INNER = REPO_ROOT / "bench/v32_kernel_sweep_inner.py"


# Shape registry (mirrored in inner script for subprocess isolation)
NICHE_SHAPES = [
    # Niche: SDPA doesn't optimize these on M5+ NAX
    "whisper-base",        # D=80, non-canonical
    "gpt-neo-d96",         # D=96, non-canonical
    "codestral-d192",      # D=192, non-canonical
    "custom-d256",         # D=256, V2 D-split territory
    "llama-decode-8k",     # qL=1, sdpa_vector path (not NAX)
    "llama-decode-32k",    # qL=1, very long kL
    "flashvsr-dense",      # D=64 small-N self-attn (Sprint 4 canary)
    "llama-prefill-2k",    # D=128 causal short
    "llama-prefill-4k",    # D=128 causal medium
    "llama-prefill-8k",    # D=128 causal long
    "ltx2-cross",          # D=64 asymmetric (V6NAX winner)
    "seedvr2-small",       # D=128 large self-attn (Phase 0 V6NAX regression)
    "cogvideox",           # D=128 very large self-attn
    # Control: canonical shapes where SDPA NAX should clearly win
    "canonical-d128-4k",
    "canonical-d64-8k",
]
BACKENDS = ["sdpa", "mfa", "auto"]


# Reset env vars for deterministic dispatch (mirrors v6nax_bench.py pattern)
RESET_ENVS = (
    "MFA_V6_USE_NAX", "MFA_V6_NAX_BQ", "MFA_V6_NAX_BK", "MFA_V6_NAX_WM",
    "MFA_V6_NAX_SINGLE_OTILE", "MFA_V6_BYPASS_TGP", "MFA_V6_USE_V33",
    "MFA_FORCE_GEN", "MFA_V6_BLOCK_R", "MFA_V6_BLOCK_C",
    "MFA_V6_EXEC_SG", "MFA_V6_BLOCK_D", "MFA_V6_BNHD_LEGACY",
    "MFA_V6_MAX_THREADS", "MFA_V6_MATMUL_EXEC_SG",
    "MFA_DISABLE_V2", "MFA_FORCE_V2", "MFA_DISABLE_V3", "MFA_FORCE_SPLITK",
    "MFA_ENABLE_V4", "MFA_ENABLE_V5",
)


def run_subprocess_bench(shape, backend, runs):
    env = {k: v for k, v in os.environ.items() if k not in RESET_ENVS}
    cmd = [sys.executable, str(INNER), "--shape", shape, "--backend", backend, "--runs", str(runs)]
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"shape": shape, "backend": backend, "error": "timeout"}
    if result.returncode != 0:
        return {"shape": shape, "backend": backend, "error": result.stderr.strip()[:500]}
    # Parse the LAST JSON-looking line of stdout
    stdout = result.stdout.strip()
    if not stdout:
        return {"shape": shape, "backend": backend, "error": "no stdout"}
    last_line = stdout.splitlines()[-1]
    try:
        return json.loads(last_line)
    except json.JSONDecodeError as e:
        return {"shape": shape, "backend": backend, "error": f"parse: {e}", "stdout_tail": last_line[:500]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--shapes", default=",".join(NICHE_SHAPES))
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--initial-cooldown", type=int, default=180,
                    help="Seconds before first bench (thermal stability).")
    ap.add_argument("--inter-config", type=int, default=30,
                    help="Seconds between (shape, backend) pairs.")
    ap.add_argument("--inter-shape", type=int, default=60,
                    help="Seconds between shapes (after all backends).")
    args = ap.parse_args()

    shapes = args.shapes.split(",")
    backends = args.backends.split(",")

    print(f"=== V32 kernel sweep ===", file=sys.stderr)
    print(f"Shapes:   {len(shapes)} ({', '.join(shapes)})", file=sys.stderr)
    print(f"Backends: {len(backends)} ({', '.join(backends)})", file=sys.stderr)
    print(f"Configs:  {len(shapes) * len(backends)}", file=sys.stderr)
    print(f"Runs/cfg: {args.runs}", file=sys.stderr)
    print(f"Initial cooldown {args.initial_cooldown}s, inter-config {args.inter_config}s, "
          f"inter-shape {args.inter_shape}s", file=sys.stderr)

    if args.initial_cooldown > 0:
        print(f"\n[sweep] Initial cooldown {args.initial_cooldown}s ...", file=sys.stderr)
        time.sleep(args.initial_cooldown)

    results = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for s_idx, shape in enumerate(shapes):
        for b_idx, backend in enumerate(backends):
            print(f"\n[sweep] [{s_idx+1}/{len(shapes)}] {shape} × {backend}", file=sys.stderr)
            r = run_subprocess_bench(shape, backend, args.runs)
            results.append(r)

            # Print per-config result
            if "error" in r:
                print(f"  ERROR: {r['error']}", file=sys.stderr)
            elif r.get("supported") is False:
                print(f"  SKIPPED: {r.get('skipped', 'unsupported')}", file=sys.stderr)
            else:
                med = r.get("median_ms", 0)
                rmse = r.get("rmse")
                rmse_str = f"rmse={rmse:.2e}" if rmse else "rmse=n/a"
                print(f"  {backend:<7}: median={med:8.2f}ms  {rmse_str}", file=sys.stderr)

            # Save incremental progress (in case of crash mid-sweep)
            out_path.write_text(json.dumps({"records": results}, indent=2))

            if args.inter_config > 0 and not (s_idx == len(shapes) - 1 and b_idx == len(backends) - 1):
                time.sleep(args.inter_config)

        if args.inter_shape > 0 and s_idx < len(shapes) - 1:
            print(f"[sweep] Inter-shape cooldown {args.inter_shape}s ...", file=sys.stderr)
            time.sleep(args.inter_shape)

    print(f"\n[sweep] Done. Wrote {len(results)} records to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
