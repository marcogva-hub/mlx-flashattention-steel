#!/usr/bin/env python3
"""Conv NAX Phase 0 — Steel-legacy baseline benchmark.

Times MLX 0.31.2's mx.conv_general path (which on M5+ runs through Steel
legacy `implicit_gemm_conv_*` kernels — no NAX, see survey §2-§3). Output
establishes the baseline that Sprint C's NAX-routed conv would need to beat.

Methodology mirrors Sprint A Phase 1.5 v2 (§4-compliant cooldowns):
  --cooldown_round 90  (between within-shape rounds; not used here, single
                        round per shape)
  --cooldown_shape 60  (between shapes)
  --cooldown_initial 180 (warmup-and-thermal-settle before any timing)

Shapes drawn from SeedVR2 VAE decoder profiling (Marco's phase0 work);
top contributors to wall-clock per `architecture_map.json` op_type_breakdown:
Conv3d_3x3x3 = 91.94% FLOPs, Conv3d_1x1x1 = 7.23% FLOPs.
"""
import argparse, json, math, os, platform, statistics, subprocess
import sys, time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

# ────────────────────────────────────────────────────────────────────────
# Shape inventory — top Conv3D shapes from SeedVR2 VAE decoder profiling.
# Format: (label, N, C_in, T, H, W, C_out, KT, KH, KW, stride_t, stride_h, stride_w)
# Based on architecture_map.json + profiling_baseline.json from
# /Users/marcomarcelino/code/SeedVR2_VAE_Flash-VAED/results/phase0/.
# ────────────────────────────────────────────────────────────────────────
SHAPES = [
    # Most-impactful Conv3D shapes by FLOPs × call frequency.
    # up_blocks.2 resnet trio: dominant up-spatial-2× stage.
    ("up2_resnet_256to256_T17_HW256_k3", 1, 256, 17, 256, 256, 256, 3, 3, 3, 1, 1, 1),
    # up_blocks.3 resnet trio: large spatial, smaller channels.
    ("up3_resnet_128to128_T17_HW512_k3", 1, 128, 17, 512, 512, 128, 3, 3, 3, 1, 1, 1),
    # up_blocks.3 resnet.0: 256→128 (channel reduction).
    ("up3_resnet0_256to128_T17_HW512_k3", 1, 256, 17, 512, 512, 128, 3, 3, 3, 1, 1, 1),
    # up_blocks.1 resnet trio: 512→512 at intermediate spatial.
    ("up1_resnet_512to512_T9_HW128_k3", 1, 512, 9, 128, 128, 512, 3, 3, 3, 1, 1, 1),
    # mid_block / up_blocks.0 resnet: 512→512 at smallest spatial.
    ("mid_resnet_512to512_T5_HW64_k3", 1, 512, 5, 64, 64, 512, 3, 3, 3, 1, 1, 1),
    # up_blocks.2 resnet.0: 512→256 (channel reduction).
    ("up2_resnet0_512to256_T17_HW256_k3", 1, 512, 17, 256, 256, 256, 3, 3, 3, 1, 1, 1),
]


def capture_conditions():
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_local": datetime.now().isoformat(),
        "platform": platform.platform(),
    }
    for cmd_name, cmd in [("sw_vers", ["sw_vers"]),
                          ("uptime", ["uptime"]),
                          ("uname", ["uname", "-a"])]:
        try:
            out[cmd_name] = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception as e:
            out[f"{cmd_name}_error"] = str(e)
    try:
        bt = subprocess.run(["sysctl", "-n", "kern.boottime"],
                            check=True, capture_output=True, text=True, timeout=5)
        out["kern_boottime_raw"] = bt.stdout.strip()
    except Exception as e:
        out["boottime_error"] = str(e)
    return out


def run_shape(label, N, C_in, T, H, W, C_out, KT, KH, KW, sT, sH, sW,
              *, n_runs, seed):
    """Time mx.conv_general for one Conv3D shape.

    MLX uses NHWC for activations and OHWIO-style (out, H, W, [D], in) for
    weights. We use NDHWC and weight shape (C_out, KT, KH, KW, C_in).
    """
    mx.random.seed(seed)
    # NDHWC layout: (N, T, H, W, C)
    x = mx.random.normal((N, T, H, W, C_in), dtype=mx.float16)
    # Conv3D weight: (C_out, KT, KH, KW, C_in) per MLX's conv_general convention
    w = mx.random.normal((C_out, KT, KH, KW, C_in), dtype=mx.float16)
    mx.eval(x, w)

    # Warmup: 1 forward to compile kernels + populate cache.
    y = mx.conv_general(
        x, w,
        stride=(sT, sH, sW),
        padding=(KT // 2, KH // 2, KW // 2),
        kernel_dilation=(1, 1, 1),
        input_dilation=(1, 1, 1),
        groups=1,
        flip=False,
    )
    mx.eval(y)

    times = []
    for _ in range(n_runs):
        mx.synchronize()
        t0 = time.perf_counter()
        y = mx.conv_general(
            x, w,
            stride=(sT, sH, sW),
            padding=(KT // 2, KH // 2, KW // 2),
            kernel_dilation=(1, 1, 1),
            input_dilation=(1, 1, 1),
            groups=1,
            flip=False,
        )
        mx.eval(y)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return {
        "shape": label,
        "N": N, "C_in": C_in, "T": T, "H": H, "W": W, "C_out": C_out,
        "kernel": [KT, KH, KW], "stride": [sT, sH, sW],
        "times_ms": times,
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "out_shape": list(y.shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_label", help="e.g. v1_S1")
    ap.add_argument("--data_path",
                    default="docs/conv-nax/conv-nax-phase0-baseline-data.json")
    ap.add_argument("--cooldown_shape", type=float, default=60.0,
                    help="Inter-shape cooldown (§4 prescribes 60s)")
    ap.add_argument("--cooldown_initial", type=float, default=180.0,
                    help="Initial warmup cooldown (§4 prescribes 180s)")
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[harness] session={args.session_label}")
    print(f"[harness] cooldowns: shape={args.cooldown_shape}s initial={args.cooldown_initial}s")
    print(f"[harness] initial cooldown {args.cooldown_initial}s")
    time.sleep(args.cooldown_initial)

    record = {
        "session_label": args.session_label,
        "phase": "C-0 (Sprint C Phase 0 baseline)",
        "methodology": "MLX 0.31.2 mx.conv_general — Steel legacy implicit_gemm_conv path (no NAX, see survey §2-§3).",
        "cooldowns": {
            "shape_s": args.cooldown_shape,
            "initial_s": args.cooldown_initial,
            "deviation_from_§4": args.cooldown_shape != 60.0 or args.cooldown_initial != 180.0,
        },
        "n_runs": args.n_runs,
        "conditions": capture_conditions(),
        "results": [],
    }

    for shape_def in SHAPES:
        res = run_shape(*shape_def, n_runs=args.n_runs, seed=args.seed)
        record["results"].append(res)
        n_out = res["out_shape"]
        print(f"  {res['shape']:<45} median={res['median_ms']:7.1f}ms "
              f"out={n_out}  stdev={res['stdev_ms']:.2f}ms")
        time.sleep(args.cooldown_shape)

    data_path = Path(args.data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        existing = json.loads(data_path.read_text())
    else:
        existing = []
    existing.append(record)
    data_path.write_text(json.dumps(existing, indent=2))
    print(f"\n[harness] session '{args.session_label}' appended to {data_path}")


if __name__ == "__main__":
    main()
