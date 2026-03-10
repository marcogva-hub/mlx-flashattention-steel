"""mlx_mfa CLI entry-point.

Usage::

    python -m mlx_mfa info           # device info + current config
    python -m mlx_mfa calibrate      # full calibration (default)
    python -m mlx_mfa calibrate --quick  # faster (5 iters, 2 warmup)
"""

from __future__ import annotations

import argparse
import sys


def _cmd_info() -> None:
    """Print device info and current kernel config."""
    import mlx_mfa
    info = mlx_mfa.get_device_info()
    cfg  = mlx_mfa.get_supported_configs()
    print(f"mlx-mfa {mlx_mfa.__version__}")
    print(f"Device : {info['device_name']}")
    print(f"Gen    : {info['gpu_family_gen']}  M3+={info['is_m3_plus']}")
    print(f"Dtypes : {cfg['dtypes']}")
    print(f"D vals : {cfg['head_dims']}")
    import os
    bk = os.environ.get("MFA_V2_FORCE_BK", "auto (gen-based)")
    print(f"V2 BK  : {bk}")


def _cmd_calibrate(quick: bool) -> None:
    """Run dispatch + kernel-config calibration."""
    from mlx_mfa.dispatch_policy import calibrate_dispatch
    if quick:
        calibrate_dispatch(warmup=2, n_iters=5, calibrate_kernel_configs=True)
    else:
        calibrate_dispatch(warmup=5, n_iters=20, calibrate_kernel_configs=True)
    print("\nRestart your Python process to pick up the calibrated config.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mlx_mfa",
                                     description="mlx-mfa device utilities")
    sub = parser.add_subparsers(dest="cmd", metavar="command")

    sub.add_parser("info", help="Show device info and current kernel config")

    cal_p = sub.add_parser("calibrate", help="Calibrate dispatch thresholds + BK")
    cal_p.add_argument("--quick", action="store_true",
                       help="Faster calibration (2 warmup / 5 iters)")

    args = parser.parse_args(argv)

    if args.cmd == "info":
        _cmd_info()
    elif args.cmd == "calibrate":
        _cmd_calibrate(args.quick)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
