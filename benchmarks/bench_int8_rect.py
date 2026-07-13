#!/usr/bin/env python3
"""One-process Stage 1 int8-MPP rectangular probe.

The C++ probe performs GPU-timestamped samples for both dtypes.  This wrapper
exists so the campaign can run fresh processes in both arm orders without
silently sharing JIT state between sessions.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import time
from pathlib import Path

from mlx_mfa import _ext


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("fp16-first", "int8-first"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = _ext.mpp_int8_rect_microbench(args.order == "int8-first")
    cells = []
    for line in raw.splitlines():
        match = re.match(
            r"(?P<label>\w+) M=(?P<M>\d+) N=(?P<N>\d+) K=(?P<K>\d+) "
            r"fp16_tflops=(?P<fp16>[-+0-9.eE]+) int8_tops=(?P<int8>[-+0-9.eE]+) "
            r"ratio=(?P<ratio>[-+0-9.eE]+)",
            line,
        )
        if match:
            values = match.groupdict()
            cells.append({
                "label": values["label"],
                "M": int(values["M"]),
                "N": int(values["N"]),
                "K": int(values["K"]),
                "fp16_tflops": float(values["fp16"]),
                "int8_tops": float(values["int8"]),
                "ratio_int8_over_fp16": float(values["ratio"]),
            })
    if len(cells) != 2:
        raise RuntimeError(f"unexpected probe output: {raw!r}")
    result = {
        "stage": "ITEM6-stage1-int8-rect",
        "order": args.order,
        "raw": raw,
        "cells": cells,
        "engagement": {
            "probe_symbol": "_ext.mpp_int8_rect_microbench",
            "fp16_arm": "raw MPP matmul2d device tensor_inline with half operands",
            "int8_arm": "raw MPP matmul2d device tensor_inline with int8_t operands and int32 destination",
            "output_sanity": "C buffer was non-zero after each GPU dispatch",
        },
        "stamp": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_mfa": importlib.metadata.version("mlx-mfa"),
            "git_head": os.popen("git rev-parse HEAD").read().strip(),
            "device_info": dict(_ext.get_device_info()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
