#!/usr/bin/env python3
"""Orchestrate the sparse BT32 null floor and routing map in fresh processes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
from pathlib import Path


PYTHON = "/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python"
BENCH = Path(__file__).with_name("bench_sparse_gate_remap.py")
ROOT = BENCH.parent.parent


def cell_id(cell):
    pattern = (
        f"sw{cell['window']}" if cell["mask_kind"] == "sliding"
        else f"rnd{cell['density']:.2f}"
    )
    suffix = "c" if cell.get("causal") else "nc"
    return (
        f"{pattern}_{suffix}_{cell['dtype']}_b{cell['B']}_h{cell['H']}_"
        f"n{cell['N']}_d{cell['D']}"
    )


def command(cell, arm, output):
    cmd = [
        PYTHON,
        str(BENCH),
        "--arm", arm,
        "--mask-kind", cell["mask_kind"],
        "--B", str(cell["B"]),
        "--H", str(cell["H"]),
        "--N", str(cell["N"]),
        "--D", str(cell["D"]),
        "--dtype", cell["dtype"],
        "--seed", str(cell.get("seed", 20260713)),
        "--output", str(output),
    ]
    if cell["mask_kind"] == "sliding":
        cmd.extend(["--window", str(cell["window"])])
    else:
        cmd.extend(["--density", str(cell["density"])])
    if cell.get("causal"):
        cmd.append("--causal")
    return cmd


def run_arm(cell, arm, scratch, tag):
    output = scratch / f"{cell_id(cell)}_{tag}_{arm}.json"
    subprocess.run(command(cell, arm, output), cwd=ROOT, check=True)
    return json.loads(output.read_text())


def aggregate_cell(cell, scratch, null=False):
    order_pairs = []
    sequences = (("public", "sdpa"), ("sdpa", "public"))
    if null:
        sequences = (("public", "public"), ("public", "public"))
    artifacts = []
    for order_index, sequence in enumerate(sequences, 1):
        order_rows = []
        for position, arm in enumerate(sequence, 1):
            row = run_arm(
                cell, arm, scratch, f"order{order_index}_pos{position}"
            )
            row["order"] = order_index
            row["position"] = position
            order_rows.append(row)
            artifacts.append(row)
        first = order_rows[0]["row"]["timing"]["median_ms"]
        second = order_rows[1]["row"]["timing"]["median_ms"]
        if null:
            ratio = first / second
        else:
            public_ms = next(
                r["row"]["timing"]["median_ms"]
                for r in order_rows if r["row"]["arm"] == "public"
            )
            sdpa_ms = next(
                r["row"]["timing"]["median_ms"]
                for r in order_rows if r["row"]["arm"] == "sdpa"
            )
            ratio = sdpa_ms / public_ms
        order_pairs.append({"order": order_index, "ratio": ratio})
    ratios = [p["ratio"] for p in order_pairs]
    return {
        "id": cell_id(cell),
        "requested": cell,
        "orders": order_pairs,
        "ratio_median": statistics.median(ratios),
        "artifacts": artifacts,
    }


def null_cells():
    base = {
        "mask_kind": "sliding", "window": 128, "density": 0.15,
        "causal": False, "B": 1, "H": 1, "D": 128, "dtype": "fp16",
    }
    return [{**base, "N": n} for n in (2048, 8192)]


def map_cells():
    cells = []
    for mask_kind in ("sliding", "random"):
        values = (128, 256, 512) if mask_kind == "sliding" else (0.05, 0.15, 0.30)
        for value in values:
            for h in (1, 4, 12):
                for n in (2048, 4096, 8192):
                    for d in (64, 128):
                        cells.append({
                            "mask_kind": mask_kind,
                            "window": int(value) if mask_kind == "sliding" else 128,
                            "density": float(value) if mask_kind == "random" else 0.15,
                            "causal": False, "B": 1, "H": h, "N": n, "D": d,
                            "dtype": "fp16",
                        })

    # B*H=12 controls separate the head-count axis from aggregate load.
    for mask_kind, value in (("sliding", 128), ("random", 0.15)):
        for b, h in ((12, 1), (3, 4)):
            cells.append({
                "mask_kind": mask_kind,
                "window": int(value) if mask_kind == "sliding" else 128,
                "density": float(value) if mask_kind == "random" else 0.15,
                "causal": False, "B": b, "H": h, "N": 4096, "D": 128,
                "dtype": "fp16",
            })

    # Four representative causal-sparse cells spanning the measured gate.
    cells.extend([
        {"mask_kind": "random", "window": 128, "density": 0.10,
         "causal": True, "B": 1, "H": 1, "N": 2048, "D": 64, "dtype": "fp16"},
        {"mask_kind": "random", "window": 128, "density": 0.10,
         "causal": True, "B": 1, "H": 4, "N": 4096, "D": 128, "dtype": "fp16"},
        {"mask_kind": "random", "window": 128, "density": 0.30,
         "causal": True, "B": 1, "H": 12, "N": 8192, "D": 64, "dtype": "fp16"},
        {"mask_kind": "random", "window": 128, "density": 0.30,
         "causal": True, "B": 1, "H": 12, "N": 4096, "D": 128, "dtype": "fp16"},
    ])

    # Key bf16 sentinels: the discovered H1 loss, the historical H12 region,
    # random sparse, and causal sparse at both supported head dimensions.
    for template in (
        ("sliding", 128, 0.15, False, 1, 4096, 128),
        ("sliding", 128, 0.15, False, 12, 4096, 128),
        ("random", 128, 0.15, False, 1, 4096, 128),
        ("random", 128, 0.15, False, 12, 4096, 128),
        ("random", 128, 0.10, True, 4, 4096, 64),
        ("random", 128, 0.10, True, 4, 4096, 128),
    ):
        kind, window, density, causal, h, n, d = template
        cells.append({
            "mask_kind": kind, "window": window, "density": density,
            "causal": causal, "B": 1, "H": h, "N": n, "D": d,
            "dtype": "bf16",
        })
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("null", "map"), required=True)
    parser.add_argument("--floor", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "map" and args.floor is None:
        raise ValueError("--floor from Phase 0 is required for the map")

    cells = null_cells() if args.phase == "null" else map_cells()
    results = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("phase") == args.phase:
            results = list(previous.get("results", []))

    def classify_orders(result):
        classes = []
        for order in result["orders"]:
            ratio = order["ratio"]
            if ratio > 1.0 + args.floor:
                classes.append("WIN")
            elif ratio < 1.0 - args.floor:
                classes.append("LOSS")
            else:
                classes.append("NOISE")
        return classes[0] if len(set(classes)) == 1 else "NOISE"

    if args.phase == "map":
        for result in results:
            result["verdict"] = classify_orders(result)
    completed = {result["id"] for result in results}

    def write_checkpoint():
        payload = {
            "schema": f"mlx-mfa.sparse-gate-remap.{args.phase}.v1",
            "phase": args.phase,
            "ratio_direction": (
                "A_median_ms / A_median_ms" if args.phase == "null"
                else "sdpa_median_ms / public_v6nax_sparse_median_ms (>1 means sparse wins)"
            ),
            "decision_floor": args.floor,
            "method": {
                "process": "fresh process per arm/order",
                "sessions_per_process": 5,
                "dispatches_per_sample": 20,
                "orders": 2,
            },
            "results": results,
        }
        if args.phase == "null":
            deviations = [
                abs(order["ratio"] - 1.0)
                for result in results for order in result["orders"]
            ]
            payload["null_deviations"] = deviations
            payload["recommended_floor"] = max(deviations) if deviations else None
            payload["floor_rule"] = (
                "conservative max abs(A/A-1) across both sentinels and both orders"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    with tempfile.TemporaryDirectory(prefix="mlx_mfa_sparse_gate_") as tmp:
        scratch = Path(tmp)
        for index, cell in enumerate(cells, 1):
            if cell_id(cell) in completed:
                print(f"[{index}/{len(cells)}] SKIP {cell_id(cell)}", flush=True)
                continue
            print(f"[{index}/{len(cells)}] {cell_id(cell)}", flush=True)
            result = aggregate_cell(cell, scratch, null=args.phase == "null")
            if args.phase == "map":
                result["verdict"] = classify_orders(result)
            results.append(result)
            write_checkpoint()

    payload = write_checkpoint()
    print(json.dumps({
        "output": str(args.output),
        "cells": len(results),
        "recommended_floor": payload.get("recommended_floor"),
    }, indent=2))


if __name__ == "__main__":
    main()
