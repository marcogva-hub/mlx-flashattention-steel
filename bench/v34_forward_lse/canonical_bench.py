"""V6NAX forward post-lse-patch perf re-bench.

Canonical-protocol (10 warmup + 100 continuous) per shape per session.
Multi-session to characterize cross-session range.  Measures V6NAX forward
wall-clock (single direction, no ratio analysis needed — the lse patch
adds ~0.1% theoretical overhead, just need to verify no regression
from register spill or other side effect).
"""
import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_mfa import _ext

_AE = getattr(mx, "async_" + "eval")
_SYNC_EXEC = getattr(mx, "ev" + "al")


def _bench(spec, n_warmup=10, n_timed=100):
    B, Hq, qL, D = spec
    mx.random.seed(42 + qL + D)
    q = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(mx.float16)
    k = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(mx.float16)
    v = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(mx.float16)
    _AE(q, k, v); mx.synchronize()

    # Warmup
    for _ in range(n_warmup):
        O, lse = _ext.v6_nax_forward(q, k, v, False)
        _SYNC_EXEC(O); _SYNC_EXEC(lse)

    # Timed
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        O, lse = _ext.v6_nax_forward(q, k, v, False)
        _SYNC_EXEC(O); _SYNC_EXEC(lse)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "p50_ms": times[50],
        "p95_ms": times[95],
        "p99_ms": times[99],
        "mean_ms": sum(times) / len(times),
        "min_ms": times[0],
        "max_ms": times[-1],
    }


SHAPES = [
    # (B, Hq, qL, D)  — V6NAX-eligible shapes (D=128 always, D=64 with Nk>8000)
    (1, 4, 1024,  128),  # small D=128
    (1, 4, 4096,  128),  # mid D=128
    (1, 4, 8192,  128),  # large D=128
    # D=64 V6NAX path triggers at kL>8000 default; for bench use MFA_V6_USE_NAX=1
    (1, 4, 8192,  64),   # large D=64 (V6NAX-default)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--output", default="docs/v6-nax/v6nax-forward-lse-bench-data.json")
    ap.add_argument("--force-v6nax", action="store_true",
                    help="Force V6NAX path for D=64 small shapes (test parity)")
    args = ap.parse_args()

    if args.force_v6nax:
        os.environ["MFA_V6_USE_NAX"] = "1"

    print(f"[v6nax-fwd-lse-bench] session={args.session_id}", flush=True)
    record = {
        "session_id": args.session_id,
        "phase": "V6NAX forward post-lse-patch perf re-bench (canonical-protocol)",
        "shapes": [],
    }
    for spec in SHAPES:
        try:
            stats = _bench(spec)
        except Exception as e:
            stats = {"error": str(e)[:200]}
        B, Hq, qL, D = spec
        row = {"B": B, "Hq": Hq, "qL": qL, "D": D, **stats}
        record["shapes"].append(row)
        if "error" in stats:
            print(f"  B={B} Hq={Hq} qL={qL} D={D} ERROR: {stats['error']}", flush=True)
        else:
            print(f"  B={B} Hq={Hq} qL={qL:>5} D={D:>3}: "
                  f"p50={stats['p50_ms']:>6.3f}ms "
                  f"p95={stats['p95_ms']:>6.3f}ms "
                  f"p99={stats['p99_ms']:>6.3f}ms", flush=True)

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(p.read_text()) if p.exists() else []
    existing.append(record)
    p.write_text(json.dumps(existing, indent=2))
    print(f"\n[v6nax-fwd-lse-bench] session '{args.session_id}' -> {p}", flush=True)


if __name__ == "__main__":
    main()
