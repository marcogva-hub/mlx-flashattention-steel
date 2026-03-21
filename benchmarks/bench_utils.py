"""Shared benchmark utilities — timing, geomean, env var management."""
import math
import os
import time
from contextlib import contextmanager

import mlx.core as mx


def med(fn, warmup=5, iters=20):
    """Median timing of `fn()` in milliseconds (warmup + measured iterations)."""
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def geomean(values):
    """Geometric mean of a list of positive floats."""
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(v, 1e-9)) for v in values) / len(values))


@contextmanager
def env_override(var, val):
    """Context manager: set env var, yield, then restore original state.

    Handles None (unset), empty string, and non-None values correctly.
    """
    prev = os.environ.get(var)
    if val is None:
        os.environ.pop(var, None)
    else:
        os.environ[var] = str(val)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = prev


def check_mfa_available():
    """Exit with error if MFA extension is not built."""
    try:
        from mlx_mfa import is_mfa_available
        if not is_mfa_available():
            print("SPEEDUP_RATIO: 0.000000")
            raise SystemExit(1)
    except ImportError:
        print("mlx_mfa not installed")
        raise SystemExit(1)
