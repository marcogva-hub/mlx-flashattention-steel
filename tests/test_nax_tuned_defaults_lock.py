"""NAX tuned-default regression lock (research/nax-autotune-m5, M5 Max).

The autoresearch (Phase 0 knob-map + Phase 1 sweep over the only LIVE tile knobs)
found, for the production-dispatched V6 NAX forward (`v6_nax_forward`, force_v6nax):
  - D=128: default BQ=64/BK=32/WM=4 is already optimal (no robust win — the nearest
    alt 32/32/2 regressed +32% at N=2048).
  - D=64:  BK=32 robustly beats the old inherited BK=64 (−2..−15% across 6 shapes ×
    {fp16,bf16}; N=8192 fp16 2.97→2.50ms, ~parity with SDPA vs ~1.21× slower before).

Regression guard = the TUNED TILE CONFIG (deterministic — a revert of the BK default
changes the kernel's baked tile, caught via the source-dump fingerprint), NOT a tight
absolute-ms threshold (timing is CI-flaky — the campaign principle: perf is
re-measure-not-locked). A generous ms ceiling guards only catastrophic (>2×) regressions.
"""
from __future__ import annotations
import os, sys, subprocess, math, time, re
import numpy as np
import mlx.core as mx
import pytest

try:
    from mlx_mfa import get_device_info
    _IS_M5 = bool(get_device_info().get("is_m5_plus"))
except Exception:
    _IS_M5 = False
try:
    from mlx_mfa._ext import v6_nax_forward
    _HAVE = True
except Exception:
    _HAVE = False

pytestmark = pytest.mark.skipif(not (_HAVE and _IS_M5),
                                reason="V6 NAX tuned-default lock: M5+ NAX + extension required")

# (D → expected tuned tile the autoresearch selected). BK=32 for both D.
_EXPECTED_TILE = {64: (32, 32, 2), 128: (64, 32, 4)}   # (BQ, BK, WM)

_DUMP_SCRIPT = """
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward
D = {D}
q = (mx.random.uniform(-1, 1, (1, 8, 8192, D)) * 0.1).astype(mx.float16)
mx.eval(q)
O, _ = v6_nax_forward(q, q, q, False, True)
mx.eval(O)
"""


@pytest.mark.parametrize("D", [64, 128])
def test_nax_default_tile_is_tuned(D):
    """Deterministic config lock: the dispatched D=64/128 NAX forward compiles with the
    autoresearch-tuned tile. A revert (e.g. D=64 BK back to 64) changes the dump → FAILS."""
    env = dict(os.environ, MFA_V6_DUMP_SOURCE="1")
    # ensure no stray tile override leaks from the environment
    for k in ("MFA_V6_NAX_BQ", "MFA_V6_NAX_BK", "MFA_V6_NAX_WM"):
        env.pop(k, None)
    r = subprocess.run([sys.executable, "-c", _DUMP_SCRIPT.format(D=D)],
                       env=env, capture_output=True, text=True)
    m = re.search(r"V6NAX source for BQ=(\d+) BK=(\d+) BD=(\d+) WM=(\d+)", r.stderr)
    assert m, f"could not read the V6NAX tile dump for D={D}:\n{r.stderr[-600:]}"
    bq, bk, wm = int(m.group(1)), int(m.group(2)), int(m.group(4))
    assert (bq, bk, wm) == _EXPECTED_TILE[D], (
        f"D={D} NAX tile drifted from the tuned default {_EXPECTED_TILE[D]} to "
        f"(BQ={bq}, BK={bk}, WM={wm}) — autoresearch regression (esp. D=64 BK must stay 32)")


def test_nax_d64_not_catastrophically_slow():
    """Generous catastrophic-regression sanity (NOT a tight perf lock — timing is CI-flaky):
    D=64 N=8192 fp16 NAX forward must stay well under 2× SDPA. Tuned ~parity (~2.5 vs ~2.45ms);
    this only trips on a gross (>2×) regression."""
    D, N = 64, 8192
    mx.random.seed(7)
    q = (mx.random.uniform(-1, 1, (1, 8, N, D)) * 0.1).astype(mx.float16)
    mx.eval(q)
    sc = 1.0 / math.sqrt(D)

    def _b(fn, w=8, it=20):
        for _ in range(w):
            mx.eval(fn())
        mx.synchronize(); ts = []
        for _ in range(it):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[len(ts) // 2] * 1e3

    t_nax = _b(lambda: v6_nax_forward(q, q, q, False, True)[0])
    t_sdpa = _b(lambda: mx.fast.scaled_dot_product_attention(q, q, q, scale=sc))
    assert t_nax < 2.0 * t_sdpa, (
        f"D=64 NAX forward catastrophically slow: {t_nax:.2f}ms vs SDPA {t_sdpa:.2f}ms "
        f"(>2× — tile regression? expected ~parity at the tuned BK=32)")
