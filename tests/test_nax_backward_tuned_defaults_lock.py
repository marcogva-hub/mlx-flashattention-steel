"""NAX BACKWARD tuned-default regression lock (research/nax-backward-autotune-m5, M5 Max).

The backward autoresearch (Phase 0 knob-map → Phase 1 sweep) found, for the default-on D=64
native backward (the dQ split kernel, `MFAV6NAXBwdQuery`):
  - D=64 dQ tile `BK 64→32` is a robust win: −4..−14% on the FULL backward (dQ+dV+dK) across
    6 shapes × {fp16,bf16}, grad-IDENTICAL (BK is perf-only for dQ). N=8192 fp16 causal:
    ~16.3ms → ~14.3ms. This makes dQ D=64 = 32/32/2, exactly the V6NAX forward's tuned config
    (the dQ had inherited the stale pre-tune BK=64).
  - D=128 backward is at the SDPA-vjp architectural floor (the default D=128 backward IS
    SDPA-vjp; the native D=128 bwd is opt-in + measured slower). NOT a tuning target — untouched.

Regression guard = the TUNED dQ TILE CONFIG (deterministic source-dump fingerprint — a revert of
the dQ BK default changes the baked tile) + a gradient-correctness assertion (vs an independent
fp32 vjp oracle, Lesson #11) + a generous catastrophic-ms ceiling (timing is CI-flaky — perf is
re-measure-not-lock; this only trips on a gross >2× regression).
"""
from __future__ import annotations
import os, sys, subprocess, math, time, re
import numpy as np
import mlx.core as mx
import pytest

try:
    from mlx_mfa import get_device_info, flash_attention
    _IS_M5 = bool(get_device_info().get("is_m5_plus"))
except Exception:
    _IS_M5 = False

pytestmark = pytest.mark.skipif(not _IS_M5,
                                reason="NAX backward tuned-default lock: M5+ NAX required")

_DUMP = """
import mlx.core as mx, math
from mlx_mfa import flash_attention
D = {D}
q = (mx.random.uniform(-1, 1, (1, 8, 2048, D)) * 0.1).astype(mx.float16)
mx.eval(q)
g = mx.grad(lambda a: flash_attention(a, q, q, scale=1.0/math.sqrt(D)).sum())(q)
mx.eval(g)
"""


def test_dq_d64_backward_tile_is_tuned():
    """Deterministic config lock: the default-on D=64 dQ backward kernel compiles with the
    autoresearch-tuned BK=32. A revert (BK back to 64) changes the dump → FAILS."""
    env = dict(os.environ, MFA_V6BWD_DUMP_SOURCE="1")
    for k in ("MFA_V6BWD_BQ", "MFA_V6BWD_BK", "MFA_V6BWD_WM"):
        env.pop(k, None)
    r = subprocess.run([sys.executable, "-c", _DUMP.format(D=64)],
                       env=env, capture_output=True, text=True, cwd="/tmp")
    bks = re.findall(r"BK[=\s]+(\d+)", r.stderr)
    assert bks, f"could not read the dQ backward tile dump:\n{r.stderr[-600:]}"
    assert all(b == "32" for b in bks), (
        f"D=64 dQ backward tile drifted from the tuned BK=32 to {bks} — "
        f"autoresearch regression (the dQ BK default must stay 32, = forward's D=64 config)")


@pytest.mark.parametrize("causal", [False, True])
def test_d64_backward_grad_correct_vs_fp32_oracle(causal):
    """The tuned D=64 backward matches an INDEPENDENT fp32 vjp oracle (Lesson #11) — BK=32 is
    perf-only, gradients unchanged. dV is the loose one (fp16 floor)."""
    mx.random.seed(0)
    D, N = 64, 4096
    sc = 1.0 / math.sqrt(D)
    f = lambda: (mx.random.uniform(-1, 1, (1, 8, N, D)) * 0.1).astype(mx.float16)
    q, k, v = f(), f(), f()
    mx.eval(q, k, v)
    gq, gk, gv = mx.grad(lambda a, b, c: flash_attention(a, b, c, scale=sc, causal=causal).sum(),
                         argnums=(0, 1, 2))(q, k, v)
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    def fwd(a, b, c):
        s = (a @ b.transpose(0, 1, 3, 2)) * sc
        if causal:
            s = mx.where(mx.arange(N)[:, None] >= mx.arange(N)[None, :], s, mx.array(-1e30, mx.float32))
        return mx.softmax(s, -1) @ c
    _, og = mx.vjp(fwd, (qf, kf, vf), (mx.ones(q.shape, mx.float32),))
    md = lambda a, b: float(np.abs(np.array(a.astype(mx.float32)) - np.array(b)).max())
    assert md(gq, og[0]) < 1e-3, f"dQ grad wrong (Δ={md(gq, og[0]):.2e})"
    assert md(gk, og[1]) < 1e-3, f"dK grad wrong (Δ={md(gk, og[1]):.2e})"
    assert md(gv, og[2]) < 2e-2, f"dV grad wrong (Δ={md(gv, og[2]):.2e})"  # dV is the loose output


def test_d64_backward_not_catastrophically_slow():
    """Generous catastrophic-regression sanity (NOT a tight perf lock — timing is CI-flaky):
    the tuned D=64 native backward must stay well under SDPA-vjp. Tuned ~2× faster; trips only
    on a gross regression (native slower than SDPA-vjp)."""
    mx.random.seed(0)
    D, N = 64, 4096
    sc = 1.0 / math.sqrt(D)
    f = lambda: (mx.random.uniform(-1, 1, (1, 8, N, D)) * 0.1).astype(mx.float16)
    q, k, v = f(), f(), f()
    mx.eval(q, k, v)

    def _b(disable):
        if disable:
            os.environ["MFA_DISABLE_V6_BACKWARD"] = "1"
        try:
            gf = mx.grad(lambda a, b, c: flash_attention(a, b, c, scale=sc, causal=True).sum(),
                         argnums=(0, 1, 2))
            for _ in range(6):
                mx.eval(gf(q, k, v))
            mx.synchronize(); ts = []
            for _ in range(15):
                mx.synchronize(); t0 = time.perf_counter(); mx.eval(gf(q, k, v)); mx.synchronize()
                ts.append(time.perf_counter() - t0)
            return sorted(ts)[len(ts) // 2] * 1e3
        finally:
            os.environ.pop("MFA_DISABLE_V6_BACKWARD", None)

    t_nat = _b(False)
    t_sdpa = _b(True)
    assert t_nat < 1.5 * t_sdpa, (
        f"D=64 native backward catastrophically slow: {t_nat:.2f}ms vs SDPA-vjp {t_sdpa:.2f}ms "
        f"(expected ~2× faster — a tile/routing regression?)")
