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
from pathlib import Path
import numpy as np
import mlx.core as mx
import pytest

_SRC = Path(__file__).resolve().parent.parent / "csrc" / "mfa_v6_nax_primitive.cpp"

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


# ─────────────────────────────────────────────────────────────────────────────
# Tier-1 #2b: dV / dK split-kernel BK confirmed-optimal at 32 (not tuned — but locked).
# Sweep verdict (M5 Max, 2026-06-18): legal BK = {32, 64} (BK=16 throws — paired-MMA
# guard, the split dV/dK have NO odd-TK tail); BK=64 is slower in all 12 cells
# (+18..+69%); dV grad-error is FLAT across legal BK (perf-only, no tradeoff). So
# BK=32 is optimal for both — no code change, but lock so a future re-inherited BK=64
# (the dQ mistake) is caught.
# ─────────────────────────────────────────────────────────────────────────────

def _default_bk_before(env_var: str) -> list[int]:
    """Every `unsigned short BK = <n>;` default immediately preceding an `env_var` read
    (the dV has 2 generator sites, dK has 1). Returns the <n> values."""
    text = _SRC.read_text()
    vals = []
    for m in re.finditer(re.escape(env_var), text):
        pre = text[:m.start()]
        d = re.findall(r"unsigned short BK = (\d+);", pre)
        assert d, f"no `unsigned short BK = N;` default precedes {env_var}"
        vals.append(int(d[-1]))
    return vals


@pytest.mark.parametrize("env_var", ["MFA_V6BWDV_BK", "MFA_V6BWDK_BK"])
def test_dvdk_split_default_bk_is_32(env_var):
    """Config-fingerprint: the split dV/dK D=64 default BK is 32 (confirmed optimal —
    BK=64 is +18..+69% slower). A revert to 64 (the dQ-style stale inheritance) fails CI."""
    vals = _default_bk_before(env_var)
    assert vals and all(v == 32 for v in vals), (
        f"{env_var}'s default BK drifted to {vals} (expected all 32 — BK=64 is slower; "
        f"re-run the dV/dK mini-sweep before changing)")


@pytest.mark.skipif(not _IS_M5, reason="M5+ NAX required")
@pytest.mark.parametrize("env_var", ["MFA_V6BWDV_BK", "MFA_V6BWDK_BK"])
def test_dvdk_bk16_throws(env_var):
    """Rule-8 valid-triple guard lock: BK=16 (TK=1, odd) is illegal for the split dV/dK
    (no odd-TK tail; paired 16x32x16 MMA) and must THROW — not fall to a removed path."""
    # NOTE: distinct k,v + argnums=(0,1,2) so dV/dK actually dispatch (an aliased
    # q=k=v with grad over arg0 only computes dQ — the dV/dK kernels never run).
    code = (
        "import math, mlx.core as mx\n"
        "from mlx_mfa import flash_attention\n"
        "mx.random.seed(0); D, N = 64, 2048; sc = 1/math.sqrt(D)\n"
        "f = lambda: (mx.random.uniform(-1,1,(1,8,N,D))*0.1).astype(mx.float16)\n"
        "q,k,v = f(),f(),f(); mx.eval(q,k,v)\n"
        "g = mx.grad(lambda a,b,c: flash_attention(a,b,c,scale=sc).sum(), argnums=(0,1,2))(q,k,v); mx.eval(g)\n"
    )
    env = dict(os.environ, **{env_var: "16"})
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd="/tmp")
    assert r.returncode != 0 and "BK must be a positive multiple of 32" in r.stderr, (
        f"{env_var}=16 did not throw the paired-MMA guard (Rule 8):\n{r.stderr[-400:]}")
