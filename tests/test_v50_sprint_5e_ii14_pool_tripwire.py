"""II-14 — buffer-pool self-consistency TRIPWIRE (permanent, stress-gated).

Runs the historical victim computation (fused dense + sparse dKdV,
all-true mask) repeatedly under MFA_POOL_STRESS=1, cross-comparing
outputs, raw partials, AND the forward (O, L) pair bit-wise.  This is
the detection harness for the II-6/II-8 stale-recycled-buffer residual:
47 consecutive stressed full-suite runs (plus ~820 amplified paired
executions) were clean at the II-14 close, statistically eliminating
the historical ~1/6 flake — see sprint-II-14-report.md.  If pool
stale-value sensitivity ever returns, this test FAILS with localization
data in /tmp/ii14_diag_*.txt.  Placed right after the victim file so it
sees the same accumulated pool state.
"""
import glob
import math
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import get_device_info

_HAS = bool(get_device_info().get("is_m5_plus", False))
pytestmark = [
    pytest.mark.skipif(not _HAS, reason="M5 only"),
    pytest.mark.skipif(os.environ.get("MFA_POOL_STRESS") != "1",
                       reason="stress-mode tripwire (MFA_POOL_STRESS=1; "
                              "release-audit pre-tag step)"),
]


def _mk(B, Hq, Hkv, N, D, dt, seed):
    mx.random.seed(seed)
    q = mx.random.normal((B, Hq, N, D)).astype(dt)
    k = mx.random.normal((B, Hkv, N, D)).astype(dt)
    v = mx.random.normal((B, Hkv, N, D)).astype(dt)
    dO = mx.random.normal((B, Hq, N, D)).astype(dt)
    mx.eval(q, k, v, dO)
    return q, k, v, dO


def test_ii14_self_consistency_diag():
    from mlx_mfa import _ext
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_v50_sprint_5d_sparse_backward_native import _convert_mask_for_v6nax_bwd_kernel

    B, H, qL, D = 1, 4, 2048, 64
    BT = 32
    NQ = NK = qL // BT
    q, k, v, dO = _mk(B, H, H, qL, D, mx.float16, 202)
    scale = 1.0 / math.sqrt(D)
    O, L = _ext.v6_nax_forward(q, k, v, False, True)
    mx.eval(O, L); mx.synchronize()
    D_vec = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1)
    mx.eval(D_vec); mx.synchronize()
    mask_all = _convert_mask_for_v6nax_bwd_kernel(
        mx.ones((NQ, NK), dtype=mx.bool_), BT, "DKDV", D)
    mx.eval(mask_all); mx.synchronize()

    def run_sparse():
        a, b = _ext.v6_nax_backward_fused_dkdv_sparse_raw(
            q, k, v, L, dO, D_vec, mask_all, scale, 4, False)
        mx.eval(a, b); mx.synchronize()
        return a, b

    def run_dense():
        a, b = _ext.v6_nax_backward_fused_dkdv_raw(
            q, k, v, L, dO, D_vec, scale, 4, False)
        mx.eval(a, b); mx.synchronize()
        return a, b

    # forward self-consistency (upstream hypothesis): recompute O, L and
    # bit-compare — if the forward itself is pool-sensitive, both backward
    # kernels inherit it.
    O2, L2 = _ext.v6_nax_forward(q, k, v, False, True)
    mx.eval(O2, L2); mx.synchronize()
    fwd_drift = []
    if not bool(mx.all(O == O2).item()):
        fwd_drift.append("FORWARD O drifted between identical calls\n")
    if not bool(mx.all(L == L2).item()):
        fwd_drift.append("FORWARD L drifted between identical calls\n")

    s1 = run_sparse(); d1 = run_dense(); s2 = run_sparse(); d2 = run_dense()

    def cmp(name, x, y, report):
        xk, yk = np.array(x[0]), np.array(y[0])
        xv, yv = np.array(x[1]), np.array(y[1])
        for tag, a, b in (("dKp", xk, yk), ("dVp", xv, yv)):
            neq = a != b
            # NaN != NaN — treat matching NaNs as equal
            both_nan = np.isnan(a) & np.isnan(b)
            neq = neq & ~both_nan
            n = int(neq.sum())
            if n:
                pos = np.argwhere(neq)
                wm_slots = np.unique(pos[:, 2]).tolist()
                heads = np.unique(pos[:, 1]).tolist()
                krows = np.unique(pos[:, 3])
                report.append(
                    f"{name}/{tag}: n_bad={n} heads={heads} wm_slots={wm_slots} "
                    f"k_rows=[{krows.min()}..{krows.max()}] n_krows={len(krows)} "
                    f"krows%32={np.unique(krows % 32).tolist()[:8]}\n"
                    f"  first: {[tuple(p) for p in pos[:5]]}\n"
                    f"  vals a={a[tuple(pos[0])]} b={b[tuple(pos[0])]}\n"
                    f"  a_nan={int(np.isnan(a).sum())} b_nan={int(np.isnan(b).sum())} "
                    f"a_inf={int(np.isinf(a).sum())} b_inf={int(np.isinf(b).sum())}\n")

    report = list(fwd_drift)
    cmp("sparse1-vs-sparse2", s1, s2, report)   # sparse nondeterminism
    cmp("dense1-vs-dense2", d1, d2, report)     # dense nondeterminism
    cmp("sparse1-vs-dense1", s1, d1, report)    # the victim's comparison
    # amplification: many more paired executions
    for it in range(5):
        sa = run_sparse(); sb = run_sparse()
        da = run_dense()
        pre = len(report)
        cmp(f"amp{it}-sp-vs-sp", sa, sb, report)
        cmp(f"amp{it}-sp-vs-dn", sa, da, report)
        if len(report) > pre and len(report) > 12:
            break  # enough evidence captured
    if report:
        n = len(glob.glob("/tmp/ii14_diag_*.txt"))
        with open(f"/tmp/ii14_diag_{n}.txt", "w") as f:
            f.writelines(report)
    assert not report, (
        "buffer-pool stale-value sensitivity RETURNED — localization in "
        f"/tmp/ii14_diag_*.txt: {report[:2]}"
    )
