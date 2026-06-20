"""Phase 1 (3a): verify split@BK=32 V6 backward — correct + fast, engagement-proven.

Sidesteps the standalone-oracle blocker by going through the PUBLIC path with eligibility
forced ON — the public path supplies the correct `force_v6nax` natural-log-lse O/L, so dV
validates by construction. Caching-immune: each arm owns DISTINCT input objects (the
shared-input MLX graph-cache is what produced the prior byteΔ=0). FULL backward (dq+dk+dv),
not grad[0] (the b01e40d dq-only artifact).

RUN IN AN INTERPRETER WHOSE `mlx_mfa._ext` IS BUILT — i.e. the editable `.venv` (3.11).
NOT a bare 3.14 venv: the only built extension is `_ext.cpython-311-darwin.so`, so under
3.14 `import mlx_mfa._ext` fails → `has_nax=False` → V6 never engages → BOTH arms run SDPA
→ byteΔ=0 and the helper (correctly) raises. The `require=` guard below turns that into a
clear `FeatureUnavailable` instead of a misleading "vacuous". (To bench in 3.14, build an
`_ext` for 3.14 first.)

Toggle (the correct one): D=64 qL>=2048 is DEFAULT-ON (gated by MFA_DISABLE_V6_BACKWARD);
D=128 is opt-in via MFA_ENABLE_V6_BACKWARD=1.
"""
from __future__ import annotations

import math
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.bench_validity import measured_speedup  # noqa: E402
import mlx_mfa  # noqa: E402


def _mk(B, H, N, D, seed):
    mx.random.seed(seed)
    f = lambda: (mx.random.normal((B, H, N, D)) * 1.0).astype(mx.float16)  # unit scale
    q, k, v = f(), f(), f()
    mx.eval(q, k, v)
    return q, k, v


def _full_bwd(q, k, v, scale, causal):
    return mx.grad(lambda q, k, v: mlx_mfa.flash_attention(q, k, v, scale=scale, causal=causal).sum(),
                   argnums=(0, 1, 2))(q, k, v)


def _fp32_oracle(q, k, v, scale, causal):
    N = q.shape[2]
    def f(q, k, v):
        qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
        s = (qf @ kf.swapaxes(-1, -2)) * scale
        if causal:
            s = s + mx.triu(mx.full((N, N), -1e30, dtype=mx.float32), k=1)
        return (mx.softmax(s, -1) @ vf).sum()
    return mx.grad(f, argnums=(0, 1, 2))(q, k, v)


def bench(B, H, N, D, causal, d64_default):
    sc = 1.0 / math.sqrt(D)
    # Distinct input objects per arm (caching-immune); same seed = same values.
    qt, kt, vt = _mk(B, H, N, D, 7)
    qb, kb, vb = _mk(B, H, N, D, 7)

    def test_arm():       # split-V6
        if d64_default:
            os.environ.pop("MFA_DISABLE_V6_BACKWARD", None)
        else:
            os.environ["MFA_ENABLE_V6_BACKWARD"] = "1"
        return _full_bwd(qt, kt, vt, sc, causal)

    def baseline_arm():   # SDPA-vjp
        if d64_default:
            os.environ["MFA_DISABLE_V6_BACKWARD"] = "1"
        else:
            os.environ.pop("MFA_ENABLE_V6_BACKWARD", None)
        return _full_bwd(qb, kb, vb, sc, causal)

    try:
        return measured_speedup(
            test_arm, baseline_arm,
            test_label=f"split-V6 D={D} causal={causal}", baseline_label="SDPA-vjp",
            oracle=lambda: _fp32_oracle(qt, kt, vt, sc, causal),
            oracle_tol=0.5,   # unit-scale fp16 backward floor (dV loosest)
            # Fail loud if V6/NAX can't engage in this interpreter (the 3.14
            # missing-`_ext` trap) instead of a misleading "vacuous" raise.
            # Consumes the canonical availability source (mlx_mfa.has_nax — DRY).
            require=lambda: mlx_mfa.has_nax(),
            require_label="V6 NAX backward (M5 + mlx_mfa._ext)",
        )
    finally:
        os.environ.pop("MFA_DISABLE_V6_BACKWARD", None)
        os.environ.pop("MFA_ENABLE_V6_BACKWARD", None)


if __name__ == "__main__":
    print(f"MLX {mx.__version__} | mlx_mfa {mlx_mfa.__version__}")
    print("D=64 (DEFAULT-ON):")
    for causal in (False, True):
        for N in (4096, 8192):
            r = bench(1, 4, N, 64, causal, d64_default=True)
            print(f"  D64 causal={causal!s:5} qL{N}: {r}  oracle_maxabs={r.oracle_max_abs:.2e}")
    print("D=128 (opt-in MFA_ENABLE_V6_BACKWARD=1):")
    for causal in (False, True):
        r = bench(1, 4, 4096, 128, causal, d64_default=False)
        print(f"  D128 causal={causal!s:5} qL4096: {r}  oracle_maxabs={r.oracle_max_abs:.2e}")
