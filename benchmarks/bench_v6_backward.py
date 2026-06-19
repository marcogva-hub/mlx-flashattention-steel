"""V6NAX backward comparative bench — routed through the engagement-asserting
`measured_speedup` helper (recurrence-prevention for the V6-vs-V6 vacuous bench).

Engagement proof: **byteΔ on the gradients**. The dispatch-trace instruments only the
FORWARD terminals (Phase-0 audit), and a backward bench's forward terminal can be
*identical* across arms (e.g. D=64 causal / D=128 route the same forward both ways)
while the BACKWARD differs — so a forward-trace check would false-positive "vacuous".
byteΔ on the returned grads is the correct signal. **FLAG: instrument the backward/vjp
terminals in `mlx_mfa.attention` (_dispatch_trace.record) for gold trace-based backward
engagement proof; until then byteΔ is the proof here.**

Correct toggle (the closure's bug): D=64 backward is DEFAULT-ON, gated by
`MFA_DISABLE_V6_BACKWARD`; D=128 is opt-in via `MFA_ENABLE_V6_BACKWARD`.
Run with the 3.14 venv (current MLX).
"""
from __future__ import annotations

import math
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.bench_validity import measured_speedup  # noqa: E402
import mlx_mfa  # noqa: E402


def _mk(B, H, N, D):
    mx.random.seed(1)
    f = lambda: (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
    q, k, v = f(), f(), f()
    mx.eval(q, k, v)
    return q, k, v


def _grad_fn(causal, D):
    sc = 1.0 / math.sqrt(D)
    return mx.grad(lambda q, k, v: mlx_mfa.flash_attention(q, k, v, scale=sc, causal=causal).sum(),
                   argnums=(0, 1, 2))


def _fp32_oracle_grads(q, k, v, causal):
    sc = 1.0 / math.sqrt(q.shape[-1])
    def f(q, k, v):
        qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
        s = (qf @ kf.swapaxes(-1, -2)) * sc
        if causal:
            N = q.shape[2]
            s = s + mx.triu(mx.full((N, N), -1e30, dtype=mx.float32), k=1)
        return (mx.softmax(s, -1) @ vf).sum()
    return mx.grad(f, argnums=(0, 1, 2))(q, k, v)


def bench(B, H, N, D, causal, d64_default_on=True):
    q, k, v = _mk(B, H, N, D)
    # NOTE: a FRESH grad fn per arm — reusing one closure lets MLX reuse the traced
    # graph across the env toggle (env is not part of its graph-cache key), which
    # collapses both arms to the same path (the helper RAISES on that — proven).

    def test_arm():       # V6 NAX-direct backward
        if d64_default_on:
            os.environ.pop("MFA_DISABLE_V6_BACKWARD", None)   # D=64 default-on
        else:
            os.environ["MFA_ENABLE_V6_BACKWARD"] = "1"        # D=128 opt-in
        return _grad_fn(causal, D)(q, k, v)

    def baseline_arm():   # SDPA-vjp
        if d64_default_on:
            os.environ["MFA_DISABLE_V6_BACKWARD"] = "1"
        else:
            os.environ.pop("MFA_ENABLE_V6_BACKWARD", None)
        return _grad_fn(causal, D)(q, k, v)

    try:
        res = measured_speedup(
            test_arm, baseline_arm,
            test_label=f"V6-bwd D={D} causal={causal}",
            baseline_label="SDPA-vjp",
            oracle=lambda: _fp32_oracle_grads(q, k, v, causal),
            oracle_tol=2e-2,   # fp16 backward floor (dV is the loosest term)
        )
    finally:
        os.environ.pop("MFA_DISABLE_V6_BACKWARD", None)
        os.environ.pop("MFA_ENABLE_V6_BACKWARD", None)
    return res


if __name__ == "__main__":
    from benchmarks.bench_validity import VacuousBenchmark
    print(f"MLX {mx.__version__} | mlx_mfa {mlx_mfa.__version__}")
    print("NOTE: on M5 the `MFA_DISABLE_V6_BACKWARD` toggle yields byteΔ=0 end-to-end "
          "(fresh inputs) → the helper RAISES 'vacuous' because the two arms are NOT "
          "provably different paths via the public flash_attention autograd. This is the "
          "FINDING: V6-backward engagement is not provable here — the gradient appears to "
          "be SDPA-vjp regardless of the carve-out (the v6_nax_backward_* _ext kernels are "
          "not demonstrably wired into the vjp on this build). FLAG: instrument the "
          "backward/vjp terminals (_dispatch_trace.record) OR bench the _ext kernel "
          "directly to settle; the b01e40d 2.55-5.75x numbers are UNVALIDATED until then.")
    for causal in (False, True):
        for N in (4096, 8192):
            try:
                r = bench(1, 4, N, 64, causal, d64_default_on=True)
                print(f"D=64 causal={causal!s:5} qL{N}: {r}")
            except VacuousBenchmark as e:
                print(f"D=64 causal={causal!s:5} qL{N}: VACUOUS (helper refused) — {str(e)[:90]}")
    for causal in (False, True):
        try:
            r = bench(1, 4, 4096, 128, causal, d64_default_on=False)
            print(f"D=128 causal={causal!s:5} qL4096 (opt-in): {r}")
        except VacuousBenchmark as e:
            print(f"D=128 causal={causal!s:5} qL4096: VACUOUS (helper refused) — {str(e)[:90]}")
