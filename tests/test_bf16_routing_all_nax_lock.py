"""bf16-routing regression lock across ALL NAX paths (Tier-1 #1 audit, M5 Max, 2026-06-18).

The 2.59.0 sparse-forward footgun (a silent `&& is_f16` eligibility gate → bf16 fell to the
~45×-slower V1 scalar kernel) lived undetected for many versions because no test asserted that
bf16 reaches the SAME fast binary as fp16. This lock closes that class: for every NAX path, it
fingerprints the dispatched binary for bf16 and asserts it is the fast path (NAX/native), NOT a
slow fallback — so a future re-introduced dtype gate (anywhere: Python eligibility OR a downstream
C++ `is_f16`) fails CI.

Audit verdict (this branch): all NAX paths route bf16 == fp16; the only silent downgrade was the
already-fixed sparse forward (locked separately in test_sparse_bf16_v2_lock.py). Sparse here is the
known-good reference cell; this module adds the dense / conv3d / GNA cells.

Method = runtime fingerprint (Lesson #14), NOT source reading.
"""
from __future__ import annotations

import math
import os
import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa.attention import _get_is_m5_plus_cached

pytestmark = pytest.mark.skipif(
    not _get_is_m5_plus_cached(),
    reason="bf16 NAX routing lock asserts M5+ kernels")


def _delta(a, b):
    mx.eval(a, b)
    return float(np.abs(np.array(a.astype(mx.float32)) - np.array(b.astype(mx.float32))).max())


def _qkv(B, H, N, D, dt, seed=0):
    mx.random.seed(seed)
    f = lambda: (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(dt)
    q, k, v = f(), f(), f()
    mx.eval(q, k, v)
    return q, k, v


def _fp32_attn_oracle(q, k, v, causal):
    """Independent fp32 forward oracle (manual softmax, NOT SDPA/vjp) — Lesson #11.
    softmax(QK^T*scale [+causal mask]) @ V, all fp32."""
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    scale = 1.0 / math.sqrt(qf.shape[-1])
    s = (qf @ kf.swapaxes(-1, -2)) * scale
    if causal:
        N = qf.shape[2]
        s = s + mx.triu(mx.full((N, N), -1e30, dtype=mx.float32), k=1)
    o = mx.softmax(s, axis=-1) @ vf
    mx.eval(o)
    return o


def test_dense_d128_bf16_routes_to_nax():
    """Dense D=128 auto (N>=2048, above the Tier-2 #1 threshold): bf16 must reach the real NAX
    matmul2d forward (byteΔ>0 vs SDPA), exactly like fp16. byteΔ==0 ⇒ it silently dropped to SDPA."""
    from mlx_mfa import flash_attention
    sc = 1.0 / math.sqrt(128)
    for dt in (mx.float16, mx.bfloat16):
        q, k, v = _qkv(1, 8, 2048, 128, dt)
        o = flash_attention(q, k, v, scale=sc)
        sd = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)
        d = _delta(o, sd)
        assert d > 1e-7, (
            f"dense D=128 {dt} is byte-identical to SDPA (Δ={d:.2e}) — it dropped to the "
            f"SDPA fallback instead of the NAX forward. A dtype gate regressed.")


@pytest.mark.parametrize("causal", [False, True])
def test_dense_d128_bf16_correct_vs_fp32(causal):
    """Tier-2 #2 axis 3 (correctness): bf16 dense D=128 NAX output is within the bf16 floor of an
    INDEPENDENT fp32 oracle.  Measured worst-case ~1.3e-3 (causal); floor 5e-3 is generous but well
    inside the bf16 range (<1e-2) and catches a real bf16 correctness regression."""
    from mlx_mfa import flash_attention
    q, k, v = _qkv(1, 8, 2048, 128, mx.bfloat16, seed=3)
    o = flash_attention(q, k, v, causal=causal)
    err = _delta(o, _fp32_attn_oracle(q, k, v, causal))
    assert err < 5e-3, f"bf16 dense D=128 (causal={causal}) wrong vs fp32 oracle (Δ={err:.2e})"


def test_dense_d128_bf16_perf_parity_vs_fp16():
    """Tier-2 #2 axis 2 (perf): bf16 dense D=128 NAX is at parity with fp16 (measured median ratio
    0.95-1.02 across shapes).  Generous ceiling 1.30× (RESULTS.md: perf re-measure-not-lock — bound,
    don't pin) — catches a future bf16-specific slowdown (e.g. a silent drop to a slow path) without
    flaking on machine/thermal variance.  Both dtypes route NAX (N=2048 >= threshold)."""
    import time
    from mlx_mfa import flash_attention

    def med_ms(q, k, v, causal, w=10, it=60):
        for _ in range(w):
            mx.eval(flash_attention(q, k, v, causal=causal))
        mx.synchronize(); ts = []
        for _ in range(it):
            mx.synchronize(); t0 = time.perf_counter()
            mx.eval(flash_attention(q, k, v, causal=causal)); mx.synchronize()
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[len(ts) // 2]

    for causal in (False, True):
        q16, k16, v16 = _qkv(1, 8, 4096, 128, mx.float16, seed=3)
        qb, kb, vb = _qkv(1, 8, 4096, 128, mx.bfloat16, seed=3)
        f16 = med_ms(q16, k16, v16, causal)
        bf = med_ms(qb, kb, vb, causal)
        assert bf < 1.30 * f16, (
            f"bf16 dense D=128 (causal={causal}) {bf*1e3:.3f} ms is >1.30× fp16 {f16*1e3:.3f} ms "
            f"(ratio {bf/f16:.2f}) — a bf16-specific perf regression (parity expected, ~1.0×).")


def test_dense_d64_force_bf16_runs_nax():
    """Forced dense D=64 NAX (v6_nax_forward, the backward-recompute/expert path): bf16 must run
    the NAX kernel (byteΔ>0 vs SDPA), like fp16."""
    from mlx_mfa._ext import v6_nax_forward
    sc = 1.0 / math.sqrt(64)
    for dt in (mx.float16, mx.bfloat16):
        q, k, v = _qkv(1, 8, 4096, 64, dt)
        o, _ = v6_nax_forward(q, k, v, False, True)
        sd = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)
        d = _delta(o, sd)
        assert d > 1e-7, f"forced D=64 NAX {dt} byte-identical to SDPA (Δ={d:.2e}) — not NAX"


def test_conv3d_bf16_routes_to_nax_hook():
    """conv3d auto-hook: an MPP-eligible bf16 conv must engage the NAX kernel
    (executed.conv3d_nax_forward++, no fallback), like fp16."""
    import mlx_mfa
    from mlx_mfa import get_hook_stats
    ec = lambda s: s["executed"].get("conv3d_nax_forward", 0)
    fc = lambda s: s["fallback"].get("conv3d_nax_forward", 0)
    for dt in (mx.float16, mx.bfloat16):
        mx.random.seed(0)
        x = (mx.random.uniform(-1, 1, (1, 8, 16, 16, 32)) * 0.1).astype(dt)  # B,T,H,W,Cin
        w = (mx.random.uniform(-1, 1, (32, 3, 3, 3, 32)) * 0.1).astype(dt)
        mx.eval(x, w)
        be, bf = ec(get_hook_stats()), fc(get_hook_stats())
        o = mx.conv_general(x, w, stride=1, padding=1)
        mx.eval(o)
        de, df = ec(get_hook_stats()) - be, fc(get_hook_stats()) - bf
        assert de > 0 and df == 0, (
            f"MPP-eligible conv3d {dt} did NOT route to the NAX kernel "
            f"(executed+={de}, fallback+={df}) — a dtype gate regressed.")
        assert bool(np.isfinite(np.array(o.astype(mx.float32))).all())


@pytest.mark.parametrize("causal", [False, True])
def test_d64_native_backward_bf16_runs_native(causal):
    """Dense D=64 NATIVE backward (default-on, N≥2048): bf16 must reach the native backward
    kernel, like fp16 — NOT a silent SDPA-vjp downgrade. Fingerprint = byteΔ between the default
    (native eligible) dQ grad and the MFA_DISABLE_V6_BACKWARD=1 (forced SDPA-vjp) grad; >0 ⇒ the
    native kernel ran. Closes the backward-shaped hole in this cross-path bf16 guard."""
    from mlx_mfa import flash_attention
    D, N = 64, 2048
    sc = 1.0 / math.sqrt(D)
    for dt in (mx.float16, mx.bfloat16):
        q, k, v = _qkv(1, 8, N, D, dt)
        gf = lambda: mx.grad(lambda a: flash_attention(a, k, v, scale=sc, causal=causal).sum())(q)
        os.environ.pop("MFA_DISABLE_V6_BACKWARD", None)
        g_native = gf()
        os.environ["MFA_DISABLE_V6_BACKWARD"] = "1"
        try:
            g_sdpa = gf()
        finally:
            os.environ.pop("MFA_DISABLE_V6_BACKWARD", None)
        d = _delta(g_native, g_sdpa)
        assert d > 1e-7, (
            f"D=64 native backward {dt} c={causal}: default dQ grad byte-identical to the "
            f"forced-SDPA-vjp grad (Δ={d:.2e}) — bf16 silently downgraded off the native backward.")


def test_gna_native_bf16_runs_native():
    """GNA: bf16 must run the native GNA kernel, like fp16 — byteΔ>0 between native-enabled and
    native-disabled (sparse-fallback) proves the native kernel engaged."""
    from mlx_mfa import flash_attention_gna
    ss, N, D = (8, 8, 8), 512, 128
    sc = 1.0 / math.sqrt(D)
    for dt in (mx.float16, mx.bfloat16):
        q, k, v = _qkv(1, 4, N, D, dt)
        on = flash_attention_gna(q, k, v, ss, (3, 3, 3), (1, 1, 1), scale=sc)
        os.environ["MFA_DISABLE_GNA_NATIVE"] = "1"
        try:
            off = flash_attention_gna(q, k, v, ss, (3, 3, 3), (1, 1, 1), scale=sc)
        finally:
            os.environ.pop("MFA_DISABLE_GNA_NATIVE", None)
        d = _delta(on, off)
        assert d > 1e-7, (
            f"GNA {dt} native-on == native-off (Δ={d:.2e}) — bf16 silently fell to the sparse "
            f"fallback instead of the native GNA kernel.")
        assert bool(np.isfinite(np.array(on.astype(mx.float32))).all())
