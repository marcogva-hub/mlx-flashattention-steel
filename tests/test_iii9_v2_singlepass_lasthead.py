"""III-9 lock — V2 single-pass non-causal last-head partial-tile read coverage.

Root cause (docs/v50/campaign-2026-06/phase3/sprint-III-9-report.md): on M5's
MFA_DIRECT_READS path the V2 single-pass forward reads K/V directly from device
memory with no bounds check (unlike the !DIRECT load_safe zero-pad). For a
PARTIAL final K-tile (S not a multiple of BK), keys past kL are read OOB. The
K-boundary mask correctly sets P=0 for those keys, but the V read still returned
NaN/stale-pool data for them — and 0 * NaN = NaN corrupted the output. Because
the LAST head (h=H-1) sits at the end of the K/V buffer, its OOB read spilled
past the buffer into freed pool memory, so the corruption was pool-history /
concurrent-allocation dependent and confined to the last head (even-D-column
MMA pattern). Forced-path only: backend="auto" routes non-causal dense to SDPA
and was always clean.

Fix (csrc/mfa_steel_fwd_v2.cpp): clamp the V direct-read key-row to the last
valid key (kL_rem-1) on the partial final tile, so OOB keys read a finite V;
P is already 0 for them, so 0*finite = 0 is the correct masked contribution.

This lock reproduces the exact trigger: forced single-pass (MFA_FORCE_SPLITK=0),
non-causal, a heterogeneous in-process allocation preamble (the pool-history
factor that made isolation tests pass falsely), a concurrent allocation before
mx.eval, and partial-S shapes — validated vs an INDEPENDENT fp32 reference.
"""
from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx

from mlx_mfa import flash_attention


@pytest.fixture(autouse=True)
def _force_single_pass(monkeypatch):
    monkeypatch.setenv("MFA_FORCE_SPLITK", "0")  # V2 single-pass (the buggy path)


def _preamble(D):
    # Heterogeneous prior dispatches → realistic buffer-pool history (the factor
    # that masked the bug under isolation/cold/warm-loop testing).
    sc = 1.0 / (D ** 0.5)
    for (n, c) in [(128, False), (128, True), (512, False), (512, True)]:
        mx.random.seed(0)
        q = mx.random.normal((1, 4, n, D)).astype(mx.float16)
        k = mx.random.normal((1, 4, n, D)).astype(mx.float16)
        v = mx.random.normal((1, 4, n, D)).astype(mx.float16)
        mx.eval(flash_attention(q, k, v, scale=sc, causal=c, backend="mfa"))


# Partial final K-tile shapes (S % BK != 0) that exposed the last-head OOB read.
# D=64 BK=64: N in {224, 992..1023}; D=128 BK=64(M3+): N=383.  Plus short-S.
_CONFIGS = [
    (224, 224, 64), (992, 992, 64), (1000, 1000, 64), (1023, 1023, 64),
    (383, 383, 128), (256, 33, 64), (256, 65, 128),
    (128, 1, 64), (128, 2, 64), (128, 3, 64), (128, 16, 64),
    (128, 1, 128), (128, 2, 128), (512, 1, 64),
]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N,S,D", _CONFIGS)
def test_v2_singlepass_noncausal_lasthead_no_oob(N, S, D, dtype):
    _preamble(D)
    sc = 1.0 / (D ** 0.5)
    mx.random.seed(1)
    q = mx.random.normal((1, 4, N, D)).astype(dtype)
    k = mx.random.normal((1, 4, S, D)).astype(dtype)
    v = mx.random.normal((1, 4, S, D)).astype(dtype)
    o = flash_attention(q, k, v, scale=sc, causal=False, backend="mfa")
    # Concurrent allocation before eval (the OOB-spill trigger).
    ref = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=sc)
    junk = mx.random.normal((1, 4, max(N, S, 512), D)).astype(mx.float32) * 3.0
    mx.eval(o, ref, junk)
    assert not bool(mx.any(mx.isnan(o)).item()), (
        f"NaN in V2 single-pass non-causal N={N} S={S} D={D} {dtype} "
        f"(last-head partial-tile OOB read regression)")
    bound = 0.03 if dtype == mx.float16 else 0.06
    mae = float(mx.abs(o.astype(mx.float32) - ref).mean().item())
    assert mae < bound, f"N={N} S={S} D={D} {dtype}: MAE {mae:.4f} >= {bound}"
