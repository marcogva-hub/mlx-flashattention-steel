"""III-9 multi-gate lock — GNA + STEEL V5 direct-V partial-tile read coverage.

§AA.5.x multi-gate: the V2 single-pass last-head OOB-V bug (fixed eb68af5) had
two sibling sites sharing the IDENTICAL unbounded MFA_DIRECT_READS V-read on the
partial final K-tile (0 * NaN-from-OOB = NaN). Iteration-2 of the III-9 sweep
found them; both fixed with the same kL_rem-1 key-row clamp:
  - GNA native (csrc/mfa_gna_fwd.cpp) — DEFAULT-REACHABLE via flash_attention_gna
    (D=128, 3D, f16/bf16, N % 32 != 0).  The serious one.
  - STEEL V5 (csrc/mfa_steel_fwd_v5.cpp) — opt-in (MFA_ENABLE_V5), partial-S.

Both validated vs an INDEPENDENT fp32 reference (element-level GNA mask + SDPA
for GNA; SDPA for V5) under the bug's trigger: heterogeneous-preamble pool
history + concurrent allocation before mx.eval + non-tile-aligned shapes.
"""
from __future__ import annotations

import itertools
import numpy as np
import pytest
import mlx.core as mx

from mlx_mfa.attention import flash_attention, flash_attention_gna


def _gna_elem_mask(seq, win, stride):
    N = int(np.prod(seq))
    coords = list(itertools.product(*[range(s) for s in seq]))
    M = np.zeros((N, N), bool)
    for qi, q in enumerate(coords):
        lohi = []
        for d in range(len(seq)):
            gb = (q[d] // stride[d]) * stride[d]
            lo = max(0, gb - (win[d] - stride[d]) // 2)
            hi = min(seq[d], gb + stride[d] + (win[d] - stride[d] + 1) // 2)
            lohi.append((lo, hi))
        for ki, kk in enumerate(coords):
            if all(lohi[d][0] <= kk[d] < lohi[d][1] for d in range(len(seq))):
                M[qi, ki] = True
    return M


def _sdpa_masked(q, k, v, mask, scale):
    add = mx.where(mx.array(mask), mx.array(0.0, mx.float32), mx.array(-1e30, mx.float32))
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=scale, mask=add)


_GNA = [((3, 3, 3), (3, 3, 3), (1, 1, 1)), ((2, 3, 7), (2, 3, 3), (1, 1, 1)),
        ((3, 5, 5), (3, 3, 3), (1, 1, 1)), ((7, 7, 3), (3, 3, 3), (1, 1, 1))]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("seq,win,stride", _GNA)
def test_gna_native_noncausal_partial_tile_no_oob(seq, win, stride, dtype):
    """GNA native (default-reachable): non-32-aligned N must not leak OOB-V NaN."""
    N = int(np.prod(seq)); D = 128; sc = 1.0 / (D ** 0.5)
    for n in (32, 64, 96):  # preamble pool history
        mx.random.seed(0)
        qq = mx.random.normal((1, 4, n, D)).astype(dtype)
        mx.eval(flash_attention_gna(qq, qq, qq, (n, 1, 1), (8, 1, 1), (1, 1, 1), scale=sc))
    mx.random.seed(1)
    q = mx.random.normal((1, 4, N, D)).astype(dtype)
    k = mx.random.normal((1, 4, N, D)).astype(dtype)
    v = mx.random.normal((1, 4, N, D)).astype(dtype)
    o = flash_attention_gna(q, k, v, seq, win, stride, scale=sc)
    ref = _sdpa_masked(q, k, v, _gna_elem_mask(seq, win, stride), sc)
    junk = mx.random.normal((1, 4, 512, D)).astype(mx.float32) * 3.0
    mx.eval(o, ref, junk)
    assert not bool(mx.any(mx.isnan(o)).item()), f"GNA NaN seq={seq} {dtype}"
    bound = 0.05 if dtype == mx.float16 else 0.08
    mae = float(mx.abs(o.astype(mx.float32) - ref).mean().item())
    assert mae < bound, f"GNA seq={seq} {dtype}: MAE {mae:.4f} >= {bound}"


_V5 = [(128, 1, 64), (128, 2, 64), (128, 3, 64), (256, 33, 64),
       (992, 992, 64), (128, 1, 128), (383, 383, 128), (256, 65, 128)]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N,S,D", _V5)
@pytest.mark.skip(reason="STEEL V4/V5 retired from build (Lot-2 chore); opt-in kernels removed")
def test_v5_noncausal_partial_tile_no_oob(N, S, D, dtype, monkeypatch):
    """STEEL V5 (opt-in): partial-S non-causal must not leak OOB-V NaN."""
    monkeypatch.setenv("MFA_ENABLE_V5", "1")
    monkeypatch.setenv("MFA_FORCE_SPLITK", "0")
    sc = 1.0 / (D ** 0.5)
    for (n, c) in [(128, False), (128, True), (256, False)]:
        mx.random.seed(0)
        q = mx.random.normal((1, 8, n, D)).astype(dtype)
        mx.eval(flash_attention(q, q, q, scale=sc, causal=c, backend="mfa"))
    mx.random.seed(1)
    q = mx.random.normal((1, 8, N, D)).astype(dtype)
    k = mx.random.normal((1, 8, S, D)).astype(dtype)
    v = mx.random.normal((1, 8, S, D)).astype(dtype)
    o = flash_attention(q, k, v, scale=sc, causal=False, backend="mfa")
    ref = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=sc)
    junk = mx.random.normal((1, 8, 512, D)).astype(mx.float32) * 3.0
    mx.eval(o, ref, junk)
    assert not bool(mx.any(mx.isnan(o)).item()), f"V5 NaN N={N} S={S} D={D} {dtype}"
    bound = 0.05 if dtype == mx.float16 else 0.08
    mae = float(mx.abs(o.astype(mx.float32) - ref).mean().item())
    assert mae < bound, f"V5 N={N} S={S} D={D} {dtype}: MAE {mae:.4f} >= {bound}"
