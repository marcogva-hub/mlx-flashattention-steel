"""CC Batch-1 — raw/function-surface dtype × forward matrix (CI-locked).

The runtime matrix (`test_dtype_backend_matrix.py`) covers only the runtime/context
surface. This is the analogue for the RAW `_ext` / function surface, where two
defect classes lived that the P-series never covered:

  Class A — silent-wrong fp32: a raw forward with a 2-way `dtype_code=0; if(bf16)=1`
    map (no fp32 branch) lets fp32 fall to the half kernel → FINITE GARBAGE (cos≈0,
    max_abs≈1e37), masked because output is allocated in the requested dtype.
    Fix = reject fp32 at the raw-entry boundary (`assert_raw_fp16_bf16_only`).
  Class B — late compile-fail: a C++-ext kernel emitted MLX's `bfloat16_t` (only
    valid on the mx.fast.metal_kernel surface) instead of native `bfloat`, so a
    reachable bf16 config late-compile-failed. Fix = emit `bfloat`.

Oracle rule (the rule every prior miss violated): a cell PASSES only if **finite
AND output.dtype == requested AND cosine vs an independent fp32 SDPA reference ≥
floor**. "runs / no exception" is never a pass. Gated cells must raise at entry.

The matrix is **table-driven** (`_CELLS`) so new findings add rows, not files.
"""
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa
from mlx_mfa._ext import (mfa_paged_varlen_forward,
                          mfa_paged_varlen_tq_forward)

assert mlx_mfa.has_nax() is True
F16, BF16, F32 = mx.float16, mx.bfloat16, mx.float32


def _cos(o, ref):
    a = np.array(o.astype(mx.float32)).reshape(-1)
    b = np.array(ref).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# ── builders + fp32 oracles per raw entry ───────────────────────────────────────
def _paged_inputs(dt, H_kv=4, D=64, bs=16, kl=(27, 19, 33)):
    base, pk, pv, tab, lens = 0, [], [], [], []
    blocks = []
    for kln in kl:
        blocks.append((kln + bs - 1) // bs)
    tot, mb = sum(blocks), max(blocks)
    PK = np.zeros((tot, bs, H_kv, D), np.float32)
    PV = np.zeros((tot, bs, H_kv, D), np.float32)
    T = np.full((len(kl), mb), -1, np.int32)
    L = np.zeros((len(kl),), np.int32)
    ks, vs = [], []
    mx.random.seed(701)
    for b, kln in enumerate(kl):
        k = mx.random.normal((1, H_kv, kln, D)).astype(dt)
        v = mx.random.normal((1, H_kv, kln, D)).astype(dt)
        mx.eval(k, v)
        ks.append(k); vs.append(v)
        kn = np.array(k.astype(mx.float32))[0].transpose(1, 0, 2)
        vn = np.array(v.astype(mx.float32))[0].transpose(1, 0, 2)
        L[b] = kln
        for lb in range(blocks[b]):
            T[b, lb] = base + lb
            s0, s1 = lb * bs, min(kln, lb * bs + bs)
            PK[base + lb, :s1 - s0] = kn[s0:s1]; PV[base + lb, :s1 - s0] = vn[s0:s1]
        base += blocks[b]
    return (ks, vs, mx.array(PK).astype(dt), mx.array(PV).astype(dt),
            mx.array(T, dtype=mx.int32), mx.array(L, dtype=mx.int32))


def _varlen_oracle(qs, ks, vs, scale):
    outs = []
    for qb, kb, vb in zip(qs, ks, vs):
        qf = np.array(qb.astype(mx.float32))[0]; kf = np.array(kb.astype(mx.float32))[0]
        vf = np.array(vb.astype(mx.float32))[0]
        H = qf.shape[0]; rep = H // kf.shape[0]; o = np.zeros((H, qf.shape[1], qf.shape[2]))
        for h in range(H):
            ss = (qf[h] @ kf[h // rep].T) * scale; ss -= ss.max(1, keepdims=True)
            e = np.exp(ss); o[h] = (e / e.sum(1, keepdims=True)) @ vf[h // rep]
        outs.append(o)
    return np.concatenate(outs, axis=1)[None]


def _run_paged_varlen(dt):
    Hq, D, bs = 8, 64, 16
    ql = [3, 1, 4]
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(99)
    qs = [mx.random.normal((1, Hq, q, D)).astype(dt) for q in ql]
    mx.eval(*qs)
    ks, vs, pk, pv, tab, lens = _paged_inputs(dt)
    qp = mx.concatenate(qs, axis=2)
    cu = mx.array([0, 3, 4, 8], dtype=mx.int32)
    toff = [0]
    for q in ql:
        toff.append(toff[-1] + (q + 31) // 32)
    O, _ = mfa_paged_varlen_forward(qp, pk, pv, cu, mx.array(toff, dtype=mx.int32),
                                    tab, lens, scale, False, bs)
    mx.eval(O)
    return O, _varlen_oracle(qs, ks, vs, scale)


def _run_paged_gather(dt):
    # public flash_attention_paged → mfa_paged_kv_gather (Nq<=4, S>=256)
    H, Nq, S, D, bs = 4, 2, 320, 64, 16
    nblk = (S + bs - 1) // bs
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(11)
    q = mx.random.normal((1, H, Nq, D)).astype(dt)
    pk = mx.random.normal((nblk, bs, H, D)).astype(dt)
    pv = mx.random.normal((nblk, bs, H, D)).astype(dt)
    tab = mx.array([list(range(nblk))], dtype=mx.int32)
    lens = mx.array([S], dtype=mx.int32)
    mx.eval(q, pk, pv)
    O = mlx_mfa.flash_attention_paged(q, pk, pv, tab, lens, scale=scale,
                                      causal=False, block_size=bs)
    mx.eval(O)
    kf = np.array(pk.astype(mx.float32)).reshape(nblk * bs, H, D)[:S].transpose(1, 0, 2)
    vf = np.array(pv.astype(mx.float32)).reshape(nblk * bs, H, D)[:S].transpose(1, 0, 2)
    qf = np.array(q.astype(mx.float32))[0]
    o2 = np.zeros((H, Nq, D))
    for h in range(H):
        ss = (qf[h] @ kf[h].T) * scale; ss -= ss.max(1, keepdims=True)
        e = np.exp(ss); o2[h] = (e / e.sum(1, keepdims=True)) @ vf[h]
    return O, o2[None]


def _reject_tq_fp32():
    # minimal fp32 q reaches the gate (after the ndim check) → must raise
    q = mx.zeros((1, 8, 8, 64), F32); pk = mx.zeros((1, 16, 4, 24), mx.uint8)
    pv = mx.zeros((1, 16, 4, 64), F16)
    cu = mx.array([0, 8], mx.int32); tile = mx.array([0, 1], mx.int32)
    tab = mx.array([[0]], mx.int32); lens = mx.array([8], mx.int32)
    cent = mx.zeros((8,), F16); ks = mx.zeros((1, 16, 4), F32)
    mfa_paged_varlen_tq_forward(q, pk, pv, cu, tile, tab, lens, cent, ks,
                                0.1, False, 16, 3, False, False, None, None, None)


# ── the matrix: (entry, dtype) -> ("ok", runner, cos_floor) | ("reject", thunk) ─
_CELLS = [
    # paged_varlen raw: fp16/bf16 OK, fp32 reject (Class A silent-wrong)
    ("paged_varlen", "fp16", ("ok", _run_paged_varlen, F16, 0.99)),
    ("paged_varlen", "bf16", ("ok", _run_paged_varlen, BF16, 0.99)),
    ("paged_varlen", "fp32", ("reject", lambda: _run_paged_varlen(F32))),
    # paged_gather via public wrapper: all 3 OK (bf16 is the Class B win)
    ("paged_gather", "fp16", ("ok", _run_paged_gather, F16, 0.99)),
    ("paged_gather", "bf16", ("ok", _run_paged_gather, BF16, 0.99)),
    ("paged_gather", "fp32", ("ok", _run_paged_gather, F32, 0.99)),
    # paged_varlen_tq raw: fp32 reject (Class A)
    ("paged_varlen_tq", "fp32", ("reject", _reject_tq_fp32)),
]


@pytest.mark.parametrize("entry,dn,spec", _CELLS, ids=[f"{e}-{d}" for e, d, _ in _CELLS])
def test_raw_matrix_cell(entry, dn, spec):
    if spec[0] == "reject":
        with pytest.raises(ValueError, match="float16"):
            spec[1]()
        return
    _, runner, dt, floor = spec
    o, ref = runner(dt)
    arr = np.array(o.astype(mx.float32))
    assert np.isfinite(arr).all(), f"{entry}+{dn}: non-finite"
    assert o.dtype == dt, f"{entry}+{dn}: dtype {o.dtype} != {dt} (forced-dtype masking)"
    cos = _cos(o, ref)
    assert cos >= floor, f"{entry}+{dn}: cos {cos:.4f} < {floor}"


# ── bites: the matrix must catch mis-classification + forced-dtype ───────────────
def test_bite_flipping_gated_cell_to_ok_fails():
    # treating paged_varlen+fp32 as an "ok" cell would run the runner, which now
    # raises → an "ok" expectation cannot silently pass.
    with pytest.raises(ValueError):
        _run_paged_varlen(F32)


def test_bite_dtype_match_is_real():
    o, _ = _run_paged_gather(BF16)
    assert o.dtype == BF16 and o.dtype != F16   # a forced-fp16 output would fail the cell
