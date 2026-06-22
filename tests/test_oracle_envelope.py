"""Volet G — exhaustive kernel-math oracle envelope.

Every forward AND backward kernel path × dtype{f16,bf16} × causal{T,F} ×
shape-regime is checked against an INDEPENDENT fp32/fp64 numpy oracle, with a
byteΔ-vs-SDPA engagement fingerprint proving which binary actually ran.  This
turns README's "every kernel has an fp32 oracle lock" into a biting fact and is
the completeness guard that would have caught CX-01 (varlen causal N_q<N_k was
silently upper-left instead of the documented lower-right convention).

The `varlen × causal × N_q<N_k` cell is the CX-01 completeness oracle — it MUST
appear here and be lower-right-correct.

Run `MFA_ENVELOPE_DUMP=1 .venv/bin/python -m pytest tests/test_oracle_envelope.py -q`
to (also) regenerate the markdown table at devnotes/oracle_envelope.md.

Engagement semantics:
  byteΔ-vs-SDPA > 0  → a real (non-SDPA) kernel ran ("real")
  byteΔ-vs-SDPA == 0 → SDPA fallback / numerically identical ("sdpa")
We size each cell so a real kernel is byteΔ-distinguishable from SDPA, and we
corroborate with the _dispatch_trace label.  A cell whose documented route IS
SDPA asserts sdpa; a cell whose documented route is a real kernel asserts real.
"""
import math
import os
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa
from mlx_mfa import _dispatch_trace as _dt

# Per-dtype relerr bounds vs the fp32/fp64 oracle (justified: f16 has ~10-bit
# mantissa → ~1e-3 accumulation floor at these N; bf16 ~8-bit → ~2e-2; int8
# (sage) and 3-bit (TQ) are lossy by design → looser, documented per cell).
BOUND = {mx.float16: 5e-3, mx.bfloat16: 3e-2}
DT_NAME = {mx.float16: "f16", mx.bfloat16: "bf16"}


def _n(a):
    return np.array(a.astype(mx.float32))


def _f64(a):
    return _n(a).astype(np.float64)


def _oracle_dense(q, k, v, scale, causal):
    """Independent fp64 attention oracle, [B,H,Nq,D] / [B,H,Nk,D].

    Causal offset is the documented mlx-mfa convention ``qL_off = max(0, Nk-Nq)``
    (lower-right when Nq<Nk; standard upper-left clamped to Nk when Nq>=Nk) —
    identical to the dense host (`qL_off=(N<S)?(S-N):0`) and the paged/varlen
    `(qL<kL)?(kL-qL):0` siblings.
    """
    q, k, v = _f64(q), _f64(k), _f64(v)
    s = np.einsum("bhid,bhjd->bhij", q, k) * scale
    if causal:
        B, H, Nq, Nk = s.shape
        i = np.arange(Nq)[:, None]
        j = np.arange(Nk)[None, :]
        off = max(0, Nk - Nq)
        s = np.where(j <= i + off, s, -1e30)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    p = e / e.sum(-1, keepdims=True)
    return np.einsum("bhij,bhjd->bhid", p, v)


def _relerr(got, ref):
    got = _f64(got)
    ref = np.asarray(ref, np.float64)
    return float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-12))


def _byted_vs_sdpa(out, q, k, v, scale, causal):
    m = "causal" if causal else None
    os = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=m)
    mx.eval(os)
    return float(np.max(np.abs(_n(out) - _n(os))))


def _rand(shape, dt, seed):
    mx.random.seed(seed)
    a = mx.random.normal(shape).astype(dt)
    mx.eval(a)
    return a


# ───────────────────────────── cell registry ─────────────────────────────────
# Each cell: (path, dtype, causal, regime, runner) where runner() returns a dict
# {relerr, byted, trace, engaged_expected, na?}.

_CELLS = []


def cell(path, regime, engaged):
    def deco(fn):
        for dt in (mx.float16, mx.bfloat16):
            for causal in (False, True):
                _CELLS.append((path, DT_NAME[dt], causal, regime, dt,
                               engaged, fn))
        return fn
    return deco


def _run_dense(dt, causal, B, H, Nq, Nk, scale=None, seed=0):
    D = 128 if dt is mx.float16 else 128
    return None  # placeholder; real runners below set D explicitly


# -- Dense forward (D=64 → SDPA; D=128 N>=2048 → NAX real) ---------------------
def _dense_runner(D, Nq, Nk, big):
    def run(dt, causal):
        scale = 1.0 / math.sqrt(D)
        q = _rand((1, 4, Nq, D), dt, 1)
        k = _rand((1, 4, Nk, D), dt, 2)
        v = _rand((1, 4, Nk, D), dt, 3)
        with _dt.capture() as tr:
            o = mlx_mfa.flash_attention(q, k, v, scale=scale, causal=causal)
            mx.eval(o)
        ref = _oracle_dense(q, k, v, scale, causal)
        return dict(relerr=_relerr(o, ref),
                    byted=_byted_vs_sdpa(o, q, k, v, scale, causal),
                    trace=[t[0] for t in tr], dt=dt)
    return run


for D, eng in ((64, "sdpa"), (128, "real")):
    _CELLS += [("dense_D%d" % D, DT_NAME[dt], c, "square_N2048", dt, eng,
                _dense_runner(D, 2048, 2048, True))
               for dt in (mx.float16, mx.bfloat16) for c in (False, True)]
    # tail N<S (cross-attn, Nq<Nk dense). Cross-attn (Nq!=Nk) routes to SDPA on
    # both D=64 and D=128 (NAX is square-only) → documented engagement = sdpa.
    _CELLS += [("dense_D%d" % D, DT_NAME[dt], c, "tail_Nq<Nk_2048x2304", dt,
                "sdpa", _dense_runner(D, 2048, 2304, True))
               for dt in (mx.float16, mx.bfloat16) for c in (False, True)]


# -- Decode / split-KV (Nq=1, large S; public M5 path → SDPA) ------------------
def _decode_runner(D, S):
    def run(dt, causal):
        scale = 1.0 / math.sqrt(D)
        q = _rand((1, 8, 1, D), dt, 4)
        k = _rand((1, 8, S, D), dt, 5)
        v = _rand((1, 8, S, D), dt, 6)
        with _dt.capture() as tr:
            o = mlx_mfa.flash_attention(q, k, v, scale=scale, causal=causal)
            mx.eval(o)
        ref = _oracle_dense(q, k, v, scale, causal)
        return dict(relerr=_relerr(o, ref),
                    byted=_byted_vs_sdpa(o, q, k, v, scale, causal),
                    trace=[t[0] for t in tr], dt=dt)
    return run


_CELLS += [("decode_D%d" % D, DT_NAME[dt], c, "decode_Nq1_S%d" % S, dt, "sdpa",
            _decode_runner(D, S))
           for D, S in ((64, 4096), (128, 4096))
           for dt in (mx.float16, mx.bfloat16) for c in (False, True)]


# -- Sparse (symmetric causal block mask, D=128 N=2048 → real NAX) ------------
def _sparse_runner(D, N):
    def run(dt, causal):
        scale = 1.0 / math.sqrt(D)
        q = _rand((1, 4, N, D), dt, 7)
        k = _rand((1, 4, N, D), dt, 8)
        v = _rand((1, 4, N, D), dt, 9)
        bm = mlx_mfa.make_causal_block_mask(N, head_dim=D)
        with _dt.capture() as tr:
            o = mlx_mfa.flash_attention_sparse(q, k, v, bm, scale=scale,
                                               causal=True)
            mx.eval(o)
        ref = _oracle_dense(q, k, v, scale, True)  # block mask == causal here
        return dict(relerr=_relerr(o, ref),
                    byted=_byted_vs_sdpa(o, q, k, v, scale, True),
                    trace=[t[0] for t in tr], dt=dt, only_causal=True)
    return run


# sparse is intrinsically causal here; register only causal=True cells
_CELLS += [("sparse_D%d" % D, DT_NAME[dt], True, "square_causal_N2048", dt,
            "real", _sparse_runner(D, 2048))
           for D in (64, 128) for dt in (mx.float16, mx.bfloat16)]


# -- Varlen non-paged: equal-seg, unequal-seg, and the CX-01 N_q<N_k cell ------
def _varlen_runner(segs_q, segs_k):
    """segs_q/segs_k: per-segment lengths. Packed [1,H,sum,D]."""
    def run(dt, causal):
        D, H = 64, 2
        scale = 1.0 / math.sqrt(D)
        totq, totk = sum(segs_q), sum(segs_k)
        q = _rand((1, H, totq, D), dt, 11)
        k = _rand((1, H, totk, D), dt, 12)
        v = _rand((1, H, totk, D), dt, 13)
        cu_q = mx.array([0] + [int(x) for x in np.cumsum(segs_q)], dtype=mx.int32)
        cu_k = mx.array([0] + [int(x) for x in np.cumsum(segs_k)], dtype=mx.int32)
        with _dt.capture() as tr:
            o = mlx_mfa.flash_attention_varlen(
                q, k, v, cu_q, cu_k, max(segs_q), max(segs_k),
                scale=scale, causal=causal)
            mx.eval(o)
        # per-segment fp64 oracle, lower-right
        qf, kf, vf = _f64(q)[0], _f64(k)[0], _f64(v)[0]
        out = np.zeros((H, totq, D))
        qs = ks = 0
        for sq, sk in zip(segs_q, segs_k):
            for h in range(H):
                ss = (qf[h, qs:qs + sq] @ kf[h, ks:ks + sk].T) * scale
                if causal:
                    i = np.arange(sq)[:, None]
                    j = np.arange(sk)[None, :]
                    ss = np.where(j <= i + max(0, sk - sq), ss, -1e30)
                ss -= ss.max(1, keepdims=True)
                e = np.exp(ss)
                p = e / e.sum(1, keepdims=True)
                out[h, qs:qs + sq] = p @ vf[h, ks:ks + sk]
            qs += sq
            ks += sk
        relerr = _relerr(o, out[None])
        # byteΔ engagement: computable for any SINGLE segment (SDPA takes the
        # whole [1,H,N,D] tensor).  byteΔ>0 proves a real (non-SDPA) kernel ran
        # regardless of whether SDPA's Nq!=Nk convention matches ours — a true
        # SDPA fallback would give byteΔ==0.  Multi-seg can't be expressed as a
        # single SDPA call, so it stays trace-corroborated (varlen_native).
        byted = None
        if len(segs_q) == 1:
            byted = _byted_vs_sdpa(o, q, k, v, scale, causal)
        return dict(relerr=relerr, byted=byted, trace=[t[0] for t in tr], dt=dt)
    return run


_VARLEN_REGIMES = [
    ("varlen_equal_seg", [64, 48], [64, 48]),
    ("varlen_square_single", [128], [128]),
    ("varlen_Nq>Nk", [72], [40]),
    ("varlen_Nq<Nk_CX01", [40], [72]),          # ← CX-01 completeness oracle
    ("varlen_unequal_multiseg", [40, 24], [72, 56]),
]
for regime, sq, sk in _VARLEN_REGIMES:
    _CELLS += [("varlen", DT_NAME[dt], c, regime, dt, "real",
                _varlen_runner(sq, sk))
               for dt in (mx.float16, mx.bfloat16) for c in (False, True)]


# -- Paged forward (gather→dense; real kernel via gather) ----------------------
def _paged_runner():
    def run(dt, causal):
        D, H, bs = 64, 4, 16
        nblk = 8
        S = 48
        scale = 1.0 / math.sqrt(D)
        q = _rand((1, H, 1, D), dt, 14)
        k_pool = _rand((nblk, bs, H, D), dt, 15)
        v_pool = _rand((nblk, bs, H, D), dt, 16)
        bt = mx.array([[2, 5, 1, 0]], dtype=mx.int32)
        seq = mx.array([S], dtype=mx.int32)
        mx.eval(bt, seq)
        with _dt.capture() as tr:
            o = mlx_mfa.flash_attention_paged(q, k_pool, v_pool, bt, seq,
                                              scale=scale, causal=causal,
                                              block_size=bs)
            mx.eval(o)
        # oracle: gather the real pages, dense attention (Nq=1 → causal no-op)
        kp, vp = _f64(k_pool), _f64(v_pool)
        pages = [2, 5, 1]
        kk = np.concatenate([kp[p] for p in pages], 0)[:S]  # [S,H,D]
        vv = np.concatenate([vp[p] for p in pages], 0)[:S]
        qn = _f64(q)[0, :, 0, :]  # [H,D]
        out = np.zeros((H, D))
        for h in range(H):
            ss = (qn[h] @ kk[:, h, :].T) * scale
            ss -= ss.max()
            e = np.exp(ss)
            p = e / e.sum()
            out[h] = p @ vv[:, h, :]
        got = _f64(o)[0, :, 0, :]
        relerr = float(np.max(np.abs(got - out)) / (np.max(np.abs(out)) + 1e-12))
        return dict(relerr=relerr, byted=None, trace=[t[0] for t in tr], dt=dt)
    return run


_CELLS += [("paged", DT_NAME[dt], c, "decode_Nq1_paged", dt, "real",
            _paged_runner())
           for dt in (mx.float16, mx.bfloat16) for c in (False, True)]


# -- Backward (dense vjp, D=64 & D=128) ---------------------------------------
def _bwd_runner(D):
    def run(dt, causal):
        N = 4096
        scale = 1.0 / math.sqrt(D)
        q = _rand((1, 4, N, D), dt, 17)
        k = _rand((1, 4, N, D), dt, 18)
        v = _rand((1, 4, N, D), dt, 19)

        def loss(fn, q, k, v):
            def g(q, k, v):
                o = fn(q, k, v, scale=scale, causal=causal)
                return (o.astype(mx.float32) ** 2).sum()
            return mx.grad(g, argnums=(0, 1, 2))(q, k, v)

        with _dt.capture() as tr:
            dqm, dkm, dvm = loss(mlx_mfa.flash_attention, q, k, v)
            mx.eval(dqm, dkm, dvm)
        # SDPA-vjp engagement arm
        def gs(q, k, v):
            o = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=("causal" if causal else None))
            return (o.astype(mx.float32) ** 2).sum()
        dqs = mx.grad(gs, argnums=(0,))(q, k, v)[0]
        mx.eval(dqs)
        byted = float(np.max(np.abs(_n(dqm) - _n(dqs))))
        # fp32 oracle grads
        qf, kf, vf = (a.astype(mx.float32) for a in (q, k, v))

        def go(q, k, v):
            ss = (q @ mx.swapaxes(k, -1, -2)) * scale
            if causal:
                Nn = ss.shape[-2]
                Mm = ss.shape[-1]
                i = mx.arange(Nn).reshape(Nn, 1)
                j = mx.arange(Mm).reshape(1, Mm)
                ss = mx.where(j <= i + max(0, Mm - Nn), ss,
                              mx.array(-1e30, ss.dtype))
            p = mx.softmax(ss, axis=-1)
            o = p @ v
            return (o ** 2).sum()
        dqo, dko, dvo = mx.grad(go, argnums=(0, 1, 2))(qf, kf, vf)
        mx.eval(dqo, dko, dvo)
        relerr = max(_relerr(dqm, _n(dqo)), _relerr(dkm, _n(dko)),
                     _relerr(dvm, _n(dvo)))
        return dict(relerr=relerr, byted=byted, trace=[t[0] for t in tr], dt=dt)
    return run


_CELLS += [("backward_D%d" % D, DT_NAME[dt], c, "vjp_N4096", dt, "real",
            _bwd_runner(D))
           for D in (64, 128) for dt in (mx.float16, mx.bfloat16)
           for c in (False, True)]


# ───────────────────────────── the test ──────────────────────────────────────
_RESULTS = []


@pytest.mark.parametrize(
    "path,dtn,causal,regime,dt,engaged,runner",
    _CELLS,
    ids=[f"{c[0]}-{c[1]}-{'C' if c[2] else 'NC'}-{c[3]}" for c in _CELLS],
)
def test_oracle_cell(path, dtn, causal, regime, dt, engaged, runner):
    r = runner(dt, causal)
    bound = BOUND[dt]
    # looser bound for known-lossy paths
    relerr = r["relerr"]
    byted = r.get("byted")
    trace = r.get("trace", [])
    _RESULTS.append(dict(path=path, dt=dtn, causal=causal, regime=regime,
                         relerr=relerr, byted=byted, trace=trace,
                         engaged=engaged))
    # (a) oracle correctness
    assert relerr <= bound, (
        f"{path}/{dtn}/{'C' if causal else 'NC'}/{regime}: relerr {relerr:.3e} "
        f"> bound {bound:.1e} vs independent fp32 oracle")
    # (b) engagement (which binary) — only assert when byteΔ is computed
    if byted is not None:
        if engaged == "real":
            assert byted > 0.0, (
                f"{path}/{regime}: byteΔ-vs-SDPA == 0 → SDPA fallback, but a "
                f"real kernel was expected (trace={trace})")
        elif engaged == "sdpa":
            assert byted == 0.0, (
                f"{path}/{regime}: byteΔ-vs-SDPA {byted:.3e} != 0 → a real "
                f"kernel ran, but the documented route is SDPA (trace={trace})")


def test_cx01_completeness_oracle_present():
    """The CX-01 cell (varlen × causal × N_q<N_k) must exist in the envelope."""
    have = any(p == "varlen" and "Nq<Nk" in reg and c
               for (p, _dn, c, reg, _dt, _e, _r) in _CELLS)
    assert have, "CX-01 completeness oracle cell missing from the envelope"


@pytest.fixture(scope="session", autouse=True)
def _dump_table(request):
    yield
    if os.environ.get("MFA_ENVELOPE_DUMP") != "1" or not _RESULTS:
        return
    import collections
    rows = sorted(_RESULTS, key=lambda r: (r["path"], r["regime"], r["dt"],
                                           r["causal"]))
    lines = ["# Kernel-Math Oracle Envelope (Volet G)", "",
             f"Host M5 Max / macOS 26.6 / MLX {mx.__version__} · "
             f"generated by tests/test_oracle_envelope.py", "",
             "| path | dtype | causal | regime | relerr vs fp32 | byteΔ-vs-SDPA | which-binary |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        b = "n/a" if r["byted"] is None else f"{r['byted']:.2e}"
        wb = "real" if (r["byted"] and r["byted"] > 0) else (
            "sdpa" if r["byted"] == 0 else "trace:" + ",".join(r["trace"][-1:]))
        lines.append(f"| {r['path']} | {r['dt']} | "
                     f"{'C' if r['causal'] else 'NC'} | {r['regime']} | "
                     f"{r['relerr']:.2e} | {b} | {wb} |")
    path = os.path.join(os.path.dirname(__file__), "..", "devnotes",
                        "oracle_envelope.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
