"""Volet P10 — CI-locked dtype × backend capability matrix (finite + correct).

P8 built this matrix with a "does it run?" probe and mis-marked `turboquant+fp32`
as "OK (fallback)" — pre-P9 tq_decode FORCED fp16 output, so the cell *ran* while
silently emitting fp16 (forced-dtype masking). P9's explicit `_msl_type` turned
that into a loud-late failure (and the fused fallback emits NON-FINITE fp32). P10
re-verifies every cell against the REAL criteria — runs **and** finite **and**
output-dtype-matches-request **and** correct vs an independent fp32 SDPA reference
— gates `turboquant+fp32` at construction, and locks the whole matrix here so it
cannot silently rot.

Verified matrix (M5 / MLX 0.31.2):
    backend     fp16            bf16            fp32
    dense       OK (cos~1.0)    OK              OK (SDPA)
    paged       OK              OK              REJECT@ctor
    sage        OK              OK              REJECT@ctor
    turboquant  OK (cos~0.98)   OK (cos~0.98)   REJECT@ctor   <- P10 gate
    hybrid      OK              OK              OK
"""
import warnings
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa

Hq, Hkv, D, S = 8, 8, 64, 48
_SCALE = 1.0 / np.sqrt(D)

# backend config + the cosine floor vs the fp32 reference (quantizing backends are
# lossy → lower floor, but still high; exact backends ~1.0).
_BACKENDS = {
    "dense":      (dict(backend="dense"), 0.99),
    "paged":      (dict(backend="paged"), 0.99),
    "sage":       (dict(backend="sage", quantized_kv=True), 0.99),
    "turboquant": (dict(backend="paged", turboquant=True), 0.90),
    "hybrid":     (dict(backend="dense", hybrid_cache=True, hybrid_enable_offload=True), 0.99),
}
_DTYPES = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}

# The locked outcome per cell: "ok" = runs finite + dtype-correct + ref-correct;
# "reject" = raises at construction.  fp16/bf16-only backends reject fp32.
_MATRIX = {
    (b, d): ("reject" if (d == "fp32" and b in ("paged", "sage", "turboquant")) else "ok")
    for b in _BACKENDS for d in _DTYPES
}


def _fixtures(dt):
    mx.random.seed(0)
    kh = mx.random.normal((1, Hkv, S, D)).astype(dt)
    vh = mx.random.normal((1, Hkv, S, D)).astype(dt)
    qp = mx.random.normal((1, Hq, S, D)).astype(dt)
    q = mx.random.normal((1, Hq, 1, D)).astype(dt)
    kn = mx.zeros((1, Hkv, 1, D), dt)
    vn = mx.zeros((1, Hkv, 1, D), dt)
    mx.eval(kh, vh, qp, q, kn, vn)
    return kh, vh, qp, q, kn, vn


def _reference(kh, vh, q, kn, vn):
    # independent fp32 SDPA over the exact decode history [history ++ new token]
    k = mx.concatenate([kh.astype(mx.float32), kn.astype(mx.float32)], axis=2)
    v = mx.concatenate([vh.astype(mx.float32), vn.astype(mx.float32)], axis=2)
    ref = mx.fast.scaled_dot_product_attention(q.astype(mx.float32), k, v, scale=_SCALE)
    mx.eval(ref)
    return np.array(ref).reshape(-1)


def _cos(o, refn):
    a = np.array(o.astype(mx.float32)).reshape(-1)
    return float(a @ refn / (np.linalg.norm(a) * np.linalg.norm(refn) + 1e-9))


def _run_cell(cfg, dt):
    rt = mlx_mfa.create_decode_runtime(
        B=1, H_q=Hq, H_kv=Hkv, D=D, max_seq_len=128, dtype=dt,
        num_blocks=16, block_size=16, **cfg)
    kh, vh, qp, q, kn, vn = _fixtures(dt)
    rt.prefill(qp, kh, vh)
    o = rt.step(q, kn, vn)
    mx.eval(o)
    return o, _reference(kh, vh, q, kn, vn)


_CELLS = [(b, d) for b in _BACKENDS for d in _DTYPES]


@pytest.mark.parametrize("b,d", _CELLS, ids=[f"{b}-{d}" for b, d in _CELLS])
def test_matrix_cell(b, d):
    cfg, cos_floor = _BACKENDS[b]
    dt = _DTYPES[d]
    expect = _MATRIX[(b, d)]
    if expect == "reject":
        # gated cell: must raise a clear capability error AT CONSTRUCTION
        with pytest.raises(ValueError, match="float16"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _run_cell(cfg, dt)
        return
    # end-to-end cell: finite + dtype-matches-request + reference-correct
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, refn = _run_cell(cfg, dt)
    arr = np.array(o.astype(mx.float32))
    assert np.isfinite(arr).all(), f"{b}+{d}: non-finite output"
    assert o.dtype == dt, f"{b}+{d}: output dtype {o.dtype} != requested {dt} (forced-dtype masking)"
    cos = _cos(o, refn)
    assert cos >= cos_floor, f"{b}+{d}: cos {cos:.3f} < floor {cos_floor}"


# ── the specific P10 repro: turboquant+fp32 rejects at construction ──────────────
def test_turboquant_fp32_rejected_at_construction():
    with pytest.raises(ValueError, match="float16"):
        mlx_mfa.create_decode_runtime(
            B=1, H_q=Hq, H_kv=Hkv, D=D, max_seq_len=128, dtype=mx.float32,
            num_blocks=16, block_size=16, backend="paged", turboquant=True)


# ── bites: the matrix test must catch mis-classification and forced-dtype ────────
def test_bite_flipping_gated_cell_to_ok_would_fail():
    # if someone claims turboquant+fp32 is "ok", the end-to-end branch would run
    # construction → which raises → so an "ok" expectation cannot silently pass.
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _run_cell(_BACKENDS["turboquant"][0], mx.float32)


def test_bite_dtype_mismatch_is_caught():
    # the dtype-equality assertion is real: a bf16 run must NOT report fp16 (the
    # P8 masking). Prove the check distinguishes dtypes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, _ = _run_cell(_BACKENDS["dense"][0], mx.bfloat16)
    assert o.dtype == mx.bfloat16
    assert o.dtype != mx.float16          # the masking dtype would fail the cell test
