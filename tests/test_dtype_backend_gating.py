"""Volet P8 — dtype × backend gated at construction (fp32 late-failure class).

P7 fixed the LocalHost fp32 over-restriction; re-run #5 found a SECOND instance of
the same class: a fp16/bf16-only backend (paged: `mfa_scatter_kv`; sage:
`mfa_quantize_per_block`) accepted `dtype=fp32` at construction (off-spec
warning) then failed LATE deep in prefill/step. P8 gates (backend, dtype) at
construction: every combo either runs end-to-end or is rejected up-front — no
construct-run-then-fail-deep.

Verified matrix (M5, MLX 0.31.2):
    backend     fp16   bf16   fp32
    dense       OK     OK     OK (SDPA)
    paged       OK     OK     REJECT@ctor   (mfa_scatter_kv fp16/bf16-only)
    sage        OK     OK     REJECT@ctor   (mfa_quantize_per_block fp16/bf16-only)
    turboquant  OK     [*]    OK (fallback)
    hybrid      OK     OK     OK (=dense base, byte-store offload dtype-agnostic)

[*] turboquant+bf16 Nq=1 decode fails late in `tq_decode` (V-gather hardcodes
    `half vout_v`). This is a SEPARATE, path-specific finding (prefill/Nq>1 works,
    so it is NOT construction-gateable without over-rejecting a working combo) —
    FLAGGED for Marco, not gated here.
"""
import warnings
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa


def _mk(n, dt, d=64):
    a = mx.random.normal((1, 8, n, d)).astype(dt)
    mx.eval(a)
    return a


def _rt(dt, **cfg):
    return mlx_mfa.create_decode_runtime(
        B=1, H_q=8, H_kv=8, D=64, max_seq_len=128, dtype=dt,
        num_blocks=16, block_size=16, **cfg)


# ── fp32 on fp16/bf16-only backends → rejected AT CONSTRUCTION (not late) ─────────
@pytest.mark.parametrize("cfg", [
    dict(backend="paged"),
    dict(backend="sage", quantized_kv=True),
])
def test_fp32_rejected_at_construction(cfg):
    with pytest.raises(ValueError, match="float16"):
        _rt(mx.float32, **cfg)


def test_paged_fp32_message_names_backend_and_dtypes():
    with pytest.raises(ValueError) as ei:
        _rt(mx.float32, backend="paged")
    msg = str(ei.value)
    assert "paged" in msg and "float16" in msg and "bfloat16" in msg


# ── direct context constructor is gated too (not only the runtime factory) ───────
def test_direct_paged_context_fp32_rejected():
    from mlx_mfa.inference import PagedInferenceContext, SageInferenceContext
    with pytest.raises(ValueError, match="float16"):
        PagedInferenceContext(num_blocks=16, block_size=16, H_kv=8, D=64, dtype=mx.float32)
    with pytest.raises(ValueError, match="float16"):
        SageInferenceContext(B=1, H_kv=8, D=64, max_seq_len=128, dtype=mx.float32)


# ── supported combos still run end-to-end (no over-rejection) ────────────────────
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16, mx.float32])
def test_dense_runs_all_dtypes(dt):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rt = _rt(dt, backend="dense")
        rt.prefill(_mk(16, dt), _mk(16, dt), _mk(16, dt))
        o = rt.step(_mk(1, dt), _mk(1, dt), _mk(1, dt))
        mx.eval(o)
        assert bool(np.isfinite(np.array(o.astype(mx.float32))).all())


def test_hybrid_fp32_offload_still_runs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rt = _rt(mx.float32, backend="dense", hybrid_cache=True, hybrid_enable_offload=True)
        rt.prefill(_mk(16, mx.float32), _mk(16, mx.float32), _mk(16, mx.float32))
        rt.hybrid_offload([0]); rt.hybrid_reload([0])
        o = rt.step(_mk(1, mx.float32), _mk(1, mx.float32), _mk(1, mx.float32))
        mx.eval(o)
        assert bool(np.isfinite(np.array(o.astype(mx.float32))).all())


@pytest.mark.parametrize("cfg", [
    dict(backend="paged"),
    dict(backend="sage", quantized_kv=True),
])
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
def test_spec_dtypes_run_on_all_backends(cfg, dt):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rt = _rt(dt, **cfg)
        rt.prefill(_mk(16, dt), _mk(16, dt), _mk(16, dt))
        o = rt.step(_mk(1, dt), _mk(1, dt), _mk(1, dt))
        mx.eval(o)
        assert bool(np.isfinite(np.array(o.astype(mx.float32))).all())
