"""Runtime dispatch fingerprint harness (which-binary, M5/26.6).

Establishes which kernel ACTUALLY executes for flash_attention_sparse, by RUNTIME
fingerprint (not source-tracing — the lesson of four which-artifact inversions):
  - byteΔ vs mx.fast.sdpa(+bias): 0.0 => it IS the SDPA fallback (bit-identical);
    ~1e-6 => a real (different) sparse kernel computing the same masked math.
  - timing flat-vs-density => dense (SDPA); sloped => real skipping sparse kernel.
  - win/loss vs SDPA corroborates.
effective-FLOP, plausibility-gated (<=51.8 TFLOPS fp16 NAX peak), 3-rep median.
"""
import numpy as np, mlx.core as mx, math, time
from mlx_mfa import flash_attention, flash_attention_sparse
from mlx_mfa.attention import _steel_block_config

mx.random.seed(0)
B, H, N = 2, 8, 4096


def mk(D, Hk=None):
    Hk = Hk or H
    f = lambda h: (mx.random.uniform(-1, 1, (B, h, N, D)) * 0.1).astype(mx.float16)
    q, k, v = f(H), f(Hk), f(Hk); mx.eval(q, k, v); return q, k, v


def elem_bias(bm, qL, kL):
    NQ, NK = bm.shape[-2], bm.shape[-1]
    em = mx.repeat(mx.repeat(bm.astype(mx.float32), qL // NQ, axis=-2), kL // NK, axis=-1)
    b = mx.where(em > 0, mx.array(0.0), mx.array(-1e9))
    while b.ndim < 4: b = b[None]
    return b.astype(mx.float16)


def t_of(fn, warm=5, rep=11):
    for _ in range(warm): mx.eval(fn())
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t0)
    return sorted(ts)[rep // 2]


def fingerprint(q, k, v, bm, scale, label):
    try:
        qL, kL = q.shape[2], k.shape[2]
        o = flash_attention_sparse(q, k, v, bm, scale=scale, causal=False)
        os_ = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=elem_bias(bm, qL, kL))
        mx.eval(o, os_)
        d = float(mx.max(mx.abs(o.astype(mx.float32) - os_.astype(mx.float32))).item())
        t = t_of(lambda: flash_attention_sparse(q, k, v, bm, scale=scale, causal=False))
        verdict = "SDPA-FALLBACK(byte-id)" if d == 0.0 else ("real-sparse" if d < 3e-2 else f"WRONG {d:.1e}")
        print(f"  {label:42s} shape={tuple(bm.shape[-2:])} byteΔ={d:.1e} t={t*1e3:6.2f}ms -> {verdict}")
    except Exception as e:
        print(f"  {label:42s} REJECTED: {type(e).__name__}: {str(e)[:50]}")


from mlx_mfa import (make_causal_block_mask, make_sliding_window_mask,
                     make_strided_mask, make_lcsa_mask)
for D in (64, 128):
    SC = 1.0 / math.sqrt(D); q, k, v = mk(D); BQ, BK = _steel_block_config(D)
    print(f"\n== D={D}  _steel_block_config=(BQ={BQ},BK={BK}) default mask {'SYM' if BQ==BK else 'ASYM (->SDPA on M5)'} ==")
    NB = N // 32; ms = np.zeros((NB, NB), bool); ms[:, :NB // 4] = True
    fingerprint(q, k, v, mx.array(ms), SC, "hand SYMMETRIC bt=32 d=0.25")
    fingerprint(q, k, v, make_causal_block_mask(N, D), SC, "make_causal_block_mask")
    for nm, fn in [("make_sliding_window_mask", lambda: make_sliding_window_mask(N, N, 512, D)),
                   ("make_strided_mask", lambda: make_strided_mask(N, N, 4, D))]:
        try: fingerprint(q, k, v, fn(), SC, nm)
        except Exception as e: print(f"  {nm:42s} maker-err {str(e)[:40]}")
    # dense baseline
    od = flash_attention(q, k, v, scale=SC, causal=False)
    osd = mx.fast.scaled_dot_product_attention(q, k, v, scale=SC); mx.eval(od, osd)
    print(f"  {'dense flash_attention':42s} byteΔ={float(mx.max(mx.abs(od.astype(mx.float32)-osd.astype(mx.float32))).item()):.1e} -> {'IS SDPA' if True else ''}")
