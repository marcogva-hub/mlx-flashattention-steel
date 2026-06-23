"""Volet P9 — native bf16 in the tq_decode V-gather + K-dequant kernels.

P8's matrix surfaced the last late cell: `turboquant` + `bf16` Nq=1 decode failed
to compile — both tq_decode kernels hardcoded `half` (`half vout_v`, `half
kout_v`), so a bf16 V-pool / bf16 output won't compile. bf16 is an in-spec dtype;
P9 templates both kernels on the cache dtype (`bfloat16_t` for bf16, byte-identical
`half` for fp16) — native bf16, no lossy bf16→fp16→bf16 round-trip. The P1 OOB
bounds guard is preserved in both dtypes.
"""
import os
import warnings
import numpy as np
import mlx.core as mx
import pytest
from mlx_mfa.tq_decode import (
    _get_v_gather_kernel, _get_k_dequant_kernel, _HEADER)
from mlx_mfa.turboquant import _get_centroids
from mlx_mfa.inference import TurboQuantPagedInferenceContext

D, Hkv, bs, bits, nb = 64, 4, 16, 3, 8
_PD = (D // 32) * 12


def _mk(n, dt):
    a = mx.random.normal((1, 8, n, 64)).astype(dt)
    mx.eval(a)
    return a


def _unpack3(byts, d):
    g, lane = d // 32, d % 32
    bl, bb = lane // 8, lane % 8
    base = g * 12
    b0, b1, b2 = byts[base + bl], byts[base + 4 + bl], byts[base + 8 + bl]
    return ((b0 >> bb) & 1) | (((b1 >> bb) & 1) << 1) | (((b2 >> bb) & 1) << 2)


# ── 1. the P8-flagged repro: turboquant + bf16 Nq=1 decode runs end-to-end ───────
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
def test_tq_bf16_decode_end_to_end(dt):
    c = TurboQuantPagedInferenceContext(num_blocks=16, block_size=16, H_kv=8, D=64,
                                        dtype=dt, tq_bits=3)
    c.prefill(_mk(16, dt), _mk(16, dt), _mk(16, dt), seq_id=0)
    o = c.step(_mk(1, dt), _mk(1, dt), _mk(1, dt), seq_id=0)
    mx.eval(o)
    assert o.dtype == dt
    assert bool(np.isfinite(np.array(o.astype(mx.float32))).all())


# ── 2. V-gather is an EXACT copy of the pool value (both dtypes) ─────────────────
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
def test_v_gather_exact(dt):
    mx.random.seed(0)
    vpool = mx.random.normal((nb, bs, Hkv, D)).astype(dt); mx.eval(vpool)
    S = bs * 2
    bt = mx.array([3, 5], dtype=mx.int32)
    params = mx.array([S, nb, 2], dtype=mx.int32)
    tot = S * Hkv * D; grid = ((tot + 255) // 256 * 256, 1, 1)
    V = _get_v_gather_kernel(D, Hkv, bs, dt)(
        inputs=[vpool, bt, params], output_shapes=[(1, Hkv, S, D)],
        output_dtypes=[dt], grid=grid, threadgroup=(256, 1, 1))[0]
    mx.eval(V)
    vn = np.array(V.astype(mx.float32))[0]
    vp = np.array(vpool.astype(mx.float32))
    orc = np.zeros((Hkv, S, D), np.float32)
    for s in range(S):
        blk, tok = s // bs, s % bs
        phys = int(bt[blk]) if blk < 2 else -1
        if 0 <= phys < nb:
            orc[:, s, :] = vp[phys, tok, :, :]
    assert np.max(np.abs(vn - orc)) == 0.0          # pure gather → exact


# ── 3. K-dequant matches the fp32 dequant oracle cast to out dtype (both) ─────────
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
def test_k_dequant_oracle(dt):
    mx.random.seed(1)
    _, cf = _get_centroids(bits)
    cent = cf.astype(mx.float16); mx.eval(cent)
    cnp = np.array(cent.astype(mx.float32))
    kpool = mx.random.randint(0, 256, (nb, bs, Hkv, _PD)).astype(mx.uint8)
    kscl = mx.random.uniform(0.5, 1.5, (nb, bs, Hkv)).astype(mx.float32)
    mx.eval(kpool, kscl)
    S = bs
    bt = mx.array([2], dtype=mx.int32)
    params = mx.array([S, nb, 1], dtype=mx.int32)
    tot = S * Hkv * D; grid = ((tot + 255) // 256 * 256, 1, 1)
    K = _get_k_dequant_kernel(D, Hkv, bs, bits, dt)(
        inputs=[kpool, kscl, cent, bt, params], output_shapes=[(1, Hkv, S, D)],
        output_dtypes=[dt], grid=grid, threadgroup=(256, 1, 1))[0]
    mx.eval(K)
    kn = np.array(K.astype(mx.float32))[0]
    kp, ks = np.array(kpool), np.array(kscl)
    orc = np.zeros((Hkv, S, D), np.float32)
    for s in range(S):
        tok = s % bs
        for h in range(Hkv):
            for d in range(D):
                orc[h, s, d] = cnp[_unpack3(kp[2, tok, h], d)] * ks[2, tok, h]
    orc_dt = np.array(mx.array(orc).astype(dt).astype(mx.float32))
    assert np.max(np.abs(kn - orc_dt)) == 0.0       # native dequant, no extra loss


# ── 4. fp16 path is byteΔ=0 vs the reconstructed pre-P9 hardcoded-half kernel ────
def test_fp16_bytedelta_zero_vs_pre_p9():
    def old_v():
        src = f"""
  const uint gid = thread_position_in_grid.x;
  const int S = params[0]; const int num_blocks = params[1]; const int n_blk = params[2];
  const uint total = (uint)S * {Hkv} * {D}; if (gid >= total) return;
  const int d=(int)(gid%{D}); const int h=(int)((gid/{D})%{Hkv}); const int s=(int)(gid/({D}*{Hkv}));
  const int blk=s/{bs}; const int tok=s%{bs}; half vout_v=(half)0;
  if (blk<n_blk){{ const int phys=block_table[blk]; if(phys>=0&&phys<num_blocks){{
    vout_v=v_pool[(ulong)phys*{bs*Hkv*D}+(ulong)tok*{Hkv*D}+(ulong)h*{D}+d]; }} }}
  Vout[((ulong)h*(ulong)S+(ulong)s)*{D}+d]=vout_v;
"""
        return mx.fast.metal_kernel(
            name="old_vgather_p9ref", input_names=["v_pool", "block_table", "params"],
            output_names=["Vout"], source=src, header=_HEADER, ensure_row_contiguous=True)
    mx.random.seed(2)
    vpool = mx.random.normal((nb, bs, Hkv, D)).astype(mx.float16); mx.eval(vpool)
    S = bs * 2
    bt = mx.array([1, 4], dtype=mx.int32)
    params = mx.array([S, nb, 2], dtype=mx.int32)
    tot = S * Hkv * D; grid = ((tot + 255) // 256 * 256, 1, 1)
    new = _get_v_gather_kernel(D, Hkv, bs, mx.float16)(
        inputs=[vpool, bt, params], output_shapes=[(1, Hkv, S, D)],
        output_dtypes=[mx.float16], grid=grid, threadgroup=(256, 1, 1))[0]
    old = old_v()(inputs=[vpool, bt, params], output_shapes=[(1, Hkv, S, D)],
                  output_dtypes=[mx.float16], grid=grid, threadgroup=(256, 1, 1))[0]
    mx.eval(new, old)
    assert np.array(new.astype(mx.float32)).tobytes() == np.array(old.astype(mx.float32)).tobytes()


# ── 5. P1 bounds guard preserved in BOTH dtypes (OOB phys → zero, no OOB read) ───
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
def test_bounds_guard_both_dtypes(dt):
    mx.random.seed(3)
    vpool = mx.random.normal((nb, bs, Hkv, D)).astype(dt)
    kpool = mx.random.randint(0, 256, (nb, bs, Hkv, _PD)).astype(mx.uint8)
    kscl = mx.ones((nb, bs, Hkv), mx.float32)
    _, cf = _get_centroids(bits); cent = cf.astype(mx.float16)
    mx.eval(vpool, kpool, kscl, cent)
    S = bs * 3
    bt = mx.array([99, -5, -1], dtype=mx.int32)      # OOB-high, OOB-low, padding
    params = mx.array([S, nb, 3], dtype=mx.int32)
    tot = S * Hkv * D; grid = ((tot + 255) // 256 * 256, 1, 1)
    V = _get_v_gather_kernel(D, Hkv, bs, dt)(
        inputs=[vpool, bt, params], output_shapes=[(1, Hkv, S, D)],
        output_dtypes=[dt], grid=grid, threadgroup=(256, 1, 1))[0]
    K = _get_k_dequant_kernel(D, Hkv, bs, bits, dt)(
        inputs=[kpool, kscl, cent, bt, params], output_shapes=[(1, Hkv, S, D)],
        output_dtypes=[dt], grid=grid, threadgroup=(256, 1, 1))[0]
    mx.eval(V, K)
    vn, kn = np.array(V.astype(mx.float32)), np.array(K.astype(mx.float32))
    assert np.isfinite(vn).all() and np.isfinite(kn).all()
    assert np.all(vn == 0) and np.all(kn == 0)       # guard skipped every OOB load


def test_bounds_guard_via_public_step_trust_indices():
    prev = os.environ.get("MFA_PAGED_TRUST_INDICES")
    os.environ["MFA_PAGED_TRUST_INDICES"] = "1"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for dt in (mx.float16, mx.bfloat16):
                c = TurboQuantPagedInferenceContext(num_blocks=16, block_size=16,
                                                    H_kv=8, D=64, dtype=dt, tq_bits=3)
                c.prefill(_mk(16, dt), _mk(16, dt), _mk(16, dt), seq_id=0)
                o = c.step(_mk(1, dt), _mk(1, dt), _mk(1, dt), seq_id=0)
                mx.eval(o)
                assert bool(np.isfinite(np.array(o.astype(mx.float32))).all())
    finally:
        if prev is None:
            os.environ.pop("MFA_PAGED_TRUST_INDICES", None)
        else:
            os.environ["MFA_PAGED_TRUST_INDICES"] = prev


# ── 6. bf16 decode is deterministic (byte-identical across runs) ─────────────────
def test_bf16_decode_deterministic():
    def run():
        mx.random.seed(7)
        c = TurboQuantPagedInferenceContext(num_blocks=16, block_size=16, H_kv=8,
                                            D=64, dtype=mx.bfloat16, tq_bits=3)
        k, v = _mk(16, mx.bfloat16), _mk(16, mx.bfloat16)
        c.prefill(k, v, _mk(16, mx.bfloat16), seq_id=0)
        q = _mk(1, mx.bfloat16)
        o = c.step(q, q, q, seq_id=0)
        mx.eval(o)
        return np.array(o.astype(mx.float32)).tobytes()
    a, b = run(), run()
    assert a == b
