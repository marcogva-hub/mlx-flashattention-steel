"""Attribute TQ step() call-time (the ~480us Python) to sub-components."""
import time, os, math
import mlx.core as mx
import mlx_mfa
from mlx_mfa import TurboQuantPagedInferenceContext

S, Hq, Hkv, D, bs, bits = 4096, 32, 8, 128, 64, 3
ctx = TurboQuantPagedInferenceContext(num_blocks=S // bs + 16, block_size=bs, H_kv=Hkv, D=D, tq_bits=bits)
scale = 1.0 / math.sqrt(D)
mx.random.seed(0)
pq = (mx.random.uniform(-1,1,(1,Hq,S,D))*0.1).astype(mx.float16)
pk = (mx.random.uniform(-1,1,(1,Hkv,S,D))*0.1).astype(mx.float16)
pv = (mx.random.uniform(-1,1,(1,Hkv,S,D))*0.1).astype(mx.float16)
ctx.prefill(pq, pk, pv)
sq = (mx.random.uniform(-1,1,(1,Hq,1,D))*0.1).astype(mx.float16)
sk = (mx.random.uniform(-1,1,(1,Hkv,1,D))*0.1).astype(mx.float16)
sv = (mx.random.uniform(-1,1,(1,Hkv,1,D))*0.1).astype(mx.float16)
mx.eval(sq, sk, sv)
from mlx_mfa.tq_decode import tq_decode_attend

def attrib():
    t = {}
    t0 = time.perf_counter()
    ctx.append(sk, sv, seq_id=0); t['append'] = (time.perf_counter()-t0)*1e6
    t0 = time.perf_counter()
    cu_q = mx.array([0, sq.shape[2]], dtype=mx.int32); t['cu_q_array'] = (time.perf_counter()-t0)*1e6
    t0 = time.perf_counter()
    bt = ctx.get_block_table([0]); t['get_block_table'] = (time.perf_counter()-t0)*1e6
    t0 = time.perf_counter()
    sl = ctx.get_seq_lens([0]); t['get_seq_lens'] = (time.perf_counter()-t0)*1e6
    t0 = time.perf_counter()
    Sn = ctx.seq_length(0); nb = (Sn + bs - 1)//bs; t['seq_length'] = (time.perf_counter()-t0)*1e6
    t0 = time.perf_counter()
    o = tq_decode_attend(sq, ctx._k_pool, ctx._v_pool_fp16, ctx._k_scales, ctx._k_centroids,
                         bt[0][:nb], Sn, scale=scale, block_size=bs, tq_bits=bits, stream=ctx.stream)
    t['tq_decode_attend_call'] = (time.perf_counter()-t0)*1e6
    return t, o

for _ in range(30):
    _, o = attrib(); mx.eval(o)
import collections
acc = collections.defaultdict(list)
for _ in range(150):
    t, o = attrib(); mx.eval(o)
    for k, val in t.items(): acc[k].append(val)
md = lambda xs: sorted(xs)[len(xs)//2]
print(f"mlx_mfa {mlx_mfa.__version__}  TQ step() call-time attribution (S={S}, median us):")
tot = 0
for k in ['append','cu_q_array','get_block_table','get_seq_lens','seq_length','tq_decode_attend_call']:
    m = md(acc[k]); tot += m
    print(f"  {k:<24} {m:8.1f}us")
print(f"  {'TOTAL call':<24} {tot:8.1f}us")
