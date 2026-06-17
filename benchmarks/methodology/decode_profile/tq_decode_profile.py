"""Phase IV-0 R.1 — TQ paged-decode step() profile (more Python orchestration:
block-table prep, pool indexing, gather/dequant + SDPA). Same call/eval attribution."""
import time
import mlx.core as mx
import mlx_mfa
from mlx_mfa import TurboQuantPagedInferenceContext

ITERS, WARMUP = 150, 30


def prof(S, Hq=32, Hkv=8, D=128, bs=64, bits=3):
    ctx = TurboQuantPagedInferenceContext(num_blocks=S // bs + 16, block_size=bs,
                                          H_kv=Hkv, D=D, tq_bits=bits)
    scale = 1.0 / (D ** 0.5)
    mx.random.seed(0)
    pq = (mx.random.uniform(-1, 1, (1, Hq, S, D)) * 0.1).astype(mx.float16)
    pk = (mx.random.uniform(-1, 1, (1, Hkv, S, D)) * 0.1).astype(mx.float16)
    pv = (mx.random.uniform(-1, 1, (1, Hkv, S, D)) * 0.1).astype(mx.float16)
    if hasattr(ctx, "prefill"):
        ctx.prefill(pq, pk, pv)
    sq = (mx.random.uniform(-1, 1, (1, Hq, 1, D)) * 0.1).astype(mx.float16)
    sk = (mx.random.uniform(-1, 1, (1, Hkv, 1, D)) * 0.1).astype(mx.float16)
    sv = (mx.random.uniform(-1, 1, (1, Hkv, 1, D)) * 0.1).astype(mx.float16)
    mx.eval(sq, sk, sv)
    for _ in range(WARMUP):
        o = ctx.step(sq, sk, sv); mx.eval(o)
    call_t, eval_t = [], []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        o = ctx.step(sq, sk, sv)
        t1 = time.perf_counter()
        mx.eval(o)
        t2 = time.perf_counter()
        call_t.append((t1 - t0) * 1e6); eval_t.append((t2 - t1) * 1e6)
    md = lambda xs: sorted(xs)[len(xs) // 2]
    c, e = md(call_t), md(eval_t)
    tot = c + e
    print(f"=== TQ step S={S} Hq={Hq} Hkv={Hkv} D={D} tq{bits}b ===")
    print(f"  step() call (Python orchestration) = {c:7.1f}us ({100*c/tot:.1f}%)")
    print(f"  eval (gather/dequant+SDPA kernels) = {e:7.1f}us ({100*e/tot:.1f}%)")
    print(f"  total = {tot:.1f}us")


print(f"mlx_mfa {mlx_mfa.__version__}")
for S in (2048, 4096, 8192):
    prof(S)
