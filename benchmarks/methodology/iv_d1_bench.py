"""IV-D1 gain bench: TQ decode step, eager-append vs deferred-append (tq_v=False).

A/B in one process on the same ctx — the ONLY difference is whether append() does
its per-step eager mx.eval. Measures the recovered MLX per-eval round-trip floor.
3 sessions via subprocess isolation; median; absolute + direction (lesson #15).

  python iv_d1_bench.py child <S>   # one session, prints JSON
  python iv_d1_bench.py parent      # 3 sessions, summarizes
"""
import sys, json, time, math, subprocess
import mlx.core as mx

WARMUP, ITERS = 40, 200
Hq, Hkv, D, bs, bits = 8, 2, 128, 64, 3


def child(S):
    from mlx_mfa import TurboQuantPagedInferenceContext
    from mlx_mfa.tq_decode import tq_decode_attend
    ctx = TurboQuantPagedInferenceContext(num_blocks=S // bs + 64, block_size=bs,
                                          H_kv=Hkv, D=D, tq_bits=bits, tq_v=False)
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

    def one_step(defer):
        ctx.append(sk, sv, seq_id=0, defer_pool_materialize=defer)
        Sn = ctx.seq_length(0); nb = (Sn + bs - 1)//bs
        bt = ctx.get_block_table([0])
        o = tq_decode_attend(sq, ctx._k_pool, ctx._v_pool_fp16, ctx._k_scales,
                             ctx._k_centroids, bt[0][:nb], Sn, scale=scale,
                             block_size=bs, tq_bits=bits, stream=ctx.stream)
        mx.eval(o)
        return o

    def bench(defer):
        for _ in range(WARMUP): one_step(defer)
        ts = []
        for _ in range(ITERS):
            t0 = time.perf_counter(); one_step(defer); ts.append((time.perf_counter()-t0)*1e6)
        return sorted(ts)[len(ts)//2]

    eager = bench(False)
    deferred = bench(True)
    print(json.dumps({"S": S, "eager_us": eager, "deferred_us": deferred}))


def parent():
    self = __file__
    for S in (2048, 4096):
        runs = []
        for _ in range(3):
            p = subprocess.run([sys.executable, self, "child", str(S)],
                               capture_output=True, text=True, timeout=600)
            runs.append(json.loads(p.stdout.strip().splitlines()[-1]))
        eg = sorted(r["eager_us"] for r in runs)[1]
        df = sorted(r["deferred_us"] for r in runs)[1]
        saved = eg - df
        print(f"S={S}: eager={eg:.1f}us  deferred={df:.1f}us  "
              f"saved={saved:.1f}us ({100*saved/eg:.1f}% of step)  "
              f"-> {eg/df:.2f}x" if df else "")
        print(f"   eager runs:    {[round(r['eager_us']) for r in runs]}")
        print(f"   deferred runs: {[round(r['deferred_us']) for r in runs]}")


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(int(sys.argv[2]))
    else:
        parent()
