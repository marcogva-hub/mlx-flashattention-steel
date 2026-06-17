"""IV-D2 gain bench: tq_v=True (DEFAULT) decode step, pre-change eager vs combined-eval.

eager (pre-change): append eager eval (floor 1, all 5 pools) + eval(o) (floor 2)  = 2 floors.
combined (IV-D2):   append deferred + eval(o, _v_pool_tq, _v_scales)               = 1 floor.
Same GPU work; the gain is removing the 2nd MLX per-eval round-trip floor. 3 sessions.

  python iv_d2_bench.py child <S>
  python iv_d2_bench.py parent
"""
import sys, json, time, math, subprocess
import mlx.core as mx

WARMUP, ITERS = 40, 200
Hq, Hkv, D, bs, bits = 8, 2, 128, 64, 3


def child(S):
    from mlx_mfa import TurboQuantPagedInferenceContext
    from mlx_mfa.tq_decode import tq_decode_attend
    from mlx_mfa.turboquant import apply_rotation
    ctx = TurboQuantPagedInferenceContext(num_blocks=S // bs + 64, block_size=bs,
                                          H_kv=Hkv, D=D, tq_bits=bits, tq_v=True)  # DEFAULT
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

    def decode(q_rot):
        Sn = ctx.seq_length(0); nb = (Sn + bs - 1)//bs
        bt = ctx.get_block_table([0])
        return tq_decode_attend(q_rot, ctx._k_pool, ctx._v_pool_fp16, ctx._k_scales,
                                ctx._k_centroids, bt[0][:nb], Sn, scale=scale,
                                block_size=bs, tq_bits=bits, stream=ctx.stream)

    def step_eager():    # pre-change: 2 floors
        ctx.append(sk, sv, seq_id=0, defer_pool_materialize=False)   # floor 1 (all 5)
        q_rot = apply_rotation(sq.astype(mx.float32), "wht").astype(mx.float16)
        o = decode(q_rot)
        mx.eval(o)                                                   # floor 2
        return o

    def step_combined():  # IV-D2: 1 floor
        ctx.append(sk, sv, seq_id=0, defer_pool_materialize=True)    # deferred
        q_rot = apply_rotation(sq.astype(mx.float32), "wht").astype(mx.float16)
        o = decode(q_rot)
        mx.eval(o, ctx._v_pool_tq, ctx._v_scales)                    # one combined floor
        return o

    def bench(fn):
        for _ in range(WARMUP): fn()
        ts = []
        for _ in range(ITERS):
            t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1e6)
        return sorted(ts)[len(ts)//2]

    print(json.dumps({"S": S, "eager_us": bench(step_eager), "combined_us": bench(step_combined)}))


def parent():
    for S in (2048, 4096):
        runs = []
        for _ in range(3):
            p = subprocess.run([sys.executable, __file__, "child", str(S)],
                               capture_output=True, text=True, timeout=600)
            runs.append(json.loads(p.stdout.strip().splitlines()[-1]))
        eg = sorted(r["eager_us"] for r in runs)[1]
        cb = sorted(r["combined_us"] for r in runs)[1]
        saved = eg - cb
        print(f"S={S} (tq_v=True default): eager={eg:.1f}us  combined={cb:.1f}us  "
              f"saved={saved:.1f}us ({100*saved/eg:.1f}%)  -> {eg/cb:.2f}x")
        print(f"   eager runs:    {[round(r['eager_us']) for r in runs]}")
        print(f"   combined runs: {[round(r['combined_us']) for r in runs]}")


if __name__ == "__main__":
    child(int(sys.argv[2])) if sys.argv[1] == "child" else parent()
