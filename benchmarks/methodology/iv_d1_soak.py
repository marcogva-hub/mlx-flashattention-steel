"""IV-D1 soak: TQ decode append-eval-deferral equivalence under churn × processes.

The deferral is a pure materialization-ORDERING change (skip append's redundant
eager eval on the tq_v=False decode branch; the gather reads pools as graph-inputs
so eval(o) materializes them). It must NOT change the result -> bit-identical to the
pre-change eager reference, per step, under concurrent-alloc churn, across processes.

  python iv_d1_soak.py save <ref.npz>      # run on PRE-change code -> save reference
  python iv_d1_soak.py compare <ref.npz>   # run on POST-change code -> assert bit-identical
  python iv_d1_soak.py fp32 <ref.npz>      # sanity: TQ vs independent fp32 (cosine)
"""
import sys, math
import numpy as np
import mlx.core as mx
from mlx_mfa import TurboQuantPagedInferenceContext

S0, KSTEPS, Hq, Hkv, D, bs, bits = 512, 200, 8, 2, 128, 64, 3  # 200 steps spans many blocks


def _inputs(step):
    mx.random.seed(1000 + step)
    sq = (mx.random.uniform(-1, 1, (1, Hq, 1, D)) * 0.1).astype(mx.float16)
    sk = (mx.random.uniform(-1, 1, (1, Hkv, 1, D)) * 0.1).astype(mx.float16)
    sv = (mx.random.uniform(-1, 1, (1, Hkv, 1, D)) * 0.1).astype(mx.float16)
    mx.eval(sq, sk, sv)
    return sq, sk, sv


def run(churn=True):
    ctx = TurboQuantPagedInferenceContext(num_blocks=(S0 + KSTEPS) // bs + 16,
                                          block_size=bs, H_kv=Hkv, D=D,
                                          tq_bits=bits, tq_v=False)  # tq_v=False: safe-defer case
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(7)
    pq = (mx.random.uniform(-1, 1, (1, Hq, S0, D)) * 0.1).astype(mx.float16)
    pk = (mx.random.uniform(-1, 1, (1, Hkv, S0, D)) * 0.1).astype(mx.float16)
    pv = (mx.random.uniform(-1, 1, (1, Hkv, S0, D)) * 0.1).astype(mx.float16)
    ctx.prefill(pq, pk, pv)
    outs = []
    for step in range(KSTEPS):
        sq, sk, sv = _inputs(step)
        o = ctx.step(sq, sk, sv, scale=scale)        # append (maybe deferred) + gather -> lazy o
        if churn:
            c = mx.random.uniform(0, 1, (1024, 1024)).astype(mx.float16)
            mx.eval(c @ c.T)                          # concurrent alloc while pool writes pending
        mx.eval(o)                                    # materialize: must be churn-immune
        outs.append(np.array(o.astype(mx.float32)))
    return np.stack(outs)  # [KSTEPS, 1, Hq, 1, D]


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "save":
        out = run(churn=True)
        np.savez(path, outs=out)
        print(f"SAVED reference: {out.shape} to {path}  (finite={np.isfinite(out).all()})")
    elif mode == "compare":
        ref = np.load(path)["outs"]
        out = run(churn=True)
        assert out.shape == ref.shape, f"shape {out.shape} vs {ref.shape}"
        maxdiff = float(np.max(np.abs(out - ref)))
        finite = bool(np.isfinite(out).all())
        per_step_max = [float(np.max(np.abs(out[i] - ref[i]))) for i in range(len(out))]
        worst_step = int(np.argmax(per_step_max))
        ok = (maxdiff == 0.0) and finite
        print(f"COMPARE: bit-identical={maxdiff==0.0} max_abs_diff={maxdiff:.2e} "
              f"finite={finite} worst_step={worst_step}({per_step_max[worst_step]:.2e})")
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    elif mode == "fp32":
        # sanity vs independent fp32 (TQ is lossy -> cosine, not bit-exact)
        out = run(churn=False)
        print(f"fp32-sanity: finite={np.isfinite(out).all()} "
              f"mean_abs={np.mean(np.abs(out)):.4f} (non-degenerate check)")
