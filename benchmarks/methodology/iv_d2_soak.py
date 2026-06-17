"""IV-D2 soak: tq_v=True default-decode combined-eval equivalence + fused-read edge.

The IV-D2 change folds the decode-unread packed-V pools into a single combined
mx.eval(o, _v_pool_tq, _v_scales) at step end (tq_v=True decode branch) — a pure
materialization-ORDERING change. Must be bit-identical to the pre-change eager path,
AND a later FUSED read of the packed-V pools must see correctly materialized data.

  python iv_d2_soak.py save <ref.npz>      # PRE-change code -> save reference (decode + fused)
  python iv_d2_soak.py compare <ref.npz>   # POST-change -> assert bit-identical
"""
import sys, math
import numpy as np
import mlx.core as mx
from mlx_mfa import TurboQuantPagedInferenceContext

S0, KSTEPS, Hq, Hkv, D, bs, bits = 512, 160, 8, 2, 128, 64, 3


def _inputs(step):
    mx.random.seed(2000 + step)
    sq = (mx.random.uniform(-1, 1, (1, Hq, 1, D)) * 0.1).astype(mx.float16)
    sk = (mx.random.uniform(-1, 1, (1, Hkv, 1, D)) * 0.1).astype(mx.float16)
    sv = (mx.random.uniform(-1, 1, (1, Hkv, 1, D)) * 0.1).astype(mx.float16)
    mx.eval(sq, sk, sv)
    return sq, sk, sv


def run():
    ctx = TurboQuantPagedInferenceContext(num_blocks=(S0 + KSTEPS) // bs + 16,
                                          block_size=bs, H_kv=Hkv, D=D,
                                          tq_bits=bits, tq_v=True)  # DEFAULT config
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(7)
    pq = (mx.random.uniform(-1, 1, (1, Hq, S0, D)) * 0.1).astype(mx.float16)
    pk = (mx.random.uniform(-1, 1, (1, Hkv, S0, D)) * 0.1).astype(mx.float16)
    pv = (mx.random.uniform(-1, 1, (1, Hkv, S0, D)) * 0.1).astype(mx.float16)
    ctx.prefill(pq, pk, pv)
    import os
    decode_outs, fused_outs = [], []
    for step in range(KSTEPS):
        sq, sk, sv = _inputs(step)
        # default decode step (combined eval folds packed-V at step end)
        o = ctx.step(sq, sk, sv, scale=scale)
        c = mx.random.uniform(0, 1, (1024, 1024)).astype(mx.float16)
        mx.eval(c @ c.T)                       # concurrent-alloc churn
        mx.eval(o)
        decode_outs.append(np.array(o.astype(mx.float32)))
        # NEW edge: every 40 steps, a FUSED read of the packed-V pools written
        # by the prior decode steps' combined evals (opt-out -> fused path).
        if (step + 1) % 40 == 0:
            os.environ["MFA_DISABLE_TQ_DECODE_SDPA"] = "1"   # force fused (reads packed-V raw)
            fq, fk, fv = _inputs(10000 + step)
            of = ctx.step(fq, fk, fv, scale=scale)
            mx.eval(of)
            fused_outs.append(np.array(of.astype(mx.float32)))
            os.environ.pop("MFA_DISABLE_TQ_DECODE_SDPA", None)
    return np.stack(decode_outs), np.stack(fused_outs)


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "save":
        d, f = run()
        np.savez(path, decode=d, fused=f)
        print(f"SAVED: decode{d.shape} fused{f.shape} finite={np.isfinite(d).all() and np.isfinite(f).all()}")
    elif mode == "compare":
        ref = np.load(path)
        d, f = run()
        dd = float(np.max(np.abs(d - ref["decode"])))
        ff = float(np.max(np.abs(f - ref["fused"])))
        finite = bool(np.isfinite(d).all() and np.isfinite(f).all())
        ok = (dd == 0.0) and (ff == 0.0) and finite
        print(f"COMPARE: decode_bit_identical={dd==0.0} (max {dd:.2e}); "
              f"fused_read_bit_identical={ff==0.0} (max {ff:.2e}); finite={finite}")
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
