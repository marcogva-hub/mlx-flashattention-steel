"""Sprint II-7 — representative-workload profiling harnesses.

Four production-shaped loops (synthetic tensors, mlx-mfa scope only):
  dit     — attention-heavy DiT-like loop (fwd + bwd, causal + non-causal)
  vae     — conv3d-heavy VAE-like loop (small-K cells from II-4)
  decode  — TurboQuant paged decode loop (rope + append + attend)
  sparse  — FlashVSR-style top-K sparse attention

Instrumentation per loop:
  - graph-build time (Python-side call until mx graph constructed)
  - eval time (mx.eval to completion)
  - optional cProfile of the Python layer (--cprofile)

Usage:
  .venv/bin/python benchmarks/profile_ii7_harnesses.py dit [--iters 50] [--cprofile]
"""
import argparse
import math
import sys
import time

import mlx.core as mx

import mlx_mfa
from mlx_mfa import flash_attention


def _timer(fn_build, iters, warmup=5):
    """Return (median_build_ms, median_eval_ms, median_total_ms)."""
    for _ in range(warmup):
        out = fn_build()
        mx.eval(out)
    builds, evals, totals = [], [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn_build()
        t1 = time.perf_counter()
        mx.eval(out)
        t2 = time.perf_counter()
        builds.append((t1 - t0) * 1e3)
        evals.append((t2 - t1) * 1e3)
        totals.append((t2 - t0) * 1e3)
    builds.sort(); evals.sort(); totals.sort()
    m = len(builds) // 2
    return builds[m], evals[m], totals[m]


def harness_dit(iters):
    """DiT-like: B=1 H=16 D=64 fp16; spatial attention at qL=4096 (64x64
    latent) + qL=8192; fwd non-causal (DiT) + bwd causal (training cell)."""
    print("== DiT-like attention loop ==")
    res = {}
    for (N, causal, bwd) in [(4096, False, False), (8192, False, False),
                             (4096, True, True), (4096, False, True)]:
        mx.random.seed(1)
        q = mx.random.normal((1, 16, N, 64), dtype=mx.float16)
        k = mx.random.normal((1, 16, N, 64), dtype=mx.float16)
        v = mx.random.normal((1, 16, N, 64), dtype=mx.float16)
        mx.eval(q, k, v)
        if bwd:
            grad = mx.grad(
                lambda a, b, c: flash_attention(a, b, c, causal=causal).sum(),
                argnums=(0, 1, 2))
            fn = lambda: grad(q, k, v)
        else:
            fn = lambda: flash_attention(q, k, v, causal=causal)
        b, e, t = _timer(fn, iters)
        tag = f"N={N} causal={causal} {'bwd' if bwd else 'fwd'}"
        res[tag] = (b, e, t)
        print(f"  {tag:34s} build={b:7.3f}ms eval={e:8.3f}ms total={t:8.3f}ms "
              f"(build={100*b/t:4.1f}%)")
    return res


def harness_vae(iters):
    """VAE-like conv3d: NDHWC, the II-4 small-K cells.  Uses the public
    conv3d_nax_forward surface (auto-routing)."""
    print("== VAE-like conv3d loop ==")
    from mlx_mfa.conv_nax import conv3d_nax_forward
    res = {}
    # (T, H, W, Cin, Cout): VAE decoder mid + upsampled blocks
    for (T, H, W, Ci, Co) in [(8, 32, 32, 256, 256), (8, 64, 64, 128, 128),
                              (16, 64, 64, 128, 128)]:
        mx.random.seed(2)
        x = mx.random.normal((1, T, H, W, Ci), dtype=mx.float16)
        w = mx.random.normal((Co, 3, 3, 3, Ci), dtype=mx.float16) * 0.05
        mx.eval(x, w)
        fn = lambda: conv3d_nax_forward(x, w, stride=(1, 1, 1),
                                        padding=(1, 1, 1))
        b, e, t = _timer(fn, iters)
        tag = f"T{T} {H}x{W} C{Ci}->{Co} (K={27*Ci})"
        res[tag] = (b, e, t)
        print(f"  {tag:34s} build={b:7.3f}ms eval={e:8.3f}ms total={t:8.3f}ms "
              f"(build={100*b/t:4.1f}%)")
    return res


def harness_decode(iters):
    """TurboQuant paged decode: prefill once, then decode steps through
    the runtime (rope + paged append + attend)."""
    print("== TurboQuant paged decode loop ==")
    from mlx_mfa.inference import TurboQuantPagedInferenceContext
    res = {}
    B, Hq, Hkv, D = 1, 32, 8, 128
    for S0 in (4096, 16384):
        ctx = TurboQuantPagedInferenceContext(
            num_blocks=max(96, S0 // 256 + 32), block_size=256,
            H_kv=Hkv, D=D, tq_bits=3)
        mx.random.seed(3)
        k0 = mx.random.normal((1, Hkv, S0, D), dtype=mx.float16)
        v0 = mx.random.normal((1, Hkv, S0, D), dtype=mx.float16)
        q0 = mx.random.normal((1, Hq, S0, D), dtype=mx.float16)
        mx.eval(k0, v0, q0)
        out = ctx.prefill(q0, k0, v0)
        mx.eval(out)

        q = mx.random.normal((1, Hq, 1, D), dtype=mx.float16)
        kn = mx.random.normal((1, Hkv, 1, D), dtype=mx.float16)
        vn = mx.random.normal((1, Hkv, 1, D), dtype=mx.float16)
        mx.eval(q, kn, vn)
        step = lambda: ctx.step(q, kn, vn)

        b, e, t = _timer(step, iters)
        tag = f"S0={S0} GQA{Hq // Hkv} D={D} tq3b"
        res[tag] = (b, e, t)
        print(f"  {tag:34s} build={b:7.3f}ms eval={e:8.3f}ms total={t:8.3f}ms "
              f"(build={100*b/t:4.1f}%)")
    return res


def harness_sparse(iters):
    """FlashVSR-style top-K sparse: audit shape B=1 H=16 N=S=4096 D=128
    k_count=64 (topk_ratio 64/4096); also LCSA block-sparse direct."""
    print("== sparse / top-K loop ==")
    from mlx_mfa import flash_attention_topk, flash_attention_sparse
    from mlx_mfa.masks import make_lcsa_mask
    res = {}
    mx.random.seed(4)
    q = mx.random.normal((1, 16, 4096, 128), dtype=mx.float16)
    k = mx.random.normal((1, 16, 4096, 128), dtype=mx.float16)
    v = mx.random.normal((1, 16, 4096, 128), dtype=mx.float16)
    mx.eval(q, k, v)

    fn = lambda: flash_attention_topk(q, k, v, topk_ratio=64.0 / 4096)
    b, e, t = _timer(fn, iters)
    res["topk k=64 N=4096 D=128"] = (b, e, t)
    print(f"  {'topk k=64 N=4096 D=128':34s} build={b:7.3f}ms eval={e:8.3f}ms "
          f"total={t:8.3f}ms (build={100*b/t:4.1f}%)")

    # LCSA: dynamic mask built from q,k PER CALL (FlashVSR pattern) —
    # mask-construction cost is part of the production loop.
    fn_mask = lambda: make_lcsa_mask(q, k, height=64, width=64,
                                     spatial_radius=8, top_k=32,
                                     head_dim=128, num_frames=1)
    b, e, t = _timer(fn_mask, iters)
    res["lcsa mask build (per-call)"] = (b, e, t)
    print(f"  {'lcsa mask build (per-call)':34s} build={b:7.3f}ms eval={e:8.3f}ms "
          f"total={t:8.3f}ms (build={100*b/t:4.1f}%)")

    mask = fn_mask()
    mx.eval(mask)
    fn2 = lambda: flash_attention_sparse(q, k, v, mask)
    b, e, t = _timer(fn2, iters)
    res["lcsa block-sparse attend"] = (b, e, t)
    print(f"  {'lcsa block-sparse attend':34s} build={b:7.3f}ms eval={e:8.3f}ms "
          f"total={t:8.3f}ms (build={100*b/t:4.1f}%)")
    return res


HARNESSES = {"dit": harness_dit, "vae": harness_vae,
             "decode": harness_decode, "sparse": harness_sparse}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("harness", choices=list(HARNESSES) + ["all"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--cprofile", action="store_true")
    args = ap.parse_args()

    names = list(HARNESSES) if args.harness == "all" else [args.harness]
    if args.cprofile:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()
    for n in names:
        HARNESSES[n](args.iters)
    if args.cprofile:
        pr.disable()
        st = pstats.Stats(pr)
        st.sort_stats("cumulative")
        print("\n== cProfile (top 25 cumulative, mlx_mfa + mlx only) ==")
        st.print_stats("mlx", 25)
