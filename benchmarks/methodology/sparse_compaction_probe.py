"""Block-sparse compacted-iteration probe (DIAGNOSTIC + gated-prototype evidence).

Phase 0 (gate): decompose the flat ~3.8ms / density-independent cost of the routed
block-sparse NAX kernel into loop-walk (per-TG, propto NK) vs grid/wave (propto NQ),
by DECOUPLING qL and kL on the REAL kernel (no standalone proxy — footgun #3).
Phase 0-floor (decisive prototype evidence): the production kernel at kL=active*BK over
the production query length IS a compacted-iteration result — compacted cost is propto
active_count regardless of block layout (the K_kb pointer jumps per active index), so this
floor is the achievable curve for any mask structure.
Phase 1: peak-memory frontier — sparse [NQ,NK] bool mask vs SDPA NxN additive mask.

Discipline (benchmark-measurement-correctness): effective FLOP; plausibility-gated vs 51.8
TFLOPS fp16 NAX peak; 3-replicate median; ablate the REAL routed kernel, never a proxy of a
different structure.
"""
import numpy as np, mlx.core as mx, math, time, gc
from mlx_mfa import flash_attention_sparse, flash_attention

B, H, D = 2, 8, 128
SCALE = 1.0 / math.sqrt(D)
BQ, BK = 32, 16


def t_run(qL, kL, active, reps=15, warm=6):
    NQ, NK = qL // BQ, kL // BK
    q = (mx.random.uniform(-1, 1, (B, H, qL, D)) * 0.1).astype(mx.float16)
    k = (mx.random.uniform(-1, 1, (B, H, kL, D)) * 0.1).astype(mx.float16)
    v = (mx.random.uniform(-1, 1, (B, H, kL, D)) * 0.1).astype(mx.float16)
    ka = min(active, NK)
    m = np.zeros((NQ, NK), bool); m[:, :ka] = True; m = mx.array(m)
    mx.eval(q, k, v, m)
    f = lambda: flash_attention_sparse(q, k, v, m, scale=SCALE, causal=False)
    for _ in range(warm): mx.eval(f())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    return sorted(ts)[reps // 2]


print("=== Phase 0 ablation A — loop-walk (fix qL=128/NQ=4, vary kL/NK, active=2) ===")
for kL in [512, 1024, 2048, 4096, 8192, 16384]:
    print(f"  NK={kL//BK:5d}  t={t_run(128, kL, 2)*1e3:7.3f} ms")
print("=== Phase 0 ablation B — grid/wave (fix kL=2048/NK=128, vary qL/NQ, active=2) ===")
for qL in [128, 512, 2048, 8192, 16384]:
    print(f"  NQ={qL//BQ:5d}  t={t_run(qL, 2048, 2)*1e3:7.3f} ms")
print("=== Phase 0 control — active=2 vs full at qL=kL=4096 (compute/memory check) ===")
print(f"  active=2    t={t_run(4096,4096,2)*1e3:.3f} ms")
print(f"  active=full t={t_run(4096,4096,256)*1e3:.3f} ms   (equal => density-independent walk, not compute/mem)")

print("\n=== Phase 0-floor — compaction floor at qL=4096 (NQ=128 production query length) ===")
print(f"  {'active':>6} {'dens':>6} {'current':>10} {'compacted-floor':>16} {'potential':>10}")
for A, kLc in [(2, 32), (8, 128), (32, 512), (64, 1024), (128, 2048), (256, 4096)]:
    t_cur = t_run(4096, 4096, A)
    t_floor = t_run(4096, kLc, kLc // BK)
    F = 4.0 * B * H * 4096 * 4096 * D * (A / 256)
    eff = F / t_floor / 1e12
    flag = " !!>PEAK" if eff > 51.8 else ""
    print(f"  {A:>6} {A/256:>6.3f} {t_cur*1e3:>8.3f}ms {t_floor*1e3:>13.3f}ms {t_cur/t_floor:>9.2f}x  (floor eff={eff:.1f} TF{flag})")

print("\n=== Phase 1 — peak-memory frontier (B=1 H=8 D=128 f16): block-mask vs NxN-mask ===")
B1 = 1
def peak_mb(fn):
    mx.synchronize()
    try: mx.reset_peak_memory()
    except Exception: pass
    o = fn(); mx.eval(o); p = mx.get_peak_memory() / 2**20; del o; gc.collect(); return p
for N in [2048, 4096, 8192, 16384, 32768]:
    NQ, NK = N // BQ, N // BK
    q = (mx.random.uniform(-1, 1, (B1, H, N, D)) * 0.1).astype(mx.float16)
    k = (mx.random.uniform(-1, 1, (B1, H, N, D)) * 0.1).astype(mx.float16)
    v = (mx.random.uniform(-1, 1, (B1, H, N, D)) * 0.1).astype(mx.float16)
    bm = (mx.zeros((NQ, NK), dtype=mx.bool_) + (mx.arange(NK)[None, :] < 32)); mx.eval(q, k, v, bm)
    p_sp = peak_mb(lambda: flash_attention_sparse(q, k, v, bm, scale=SCALE, causal=False))
    am = mx.zeros((B1, H, N, N), dtype=mx.float16); mx.eval(am)
    p_sd = peak_mb(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=am)); del am; gc.collect()
    print(f"  N={N:6d}  sparse {p_sp:8.1f} MB | SDPA(NxN-mask) {p_sd:8.1f} MB | {p_sd/p_sp:.2f}x | NxN fp16 mask = {B1*H*N*N*2/2**30:.2f} GB")
