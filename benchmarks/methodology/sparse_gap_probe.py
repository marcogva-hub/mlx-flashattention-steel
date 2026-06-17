"""Sparse NAX gap-decomposition probe (DIAGNOSTIC, marked micro-probe).

Decomposes the routed block-sparse NAX-matmul2d forward's effective throughput
into: (a) density-independent fixed overhead [intercept = gather/mask/launch,
NOT matmul2d-setup-addressable], (b) per-active-block work [slope]; and bounds
the matmul2d-form penalty (the simdgroup-addressable part) by comparing the
sparse slope-work to the SAME work done by real simdgroup-form fused kernels
(STEEL dense backend='mfa', Apple SDPA) — NO standalone proxy (skill #3).

Discipline (benchmark-measurement-correctness): effective FLOP only; causal-½ is
a work factor not a throughput factor (all runs NON-causal, apples-to-apples);
plausibility-gated vs the ~51.8 TFLOPS fp16 NAX peak; regime checked via the
t-vs-density linear R^2 (Delta time must track Delta work); 3-replicate median;
fp32 sanity on the d=1.0 (all-active == dense) case.
"""
import math, time, statistics
import numpy as np
import mlx.core as mx
import mlx_mfa
from mlx_mfa import flash_attention, flash_attention_sparse

B, H, N, D = 2, 8, 4096, 128
SCALE = 1.0 / math.sqrt(D)
PEAK = 51.8  # fp16 NAX matmul2d peak (Day-J), plausibility ceiling
BQ, BK = 32, 16
NQ, NK = N // BQ, N // BK  # 128, 256
F_DENSE = 4.0 * B * H * N * N * D  # non-causal dense effective FLOPs

mx.random.seed(0)
q = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
k = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
v = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
mx.eval(q, k, v)


def rect_mask(d):
    """[NQ,NK] bool mask; each Q-block attends the first round(d*NK) K-blocks.
    Active fraction is exactly round(d*NK)/NK -> linear in d, deterministic."""
    ka = max(1, round(d * NK))
    m = np.zeros((NQ, NK), dtype=bool)
    m[:, :ka] = True
    return mx.array(m), ka / NK


def med3(fn):
    reps = []
    for _ in range(3):
        for _ in range(8):
            mx.eval(fn())
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t0)
        reps.append(sorted(ts)[10])
    reps.sort()
    return reps[1], (max(reps) - min(reps)) / reps[1]


def gate(name, t, flops):
    eff = flops / t / 1e12
    flag = "  !!ABOVE PEAK (artifact)" if eff > PEAK else ""
    print(f"  {name:<34} {t*1e3:7.3f} ms   eff={eff:6.2f} TFLOPS{flag}")
    return eff


print(f"mlx_mfa {mlx_mfa.__version__} | B{B} H{H} N{N} D{D} f16 NON-causal | peak={PEAK}\n")

# --- Dense anchors (simdgroup form): the realistic per-work ceiling ----------
print("=== dense anchors (simdgroup form, full N^2 work) ===")
t_sdpa, cv = med3(lambda: flash_attention(q, k, v, scale=SCALE, causal=False))   # SDPA (M5 default)
eff_sdpa = gate(f"SDPA (auto/default) cv={cv:.2f}", t_sdpa, F_DENSE)
try:
    t_steel, cv = med3(lambda: flash_attention(q, k, v, scale=SCALE, causal=False, backend="mfa"))
    eff_steel = gate(f"STEEL dense backend=mfa cv={cv:.2f}", t_steel, F_DENSE)
except Exception as e:
    t_steel, eff_steel = None, None
    print(f"  STEEL dense backend=mfa: N/A ({e})")

# --- fp32 sanity: sparse all-active (d=1.0) must equal dense ------------------
m_full, _ = rect_mask(1.0); mx.eval(m_full)
o_sp = flash_attention_sparse(q, k, v, m_full, scale=SCALE, causal=False)
o_rf = flash_attention(q, k, v, scale=SCALE, causal=False)
mx.eval(o_sp, o_rf)
mad = float(mx.max(mx.abs(o_sp.astype(mx.float32) - o_rf.astype(mx.float32))).item())
print(f"\n  [fp32 sanity] sparse(d=1.0) vs dense max_abs_err = {mad:.2e}  ({'OK' if mad < 3e-2 else 'FAIL'})")

# --- Density sweep: Amdahl decomposition -------------------------------------
print("\n=== sparse NAX-matmul2d density sweep (Amdahl: t = a + b*d) ===")
ds = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
xs, ys = [], []
for d in ds:
    m, dexact = rect_mask(d); mx.eval(m)
    t, cv = med3(lambda: flash_attention_sparse(q, k, v, m, scale=SCALE, causal=False))
    eff = gate(f"d={dexact:.3f} cv={cv:.2f}", t, F_DENSE * dexact)
    xs.append(dexact); ys.append(t)

# linear fit t(d) = a + b*d
xa, ya = np.array(xs), np.array(ys)
b, a = np.polyfit(xa, ya, 1)
yhat = a + b * xa
ss_res = float(np.sum((ya - yhat) ** 2)); ss_tot = float(np.sum((ya - ya.mean()) ** 2))
r2 = 1 - ss_res / ss_tot
print(f"\n  fit: t(d) = {a*1e3:.3f}ms (intercept a) + {b*1e3:.3f}ms * d   R^2={r2:.4f}")
print(f"  regime check (Delta time tracks Delta work): R^2 {'OK >=0.97' if r2 >= 0.97 else 'WEAK <0.97 (overhead-contaminated?)'}")

# --- Decomposition --------------------------------------------------------
t_full = ys[-1]                      # measured t_sparse(d=1.0), full N^2 work
work_at_full = t_full - a            # slope-attributable (per-work) at d=1.0
print("\n=== decomposition of t_sparse(d=1.0) ===")
print(f"  fixed overhead a (intercept)         = {a*1e3:7.3f} ms  ({a/t_full*100:4.1f}%)  [gather/mask/launch — NOT simdgroup-addressable]")
print(f"  per-work term (t_full - a)           = {work_at_full*1e3:7.3f} ms  ({work_at_full/t_full*100:4.1f}%)  [matmul2d-setup + GEMM + softmax + V-gather]")
print(f"  -- per-work vs simdgroup-form ceilings (same full N^2 work) --")
print(f"     SDPA dense (register coop-tensor) = {t_sdpa*1e3:7.3f} ms")
if t_steel: print(f"     STEEL dense (simdgroup_matrix)    = {t_steel*1e3:7.3f} ms")
ceil_ref = t_steel if t_steel else t_sdpa
print(f"  matmul2d-form penalty bound = (t_full - a) / simdgroup-dense = {work_at_full/ceil_ref:.2f}x")
print(f"     (>1.3x => matmul2d per-work penalty material => simdgroup lever has a measured premise;")
print(f"      ~1.0x => per-work already at simdgroup rate => gap is the intercept, lever NOT warranted)")

# --- Phase 1: realistic ideal-sparse ceiling vs measured ---------------------
print("\n=== Phase 1: realistic ideal-sparse ceiling (NOT dense-44.9) ===")
for d in [0.25, 0.5]:
    m, dexact = rect_mask(d); mx.eval(m)
    t, _ = med3(lambda: flash_attention_sparse(q, k, v, m, scale=SCALE, causal=False))
    ideal = dexact * t_sdpa            # active-compute floor at SDPA per-work rate (+ ~0 unavoidable gather)
    eff_meas = F_DENSE * dexact / t / 1e12
    eff_ideal = F_DENSE * dexact / ideal / 1e12
    print(f"  d={dexact:.3f}: measured {t*1e3:6.3f}ms ({eff_meas:.2f} TF) | ideal-sparse {ideal*1e3:6.3f}ms ({eff_ideal:.2f} TF) "
          f"| addressable gap {(t-ideal)*1e3:+.3f}ms ({t/ideal:.2f}x slower than ideal-sparse)")
print(f"\n  (ideal-sparse = d * SDPA-dense-time; the realistic ceiling is SDPA's per-work rate on the")
print(f"   active fraction, NOT the full dense 44.9. unavoidable gather/mask ~ intercept-floor, small.)")
