"""Phase IV-0 R.1 — decode/small-q hot-path profile on M5/26.6.

Attributes decode wall-clock to layers:
  t_call  = flash_attention(...) Python call (dispatch decision + lazy graph build; no GPU yet)
  t_eval  = mx.eval(out)  (GPU kernel + lazy-exec)
Reducible Python overhead = flash_attention total - raw mx.fast.sdpa total (the path it routes to
on M5). If that delta is small vs total -> Python dispatch is already in the noise (definitive).
"""
import time
import mlx.core as mx
import mlx_mfa
from mlx_mfa import flash_attention

ITERS = 200
WARMUP = 30


def _bench_segments(fn_call, q, k, v):
    # warmup
    for _ in range(WARMUP):
        o = fn_call(q, k, v); mx.eval(o)
    call_t, eval_t, total_t = [], [], []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        o = fn_call(q, k, v)              # Python dispatch + lazy graph build
        t1 = time.perf_counter()
        mx.eval(o)                         # GPU exec
        t2 = time.perf_counter()
        call_t.append((t1 - t0) * 1e6)     # us
        eval_t.append((t2 - t1) * 1e6)
        total_t.append((t2 - t0) * 1e6)
    md = lambda xs: sorted(xs)[len(xs) // 2]
    return md(call_t), md(eval_t), md(total_t)


def run(label, S, D=128, Hq=8, Hkv=8, gqa=False):
    B = 1
    if gqa:
        Hq, Hkv = 32, 8
    scale = 1.0 / (D ** 0.5)
    mx.random.seed(0)
    q = (mx.random.uniform(-1, 1, (B, Hq, 1, D)) * 0.1).astype(mx.float16)   # N_q=1 decode
    k = (mx.random.uniform(-1, 1, (B, Hkv, S, D)) * 0.1).astype(mx.float16)
    v = (mx.random.uniform(-1, 1, (B, Hkv, S, D)) * 0.1).astype(mx.float16)
    mx.eval(q, k, v)

    def fa(q, k, v): return flash_attention(q, k, v, scale=scale, causal=False)
    def sdpa(q, k, v): return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    fa_call, fa_eval, fa_total = _bench_segments(fa, q, k, v)
    sd_call, sd_eval, sd_total = _bench_segments(sdpa, q, k, v)
    overhead = fa_total - sd_total
    print(f"\n=== {label} (Hq={Hq} Hkv={Hkv} S={S} D={D} N_q=1) ===")
    print(f"  flash_attention: call={fa_call:7.1f}us  eval={fa_eval:7.1f}us  total={fa_total:7.1f}us")
    print(f"  raw mx.sdpa    : call={sd_call:7.1f}us  eval={sd_eval:7.1f}us  total={sd_total:7.1f}us")
    print(f"  mlx-mfa wrapper overhead vs raw SDPA = {overhead:.1f}us "
          f"({100*overhead/fa_total:.1f}% of fa total)")
    print(f"  flash_attention Python-dispatch (call) = {fa_call:.1f}us "
          f"({100*fa_call/fa_total:.1f}% of fa total); GPU/eval = {100*fa_eval/fa_total:.1f}%")
    return {"label": label, "fa_call": fa_call, "fa_eval": fa_eval, "fa_total": fa_total,
            "sdpa_total": sd_total, "overhead_us": overhead}


print(f"mlx_mfa {mlx_mfa.__version__} | mlx {mx.__version__}")
rows = []
for S in (2048, 4096, 16384):
    rows.append(run(f"decode S={S} D=128 MHA", S, D=128))
rows.append(run("decode S=4096 D=128 GQA32:8", 4096, D=128, gqa=True))
rows.append(run("decode S=4096 D=64 MHA", 4096, D=64))

print("\n\n=== SUMMARY: reducible Python overhead vs irreducible kernel ===")
for r in rows:
    print(f"  {r['label']:<28} fa_total={r['fa_total']:7.1f}us  "
          f"py_dispatch={r['fa_call']:6.1f}us  wrapper_vs_sdpa={r['overhead_us']:+6.1f}us")
