#!/usr/bin/env python3
"""Volet A Phase 1 — hardened measurement campaign (spec §4 gates 6+7).

Re-judges the P1 ratios at PROMOTION standard AND through the SHIPPED public path
(flash_attention_sparse with MFA_SPARSE_NAX_EXTENDED=1 + auto_pad), proving the
documented API reaches the kernel (§Z). Solo arm-per-process (autoresearch
contract): each process times ONE arm; the parent interleaves launches to null
pool drift; medians across ≥procs with declared outliers.

Arms per cell:
  cand         flash_attention_sparse(..., auto_pad=True)  [extended env]  — shipped path
  sdpa_nomask  mx.fast.sdpa(q,k,v)                          — prod champion
  sdpa_mask    mx.fast.sdpa(q,k,v, mask=bias)               — semantic equiv (may OOM @144k)
  raw_dense    _sparse_fallback_sdpa_perhead(...)           — cutoff-cell overhead ref
Engagement (cand): byteΔ vs forced scalar_fallback > 0. Correctness: cos vs fp32
gold (row-subsample at N≥62752 to fit RAM — the full gold bias is 126 GB @144k).
Peak mem: mx.get_peak_memory() per cell. Stamp: MLX 0.31.2 / M5 Max / macOS 27.

  single '<cell>' <arm>          # one arm, fresh process, prints JSON
  parent <out.jsonl> [--floors]  # campaign driver (self-resuming)
"""
from __future__ import annotations
import json, math, os, subprocess, sys, time

os.environ.setdefault("MFA_SILENCE_NAX_WARNING", "1")
BT = 32
GOLD_ROWS = 512                 # row-subsample for the fp32 oracle on heavy N
MEM_BUDGET_GB = 100.0


# ----------------------------------------------------------------- mask builders
def _band(nq, nk, d):
    import mlx.core as mx
    h = max(0, int(round(d * nk / 2.0)))
    ii = mx.arange(nq)[:, None]; jj = mx.arange(nk)[None, :]
    center = (ii * nk) // max(1, nq)
    m = (mx.abs(jj - center) <= h) | (mx.eye(nq, nk, dtype=mx.float32) > 0.5)
    return m.astype(mx.bool_)


def _random(nq, nk, d, seed=0):
    import mlx.core as mx
    mx.random.seed(seed)
    m = (mx.random.uniform(shape=(nq, nk)) < d)
    m = (m | m.T) | (mx.eye(nq, nk, dtype=mx.float32) > 0.5)
    return m.astype(mx.bool_)


def _lcsa(q, k, N, D):
    import mlx.core as mx
    from mlx_mfa.masks import make_lcsa_mask
    h = int(math.isqrt(N))
    while h > 1 and N % h:
        h -= 1
    W = N // h
    nq = N // BT
    top_k = max(1, int(round(0.10 * nq)))
    m = make_lcsa_mask(q, k, height=h, width=W, spatial_radius=9, top_k=top_k,
                       head_dim=D, num_frames=1).astype(mx.bool_)
    return m, {"H": h, "W": W, "spatial_radius": 9, "top_k": top_k, "topk_ratio": 2.0}


def _build(cell):
    import mlx.core as mx
    B, H, N, D = cell["B"], cell["H"], cell["N"], cell["D"]
    dt = {"float16": mx.float16, "bfloat16": mx.bfloat16}[cell["dtype"]]
    mx.random.seed(0)
    mk = lambda n: (mx.random.normal((B, H, n, D)) * 0.1).astype(dt)
    q, k, v = mk(N), mk(N), mk(N); mx.eval(q, k, v)
    nq = (N + BT - 1) // BT
    kind = cell["mask"]; d = cell.get("density", 0.10)
    if kind == "sliding":
        bm, meta = _band(nq, nq, d), {}
    elif kind == "random":
        bm, meta = _random(nq, nq, d), {}
    elif kind == "lcsa":
        bm, meta = _lcsa(q, k, N, D)
    else:
        raise ValueError(kind)
    mx.eval(bm)
    return q, k, v, bm, meta


# ----------------------------------------------------------------- timing / oracle
def _med(fn, warm, iters):
    import mlx.core as mx
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts


def _adaptive(fn):
    import mlx.core as mx
    mx.eval(fn()); mx.synchronize()
    t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
    one = (time.perf_counter() - t0) * 1000.0
    ts = _med(fn, 1, 3) if one > 60.0 else _med(fn, 3, 12)
    return ts


def _cosf(a, b_flat):
    import mlx.core as mx, numpy as np
    a = np.asarray(a.astype(mx.float32)).ravel().astype(np.float64)
    return float(np.dot(a, b_flat) / (np.linalg.norm(a) * np.linalg.norm(b_flat) + 1e-30))


def _maxabs(a, b):
    import mlx.core as mx, numpy as np
    return float(np.abs(np.asarray(a.astype(mx.float32)) - np.asarray(b.astype(mx.float32))).max())


def _gold_cos(o, q, k, v, bm, scale, causal, N):
    """fp32 gold cos; full bias when it fits, else a GOLD_ROWS query-row subsample."""
    import mlx.core as mx, numpy as np
    full_bias_gb = (N * N * 4) / 1e9
    if full_bias_gb <= MEM_BUDGET_GB:
        em = mx.repeat(mx.repeat(bm, BT, 0), BT, 1)[:N, :N]
        if causal:
            idx = mx.arange(N); em = em & (idx[None, :] <= idx[:, None])
        bias = mx.where(em, mx.array(0.0, mx.float32), mx.array(float("-inf"), mx.float32))
        g = mx.fast.scaled_dot_product_attention(q.astype(mx.float32), k.astype(mx.float32),
                                                  v.astype(mx.float32), scale=scale, mask=bias)
        mx.eval(g)
        return _cosf(o, np.asarray(g).ravel().astype(np.float64)), "gold_fp32_full"
    # row-subsample: first GOLD_ROWS query rows vs all keys
    r = GOLD_ROWS
    emr = mx.repeat(mx.repeat(bm[: (r + BT - 1) // BT], BT, 0), BT, 1)[:r, :N]
    if causal:
        iq = mx.arange(r)[:, None]; ik = mx.arange(N)[None, :]; emr = emr & (ik <= iq)
    biasr = mx.where(emr, mx.array(0.0, mx.float32), mx.array(float("-inf"), mx.float32))
    gr = mx.fast.scaled_dot_product_attention(
        q[:, :, :r].astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=scale, mask=biasr)
    mx.eval(gr)
    osub = o[:, :, :r]
    return _cosf(osub, np.asarray(gr).ravel().astype(np.float64)), f"gold_fp32_rows{r}"


# ----------------------------------------------------------------- one arm
def measure(cell, arm):
    import mlx.core as mx, numpy as np
    from mlx_mfa import _ext, attention as A
    from mlx_mfa.attention import flash_attention_sparse
    B, H, N, D = cell["B"], cell["H"], cell["N"], cell["D"]
    causal = bool(cell.get("causal", False)); scale = 1.0 / math.sqrt(D)
    os.environ["MFA_SPARSE_NAX_EXTENDED"] = "1"
    if "cutoff" in cell:
        os.environ["MFA_SPARSE_D_DENSE_CUTOFF"] = str(cell["cutoff"])
    res = {"cell_id": cell["cell_id"], "arm": arm, "N": N, "H": H, "dtype": cell["dtype"],
           "mask": cell["mask"], "density": cell.get("density"), "causal": causal, "error": None}
    try:
        q, k, v, bm, meta = _build(cell)
        res["density_actual"] = round(float(mx.mean(bm.astype(mx.float32)).item()), 4)
        res["mask_meta"] = meta
        mx.reset_peak_memory()
        ap = bool(cell.get("auto_pad", False))
        if arm == "cand":
            fn = lambda: flash_attention_sparse(q, k, v, bm, scale=scale, causal=causal, auto_pad=ap)
        elif arm == "sdpa_nomask":
            fn = lambda: mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask="causal" if causal else None)
        elif arm == "sdpa_mask":
            if (N * N * 2) / 1e9 > MEM_BUDGET_GB:
                res["ms"] = None; res["skipped"] = f"sdpa+mask bias {(N*N*2)/1e9:.1f}GB > budget"
                return res
            em = mx.repeat(mx.repeat(bm, BT, 0), BT, 1)[:N, :N]
            if causal:
                idx = mx.arange(N); em = em & (idx[None, :] <= idx[:, None])
            bias = mx.where(em, mx.array(0.0, q.dtype), mx.array(-6e4, q.dtype)); mx.eval(bias)
            fn = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=bias)
        elif arm == "raw_dense":
            fn = lambda: A._sparse_fallback_sdpa_perhead(q, k, v, bm, scale, causal)
        else:
            raise ValueError(arm)
        ts = _adaptive(fn)
        res.update(ms=round(ts[len(ts) // 2], 4), ms_min=round(ts[0], 4), ms_max=round(ts[-1], 4),
                   reps=len(ts), peak_gb=round(mx.get_peak_memory() / 1e9, 3))
        if arm == "cand":
            o = fn(); mx.eval(o)
            # engagement forces the KERNEL (v6nax vs scalar). The kernel needs
            # block-aligned tensors, so pad when the cell is the non-aligned pad
            # case (auto_pad handles this inside the wrapper; here we pad for the
            # direct _ext fingerprint calls). Gold below uses the ORIGINAL N.
            qe, ke, ve = q, k, v
            if ap and (N % BT):
                Np = ((N + BT - 1) // BT) * BT
                _pd = lambda x: mx.pad(x, [(0, 0), (0, 0), (0, Np - x.shape[2]), (0, 0)])
                qe, ke, ve = _pd(q), _pd(k), _pd(v)
            sca = _ext.sparse_attention_forward(qe, ke, ve, bm, BT, causal, scale, "scalar_fallback", False, 0)
            mx.eval(sca)
            nax = _ext.sparse_attention_forward(qe, ke, ve, bm, BT, causal, scale, "v6nax_sparse", False, 0)
            mx.eval(nax)
            res["byteDelta_v6nax_vs_scalar"] = _maxabs(nax, sca)
            res["engaged"] = res["byteDelta_v6nax_vs_scalar"] > 0.0
            cos, oracle = _gold_cos(o, q, k, v, bm, scale, causal, N)
            res["cos_vs_gold"] = round(cos, 6); res["oracle"] = oracle
            res["correct"] = cos >= 0.999
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:180]}"
    res["ts_epoch"] = int(time.time())
    return res


# ----------------------------------------------------------------- cells
def _cells():
    C = []
    for N in (16384, 32768, 62752, 144288):
        for mask, d in (("sliding", 0.10), ("lcsa", 0.10), ("sliding", 0.50)):
            C.append({"B": 1, "H": 40, "N": N, "D": 128, "dtype": "float16",
                      "mask": mask, "density": d, "causal": False})
    C.append({"B": 1, "H": 40, "N": 16384, "D": 128, "dtype": "bfloat16", "mask": "sliding", "density": 0.10, "causal": False})
    C.append({"B": 1, "H": 40, "N": 62752, "D": 128, "dtype": "bfloat16", "mask": "sliding", "density": 0.10, "causal": False})
    C.append({"B": 1, "H": 40, "N": 32768, "D": 128, "dtype": "float16", "mask": "sliding", "density": 0.10, "causal": True})
    C.append({"B": 1, "H": 40, "N": 16384, "D": 128, "dtype": "float16", "mask": "random", "density": 0.95, "causal": False, "cutoff": 0.85})
    C.append({"B": 1, "H": 40, "N": 62730, "D": 128, "dtype": "float16", "mask": "sliding", "density": 0.10, "causal": False, "auto_pad": True})
    for i, c in enumerate(C):
        c["cell_id"] = f"{c['mask']}_N{c['N']}_{c['dtype'][:4]}_d{c.get('density')}_c{int(c['causal'])}" + ("_cut" if "cutoff" in c else "")
    return C


def _arms(cell):
    if "cutoff" in cell:
        return ("cand", "raw_dense", "sdpa_mask")
    return ("cand", "sdpa_nomask", "sdpa_mask")


def _procs(N):
    return 5 if N >= 62752 else 8


def parent(out):
    cells = _cells(); self = os.path.abspath(__file__)
    done = {}
    if os.path.exists(out):
        for l in open(out):
            try:
                r = json.loads(l); done[(r["cell_id"], r["arm"], r["proc"])] = 1
            except Exception:
                pass
    print(f"campaign: {len(cells)} cells → {out}", flush=True)
    for spec in cells:
        P = _procs(spec["N"])
        for p in range(P):
            for arm in _arms(spec):
                if (spec["cell_id"], arm, p) in done:
                    continue
                r = subprocess.run([sys.executable, self, "single", json.dumps(spec), arm],
                                   capture_output=True, text=True, timeout=2400)
                rec = None
                for ln in reversed((r.stdout or "").strip().splitlines()):
                    if ln.startswith("{"):
                        rec = json.loads(ln); break
                if rec is None:
                    rec = {"cell_id": spec["cell_id"], "arm": arm, "error": "no-json",
                           "stderr": (r.stderr or "")[-200:]}
                rec["proc"] = p
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        print(f"  {spec['cell_id']}: {P} procs × {len(_arms(spec))} arms done", flush=True)
    print("campaign done", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "single":
        print(json.dumps(measure(json.loads(sys.argv[2]), sys.argv[3])))
    elif sys.argv[1] == "parent":
        parent(sys.argv[2])
