#!/usr/bin/env python3
"""Release smoke: every ``metal_kernel``-built kernel x every MLX version of the ABI table.

Why (2.62.3): the V6NAX sparse kernel is compiled at runtime by MLX's ``metal_kernel``
JIT, which prepends MLX's OWN Metal headers — so the kernel source can break when MLX
changes those headers, independently of our build.  2.62.2 added MLX 0.32.1/0.32.2 to the
ABI table (BLD-01) and its install smoke only exercised dense kernels: the sparse kernel
did not compile on those versions and `flash_attention_sparse` raised on its default
route.  This tool closes that gap and is a release gate (release audit + publish.yml).

What it does, per MLX version X of ``csrc/cmake/MlxNanobindAbi.cmake``:
  1. fresh venv; ``pip install <sdist>`` with BUILD ISOLATION and ``PIP_CONSTRAINT`` =
     ``mlx==X`` (the isolated build env and the runtime both get MLX X — the build's MLX
     is read back from ``_ext._mlx_build_version()``, so the matrix proves which binary);
  2. runs ``--probe`` in that venv from a neutral cwd (never the checkout): one call per
     kernel family, each compared with an independent reference;
  3. records everything in a receipt ``release-gate/metal-kernel-matrix-<version>.json``
     (checked by scripts/check_metal_kernel_matrix.py).

Usage:
  .venv/bin/python scripts/metal_kernel_matrix_smoke.py --sdist dist/mlx_mfa-X.tar.gz
  .venv/bin/python scripts/metal_kernel_matrix_smoke.py --sdist ... --versions 0.32.2
  <venv>/bin/python scripts/metal_kernel_matrix_smoke.py --probe      # internal
Exit code 0 only if every kernel passes on every version.  Takes minutes (one build per
MLX version) — run it detached (nohup) on an M5+ host.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ABI_TABLE = os.path.join(_REPO, "csrc", "cmake", "MlxNanobindAbi.cmake")

# Every production ``metal_kernel`` call site -> the probe that exercises it.  Locked by
# tests/test_metal_kernel_matrix_tool.py against the source (a new site without a probe
# fails the suite).  Dev-only modules excluded from the package are not listed.
SITES = {
    "csrc/mfa_sparse_attention.cpp:sparse_attention_forward": ("sparse_v6nax", "sparse_scalar"),
    "csrc/mfa_sparse_attention.cpp:sparse_attention_forward_with_lse": ("sparse_v6nax_lse", "sparse_scalar_lse"),
    "csrc/mfa_gna_nax.cpp": ("gna_nax",),
    "csrc/mfa_ffn_nax.cpp": ("ffn_nax",),
    "csrc/mfa_qmm_nax.cpp": ("qmm_nax",),
    "csrc/mfa_conv_nax.cpp:dispatch_pointwise_fast_path": ("conv_pointwise",),
    "csrc/mfa_conv_nax.cpp:conv3d_mpp_dispatch": ("conv_mpp",),
    "csrc/mfa_conv_nax.cpp:conv3d_nax_forward": ("conv_im2col",),
    "mlx_mfa/conv_nax.py": ("conv_py_im2col", "conv_py_pointwise"),
    "mlx_mfa/attention.py": ("topk_bisect",),
    "mlx_mfa/tq_decode.py": ("tq_decode",),
}
# Kernels built by OUR ShaderCache (not metal_kernel) that embed the same shared NAX
# helpers block (csrc/mfa/v6_nax/NAAttentionKernel.cpp) — probed too, so a helpers resync
# can never silently change the dense default route.
SHARED_HELPER_CONSUMERS = {
    "csrc/mfa/v6_nax/NAAttentionKernel.cpp:forward": ("dense_v6nax_fwd",),
    "csrc/mfa/v6_nax/NAAttentionKernel.cpp:backward": ("dense_v6nax_bwd",),
}
PROBES = tuple(dict.fromkeys(p for ps in (*SITES.values(), *SHARED_HELPER_CONSUMERS.values())
                             for p in ps))


def abi_table_versions() -> list[str]:
    text = open(_ABI_TABLE, encoding="utf-8").read()
    return re.findall(r'^\s*"(\d+\.\d+\.\d+)=v[\d.]+=\d+"', text, re.M)


# ─────────────────────────────────────────────────────────────────────── probe (in venv)
def _probe() -> dict:
    import mlx.core as mx
    import numpy as np
    import mlx_mfa
    from mlx_mfa import _ext

    def rel(a, b):
        a = np.asarray(mx.array(a).astype(mx.float32)); b = np.asarray(mx.array(b).astype(mx.float32))
        return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-6))

    dump = {}
    current = [None]

    def keep(out):
        # First output of each probe is kept for --dump (bit-identity across builds).
        key = current[0] if current[0] not in dump else f"{current[0]}#{len(dump)}"
        dump[key] = np.asarray(mx.array(out).astype(mx.float32))

    def ok_close(out, ref, tol):
        mx.eval(out, ref)
        keep(out)
        finite = bool(mx.all(mx.isfinite(mx.array(out).astype(mx.float32))).item())
        r = rel(out, ref)
        return finite and r < tol, f"rel={r:.2e} finite={finite}"

    def sparse_ref(q, k, v, bm, bt, scale):
        N, S = q.shape[2], k.shape[2]
        with mx.stream(mx.cpu):
            tok = mx.repeat(mx.repeat(bm, bt, axis=-2), bt, axis=-1)[..., :N, :S]
            s = mx.where(tok, (q.astype(mx.float32) @ mx.swapaxes(k.astype(mx.float32), -1, -2)) * scale,
                         float("-inf"))
            o = mx.softmax(s, axis=-1) @ v.astype(mx.float32)
            o = mx.where(mx.any(tok, axis=-1, keepdims=True), o, 0.0)
            mx.eval(o)
        return o

    mx.random.seed(0)
    res = {}

    def run(name, fn):
        current[0] = name
        try:
            ok, detail = fn()
            res[name] = {"ok": bool(ok), "detail": detail}
        except Exception as e:  # recorded, never swallowed: the matrix FAILS on it
            res[name] = {"ok": False, "detail": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}

    N, D, H = 4096, 128, 12
    q, k, v = (mx.random.normal((1, H, N, D)).astype(mx.float16) for _ in range(3))
    nb = N // 32
    bm = (mx.random.uniform(shape=(nb, nb)) < 0.1) | mx.eye(nb, dtype=mx.bool_)
    sc = D ** -0.5
    ref32 = None

    def p_sparse_v6nax():
        # Public default route; which-binary by RUNTIME fingerprint: byte-identical to the
        # explicit v6nax_sparse kernel and different from the scalar kernel.
        nonlocal ref32
        ref32 = sparse_ref(q, k, v, bm, 32, sc)
        o = mlx_mfa.flash_attention_sparse(q, k, v, bm, scale=sc)
        o_nax = _ext.sparse_attention_forward(q, k, v, bm, block_tile=32, scale=sc,
                                              kernel_version="v6nax_sparse")
        o_sc = _ext.sparse_attention_forward(q, k, v, bm, block_tile=32, scale=sc,
                                             kernel_version="scalar_fallback")
        mx.eval(o, o_nax, o_sc)
        d_nax = float(mx.abs(o.astype(mx.float32) - o_nax.astype(mx.float32)).max().item())
        d_sc = float(mx.abs(o_nax.astype(mx.float32) - o_sc.astype(mx.float32)).max().item())
        engaged = d_nax == 0.0 and d_sc > 0.0
        ok, d = ok_close(o, ref32, 1e-2)
        return ok and engaged, f"{d} byteD(pub,nax)={d_nax:.1e} byteD(nax,scalar)={d_sc:.1e}"
    run("sparse_v6nax", p_sparse_v6nax)

    def p_sparse_scalar():
        o = _ext.sparse_attention_forward(q, k, v, bm, block_tile=32, scale=sc,
                                          kernel_version="scalar_fallback")
        return ok_close(o, ref32 if ref32 is not None else sparse_ref(q, k, v, bm, 32, sc), 1e-2)
    run("sparse_scalar", p_sparse_scalar)

    bm16 = mx.repeat(mx.repeat(bm, 2, axis=-2), 2, axis=-1)              # same token mask, BT16

    def p_sparse_v6nax_lse():
        # BT32 D128 fp16 -> the V6NAX LSE source; which-binary: differs (byteD > 0) from the
        # scalar LSE kernel (BT16) on the same token mask.
        o, _l = _ext.sparse_attention_forward_with_lse(q, k, v, bm, block_tile=32, scale=sc)
        o_sc, _ = _ext.sparse_attention_forward_with_lse(q, k, v, bm16, block_tile=16, scale=sc)
        mx.eval(o, o_sc)
        bd = float(mx.abs(o.astype(mx.float32) - o_sc.astype(mx.float32)).max().item())
        ok, d = ok_close(o, ref32 if ref32 is not None else sparse_ref(q, k, v, bm, 32, sc), 1e-2)
        return ok and bd > 0.0, f"{d} byteD(nax_lse,scalar_lse)={bd:.1e}"
    run("sparse_v6nax_lse", p_sparse_v6nax_lse)

    def p_sparse_scalar_lse():
        o, _l = _ext.sparse_attention_forward_with_lse(q, k, v, bm16, block_tile=16, scale=sc)
        return ok_close(o, ref32 if ref32 is not None else sparse_ref(q, k, v, bm, 32, sc), 1e-2)
    run("sparse_scalar_lse", p_sparse_scalar_lse)

    def p_gna_nax():
        seq, win, st = (2, 8, 16), (1, 3, 5), (1, 1, 2)
        Ng = seq[0] * seq[1] * seq[2]
        g = [mx.random.normal((1, 1, Ng, 128)).astype(mx.float16) for _ in range(3)]
        nax = _ext.mfa_gna_nax_forward(*g, *seq, *win, *st, 128 ** -0.5)
        steel = _ext.mfa_gna_forward(*g, 128 ** -0.5, *seq, *win, *st)
        return ok_close(nax, steel, 2e-2)
    run("gna_nax", p_gna_nax)

    def p_ffn_nax():
        x = (mx.random.normal((2, 32, 256)) * 0.05).astype(mx.float16)
        w = (mx.random.normal((128, 256)) * 0.02).astype(mx.float16)
        b = (mx.random.normal((128,)) * 0.01).astype(mx.float16)
        y = _ext.v6_nax_linear(x, w, b, False)
        ref = x.astype(mx.float32) @ w.astype(mx.float32).T + b.astype(mx.float32)
        return ok_close(y, ref, 2e-2)
    run("ffn_nax", p_ffn_nax)

    def p_qmm_nax():
        x = mx.random.normal((128, 512)).astype(mx.float16)
        w = mx.random.normal((128, 512)).astype(mx.float16)
        wq, s, bi = mx.quantize(w, group_size=64, bits=4)
        y = _ext.v6_nax_quantized_matmul(x, wq, s, bi, 64, 4)
        ref = mx.quantized_matmul(x, wq, s, bi, transpose=True, group_size=64, bits=4)
        return ok_close(y, ref, 2e-2)
    run("qmm_nax", p_qmm_nax)

    def conv_ref(x, w, pad):
        with mx.stream(mx.cpu):
            r = mx.conv_general(x.astype(mx.float32), w.astype(mx.float32), stride=1, padding=pad)
            mx.eval(r)
        return r

    xc = (mx.random.normal((1, 8, 32, 32, 128)) * 0.5).astype(mx.float16)
    w3 = (mx.random.normal((128, 3, 3, 3, 128)) * 0.1).astype(mx.float16)
    w1 = (mx.random.normal((128, 1, 1, 1, 128)) * 0.1).astype(mx.float16)
    xo = (mx.random.normal((1, 6, 30, 30, 64)) * 0.5).astype(mx.float16)   # H_out % 8 != 0
    w3o = (mx.random.normal((64, 3, 3, 3, 64)) * 0.1).astype(mx.float16)
    kw = dict(stride=(1, 1, 1), dilation=(1, 1, 1))
    p1, p0 = (1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)

    def with_env(name, fn):
        os.environ[name] = "1"
        try:
            out = fn(); mx.eval(out)
            return out
        finally:
            del os.environ[name]

    def p_conv_sub(x, w, pad, ref_pad, opt_out):
        # Conv sub-path selection is a deterministic function of the shape (the probe
        # shapes sit inside each sub-path's eligibility envelope, csrc/mfa_conv_nax.cpp),
        # and no conv/kernel C++ site catches (locked by the tool's test), so a kernel
        # that fails to build RAISES here.  byteD vs the opt-out arm is informational only:
        # MPP / pointwise / im2col are bit-identical at fp16 for these shapes (measured on
        # M5 / MLX 0.31.2: byteD 0.0 with a 1.74 vs 2.51 ms path split), so it cannot prove
        # engagement.
        out = _ext.conv3d_nax_forward(x, w, padding=pad, **kw); mx.eval(out)
        ok, d = ok_close(out, conv_ref(x, w, ref_pad), 2e-2)
        if opt_out is None:
            return ok, d
        alt = with_env(opt_out, lambda: _ext.conv3d_nax_forward(x, w, padding=pad, **kw))
        ok2, _ = ok_close(alt, conv_ref(x, w, ref_pad), 2e-2)
        bd = float(mx.abs(out.astype(mx.float32) - alt.astype(mx.float32)).max().item())
        return ok and ok2, f"{d} byteD(vs {opt_out})={bd:.1e} (info)"
    run("conv_mpp", lambda: p_conv_sub(xc, w3, p1, 1, "MFA_DISABLE_CONV3D_MPP"))
    run("conv_pointwise", lambda: p_conv_sub(xc, w1, p0, 0, "MFA_CONV_NAX_NO_FAST_PATH"))
    run("conv_im2col", lambda: p_conv_sub(xo, w3o, p1, 1, None))   # H_out % 8 != 0: not MPP

    def p_conv_py(w, x, pad):
        os.environ["MFA_CONV_NAX_USE_PYTHON_LEGACY"] = "1"
        try:
            from mlx_mfa.conv_nax import conv3d_nax_forward
            out = conv3d_nax_forward(x, w, stride=(1, 1, 1), padding=(pad, pad, pad))
            return ok_close(out, conv_ref(x, w, pad), 2e-2)
        finally:
            del os.environ["MFA_CONV_NAX_USE_PYTHON_LEGACY"]
    run("conv_py_im2col", lambda: p_conv_py(w3o, xo, 1))
    run("conv_py_pointwise", lambda: p_conv_py(w1, xc, 0))

    def p_topk():
        # Bisection kernel (default) vs the mx.topk path: both are top-k approximations that
        # may pick different boundary elements on fp16 score ties (documented), so the gate
        # is per-row: >= 99% of rows agree to 2e-2 and everything is finite.
        qt, kt, vt = (mx.random.normal((1, 4, 1024, 64)).astype(mx.float16) for _ in range(3))
        o = mlx_mfa.flash_attention_topk(qt, kt, vt, topk_ratio=0.1)
        ref = with_env("MFA_DISABLE_TOPK_BISECT",
                       lambda: mlx_mfa.flash_attention_topk(qt, kt, vt, topk_ratio=0.1))
        mx.eval(o)
        finite = bool(mx.all(mx.isfinite(o.astype(mx.float32))).item())
        row = mx.abs(o.astype(mx.float32) - ref.astype(mx.float32)).max(axis=-1)
        agree = float(mx.mean(row < 2e-2).item())
        return finite and agree >= 0.99, f"rows_agree={agree:.4f} finite={finite}"
    run("topk_bisect", p_topk)

    def p_tq():
        from mlx_mfa.tq_decode import tq_decode_attend, _packed_d
        nbk, bs, Hkv, Dt, bits = 4, 16, 2, 64, 4
        qd = mx.random.normal((1, 4, 1, Dt)).astype(mx.float16)
        ktq = mx.random.randint(0, 255, (nbk, bs, Hkv, _packed_d(Dt, bits))).astype(mx.uint8)
        vp = mx.random.normal((nbk, bs, Hkv, Dt)).astype(mx.float16)
        ks = mx.ones((nbk, bs, Hkv), mx.float32)
        cent = mx.linspace(-1, 1, 2 ** bits).astype(mx.float16)
        bt = mx.array([0, 1, 2], mx.int32)
        o = tq_decode_attend(qd, ktq, vp, ks, cent, bt, 40, block_size=bs, tq_bits=bits)
        mx.eval(o)
        finite = bool(mx.all(mx.isfinite(o.astype(mx.float32))).item())
        return finite and o.shape == (1, 4, 1, Dt), f"finite={finite} shape={tuple(o.shape)}"
    run("tq_decode", p_tq)

    def p_dense_fwd():
        # Default public dense route on M5: D=128 -> v6_nax_forward (dispatch-map lock);
        # engagement by the dispatch trace terminal, correctness vs a CPU fp32 oracle.
        from mlx_mfa import _dispatch_trace as dt
        with dt.capture() as cap:
            o = mlx_mfa.flash_attention(q, k, v, scale=sc)
            mx.eval(o)
        with mx.stream(mx.cpu):
            ref = mx.fast.scaled_dot_product_attention(
                q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=sc)
            mx.eval(ref)
        ok, d = ok_close(o, ref, 1e-2)
        term = cap[-1][0] if cap else None
        return ok and term == "nax_dense", f"{d} terminal={term}"
    run("dense_v6nax_fwd", p_dense_fwd)

    def p_dense_bwd():
        # Default D=64 backward on M5 (qL >= 2048, docs/reference/dispatch-map.md):
        # V6NAX split dQ/dV/dK — engagement by the trace terminal.
        from mlx_mfa import _dispatch_trace as dt
        qb, kb, vb = (mx.random.normal((1, 4, 2048, 64)).astype(mx.float16) for _ in range(3))
        g = mx.random.normal((1, 4, 2048, 64)).astype(mx.float16)
        f = lambda a, b, c: (mlx_mfa.flash_attention(a, b, c) * g).sum()
        with dt.capture() as cap:
            grads = mx.grad(f, argnums=(0, 1, 2))(qb, kb, vb)
            mx.eval(grads)
        with mx.stream(mx.cpu):
            r = lambda a, b, c: (mx.fast.scaled_dot_product_attention(
                a, b, c, scale=64 ** -0.5) * g.astype(mx.float32)).sum()
            refs = mx.grad(r, argnums=(0, 1, 2))(*(x.astype(mx.float32) for x in (qb, kb, vb)))
            mx.eval(refs)
        oks = [ok_close(a, b, 2e-2) for a, b in zip(grads, refs)]
        engaged = any(b == "v6_split_backward" for b, _ in cap)
        return all(o for o, _ in oks) and engaged, \
            " ".join(d for _, d in oks) + f" v6_split_backward={engaged}"
    run("dense_v6nax_bwd", p_dense_bwd)

    if os.environ.get("MFA_MATRIX_DUMP"):
        np.savez(os.environ["MFA_MATRIX_DUMP"], **dump)
    return {"mlx": mx.__version__, "mlx_mfa": mlx_mfa.__version__,
            "build_mlx": _ext._mlx_build_version(), "has_nax": bool(mlx_mfa.has_nax()),
            "results": res}


# ─────────────────────────────────────────────────────────────────────── driver
def _sh(cmd, env=None, cwd=None):
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)


def _run_version(ver: str, sdist: str, work: str, base_python: str,
                 dump_dir: str | None = None) -> dict:
    venv = os.path.join(work, f"venv-{ver}")
    t0 = time.time()
    r = _sh([base_python, "-m", "venv", venv])
    if r.returncode:
        return {"error": f"venv: {r.stderr[-400:]}"}
    py = os.path.join(venv, "bin", "python")
    cons = os.path.join(work, f"constraints-{ver}.txt")
    with open(cons, "w") as f:
        f.write(f"mlx=={ver}\n")
    env = dict(os.environ, PIP_CONSTRAINT=cons, PIP_DISABLE_PIP_VERSION_CHECK="1")
    env.pop("PYTHONPATH", None)
    r = _sh([py, "-m", "pip", "install", "-q", "--no-cache-dir", sdist], env=env)
    if r.returncode:
        return {"error": f"install (isolated, mlx=={ver}): {r.stderr[-800:]}"}
    penv = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        penv["MFA_MATRIX_DUMP"] = os.path.join(os.path.abspath(dump_dir), f"mlx-{ver}.npz")
    r = _sh([py, os.path.abspath(__file__), "--probe"], cwd=work, env=penv)
    if r.returncode:
        return {"error": f"probe crashed: {r.stderr[-800:]}"}
    out = json.loads(r.stdout.strip().splitlines()[-1])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--sdist")
    ap.add_argument("--versions", default=None, help="comma list (default: the ABI table)")
    ap.add_argument("--work", default=None)
    ap.add_argument("--receipt", default=None)
    ap.add_argument("--dump-dir", default=None,
                    help="save every probe's first output per version (bit-identity checks)")
    args = ap.parse_args()

    if args.probe:
        print(json.dumps(_probe()))
        return 0

    if not args.sdist or not os.path.exists(args.sdist):
        print("--sdist <path to the built sdist> is required", file=sys.stderr)
        return 2
    sdist = os.path.abspath(args.sdist)
    m = re.search(r"mlx_mfa-(\d+\.\d+\.\d+)\.tar\.gz$", sdist)
    pkg_version = m.group(1) if m else "unknown"
    versions = args.versions.split(",") if args.versions else abi_table_versions()
    work = args.work or tempfile.mkdtemp(prefix="mfa-kernel-matrix-")
    base_python = getattr(sys, "_base_executable", sys.executable)
    matrix = {}
    for ver in versions:
        print(f"[matrix] MLX {ver} ...", flush=True)
        res = _run_version(ver, sdist, work, base_python, args.dump_dir)
        matrix[ver] = res
        if "error" in res:
            print(f"[matrix] MLX {ver}: ERROR {res['error'][:300]}", flush=True)
            continue
        bad = [k for k, v in res["results"].items() if not v["ok"]]
        wrong_build = res.get("build_mlx") != ver or res.get("mlx") != ver
        print(f"[matrix] MLX {ver}: build={res.get('build_mlx')} runtime={res.get('mlx')} "
              f"{'ALL OK' if not bad and not wrong_build else 'FAIL ' + str(bad)} "
              f"({res.get('seconds')} s)", flush=True)
    all_pass = all(
        "error" not in r and r.get("build_mlx") == v and r.get("mlx") == v
        and set(r["results"]) >= set(PROBES) and all(x["ok"] for x in r["results"].values())
        for v, r in matrix.items())
    git_sha = _sh(["git", "-C", _REPO, "rev-parse", "HEAD"]).stdout.strip() or "UNKNOWN"
    git_dirty = bool(_sh(["git", "-C", _REPO, "status", "--porcelain", "--",
                          "csrc", "mlx_mfa"]).stdout.strip())
    try:
        import mlx_mfa
        device = mlx_mfa.get_device_info().get("device_name", "?")
    except Exception as e:
        device = f"unknown ({type(e).__name__})"
    receipt = {
        "release_version": pkg_version,
        "sdist_sha256": hashlib.sha256(open(sdist, "rb").read()).hexdigest(),
        "git_sha": git_sha,
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "device": device,
        "abi_table_versions": abi_table_versions(),
        "probes": list(PROBES),
        "git_dirty": git_dirty,
        "matrix": matrix,
        "matrix_sha256": hashlib.sha256(json.dumps(matrix, sort_keys=True).encode()).hexdigest(),
        "all_pass": bool(all_pass and set(versions) >= set(abi_table_versions())),
    }
    if args.receipt is None and git_dirty:
        # A release receipt binds to git_sha: refuse to write one for uncommitted source.
        print("[matrix] csrc/ or mlx_mfa/ has uncommitted changes — commit first, or pass "
              "--receipt <scratch path> for a non-release run.", file=sys.stderr)
        return 2
    path = args.receipt or os.path.join(_REPO, "release-gate",
                                        f"metal-kernel-matrix-{pkg_version}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(receipt, f, indent=1)
    print(f"[matrix] receipt -> {os.path.relpath(path, _REPO)}  all_pass={receipt['all_pass']}")
    return 0 if receipt["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
