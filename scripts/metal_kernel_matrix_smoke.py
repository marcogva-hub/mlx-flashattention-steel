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
    "csrc/mfa_sparse_attention.cpp:sparse_attention_forward": ("sparse_v6nax", "sparse_variants"),
    "csrc/mfa_sparse_attention.cpp:sparse_attention_forward_with_lse": ("sparse_lse_variants",
                                                                        "sparse_lse_fingerprint"),
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
# can never silently change the dense routes (not MLX-version-sensitive; bit-identity).
SHARED_HELPER_CONSUMERS = {
    "csrc/mfa/v6_nax/NAAttentionKernel.cpp:forward": ("dense_v6nax_fwd",),
    "csrc/mfa/v6_nax/NAAttentionKernel.cpp:backward": ("dense_v6nax_bwd",),
}
PROBES = tuple(dict.fromkeys(p for ps in (*SITES.values(), *SHARED_HELPER_CONSUMERS.values())
                             for p in ps))

# Pre-existing defects the matrix must keep SEEING (strict: the cell must still fail; if it
# passes the probe FAILS until the entry is removed).  Every entry is reported in the
# receipt and by the checker.  Adding one needs a maintainer decision.
KNOWN_FAILURES: dict = {}   # (2.63.0: conv_pointwise/bf16 fixed — the matmul2d source follows the dtype)


def abi_table_versions() -> list[str]:
    text = open(_ABI_TABLE, encoding="utf-8").read()
    return re.findall(r'^\s*"(\d+\.\d+\.\d+)=v[\d.]+=\d+"', text, re.M)


# ─────────────────────────────────────────────────────────────────────── probe (in venv)
def _probe() -> dict:
    """Runs inside the per-version venv.  Every probe is a SWEEP over the variant axes the
    kernel source is specialised on (metal_kernel compiles one library per dtype / D /
    causal / mask rank / tile / flag — a construct behind a branch only builds when that
    variant is instantiated), each cell checked against an independent reference."""
    import math

    # Knobs that would turn probes into vacuous passes must not leak in (the driver strips
    # them; recorded so the checker can refuse a receipt produced with any set).  Read
    # BEFORE importing mlx_mfa, which seeds its own MFA_* tuning defaults on import.
    leaked_env = sorted(k for k in os.environ if k.startswith("MFA_") and k != "MFA_MATRIX_DUMP")

    import mlx.core as mx
    import numpy as np
    import mlx_mfa
    from mlx_mfa import _ext

    dump, current, res = {}, [None], {}
    f32 = lambda a: mx.array(a).astype(mx.float32)

    def keep(out):
        key = current[0] if current[0] not in dump else f"{current[0]}#{len(dump)}"
        dump[key] = np.asarray(f32(out))

    def check(out, ref, tol):
        """(ok, detail): finite and max|out-ref| / max|ref| < tol; output kept for --dump."""
        mx.eval(out, ref)
        keep(out)
        finite = bool(mx.all(mx.isfinite(f32(out))).item())
        r = float(mx.abs(f32(out) - f32(ref)).max().item()) / max(float(mx.abs(f32(ref)).max().item()), 1e-6)
        return finite and r < tol, f"rel={r:.1e}{'' if finite else ' NONFINITE'}"

    def bytediff(a, b):
        mx.eval(a, b)
        return float(mx.abs(f32(a) - f32(b)).max().item())

    def sweep(cells):
        """cells: [(label, thunk -> (ok, detail))].  Every exception is a FAILED cell.  A
        cell listed in KNOWN_FAILURES must still fail (strict): if it passes, the marker is
        stale and the cell FAILS; its expected failure is reported, never hidden."""
        bad, known, n = [], [], 0
        for label, thunk in cells:
            name = current[0].split("/")[0]
            current[0] = f"{name}/{label}"
            try:
                ok, d = thunk()
            except Exception as e:  # recorded, never swallowed: the cell FAILS
                ok, d = False, f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
            n += 1
            if current[0] in KNOWN_FAILURES:
                if ok:
                    bad.append(f"{label}: KNOWN FAILURE NOW PASSES — remove it from KNOWN_FAILURES")
                else:
                    known.append(f"{label}: {d}")
            elif not ok:
                bad.append(f"{label}: {d}")
            current[0] = name
        return not bad, (f"{n - len(bad) - len(known)}/{n} cells ok"
                         + (f"; KNOWN-FAIL (expected) {' | '.join(known)}" if known else "")
                         + ("; FAILED " + " | ".join(bad) if bad else ""))

    def run(name, fn):
        current[0] = name
        try:
            ok, detail = fn()
            res[name] = {"ok": bool(ok), "detail": detail}
        except Exception as e:  # recorded, never swallowed: the matrix FAILS on it
            res[name] = {"ok": False, "detail": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}

    def with_env(pairs, fn):
        prev = {k: os.environ.get(k) for k in pairs}
        os.environ.update(pairs)
        try:
            out = fn()
            mx.eval(out)
            return out
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def rnd(shape, dt=mx.float16, s=1.0):
        return (mx.random.normal(shape) * s).astype(dt)

    mx.random.seed(0)

    # ─────────────────────────────── block-sparse (V6NAX / scalar, forward / LSE)
    def sparse_ref(q, k, v, bm, bt, scale, causal=False, lse=False):
        N, S = q.shape[2], k.shape[2]
        with mx.stream(mx.cpu):
            tok = mx.repeat(mx.repeat(bm, bt, axis=-2), bt, axis=-1)[..., :N, :S]
            if q.shape[1] != k.shape[1]:                               # GQA
                rep = q.shape[1] // k.shape[1]
                k, v = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
            if causal:
                tok = tok & (mx.arange(S)[None, :] <= mx.arange(N)[:, None])
            s = mx.where(tok, (f32(q) @ mx.swapaxes(f32(k), -1, -2)) * scale, float("-inf"))
            o = mx.softmax(s, axis=-1) @ f32(v)
            o = mx.where(mx.any(tok, axis=-1, keepdims=True), o, 0.0)
            L = mx.logsumexp(s, axis=-1)
            mx.eval(o, L)
        return (o, L) if lse else o

    def mask(nq, nk, density=0.15, lead=()):
        m = mx.random.uniform(shape=(*lead, nq, nk)) < density
        return m | mx.eye(nq, nk, dtype=mx.bool_)

    # Public default route (M5 NAX) + which-binary: byte-identical to the explicit V6NAX
    # kernel and different from the scalar kernel.
    N, D, H = 4096, 128, 12
    q, k, v = rnd((1, H, N, D)), rnd((1, H, N, D)), rnd((1, H, N, D))
    bm = mask(N // 32, N // 32, 0.1)
    sc = D ** -0.5

    def p_sparse_public():
        ref = sparse_ref(q, k, v, bm, 32, sc)
        o = mlx_mfa.flash_attention_sparse(q, k, v, bm, scale=sc)
        o_nax = _ext.sparse_attention_forward(q, k, v, bm, block_tile=32, scale=sc,
                                              kernel_version="v6nax_sparse")
        o_sc = _ext.sparse_attention_forward(q, k, v, bm, block_tile=32, scale=sc,
                                             kernel_version="scalar_fallback")
        d_nax, d_sc = bytediff(o, o_nax), bytediff(o_nax, o_sc)
        ok, d = check(o, ref, 1e-2)
        return ok and d_nax == 0.0 and d_sc > 0.0, \
            f"{d} byteD(pub,nax)={d_nax:.1e} byteD(nax,scalar)={d_sc:.1e}"
    run("sparse_v6nax", p_sparse_public)

    def sparse_cell(dt, Dh, causal, ndim, Hq=2, Hk=2, window=None, kv_valid=None):
        """V6NAX and scalar forward for one variant; which-binary: they differ (byteD>0).
        ``kv_valid`` (2.63.0): the in-kernel pad-key mask variant (kL-32 < kv_valid < kL);
        V6NAX-only, so it is checked against the reference alone."""
        Ns = 2048                                                  # 2-D mask = 4096 B (min)
        qs, ks, vs = rnd((1, Hq, Ns, Dh), dt), rnd((1, Hk, Ns, Dh), dt), rnd((1, Hk, Ns, Dh), dt)
        nb = Ns // 32
        lead = {2: (), 3: (Hq,), 4: (1, Hq)}[ndim]
        m = mask(nb, nb, 0.15, lead)
        scale = Dh ** -0.5
        kw = dict(block_tile=32, causal=causal, scale=scale)
        if window is not None:
            qi, kb = mx.arange(nb)[:, None], mx.arange(nb)[None, :]
            m = ((kb * 32 + 31) >= (qi * 32 + 16 - window)) & ((kb * 32) <= (qi * 32 + 16 + window))
            if causal:
                m = m & ((kb * 32) <= ((qi + 1) * 32 - 1))
            kw.update(structured_window_probe=True, structured_window_size=window)
        if kv_valid is not None:
            kw.update(kv_valid_len=kv_valid)
            with mx.stream(mx.cpu):                               # keys >= kv_valid do not exist
                ks_ref, vs_ref = ks[:, :, :kv_valid], vs[:, :, :kv_valid]
            ref = sparse_ref(qs, ks_ref, vs_ref, m, 32, scale, causal)
        else:
            ref = sparse_ref(qs, ks, vs, m, 32, scale, causal)
        o_nax = _ext.sparse_attention_forward(qs, ks, vs, m, kernel_version="v6nax_sparse", **kw)
        ok, d = check(o_nax, ref, 2e-2)
        if window is not None or kv_valid is not None:
            return ok, d
        o_sc = _ext.sparse_attention_forward(qs, ks, vs, m, kernel_version="scalar_fallback",
                                             block_tile=32, causal=causal, scale=scale)
        ok2, d2 = check(o_sc, ref, 2e-2)
        bd = bytediff(o_nax, o_sc)
        return ok and ok2 and bd > 0.0, f"nax {d} scalar {d2} byteD={bd:.1e}"

    cells = [(f"{'bf16' if dt == mx.bfloat16 else 'f16'}_D{Dh}_{'c' if c else 'nc'}_M2",
              (lambda dt=dt, Dh=Dh, c=c: sparse_cell(dt, Dh, c, 2)))
             for dt in (mx.float16, mx.bfloat16) for Dh in (64, 128) for c in (False, True)]
    cells += [(f"f16_D128_{'c' if c else 'nc'}_M{nd}", (lambda c=c, nd=nd: sparse_cell(mx.float16, 128, c, nd)))
              for nd in (3, 4) for c in (False, True)]
    cells += [("f16_D64_nc_GQA4:2", lambda: sparse_cell(mx.float16, 64, False, 2, Hq=4, Hk=2)),
              ("f16_D128_nc_window", lambda: sparse_cell(mx.float16, 128, False, 2, window=96)),
              ("bf16_D64_c_window", lambda: sparse_cell(mx.bfloat16, 64, True, 2, window=96)),
              ("f16_D128_nc_kvvalid2041", lambda: sparse_cell(mx.float16, 128, False, 2, kv_valid=2041)),
              ("bf16_D64_c_kvvalid2030", lambda: sparse_cell(mx.bfloat16, 64, True, 2, kv_valid=2030))]

    def sparse_auto_pad_public():
        # 2.63.0 public auto_pad route: non-aligned N=4090 is padded to 4096 and the pad keys
        # masked in-kernel (kv_valid_len); engagement by the dispatch-trace terminal.
        from mlx_mfa import _dispatch_trace as dtr
        Na = 4090
        qa, ka, va = q[:, :, :Na], k[:, :, :Na], v[:, :, :Na]
        bma = bm[: (Na + 31) // 32, : (Na + 31) // 32]
        with dtr.capture() as cap:
            o = mlx_mfa.flash_attention_sparse(qa, ka, va, bma, scale=sc, auto_pad=True)
            mx.eval(o)
        ok, d = check(o, sparse_ref(qa, ka, va, bma, 32, sc), 1e-2)
        engaged = any(b == "v6nax_sparse" for b, _ in cap)
        return ok and engaged and o.shape[2] == Na, f"{d} v6nax_sparse={engaged}"
    cells += [("public_auto_pad_N4090", sparse_auto_pad_public)]
    run("sparse_variants", lambda: sweep(cells))

    def lse_cell(dt, Dh, causal, bt, ndim=2):
        """BT32 -> V6NAX LSE source, BT16 -> scalar LSE source; O and natural-log L checked."""
        Ns = 2048
        qs, ks, vs = rnd((1, 2, Ns, Dh), dt), rnd((1, 2, Ns, Dh), dt), rnd((1, 2, Ns, Dh), dt)
        nb = Ns // bt
        m = mask(nb, nb, 0.15, {2: (), 3: (2,)}[ndim])
        scale = Dh ** -0.5
        o, L = _ext.sparse_attention_forward_with_lse(qs, ks, vs, m, block_tile=bt,
                                                       causal=causal, scale=scale)
        ref_o, ref_L = sparse_ref(qs, ks, vs, m, bt, scale, causal, lse=True)
        ok, d = check(o, ref_o, 2e-2)
        mx.eval(L)
        dl = float(mx.abs(f32(L).reshape(ref_L.shape) - ref_L).max().item())
        return ok and dl < 2e-2, f"{d} L_absdiff={dl:.1e}"

    cells = [(f"{'bf16' if dt == mx.bfloat16 else 'f16'}_D{Dh}_{'c' if c else 'nc'}_BT32",
              (lambda dt=dt, Dh=Dh, c=c: lse_cell(dt, Dh, c, 32)))
             for dt in (mx.float16, mx.bfloat16) for Dh in (64, 128) for c in (False, True)]
    cells += [("f16_D128_nc_BT32_M3", lambda: lse_cell(mx.float16, 128, False, 32, 3))]
    cells += [(f"{'bf16' if dt == mx.bfloat16 else 'f16'}_D64_{'c' if c else 'nc'}_BT16",
               (lambda dt=dt, c=c: lse_cell(dt, 64, c, 16)))
              for dt in (mx.float16, mx.bfloat16) for c in (False, True)]
    run("sparse_lse_variants", lambda: sweep(cells))

    def p_lse_fingerprint():
        # which-binary for the LSE pair: BT32 (V6NAX LSE) != BT16 (scalar LSE), same tokens.
        bm16 = mx.repeat(mx.repeat(bm, 2, axis=-2), 2, axis=-1)
        o, _ = _ext.sparse_attention_forward_with_lse(q, k, v, bm, block_tile=32, scale=sc)
        o_sc, _ = _ext.sparse_attention_forward_with_lse(q, k, v, bm16, block_tile=16, scale=sc)
        bd = bytediff(o, o_sc)
        ok, d = check(o, sparse_ref(q, k, v, bm, 32, sc), 1e-2)
        return ok and bd > 0.0, f"{d} byteD(nax_lse,scalar_lse)={bd:.1e}"
    run("sparse_lse_fingerprint", p_lse_fingerprint)

    # ─────────────────────────────── GNA NAX (exact per-element window oracle)
    def gna_mask(seq, win, st):
        Ng = int(np.prod(seq))
        coords = np.array(np.unravel_index(np.arange(Ng), seq)).T
        M = np.ones((Ng, Ng), bool)
        for d in range(len(seq)):
            pos = coords[:, d]
            gb = (pos // st[d]) * st[d]
            lo = np.maximum(gb - (win[d] - st[d]) // 2, 0)
            hi = np.minimum(gb + st[d] + (win[d] - st[d] + 1) // 2, seq[d])
            M &= (pos[None, :] >= lo[:, None]) & (pos[None, :] < hi[:, None])
        return M

    def gna_cell(dt, Dh, env=None, Hq=2, Hk=2, seq=(2, 8, 16), win=(1, 3, 5), st=(1, 1, 2)):
        Ng = int(np.prod(seq))
        qg, kg, vg = rnd((1, Hq, Ng, Dh), dt), rnd((1, Hk, Ng, Dh), dt), rnd((1, Hk, Ng, Dh), dt)
        scale = Dh ** -0.5
        call = lambda: _ext.mfa_gna_nax_forward(qg, kg, vg, *seq, *win, *st, scale)
        out = with_env(env, call) if env else call()
        with mx.stream(mx.cpu):
            kk, vv = kg, vg
            if Hq != Hk:
                kk, vv = mx.repeat(kg, Hq // Hk, axis=1), mx.repeat(vg, Hq // Hk, axis=1)
            s = (f32(qg) @ mx.swapaxes(f32(kk), -1, -2)) * scale
            s = mx.where(mx.array(gna_mask(seq, win, st))[None, None], s, float("-inf"))
            ref = mx.softmax(s, axis=-1) @ f32(vv)
            mx.eval(ref)
        return check(out, ref, 2e-2)

    cells = [(f"{'bf16' if dt == mx.bfloat16 else 'f16'}_D{Dh}", (lambda dt=dt, Dh=Dh: gna_cell(dt, Dh)))
             for dt in (mx.float16, mx.bfloat16) for Dh in (64, 128)]
    cells += [("f16_D128_BQ32WM2(N>=8192 tile)",
               lambda: gna_cell(mx.float16, 128, {"MFA_GNA_NAX_BQ": "32", "MFA_GNA_NAX_WM": "2"})),
              ("f16_D128_precompute_range", lambda: gna_cell(mx.float16, 128, {"MFA_GNA_NAX_PRECOMPUTE_RANGE": "1"})),
              ("f16_D128_swizzle1", lambda: gna_cell(mx.float16, 128, {"MFA_GNA_NAX_SWIZZLE_LOG": "1"})),
              ("f16_D64_GQA4:2", lambda: gna_cell(mx.float16, 64, Hq=4, Hk=2))]
    run("gna_nax", lambda: sweep(cells))

    # ─────────────────────────────── FFN / QMM NAX
    def ffn_cell(dt, gelu, M=64):
        x, w, b = rnd((2, M, 256), dt, 0.5), rnd((128, 256), dt, 0.05), rnd((128,), dt, 0.1)
        y = _ext.v6_nax_linear(x, w, b, gelu)
        with mx.stream(mx.cpu):
            ref = f32(x) @ f32(w).T + f32(b)
            if gelu:
                ref = 0.5 * ref * (1 + mx.tanh(math.sqrt(2 / math.pi) * (ref + 0.044715 * ref ** 3)))
            mx.eval(ref)
        return check(y, ref, 2e-2)

    cells = [(f"{'bf16' if dt == mx.bfloat16 else 'f16'}_gelu{int(g)}", (lambda dt=dt, g=g: ffn_cell(dt, g)))
             for dt in (mx.float16, mx.bfloat16) for g in (False, True)]
    cells += [("f16_M40(partial tile)", lambda: ffn_cell(mx.float16, False, M=40))]
    run("ffn_nax", lambda: sweep(cells))

    def qmm_cell(dt, bits, gs):
        x, w = rnd((128, 512), dt), rnd((128, 512), dt)
        wq, s, bi = mx.quantize(w, group_size=gs, bits=bits)
        y = _ext.v6_nax_quantized_matmul(x, wq, s, bi, gs, bits)
        with mx.stream(mx.cpu):
            ref = f32(x) @ f32(mx.dequantize(wq, s, bi, group_size=gs, bits=bits)).T
            mx.eval(ref)
        return check(y, ref, 2e-2)

    cells = [(f"f16_b{b}_g{g}", (lambda b=b, g=g: qmm_cell(mx.float16, b, g)))
             for b in (4, 8) for g in (32, 64, 128)]
    cells += [("bf16_b4_g64", lambda: qmm_cell(mx.bfloat16, 4, 64)),
              ("bf16_b8_g128", lambda: qmm_cell(mx.bfloat16, 8, 128))]
    run("qmm_nax", lambda: sweep(cells))

    # ─────────────────────────────── conv3d NAX (C++ and legacy Python)
    def conv_ref(x, w, stride, pad6):
        with mx.stream(mx.cpu):
            xp = mx.pad(f32(x), [(0, 0), (pad6[0], pad6[1]), (pad6[2], pad6[3]), (pad6[4], pad6[5]), (0, 0)])
            r = mx.conv_general(xp, f32(w), stride=stride)
            mx.eval(r)
        return r

    def conv_cell(xs, ws, dt, pad6=(1, 1, 1, 1, 1, 1), stride=(1, 1, 1), python=False):
        # Sub-path selection is a deterministic function of the shape (the cells sit inside
        # each sub-path's envelope, csrc/mfa_conv_nax.cpp) and no C++ kernel site catches, so
        # a kernel that fails to build RAISES here.  (MPP / pointwise / im2col are
        # bit-identical at fp16 for these shapes, so byteD cannot prove the sub-path.)
        x, w = rnd(xs, dt, 0.5), rnd(ws, dt, 0.1)
        if python:
            from mlx_mfa.conv_nax import conv3d_nax_forward as fwd
            sym = (pad6[0], pad6[2], pad6[4])                       # legacy: symmetric 3-tuple
            assert pad6 == (sym[0], sym[0], sym[1], sym[1], sym[2], sym[2]), pad6
            out = with_env({"MFA_CONV_NAX_USE_PYTHON_LEGACY": "1"},
                           lambda: fwd(x, w, stride=stride, padding=sym))
        else:
            out = _ext.conv3d_nax_forward(x, w, stride=stride, padding=pad6, dilation=(1, 1, 1))
        return check(out, conv_ref(x, w, stride, pad6), 2e-2)

    F16, BF16 = mx.float16, mx.bfloat16
    run("conv_mpp", lambda: sweep([
        ("f16_KT3_tile8", lambda: conv_cell((1, 8, 32, 32, 128), (128, 3, 3, 3, 128), F16)),
        ("bf16_KT3_tile8", lambda: conv_cell((1, 8, 32, 32, 128), (128, 3, 3, 3, 128), BF16)),
        ("f16_KT3_tile16", lambda: conv_cell((1, 8, 64, 64, 64), (64, 3, 3, 3, 64), F16)),
        ("f16_KT3_padT0", lambda: conv_cell((1, 8, 32, 32, 64), (64, 3, 3, 3, 64), F16, (0, 0, 1, 1, 1, 1))),
        ("f16_KT4_sT2", lambda: conv_cell((1, 9, 34, 34, 64), (64, 4, 3, 3, 64), F16, (0, 0, 0, 0, 0, 0), (2, 1, 1))),
    ]))
    run("conv_pointwise", lambda: sweep([
        ("f16", lambda: conv_cell((1, 8, 32, 32, 128), (128, 1, 1, 1, 128), F16, (0,) * 6)),
        ("bf16", lambda: conv_cell((1, 8, 32, 32, 128), (128, 1, 1, 1, 128), BF16, (0,) * 6)),
    ]))
    run("conv_im2col", lambda: sweep([          # fp16 only: bf16 raises there by design
        ("f16_HW30(not MPP)", lambda: conv_cell((1, 6, 30, 30, 64), (64, 3, 3, 3, 64), F16)),
        ("f16_K133", lambda: conv_cell((1, 4, 16, 16, 64), (64, 1, 3, 3, 64), F16, (0, 0, 1, 1, 1, 1))),
    ]))
    # Legacy Python orchestrator (MFA_CONV_NAX_USE_PYTHON_LEGACY=1): fp16-only by design.
    run("conv_py_im2col", lambda: sweep([
        ("f16", lambda: conv_cell((1, 6, 30, 30, 64), (64, 3, 3, 3, 64), F16, python=True)),
    ]))
    run("conv_py_pointwise", lambda: sweep([
        ("f16", lambda: conv_cell((1, 8, 32, 32, 128), (128, 1, 1, 1, 128), F16, (0,) * 6, python=True)),
    ]))

    # ─────────────────────────────── top-k bisection threshold kernel
    def topk_cell(dt, Dh):
        # Bisection (default) vs the mx.topk path: both are top-k approximations that may
        # pick different boundary elements on fp16 score ties (documented), so the gate is
        # per-row: >= 99% of rows agree to 2e-2 and everything is finite.
        qt, kt, vt = rnd((1, 4, 1024, Dh), dt), rnd((1, 4, 1024, Dh), dt), rnd((1, 4, 1024, Dh), dt)
        o = mlx_mfa.flash_attention_topk(qt, kt, vt, topk_ratio=0.1)
        mx.eval(o)
        keep(o)
        ref = with_env({"MFA_DISABLE_TOPK_BISECT": "1"},
                       lambda: mlx_mfa.flash_attention_topk(qt, kt, vt, topk_ratio=0.1))
        finite = bool(mx.all(mx.isfinite(f32(o))).item())
        agree = float(mx.mean(mx.abs(f32(o) - f32(ref)).max(axis=-1) < 2e-2).item())
        return finite and agree >= 0.99, f"rows_agree={agree:.4f}{'' if finite else ' NONFINITE'}"

    run("topk_bisect", lambda: sweep([
        (f"{'bf16' if dt == mx.bfloat16 else 'f16'}_D{Dh}", (lambda dt=dt, Dh=Dh: topk_cell(dt, Dh)))
        for dt in (mx.float16, mx.bfloat16) for Dh in (64, 128)]))

    # ─────────────────────────────── TurboQuant paged decode (Python ground truth)
    def tq_cell(bits, Dh, Hq=8, Hkv=2, S0=256, BS=64):
        from mlx_mfa.inference import TurboQuantPagedInferenceContext
        from mlx_mfa.tq_decode import tq_decode_attend
        from mlx_mfa.turboquant import (apply_rotation, unpack_indices, unpack_3bit_optimal,
                                        dequantize_from_indices, _compute_packed_d)
        ctx = TurboQuantPagedInferenceContext(num_blocks=S0 // BS + 4, block_size=BS, H_kv=Hkv,
                                              D=Dh, tq_bits=bits)
        k0, v0, q0 = rnd((1, Hkv, S0, Dh)), rnd((1, Hkv, S0, Dh)), rnd((1, Hq, S0, Dh))
        mx.eval(ctx.prefill(q0, k0, v0))
        qr = apply_rotation(f32(rnd((1, Hq, 1, Dh))), "wht").astype(mx.float16)
        S = ctx.seq_length(0)
        tbl = ctx.get_block_table([0])[0][:(S + BS - 1) // BS]
        scale = Dh ** -0.5
        out = tq_decode_attend(qr, ctx._k_pool, ctx._v_pool_fp16, ctx._k_scales, ctx._k_centroids,
                               tbl, S, scale=scale, block_size=BS, tq_bits=bits)
        p, s = ctx._k_pool[tbl], ctx._k_scales[tbl]
        nbk = p.shape[0]
        if bits == 3:
            idx = unpack_3bit_optimal(p.reshape(nbk * BS * Hkv, _compute_packed_d(Dh, 3)), Dh)
        else:
            idx = unpack_indices(p.reshape(-1), nbk * BS * Hkv * Dh, bits).reshape(nbk * BS * Hkv, Dh)
        Kd = (dequantize_from_indices(idx, bits) * s.reshape(-1)[:, None]).reshape(nbk * BS, Hkv, Dh)[:S]
        K = mx.transpose(Kd, (1, 0, 2))[None].astype(mx.float16)
        V = mx.transpose(ctx._v_pool_fp16[tbl].reshape(-1, Hkv, Dh)[:S], (1, 0, 2))[None]
        with mx.stream(mx.cpu):
            rep = Hq // Hkv
            K32, V32 = mx.repeat(f32(K), rep, axis=1), mx.repeat(f32(V), rep, axis=1)
            ref = mx.softmax((f32(qr) @ mx.swapaxes(K32, -1, -2)) * scale, axis=-1) @ V32
            mx.eval(ref)
        return check(out, ref, 2e-2)

    run("tq_decode", lambda: sweep([(f"b{b}_D128", (lambda b=b: tq_cell(b, 128))) for b in (2, 3, 4)]
                                   + [("b4_D64", lambda: tq_cell(4, 64))]))

    # ─────────────────────────────── dense V6 (ShaderCache consumers of the shared helpers)
    def dense_ref(qd, kd, vd, causal, scale):
        with mx.stream(mx.cpu):
            r = mx.fast.scaled_dot_product_attention(f32(qd), f32(kd), f32(vd), scale=scale,
                                                     mask="causal" if causal else None)
            mx.eval(r)
        return r

    def dense_fwd_public():
        from mlx_mfa import _dispatch_trace as dt
        with dt.capture() as cap:
            o = mlx_mfa.flash_attention(q, k, v, scale=sc)
            mx.eval(o)
        ok, d = check(o, dense_ref(q, k, v, False, sc), 1e-2)
        term = cap[-1][0] if cap else None
        return ok and term == "nax_dense", f"{d} terminal={term}"

    def dense_fwd_cell(dt, Dh, causal):
        qd, kd, vd = rnd((1, 4, 1024, Dh), dt), rnd((1, 4, 1024, Dh), dt), rnd((1, 4, 1024, Dh), dt)
        o, _L = _ext.v6_nax_forward(qd, kd, vd, causal, True, -1.0)
        return check(o, dense_ref(qd, kd, vd, causal, Dh ** -0.5), 2e-2)

    cells = [("public_D128_f16(nax_dense)", dense_fwd_public)]
    cells += [(f"{'bf16' if dt == mx.bfloat16 else 'f16'}_D{Dh}_{'c' if c else 'nc'}",
               (lambda dt=dt, Dh=Dh, c=c: dense_fwd_cell(dt, Dh, c)))
              for dt in (mx.float16, mx.bfloat16) for Dh in (64, 128) for c in (False, True)]
    run("dense_v6nax_fwd", lambda: sweep(cells))

    def dense_bwd_cell(dt, Dh, causal, env=None):
        # D=64, qL >= 2048: default V6NAX split backward (docs/reference/dispatch-map.md);
        # D=128 needs MFA_ENABLE_V6_BACKWARD=1.  Engagement by the trace terminal.
        from mlx_mfa import _dispatch_trace as dtr
        qb, kb, vb = rnd((1, 2, 2048, Dh), dt), rnd((1, 2, 2048, Dh), dt), rnd((1, 2, 2048, Dh), dt)
        g = rnd((1, 2, 2048, Dh), dt)
        f = lambda a, b, c: (mlx_mfa.flash_attention(a, b, c, causal=causal) * g).sum()
        with dtr.capture() as cap:
            grads = with_env(env, lambda: mx.grad(f, argnums=(0, 1, 2))(qb, kb, vb)) if env \
                else mx.grad(f, argnums=(0, 1, 2))(qb, kb, vb)
            mx.eval(grads)
        with mx.stream(mx.cpu):
            r = lambda a, b, c: (mx.fast.scaled_dot_product_attention(
                a, b, c, scale=Dh ** -0.5, mask="causal" if causal else None) * f32(g)).sum()
            refs = mx.grad(r, argnums=(0, 1, 2))(f32(qb), f32(kb), f32(vb))
            mx.eval(refs)
        oks = [check(a, b, 3e-2) for a, b in zip(grads, refs)]
        engaged = any(b == "v6_split_backward" for b, _ in cap)
        return all(o for o, _ in oks) and engaged, \
            " ".join(d for _, d in oks) + f" v6_split_backward={engaged}"

    run("dense_v6nax_bwd", lambda: sweep([
        ("f16_D64_nc", lambda: dense_bwd_cell(mx.float16, 64, False)),
        ("bf16_D64_nc", lambda: dense_bwd_cell(mx.bfloat16, 64, False)),
        ("f16_D64_c", lambda: dense_bwd_cell(mx.float16, 64, True)),
        ("f16_D128_nc_optin", lambda: dense_bwd_cell(mx.float16, 128, False, {"MFA_ENABLE_V6_BACKWARD": "1"})),
    ]))

    if os.environ.get("MFA_MATRIX_DUMP"):
        np.savez(os.environ["MFA_MATRIX_DUMP"], **dump)
    return {"mlx": mx.__version__, "mlx_mfa": mlx_mfa.__version__,
            "build_mlx": _ext._mlx_build_version(), "has_nax": bool(mlx_mfa.has_nax()),
            "mfa_env": leaked_env, "cells_dumped": len(dump), "results": res}


# ─────────────────────────────────────────────────────────────────────── driver
def _sh(cmd, env=None, cwd=None):
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)


def _clean_env(**extra) -> dict:
    """Caller env minus PYTHONPATH and every MFA_* knob (an inherited opt-out would turn a
    probe into a vacuous pass), plus ``extra``."""
    env = {k: v for k, v in os.environ.items()
           if k != "PYTHONPATH" and not k.startswith("MFA_")}
    env.update(extra)
    return env


# Source members of the sdist that must equal the git HEAD tree (what the receipt binds to).
_BOUND_PREFIXES = ("csrc/", "mlx_mfa/")
_BOUND_FILES = ("CMakeLists.txt", "pyproject.toml")


def _sdist_vs_head(sdist: str) -> list[str]:
    """Mismatches between the sdist's build inputs and ``git HEAD`` ([] = bound)."""
    import tarfile
    out = []
    with tarfile.open(sdist) as tf:
        members = {}
        for m in tf.getmembers():
            if not m.isfile():
                continue
            rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
            if rel.startswith(_BOUND_PREFIXES) or rel in _BOUND_FILES:
                members[rel] = tf.extractfile(m).read()
    for rel, data in sorted(members.items()):
        head = subprocess.run(["git", "-C", _REPO, "show", f"HEAD:{rel}"], capture_output=True)
        if head.returncode != 0:
            out.append(f"{rel}: not tracked at HEAD")
        elif head.stdout != data:
            out.append(f"{rel}: differs from HEAD")
    tracked = _sh(["git", "-C", _REPO, "ls-tree", "-r", "--name-only", "HEAD", "--",
                   "csrc", "mlx_mfa", *_BOUND_FILES]).stdout.split()
    text = open(os.path.join(_REPO, "pyproject.toml"), encoding="utf-8").read()
    section = text[text.index("[tool.scikit-build.sdist]"):]
    excluded = set(re.findall(r'"([^"]+)"', re.search(r"^exclude\s*=\s*\[(.*?)\]",
                                                        section, re.M | re.S).group(1)))
    out += [f"{rel}: tracked at HEAD, missing from sdist" for rel in tracked
            if rel not in members and rel not in excluded and "__pycache__" not in rel]
    return out


def _run_version(ver: str, sdist: str, work: str, base_python: str,
                 dump_dir: str | None = None) -> dict:
    venv = os.path.join(work, f"venv-{ver}")
    t0 = time.time()
    r = _sh([base_python, "-m", "venv", "--clear", venv])
    if r.returncode:
        return {"error": f"venv: {r.stderr[-400:]}"}
    py = os.path.join(venv, "bin", "python")
    cons = os.path.join(work, f"constraints-{ver}.txt")
    with open(cons, "w") as f:
        f.write(f"mlx=={ver}\n")
    # PIP_CONSTRAINT stops applying to isolated build envs at pip 26.2; PIP_BUILD_CONSTRAINT
    # is the forward form (the build_mlx == runtime check still catches any mismatch).
    env = _clean_env(PIP_CONSTRAINT=cons, PIP_BUILD_CONSTRAINT=cons,
                     PIP_DISABLE_PIP_VERSION_CHECK="1")
    r = _sh([py, "-m", "pip", "install", "-q", "--no-cache-dir", sdist], env=env)
    if r.returncode:
        return {"error": f"install (isolated, mlx=={ver}): {r.stderr[-800:]}"}
    penv = _clean_env()
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
        and r.get("mlx_mfa") == pkg_version and not r.get("mfa_env")
        and set(r["results"]) >= set(PROBES) and all(x["ok"] for x in r["results"].values())
        for v, r in matrix.items())
    sdist_mismatch = _sdist_vs_head(sdist)
    git_sha = _sh(["git", "-C", _REPO, "rev-parse", "HEAD"]).stdout.strip() or "UNKNOWN"
    git_dirty = bool(_sh(["git", "-C", _REPO, "status", "--porcelain", "--",
                          "csrc", "mlx_mfa", *_BOUND_FILES]).stdout.strip())
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
        "known_failures": KNOWN_FAILURES,
        "git_dirty": git_dirty,
        "sdist_matches_head": not sdist_mismatch,
        "sdist_mismatch": sdist_mismatch[:20],
        "matrix": matrix,
        "matrix_sha256": hashlib.sha256(json.dumps(matrix, sort_keys=True).encode()).hexdigest(),
        "all_pass": bool(all_pass and set(versions) >= set(abi_table_versions())),
    }
    if args.receipt is None and (git_dirty or sdist_mismatch):
        # A release receipt binds to git_sha: refuse to write one for uncommitted source or
        # for an sdist whose build inputs are not the HEAD tree.
        print("[matrix] refusing to write a release receipt: "
              + ("uncommitted csrc/mlx_mfa/build-file changes; " if git_dirty else "")
              + (f"sdist != HEAD ({sdist_mismatch[:3]}); " if sdist_mismatch else "")
              + "commit + rebuild the sdist, or pass --receipt <scratch path>.", file=sys.stderr)
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
