#!/usr/bin/env python3
"""Robust benchmark pool runner — survives long runs + partial kills/crashes.

Why this exists (Sprint III-11 infra fix): the methodology dispatcher emits its
JSON only at the very end, so any kill/crash mid-run loses the whole batch; and
harness-tracked background jobs get killed at a ~5-8 min duration limit. This
runner fixes both:

  1. INCREMENTAL PERSISTENCE — each measurement spec's result is appended to a
     JSONL file the instant it completes (and the spec is marked done), so a
     kill/crash keeps every prior result. Re-running SKIPS already-done specs
     (idempotent resume).
  2. DETACHMENT — launch via `nohup python robust_pool_runner.py <catalog> &`
     (RULE 12a) so it is NOT subject to the harness background-job kill limit.
     Poll the .jsonl for progress.
  3. PER-SPEC ISOLATION — every (spec, role, session) runs in a fresh subprocess
     (defeats pipeline-cache contamination + lets one bad spec fail without
     killing the run; the parent try/excepts each spec).

Protocol per CLAUDE_V6_NAX §4: strict 4s-cooldown (8 iters) for >=1.5ms shapes,
canonical 10-warmup + 100-continuous for sub-1.5ms. 3 sessions, median, verdict
from cross-session ratio range (CONFIDENT <10%, BOUNDARY <20%, else HIGH_VAR).

Usage:
  child:  robust_pool_runner.py child <spec_json> <role> <protocol>   # prints median ms
  parent: robust_pool_runner.py <catalog.json> <out.jsonl>            # orchestrates
"""
import os, sys, json, time, statistics, subprocess

STRICT_COOLDOWN_S = 4.0
STRICT_WARMUP, STRICT_ITERS = 4, 8
CANON_WARMUP, CANON_ITERS = 10, 100
EST_STRICT_MS = 1.5


# ---------------------------------------------------------------------------
# CHILD: one (spec, role, protocol) measurement in an isolated process.
# ---------------------------------------------------------------------------
def _child(spec, role, protocol):
    # role-dependent env MUST be set before importing mlx_mfa.
    if spec["path"] == "conv" and role == "baseline" and spec.get("baseline") == "legacy":
        os.environ["MFA_DISABLE_CONV3D_MPP"] = "1"
    import mlx.core as mx

    def make_attn():
        import mlx.core as _mx
        dt = getattr(_mx, spec.get("dtype", "float16"))
        B, H, qL, kL, D = spec["B"], spec["H"], spec["qL"], spec["kL"], spec["D"]
        Hk = spec.get("Hk", H)
        _mx.random.seed(0)
        q = _mx.random.normal((B, H, qL, D)).astype(dt)
        k = _mx.random.normal((B, Hk, kL, D)).astype(dt)
        v = _mx.random.normal((B, Hk, kL, D)).astype(dt)
        return q, k, v

    def make_conv():
        dt = getattr(mx, spec.get("dtype", "float16"))
        T = spec["T"]
        mx.random.seed(0)
        x = mx.random.normal((1, T, 64, 64, 128)).astype(dt)
        w = mx.random.normal((128, 3, 3, 3, 128)).astype(dt)
        return x, w

    # Resolve the callable for this (path, role).
    if spec["path"] == "attn":
        import mlx_mfa
        scale = 1.0 / (spec["D"] ** 0.5)
        causal = spec.get("causal", False)
        mask = "causal" if causal else None
        is_bwd = spec.get("is_backward", False)
        if role == "target":
            if is_bwd:
                def f(q, k, v): return mlx_mfa.flash_attention(q, k, v, scale=scale, causal=causal).sum()
            else:
                def f(q, k, v): return mlx_mfa.flash_attention(q, k, v, scale=scale, causal=causal)
        else:  # baseline = SDPA / SDPA-vjp
            if is_bwd:
                def f(q, k, v): return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask).sum()
            else:
                def f(q, k, v): return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        mk = make_attn
        wrap_grad = is_bwd
    elif spec["path"] == "conv":
        from mlx_mfa.conv_nax import conv3d_nax_forward
        base = spec.get("baseline", "legacy")
        if role == "target":
            def f(x, w): return conv3d_nax_forward(x, w, padding=(1, 1, 1))
        elif base == "conv_general":
            def f(x, w): return mx.conv_general(x, w, stride=1, padding=1)
        else:  # legacy = conv3d_nax with MPP disabled (env set above)
            def f(x, w): return conv3d_nax_forward(x, w, padding=(1, 1, 1))
        mk = make_conv
        wrap_grad = False
    else:
        raise ValueError(f"unknown path {spec['path']!r}")

    if spec["path"] == "attn" and wrap_grad:
        base_f = f
        def call(*a): return mx.grad(base_f, argnums=(0, 1, 2))(*a)
    else:
        call = f

    args = mk()
    for _ in range(STRICT_WARMUP if protocol == "strict" else CANON_WARMUP):
        mx.eval(call(*args))

    if protocol == "strict":
        ts = []
        for _ in range(STRICT_ITERS):
            args = mk(); mx.eval(*args)
            t0 = time.perf_counter(); mx.eval(call(*args)); ts.append((time.perf_counter() - t0) * 1000)
            time.sleep(STRICT_COOLDOWN_S)
        print(statistics.median(ts))
    else:  # canonical continuous
        mx.eval(call(*args))
        t0 = time.perf_counter()
        for _ in range(CANON_ITERS):
            mx.eval(call(*args))
        print(((time.perf_counter() - t0) * 1000) / CANON_ITERS)


# ---------------------------------------------------------------------------
# PARENT: orchestrate the catalog with incremental persistence + resume.
# ---------------------------------------------------------------------------
def _one_session(spec, role, protocol):
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "child", json.dumps(spec), role, protocol],
        capture_output=True, text=True,
        env={**os.environ, "MLX_MFA_REPO_ROOT": os.environ.get("MLX_MFA_REPO_ROOT", os.getcwd())},
    )
    line = out.stdout.strip().splitlines()
    if not line:
        raise RuntimeError(out.stderr.strip()[-300:] or "no output")
    return float(line[-1])


def _parent(catalog_path, out_path):
    specs = json.load(open(catalog_path))
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            try: done.add(json.loads(ln)["id"])
            except Exception: pass
    for spec in specs:
        sid = spec["id"]
        if sid in done:
            print(f"SKIP {sid} (done)", flush=True); continue
        rec = {"id": sid, "spec": spec, "ts": time.time()}
        try:
            # protocol: probe one target session, pick strict if >=1.5ms.
            probe = _one_session(spec, "target", "strict" if spec.get("force_strict") else "canonical")
            protocol = "strict" if probe >= EST_STRICT_MS or spec.get("force_strict") else "canonical"
            tgt, base = [], []
            for _ in range(3):
                tgt.append(_one_session(spec, "target", protocol))
                base.append(_one_session(spec, "baseline", protocol))
            ratios = [b / t for b, t in zip(base, tgt)]  # speedup of target vs baseline
            med = statistics.median(ratios)
            rng = (max(ratios) - min(ratios)) / med * 100 if med else float("inf")
            verdict = "CONFIDENT" if rng < 10 else ("BOUNDARY" if rng < 20 else "HIGH_VARIANCE")
            rec.update(protocol=protocol, median_target_ms=tgt, median_baseline_ms=base,
                       ratios=ratios, median_ratio=med, range_pct=rng, verdict=verdict)
        except Exception as e:
            rec.update(error=str(e)[-400:])
        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
        r = rec.get("median_ratio")
        print(f"DONE {sid}: " + (f"{r:.3f}x {rec.get('verdict')} ({rec.get('range_pct',0):.1f}%)"
                                 if r else f"ERROR {rec.get('error','')[:120]}"), flush=True)
    print("POOL COMPLETE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "child":
        _child(json.loads(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        _parent(sys.argv[1], sys.argv[2])
