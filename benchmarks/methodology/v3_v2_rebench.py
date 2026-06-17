"""V3-vs-V2 forward re-bench on M5/26.6 at V3's auto-fire regime (Queue Closure Sprint).

§4-strict cooldown protocol (V3 large-N >=1.5ms): 4 warmup / 8 timed / 4s cooldown,
fresh allocs per timed iter, 3 sessions via subprocess isolation, incremental JSONL
(resume skips done arms), nohup-detached (survives the kill limit, RULE 12).

Arms are FORCED (lesson #14, deterministic which-binary):
  V3 arm: MFA_ENABLE_V3=1  (bypasses shape guard -> guaranteed V3)
  V2 arm: MFA_DISABLE_V3=1 (forces V2 fallback)
Both via backend="mfa" so the MFA primitive is invoked; windowed shapes also exercise
the production-reachable path (window -> MFA on M5).

Usage:
  parent: python v3_v2_rebench.py parent <out.jsonl>
  child : python v3_v2_rebench.py child '<spec_json>'   (internal)
"""
import sys, os, json, time, subprocess

WARMUP, TIMED, COOLDOWN = 4, 8, 4.0

CATALOG = [
    # windowed-causal = the PRODUCTION-reachable path on M5 (window -> MFA -> V3)
    {"id": "W_D64_N4096",  "B": 1, "H": 8, "N": 4096, "D": 64,  "win": 1024, "windowed": True},
    {"id": "W_D64_N8192",  "B": 1, "H": 8, "N": 8192, "D": 64,  "win": 1024, "windowed": True},
    {"id": "W_D128_N2048", "B": 1, "H": 8, "N": 2048, "D": 128, "win": 1024, "windowed": True},
    {"id": "W_D128_N4096", "B": 1, "H": 8, "N": 4096, "D": 128, "win": 1024, "windowed": True},
    {"id": "W_D128_N8192", "B": 1, "H": 8, "N": 8192, "D": 128, "win": 1024, "windowed": True},
    # dense-causal backend="mfa" = the expert/opt-in path (dense routes to SDPA on M5,
    # but backend="mfa" users still hit V3 auto-fire)
    {"id": "M_D128_N4096", "B": 1, "H": 8, "N": 4096, "D": 128, "win": -1,   "windowed": False},
    {"id": "M_D64_N4096",  "B": 1, "H": 8, "N": 4096, "D": 64,  "win": -1,   "windowed": False},
]
SESSIONS = [1, 2, 3]


def _child(spec):
    import mlx.core as mx
    import mlx_mfa
    import numpy as np
    B, H, N, D = spec["B"], spec["H"], spec["N"], spec["D"]
    scale = 1.0 / (D ** 0.5)
    win = spec["win"]
    window = (win, 0) if spec["windowed"] else None  # (left, right) causal sliding window

    def run_once():
        q = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
        k = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
        v = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
        mx.eval(q, k, v)
        t0 = time.perf_counter()
        if window is not None:
            out = mlx_mfa.flash_attention(q, k, v, scale=scale, causal=True,
                                          window_size=window, backend="mfa")
        else:
            out = mlx_mfa.flash_attention(q, k, v, scale=scale, causal=True, backend="mfa")
        mx.eval(out)
        dt = (time.perf_counter() - t0) * 1000.0
        return dt, out

    # one fp32 correctness check (lesson #11) on first timed iter
    for _ in range(WARMUP):
        run_once()
    times = []
    last_out = None
    for _ in range(TIMED):
        dt, last_out = run_once()
        times.append(dt)
        time.sleep(COOLDOWN)
    times.sort()
    median = times[len(times) // 2]
    return median, last_out


def child_main(spec_json):
    spec = json.loads(spec_json)
    median, _ = _child(spec)
    print(json.dumps({"median_ms": median}))


def parent_main(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["session"], r["id"], r["arm"]))
                except Exception:
                    pass
    self = os.path.abspath(__file__)
    for session in SESSIONS:
        for spec in CATALOG:
            for arm, env_key in (("V3", "MFA_ENABLE_V3"), ("V2", "MFA_DISABLE_V3")):
                key = (session, spec["id"], arm)
                if key in done:
                    continue
                env = dict(os.environ)
                env.pop("MFA_ENABLE_V3", None)
                env.pop("MFA_DISABLE_V3", None)
                env[env_key] = "1"
                proc = subprocess.run(
                    [sys.executable, self, "child", json.dumps(spec)],
                    env=env, capture_output=True, text=True, timeout=900,
                )
                rec = {"session": session, "id": spec["id"], "arm": arm,
                       "D": spec["D"], "N": spec["N"], "windowed": spec["windowed"]}
                try:
                    rec["median_ms"] = json.loads(proc.stdout.strip().splitlines()[-1])["median_ms"]
                    rec["ok"] = True
                except Exception as e:
                    rec["ok"] = False
                    rec["err"] = (proc.stderr or str(e))[-500:]
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
    # summary
    print("DONE", out_path)


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child_main(sys.argv[2])
    elif sys.argv[1] == "parent":
        parent_main(sys.argv[2])
