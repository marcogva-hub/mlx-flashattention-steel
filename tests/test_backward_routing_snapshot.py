"""Backward routing-equivalence golden — guards the V6-split-vs-SDPA-vjp default.

The forward routing net (`test_routing_equivalence_snapshot.py`) covers forward terminals
only. The D=64 split-V6 backward is the PUBLIC DEFAULT (verified 2.16-3.05× vs SDPA-vjp,
2026-06-19) — this snapshot freezes which backward kernel each cell routes to, so a future
change (e.g. flipping the D=64 default, or D=128 going default) is caught.

Captured via `_dispatch_trace` at the backward vjp terminals (added this pass): the last
backward record per `mx.grad(flash_attention)` call — `v6_split_backward` / `sdpa_vjp` /
`steel_backward`, or `<none>` (SDPA forward took the non-custom path → MLX SDPA autograd).

Device-stamped (M5). M1/M4 tier needs Marco's golden (as with the forward net). Regenerate
(intentional change only): `MFA_REGEN_BWD_ROUTING_GOLDEN=1`.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mlx.core as mx
import pytest

import mlx_mfa
from mlx_mfa import _dispatch_trace as _dt

_GOLDEN = Path(__file__).parent / "backward_routing_golden.json"
_BWD = ("v6_split_backward", "sdpa_vjp", "steel_backward")


def _device_key() -> str:
    try:
        info = mlx_mfa.get_device_info()
        gen = info.get("gpu_family_gen", "?")
        m5 = "1" if str(info.get("is_m5_plus", False)) in ("True", "1", "true") else "0"
        return f"gen{gen}_m5{m5}"
    except Exception:
        return "unknown"


# (id, D, causal, qL, env) — env is the toggle that selects the path.
_CELLS = []
def _add(cid, D, causal, qL, env=None):
    _CELLS.append((cid, D, causal, qL, env or {}))

for causal in (False, True):
    c = "caus" if causal else "nc"
    _add(f"d64_{c}_qL4096_default", 64, causal, 4096)          # default-on -> v6_split
    _add(f"d64_{c}_qL8192_default", 64, causal, 8192)          # default-on -> v6_split
_add("d64_caus_qL1024_default", 64, True, 1024)               # <2048 -> sdpa
_add("d64_caus_qL4096_disabled", 64, True, 4096, {"MFA_DISABLE_V6_BACKWARD": "1"})  # -> sdpa
_add("d128_caus_qL4096_default", 128, True, 4096)             # opt-in off -> sdpa
_add("d128_caus_qL4096_enabled", 128, True, 4096, {"MFA_ENABLE_V6_BACKWARD": "1"})  # -> v6_split (slower)
_add("d128_nc_qL4096_enabled", 128, False, 4096, {"MFA_ENABLE_V6_BACKWARD": "1"})


def _bwd_terminal(D, causal, qL, env) -> str:
    for k in ("MFA_DISABLE_V6_BACKWARD", "MFA_ENABLE_V6_BACKWARD"):
        os.environ.pop(k, None)
    for k, val in env.items():
        os.environ[k] = val
    try:
        mx.random.seed(0)
        f = lambda: (mx.random.normal((1, 4, qL, D)) * 1.0).astype(mx.float16)
        q, k, v = f(), f(), f()
        mx.eval(q, k, v)
        sc = 1.0 / math.sqrt(D)
        with _dt.capture() as tr:
            g = mx.grad(lambda q, k, v: mlx_mfa.flash_attention(q, k, v, scale=sc, causal=causal).sum(),
                        argnums=(0, 1, 2))(q, k, v)
            mx.eval(g)
        bwd = [t[0] for t in tr if t[0] in _BWD]
        return bwd[-1] if bwd else "<none>"
    finally:
        for k in ("MFA_DISABLE_V6_BACKWARD", "MFA_ENABLE_V6_BACKWARD"):
            os.environ.pop(k, None)


if os.environ.get("MFA_REGEN_BWD_ROUTING_GOLDEN") == "1":
    g = {"_device": _device_key(),
         "backward_routing": {cid: _bwd_terminal(D, c, qL, env)
                              for cid, D, c, qL, env in _CELLS}}
    _GOLDEN.write_text(json.dumps(g, indent=2, sort_keys=True) + "\n")
    print(f"[bwd-routing-golden] regenerated ({len(g['backward_routing'])} cells) on {g['_device']}")


def _load():
    if not _GOLDEN.exists():
        pytest.skip("golden missing — MFA_REGEN_BWD_ROUTING_GOLDEN=1 to create")
    return json.loads(_GOLDEN.read_text())


@pytest.mark.parametrize("cid,D,causal,qL,env", _CELLS, ids=[c[0] for c in _CELLS])
def test_backward_routing_matches_golden(cid, D, causal, qL, env):
    g = _load()
    if g.get("_device") != _device_key():
        pytest.skip(f"golden device {g.get('_device')} != {_device_key()} — backward routing "
                    f"is hardware-specific; regenerate on this device")
    expected = g["backward_routing"].get(cid)
    if expected is None:
        pytest.fail(f"cell {cid!r} not in golden — regenerate")
    actual = _bwd_terminal(D, causal, qL, env)
    assert actual == expected, (
        f"BACKWARD ROUTING REGRESSION at {cid}: golden {expected!r}, now {actual!r}. "
        f"Intentional (V6-backward wiring change)? regenerate with MFA_REGEN_BWD_ROUTING_GOLDEN=1.")
