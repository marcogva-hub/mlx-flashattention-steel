"""R8 / STA-05 / RES-01 (+ hypotheses STA-06 / BLD-12) and the async_v2 retirement
(code review 2026-09).

AOT cache (~/.mlx_mfa/metallib): the loader matched files named only by
(D, BK, is_m3_plus, dtype, causal) — no block_q / n_warps / steel_msl_mode, no
mlx-mfa version, no source hash — then cached them under the FULL KernelKey.  With
MFA_V2_BQ64=1 a BQ32 kernel was served for BQ64 host geometry (err 0.95), and any
metallib built before a kernel fix kept loading after an upgrade.  Now the filename
carries the full geometry, the mlx-mfa version and an FNV-1a-64 hash of the generated
MSL source: a metallib built from ANY other source/version/geometry is never matched
(content-addressed invalidation, no user file is deleted).  The canonical name is
emitted by the C++ loader itself in the MFA_DEBUG_SHADERS header (`aot=...`), which
compile_metallib reads — one source of truth.

async_v2.metallib (loaded BEFORE the JIT on macOS 14/15, source frozen 2026-03-11,
BLD-04): retired — file, loader, MFA_DISABLE_ASYNC knob, build workflow.

Every AOT test redirects NSHomeDirectory() with CFFIXED_USER_HOME to a tmp dir: the
user's real ~/.mlx_mfa is never touched.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

CHILD = r'''
import json, mlx.core as mx, mlx_mfa
from mlx_mfa import attention as A
A._auto_warmup_done = True
mx.random.seed(0)
B, H, N, D = 1, 8, 256, 64
q, k, v = (mx.random.normal((B, H, N, D)).astype(mx.float16) for _ in range(3))
o = mlx_mfa.flash_attention(q, k, v, causal=False, backend="mfa")
qf, kf, vf = (t.astype(mx.float32) for t in (q, k, v))
ref = mx.softmax((qf @ kf.swapaxes(-1, -2)) * (D ** -0.5), axis=-1) @ vf
print(json.dumps({"err": mx.max(mx.abs(o.astype(mx.float32) - ref)).item()}))
'''
_HDR = re.compile(r"=== MFA Shader \[steel_fwd_v2[^\]]*?aot=(\S+?)\] ===\n(.*?)\n=== END MFA Shader ===",
                  re.S)


def _xcrun_ok():
    if shutil.which("xcrun") is None:
        return False
    return subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"],
                          capture_output=True).returncode == 0


needs_ext = pytest.mark.skipif(
    subprocess.run([sys.executable, "-c", "import mlx_mfa._ext"], capture_output=True).returncode != 0,
    reason="MFA extension required")
needs_xcrun = pytest.mark.skipif(not _xcrun_ok(), reason="xcrun metal required to build a metallib")


def _run(home: Path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MFA_") and k != "CFFIXED_USER_HOME"}
    env.update({"MFA_FORCE_SPLITK": "0", "MFA_DEBUG_SHADERS": "1", "CFFIXED_USER_HOME": str(home)})
    env.update(extra)
    r = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=110)
    assert r.returncode == 0, r.stderr[-800:]
    m = _HDR.search(r.stderr)
    return {"err": json.loads(r.stdout.strip().splitlines()[-1])["err"],
            "jit": m is not None,
            "aot_name": m.group(1) if m else None,
            "source": m.group(2) if m else None}


def _metallib_dir(home: Path) -> Path:
    d = home / ".mlx_mfa" / "metallib"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    """JIT run in an empty fake home -> the canonical AOT name + source for this key."""
    home = tmp_path_factory.mktemp("home_capture")
    res = _run(home)
    assert res["jit"] and res["aot_name"], "loader must emit aot=<name> for an AOT-eligible key"
    return res


def _build(source: str, dest: Path):
    from mlx_mfa.compile_metallib import _compile_source_to_metallib
    assert _compile_source_to_metallib(source, str(dest)), "xcrun metal failed"


@needs_ext
def test_aot_name_carries_geometry_version_and_source_hash(captured):
    import mlx_mfa
    name = captured["aot_name"]
    for field in ("_BQ", "_BK", "_W", "_msl", f"_v{mlx_mfa.__version__}_", "_h"):
        assert field in name, (field, name)


@needs_ext
@needs_xcrun
def test_matching_aot_metallib_is_loaded(captured, tmp_path):
    """Engagement: a metallib built from the exact current source IS used (no JIT)."""
    _build(captured["source"], _metallib_dir(tmp_path) / captured["aot_name"])
    res = _run(tmp_path)
    assert not res["jit"], "AOT file with the canonical name must be loaded"
    assert res["err"] < 1e-2


@needs_ext
@needs_xcrun
def test_geometry_mismatch_is_not_loaded(captured, tmp_path):
    """STA-05: the BQ32 build must NOT serve a BQ64 KernelKey (MFA_V2_BQ64=1)."""
    _build(captured["source"], _metallib_dir(tmp_path) / captured["aot_name"])
    res = _run(tmp_path, MFA_V2_BQ64="1")
    assert res["jit"] and res["aot_name"] != captured["aot_name"]
    assert res["err"] < 1e-2


@needs_ext
@needs_xcrun
@pytest.mark.parametrize("stale", ["legacy_name", "other_version", "other_source"])
def test_stale_metallibs_are_ignored(captured, tmp_path, stale):
    """STA-06/BLD-12 class: a metallib from an older build/source never loads."""
    d = _metallib_dir(tmp_path)
    if stale == "legacy_name":        # pre-2.62.2 scheme (what old installs left in ~)
        name = "v2_D64_BK64_M1_dtype0_causal0.metallib"
    elif stale == "other_version":
        name = re.sub(r"_v[^_]+_h", "_v0.0.0_h", captured["aot_name"])
    else:
        name = re.sub(r"_h[0-9a-f]+\.metallib$", "_h0123456789abcdef.metallib", captured["aot_name"])
    _build(captured["source"], d / name)
    res = _run(tmp_path)
    assert res["jit"], f"{stale}: {name} must be ignored"
    assert res["err"] < 1e-2


def test_async_v2_metallib_retired():
    """BLD-04 (decision Marco): no shipped precompiled async kernel, loader or knob."""
    assert not (ROOT / "mlx_mfa" / "precompiled").exists()
    sc = (ROOT / "csrc" / "shader_cache.mm").read_text()
    assert "try_async_pipeline" not in sc and '@"async_v2.metallib"' not in sc
    assert not (ROOT / ".github" / "workflows" / "build-metallib.yml").exists()
    from mlx_mfa import _knobs
    assert "MFA_DISABLE_ASYNC" not in _knobs.KNOWN_KNOBS
    assert "MFA_DISABLE_ASYNC" not in (ROOT / "ENV_VARS.md").read_text()


@needs_ext
def test_built_extension_has_no_async_loader():
    import mlx_mfa._ext as e
    blob = Path(e.__file__).read_bytes()
    assert b"async_v2.metallib" not in blob and b"mlx_mfa_v2_async_attention" not in blob
