"""compile_metallib — AOT compilation of common STEEL V2 Metal kernels.

Usage::

    from mlx_mfa import compile_metallib
    compile_metallib()

Or from the command line::

    python -m mlx_mfa.compile_metallib

After compilation, common STEEL V2 forward kernels are cached as precompiled
AIR metallibs in ``~/.mlx_mfa/metallib/``.  The C++ ShaderCache loads them on
subsequent runs to reduce JIT cold-start overhead for production paths.

Scope note:
  - Covers STEEL V1 (M3+ D≤128 causal) and V2 / V2 D-split.
  - Sage/paged/varlen AOT deferred: these kernels have per-request varying
    configs (block_size, max_blocks, quantization) that make static AOT
    impractical. They compile on first use via JIT with acceptable latency.

Compiled configs cover (standard V2):
  - D=64  BK=64  f16/bf16  causal/noncausal  (all gens)
  - D=128 BK=32  f16/bf16  causal/noncausal  (M1/M2)
  - D=128 BK=64  f16/bf16  causal/noncausal  (M3+, if is_m3_plus)

Compiled configs cover (V2 D-split):
  - D=256 BK=32/64 f16/bf16  causal/noncausal  (M1/M2 or M3+)
  - D=512 BK=32/64 f16/bf16  causal/noncausal  (M1/M2 or M3+)

Filename scheme (R8, review 2026-09 — content-addressed; the C++ ShaderCache is the
single source of truth and prints it as ``aot=<name>`` in the MFA_DEBUG_SHADERS header)::

    v2[_dsplit]_D{D}_BQ{BQ}_BK{BK}_BD{BD}_W{warps}_M{m3}_dtype{dt}_causal{0|1}
        _msl{steel_msl_mode}_v{mlx-mfa version}_h{FNV-1a-64 of the MSL source}.metallib

A metallib built by another mlx-mfa version, from another kernel source, or for
another geometry (e.g. MFA_V2_BQ64) is never matched, so it can never be served
for the wrong kernel.  Files from the pre-2.62.2 scheme are simply ignored.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Optional


# ---------------------------------------------------------------------------
# Default metallib cache directory
# ---------------------------------------------------------------------------

_DEFAULT_DIR = os.path.expanduser("~/.mlx_mfa/metallib")


# ---------------------------------------------------------------------------
# compile_metallib
# ---------------------------------------------------------------------------

def compile_metallib(
    output_dir: Optional[str] = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Pre-compile common STEEL V2 kernel configs to precompiled AIR metallibs.

    Parameters
    ----------
    output_dir : str, optional
        Where to save the metallibs.  Defaults to ``~/.mlx_mfa/metallib``.
    force : bool
        If True, recompile even if the metallib already exists.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Mapping filename -> True (compiled/already-exists) or False (failed).
    """
    if output_dir is None:
        output_dir = _DEFAULT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Check MFA extension availability
    try:
        import mlx_mfa._ext  # noqa: F401
        ext_ok = True
    except ImportError:
        ext_ok = False

    if not ext_ok:
        if verbose:
            print("[compile_metallib] MFA C++ extension not available; skipping.")
        return {}

    # Check xcrun metal availability
    if shutil.which("xcrun") is None or not _xcrun_metal_available():
        if verbose:
            print("[compile_metallib] xcrun metal not found; skipping.")
        return {}

    # Determine device config
    try:
        from mlx_mfa import get_device_info
        info = get_device_info()
        is_m3_plus = info.get("is_m3_plus", False)
    except Exception:
        is_m3_plus = False


    # V2 block sizes (must match select_steel_v2_block_config in C++)
    bk_d64 = 64                           # D=64: BK=64 all gens
    bk_d128 = 64 if is_m3_plus else 32   # D=128: BK=64 M3+, BK=32 M1/M2

    # Configs: (D, BK, dtype_code, causal, mlx_dtype_name)
    configs = [
        (64,   bk_d64,  0, True,  "float16"),
        (64,   bk_d64,  0, False, "float16"),
        (64,   bk_d64,  1, True,  "bfloat16"),
        (64,   bk_d64,  1, False, "bfloat16"),
        (128,  bk_d128, 0, True,  "float16"),
        (128,  bk_d128, 0, False, "float16"),
        (128,  bk_d128, 1, True,  "bfloat16"),
        (128,  bk_d128, 1, False, "bfloat16"),
    ]

    results: dict = {}

    for D, BK, dtype_code, causal, dtype_name in configs:
        label = f"v2_D{D}_BK{BK}_dtype{dtype_code}_causal{int(causal)}"
        captured = _capture_shader_source(D, BK, dtype_name, causal, is_m3_plus)
        _compile_one(captured, label, output_dir, force, verbose, results)

    # ── V2 D-split configs (D=256/512) ─────────────────────────────────────
    # BK from select_steel_v2_block_config(128, is_m3_plus) — same as D=128.
    bk_dsplit = 64 if is_m3_plus else 32

    dsplit_configs = [
        (256, bk_dsplit, 0, True,  "float16"),
        (256, bk_dsplit, 0, False, "float16"),
        (256, bk_dsplit, 1, True,  "bfloat16"),
        (256, bk_dsplit, 1, False, "bfloat16"),
        (512, bk_dsplit, 0, True,  "float16"),
        (512, bk_dsplit, 0, False, "float16"),
        (512, bk_dsplit, 1, True,  "bfloat16"),
        (512, bk_dsplit, 1, False, "bfloat16"),
    ]

    for D, BK, dtype_code, causal, dtype_name in dsplit_configs:
        label = f"v2_dsplit_D{D}_BK{BK}_dtype{dtype_code}_causal{int(causal)}"
        captured = _capture_dsplit_shader_source(D, BK, dtype_name, causal, is_m3_plus)
        _compile_one(captured, label, output_dir, force, verbose, results)

    if verbose:
        n_ok = sum(1 for v in results.values() if v)
        print(f"[compile_metallib] {n_ok}/{len(results)} configs compiled -> {output_dir}")

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEADER_AOT = re.compile(r"aot=(\S+?)\]")


def _compile_one(captured, label, output_dir, force, verbose, results) -> None:
    """Compile one captured (source, aot_name) pair into ``output_dir/aot_name``."""
    # Failures have no canonical (source-hashed) name: key them as an explicitly
    # UNRESOLVED filename (ok=False, no file) to keep the documented
    # {filename: bool} contract without inventing a loadable name.
    if captured is None:
        # e.g. causal D<=128 on M3+ runs STEEL V1 (m3_prefers_v1) — no V2 kernel.
        if verbose:
            print(f"[compile_metallib] {label}: FAILED (no V2 kernel captured)")
        results[f"unresolved_{label}.metallib"] = False
        return
    source, filename = captured
    if not filename or filename == "-":
        # The C++ loader declared this key not AOT-eligible: nothing to precompile.
        if verbose:
            print(f"[compile_metallib] {label}: FAILED (key not AOT-eligible)")
        results[f"unresolved_{label}.metallib"] = False
        return
    metallib_path = os.path.join(output_dir, filename)
    if os.path.exists(metallib_path) and not force:
        if verbose:
            print(f"[compile_metallib] Already compiled: {filename}")
        results[filename] = True
        return
    if verbose:
        print(f"[compile_metallib] Compiling: {filename} ...", end=" ", flush=True)
    ok = _compile_source_to_metallib(source, metallib_path)
    results[filename] = ok
    if verbose:
        print("ok" if ok else "FAILED (xcrun)")


def _capture_env() -> dict:
    """Env for the capture subprocess: debug-shader dump ON, and NSHomeDirectory()
    redirected to an EMPTY dir so an existing AOT file can never short-circuit the
    JIT (the header — source + canonical aot name — is only printed on JIT)."""
    env = dict(os.environ)
    env["MFA_DEBUG_SHADERS"] = "1"
    env.pop("MFA_DISABLE_V2", None)
    env["CFFIXED_USER_HOME"] = tempfile.mkdtemp(prefix="mlx_mfa_aot_capture_")
    return env


def _parse_capture(stderr: str, label_regex: str):
    """-> (source, aot_name) from the MFA_DEBUG_SHADERS dump, or None."""
    m = re.search(
        rf"=== MFA Shader \[({label_regex}[^\]]*)\] ===\n(.*?)=== END MFA Shader ===",
        stderr, re.DOTALL)
    if not m:
        return None
    a = _HEADER_AOT.search("[" + m.group(1) + "]")
    return m.group(2).strip(), (a.group(1) if a else None)

def _xcrun_metal_available() -> bool:
    """Return True if xcrun metal can be invoked without error."""
    try:
        r = subprocess.run(
            ["xcrun", "metal", "--version"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _capture_shader_source(
    D: int,
    BK: int,
    dtype_name: str,
    causal: bool,
    is_m3_plus: bool,
) -> Optional[tuple]:
    """Launch a subprocess that calls flash_attention with MFA_DEBUG_SHADERS=1
    and extract (V2 kernel source, canonical AOT filename) from stderr."""
    N = 4096
    scale = 1.0 / (D ** 0.5)

    # Build the subprocess script as list of lines to avoid any hook issues
    lines = [
        "import mlx.core as mx, mlx_mfa",
        f"q = mx.zeros([1, 1, {N}, {D}], dtype=mx.{dtype_name})",
        (
            f"r = mlx_mfa.flash_attention(q, q, q, scale={scale:.8f},"
            f" causal={causal}, backend='mfa')"
        ),
        "mx.eval(r)",  # mx.synchronize() does not trigger lazy evaluation
    ]
    script = "\n".join(lines)

    env = _capture_env()
    if D == 128:
        env["MFA_V2_FORCE_BK"] = str(BK)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env, capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return None

    # "=== MFA Shader [steel_fwd_v2 ... aot=<name>] ===\n<source>\n=== END MFA Shader ==="
    return _parse_capture(result.stderr, "steel_fwd_v2")


def _capture_dsplit_shader_source(
    D: int,
    BK: int,
    dtype_name: str,
    causal: bool,
    is_m3_plus: bool,
) -> Optional[tuple]:
    """Like _capture_shader_source but for the V2 D-split kernel (D=256/512)."""
    N = 256  # short sequence — just triggers one JIT compile
    scale = 1.0 / (D ** 0.5)
    label = f"steel_v2_dsplit{D}"

    lines = [
        "import mlx.core as mx, mlx_mfa",
        f"q = mx.zeros([1, 1, {N}, {D}], dtype=mx.{dtype_name})",
        (
            f"r = mlx_mfa.flash_attention(q, q, q, scale={scale:.8f},"
            f" causal={causal}, backend='mfa')"
        ),
        "mx.eval(r)",  # mx.synchronize() does not trigger lazy evaluation
    ]
    script = "\n".join(lines)

    env = _capture_env()

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env, capture_output=True, text=True, timeout=120,
        )
    except Exception:
        return None

    return _parse_capture(result.stderr, re.escape(label))


def _compile_source_to_metallib(source: str, output_path: str) -> bool:
    """Compile a Metal source string to a .metallib file via xcrun metal/metallib."""
    with tempfile.TemporaryDirectory() as tmp:
        src_file = os.path.join(tmp, "kernel.metal")
        air_file = os.path.join(tmp, "kernel.air")

        with open(src_file, "w") as f:
            f.write(source)

        try:
            subprocess.run(
                [
                    "xcrun", "metal",
                    "-target", "air64-apple-macos15.0",
                    "-c", src_file, "-o", air_file,
                ],
                check=True, capture_output=True, timeout=120,
            )
            subprocess.run(
                ["xcrun", "metallib", air_file, "-o", output_path],
                check=True, capture_output=True, timeout=30,
            )
            return True
        except subprocess.CalledProcessError as e:
            # CC-26 (audit): a compile FAILURE was swallowed to a bare False with
            # no diagnostic.  Surface the compiler stderr (loud) so it is not
            # masked; the caller still tallies n_ok/total and the CLI exits
            # non-zero (below) when any config fails.
            import sys
            err = (e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or ""))
            print(f"[compile_metallib] COMPILE FAILED ({output_path}):\n{err.strip()}",
                  file=sys.stderr)
            return False
        except Exception as e:
            import sys
            print(f"[compile_metallib] COMPILE ERROR ({output_path}): "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            return False


# ---------------------------------------------------------------------------
# __main__ entry point:  python -m mlx_mfa.compile_metallib
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-compile common STEEL V2 Metal kernels to AIR metallibs."
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help=f"Output directory (default: {_DEFAULT_DIR})",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Recompile even if metallib already exists.",
    )
    args = parser.parse_args()
    results = compile_metallib(output_dir=args.output_dir, force=args.force)
    # CC-26 (audit): exit non-zero if any config failed to compile, so a build
    # step invoking this can't silently treat a compile failure as success.
    import sys
    sys.exit(0 if all(results.values()) else 1)
