"""compile_metallib — AOT compilation of common STEEL V2 Metal kernels.

Usage::

    from mlx_mfa import compile_metallib
    compile_metallib()

Or from the command line::

    python -m mlx_mfa.compile_metallib

After compilation, common STEEL V2 forward kernels are cached as precompiled
AIR metallibs in ``~/.mlx_mfa/metallib/``.  The C++ ShaderCache loads them on
subsequent runs, reducing cold-start latency from ~50ms to ~5ms per kernel.

Compiled configs cover:
  - D=64  BK=64  f16/bf16  causal/noncausal  (all gens)
  - D=128 BK=32  f16/bf16  causal/noncausal  (M1/M2)
  - D=128 BK=64  f16/bf16  causal/noncausal  (M3+, if is_m3_plus)

Filename scheme (matches C++ ShaderCache lookup)::

    v2_D{D}_BK{BK}_M{is_m3_plus}_dtype{dtype_code}_causal{0|1}.metallib
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

    m3 = 1 if is_m3_plus else 0

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
        filename = f"v2_D{D}_BK{BK}_M{m3}_dtype{dtype_code}_causal{int(causal)}.metallib"
        metallib_path = os.path.join(output_dir, filename)

        if os.path.exists(metallib_path) and not force:
            if verbose:
                print(f"[compile_metallib] Already compiled: {filename}")
            results[filename] = True
            continue

        if verbose:
            print(f"[compile_metallib] Compiling: {filename} ...", end=" ", flush=True)

        source = _capture_shader_source(D, BK, dtype_name, causal, is_m3_plus)
        if source is None:
            if verbose:
                print("FAILED (source generation)")
            results[filename] = False
            continue

        ok = _compile_source_to_metallib(source, metallib_path)
        results[filename] = ok
        if verbose:
            print("ok" if ok else "FAILED (xcrun)")

    if verbose:
        n_ok = sum(1 for v in results.values() if v)
        print(f"[compile_metallib] {n_ok}/{len(results)} configs compiled -> {output_dir}")

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
) -> Optional[str]:
    """Launch a subprocess that calls flash_attention with MFA_DEBUG_SHADERS=1
    and extract the V2 kernel source from stderr."""
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
        "mx.synchronize()",
    ]
    script = "\n".join(lines)

    env = dict(os.environ)
    env["MFA_DEBUG_SHADERS"] = "1"
    env.pop("MFA_DISABLE_V2", None)
    if D == 128:
        env["MFA_V2_FORCE_BK"] = str(BK)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env, capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return None

    stderr = result.stderr
    # Parse: "=== MFA Shader [steel_fwd_v2 ...] ===\n<source>\n=== END MFA Shader ==="
    pattern = (
        r"=== MFA Shader \[steel_fwd_v2[^\]]*\] ===\n"
        r"(.*?)"
        r"=== END MFA Shader ==="
    )
    m = re.search(pattern, stderr, re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


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
        except Exception:
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
    compile_metallib(output_dir=args.output_dir, force=args.force)
