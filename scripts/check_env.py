#!/usr/bin/env python3
"""Pre-build environment check for mlx-mfa."""

import sys
import subprocess


def check(name, fn):
    try:
        result = fn()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def info(name, fn):
    """Informational line: printed, never part of the pass/fail verdict."""
    try:
        print(f"  [INFO] {name}: {fn()}")
    except Exception as e:
        print(f"  [INFO] {name}: {e}")


def _verified_mlx_nanobind_table():
    """{mlx_version: nanobind_tag} from csrc/cmake/MlxNanobindAbi.cmake (the build's SoT)."""
    import os
    import re
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                        "csrc", "cmake", "MlxNanobindAbi.cmake")
    with open(path) as f:
        return dict(re.findall(r'"([0-9]+\.[0-9]+\.[0-9]+)=(v[0-9.]+)=[0-9]+"', f.read()))


def main():
    print("mlx-mfa environment check")
    print("=" * 50)
    ok = True

    # Python
    ok &= check("Python", lambda: sys.version.split()[0])

    # Platform
    import platform
    ok &= check("Platform", lambda: f"{platform.system()} {platform.machine()}")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("  [WARN] mlx-mfa requires macOS on Apple Silicon (arm64)")

    # MLX (version lives on mlx.core since ≥0.19).  BLD-01/A12 (review 2026-09): the
    # build maps each VERIFIED MLX release to its nanobind tag and FAILS on anything
    # else (csrc/cmake/MlxNanobindAbi.cmake) — check the same table here so an
    # unbuildable MLX is reported BEFORE the build, not as a CMake FATAL.
    def check_mlx():
        import mlx.core
        v = mlx.core.__version__
        table = _verified_mlx_nanobind_table()
        if v not in table:
            raise RuntimeError(
                f"MLX {v} is not in the verified nanobind-ABI table {sorted(table)} — "
                "the build will refuse it (install one of those MLX releases)")
        return f"{v} -> nanobind {table[v]} (verified table)"
    ok &= check("MLX", check_mlx)

    # MLX include path (mlx is a namespace pkg; use mlx.__path__)
    def check_mlx_headers():
        import mlx, os
        base = list(mlx.__path__)[0]  # site-packages/mlx/
        inc = os.path.join(base, "include", "mlx", "array.h")
        assert os.path.exists(inc), f"Not found: {inc}"
        return inc
    ok &= check("MLX headers", check_mlx_headers)

    # MLX lib
    def check_mlx_lib():
        import mlx, os, glob
        base = list(mlx.__path__)[0]
        libs = glob.glob(os.path.join(base, "lib", "libmlx*"))
        assert libs, f"No libmlx found in {base}/lib/"
        return libs[0]
    ok &= check("MLX library", check_mlx_lib)

    # nanobind — NOT a requirement (A12, review 2026-09: this line failed CI since
    # 2026-06): the build FetchContents the nanobind tag ABI-matched to the detected
    # MLX (csrc/cmake/MlxNanobindAbi.cmake) and never uses a pip nanobind.
    def check_nanobind():
        try:
            ver = __import__("nanobind").__version__
        except ImportError:
            return "pip nanobind not installed — not needed (CMake fetches the ABI-matched tag)"
        return f"pip nanobind {ver} present — unused by the build (CMake fetches the ABI-matched tag)"
    info("nanobind", check_nanobind)

    # CMake
    def check_cmake():
        r = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
        return r.stdout.split("\n")[0]
    ok &= check("CMake", check_cmake)

    # Xcode CLT
    def check_xcode():
        r = subprocess.run(["xcode-select", "-p"], capture_output=True, text=True)
        return r.stdout.strip()
    ok &= check("Xcode CLT", check_xcode)

    print("=" * 50)
    if ok:
        print("All checks passed. Ready to build.")
    else:
        print("Some checks failed. Fix issues above before building.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
