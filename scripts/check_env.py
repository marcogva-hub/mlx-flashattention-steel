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

    # MLX
    ok &= check("MLX", lambda: __import__("mlx").__version__)

    # MLX include path
    def check_mlx_headers():
        import mlx, os
        inc = os.path.join(os.path.dirname(mlx.__file__), "include", "mlx", "mlx.h")
        assert os.path.exists(inc), f"Not found: {inc}"
        return inc
    ok &= check("MLX headers", check_mlx_headers)

    # MLX lib
    def check_mlx_lib():
        import mlx, os, glob
        base = os.path.dirname(mlx.__file__)
        libs = glob.glob(os.path.join(base, "lib", "libmlx*")) + \
               glob.glob(os.path.join(base, "libmlx*"))
        assert libs, f"No libmlx found in {base}"
        return libs[0]
    ok &= check("MLX library", check_mlx_lib)

    # nanobind
    ok &= check("nanobind", lambda: __import__("nanobind").__version__)

    # nanobind cmake dir
    def check_nb_cmake():
        r = subprocess.run(
            [sys.executable, "-m", "nanobind", "--cmake_dir"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout.strip()
    ok &= check("nanobind cmake", check_nb_cmake)

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
