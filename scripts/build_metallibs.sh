#!/usr/bin/env bash
# build_metallibs.sh — Pre-compile common STEEL V2 Metal kernels to AIR metallibs.
#
# Usage:
#   ./scripts/build_metallibs.sh [--force] [--output-dir DIR]
#
# After running, precompiled metallibs are stored in ~/.mlx_mfa/metallib/.
# The ShaderCache C++ layer loads them on startup, reducing cold-start JIT
# latency from ~50ms to ~2-5ms per kernel.
#
# Kernels compiled:
#   Standard V2 (D=64/128): mlx_mfa_v2_attention
#   D-split V2  (D=256/512): mlx_mfa_v2_dsplit_attention
#
# Note: simdgroup_async_copy is not available on macOS 26 (Darwin 25.x) in
# either the runtime (newLibraryWithSource:) or offline (xcrun metal) compilers.
# The compiled metallibs use standard per-lane threadgroup loads — the same
# code path as the JIT kernel — but benefit from the offline compiler's more
# aggressive optimization passes (inlining, LICM, instruction scheduling).
#
# Requirements:
#   - mlx-mfa installed in the active Python environment
#   - Xcode Command Line Tools (xcrun metal)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Determine Python executable
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  VENV_PY="${REPO_ROOT}/.venv/bin/python"
  if [[ -x "${VENV_PY}" ]]; then
    PYTHON="${VENV_PY}"
  elif command -v python3 &>/dev/null; then
    PYTHON="python3"
  else
    echo "ERROR: no Python found. Set PYTHON= or activate a virtualenv." >&2
    exit 1
  fi
fi

echo "Using Python: ${PYTHON}"
echo "mlx-mfa version: $("${PYTHON}" -c 'import mlx_mfa; print(mlx_mfa.__version__)' 2>/dev/null || echo 'not installed')"

# Check xcrun metal
if ! xcrun metal --version &>/dev/null; then
  echo "ERROR: xcrun metal not found. Install Xcode Command Line Tools." >&2
  exit 1
fi

# Forward all args to compile_metallib module
exec "${PYTHON}" -m mlx_mfa.compile_metallib "$@"
