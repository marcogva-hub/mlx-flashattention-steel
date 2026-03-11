#!/usr/bin/env bash
# build_async_metallib.sh — Compile async_v2_kernel.metal to async_v2.metallib.
#
# Usage:
#   ./scripts/build_async_metallib.sh [--output-dir DIR]
#
# Produces mlx_mfa/precompiled/async_v2.metallib (or --output-dir).
# The metallib contains two entry points:
#   mlx_mfa_v2_async_attention        (D=64,  BK=64)
#   mlx_mfa_v2_async_attention_d128   (D=128, BK=32)
#
# Function constants (set via MTLFunctionConstantValues at runtime):
#   index 0 (bool)   FC_CAUSAL      — causal masking on/off
#   index 1 (ushort) FC_GQA_FACTOR  — GQA ratio (1 = standard MHA)
#
# Requirements:
#   - Xcode Command Line Tools (xcrun metal)
#   - macOS ≤15 / Xcode ≤16 for simdgroup_async_copy support.
#     On macOS 26 the xcrun metal compiler rejects __asm("air.simdgroup_async_copy_2d...")
#     The script exits non-zero; the .metal source is still preserved for CI builds.
#
# Note: This metallib ships with the package in mlx_mfa/precompiled/.
# ShaderCache::try_precompiled_pipeline() checks for it first, falling back
# to the synchronous AOT metallib, then JIT compilation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

METAL_SRC="${REPO_ROOT}/csrc/async_v2_kernel.metal"
OUTPUT_DIR="${1:-${REPO_ROOT}/mlx_mfa/precompiled}"

# Allow override via --output-dir flag
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir|-o) OUTPUT_DIR="$2"; shift 2 ;;
    *) shift ;;
  esac
done

OUTPUT_FILE="${OUTPUT_DIR}/async_v2.metallib"

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

echo "Building async V2 metallib..."
echo "  Source:  ${METAL_SRC}"
echo "  Output:  ${OUTPUT_FILE}"
echo ""

# Check xcrun metal
if ! xcrun metal --version &>/dev/null; then
  echo "ERROR: xcrun metal not found. Install Xcode Command Line Tools." >&2
  exit 1
fi

METAL_VERSION="$(xcrun metal --version 2>&1 | head -1)"
echo "  Compiler: ${METAL_VERSION}"
echo ""

# Compile .metal → .air
AIR_FILE="$(mktemp /tmp/async_v2_XXXX.air)"
trap 'rm -f "${AIR_FILE}"' EXIT

echo "Step 1: Compiling Metal source to AIR..."
if ! xcrun metal \
    -target air64-apple-macos15.0 \
    -std=metal3.1 \
    -c "${METAL_SRC}" \
    -o "${AIR_FILE}" 2>&1; then
  echo ""
  echo "NOTE: xcrun metal rejected the __asm simdgroup_async_copy intrinsics."
  echo "This is expected on macOS 26 (Darwin 25.x) where Apple removed the"
  echo "simdgroup_async_copy runtime and offline compiler support."
  echo ""
  echo "The source file csrc/async_v2_kernel.metal has been preserved."
  echo "To compile, use Xcode ≤16 on macOS ≤15, or a GitHub Actions macos-14 runner."
  echo ""
  echo "The runtime fallback chain is:"
  echo "  async_v2.metallib (not available) → sync AOT metallib → JIT"
  exit 1
fi

# Link .air → .metallib
echo "Step 2: Linking AIR to metallib..."
xcrun metallib "${AIR_FILE}" -o "${OUTPUT_FILE}"

echo ""
echo "Success! Metallib written to: ${OUTPUT_FILE}"
echo ""
echo "Entry points:"
echo "  mlx_mfa_v2_async_attention        (D=64,  BQ=32, BK=64)"
echo "  mlx_mfa_v2_async_attention_d128   (D=128, BQ=32, BK=32)"
echo ""
echo "Function constants:"
echo "  index 0 (bool)   FC_CAUSAL"
echo "  index 1 (ushort) FC_GQA_FACTOR"
