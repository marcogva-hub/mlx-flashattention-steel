#!/usr/bin/env bash
# check_venv.sh — verify canonical .venv has all required release-flow tools.
#
# Per CLAUDE.md "Canonical Python environment" and CLAUDE_V6_NAX.md §X.5
# (pre-tag tool-availability audit, added 2026-05-13).  Run before any
# release flow (version bump, pytest, build, twine upload).
#
# Usage:
#   bash scripts/check_venv.sh        # check; install missing if any
#   bash scripts/check_venv.sh --no-install   # check-only; exit non-zero on missing
#
# Exit codes:
#   0  — canonical .venv verified, all tools present (post-install if needed)
#   1  — .venv missing (cannot be auto-recreated by this script)
#   2  — a tool is missing and --no-install was passed
#   3  — a tool failed verification after install (transient or upstream issue)

set -euo pipefail

# Resolve repo root from script location, regardless of CWD
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
cd "${REPO_ROOT}"

NO_INSTALL=0
if [[ "${1:-}" == "--no-install" ]]; then
  NO_INSTALL=1
fi

VENV_PY=".venv/bin/python"

if ! test -f "${VENV_PY}"; then
  echo "❌ ${VENV_PY} missing — canonical venv not present in ${REPO_ROOT}"
  echo "   To create:"
  echo "     python3 -m venv .venv"
  echo "     .venv/bin/pip install --upgrade pip"
  echo "     CMAKE_ARGS=\"-DPython_EXECUTABLE=.venv/bin/python\" \\"
  echo "       .venv/bin/python -m pip install --no-build-isolation -e ."
  exit 1
fi

# Tools that ship as binaries in .venv/bin/
BINARY_TOOLS=("twine" "pytest")
# Tools that ship as Python modules only (no .venv/bin/<tool> exists)
MODULE_TOOLS=("build")

# Note: under `set -u`, expanding an empty array via "${arr[@]}" raises
# "unbound variable".  Use "${arr[@]+"${arr[@]}"}" to safely expand
# (no-op if empty, full expansion if populated).  See Bash FAQ #112.
missing_binaries=()
missing_modules=()

for tool in "${BINARY_TOOLS[@]}"; do
  if ! test -f ".venv/bin/${tool}"; then
    missing_binaries+=("${tool}")
  fi
done

for mod in "${MODULE_TOOLS[@]}"; do
  if ! "${VENV_PY}" -c "import ${mod}" 2> /dev/null; then
    missing_modules+=("${mod}")
  fi
done

missing_total=(
  ${missing_binaries[@]+"${missing_binaries[@]}"}
  ${missing_modules[@]+"${missing_modules[@]}"}
)

if [[ ${#missing_total[@]} -gt 0 ]]; then
  echo "⚠ Missing tools in .venv: ${missing_total[*]}"

  if [[ ${NO_INSTALL} -eq 1 ]]; then
    echo "   --no-install passed; not installing.  Run without --no-install"
    echo "   or: .venv/bin/pip install ${missing_total[*]}"
    exit 2
  fi

  echo "Installing in-place via .venv/bin/pip..."
  .venv/bin/pip install --quiet "${missing_total[@]}"
  echo "✓ Installed: ${missing_total[*]}"
fi

# Verify all tools actually work post-install.  Array-driven so adding
# a new tool requires editing the BINARY_TOOLS/MODULE_TOOLS arrays only —
# no second edit site here (per pre-commit /mlx-code-review MEDIUM).
for tool in "${BINARY_TOOLS[@]}"; do
  if ! ".venv/bin/${tool}" --version > /dev/null 2>&1; then
    echo "❌ ${tool} --version failed post-install"
    exit 3
  fi
done

# Build a single `import a; import b; ...` line and run it once.
import_cmd=""
for mod in "${MODULE_TOOLS[@]}"; do
  import_cmd+="import ${mod}; "
done
if [[ -n "${import_cmd}" ]]; then
  if ! "${VENV_PY}" -c "${import_cmd}" 2> /dev/null; then
    echo "❌ Module import failed post-install (one of: ${MODULE_TOOLS[*]})"
    exit 3
  fi
fi

# Verify mlx is importable (catches editable-install drift)
if ! "${VENV_PY}" -c "import mlx.core, mlx_mfa" 2> /dev/null; then
  echo "⚠ mlx / mlx_mfa not importable — extension may need rebuild:"
  echo "    CMAKE_ARGS=\"-DPython_EXECUTABLE=.venv/bin/python\" \\"
  echo "      .venv/bin/python -m pip install --no-build-isolation -e ."
  # Non-fatal: release flow may still be able to operate on a non-built
  # source tree (e.g., wheel build doesn't need a pre-built extension).
fi

echo "✓ .venv canonical environment verified"
echo "  Python:  $("${VENV_PY}" --version 2>&1)"
echo "  Tools:   twine + build + pytest available"
