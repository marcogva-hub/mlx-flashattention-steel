#!/usr/bin/env bash
# 3-session §4-strict matched-workload-family validation driver.
# Each session is its own subprocess (per DM7).  Runlogs captured to
# docs/methodology/matched-workload-runlog-M{1,2,3}.txt.
#
# Expected wall-clock per session ≈ 30 min (initial 180s + 7 shapes ×
# (3s measurement + 180s round cooldowns) + 6 × 60s inter-shape) ≈
# 1821s.  Three sessions ≈ 91 min total.

set -euo pipefail

cd "$(dirname "$0")/../.."

PY=".venv/bin/python"
HARNESS="bench/methodology/matched_workload_harness.py"
OUT_DIR="docs/methodology"
DATA_JSON="$OUT_DIR/matched-workload-data.json"

# Inter-session gap: 60s for subprocess teardown + GPU power-state
# stabilization between sessions.  Cleaner than chaining processes back-
# to-back which could leak state.
INTER_SESSION_GAP=60

mkdir -p "$OUT_DIR"

echo "================================================================"
echo "matched-workload-family 3-session run starting: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

for i in 1 2 3; do
  sid="M${i}"
  log="$OUT_DIR/matched-workload-runlog-${sid}.txt"
  echo ""
  echo "[driver] session ${sid} starting (log: ${log}) at $(date -u +%H:%M:%S)"
  "$PY" "$HARNESS" --session-id "$sid" --output "$DATA_JSON" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  echo "[driver] session ${sid} exit=${rc} at $(date -u +%H:%M:%S)"
  if [ "$rc" -ne 0 ]; then
    echo "[driver] ABORTING: session ${sid} failed with exit ${rc}" >&2
    exit "$rc"
  fi
  if [ "$i" -ne 3 ]; then
    echo "[driver] inter-session gap ${INTER_SESSION_GAP}s..."
    sleep "$INTER_SESSION_GAP"
  fi
done

echo ""
echo "================================================================"
echo "all 3 sessions complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

echo "[driver] running analysis..."
"$PY" bench/methodology/matched_workload_analysis.py 2>&1

echo "[driver] done."
