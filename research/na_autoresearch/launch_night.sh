#!/usr/bin/env bash
# Launch the NA-autoresearch night runner DETACHED under caffeinate.
# The runner is self-sufficient: this session may end, the sweep continues until
# the queue+zoom+backlog are exhausted, the watchdog STOPs, or 07:00.
#
# Usage:
#   research/na_autoresearch/launch_night.sh <queue.jsonl> [backlog.jsonl]
#
# Deadline defaults to the next 07:00 local. Run dir is dated under
# benchmarks/results/autoresearch/. RULE 12: pre-launch health is the runner's
# own head block (day-floors + RAM refuse); we still echo the machine state here.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${NAAR_PY:-.venv/bin/python}"

QUEUE="${1:?usage: launch_night.sh <queue.jsonl> [backlog.jsonl]}"
BACKLOG="${2:-}"

now=$(date +%s)
today7=$(date -v7H -v0M -v0S +%s)
if [ "$now" -lt "$today7" ]; then DEADLINE=$today7; else DEADLINE=$(date -v+1d -v7H -v0M -v0S +%s); fi

RUN_DIR="benchmarks/results/autoresearch/night-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "=== pre-launch machine state (RULE 12) ==="
pmset -g 2>/dev/null | grep -i powermode || true
memory_pressure 2>/dev/null | tail -1 || true
ioreg -r -d 1 -c AGXAccelerator 2>/dev/null | grep -oE '"Device Utilization %"=[0-9]+' | head -1 || true
echo "  deadline (07:00): $(date -r $DEADLINE '+%Y-%m-%d %H:%M:%S')  ($((($DEADLINE-now)/3600))h $(((($DEADLINE-now)%3600)/60))m from now)"
echo "  queue: $QUEUE ($(wc -l < "$QUEUE") cells)  backlog: ${BACKLOG:-none}"
echo "  run dir: $RUN_DIR"

BACKLOG_ARG=()
[ -n "$BACKLOG" ] && BACKLOG_ARG=(--backlog "$BACKLOG")

MFA_SILENCE_NAX_WARNING=1 \
  nohup caffeinate -dims "$PY" -m research.na_autoresearch.runner.night_runner \
    --queue "$QUEUE" --out "$RUN_DIR" ${BACKLOG_ARG[@]+"${BACKLOG_ARG[@]}"} \
    --deadline-epoch "$DEADLINE" > "$RUN_DIR/driver.log" 2>&1 &
PID=$!
echo "$PID" > "$RUN_DIR/runner.pid"
echo "=== LAUNCHED detached PID $PID → $RUN_DIR ==="
echo "  tail:   tail -f $RUN_DIR/runner.log"
echo "  stop:   touch $RUN_DIR/STOP   (clean stop after the current cell)"
echo "  report: $RUN_DIR/morning_report.md  (written on stop)"
