#!/usr/bin/env bash
# DAY-3 launcher (2026-09-30): the August runner (--day3) DETACHED via LaunchAgent (the
# validated mechanism — a nohup wrapper dies with the terminal), under caffeinate, plus a
# separate macmon LaunchAgent (light telemetry, 5 s). Stops by EXHAUSTION (no deadline
# pressure: deadline set far away); Marco controls the window with PAUSE / STOP.
#
#   launch_day3.sh <label-suffix> <queue.jsonl> <run_dir> [--no-zoom] [--no-macmon] [--tail-queue <q>]
#   stop agents:  launchctl bootout gui/$(id -u)/com.marco.mfa.<suffix>{,-macmon}
set -euo pipefail
SUFFIX="${1:?label suffix}"; QUEUE="${2:?queue}"; RUN="${3:?run dir}"; shift 3
ZOOM=""; MACMON=1; TAIL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --no-zoom) ZOOM="--no-zoom" ;;
    --no-macmon) MACMON=0 ;;
    --tail-queue) TAIL="$2"; shift ;;
    *) echo "unknown $1" >&2; exit 2 ;;
  esac
  shift
done
MFA=/Users/marcomarcelino/code/mlx-mfa-v2
ATTIC=/Users/marcomarcelino/code/mlx-mfa-attic/na-autoresearch
PY=$MFA/.venv/bin/python
LABEL=com.marco.mfa.$SUFFIX
AGENTS=$HOME/Library/LaunchAgents
mkdir -p "$RUN/telemetry"
DEADLINE=$(( $(date +%s) + 24*3600 ))

plist_job="$RUN/$SUFFIX.job.launchagent.plist"
cat > "$plist_job" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>KeepAlive</key><false/>
  <key>RunAtLoad</key><true/>
  <key>ProcessType</key><string>Interactive</string>
  <key>WorkingDirectory</key><string>$MFA</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PYTHONPATH</key><string>$ATTIC</string>
    <key>PYTHONUNBUFFERED</key><string>1</string>
    <key>MFA_SILENCE_NAX_WARNING</key><string>1</string>
    <key>NAAR_FLOORS_B1H12</key><string>1</string>
    <key>DAY3_MFA_ROOT</key><string>$MFA</string>
  </dict>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/caffeinate</string><string>-dims</string>
    <string>$PY</string><string>-m</string><string>research.na_autoresearch.runner.night_runner</string>
    <string>--queue</string><string>$QUEUE</string>
    <string>--out</string><string>$RUN</string>
    <string>--day3</string>
    <string>--deadline-epoch</string><string>$DEADLINE</string>
    $( [ -n "$ZOOM" ] && echo "<string>$ZOOM</string>" )
    $( [ -n "$TAIL" ] && echo "<string>--tail-queue</string><string>$TAIL</string>" )
  </array>
  <key>StandardOutPath</key><string>$RUN/driver.log</string>
  <key>StandardErrorPath</key><string>$RUN/driver.err</string>
</dict></plist>
EOF
plutil -lint "$plist_job" >/dev/null

if [ "$MACMON" = 1 ]; then
  plist_mm="$RUN/$SUFFIX.macmon.launchagent.plist"
  cat > "$plist_mm" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL-macmon</string>
  <key>KeepAlive</key><false/>
  <key>RunAtLoad</key><true/>
  <key>ProgramArguments</key><array>
    <string>/opt/homebrew/bin/macmon</string><string>-i</string><string>5000</string><string>pipe</string>
  </array>
  <key>StandardOutPath</key><string>$RUN/telemetry/macmon.jsonl</string>
  <key>StandardErrorPath</key><string>$RUN/telemetry/macmon.err</string>
</dict></plist>
EOF
  plutil -lint "$plist_mm" >/dev/null
  cp "$plist_mm" "$AGENTS/$LABEL-macmon.plist"
  launchctl bootstrap "gui/$(id -u)" "$AGENTS/$LABEL-macmon.plist"
  sleep 7        # a fresh macmon sample for the runner's host preflight
fi
cp "$plist_job" "$AGENTS/$LABEL.plist"
launchctl bootstrap "gui/$(id -u)" "$AGENTS/$LABEL.plist"
echo "LAUNCHED $LABEL → $RUN  (queue $(wc -l < "$QUEUE") cells)"
echo "  pause : touch $RUN/PAUSE"
echo "  resume: rm $RUN/PAUSE"
echo "  stop  : touch $RUN/STOP"
echo "  status: cat $RUN/status.txt"
