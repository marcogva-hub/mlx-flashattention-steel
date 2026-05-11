#!/usr/bin/env bash
# Sprint G dispatch v6 cross-session A/B/A re-bench.
# Round 1: feat/v6-nax (v2.30.0, dispatch v5)
# Round 2: experiment/sprint-g-rebench-thermal-stable (dispatch v6 reapplied)
# Round 3: feat/v6-nax (thermal validation)
#
# Each round: clean checkout + force rebuild + bench in subprocess.
set -e
cd /Users/marcomarcelino/code/mlx-mfa-v2

OUT=/Users/marcomarcelino/code/mlx-mfa-v2/outputs/sprint-g-rebench-thermal-stable.json
LOG=/Users/marcomarcelino/code/mlx-mfa-v2/outputs/sprint_g_rebench.log
rm -f "$OUT"

echo "=== Sprint G re-bench start $(date) ===" | tee "$LOG"

# Round 1: feat/v6-nax (baseline, dispatch v5)
echo "" | tee -a "$LOG"
echo "=== R1: feat/v6-nax (dispatch v5, baseline) ===" | tee -a "$LOG"
git checkout feat/v6-nax 2>&1 | tail -1 | tee -a "$LOG"
CMAKE_ARGS="-DPython_EXECUTABLE=/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python" \
  /Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python -m pip install \
  --no-build-isolation -e . 2>&1 | tail -2 | tee -a "$LOG"
/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python /tmp/sprint_g_round_bench.py \
  --label "v5_R1" --output "$OUT" --pre-cooldown 300 --runs 3 2>&1 | tee -a "$LOG"

# Inter-round cooldown
echo "" | tee -a "$LOG"
echo "[wrapper] inter-round cooldown 120s" | tee -a "$LOG"
sleep 120

# Round 2: experiment branch (dispatch v6)
echo "" | tee -a "$LOG"
echo "=== R2: experiment/sprint-g-rebench-thermal-stable (dispatch v6) ===" | tee -a "$LOG"
git checkout experiment/sprint-g-rebench-thermal-stable 2>&1 | tail -1 | tee -a "$LOG"
CMAKE_ARGS="-DPython_EXECUTABLE=/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python" \
  /Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python -m pip install \
  --no-build-isolation -e . 2>&1 | tail -2 | tee -a "$LOG"
/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python /tmp/sprint_g_round_bench.py \
  --label "v6_R2" --output "$OUT" --pre-cooldown 0 --runs 3 2>&1 | tee -a "$LOG"

# Inter-round cooldown
echo "" | tee -a "$LOG"
echo "[wrapper] inter-round cooldown 120s" | tee -a "$LOG"
sleep 120

# Round 3: feat/v6-nax (thermal validation)
echo "" | tee -a "$LOG"
echo "=== R3: feat/v6-nax (thermal validation) ===" | tee -a "$LOG"
git checkout feat/v6-nax 2>&1 | tail -1 | tee -a "$LOG"
CMAKE_ARGS="-DPython_EXECUTABLE=/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python" \
  /Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python -m pip install \
  --no-build-isolation -e . 2>&1 | tail -2 | tee -a "$LOG"
/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python /tmp/sprint_g_round_bench.py \
  --label "v5_R3" --output "$OUT" --pre-cooldown 0 --runs 3 2>&1 | tee -a "$LOG"

# Restore experiment branch
echo "" | tee -a "$LOG"
echo "=== Restore experiment branch ===" | tee -a "$LOG"
git checkout experiment/sprint-g-rebench-thermal-stable 2>&1 | tail -1 | tee -a "$LOG"
CMAKE_ARGS="-DPython_EXECUTABLE=/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python" \
  /Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python -m pip install \
  --no-build-isolation -e . 2>&1 | tail -2 | tee -a "$LOG"

echo "=== Sprint G re-bench done $(date) ===" | tee -a "$LOG"
