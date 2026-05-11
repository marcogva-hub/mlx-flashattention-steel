#!/usr/bin/env bash
# V34 vs legacy A/B/A cross-session bench wrapper.
# 5 production shapes × {legacy, v34, legacy_validation} rounds.
# 90s inter-round cooldown, 30s inter-shape cooldown for thermal stability.
set -e
cd /Users/marcomarcelino/code/mlx-mfa-v2

OUT=outputs/v34-aba.json
LOG=outputs/v34_aba.log
SHAPES=("FlashVSR-dense" "LTX2-cross" "SeedVR2-small" "CogVideoX" "SeedVR2-large")

mkdir -p outputs
rm -f "$OUT"

echo "=== V34 A/B/A bench start $(date) ===" | tee "$LOG"

# Initial 90s cooldown
echo "[wrapper] initial 90s cooldown" | tee -a "$LOG"
sleep 90

for shape in "${SHAPES[@]}"; do
  echo "" | tee -a "$LOG"
  echo "=== shape=$shape ===" | tee -a "$LOG"
  # Round 1: legacy
  .venv/bin/python bench/v34_bench.py --shape "$shape" --mode legacy --runs 3 \
                   --include-sdpa --output "$OUT" 2>&1 | tee -a "$LOG"
  sleep 60
  # Round 2: v34
  .venv/bin/python bench/v34_bench.py --shape "$shape" --mode v34 --runs 3 \
                   --include-sdpa --output "$OUT" 2>&1 | tee -a "$LOG"
  sleep 60
  # Round 3: legacy validation
  .venv/bin/python bench/v34_bench.py --shape "$shape" --mode legacy --runs 3 \
                   --include-sdpa --output "$OUT" 2>&1 | tee -a "$LOG"
  echo "[wrapper] inter-shape cooldown 30s" | tee -a "$LOG"
  sleep 30
done

echo "=== V34 A/B/A bench done $(date) ===" | tee -a "$LOG"
