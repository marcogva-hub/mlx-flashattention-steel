#!/usr/bin/env bash
set -e
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH
cd /Users/marcomarcelino/code/mlx-mfa-v2

OUT_DIR=outputs/diagnostic
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/phase-a3-warmup-test.log"
WARMUP_OUT="$OUT_DIR/a3-postwarmup-seedvr2small.json"
rm -f "$WARMUP_OUT"

echo "=== Phase A.3.1 ramp-up test start $(date) ===" | tee "$LOG"
echo "[A.3.1] 60s short cooldown ..." | tee -a "$LOG"
sleep 60

echo "[A.3.1] Running 30s sustained matmul workload ..." | tee -a "$LOG"
.venv/bin/python /tmp/a3_warmup.py 2>&1 | tee -a "$LOG"

echo "[A.3.1] Bench SeedVR2-small immediately post-warmup ..." | tee -a "$LOG"
.venv/bin/python bench/v34_bench.py \
    --shape SeedVR2-small --mode legacy --runs 5 \
    --include-sdpa --output "$WARMUP_OUT" 2>&1 | tee -a "$LOG"

echo "=== Phase A.3.1 done $(date) ===" | tee -a "$LOG"
