#!/usr/bin/env bash
# Phase A.1 — PSO cache hypothesis test
#
# Procedure: clear Python Metal PSO cache, run cold legacy bench on 3
# D=128 production shapes, immediately re-run warm legacy bench. Compare
# to v2.31.0 (275.6/3669/6780 ms) and Phase 0 (167.75/2344/3982 ms).
#
# Discriminant:
#   Cold ≈ v2.31.0 AND Warm ≈ Phase 0  → PSO cache CONFIRMED
#   Cold ≈ Warm ≈ Phase 0              → PSO REJECTED
#   Both slow                          → other culprit (e.g., driver state)
set -e

export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH

cd "$(/usr/bin/dirname "$0")/.."
REPO_ROOT="$(pwd)"

# Cache path on macOS 26 (verified 2026-05-06 in Phase A.0)
PY_METAL_CACHE=/var/folders/c2/pwjb45v12rl4tf2k56vvh_300000gn/C/org.python.python/com.apple.metal
PY_METALFE_CACHE=/var/folders/c2/pwjb45v12rl4tf2k56vvh_300000gn/C/org.python.python/com.apple.metalfe

OUT_DIR="$REPO_ROOT/outputs/diagnostic"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/phase-a1-pso-aba.log"

# Cold-bench output files (separate JSONs so we don't mix records)
COLD_S="$OUT_DIR/a1-cold-seedvr2small.json"
COLD_C="$OUT_DIR/a1-cold-cogvideox.json"
COLD_L="$OUT_DIR/a1-cold-seedvr2large.json"
WARM_S="$OUT_DIR/a1-warm-seedvr2small.json"
WARM_C="$OUT_DIR/a1-warm-cogvideox.json"
WARM_L="$OUT_DIR/a1-warm-seedvr2large.json"
rm -f "$COLD_S" "$COLD_C" "$COLD_L" "$WARM_S" "$WARM_C" "$WARM_L"

echo "=== Phase A.1 PSO cache A/B start $(/bin/date) ===" | /usr/bin/tee "$LOG"

# --- Step 1: snapshot cache, then clear ---
echo | /usr/bin/tee -a "$LOG"
echo "[A.1] Cache size BEFORE clear:" | /usr/bin/tee -a "$LOG"
/usr/bin/du -sh "$PY_METAL_CACHE" 2>&1 | /usr/bin/tee -a "$LOG"
/usr/bin/du -sh "$PY_METALFE_CACHE" 2>&1 | /usr/bin/tee -a "$LOG"

echo "[A.1] Clearing $PY_METAL_CACHE ..." | /usr/bin/tee -a "$LOG"
/bin/rm -rf "$PY_METAL_CACHE"/* 2>&1 | /usr/bin/tee -a "$LOG" || true
echo "[A.1] Clearing $PY_METALFE_CACHE ..." | /usr/bin/tee -a "$LOG"
/bin/rm -rf "$PY_METALFE_CACHE"/* 2>&1 | /usr/bin/tee -a "$LOG" || true

echo "[A.1] Cache size AFTER clear:" | /usr/bin/tee -a "$LOG"
/usr/bin/du -sh "$PY_METAL_CACHE" 2>&1 | /usr/bin/tee -a "$LOG"
/usr/bin/du -sh "$PY_METALFE_CACHE" 2>&1 | /usr/bin/tee -a "$LOG"

# --- Step 2: 3-min cooldown ---
echo | /usr/bin/tee -a "$LOG"
echo "[A.1] 180s initial cooldown ..." | /usr/bin/tee -a "$LOG"
/bin/sleep 180

# --- Step 3: COLD bench (cache empty at start of first subprocess) ---
echo | /usr/bin/tee -a "$LOG"
echo "=== A.1 COLD BENCH (cleared cache) ===" | /usr/bin/tee -a "$LOG"

echo "[A.1] cold SeedVR2-small ..." | /usr/bin/tee -a "$LOG"
"$REPO_ROOT/.venv/bin/python" bench/v6nax_bench.py \
    --shape SeedVR2-small --mode legacy --runs 5 \
    --include-sdpa --output "$COLD_S" 2>&1 | /usr/bin/tee -a "$LOG"
/bin/sleep 60

echo "[A.1] cold CogVideoX ..." | /usr/bin/tee -a "$LOG"
"$REPO_ROOT/.venv/bin/python" bench/v6nax_bench.py \
    --shape CogVideoX --mode legacy --runs 5 \
    --include-sdpa --output "$COLD_C" 2>&1 | /usr/bin/tee -a "$LOG"
/bin/sleep 60

echo "[A.1] cold SeedVR2-large ..." | /usr/bin/tee -a "$LOG"
"$REPO_ROOT/.venv/bin/python" bench/v6nax_bench.py \
    --shape SeedVR2-large --mode legacy --runs 5 \
    --include-sdpa --output "$COLD_L" 2>&1 | /usr/bin/tee -a "$LOG"

echo | /usr/bin/tee -a "$LOG"
echo "[A.1] Cache size AFTER cold bench:" | /usr/bin/tee -a "$LOG"
/usr/bin/du -sh "$PY_METAL_CACHE" 2>&1 | /usr/bin/tee -a "$LOG"
/usr/bin/du -sh "$PY_METALFE_CACHE" 2>&1 | /usr/bin/tee -a "$LOG"

# --- Step 4: short cooldown then WARM bench ---
echo | /usr/bin/tee -a "$LOG"
echo "[A.1] 30s cooldown before warm bench ..." | /usr/bin/tee -a "$LOG"
/bin/sleep 30

echo | /usr/bin/tee -a "$LOG"
echo "=== A.1 WARM BENCH (cache populated by cold pass) ===" | /usr/bin/tee -a "$LOG"

echo "[A.1] warm SeedVR2-small ..." | /usr/bin/tee -a "$LOG"
"$REPO_ROOT/.venv/bin/python" bench/v6nax_bench.py \
    --shape SeedVR2-small --mode legacy --runs 5 \
    --include-sdpa --output "$WARM_S" 2>&1 | /usr/bin/tee -a "$LOG"
/bin/sleep 60

echo "[A.1] warm CogVideoX ..." | /usr/bin/tee -a "$LOG"
"$REPO_ROOT/.venv/bin/python" bench/v6nax_bench.py \
    --shape CogVideoX --mode legacy --runs 5 \
    --include-sdpa --output "$WARM_C" 2>&1 | /usr/bin/tee -a "$LOG"
/bin/sleep 60

echo "[A.1] warm SeedVR2-large ..." | /usr/bin/tee -a "$LOG"
"$REPO_ROOT/.venv/bin/python" bench/v6nax_bench.py \
    --shape SeedVR2-large --mode legacy --runs 5 \
    --include-sdpa --output "$WARM_L" 2>&1 | /usr/bin/tee -a "$LOG"

echo | /usr/bin/tee -a "$LOG"
echo "=== Phase A.1 done $(/bin/date) ===" | /usr/bin/tee -a "$LOG"
