# AUTORESEARCH — mlx-mfa D=256 V2 D-split Kernel Optimization

## Objective

Maximize `SPEEDUP_RATIO` from `benchmarks/bench_dispatch_d256_kernel.py`.

This ratio is the **geometric mean** of V2 D-split vs SDPA at D=256 causal
for N ∈ {2048, 4096, 8192} (B=2, H=8, f16 — production-like profile).

Baseline on M1 Max (decision pass 2026-03-12, unmodified code):
- N=2048: ~0.94x
- N=4096: ~0.98x
- N=8192: ~1.01–1.11x
- Geomean baseline: record after first bench run

**Target: geomean > 1.02x consistently across all 3 N values.**

---

## Context — internalize before iteration 1

### Kernel architecture

The D=256 path uses `generate_steel_v2_dsplit_source()` in `mfa_steel_fwd_v2.cpp`.
It processes D=256 in 2 passes (D_SPLITS=2, BD_HALF=128 per pass).
Each pass is equivalent to a D=128 attention kernel.

Block config is set by `select_steel_v2_dsplit_block_config()` (lines ~100–115):

```cpp
SteelV2BlockConfig select_steel_v2_dsplit_block_config(bool is_m3_plus) {
  int forced_bk = 0;
  if (const char* env = std::getenv("MFA_V2_FORCE_BK_D256")) {
    const int parsed = std::atoi(env);
    if (parsed == 32 || parsed == 64) forced_bk = parsed;
  }
  const int bk = forced_bk ? forced_bk : (is_m3_plus ? 64 : 32);
  return {32, bk, 128, 4, 1};  // BQ=32, BK=bk, BD=128, WM=4, WN=1
}
```

`is_m3_plus=false` on M1 Max → default BK=32.

The return type `SteelV2BlockConfig` has fields: `{BQ, BK, BD, WM, WN}`.

### TGP budget (CRITICAL constraint)

Threadgroup memory (TGP) must stay under 32 KB.

For the D-split kernel, TGP is dominated by KV_smem:
- `KV_smem = BQ * (BD_HALF + padQ) + BK * (BD_HALF + padKV)` elements of T (f16 = 2 bytes)
- padQ = padKV = 16/sizeof(T) = 8 elements (bank-conflict padding)
- With BQ=32, BK=32: TGP ≈ 32*(128+8)*2 + 32*(128+8)*2 = 17,408 bytes ✓
- With BQ=32, BK=64: TGP ≈ 32*(128+8)*2 + 64*(128+8)*2 = 26,112 bytes ✓ (under 32KB)
- With BQ=64, BK=32: TGP ≈ 64*(128+8)*2 + 32*(128+8)*2 = 26,112 bytes ✓
- With BQ=64, BK=64: TGP ≈ 64*(128+8)*2 + 64*(128+8)*2 = 34,816 bytes ✗ (EXCEEDS 32KB)

Rule: BQ * (BD_HALF + 8) + BK * (BD_HALF + 8) must be ≤ 32768 / sizeof(T) = 16384 elements.
With BD_HALF=128 and padding=8: (BQ + BK) * 136 ≤ 16384 → BQ + BK ≤ 120.

Also: WM = TGP_SIZE / 32, where TGP_SIZE = WM * WN * 32. And TQ = BQ / (WM * 8) must be ≥ 1.
So BQ ≥ WM * 8.

### What BK controls

BK = number of K-sequence tokens processed per K-tile.
Larger BK → fewer K-tile iterations → less loop overhead but larger TGP.
The D-split already loops D_SPLITS=2 times, so K-tile count matters.

BK=32: N/32 K-tile iterations per D-split pass. Default M1.
BK=64: N/64 K-tile iterations. Half the loop overhead. `MFA_V2_FORCE_BK_D256=64` tests this.

### What BQ controls

BQ = number of Q rows processed per threadgroup.
Larger BQ → more Q parallelism → better occupancy for long N.
Smaller BQ → fewer registers per threadgroup → potentially less spill on M1.

### `enable_unroll` in D-split

Currently hardcoded `true` in `generate_steel_v2_dsplit_source()` (line ~858):
```cpp
const bool enable_unroll = true;
```
This is different from V1 where D=256 disables unroll on M1 due to register spill.
In D-split, each half is BD_HALF=128 wide (TD_HALF=16), which is safe to unroll.
However, if register pressure is an issue at specific BQ/BK combos, try `false`.

### The TGP padding

`pad_expr = no_padding ? "0" : "16 / sizeof(T)"` — currently 8 elements (16 bytes) per row.
Removing padding (MFA_NO_PADDING=1) eliminates ~4% TGP overhead but risks bank conflicts.
Reducing to 4 or 0 could help TGP budget and allow larger BK or BQ.

---

## Scope — what you MAY modify

**ONE file only: `csrc/mfa_steel_fwd_v2.cpp`**

Specifically, your search space:
1. `select_steel_v2_dsplit_block_config()` — BQ, BK, WM values (main lever)
2. `enable_unroll` in `generate_steel_v2_dsplit_source()` — try `false` if BQ/BK changes cause issues
3. `pad_expr` override for D-split only — try reduced padding for larger tile budget
4. The env var hook (`MFA_V2_FORCE_BK_D256`) — can be extended to also handle BQ

**Do NOT modify anything else.** Specifically: NEVER touch tests/, dispatch_policy.py,
benchmarks/, docs/, devnotes/, pyproject.toml, CMakeLists.txt, or any other .cpp/.hpp/.metal.

---

## Verification command (metric)

```bash
cd ~/code/mlx-mfa-v2
.venv/bin/python benchmarks/bench_dispatch_d256_kernel.py 2>&1
```

Parse the **last line**:
```
SPEEDUP_RATIO: X.XXXXXX
```

Maximize it. Baseline ≈ 0.97–0.99x (record on first run).

---

## Build command (after each .cpp modification)

```bash
cd ~/code/mlx-mfa-v2
CMAKE_ARGS="-DPython_EXECUTABLE=.venv/bin/python" \
  .venv/bin/pip install --no-build-isolation -e . 2>&1 | tail -5
```

Build takes < 1 min. If it exits non-zero → the change broke compilation → git reset immediately.

---

## Guard command

```bash
cd ~/code/mlx-mfa-v2
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -5
```

Tests must pass before any commit.

---

## Loop protocol

### Phase 0 — Setup (once)

```bash
cd ~/code/mlx-mfa-v2
git checkout -b autoresearch/d256-kernel-$(date +%Y%m%d)
```

Run bench once to record baseline. Initialize `results.tsv` (untracked, do NOT commit):
```
iteration	hyp	bq	bk	wm	ratio	keep	notes
0	baseline	32	32	4	<RECORD>	Y	unmodified code
```

### Phase 1 — Each iteration

```
1. Read git log + results.tsv (understand tried configs and pattern)
2. Form ONE hypothesis (a specific BQ/BK/WM/padding/unroll change)
3. Calculate TGP: (BQ + BK) * 136 * 2 bytes. Reject if > 32KB before editing.
4. Edit select_steel_v2_dsplit_block_config() in csrc/mfa_steel_fwd_v2.cpp
5. Build: CMAKE_ARGS="-DPython_EXECUTABLE=.venv/bin/python" .venv/bin/pip install --no-build-isolation -e .
   - If build fails → git checkout csrc/mfa_steel_fwd_v2.cpp → record as build_fail in tsv
6. Run bench → parse SPEEDUP_RATIO
7. Run guard → check tests pass
8. Decision:
   - ratio improved (>=0.5% relative gain) AND tests pass → git commit "experiment: BQ=X BK=Y WM=Z ratio=X.XXx"
   - ratio worse OR tests fail → git checkout csrc/mfa_steel_fwd_v2.cpp → do NOT commit
9. Record in results.tsv: iteration, hyp, bq, bk, wm, ratio, keep (Y/N), notes
10. Goto 1
```

### Suggested exploration order (first 10 iterations)

```
iter 1:  BQ=32 BK=64  WM=4  — double K-tile width (already has env var, but hardcode for clean test)
iter 2:  BQ=16 BK=64  WM=2  — smaller Q, wider K, reduced register pressure
iter 3:  BQ=16 BK=32  WM=2  — smaller Q only, baseline K
iter 4:  BQ=32 BK=48  WM=4  — intermediate BK (if 64 regresses)
iter 5:  BQ=32 BK=32  WM=4 + enable_unroll=false — disable unroll on M1
iter 6:  BQ=32 BK=64  WM=4 + enable_unroll=false — BK=64 without unroll
iter 7:  BQ=32 BK=32  WM=4 + pad=0  — remove padding (try MFA_NO_PADDING=1 first)
iter 8:  BQ=32 BK=64  WM=4 + pad=0  — BK=64 without padding overhead
iter 9:  BQ=48 BK=32  WM=4  — wider Q (BQ must be divisible by WM*8=32 → 48 not valid if BQ/32<1 → try BQ=64 BK=32 instead, check TGP: 96*136*2=26112 ✓)
iter 10: BQ=64 BK=32  WM=8  — max Q parallelism, WM must equal BQ/8=8 → TGP_SIZE=256
```

TGP quick reference:
- BQ=32 BK=32: 17,408B ✓
- BQ=32 BK=64: 26,112B ✓
- BQ=16 BK=64: 21,760B ✓
- BQ=64 BK=32: 26,112B ✓
- BQ=64 BK=64: 34,816B ✗ EXCEEDS

### Stuck recovery (5 consecutive discards)

- Re-read bench output: which N shows the most variance? That's where the gain is hiding.
- Try combinations of best near-misses (e.g., BK=48 + unroll=false)
- Try reducing `BD_HALF`: changing `const int BD_HALF = 128` to 64 creates D_SPLITS=4 for D=256
  (more passes but each is D=64-wide and fits better in registers). This is a more radical change.
- Try `WN=2` (currently hardcoded to 1) — changes the GEMM shape

---

## NEVER STOP

Once the loop begins, DO NOT pause to ask. DO NOT check in mid-loop.
The loop runs until manually interrupted. If stuck, think harder and try more radical changes.

---

## What success looks like

- geomean > 1.02x stably → win, document in devnotes/d256-design-track/
- geomean 0.99–1.02x → marginal, keep searching
- geomean < baseline across 15 iterations → document as ceiling reached, current BQ=32/BK=32 is optimal for M1 Max D=256
