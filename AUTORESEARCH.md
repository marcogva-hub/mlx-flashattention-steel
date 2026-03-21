# AUTORESEARCH — mlx-mfa D=512 VAE Dispatch Optimization

## Objective

Maximize `SPEEDUP_RATIO` from `benchmarks/bench_dispatch_d512_vae.py`.

This ratio is the **geometric mean speedup** of the current auto-dispatch policy
vs baseline SDPA across 6 D=512 VAE-path configs (N=64–512, B=1, H=8, f16).

- **Baseline**: all D=512 configs route to SDPA → ratio ≈ 1.000
- **Improvement**: any ratio > 1.000 means you found a regime where MFA wins
- **Regression**: ratio < baseline means you made things worse

**Prior context you must internalize before the first iteration:**
- D=512 decision pass (2026-03-12, N=1024–8192): 0/32 wins, best 0.813x (B=1 H=1)
- The VAE-path sequences N=64–512 have **never been benchmarked** — this is new territory
- `_D512_CONSERVATIVE_MIN_N = 999_999` is intentionally conservative based on N>=1024 data
- Under-occupied grids (B=1, H=1) showed the best D=512 MFA ratios in the prior pass
- The env var `MFA_FORCE_D512_PATH=1|mfa` forces MFA for any D=512 call (for testing)

---

## Scope — what you MAY modify

**ONE file only: `mlx_mfa/dispatch_policy.py`**

Specifically, your search space within that file:
1. `_D512_CONSERVATIVE_MIN_N` — lower from 999_999 to test a threshold
2. `_DEFAULT_THRESHOLDS[(512, True)]` and `[(512, False)]` — add N-based routing
3. The `_d512_min_n()` function body — add causal/profile-specific differentiation
4. Environment variable setdefault patterns (following style of `_load_calibrated_kernel_config`)
5. The split-K routing in `should_use_splitk()` for D=512 (currently falls through to C++ heuristic)

**Do NOT modify anything else.**
Specifically: NEVER touch `.cpp`, `.hpp`, `.metal`, `.metallib`, `tests/`, `benchmarks/`
(except reading them), `pyproject.toml`, `CMakeLists.txt`, or any file other than
`mlx_mfa/dispatch_policy.py`.

---

## Verification command (metric)

```bash
cd ~/code/mlx-mfa-v2
.venv/bin/python benchmarks/bench_dispatch_d512_vae.py 2>&1
```

Parse the **last line** for the scalar:
```
SPEEDUP_RATIO: X.XXXXXX
```

Higher is better. Baseline ≈ 1.000.

Target runtime: < 2 minutes.

---

## Guard command (regression check)

```bash
cd ~/code/mlx-mfa-v2
.venv/bin/python -m pytest tests/ -q --timeout=120 2>&1 | tail -10
```

The guard MUST pass (no test failures, no errors) before any commit.
If the guard fails: **immediately git reset** and record as discard.

---

## Loop protocol

### Phase 0 — Setup (run once before loop begins)

```bash
cd ~/code/mlx-mfa-v2
git checkout -b autoresearch/d512-vae-$(date +%Y%m%d)
```

Initialize `results.tsv` (untracked):
```
iteration\thyp\tratio\tkeep\tnotes
0\tbaseline\t<run bench and record>\tY\tbaseline
```

### Phase 1 — Each iteration

```
1. Read git log + results.tsv → understand what was tried and what was learned
2. Form ONE hypothesis (a specific, testable change to dispatch_policy.py)
3. Make ONE atomic edit to dispatch_policy.py
4. Run bench → parse SPEEDUP_RATIO
5. Run guard → check all tests pass
6. Decision:
   - ratio improved (>=0.5% relative gain) AND tests pass → git commit "experiment: <brief hypothesis>"
   - ratio worse OR tests fail → git checkout mlx_mfa/dispatch_policy.py (do NOT commit)
7. Record row in results.tsv: iteration, hypothesis, ratio, keep (Y/N), notes
8. Goto 1
```

### Stuck recovery (after 5 consecutive discards)

If 5 consecutive iterations are discarded, do the following BEFORE forming the next hypothesis:
- Re-read bench output line-by-line — identify which individual configs (if any) are close to 1.0x
- Check if causal vs non-causal shows a consistent pattern in the ratios
- Re-read the D=512 occupancy column in bench output (tgs, occ values)
- Try the OPPOSITE of what recently failed
- Try: add `_d512_min_n()` logic that routes ONLY the specific (N, causal) pair that
  showed the highest individual ratio, leaving all others at SDPA
- Try: modify `should_use_splitk()` for D=512 to force split-K off (`MFA_FORCE_SPLITK=0`
  style env injection) to see if split-K overhead is hurting the short-N VAE configs
- Try: attempt a lower but still conservative threshold (e.g., `_D512_CONSERVATIVE_MIN_N = 256`)
  — the hypothesis being that very short N might behave differently than N>=1024

---

## NEVER STOP

Once the loop begins, **DO NOT pause to ask if you should continue**.
DO NOT ask "should I proceed?" or "is this a good stopping point?".
The human is away. Continue until manually interrupted.
The loop runs until the human stops it, period.

If you genuinely exhaust all reasonable hypotheses, start combining near-misses
or trying more radical restructuring of `_d512_min_n()` logic.

---

## What success looks like

- ratio > 1.020 on any stable iteration → meaningful win, continue to verify reproducibility
- ratio > 1.050 → strong signal, worth promoting to `devnotes/`
- ratio stays at 1.000±0.005 across 20 iterations → confirmed: no D=512 win exists
  at VAE-path lengths on M1 Max; document this finding

The negative finding (no win exists) is also scientifically valid output.
Document it clearly in `results.tsv` notes at end of session.
