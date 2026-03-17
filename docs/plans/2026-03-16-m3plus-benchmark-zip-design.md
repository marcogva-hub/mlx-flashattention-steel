# M3+ Portable Benchmark ZIP — Design

**Date**: 2026-03-16
**Version**: v2.10.0
**Audience**: Third-party tester with M4 Max hardware
**Objective**: Validate M3+ performance assumptions, determine development priorities

## Problem

All mlx-mfa benchmarks were run on M1 Max. The M3+ code paths (larger BK tiles,
V4 direct-device K reads, V5 D-blocked kernel) are designed but unvalidated on
actual M3+ hardware. We need a turnkey package a third party can run to answer:

1. Does V2 BK=64 (vs BK=32) deliver the expected speedup for D=128?
2. Does V4 (direct device K reads) benefit from M3+ L2 cache?
3. Does V5 (D-blocked, BK=128) overcome barrier overhead with M3+ occupancy?
4. Are split-K thresholds correctly calibrated for M4 Max core count?

## Design

### ZIP Structure

```
mlx-mfa-bench/
  run_benchmarks.sh            # Main entrypoint (POSIX-safe bash)
  collect_metadata.sh          # Hardware/software fingerprint
  bench_m3plus_hypotheses.py   # Hypothesis-focused runner
  benchmarks/
    bench_v2_final.py          # Canonical dense/window/splitk benchmark
    bench_attention.py         # Lightweight forward validation
  README.md                    # Instructions for tester
```

### run_benchmarks.sh

Bash script following defensive patterns (set -euo pipefail, trap cleanup):

1. **Preflight**: Verify macOS arm64, Python ≥3.10, cmake, clang
2. **Venv**: Create `.venv`, install `mlx-mfa` from PyPI with `--no-binary mlx-mfa`
3. **Metadata**: Run `collect_metadata.sh` → `results/metadata.txt`
4. **Canonical bench**: `bench_v2_final.py --section all --save` → `results/RESULTS.md`
5. **Hypothesis bench**: `bench_m3plus_hypotheses.py` → `results/M3PLUS_HYPOTHESES.md`
6. **Summary**: Print location of results/, estimated total time ~20 min

### bench_m3plus_hypotheses.py

~150 lines. Tests 4 hypotheses by toggling env vars and comparing timings:

| Hypothesis | Env Var | Comparison | Configs |
|-----------|---------|------------|---------|
| H1: V2 BK=64 vs BK=32 | `MFA_V2_FORCE_BK=32` vs default | D=128, N=2048/4096/8192, causal | B=2 H=8 f16 |
| H2: V4 direct K reads | `MFA_ENABLE_V4=1` vs V2 default | D=64/128, N=2048/4096/8192, causal | B=2 H=8 f16 |
| H3: V5 D-blocked | `MFA_ENABLE_V5=1` vs V2 default | D=64/128, N=2048/4096/8192, causal | B=2 H=8 f16 |
| H4: Split-K threshold | `MFA_FORCE_SPLITK=1` vs auto | D=64/128, B=1 H=2 (small grid), N=512/1024/2048 | f16 causal |

Output: Markdown table per hypothesis with median_ms, speedup_vs_sdpa, speedup_vs_v2.

### collect_metadata.sh

Captures: macOS version, chip (sysctl hw.optional), Metal GPU family,
Python version, mlx version, mlx-mfa version, cmake version, GPU core count.

### README.md

One-page: prerequisites (Xcode CLT), single command to run, what to send back,
expected runtime, troubleshooting (cmake not found, etc.).

## Non-goals

- No CI/CD integration (one-shot use)
- No automatic upload of results
- No Docker/Nix (unnecessary complexity)
- No backward pass benchmarks (forward-only for M3+ hypothesis testing)
- No V3 benchmarks (already proven slower than V2 due to occupancy)

## Success Criteria

Tester runs `./run_benchmarks.sh`, waits ~20 min, sends back `results/` folder.
We get definitive answers on all 4 hypotheses.
