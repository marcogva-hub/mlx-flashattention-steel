# Canonical re-bench results (Sprint Option beta, v2.36.1)

**Methodology**: docs/methodology/canonical-protocol.md - 10 warmup + 100 continuous timed iters, V2 and SDPA back-to-back per shape, 3 subprocess-isolated sessions.
**Hardware**: M5 Max 128GB, macOS 26 (fan profile iStat performance).
**Sessions**: 3.

## Per-shape verdict

| Shape | density | qL*kL*D | V2 p50 ms | SDPA p50 ms | Ratio | V2 range % | Ratio range % | Verdict | V2 default | s4-strict ratio range % | s4-strict flag |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|---:|:--:|
| lcsa_small_seq4k | 0.239 | 2.15e+09 | 1.143 | 2.573 | 2.25x | 15.1% | 18.1% | BOUNDARY | YES | 26.0% | HIGH |
| lcsa_small_seq4k_sparse | 0.067 | 2.15e+09 | 0.695 | 2.548 | 3.66x | 7.4% | 5.1% | CONFIDENT | YES | 4.7% | CONFIDENT |
| lcsa_mid_seq8k | 0.119 | 8.59e+09 | 1.219 | 6.441 | 5.29x | 3.5% | 4.0% | CONFIDENT | YES | 8.6% | CONFIDENT |
| lcsa_mid_seq8k_sparse | 0.030 | 8.59e+09 | 0.612 | 6.443 | 10.53x | 2.9% | 1.8% | CONFIDENT | YES | 37.3% | HIGH |
| lcsa_large_seq16k | 0.120 | 3.44e+10 | 2.027 | 12.715 | 6.27x | 0.6% | 1.3% | CONFIDENT | YES | 3.3% | CONFIDENT |
| lcsa_large_seq16k_sparse | 0.030 | 3.44e+10 | 0.916 | 12.702 | 13.86x | 1.6% | 3.6% | CONFIDENT | YES | 5.8% | CONFIDENT |
| lcsa_mid_seq8k_very_sparse | 0.011 | 8.59e+09 | 0.517 | 6.435 | 12.59x | 6.5% | 6.6% | CONFIDENT | YES | 46.0% | HIGH |

## Threshold calibration for decide_auto_version()

- **No threshold**: see rationale.
- **Rationale**: All shapes CONFIDENT/BOUNDARY under canonical methodology - V2 ships unconditionally for the tested shape regime
- Eligible work range: [2.15e+09, 3.44e+10]
- Clean inflection: True

## Per-session samples

### lcsa_small_seq4k

| Session | V2 p50 ms | V2 p95 ms | V2 p99 ms | SDPA p50 ms | Ratio |
|---|---:|---:|---:|---:|---:|
| C1 | 0.970 | 1.134 | 1.191 | 2.578 | 2.66x |
| C2 | 1.143 | 1.337 | 1.426 | 2.573 | 2.25x |
| C3 | 1.143 | 1.319 | 1.375 | 2.573 | 2.25x |

### lcsa_small_seq4k_sparse

| Session | V2 p50 ms | V2 p95 ms | V2 p99 ms | SDPA p50 ms | Ratio |
|---|---:|---:|---:|---:|---:|
| C1 | 0.715 | 0.843 | 1.034 | 2.586 | 3.62x |
| C2 | 0.663 | 0.759 | 0.820 | 2.523 | 3.81x |
| C3 | 0.695 | 0.810 | 0.847 | 2.548 | 3.66x |

### lcsa_mid_seq8k

| Session | V2 p50 ms | V2 p95 ms | V2 p99 ms | SDPA p50 ms | Ratio |
|---|---:|---:|---:|---:|---:|
| C1 | 1.219 | 1.277 | 1.379 | 6.441 | 5.29x |
| C2 | 1.220 | 1.273 | 1.339 | 6.429 | 5.27x |
| C3 | 1.177 | 1.285 | 1.328 | 6.453 | 5.48x |

### lcsa_mid_seq8k_sparse

| Session | V2 p50 ms | V2 p95 ms | V2 p99 ms | SDPA p50 ms | Ratio |
|---|---:|---:|---:|---:|---:|
| C1 | 0.612 | 0.689 | 0.743 | 6.443 | 10.53x |
| C2 | 0.611 | 0.717 | 0.795 | 6.439 | 10.54x |
| C3 | 0.629 | 0.736 | 0.820 | 6.504 | 10.35x |

### lcsa_large_seq16k

| Session | V2 p50 ms | V2 p95 ms | V2 p99 ms | SDPA p50 ms | Ratio |
|---|---:|---:|---:|---:|---:|
| C1 | 2.031 | 2.105 | 2.204 | 12.696 | 6.25x |
| C2 | 2.027 | 2.121 | 2.233 | 12.715 | 6.27x |
| C3 | 2.019 | 2.098 | 2.131 | 12.786 | 6.33x |

### lcsa_large_seq16k_sparse

| Session | V2 p50 ms | V2 p95 ms | V2 p99 ms | SDPA p50 ms | Ratio |
|---|---:|---:|---:|---:|---:|
| C1 | 0.920 | 0.992 | 1.059 | 12.696 | 13.81x |
| C2 | 0.916 | 0.972 | 1.100 | 12.702 | 13.86x |
| C3 | 0.905 | 1.276 | 1.725 | 12.956 | 14.31x |

### lcsa_mid_seq8k_very_sparse

| Session | V2 p50 ms | V2 p95 ms | V2 p99 ms | SDPA p50 ms | Ratio |
|---|---:|---:|---:|---:|---:|
| C1 | 0.517 | 0.549 | 0.616 | 6.435 | 12.44x |
| C2 | 0.484 | 0.553 | 0.577 | 6.431 | 13.27x |
| C3 | 0.518 | 0.590 | 1.436 | 6.523 | 12.59x |

## Comparison vs section-4-strict (v2.36.0 baseline)

| Shape | section-4-strict range | canonical range | Direction |
|---|---:|---:|:--:|
| lcsa_small_seq4k | 26.0% | 18.1% | IMPROVED |
| lcsa_small_seq4k_sparse | 4.7% | 5.1% | WORSE |
| lcsa_mid_seq8k | 8.6% | 4.0% | IMPROVED |
| lcsa_mid_seq8k_sparse | 37.3% | 1.8% | IMPROVED |
| lcsa_large_seq16k | 3.3% | 1.3% | IMPROVED |
| lcsa_large_seq16k_sparse | 5.8% | 3.6% | IMPROVED |
| lcsa_mid_seq8k_very_sparse | 46.0% | 6.6% | IMPROVED |

## Session conditions

### C1
- **timestamp_utc**: `2026-05-12T23:24:50.856728+00:00`
- **platform**: `macOS-26.5-arm64-arm-64bit`
- **mfa_lcsa_kernel_version_env**: `v2`
- **sw_vers**: `ProductName:		macOS
ProductVersion:		26.5
BuildVersion:		25F5068a`
- **uptime**: `0:24  up 6 days, 12:21, 4 users, load averages: 1.35 1.95 2.34`
- **uname**: `Darwin MBP-de-Marco.lan 25.5.0 Darwin Kernel Version 25.5.0: Thu Apr 23 21:23:38 PDT 2026; root:xnu-12377.121.5~4/RELEASE_ARM64_T6050 arm64`
- **boottime**: `{ sec = 1778065444, usec = 147247 } Wed May  6 12:04:04 2026`
- **mlx_version**: `0.31.2`
- **mlx_mfa_version**: `2.36.0`
- **protocol**:
  - name: `canonical_warmup_continuous`
  - warmup_iterations: `10`
  - timed_iterations: `100`
  - inter_shape_settle_s: `5.0`
  - reference_doc: `docs/methodology/canonical-protocol.md`

### C2
- **timestamp_utc**: `2026-05-12T23:25:57.804827+00:00`
- **platform**: `macOS-26.5-arm64-arm-64bit`
- **mfa_lcsa_kernel_version_env**: `v2`
- **sw_vers**: `ProductName:		macOS
ProductVersion:		26.5
BuildVersion:		25F5068a`
- **uptime**: `0:25  up 6 days, 12:22, 4 users, load averages: 1.22 1.78 2.24`
- **uname**: `Darwin MBP-de-Marco.lan 25.5.0 Darwin Kernel Version 25.5.0: Thu Apr 23 21:23:38 PDT 2026; root:xnu-12377.121.5~4/RELEASE_ARM64_T6050 arm64`
- **boottime**: `{ sec = 1778065444, usec = 147247 } Wed May  6 12:04:04 2026`
- **mlx_version**: `0.31.2`
- **mlx_mfa_version**: `2.36.0`
- **protocol**:
  - name: `canonical_warmup_continuous`
  - warmup_iterations: `10`
  - timed_iterations: `100`
  - inter_shape_settle_s: `5.0`
  - reference_doc: `docs/methodology/canonical-protocol.md`

### C3
- **timestamp_utc**: `2026-05-12T23:27:04.778174+00:00`
- **platform**: `macOS-26.5-arm64-arm-64bit`
- **mfa_lcsa_kernel_version_env**: `v2`
- **sw_vers**: `ProductName:		macOS
ProductVersion:		26.5
BuildVersion:		25F5068a`
- **uptime**: `0:27  up 6 days, 12:23, 4 users, load averages: 1.03 1.62 2.15`
- **uname**: `Darwin MBP-de-Marco.lan 25.5.0 Darwin Kernel Version 25.5.0: Thu Apr 23 21:23:38 PDT 2026; root:xnu-12377.121.5~4/RELEASE_ARM64_T6050 arm64`
- **boottime**: `{ sec = 1778065444, usec = 147247 } Wed May  6 12:04:04 2026`
- **mlx_version**: `0.31.2`
- **mlx_mfa_version**: `2.36.0`
- **protocol**:
  - name: `canonical_warmup_continuous`
  - warmup_iterations: `10`
  - timed_iterations: `100`
  - inter_shape_settle_s: `5.0`
  - reference_doc: `docs/methodology/canonical-protocol.md`
