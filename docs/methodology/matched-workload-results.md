# Methodology sprint - matched-workload-family validation results

**Methodology**: V2 -> SDPA+bias -> V2 A/B/A with matched-workload-family cooldowns (50ms warmup gap, small sparse_attention_nax dispatch with B=1 H=4 qL=kL=2048 D=64 BT=16 density=0.10).
**Hardware**: M5 Max 128GB, macOS 26.5.
**Hypothesis under test**: matched-workload warmup eliminates GPU power-state downclock variance WITHOUT competing for L2 cache against the measured kernel.

## Verdict: **REGRESSION**

> 3 previously-CONFIDENT shape(s) regressed under matched-workload protocol. Warmup may still be polluting cache. Debug; do not ship.

- HIGH->CONFIDENT resolved: **0/3**
- HIGH remaining:           3/3
- CONFIDENT shapes regressed: 3/4
- Total CONFIDENT: 1/7
- Total HIGH:      6/7
- Median ratio >=1.2x: 7/7

## Per-shape results (cross-session medians)

| Shape | density | V2 ms | SDPA ms | Ratio | V2 range % | Ratio range % | Drift max | Flag (new) | v2.36.0 range | Flag (v2.36) | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|:--:|---:|
| lcsa_small_seq4k | 0.239 | 1.477 | 2.704 | 1.77x | 118.0% | 28.3% | 31.4% | HIGH | 26.0% | HIGH | +2.3% |
| lcsa_small_seq4k_sparse | 0.067 | 1.517 | 2.714 | 1.74x | 18.0% | 36.5% | 24.3% | HIGH | 4.7% | CONFIDENT | +31.8% |
| lcsa_mid_seq8k | 0.119 | 2.302 | 6.621 | 2.89x | 49.6% | 40.3% | 37.2% | HIGH | 8.6% | CONFIDENT | +31.7% |
| lcsa_mid_seq8k_sparse | 0.030 | 1.758 | 6.727 | 3.83x | 57.4% | 41.8% | 105.0% | HIGH | 37.3% | HIGH | +4.5% |
| lcsa_large_seq16k | 0.120 | 2.571 | 13.061 | 5.08x | 34.8% | 31.5% | 27.0% | HIGH | 3.3% | CONFIDENT | +28.2% |
| lcsa_large_seq16k_sparse | 0.030 | 1.609 | 12.861 | 7.93x | 6.1% | 7.2% | 127.5% | CONFIDENT | 5.8% | CONFIDENT | +1.4% |
| lcsa_mid_seq8k_very_sparse | 0.011 | 1.181 | 6.570 | 5.56x | 42.4% | 28.2% | 102.6% | HIGH | 46.0% | HIGH | -17.8% |

## Axis-2 path-entered verification (warmup counter)

| Session | Warmup dispatches | Cooldown intervals | Avg fires per interval | Single dispatch us |
|---|---:|---:|---:|---:|
| M1 | 29962 | 21 | 1426.8 | 3198.5 |
| M2 | 30453 | 21 | 1450.1 | 3307.0 |
| M3 | 31318 | 21 | 1491.3 | 2445.3 |

Expected: >=1600 dispatches per 90s cooldown (50ms gap). Initial 180s cooldown adds ~3500 more. Per-shape inter-shape 60s cooldowns add ~1100 each. Total per session ~ initial 3500 + per-shape 6x1100 + per-round (6x2)x1700 ~= 30k dispatches per session.

## Per-session samples

### lcsa_small_seq4k

| Session | V2 ms | SDPA ms | Ratio | Drift % |
|---|---:|---:|---:|---:|
| M1 | 1.383 | 2.704 | 1.95x | 1.3% |
| M2 | 3.126 | 4.548 | 1.46x | 31.4% |
| M3 | 1.477 | 2.610 | 1.77x | 12.6% |

### lcsa_small_seq4k_sparse

| Session | V2 ms | SDPA ms | Ratio | Drift % |
|---|---:|---:|---:|---:|
| M1 | 1.517 | 2.644 | 1.74x | 8.2% |
| M2 | 1.643 | 2.714 | 1.65x | 1.5% |
| M3 | 1.371 | 3.135 | 2.29x | 24.3% |

### lcsa_mid_seq8k

| Session | V2 ms | SDPA ms | Ratio | Drift % |
|---|---:|---:|---:|---:|
| M1 | 3.144 | 6.621 | 2.11x | 6.4% |
| M2 | 2.302 | 6.657 | 2.89x | 37.2% |
| M3 | 2.002 | 6.552 | 3.27x | 30.9% |

### lcsa_mid_seq8k_sparse

| Session | V2 ms | SDPA ms | Ratio | Drift % |
|---|---:|---:|---:|---:|
| M1 | 1.758 | 6.727 | 3.83x | 44.3% |
| M2 | 2.496 | 6.856 | 2.75x | 10.2% |
| M3 | 1.488 | 6.463 | 4.35x | 105.0% |

### lcsa_large_seq16k

| Session | V2 ms | SDPA ms | Ratio | Drift % |
|---|---:|---:|---:|---:|
| M1 | 3.098 | 13.200 | 4.26x | 27.0% |
| M2 | 2.571 | 13.061 | 5.08x | 17.4% |
| M3 | 2.204 | 12.922 | 5.86x | 11.9% |

### lcsa_large_seq16k_sparse

| Session | V2 ms | SDPA ms | Ratio | Drift % |
|---|---:|---:|---:|---:|
| M1 | 1.608 | 13.037 | 8.11x | 127.5% |
| M2 | 1.707 | 12.861 | 7.54x | 9.9% |
| M3 | 1.609 | 12.753 | 7.93x | 6.6% |

### lcsa_mid_seq8k_very_sparse

| Session | V2 ms | SDPA ms | Ratio | Drift % |
|---|---:|---:|---:|---:|
| M1 | 1.669 | 6.700 | 4.01x | 102.6% |
| M2 | 1.181 | 6.570 | 5.56x | 87.1% |
| M3 | 1.169 | 6.528 | 5.59x | 81.6% |

## Session conditions

### M1
- **timestamp_utc**: `2026-05-12T20:32:00.305406+00:00`
- **platform**: `macOS-26.5-arm64-arm-64bit`
- **mfa_lcsa_kernel_version_env**: `v2`
- **sw_vers**: `ProductName:		macOS
ProductVersion:		26.5
BuildVersion:		25F5068a`
- **uptime**: `21:32  up 6 days,  9:28, 4 users, load averages: 2.81 3.27 3.01`
- **uname**: `Darwin MBP-de-Marco.lan 25.5.0 Darwin Kernel Version 25.5.0: Thu Apr 23 21:23:38 PDT 2026; root:xnu-12377.121.5~4/RELEASE_ARM64_T6050 arm64`
- **boottime**: `{ sec = 1778065444, usec = 147247 } Wed May  6 12:04:04 2026`
- **mlx_version**: `0.31.2`
- **mlx_mfa_version**: `2.36.0`
- **warmup_config**:
  - B: `1`
  - Hq: `4`
  - Hk: `4`
  - qL: `2048`
  - kL: `2048`
  - D: `64`
  - density: `0.1`
  - BT: `16`
  - seed: `42424242`

### M2
- **timestamp_utc**: `2026-05-12T21:03:02.347809+00:00`
- **platform**: `macOS-26.5-arm64-arm-64bit`
- **mfa_lcsa_kernel_version_env**: `v2`
- **sw_vers**: `ProductName:		macOS
ProductVersion:		26.5
BuildVersion:		25F5068a`
- **uptime**: `22:03  up 6 days,  9:59, 4 users, load averages: 3.31 4.04 3.41`
- **uname**: `Darwin MBP-de-Marco.lan 25.5.0 Darwin Kernel Version 25.5.0: Thu Apr 23 21:23:38 PDT 2026; root:xnu-12377.121.5~4/RELEASE_ARM64_T6050 arm64`
- **boottime**: `{ sec = 1778065444, usec = 147247 } Wed May  6 12:04:04 2026`
- **mlx_version**: `0.31.2`
- **mlx_mfa_version**: `2.36.0`
- **warmup_config**:
  - B: `1`
  - Hq: `4`
  - Hk: `4`
  - qL: `2048`
  - kL: `2048`
  - D: `64`
  - density: `0.1`
  - BT: `16`
  - seed: `42424242`

### M3
- **timestamp_utc**: `2026-05-12T21:34:04.254731+00:00`
- **platform**: `macOS-26.5-arm64-arm-64bit`
- **mfa_lcsa_kernel_version_env**: `v2`
- **sw_vers**: `ProductName:		macOS
ProductVersion:		26.5
BuildVersion:		25F5068a`
- **uptime**: `22:34  up 6 days, 10:30, 4 users, load averages: 2.65 2.59 2.71`
- **uname**: `Darwin MBP-de-Marco.lan 25.5.0 Darwin Kernel Version 25.5.0: Thu Apr 23 21:23:38 PDT 2026; root:xnu-12377.121.5~4/RELEASE_ARM64_T6050 arm64`
- **boottime**: `{ sec = 1778065444, usec = 147247 } Wed May  6 12:04:04 2026`
- **mlx_version**: `0.31.2`
- **mlx_mfa_version**: `2.36.0`
- **warmup_config**:
  - B: `1`
  - Hq: `4`
  - Hk: `4`
  - qL: `2048`
  - kL: `2048`
  - D: `64`
  - density: `0.1`
  - BT: `16`
  - seed: `42424242`
