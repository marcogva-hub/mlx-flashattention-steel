# Sprint B coop-rewrite — §4-strict 3-session results

**Methodology**: 3 subprocess-isolated sessions, §4 cooldowns (180s initial, 60s inter-shape, 90s inter-round). A/B/A pattern V2 → V1 → V2, 5 runs/direction.
**Hardware**: M5 Max 128GB, macOS 26.5.

## Verdict: **SHIP_OPT_IN**

> v2.35.0 opt-in release; MFA_LCSA_KERNEL_VERSION=v2 default-off

- Production shape wins (V2/V1 ≥ 1.2× AND range < 10%): 2/7
- Density 0.20 SDPA/V2 ratio: 3.74×

## Production shapes (cross-session medians)

| Shape | density | V2 ms | V1 ms | SDPA ms | V1/V2 | SDPA/V2 | range% | drift% | flag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| lcsa_small_seq4k | 0.239 | 1.14 | 38.62 | 2.56 | 33.93× | 2.24× | 7.3% | 37.4% | CONFIDENT |
| lcsa_small_seq4k_sparse | 0.067 | 0.97 | 11.29 | 2.57 | 11.44× | 2.64× | 5.4% | 61.6% | CONFIDENT |
| lcsa_mid_seq8k | 0.119 | 1.52 | 51.52 | 6.45 | 35.79× | 4.35× | 12.1% | 36.1% | BOUNDARY |
| lcsa_mid_seq8k_sparse | 0.030 | 1.10 | 13.48 | 6.46 | 12.23× | 5.85× | 35.0% | 65.0% | HIGH |
| lcsa_large_seq16k | 0.120 | 2.06 | 103.91 | 12.73 | 50.46× | 6.18× | 20.3% | 39.5% | HIGH |
| lcsa_large_seq16k_sparse | 0.030 | 1.10 | 27.15 | 12.83 | 24.63× | 11.57× | 18.6% | 38.7% | BOUNDARY |
| lcsa_mid_seq8k_very_sparse | 0.011 | 0.63 | 5.42 | 6.45 | 8.54× | 10.29× | 32.6% | 60.8% | HIGH |

## Density sweep — lcsa_mid_seq8k

| density | V2 ms | V1 ms | SDPA ms | V1/V2 | SDPA/V2 | range% |
|---:|---:|---:|---:|---:|---:|---:|
| 0.011 | 0.71 | 5.15 | 6.49 | 7.28× | 9.24× | 11.4% |
| 0.030 | 0.82 | 13.19 | 6.48 | 16.01× | 7.86× | 44.3% |
| 0.049 | 0.85 | 21.39 | 6.46 | 24.95× | 7.54× | 13.6% |
| 0.102 | 1.18 | 43.64 | 6.45 | 37.13× | 5.49× | 3.6% |
| 0.199 | 1.72 | 85.48 | 6.45 | 49.27× | 3.74× | 26.0% |
| 0.500 | 3.32 | 211.06 | 6.45 | 63.59× | 1.95× | 1.0% |

## Per-shape per-session samples

### lcsa_small_seq4k

| Session | V2 ms | V1 ms | SDPA ms | V1/V2 | SDPA/V2 | drift% |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 1.14 | 38.73 | 2.56 | 33.93× | 2.24× | 37.4% |
| C2 | 1.16 | 38.62 | 2.58 | 33.31× | 2.22× | 36.3% |
| C3 | 1.07 | 38.46 | 2.56 | 35.90× | 2.39× | 32.7% |

### lcsa_small_seq4k_sparse

| Session | V2 ms | V1 ms | SDPA ms | V1/V2 | SDPA/V2 | drift% |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 0.96 | 11.29 | 2.54 | 11.73× | 2.64× | 61.6% |
| C2 | 1.03 | 11.35 | 2.58 | 11.02× | 2.51× | 61.1% |
| C3 | 0.97 | 11.11 | 2.57 | 11.44× | 2.65× | 57.8% |

### lcsa_mid_seq8k

| Session | V2 ms | V1 ms | SDPA ms | V1/V2 | SDPA/V2 | drift% |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 1.52 | 54.31 | 6.61 | 35.79× | 4.35× | 32.6% |
| C2 | 1.60 | 51.52 | 6.45 | 32.15× | 4.03× | 36.1% |
| C3 | 1.42 | 51.20 | 6.45 | 36.17× | 4.55× | 33.2% |

### lcsa_mid_seq8k_sparse

| Session | V2 ms | V1 ms | SDPA ms | V1/V2 | SDPA/V2 | drift% |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 1.10 | 13.48 | 6.45 | 12.23× | 5.85× | 58.6% |
| C2 | 0.85 | 13.57 | 6.46 | 16.03× | 7.64× | 33.6% |
| C3 | 1.16 | 13.19 | 6.47 | 11.40× | 5.59× | 65.0% |

### lcsa_large_seq16k

| Session | V2 ms | V1 ms | SDPA ms | V1/V2 | SDPA/V2 | drift% |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 2.06 | 103.91 | 12.73 | 50.46× | 6.18× | 0.7% |
| C2 | 2.55 | 103.23 | 12.72 | 40.53× | 4.99× | 39.5% |
| C3 | 2.04 | 103.97 | 12.74 | 50.99× | 6.25× | 3.1% |

### lcsa_large_seq16k_sparse

| Session | V2 ms | V1 ms | SDPA ms | V1/V2 | SDPA/V2 | drift% |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 1.10 | 27.15 | 12.75 | 24.63× | 11.57× | 34.1% |
| C2 | 1.24 | 27.00 | 12.85 | 21.78× | 10.36× | 38.7% |
| C3 | 1.03 | 27.55 | 12.83 | 26.87× | 12.51× | 26.2% |

### lcsa_mid_seq8k_very_sparse

| Session | V2 ms | V1 ms | SDPA ms | V1/V2 | SDPA/V2 | drift% |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 0.91 | 5.57 | 6.45 | 6.11× | 7.08× | 60.8% |
| C2 | 0.62 | 5.42 | 6.44 | 8.78× | 10.44× | 37.9% |
| C3 | 0.63 | 5.37 | 6.46 | 8.54× | 10.29× | 24.2% |

## Session conditions (3 sessions)

### C1
- **timestamp_utc**: `2026-05-12T09:39:34.785412+00:00`
- **platform**: `macOS-26.5-arm64-arm-64bit`
- **sw_vers**: `ProductName:		macOS
ProductVersion:		26.5
BuildVersion:		25F5068a`
- **uptime**: `10:39  up 5 days, 22:36, 4 users, load averages: 1.85 3.61 3.78`
- **uname**: `Darwin MBP-de-Marco.lan 25.5.0 Darwin Kernel Version 25.5.0: Thu Apr 23 21:23:38 PDT 2026; root:xnu-12377.121.5~4/RELEASE_ARM64_T6050 arm64`
- **boottime**: `{ sec = 1778065444, usec = 34302 } Wed May  6 12:04:04 2026`
- **mlx_version**: `0.31.2`
- **mlx_mfa_version**: `2.34.0`

### C2
- **timestamp_utc**: `2026-05-12T09:55:11.016285+00:00`
- **platform**: `macOS-26.5-arm64-arm-64bit`
- **sw_vers**: `ProductName:		macOS
ProductVersion:		26.5
BuildVersion:		25F5068a`
- **uptime**: `10:55  up 5 days, 22:51, 4 users, load averages: 1.65 1.82 2.40`
- **uname**: `Darwin MBP-de-Marco.lan 25.5.0 Darwin Kernel Version 25.5.0: Thu Apr 23 21:23:38 PDT 2026; root:xnu-12377.121.5~4/RELEASE_ARM64_T6050 arm64`
- **boottime**: `{ sec = 1778065444, usec = 34302 } Wed May  6 12:04:04 2026`
- **mlx_version**: `0.31.2`
- **mlx_mfa_version**: `2.35.0`

### C3
- **timestamp_utc**: `2026-05-12T10:10:47.195820+00:00`
- **platform**: `macOS-26.5-arm64-arm-64bit`
- **sw_vers**: `ProductName:		macOS
ProductVersion:		26.5
BuildVersion:		25F5068a`
- **uptime**: `11:10  up 5 days, 23:07, 4 users, load averages: 1.73 1.85 1.97`
- **uname**: `Darwin MBP-de-Marco.lan 25.5.0 Darwin Kernel Version 25.5.0: Thu Apr 23 21:23:38 PDT 2026; root:xnu-12377.121.5~4/RELEASE_ARM64_T6050 arm64`
- **boottime**: `{ sec = 1778065444, usec = 34302 } Wed May  6 12:04:04 2026`
- **mlx_version**: `0.31.2`
- **mlx_mfa_version**: `2.35.0`
