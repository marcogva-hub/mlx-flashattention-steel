# Sprint B §4 re-bench — results

**Methodology**: §4-strict 3-session subprocess-isolated re-bench
of Sprint B Phase 1.5 ship envelope. Cooldowns 180/60/90s. A/B/A pattern, A = sparse_attention_dispatch (cache-HIT pattern), B = mx.fast.scaled_dot_product_attention(mask=bias). Ratio convention: `ratio_sdpa_over_nax > 1.0` → NAX faster.

**Variance summary**: 6 confident (<10%), 1 boundary (10-20%), 0 high (>20%) out of 7 shapes.
**Max |delta| vs single-session**: 6.9%
**Niche overturned**: no

**Decision** (per `lcsa-nax-rebench-decisions.md` §E and prompt §D.3 action matrix): **DOC_UPDATE_WITH_CAVEATS**

> 1 boundary shape(s) (10-20% range). Update ship-verdict with §4 numbers + boundary caveats; no tag.

## Per-shape results

| Shape | n_sess | median ratio | range % | A/B/A drift max | flag | Phase1.5 ref | Δ % |
|---|---:|---:|---:|---:|:--:|---:|---:|
| lcsa_small_seq4k | 3 | 0.99× | 0.1% | 1.0% | CONFIDENT | 0.96× (d=0.1) | +3.0% |
| lcsa_small_seq4k_sparse | 3 | 0.99× | 3.2% | 6.8% | CONFIDENT | 0.96× (d=0.1) | +3.1% |
| lcsa_mid_seq8k | 3 | 1.00× | 1.7% | 1.0% | CONFIDENT | 0.98× (d=0.1) | +2.1% |
| lcsa_mid_seq8k_sparse | 3 | 1.01× | 0.7% | 2.7% | CONFIDENT | 0.96× (d=0.03) | +4.8% |
| lcsa_large_seq16k | 3 | 0.99× | 1.8% | 0.9% | CONFIDENT | 0.95× (d=0.1) | +4.3% |
| lcsa_large_seq16k_sparse | 3 | 1.00× | 0.4% | 0.7% | CONFIDENT | 1.00× (d=0.03) | -0.4% |
| lcsa_mid_seq8k_very_sparse | 3 | 2.28× | 10.0% | 21.0% | BOUNDARY | 2.45× (d=0.01) | -6.9% |

## Per-session samples (full data)

### lcsa_small_seq4k

| Session | ratio | A/B/A drift |
|---|---:|---:|
| S1 | 0.988× | 1.0% |
| S2 | 0.988× | 0.4% |
| S3 | 0.989× | 0.5% |

### lcsa_small_seq4k_sparse

| Session | ratio | A/B/A drift |
|---|---:|---:|
| S1 | 0.981× | 6.8% |
| S2 | 0.990× | 2.8% |
| S3 | 1.013× | 5.9% |

### lcsa_mid_seq8k

| Session | ratio | A/B/A drift |
|---|---:|---:|
| S1 | 1.001× | 0.7% |
| S2 | 1.012× | 1.0% |
| S3 | 0.995× | 0.4% |

### lcsa_mid_seq8k_sparse

| Session | ratio | A/B/A drift |
|---|---:|---:|
| S1 | 1.004× | 0.8% |
| S2 | 1.011× | 0.0% |
| S3 | 1.006× | 2.7% |

### lcsa_large_seq16k

| Session | ratio | A/B/A drift |
|---|---:|---:|
| S1 | 0.988× | 0.1% |
| S2 | 0.991× | 0.9% |
| S3 | 1.006× | 0.8% |

### lcsa_large_seq16k_sparse

| Session | ratio | A/B/A drift |
|---|---:|---:|
| S1 | 1.001× | 0.4% |
| S2 | 0.996× | 0.1% |
| S3 | 0.996× | 0.7% |

### lcsa_mid_seq8k_very_sparse

| Session | ratio | A/B/A drift |
|---|---:|---:|
| S1 | 2.059× | 21.0% |
| S2 | 2.281× | 2.0% |
| S3 | 2.288× | 2.8% |

## Session conditions

### S1

- **timestamp_utc**: `2026-05-12T08:23:43.118101+00:00`
- **platform**: `macOS-26.5-arm64-arm-64bit`
- **sw_vers**: `ProductName:		macOS
ProductVersion:		26.5
BuildVersion:		25F5068a`
- **uptime**: `9:23  up 5 days, 21:20, 4 users, load averages: 2.62 2.85 2.43`
- **uname**: `Darwin MBP-de-Marco.lan 25.5.0 Darwin Kernel Version 25.5.0: Thu Apr 23 21:23:38 PDT 2026; root:xnu-12377.121.5~4/RELEASE_ARM64_T6050 arm64`
- **boottime**: `{ sec = 1778065444, usec = 34302 } Wed May  6 12:04:04 2026`
- **mlx_version**: `0.31.2`
- **mlx_mfa_version**: `2.34.0`

### S2

- **timestamp_utc**: `2026-05-12T08:32:44.521177+00:00`
- **platform**: `macOS-26.5-arm64-arm-64bit`
- **sw_vers**: `ProductName:		macOS
ProductVersion:		26.5
BuildVersion:		25F5068a`
- **uptime**: `9:32  up 5 days, 21:29, 4 users, load averages: 1.77 1.90 2.08`
- **uname**: `Darwin MBP-de-Marco.lan 25.5.0 Darwin Kernel Version 25.5.0: Thu Apr 23 21:23:38 PDT 2026; root:xnu-12377.121.5~4/RELEASE_ARM64_T6050 arm64`
- **boottime**: `{ sec = 1778065444, usec = 34302 } Wed May  6 12:04:04 2026`
- **mlx_version**: `0.31.2`
- **mlx_mfa_version**: `2.34.0`

### S3

- **timestamp_utc**: `2026-05-12T08:41:45.889646+00:00`
- **platform**: `macOS-26.5-arm64-arm-64bit`
- **sw_vers**: `ProductName:		macOS
ProductVersion:		26.5
BuildVersion:		25F5068a`
- **uptime**: `9:41  up 5 days, 21:38, 4 users, load averages: 2.88 2.23 2.12`
- **uname**: `Darwin MBP-de-Marco.lan 25.5.0 Darwin Kernel Version 25.5.0: Thu Apr 23 21:23:38 PDT 2026; root:xnu-12377.121.5~4/RELEASE_ARM64_T6050 arm64`
- **boottime**: `{ sec = 1778065444, usec = 34302 } Wed May  6 12:04:04 2026`
- **mlx_version**: `0.31.2`
- **mlx_mfa_version**: `2.34.0`
