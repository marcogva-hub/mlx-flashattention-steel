# DiT/UNet Dispatch Audit Report

**Date**: 2026-04-06
**Hardware**: Apple M1 Max (32 GPU cores, gen 13)
**Version**: v2.26.0
**MLX**: 0.31.1, macOS 26, Python 3.11, f16

## 1) Self-Attention (Non-Causal)

All 11 self-attention shapes correctly routed. MFA wins 1.17-1.65x for
N >= 4096 (above the 2048 threshold). SDPA wins for N <= 1024.

| Shape | D | N | MFA ms | SDPA ms | Ratio | Winner | Dispatch |
|-------|--:|------:|-------:|--------:|------:|--------|:--------:|
| UNet smallest | 128 | 64 | 0.42 | 0.35 | 0.84x | SDPA | SDPA |
| UNet tiny | 128 | 256 | 0.52 | 0.39 | 0.76x | SDPA | SDPA |
| UNet high-res D=64 | 64 | 1024 | 0.87 | 0.68 | 0.79x | SDPA | SDPA |
| UNet low-res D=64 | 64 | 4096 | 4.33 | 5.08 | 1.17x | MFA | MFA |
| UNet low-res D=128 | 128 | 4096 | 6.90 | 9.70 | 1.41x | MFA | MFA |
| UNet B=2 CFG D=64 | 64 | 4096 | 6.73 | 9.29 | 1.38x | MFA | MFA |
| UNet mid-res D=64 | 64 | 16384 | 48.5 | 72.1 | 1.49x | MFA | MFA |
| SeedVR2 DiT bs=29 | 128 | 26730 | 598 | 988 | 1.65x | MFA | MFA |
| CogVideoX | 128 | 70200 | 6135 | 10007 | 1.63x | MFA | MFA |
| Wan2.1 | 128 | 100000 | 16393 | 26839 | 1.64x | MFA | MFA |
| SeedVR2 DiT bs=497 | 128 | 111375 | 10185 | 16738 | 1.64x | MFA | MFA |

**Key finding**: MFA delivers consistent 1.63-1.65x speedup for large DiT
shapes (N >= 26K). The V2 kernel's tile-based approach scales well with
sequence length — the speedup ratio is stable, not increasing, because both
MFA and SDPA scale as O(N^2) but MFA has lower constant factor from tiled
register blocking.

## 2) Cross-Attention (N_q != N_kv)

Cross-attention reveals an asymmetric dispatch gap. The dispatch previously
only considered N_q (query length), not N_kv (key/value length).

| Shape | D | N_q | N_kv | MFA ms | SDPA ms | Ratio | Winner |
|-------|--:|------:|-----:|-------:|--------:|------:|--------|
| SD/SDXL D=64 | 64 | 4096 | 77 | 1.09 | 0.76 | 0.70x | SDPA |
| SD/SDXL D=128 | 128 | 4096 | 77 | 0.65 | 1.05 | 1.60x | MFA |
| DiT x CLIP-77 | 128 | 70200 | 77 | 21.0 | 14.8 | 0.70x | SDPA |
| CogVideoX text | 128 | 70200 | 226 | 48.0 | 37.7 | 0.79x | SDPA |
| Wan2.1 text | 128 | 100000 | 512 | 172 | 141 | 0.82x | SDPA |
| LTX-2 v->a | 64 | 14000 | 2000 | 37.6 | 30.0 | 0.80x | SDPA |
| LTX-2 a->v | 64 | 2000 | 14000 | 3.47 | 29.8 | 8.59x | MFA |

### Root cause analysis

**Small N_kv (77-512 tokens)**: MFA processes K in BK-sized tiles (BK=32 for
D=128). With N_kv=77, there are only ceil(77/32)=3 K-tiles per Q-tile.
The per-tile overhead (barrier syncs, TGP loads) is fixed, but with only 3
tiles, it's not amortized over enough compute to beat SDPA's fused matmul.

**Large N_kv, small N_q (LTX-2 audio->video)**: MFA's Q-centric design shines
here. Only ceil(2000/32)=63 Q-tiles launch, each iterating over 438 K-tiles.
SDPA materializes the full 2000x14000 attention matrix (56M elements), while
MFA keeps it in registers/TGP (32x32 tiles = 1024 elements at a time). Result:
8.59x speedup.

## 3) Dispatch Changes

### 3.1) New `kv_seq_len` parameter

Added `kv_seq_len: Optional[int]` to `should_use_mfa()`. When `None`
(self-attention), behavior is unchanged. When set (cross-attention), two
new rules apply:

### 3.2) Small-KV rule

```
if N_kv <= 512 and N_q > 8192: return SDPA
```

Prevents routing to MFA when there are too few K-tiles to amortize tile
overhead. Threshold N_kv=512 matches the observed crossover; N_q>8192
ensures small self-attention shapes (N_q=N_kv=4096) are unaffected.

### 3.3) Large-KV rule

```
if N_kv >= 4096 and N_q <= 4096: return MFA
```

Forces MFA for cross-attention where N_kv >> N_q. Flash attention processes
few Q-tiles iterating over many K-tiles — much more efficient than SDPA's
materialized attention matrix.

### 3.4) Files modified

- `mlx_mfa/dispatch_policy.py`: `should_use_mfa()` signature + cross-attn logic
- `mlx_mfa/attention.py`: passes `kv_seq_len=k.shape[2]` to dispatch, updated
  cache key to include KV length

## 4) Unresolved cases

| Shape | D | N_q | N_kv | Ratio | Note |
|-------|--:|-----:|-----:|------:|------|
| LTX-2 v->a | 64 | 14000 | 2000 | 0.80x | N_kv=2000 is above 512 threshold; falls through to standard self-attn-like dispatch which routes MFA. SDPA is 20% faster here. Fixing requires a more nuanced N_kv vs BK analysis. |
| SD/SDXL D=64 | 64 | 4096 | 77 | 0.70x | N_q=4096 is below 8192 threshold; dispatches to MFA via standard table. SDPA is 30% faster. Small absolute time (1ms) makes this low-impact. |

These are left as known suboptimalities. A future audit with per-D per-BK
thresholds could close them.

## 5) M5 Max Calibration Notes

The M3+ threshold table currently disables non-causal MFA entirely
(`(D, False): 999_999`). DiT models on M3+ will use SDPA. When M5 Max
hardware is available, re-run this audit:

1. Self-attention: verify that M5+ SDPA is still faster for non-causal
   (it might not be — M5 tensor API could change the balance).
2. Cross-attention small-KV rule: thresholds may differ on M5+ due to
   different L2 cache behavior.
3. Consider the LTX-2 v->a case (N_kv=2000): M5+ may tip this to MFA.

Raw benchmark data: `docs/audit_dit_dispatch_results.json`
