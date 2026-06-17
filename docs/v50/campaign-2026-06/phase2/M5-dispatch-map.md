# M5 Max Authoritative Dispatch Map (Sprint II-1, 2026-06-12)

Every routing decision below is backed by a fresh M5 Max measurement
(3-block median, warmup-trimmed; raw data /tmp/ii1_*.json archived in the
II-1 report tables).  Grid: D∈{64,128,256,512} × {fp16,bf16} ×
{causal,nc} × N∈{1024,4096,16384} × {MHA B1H8, GQA3 B2H24} forward;
D∈{64,128,256} × dtypes × causality × N∈{2048,8192} backward; decode
q_len=1 D∈{64,128,256} S∈{4096,16384}; odd-D {80,96,160}.

## Forward (dense): SDPA EVERYWHERE on M5 — 44/44 cells HOLD

Post Phase-I D=256-causal fix, NO dense forward cell routes to a custom
kernel on M5.  Auto tracks raw SDPA within noise at every cell (the 3
flagged cells re-probed to noise; D512-c-N4096 re-probe: 6.91 vs 6.90).
Forced-MFA columns (V2; V3/V4/V5 per Sprint C): 3-4× behind SDPA at
D≤128; 1.4-3.4× behind at D=256.  Custom forward kernels on M5 serve
ONLY: sparse/LCSA, GNA, sage, paged/varlen/TQ, flash-decode (N≤4), and
all M1-M4 hardware (whose branches are untouched and hardware-gated).

Consequence: BK micro-tuning (BK=32/16 @ D=128, D=256 BK=8) is MOOT for
M5 dense dispatch — those configs only execute on M1-M4 branches or
opt-in paths.  Recorded as not-applicable rather than re-tuned.

## Backward: II-0 promotion + HOLD elsewhere — 24/24 cells validated

| Cell family | Routing | Evidence |
|---|---|---|
| D=64 causal (fp16) qL≥2048 | **V6NAX NAX-direct (default-on, II-0)** | 2.19×/2.71× at N=2048/8192 |
| D=64 causal (bf16) qL≥2048 | **V6NAX (default-on)** | 1.33×/2.68× |
| D=64 non-causal | SDPA-vjp | 0.99-1.01× parity both dtypes |
| D=128 all | SDPA-vjp | 0.99-1.01× (V6NAX loses 0.46-0.58× per Phase I) |
| D=256 all | SDPA-vjp | 0.98-1.00× |

## Decode (q_len=1): SDPA HOLD — 6/6

sdpa_vector_2pass covers D≤256 decode optimally (0.24-1.31ms range);
auto at parity.  The 2026-03 "decode coverage gap" is CLOSED by Apple.

## Odd head dims (80/96/160): SDPA HOLD, diff=0.0

D=80 has a native NAX variant (metallib); 96/160 route well unfused.
No custom-kernel gap worth building (parity at all three).

## Map summary

**Zero INVERT cells remain.**  M5 dispatch truth: SDPA forward
everywhere dense; V6NAX backward D=64-causal; SDPA-vjp backward
elsewhere; custom kernels own the structured paths (sparse/paged/
TQ/GNA/sage/decode≤4) where Apple has no coverage (re-verified against
MLX 0.31.2 in Sprint C Track 4).
