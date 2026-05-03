# V6 NAX Roofline Analysis — M5 Max

**Hardware**: Apple M5 Max
- NAX peak compute: ~70 TFLOPS (FP16) — *estimated based on M5 family specs*
- Unified memory bandwidth: 614 GB/s
- Ridge point: 70 TFLOPS / 614 GB/s ≈ **114 FLOPS/byte**

A workload below 114 FLOPS/byte is **bandwidth-bound** (limited by memory). Above 114 it's **compute-bound** (limited by NAX TFLOPS).

## Per-shape arithmetic intensity

For attention `softmax(Q@K^T)@V`:
- **FLOPS**: `4 * B * H * N_q * N_kv * D` (Q@K^T + P@V, each `2BHN_qN_kvD`)
- **BYTES**: `2 * B * H * D * (2*N_q + 2*N_kv)` (read Q,K,V, write O — FP16 = 2 bytes)
- **AI** = FLOPS / BYTES = `2 * D * N_q * N_kv / (D * (2*N_q + 2*N_kv))` = `N_q*N_kv / (N_q + N_kv)`

For self-attention (N_q == N_kv == N): **AI = N/2**

| Shape | N_q | N_kv | D | H | AI (flops/byte) | Class |
|-------|----:|-----:|--:|--:|----------------:|-------|
| FlashVSR-dense |   4096 |   4096 |  64 | 10 | 2048 | **compute-bound** (AI >> 114) |
| SeedVR2-small  |  26730 |  26730 | 128 | 20 | 13365 | **compute-bound** |
| CogVideoX      |  70200 |  70200 | 128 | 30 | 35100 | **compute-bound** |
| SeedVR2-large  | 111375 | 111375 | 128 | 20 | 55687 | **compute-bound** |
| LTX2-cross     |   2048 |  14000 |  64 |  8 | 1786 | **compute-bound** |

**All production shapes are compute-bound on M5 Max.** This means kernel performance is limited by NAX TFLOPS utilization, not memory bandwidth. A perfect kernel would saturate the NAX units.

## Theoretical kernel time

For each shape, theoretical lower bound = `FLOPS / 70e12 * 1000` ms.

| Shape | FLOPS (G) | Theoretical (ms @ 70 TFLOPS) | Theoretical @ 50% util |
|-------|----------:|-----------------------------:|----------------------:|
| FlashVSR-dense | 4.29e9 / 1000 = 4.29 | **0.061** | 0.123 |
| SeedVR2-small | 4 * 1 * 20 * 26730 * 26730 * 128 = 7.31e12 → 7.31 | **104.4** | 208.7 |
| CogVideoX | 4 * 1 * 30 * 70200 * 70200 * 128 = 7.57e13 → 75.7 | **1081.4** | 2162.9 |
| SeedVR2-large | 4 * 1 * 20 * 111375 * 111375 * 128 = 1.27e14 → 127.0 | **1814.4** | 3628.7 |
| LTX2-cross | 4 * 1 * 8 * 2048 * 14000 * 64 = 5.87e10 → 0.0587 | **0.84** | 1.68 |

## Efficiency vs measured (Phase 3B V6 tuned numbers)

| Shape | Theoretical (70 TFLOPS) | V6 tuned | SDPA | V6 efficiency | SDPA efficiency |
|-------|------------------------:|---------:|-----:|--------------:|----------------:|
| FlashVSR-dense | 0.06 ms | 1.48 ms | 0.91 ms | **4.1%** | **6.7%** |
| SeedVR2-small | 104.4 ms | 231.27 ms | 205.63 ms | **45.1%** | **50.8%** |
| CogVideoX | 1081.4 ms | 2870.48 ms | 2507.00 ms | **37.7%** | **43.1%** |
| SeedVR2-large | 1814.4 ms | 4659.28 ms | 4493.96 ms | **38.9%** | **40.4%** |
| LTX2-cross | 0.84 ms | — | 1.31 ms | — | **64.1%** |

## Findings

1. **None of the kernels reach > 50% NAX efficiency** even with Apple's tuning. SDPA peaks at 50.8% on SeedVR2-small. Real-world attention (with memory traffic, online softmax, etc.) doesn't hit theoretical peak.

2. **The V6/SDPA gap is the efficiency gap**: SDPA is 5-7 percentage points more efficient than V6 tuned on most shapes. On SeedVR2-large the gap is only **1.5 percentage points** (38.9% vs 40.4%) — V6 is essentially at parity.

3. **FlashVSR-dense efficiency is brutal** — 4-7% for both V6 and SDPA. Small workloads can't amortize fixed overhead (kernel launch, function-constant specialization, etc.). No tile config can fix this — it's a hardware floor.

4. **Headroom estimate**: assuming Apple's NAX SDPA achieves ~50% of theoretical peak (well-tuned), the ABSOLUTE ceiling on M5 Max is ~2× our current SDPA times. V6 tuned at ~38-45% efficiency has ~5-15% headroom to match SDPA, plus another ~10-20% to approach Apple's tuning.

5. **Conclusion**: V6 NAX is in the **compute-bound regime** for all production shapes. Further gains come from:
   - Better tile selection (Axe 1 extended sweep)
   - Reducing fixed overhead (Axe 9, mostly for small shapes)
   - Apple-level kernel hand-tuning (likely the irreducible 5-7 percentage points)
