# Sprint III-12 — TQ Decode Re-Bench + Distributional README (pre-v2.55.0)

**Date:** 2026-06-16
**Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `564040d` (clean), macOS 26.6 (25G5028f), Apple M5 Max 128GB, mlx 0.31.2.

## HEADLINE FINDING: the TQ paged-decode "6–14× faster" claim is INVERTED.

### R.1 — harness fix
`bench_turboquant_full.py::_build_tq_pool` declared `packed_D = D // 2` (a 4-bit assumption); the
bench uses `bits=3`, so `pack_k_for_metal` emits `D*3/8 = 48` bytes, which couldn't broadcast into
the `(chunk,H_kv,64)` pool slot → the III-11 crash. Fixed to `packed_D = (D*bits+7)//8` (and the
same fix in `_kv_memory_mb`'s memory estimate). The timed lambdas are **attend-only** (pool build is
outside them; warmup=3, iters=10, median) — so the numbers below are kernel latency, not build cost.

### R.2 — TQ decode re-measured on 26.6 (full 9-config matrix, attend-only latency)
| Config | fp16 ms | TQ-fused (P3) ms | **P3/fp16** | cos | fp16 MB | KV-TQ MB |
|---|---|---|---|---|---|---|
| Llama-8B 1seq 2K | 0.73 | 3.25 | **4.5×** | 0.965 | 8.0 | 1.6 |
| Llama-8B 1seq 8K | 0.53 | 11.34 | **21.6×** | 0.965 | 32 | 6.5 |
| Llama-8B 8seq 4K | 1.61 | 10.13 | **6.3×** | 0.966 | 128 | 26 |
| **Qwen-7B 1seq 8K** | 0.41 | 10.61 | **25.8×** | 0.965 | 16 | 3.25 |
| Qwen-7B 4seq 4K | 0.67 | 5.53 | **8.3×** | 0.965 | 32 | 6.5 |
| Mixed 8seq hetero | 1.74 | 6.50 | **3.7×** | 0.965 | 60 | 12.2 |

**`P3/fp16` is the ratio of TQ latency to fp16 latency — i.e. TQ paged decode is 1.6–26× SLOWER than
fp16 paged decode, NOT faster.** The README "6.0× (S=4K) to 14.4× (S=16K) faster" is this **slowdown
ratio mislabeled as a speedup**: RESULTS.md cited "Qwen-7B 1seq 8K = 14.4×" — the exact config that
now measures 25.8× P3/fp16 (the slowdown grew because Apple's fp16 paged decode got faster on 26.6).

**Run-to-run stability (R.2 requirement, MEASURED not assumed):** two independent runs agree within
~5% (Llama 1seq 8K 22.5× vs 21.6×; 2K 4.2× vs 4.5×). So TQ IS OS/run-stable (memory-bound, unlike
the ±30–40% compute-bound paths) — but the stable number is a **slowdown**.

### R.3 — promotion verdict
TQ paged decode's value is **memory** (4–5× KV reduction: 16–128 MB → 3.25–26 MB) at preserved
quality (cos 0.96–0.99), at a **latency cost** (decompression dominates single-token decode). It is
an **explicit opt-in API** (`flash_attention_paged_varlen_turboquant`) for memory-constrained
long-context / high-concurrency decode — NOT an auto-routed latency optimization, so there is no
harmful auto-promotion. **The "faster" framing is wrong; the memory-feature framing is correct.**

## Disposition
This is a pre-existing documentation error (not introduced by v2.55.0), but it is the repo's flagship
claim and must be corrected before publishing. README updated to the honest story: TQ trades decode
latency for a ~4–5× KV-memory reduction at ~0.96 cosine — a memory feature, not a speedup.
