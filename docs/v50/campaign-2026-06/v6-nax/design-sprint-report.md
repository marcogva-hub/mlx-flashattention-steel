# V6 NAX / dequant-in-GEMM — Architecture & Design Sprint (Phase 0 measured)

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `fd87f16` (post-v2.56.0), macOS 26.6 (25G5028f), Metal toolchain 32023.864,
Apple M5 Max 128GB, mlx 0.31.2. **No production kernel written** (design + throwaway premise probes only).

## Headline verdict: **NOT a green light to implement on 26.6.**

**Dense fp16/bf16 attention = HOSTILE** (the predicted "1.3–1.5× prefill" win does not exist — that
range is the gap by which the MPP-matmul2d V6 path *loses* to Apple SDPA). **The only numerically-
viable lever (INT8 quantized-KV) = SCOPED + deferred** — it competes with the already-shipped,
IV-D1/D2-optimized TQ-decode-via-SDPA path in the sync-floor-bound decode regime. A money-saving
Phase-0 outcome: **do not build the V6 NAX matmul2d attention kernel for dense — measured to lose to
SDPA before a line of it was written.**

> **Doc gap:** the named `v6_nax_reference_synthesis_prediction.md` does not exist in either repo.
> Worked from the actual `docs/v6-nax/` analyses (esp. `apple-sdpa-nax-analysis.md`, `sprint-3-3`),
> the Day-J data, and the prompt's stated prediction ranges as the targets to validate.

## Phase 0 — the five measured premises (M5/26.6)

### P0.1 — The missing denominator → **HOSTILE**
Effective attention throughput of the current paths (probe `/tmp` archived → `v6nax_p0.py`), apparent
TFLOPS = `4·B·H·N²·D / t` (causal upper bound; real ≈ ½ after the triangular halving):

| shape | Apple SDPA (auto, M5 default) | mlx-mfa flash (backend=mfa) |
|---|---|---|
| B2 H8 N2048 D128 | 0.73 ms → **47 TFLOPS** (91% of NAX peak) | 1.90 ms → 18 TFLOPS |
| B2 H8 N4096 D128 | 1.79 ms → **77 TFLOPS** (148%) | 6.07 ms → 23 TFLOPS |
| B2 H8 N8192 D128 | 6.23 ms → **88 TFLOPS** (170%) | 23.5 ms → 23 TFLOPS |
| B1 H32 N4096 D128 | 3.27 ms → 84 TFLOPS | 11.7 ms → 24 TFLOPS |
| B2 H8 N4096 D64 | 0.84 ms → 82 TFLOPS | 2.77 ms → 25 TFLOPS |

vs the Day-J NAX matmul2d peak **51.8 TFLOPS (fp16, M=128 sweet spot)**. Even halved for causal,
SDPA's effective useful throughput (~24–44 TFLOPS) is **at or above the NAX-matmul2d peak** — because
Apple SDPA already runs on the Neural Accelerators via the **raw `metal_simdgroup_matrix` NAX path**
(`steel_attention_nax.h`), fuses softmax, and **avoids the MPP scheduling overhead** that the
matmul2d/cooperative-tensor layer adds (`apple-sdpa-nax-analysis.md`). **There is no headroom for a
V6-NAX-via-matmul2d kernel to beat SDPA on dense fp16/bf16.** Corroborated by the recorded
`sprint-3-3` data: V6/SDPA was ~2× slower, improved only to ~1.3–1.5× slower.

### P0.2 — Tile tension → **confirms the penalty** (Day-J, M5/26.6)
Per-core GFLOPS: M=16 → 188, **M=32 (flash-native BQ) → 310**, M=64 sq → 610, **M=128 (sweet spot) →
~1295**. So the flash-native BQ=32 tile runs at **~24% of the M=128 sweet spot** (matches Day-J's
flag). Winning NAX throughput would require restructuring to BQ=128 (4× the flash tile → register-
pressure + softmax-granularity changes) — and even then it competes with SDPA, already at NAX-peak.

### P0.3 — Dispatch / MPP-overhead crossover → **no crossover for dense**
The MPP matmul2d layer carries per-op scheduling overhead the raw-simdgroup SDPA path avoids
(`apple-sdpa-nax-analysis.md`). P0.1 shows SDPA wins at **every** measured N (2048→8192) — there is
no N at which V6-MPP overtakes SDPA for dense attention. (The 0.3–0.5 ms MTL4 per-*kernel-launch*
overhead is one launch per attention call, same as SDPA — not the differentiator; the MPP per-op
overhead inside the kernel is.)

### P0.4 — 26.6-available primitives → **int8 matmul2d YES; fully-fused dequant-load UNVERIFIED (not required)**
- **int8 matmul2d (int8 → int32 accumulate) IS available on 26.6** — proven by the shipping
  `csrc/mpp_int8_bench.mm` (`matmul2d<desc>` + `get_*_cooperative_tensor<int8_t,int8_t,int32_t>()`,
  toolchain 32023.864). The dequant-by-scales step runs in the ALU post-int32-accumulate (26.6-fine;
  it's what P0.5 modeled).
- The **fully-fused dequant-inside-the-cooperative-tensor-load** (WWDC26 session 330) could not be
  verified from headers on this 26.6 system (likely Metal 4.1 / macOS 27 — Rigel). **Not required**:
  the int8-matmul2d + ALU-dequant-scales path is 26.6-buildable and accurate (P0.5). A 4.1 fully-fused
  variant is a **27-gated future** optimization, not the 26.6 design.

### P0.5 — INT8 attention accuracy → **GREEN**
INT8-quantized QK^T attention (per-row symmetric int8, int32 accumulate, dequant by row scales,
softmax, fp16 P@V) vs fp32 reference: **cosine 1.00000, max_rel_err 1e-4** at D=64/128, N=2048/4096.
The softmax normalization makes attention robust to QK^T quantization — the precision-sensitivity
concern does **not** block the INT8 lever at these shapes.

## Predicted-vs-measured log (the first checkpoint)

| Prediction (prompt/synthesis) | Measured premise | Verdict |
|---|---|---|
| Dense fp16 prefill **1.3–1.5× gain** | SDPA already at/above NAX-matmul2d peak; V6-MPP records 1.3–1.5× *slower* | **INVERTED → HOSTILE** |
| INT8 **1.2–1.9× gain** | INT8 QK^T accuracy green (cos 1.0); int8 matmul2d 26.6-available | viability GREEN, but **headroom unmeasured vs the existing TQ-decode-via-SDPA path** |
| Decode **~1.0×** | decode is sync-floor-bound (IV-0); IV-D1/D2 already optimized the TQ path | confirmed — no NAX decode win |

## Phase 1 — design implication (grounded in the measured premises)

The premises rewrite the design from "build a V6 NAX attention backend" to:

1. **Dense fp16/bf16: do NOT build V6 NAX.** Keep-all-paths is already satisfied — the M5 default
   routes dense → Apple SDPA (the NAX-peak path). No new path, no routing change. The existing V6 NAX
   primitive (`mfa_v6_nax_primitive.cpp`) stays as the backend=mfa/expert + research path (and is now
   int64-safe post-A3-1); it is not promoted.
2. **The only candidate worth a future design: INT8/TQ quantized-KV attention via int8 matmul2d** —
   the regime SDPA cannot serve directly (it needs dequantized inputs). int8 matmul2d is 26.6-available
   and accuracy-green. **But before any kernel:** a dedicated measure-gate vs the *current* quantized
   path (TQ-decode gather/dequant+SDPA, IV-D1/D2-optimized, near the dense-decode floor) is required —
   the headroom is unmeasured and the decode regime is sync-floor-bound, so this could itself come back
   hostile. That gate is the prerequisite, not a kernel.
3. **AUTO routing predicate (if the INT8 lever ever greenlights):** `V6-NAX-int8` selected ONLY for
   `quantized-KV AND prefill-or-large-batch-decode AND N ≥ <measured-crossover>` where it beats
   TQ-via-SDPA; dense and small-N stay SDPA. Predicate is undefined until the §2 gate measures the
   crossover — do not pre-commit it.

## Phased ladder (if/when the INT8 lever is greenlit — NOT this sprint, NOT the next without the gate)
0. **Gate (prerequisite):** int8-matmul2d QK^T-over-TQ-KV micro-vs-TQ-decode-via-SDPA on M5/26.6
   (Pattern #6). If it doesn't beat the existing path → STOP (hostile, don't build).
1. QK^T-only int8 NAX fragment vs fp32 (correctness) + vs TQ-SDPA (perf).
2. + P@V; 3. fuse; 4. (27-gated) fully-fused dequant-in-cooperative-tensor-load.
Each gate: independent fp32 oracle + three-axis + the `/mlx-mfa-nax-matmul2d-correctness` footgun
pre-flight (type-matching, device-non-const, int8_t/int32_t, K%16, 128-byte align, int64 offsets).

## Disposition

**V6 NAX dense attention: DECLINED on 26.6** (SDPA already owns the NAX win; measured, money-saving).
**V6 NAX int8/quantized-KV: not greenlit** — gated on a §0 measure-vs-TQ-SDPA check that must pass
first. No kernel written. No code changed. The existing V6 NAX primitive remains the expert/research
path (keep-all-paths). The real near-term wins were already shipped (v2.56.0: IV-D1/D2 decode + the
correctness hardening); incremental optimization is at the floor (IV-OPT). This Phase 0 saved an
L/XL kernel build that would have lost to SDPA.
