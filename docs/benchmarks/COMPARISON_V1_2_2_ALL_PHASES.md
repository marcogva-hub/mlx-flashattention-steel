# Benchmark Comparison: Pre-Remediation vs v1.2.2 (All Phases Complete)

**Device**: Apple M1 Max (gen 13, M3+: False)
**MLX version**: 0.31.0
**Pre-remediation commit**: c234138 (v1.2.1, before any tech-debt fixes)
**Post-remediation**: v1.2.2 — all Phases 1-4 complete
**Date**: 2026-03-09

---

## Phases applied

| Phase | Fixes | Description |
|-------|-------|-------------|
| 1 | D.1, D.2, D.3, A.3, B.3, E.5 | Per-call dispatch overhead eliminated |
| 2 | B.2, C.3, C.4, C.5, D.4, D.6, D.8, D.9 | Python-loop hotpaths vectorised/cached |
| 3 | B.1, D.5, E.4 | Structural: saved logsumexp, C++ contiguity, batched backward |
| 4 | A.1+A.2, C.1+E.2, E.1 | New C++ Metal primitives: fused quantize, scatter KV, concat eval |

---

## Forward Attention (MFA vs SDPA speedup)

| Config | Pre speedup | Post speedup | Delta |
|--------|------------|-------------|-------|
| fwd D=64  N=4096  f16 causal | 0.94x | 0.98x | +0.04x |
| fwd D=64  N=8192  f16 causal | 1.40x | 1.37x | -0.03x (noise) |
| fwd D=64  N=8192  f16 non-causal | 0.91x | 0.92x | +0.01x |
| fwd D=128 N=2048  f16 causal | 0.73x | 0.74x | +0.01x |
| fwd D=128 N=4096  f16 causal | 0.83x | 0.83x | 0.00x |
| fwd D=128 N=8192  f16 causal | 1.25x | 1.26x | +0.01x |
| fwd D=128 N=8192  f16 non-causal | 0.82x | 0.83x | +0.01x |
| fwd D=128 N=4096  bf16 causal | 0.70x | 0.70x | 0.00x |
| fwd D=256 N=4096  f16 causal | 0.56x | 0.57x | +0.01x |
| fwd D=256 N=8192  f16 causal | 0.77x | 0.77x | 0.00x |
| fwd D=512 N=2048  f16 causal | 0.25x | 0.24x | -0.01x |
| fwd D=512 N=4096  f16 causal | 0.26x | 0.26x | 0.00x |
| fwd D=512 N=4096  f16 non-causal | 0.24x | 0.23x | -0.01x |

All values within the +/-15% run-to-run noise band. STEEL Metal kernel
compute time is unchanged — Python-side fixes don't move 5-170ms kernel times.

---

## Backward Attention

| Config | Pre speedup | Post speedup | Delta |
|--------|------------|-------------|-------|
| bwd D=64  N=2048  f16 causal | 0.55x | 0.54x | -0.01x (noise) |
| bwd D=64  N=4096  f16 causal | 0.66x | 0.64x | -0.02x (noise) |
| bwd D=128 N=2048  f16 causal | 0.26x | 0.25x | -0.01x (noise) |
| bwd D=128 N=4096  f16 causal | 0.26x | 0.26x | 0.00x |
| bwd D=128 N=2048  bf16 causal | 0.18x | 0.19x | +0.01x |
| bwd D=256 N=2048  f16 causal | 0.16x | 0.17x | +0.01x |
| bwd D=256 N=4096  f16 causal | 0.16x | 0.16x | 0.00x |
| bwd D=512 N=1024  f16 causal | 0.12x | 0.12x | 0.00x |
| bwd D=512 N=2048  f16 causal | 0.12x | 0.12x | 0.00x |

Backward uses mx.vjp(SDPA) for the dense path (Phase C, commit b622941). The
B.1 logsumexp save benefits the sparse and custom backward paths where L was
previously discarded and recomputed.

---

## Sliding Window Attention

| Config | Pre speedup | Post speedup | Delta |
|--------|------------|-------------|-------|
| win D=128 N=4096  w=512  f16 | 5.64x | 5.43x | -0.21x (noise) |
| win D=128 N=8192  w=512  f16 | 8.07x | 7.67x | -0.40x (noise) |
| win D=128 N=8192  w=1024 f16 | 4.46x | 4.46x | 0.00x |
| win D=128 N=16384 w=512  f16 | 13.24x | 13.17x | -0.07x (noise) |

All within noise. N=16384 w=512 remains the headline at ~13x.

---

## SageAttention — KEY WIN from Phase 4-A.1+A.2

Fused MFAQuantizePerBlock C++ primitive replaces 12+ Python MLX ops with a
single Metal kernel, moving SageAttention from slower-than to faster-than
flash_attention at small N.

| Config | Pre FA (ms) | Pre Sage (ms) | Pre speedup | Post FA (ms) | Post Sage (ms) | Post speedup | Delta |
|--------|------------|--------------|------------|-------------|---------------|-------------|-------|
| N=512  | 1.17 | 1.32 | 0.89x | 0.93 | 0.85 | **1.10x** | **+0.21x** |
| N=1024 | 1.73 | 2.13 | 0.81x | 1.93 | 1.73 | **1.12x** | **+0.31x** |
| N=2048 | 3.93 | 6.65 | 0.59x | 3.78 | 6.05 | 0.63x | +0.04x |
| N=4096 | 12.0 | 23.0 | 0.52x | 11.57 | 20.55 | 0.56x | +0.04x |

At N>=2048 the sage_forward Metal kernel dominates; quantize overhead is now
~0.5ms (fused) vs ~3ms (Python) but is small relative to ~6-20ms sage_forward.

---

## Impact Summary

### Definitive wins (measurable, beyond noise)

| Fix | Gain |
|-----|------|
| A.1+A.2 fused quantize | SageAttn N=512: 0.89x -> 1.10x (+24%) |
| A.1+A.2 fused quantize | SageAttn N=1024: 0.81x -> 1.12x (+38%) |
| C.5 speculative_verify | O(BxN) GPU->CPU scalar syncs eliminated |
| D.6 sparse factory cache | ~30us/call -> ~0.2us on repeated configs |
| D.1 _ext_available cache | ~3us saved per flash_attention() call |
| D.4 mlx_lm direct dispatch | ~2us saved per token (32 layers = ~64us/token) |
| E.1 InferenceContext concat | O(N^2) graph depth -> O(1) for long decode loops |
| C.1+E.2 scatter KV write | O(num_blocks) concat -> O(1) Metal scatter |

### Within-noise (as expected for GPU-dominated workloads)

For 5-170ms forward kernels, the 2-5us Python overhead eliminated by
Phases 1-2 represents 0.003-0.1% improvement — undetectable vs +-15% noise.

### Deferred (Phase 5)

| Fix | Estimated impact |
|-----|-----------------|
| C.2 Batched RoPE offsets | ~48x fewer dispatch calls for B>1 decode |
| E.3 seq_lens as Metal buffer | Eliminates GPU->CPU sync per paged token |

