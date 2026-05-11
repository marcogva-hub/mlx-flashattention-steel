# Phase 1.2 — Decisions Companion

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_2`

Decisions D18-D22 made during Phase 1.2. Continues numbering from
`conv-nax-phase1_1-decisions.md` (D1-D17).

---

## D18 — M-chunking to avoid MPP int32 byte-address overflow

**Context.** Phase 1.1 HANDOFF Pitfall 5: `mlx_mfa.conv_nax` produced
~47% NaN at M=147456 (up1_resnet shape). Root-causing in Phase 1.2
revealed:

- MPP `matmul2d` uses int32 internally for byte-address arithmetic.
- NaN starts deterministically at byte offset ≥ 2^31 in the A buffer.
- For K=13824 f16: NaN onset at row 77696 (= 2^31 / (K × 2 bytes)).
- Verified linear (each M > 77696 produces exactly `M - 77696` NaN rows).
- 5-run determinism confirmed: same NaN count every time.

**Decision.** Implement M-chunking at dispatch level. Each chunk's
im2col output buffer satisfies `chunk_M × K × dtype_bytes < 2^31 × 0.875`.
The 0.875 safety margin accounts for matmul2d's internal address tricks
(cooperative tensor + per-tile + per-thread offsets) which may add up to
~12% address-space overhead.

**Implementation.**
- `_compute_chunk_layout(M, K, dtype_bytes)` returns
  `[(m_offset, m_chunk), ...]` summing to M_total, M_TILE-aligned.
- For K=13824 f16: max chunk_M = 65504 (2047 tiles). up1_resnet
  M=147456 splits into 3 chunks of 49152 each.
- `_im2col3d_source` takes `m_offset` + `m_chunk` as compile-time
  constants so the kernel reads input X at `m_global = m_local + m_offset`.

**Rejected alternatives.**
- `dextents<int64_t, 2>` in MPP wrappers — Metal compiler rejects;
  MPP requires int32.
- Output pointer offsetting (write to global buffer at offset) —
  doesn't help because the BUG is in MPP's internal addressing,
  triggered by the **buffer size** (full M, not chunked), not by
  the write target.
- Algorithm change (matmul replacement) — out of scope for Phase 1.2;
  loses 32% perf gate gain.

**Validation.** up1_resnet (M=147456) post-fix: rel_err = 3.23e-5
vs `mx.conv_general` (within FP16 noise floor). 3-session bit-exact
reproduction: rmse=1.1613434181e-03 identical to 10 decimals.

---

## D19 — Chunk safety headroom = 0.875

**Context.** D18 establishes the int32 byte-address overflow at 2^31.
Empirical testing during Phase 1.2 root-cause:

- `0.95 × 2^31` headroom (used in an initial draft): still fired NaN
  at M close to the boundary on some kernels — internal MPP addressing
  adds offsets beyond the buffer size.
- `0.875 × 2^31` headroom: clean across all tested shapes (mid_resnet,
  up1_resnet, plus probe shapes in microbench).

**Decision.** `SAFETY_HEADROOM = 0.875` as a module-level constant in
`mlx_mfa/conv_nax.py`. Documented in code as accounting for matmul2d's
internal address tricks (cooperative tensor + tile offsets +
per-thread fragment addresses).

**Rationale.** Better to chunk one extra time on the boundary than
to fire NaN silently. The chunking overhead (extra kernel dispatch +
small concat) is ~50-100 µs — negligible vs ~85 ms matmul.

---

## D20 — Asymmetric padding API: tuple-of-pairs

**Context.** Causal video decoders use `pad_T = (K_T-1, 0)` — left-pad
by K_T-1 frames, no right padding — so the convolution at time t only
sees frames [t-K_T+1, t]. This is critical for autoregressive video
generation.

**Decision.** `padding` parameter accepts three forms:

1. `int` — symmetric across all 3 spatial axes (e.g. `padding=1`)
2. `3-tuple of int` — per-axis symmetric (e.g. `padding=(1, 1, 1)`)
3. `3-tuple of (left, right) pairs` — fully asymmetric
   (e.g. `padding=((K_T-1, 0), (1, 1), (1, 1))`)

Plus a convenience kwarg `causal_pad_t: bool` that auto-substitutes
`pT = (K_T-1, 0)` when True.

**Implementation.**
- `_normalize_padding()` converts all three forms to the canonical
  `((pT_l, pT_r), (pH_l, pH_r), (pW_l, pW_r))` representation.
- The im2col kernel uses only the **left** paddings for input-coord
  translation: `t_in = t_out * sT + k_t * dT - pT_l`. The right
  paddings affect T_out computation Python-side.
- ConvKey includes all 6 pad values so asymmetric configurations
  cache distinctly.

**Rejected.**
- Single `pad_T_low, pad_T_high` flat keyword args — visually noisy
  for the common 6-arg case.
- Match PyTorch's `pad` argument (flat 6-tuple) — too easy to mix up
  order with 3-tuple-of-pairs.

**Validation.** Two tests:
- `test_mid_resnet_causal_pad_t` — explicit asymmetric form vs
  `mx.conv_general` with low/high padding lists.
- `test_mid_resnet_causal_pad_t_flag` — `causal_pad_t=True` flag is
  bit-exact equivalent to explicit asymmetric form.

---

## D21 — K_T=1 routing through general path (Phase 1.2)

**Context.** Original prompt §C.1 calls for K_T=1 specialized routing.
When K_T=1, the convolution is effectively 2D per temporal slice:
K = K_H × K_W × C_in (not 27 × C_in for 3×3×3).

**Decision.** Phase 1.2 routes K_T=1 through the **general path** —
the source-gen emits `K = K_H × K_W × C_in × 1` (compile-time
constant), which already handles the smaller K correctly.

**Rationale.** No code change needed beyond the existing source-gen.
The general path's im2col addressing computes
`t_in = t_out * sT + k_t * dT - pT_l` with `k_t ∈ [0, K_T)`. At K_T=1,
`k_t = 0` always, so `t_in = t_out * sT - pT_l`. This degenerates
correctly.

**Phase 1.4 fast path (separate).** The 1×1×1 special case
(K_T = K_H = K_W = 1) skips im2col entirely (pure pointwise matmul).
K_T=1 alone does NOT qualify for that fast path — there's still
spatial expansion in H, W.

**Validation.** `test_kt1_routing` (B=1, T=5, H=64, W=64, K=(1,3,3),
padding=(0,1,1)): rel_err < 1e-4 vs `mx.conv_general`.

---

## D22 — Phase 1.2 total budget = 16 GB im2col

**Context.** Phase 1.1's `_sanity_asserts` had a loose 8 GB single-chunk
budget that DIDN'T catch up1_resnet's failure (its im2col is 4 GB,
below the 8 GB bar, but MPP failed at row 77696 due to int32
addressing).

**Decision.** Phase 1.2 replaces the budget with two-tier check:

1. **Per-chunk** (enforced in `_compute_chunk_layout`):
   `chunk_M × K × dtype_bytes < 2^31 × 0.875` (~1.88 GB).
   Below this: int32 byte-address safety. Above: chunked.

2. **Total** (sanity-assert): `M_total × K × dtype_bytes < 16 GB`.
   Above this: explicit ValueError directing to `mx.conv_general` until
   Phase 1.3 adds streaming for shapes beyond the GPU memory budget.

**Rationale.** With auto-chunking, the bottleneck shifts from
"single-chunk budget" to "total memory budget". 16 GB is generous
relative to MLX's typical workloads. Phase 1.3 will tighten this
with working-set instrumentation.

**Validation.** All Phase 1.2 production shapes are under 16 GB
(up3_resnet0_full at 62.7 GB remains skipped — explicitly out of scope).
