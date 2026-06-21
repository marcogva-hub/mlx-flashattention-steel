# Backward Family — Verified Per-Kernel Spec (audit B3, durable reference)

RUNTIME-verified on M5/26.6. Gradient correctness vs an INDEPENDENT fp32 gradient oracle (`mx.vjp` of
a MANUAL pure-mlx fp32 forward — not another kernel, lesson #11; oracle trusted via ~1e-7 agreement
with SDPA-vjp on the SDPA paths + an FD sign/scale check). Per-gradient which-binary by byte-identity
vs SDPA-vjp (native is byte-DISTINCT, Δ>0; SDPA-vjp is Δ=0). Locked by
`tests/test_backward_family_lock.py` (6 cells). Labels: [V]erified/[D]educed.

## The backward is a MIX — per-(path × gradient) which-binary map [V]

| Path (how reached) | dQ | dK | dV |
|---|---|---|---|
| **dense D=128** (any) | SDPA-vjp | SDPA-vjp | SDPA-vjp |
| **dense D=64 causal/non-causal, N≥2048** (DEFAULT-ON) | **NATIVE** | **NATIVE** | **NATIVE** |
| **sparse backward DEFAULT** (no env) | SDPA-vjp | SDPA-vjp | SDPA-vjp |
| **sparse opt-in** `MFA_ENABLE_V6_BACKWARD=1`, bt≥64 (**hybrid**) | SDPA-vjp | SDPA-vjp | **NATIVE** |
| **sparse opt-in** `MFA_V6_BWD_SPARSE_NATIVE=1`, bt≥64 (full-native) | NATIVE | NATIVE | NATIVE |

Confirms + completes B1's "hybrid" glimpse: the sparse opt-in hybrid is **native dV only** (dQ/dK stay
SDPA-vjp). The full-native opt-in makes all three native.

## Native backward kernels (when NATIVE above) — `_v6nax_backward_vjp` / sparse orchestrators

- **dQ**: `v6_nax_backward_query` (single Primitive). [V]
- **dK/dV**: `v6_nax_backward_kv`, or fused `v6_nax_backward_fused_dkdv_raw` / split `v6_nax_backward_dv_raw`
  (**AUTO (`MFA_V6_BWD_KERNEL=auto`): split for every D** — D∈{64,128}; fused is opt-in via `=fused` only.
  Corrected H-03/M5: `auto` no longer picks fused at D=64 — the fused-BK16 D=64 edge was withdrawn (fused is
  now only parity-with-split, not faster); fused still regresses 3-7% at D=128. D=64 backward default = split-V6,
  **2.16–3.05× vs SDPA-vjp** (M5 Max / macOS 26.6 / MLX 0.31.2). [D — source]
- **sparse**: `v6_nax_backward_query_sparse_raw` (dQ), `v6_nax_backward_dv_sparse_raw` (dV),
  `v6_nax_backward_fused_dkdv_sparse_raw` / `v6_nax_backward_dk_sparse_raw` (dK). [V — source]
- All are NAX `matmul2d` cooperative-tensor; recompute O/L from the saved sparse-LSE (consistent
  sparse-LSE forward + sparse backward — Pattern #5). Faithful FA-2 backward (7-GEMM dQ/dKV; no
  inspired-by deviation). [V]

## Constraints / valid regimes [V]
- Dense native carveout (`dispatch_policy._v6nax_backward_carveout`): D=64, N≥2048, fp16/bf16, causal
  OR non-causal → **default-on**; D=128 + everything else → opt-in `MFA_ENABLE_V6_BACKWARD=1`. Outer
  guards: softcap==0, alibi_slopes is None, not return_lse.
- Sparse hybrid eligibility: `MFA_ENABLE_V6_BACKWARD=1` AND D∈{64,128} AND N≥2048 AND ndim==2 AND
  **bt≥64** (III-4 D16 fix: bt<64 OR-downsamples the mask → wrong dV/dK, so bt≥64 required for the
  native sparse backward; finer masks route to SDPA-vjp). Non-GQA only (sparse PoC).
- Full-native: + `MFA_V6_BWD_SPARSE_NATIVE=1` (declined-on-perf, Pattern #6: native < SDPA-vjp dense
  at VSR shapes).

## Gradient correctness (fp32 oracle, all edges) — LOCKED [V]
Every path × every gradient: **err ≤ 1.2e-4** (dV highest, accumulation-heavy; dQ/dK ≤ 5e-7), all
finite. 6 locked cells (dense D128/D64-causal/D64-noncausal, sparse default/hybrid/full-native).

## Selection-threshold audit
| Threshold | Verdict |
|---|---|
| dense carveout `N≥2048` floor | **MEASURED** — lowered 4096→2048 after v2.39.1 BK=16 fix reached parity (documented, 3-session). Not arbitrary. [V] |
| sparse hybrid `bt≥64` gate | **CORRECTNESS** gate (mask OR-downsample, III-4 D16), not perf/overflow. [V] |
| D=128 default-off (opt-in) | measured: D=128 V6NAX backward SLOWER than SDPA-vjp (declined). [V] |

No arbitrary/overflow threshold. **Phase-E carry-forward** (open): the sparse V1↔V2 `2^31` work
threshold's PERF validity (is V1 ever faster than V2; over-use below threshold) — perf question, not
resolved here (the overflow concern was resolved benign in B2).

## Comment sweep
Backward family comments are **fresh** — the "PoC" labels (sparse-native dV, Prompt 5b) accurately
describe the declined-on-perf path; "will transpose / will consume lse / for now" are accurate in-call
descriptions. No corrections needed (as B2's dense).
