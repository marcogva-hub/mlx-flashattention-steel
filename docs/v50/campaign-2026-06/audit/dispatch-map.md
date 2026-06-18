# mlx-mfa Runtime Dispatch Map — M5 Max / macOS 26.6 (authoritative, test-locked)

**Status:** DURABLE current-state artifact. Every cell is established by RUNTIME FINGERPRINT
(byte-identity vs a known reference; density signature; conv hook telemetry) — NOT source-tracing
(the lesson of four which-binary inversions). Locked by `tests/test_dispatch_map_lock.py` (CI fails on
unintentional drift). Provenance: audit Phase A (`docs/v50/campaign-2026-06/audit/phase-A-dispatch-map-report.md`),
extends the cartography (`runtime-dispatch-cartography-report.md`). Hardware: M5 Max, mlx 0.31.2.

**Fingerprint key:** byteΔ vs SDPA reference **== 0.0** ⇒ the path *is* that kernel (the SDPA fallback
is literally `mx.fast.sdpa`); **~1e-6** ⇒ a different real kernel (same math, different rounding);
conv via `get_hook_stats()` executed/fallback counters.

## The map

| Entry | Input class (decision boundary) | **Kernel that runs** | Fingerprint | Class |
|---|---|---|---|---|
| `flash_attention` | `backend="auto"`, dense | **Apple SDPA** | Δ=0.0 vs sdpa | routed-as-intended |
| `flash_attention` | `backend="mfa"`, dense | **STEEL** (V2 default / V3 cond-auto) | Δ=1.9e-6 vs sdpa (real) | routed-as-intended (expert) |
| `flash_attention_sparse` | **D=128**, built-in maker mask (causal/sliding/strided/lcsa → [N/32,N/32], symmetric since **Phase F**), density < 0.78 | **real NAX sparse** (wins 1.7–4.2×) | Δ=3.8e-6; sloped | **routed-as-intended (gotcha 1 FIXED — Phase F)** |
| `flash_attention_sparse` | **D=128**, symmetric mask, density ≥ 0.78 (ceiling) | **dense Apple SDPA** (density gate) | Δ=0.0 vs sdpa+bias | routed-as-intended (NAX loses near-dense) |
| `flash_attention_sparse` | **D=128**, asymmetric/custom mask (bt_q≠bt_k) OR mask_bytes<4096 | **dense Apple SDPA** | Δ=0.0 vs sdpa+bias; flat | routed-as-intended (residual SDPA edges) |
| `flash_attention_sparse` | **D=64**, default (symmetric) | **real V2 NAX sparse** (since **Phase F**: V2 always, not V1) | Δ=3.8e-6; sloped; ~9× vs old V1 | **routed-as-intended (gotcha 2 FIXED — Phase F)** |
| `flash_attention_gna` | D=128 3D f16 | **native GNA kernel** | Δ=7.3e-2 vs block-bias-SDPA (≠0 → not fallback) | routed-as-intended |
| `flash_attention_topk` | — | own path (topk + SDPA) | Δ=1.9e-6 @ ratio=1.0 | routed-as-intended |
| `sage_attention` | — | int8 sage kernel | Δ=1.1e-3 vs sdpa | routed-as-intended |
| `flash_attention_kvcache` | decode N_q=1 | **Apple SDPA** (gather + SDPA) | Δ=0.0 vs sdpa | routed-as-intended (sync-floor regime) |
| `mx.grad(flash_attention)` | dense | **SDPA-vjp** | dQ Δ=0.0 vs sdpa-vjp | routed-as-intended |
| `mx.grad(flash_attention_sparse)` | default (no env) | **dense SDPA-vjp** | dQ Δ=0.0 vs sdpa-vjp+bias | **routed-but-suboptimal (gotcha 3): sparse fwd, DENSE bwd** |
| `mx.grad(flash_attention_sparse)` | `MFA_ENABLE_V6_BACKWARD=1` + D∈{64,128} + N≥2048 + ndim==2 + bt≥64 | hybrid: **dV native** + dQ/dK SDPA-vjp | dQ Δ=0.0 (SDPA-vjp by design); dV native | declined-on-perf (opt-in; Pattern #6) |
| `mx.grad(flash_attention_sparse)` | `MFA_V6_BWD_SPARSE_NATIVE=1` (+ above) | full-native sparse | — | declined-on-perf (opt-in; Pattern #6) |
| `mx.conv_general` | Conv3D eligible (C%16==0 & ≥32, HW%8==0, B=1, pad=(1,1,1), f16) | **NAX conv kernel** (matmul2d) | `executed.conv3d_nax_forward++` | routed-as-intended (auto-hook) |
| `mx.conv_general` | Conv3D ineligible | `mx.conv_general` | `fallback.conv3d_nax_forward++` | by-design fallback |

## Runtime guards (each runtime-confirmed)

1. **Dense default → SDPA** — `dispatch_policy._M5_NAX_THRESHOLDS = 999999` (always SDPA); `backend="mfa"` overrides to STEEL (legacy-on-M5: SDPA 3–4× faster).
2. **Sparse maker masks → symmetric → NAX (Phase F)** — built-in D=128 makers emit symmetric 32×32 (`masks.py::_bq_bk(128)=(32,32)`); the auto-route (`attention.py`, `bt_q==bt_k`) sends them to the real NAX kernel. Density gate: ≥ `_nax_sparse_density_ceiling()` (0.78, env `MFA_NAX_SPARSE_DENSITY_CEILING`) → SDPA fallback (NAX loses near-dense).
3. **Sparse asymmetric/custom or small (<4096 bytes) → SDPA fallback** — `_sparse_fallback_sdpa_perhead` on M5+ (asymmetric STEEL kernel disabled by the `(long)p->NK` miscompile, `docs/v6-nax/sparse-bug-investigation.md`; small masks excluded by NAX device-pointer lowering). Validator accepts EITHER geometry then exact-tile-splits to kernel geometry.
4. **Sparse V1/V2 selection (Phase F)** — `decide_auto_version` routes D∈{64,128} → V2 (matmul2d) always; the old `qL*kL*D≥2^31` work-product gate is RETIRED (V1-scalar was never fastest). V1 kept only as the genuine fallback (D∉{64,128}).
5. **Sparse backward hybrid gate** — `MFA_ENABLE_V6_BACKWARD=1` AND D∈{64,128} AND N≥2048 AND ndim==2 AND **bt≥64** (III-4 D16 fix); else dense SDPA-vjp.
6. **Conv3D NAX MPP gate** — C%16==0 & ≥32, HW%8==0, B=1, pad=(1,1,1), f16/bf16.

## Gotchas — status after Phase F (2026-06-18)

1. **D=128 sparse + built-in maker → silent dense SDPA** — **FIXED (Phase F):** makers now emit symmetric 32×32 → NAX (1.7–4.2× at d<0.78). Residual SDPA only for asymmetric/custom, small (<4096 bytes), or dense (≥0.78) masks — all intentional now.
2. **D=64 sparse → slow** — **FIXED (Phase F):** `decide_auto_version` routes D=64 → V2 (was V1 via the 2^31 gate); ~9× faster.
3. **Sparse backward is dense by default** — UNCHANGED (declined-on-perf, Pattern #6): `mx.grad(flash_attention_sparse)` gets the sparse forward win but a dense SDPA-vjp backward unless the opt-in env + bt≥64 is set. Mild (correct, just not sparse-accelerated).

No NEW catastrophic silent-fallbacks found in backward / GNA / paged / conv beyond these: backward=SDPA-vjp (intended), GNA=native (intended), decode=SDPA (intended, sync-floor regime), conv=NAX-when-eligible (intended).

## Notes / deferred (Phase B)
- The GNA Δ=7.3e-2 is my crude block-mask reference over-approximating GNA's exact per-element window
  — a **Phase-B correctness-reference** item, NOT a dispatch issue (GNA native provably runs, Δ≠0).
- `backend="mfa"` STEEL *variant* selection (V2/V3/V4/V5/split-K/dsplit/flash_decode) — enumerated
  from `eval_gpu`; the default is real STEEL (Δ=1.9e-6). Per-variant routing fingerprint (env-toggle
  + timing-match) deferred to Phase B/E; not user-facing default (dense default = SDPA).
- Sparse-backward dV-native (hybrid/full-native opt-in) — dQ confirmed SDPA-vjp; dV-native fingerprint
  deferred to Phase B (declined-on-perf research path).
