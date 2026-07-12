# mlx-mfa Runtime Dispatch Map — M5 Max / macOS 26.6 (authoritative, test-locked)

**Status:** DURABLE current-state artifact. Every cell is established by RUNTIME FINGERPRINT
(byte-identity vs a known reference; density signature; conv hook telemetry) — NOT source-tracing
(the lesson of four which-binary inversions). Locked by `tests/test_dispatch_map_lock.py` (CI fails on
unintentional drift). Provenance: audit Phase A (`.doc-archive/docs/v50/campaign-2026-06/audit/phase-A-dispatch-map-report.md`),
extends the cartography (`runtime-dispatch-cartography-report.md`). Hardware: M5 Max, mlx 0.31.2.

**Fingerprint key:** byteΔ vs SDPA reference **== 0.0** ⇒ the path *is* that kernel (the SDPA fallback
is literally `mx.fast.sdpa`); **~1e-6** ⇒ a different real kernel (same math, different rounding);
conv via `get_hook_stats()` executed/fallback counters.

## The map

| Entry | Input class (decision boundary) | **Kernel that runs** | Fingerprint | Class |
|---|---|---|---|---|
| `flash_attention` | `backend="auto"`, dense **D=128**, **N≥2048** (N==S, fp16/bf16, no window/bias) | **NAX matmul2d** (`v6_nax_forward`, F-2) | Δ=1.9e-6 vs sdpa (real) | routed-as-intended (parity-to-modest-win; all scales; bwd=SDPA-vjp) |
| `flash_attention` | `backend="auto"`, dense **D=128**, **N<2048** (Tier-2 #1 threshold) | **Apple SDPA** | Δ=0.0 vs sdpa | routed-as-intended (NAX loses small-N: N=512 16-36%, N=1024 3-17%; `MFA_V6_DENSE_MIN_N`) |
| `flash_attention` | `backend="auto"`, dense **D=64** (non-causal, or causal with B·H<4 or N<`v3_min_N`) / cross-attn / windowed / `MFA_DISABLE_V6_DENSE=1` | **Apple SDPA** | Δ=0.0 vs sdpa | routed-as-intended (NAX loses at D=64) |
| `flash_attention` | `backend="auto"`, dense **D=64**, **causal & B·H≥4 & N≥`v3_min_N`(=4096)** | **NAX tier (M5): Apple SDPA** — `should_use_mfa(D=64,causal,has_nax=True)` returns False, so the "MFA primitive" carve-out does NOT engage; that real-kernel path only exists on the M3/M4 tier where `has_nax=False`. Backward: **V6 split** (`v6_nax_backward`, default-on D=64 qL≥2048), NOT SDPA-vjp. | **NAX tier: Δ=0.0 vs sdpa** (byteΔ=0 → SDPA fallback; M3/M4 tier: byteΔ>0 real MFA primitive) | routed-as-intended (M2/H-02 label fix: M5/NAX forward = SDPA byteΔ=0, math correct; the byteΔ>0 MFA-primitive case is the M3/M4 tier only; D=64 backward = V6 split by default) |
| `flash_attention` | `backend="mfa"`, dense | **simdgroup STEEL** (V2 default / V3 cond-auto) | Δ=1.9e-6 vs sdpa (real) | routed-as-intended (expert; legacy-on-M5) |
| `flash_attention_sparse` | **D=128**, built-in maker mask (causal/sliding/strided/lcsa → [N/32,N/32], symmetric since **Phase F**), density < 0.78 | **real NAX sparse** (wins 1.7–4.2×) | Δ=3.8e-6; sloped | **routed-as-intended (gotcha 1 FIXED — Phase F)** |
| `flash_attention_sparse` | **D=128**, symmetric mask, density ≥ 0.78 (ceiling) | **dense Apple SDPA** (density gate) | Δ=0.0 vs sdpa+bias | routed-as-intended (NAX loses near-dense) |
| `flash_attention_sparse` | **D=128**, asymmetric/custom mask (bt_q≠bt_k) OR mask_bytes<4096 | **dense Apple SDPA** | Δ=0.0 vs sdpa+bias; flat | routed-as-intended (residual SDPA edges) |
| `flash_attention_sparse` | **D=64**, default (symmetric) | **real V2 NAX sparse** (since **Phase F**: V2 always, not V1) | Δ=3.8e-6; sloped; ~9× vs old V1 | **routed-as-intended (gotcha 2 FIXED — Phase F)** |
| `flash_attention_sparse` | **D∈{64,128}**, **bf16**, symmetric, BT=32 | **real V2 NAX sparse** (since the bf16 fix; was the V1 scalar fallback) | bf16 Δ=3.3e-5 vs fp32 oracle; 1.2–6.5× vs SDPA | **routed-as-intended (gotcha 4 FIXED)** |
| `flash_attention_gna` | D=128 3D f16 | **native GNA kernel** | Δ=7.3e-2 vs block-bias-SDPA (≠0 → not fallback) | routed-as-intended |
| `flash_attention_topk` | — | own path (topk + SDPA) | Δ=1.9e-6 @ ratio=1.0 | routed-as-intended |
| `sage_attention` | — | int8 sage kernel | Δ=1.1e-3 vs sdpa | routed-as-intended |
| `flash_attention_kvcache` | decode N_q=1 | **Apple SDPA** (gather + SDPA) | Δ=0.0 vs sdpa | routed-as-intended (sync-floor regime) |
| `mx.grad(flash_attention)` | dense **D=64**, qL≥2048, fp16/bf16 (causal OR non-causal) | **V6 split** (`v6_nax_backward`, default-on; opt-out `MFA_DISABLE_V6_BACKWARD=1`) | native (≠ sdpa-vjp) | routed-as-intended (M2/H-02: D=64 backward default = V6 split, 2.16–3.05× vs SDPA-vjp) |
| `mx.grad(flash_attention)` | dense **D=128** / D=64 small-qL / opt-out | **SDPA-vjp** | dQ Δ=0.0 vs sdpa-vjp | routed-as-intended |
| `mx.grad(flash_attention_sparse)` | default (no env) | **dense SDPA-vjp** | dQ Δ=0.0 vs sdpa-vjp+bias | **routed-but-suboptimal (gotcha 3): sparse fwd, DENSE bwd** |
| `mx.grad(flash_attention_sparse)` | `MFA_ENABLE_V6_BACKWARD=1` + D∈{64,128} + N≥2048 + ndim==2 + bt≥64 | hybrid: **dV native** + dQ/dK SDPA-vjp | dQ Δ=0.0 (SDPA-vjp by design); dV native | declined-on-perf (opt-in; Pattern #6) |
| `mx.grad(flash_attention_sparse)` | `MFA_V6_BWD_SPARSE_NATIVE=1` (+ above) | full-native sparse | — | declined-on-perf (opt-in; Pattern #6) |
| `mx.conv_general` | Conv3D eligible (C%16==0 & ≥32, HW%8==0, B=1, pad=(1,1,1), f16) | **NAX conv kernel** (matmul2d) | `executed.conv3d_nax_forward++` | routed-as-intended (auto-hook) |
| `mx.conv_general` | `MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE=1`, fp16 `B=1`, `T∈{4,5}`, `108×132`, `C_in=C_out=512`, k=3³, stride 1, temporal pad 0/spatial pad 1 | **NAX MPP spatial-pad → slice** | `executed.conv3d_nax_spatial_pad_slice++` | opt-in β3; candidate default-on only after stable-macOS revalidation |
| `flash_attention_varlen` | `MFA_ENABLE_VARLEN_NAX=1`, `B=1`, D128, f16/bf16, causal or non-causal, GQA 2/4/8, 20/24 equal-Q/K segments, `35018≤total≤35250` | **V6 NAX packed-varlen**, fixed BQ32/BK32/WM2 | dispatch trace `varlen_v6nax` | opt-in β3; fixed tiles are explicitly coherent across MSL generation and host dispatch |
| `mx.conv_general` | Conv3D ineligible | `mx.conv_general` | `fallback.conv3d_nax_forward++` | by-design fallback |

## Runtime guards (each runtime-confirmed)

1. **Dense default → NAX (D=128, N≥2048) / SDPA (else)** — F-2 (Change 3): D=128 dense auto routes
   to the NAX matmul2d forward (`v6_nax_forward`, parity-to-modest-win, all scales via the
   plumbed scale arg, backward via SDPA-vjp) **for N≥2048**; below that the Apple SDPA kernel is
   faster (Tier-2 #1: N=512 16-36%, N=1024 3-17% — crossover governed by N alone, not N·B·H;
   threshold `MFA_V6_DENSE_MIN_N`=2048, =0 forces all-N NAX).  D=64 + cross-attn + windowed +
   opt-out stay SDPA (NAX loses 1.17–1.22× at D=64).  `backend="mfa"` overrides to simdgroup STEEL (legacy-on-M5:
   SDPA 2–4× faster).
2. **Sparse maker masks → symmetric → NAX (Phase F)** — built-in D=128 makers emit symmetric 32×32 (`masks.py::_bq_bk(128)=(32,32)`); the auto-route (`attention.py`, `bt_q==bt_k`) sends them to the real NAX kernel. Density gate: ≥ `_nax_sparse_density_ceiling()` (0.78, env `MFA_NAX_SPARSE_DENSITY_CEILING`) → SDPA fallback (NAX loses near-dense).
3. **Sparse asymmetric/custom or small (<4096 bytes) → SDPA fallback** — `_sparse_fallback_sdpa_perhead` on M5+ (asymmetric STEEL kernel disabled by the `(long)p->NK` miscompile, `.doc-archive/docs/v6-nax/sparse-bug-investigation.md`; small masks excluded by NAX device-pointer lowering). Validator accepts EITHER geometry then exact-tile-splits to kernel geometry.
4. **Sparse V1/V2 selection (Phase F)** — `decide_auto_version` routes D∈{64,128} → V2 (matmul2d) always; the old `qL*kL*D≥2^31` work-product gate is RETIRED (V1-scalar was never fastest). V1 kept only as the genuine fallback (D∉{64,128}).
5. **Sparse backward hybrid gate** — `MFA_ENABLE_V6_BACKWARD=1` AND D∈{64,128} AND N≥2048 AND ndim==2 AND **bt≥64** (III-4 D16 fix); else dense SDPA-vjp.
6. **Conv3D NAX MPP gate** — C%16==0 & ≥32, HW%8==0, B=1, pad=(1,1,1), f16/bf16.

## Gotchas — status after Phase F (2026-06-18)

1. **D=128 sparse + built-in maker → silent dense SDPA** — **FIXED (Phase F):** makers now emit symmetric 32×32 → NAX (1.7–4.2× at d<0.78). Residual SDPA only for asymmetric/custom, small (<4096 bytes), or dense (≥0.78) masks — all intentional now.
2. **D=64 sparse → slow** — **FIXED (Phase F):** `decide_auto_version` routes D=64 → V2 (was V1 via the 2^31 gate); ~9× faster.
3. **Sparse backward is dense by default** — UNCHANGED (declined-on-perf, Pattern #6): `mx.grad(flash_attention_sparse)` gets the sparse forward win but a dense SDPA-vjp backward unless the opt-in env + bt≥64 is set. Mild (correct, just not sparse-accelerated).
4. **bf16 sparse → slow V1 scalar fallback** — **FIXED (bf16-sparse-v2, 2026-06-18):** V2 eligibility was `&& is_f16`, so bf16 silently fell to V1 (up to ~50× slower than plain SDPA-with-mask — strictly dominated). The V2 `mma` is dtype-templated with fp32 accumulation, so removing the gate routes bf16 → V2 (faithful 3.3e-5 vs fp32 oracle; 1.2–6.5× vs SDPA, parity with fp16). Note: gotcha 2's "D=64 sparse FIXED (Phase F)" was only true for fp16; this closes the bf16 case. Locked by `tests/test_sparse_bf16_v2_lock.py`.

No NEW catastrophic silent-fallbacks found in backward / GNA / paged / conv beyond these: backward=SDPA-vjp (intended), GNA=native (intended), decode=SDPA (intended, sync-floor regime), conv=NAX-when-eligible (intended).

## bf16 dtype-routing audit (Tier-1 #1, 2026-06-18)

After the gotcha-4 sparse fix, a full **(path × dtype) routing audit** fingerprinted the dispatched
binary for `{fp16, bf16}` on every NAX path (Lesson #14 — runtime fingerprint, not source-trust).
**Verdict: the sparse forward was the ONLY silent bf16 downgrade; all other NAX paths route bf16 ==
fp16.** Per-path fingerprint:

| Path | bf16 fingerprint | classification |
|---|---|---|
| Dense fwd D=128 (auto) | NAX, Δ=1.5e-5 vs SDPA (≠0) | fast-path (= fp16) |
| Dense fwd D=64 (force/recompute) | NAX, Δ=1.5e-5 (≠0) | fast-path |
| Dense backward **D=128** | SDPA-vjp (default; native D=128 bwd opt-in + slower) | by-design floor; symmetric |
| Dense backward **D=64** native (default-on, N≥2048) | NATIVE bwd, byteΔ>0 vs forced-SDPA-vjp (both dtypes, causal + non-causal) | fast-path (**2.16–3.05× vs SDPA-vjp at qL≥4096**, ~1.5–1.7× @qL2048; M5 Max / macOS 26.6 / MLX 0.31.2; dQ tile tuned `BK 64→32`, Tier-1 #2) |
| Sparse fwd V2 | NAX, Δ=6.1e-5 vs forced-V1 | fast-path (gotcha-4 fix holds) |
| Sparse backward hybrid (opt-in) | runs, finite grad | by-design SDPA-vjp; symmetric |
| conv3d NAX (MPP-eligible) | `executed.conv3d_nax_forward++`, 0 fallback | fast-path (auto-hook) |
| conv3d legacy im2col (raw C++ only) | **loud raise** (`mfa_conv_nax.cpp` bf16 guard) | by-design loud (Rule 8); not auto-routed |
| GNA native | native, Δ=6.6e-2 vs sparse-fallback | fast-path |
| paged-varlen fused | fused kernel, err 5.6e-3 vs fp32 oracle | fast-path (`dtype_code` 0/1 symmetric) |

Locked by `tests/test_bf16_routing_all_nax_lock.py` (dense D=128 / D=64 / conv3d / GNA) +
`tests/test_sparse_bf16_v2_lock.py` (sparse) — a future re-added `is_f16` gate (Python eligibility OR
a downstream C++ gate) on any of these fails CI.

## Hardware-tier map + V6 purification (Phase F-3, 2026-06-18)

| Tier | Dense forward kernel family | Notes |
|---|---|---|
| **M1–M4** | **standalone simdgroup STEEL** (V1/V2/V3/split-K/dsplit/flash_decode) | the validated dense tier — `m3_prefers_v1`, `v3_min_N`, RESULTS.md. UNTOUCHED by F-3. (V4/V5 retired from build — Lot-2.) |
| **M5+** | **NAX matmul2d** (`v6_nax_forward`) for D=128 `auto`; **Apple SDPA** for **all** D=64 dense `auto` (incl. causal-large-N: `should_use_mfa(D=64, has_nax=True)`=False → byteΔ=0; the V3 cond-auto MFA-primitive path is M3/M4-tier only — see the dense-D64 rows above) | NAX/V6 = the M5+ tier (F-2). |
| any (expert) | `backend="mfa"` → simdgroup STEEL | **legacy-reachable, loses on M5** (SDPA 2–4× faster). |

**V6 is now PURE NAX (F-3).** The simdgroup-*within*-V6 fallback (the old `MFAV6Forward`
`use_v6nax=false` path) was a **diverged, D=64-BROKEN duplicate** (D=64 N=4096 gave max-abs-err
≈512 vs fp32) of the standalone family, **unreachable from production Python** (every V6 entry
forces NAX; NAX-ineligible dense → the existing dispatch: D=64 → SDPA). It is **removed**:
`MFAV6Forward` serves only NAX (D∈{64,128}, valid GQA); invalid GQA raises (Rule 8) rather than
silently dispatching the removed broken path. The standalone simdgroup family (the M1–M4 tier) is
untouched.

**V4/V5 RETIRED FROM BUILD (Lot-2, off `3933c5f`).** Both were standalone experimental opt-ins
(`MFA_ENABLE_V4` / `MFA_ENABLE_V5`), **never auto-selected on any tier** (compiled-but-unrouted) —
verified-not-routed at source before removal (no auto-route in `dispatch_policy`; only the explicit
env gates in `eval_gpu`). **V5 was M5-validated *pour la forme* and showed no advantage anywhere in
its envelope (3.1–4.4× slower than the routed NAX/SDPA default across D=64/128 × fp16/bf16 ×
causal/nc × N∈{4096,8192}), correct ≤5.6e-4** — so the retirement is measured, not assumed. The
`.cpp`/`.hpp` sources are dropped from `CMakeLists.txt` and the dispatch + env gates removed;
source recoverable via the `archive/v4-v5-prototypes` tag + git history (keep-all-paths). The
standalone simdgroup family that *is* auto-reachable (V1/V2/V3/split-K/dsplit/flash_decode) is
untouched.

### V3 — partial-route status (RETAINED, intentional)

V3 is the one conditionally-auto-routed standalone STEEL variant — **KEPT, not a removal
candidate.** Exact gate predicate (`csrc/mfa_attention.cpp` ~607–611, source of truth):

```
route V3  ⟺  !MFA_ENV(MFA_DISABLE_V3)
             AND causal
             AND B*H >= 4                       # sufficient parallelism (sweep: V3 wins all B·H≥4)
             AND N >= v3_min_N                   # v3_min_N = 4096 (D=64) / 2048 (D=128)
```

`MFA_DISABLE_V3=1` forces V2 (debug/bench escape hatch). On M5 the production dense default routes
to NAX/SDPA *before* this path; V3 is reached on M1–M4 and via `backend="mfa"`.

**Falsified-claims caveat (Lesson #15):** the V3 perf numbers embedded in the source comments and
RESULTS.md (e.g. "geomean V3/SDPA 1.47×", "V3 ~32% faster") were measured on **older
hardware/MLX/macOS** and have been *re-stated/falsified before* (see III-11/III-12b honest-perf
re-statement). **Do NOT inherit them.** Any future V3 perf claim — for a release note, a routing
change, or a removal decision — MUST be re-measured under current rules (absolute ms, 3-session
§AA.4, current M5/26.x), not copied from these comments. The gate predicate above is current; the
*magnitudes* in the comments are historical.

## Notes / deferred (Phase B)
- The GNA Δ=7.3e-2 is my crude block-mask reference over-approximating GNA's exact per-element window
  — a **Phase-B correctness-reference** item, NOT a dispatch issue (GNA native provably runs, Δ≠0).
- `backend="mfa"` STEEL *variant* selection (V2/V3/split-K/dsplit/flash_decode) — enumerated
  from `eval_gpu`; the default is real STEEL (Δ=1.9e-6). Per-variant routing fingerprint (env-toggle
  + timing-match) deferred to Phase B/E; not user-facing default (dense default = SDPA).
- Sparse-backward dV-native (hybrid/full-native opt-in) — dQ confirmed SDPA-vjp; dV-native fingerprint
  deferred to Phase B (declined-on-perf research path).
