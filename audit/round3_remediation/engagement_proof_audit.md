# Engagement-Proof & Anti-Vacuity Audit — Volet A

> Enumeration-first artifact for volet A of the audit-remediation campaign.
> Host **M5 Max / macOS 26.6 / MLX 0.31.2**, branch `fix/audit-remediation`,
> base HEAD `dafdbce` (2.61.0). Read-only enumeration + telemetry/test-only
> fixes; the forward/backward **binaries are byteΔ-identical before vs after**
> (proven below). All line numbers verified at source (RULE 16 — the audit
> leads were treated as hypotheses).

The contract: every terminal `_dispatch_trace` record must name the **binary
that actually executed**, every engagement assertion must rest on a runtime
which-binary proof where one exists, and no spurious record may mislead a
future test. Each finding below maps from **untrustworthy → trustworthy**.

---

## Part 1 — Every terminal `_dispatch_trace` record site

Enumerated by `grep -n "_dtrace.record(" mlx_mfa/attention.py` (the sole
recorder; the C++ variant pick inside `mfa_primitive` is a separate layer
captured behaviorally, per the module docstring). **Intended route vs binary
that actually runs the forward**, and whether they match.

### Forward terminals (in `flash_attention`)

| line | recorded | binary that runs | match? |
|---|---|---|---|
| 549 | `sdpa` / backend=sdpa forced | `mx.fast.sdpa` | ✓ |
| 700 | `sdpa` / return_attn_weights | `mx.fast.sdpa` | ✓ |
| 709 | `sdpa` / dropout | `mx.fast.sdpa` | ✓ |
| 727 | `mfa_bias_native` / attn_bias 1/2 | `_mfa_attn_bias_forward` → `_ext` bias kernel | ✓ (source) |
| 754 | `sdpa` / attn_bias 0/3 or unavail | `mx.fast.sdpa` | ✓ |
| 763 | `sage` / backend=sage | `sage_attention` → `_ext` sage kernel | ✓ (source) |
| 932 | `sdpa` / softcap (not mfa) | `_softcap_sdpa_ref` | ✓ |
| 935 | `sdpa` / alibi (not mfa/f32) | `_alibi_sdpa_ref` | ✓ |
| 943 | `sdpa` / return_lse not mfa-capable | `_fallback_sdpa_with_lse` | ✓ |
| 960 | `nax_dense` | `_make_v6nax_dense_custom` → `v6_nax_forward` (NAX) | ✓ |
| 962 | `_be` (=`sdpa`) | `_fallback_sdpa` | ✓ |
| 968 | `sdpa` / alibi f32 | `_alibi_sdpa_ref` | ✓ |
| 970 | `mfa_alibi` | `_mfa_alibi_forward` → `_ext` alibi kernel | ✓ (source) |
| 1000 | `sdpa` / window f32 | `mx.fast.sdpa` (masked) | ✓ |
| 1027 | `mfa_primitive` / return_lse | `_make_mfa_custom_lse` → `mfa_forward_with_lse` (REAL kernel) | ✓ |
| **1030** | **was `mfa_primitive` always** | `_mfa_forward`→`_make_mfa_custom`: **Apple SDPA** when `_v6nax_eligible & not force_kernel` (`attention.py:5642`), else `mfa_forward_with_lse` | **✗ → FIXED (CX-08)** |

**The single mismatch was line 1030** (the V6NAX-backward carve-out: forward
runs Apple SDPA, label said `mfa_primitive`). Fixed: the terminal now records
`apple_sdpa` for the eligible-and-not-forced subset (gated on
`_dtrace.recording()` → zero production overhead), `mfa_primitive` otherwise.
Forced `backend="mfa"` (`force_kernel=True`) skips the SDPA branch in `_impl`,
so it correctly stays `mfa_primitive`. Site 1027 was **already correct**
(`_make_mfa_custom_lse` runs the real `mfa_forward_with_lse`).

### Backward terminals (inside `_make_mfa_custom._backward` / vjp — fire on grad)

| line | recorded | binary | match? |
|---|---|---|---|
| 5728 | `v6_split_backward` | `_v6nax_backward_vjp` → V6NAX split kernels | ✓ |
| 5744 | `steel_backward` | `mfa_steel_backward` | ✓ |
| 5747 | `sdpa_vjp` | `mx.vjp(_fallback_sdpa)` | ✓ |
| 5754 | `sdpa_vjp` / softcap | `mx.vjp(_softcap_sdpa_ref)` | ✓ |

These are backward-only and intended; not part of the forward-terminal
mislabel. (The forward of a carve-out call records `apple_sdpa`; the grad of
the same call then records `v6_split_backward` — both true, different phases.)

### Varlen terminals (in `flash_attention_varlen`)

| 6198 `varlen_empty` · 6205/6244 `varlen_split_concat` · 6241 `varlen_native` | all ✓ |

---

## Part 2 — Every engagement / which-binary test, classified

Grep: `_dispatch_trace`, `mfa_primitive`/`apple_sdpa`, `byteΔ`/`byte_delta`,
`_assert_engaged`, `.capture()`.

| test file | class (before → after) | proof mechanism |
|---|---|---|
| `test_fingerprint_discipline.py` | byteΔ-proven | byteΔ>0 vs SDPA; proven-to-bite |
| `test_dispatch_map_lock.py` | byteΔ-proven | per-cell byte fingerprint |
| `test_sparse_family_correctness_lock.py` | byteΔ-proven | independent fp32 oracle |
| `test_backward_family_lock.py` | byteΔ-proven | fp32-vjp oracle + per-grad byteΔ |
| `test_b4_family_lock.py` | byteΔ-proven | per-kernel oracles |
| `test_v6_nax_forward_lock.py` | byteΔ-proven | fp32 + force-the-binary |
| `test_bf16_routing_all_nax_lock.py` | byteΔ-proven | byteΔ vs SDPA |
| `test_nax_routing_threshold_lock.py` | byteΔ-proven | byteΔ threshold |
| `test_sparse_bf16_v2_lock.py` | byteΔ-proven | byteΔ + fp32 oracle |
| `test_paged_oob_guard.py` | byteΔ-proven | `_ext` direct + gather oracle |
| `test_causal_maskzone_split_lock.py` | trace-label + fp64 oracle | asserts `mfa_primitive` (forced) + fp64 |
| `test_routing_equivalence_snapshot.py` | trace-label-trusted | golden snapshot (**labels corrected**) |
| `test_backward_routing_snapshot.py` | trace-label-trusted | backward golden |
| `test_attention.py` (warmup/L1818) | trace-label (`tr[-1]`) | asserts `tr[-1]` (robust) |
| **`test_dense_steel_family_lock.py`** | **trace/oracle → byteΔ-proven + source-predicate** | **added `_assert_engaged` (byteΔ>0 vs SDPA) + explicit RUNTIME-INDISTINGUISHABLE annotation (CC-07)** |
| **`test_v50_sprint_5b_section_b_topk_bisect.py`** | **isfinite-only → oracle-backed** | **numpy fp64 kth-largest + count oracle + top-k attention oracle (CC-06)** |
| **`test_engagement_proof_guard.py`** (NEW) | meta-guard | greps suite for fragile `tr[0]`/`len==1` reliance |

---

## Part 3 — Findings closed (completeness oracles → trustworthy)

| finding | sev | untrustworthy state | change | trustworthy now |
|---|---|---|---|---|
| **CX-08** | MED | terminal `mfa_primitive` while Apple SDPA runs the D=64 carve-out forward; golden locked the wrong label | `attention.py:1030` records `apple_sdpa` for eligible-not-forced; golden regenerated (10 D=64 N≥2048 cells flipped); cross-checked vs `dispatch-map.md:20` ("M5/NAX forward = SDPA") — **agree** | terminal = actual binary; golden + dispatch-map consistent |
| **CC-07** | MED | dense-STEEL correctness cells assert no which-binary (source-predicate only, implicit) | `_assert_engaged` byteΔ>0-vs-SDPA added to every forced cell; `# RUNTIME-INDISTINGUISHABLE` annotation made explicit on the byte-identical inter-variant lock | engagement byteΔ-proven; source-trust limitation explicit |
| **CC-15** | LOW | 8 spurious `('sdpa','fallback (not use_mfa)')` records per first MFA call could mislead `tr[0]`/`len==1` consumers | root cause = `_auto_warmup_background` (NOT the vjp closure the finding claimed); wrapped in `_dtrace.reentrant()` → records tagged `[reentrant]`; added suite-wide guard meta-test | records tagged + unmistakable; `tr[-1]` contract intact; guard prevents new fragile reliance |
| **CC-06** | MED | topk-bisect: 5/8 isfinite-only, 1 loose `diff<5.0` self-compare | numpy fp64 kth-largest+count oracle on the threshold cell; numpy top-k attention oracle on the output cells; self-compare cell now asserts BOTH paths vs the oracle | independent oracle; +0.5 perturbation bites |
| **CC-22** | LOW | doc-example `NameError` → silent skip (broken example masked) | only `# illustrative-fragment`-tagged snippets may skip; untagged NameError/IndentationError now FAIL; the 4 genuine fragments tagged in README/SERVING_GUIDE/API_MANUAL | broken examples fail loudly; fragments explicitly tagged |
| **CC-23** | LOW | native sparse-backward untested on M5 (honest skip) | documented below + `TODO(volet-D)` | gap tracked, tied to the M5 gate |

---

## Part 4 — Validation (every lock proven to BITE)

1. **Full suite:** `2333 passed, 91 skipped, 0 failed, 0 XPASS, 0 xfail` (56s).
   Baseline 2330 + 3 new guard tests. Collection (2333+91 = 2424) ≥ 1800. ✓
2. **BITE — trace label:** under `MFA_DISABLE_V6_DENSE=1` the D=128 N=2048
   cells reroute → `test_routing_equivalence_snapshot` raises
   `ROUTING REGRESSION at d128_N2048_caus_f16: golden ['nax_dense',...], now
   ['sdpa',...]`. The golden lock bites on a D128 label change. ✓ (env-scoped, auto-reverted)
3. **BITE — byteΔ engagement:** feeding the SDPA bytes (byteΔ=0) into
   `_assert_engaged` raises `ENGAGEMENT FAILURE: byteΔ-vs-SDPA == 0.0`. ✓
4. **BITE — topk oracle:** `threshold + 0.5` → p95|τ−kth| = 0.500, median
   count = 18 → both oracle assertions FAIL (kernel-correct values: err 0.0,
   count 64). ✓
5. **BITE — doc example:** an untagged `flash_attention(qqq,kkk,vvv)` snippet
   raises `NameError` (→ `pytest.fail`), no longer a silent skip. ✓
6. **dispatch-map byteΔ spot-check:** `test_dispatch_map_lock.py` (28 cells)
   passes; D128/N4096→NAX (Δ≠0), D128/N≤1024→SDPA (Δ=0), D64→SDPA (Δ=0) all
   match. ✓
7. **byteΔ-IDENTITY (no compute changed):** 48-cell forward+grad hash sweep
   over the dense + carve-out envelope (D∈{64,128,256}, N∈{512,2048,4096},
   causal both, f16/bf16) — **0 diffs before vs after**. The edits are
   telemetry-only: `record()` is a no-op without an open `capture()`, the A1
   eligibility re-eval is gated on `recording()`, and `reentrant()` only tags
   records. ✓

---

## Part 5 — CC-23: native sparse-backward M5 coverage gap (documented, not fabricated)

Native sparse-backward (`mfa_sparse_attention` backward / the STEEL sparse VJP)
is M1–M4-only. On this M5/NAX host, symmetric-mask `flash_attention_sparse`
backward routes to **SDPA-vjp** (the native sparse-backward kernel does not
engage), so the native kernel has **zero correctness coverage on M5**
(`tests/test_attention.py:1463` skips it honestly with that reason). This is a
real, un-fillable-on-this-host gap — not masked, not fabricated.

**`TODO(volet-D)`** — tie this to the volet-D M5/NAX release gate: either (a)
add an M1/M4 CI lane (or a self-hosted runner) that exercises native
sparse-backward vs an fp32-vjp oracle, or (b) record the gap explicitly in the
release-gate manifest so it is a conscious ship decision, not an invisible
hole. Cross-ref the CI green-but-untested gap (the M5-only locks never run on
the macOS-14 GitHub runner).

---

*Telemetry/test-only. No kernel math, dispatch decision, or valid-path output
changed (Part 4 #7). Commit on `fix/audit-remediation` only — not merged,
tagged, or published.*
