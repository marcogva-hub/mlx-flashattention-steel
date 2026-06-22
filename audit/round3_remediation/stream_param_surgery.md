# Stream/Device Param Surgery — CX-06 (pre-cut)

> Branch `fix/audit-remediation`, base HEAD `1632587` (after volets A–E), host
> M5 Max / macOS 26.6 / MLX 0.31.2. Every exported `_ext` binding that exposed a
> `stream` / `StreamOrDevice` / `device` parameter, verified non-functional and
> removed so no binding advertises a parameter it cannot honor. byteΔ-identity on
> the 48-cell dense+grad valid envelope (a no-op param removed changes nothing).
> Line numbers verified at source (RULE 16).

## Enumeration (grep `csrc/bindings.cpp` — the only binding TU)

**16 bindings exposed a `stream` param.** "Functional?" = passing a real
`mx.Stream` → does it work? All 16: **NO** (TypeError on the `StreamOrDevice`
variant — no caster registered in this extension's nanobind domain — or, for the
2 `nb::object` ones, silently accepted and ignored). None thread the stream.

| binding | C++ kind | param type | functional? (real mx.Stream) | removed? |
|---|---|---|---|---|
| `mfa_attention_forward` | free-fn | `optional<StreamOrDevice>` | TypeError [incompatible] | ✓ |
| `mfa_attention_alibi_forward` | free-fn | `optional<StreamOrDevice>` | TypeError | ✓ |
| `mfa_attention_bias_forward` | free-fn | `optional<StreamOrDevice>` | TypeError | ✓ |
| `mfa_attention_rope_forward` | free-fn | `optional<StreamOrDevice>` | TypeError | ✓ |
| `mfa_attention_sparse_forward` | lambda | `optional<StreamOrDevice>` | TypeError | ✓ |
| `mfa_attention_sparse_forward_with_lse` | lambda | `optional<StreamOrDevice>` | TypeError | ✓ |
| `mfa_gna_forward` | lambda | `optional<StreamOrDevice>` | TypeError | ✓ |
| `mfa_attention_varlen_forward` | lambda | `optional<StreamOrDevice>` | TypeError | ✓ |
| `mfa_paged_kv_gather` | lambda | `optional<StreamOrDevice>` | TypeError | ✓ |
| `mfa_paged_steel_forward` | lambda | `optional<StreamOrDevice>` | TypeError | ✓ |
| `mfa_sage_forward` | lambda | `optional<StreamOrDevice>` | TypeError | ✓ |
| `mfa_quantize_per_block` | lambda | `optional<StreamOrDevice>` | TypeError | ✓ |
| `mfa_smooth_quantize_k` | lambda | `optional<StreamOrDevice>` | TypeError | ✓ |
| `mfa_scatter_kv` | lambda | `optional<StreamOrDevice>` | TypeError | ✓ |
| `mfa_paged_varlen_forward` | lambda | **`nb::object`** (ignored) | **accepted + ignored** (always default stream) | ✓ |
| `mfa_paged_varlen_tq_forward` | lambda | **`nb::object`** (ignored) | **accepted + ignored** | ✓ |

**16 found · 16 removed · 0 flagged functional.** The cited trio
(`mfa_attention_forward`, `mfa_quantize_per_block`, `mfa_paged_kv_gather`) are a
subset; the full surface is the 16 above. The 2 `nb::object` paged-varlen
bindings were the worst — they accepted *any* object as `stream` and silently
discarded it (`auto s = default_stream(gpu)`; the object never read).

## How removed (all in `csrc/bindings.cpp` — no free-fn signatures touched)
- **4 free-fns** (`&mlx_mfa::fn` binds): wrapped in a lambda calling the free fn
  *without* stream (the C++ param defaults to `std::nullopt` = the only path that
  ever worked); no change to the free function or its (zero) other callers.
- **10 `StreamOrDevice` lambdas**: dropped the `optional<StreamOrDevice> stream`
  param + `nb::arg("stream")`; the forwarding either passes `std::nullopt` to the
  host fn (3 pass-through cases) or resolves `auto s = default_stream(Device::gpu)`
  (7 resolve cases) — both are exactly the old omit-path value.
- **2 `nb::object` lambdas**: dropped the param + `nb::arg`; body already used
  `default_stream(gpu)` (unchanged).
- Removed the volet-C CX-06 docstring note from `mfa_attention_forward` (the param
  it described is gone).

## Internal callers fixed (RULE-16 correction)
The volet-C note expected "no internal caller passes stream" — **wrong**: a broad
scan (keyword *and* trailing-positional) of `mlx_mfa/` found **3** raw-binding
calls that passed a (no-op) stream, which broke once the params were removed:
- `attention.py` `_sage_fwd(..., window_left, window_right, stream)` ×2 (positional)
- `attention.py` `_gna_fwd(..., stream=stream)` (keyword)
All three dropped the stream arg (the kernels always ran on the default stream).
Every `stream=` elsewhere in `mlx_mfa/` is to the **public Python API**
(`flash_attention` etc.), which keeps its own `stream` param — out of scope.

## Validation (bite-proven)
1. `_ext` rebuilds clean (absolute `-DPython_EXECUTABLE` path), loads, `has_nax()` True.
2. **byteΔ-identity**: 48-cell dense+grad sweep (D∈{64,128,256} × N × causal × fp16/bf16
   forward + grad) — **0 diffs** vs the pre-surgery baseline. A removed no-op param changes nothing.
3. **All 16 bindings**: signature carries no `stream` (16/16); passing `stream=` raises
   `TypeError` (16/16). Locked by `tests/test_cx06_no_stream_param.py`
   (`test_binding_rejects_stream_kwarg`).
4. **Regression lock**: `test_no_internal_caller_passes_stream_to_a_binding` re-runs the
   broad scan → 0 (re-adding a `stream=` call to a raw binding fails CI).
5. **Bite proof** (C++ rebuild, post-commit, RULE-2b-safe): re-adding the `stream` param to
   one binding → `test_binding_rejects_stream_kwarg[that-binding]` FAILS (the kwarg is now
   accepted); `git checkout` + rebuild restores it. Result recorded below.
6. Full suite `2404 passed, 91 skipped, 0 failed, 0 XPASS`.

Bite-proof result: **PASS** (post-commit, RULE-2b-safe). Re-adding the
`StreamOrDevice stream` param to `mfa_quantize_per_block` + rebuild →
`test_binding_signature_has_no_stream_param[mfa_quantize_per_block]` **FAILED**
(the signature again advertises `stream`); `git checkout` + rebuild restored it →
33/33 locks pass, byteΔ-identical. (The signature lock is the stronger guard: the
`rejects_kwarg` lock alone would *not* bite on a re-added `StreamOrDevice` param,
since that param is itself still non-functional — exactly the CX-06 bug — which is
why the honest fix is removal, not "document it".)

## Micro-break
Raw `_ext` callers passing `stream=None` (or positionally) now get TypeError. This is a
raw-`_ext` surface and the param was always a no-op (ran on default stream). Public
`mlx_mfa.*` APIs are unaffected.

---
*Signature cleanup only; no op's execution stream or output changed (#2). Commit on
`fix/audit-remediation` only.*
