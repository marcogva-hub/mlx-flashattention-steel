# Feature coverage

Status reflects the current public surface, not the existence of an internal
probe.

| Feature | Public support | Native families | Fallback/limit |
|---|---|---|---|
| dense self/cross attention | yes | STEEL and V6 NAX | MLX SDPA covers unsupported or non-winning cells |
| causal/non-causal | yes | path-dependent | bottom-right-aligned, zero-clamped convention |
| GQA | yes when `Hq % Hkv == 0` | dense, sparse, GNA, varlen subsets | route-specific validation applies |
| D64/D128 | yes | broadest native coverage | exact route is shape/feature dependent |
| D256 | public correction | limited STEEL/expert coverage | automatic route may prefer MLX |
| D512 | public correction | no direct production kernel | delegated to MLX; direct expert rejects |
| fp16/bf16 | yes | principal native dtypes | beta routes remain envelope-specific |
| fp32 | public fallback coverage | selected scalar/reference paths | not evidence of NAX fp32 support |
| block sparse | yes | V6 NAX BT32, scalar coverage | measured gate; BT64 expands conditionally |
| sliding/local masks | yes | sparse or STEEL window path | route depends on public entry |
| GNA 3D | yes | V6 NAX and STEEL | 1D/2D use existing fallback behavior |
| packed varlen | yes | STEEL; narrow V6 NAX opt-in | split/concat for fp32/D512/outside gate |
| paged KV | yes | paged STEEL/TurboQuant kernels | metadata validation is default |
| dense backward | yes | D64 V6 split default; D128 opt-in | SDPA VJP elsewhere |
| sparse backward | yes | hybrid/full-native opt-ins | default remains SDPA VJP outside gates |
| additive bias | yes | native supported bias modes | unsupported modes delegate |
| RoPE | yes | separate MLX RoPE and retained fused expert path | fused route is not default |
| Sage int8 KV | yes | native Sage kernels | inference-oriented |
| TurboQuant | yes | packed paged kernels plus Python compression | supported bits/layouts are API-validated |
| SVDQuant | yes | W4A16 module composition | linear replacement is explicit |
| Conv3D hook | yes | M5 NAX/MPP eligible cells | original `mx.conv_general` otherwise |
| qmm V6 NAX | expert coverage | direct native binding | not public-default routed |
| FFN/GELU V6 NAX | expert coverage | direct native binding | not public-default routed |

## Production, beta and research

- **Default**: public auto routes with permanent dispatch/correctness locks.
- **Beta opt-in**: varlen V6 NAX, Conv3D pad/slice variants, sparse full-native
  backward, D128 backward and GNA `_pr1`.
- **Expert/research**: tile overrides, source dumps, qmm/linear probes and
  build-time int8/fp8 characterization.

No beta-OS measurement is promoted to a stable-OS claim without revalidation.
