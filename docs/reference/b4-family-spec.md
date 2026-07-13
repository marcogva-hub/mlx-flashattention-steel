# Specialized Kernel Families

This specification covers public families outside plain dense and block-sparse attention.

## GNA

`flash_attention_gna` implements neighborhood self-attention over an N-dimensional sequence shape. Q and KV may use GQA, but batch, sequence length, and head dimension must agree.

On NAX-capable M5 systems, 3-D f16/bf16 calls select `gna_v6nax` for D=128 with N>=2048 and D=64 with N>=4096. D=128 3-D cells outside the NAX threshold can use `gna_steel`. All other supported cells build a GNA block mask and enter sparse attention. `MFA_DISABLE_GNA_NATIVE=1` forces the differentiable sparse route.

## Conv3D

Transparent Conv3D hooks inspect layout, channel, kernel, stride, dilation, group, and dtype constraints before dispatch. Native MPP is the default only inside its gate. Pad-and-slice extensions remain β3 opt-ins. Hook telemetry records engagement and fallback counts.

## Top-k attention

The public top-k surface computes dynamic sparse selection and chooses among native and compositional helpers. Disable controls keep a correctness fallback available. Top-k masks are not interchangeable with caller-provided block masks.

## SageAttention and quantization

SageAttention quantizes selected intermediates but has its own scale and packing contract. General quantized matmul and quantization utilities are separate expert families; neither implies that public attention performs integer MMA.

## Paged attention

Paged kernels gather K/V through a block table. Public validation checks metadata capacity and shape unless a trust knob explicitly disables host checks. Page size must match the pool layout.

## Packed varlen

STEEL packed varlen serves f16/bf16 D<=256. D=512 and fp32 use per-segment split-concat. The V6 NAX variant is a narrow opt-in described in [dispatch-map.md](dispatch-map.md).

For causal segments with qL>kL, the public path intentionally keeps explicit per-segment SDPA even though the expert STEEL kernel implements the same causal convention correctly; measurement favored split-concat for that regime.
