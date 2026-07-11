# Paged Varlen Multi-Seq Benchmark

- commit: `e2656df`
- mlx: `0.31.2`
- device: `Device(gpu, 0)`
- sessions/arm: `5`, warmup: `3`, iters/session: `12`

| scenario | causal | dtype | D | page-native ms | materialize ms | sdpa/seq ms | native/materialize | public/materialize |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode4_hetero_kv | yes | fp16 | 64 | 2.0777 | 0.4506 | 0.2675 | 4.611x | 3.143x |
| decode4_hetero_kv | no | fp16 | 64 | 1.9646 | 0.4448 | 0.2317 | 4.416x | 3.104x |
| decode4_hetero_kv | yes | fp16 | 128 | 3.8664 | 0.5631 | 0.2837 | 6.867x | 2.763x |
| decode4_hetero_kv | no | fp16 | 128 | 3.8615 | 0.5650 | 0.2349 | 6.835x | 2.719x |
| decode4_hetero_kv | yes | bf16 | 64 | 1.9690 | 0.4540 | 0.2765 | 4.337x | 2.997x |
| decode4_hetero_kv | no | bf16 | 64 | 1.9180 | 0.4285 | 0.2373 | 4.476x | 3.194x |
| decode4_hetero_kv | yes | bf16 | 128 | 3.7806 | 0.5620 | 0.2787 | 6.727x | 2.716x |
| decode4_hetero_kv | no | bf16 | 128 | 3.6744 | 0.5633 | 0.2348 | 6.523x | 2.716x |
| decode8_hetero_kv | yes | fp16 | 64 | 1.9179 | 0.5784 | 0.3492 | 3.316x | 2.803x |
| decode8_hetero_kv | no | fp16 | 64 | 1.9670 | 0.5721 | 0.2596 | 3.438x | 2.813x |
| decode8_hetero_kv | yes | fp16 | 128 | 3.7211 | 0.8230 | 0.3941 | 4.521x | 2.384x |
| decode8_hetero_kv | no | fp16 | 128 | 3.9733 | 0.8402 | 0.2944 | 4.729x | 2.321x |
| decode8_hetero_kv | yes | bf16 | 64 | 2.0534 | 0.6168 | 0.3886 | 3.329x | 2.809x |
| decode8_hetero_kv | no | bf16 | 64 | 2.0198 | 0.6164 | 0.2851 | 3.277x | 2.789x |
| decode8_hetero_kv | yes | bf16 | 128 | 3.8893 | 0.8316 | 0.3816 | 4.677x | 2.318x |
| decode8_hetero_kv | no | bf16 | 128 | 3.8768 | 0.8295 | 0.3134 | 4.674x | 2.312x |
| prefill4_hetero_qkv | yes | fp16 | 64 | 2.0067 | 0.5869 | 0.5609 | 3.419x | 3.971x |
| prefill4_hetero_qkv | no | fp16 | 64 | 1.9879 | 0.5704 | 0.2893 | 3.485x | 4.025x |
| prefill4_hetero_qkv | yes | fp16 | 128 | 3.6171 | 0.9319 | 0.7640 | 3.882x | 4.494x |
| prefill4_hetero_qkv | no | fp16 | 128 | 3.6677 | 0.9395 | 0.4995 | 3.904x | 4.341x |
| prefill4_hetero_qkv | yes | bf16 | 64 | 1.9458 | 0.5794 | 0.5761 | 3.358x | 3.980x |
| prefill4_hetero_qkv | no | bf16 | 64 | 1.8908 | 0.5853 | 0.2799 | 3.230x | 3.961x |
| prefill4_hetero_qkv | yes | bf16 | 128 | 3.7525 | 0.9775 | 0.8366 | 3.839x | 4.217x |
| prefill4_hetero_qkv | no | bf16 | 128 | 3.6134 | 0.9319 | 0.4986 | 3.878x | 4.416x |
| mixed8_hetero_qkv | yes | fp16 | 64 | 1.1561 | 0.5149 | 0.4466 | 2.245x | 3.080x |
| mixed8_hetero_qkv | no | fp16 | 64 | 1.2183 | 0.5327 | 0.3216 | 2.287x | 3.063x |
| mixed8_hetero_qkv | yes | fp16 | 128 | 2.0366 | 0.8287 | 0.6350 | 2.458x | 3.012x |
| mixed8_hetero_qkv | no | fp16 | 128 | 2.1109 | 0.8295 | 0.4769 | 2.545x | 3.098x |
| mixed8_hetero_qkv | yes | bf16 | 64 | 1.2270 | 0.5596 | 0.4766 | 2.192x | 2.911x |
| mixed8_hetero_qkv | no | bf16 | 64 | 1.2218 | 0.5653 | 0.3343 | 2.162x | 2.888x |
| mixed8_hetero_qkv | yes | bf16 | 128 | 2.0823 | 0.8628 | 0.6428 | 2.414x | 2.944x |
| mixed8_hetero_qkv | no | bf16 | 128 | 2.0366 | 0.8015 | 0.4716 | 2.541x | 3.068x |
