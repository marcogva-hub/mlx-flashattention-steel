| dtype | D | N | window | arm | median ms | cos vs SDPA |
|---|---:|---:|---|---|---:|---:|
| fp16 | 128 | 4096 | small | gna_nax | 0.2624 | 1.000000 |
| fp16 | 128 | 4096 | small | sdpa_masked | 0.6109 | 1.000000 |
| fp16 | 128 | 4096 | small | gna_steel | 0.4040 | 1.000000 |
| fp16 | 128 | 4096 | large | gna_nax | 0.3715 | 1.000000 |
| fp16 | 128 | 4096 | large | sdpa_masked | 0.6122 | 1.000000 |
| fp16 | 128 | 4096 | large | gna_steel | 0.5485 | 1.000000 |
| bf16 | 128 | 4096 | small | gna_nax | 0.2604 | 1.000000 |
| bf16 | 128 | 4096 | small | sdpa_masked | 0.6112 | 1.000000 |
| bf16 | 128 | 4096 | small | gna_steel | 0.3934 | 0.999999 |
| bf16 | 128 | 4096 | large | gna_nax | 0.3756 | 1.000000 |
| bf16 | 128 | 4096 | large | sdpa_masked | 0.6174 | 1.000000 |
| bf16 | 128 | 4096 | large | gna_steel | 0.5471 | 0.999999 |
