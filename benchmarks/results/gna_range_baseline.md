| dtype | D | N | window | arm | median ms | cos vs SDPA |
|---|---:|---:|---|---|---:|---:|
| fp16 | 128 | 4096 | small | gna_nax | 0.2605 | 1.000000 |
| fp16 | 128 | 4096 | small | sdpa_masked | 0.6005 | 1.000000 |
| fp16 | 128 | 4096 | small | gna_steel | 0.4021 | 1.000000 |
| fp16 | 128 | 4096 | large | gna_nax | 0.3867 | 1.000000 |
| fp16 | 128 | 4096 | large | sdpa_masked | 0.6170 | 1.000000 |
| fp16 | 128 | 4096 | large | gna_steel | 0.5495 | 1.000000 |
| bf16 | 128 | 4096 | small | gna_nax | 0.2605 | 1.000000 |
| bf16 | 128 | 4096 | small | sdpa_masked | 0.6152 | 1.000000 |
| bf16 | 128 | 4096 | small | gna_steel | 0.3960 | 0.999999 |
| bf16 | 128 | 4096 | large | gna_nax | 0.3844 | 1.000000 |
| bf16 | 128 | 4096 | large | sdpa_masked | 0.6071 | 1.000000 |
| bf16 | 128 | 4096 | large | gna_steel | 0.5420 | 0.999999 |
