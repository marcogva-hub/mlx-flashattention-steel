| dtype | D | N | window | arm | median ms | cos vs SDPA |
|---|---:|---:|---|---|---:|---:|
| fp16 | 128 | 4096 | small | gna_nax | 0.4087 | 1.000000 |
| fp16 | 128 | 4096 | small | sdpa_masked | 0.6167 | 1.000000 |
| fp16 | 128 | 4096 | small | gna_steel | 0.3985 | 1.000000 |
| fp16 | 128 | 4096 | large | gna_nax | 0.3725 | 1.000000 |
| fp16 | 128 | 4096 | large | sdpa_masked | 0.6042 | 1.000000 |
| fp16 | 128 | 4096 | large | gna_steel | 0.5520 | 1.000000 |
| bf16 | 128 | 4096 | small | gna_nax | 0.2605 | 1.000000 |
| bf16 | 128 | 4096 | small | sdpa_masked | 0.6096 | 1.000000 |
| bf16 | 128 | 4096 | small | gna_steel | 0.3964 | 0.999999 |
| bf16 | 128 | 4096 | large | gna_nax | 0.3701 | 1.000000 |
| bf16 | 128 | 4096 | large | sdpa_masked | 0.6184 | 1.000000 |
| bf16 | 128 | 4096 | large | gna_steel | 0.5513 | 0.999999 |
