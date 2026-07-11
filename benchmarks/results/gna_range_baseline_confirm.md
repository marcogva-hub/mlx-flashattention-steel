| dtype | D | N | window | arm | median ms | cos vs SDPA |
|---|---:|---:|---|---|---:|---:|
| fp16 | 128 | 4096 | small | gna_nax | 0.2608 | 1.000000 |
| fp16 | 128 | 4096 | small | sdpa_masked | 0.6140 | 1.000000 |
| fp16 | 128 | 4096 | small | gna_steel | 0.3998 | 1.000000 |
| fp16 | 128 | 4096 | large | gna_nax | 0.3799 | 1.000000 |
| fp16 | 128 | 4096 | large | sdpa_masked | 0.7228 | 1.000000 |
| fp16 | 128 | 4096 | large | gna_steel | 0.5520 | 1.000000 |
| bf16 | 128 | 4096 | small | gna_nax | 0.2601 | 1.000000 |
| bf16 | 128 | 4096 | small | sdpa_masked | 0.6108 | 1.000000 |
| bf16 | 128 | 4096 | small | gna_steel | 0.3953 | 0.999999 |
| bf16 | 128 | 4096 | large | gna_nax | 0.3813 | 1.000000 |
| bf16 | 128 | 4096 | large | sdpa_masked | 0.6177 | 1.000000 |
| bf16 | 128 | 4096 | large | gna_steel | 0.5491 | 0.999999 |
