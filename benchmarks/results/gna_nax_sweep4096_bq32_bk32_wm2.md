| dtype | D | N | window | arm | median ms | cos vs SDPA |
|---|---:|---:|---|---|---:|---:|
| fp16 | 64 | 4096 | small | gna_nax | 0.4753 | 1.000000 |
| fp16 | 64 | 4096 | small | sdpa_masked | 0.5492 | 1.000000 |
| fp16 | 64 | 4096 | large | gna_nax | 0.2683 | 1.000000 |
| fp16 | 64 | 4096 | large | sdpa_masked | 0.4502 | 1.000000 |
| fp16 | 128 | 4096 | small | gna_nax | 0.3607 | 1.000000 |
| fp16 | 128 | 4096 | small | sdpa_masked | 0.6939 | 1.000000 |
| fp16 | 128 | 4096 | small | gna_steel | 0.4025 | 1.000000 |
| fp16 | 128 | 4096 | large | gna_nax | 0.4281 | 1.000000 |
| fp16 | 128 | 4096 | large | sdpa_masked | 0.6951 | 1.000000 |
| fp16 | 128 | 4096 | large | gna_steel | 0.5791 | 1.000000 |
