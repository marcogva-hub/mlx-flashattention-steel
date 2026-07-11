| dtype | D | N | window | arm | median ms | cos vs SDPA |
|---|---:|---:|---|---|---:|---:|
| fp16 | 64 | 2048 | small | gna_nax | 0.4687 | 1.000000 |
| fp16 | 64 | 2048 | small | sdpa_masked | 0.2830 | 1.000000 |
| fp16 | 64 | 2048 | large | gna_nax | 0.4278 | 1.000000 |
| fp16 | 64 | 2048 | large | sdpa_masked | 0.2702 | 1.000000 |
| fp16 | 64 | 8192 | small | gna_nax | 0.4549 | 1.000000 |
| fp16 | 64 | 8192 | small | sdpa_masked | 1.0046 | 1.000000 |
| fp16 | 64 | 8192 | large | gna_nax | 0.5433 | 1.000000 |
| fp16 | 64 | 8192 | large | sdpa_masked | 0.9659 | 1.000000 |
| fp16 | 128 | 2048 | small | gna_nax | 0.3893 | 1.000000 |
| fp16 | 128 | 2048 | small | sdpa_masked | 0.4090 | 1.000000 |
| fp16 | 128 | 2048 | small | gna_steel | 0.2864 | 1.000000 |
| fp16 | 128 | 2048 | large | gna_nax | 0.5109 | 1.000000 |
| fp16 | 128 | 2048 | large | sdpa_masked | 0.3825 | 1.000000 |
| fp16 | 128 | 2048 | large | gna_steel | 0.4207 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_nax | 0.3713 | 1.000000 |
| fp16 | 128 | 8192 | small | sdpa_masked | 1.3748 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_steel | 0.7984 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_nax | 0.6612 | 1.000000 |
| fp16 | 128 | 8192 | large | sdpa_masked | 1.4049 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_steel | 1.1044 | 1.000000 |
