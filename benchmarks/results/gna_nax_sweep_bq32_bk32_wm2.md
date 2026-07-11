| dtype | D | N | window | arm | median ms | cos vs SDPA |
|---|---:|---:|---|---|---:|---:|
| fp16 | 64 | 2048 | small | gna_nax | 0.4176 | 1.000000 |
| fp16 | 64 | 2048 | small | sdpa_masked | 0.3886 | 1.000000 |
| fp16 | 64 | 2048 | large | gna_nax | 0.2954 | 1.000000 |
| fp16 | 64 | 2048 | large | sdpa_masked | 0.3081 | 1.000000 |
| fp16 | 64 | 8192 | small | gna_nax | 0.3827 | 1.000000 |
| fp16 | 64 | 8192 | small | sdpa_masked | 0.9966 | 1.000000 |
| fp16 | 64 | 8192 | large | gna_nax | 0.4551 | 1.000000 |
| fp16 | 64 | 8192 | large | sdpa_masked | 0.9804 | 1.000000 |
| fp16 | 128 | 2048 | small | gna_nax | 0.4585 | 1.000000 |
| fp16 | 128 | 2048 | small | sdpa_masked | 0.4461 | 1.000000 |
| fp16 | 128 | 2048 | small | gna_steel | 0.2970 | 1.000000 |
| fp16 | 128 | 2048 | large | gna_nax | 0.4288 | 1.000000 |
| fp16 | 128 | 2048 | large | sdpa_masked | 0.3844 | 1.000000 |
| fp16 | 128 | 2048 | large | gna_steel | 0.4451 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_nax | 0.3386 | 1.000000 |
| fp16 | 128 | 8192 | small | sdpa_masked | 1.4088 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_steel | 0.7920 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_nax | 0.5062 | 1.000000 |
| fp16 | 128 | 8192 | large | sdpa_masked | 1.3834 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_steel | 1.1032 | 1.000000 |
