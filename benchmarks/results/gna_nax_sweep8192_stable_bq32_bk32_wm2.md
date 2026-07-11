| dtype | D | N | window | arm | median ms | cos vs SDPA |
|---|---:|---:|---|---|---:|---:|
| fp16 | 64 | 8192 | small | gna_nax | 0.2871 | 1.000000 |
| fp16 | 64 | 8192 | small | sdpa_masked | 0.9634 | 1.000000 |
| fp16 | 64 | 8192 | large | gna_nax | 0.3418 | 1.000000 |
| fp16 | 64 | 8192 | large | sdpa_masked | 0.9838 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_nax | 0.2915 | 1.000000 |
| fp16 | 128 | 8192 | small | sdpa_masked | 1.3861 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_steel | 0.8294 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_nax | 0.4363 | 1.000000 |
| fp16 | 128 | 8192 | large | sdpa_masked | 1.3885 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_steel | 1.0884 | 1.000000 |
