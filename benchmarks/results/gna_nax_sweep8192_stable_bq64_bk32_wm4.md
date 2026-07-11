| dtype | D | N | window | arm | median ms | cos vs SDPA |
|---|---:|---:|---|---|---:|---:|
| fp16 | 64 | 8192 | small | gna_nax | 0.2920 | 1.000000 |
| fp16 | 64 | 8192 | small | sdpa_masked | 0.9767 | 1.000000 |
| fp16 | 64 | 8192 | large | gna_nax | 0.3732 | 1.000000 |
| fp16 | 64 | 8192 | large | sdpa_masked | 0.9763 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_nax | 0.2910 | 1.000000 |
| fp16 | 128 | 8192 | small | sdpa_masked | 1.3877 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_steel | 0.8038 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_nax | 0.5070 | 1.000000 |
| fp16 | 128 | 8192 | large | sdpa_masked | 1.3899 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_steel | 1.0889 | 1.000000 |
