| dtype | D | N | window | arm | median ms | cos vs SDPA |
|---|---:|---:|---|---|---:|---:|
| fp16 | 64 | 4096 | small | gna_nax | 0.4704 | 1.000000 |
| fp16 | 64 | 4096 | small | sdpa_masked | 0.4816 | 1.000000 |
| fp16 | 64 | 4096 | large | gna_nax | 0.4635 | 1.000000 |
| fp16 | 64 | 4096 | large | sdpa_masked | 0.4669 | 1.000000 |
| fp16 | 128 | 4096 | small | gna_nax | 0.2709 | 1.000000 |
| fp16 | 128 | 4096 | small | sdpa_masked | 0.6789 | 1.000000 |
| fp16 | 128 | 4096 | small | gna_steel | 0.3990 | 1.000000 |
| fp16 | 128 | 4096 | large | gna_nax | 0.3664 | 1.000000 |
| fp16 | 128 | 4096 | large | sdpa_masked | 0.6733 | 1.000000 |
| fp16 | 128 | 4096 | large | gna_steel | 0.5960 | 1.000000 |
