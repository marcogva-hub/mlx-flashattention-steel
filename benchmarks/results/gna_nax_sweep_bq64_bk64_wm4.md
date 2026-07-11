| dtype | D | N | window | arm | median ms | cos vs SDPA |
|---|---:|---:|---|---|---:|---:|
| fp16 | 64 | 2048 | small | gna_nax | 0.5831 | 1.000000 |
| fp16 | 64 | 2048 | small | sdpa_masked | 0.2732 | 1.000000 |
| fp16 | 64 | 2048 | large | gna_nax | 0.4293 | 1.000000 |
| fp16 | 64 | 2048 | large | sdpa_masked | 0.2692 | 1.000000 |
| fp16 | 64 | 8192 | small | gna_nax | 0.4598 | 1.000000 |
| fp16 | 64 | 8192 | small | sdpa_masked | 1.0946 | 1.000000 |
| fp16 | 64 | 8192 | large | gna_nax | 0.5602 | 1.000000 |
| fp16 | 64 | 8192 | large | sdpa_masked | 0.9789 | 1.000000 |
| fp16 | 128 | 2048 | small | gna_nax | 0.4798 | 1.000000 |
| fp16 | 128 | 2048 | small | sdpa_masked | 0.3921 | 1.000000 |
| fp16 | 128 | 2048 | small | gna_steel | 0.2870 | 1.000000 |
| fp16 | 128 | 2048 | large | gna_nax | 0.5103 | 1.000000 |
| fp16 | 128 | 2048 | large | sdpa_masked | 0.4368 | 1.000000 |
| fp16 | 128 | 2048 | large | gna_steel | 0.4391 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_nax | 0.4079 | 1.000000 |
| fp16 | 128 | 8192 | small | sdpa_masked | 1.4191 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_steel | 0.8045 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_nax | 0.8192 | 1.000000 |
| fp16 | 128 | 8192 | large | sdpa_masked | 1.4027 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_steel | 1.1307 | 1.000000 |
