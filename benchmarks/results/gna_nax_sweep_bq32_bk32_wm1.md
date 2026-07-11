| dtype | D | N | window | arm | median ms | cos vs SDPA |
|---|---:|---:|---|---|---:|---:|
| fp16 | 64 | 2048 | small | gna_nax | 0.3799 | 1.000000 |
| fp16 | 64 | 2048 | small | sdpa_masked | 0.2634 | 1.000000 |
| fp16 | 64 | 2048 | large | gna_nax | 0.4917 | 1.000000 |
| fp16 | 64 | 2048 | large | sdpa_masked | 0.2600 | 1.000000 |
| fp16 | 64 | 8192 | small | gna_nax | 0.4809 | 1.000000 |
| fp16 | 64 | 8192 | small | sdpa_masked | 0.9729 | 1.000000 |
| fp16 | 64 | 8192 | large | gna_nax | 0.6646 | 1.000000 |
| fp16 | 64 | 8192 | large | sdpa_masked | 0.9996 | 1.000000 |
| fp16 | 128 | 2048 | small | gna_nax | 0.4183 | 1.000000 |
| fp16 | 128 | 2048 | small | sdpa_masked | 0.4250 | 1.000000 |
| fp16 | 128 | 2048 | small | gna_steel | 0.2813 | 1.000000 |
| fp16 | 128 | 2048 | large | gna_nax | 0.7497 | 1.000000 |
| fp16 | 128 | 2048 | large | sdpa_masked | 0.4254 | 1.000000 |
| fp16 | 128 | 2048 | large | gna_steel | 0.4399 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_nax | 0.4411 | 1.000000 |
| fp16 | 128 | 8192 | small | sdpa_masked | 1.4082 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_steel | 0.8039 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_nax | 1.0706 | 1.000000 |
| fp16 | 128 | 8192 | large | sdpa_masked | 1.3767 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_steel | 1.0990 | 1.000000 |
