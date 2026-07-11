| dtype | D | N | window | arm | median ms | cos vs SDPA |
|---|---:|---:|---|---|---:|---:|
| fp16 | 64 | 2048 | small | gna_nax | 0.4069 | 1.000000 |
| fp16 | 64 | 2048 | small | sdpa_masked | 0.2630 | 1.000000 |
| fp16 | 64 | 2048 | large | gna_nax | 0.5505 | 1.000000 |
| fp16 | 64 | 2048 | large | sdpa_masked | 0.2612 | 1.000000 |
| fp16 | 64 | 8192 | small | gna_nax | 0.3164 | 1.000000 |
| fp16 | 64 | 8192 | small | sdpa_masked | 1.0118 | 1.000000 |
| fp16 | 64 | 8192 | large | gna_nax | 0.4065 | 1.000000 |
| fp16 | 64 | 8192 | large | sdpa_masked | 0.9832 | 1.000000 |
| fp16 | 128 | 2048 | small | gna_nax | 0.2762 | 1.000000 |
| fp16 | 128 | 2048 | small | sdpa_masked | 0.4144 | 1.000000 |
| fp16 | 128 | 2048 | small | gna_steel | 0.2747 | 1.000000 |
| fp16 | 128 | 2048 | large | gna_nax | 0.3253 | 1.000000 |
| fp16 | 128 | 2048 | large | sdpa_masked | 0.3872 | 1.000000 |
| fp16 | 128 | 2048 | large | gna_steel | 0.4109 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_nax | 0.3090 | 1.000000 |
| fp16 | 128 | 8192 | small | sdpa_masked | 1.3832 | 1.000000 |
| fp16 | 128 | 8192 | small | gna_steel | 0.8017 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_nax | 0.5273 | 1.000000 |
| fp16 | 128 | 8192 | large | sdpa_masked | 1.3986 | 1.000000 |
| fp16 | 128 | 8192 | large | gna_steel | 1.1113 | 1.000000 |
