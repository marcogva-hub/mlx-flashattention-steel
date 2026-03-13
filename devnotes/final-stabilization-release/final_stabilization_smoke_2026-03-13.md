# Final Stabilization Smoke Pass (2026-03-13)

## Scope

Final pre-merge verification for stabilization/polish changes only.
No new kernel experiments or dispatch-policy expansions were included.

## Tests Run

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest tests/ -q` | `698 passed, 12 warnings` |
| `.venv/bin/python -m pytest tests/test_inference_context.py -q` | `21 passed` |
| `.venv/bin/python -m pytest tests/test_sage_attention.py -q` | `30 passed` |
| `.venv/bin/python -m pytest tests/test_mlx_lm_integration.py -q` | `38 passed, 2 warnings` |

## Targeted Smoke Checks (separate processes)

| Path | Command style | Result |
|---|---|---|
| V2 production auto route | standalone `python - <<'PY'` using `flash_attention(..., backend="auto")` on `D=128,N=2048,causal=True` | pass (`v2_auto_smoke_ok`) |
| Sage specialized decode | standalone `python - <<'PY'` using `create_decode_runtime(backend="sage", quantized_kv=True)` + prefill/step | pass (`sage_decode_smoke_ok`) |
| Paged + shared-prefix runtime flows | standalone `python - <<'PY'` using paged runtime prefill/step and dense runtime `prefill_shared_prefix` + `decode_from_shared_prefix` | pass (`paged_sharedprefix_smoke_ok`) |

## Not Re-run in this stabilization pass

- Heavy benchmark matrices (D=256/D=512/native-backward/experimental triage) were not re-run.
- Rationale: out of scope for a polish/merge pass and no performance-path logic changes were introduced.

