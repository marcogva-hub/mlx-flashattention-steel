# Release Verification — v2.9.2 (2026-03-13)

## Tests run

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest tests/ -q` | `698 passed, 12 warnings` |
| `.venv/bin/python -m pytest tests/test_inference_context.py -q` | `21 passed` |
| `.venv/bin/python -m pytest tests/test_sage_attention.py -q` | `30 passed` |
| `.venv/bin/python -m pytest tests/test_mlx_lm_integration.py -q` | `38 passed, 2 warnings` |

## Smoke checks (separate processes)

| Path | Result |
|---|---|
| V2 dense production auto route (`flash_attention(..., backend="auto")`) | pass |
| Sage specialized decode (`create_decode_runtime(backend="sage", quantized_kv=True)`) | pass |
| paged/shared-prefix runtime flows (`create_decode_runtime(...)`) | pass |

## Intentionally skipped

- No heavy benchmark matrices were re-run in this release-prep pass.
- Reason: this run is limited to verification, packaging, and publish readiness; no kernel or dispatch changes were introduced.
