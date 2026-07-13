# Documentation Claim Lock Map

This index connects published assertions to executable guards.

| Published assertion | Primary lock |
|---|---|
| package version and API export count | `tests/test_doc_accuracy_guards.py` |
| published-file allowlist and sdist contents | `tests/test_publish_surface_guard.py` |
| runtime route table | `tests/test_dispatch_map_lock.py` |
| performance claim identifiers | `tests/test_perf_claims_doc_sync.py` |
| public claim reachability | `tests/test_release_notes_perf_claims.py` |
| documentation examples do not leak environment state | `tests/test_doc_examples.py` |
| strict knob names and values | `tests/test_knob_registry.py`, `tests/test_audit_remediation_r3.py` |
| D=512 delegation | `tests/test_d512_test_surface.py`, varlen D512 delegation lock |
| sparse NAX route boundaries | sparse gate patch and dispatch-map lock tests |
| packed-varlen tile coherence | packed-varlen public routing locks |
| causal varlen qL>kL | public split-concat and expert STEEL correctness locks |
| GNA NAX route and escape | GNA routing and dispatch-map locks |
| no-LSE sparse conservation | sparse LSE forward locks |

## Lock semantics

A math-only comparison does not prove engagement. Route locks assert a terminal or compare against a known fallback fingerprint. Documentation that names a kernel must cite one of those locks.

Performance prose is additionally constrained by the measurement contract in [BENCHMARKS.md](BENCHMARKS.md). The claim registry preserves executable identifiers but delegates current numerical results to [RESULTS.md](../../RESULTS.md).
