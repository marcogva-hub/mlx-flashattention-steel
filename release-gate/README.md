# release-gate/ — M5/NAX pre-tag gate receipts (tracked, sdist-excluded)

The mandatory M5/NAX release gate (`scripts/release_m5_nax_gate.py`, CLAUDE_V6_NAX.md
§AA.8) can only run on a real M5+ host — GitHub-hosted runners are M1 (NAX never
engages) and a self-hosted M5 runner on this PUBLIC repo is a security risk.

So the gate is run **by the maintainer on M5** at release time. It writes a
receipt here: `m5-gate-<version>.json` (git_sha, has_nax, is_m5_plus, gate verdict,
byteΔ fingerprints + their sha256, MLX/hardware/date). **Commit the receipt** as a
release-prep step before dispatching `publish.yml`.

`publish.yml` runs `scripts/check_m5_gate_fingerprint.py` as a FATAL precondition:
it BLOCKS the publish unless a receipt exists for the pyproject version, is PASS +
NAX-live + M5, and is FRESH — its `git_sha` is an ancestor of HEAD with **no
`csrc/` or `mlx_mfa/` change since** (so the published source == the gated source).

This directory is **tracked** (publish.yml's fresh checkout must see the receipt)
but **sdist-excluded** (it never ships to users).

> A missing receipt is the correct default for a held/unreleased version — it
> means "the M5 gate has not certified this exact source for release yet."

## metal_kernel x MLX-version matrix receipt (since 2.62.3)

`metal-kernel-matrix-<version>.json` is written by `scripts/metal_kernel_matrix_smoke.py`
(M5, from a clean committed tree): the release sdist is installed in isolation against
**every MLX version of the nanobind ABI table** and every shipped `metal_kernel` kernel is
swept over its variant axes against independent references. `publish.yml` GATE 6 and the
release audit (Check 10) run `scripts/check_metal_kernel_matrix.py`, which BLOCKS unless
the receipt covers every ABI-table version and probe, passes (strict known failures are
reported, never hidden), is bound to the HEAD tree (sdist build inputs == git), and is
fresh (no `csrc/`, `mlx_mfa/`, `CMakeLists.txt` or `pyproject.toml` change since).

Why: 2.62.2 opened MLX 0.32.1/0.32.2 while its install smoke exercised dense kernels only;
every NAX `metal_kernel` kernel failed to build there (MSL 4.1 on macOS 27) and shipped.

```bash
.venv/bin/python -m build --sdist
.venv/bin/python scripts/metal_kernel_matrix_smoke.py --sdist dist/mlx_mfa-<version>.tar.gz
git add release-gate/metal-kernel-matrix-<version>.json
```
