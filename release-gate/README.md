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
