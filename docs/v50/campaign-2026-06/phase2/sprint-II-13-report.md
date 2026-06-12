# Sprint II-13 — Systematic Hook-Coverage Audit (2026-06-12)

**Status**: COMPLETE — coverage VERIFIED COMPLETE on M5 (zero remaining
gaps); Pattern #8 detection is now structural.

## R.1/R.2 — Enumeration + cross-check

`hook-coverage/01-call-paths.md` (committed): every user-facing entry
point (8 nn modules + direct mx ops + the VSR-portfolio idiom)
enumerated by source introspection and classified.  install_hooks()
patches exactly {mx.conv_general, mx.conv3d}.

Findings:
- **COVERED**: nn.Conv3d, mx.conv3d, mx.conv_general (5D), the
  SeedVR2-class portfolio idiom (Phase-96 telemetry corroborates).
- **CORRECTLY-UNHOOKED** (each with the verified reason): conv1d/2d
  (no acceleration exists), ConvTranspose3d (input-dilation class,
  envelope-ineligible), nn.RoPE/mx.fast.rope (mx.fast.rope IS the
  optimum — the repo's own fused rope was declined 4x against it),
  nn.QuantizedLinear (SVDQuant is an explicit tier-3 API),
  nn.MultiHeadAttention/mx.fast.sdpa **on M5 by Pattern #6** (Apple
  SDPA NAX owns dense forward; hooking it would route users to slower
  paths).
- **FLAGGED-GATED (not a fix)**: an M1/M2-gated SDPA hook (MFA fwd
  historically 1.6-2.2x on M1) is the one candidate this audit
  surfaces — it requires an M1 bench this machine cannot run.
  Recorded for the ledger; Pattern #6 forbids enabling unbenched.

## R.3 — Gaps fixed

None remaining: the single real gap of this class (nn.Conv3d) was
fixed in II-7 at the structural chokepoint (patching mx.conv3d itself
covers both the nn module and direct callers by construction).  The
audit confirms no other entry point reaches an accelerated op
unhooked.

## R.4 — Telemetry-backed enforcement (tests/test_phase2_ii13_hook_coverage.py)

1. **Engagement tests** — nn.Conv3d-as-a-user-would, direct mx.conv3d,
   and mx.conv_general each assert `executed > 0` via hook telemetry:
   the exact test class that would have caught the II-7 gap on day one.
2. **Completeness check** — install_hooks() must patch every op in the
   `_EXPECTED_PATCHED` registry (marker-verified); adding an
   acceleration without registering its surface fails the test.
3. **Anti-silence check** — an ineligible call must COUNT as fallback
   in telemetry (Pattern #8's original silent-fallback half).

## R.5 — Pattern #8 catalogue

Pattern #8 now reads: TWO exhibits — (1) v2.50.1 KD-6 silent-block
(hook crash absorbed downstream), (2) II-7 nn.Conv3d coverage gap
(hook patched a surface users don't call).  Detection is now
STRUCTURAL: telemetry-backed engagement tests + the patched-op
completeness registry + the anti-silence fallback check (this sprint),
mirroring Pattern #9's gate-#9 automation (II-8 addendum).

Suite: 1409 passed + 1 skipped x2.
