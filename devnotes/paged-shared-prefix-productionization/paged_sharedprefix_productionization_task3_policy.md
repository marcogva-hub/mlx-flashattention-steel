# Task 3 Policy Decision — Paged Decode Auto Routing

Date: 2026-03-12  
Artifact: `devnotes/paged_sharedprefix_matrix_latest.json`

Decision:
- Keep paged decode **explicit-only** for now.
- Do **not** add broad auto paging logic in `backend="auto"`.
- Do **not** add `MFA_FORCE_PAGED_DECODE` in this pass.

Why:
- Paged decode steady-state matrix (`paged_step`) is not benchmark-backed for auto
  promotion: `clear_win=0`, `maybe_win=1`, `no_win=1`, `losing=28`.
- Paged setup path (`paged_setup`) is also consistently slower (`losing=10/10`).
- The one near-threshold row (`~1.04x`) is not strong enough to justify default
  policy complexity or broad routing risk.

Operational guidance:
- Keep using paged mode as an explicit serving/runtime optimization where memory
  behavior is a primary concern.
- Revisit auto policy only after a higher-fidelity matrix shows a stable and
  narrow paged winning regime.
