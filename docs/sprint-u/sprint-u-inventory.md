# Sprint U — Inventory

## Foundation
- master tip pre-Sprint-U: `8ca029f` (v2.35.0 production)
- Sprint U goal: codify auto-default principle + unify main user path

## New artifacts

| File | Section | Purpose |
|---|---|---|
| `docs/RELEASE_PHILOSOPHY.md` (207 LOC) | A | Canonical auto-default principle |
| `CLAUDE.md` (amendment) | A | Auto-default principle reminder for CC |
| `CLAUDE_V6_NAX.md` §5.X (amendment) | A | Pre-tag auto-default audit checklist |
| `mlx_mfa/attention.py` (modified) | B | M5+ symmetric-BT mask auto-routes to dispatcher |
| `tests/test_sprint_u_sparse_routing.py` (4 tests) | B | Three-axis validation |
| `mlx_mfa/_auto_hooks.py` (NEW, 222 LOC) | C | install_hooks / uninstall_hooks / hooks_status + Conv3D hook |
| `mlx_mfa/__init__.py` (modified) | C | enable() / disable() / hooks_status() + auto-install at import |
| `tests/test_sprint_u_auto_hooks.py` (9 tests) | C | Three-axis validation |
| `README.md` (rewritten) | D | New Minimal Usage + Three usage levels + Disabling section |
| `CHANGELOG.md` [2.36.0] | D | Full release entry with migration notes |
| `pyproject.toml` 2.35.0 → 2.36.0 | E | Multi-SoT version bump |
| `mlx_mfa/__init__.py:30` 2.35.0 → 2.36.0 | E | Multi-SoT version bump |
| `docs/sprint-u/sprint-u-{inventory,decisions,results}.md` | F | 5-deliverables docs |
| `devnotes/SESSION_LOG.md` entry | F | Sprint log |

## Test count growth
- Pre-Sprint-U baseline: 52 LCSA + integration + V2 tests
- Sprint U adds:
  - 4 Section B sparse-auto-routing tests
  - 9 Section C auto-hook tests
- Total: **65/65 pass** at v2.36.0

## Hardware
- M5 Max 128GB, macOS 26.5, iStat performance fan profile
- MLX 0.31.2, mlx_mfa 2.36.0
