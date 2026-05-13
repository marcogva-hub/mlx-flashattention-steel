# mlx-mfa specialized skills — installation

The three mlx-mfa skills live in `~/.claude/skills/` (Claude Code's
user-skill directory).  They are NOT checked into the `mlx-mfa-v2`
repo — the repo only documents their existence (`docs/skills/README.md`)
and references their behavior in CLAUDE_V6_NAX.md §AA.

## Fresh-machine installation

### Prerequisites

- Claude Code installed and configured
- `~/code/mlx-mfa-v2` checkout (or wherever your worktree lives)
- `.venv/` set up per `CLAUDE.md` "Canonical Python environment"
  (Sprint 1) — `bash scripts/check_venv.sh` should pass

### Option 1 — clone from a dotfiles repo

If you have a dotfiles / claude-config repo containing the skills:

```bash
cd ~/.claude/skills
git clone <your-skills-repo>/mlx-mfa-perf-audit
git clone <your-skills-repo>/mlx-mfa-release-audit
git clone <your-skills-repo>/mlx-mfa-bench-methodology
```

Each skill directory contains:
- `SKILL.md` — frontmatter + methodology (Claude Code loads this on
  invocation)
- `audit_runner.py` (perf-audit, release-audit) or
  `protocol_dispatcher.py` (bench-methodology) — CLI helper
- `README.md` — invocation patterns + examples

### Option 2 — manual transcription

If you don't have a dotfiles repo, copy each skill's contents from
the mlx-mfa-v2 Sprint 3 PR / commit history.  Sprint 3 commit
references:

- `feat(skills): /mlx-mfa-perf-audit skill via /skill-creator`
- `feat(skills): /mlx-mfa-release-audit skill via /skill-creator`
- `feat(skills): /mlx-mfa-bench-methodology skill via /skill-creator`

The repo's `docs/skills/README.md` reproduces the SKILL.md outline
for each.  Full source is in git history of the Sprint 3 branch.

## Repo-location override

Both `mlx-mfa-release-audit` and `mlx-mfa-bench-methodology` runners
default to `REPO_ROOT = ~/code/mlx-mfa-v2`.  Override via env var
if your checkout lives elsewhere:

```bash
export MLX_MFA_REPO_ROOT=~/dev/mlx-mfa-v2
# All subsequent skill invocations resolve REPO_ROOT from this env var
```

The runners validate `$MLX_MFA_REPO_ROOT/pyproject.toml` exists
before proceeding; if not, they exit with an actionable error.

`mlx-mfa-perf-audit` has no path coupling — it imports `mlx_mfa`
directly via Python's import resolution.

## Verification

After installation, smoke-test each skill from a fresh shell:

```bash
# Skill 1: perf-audit (dry-run on a known-reachable claim)
.venv/bin/python ~/.claude/skills/mlx-mfa-perf-audit/audit_runner.py \
  --claim-id smoke_test \
  --shape B=1,H=4,qL=4096,kL=4096,D=64,dtype=float16 \
  --env MFA_ENABLE_V34_BACKWARD=1 \
  --expected v34_backward \
  --documented-ratio 1.82 \
  --baseline sdpa_vjp
# Expected: verdict REACHABLE, exit 0

# Skill 2: release-audit (against current released v2.37.3)
.venv/bin/python ~/.claude/skills/mlx-mfa-release-audit/audit_runner.py \
  --target-version 2.37.3
# Expected: most checks PASS; ADVISORY on skill log if skills are
# brand-new in your environment; BLOCKED on test_suite if unrelated
# flakes exist

# Skill 3: bench-methodology (small canonical-regime shape)
.venv/bin/python ~/.claude/skills/mlx-mfa-bench-methodology/protocol_dispatcher.py \
  --shapes '[{"B":1,"H":4,"qL":2048,"kL":2048,"D":64,"is_backward":false}]' \
  --baseline sdpa --target flash_attention
# Expected: protocol_used = canonical_warmup_continuous (sub-1.5ms);
# verdict CONFIDENT or BOUNDARY
```

## Skill availability check

Claude Code lists installed skills.  After installation, verify
they're discoverable:

```
# In a Claude Code session:
mlx-mfa-perf-audit, mlx-mfa-release-audit, mlx-mfa-bench-methodology
# should all appear in the skill list (Available skills section)
```

The three skills are listed in the `Available skills` panel when
Claude Code starts a session in any directory.  Invocation patterns
match each SKILL.md's frontmatter `description` field.

## Updating skills

Skills are versioned only via git history of the skill directory.
For breaking changes:
1. Update the SKILL.md `description` to reflect new behavior
2. Bump the runner script (`audit_runner.py` / `protocol_dispatcher.py`)
3. Verify smoke-test commands above still produce expected output
4. Commit to the dotfiles / claude-config repo

The mlx-mfa-v2 repo's `docs/skills/README.md` should be updated
in sync if the new behavior diverges from the existing description.
