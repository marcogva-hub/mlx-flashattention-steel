"""Phase II-8 addendum — gate #9 PROGRAMMATIC: paired-MMA stride parity.

Pattern #9 (generator/dispatch tile-constant mismatch) has struck three
times through the SAME mechanism:
  1. KD-5: dKV-split dispatch BK != generator's hardcoded BK override.
  2. v2.39.1: fused-backward BK lowered to 16 on the Primitive side;
     the generator's paired 16x32x16 MMA (`ik += 2` over TK) reads past
     the tile at TK=1 — silent dK/dV corruption (II-6 CRITICAL).
  3. II-8 sweep: the V34 FORWARD has the same paired loop and an
     unguarded MFA_V6_V34_BK env override (fixed with the loud guard).

This test makes the class structurally detected (CI-static loud-failure
semantics, like the cache-key invariant tests): every `ik += 2`
paired-MMA emission site in the generators must have its BK source
guarded by a `% 32` evenness check on the dispatch side.

If you add a paired-MMA loop or a new BK knob and this test fails:
either route the BK through a guarded resolution point, or add the
`BK % 32` loud guard next to the new knob, then register the prefix in
the maps below.  Do NOT weaken the test.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_GEN = _REPO / "csrc" / "mfa" / "v6_nax" / "NAAttentionKernel.cpp"
_PRIM = _REPO / "csrc" / "mfa_v6_nax_primitive.cpp"
_ATTN = _REPO / "csrc" / "mfa_attention.cpp"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class TestPairedMMAStrideParity:
    def test_all_paired_sites_enumerated_and_guarded(self):
        gen = _read(_GEN)
        prim = _read(_PRIM)

        # 1. Enumerate every paired-MMA loop and its TK macro prefix.
        sites = re.findall(r"for \(short ik = 0; ik < (\w+_TK); ik \+= 2\)", gen)
        assert sites, "no paired-MMA sites found — parser broken or code moved"
        prefixes = sorted({s.rsplit("_TK", 1)[0] for s in sites})

        # Known prefixes -> the guard that covers their BK source.
        #   - V34BWD*/V34BWDF/V34BWDK/V34BWDV/V34BWDKV: every backward
        #     Primitive resolves BK then calls
        #     compile_v34_backward_pipeline(), which hard-rejects
        #     BK % 32 != 0 (II-6 guard).
        #   - V34 (forward): guarded at both MFA_V6_V34_BK env sites
        #     (II-8 addendum guard).
        backward_prefixes = {p for p in prefixes if p.startswith("V34BWD")}
        forward_prefixes = {p for p in prefixes if p == "V34"}
        unknown = set(prefixes) - backward_prefixes - forward_prefixes
        assert not unknown, (
            f"NEW paired-MMA prefixes {sorted(unknown)} found in the "
            f"generator with no registered BK guard — Pattern #9 risk. "
            f"Add a BK % 32 guard at the dispatch-side BK source and "
            f"register the prefix here."
        )

        # 2. The backward chokepoint guard exists and is in the shared
        #    helper every backward Primitive flows through.
        m = re.search(
            r"void\* compile_v34_backward_pipeline\((?:.|\n)*?\{((?:.|\n)*?)\n  // Build memoryPrecisions",
            prim,
        )
        assert m, "compile_v34_backward_pipeline not found"
        assert "BK % 32 != 0" in m.group(1), (
            "the II-6 BK % 32 guard is missing from "
            "compile_v34_backward_pipeline — backward paired-MMA sites "
            "are unprotected"
        )

        # 3. Every backward Primitive's pipeline compile goes through the
        #    guarded helper (no direct v34_compile bypasses for backward).
        n_helper_calls = len(re.findall(r"compile_v34_backward_pipeline\(", prim)) - 1
        n_bwd_env_knobs = len(re.findall(r'getenv\("MFA_V34BWD\w*_BK"\)', prim))
        assert n_helper_calls >= n_bwd_env_knobs, (
            f"{n_bwd_env_knobs} backward BK env knobs but only "
            f"{n_helper_calls} guarded-helper call sites — a backward "
            f"Primitive may bypass the BK guard"
        )

        # 4. The forward env knob is guarded at EVERY site where it is read.
        n_fwd_knobs = len(re.findall(r'getenv\("MFA_V6_V34_BK"\)', prim))
        n_fwd_guards = len(re.findall(
            r"BK must be a positive multiple of 32 \(paired", prim))
        assert n_fwd_knobs > 0, "forward MFA_V6_V34_BK knob disappeared — update this test"
        assert n_fwd_guards >= n_fwd_knobs, (
            f"{n_fwd_knobs} forward MFA_V6_V34_BK read sites but only "
            f"{n_fwd_guards} paired-MMA BK guards — an unguarded forward "
            f"override path exists (Pattern #9 third-site class)"
        )

    def test_kd5_dispatch_generator_bk_parity_locked(self):
        """KD-5 lock: the dispatch-side BK expression must mirror the
        generator's D-conditional override (the original Pattern #9)."""
        attn = _read(_ATTN)
        assert re.search(r"const int BK = \(D <= 64\) \? cfg\.BK : 16;", attn), (
            "the KD-5 dispatch-side BK expression changed — re-verify it "
            "mirrors the generator override before updating this lock "
            "(see docs/v50/campaign-2026-06/ KD-5 ledger)"
        )
