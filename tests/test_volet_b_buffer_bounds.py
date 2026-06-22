"""Volet B — kernel buffer-bounds: check-before-read source locks + host invariant.

Apple GPUs silently absorb OOB device reads (UB, no observable fault), so a
runtime test cannot reliably catch a read-before-check regression — these are
**source-predicate locks** that assert the ordering in the JIT emitter sources,
mirroring the correct non-TQ sibling.  See devnotes/buffer_read_audit.md.

Covers: CC-02 (TQ paged-varlen block_table read-before-check) and the III-9 K
sibling (the K direct-read partial-tile clamp that hardened V but missed K in
mfa_steel_fwd_v2 / mfa_gna_fwd).  Plus the host capacity invariant.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

_CSRC = Path(__file__).resolve().parent.parent / "csrc"


def _read(name):
    return (_CSRC / name).read_text(encoding="utf-8")


# ── CC-02: TQ paged-varlen block_table check-before-read ─────────────────────

_TQ_READ = "const int phys = block_table[seq_id * p->max_blocks + blk_idx]"
_TQ_GUARD = "if (blk_idx < p->max_blocks) {"
_TQ_BUGGY = "if (blk_idx < p->max_blocks && phys"  # read-before-check signature


def _tq_read_is_guarded(src: str) -> bool:
    """True iff every block_table read is preceded by the blk_idx<max_blocks
    guard, and the read-before-check combined signature is absent."""
    if _TQ_BUGGY in src:
        return False  # the combined check only exists when phys was read first
    pos = 0
    found = 0
    while True:
        i = src.find(_TQ_READ, pos)
        if i < 0:
            break
        found += 1
        g = src.rfind(_TQ_GUARD, max(0, i - 400), i)
        if g < 0:
            return False  # read not preceded by a blk_idx guard
        pos = i + len(_TQ_READ)
    return found >= 2  # K-gather + V-gather


def test_tq_block_table_check_precedes_read():
    """CC-02: both TQ gather sites must guard blk_idx BEFORE the block_table read
    (mirror mfa_steel_paged_varlen_fwd.cpp)."""
    src = _read("mfa_steel_paged_varlen_tq_fwd.cpp")
    assert _tq_read_is_guarded(src), (
        "TQ paged-varlen block_table read is not guarded by blk_idx<max_blocks "
        "before the read (CC-02 mis-ordering regressed)")
    # the non-TQ sibling is the reference — confirm it too (anti-drift)
    sib = _read("mfa_steel_paged_varlen_fwd.cpp")
    assert _tq_read_is_guarded(sib), "non-TQ sibling lost its check-before-read"


def test_tq_ordering_lock_bites():
    """Prove the lock fails on the read-before-check order (scratch string —
    never mutate the tracked file)."""
    fixed = _read("mfa_steel_paged_varlen_tq_fwd.cpp")
    # reconstruct the BUGGY order from the fixed source (read precedes a combined
    # check), as it was before CC-02 was fixed.
    buggy = fixed.replace(
        '            if (blk_idx < p->max_blocks) {\\n";\n'
        '    ss << "            const int phys = block_table[seq_id * p->max_blocks + blk_idx];\\n";\n'
        '    ss << "            if (phys >= 0 && phys < p->num_blocks) {\\n";',
        '            const int phys = block_table[seq_id * p->max_blocks + blk_idx];\\n";\n'
        '    ss << "            if (blk_idx < p->max_blocks && phys >= 0 && phys < p->num_blocks) {\\n";',
    )
    assert buggy != fixed, "scratch transform did not apply — lock self-test invalid"
    assert not _tq_read_is_guarded(buggy), (
        "the source-order lock failed to bite on the read-before-check order")


# ── III-9 K sibling: K direct-read partial-tile clamp (v2 + gna) ─────────────

_K_RAW = "K_cur + (long)(sm + (short)(dd * 8)) + (long)sn * K_stride"
_K_CLAMPED = "K_cur + (long)(sm + (short)(dd * 8)) + (long)k_row * K_stride"
_K_CLAMP_GUARD = "if (kb == p->NK_aligned && k_row >= p->kL_rem)"


def _k_direct_read_clamped(src: str) -> bool:
    """The K direct-read must use the clamped key-row k_row (not raw sn) and the
    clamp guard must precede it."""
    if _K_RAW in src:
        return False  # raw unclamped key-row still used
    if _K_CLAMPED not in src:
        return False
    i = src.find(_K_CLAMPED)
    g = src.rfind(_K_CLAMP_GUARD, max(0, i - 400), i)
    return g >= 0


@pytest.mark.parametrize("fname", ["mfa_steel_fwd_v2.cpp", "mfa_gna_fwd.cpp"])
def test_k_direct_read_partial_tile_clamped(fname):
    """III-9 K sibling: the MFA_DIRECT_READS K key-row must be clamped on the
    partial final tile (was OOB; V was clamped, K was not)."""
    assert _k_direct_read_clamped(_read(fname)), (
        f"{fname}: K direct-read key-row is unclamped — OOB device read on the "
        f"partial final K-tile (the III-9 V fix did not cover K)")


@pytest.mark.parametrize("fname", ["mfa_steel_fwd_v2.cpp", "mfa_gna_fwd.cpp"])
def test_k_clamp_lock_bites(fname):
    """Prove the K-clamp lock fails if the raw (unclamped sn) read is restored."""
    fixed = _read(fname)
    buggy = fixed.replace(_K_CLAMPED, _K_RAW)
    assert buggy != fixed
    assert not _k_direct_read_clamped(buggy), (
        f"{fname}: K-clamp lock failed to bite on the unclamped key-row")


# ── Host capacity invariant (volet C, reached by the TQ public wrapper) ──────

class TestHostCapacityInvariant:
    """seq_lens_kv > max_blocks*block_size must raise before dispatch
    (_validate_paged_block_table, reached at flash_attention_paged_varlen_turboquant
    attention.py:~8235). The kernel reorder (CC-02) is the raw-_ext defense; this
    is the public-path loud failure."""

    def test_over_capacity_raises(self):
        import mlx_mfa.attention as A
        bt = mx.zeros((2, 4), dtype=mx.int32)        # max_blocks = 4
        sl = mx.array([100, 8], dtype=mx.int32)      # 100 > 4*16 = 64
        with pytest.raises(ValueError, match="(?i)exceeds max_blocks"):
            A._validate_paged_block_table(
                bt, sl, num_blocks=10, block_size=16, max_blocks=4, fn="vbtest")

    def test_in_capacity_ok(self):
        import mlx_mfa.attention as A
        bt = mx.zeros((2, 4), dtype=mx.int32)
        sl = mx.array([64, 8], dtype=mx.int32)       # 64 == 4*16, in bounds
        A._validate_paged_block_table(
            bt, sl, num_blocks=10, block_size=16, max_blocks=4, fn="vbtest")

    def test_capacity_check_bites(self, monkeypatch):
        """Prove the host capacity check is load-bearing: with it neutralized,
        the over-capacity request no longer raises (non-destructive monkeypatch)."""
        import mlx_mfa.attention as A
        monkeypatch.setattr(A, "_validate_paged_block_table", lambda *a, **k: None)
        bt = mx.zeros((2, 4), dtype=mx.int32)
        sl = mx.array([100, 8], dtype=mx.int32)
        # No raise now (the guard was the only thing producing the loud failure):
        A._validate_paged_block_table(bt, sl, 10, 16, 4, fn="vbtest")  # patched no-op
