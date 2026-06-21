"""Audit M-02 lock — split-K calibration key carries the window SIZE.

THE BUG (reproduced, audit 2026-06-21): the split-K calibration key serialized
the sliding window as a single bit (`_W0`/`_W1`).  Distinct windows (256 vs 512)
therefore collided on one key, and on load `os.environ.setdefault` kept the
FIRST measurement and silently dropped the second — so one window ran with the
other window's (wrong) calibration.

THE FIX: the key now carries the window size (`_W{left}_{right}`).  Distinct
windows → distinct keys.  Old (schema < 2) windowed entries lost the size on
disk and CANNOT be recovered, so they are INVALIDATED on load (pruned →
re-calibrated lazily), never mis-applied; non-windowed entries are retained.

Bite-proof: on the PRE-fix code, `_splitk_env_key` had no window-size param and
both windows produced `_W1` — the distinctness assertions below fail, and the
old-format file would mis-apply window 256's value to window 512.
"""
import json

import pytest

from mlx_mfa import dispatch_policy as dp


@pytest.fixture(autouse=True)
def _clean_splitk_env(monkeypatch):
    """Isolate: clear any MFA_SPLITK_* + reset the one-time warn flag."""
    import os
    for k in list(os.environ):
        if k.startswith("MFA_SPLITK_MAX_N_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(dp, "_WARNED_CALIB_SCHEMA", False)
    yield


# ── The key fix: distinct windows → distinct keys (serialize + lookup) ────────

def test_distinct_windows_serialize_to_distinct_keys():
    k256 = dp._splitk_env_key(64, True, has_alibi=False, window_left=256, window_right=0)
    k512 = dp._splitk_env_key(64, True, has_alibi=False, window_left=512, window_right=0)
    knone = dp._splitk_env_key(64, True, has_alibi=False)
    assert k256 != k512, "window 256 and 512 STILL collide on the same key (the M-02 bug)"
    assert k256 != knone and k512 != knone
    # non-windowed keeps the legacy _W0 suffix (so non-windowed on-disk entries stay valid)
    assert knone.endswith("_W0")
    assert k256.endswith("_W256_0") and k512.endswith("_W512_0")


def test_v2_roundtrip_both_windows_survive(tmp_path, monkeypatch):
    """calibrate→serialize→load: window 256 (max_N 2048) and 512 (max_N 8192)
    BOTH survive with their OWN values (the pre-fix bug dropped the second)."""
    table = tmp_path / "dispatch.json"
    payload = {
        "calibration_schema_version": dp._CALIBRATION_SCHEMA_VERSION,
        "thresholds": [],
        "splitk_thresholds": [
            {"D": 64, "causal": True, "has_alibi": False,
             "window_left": 256, "window_right": 0, "max_N": 2048},
            {"D": 64, "causal": True, "has_alibi": False,
             "window_left": 512, "window_right": 0, "max_N": 8192},
        ],
    }
    table.write_text(json.dumps(payload))
    monkeypatch.setenv("MLX_MFA_DISPATCH_TABLE", str(table))
    dp._load_calibrated_kernel_config()
    import os
    assert os.environ.get(dp._splitk_env_key(64, True, has_alibi=False, window_left=256, window_right=0)) == "2048"
    assert os.environ.get(dp._splitk_env_key(64, True, has_alibi=False, window_left=512, window_right=0)) == "8192"


# ── Migration: old size-less file pruned + warned, not mis-applied ───────────

def test_old_schema_windowed_entries_pruned_and_warned(tmp_path, monkeypatch, recwarn):
    """An old (schema-less) file with size-less windowed entries: loads WITHOUT
    crashing, the windowed entries are INVALIDATED (not read back as any window's
    calibration), a one-time warning fires, and non-windowed entries are RETAINED."""
    table = tmp_path / "old_dispatch.json"
    payload = {
        "generated": "old",
        # no calibration_schema_version → treated as v1
        "splitk_thresholds": [
            {"D": 64, "causal": True, "has_alibi": False, "has_window": True, "max_N": 2048},   # windowed → prune
            {"D": 64, "causal": True, "has_alibi": False, "has_window": True, "max_N": 8192},   # windowed → prune
            {"D": 64, "causal": True, "has_alibi": False, "has_window": False, "max_N": 4096},  # non-windowed → retain
        ],
    }
    table.write_text(json.dumps(payload))
    monkeypatch.setenv("MLX_MFA_DISPATCH_TABLE", str(table))

    dp._load_calibrated_kernel_config()  # must NOT raise

    import os
    # windowed legacy entry must NOT be mis-applied to ANY window key
    assert os.environ.get(dp._splitk_env_key(64, True, has_alibi=False, window_left=256, window_right=0)) is None
    assert os.environ.get(dp._splitk_env_key(64, True, has_alibi=False, window_left=512, window_right=0)) is None
    # the old size-less _W1 key must not be set either
    assert os.environ.get("MFA_SPLITK_MAX_N_D64_C1_A0_W1") is None
    # non-windowed legacy entry IS retained
    assert os.environ.get(dp._splitk_env_key(64, True, has_alibi=False)) == "4096"
    # one-time loud warning fired
    msgs = [str(w.message) for w in recwarn.list if "calibration schema" in str(w.message)]
    assert msgs, "expected a one-time invalidation warning for old windowed calib entries"
    assert "INVALIDATED" in msgs[0]


def test_old_schema_warning_is_one_time(tmp_path, monkeypatch, recwarn):
    table = tmp_path / "old_dispatch.json"
    table.write_text(json.dumps({
        "splitk_thresholds": [
            {"D": 64, "causal": True, "has_alibi": False, "has_window": True, "max_N": 2048},
        ],
    }))
    monkeypatch.setenv("MLX_MFA_DISPATCH_TABLE", str(table))
    dp._load_calibrated_kernel_config()
    dp._load_calibrated_kernel_config()  # second load
    msgs = [w for w in recwarn.list if "calibration schema" in str(w.message)]
    assert len(msgs) == 1, f"expected exactly ONE warning across two loads, got {len(msgs)}"


def test_saved_payload_carries_schema_version(tmp_path, monkeypatch):
    """calibrate_dispatch persists the schema version (so future loads can migrate)."""
    out = tmp_path / "out.json"
    # head_dims=[] → no GPU crossover work; just exercise the save + schema stamp.
    dp.calibrate_dispatch(head_dims=[], save_path=str(out),
                          calibrate_splitk=False, calibrate_kernel_configs=False)
    data = json.loads(out.read_text())
    assert data.get("calibration_schema_version") == dp._CALIBRATION_SCHEMA_VERSION


# ── B (audit follow-up): Python↔C++ split-K key byte-identity contract ───────
# The key format is built in TWO languages (dispatch_policy._splitk_env_key and
# the C++ build_splitk_env_key the dispatch lookup uses).  A one-sided edit would
# silently desync → calibration written under one key, looked up under another →
# the mis-calibration class returns.  This locks them byte-for-byte.
#
# NON-VACUOUS: the C++ binding `_ext._splitk_env_key_cpp` calls the SAME
# `build_splitk_env_key` the real dispatch lookup uses (csrc/mfa_attention.cpp),
# not a parallel copy.  PROVES: key identity — window-512 calibration is read
# back under the 512 key, not 256's (the partial axis-(b) signal).  DOES NOT
# prove C++ dispatch actually entered split-K for that window (no split-K trace
# exists; bounded by split-K being output-invariant — a wrong key costs perf,
# never correctness).

_KEY_MATRIX = [
    (64, True, False, 256, 0),    # the collision pair — window 256 ...
    (64, True, False, 512, 0),    # ... vs window 512 (must be DISTINCT)
    (64, True, False, -1, -1),    # non-windowed (_W0)
    (64, False, False, 256, 0),   # non-causal windowed
    (128, True, False, 512, 0),   # D=128 windowed
    (128, False, False, -1, -1),  # D=128 non-windowed
    (64, True, True, 256, 0),     # alibi + windowed
]


def _ext_keyfn():
    try:
        from mlx_mfa._ext import _splitk_env_key_cpp
        return _splitk_env_key_cpp
    except Exception:
        return None


@pytest.mark.skipif(_ext_keyfn() is None,
                    reason="needs compiled _ext._splitk_env_key_cpp")
@pytest.mark.parametrize("D,causal,alibi,wl,wr", _KEY_MATRIX)
def test_python_cpp_splitk_key_byte_identical(D, causal, alibi, wl, wr):
    cpp = _ext_keyfn()
    py = dp._splitk_env_key(D, causal, has_alibi=alibi, window_left=wl, window_right=wr)
    c = cpp(D, causal, alibi, wl, wr)
    assert py == c, (
        f"Python↔C++ split-K key DESYNC for D={D} causal={causal} alibi={alibi} "
        f"win=({wl},{wr}): python={py!r} cpp={c!r} — calibration would be written "
        f"under one key and looked up under another (the M-02 mis-calibration class).")


@pytest.mark.skipif(_ext_keyfn() is None, reason="needs compiled _ext")
def test_python_cpp_keys_keep_256_512_distinct():
    """The collision pair must be distinct in BOTH languages (the core M-02 lock)."""
    cpp = _ext_keyfn()
    k256_py = dp._splitk_env_key(64, True, has_alibi=False, window_left=256, window_right=0)
    k512_py = dp._splitk_env_key(64, True, has_alibi=False, window_left=512, window_right=0)
    k256_c = cpp(64, True, False, 256, 0)
    k512_c = cpp(64, True, False, 512, 0)
    assert k256_py != k512_py and k256_c != k512_c, "256/512 collided again"
    assert k256_py == k256_c and k512_py == k512_c, "Python↔C++ desync on the collision pair"
