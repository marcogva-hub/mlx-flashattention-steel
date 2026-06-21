"""Thin shim so `bench/` scripts can `from _bench_guard import require_nax_or_die`.

Single source of truth lives in `benchmarks/_bench_guard.py` (audit H7/H-09).
DEV/BENCH-ONLY (sdist-excluded).  `bench/` is not a package; a script run as
`python bench/foo.py` has `bench/` on sys.path[0] and imports THIS module.  We
load the canonical implementation by explicit PATH under a DISTINCT module name
(`_bench_guard_canonical`) to avoid the `_bench_guard`-vs-`_bench_guard` name
collision, then re-export.
"""
from __future__ import annotations

import importlib.util as _ilu
import os as _os
import sys as _sys

_BENCHMARKS = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "benchmarks")
# benchmarks/ on path so the canonical can import its sibling bench_validity.
if _BENCHMARKS not in _sys.path:
    _sys.path.insert(0, _BENCHMARKS)

_spec = _ilu.spec_from_file_location(
    "_bench_guard_canonical", _os.path.join(_BENCHMARKS, "_bench_guard.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FeatureUnavailable = _mod.FeatureUnavailable
require_nax_or_die = _mod.require_nax_or_die
require_accel_or_die = _mod.require_accel_or_die
nax_active = _mod.nax_active
