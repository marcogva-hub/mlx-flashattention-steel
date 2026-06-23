"""Volet P2 — durable class-method + mx.fast.metal_kernel classifier guard
(P0 Task 4).

The function/raw guard (enumerate_api_surface) structurally cannot see class
methods or JIT kernels — which is how CX-TQ-DECODE-01 (an unguarded tq_decode
page load reached only via a class method + mx.fast.metal_kernel) survived 11
rounds. These bites lock the third-surface guard: a NEW public method that
reaches a computational/kernel/raw call, or a NEW page-indexed metal_kernel
without a bounds record, fails enumeration loudly.
"""
import importlib.util
import pytest


def _enum():
    spec = importlib.util.spec_from_file_location(
        "_enum_p2", "scripts/enumerate_api_surface.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── clean state ─────────────────────────────────────────────────────────────────
def test_clean_state_no_offenders_and_counts():
    m = _enum()
    cm_off, n_cm = m.class_method_offenders()
    assert cm_off == [], f"class-method offenders: {cm_off}"
    assert n_cm == 29, f"expected 29 computational class-methods, got {n_cm}"
    assert m.metal_kernel_offenders() == []
    assert len(m.metal_kernel_sites()) == 9


def test_promotion_rule_reproduces_p0_set():
    # every method the property flags as reaching must be in COMPUTATIONAL_CLASS_
    # METHODS (no reaching method hides; no bookkeeping method falsely promoted).
    m = _enum()
    flagged = set()
    for cn, cls in m._exported_classes():
        for nm in m._project_methods(cls):
            if m._method_reaches(cls, nm):
                flagged.add(f"{cn}.{nm}")
    assert flagged <= set(m.COMPUTATIONAL_CLASS_METHODS), \
        f"reaching methods not classified: {flagged - set(m.COMPUTATIONAL_CLASS_METHODS)}"


# ── bite 1: move a computational method into the reviewed set → fail ─────────────
def test_bite_computational_method_moved_to_reviewed():
    m = _enum()
    key = "TurboQuantPagedInferenceContext.step"
    m.COMPUTATIONAL_CLASS_METHODS.pop(key)
    m.REVIEWED_NONCOMPUTATIONAL_CLASS_METHODS[key] = "BITE"
    off, _ = m.class_method_offenders()
    assert any(key in o for o in off), f"guard did not flag {key} moved to reviewed: {off}"


# ── bite 2: synthetic public method that reaches compute, unclassified → fail ────
def test_bite_synthetic_reaching_method_unclassified():
    m = _enum()
    import mlx_mfa

    def _synthetic_attend(self, q, k, v):           # noqa: ANN001
        return mlx_mfa.flash_attention(q, k, v)      # reaches a computational entry
    _synthetic_attend.__module__ = "mlx_mfa.attention"   # mark as project method
    mlx_mfa.DenseKVCache.p2_synthetic_attend = _synthetic_attend
    try:
        off, _ = m.class_method_offenders()
        assert any("DenseKVCache.p2_synthetic_attend" in o for o in off), \
            f"guard did not flag the synthetic reaching method: {off}"
    finally:
        del mlx_mfa.DenseKVCache.p2_synthetic_attend


# ── bite 3: synthetic page-indexed metal_kernel with no record → fail ────────────
def test_bite_synthetic_page_indexed_kernel_unrecorded():
    m = _enum()
    orig = m.metal_kernel_sites
    m.metal_kernel_sites = lambda: [("mlx_mfa/fake_new.py", "_fake_paged_kernel", True)]
    try:
        off = m.metal_kernel_offenders()
        assert any("fake_new.py" in o for o in off), \
            f"guard did not flag the unrecorded page-indexed kernel: {off}"
    finally:
        m.metal_kernel_sites = orig


def test_bite_recorded_page_kernel_downgraded_to_unguarded():
    m = _enum()
    m.METAL_KERNELS["mlx_mfa/tq_decode.py:_get_k_dequant_kernel"]["page_bounds"] = "unguarded"
    off = m.metal_kernel_offenders()
    assert any("_get_k_dequant_kernel" in o for o in off), \
        f"guard did not flag a page kernel marked unguarded: {off}"


# ── bite 4: stale entry (method no longer exists) → fail ─────────────────────────
def test_bite_stale_computational_entry():
    m = _enum()
    m.COMPUTATIONAL_CLASS_METHODS["DenseKVCache.method_that_does_not_exist"] = "BITE"
    off, _ = m.class_method_offenders()
    assert any("method_that_does_not_exist" in o and "stale" in o for o in off), \
        f"guard did not flag the stale entry: {off}"
