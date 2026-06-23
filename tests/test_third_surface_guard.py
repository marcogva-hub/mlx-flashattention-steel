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
    # 36 = 29 P0 hand-audited + 7 property-derived cache-append delegators.
    assert n_cm == 36, f"expected 36 computational class-methods, got {n_cm}"
    assert m.metal_kernel_offenders() == []
    assert len(m.metal_kernel_sites()) == 9


def test_promotion_rule_property_complete():
    # P3: the property rule must reproduce ALL classified computational methods
    # (incl. the 4 previously explicit-only — by property, not the list), and
    # flag nothing that isn't classified.
    m = _enum()
    flagged = set()
    for cn, cls in m._exported_classes():
        for nm in m._project_methods(cls):
            if m._method_reaches(cls, nm):
                flagged.add(f"{cn}.{nm}")
    comp = set(m.COMPUTATIONAL_CLASS_METHODS)
    # every classified computational method is independently DERIVED by property
    assert comp <= flagged, f"NOT derived by property (explicit-only crutch): {comp - flagged}"
    # nothing reaching is unclassified
    assert flagged <= comp, f"reaching but unclassified: {flagged - comp}"
    # the 4 round-12 NO-GO methods specifically
    for k in ("DecodeRuntime.prefill", "DecodeRuntime.step",
              "DenseKVCache.append", "PagedKVCache.append"):
        assert k in flagged, f"{k} still not derived by property"


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


# ── P3 property-completeness bites: the 4 patterns must each bite ────────────────
def _inject(cls_name, meth_name, fn):
    """Attach fn as a project method on an exported class; returns a cleanup."""
    import mlx_mfa
    fn.__module__ = "mlx_mfa.attention"
    cls = getattr(mlx_mfa, cls_name)
    setattr(cls, meth_name, fn)
    return lambda: delattr(cls, meth_name)


def test_bite_cross_object_delegation_in_reviewed():
    # Codex's synthetic: a DecodeRuntime method delegating to self.context.step
    # (cross-object delegation) placed in reviewed MUST be flagged.
    m = _enum()

    def delegated_public(self, *a, **k):            # noqa: ANN001
        return self.context.step(*a, **k)
    cleanup = _inject("DecodeRuntime", "delegated_public", delegated_public)
    m.REVIEWED_NONCOMPUTATIONAL_CLASS_METHODS["DecodeRuntime.delegated_public"] = "BITE"
    try:
        off, _ = m.class_method_offenders()
        assert any("DecodeRuntime.delegated_public" in o for o in off), \
            f"cross-object delegation in reviewed not flagged: {off}"
    finally:
        cleanup()


def test_bite_state_production_in_reviewed():
    # a method writing self._v from a v_new input, placed in reviewed → flagged.
    m = _enum()

    def state_writer(self, k_new, v_new):           # noqa: ANN001
        self._v[:, :, 0:1, :] = v_new
        return None
    cleanup = _inject("DenseKVCache", "state_writer", state_writer)
    m.REVIEWED_NONCOMPUTATIONAL_CLASS_METHODS["DenseKVCache.state_writer"] = "BITE"
    try:
        off, _ = m.class_method_offenders()
        assert any("DenseKVCache.state_writer" in o for o in off), \
            f"state-production in reviewed not flagged: {off}"
    finally:
        cleanup()


def test_bite_raw_ext_call_in_reviewed():
    # a method calling a raw _ext binding, placed in reviewed → flagged.
    m = _enum()

    def raw_caller(self, q, k, v):                   # noqa: ANN001
        from mlx_mfa import _ext
        return _ext.mfa_scatter_kv(q, k, v, v)
    cleanup = _inject("DenseKVCache", "raw_caller", raw_caller)
    m.REVIEWED_NONCOMPUTATIONAL_CLASS_METHODS["DenseKVCache.raw_caller"] = "BITE"
    try:
        off, _ = m.class_method_offenders()
        assert any("DenseKVCache.raw_caller" in o for o in off), \
            f"raw _ext call in reviewed not flagged: {off}"
    finally:
        cleanup()


def test_bite_intra_class_delegation_in_reviewed():
    # a method calling self.<computational_method>(), placed in reviewed → flagged.
    m = _enum()

    def intra_caller(self, *a, **k):                 # noqa: ANN001
        return self.append(*a, **k)                  # DenseKVCache.append is computational
    cleanup = _inject("DenseKVCache", "intra_caller", intra_caller)
    m.REVIEWED_NONCOMPUTATIONAL_CLASS_METHODS["DenseKVCache.intra_caller"] = "BITE"
    try:
        off, _ = m.class_method_offenders()
        assert any("DenseKVCache.intra_caller" in o for o in off), \
            f"intra-class delegation in reviewed not flagged: {off}"
    finally:
        cleanup()


def test_no_over_promotion_getters_clean():
    # pure getters/bookkeeping stay clean; SVDQuantLinear.__call__ stays reviewed.
    m = _enum()
    import mlx_mfa
    for cn, nm in [("DenseKVCache", "reset"), ("DenseKVCache", "seq_length"),
                   ("PagedKVCache", "seq_length")]:
        cls = getattr(mlx_mfa, cn)
        if nm in m._project_methods(cls):
            assert not m._method_reaches(cls, nm), f"{cn}.{nm} falsely promoted"
    assert "SVDQuantLinear.__call__" in m.REVIEWED_NONCOMPUTATIONAL_CLASS_METHODS
