# CLAUDE_V6_NAX.md — Guardrails pour le travail V6 NAX

Ce fichier est lu en complément de `CLAUDE.md` quand tu travailles sur **V6 NAX** (path d'attention pour Apple M5+ Neural Accelerators). Il contient des règles comportementales spécifiques à cette zone du code, accumulées depuis plusieurs sessions où des artifacts méthodologiques ont contaminé des décisions de shipping.

**Lis ce fichier intégralement avant de toucher à `csrc/mfa/v6_nax/` ou `csrc/mfa_v6_nax_primitive.cpp`.**

---

## 1. Sources de référence Apple — accès direct obligatoire

### 1.1 Repo MLX local

Le code source MLX (incluant les headers internes que ce projet utilise) est cloné en local :

```
~/code/mlx-source
```

**Tu DOIS t'y référer chaque fois qu'il y a une question sur le comportement Apple.** Pas de devinettes, pas d'hypothèses sur ce que MLX/MPP fait sans regarder le code.

Fichiers Apple critiques pour V6 NAX :
- `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h` — kernel NAX de référence Apple (le pattern à suivre)
- `mlx/backend/metal/kernels/steel/attn/nax.h` — abstractions `NAXFrag`, `NAXTile`, `mma()`, `row_reduce`, `row_bin_op`
- `mlx/backend/metal/kernels/steel/attn/transforms.h` — `MaxOp`, `SumOp`, `ExpSubOp`, `MulOp`, `BlockSwizzle`
- `mlx/backend/metal/kernels/steel/attn/params.h` — `AttnParams` struct
- `mlx/backend/metal/kernels/steel/attn/loader.h` — patterns de chargement Q/K/V

Avant d'écrire du code MSL touchant à NAX, lis le fichier Apple correspondant. Cite avec `file:line` dans tes commits.

### 1.2 MPP — limites connues

`mpp::tensor_ops::matmul2d` impose des contraintes connues :
- `get_left_input_cooperative_tensor` → require `execution_simdgroups<1>` (static_assert dans `MPPTensorOpsMatMul2dImpl.h`)
- `reduce_rows` → require `execution_simdgroups<1>` (static_assert)
- `correction.map_iterator(cO_0_it)` → require shared scope

**Ces contraintes sont définitives.** Si tu te trouves à essayer de les contourner via threadgroup memory bridges, **arrête et reconsidère**. La voie qui marche est de sortir de la couche cooperative_tensor MPP et d'opérer au niveau NAXFrag (cf. `nax.h`), comme Apple le fait dans `steel_attention_nax.h`.

### 1.3 Apple n'utilise PAS `mpp::tensor_ops::matmul2d` dans `steel_attention_nax.h`

Vérifie par toi-même : `grep -n "matmul2d\|tensor_ops" ~/code/mlx-source/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h`

Apple utilise :
- `NAXTile<T, TQ, TD>` (wrapper sur fragments NAX 16×16)
- `NAXFrag_t::mma()` (MMA direct sur les fragments)
- `metal::vec<AccumType, kRowsPT>` pour le softmax state
- `Stile.template row_reduce<MaxOp>(new_max)` (qui fait simd_shuffle_xor en interne)
- `Otile.template row_bin_op<MulOp>(factor)` pour l'application row-wise

Ce sont les building blocks de référence. Tout ce qui s'éloigne de ce pattern est une dette technique.

---

## 2. Méthodologie obligatoire — cinq artifacts déjà rencontrés

Cinq fois en cette campagne, des décisions de shipping ont été contaminées par des artifacts méthodologiques. Internalise les leçons :

### Artifact #1 — Pipeline cache contamination within-session
**Sprint G dispatch v6** : v6 mesuré -11.7% within-session, +14.3% en cross-session contrôlé.
**Cause** : la pipeline cachée de la première mesure pollue les suivantes.
**Règle** : pour toute décision shipping, **subprocess isolation obligatoire** via `subprocess.run()` séparé pour chaque round.

### Artifact #2 — Cache key bit-shift overflow
**MFA_V6_MATMUL_EXEC_SG** : `axis_flags << 24` overflow uint32_t pour axis_flags ≥ 0x100, le shader cache retourne la pipeline `<1>` cachée pour toutes les valeurs N.
**Règle** : pour chaque nouveau paramètre exposé via env var ou config, utilise un **champ dédié** dans la cache key, jamais de bit-packing dans des champs partagés. Vérifie `V6Key` dans `mfa_v6_nax_primitive.cpp:54-79`.

### Artifact #3 — Cold-start vs warmed-up comparison
**FlashVSR Sprint G claim** : "v6 -6.4%" était v5 cold-start (1.60ms) vs v6 warmed-up (1.15ms). Quand v5 est aussi warmed-up : 1.14ms = identique à v6.
**Règle** : warmup explicite avant chaque mesure. Pour les A/B/A, le round 1 et le round 3 (même config) doivent dériver à <10% sinon le bench est invalide.

### Artifact #4 — Env var change without kernel cache invalidation
**V33 SG=1 baseline** : `os.environ["MFA_V6_EXEC_SG"] = "2"` puis dispatch dans le même process Python ne re-compile pas le kernel — le cache retourne la pipeline `<1>`. RMSE rapportée 5e-3, RMSE réelle 2.5e-2.
**Règle** : tout changement d'env var qui affecte le code généré nécessite **un nouveau process Python**. Pas de boucle in-process sur env vars.

### Artifact #5 — Cross-session perf claims publishable only after multi-condition repro

**Findings v2.31.0 → v2.32.0 cross-session diagnostic** (2026-05-06) :
36-43% drift sur le legacy D=128 path entre le bench v2.31.0 (02:48 AM,
post-overnight idle) et la re-bench Phase 0 à 13:24 PM le même jour.
Hypothèse PSO cache testée et **rejetée** (cold-cache et warm-cache
benches produisent des timings identiques à ±2%). Hypothèse GPU ramp-up /
P-state testée et **rejetée** (post-30s-aggressive-warmup bench match
no-warmup bench à ±2%). Le drift n'est **pas un artifact transient
manipulable** — c'est un offset steady-state entre v2.31.0 et la session
courante, au-delà de la discrimination session-feasible.

**Sub-rule 5a — Metal PSO cache path on macOS 26+**

Le cache a été déplacé de `~/Library/Caches/com.apple.metal/`
(empty/obsolete sur macOS 26) vers per-application :

```
$DARWIN_USER_CACHE_DIR/<bundle-id>/com.apple.metal/
```

Pour notre `.venv/bin/python` bench process : bundle `org.python.python`.
Resolve via `getconf DARWIN_USER_CACHE_DIR`. Tout step "clear cache" dans
les scripts de diagnostic doit utiliser ce path, pas l'ancien
`~/Library/Caches`.

**Sub-rule 5b — Marketing-grade benchmark publication discipline**

Avant de publier des perf claims dans CHANGELOG / README / PyPI :

1. **Cross-session repro across 3+ sessions** with different times of
   day and different pre-bench states (cold-boot morning vs mid-day
   sustained vs after long idle).
2. **Document each session's conditions** : time of day, hardware
   uptime at bench start, Metal cache size before clear, macOS
   version (`sw_vers`), `GPU Active` percentage from `sudo powermetrics`
   in idle (must be < 5%).
3. **Use median of session medians**, not within-session statistics.
4. **Single-session bench results are STAGING data**, not publication
   data. Always pair with at least one re-bench in a different session.

**Why** : a single well-controlled within-session A/B/A is sufficient
for *engineering decisions* (e.g., dispatch choice within the project),
but insufficient for *external publication*. v2.31.0's perf claims —
based on a single A/B/A session — depended on measurement-time conditions
we cannot reproduce on demand.

**Anti-pattern** : single-session bench within hours of code changes.
The pipeline cache state and chip thermal regime at that moment may not
represent typical user experience.

**v2.32.0 strategic implication** : on M5+ NAX, forward attention on
canonical shapes (D∈{64,128}, qL>8) routes to `mx.fast.scaled_dot_product_
attention` which uses Apple's `steel_attention_nax.h`. mlx-mfa's V34
NAX-direct path matches but does not beat Apple's kernel cross-session,
and Apple's kernel benefits from continuous upstream tuning. Routing
to SDPA preserves mlx-mfa as a unified toolkit while stopping unnecessary
competition with Apple on shapes Apple covers well.

### Conséquence pratique

Pour TOUTE décision de shipping :

```python
# OBLIGATOIRE
subprocess.run(["python", "bench_script.py", "--mode=v1"], env=env_v1)
subprocess.run(["python", "bench_script.py", "--mode=v2"], env=env_v2)
subprocess.run(["python", "bench_script.py", "--mode=v1"], env=env_v1)  # validation thermique

# INTERDIT pour shipping
os.environ["X"] = "v1"
result_v1 = bench()
os.environ["X"] = "v2"
result_v2 = bench()  # CONTAMINÉ par cache pipeline
```

---

## 3. Multi-run obligatoire

Les variances run-to-run sur M5 Max peuvent atteindre 5-15%, parfois plus. Aucune décision shipping single-run :

| Delta observé | Runs minimum | Cross-bench |
|---|---|---|
| > 30% | 1 (signal très fort) | recommandé |
| 15–30% | 3 | obligatoire |
| 5–15% | **5** | **obligatoire** |
| < 5% | 5 | obligatoire, mais n'expect pas de shipping |

Médiane des médianes pour la stabilité (pas la moyenne).

---

## 3.5 Dispatch-path modification — three-axis validation rule

Any patch that modifies a dispatch decision, routing logic, or kernel
selection path must validate three distinct axes before shipping:

1. **Output sanity** — correctness oracle (PyTorch CPU FP32 cross-check,
   RMSE bar, sentinel-fill coverage gate). Catches: physically
   impossible outputs, addressing bugs that leave gaps, kernel
   miscompiles that produce garbage.
2. **Path entered** — perf or sanity A/B bench that detects whether the
   new path is actually taken. Catches: dispatch elision (silent no-op
   overrides, Python `__call__` type-vs-instance dunder lookup
   gotchas, fallback paths that engage when they shouldn't, env-var
   propagation gaps between Python and C++).

   **Critical sub-rule (added 2026-05-13 post-v2.37.0/v2.37.1 silent
   integration bug):** the path-entered exercise MUST use the public
   user-facing API path (e.g., `flash_attention(...)` with default
   `backend="auto"` + the documented env vars), not just forced or
   internal paths (e.g., `backend="mfa"` override or direct calls to
   `_ext.*` C++ bindings).

   Rationale: tests that force the MFA path bypass `should_use_mfa()`
   and therefore cannot detect dispatch-gate regressions on the
   user-facing surface. v2.37.0/v2.37.1 shipped with 100% test pass
   yet the documented perf claim was unreachable because every
   correctness test used `backend="mfa"` while users call with the
   default `backend="auto"`. Reference incident:
   `docs/v6-nax/v2.37.x-perf-claim-audit.md` and §Z below.

   **Required test pattern** (apply to any new auto-routing feature):

   ```python
   # INSUFFICIENT — only tests forced path; bypasses should_use_mfa()
   def test_routes_when_forced():
       out = flash_attention(q, k, v, backend="mfa")
       # ...

   # REQUIRED — also tests default path the user actually calls
   def test_routes_via_default_api(monkeypatch):
       monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "1")
       out = flash_attention(q, k, v)  # default backend="auto"
       # instrument or differential-bench to verify the new kernel fires
   ```

   Both patterns are required. The default-backend test is the
   minimum bar for "path entered" axis to be considered satisfied.
3. **Edges preserved** — semantic edge-case tests for NaN propagation,
   all-zero / all-masked inputs, denormal inputs, boundary conditions.
   Catches: optimizations that are bit-exact on mainline cases but
   break edge semantics other code depends on.

**Mainline correctness alone is insufficient.** The Sprint C → D →
v2.33.1 arc surfaced one silent bug per axis, each caught by the
corresponding gate class. All three axes are mandatory for
dispatch-path patches.

### Practical checklist

Before tagging a release that modifies dispatch:

- [ ] **Output sanity gate**: smoke test with sentinel-fill + oracle
      RMSE check on a small shape, BEFORE any production-shape timing.
      The smoke gate's pre-flight signature must include a non-trivial
      correctness verification — not just "did it run".
- [ ] **Path-entered gate**: A/B perf comparison between the old and new
      paths on at least one representative shape, **using the public
      user-facing API path** (`flash_attention(...)` with default
      `backend="auto"` + documented env vars), not forced backends or
      direct `_ext.*` calls. If perf ratio is ~1.00× when the new path
      is supposed to be faster, the new path isn't actually engaged
      (dead override / fallback engagement / dispatch gate blocking
      the routing). See §Z for the broader rule on public API path
      validation.
- [ ] **Edges preserved gate**: run the full pre-existing test suite,
      with NaN/Inf checks active. Any test that was passing before the
      patch and now fails — even if "the new behavior is reasonable" —
      indicates an edge-case semantic shift that downstream code may
      rely on.

### Worked examples (silent bugs caught by each axis)

**Sprint C Phase 1.1 (Output sanity).** `bench/conv_nax_matmul2d_microbench.py`
v1 reported 101 TFLOPS on M5 Max (NAX FP16 peak is 38 TF). Microbench passed
because there was no correctness check on its output — just timing. The
methodology bug was caught by adding a sentinel-fill + `mx.matmul(A.f32, B.f32)`
oracle check on a tiny shape (M=128, K=64, N=64) as a pre-flight gate.
After fix: RMSE=0 on the smoke shape, production timings physically plausible
(43 TF on `mid_resnet`). Reference: `docs/conv-nax/conv-nax-phase1_1-microbench-blocker.md`.

**Sprint D Track C (Path entered).** `patch_seedvr2_vae(model)` initially used
`object.__setattr__(mod, "__call__", patched_fn)`. Python's `__call__`
resolution looks up the TYPE, not the instance — the override was dead.
All 4 correctness tests passed (because both paths called the same
class-level `__call__`). The A/B perf bench in `bench/conv_nax_patcher_ab.py`
measured speedup 1.00× and revealed the dead override. Fix: `mod.__class__ = ...`
swap to a dynamically-created subclass with overridden `__call__`. After
fix: 2.29× speedup (matches Phase 1.5 `mid_resnet` 2.26× ratio).
Reference: `docs/conv-nax/conv-nax-prod-decisions.md` D34.

**v2.37.0/v2.37.1 (Path entered, public API sub-rule).** V34 backward
kernels shipped as SHIP_OPT_IN with the documented claim "D=64 qL ≥ 2048:
1.4-1.85× faster than SDPA-vjp". All 8 V34 backward correctness tests
passed — but every test forced `backend="mfa"`, which bypasses
`should_use_mfa()`. For non-causal D ∈ {64, 128}, `should_use_mfa()`
returns False, and `flash_attention()` returned via `_fallback_sdpa()`
BEFORE the V34 backward env-var check could engage the custom-vjp
chain. Users following the documented API setup
(`MFA_ENABLE_V34_BACKWARD=1` + `mx.grad(flash_attention(...))`) got
SDPA-vjp silently. Direct `_ext.v6_nax_*` kernel calls confirmed the
1.81× speedup existed at kernel level — unreachable through the
public API. Caught only by manual investigation; not by the test suite.
After fix (v2.37.2): narrow carve-out in `flash_attention()` engages
V34 when env + shape qualify; differential benches via the default
`backend="auto"` path now show 1.81-1.82× speedup. Reference:
`docs/releases/v2.37.2-release-notes.md` and
`docs/v6-nax/v2.37.x-perf-claim-audit.md`.

**v2.33.1 patch (Edges preserved).** Initial fast-fallback design substituted
bool mask for float bias to skip `mx.where` (~1.3 ms saved unconditionally).
Bit-exact on normal cases — all correctness equivalence tests passed. But
MLX SDPA with all-False bool mask produces finite garbage (no attention),
while the float-bias all-`-inf` row produces NaN softmax (the semantic
downstream code depends on for "no information available" detection).
Caught by the existing `test_all_false_mask_row_gives_nan_or_zero` test
failing on the first revision. Revised patch caches the FLOAT BIAS (not
bool mask), preserving the NaN-for-fully-masked-rows contract.
Reference: `docs/sparse-fallback-audit.md` + commit `9e0ab6a`.

### When to apply

This rule applies to any patch that:
- Changes the kernel selection (`if config_X: use_kernel_A else: use_kernel_B`)
- Routes through a different fallback or fast path
- Swaps `__call__` / `forward` methods (patchers, decorators)
- Modifies the M1-vs-M3 vs M5+ hardware dispatch
- Adds or removes a cache layer (id-keyed, lru-keyed, content-keyed)
- Inlines or unrolls a previous indirect dispatch

It does NOT apply to:
- Pure refactors that preserve the exact dispatch graph
- New API surfaces that don't touch existing routing
- Documentation, tests, build-system changes

When in doubt, apply the three axes anyway — the cost is small (a
focused A/B bench + a `find . -name "test_*.py" -exec` of the existing
edge tests), the cost of a silent bug shipping to PyPI is large.

---

## 4. Benchmark protocol — dual regime (M5 Max)

Le M5 Max sous Marco utilise le profil de ventilation **iStat Menus performance** (pas le profil Apple par défaut qui throttle agressivement). Température GPU sous charge ~70°C.

**Two protocols are canonical, selected by measured-kernel wall-clock regime.**
§4.1 (§4-strict cooldown) for ≥1.5ms kernels; §4.2 (canonical warmup +
continuous) for sub-1.5ms kernels. They are complementary, not competing —
they serve different measurement regimes.

### §4.1 — §4-strict cooldown protocol (canonical for ≥1.5ms wall-clock kernels)

Cooldowns recommandés :
- 3 min initial cool-down avant le premier round
- 90s entre rounds (A→B, B→A)
- 60s entre shapes pendant un round
- Si drift R1↔R3 > 10%, étendre les cooldowns

This protocol is canonical for shapes where wall-clock ≥ 1.5ms. The
kernel's own runtime keeps the GPU busy enough to dampen power-state
variance during measurement; the §4 idle cooldowns then ensure thermal
stability across runs. Cross-session range CONFIDENT (<10%) is
achievable under this protocol.

### §4.2 — Canonical warmup + continuous (sub-1.5ms wall-clock kernels)

For shapes where wall-clock < 1.5ms, the §4-strict protocol fails per
two REGRESSION sprints (mx.matmul v2.36.0, matched-workload 2026-05-12)
and per web research convergence (Apple Developer Forums thread 692062,
Feng et al. arXiv 2501.14925, MLX docs, WWDC25 Session 315, Draw Things
MFA v2.5 NA release post Nov 2025, MLX GitHub Discussion #1571). See
`docs/methodology/canonical-protocol.md` for full methodology and
rationale.

Specification:
- 10 warmup iters + 100 continuous timed iters per direction per shape
- `mx.eval` synchronisation inside both loops
- p50 / p95 / p99 / mean / min / max stats per direction
- Ratio analysis V2/SDPA cross-session (more stable than absolute)
- Verdict: CONFIDENT <10% / BOUNDARY 10-20% / HIGH_VARIANCE >20% on
  cross-session **ratio** range (not absolute timing range)

### §4.3 — Protocol selection rule

Pre-bench, estimate wall-clock per shape from prior data:

| Regime | Protocol |
|---|---|
| ≥ 1.5ms | §4-strict (§4.1) |
| < 1.5ms | canonical warmup + continuous (§4.2) |
| Unknown | run §4.1 first, switch to §4.2 if wall-clock < 1.5ms confirmed |

### §4.X — Sub-1ms kernel measurement caveat (M5 Max) — RESOLVED via §4.2

The sub-1ms variance issue surfaced in v2.36.0 V2-only re-bench is now
mechanistically explained and methodologically addressed via §4.2
(canonical warmup + continuous protocol). This section preserves the
historical record of the resolution.

Kernels with wall-clock median ≤ 1.4 ms exhibit cross-session variance
from **GPU power-state cycling during §4 idle cooldowns**. The 90s/60s/180s
periods are sufficient for thermal stability on ≥ 2ms kernels but
introduce power-state-cycle variance for sub-1.5ms kernels.

**Empirical anchor.** v2.36.0 V2-only re-bench Section D control bench
(2026-05-12) measured `lcsa_mid_seq8k_sparse`:

- R1 (first dispatch in fresh session) : 0.87 ms
- R2 (after 90s idle) : 1.82 ms — **109% slowdown**, not warmup
- R3 (after another 90s idle) : 1.91 ms

This is the opposite of typical cache-warmup behavior (warmup makes
things faster, not slower). The mechanism is M5 Max aggressive GPU
power management — 90s idle is sufficient for the GPU to downclock
its clocks/voltage. The next dispatch hits a cooler power state and
runs slower.

**Diagnostic separator (M5 Max):**

- Wall-clock median ≤ 1.4 ms → power-state-sensitive, §4 cooldowns CONFOUND measurement
- Wall-clock median ≥ 2.0 ms → §4 cooldowns work fine, kernel keeps GPU busy enough

**Resolution path** (closed 2026-05-13):

Two warmup-during-cooldown protocols were tested and both produced a
**REGRESSION verdict**:

| Protocol | Date | HIGH→CONFIDENT | CONFIDENT regressed | Verdict |
|---|---|:--:|:--:|:--:|
| v2.36.0 — 256×256 FP16 matmul, 50ms gap | 2026-05-12 | 2/3 | 3/4 | REGRESSION |
| Matched-workload family — sparse_attention_nax B=1 H=4 qL=kL=2048 D=64 BT=16, 50ms gap | 2026-05-12 | **0/3** | **3/4** | REGRESSION |

Web research convergence (6 sources cited in
`docs/methodology/canonical-protocol.md`) confirmed Apple's hardware
design intentionally excludes userspace P-state lock. No warmup-during-
cooldown approach can fully resolve sub-1ms variance under §4-strict-
style protocols. The canonical Apple Silicon methodology (warmup +
continuous, ratio analysis — §4.2) is the appropriate tool for this
regime.

**Consolidated finding**: every warmup mechanism that holds GPU power
state above the < 100ms downclock threshold inevitably perturbs the
measured kernel's cache state in a shape-specific way. The variance is
real (not a measurement artifact). The fix is not a better warmup but
a different measurement regime: §4.2 measures continuous back-to-back
iterations, bypassing the cooldown-induced power-state cycling entirely.

**Path-forward registry (CLOSED 2026-05-13):**

| Option | Status |
|---|---|
| 1. Matched-workload family | FALSIFIED 2026-05-12 |
| 2. Heartbeat register-only warmup | **SKIPPED** per Marco strategic decision (Option β: methodology pivot) |
| 3. Metal API power-state lock | deferred — low-EV given confirmed userspace exclusion |
| 4. Shape-aware ≥X-ms default | **ACTIVATED 2026-05-13 via v2.36.1** |

Sub-1ms methodology thread closed. Future sub-ms kernel work follows
§4.2 canonical protocol. v2.36.1 ships shape-aware
`decide_auto_version()` calibrated empirically from §4.2 data; V2
default activates for shapes where cross-session ratio range is
CONFIDENT under canonical methodology.

References:
- `docs/methodology/canonical-protocol.md` (canonical §4.2 spec + 6 web research sources)
- `docs/methodology/canonical-bench-results.md` (v2.36.1 calibration data)
- `docs/methodology/matched-workload-results.md` (REGRESSION verdict 2026-05-12)
- `docs/methodology/matched-workload-diagnostic.md` (option 1 falsification analysis)
- `bench/methodology/canonical_warmup_continuous_harness.py` (§4.2 reference implementation)
- `bench/methodology/matched_workload_harness.py` (historical option 1 implementation)
- prior `experiment/methodology-sub1ms-protocol` branch (v2.36.0 matmul protocol artifacts, preserved for archaeology — not merged to master)

---

## 4.5 V34 forward mechanistic findings (référence canonique)

Les gains V34 forward documentés en v2.32.0 (+18-40% vs prédécesseurs)
ont été décomposés empiriquement en investigation 2026-05-12
(`docs/v6-nax/v34-forward-mechanisms.md`) :

| Hypothèse | Statut | Mécanisme |
|---|---|---|
| B — cross-SG sync elim | **CONFIRMÉE** (structurelle + bundle) | V34: `simdgroup_barrier(mem_none)` only; predecessors: `threadgroup_barrier(mem_threadgroup)` |
| C — simd_shuffle_xor vs MPP reduce | **CONFIRMÉE** (structurelle + bundle) | V34: `NAXFrag::row_reduce` → `simd_shuffle_xor`; predecessors: `mpp::reduce_rows` |
| E — Apple defaults mis-tunés pour M5 | **CONFIRMÉE** (structurelle + bundle) | V34: BQ/BK/WM tunés M5; predecessor: MPP autotune par défaut |
| **B+C+E aggregate** | **CONFIRMÉE: ratio 1.184× (+18%)** | Probe `MFA_V6_USE_V34=0` vs `=1` sur 3 shapes ≥1.4ms |
| A — TGP occupancy | **FALSIFIÉE au baseline + REVERSE à SG=8** | V34's default `EXEC_SG=4` est sub-optimal pour mid_d128; SG=8 gagne +32% |
| D — register pressure | **NULL** | V34 tile defaults ne sont pas register-bottlenecked |

### Mécanismes à appliquer dans tout nouveau kernel NAX-direct sur M5+

1. **Cross-SG sync minimization** — utiliser `simdgroup_barrier(mem_none)`
   pour les barrières intra-SG; réserver `threadgroup_barrier(mem_threadgroup)`
   aux cas où l'accumulation cross-SG est strictement nécessaire (idéalement
   ≤1 par K-tile).
2. **NAXFrag::row_reduce** pour les réductions softmax row-wise plutôt que
   `mpp::reduce_rows`. Le shuffle-xor pattern est plus rapide que la cooperative
   tensor reduction.
3. **M5-tuned BQ/BK/WM defaults** — ne pas hériter des Apple MPP defaults
   aveuglément. V34 forward defaults reference: BQ=32/BK=32/WM=2 (D=64),
   BQ=64/BK=32/WM=4 (D=128). À noter (anti-pattern A) : EXEC_SG=4 est
   sub-optimal pour D=128 mid shapes; SG=8 unlock +32% sur mid_d128.

### Anti-patterns identifiés

- **V34's default `EXEC_SG=4` for D=128 mid shapes** : sous-tuné. Un
  follow-up patch pourrait introduire une heuristique shape-aware
  (`EXEC_SG=8` pour qL∈[2048, 4096], `EXEC_SG=4` pour qL≥8192 où le
  baseline est déjà saturé). Voir `docs/v6-nax/v34-forward-mechanisms.md`
  §"Implications".

- **Hériter des Apple MPP autotune defaults aveuglément** : H. E confirmée
  empiriquement. Tout nouveau kernel NAX-direct doit explicitement
  caractériser ses tile-shape defaults pour M5 Max.

Référence implémentation : `csrc/mfa/v6_nax/NAAttentionKernel.cpp`
`createV34Source()` (lignes 2307-3671) + `csrc/mfa_v6_nax_primitive.cpp`
`generate_v6_source()` (env knob dispatch).

---

## 5. Correctness avant tout

Avant TOUTE mesure de timing :
- **Tile coverage 100%** via `MFA_V6_SENTINEL_FILL=1` + `bench/v6_coverage_diagnostic_v2.py`
- **RMSE FP32 < 1e-3** vs `mx.fast.scaled_dot_product_attention` (FP32 reference)

Si une seule shape fail correctness, **skip cette config**, ne tente pas de la shipper.

---

## 5.X Pre-tag auto-default audit (Sprint U / v2.36.0+)

Before any PyPI release, CC verifies (in addition to multi-SoT version
audit per v2.33.x lesson):

- [ ] New code paths that ship validated optimizations are auto-routed
      through existing mlx-mfa public surfaces (`flash_attention*`,
      `sparse_attention*`, `conv3d_nax_forward`, etc.)
- [ ] Auto-on-import hooks register the new optimization if it requires
      hooking external `mx.*` surfaces (see `mlx_mfa/_auto_hooks.py`)
- [ ] Env-var opt-in present only for:
      (a) transitional state (validation pending)
      (b) escape hatch (e.g., `MFA_DISABLE_AUTO_HOOKS` for benchmarking)
      (c) A/B comparison knob (e.g., `MFA_LCSA_KERNEL_VERSION` pre-graduation)
- [ ] Named patchers documented as expert-mode, not primary path
- [ ] README primary usage path is `import mlx_mfa` + normal MLX usage
- [ ] Migration documented in CHANGELOG if optimization graduates from
      opt-in to default

See `docs/RELEASE_PHILOSOPHY.md` for the full principle.

### §X.5 — Tool availability verification (added 2026-05-13)

Before any version bump or release flow, the canonical `.venv/` must
have all required release-flow tools available.  This was added after
the v2.37.3 release session surfaced a false "twine not found"
diagnosis caused by using `which twine` (which searches `$PATH`, not
the venv) instead of `.venv/bin/twine` — and after the Sprint 1
venv-consolidation cleanup eliminated the ambiguous legacy `venv/`
that accumulated alongside `.venv/`.

Pre-tag gate:

- [ ] `bash scripts/check_venv.sh` runs clean (exit 0)
- [ ] `.venv/bin/twine` and `.venv/bin/pytest` exist as binaries
- [ ] `.venv/bin/python -c "import build"` succeeds (note: `build` is a
      Python module only, NOT a binary — checking for `.venv/bin/build`
      will always fail and is the wrong test)
- [ ] `.venv/bin/python -c "import mlx.core, mlx_mfa"` succeeds (catches
      editable-install drift after kernel changes)

CI variant (check-only, no auto-install):

```bash
bash scripts/check_venv.sh --no-install
# Exit 0  → all tools present
# Exit 2  → at least one tool missing; install before tagging
```

The script is array-driven: adding a new release-flow tool (e.g.,
`ruff` for linting, `mypy` for type-check gate) is a one-line edit
to the `BINARY_TOOLS` or `MODULE_TOOLS` array in the script — no
second edit needed.

Reference: `CLAUDE.md` "Canonical Python environment" section + Sprint 1
venv-consolidation (`chore/venv-consolidation`, 2026-05-13).

---

## §Z. Public API path testing rule (added 2026-05-13)

Every performance claim documented in release notes, CHANGELOG entries,
README, training guides, or any user-facing public doc MUST be
reproducible via the documented user-facing API call path with the
same env vars / configuration a user would set, NOT via internal
kernel benchmarks or forced-backend (e.g., `backend="mfa"`)
measurements.

### Reference incident

v2.37.0 / v2.37.1 silent integration bug (2026-05-13).  Release notes
documented "D=64 qL ≥ 2048: V34 backward is 1.4-1.85× FASTER than
SDPA-vjp."  At kernel level the speedup existed (direct
`_ext.v6_nax_backward_dv_raw` calls achieved 1.81-1.82× faster than
SDPA-vjp).  Through the documented public API
(`mx.grad(flash_attention(...))` with `MFA_ENABLE_V34_BACKWARD=1`),
the speedup was unreachable: `should_use_mfa()` returns False for
non-causal D ∈ {64, 128}, `flash_attention()` returned via
`_fallback_sdpa()` before the V34 backward env-var check could
engage the custom-vjp.  Users following the docs got SDPA-vjp
silently.  See `docs/v6-nax/v2.37.x-perf-claim-audit.md` and
v2.37.2 release notes.

### Reproducibility template

For every "X× faster" or "Y% speedup" or "Z ms vs W ms" claim, run
this audit checklist before tagging:

- [ ] What is the documented public API call the user makes?
      (`mx.grad(flash_attention(...))`, `mlx_mfa.flash_attention(...)`,
      etc.)
- [ ] What env vars are documented as required?
      (`MFA_ENABLE_V34_BACKWARD=1`, `MFA_LCSA_KERNEL_VERSION=v34`, etc.)
- [ ] What's the documented shape regime?  (D=64 + qL ≥ 4096, sparse
      density ≥ X, etc.)
- [ ] Reproduce the measurement using ONLY the documented API + env +
      shape.  Verify the kernel claimed responsible for the speedup
      actually engages (instrument the dispatch path with a counter
      or differential bench: if "feature ON" and "feature OFF" produce
      identical timings, the feature isn't engaging).
- [ ] Compare against the documented baseline (typically SDPA-vjp or
      vanilla SDPA) using the same public API path.
- [ ] If kernel doesn't engage via documented path → claim is
      unreachable → either fix routing (§3.5 axis 2) OR correct /
      remove the claim from user-facing docs.

### What this rule prohibits

- Documenting a kernel-isolation benchmark result as if it were
  end-to-end user perf (e.g., bench `_ext.v6_nax_backward_dk(...)` and
  claim "X× faster backward" without checking that
  `mx.grad(flash_attention(...))` actually routes to it).
- Documenting a forced-backend (`backend="mfa"`) result as if it
  applied to default user behavior.
- Stating shape regime in vague terms ("D=64 fast") without specifying
  the exact public-API engagement criteria (which env vars, which
  routing conditions, which shape thresholds).
- Quoting kernel-only timing tables in release notes / README /
  training guides without a paired end-to-end measurement via the
  public API.

### What this rule requires

- Every release that contains perf claims runs the perf-claim
  reachability test suite (`tests/test_release_notes_perf_claims.py`)
  before tagging.  The suite is the executable form of this rule.
- Every perf-claim docstring / CHANGELOG entry includes a "Reproduce"
  snippet showing the exact public API call + env setup the claim
  rests on.  Example:

  ```python
  # Reproduce: D=64 qL=8192 V34 backward 1.81× faster than SDPA-vjp
  import os; os.environ["MFA_ENABLE_V34_BACKWARD"] = "1"
  import mlx.core as mx, mlx_mfa
  q = mx.random.normal((1, 4, 8192, 64), dtype=mx.float16)
  k = mx.random.normal((1, 4, 8192, 64), dtype=mx.float16)
  v = mx.random.normal((1, 4, 8192, 64), dtype=mx.float16)
  def loss(q, k, v):
      return mlx_mfa.flash_attention(q, k, v).sum()  # default backend
  dQ, dK, dV = mx.grad(loss, argnums=(0, 1, 2))(q, k, v)
  ```

- If a perf finding is genuinely kernel-isolation-only (research
  characterization, autoresearch sweep, profiling-grade microbench),
  it goes in research docs (`docs/v6-nax/*-investigation.md`,
  `devnotes/*`), NOT in user-facing release notes / README /
  training guides.

### Crosswalk to existing rules

- §3.5 axis 2 (path entered) — public API sub-rule: same content
  applied to tests rather than release notes.
- §5.X (pre-tag auto-default audit) — extended below: pre-tag audit
  must also pass perf-claim reachability checks.
- §AA (skill invocation checkpoints, below) — `/mlx-code-review`
  invocation is mandatory post-doc-creation with perf claims.

### Updated pre-tag checklist (extends §5.X)

Before any PyPI release that mentions perf in user-facing docs:

- [ ] Run `pytest tests/test_release_notes_perf_claims.py -v` — ALL
      parameterized claims pass.
- [ ] For each new claim added since last release: add a parameterized
      entry to `PERF_CLAIMS` in that test file.
- [ ] Each claim's "Reproduce" snippet in release notes is a valid
      executable script under the project venv.

---

## §AA. Skill invocation checkpoints (added 2026-05-13)

CC has access to specialized slash-command skills (e.g.,
`/mlx-code-review`, `/metal-kernel-dev`, `/mlx-debug-forensics`,
`/repo-release-prep`).  These were underused in the v2.37.0/v2.37.1
sprint (only one `/mlx-code-review` invocation across the whole
SHIP_OPT_IN chain), which contributed to the silent integration bug
shipping.  This section codifies mandatory invocation points to
ensure systematic use.

### Mandatory invocations

| Trigger | Skill | Why |
|---|---|---|
| Pre-kernel-design / new kernel write (>200 LOC of generator) | `/metal-kernel-dev` | Register budget, tile layout, NAX primitives availability |
| Post-kernel impl, pre-correctness-tests | `/mlx-debug-forensics` | Anticipate silent corruption modes (NAX cooperative-tensor edge cases, transposeState, padding) |
| Post-bench discovery of "X× speedup" (kernel OR forced-backend) | `/mlx-code-review` | Methodology audit, public API path engagement verification (§Z) |
| Any FALSIFIED outcome (REGRESSION verdict, hypothesis disproved) | `/mlx-code-review` | Audit that the falsification methodology is sound (not noise / methodology bug) |
| Pre-merge to master (any branch) | `/mlx-code-review` | Final review of diff against repo conventions, code quality, test coverage |
| Pre-version-bump (any release) | `/mlx-code-review` (full) + `/repo-release-prep` | Multi-SoT version audit, CHANGELOG correctness, release notes generation |
| Pre-release tag | `/repo-release-prep` | Multi-SoT audit, release artifact preparation |
| Register-spill suspicion / WM-BK-BQ default change | `/metal-kernel-dev` | Register pressure math, NAXFrag scope sizing, tile-fit verification |
| Post-doc-creation (README / quickstart / training guides) with perf claims | `/mlx-code-review` | Verify §Z compliance: every claim reachable via public API |

### Recommended invocations (not mandatory)

| Trigger | Skill |
|---|---|
| Model integration questions (FlashVSR, SeedVR2, SparkVSR, etc.) | `/mlx-model-porting` |
| Architectural cross-sprint planning | `/cc-prompt-orchestrator` |
| User-facing TUI work | `/inference-console` |

### What this rule prohibits

- Shipping a perf claim to PyPI without `/mlx-code-review` audit of
  the claim's reachability.
- Merging kernel changes to master without `/mlx-code-review` on the
  full diff.
- Changing WM / BK / BQ defaults without `/metal-kernel-dev` audit of
  register pressure math.
- Implicit "I read the code, looks fine" review in lieu of
  `/mlx-code-review`.

### Invocation protocol

When a checkpoint fires:

1. CC invokes the skill with relevant context (file paths, recent
   changes, bench data, hypothesis under test).
2. CC acts on the skill's findings (applies fixes, flags concerns,
   updates STATUS doc).
3. CC logs the invocation in the relevant sprint deliverable doc
   (decisions.md, status.md, audit.md) with a brief summary of
   findings — institutional memory across sessions.

Skill invocations are visible institutional memory: they appear in
the session log and let future CC sessions audit the rigor of prior
work.

### Reference incident retrospective

v2.37.0/v2.37.1: `/mlx-code-review` applied post-Phase 2 Section E
(test-coverage review) would likely have surfaced the gap — every V34
backward correctness test forced `backend="mfa"`, never exercising the
default backend the perf claim implicitly relied on.  §AA forces that
review checkpoint by name.

### §AA.1 — Mandatory blocking enforcement (added 2026-05-13, Sprint 4)

The mandatory invocations in the table above are NOT advisory.  They
are **blocking gates**.  If a checkpoint is reached and the required
skill is not invoked, CC MUST:

1. **Halt the current workflow** before proceeding with the next action
2. **Surface the gap explicitly** via the STATUS doc or session message
   (e.g., "§AA violation: /mlx-mfa-perf-audit not invoked at perf
   discovery checkpoint; halting before merge.")
3. **Invoke the required skill** before continuing
4. **Log the invocation** in the relevant sprint deliverable doc

**"Halt the current workflow" means** (mechanical definition, not
judgment-based):

- Do NOT start the next tool call (Edit / Write / Bash / git commit)
  that would advance the workflow toward the gated outcome (merge,
  tag, build, upload).
- Complete the in-flight READ-ONLY tool call (Read / Grep / Glob /
  Bash `git status` etc.) if any — these don't mutate state.
- Switch context to invoke the required skill BEFORE any state-
  mutating action.
- Partial state from before the halt (uncommitted edits, staged
  files) is NOT rolled back — the skill invocation operates on the
  current working tree as-is.

Violations of §AA mandatory checkpoints constitute a **procedural
failure** equivalent to a failing three-axis test (§3.5).  They block
merge to master, block version bumps, block PyPI releases.

**There is no manual override.**  If a checkpoint is genuinely
inapplicable (e.g., a doc-only release with no kernel changes), CC
documents the inapplicability in the sprint deliverable doc with
reasoning, but the audit (§AA.2) still records the decision.

### §AA.2 — Skill invocation evidence in sprint docs (added 2026-05-13)

Every sprint deliverable doc (`decisions.md`, `results.md`, `status.md`,
`docs/audits/*.md`, `docs/sprints/*.md`) MUST include a "Skill
invocations" section in this format:

```markdown
## Skill invocations

| Skill | Decision point | Timestamp (ISO) | Findings count | Action taken |
|---|---|---|---|---|
| /mlx-code-review | pre-merge of audit_runner.py | 2026-05-13T14:23Z | 2 MEDIUM, 3 LOW | MEDIUM fixed before commit |
| /mlx-mfa-perf-audit | claim audit v2.37.4_d64 | 2026-05-13T15:01Z | 1 (REACHABLE) | Verdict captured in CHANGELOG |
```

Empty or missing section → **audit fails**.  The
`/mlx-mfa-release-audit` skill's Check 5 (skill invocation log audit)
verifies this section is populated before any version bump.

**Check 5 enforcement boundary (honest scope):** Check 5 currently
verifies *presence* of a populated table with non-empty rows.  It
does NOT cross-reference each row against the §AA.3 checkpoint
category matrix.  Coverage of all mandatory checkpoints is on the
sprint author — a sprint that genuinely had a perf discovery and
silently skipped `/mlx-mfa-perf-audit` could still pass Check 5 by
listing only other invocations.  Future audit work may tighten
Check 5 to flag missing categories; for now, the rule depends on
sprint-author honesty for category coverage, while Check 5
mechanizes the floor (no empty / missing tables).

Templates for sprint deliverable docs live in `docs/templates/`:
- `sprint_decisions_template.md`
- `sprint_status_template.md`

These templates pre-include the Skill-invocations table so CC cannot
forget to fill it.  When starting a new sprint deliverable, copy from
the template.

### §AA.3 — Reference to mlx-mfa-* skills (Sprint 3 deliverables)

The mlx-mfa-specialized skills created in Sprint 3 (see
`docs/skills/README.md`) automate specific §AA checkpoints:

| Checkpoint | General skill | Automation skill (preferred) |
|---|---|---|
| Post-bench "X× speedup" / "Y% speedup" discovery | /mlx-code-review | **/mlx-mfa-perf-audit** |
| Pre-version-bump | /mlx-code-review + /repo-release-prep | **/mlx-mfa-release-audit** |
| Pre-bench sub-ms work | /metal-kernel-dev | **/mlx-mfa-bench-methodology** |
| New kernel write (>200 LOC source generator) | /metal-kernel-dev | **/mlx-mfa-kernel-design** (UNBLOCKED v2.38.x Phase B; mandatory) |
| **Before audit-prescribed kernel sprint (§AA.5 premise validation)** | manual `dir(mx)` + bench | **/mlx-mfa-apple-primitives-coverage** (added 2026-05-14 post-Sprint-3+4 retrospective) |

When an mlx-mfa-* automation skill exists for a checkpoint, invocation
of that skill satisfies the §AA mandatory requirement.  The general
skill remains available for cases not covered by the specialized
skill — e.g., reviewing a Python refactor that doesn't touch a
documented perf claim still needs `/mlx-code-review`, not
`/mlx-mfa-perf-audit`.

### §AA.4 — Pre-tag enforcement via /mlx-mfa-release-audit

Before any version bump (and therefore any PyPI release), CC MUST
invoke:

```
/mlx-mfa-release-audit target_version=<new_version>
```

If verdict is `BLOCKED`, the release flow halts.  CC does NOT
proceed to:
- Multi-SoT version bump (`pyproject.toml`, `mlx_mfa/__init__.py`,
  `README.md`)
- `git tag vX.Y.Z`
- `python -m build`
- `twine upload`
- `gh release create`

The blocking findings are addressed and `/mlx-mfa-release-audit` is
re-invoked until `GREEN` (or `GREEN_WITH_ADVISORY` with documented
advisory-acceptance).

This is the **canonical pre-tag gate**.  The earlier §X manual
checklist remains as a backup / documentation reference, but
`/mlx-mfa-release-audit` is now the mechanical enforcer.  In case of
disagreement between the manual checklist and the skill's verdict
(e.g., the skill flags BLOCKED but a human reviewer believes the
release is fine), the skill is authoritative — the disagreement
means either:
- The skill caught something the human missed (most likely), OR
- The skill needs to be updated (Sprint-3-style amendment, separate
  branch)

Either way, the release flow halts until the disagreement is
resolved.

**Concrete past case anchoring the "skill needs updating" path:**
Sprint 4 itself amended `/mlx-mfa-release-audit` Check 3 from
probing a hardcoded `_hooks_installed` attribute to defensive
introspection of any `*install*`/`*hook*` boolean attribute, after
the actual attribute name in `_auto_hooks.py` turned out to be
`_HOOKS_INSTALLED` (uppercase).  Without the disagreement-resolution
clause, the skill would have produced a false BLOCKED that humans
would have manually overridden — instead, the skill got fixed.

---

### §AA.5 — Premise validation discipline (added 2026-05-14, Sprint 3+4 retrospective)

Before committing to an audit-prescribed implementation, **verify
empirically that the audit's premise still holds**.  Three sprints in
v2.50 (Sprint 1 density threshold, Sprint 2 rope NAX, Sprint 3 top-K)
discovered that the audit's framing was at least partially inverted —
Apple primitives delivered the win in 30-50 LOC rather than the
prescribed L-effort kernel build.  One sprint (Sprint 4) discovered
the audit's scope estimate was 1.5-2× too optimistic.

**Pattern that must trigger the premise-validation check**:
- Audit prescribes "build new Metal kernel" or "extend kernel
  significantly" at S/M/L effort
- Premise check (~30 min) before committing to implementation:
  1. List all `mx.fast.*`, `mx.*` primitives that operate on this
     operand pattern (use `dir(mx)`, `dir(mx.fast)`)
  2. Bench candidate primitive-based dispatch paths against the
     audit's measured regression
  3. Decompose audit's measurement by component (matmul vs sort vs
     softmax vs reduce) to identify the actual bottleneck
  4. Determine outcome:
     - **Full inversion**: Apple primitive recovers the full
       regression → ship dispatch fix, no kernel work
     - **Partial inversion**: Apple primitive recovers part of the
       regression → ship dispatch fix + document why kernel work
       remains needed
     - **Confirmation**: audit's prescription is empirically correct
       → proceed with kernel implementation

**§AA.5 is a BLOCKING gate**: cannot proceed to implementation phase
of a kernel-build sprint without a §AA.5 premise check section in
the sprint deliverable doc.  Templates updated in §10.1 to require a
"Premise validation" section adjacent to "Skill invocations".

### Why §AA.5 was needed

The audit's effort estimate is necessarily produced without the
kernel-implementation-time investigation that surfaces these
inversions.  Sprint 1 demonstrated this acutely: the audit
prescribed bool-mask substitution (Layer 1 from a 2025 doc); the
v2.50 empirical check on MLX 0.31 found bool 1.085× SLOWER than
float bias (FALSIFIED), and the actual win came from a one-line
density threshold change.  Without §AA.5, Sprint 1 would have
shipped a no-op bool-mask substitution and missed the 6× win.

### §AA.5 evidence template

Every kernel-build sprint deliverable doc MUST include:

```markdown
## §AA.5 Premise validation

**Audit prescription**: <verbatim from audit>

**Available Apple primitives checked**:
- `mx.<primitive_1>`: <signature, applicable scope>
- `mx.fast.<primitive_2>`: <signature, applicable scope>
- ...

**Candidate primitive-based paths benched**:
| Path | Latency | Speedup vs current |
|---|---|---|
| <current> | X ms | 1.00× |
| <candidate A> | Y ms | N× |
| ...

**Premise verdict**: <FULL_INVERSION / PARTIAL_INVERSION / CONFIRMATION>

**Rationale**: <one paragraph explaining what the empirical data tells us>
```

If verdict is FULL_INVERSION: ship dispatch fix; kernel sprint cancelled.
If PARTIAL_INVERSION: ship dispatch fix AND scope-correct the kernel
work needed for the remainder.
If CONFIRMATION: proceed with kernel implementation as scoped.

### Related: audit framing inversions catalogue

See `docs/v50/audit-framing-inversions.md` for the empirically-validated
list of audit framing inversions through v2.50.  Update this doc each
time a sprint surfaces a new inversion or confirms an audit prediction.

### §AA.5.x — Multi-gate audit requirement (added 2026-05-14, v2.50 Prompt 4 retrospective)

**Rule**: when an investigation surfaces a kernel-input compatibility
issue (e.g., LSE convention, scale convention, dtype packing, buffer
layout), the fix MUST enumerate ALL dispatch sites that produce that
input — not just the one the failing test touches.

**Why**: v2.50 Prompt 4 found that the V34 backward dV residual was
caused by an "incomplete-fix dispatch-chain" — two upstream gates
were patched to route to V34 forward (natural-log LSE), but a third
gate in `MFAV6Forward::eval_gpu()` continued routing causal forward
to STEEL legacy (log2-domain LSE).  The V34 backward consumed the
log2 LSE as if it were natural-log, producing ~0.4 dV residual.  Each
of the three dispatch sites read correct in isolation; only the
unfixed third site's interaction with the V34 backward exposed the
incompleteness.  See Pattern #5 in
`docs/v50/audit-framing-inversions.md`.

**Audit checklist** before declaring any kernel-input fix complete:

1. **Identify the input contract**.  What format/convention does the
   consumer kernel expect?  Document it explicitly (e.g., "V34 backward
   consumes lse in natural-log domain").

2. **Enumerate ALL producer sites**.  Use `grep -rn "set_output\|write.*lse"`
   or equivalent for the input.  For each producer:
   - Is it currently producing the expected format?
   - Could it dispatch through this code path for the failing test?

3. **Verify each producer site individually**.  Use sentinel writes
   per `docs/methodology/kernel-debugging.md` §2 to confirm which
   producer is actually active for the failing test.

4. **Cross-check with eligibility gates**.  Producer sites are often
   guarded by eligibility predicates (`_v34_eligible`, `_v34_backward_carveout`,
   etc.).  A fix to one gate is insufficient if another gate routes
   the failing test to a different producer.

5. **Document the gate-set in the fix's commit message**.  Future
   investigators encountering related residuals will know which gates
   were verified at fix-time vs which need fresh verification.

**Anti-pattern**: don't trust "the failing test now passes" as proof
of completeness.  The test may have stopped exercising the buggy
producer path while leaving the bug latent in other dispatch routes.

**Cross-references**:
- `docs/methodology/kernel-debugging.md` (sentinel writes + LSE
  consistency techniques)
- `docs/v50/audit-framing-inversions.md` §6 (Pattern #5 catalogue entry)
- `docs/v6-nax/v50-prompt4-sectionb-dv-residual-RESOLVED.md` (full
  empirical case)

---

## 6. Scope discipline — pas de re-escalade prématurée

Quatre sessions successives ont eu CC qui a re-escaladé un sprint en "multi-session scope" malgré un mandat explicite d'aller au bout. Internalise :

### 6.1 Si tu estimes "ce sprint nécessite plus de temps que prévu"

**Ne re-escalade PAS.** Au lieu :
1. Sub-scope toi-même au plus petit incrément qui livre quelque chose de testable
2. Documente précisément ce qui rentre dans le scope sub-scope
3. Continue à travailler sur le sub-scope jusqu'à le livrer
4. À la fin, documente ce qui reste à faire pour les futures sessions

### 6.2 Si tu estimes "ce que je tente ne marche pas"

**Pivote, ne rends pas la main.** Au lieu :
1. Documente précisément pourquoi ça ne marche pas (avec citations file:line)
2. Bascule vers la Phase contingence définie dans le prompt
3. Si pas de contingence définie : **demande au prompt suivant** mais en livrant le diagnostic, pas en disant "à investiguer"

### 6.3 Si tu estimes "il manque des informations Apple internes"

**Ne suppose pas.** Au lieu :
1. Va dans `~/code/mlx-source` et lis le code
2. Cherche le pattern dans `steel_attention_nax.h` ou `nax.h`
3. Si vraiment l'information n'est pas dans le repo MLX, écris un probe MSL pour reverse-engineer empiriquement

### 6.4 Trois exemples de re-escalade injustifiée à ne PAS reproduire

- "10-15h focused work, multi-session scope" — alors que 80% du code existe déjà dans le repo (`csrc/mfa_steel_fwd_v6_nax.cpp`)
- "5e-3 RMSE bloque, multi-session scope" — alors que la mesure était fausse (cache cross-process)
- "Phase 2 escalated to multi-session" — alors que le mandat budget était explicite et généreux

---

## 7. Source generator — patterns établis

Le code V6 NAX est généré par un C++ source generator (`csrc/mfa/v6_nax/NAAttentionKernel.cpp`, ~3338 LOC) qui produit du MSL au runtime via templates `{{VAR}}` et `replace_all` post-substitution.

### 7.1 Cohérence cache key ↔ source generator

**Tout paramètre qui modifie le source MSL DOIT modifier la cache key.** Sinon le shader cache retourne une pipeline obsolète. Pattern actuel dans `mfa_v6_nax_primitive.cpp:54-79` :

```cpp
struct V6Key {
    int head_dim, Hq, Hk, dtype;
    bool isCausal;
    uint32_t R, C, qbs, kbs, vbs, obs;
    bool use_v33 = false;       // dedicated field, no bit-packing
    bool operator==(const V6Key& o) const { ... }
};
```

Pour ajouter un paramètre (ex: `use_v34`), ajoute un champ dédié, mets à jour `operator==` ET `V6KeyHash::operator()`. Pas de raccourcis.

### 7.2 Post-generation rewriting

Le code utilise `replace_all` après génération pour des modifications complexes (BHND rewriter, etc.). Lis les sites existants dans `mfa_v6_nax_primitive.cpp` avant d'en ajouter un nouveau. Préfère générer le bon code directement quand possible plutôt que de patcher après.

---

## 8. Tests à respecter

Avant tout commit qui touche V6 NAX :

```bash
.venv/bin/python -m pytest tests/ -x -v --tb=short -k "v6_nax or attention" 2>&1 | tail -50
```

Tests connus comme flaky pré-existants : 5 dans le fichier `tests/test_v6_nax*.py` qui failent sur `feat/v6-nax` aussi. Ne les attribue pas à ton sprint si ils étaient déjà rouges avant.

Tile coverage sur les 5 shapes production :
```bash
MFA_V6_SENTINEL_FILL=1 .venv/bin/python bench/v6_coverage_diagnostic_v2.py
```

---

## 9. Branche et merge policy

- Toujours travailler sur une branche `experiment/<sprint-name>` partant de `feat/v6-nax`
- **Jamais de push.** Marco merge manuellement après review.
- Commits petits et atomiques. Un commit par sous-système, pas un mega-commit final.
- Messages de commit avec contexte : "feat(v6-nax): X — Y%; Z replaces W". Pas de "WIP".

---

## 10. Documentation requise par sprint

À chaque sprint, livre :

1. **Inventaire** dans `docs/v6-nax/<sprint>-inventory.md` — citations file:line de tout ce qui est touché
2. **Décisions techniques** dans `docs/v6-nax/<sprint>-decisions.md` — pour chaque choix non-trivial : contexte, options considérées, choix retenu, raisonnement
3. **Résultats** dans `docs/v6-nax/<sprint>-results.md` — bench numbers cross-session, par shape, avec RMSE
4. **Raw data** dans `docs/v6-nax/<sprint>-data.json`
5. **SESSION_LOG entry** dans `devnotes/SESSION_LOG.md` — résumé exécutif lisible en 2 minutes

Marco doit pouvoir comprendre TOUS les choix techniques en lisant les rapports, sans avoir à interpréter le code.

### §10.1 — Templates (added 2026-05-13, Sprint 4)

Use the templates in `docs/templates/` as starting points:

- `docs/templates/sprint_decisions_template.md` — pre-includes the
  §AA.2 Skill invocations table so it cannot be forgotten
- `docs/templates/sprint_status_template.md` — same pattern for
  status docs

Per §AA.2, EVERY sprint deliverable doc must include a populated
Skill invocations table.  The templates make this mechanical: copy
the template, fill in the rows as the sprint progresses, ship.

If a §AA mandatory checkpoint did not fire during the sprint,
document inapplicability with one row (`N/A | N/A | inapplicable:
<reason>`) — silent absence is indistinguishable from forgetting.

---

## 11. Si tu te retrouves à inventer une approche non-Apple

C'est probablement le signal qu'il faut s'arrêter et lire `~/code/mlx-source/mlx/backend/metal/kernels/steel/attn/`.

Apple a pensé à ces problèmes. Leur code est dans le repo. Tu n'as pas besoin de réinventer si ça existe déjà.

Le pattern hybride V33 (TG-memory bridge entre cooperative_tensor MPP) **est une approche non-Apple**. Apple ne fait pas ça. Apple opère un niveau en dessous (NAXFrag direct). C'est pour ça que le V33 hybrid a échoué et que V34 NAX-direct est la suite.

---

## 12. Quand demander à Marco

Marco te dit toujours "autonomie complète, pas de demande de permission". MAIS :

- Si tu trouves un bug dans le code de production qui n'était pas le scope de ton sprint, **commit fix séparé avec message clair**, ne le mélange pas avec le sprint
- Si tu modifies un fichier hors scope V6 NAX (par ex `mfa_attention.cpp`), **mention dans le rapport** et explique pourquoi
- Si Apple change `nax.h` ou `steel_attention_nax.h` entre deux sessions et que ça casse le code, **adapte et documente** le diff

Mais ne demande pas de permission pour des décisions techniques de scope sprint. Tranche, exécute, documente.

---

## Ajout suggéré dans CLAUDE.md global

Ajoute cette ligne dans le CLAUDE.md global, sous "Source layout" :

```
## V6 NAX work — guardrails séparés
Si ton sprint touche `csrc/mfa/v6_nax/` ou `csrc/mfa_v6_nax_primitive.cpp`,
lis `CLAUDE_V6_NAX.md` à la racine du repo avant de coder. Il contient
les règles méthodologiques accumulées depuis les sprints v2.27.0 à v2.30.x.

MLX source local : `~/code/mlx-source` — référence Apple obligatoire pour
toute question sur le comportement MPP/NAX.
```

---

### §AA.6 — Stable-but-unverified production-critical code in audit scope (added 2026-05-15, v2.50.1 Prompt 5g retrospective)

**Rule**: pre-release audit scope must include stable code that is
production-critical, not just code modified since the last release.
This specifically applies to:

- **Auto-hooks patching MLX primitives globally** (Pattern #8
  vulnerability — e.g., `_auto_hooks.py::_patched_conv_general`).
- **Dispatch policy decisions** consumed by multiple code paths.
- **Cache key construction** (Pattern #5 vulnerability).
- **Compile pipeline branching logic** (Pattern #5 vulnerability).

**Audit framework** must explicitly include "stable production-critical
code" as a separate scope category alongside "modifications since
last release" in inventory phase.

**Rationale**: Prompt 5e audit scope was "modifications since v2.39.1"
which missed `_auto_hooks.py::conv3d_nax_forward` (unchanged since
v2.36.0).  The bug — a dtype-mismatch silent break where the C++ NAX
kernel required `x.dtype == w.dtype` but the Python eligibility check
only verified the weight side — survived from v2.36.0 through v2.50.0
because audit scope excluded stable code regardless of production
criticality.  Every VSR VAE encoder call hit `RuntimeError` for ~6
weeks of production releases; user pipelines silently absorbed the
exception, masquerading as "everything works" while the M5 Neural
Engine NAX Conv3D acceleration NEVER engaged.  See Pattern #8 in
`docs/v50/audit-framing-inversions.md` for the full case study and
`docs/v50/known-debt-v2.50.md` KD-6/KD-7.

**Concrete checklist** for stable-but-unverified production-critical
code:

1. **Inventory all global hooks** (`grep -rn 'mx\.\(.*\) = ' src/`).
   Document each hook's: patched primitive, eligibility contract,
   fallback path.
2. **Compare hook eligibility vs patched primitive's documented
   contract**.  Tighter eligibility = Pattern #8 risk.
3. **Verify hook telemetry** is in place (e.g.,
   `mlx_mfa.get_hook_stats()`).  Releases without telemetry are
   flying blind.
4. **Run integration smoke tests** that assert hook engagement
   (`executed > 0`, `fallback == 0`) for representative production
   patterns.
5. **Cross-version bench delta check**: if v(N+1) "added optimization"
   shows no measurable perf delta vs v(N), investigate whether the
   optimization actually engages — it may be a Pattern #8 ghost.

### §AA.7 — Dispatch/source constant parity is in audit scope (added 2026-06 campaign Sprint B)

Any dimension constant (block size, tile size, SIMD/threadgroup size,
threadgroup-memory size) that appears in BOTH the dispatch-side
launch-grid computation AND the source-side kernel generator MUST be
verified equal per dispatch-table cell (D × dtype × causal × kernel
direction × hardware-gen branch).

A divergence produces **silent partial-write corruption** that presents
as a hardware-specific "limitation" and tends to be misfiled as
deprecated debt: the bug only manifests on cells where the two values
differ, so the developer's machine (taking the consistent branch)
shows correct behavior.  Reference case: KD-5 — `MFASteelBwdDKV`
dispatched with cfg.BK=32 (M3+, D=128) while the generator hardcodes
BK=16 for D>64; carried for weeks as "deprecated STEEL backward debt",
fixed by one expression in the 2026-05 whole-repo review.  See
Pattern #9 in `docs/v50/audit-framing-inversions.md`.

**Enforcement**: `/mlx-mfa-release-audit` Check 9 (source/dispatch
block-dimension consistency) runs the per-(kernel, constant) parity
checklist at every pre-tag gate.  Any mismatch is a CRITICAL finding.

**Scope extension of §AA.6**: stable-but-unverified kernels with
hardcoded generator constants are explicitly in audit scope — "this
kernel hasn't changed in N releases" is not an exemption, because the
divergence may be activated by a DISPATCH-side change (new hardware
branch, new cfg table entry) without any generator edit.

**KD-ledger discipline corollary**: a KD entry describing a kernel as
fundamentally limited MUST record the failure MECHANISM, not just the
symptom.  KD-5 sat as accepted debt precisely because it recorded
"zeroes blocks at D=128 N≥2048" (symptom) with a speculative theory
("tile-loop termination bug") instead of a derived mechanism.  A KD
entry without a mechanism is an OPEN INVESTIGATION, not a verdict.
