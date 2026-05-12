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
      paths on at least one representative shape. If perf ratio is
      ~1.00× when the new path is supposed to be faster, the new path
      isn't actually engaged (dead override / fallback engagement).
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

## 4. Thermal protocol M5 Max

Le M5 Max sous Marco utilise le profil de ventilation **iStat Menus performance** (pas le profil Apple par défaut qui throttle agressivement). Température GPU sous charge ~70°C, drift R1↔R3 typiquement <6%.

Cooldowns recommandés :
- 3 min initial cool-down avant le premier round
- 90s entre rounds (A→B, B→A)
- 60s entre shapes pendant un round
- Si drift R1↔R3 > 10%, étendre les cooldowns

### §4.X — Sub-1ms kernel measurement caveat (M5 Max)

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

**Methodology guidance for sub-1.5ms kernels (under active investigation):**
the naive continuous-workload protocol (interleaved warmup dispatches
during cooldown periods) was tested in the 2026-05-12 methodology sprint
and produced a **REGRESSION verdict** — it resolved variance on 2/3 v2.36.0
HIGH-variance shapes but regressed 3/4 v2.36.0 CONFIDENT shapes. See
`docs/methodology/sub1ms-protocol-diagnostic.md` for the full analysis.

Root cause of the regression: the warmup workload (small matmul, ~50µs
dispatch on a 256×256 FP16 shape) competes with the measured kernel for
GPU L2 / scheduling state. A small workload sufficient to hold GPU power
state high is still large enough to perturb cache state in a way that
matters for sub-1.5ms shapes.

Until a clean methodology is established, the practical guidance is:

- **For kernels ≥ 2.0 ms wall-clock**: §4 cooldowns work fine, ship via
  standard 3-session §4-strict protocol.
- **For sub-1.5 ms kernels**: cross-session variance characterization is
  not yet possible under either idle-cooldown or continuous-workload
  protocol. SHIP_OPT_IN is the conservative verdict regardless of perf
  ratio magnitude. Explicit opt-in via env var (`MFA_LCSA_KERNEL_VERSION=v2`
  for the v2.35.0 case) preserves user agency.

Candidate next-sprint ideas tracked in
`docs/methodology/sub1ms-protocol-diagnostic.md` §"Path forward":
matched-workload-family warmup, sub-millisecond heartbeat-only warmup,
Metal power-state API investigation, accept-the-trade-off (≥1.5ms
shapes only get auto-default).

References:
- `docs/methodology/sub1ms-protocol-results.md` (REGRESSION analysis)
- `docs/methodology/sub1ms-protocol-diagnostic.md` (root cause + path forward)
- `docs/methodology/downclock-threshold-data.json` (Section B empirical anchor)
- `docs/lcsa-nax/v2-only-rebench-diagnostic.md` (originating STOP diagnostic)

---

## 5. Correctness avant tout

Avant TOUTE mesure de timing :
- **Tile coverage 100%** via `MFA_V6_SENTINEL_FILL=1` + `bench/v6_coverage_diagnostic_v2.py`
- **RMSE FP32 < 1e-3** vs `mx.fast.scaled_dot_product_attention` (FP32 reference)

Si une seule shape fail correctness, **skip cette config**, ne tente pas de la shipper.

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
