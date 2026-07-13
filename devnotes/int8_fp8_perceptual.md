# ITEM 6 — int8/fp8-NA compute et probe perceptuel

Date: 2026-07-13. Branche: `explore/int8-na-perceptual`. Résultats M5 Max/macOS 27 beta, MLX 0.31.2,
mlx-mfa 2.61.0; indicatif beta3, pas une décision de routage.

## §AA.5 et invocations

| Point | Vérification |
|---|---|
| Substrat int8/fp8 | `csrc/mpp_int8_bench.mm`, `CMakeLists.txt`, headers MLX sous `/tmp/mlx/mlx-main` |
| Adversaire compute | même probe MPP `matmul2d` fp16, mêmes rectangles et mêmes timestamps GPU |
| Qualité | classes/poids SeedVR2 réels, entrées synthétiques shape-faithful, VAE réel pour SSIM |
| Skills | `metal-kernel-dev`, `benchmark-harness-builder`, `mlx-validation-runner` |

Le repo externe SeedVR2 a été lu uniquement et n'a pas été modifié. HEAD observé: `ced265d`.

## Étape 0 — faits matériels

`csrc/mpp_int8_bench.mm:8-25,59-175` confirme que Metal 4/M5 accepte des opérandes `int8_t` dans
les formes MPP testées, avec accumulation int32 dans la forme cooperative. Le type `char` est
incorrect pour cette sélection MPP; le probe exige `int8_t` [VERIFIED].

Le probe historique `mpp_int8_microbench()` est un throughput probe `64x64x128`, pas un GEMM rectangulaire.
Il conserve deux références historiques, mais elles ne sont pas utilisées comme résultat de cette étape:
`233 TOPS / 124 TFLOPS = 1.88x` en forme register-fill, et des résultats device-tensor plus récents
autour de `1.37-1.46x` [VERIFIED].

Le probe fp8 sous `mpp_fp8_microbench()` compile sous Metal 4.1, mais les types sont des formats MX
block-scaled. Le binding tensor-argument ne transmet pas les extents, et le source note lui-même que
l'engagement et les opérandes corrects ne sont pas prouvés sans plan de scales MX dans le shader. Le run
du 13 juillet a affiché `0.861x` (E4M3/f16), `0.907x` (E4M3/f32), `0.862x` (E5M2/f16) et `1.243x`
pour int8 sur ce probe instruction-level; ces valeurs fp8 ne sont **pas** des speedups acceptés
[VERIFIED]. MLX `quantized_nax` et `fp_quantized_nax` concernent le qmm affine/MX packed, pas un raw
fp8 GEMM général [VERIFIED]. Conclusion: int8 est V6/Metal 4.0; fp8 raw reste V7/Metal 4.1 et hors
surface fiable de ce run [DEDUCED].

## Étape 1 — GEMM int8 réel

Le nouveau symbole dev-only `_ext.mpp_int8_rect_microbench(bool)` est ajouté derrière
`MFA_BUILD_PROBES=ON`. Il construit les tensors `tensor_inline` avec extents explicites, accumule les
K-tiles de 128 avec `multiply_accumulate`, remet C à zéro à chaque dispatch et vérifie la sortie contre
l'oracle analytique des buffers constants. Les bras sont des sources MSL distinctes: `half -> half` et
`int8_t -> int32_t`; le probe est donc engagé et non un fallback MLX [VERIFIED].

Chaque ordre a été exécuté dans 5 processus frais (`fp16-first`, puis `int8-first`), avec 20 dispatches
GPU timestampés par bras et 20 répétitions MPP par dispatch. Les raw JSON sont sous `benchmarks/results/`
et restent non suivis [VERIFIED].

| Rectangle | Médiane int8/fp16 sur 10 sessions | Min-max | Verdict porte 1.2x |
|---|---:|---:|---|
| Attention `M=2048,K=128,N=2048` | `1.084x` | `1.059-1.105x` | fermé, `<1.2x` |
| FFN-up `M=2048,K=1536,N=8960` | `0.968x` | `0.951-1.171x` | fermé, `<1.2x` |

Le compute int8 ne franchit donc pas la porte `>=1.2x` sur les deux formes demandées. L'attention ne
justifie aucun chantier int8; le rectangle FFN est même plus lent en médiane malgré la capacité MPP
int8 [VERIFIED]. La variance initiale observée avant le lock `multiply_accumulate` et lors d'un lancement
parallèle de processus est rejetée; elle provenait du probe non accumulant puis d'un harness qui n'attendait
pas les processus. La série finale est séquentielle et chaque sortie est validée [VERIFIED].

## Étape 2 — carte fake-quant perceptuelle

Le harness `benchmarks/bench_int8_fp8_perceptual.py` charge les classes et poids de production SeedVR2,
avec une entrée synthétique shape-faithful `B=1, T_lat=4, H=54, W=66, D=2560, Hq=20, d=128, texte=58`.
Ce n'est pas un clip archival complet `T_lat=38`: c'est un proxy modèle-réel explicite. Le VAE réel est
néanmoins utilisé pour décoder les trois premières tranches latentes et calculer SSIM/PSNR par frame
[VERIFIED]. Les wrappers quantifient puis déquantifient en fp32 avant de recaster; aucun kernel rapide
int8 n'est revendiqué.

| Site | Granularité | Cos intermédiaire | SSIM min | PSNR |
|---|---|---:|---:|---:|
| QK | per-tensor | 0.999974261 | 0.999946719 | 59.33 dB |
| QK | block32 | **0.999983717** | **0.999967623** | **61.45 dB** |
| QK | block64 | 0.999982543 | 0.999963710 | 61.04 dB |
| QK | block128 | 0.999981604 | 0.999963124 | 60.95 dB |
| QK | per-channel | 0.999980307 | 0.999961498 | 60.68 dB |
| PV | per-tensor | 0.999963745 | 0.999919848 | 57.67 dB |
| PV | block32 | **0.999982549** | **0.999964933** | **61.14 dB** |
| PV | block64 | 0.999981783 | 0.999963696 | 60.95 dB |
| PV | block128 | 0.999981339 | 0.999963206 | 60.89 dB |
| PV | per-channel | 0.999981204 | 0.999963564 | 60.88 dB |
| FFN | per-tensor | 0.997218044 | 0.988473044 | 37.69 dB |
| FFN | block32 | **0.999926210** | **0.999835965** | **55.14 dB** |
| FFN | block64 | 0.999895383 | 0.999814671 | 53.94 dB |
| FFN | block128 | 0.999840439 | 0.999676382 | 52.07 dB |
| FFN | per-channel | 0.999856840 | 0.998942454 | 50.73 dB |
| QK+PV+FFN | per-tensor | 0.997194827 | 0.987751365 | 37.60 dB |
| QK+PV+FFN | block32 | **0.999923751** | **0.999808078** | **54.79 dB** |
| QK+PV+FFN | block64 | 0.999893128 | 0.999811414 | 53.86 dB |
| QK+PV+FFN | block128 | 0.999836753 | 0.999652094 | 51.98 dB |
| QK+PV+FFN | per-channel | 0.999853132 | 0.998936490 | 50.58 dB |

Les sites QK/PV tolèrent toutes les granularités testées dans ce proxy; la meilleure cellule est block32.
La quantification FFN répétée sur les 32 blocs montre une sensibilité supérieure: per-tensor est clairement
dégradé, block32 est la meilleure granularité testée, et per-channel descend sous `0.999` SSIM min.
Ces chiffres sont une carte de qualité, pas un seuil de décision imposé à Marco [VERIFIED].

### Pareto vitesse/qualité

Le seul speedup disponible est le plafond micro MPP de l'étape 1, pas une vitesse fake-quant: environ
`1.084x` attention et `0.968x` FFN. Le fake-quant n'a volontairement aucun chemin rapide et ne peut donc
pas être présenté comme un gain end-to-end. La carte indique une qualité viable en block32, mais aucun
point ne combine ici une vitesse réelle `>=1.2x` avec cette qualité [VERIFIED].

## Étape 3 — porte économique et verdict

La porte compute est fermée: aucune forme représentative ne tient `1.2x` en mesure rectangulaire avec
opérandes réellement engagés. La porte qualité fake-quant est, elle, partiellement favorable: block32
est la meilleure candidate perceptuelle, surtout pour QK/PV et en combiné; per-tensor est rejetée par la
dégradation observée. L'étape 3, qui exigerait les deux portes, n'est donc pas exécutée. Aucun kernel,
routage, SageAttention ou changement de défaut n'est ajouté [VERIFIED].

Recommandation à Marco: conserver int8 comme substrat de recherche uniquement; ne pas relancer Sage ou
écrire un kernel int8 sans une nouvelle hypothèse qui améliore le rectangle FFN et absorbe les coûts de
packing/scales. Si une décision perceptuelle ouvre ce chantier plus tard, commencer par block32 et refaire
la validation sur `T_lat=38`/clip archival réel, puis mesurer le pipeline complet. fp8 nécessite un probe
Metal 4.1 avec plan de scales MX et ne doit pas être mélangé à ce verdict [DEDUCED].

## Red-team

- Le probe MPP compare deux kernels raw MPP, pas `mx.matmul` de production: le résultat est un plafond
  de datapath, non un speedup end-to-end [VERIFIED].
- Le proxy utilise des poids réels mais des activations synthétiques `T_lat=4`; les SSIM ne ferment pas
  la question du clip archival `T_lat=38` [VERIFIED].
- Le fake-quant est symétrique int8 et ne modélise pas les scales/packing exacts de `quantized_nax`; une
  implémentation future doit matcher cette simulation puis être revalidée [DEDUCED].
- Les premières mesures non accumulantes et la tentative concurrente sont explicitement exclues; seuls
  les JSON `int8_rect_final_*` et `fake_int8_{attention_qk,attention_pv,ffn,combined}_*.json` servent au
  tableau [VERIFIED].

## Stamp

- Python 3.11.14; MLX 0.31.2; mlx-mfa 2.61.0.
- Apple M5 Max, 40 GPU cores, macOS 27.0 beta, Metal 32023.918.
- mlx-mfa HEAD au run: `a461b59`.
- SeedVR2 HEAD lu en lecture seule: `ced265d`.
- Aucun push, tag, release, routage ou modification du repo SeedVR2.
- Validation finale: `3488 passed, 93 skipped, 3 warnings`; `py_compile` des deux harnesses et `git diff --check` verts.
