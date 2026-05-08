# Synthèse finale - Modalité clavier

Ce document résume la modalité **Keyboard Dynamics** retenue pour le prototype multimodal de détection exploratoire de signes compatibles avec Parkinson.

Le modèle ne fournit pas de diagnostic médical. Il produit uniquement un score de risque expérimental destiné à être combiné avec d’autres modalités, comme la voix ou le dessin.

## Objectif

La modalité clavier mesure la dynamique de frappe d’un utilisateur pendant un test standardisé dans le navigateur. Les événements `keydown` et `keyup` sont transformés en features temporelles, puis passés à un modèle entraîné sur NeuroQWERTY.

L’objectif n’est pas de reproduire exactement un article, mais d’obtenir une baseline robuste, compatible avec l’application web et évaluée sans fuite de données entre sujets.

## Données utilisées

Le dataset retenu est **NeuroQWERTY / MIT-CS1PD / MIT-CS2PD**.

Il contient :

- `85` sujets au total ;
- `116` sessions disponibles ;
- des logs bruts de frappe ;
- un label contrôle/Parkinson ;
- un protocole encadré et documenté.

Le dataset Tappy a aussi été exploré comme source complémentaire. Il n’a pas été retenu pour le modèle final, car son protocole est moins contrôlé et moins proche du test navigateur standardisé visé.

## Méthode retenue

La méthode finale retenue est :

```text
agg_timing_xgb
```

Elle utilise :

- segmentation des sessions en fenêtres de frappe ;
- features temporelles agrégées sur `hold_time` et `flight_time` ;
- statistiques de tendance, dispersion et quantiles ;
- variations locales via différences successives ;
- autocorrélations simples ;
- modèle `XGBoost` ;
- agrégation des scores par moyenne des segments.

Le modèle exporté est :

```text
models/keyboard_dynamics_neuroqwerty_agg_timing_xgb.joblib
```

L’application Flask charge directement cet artefact avec :

```text
src/modalities/keyboard/predictor.py
```

## Résultats principaux

L’évaluation finale du notebook 10 donne :

| Métrique | Moyenne | Écart-type |
|---|---:|---:|
| Accuracy | 0.724 | 0.061 |
| Balanced accuracy | 0.715 | 0.074 |
| F1 macro | 0.706 | 0.080 |
| F1 Parkinson | 0.712 | 0.139 |
| Précision Parkinson | 0.762 | 0.095 |
| Rappel Parkinson | 0.715 | 0.226 |
| ROC-AUC | 0.834 | 0.039 |
| PR-AUC | 0.882 | 0.055 |

La matrice de confusion agrégée est :

| Classe réelle | Prédit contrôle | Prédit Parkinson |
|---|---:|---:|
| Contrôle | 40 | 15 |
| Parkinson | 17 | 43 |

Le score continu est plus intéressant que la décision binaire seule. C’est pour cette raison que la modalité clavier est surtout pertinente dans une fusion tardive multimodale.

## Comparaisons effectuées

Plusieurs pistes ont été testées avant de fixer le modèle final :

| Piste | Conclusion |
|---|---|
| Features classiques + modèles sklearn | Baseline utile, mais performances modestes. |
| PSO + sélection de features | Intéressant en exploration, mais scores optimistes si la sélection n’est pas strictement imbriquée. |
| HistGB timing-only + PSO | Plus propre vis-à-vis du layout clavier, mais moins performant. |
| XGBoost + features ondelettes | Améliore certaines métriques, mais sélection instable selon les folds. |
| DWT/DTW + XGBoost | Ne bat pas `agg_timing_xgb` en validation répétée. |

La dernière expérimentation DWT/DTW a confirmé que les variantes plus complexes n’apportent pas de gain robuste par rapport à `agg_timing_xgb`.

## Méthodologie d’évaluation

L’évaluation utilise une séparation groupée par sujet : un même `pID` ne peut jamais être à la fois dans l’entraînement et dans la validation.

C’est un point central, car les données contiennent plusieurs sessions par sujet. Un split classique par session créerait une fuite de sujet et donnerait des scores trop optimistes.

Les métriques principales sont :

- `ROC-AUC`, pour mesurer la qualité du score continu ;
- `PR-AUC`, utile avec des classes potentiellement déséquilibrées ;
- `F1 macro`, pour observer la décision binaire ;
- `balanced accuracy`, pour limiter l’effet du déséquilibre de classes.

## Intégration navigateur

Dans l’application :

1. l’utilisateur ouvre `/keyboard` ;
2. il recopie un texte standardisé ;
3. le navigateur capture les événements de frappe ;
4. le backend reconstruit les frappes valides ;
5. les features `agg_timing_xgb` sont extraites ;
6. le modèle produit un score ;
7. le résultat est stocké en session ;
8. la page `/results` peut fusionner ce score avec les autres modalités.

Le résultat retourné respecte le contrat commun `PredictionResult` :

```python
{
    "modality": "keyboard",
    "status": "ok",
    "score": 0.62,
    "confidence": 0.9,
    "label": "elevated",
    "details": {},
    "warnings": []
}
```

## Notebooks associés

Les notebooks les plus directement liés au résultat final sont :

```text
notebooks/keyboard/10_keyboard_dynamics_neuroqwerty_final_agg_timing_xgb.ipynb
notebooks/keyboard/11_keyboard_dynamics_neuroqwerty_dwt_dtw_repeated_auc.ipynb
```

Le notebook `06_keyboard_dynamics_neuroqwerty_rigorous_selection.ipynb` documente la validation stricte avec folds groupés. Les autres notebooks correspondent aux explorations intermédiaires et aux pistes non retenues.

## Parcours expérimental

Le cheminement expérimental, les choix méthodologiques, les essais intermédiaires et les pistes non retenues sont détaillés dans :

```text
docs/keyboard/parcours_experimental_keyboard_dynamics.md
```

## Limites

Les limites principales sont :

- petit nombre de sujets ;
- absence de validation clinique externe ;
- protocole navigateur différent du protocole NeuroQWERTY original ;
- score clavier insuffisant comme test autonome ;
- sensibilité possible au type de clavier et aux conditions de saisie ;
- usage démonstration/prévention exploratoire uniquement.

## Références principales

- Giancardo et al., “Computer keyboard interaction as an indicator of early Parkinson’s disease”, *Scientific Reports*, 2016 : https://www.nature.com/articles/srep34468
- NeuroQWERTY MIT-CSXPD Dataset, PhysioNet : https://physionet.org/content/nqmitcsxpd/
- Arroyo-Gallego et al., “Detecting Motor Impairment in Early Parkinson's Disease via Natural Typing Interaction With Keyboards”, 2018 : https://pubmed.ncbi.nlm.nih.gov/29581092/
- Padate, Chavan et Abin, “Parkinson’s Disease Detection Using Keystroke Dynamics with PSO-Based Feature Selection and Ensemble Voting Classifier”, 2025 : https://www.atlantis-press.com/proceedings/icsiaiml-25/126021170
- Chen et Guestrin, “XGBoost: A Scalable Tree Boosting System”, 2016 : https://arxiv.org/abs/1603.02754
